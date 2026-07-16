"""Top-level Vesper backtest entry point.

Pipeline:

1. Build (or load) the synthetic universe (``scripts.synthetic_generator``).
2. Parse "filings" and compute NLP information-decay per release date.
3. Aggregate to weekly observation cadence per ticker.
4. Compute idiosyncratic-return targets from synthetic market + sector data.
5. Train the regularised alpha model with purged time-series CV.
6. Score weekly alpha predictions.
7. Sector-neutralise and feed into the convex portfolio optimizer.
8. Iterate weekly, log PnL and diagnostics.

Run with::

    python main.py --data-dir data

This script is deliberately *self-contained*: it does not need internet
access. The synthetic data generator produces the same artefacts every
run, so the backtest is deterministic.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Project modules
from data_pipeline.graph_builder import build_supply_chain_graph_from_json
from data_pipeline.mda_parser import clean_text
from features.nlp_decay import compute_information_decay
from features.shock_propagation import propagate_shock_scores
from alpha_model.cross_sectional_model import AlphaModel, build_training_matrix
from alpha_model.target_formulation import compute_residual_returns_rolling
from portfolio.convex_optimizer import (
    ConvexPortfolioOptimizer,
    PortfolioConstraints,
    TransactionCostConfig,
    build_alpha_layer,
)
from portfolio.factor_neutralization import neutralize_to_sectors
from scripts.synthetic_generator import generate_synthetic_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("vesper.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacktestResult:
    weights: pd.DataFrame
    weekly_pnl: pd.DataFrame
    diagnostics: dict[str, float]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _weekly_incorporation(
    weekly_baseline_df: pd.DataFrame,
    feature_series: pd.DataFrame,
    *,
    weekly_date_col: str = "date",
    ticker_col: str = "ticker",
    feature_cols: tuple[str, ...] = ("nlp_decay_score",),
) -> pd.DataFrame:
    """Forward-fill per-release features onto the weekly grid.

    Returns:
        DataFrame indexed ``(date, ticker)`` carrying per-week feature values.
    """
    base = weekly_baseline_df[[weekly_date_col, ticker_col]].copy()
    base = base.set_index([weekly_date_col, ticker_col]).sort_index()
    merged = base.join(feature_series, how="left")
    # We deliberately do not impose a `ffill` — features strictly NaN until
    # the first release; this is the *look-ahead firewall*.
    return merged[list(feature_cols)]


def _weekly_nlp_features(
    filings_df: pd.DataFrame,
    weekly_features_df: pd.DataFrame,
    *,
    release_column: str = "release_date",
    ticker_col: str = "ticker",
    feature_col: str = "nlp_decay_score",
) -> pd.DataFrame:
    """Re-key NLP features from release_date to the weekly feature date.

    This is a strict asof merge: each weekly observation only sees features
    that were *publicly released* on or before its date. Future releases are
    ignored. The helper is *defensive* about column ordering, duplicate
    columns, and index state because it sits at a structural pin in the
    pipeline (downstream code expects ``(date, ticker)`` exactly).

    Args:
        filings_df: Filings table (output of :func:`compute_information_decay`),
            with columns ``release_column``, ``ticker_col``, and ``feature_col``.
        weekly_features_df: Weekly features frame indexed by ``(date, ticker)``.

    Returns:
        Forward-fillable frame indexed by ``(date, ticker)`` with
        ``feature_col``. If either frame is empty, returns an empty frame
        with the correct MultiIndex.
    """
    # Always operate with a (date, ticker) -> feature_col shape at the end.
    date_col = "date"

    # Drop duplicate columns defensively — this can arise if the caller has
    # passed a frame where the index names collided with explicit columns.
    filings_clean = filings_df.loc[:, ~filings_df.columns.duplicated()].copy()
    weekly_clean = weekly_features_df.loc[:, ~weekly_features_df.columns.duplicated()].copy()

    # Validate required columns.
    needed_filings = {release_column, ticker_col, feature_col}
    if filings_clean.empty:
        missing = needed_filings
    else:
        missing = needed_filings - set(filings_clean.columns)
    if missing:
        raise ValueError(
            f"filings_df is missing required columns {sorted(missing)}; "
            f"have {list(filings_clean.columns)}"
        )

    needed_weekly = {date_col, ticker_col}
    if weekly_clean.empty:
        missing_weekly = needed_weekly
    else:
        missing_weekly = needed_weekly - set(weekly_clean.columns)
    if missing_weekly:
        # If index names supply the missing columns (e.g., caller passed a
        # MultiIndexed frame without ``reset_index``), re-key here.
        if (
            len(weekly_clean.index.names) == 2
            and date_col in (weekly_clean.index.names or [])
            and ticker_col in (weekly_clean.index.names or [])
        ):
            weekly_clean = weekly_clean.reset_index()
            missing_weekly = set()
        else:
            raise ValueError(
                f"weekly_features_df is missing required columns "
                f"{sorted(missing_weekly)}; have {list(weekly_clean.columns)}"
            )

    # Sort NLP data (merge_asof requires sorted right key).
    nlp = filings_clean[[release_column, ticker_col, feature_col]].sort_values(
        [ticker_col, release_column]
    )

    # Sort weekly data (merge_asof requires sorted left key per group).
    weekly_long = weekly_clean[[date_col, ticker_col]].drop_duplicates().sort_values(
        [ticker_col, date_col]
    )

    if weekly_long.empty:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=[date_col, ticker_col]
        )
        return pd.DataFrame(
            {feature_col: pd.Series([], dtype=float, index=empty_idx)}
        )

    merged_per_ticker: list[pd.DataFrame] = []
    for ticker, group in weekly_long.groupby(ticker_col, sort=False):
        sub_nlp = nlp[nlp[ticker_col] == ticker].sort_values(release_column)
        if sub_nlp.empty:
            group = group.copy()
            group[feature_col] = np.nan
            merged_per_ticker.append(group)
            continue
        out = pd.merge_asof(
            group,
            sub_nlp[[release_column, feature_col]],
            left_on=date_col,
            right_on=release_column,
            direction="backward",
        )
        merged_per_ticker.append(out)

    merged = pd.concat(merged_per_ticker, ignore_index=True)
    out = (
        merged.set_index([date_col, ticker_col])[[feature_col]]
        .rename_axis([date_col, ticker_col])
        .sort_index()
    )
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_backtest(data_dir: str | Path = "data") -> BacktestResult:
    """Run the full research backtest against synthetic artefacts.

    Args:
        data_dir: Directory holding (or to receive) synthetic artefacts.

    Returns:
        :class:`BacktestResult`.
    """
    data_dir = Path(data_dir)
    expected_files = {
        "price_panel",
        "weekly_returns",
        "weekly_sector_returns",
        "filings",
        "supply_chain",
    }
    if not (data_dir / "price_panel.parquet").exists():
        logger.info("Synthetic artefacts missing — generating under %s ...", data_dir)
        generate_synthetic_dataset(data_dir)
    else:
        logger.info("Loading synthetic artefacts from %s", data_dir)

    price = pd.read_parquet(data_dir / "price_panel.parquet")
    weekly = pd.read_parquet(data_dir / "weekly_returns.parquet")
    weekly_sector_ret_long = pd.read_parquet(
        data_dir / "weekly_sector_returns.parquet"
    )
    if "date" not in weekly_sector_ret_long.columns:
        weekly_sector_ret_long = weekly_sector_ret_long.rename(
            columns={weekly_sector_ret_long.columns[0]: "date"}
        )
    weekly_sector_ret = weekly_sector_ret_long.set_index("date").sort_index()
    filings_raw = pd.read_parquet(data_dir / "filings.parquet")
    graph = build_supply_chain_graph_from_json(data_dir / "supply_chain.json")

    # ----- 1. NLP feature: per-filing information decay. -----
    filings_clean = filings_raw.copy()
    # Pre-cleaning hook (no HTML in synthetic data but we keep parity):
    filings_clean["text_clean"] = filings_clean["text_clean"].astype(str).apply(clean_text)
    nlp_decay = compute_information_decay(
        filings_clean, text_column="text_clean"
    )
    nlp_decay = nlp_decay.rename_axis(index=["release_date", "ticker"])

    # ----- 2. Build weekly feature layer keyed by (date, ticker). -----
    weekly_long = weekly[["date", "ticker", "raw_ret", "sector", "beta"]].copy()
    weekly_features_index = weekly_long.set_index(["date", "ticker"]).sort_index()

    nlp_features_weekly = _weekly_nlp_features(
        nlp_decay.reset_index(),
        weekly_features_index.reset_index(),
        release_column="release_date",
        ticker_col="ticker",
        feature_col="nlp_decay_score",
    )

    # ----- 3. Graph propagated shock: weekly. -----
    # For each (date, ticker) observation in the weekly panel, attach the most-
    # recent direct shock from nlp_decay whose ``release_date`` <= ``date``,
    # then propagate that shock across the supply-chain graph.
    # ``merge_asof`` requires numerical/datetime join keys, so we partition by
    # ticker first (12 tickers) and apply a backward asof merge per ticker.
    feature_release = nlp_decay.reset_index()
    feature_release = feature_release.rename(
        columns={feature_release.columns[0]: "release_date"}
    )
    # Drop duplicates defensively (in case the upstream rename was a no-op).
    if not feature_release.empty:
        feature_release = feature_release.loc[
            :, ~feature_release.columns.duplicated()
        ].copy()
        # Ensure expected columns.
        needed_release = {"release_date", "ticker", "nlp_decay_score"}
        miss = needed_release - set(feature_release.columns)
        if miss:
            raise RuntimeError(
                f"feature_release is missing columns {sorted(miss)}: "
                f"have {list(feature_release.columns)}"
            )

    weekly_dates = sorted(weekly_long["date"].unique())
    weekly_pairs = weekly_long[["date", "ticker"]].drop_duplicates().sort_values(
        ["date", "ticker"]
    )

    # Per-ticker backward asof merge to attach direct shocks.
    direct_frames: list[pd.DataFrame] = []
    for ticker, sub_pair in weekly_pairs.groupby("ticker", sort=False):
        sub_pair_sorted = sub_pair.sort_values("date").copy()
        if feature_release.empty:
            sub_pair_sorted["nlp_decay_score"] = np.nan
            direct_frames.append(sub_pair_sorted)
            continue
        sub_nlp = (
            feature_release.loc[feature_release["ticker"] == ticker]
            .sort_values("release_date")
        )
        if sub_nlp.empty:
            sub_pair_sorted["nlp_decay_score"] = np.nan
            direct_frames.append(sub_pair_sorted)
            continue
        merged = pd.merge_asof(
            sub_pair_sorted,
            sub_nlp[["release_date", "nlp_decay_score"]],
            left_on="date",
            right_on="release_date",
            direction="backward",
        )
        direct_frames.append(merged)

    if direct_frames:
        direct_df = (
            pd.concat(direct_frames, ignore_index=True)
            .set_index(["date", "ticker"])
            .sort_index()
        )
    else:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=["date", "ticker"]
        )
        direct_df = pd.DataFrame(
            {"nlp_decay_score": pd.Series([], dtype=float, index=empty_idx)}
        )

    # Now propagate each week's direct shocks through the supply-chain graph.
    graph_shock_records: list[dict[str, object]] = []
    for week in weekly_dates:
        try:
            week_block = direct_df.loc[(week,)]
        except KeyError:
            continue
        if week_block.empty:
            continue
        direct_series = week_block["nlp_decay_score"]
        propagate = propagate_shock_scores(direct_series, graph)
        for ticker, row in propagate.iterrows():
            graph_shock_records.append(
                {
                    "date": week,
                    "ticker": ticker,
                    "graph_shock_score": float(row["graph_shock_score"]),
                }
            )
    if graph_shock_records:
        graph_shock_df = (
            pd.DataFrame(graph_shock_records)
            .set_index(["date", "ticker"])
            .sort_index()
        )
    else:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=["date", "ticker"]
        )
        graph_shock_df = pd.DataFrame(
            {"graph_shock_score": pd.Series([], dtype=float, index=empty_idx)}
        )

    weekly_features = weekly_features_index.join(
        nlp_features_weekly, how="left"
    ).join(graph_shock_df, how="left")
    # Replace NaNs with 0 for features (no signal == no contribution); this is
    # *not* a leakage because NaN-to-0 means "the feature was 0 in the model
    # matrix" — semantically equivalent to "no observation yet".
    weekly_features[["nlp_decay_score", "graph_shock_score"]] = (
        weekly_features[["nlp_decay_score", "graph_shock_score"]].fillna(0.0)
    )

    # ----- 4. Compute idiosyncratic-return targets (weekly). -----
    idio_df = compute_residual_returns_rolling(
        weekly, weekly_sector_ret,
        ticker_column="ticker",
        sector_column="sector",
        date_column="date",
        raw_return_column="raw_ret",
        window=8,
    )

    # ----- 5. Build training matrix: features_t -> forward 1-week idio. -----
    training = build_training_matrix(
        weekly_features.fillna(0.0),
        idio_df.fillna(0.0),
        feature_columns=("nlp_decay_score", "graph_shock_score"),
        horizon=1,
        target_column="idiox_ret",
    )
    training = training.dropna(subset=["target"]).reset_index()

    # ----- 6. Purged-CV alpha model fit. -----
    alpha_model = AlphaModel(
        feature_columns=("nlp_decay_score", "graph_shock_score"),
        ridge_alpha=5.0,
        n_splits=4,
        gap_groups=1,
        random_state=42,
    )
    alpha_model.fit(training, date_column="date")

    # ----- 7. Weekly portfolio loop. -----
    constraints = PortfolioConstraints(
        gross_exposure=0.03,
        enforce_dollar_neutral=True,
        enforce_beta_neutral=True,
    )
    costs = TransactionCostConfig(
        base_bps_slippage=5.0,
        short_borrow_bps_annual=20.0,
        turnover_gamma=0.5,
        daily_turnover_cap=None,
    )
    optimizer = ConvexPortfolioOptimizer(constraints=constraints, costs=costs)

    weights_records: list[dict[str, object]] = []
    pnl_records: list[dict[str, object]] = []

    universe_tickers = sorted(set(weekly_long["ticker"]).intersection(graph.nodes))
    tradeable_tickers = [
        t for t in universe_tickers if t in graph.graph.get("point_in_time_universe", universe_tickers)
    ]
    prev_weights: np.ndarray | None = None
    portfolio_returns = np.zeros(len(weekly_dates))

    for week_idx, week in enumerate(weekly_dates):
        feats_w = weekly_features.loc[(week,)].reindex(tradeable_tickers)
        # Drop tickers with missing features AFTER warmup (i.e., for which
        # the convex allocator has no signal).
        if feats_w[["nlp_decay_score", "graph_shock_score"]].isna().any(axis=1).any():
            feats_w = feats_w.fillna(0.0)
        if feats_w.empty:
            prev_weights = None
            continue
        scores = alpha_model.predict(feats_w.reset_index())

        # Sector dummy construction (column for every sector).
        sectors_for_universe = sorted(set(graph.nodes[t].get("sector", "UNKNOWN") for t in tradeable_tickers))
        sector_dummies = pd.DataFrame(
            {
                t: {s: 1.0 if graph.nodes[t].get("sector", "UNKNOWN") == s else 0.0
                    for s in sectors_for_universe}
                for t in tradeable_tickers
            }
        ).T
        sector_dummies.index.name = "ticker"

        # Pre-built alignment layer (the *firewall* the Thinker flagged).
        betas = pd.Series(
            [float(weekly.loc[(week, t), "beta"]) if (week, t) in weekly.index else 1.0
             for t in tradeable_tickers],
            index=tradeable_tickers,
            dtype=float,
        )
        layer = build_alpha_layer(
            alpha=scores,
            beta=betas,
            sector_dummies=sector_dummies,
        )
        neutral = neutralize_to_sectors(layer["alpha"], sector_dummies)

        if prev_weights is None or prev_weights.shape[0] != len(tradeable_tickers):
            prev = np.zeros(len(tradeable_tickers))
        else:
            prev = prev_weights

        # Solve convex allocator.
        w_opt = optimizer.solve(
            alpha=neutral.values,
            beta=layer["beta"].values,
            prev_weights=prev,
        )

        # Realised return for the next week:
        next_week = weekly_dates[week_idx + 1] if week_idx + 1 < len(weekly_dates) else None
        if next_week is not None:
            realised = np.array(
                [float(weekly.loc[(next_week, t), "raw_ret"]) if (next_week, t) in weekly.index else 0.0
                 for t in tradeable_tickers],
                dtype=float,
            )
            pnl = optimizer.apply_costs_to_pnl(w_opt, realised, prev_weights=prev, trading_days=1)
            portfolio_returns[week_idx] = pnl["net_pnl"]
            pnl_records.append({"date": next_week, **pnl})

        for ticker, weight in zip(tradeable_tickers, w_opt):
            weights_records.append({"date": week, "ticker": ticker, "weight": float(weight)})

        prev_weights = w_opt

    weights_df = pd.DataFrame(weights_records).set_index(["date", "ticker"]).sort_index()
    pnl_df = pd.DataFrame(pnl_records).set_index("date").sort_index()

    diagnostics = {
        "n_periods": len(weekly_dates),
        "n_tradeable_tickers": len(tradeable_tickers),
        "avg_weekly_net_pnl": float(np.nanmean(portfolio_returns)) if len(portfolio_returns) else float("nan"),
        "total_net_pnl": float(np.nansum(portfolio_returns)),
        "realised_turnover_weekly": float(
            sum(p.get("turnover", 0.0) for p in pnl_records)
            / max(len(pnl_records), 1)
        )
        if pnl_records
        else 0.0,
        "max_gross_weight": float(weights_df["weight"].abs().max()) if not weights_df.empty else 0.0,
        "dollar_neutrality_violation": float((weights_df.groupby("date")["weight"].sum()).abs().max())
        if not weights_df.empty
        else 0.0,
    }
    logger.info("Backtest complete: %s", {k: round(v, 6) if isinstance(v, float) else v for k, v in diagnostics.items()})

    return BacktestResult(weights=weights_df, weekly_pnl=pnl_df, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Vesper backtest.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory holding synthetic artefacts (created if missing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_backtest(data_dir=args.data_dir)
    # Persist a small report next to the data artefacts.
    report_path = Path(args.data_dir) / "backtest_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not result.weekly_pnl.empty:
        result.weekly_pnl.to_csv(report_path)
        logger.info("Wrote weekly PnL report to %s", report_path)
    diagnostics_path = Path(args.data_dir) / "backtest_diagnostics.txt"
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        for k, v in result.diagnostics.items():
            handle.write(f"{k} = {v}\n")
    logger.info("Wrote diagnostics to %s", diagnostics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
