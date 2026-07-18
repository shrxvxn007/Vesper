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

    python -m vesper.main --data-dir data

This script is deliberately *self-contained*: it does not need internet
access. The synthetic data generator produces the same artefacts every
run, so the backtest is deterministic.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Project modules
from vesper.data_pipeline.graph_builder import build_supply_chain_graph_from_json
from vesper.data_pipeline.mda_parser import clean_text, extract_mda_section
from vesper.data_pipeline.sec_scraper import SECScraper, SECScraperError
from vesper.features.nlp_decay import compute_information_decay
from vesper.features.shock_propagation import propagate_shock_scores
from vesper.alpha_model.cross_sectional_model import AlphaModel, build_training_matrix
from vesper.alpha_model.target_formulation import compute_residual_returns_rolling
from vesper.portfolio.convex_optimizer import (
    ConvexPortfolioOptimizer,
    PortfolioConstraints,
    TransactionCostConfig,
    build_alpha_layer,
)
from vesper.portfolio.factor_neutralization import neutralize_to_sectors
from vesper.scripts.synthetic_generator import generate_synthetic_dataset

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
    # Per-(date, ticker) NLP decay + graph shock features actually consumed
    # by the alpha model. Exposed so downstream notebooks can correlate
    # signals with weights / PnL without recomputing the pipeline.
    weekly_features: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Live-EDGAR plumbing
# ---------------------------------------------------------------------------

# 10-digit zero-padded SEC CIKs for the 12-ticker universe. These map the
# synthetic ticker labels to real EDGAR issuers. Confidence: high — these are
# the standard EDGAR records for the named large-cap tech issuers.
TICKER_TO_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AVGO": "0001730168",
    "ORCL": "0001341439",
    "CRM":  "0001108524",
    "ADBE": "0000796343",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "INTC": "0000050863",
}

PLACEHOLDER_USER_AGENT: str = "Vesper Research research@example.com"
DEFAULT_USER_AGENT_ENV: str = "VESPER_SEC_USER_AGENT"
DEFAULT_REAL_EDGAR_CACHE: str = "real_edgar"
REAL_EDGAR_CACHE_TTL_SECONDS: int = 24 * 60 * 60  # 24 hours


def _resolve_user_agent(env_var: str = DEFAULT_USER_AGENT_ENV) -> str:
    """Return the SEC User-Agent, sourcing it from ``env_var`` if set.

    Args:
        env_var: Name of the environment variable to read first.

    Returns:
        User-Agent string. If unset or set to the placeholder, logs a loud
        warning — SEC's strict fair-access policy will reject placeholder
        user agents with HTTP 403.
    """
    ua = os.environ.get(env_var, "").strip() or PLACEHOLDER_USER_AGENT
    if ua == PLACEHOLDER_USER_AGENT:
        logger.warning(
            "SEC User-Agent is the placeholder %r. Set the %s environment "
            "variable to a string of the form "
            "'Company Name AdminContact@<your-domain>' before --real-edgar "
            "requests will succeed. EDGAR will respond 403 otherwise.",
            PLACEHOLDER_USER_AGENT,
            env_var,
        )
    return ua


def pull_real_filings(
    scraper: SECScraper,
    cache_dir: str | Path,
    *,
    max_filings_per_ticker: int = 8,
    cache_ttl_seconds: int = REAL_EDGAR_CACHE_TTL_SECONDS,
) -> pd.DataFrame:
    """Pull real EDGAR filings for the 12-ticker universe and cache them.

    This function:
    1. Reuses ``cache_dir / "filings_metadata.parquet"`` if it exists and
       its mtime is younger than ``cache_ttl_seconds``.
    2. Otherwise, fetches metadata from EDGAR for each ticker in
       :data:`TICKER_TO_CIK` with graceful per-ticker failure.
    3. Hydrates the MD&A text of any filing whose ``accession_number`` is
       not already cached at ``cache_dir / "texts" / "<accession>.txt"``.

    Args:
        scraper: A configured :class:`SECScraper` with ``allow_online=True``.
        cache_dir: Local cache root (typically ``data/real_edgar``).
        max_filings_per_ticker: Per-ticker cap on filings to fetch.
        cache_ttl_seconds: How long the metadata cache is considered fresh.

    Returns:
        :class:`pandas.DataFrame` with one row per filing. Columns:
        ``ticker``, ``form_type``, ``release_date``, ``period_end_date``,
        ``accession_number``, ``url``, ``text_clean``,
        ``is_synthetic_shock``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_dir = cache_dir / "texts"
    text_dir.mkdir(exist_ok=True)
    meta_path = cache_dir / "filings_metadata.parquet"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")

    use_cache = (
        meta_path.exists()
        and (pd.Timestamp.now().timestamp() - meta_path.stat().st_mtime) < cache_ttl_seconds
    )

    if use_cache:
        logger.info("Loading EDGAR metadata cache from %s", meta_path)
        meta = pd.read_parquet(meta_path)
    else:
        logger.info("Fetching EDGAR metadata for %d tickers", len(TICKER_TO_CIK))
        meta_frames: list[pd.DataFrame] = []
        for ticker, cik in TICKER_TO_CIK.items():
            try:
                df = scraper.fetch_recent_filings(
                    cik, form_types=("10-Q", "10-K"), limit=max_filings_per_ticker
                )
            except SECScraperError as exc:
                logger.warning("EDGAR fetch failed for %s (CIK %s): %s", ticker, cik, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - last-line defence
                logger.warning("Unexpected EDGAR error for %s: %s", ticker, exc)
                continue
            if df.empty:
                logger.info("No filings returned for %s (CIK %s)", ticker, cik)
                continue
            df = df.reset_index()
            df["ticker"] = ticker
            meta_frames.append(df)

        if not meta_frames:
            meta = pd.DataFrame(
                columns=[
                    "ticker", "cik", "form_type", "filing_date",
                    "period_end_date", "accession_number",
                    "primary_document", "url",
                ]
            )
        else:
            meta = pd.concat(meta_frames, ignore_index=True)

        # Atomic write so partial failures don't corrupt the cache.
        try:
            meta.to_parquet(meta_tmp)
            os.replace(meta_tmp, meta_path)
            logger.info("Wrote EDGAR metadata cache to %s", meta_path)
        except Exception as exc:  # noqa: BLE001 - last-line defence
            logger.warning("Failed to persist EDGAR metadata cache: %s", exc)
            if meta_tmp.exists():
                try:
                    meta_tmp.unlink()
                except OSError:
                    pass

    # ---- Hydrate per-filing text (cached per-accession) ----
    # ``itertuples()`` is the project-approved alternative to ``iterrows()``
    # for row-level side-effect loops (faster, no per-row Series wrapping).
    texts: list[str] = [
        _hydrate_one_filing(row, scraper, text_dir) for row in meta.itertuples()
    ] if len(meta) > 0 else []
    # ``texts`` stays aligned with ``meta`` even when ``meta`` is empty
    # (the early empty-cache branch above sets columns with no rows).

    meta["text_clean"] = texts
    meta["is_synthetic_shock"] = False

    # Normalise ``filing_date`` -> ``release_date`` to match the synthetic
    # filings schema so downstream ``compute_information_decay`` sees the
    # same column it expects.
    if "release_date" not in meta.columns and "filing_date" in meta.columns:
        meta = meta.rename(columns={"filing_date": "release_date"})

    return meta


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


def _week_shock_records(
    week: pd.Timestamp,
    propagate: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build graph-shock records for one week, defensively discovering the
    score column name.

    ``propagate_shock_scores`` returns a DataFrame indexed by ticker
    (whose ``index.name`` is conventionally ``"ticker"`` but the values
    flow through regardless of that label) with the propagated score as
    the *last* column by positional contract. Instead of hardcoding
    ``propagate["graph_shock_score"]``, this helper reads the rightmost
    column name dynamically so a benign upstream rename in
    ``propagate_shock_scores`` (e.g., ``graph_shock_score`` ->
    ``transmitted_risk``) no longer silently raises a ``KeyError`` here.

    The output ``ticker`` key is always emitted under the literal name
    ``"ticker"`` regardless of the propagator's index name — downstream
    consumers expect the canonical key, and ``propagate.index.to_numpy()``
    already surfaces the ticker strings without needing the label.

    Args:
        week: The week date (timestamp) for the records.
        propagate: Output of :func:`vesper.features.shock_propagation
            .propagate_shock_scores`; expected to be indexed by ticker
            with the propagated score as the rightmost column.

    Returns:
        List of dicts, one per row of ``propagate``. Each dict carries
        ``date``, ``ticker`` (sourced from the index values), and the
        propagated-score column using the *discovered* name (last column).
        The score key mirrors the column emitted by the propagator, so a
        downstream ``pd.DataFrame`` round-trip picks up whatever name the
        propagator produced.
    """
    # Ticker values are sourced via ``propagate.index.to_numpy()`` rather
    # than a hypothetical ``propagate[ticker_name]`` access (which would
    # require a ``reset_index()`` and would silently re-introduce a
    # KeyError if upstream renames ``index.name``). The score column is
    # discovered as ``columns[-1]`` by positional contract.
    score_name = propagate.columns[-1]
    return [
        {
            "date": week,
            "ticker": str(ticker),
            score_name: float(score),
        }
        for ticker, score in zip(
            propagate.index.to_numpy(),
            propagate[score_name].to_numpy(),
        )
    ]


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

    # Sort NLP data (merge_asof requires the on-key column to be globally
    # monotonic. With ``by=ticker_col`` and ``on=date``, the ``date`` column
    # cycles per ticker (AAPL➡Dec, MSFT➡Jan); only the on-key needs global
    # monotonicity, so we sort on-column-first. Empirically validated in
    # ``/tmp/vesper_probe_merge_asof.py``.
    nlp = filings_clean[[release_column, ticker_col, feature_col]].sort_values(
        [release_column, ticker_col]
    )

    # Sort weekly data (same rationale as above).
    weekly_long = weekly_clean[[date_col, ticker_col]].drop_duplicates().sort_values(
        [date_col, ticker_col]
    )

    if weekly_long.empty:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=[date_col, ticker_col]
        )
        return pd.DataFrame(
            {feature_col: pd.Series([], dtype=float, index=empty_idx)}
        )

    # Single global backward-asof. ``by=ticker_col`` makes pandas partition the
    # asof by ticker (exact-match key) without any explicit groupby + concat.
    # Equivalent to the prior per-ticker loop on every test in
    # ``tests/test_lookahead.py``; faster and C-path.
    #
    # NOTE: the right-frame MUST keep ``ticker_col`` — ``merge_asof(by=...)``
    # requires the by-key to be present on both sides. Dropping it earlier
    # was a silent-consistency regression; see ``tests/test_lookahead.py``.
    merged = pd.merge_asof(
        weekly_long,
        nlp,
        left_on=date_col,
        right_on=release_column,
        by=ticker_col,
        direction="backward",
        allow_exact_matches=True,
    )
    out = (
        merged.set_index([date_col, ticker_col])[[feature_col]]
        .rename_axis([date_col, ticker_col])
        .sort_index()
    )
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_backtest(
    data_dir: str | Path = "data",
    *,
    external_filings: pd.DataFrame | None = None,
    model_type: Literal["ridge", "xgboost"] = "ridge",
) -> BacktestResult:
    """Run the full research backtest against synthetic artefacts.

    Args:
        data_dir: Directory holding (or to receive) synthetic artefacts.
        external_filings: Optional filings frame to use instead of
            ``data/filings.parquet``. Expected columns when provided:
            ``ticker``, ``form_type``, ``release_date``, ``period_end_date``,
            and ``text_clean``. Typically the output of
            :func:`pull_real_filings`.

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
    if external_filings is not None and not external_filings.empty:
        logger.info("Using external_filings (%d rows)", len(external_filings))
        filings_raw = external_filings.copy()
        # Defensive column normalisation: callers using ``pull_real_filings``
        # already produce ``release_date``, but other sources might not.
        if "release_date" not in filings_raw.columns and "filing_date" in filings_raw.columns:
            filings_raw = filings_raw.rename(columns={"filing_date": "release_date"})
    else:
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

    # Single global backward-asof keyed by ticker (replaces the prior
    # per-ticker groupby loop). ``merge_asof(by=...)`` partitions the asof
    # by exact-match key without an explicit groupby.
    if feature_release.empty or weekly_pairs.empty:
        empty_idx = pd.MultiIndex.from_arrays(
            [[], []], names=["date", "ticker"]
        )
        direct_df = pd.DataFrame(
            {"nlp_decay_score": pd.Series([], dtype=float, index=empty_idx)}
        )
    else:
        # ``merge_asof(by="ticker", on="date")`` requires the on-column
        # (``date``) to be globally monotonic, not the by-column. Sort
        # on-key first; ``ticker`` then partitions within each date.
        # Note: ``merge_asof`` is happy with the by-column non-monotonic
        # globally, but not with the on-column non-monotonic globally.
        left = weekly_pairs.sort_values(["date", "ticker"])
        right = feature_release.sort_values(["release_date", "ticker"])
        # ``merge_asof(by="ticker")`` requires ``ticker`` on the right frame
        # for the exact-match partition key. Slicing ``right`` to drop the
        # column silently regresses to ``KeyError: 'ticker'``.
        merged = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="release_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )
        direct_df = (
            merged
            .set_index(["date", "ticker"])
            .sort_index()
        )

    # Now propagate each week's direct shocks through the supply-chain graph.
    # ``propagate`` is indexed by ticker; we lift the index to a column once
    # and then iterate the columns as numpy arrays via ``zip`` — equivalent to
    # the previous ``propagate.iterrows()`` row walk but vectorised.
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
        # Defensive rename-discovery lives in ``_week_shock_records`` so a
        # benign upstream rename in ``propagate_shock_scores`` (e.g.,
        # ``ticker`` -> ``asset_id``, ``graph_shock_score`` ->
        # ``transmitted_risk``) no longer silently raises a ``KeyError``
        # here. See ``tests/test_main_step3_defensive.py``.
        graph_shock_records.extend(_week_shock_records(week, propagate))
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
    logger.info(
        "Fitting %s alpha model on %d training rows",
        model_type,
        len(training),
    )
    alpha_model = AlphaModel(
        feature_columns=("nlp_decay_score", "graph_shock_score"),
        model_type=model_type,
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

    return BacktestResult(
        weights=weights_df,
        weekly_pnl=pnl_df,
        diagnostics=diagnostics,
        weekly_features=weekly_features,
    )


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
    parser.add_argument(
        "--real-edgar",
        action="store_true",
        help=(
            "Replace the synthetic filings source with a real pull from SEC "
            "EDGAR. Requires a compliant User-Agent via "
            "--user-agent-env or the VESPER_SEC_USER_AGENT env var. "
            "Filings are cached under <data-dir>/real_edgar/ for 24 hours."
        ),
    )
    parser.add_argument(
        "--user-agent-env",
        type=str,
        default=DEFAULT_USER_AGENT_ENV,
        help=(
            "Name of the environment variable holding the SEC User-Agent. "
            "Defaults to VESPER_SEC_USER_AGENT. The User-Agent must be of "
            "the form 'Company Name AdminContact@<your-domain>'."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("ridge", "xgboost"),
        default="ridge",
        help=(
            "Alpha-model backend. 'ridge' (default) is StandardScaler + Ridge. "
            "'xgboost' uses shallow trees (max_depth=2) with heavy L2 "
            "regularisation (reg_lambda=10). Requires the optional xgboost>=2.0 "
            "package; passing it without xgboost installed produces a clear "
            "ImportError on startup."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    external_filings: pd.DataFrame | None = None
    if args.real_edgar:
        user_agent = _resolve_user_agent(env_var=args.user_agent_env)
        scraper = SECScraper(user_agent=user_agent, allow_online=True)
        external_filings = pull_real_filings(
            scraper,
            cache_dir=Path(args.data_dir) / DEFAULT_REAL_EDGAR_CACHE,
        )

    result = run_backtest(
        data_dir=args.data_dir,
        external_filings=external_filings,
        model_type=args.model_type,
    )
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
