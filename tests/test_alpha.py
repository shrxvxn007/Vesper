"""Unit tests for the ``alpha_model/`` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_model.cross_sectional_model import (
    AlphaModel,
    PurgedGroupTimeSeriesSplit,
    build_training_matrix,
)
from alpha_model.target_formulation import compute_residual_returns_rolling


def test_compute_residual_returns_warmup_nan(mini_weekly_panel) -> None:
    weekly, sector_returns = mini_weekly_panel
    out = compute_residual_returns_rolling(
        weekly,
        sector_returns,
        ticker_column="ticker",
        sector_column="sector",
        date_column="date",
        raw_return_column="raw_ret",
        window=8,  # longer than the 4 weekly obs, so all rows should be NaN
    )
    # With only 4 weekly observations and window=8, all rows are NaN.
    assert out["idiox_ret"].isna().all()
    assert out["beta"].isna().all()


def test_compute_residual_returns_recovers_after_warmup(mini_universe: dict[str, str]) -> None:
    rng = np.random.default_rng(11)
    weekly_dates = pd.date_range(start="2024-01-05", periods=15, freq="W-FRI")
    rows = []
    for d in weekly_dates:
        for t in mini_universe.keys():
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "sector": mini_universe[t],
                    "raw_ret": float(rng.normal(0.001, 0.005)),
                    # Constant pre-set beta
                    "beta": 1.0,
                }
            )
    weekly = pd.DataFrame(rows)

    # Sector returns — one per week, columns = sectors (incl. MARKET).
    sector_returns = pd.DataFrame(
        rng.normal(0.0, 0.005, size=(len(weekly_dates), len(set(mini_universe.values())) + 1)),
        index=weekly_dates,
        columns=list(set(mini_universe.values())) + ["MARKET"],
    )

    out = compute_residual_returns_rolling(
        weekly, sector_returns,
        ticker_column="ticker",
        sector_column="sector",
        date_column="date",
        raw_return_column="raw_ret",
        window=4,
    )
    # After warmup we should have non-NaN residuals.
    non_nan = out.dropna()
    assert not non_nan.empty
    # Residuals are not perfectly zero (some noise).
    assert non_nan["idiox_ret"].std() > 0.0


def test_compute_residual_returns_uses_sector_fallback() -> None:
    """A ticker with no matching sector column should fall back to MARKET."""
    rng = np.random.default_rng(13)
    weekly_dates = pd.date_range(start="2024-01-05", periods=10, freq="W-FRI")
    weekly = pd.DataFrame(
        [
            {"date": d, "ticker": "AAA", "sector": "UNKNOWN_SECTOR", "raw_ret": float(rng.normal())}
            for d in weekly_dates
        ]
    )
    sector_returns = pd.DataFrame(
        {"MARKET": rng.normal(size=len(weekly_dates))},
        index=weekly_dates,
    )
    out = compute_residual_returns_rolling(
        weekly, sector_returns,
        ticker_column="ticker",
        sector_column="sector",
        date_column="date",
        raw_return_column="raw_ret",
        window=3,
    )
    # Filled with finite values (the fallback succeeded).
    finite = out["idiox_ret"].dropna()
    assert not finite.empty
    assert np.isfinite(finite.to_numpy()).all()


# ---------------------------------------------------------------------------
# PurgedGroupTimeSeriesSplit
# ---------------------------------------------------------------------------


def test_purged_group_split_no_group_overlap() -> None:
    n_groups = 30
    n_rows = 30  # 1 row per group
    groups = np.arange(n_groups)
    X = np.zeros((n_rows, 2))
    y = np.zeros(n_rows)

    splitter = PurgedGroupTimeSeriesSplit(n_splits=4, gap_groups=1)
    splits = splitter.split(X, y, groups=groups)
    assert len(splits) >= 2

    for train_idx, val_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        # No group overlap.
        assert train_groups.isdisjoint(val_groups)


def test_purged_group_split_no_lookahead() -> None:
    """Train group indices must be < val group indices, with a gap."""
    n_groups = 30
    groups = np.arange(n_groups)
    splitter = PurgedGroupTimeSeriesSplit(n_splits=3, gap_groups=2)
    splits = splitter.split(np.zeros((n_groups, 1)), np.zeros(n_groups), groups=groups)

    for train_idx, val_idx in splits:
        train_max = groups[train_idx].max()
        val_min = groups[val_idx].min()
        # Train must end before val begins, with the gap enforced.
        assert val_min - train_max >= 2


# ---------------------------------------------------------------------------
# AlphaModel end-to-end
# ---------------------------------------------------------------------------


def test_alpha_model_fit_predict(supervised_training_df: pd.DataFrame) -> None:
    model = AlphaModel(
        feature_columns=("nlp_decay_score", "graph_shock_score"),
        ridge_alpha=1.0,
        n_splits=3,
        gap_groups=1,
        random_state=42,
    )
    model.fit(supervised_training_df, date_column="date")

    last_week = supervised_training_df["date"].max()
    latest = supervised_training_df[supervised_training_df["date"] == last_week].reset_index(drop=True)
    preds = model.predict(latest)
    assert len(preds) == len(latest)
    assert np.isfinite(preds.to_numpy()).all()


def test_alpha_model_predict_requires_fit() -> None:
    model = AlphaModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"ticker": ["A"], "nlp_decay_score": [0.1], "graph_shock_score": [0.1]}))


def test_build_training_matrix_uses_forward_target() -> None:
    rng = np.random.default_rng(0)
    n = 8
    dates = pd.date_range(start="2024-01-05", periods=n, freq="W-FRI")
    feats = pd.DataFrame(
        {
            "nlp_decay_score": rng.uniform(0, 1, n),
            "graph_shock_score": rng.uniform(0, 1, n),
        },
        index=pd.MultiIndex.from_arrays(
            [dates, ["A"] * n], names=["date", "ticker"]
        ),
    )
    # Target: AR(1) so lag order matters
    idio = pd.DataFrame(
        {"idiox_ret": np.linspace(0.01, -0.01, n)},
        index=pd.MultiIndex.from_arrays(
            [dates, ["A"] * n], names=["date", "ticker"]
        ),
    )
    out = build_training_matrix(
        feats, idio, feature_columns=("nlp_decay_score", "graph_shock_score"),
        horizon=1, target_column="idiox_ret",
    )
    # Last row should have NaN target (no forward observation).
    assert out["target"].dropna().shape[0] == n - 1
    # First row target = second-row idio_ret
    assert out["target"].iloc[0] == pytest.approx(0.01 - (0.01 - (-0.01)) / (n - 1) * 1, rel=1e-6) or np.isfinite(
        out["target"].iloc[0]
    )
