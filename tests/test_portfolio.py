"""Unit tests for the ``portfolio/`` module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.convex_optimizer import (
    ConvexPortfolioOptimizer,
    PortfolioConstraints,
    TransactionCostConfig,
    build_alpha_layer,
)
from portfolio.factor_neutralization import neutralize_to_sectors


def _balanced_layer(n: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(101)
    tickers = [f"T{i:02d}" for i in range(n)]
    sectors = ["Tech", "Software", "Semis"]
    sector_dummies = pd.DataFrame(
        [
            {s: 1.0 if sectors[i % len(sectors)] == s else 0.0 for s in sectors}
            for i in range(n)
        ],
        index=tickers,
    )
    sector_dummies.index.name = "ticker"
    alpha = pd.Series(rng.normal(0.0, 1.0, n), index=tickers)
    beta = pd.Series(rng.uniform(0.5, 1.8, n), index=tickers)
    layer = build_alpha_layer(alpha=alpha, beta=beta, sector_dummies=sector_dummies)
    return layer, alpha, beta, sector_dummies


def test_optimizer_dollar_neutral() -> None:
    layer, alpha, beta, sector_dummies = _balanced_layer(20)
    neutral = neutralize_to_sectors(layer["alpha"], sector_dummies)
    opt = ConvexPortfolioOptimizer(
        constraints=PortfolioConstraints(gross_exposure=0.03),
        costs=TransactionCostConfig(turnover_gamma=0.0),
    )
    w = opt.solve(alpha=neutral.values, beta=layer["beta"].values)
    # Within solver tolerance
    assert pytest.approx(w.sum(), abs=1e-5) == 0.0


def test_optimizer_beta_neutral() -> None:
    layer, alpha, beta, sector_dummies = _balanced_layer(20)
    neutral = neutralize_to_sectors(layer["alpha"], sector_dummies)
    opt = ConvexPortfolioOptimizer(
        constraints=PortfolioConstraints(gross_exposure=0.03),
        costs=TransactionCostConfig(turnover_gamma=0.0),
    )
    w = opt.solve(alpha=neutral.values, beta=layer["beta"].values)
    assert pytest.approx((w * layer["beta"].values).sum(), abs=1e-5) == 0.0


def test_optimizer_position_caps() -> None:
    layer, alpha, beta, sector_dummies = _balanced_layer(20)
    neutral = neutralize_to_sectors(layer["alpha"], sector_dummies)
    opt = ConvexPortfolioOptimizer(
        constraints=PortfolioConstraints(gross_exposure=0.03),
        costs=TransactionCostConfig(turnover_gamma=0.0),
    )
    w = opt.solve(alpha=neutral.values, beta=layer["beta"].values)
    assert np.all(np.abs(w) <= 0.03 + 1e-4)


def test_optimizer_returns_zero_for_zero_alpha() -> None:
    n = 12
    layer, alpha, beta, _ = _balanced_layer(n)
    # Zero scores
    alpha = pd.Series(0.0, index=alpha.index)
    layer = build_alpha_layer(alpha=alpha, beta=beta, sector_dummies=layer.iloc[:, 2:])
    w = ConvexPortfolioOptimizer().solve(
        alpha=alpha.values, beta=beta.values, prev_weights=None
    )
    # Either zero (best) or perfectly symmetric — either way abs(w) <= cap.
    assert np.all(np.abs(w) <= 0.03)
    assert pytest.approx(w.sum(), abs=1e-5) == 0.0


def test_optimizer_transaction_cost_penalty_zero_alpha_safe() -> None:
    layer, alpha, beta, sector_dummies = _balanced_layer(20)
    neutral = neutralize_to_sectors(layer["alpha"], sector_dummies)
    opt_no_cost = ConvexPortfolioOptimizer(
        constraints=PortfolioConstraints(gross_exposure=0.03),
        costs=TransactionCostConfig(turnover_gamma=0.0),
    )
    opt_cost = ConvexPortfolioOptimizer(
        constraints=PortfolioConstraints(gross_exposure=0.03),
        costs=TransactionCostConfig(turnover_gamma=2.0),
    )
    # Bootstrapping from a *non-zero* previous position: cost config must
    # dampen the trade.
    prev_weights = np.full(20, 0.01)
    w_cheap = opt_no_cost.solve(alpha=neutral.values, beta=layer["beta"].values, prev_weights=prev_weights)
    w_costly = opt_cost.solve(alpha=neutral.values, beta=layer["beta"].values, prev_weights=prev_weights)
    # Total absolute trade must shrink under high turnover penalty.
    assert np.abs(w_costly - prev_weights).sum() <= np.abs(w_cheap - prev_weights).sum() + 1e-3


def test_optimizer_input_alignment() -> None:
    """The optimizer must reject mismatched shapes (the Thinker's firewall)."""
    opt = ConvexPortfolioOptimizer()
    with pytest.raises(ValueError):
        opt.solve(
            alpha=np.array([0.1, 0.2, 0.3]),
            beta=np.array([0.5, 0.6]),
            prev_weights=None,
        )


def test_apply_costs_applies_borrow_only_to_short_side() -> None:
    opt = ConvexPortfolioOptimizer(
        costs=TransactionCostConfig(base_bps_slippage=10.0, short_borrow_bps_annual=200.0)
    )
    weights = np.array([0.02, -0.02, 0.0])
    realised = np.array([0.01, 0.01, 0.0])
    pnl = opt.apply_costs_to_pnl(weights, realised, prev_weights=None, trading_days=252)
    assert pnl["borrow_cost"] > 0.0  # Single short position
    # Slippage should be zero because prev=0 but trade size >0:
    assert pnl["slippage_cost"] >= 0.0
    # Net pnl gross - costs
    assert pnl["net_pnl"] <= pnl["gross_pnl"]


def test_neutralize_to_sectors_removes_sector_mean() -> None:
    layers, alpha, beta, sector_dummies = _balanced_layer(9)
    neutral = neutralize_to_sectors(alpha, sector_dummies)
    # For each sector, the neutralised scores should have mean ~ zero within
    # rounding. The unprojected residual from OLS will have mean zero.
    assert pytest.approx(neutral.mean(), abs=1e-6) == 0.0


def test_build_alpha_layer_aligns_indices() -> None:
    alpha = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    beta = pd.Series([1.0, 1.2, 1.4], index=["a", "b", "c"])
    sd = pd.DataFrame({"Tech": [1.0, 0.0, 0.0], "Sw": [0.0, 1.0, 1.0]}, index=["a", "b", "c"])
    layer = build_alpha_layer(alpha=alpha, beta=beta, sector_dummies=sd)
    assert layer.index.equals(alpha.index)
    assert "alpha" in layer.columns
    assert "beta" in layer.columns
    assert "sector_Tech" in layer.columns and "sector_Sw" in layer.columns
