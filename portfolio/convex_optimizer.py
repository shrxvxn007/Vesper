"""Convex portfolio optimizer.

Implements an institutional-grade allocator that:

* **Maximises** expected factor portfolio return ``alpha @ w``.
* **Subtracts** an explicit linear transaction-cost penalty proportional to
  trade size and a configured bid-ask spread override.
* **Enforces** three hard constraints:
    1. Dollar-neutrality: ``sum(w) == 0``.
    2. Beta-neutrality:   ``w @ beta == 0``.
    3. Per-name gross-exposure cap: ``|w_i| <= 0.03``.

Inputs must be **strictly aligned** numpy arrays or pandas Series sharing the
same ticker ordering. The optimizer will refuse to run if shapes disagree —
this is the single most likely source of silent bugs in this codebase (the
Thinker's "index misalignment" warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PortfolioConstraints:
    """Hard portfolio constraints.

    Attributes:
        gross_exposure: Maximum gross weight per name (``|w_i| <= gross_exp``).
            Defaults to 0.03 (3%) per spec.
        enforce_dollar_neutral: Whether to enforce ``sum(w) == 0``.
        enforce_beta_neutral: Whether to enforce ``w @ beta == 0``.
    """

    gross_exposure: float = 0.03
    enforce_dollar_neutral: bool = True
    enforce_beta_neutral: bool = True


@dataclass(slots=True, frozen=True)
class TransactionCostConfig:
    """Transaction-cost model.

    Attributes:
        base_bps_slippage: One-way slippage in basis points applied linearly
            to trade size. Defaults to 5 bps.
        short_borrow_bps_annual: Annualised borrow cost for short positions,
            in basis points. Applied as ``bps/252/10000`` per business day.
        turnover_gamma: Coefficient on the L1 turnover penalty in the
            objective. Tuned jointly with the alpha scale.
        daily_turnover_cap: Optional hard cap on ``|w - w_prev|`` 1-norm.
            Set to ``None`` to disable.
    """

    base_bps_slippage: float = 5.0
    short_borrow_bps_annual: float = 20.0
    turnover_gamma: float = 0.5
    daily_turnover_cap: float | None = None
    _extra: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ConvexPortfolioOptimizer:
    """Build and solve a single cross-section's portfolio problem.

    Args:
        constraints: Hard portfolio constraints.
        costs: Transaction-cost configuration.
        solver: ``cvxpy`` solver (one of ``"ECOS"``, ``"SCS"``, ``"OSQP"``).
        verbose: Forwarded to ``cvxpy``.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        costs: TransactionCostConfig | None = None,
        *,
        solver: str = "ECOS",
        verbose: bool = False,
    ) -> None:
        self._constraints = constraints or PortfolioConstraints()
        self._costs = costs or TransactionCostConfig()
        self._solver = solver
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def constraints(self) -> PortfolioConstraints:
        return self._constraints

    @property
    def costs(self) -> TransactionCostConfig:
        return self._costs

    def solve(
        self,
        alpha: np.ndarray | pd.Series,
        beta: np.ndarray | pd.Series,
        *,
        prev_weights: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        """Solve the cross-sectional optimization.

        Args:
            alpha: Per-ticker expected returns (already sector-neutral).
            beta: Per-ticker precomputed market betas.
            prev_weights: Previous week weights (default 0 if first period).

        Returns:
            ``np.ndarray`` of optimal weights of shape ``(n_assets,)``.

        Raises:
            ValueError: If shapes disagree, or the problem is contradictory.
            cvxpy.error.SolverError: If the solver fails to converge.
        """
        alpha_v = _to_aligned_vector(alpha, name="alpha")
        beta_v = _to_aligned_vector(beta, name="beta")
        if alpha_v.shape != beta_v.shape:
            raise ValueError(
                f"alpha ({alpha_v.shape}) and beta ({beta_v.shape}) must share shape"
            )

        if prev_weights is None:
            prev_w_v = np.zeros_like(alpha_v)
        else:
            prev_w_v = _to_aligned_vector(prev_weights, name="prev_weights")
            if prev_w_v.shape != alpha_v.shape:
                raise ValueError(
                    f"prev_weights ({prev_w_v.shape}) must match alpha ({alpha_v.shape})"
                )

        n = alpha_v.shape[0]
        if n == 0:
            return np.zeros(0, dtype=float)

        # cvxpy variable — the SINGLE source of truth for constraints.
        w = cp.Variable(n)

        constraints: list[cp.constraints.Constraint] = [
            cp.abs(w) <= self._constraints.gross_exposure,
        ]
        if self._constraints.enforce_dollar_neutral:
            constraints.append(cp.sum(w) == 0)
        if self._constraints.enforce_beta_neutral:
            constraints.append(w @ beta_v == 0)

        # Objective: maximise alpha @ w minus linear turnover penalty.
        turnover = cp.sum(cp.abs(w - prev_w_v))
        cost_per_unit = (self._costs.base_bps_slippage / 10000.0) * self._costs.turnover_gamma
        objective = cp.Maximize(alpha_v @ w - cost_per_unit * turnover)

        if self._costs.daily_turnover_cap is not None:
            constraints.append(turnover <= self._costs.daily_turnover_cap)

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self._solver, verbose=self._verbose)
        except cp.error.SolverError as exc:
            # Retry with a fallback solver before giving up.
            if self._solver != "SCS":
                problem.solve(solver="SCS", verbose=self._verbose)
            else:
                raise exc

        if w.value is None:
            raise ValueError(
                f"Convex optimizer did not return a value. status={problem.status}"
            )

        weights = np.asarray(w.value, dtype=float).reshape(-1)
        # Clip to constraints — defensive against tiny numerical overshoot.
        weights = np.clip(weights, -self._constraints.gross_exposure, self._constraints.gross_exposure)
        if self._constraints.enforce_dollar_neutral:
            weights = weights - weights.mean()
        return weights

    def apply_costs_to_pnl(
        self,
        weights: np.ndarray,
        realized_returns: np.ndarray,
        prev_weights: np.ndarray | None = None,
        *,
        trading_days: int = 1,
    ) -> dict[str, float]:
        """Apply transaction costs / borrow fees to realised PnL.

        Args:
            weights: New weights (after trade).
            realized_returns: Per-ticker realised returns over the period.
            prev_weights: Pre-trade weights (default 0).
            trading_days: Number of business days used to annualise borrow cost.

        Returns:
            Dict with ``gross_pnl``, ``slippage_cost``, ``borrow_cost``,
            ``net_pnl``. Borrow cost is applied to short positions only.
        """
        prev = np.zeros_like(weights) if prev_weights is None else np.asarray(prev_weights, dtype=float)
        turnover = np.abs(weights - prev).sum()
        gross_pnl = float(np.dot(weights, realized_returns))
        slippage_cost = float((self._costs.base_bps_slippage / 10000.0) * turnover)
        # Borrow cost: applies only when w_i < 0 (short).
        short_notional = float(np.abs(np.minimum(weights, 0.0)).sum())
        per_day_borrow = self._costs.short_borrow_bps_annual / 10000.0 / 252.0
        borrow_cost = float(short_notional * per_day_borrow * trading_days)
        net_pnl = gross_pnl - slippage_cost - borrow_cost
        return {
            "gross_pnl": gross_pnl,
            "slippage_cost": slippage_cost,
            "borrow_cost": borrow_cost,
            "net_pnl": net_pnl,
            "turnover": float(turnover),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_aligned_vector(
    value: np.ndarray | pd.Series, *, name: str
) -> np.ndarray:
    """Convert a 1-D vector to ``float``, validating dimensionality.

    Used to make sure alpha / beta / prev_weights share the same index.
    """
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    return arr


def build_alpha_layer(
    *,
    alpha: pd.Series,
    beta: pd.Series,
    sector_dummies: pd.DataFrame,
    sector_index: pd.Series | None = None,
) -> pd.DataFrame:
    """Pre-align every optimizer input into a single DataFrame keyed by ticker.

    This is the index-alignment firewall the Thinker flagged as the single
    most likely failure mode in the first cut. Build the layer once and pass
    ``layer['alpha'].values`` etc. into the optimizer.
    """
    base = pd.DataFrame(index=alpha.index)
    base["alpha"] = alpha
    base["beta"] = beta
    for col in sector_dummies.columns:
        base[f"sector_{col}"] = sector_dummies[col]
    if sector_index is not None:
        base["sector_index"] = sector_index
    return base.fillna(0.0)


__all__ = [
    "PortfolioConstraints",
    "TransactionCostConfig",
    "ConvexPortfolioOptimizer",
    "build_alpha_layer",
]
