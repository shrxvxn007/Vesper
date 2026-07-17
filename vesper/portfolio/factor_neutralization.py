"""Cross-sectional factor neutralization.

Removes sector exposure from a vector of raw alpha scores by regressing them
against a one-hot sector dummy matrix and taking the residual. Operates on a
single cross-section per call (one date); pass each date's alpha Series in
turn.

Typical usage::

    sector_dummies = pd.get_dummies(alpha_df["sector"]).reindex(index=alpha_df.index, fill_value=0)
    neutralized = neutralize_to_sectors(alpha_df["alpha"], sector_dummies)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_to_sectors(
    alpha: pd.Series,
    sector_dummies: pd.DataFrame,
    *,
    add_intercept: bool = True,
) -> pd.Series:
    """Subtract the OLS fit of ``alpha ~ sector_dummies`` from ``alpha``.

    Args:
        alpha: Per-ticker alpha scores indexed by ticker.
        sector_dummies: Per-ticker one-hot sector matrix aligned to ``alpha``'s
            index. Columns must not include the intercept.
        add_intercept: If ``True`` (default) include an intercept in the
            regression so the residual's mean is zero.

    Returns:
        :class:`pandas.Series` indexed by ``alpha.index`` of the same shape
        with sector effects (and intercept, if included) removed.

    Raises:
        ValueError: If ``alpha`` and ``sector_dummies`` are not aligned.
    """
    if not alpha.index.equals(sector_dummies.index):
        raise ValueError(
            f"alpha.index and sector_dummies.index must match exactly; "
            f"got len(alpha)={len(alpha)} and len(dummies)={len(sector_dummies)} with diff "
            f"{set(alpha.index) ^ set(sector_dummies.index)}"
        )

    if alpha.empty:
        return alpha.copy()

    y = alpha.to_numpy(dtype=float)
    X = sector_dummies.to_numpy(dtype=float)
    if add_intercept:
        X = np.column_stack([np.ones(len(y), dtype=float), X])

    # Solve via least squares: beta = (X'X)^-1 X'y, residual = y - X beta.
    # Use ``np.linalg.lstsq`` for numerical robustness. If X is rank-deficient
    # (because some ticker has no sector — shouldn't happen with dummy fill,
    # but defensively guard), add a small ridge term.
    rank_X = np.linalg.matrix_rank(X)
    if rank_X < X.shape[1]:
        XtX = X.T @ X + np.eye(X.shape[1]) * 1e-6
        beta = np.linalg.solve(XtX, X.T @ y)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    return pd.Series(residual, index=alpha.index, name="alpha_neutral")


__all__ = ["neutralize_to_sectors"]
