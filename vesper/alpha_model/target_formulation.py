"""Idiosyncratic return target formulation.

The ML target is the **idiosyncratic** component of weekly stock returns —
the residual from regressing raw stock returns against a sector-equivalent
index return over a rolling historical window. We deliberately avoid raw
returns as targets because they embed market/sector common factors.

Mathematically, for each ticker ``i`` on each date ``t``::

    r_{i,t} = alpha_i + beta_i * S_{g(i),t} + eps_{i,t}

where ``S_{g(i), t}`` is the sector-index return for ticker ``i``'s sector
``g(i)``. We estimate ``(alpha_i, beta_i)`` by OLS using the trailing
``window`` observations and take ``eps = r - alpha - beta*S`` as our target.

Implementation notes:
    * Sector returns are pivoted to long form keyed by ``(date, sector)``
      and merged on ``(date, sector)`` so we never lose row alignment.
    * OLS coefficients use sample variance/covariance (``ddof=1``) so they
      match ``sklearn``/``statsmodels``.
    * Per ticker we use the rolling-sum closed-form OLS: rolling sums of
      ``r``, ``s``, ``r*s``, and ``s*s`` give us ``var``, ``cov``, and the
      resulting ``alpha``/``beta`` in a fully vectorised way without the
      ``rolling.apply`` quirks (which has historically had subtle issues
      with how it interprets the closure's argument shape).
    * During the first ``window-1`` observations for each ticker the
      residual, beta, and intercept are all ``NaN`` — we *drop* these rows
      rather than substituting fabricated values.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

DEFAULT_ROLLING_WINDOW: Final[int] = 60  # daily observations
DEFAULT_BROAD_SECTOR_FALLBACK: Final[str] = "MARKET"
_EPS_VAR: Final[float] = 1e-12  # floor on sector variance to guard divide-by-zero


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_residual_returns_rolling(
    returns_df: pd.DataFrame,
    sector_returns_df: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    sector_column: str = "sector",
    date_column: str = "date",
    raw_return_column: str = "raw_ret",
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    """Compute ticker-level rolling-OLS residual returns.

    Args:
        returns_df: Long-form panel (one row per ``(date, ticker)``) with
            columns ``date_column``, ``ticker_column``, ``sector_column``,
            and ``raw_return_column``. May also accept wide-form (columns =
            tickers) but only the long-form path is documented.
        sector_returns_df: Sector returns panel. Two accepted layouts:
            ``(A)`` long-form DataFrame with columns ``date_column``,
            ``sector``, and ``sector_ret``; ``(B)`` wide-form indexed by
            ``date_column`` with one column per sector.
        ticker_column, sector_column, date_column, raw_return_column:
            Column name overrides.
        window: Rolling window length (in observations). Defaults to 60.

    Returns:
        :class:`pandas.DataFrame` indexed by ``(date_column, ticker_column)``
        with columns ``idiox_ret``, ``beta``, and ``alpha_intercept``. Rows
        inside the warmup window are ``NaN``.
    """
    long_returns = _ensure_long_form(
        returns_df,
        ticker_column=ticker_column,
        sector_column=sector_column,
        date_column=date_column,
        value_column=raw_return_column,
    )
    sector_long = _ensure_sector_long(sector_returns_df, date_column=date_column)

    merged = long_returns.merge(
        sector_long,
        on=[date_column, sector_column],
        how="left",
    )

    # Fallback: if a ticker's sector isn't present in ``sector_returns_df``,
    # borrow the ``MARKET`` sector return for the same date.
    if DEFAULT_BROAD_SECTOR_FALLBACK in set(sector_long[sector_column]):
        market_ret_series = (
            sector_long.loc[sector_long[sector_column] == DEFAULT_BROAD_SECTOR_FALLBACK]
            .set_index(date_column)["sector_ret"]
        )
        nan_mask = merged["sector_ret"].isna()
        if nan_mask.any():
            merged.loc[nan_mask, "sector_ret"] = merged.loc[nan_mask, date_column].map(
                market_ret_series
            )

    # Per-ticker rolling OLS via closed-form rolling sums. ``groupby`` over
    # tickers is the only iteration; everything else below is pandas-vectorised.
    out_records: list[pd.DataFrame] = []
    for _, group in merged.groupby(ticker_column, sort=False):
        sub = group.sort_values(date_column).reset_index(drop=True).copy()
        idio, beta_col, alpha_col = _vectorised_ols_window(
            sub[raw_return_column].to_numpy(dtype=float),
            sub["sector_ret"].to_numpy(dtype=float),
            window=window,
        )
        sub["idiox_ret"] = idio
        sub["beta"] = beta_col
        sub["alpha_intercept"] = alpha_col
        out_records.append(
            sub[[date_column, ticker_column, "idiox_ret", "beta", "alpha_intercept"]]
        )

    out = pd.concat(out_records, ignore_index=True)
    return out.set_index([date_column, ticker_column]).sort_index()


# Backward-compatible alias.
def compute_idiosyncratic_returns(
    returns_df: pd.DataFrame,
    sector_returns_df: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    sector_column: str = "sector",
    date_column: str = "date",
    raw_return_column: str = "raw_ret",
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    return compute_residual_returns_rolling(
        returns_df,
        sector_returns_df,
        ticker_column=ticker_column,
        sector_column=sector_column,
        date_column=date_column,
        raw_return_column=raw_return_column,
        window=window,
    )


# ---------------------------------------------------------------------------
# Vectorised OLS via rolling sums
# ---------------------------------------------------------------------------


def _vectorised_ols_window(
    r: np.ndarray, s: np.ndarray, *, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(idiox_ret, beta, alpha)`` using rolling-sum OLS algebra.

    Args:
        r: 1-D array of raw returns for one ticker.
        s: 1-D array of sector returns for one ticker, aligned to ``r``.
        window: Sliding-window length.

    Returns:
        Tuple ``(idio_ret, beta, alpha)`` of 1-D arrays aligned to ``r``.
        Warmup indices are filled with NaN.
    """
    n = r.shape[0]
    idio = np.full(n, np.nan, dtype=float)
    beta = np.full(n, np.nan, dtype=float)
    alpha = np.full(n, np.nan, dtype=float)
    if n < window:
        return idio, beta, alpha

    # Rolling window sums via pandas Series (vectorised; uses C path).
    r_s = pd.Series(r)
    s_s = pd.Series(s)
    sum_r = r_s.rolling(window=window, min_periods=window).sum().to_numpy()
    sum_s = s_s.rolling(window=window, min_periods=window).sum().to_numpy()
    sum_rs = (r_s * s_s).rolling(window=window, min_periods=window).sum().to_numpy()
    sum_ss = (s_s * s_s).rolling(window=window, min_periods=window).sum().to_numpy()

    # Sample variance/covariance (ddof=1).
    valid_sum = ~np.isnan(sum_r)
    # mean_s_valid
    mean_r = np.where(valid_sum, sum_r / window, np.nan)
    mean_s = np.where(valid_sum, sum_s / window, np.nan)
    var_s = np.where(valid_sum, (sum_ss - sum_s * sum_s / window) / (window - 1), np.nan)
    cov = np.where(valid_sum, (sum_rs - sum_r * sum_s / window) / (window - 1), np.nan)

    # Mask: warmup OK AND variance passes the floor.
    has_var = valid_sum & np.isfinite(var_s) & (var_s > _EPS_VAR)
    if not has_var.any():
        return idio, beta, alpha

    beta_v = cov / var_s
    alpha_v = mean_r - beta_v * mean_s

    # Residual: at index t use r[t] - alpha[t] - beta[t] * s[t]. The
    # rolling sums at index t use r[t-W+1:t+1] for slope estimates, and
    # the trailing element of that window is r[t], so the *point estimate*
    # at index t is r[t] - alpha[t] - beta[t] * s[t]. Using r[has_var]
    # indexes directly with the boolean mask (rather than by rolling
    # position) gives exactly the right assignment.
    idio_v = r - alpha_v - beta_v * s

    idio[has_var] = idio_v[has_var]
    beta[has_var] = beta_v[has_var]
    alpha[has_var] = alpha_v[has_var]
    return idio, beta, alpha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_long_form(
    df: pd.DataFrame,
    *,
    ticker_column: str,
    sector_column: str,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Reshape to long form (one row per ticker per date)."""
    if ticker_column in df.columns:
        return df[
            [date_column, ticker_column, sector_column, value_column]
        ].copy()
    wide = df.copy()
    if date_column in wide.index.names:
        wide = wide.reset_index()
    long = wide.melt(
        id_vars=[c for c in [date_column, sector_column] if c in wide.columns],
        var_name=ticker_column,
        value_name=value_column,
    )
    if sector_column not in long.columns:
        long[sector_column] = DEFAULT_BROAD_SECTOR_FALLBACK
    return long


def _ensure_sector_long(
    sector_df: pd.DataFrame, *, date_column: str
) -> pd.DataFrame:
    """Reshape sector returns to ``(date, sector, sector_ret)`` long form."""
    if {"sector", "sector_ret"}.issubset(sector_df.columns):
        out = sector_df.copy()
        if date_column not in out.columns:
            out = out.rename(columns={out.columns[0]: date_column})
        return out[[date_column, "sector", "sector_ret"]].copy()

    if date_column in sector_df.columns:
        wide = sector_df.set_index(date_column)
    else:
        wide = sector_df.copy()
    long = wide.reset_index().melt(
        id_vars=[wide.index.name or "index"],
        var_name="sector",
        value_name="sector_ret",
    )
    long = long.rename(columns={wide.index.name or "index": date_column})
    return long[[date_column, "sector", "sector_ret"]].copy()


__all__ = [
    "compute_residual_returns_rolling",
    "compute_idiosyncratic_returns",
    "DEFAULT_ROLLING_WINDOW",
    "DEFAULT_BROAD_SECTOR_FALLBACK",
]
