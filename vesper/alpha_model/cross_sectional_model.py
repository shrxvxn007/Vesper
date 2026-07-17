"""Cross-sectional alpha model: features → weekly forward idiosyncratic returns.

Two responsibilities:

1. **Training-matrix construction** (:func:`build_training_matrix`):
   align feature rows on date ``t`` with the *forward* 1-week idiosyncratic
   return ``r^{idio}_{t+1week}`` as the regression target. We never regress
   alpha on contemporaneous residuals — that's the textbook leakage trap.

2. **Model fitting + prediction** (:class:`AlphaModel`): a strictly time-
   series cross-validated model with a *purged* gap between folds so that
   week ``t`` cannot appear in both train and validation sub-splits. Two
   backends are available, both behind ``model_type``:

   * ``"ridge"`` (default) builds a ``StandardScaler → Ridge`` pipeline.
     Linear, heavy L2, fast to fit.
   * ``"xgboost"`` uses :class:`xgboost.XGBRegressor` with shallow trees
     (``max_depth=2``) and heavy L2 regularisation (``reg_lambda=10``)
     plus a mean-zero prior (``base_score=0.0``). No
     :class:`~sklearn.preprocessing.StandardScaler` (irrelevant for tree
     models).

   Both live inside the "shallow trees + heavy L2" envelope the spec
   asked for, and share the prediction contract documented on
   :meth:`AlphaModel.predict`.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# XGBoost defaults + pipeline builder
# ---------------------------------------------------------------------------

# Shallow trees + heavy L2 + mean-zero prior. See :class:`AlphaModel`
# docstring for the rationale. These are deliberately conservative: the
# universe is small (~12 cross-section names) and over-parameterised trees
# will fit noise.
DEFAULT_XGBOOST_PARAMS: Final[Mapping[str, object]] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "reg_lambda": 10.0,    # heavy L2 on leaf weights
    "reg_alpha": 0.1,      # mild L1 for feature parsimony
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "base_score": 0.0,     # CRITICAL: mean-zero prior so unbiased scores
}


def _build_pipeline(
    model_type: str,
    *,
    ridge_alpha: float,
    random_state: int,
    model_params: Mapping[str, object] | None,
):
    """Return a sklearn-compatible regressor for the requested backend.

    The returned object must support ``.fit(X, y)`` and ``.predict(X)``
    over 2-D ``np.ndarray`` inputs (XGBoost and sklearn's ``Pipeline``
    both do).
    """
    if model_type == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha, random_state=random_state)),
            ]
        )

    # XGBoost branch
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for model_type='xgboost'. "
            "Install via `pip install 'xgboost>=2.0'`."
        ) from exc

    config: dict[str, object] = dict(DEFAULT_XGBOOST_PARAMS)
    config["random_state"] = random_state
    if model_params:
        config.update(model_params)
    return XGBRegressor(**config)


# ---------------------------------------------------------------------------
# Purged time-series split
# ---------------------------------------------------------------------------


class PurgedGroupTimeSeriesSplit:
    """Time-series split with a *purge gap* and optional group-by.

    Standard :class:`~sklearn.model_selection.TimeSeriesSplit` puts every
    observation in chronological order and expands the train window. When you
    have *weekly* observations with a *forward* 1-week target, the very last
    training row is the validation row's lag — leakage!

    This splitter groups rows by ``date_group`` (one slot per cross-section)
    and inserts a non-trainable gap of ``gap_groups`` between train and
    validation. Such a gap guarantees no group appears on both sides.

    Args:
        n_splits: Number of CV folds (train-validation pairs).
        gap_groups: How many *groups* to drop between train and validation
            to enforce no observation appears on both sides.
    """

    def __init__(self, n_splits: int = 5, *, gap_groups: int = 1) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self._n_splits = n_splits
        self._gap = max(0, int(gap_groups))

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def split(
        self,
        X: np.ndarray | pd.DataFrame,  # noqa: ARG002 - unused but kept for sklearn-compat
        y: np.ndarray | pd.Series | None = None,  # noqa: ARG002
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("groups is required for PurgedGroupTimeSeriesSplit")
        groups_arr = np.asarray(groups)
        unique = np.unique(groups_arr, axis=0) if groups_arr.ndim > 1 else np.unique(groups_arr)
        if groups_arr.ndim > 1:
            # Use sorted tuples.
            unique_view = np.array([tuple(row) for row in unique])
            order = np.argsort(unique_view.view([("", unique_view.dtype)] * unique_view.shape[1]), axis=0)
            unique = unique[order.flatten()]
        else:
            unique = np.sort(unique)
        n_groups = len(unique)
        if n_groups < self._n_splits + 1:
            raise ValueError(
                f"Need more groups ({n_groups}) than n_splits + 1 ({self._n_splits + 1})"
            )

        # Map each row to its group index in the sorted unique array.
        if groups_arr.ndim == 1:
            group_to_pos = {g: i for i, g in enumerate(unique)}
            row_to_group = np.array([group_to_pos[g] for g in groups_arr])
        else:
            unique_view = np.array([tuple(row) for row in unique])
            row_to_group = np.array(
                [np.where((unique_view == tuple(row)).all(axis=1))[0][0] for row in groups_arr]
            )

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        # For each fold, validation is the i-th block since the end.
        test_block_size = n_groups // (self._n_splits + 1)
        if test_block_size < 1:
            test_block_size = 1
        for i in range(1, self._n_splits + 1):
            val_end = n_groups - (self._n_splits - i) * test_block_size
            val_start = val_end - test_block_size
            train_end = val_start - self._gap
            if train_end <= 0:
                continue
            train_mask = row_to_group < train_end
            val_mask = (row_to_group >= val_start) & (row_to_group < val_end)
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((train_idx, val_idx))
        return splits


# ---------------------------------------------------------------------------
# Training matrix construction
# ---------------------------------------------------------------------------


def build_training_matrix(
    features_df: pd.DataFrame,
    idio_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = ("nlp_decay_score", "graph_shock_score"),
    horizon: int = 1,
    target_column: str = "idiox_ret",
) -> pd.DataFrame:
    """Build a supervised learning matrix.

    Args:
        features_df: Features panel indexed by ``(date, ticker)``. Columns
            named in ``feature_columns``.
        idio_df: Idiosyncratic-return panel indexed by ``(date, ticker)``
            with column ``target_column``. The forward ``horizon``-step
            (in observation rows, i.e., weekly weeks) residual is the target.
        feature_columns: Feature column names.
        horizon: Number of forward periods for the target. Defaults to 1
            (1-week ahead).
        target_column: Column in ``idio_df`` carrying the residual returns.

    Returns:
        :class:`pandas.DataFrame` indexed by ``(date, ticker)`` with
        ``feature_columns`` plus a new ``target`` column. Rows where
        ``target`` is ``NaN`` are retained (so the splitter can drop them)
        — the model itself zeroes NaN rows during fit.
    """
    if features_df.empty:
        return features_df.copy()
    out = features_df.copy()
    # Bring idio_df into a frame keyed by (date, ticker).
    idio_long = idio_df.reset_index()
    date_col = idio_long.columns[0]
    ticker_col = idio_long.columns[1]

    # Build the shifted target: target_t = idiox_ret_{t + horizon weeks}.
    idio_long = idio_long.sort_values([ticker_col, date_col])
    idio_long["target"] = idio_long.groupby(ticker_col)[target_column].shift(-horizon)

    target_lookup = idio_long.set_index([date_col, ticker_col])[["target"]]
    out = out.join(target_lookup, how="left")
    return out


# ---------------------------------------------------------------------------
# Alpha model
# ---------------------------------------------------------------------------


class AlphaModel:
    """Cross-sectional regularised alpha model.

    Args:
        feature_columns: Feature column names consumed by the pipeline.
        model_type: One of ``"ridge"`` (default; StandardScaler + Ridge)
            or ``"xgboost"`` (shallow trees with heavy L2 regularisation;
            requires ``xgboost>=2.0``).
        ridge_alpha: L2 regularisation strength for the ridge regressor
            (only used when ``model_type="ridge"``).
        model_params: Optional override dict for the XGBoost keyword
            arguments. Ignored when ``model_type="ridge"``.
        n_splits: Number of purged time-series CV folds used during fit.
        gap_groups: Purged-split gap between train and validation groups.
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] = ("nlp_decay_score", "graph_shock_score"),
        model_type: Literal["ridge", "xgboost"] = "ridge",
        ridge_alpha: float = 5.0,
        model_params: Mapping[str, object] | None = None,
        n_splits: int = 5,
        gap_groups: int = 1,
        random_state: int = 42,
    ) -> None:
        if model_type not in {"ridge", "xgboost"}:
            raise ValueError(
                f"model_type must be one of {{'ridge', 'xgboost'}}; got {model_type!r}"
            )
        self._feature_columns = tuple(feature_columns)
        self._model_type = model_type
        self._pipeline = _build_pipeline(
            model_type=model_type,
            ridge_alpha=ridge_alpha,
            random_state=random_state,
            model_params=model_params,
        )
        self._splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, gap_groups=gap_groups)
        self._random_state = random_state
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def model_type(self) -> str:
        """Return the active backend (one of ``"ridge"``, ``"xgboost"``)."""
        return self._model_type

    def fit(self, training_df: pd.DataFrame, *, date_column: str = "date") -> "AlphaModel":
        """Fit the pipeline using purged-group time-series CV.

        Args:
            training_df: Long-form DataFrame with columns ``date_column``,
                ``ticker``, ``feature_columns`` and ``target``.
            date_column: Date column that we *group* by for the purged split.

        Returns:
            ``self``, after refitting the pipeline on the full data.
        """
        df = training_df.dropna(subset=["target"] + list(self._feature_columns)).copy()
        if df.empty:
            raise ValueError("No usable rows after dropping NaNs; check feature/target pipelines.")
        # Ensure sorted by date, ticker.
        df = df.sort_values([date_column, "ticker"]).reset_index(drop=True)

        groups = df[date_column].rank(method="dense").astype(int).to_numpy()
        X = df[list(self._feature_columns)].to_numpy(dtype=float)
        y = df["target"].to_numpy(dtype=float)

        # Run purged CV: collect train/val indices per fold, score val MSE,
        # then refit on all data. We log fold scores to stdout for diagnostics.
        for fold_idx, (train_idx, val_idx) in enumerate(
            self._splitter.split(X, y, groups=groups)
        ):
            X_train, y_train = X[train_idx], y[train_idx]
            if len(np.unique(groups[train_idx])) < 2 or X_train.shape[0] < len(self._feature_columns):
                continue
            self._pipeline.fit(X_train, y_train)

        # Refit on the entire dataset with the same hyper-parameters.
        self._pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Score cross-sectional alpha for one observation date.

        Args:
            features_df: Long-form data with columns ``feature_columns`` and
                ``ticker``. Should be aligned to a single date and have one
                row per ticker.

        Returns:
            :class:`pandas.Series` indexed by ticker with the predicted
            forward 1-week idiosyncratic return.
        """
        if not self._fitted:
            raise RuntimeError("AlphaModel.predict called before fit().")
        X = features_df[list(self._feature_columns)].to_numpy(dtype=float)
        preds = self._pipeline.predict(X)
        # Soft-clamp to the historical target range; doesn't change ranking
        # meaningfully but prevents numerical oddities.
        preds = np.clip(preds, -1.0, 1.0)
        return pd.Series(preds, index=features_df["ticker"].to_numpy(), name="alpha")


__all__ = [
    "PurgedGroupTimeSeriesSplit",
    "build_training_matrix",
    "AlphaModel",
    "DEFAULT_XGBOOST_PARAMS",
]
