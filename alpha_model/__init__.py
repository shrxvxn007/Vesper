"""alpha_model: idiosyncratic-return targeting and cross-sectional ML model."""

from alpha_model.cross_sectional_model import AlphaModel, build_training_matrix
from alpha_model.target_formulation import (
    compute_idiosyncratic_returns,
    compute_residual_returns_rolling,
)

__all__ = [
    "compute_idiosyncratic_returns",
    "compute_residual_returns_rolling",
    "AlphaModel",
    "build_training_matrix",
]
