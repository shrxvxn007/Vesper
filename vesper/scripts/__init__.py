"""scripts: synthetic data generators and CLI entry points."""

from .synthetic_generator import (
    SyntheticGenerator,
    generate_synthetic_dataset,
)

__all__ = [
    "SyntheticGenerator",
    "generate_synthetic_dataset",
]
