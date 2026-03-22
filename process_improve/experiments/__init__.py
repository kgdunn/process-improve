"""Designed experiments: factorial designs, linear models, and optimization."""

from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.models import Model, lm, predict, summary
from process_improve.experiments.structures import (
    Column,
    Expt,
    c,
    expand_grid,
    gather,
    supplement,
)

__all__ = [
    "Column",
    "Expt",
    "Model",
    "c",
    "expand_grid",
    "full_factorial",
    "gather",
    "lm",
    "predict",
    "summary",
    "supplement",
]
