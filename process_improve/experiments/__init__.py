"""Designed experiments: factorial designs, linear models, optimization, and design generation."""

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Constraint, DesignResult, Factor
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
    "Constraint",
    "DesignResult",
    "Expt",
    "Factor",
    "Model",
    "analyze_experiment",
    "c",
    "evaluate_design",
    "expand_grid",
    "full_factorial",
    "gather",
    "generate_design",
    "lm",
    "predict",
    "summary",
    "supplement",
]
