"""Designed experiments: factorial designs, linear models, optimization, and design generation."""

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.augment import augment_design
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Constraint, DesignResult, Factor
from process_improve.experiments.models import Model, lm, predict, summary
from process_improve.experiments.optimization import optimize_responses
from process_improve.experiments.structures import (
    Column,
    Expt,
    c,
    expand_grid,
    gather,
    supplement,
)
from process_improve.experiments.visualization import visualize_doe

__all__ = [
    "Column",
    "Constraint",
    "DesignResult",
    "Expt",
    "Factor",
    "Model",
    "analyze_experiment",
    "augment_design",
    "c",
    "evaluate_design",
    "expand_grid",
    "full_factorial",
    "gather",
    "generate_design",
    "lm",
    "optimize_responses",
    "predict",
    "summary",
    "supplement",
    "visualize_doe",
]
