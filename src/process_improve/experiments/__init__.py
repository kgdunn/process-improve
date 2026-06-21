"""Designed experiments: factorial designs, linear models, optimization, and design generation."""

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.augment import augment_design
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.designs_omars import is_omars, omars_properties
from process_improve.experiments.designs_omars_ilp import generate_omars
from process_improve.experiments.evaluate import evaluate_all, evaluate_design
from process_improve.experiments.factor import Constraint, DesignResult, Factor, Response, ResponseGoal
from process_improve.experiments.knowledge import doe_knowledge
from process_improve.experiments.models import Model, lm, predict, summary
from process_improve.experiments.omars import OmarsResult, analyze_omars
from process_improve.experiments.optimization import optimize_responses
from process_improve.experiments.strategy import recommend_strategy
from process_improve.experiments.structures import (
    Column,
    Expt,
    c,
    expand_grid,
    gather,
    supplement,
)
from process_improve.experiments.visualization import main_effects_plot, visualize_doe

__all__ = [
    "Column",
    "Constraint",
    "DesignResult",
    "Expt",
    "Factor",
    "Model",
    "OmarsResult",
    "Response",
    "ResponseGoal",
    "analyze_experiment",
    "analyze_omars",
    "augment_design",
    "c",
    "doe_knowledge",
    "evaluate_all",
    "evaluate_design",
    "expand_grid",
    "full_factorial",
    "gather",
    "generate_design",
    "generate_omars",
    "is_omars",
    "lm",
    "main_effects_plot",
    "omars_properties",
    "optimize_responses",
    "predict",
    "recommend_strategy",
    "summary",
    "supplement",
    "visualize_doe",
]
