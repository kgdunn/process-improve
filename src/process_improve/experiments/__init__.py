"""Designed experiments: factorial designs, linear models, optimization, and design generation."""

from process_improve.experiments._moment_aberration import (
    MomentAberrationResult,
    NotTwoLevelError,
    moment_aberration,
)
from process_improve.experiments.allocation import allocate_runs
from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.augment import augment_design
from process_improve.experiments.compare import DesignComparison, compare_designs
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_constrained import rename_factors
from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.designs_omars import is_omars, omars_properties
from process_improve.experiments.designs_omars_ilp import generate_omars
from process_improve.experiments.evaluate import evaluate_all, evaluate_design
from process_improve.experiments.factor import Constraint, DesignResult, Factor, Response, ResponseGoal
from process_improve.experiments.knowledge import doe_knowledge
from process_improve.experiments.least_squares_means import Comparisons, least_squares_means
from process_improve.experiments.marginal_means import marginal_means_prediction
from process_improve.experiments.models import Model, lm, predict, summary
from process_improve.experiments.multistratum import multistratum_anova
from process_improve.experiments.omars import OmarsResult, analyze_omars
from process_improve.experiments.omars_trade_off import (
    OmarsTradeOffTableEntry,
    box_behnken_runs,
    definitive_screening_runs,
    get_omars_trade_off_table_entry,
    omars_anchor_entry,
    omars_minimum_runs,
    omars_trade_off_table,
)
from process_improve.experiments.optimization import optimize_responses
from process_improve.experiments.projections import evaluate_projections
from process_improve.experiments.random_effects import (
    expected_mean_squares,
    intraclass_correlation_interval,
    variance_components,
)
from process_improve.experiments.region import DesignRegion
from process_improve.experiments.run_order import run_order_trends, time_trends, trend_robust_run_order
from process_improve.experiments.separating import (
    NotSeparableError,
    constraint_through_cuts,
    separating_constraint,
)
from process_improve.experiments.sequential import NoSteepestDirectionError, next_design, steepest_path
from process_improve.experiments.simulations import grocery, manufacture, popcorn
from process_improve.experiments.strategy import recommend_strategy
from process_improve.experiments.structures import (
    Column,
    Expt,
    c,
    expand_grid,
    gather,
    supplement,
)
from process_improve.experiments.trade_off import TradeOffTableEntry, get_trade_off_table_entry, trade_off_table
from process_improve.experiments.true_process import TrueProcess
from process_improve.experiments.visualization import main_effects_plot, visualize_doe

__all__ = [
    "Column",
    "Comparisons",
    "Constraint",
    "DesignComparison",
    "DesignRegion",
    "DesignResult",
    "Expt",
    "Factor",
    "Model",
    "MomentAberrationResult",
    "NoSteepestDirectionError",
    "NotSeparableError",
    "NotTwoLevelError",
    "OmarsResult",
    "OmarsTradeOffTableEntry",
    "Response",
    "ResponseGoal",
    "TradeOffTableEntry",
    "TrueProcess",
    "allocate_runs",
    "analyze_experiment",
    "analyze_omars",
    "augment_design",
    "box_behnken_runs",
    "c",
    "compare_designs",
    "constraint_through_cuts",
    "definitive_screening_runs",
    "doe_knowledge",
    "evaluate_all",
    "evaluate_design",
    "evaluate_projections",
    "expand_grid",
    "expected_mean_squares",
    "full_factorial",
    "gather",
    "generate_design",
    "generate_omars",
    "get_omars_trade_off_table_entry",
    "get_trade_off_table_entry",
    "grocery",
    "intraclass_correlation_interval",
    "is_omars",
    "least_squares_means",
    "lm",
    "main_effects_plot",
    "manufacture",
    "marginal_means_prediction",
    "moment_aberration",
    "multistratum_anova",
    "next_design",
    "omars_anchor_entry",
    "omars_minimum_runs",
    "omars_properties",
    "omars_trade_off_table",
    "optimize_responses",
    "popcorn",
    "predict",
    "recommend_strategy",
    "rename_factors",
    "run_order_trends",
    "separating_constraint",
    "steepest_path",
    "summary",
    "supplement",
    "time_trends",
    "trade_off_table",
    "trend_robust_run_order",
    "variance_components",
    "visualize_doe",
]
