"""Designed experiments: factorial designs, linear models, optimization, and design generation.

Every public name is imported from its module when it is first asked for (PEP 562), not when the package is:
``from process_improve.experiments import generate_design`` loads the design generators and nothing of the analysis,
which needs statsmodels, or of the functional fits, which need scikit-learn (#1235). The names and where they live
are in :data:`_WHERE`; the imports under ``TYPE_CHECKING`` say the same to a type checker, and
``tests/test_lazy_imports.py`` keeps the two in step.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

from process_improve import _lazy

if TYPE_CHECKING:
    from process_improve.experiments._moment_aberration import (
        MomentAberrationResult,
        NotTwoLevelError,
        moment_aberration,
    )
    from process_improve.experiments.across_blocks import BlockedFit, fit_across_blocks
    from process_improve.experiments.allocation import allocate_runs
    from process_improve.experiments.analysis import analyze_experiment
    from process_improve.experiments.augment import augment_design
    from process_improve.experiments.compare import DesignComparison, compare_designs
    from process_improve.experiments.crossed_arrays import cross_arrays, signal_to_noise
    from process_improve.experiments.designs import generate_design
    from process_improve.experiments.designs_constrained import rename_factors
    from process_improve.experiments.designs_factorial import full_factorial
    from process_improve.experiments.designs_omars import is_omars, omars_properties
    from process_improve.experiments.designs_omars_ilp import generate_omars
    from process_improve.experiments.designs_screening import defining_relation
    from process_improve.experiments.evaluate import evaluate_all, evaluate_design
    from process_improve.experiments.factor import Constraint, DesignResult, Factor, Response, ResponseGoal
    from process_improve.experiments.functional import FunctionalFit, analyze_functional_experiment
    from process_improve.experiments.gaussian_process import GaussianProcessFit, fit_gaussian_process
    from process_improve.experiments.incomplete_blocks import bibd_analysis
    from process_improve.experiments.knowledge import doe_knowledge
    from process_improve.experiments.layouts import estimate_missing, tukey_nonadditivity
    from process_improve.experiments.least_squares_means import Comparisons, least_squares_means
    from process_improve.experiments.marginal_means import marginal_means_prediction
    from process_improve.experiments.model_comparison import compare_models, group_test
    from process_improve.experiments.models import Model, lm, predict, summary
    from process_improve.experiments.multistratum import multistratum_anova
    from process_improve.experiments.next_step import recommend_next_step
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
    from process_improve.experiments.pseudocomponents import from_pseudocomponents, to_pseudocomponents
    from process_improve.experiments.random_effects import (
        expected_mean_squares,
        intraclass_correlation_interval,
        sampling_scheme_variance,
        variance_components,
    )
    from process_improve.experiments.region import DesignRegion
    from process_improve.experiments.run_order import run_order_trends, time_trends, trend_robust_run_order
    from process_improve.experiments.run_sheet import read_run_sheet, run_sheet
    from process_improve.experiments.separating import (
        NotSeparableError,
        constraint_through_cuts,
        separating_constraint,
    )
    from process_improve.experiments.sequential import NoSteepestDirectionError, next_design, steepest_path
    from process_improve.experiments.simulate import simulate_design_performance
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
    from process_improve.experiments.surface_checks import (
        pooled_pure_error,
        stationary_point_region,
        third_order_contrasts,
    )
    from process_improve.experiments.three_level_components import interaction_components
    from process_improve.experiments.trade_off import TradeOffTableEntry, get_trade_off_table_entry, trade_off_table
    from process_improve.experiments.true_process import TrueProcess
    from process_improve.experiments.variance_dispersion import variance_dispersion
    from process_improve.experiments.visualization import main_effects_plot, visualize_doe
    from process_improve.experiments.yates import reverse_yates, yates_table

#: Each module of the package that has public names, and the names: what ``from process_improve.experiments import``
#: offers.
_WHERE = {
    "_moment_aberration": ("moment_aberration", "MomentAberrationResult", "NotTwoLevelError"),
    "across_blocks": ("BlockedFit", "fit_across_blocks"),
    "allocation": ("allocate_runs",),
    "analysis": ("analyze_experiment",),
    "augment": ("augment_design",),
    "compare": ("compare_designs", "DesignComparison"),
    "crossed_arrays": ("cross_arrays", "signal_to_noise"),
    "designs": ("generate_design",),
    "designs_constrained": ("rename_factors",),
    "designs_factorial": ("full_factorial",),
    "designs_omars": ("is_omars", "omars_properties"),
    "designs_omars_ilp": ("generate_omars",),
    "designs_screening": ("defining_relation",),
    "evaluate": ("evaluate_all", "evaluate_design"),
    "factor": ("Constraint", "DesignResult", "Factor", "Response", "ResponseGoal"),
    "functional": ("analyze_functional_experiment", "FunctionalFit"),
    "gaussian_process": ("fit_gaussian_process", "GaussianProcessFit"),
    "incomplete_blocks": ("bibd_analysis",),
    "knowledge": ("doe_knowledge",),
    "layouts": ("estimate_missing", "tukey_nonadditivity"),
    "least_squares_means": ("Comparisons", "least_squares_means"),
    "marginal_means": ("marginal_means_prediction",),
    "model_comparison": ("compare_models", "group_test"),
    "models": ("lm", "Model", "predict", "summary"),
    "multistratum": ("multistratum_anova",),
    "next_step": ("recommend_next_step",),
    "omars": ("analyze_omars", "OmarsResult"),
    "omars_trade_off": (
        "box_behnken_runs",
        "definitive_screening_runs",
        "get_omars_trade_off_table_entry",
        "omars_anchor_entry",
        "omars_minimum_runs",
        "omars_trade_off_table",
        "OmarsTradeOffTableEntry",
    ),
    "optimization": ("optimize_responses",),
    "projections": ("evaluate_projections",),
    "pseudocomponents": ("from_pseudocomponents", "to_pseudocomponents"),
    "random_effects": (
        "expected_mean_squares",
        "intraclass_correlation_interval",
        "sampling_scheme_variance",
        "variance_components",
    ),
    "region": ("DesignRegion",),
    "run_order": ("run_order_trends", "time_trends", "trend_robust_run_order"),
    "run_sheet": ("read_run_sheet", "run_sheet"),
    "separating": ("constraint_through_cuts", "NotSeparableError", "separating_constraint"),
    "sequential": ("next_design", "NoSteepestDirectionError", "steepest_path"),
    "simulate": ("simulate_design_performance",),
    "simulations": ("grocery", "manufacture", "popcorn"),
    "strategy": ("recommend_strategy",),
    "structures": ("c", "Column", "expand_grid", "Expt", "gather", "supplement"),
    "surface_checks": ("pooled_pure_error", "stationary_point_region", "third_order_contrasts"),
    "three_level_components": ("interaction_components",),
    "trade_off": ("get_trade_off_table_entry", "trade_off_table", "TradeOffTableEntry"),
    "true_process": ("TrueProcess",),
    "variance_dispersion": ("variance_dispersion",),
    "visualization": ("main_effects_plot", "visualize_doe"),
    "yates": ("reverse_yates", "yates_table"),
}

_MODULE_OF = {name: module for module, names in _WHERE.items() for name in names}

#: The names that are both a function of the package and the module the function lives in.
_SAME_NAME = frozenset(name for name, module in _MODULE_OF.items() if name == module)


class _Package(types.ModuleType):
    """Module type of this package, for the names in :data:`_SAME_NAME`.

    Importing a module makes the import system set it as its package's attribute. For a module named as the
    function it holds, the package's attribute has always been the function: so the function is set in the
    module's place, whoever imports the module and however. The module itself stays in ``sys.modules``.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if name in _SAME_NAME and isinstance(value, types.ModuleType):
            value = getattr(value, name)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a class, a function or a module of the package
    """Import the module that holds one of the package's public names, the first time the name is asked for.

    Any other module of the package is its attribute too, as when the package imported them all:
    ``experiments.models.lm``.
    """
    if name in _MODULE_OF:
        value = getattr(_lazy.load(f"{__name__}.{_MODULE_OF[name]}"), name)
        globals()[name] = value  # found in the package itself from here on
        return value
    return _lazy.submodule(__name__, name)


def __dir__() -> list[str]:
    return _lazy.listing(sys.modules[__name__], _MODULE_OF)


__all__ = [
    "BlockedFit",
    "Column",
    "Comparisons",
    "Constraint",
    "DesignComparison",
    "DesignRegion",
    "DesignResult",
    "Expt",
    "Factor",
    "FunctionalFit",
    "GaussianProcessFit",
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
    "analyze_functional_experiment",
    "analyze_omars",
    "augment_design",
    "bibd_analysis",
    "box_behnken_runs",
    "c",
    "compare_designs",
    "compare_models",
    "constraint_through_cuts",
    "cross_arrays",
    "defining_relation",
    "definitive_screening_runs",
    "doe_knowledge",
    "estimate_missing",
    "evaluate_all",
    "evaluate_design",
    "evaluate_projections",
    "expand_grid",
    "expected_mean_squares",
    "fit_across_blocks",
    "fit_gaussian_process",
    "from_pseudocomponents",
    "full_factorial",
    "gather",
    "generate_design",
    "generate_omars",
    "get_omars_trade_off_table_entry",
    "get_trade_off_table_entry",
    "grocery",
    "group_test",
    "interaction_components",
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
    "pooled_pure_error",
    "popcorn",
    "predict",
    "read_run_sheet",
    "recommend_next_step",
    "recommend_strategy",
    "rename_factors",
    "reverse_yates",
    "run_order_trends",
    "run_sheet",
    "sampling_scheme_variance",
    "separating_constraint",
    "signal_to_noise",
    "simulate_design_performance",
    "stationary_point_region",
    "steepest_path",
    "summary",
    "supplement",
    "third_order_contrasts",
    "time_trends",
    "to_pseudocomponents",
    "trade_off_table",
    "trend_robust_run_order",
    "tukey_nonadditivity",
    "variance_components",
    "variance_dispersion",
    "visualize_doe",
    "yates_table",
]
