"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for designed experiments.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from patsy import PatsyError
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

logger = logging.getLogger(__name__)

# Per the ENG-11 error-handling style guide, every tool wrapper narrows
# its ``except`` to this canonical set so that anything *outside* this
# set propagates up to ``mcp_server._serialise_tool_error`` and gets
# redacted before reaching the caller (SEC-18 / #267).
_TOOL_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    np.linalg.LinAlgError,
    PatsyError,
)

_EXPERIMENTS_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _EXPERIMENTS_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# create_factorial_design
# ---------------------------------------------------------------------------


class CreateFactorialDesignInput(BaseModel):
    """Input contract for ``create_factorial_design``."""

    model_config = ConfigDict(extra="forbid")

    n_factors: int = Field(
        ...,
        ge=2,
        le=10,
        description="Number of factors in the design (2 to 10).",
    )
    factor_names: list[str] | None = Field(
        None,
        description=(
            "Optional list of factor names. Must have exactly n_factors entries. "
            "If not provided, factors are named A, B, C, ..."
        ),
    )


@tool_spec(
    name="create_factorial_design",
    description=(
        "Create a full factorial (2^k) experimental design with factors coded as -1 and +1. "
        "Returns the design matrix as a list of dictionaries (one per run), the number of runs, "
        "the number of factors, and the factor names. "
        "Use this when planning a designed experiment to systematically explore the effect of "
        "2 to 10 factors, each at two levels."
    ),
    input_model=CreateFactorialDesignInput,
    examples="""
    # "Create a 2-factor full factorial design"
        -> ``create_factorial_design(n_factors=2)``

    # "Create a 3-factor design with named factors"
        -> ``create_factorial_design(n_factors=3, factor_names=["Temperature", "Pressure", "Time"])``
    """,
    category="experiments",
)
def create_factorial_design(spec: CreateFactorialDesignInput) -> dict[str, Any]:
    """Create a full factorial design."""
    try:
        from process_improve.experiments.designs_factorial import full_factorial  # noqa: PLC0415

        columns = full_factorial(spec.n_factors, names=spec.factor_names)
        design = pd.concat(columns, axis=1)
        names = list(design.columns)
        return clean({
            "design": design.to_dict(orient="records"),
            "n_runs": len(design),
            "n_factors": spec.n_factors,
            "factor_names": names,
        })
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool create_factorial_design failed")
        return {"error": str(e)}


_register("create_factorial_design")


# ---------------------------------------------------------------------------
# fit_linear_model
# ---------------------------------------------------------------------------


class FitLinearModelInput(BaseModel):
    """Input contract for ``fit_linear_model``."""

    model_config = ConfigDict(extra="forbid")

    formula: str = Field(
        ...,
        description=(
            "Model formula in Wilkinson notation, e.g. 'y ~ A*B*C'. "
            "The left-hand side is the response variable name, the right-hand "
            "side specifies the terms. '*' expands to main effects and all "
            "interactions; ':' denotes a specific interaction; '+' adds terms."
        ),
    )
    data: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dictionaries, one per experimental run. Each dict must "
            "contain keys for every factor and the response variable referenced "
            "in the formula. Example: [{'A': -1, 'B': -1, 'y': 45.2}, ...]"
        ),
    )


@tool_spec(
    name="fit_linear_model",
    description=(
        "Fit a linear model to experimental data using a formula specification. "
        "The formula uses Wilkinson notation, e.g. 'y ~ A*B*C' for a full factorial model "
        "with all main effects and interactions. Use 'y ~ A + B' for main effects only, or "
        "'y ~ A*B' for main effects plus the A:B interaction. "
        "The data should be a list of dictionaries, where each dictionary represents one "
        "experimental run with column names as keys. Factor columns should contain -1/+1 "
        "coded values (or real values), and there must be a response column matching the "
        "left-hand side of the formula. "
        "Returns the fitted coefficients (name, estimate), R-squared, and a text summary."
    ),
    input_model=FitLinearModelInput,
    examples="""
    # "Fit a model y ~ A*B to my 2^2 factorial data"
        -> ``fit_linear_model(formula="y ~ A*B",
                data=[{"A":-1,"B":-1,"y":28}, {"A":1,"B":-1,"y":36},
                      {"A":-1,"B":1,"y":18}, {"A":1,"B":1,"y":31}])``

    # "Fit a main-effects-only model"
        -> ``fit_linear_model(formula="y ~ A + B + C", data=[...])``
    """,
    category="experiments",
)
def fit_linear_model(spec: FitLinearModelInput) -> dict[str, Any]:
    """Fit a linear model to experimental data."""
    try:
        from process_improve.config import settings  # noqa: PLC0415
        from process_improve.experiments.models import lm, validate_formula_is_safe  # noqa: PLC0415
        from process_improve.experiments.structures import Expt  # noqa: PLC0415

        if len(spec.data) > settings.max_matrix_rows:
            return {
                "error": (
                    f"data has {len(spec.data)} rows; the cap is "
                    f"settings.max_matrix_rows={settings.max_matrix_rows}."
                )
            }
        if len(spec.formula) > settings.max_formula_chars:
            return {
                "error": (
                    f"formula is {len(spec.formula)} chars; the cap is "
                    f"settings.max_formula_chars={settings.max_formula_chars}."
                )
            }

        df = pd.DataFrame(spec.data)

        # Patsy evaluates formula terms as Python expressions, so a formula from an
        # untrusted caller is a code-execution vector. Only allow a plain Wilkinson
        # formula over the columns actually present in the data.
        validate_formula_is_safe(spec.formula, df.columns)

        expt_data = Expt(df)
        expt_data.pi_title = None
        expt_data.pi_source = None
        expt_data.pi_units = None

        model = lm(spec.formula, expt_data)

        params = model.get_parameters(drop_intercept=True)
        if isinstance(params, pd.Series):
            coefficients = [
                {"name": str(name), "estimate": float(value)}
                for name, value in params.items()
            ]
        else:
            coefficients = params.to_dict(orient="records")

        r2 = float(model._OLS.rsquared)

        smry = model.summary(print_to_screen=False)
        summary_text = str(smry)

        return clean({
            "coefficients": coefficients,
            "r2": r2,
            "summary_text": summary_text,
        })
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool fit_linear_model failed")
        return {"error": str(e)}


_register("fit_linear_model")


# ---------------------------------------------------------------------------
# generate_design
# ---------------------------------------------------------------------------


class GenerateDesignInput(BaseModel):
    """Input contract for ``generate_design``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of factor specifications (name, type, low/high or levels, units).",
    )
    design_type: Literal[
        "full_factorial",
        "fractional_factorial",
        "plackett_burman",
        "box_behnken",
        "ccd",
        "dsd",
        "d_optimal",
        "i_optimal",
        "a_optimal",
        "mixture",
        "taguchi",
    ] | None = Field(
        None,
        description="Design type. If omitted, auto-selected based on factors and budget.",
    )
    budget: int | None = Field(
        None,
        ge=1,
        description="Maximum number of experimental runs.",
    )
    center_points: int = Field(
        3,
        ge=0,
        description="Number of center point replicates (default: 3).",
    )
    replicates: int = Field(
        1,
        ge=1,
        description="Number of full replicates (default: 1).",
    )
    resolution: int | None = Field(
        None,
        ge=3,
        le=5,
        description="Minimum resolution for fractional factorials (3, 4, or 5).",
    )
    alpha: Literal["rotatable", "face_centered", "orthogonal"] | None = Field(
        None,
        description="Axial distance for CCD designs.",
    )
    random_seed: int = Field(
        42,
        description="Seed for reproducible randomization (default: 42).",
    )


@tool_spec(
    name="generate_design",
    description=(
        "Generate an experimental design matrix for a designed experiment. "
        "Supports full factorial, fractional factorial, Plackett-Burman, Box-Behnken, "
        "Central Composite (CCD), Definitive Screening (DSD), D-optimal, mixture, "
        "and Taguchi designs. "
        "Each factor needs a name and type ('continuous', 'categorical', or 'mixture'). "
        "Continuous factors require 'low' and 'high' bounds. Categorical factors require 'levels'. "
        "If design_type is not specified, one is auto-selected based on the number of factors and budget. "
        "Returns the design matrix in both coded (-1/+1) and actual units, run order, and metadata."
    ),
    input_model=GenerateDesignInput,
    examples="""
    # "Create a 2-factor CCD for Temperature (150-200 degC) and Pressure (1-5 bar)"
        -> ``generate_design(factors=[{"name": "Temperature", "low": 150, "high": 200, "units": "degC"},
                                      {"name": "Pressure", "low": 1, "high": 5, "units": "bar"}],
                             design_type="ccd", alpha="rotatable")``

    # "Screen 7 factors with minimal runs"
        -> ``generate_design(factors=[{"name": "A", "low": -1, "high": 1}, ...7 factors...],
                             design_type="plackett_burman")``

    # "Create a 2^(5-2) fractional factorial at resolution III"
        -> ``generate_design(factors=[{"name": f, "low": -1, "high": 1} for f in "ABCDE"],
                             design_type="fractional_factorial", resolution=3)``
    """,
    category="experiments",
)
def generate_design_tool(spec: GenerateDesignInput) -> dict[str, Any]:
    """Generate an experimental design."""
    try:
        from process_improve.experiments.designs import generate_design  # noqa: PLC0415
        from process_improve.experiments.factor import Factor  # noqa: PLC0415

        factor_objects = [Factor(**f) for f in spec.factors]

        result = generate_design(
            factors=factor_objects,
            design_type=spec.design_type,
            budget=spec.budget,
            center_points=spec.center_points,
            replicates=spec.replicates,
            resolution=spec.resolution,
            alpha=spec.alpha,
            random_seed=spec.random_seed,
        )

        design_coded = result.design.drop(columns=["RunOrder"], errors="ignore")
        design_actual = result.design_actual.drop(columns=["RunOrder"], errors="ignore")

        output: dict[str, Any] = {
            "design_coded": design_coded.to_dict(orient="records"),
            "design_actual": design_actual.to_dict(orient="records"),
            "run_order": result.run_order,
            "design_type": result.design_type,
            "n_runs": result.n_runs,
            "n_factors": result.n_factors,
            "factor_names": result.factor_names,
        }
        if result.generators:
            output["generators"] = result.generators
        if result.defining_relation:
            output["defining_relation"] = result.defining_relation
        if result.resolution is not None:
            output["resolution"] = result.resolution
        if result.alpha is not None:
            output["alpha"] = result.alpha

        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool generate_design failed")
        return {"error": str(e)}


_register("generate_design")


# ---------------------------------------------------------------------------
# evaluate_design
# ---------------------------------------------------------------------------


class EvaluateDesignInput(BaseModel):
    """Input contract for ``evaluate_design``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dictionaries, one per experimental run. Each dict maps "
            "factor name to coded value. Example: [{'A': -1, 'B': -1}, ...]"
        ),
    )
    model: Literal["main_effects", "interactions", "quadratic"] | None = Field(
        None,
        description=(
            "Model type to evaluate against. 'main_effects' = main effects only, "
            "'interactions' = main effects + 2-factor interactions (default), "
            "'quadratic' = interactions + squared terms."
        ),
    )
    metric: str | list[str] = Field(
        "d_efficiency",
        description=(
            "One or more metric names to compute. Default: 'd_efficiency'. "
            "Options include d_efficiency, i_efficiency, g_efficiency, "
            "prediction_variance, vif, condition_number, power, "
            "degrees_of_freedom, alias_structure, confounding, resolution, "
            "defining_relation, clear_effects, minimum_aberration."
        ),
    )
    effect_size: float | None = Field(
        None,
        description="Expected effect size for power calculation.",
    )
    alpha: float = Field(
        0.05,
        description="Significance level (default 0.05).",
    )
    sigma: float | None = Field(
        None,
        description="Estimated noise standard deviation.",
    )


@tool_spec(
    name="evaluate_design",
    description=(
        "Evaluate the quality of an experimental design matrix by computing metrics such as "
        "D-efficiency, G-efficiency, I-efficiency, VIF, condition number, alias structure, "
        "confounding pattern, resolution, power, prediction variance, degrees of freedom, "
        "clear effects, and minimum aberration. "
        "The design_matrix should be a list of dictionaries with factor names as keys and "
        "coded values (-1/+1) as values. "
        "Use this after generating a design to check if it meets quality criteria, or to "
        "compare alternative designs."
    ),
    input_model=EvaluateDesignInput,
    examples="""
    # "What is the D-efficiency of my 2^3 factorial design?"
        -> ``evaluate_design(design_matrix=[{"A":-1,"B":-1,"C":-1}, ...],
                metric="d_efficiency", model="interactions")``

    # "Check VIF and condition number"
        -> ``evaluate_design(design_matrix=[...],
                metric=["vif", "condition_number"], model="interactions")``

    # "What is the power to detect an effect of size 2 with noise SD of 1?"
        -> ``evaluate_design(design_matrix=[...],
                metric="power", effect_size=2.0, sigma=1.0)``
    """,
    category="experiments",
)
def evaluate_design_tool(spec: EvaluateDesignInput) -> dict[str, Any]:
    """Evaluate design quality."""
    try:
        from process_improve.experiments.evaluate import evaluate_design  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        result = evaluate_design(
            df,
            model=spec.model,
            metric=spec.metric,
            effect_size=spec.effect_size,
            alpha=spec.alpha,
            sigma=spec.sigma,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool evaluate_design failed")
        return {"error": str(e)}


_register("evaluate_design")


# ---------------------------------------------------------------------------
# analyze_experiment
# ---------------------------------------------------------------------------


class AnalyzeExperimentInput(BaseModel):
    """Input contract for ``analyze_experiment``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dicts, one per run. Must contain factor columns and "
            "optionally the response column. Example: "
            "[{'A': -1, 'B': -1, 'y': 28}, {'A': 1, 'B': -1, 'y': 36}, ...]"
        ),
    )
    response_column: str = Field(
        ...,
        description="Name of the response column in the design_matrix.",
    )
    model: str | None = Field(
        None,
        description=(
            "Model type ('main_effects', 'interactions' default, or 'quadratic') "
            "or an explicit Wilkinson formula string (e.g. 'y ~ A*B'). "
            "Formulas are validated by validate_formula_is_safe at the dispatch site."
        ),
    )
    analysis_type: str | list[str] = Field(
        "anova",
        description=(
            "One or more analysis types to run. Default: 'anova'. "
            "Options: anova, effects, coefficients, significance, "
            "residual_diagnostics, lack_of_fit, curvature_test, "
            "model_selection, box_cox, lenth_method, confidence_intervals, "
            "prediction, confirmation_test."
        ),
    )
    significance_level: float = Field(
        0.05,
        description="Significance level (default 0.05).",
    )
    transform: Literal["log", "sqrt", "inverse", "box_cox"] | None = Field(
        None,
        description="Optional response transform before fitting.",
    )
    new_points: list[dict[str, Any]] | None = Field(
        None,
        description="New factor settings for prediction or confirmation.",
    )
    observed_at_new: list[float] | None = Field(
        None,
        description="Observed values at new_points (for confirmation testing).",
    )


@tool_spec(
    name="analyze_experiment",
    description=(
        "Fit a model to experimental data and run statistical analyses. "
        "Supports ANOVA, effects, coefficients with p-values, significance testing, "
        "residual diagnostics (Shapiro-Wilk, Durbin-Watson, Breusch-Pagan, Cook's distance), "
        "lack-of-fit test, curvature test (center points vs factorial points), "
        "stepwise model selection (AIC/BIC), Box-Cox transformation, "
        "Lenth's method (PSE for unreplicated factorials), confidence intervals, "
        "prediction with prediction intervals, and confirmation run testing. "
        "Always returns a model summary with R-squared, adj-R-squared, pred-R-squared, and adequate precision. "
        "The design_matrix should contain factor columns with coded values (-1/+1). "
        "The response can be in a separate column or included in design_matrix."
    ),
    input_model=AnalyzeExperimentInput,
    examples="""
    # "Run ANOVA on my 2^2 factorial experiment"
        -> ``analyze_experiment(design_matrix=[{"A":-1,"B":-1,"y":28}, ...],
                response_column="y", analysis_type="anova")``

    # "Check residual diagnostics and lack of fit"
        -> ``analyze_experiment(design_matrix=[...], response_column="y",
                analysis_type=["residual_diagnostics", "lack_of_fit"])``

    # "Use Lenth's method on my unreplicated factorial"
        -> ``analyze_experiment(design_matrix=[...], response_column="y",
                analysis_type="lenth_method")``
    """,
    category="experiments",
)
def analyze_experiment_tool(spec: AnalyzeExperimentInput) -> dict[str, Any]:
    """Analyze experimental data."""
    try:
        from process_improve.experiments.analysis import analyze_experiment  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        np_df = pd.DataFrame(spec.new_points) if spec.new_points else None

        result = analyze_experiment(
            design_matrix=df,
            response_column=spec.response_column,
            model=spec.model,
            analysis_type=spec.analysis_type,
            significance_level=spec.significance_level,
            transform=spec.transform,
            new_points=np_df,
            observed_at_new=spec.observed_at_new,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool analyze_experiment failed")
        return {"error": str(e)}


_register("analyze_experiment")


# ---------------------------------------------------------------------------
# optimize_responses
# ---------------------------------------------------------------------------


class OptimizeResponsesInput(BaseModel):
    """Input contract for ``optimize_responses``."""

    model_config = ConfigDict(extra="forbid")

    fitted_models: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "One or more fitted models from analyze_experiment. Each entry must "
            "include 'coefficients' (list of {term, coefficient}), 'factor_names' "
            "(list of strings), and optionally 'response_name', 'mse_residual', 'r_squared'."
        ),
    )
    goals: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Per-response optimisation goals. Each entry: response (str), "
            "goal ('maximize'|'minimize'|'target'), low, high, optional target, "
            "weight, importance. Required for 'desirability' method."
        ),
    )
    method: Literal[
        "desirability",
        "steepest_ascent",
        "steepest_descent",
        "stationary_point",
        "canonical_analysis",
    ] = Field(
        "desirability",
        description="Optimisation method (default: 'desirability').",
    )
    factor_ranges: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            'Factor bounds in actual units, e.g. {"Temperature": {"low": 150, "high": 200}}. '
            "Used to convert coded settings to actual units in the output."
        ),
    )
    step_size: float = Field(
        0.5,
        description="Step size in coded units for steepest ascent/descent (default 0.5).",
    )
    n_steps: int = Field(
        10,
        ge=1,
        description="Number of steps for steepest ascent/descent (default 10).",
    )
    desirability_weights: list[float] | None = Field(
        None,
        description="Importance weights for composite desirability (overrides per-goal importance).",
    )


@tool_spec(
    name="optimize_responses",
    description=(
        "Find optimal factor settings for one or multiple responses from fitted experimental models. "
        "Supports several methods: 'desirability' (Derringer-Suich desirability functions for single or "
        "multi-response optimisation), 'steepest_ascent' / 'steepest_descent' (move along the gradient "
        "of a first-order model), 'stationary_point' (locate the optimum of a second-order model), "
        "'canonical_analysis' (eigenvalue decomposition to classify the response surface shape). "
        "Each fitted_model must include coefficients (as returned by analyze_experiment with "
        "analysis_type='coefficients'), factor_names, and response_name. "
        "For desirability, each goal specifies whether to maximize, minimize, or target a value."
    ),
    input_model=OptimizeResponsesInput,
    examples="""
    # "Find the stationary point of my quadratic model"
        -> ``optimize_responses(fitted_models=[{"response_name": "yield",
                "coefficients": [{"term": "Intercept", "coefficient": 40},
                    {"term": "A", "coefficient": 5.25}, {"term": "B", "coefficient": -2},
                    {"term": "I(A ** 2)", "coefficient": -3}, {"term": "I(B ** 2)", "coefficient": -1.5},
                    {"term": "A:B", "coefficient": 1.5}],
                "factor_names": ["A", "B"]}],
            method="stationary_point")``

    # "Optimize two responses using desirability"
        -> ``optimize_responses(fitted_models=[model1, model2],
                goals=[{"response": "yield", "goal": "maximize", "low": 30, "high": 50},
                       {"response": "cost", "goal": "minimize", "low": 10, "high": 40}],
                method="desirability")``

    # "Generate a steepest ascent path from a first-order model"
        -> ``optimize_responses(fitted_models=[model],
                method="steepest_ascent", step_size=0.5, n_steps=8,
                factor_ranges={"Temperature": {"low": 150, "high": 200}})``
    """,
    category="experiments",
)
def optimize_responses_tool(spec: OptimizeResponsesInput) -> dict[str, Any]:
    """Optimize experimental responses."""
    try:
        from process_improve.experiments.optimization import optimize_responses  # noqa: PLC0415

        result = optimize_responses(
            fitted_models=spec.fitted_models,
            goals=spec.goals,
            method=spec.method,
            factor_ranges=spec.factor_ranges,
            step_size=spec.step_size,
            n_steps=spec.n_steps,
            desirability_weights=spec.desirability_weights,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool optimize_responses failed")
        return {"error": str(e)}


_register("optimize_responses")


# ---------------------------------------------------------------------------
# augment_design
# ---------------------------------------------------------------------------


class AugmentDesignInput(BaseModel):
    """Input contract for ``augment_design``."""

    model_config = ConfigDict(extra="forbid")

    existing_design: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "Current design matrix as list of dicts with factor names as keys "
            "and coded values (-1/+1) as values. "
            "Example: [{'A': -1, 'B': -1}, {'A': 1, 'B': -1}, ...]"
        ),
    )
    augmentation_type: Literal[
        "foldover",
        "semifold",
        "add_center_points",
        "add_axial_points",
        "add_runs_optimal",
        "upgrade_to_rsm",
        "add_blocks",
        "replicate",
    ] = Field(
        ...,
        description="Type of augmentation to apply to the design.",
    )
    target_model: Literal["main_effects", "interactions", "quadratic"] | None = Field(
        None,
        description=(
            "Desired model after augmentation. Used by 'add_runs_optimal' "
            "and 'upgrade_to_rsm'. Default: 'interactions'."
        ),
    )
    n_additional_runs: int | None = Field(
        None,
        ge=1,
        description=(
            "Budget for additional runs. Interpretation depends on type: "
            "number of center points, D-optimal runs, replicates, or blocks."
        ),
    )
    fold_on: str | None = Field(
        None,
        description=(
            "Factor name to fold on (semifold only). "
            "If omitted, the best factor is auto-selected."
        ),
    )
    alpha: str | float | None = Field(
        None,
        description=(
            "Axial distance for add_axial_points or upgrade_to_rsm. "
            "'rotatable', 'face_centered', 'orthogonal', or a numeric value."
        ),
    )
    generators: list[str] | None = Field(
        None,
        description=(
            "Generator strings from the original fractional factorial design "
            "(e.g. ['D=ABC']). Needed for foldover/semifold alias analysis."
        ),
    )


@tool_spec(
    name="augment_design",
    description=(
        "Extend or modify an existing experimental design. Supports foldover (de-alias all "
        "2-factor interactions), semifold (de-alias specific interactions with fewer runs), "
        "adding center points (test for curvature), adding axial/star points (upgrade to CCD "
        "for response surface modeling), D-optimal augmentation (add runs to maximize information), "
        "upgrade to RSM (convert screening design to response surface design), add blocks "
        "(retroactively confound block effects with high-order interactions), and replication "
        "(improve precision estimates). "
        "Always returns the augmented design matrix plus an explanation of what changed in the "
        "alias structure and design properties."
    ),
    input_model=AugmentDesignInput,
    examples="""
    # "Fold over my 2^(4-1) design to de-alias two-factor interactions"
        -> ``augment_design(existing_design=[...], augmentation_type="foldover",
                generators=["D=ABC"])``

    # "Add 5 center points to test for curvature"
        -> ``augment_design(existing_design=[...], augmentation_type="add_center_points",
                n_additional_runs=5)``

    # "Upgrade my screening design to a CCD for response surface modeling"
        -> ``augment_design(existing_design=[...], augmentation_type="upgrade_to_rsm",
                alpha="rotatable", target_model="quadratic")``

    # "Add 6 D-optimal runs to improve my design"
        -> ``augment_design(existing_design=[...], augmentation_type="add_runs_optimal",
                n_additional_runs=6, target_model="interactions")``
    """,
    category="experiments",
)
def augment_design_tool(spec: AugmentDesignInput) -> dict[str, Any]:
    """Augment an existing design."""
    try:
        from process_improve.experiments.augment import augment_design  # noqa: PLC0415

        df = pd.DataFrame(spec.existing_design)
        result = augment_design(
            existing_design=df,
            augmentation_type=spec.augmentation_type,
            target_model=spec.target_model,
            n_additional_runs=spec.n_additional_runs,
            fold_on=spec.fold_on,
            alpha=spec.alpha,
            generators=spec.generators,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool augment_design failed")
        return {"error": str(e)}


_register("augment_design")


# ---------------------------------------------------------------------------
# visualize_doe
# ---------------------------------------------------------------------------


class VisualizeDoeInput(BaseModel):
    """Input contract for ``visualize_doe``."""

    model_config = ConfigDict(extra="forbid")

    plot_type: Literal[
        "pareto", "half_normal", "daniel",
        "main_effects", "interaction", "perturbation",
        "residuals_vs_fitted", "normal_probability",
        "residuals_vs_order", "box_cox",
        "contour", "surface_3d", "prediction_variance",
        "cube_plot", "square_plot",
        "desirability_contour", "overlay",
        "ridge_trace", "steepest_ascent_path",
        "fds_plot", "power_curve",
    ] = Field(
        ...,
        description="Type of DOE plot to generate.",
    )
    analysis_results: dict[str, Any] | None = Field(
        None,
        description=(
            "Results from fit_linear_model or analyze_experiment. "
            "Should contain keys like 'coefficients', 'effects', "
            "'residual_diagnostics', 'lenth_method', 'model_summary'."
        ),
    )
    design_data: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Raw design matrix as a list of dicts (one per run). "
            "Factor columns should be coded -1/+1."
        ),
    )
    response_column: str | None = Field(
        None,
        description="Name of the response column in design_data.",
    )
    factors_to_plot: list[str] | None = Field(
        None,
        description=(
            "Which factors to plot (2 for contour/interaction, "
            "3 for cube_plot). If omitted, inferred from data."
        ),
    )
    hold_values: dict[str, float] | None = Field(
        None,
        description=(
            "Coded values for factors not being plotted "
            "(default 0 = centre). E.g. {'C': 0.5}."
        ),
    )
    highlight_significant: bool = Field(
        True,
        description="Highlight significant effects on Pareto/half-normal plots.",
    )
    confidence_level: float = Field(
        0.95,
        description="Confidence level for thresholds (default 0.95).",
    )
    backend: Literal["both", "plotly", "echarts"] = Field(
        "both",
        description="Which rendering backend(s) to include in output.",
    )


@tool_spec(
    name="visualize_doe",
    description=(
        "Generate DOE visualisations from analysis results or design data. "
        "Supports 21 plot types: significance plots (pareto, half_normal, daniel), "
        "factor-effect plots (main_effects, interaction, perturbation), "
        "diagnostic plots (residuals_vs_fitted, normal_probability, residuals_vs_order, box_cox), "
        "response-surface plots (contour, surface_3d, prediction_variance), "
        "cube plot (cube_plot), square plot (square_plot), "
        "optimisation plots (desirability_contour, overlay, ridge_trace, steepest_ascent_path), "
        "and design-quality plots (fds_plot, power_curve). "
        "Returns both Plotly and ECharts configurations for dual-backend rendering. "
        "Pass analysis_results from fit_linear_model or analyze_experiment, or raw design_data."
    ),
    input_model=VisualizeDoeInput,
    examples="""
    # "Show me a Pareto chart of my effects"
        -> ``visualize_doe(plot_type="pareto",
                analysis_results={"effects": {"A": 5.2, "B": -3.1, "A:B": 1.0}})``

    # "Draw a contour plot of Temperature vs Pressure"
        -> ``visualize_doe(plot_type="contour",
                analysis_results={"coefficients": [...]},
                factors_to_plot=["Temperature", "Pressure"])``
    """,
    category="experiments",
)
def visualize_doe_tool(spec: VisualizeDoeInput) -> dict[str, Any]:
    """Generate a DOE visualisation."""
    try:
        from process_improve.experiments.visualization import visualize_doe  # noqa: PLC0415

        result = visualize_doe(
            plot_type=spec.plot_type,
            analysis_results=spec.analysis_results,
            design_data=spec.design_data,
            response_column=spec.response_column,
            factors_to_plot=spec.factors_to_plot,
            hold_values=spec.hold_values,
            highlight_significant=spec.highlight_significant,
            confidence_level=spec.confidence_level,
            backend=spec.backend,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool visualize_doe failed")
        return {"error": str(e)}


_register("visualize_doe")


# ---------------------------------------------------------------------------
# doe_knowledge
# ---------------------------------------------------------------------------


class DoeKnowledgeInput(BaseModel):
    """Input contract for ``doe_knowledge``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        "",
        description=(
            "Natural-language query, e.g. 'What is design resolution?' or "
            "'funnel shaped residuals'."
        ),
    )
    topic: Literal[
        "design_selection",
        "design_properties",
        "design_types",
        "analysis_methods",
        "interpretation",
        "troubleshooting",
        "diagnostics",
        "optimization",
        "statistical_concepts",
        "screening",
        "response_surface",
        "worked_examples",
        "",
    ] = Field(
        "",
        description="Restrict search to a topic. Leave empty for broad search.",
    )
    context: dict[str, Any] | None = Field(
        None,
        description=(
            "Experimental context for design-selection queries. "
            "Keys: n_factors (int), budget (int), goal ('screening'|'optimization'), "
            "sequential (bool), curvature_important (bool), has_hard_to_change (bool)."
        ),
    )
    detail_level: Literal["novice", "intermediate", "expert"] = Field(
        "intermediate",
        description="Depth of explanation: novice, intermediate, or expert.",
    )


@tool_spec(
    name="doe_knowledge",
    description=(
        "Retrieve DOE (Design of Experiments) domain knowledge: design-type descriptions, "
        "design-selection decision logic, statistical concept definitions, residual-diagnostic "
        "troubleshooting guides, interpretation guidance, and worked examples. "
        "Use this tool whenever the user asks a conceptual DOE question, needs help choosing "
        "a design, or wants to understand how to interpret DOE results."
    ),
    input_model=DoeKnowledgeInput,
    examples="""
    # "Which design should I use for 7 screening factors with a budget of 15 runs?"
        -> ``doe_knowledge(query="screening 7 factors 15 runs",
                topic="design_selection",
                context={"n_factors": 7, "budget": 15, "goal": "screening"})``

    # "What is design resolution?"
        -> ``doe_knowledge(query="What is design resolution?",
                topic="statistical_concepts")``
    """,
    category="experiments",
)
def doe_knowledge_tool(spec: DoeKnowledgeInput) -> dict[str, Any]:
    """Query the DOE knowledge graph."""
    try:
        from process_improve.experiments.knowledge import doe_knowledge  # noqa: PLC0415

        return clean(doe_knowledge(
            query=spec.query,
            topic=spec.topic,
            context=spec.context,
            detail_level=spec.detail_level,
        ))
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool doe_knowledge failed")
        return {"error": str(e)}


_register("doe_knowledge")


# ---------------------------------------------------------------------------
# recommend_strategy
# ---------------------------------------------------------------------------


class RecommendStrategyInput(BaseModel):
    """Input contract for ``recommend_strategy``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="All candidate experimental factors.",
    )
    responses: list[dict[str, Any]] | None = Field(
        None,
        description="Response variables with optimisation goals.",
    )
    budget: int | None = Field(
        None,
        ge=1,
        description="Total run budget across all stages. Omit for ideal allocation.",
    )
    constraints: list[dict[str, Any]] | None = Field(
        None,
        description="Factor-space constraints. Each entry: expression (str), optional type ('linear'|'nonlinear').",
    )
    hard_to_change_factors: list[str] | None = Field(
        None,
        description="Factor names that are expensive to reset (triggers split-plot).",
    )
    prior_knowledge: str | None = Field(
        None,
        description=(
            "Free-text description of prior knowledge, e.g. "
            "'Published literature confirms Temperature and pH are significant.' "
            "or 'No prior data - first time running this process.'"
        ),
    )
    existing_data: list[dict[str, Any]] | None = Field(
        None,
        description="Prior experimental data as list of dicts (optional).",
    )
    domain: Literal[
        "pharma_formulation",
        "fermentation",
        "food_science",
        "extraction",
        "analytical_method",
        "cell_culture",
        "bioprocess",
        "general",
    ] | None = Field(
        None,
        description="Application domain for domain-specific adjustments. Default: 'general'.",
    )
    detail_level: Literal["novice", "intermediate"] = Field(
        "intermediate",
        description="Depth of explanations in the output.",
    )


@tool_spec(
    name="recommend_strategy",
    description=(
        "Recommend a multi-stage experimental strategy given a DOE problem description. "
        "Given factors, responses, budget, constraints, domain, and prior knowledge, "
        "applies deterministic decision rules to recommend a staged experimental plan "
        "(screening then optimisation then confirmation). "
        "Returns a structured strategy with stage-by-stage design types, estimated run counts, "
        "transition rules, budget allocation, assumptions, risks, and alternative approaches. "
        "Use this when the user asks 'How should I plan my experiments?' or 'What design strategy "
        "should I use for N factors?'"
    ),
    input_model=RecommendStrategyInput,
    examples="""
    # "I have 7 factors - how do I plan my experiments?"
        -> ``recommend_strategy(factors=[{"name": "A", "low": 0, "high": 100}, ...7 factors...],
                budget=40, domain="general")``

    # "Optimize fermentation with 7 factors in ~40 runs"
        -> ``recommend_strategy(factors=[{"name": "pH", "low": 5, "high": 8}, ...],
                responses=[{"name": "Yield", "goal": "maximize"}],
                budget=40, domain="fermentation")``
    """,
    category="experiments",
)
def recommend_strategy_tool(spec: RecommendStrategyInput) -> dict[str, Any]:
    """Recommend a multi-stage experimental strategy."""
    try:
        from process_improve.experiments.factor import Constraint, Factor, Response  # noqa: PLC0415
        from process_improve.experiments.strategy import recommend_strategy  # noqa: PLC0415

        factor_objects = [Factor(**f) for f in spec.factors]
        response_objects = [Response(**r) for r in spec.responses] if spec.responses else None
        constraint_objects = (
            [Constraint(**c) for c in spec.constraints] if spec.constraints else None
        )

        df = pd.DataFrame(spec.existing_data) if spec.existing_data else None

        result = recommend_strategy(
            factors=factor_objects,
            responses=response_objects,
            budget=spec.budget,
            constraints=constraint_objects,
            hard_to_change_factors=spec.hard_to_change_factors,
            prior_knowledge=spec.prior_knowledge,
            existing_data=df,
            domain=spec.domain,
            detail_level=spec.detail_level,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool recommend_strategy failed")
        return {"error": str(e)}


_register("recommend_strategy")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_experiments_tool_specs() -> list[dict]:
    """Return tool specs for all experiments tools registered in this module."""
    return get_tool_specs(names=_EXPERIMENTS_TOOL_NAMES)
