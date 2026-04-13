"""(c) Kevin Dunn, 2010-2025. MIT License.

Agent-callable tool wrappers for designed experiments.

Each function in this module is decorated with ``@tool_spec`` so it can be
passed directly to an LLM tool-use API (e.g. Anthropic ``tools=``).
The wrappers accept plain JSON-serialisable inputs (lists of dicts, strings,
integers) and always return JSON-serialisable ``dict`` results.

Import all specs at once::

    from process_improve.experiments.tools import get_experiments_tool_specs
    # or get everything registered so far
    from process_improve.tool_spec import get_tool_specs

Dispatch a tool call returned by the model::

    from process_improve.tool_spec import execute_tool_call
    result = execute_tool_call(block.name, block.input)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPERIMENTS_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _EXPERIMENTS_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_spec(
    name="create_factorial_design",
    description=(
        "Create a full factorial (2^k) experimental design with factors coded as -1 and +1. "
        "Returns the design matrix as a list of dictionaries (one per run), the number of runs, "
        "the number of factors, and the factor names. "
        "Use this when planning a designed experiment to systematically explore the effect of "
        "2 to 10 factors, each at two levels."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "n_factors": {
                    "type": "integer",
                    "description": "Number of factors in the design (2 to 10).",
                    "minimum": 2,
                    "maximum": 10,
                },
                "factor_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of factor names. Must have exactly n_factors entries. "
                        "If not provided, factors are named A, B, C, ..."
                    ),
                },
            },
            "required": ["n_factors"],
        }
    },
    examples="""
    # "Create a 2-factor full factorial design"
        -> ``create_factorial_design(n_factors=2)``

    # "Create a 3-factor design with named factors"
        -> ``create_factorial_design(n_factors=3, factor_names=["Temperature", "Pressure", "Time"])``
    """,
    category="experiments",
)
def create_factorial_design(
    *,
    n_factors: int,
    factor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Create a full factorial design; see tool spec for details."""
    try:
        from process_improve.experiments.designs_factorial import full_factorial  # noqa: PLC0415

        columns = full_factorial(n_factors, names=factor_names)
        # full_factorial returns a list of Series; combine into a DataFrame
        design = pd.concat(columns, axis=1)
        names = list(design.columns)
        return clean(
            {
                "design": design.to_dict(orient="records"),
                "n_runs": len(design),
                "n_factors": n_factors,
                "factor_names": names,
            }
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("create_factorial_design")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": (
                        "Model formula in Wilkinson notation, e.g. 'y ~ A*B*C'. "
                        "The left-hand side is the response variable name, the right-hand "
                        "side specifies the terms. '*' expands to main effects and all "
                        "interactions; ':' denotes a specific interaction; '+' adds terms."
                    ),
                },
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "description": (
                        "List of dictionaries, one per experimental run. Each dict must "
                        "contain keys for every factor and the response variable referenced "
                        "in the formula. Example: [{'A': -1, 'B': -1, 'y': 45.2}, ...]"
                    ),
                    "minItems": 2,
                },
            },
            "required": ["formula", "data"],
        }
    },
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
def fit_linear_model(
    *,
    formula: str,
    data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit a linear model to experimental data; see tool spec for details."""
    try:
        from process_improve.experiments.models import lm  # noqa: PLC0415
        from process_improve.experiments.structures import Expt  # noqa: PLC0415

        df = pd.DataFrame(data)
        expt_data = Expt(df)
        expt_data.pi_title = None
        expt_data.pi_source = None
        expt_data.pi_units = None

        model = lm(formula, expt_data)

        # Get coefficients
        params = model.get_parameters(drop_intercept=True)
        if isinstance(params, pd.Series):
            coefficients = [
                {"name": str(name), "estimate": float(value)}
                for name, value in params.items()
            ]
        else:
            coefficients = params.to_dict(orient="records")

        # Get R-squared from the underlying OLS model
        r2 = float(model._OLS.rsquared)

        # Get summary text
        smry = model.summary(print_to_screen=False)
        summary_text = str(smry)

        return clean(
            {
                "coefficients": coefficients,
                "r2": r2,
                "summary_text": summary_text,
            }
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("fit_linear_model")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "factors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Factor name (e.g. 'Temperature')."},
                            "type": {
                                "type": "string",
                                "enum": ["continuous", "categorical", "mixture"],
                                "description": "Factor type. Default: 'continuous'.",
                            },
                            "low": {"type": "number", "description": "Low level (required for continuous)."},
                            "high": {"type": "number", "description": "High level (required for continuous)."},
                            "levels": {
                                "type": "array",
                                "description": "Explicit levels (required for categorical).",
                            },
                            "units": {"type": "string", "description": "Engineering units (optional)."},
                        },
                        "required": ["name"],
                    },
                    "description": "List of factor specifications.",
                    "minItems": 1,
                },
                "design_type": {
                    "type": "string",
                    "enum": [
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
                    ],
                    "description": (
                        "Design type. If omitted, auto-selected based on factors and budget."
                    ),
                },
                "budget": {
                    "type": "integer",
                    "description": "Maximum number of experimental runs.",
                    "minimum": 1,
                },
                "center_points": {
                    "type": "integer",
                    "description": "Number of center point replicates (default: 3).",
                    "minimum": 0,
                },
                "replicates": {
                    "type": "integer",
                    "description": "Number of full replicates (default: 1).",
                    "minimum": 1,
                },
                "resolution": {
                    "type": "integer",
                    "description": "Minimum resolution for fractional factorials (3, 4, or 5).",
                    "minimum": 3,
                    "maximum": 5,
                },
                "alpha": {
                    "type": "string",
                    "enum": ["rotatable", "face_centered", "orthogonal"],
                    "description": "Axial distance for CCD designs.",
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Seed for reproducible randomization (default: 42).",
                },
            },
            "required": ["factors"],
        }
    },
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
def generate_design_tool(  # noqa: PLR0913
    *,
    factors: list[dict[str, Any]],
    design_type: str | None = None,
    budget: int | None = None,
    center_points: int = 3,
    replicates: int = 1,
    resolution: int | None = None,
    alpha: str | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Generate an experimental design; see tool spec for details."""
    try:
        from process_improve.experiments.designs import generate_design  # noqa: PLC0415
        from process_improve.experiments.factor import Factor  # noqa: PLC0415

        # Convert raw dicts to Factor objects
        factor_objects = [Factor(**f) for f in factors]

        result = generate_design(
            factors=factor_objects,
            design_type=design_type,
            budget=budget,
            center_points=center_points,
            replicates=replicates,
            resolution=resolution,
            alpha=alpha,
            random_seed=random_seed,
        )

        # Build JSON-serializable output
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
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("generate_design")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "design_matrix": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "description": (
                        "List of dictionaries, one per experimental run. Each dict maps "
                        "factor name to coded value. Example: [{'A': -1, 'B': -1}, ...]"
                    ),
                    "minItems": 2,
                },
                "model": {
                    "type": "string",
                    "enum": ["main_effects", "interactions", "quadratic"],
                    "description": (
                        "Model type to evaluate against. 'main_effects' = main effects only, "
                        "'interactions' = main effects + 2-factor interactions (default), "
                        "'quadratic' = interactions + squared terms."
                    ),
                },
                "metric": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": [
                                "d_efficiency",
                                "i_efficiency",
                                "g_efficiency",
                                "prediction_variance",
                                "vif",
                                "condition_number",
                                "power",
                                "degrees_of_freedom",
                                "alias_structure",
                                "confounding",
                                "resolution",
                                "defining_relation",
                                "clear_effects",
                                "minimum_aberration",
                            ],
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    ],
                    "description": (
                        "One or more metric names to compute. Default: 'd_efficiency'."
                    ),
                },
                "effect_size": {
                    "type": "number",
                    "description": "Expected effect size for power calculation.",
                },
                "alpha": {
                    "type": "number",
                    "description": "Significance level (default 0.05).",
                },
                "sigma": {
                    "type": "number",
                    "description": "Estimated noise standard deviation.",
                },
            },
            "required": ["design_matrix", "metric"],
        }
    },
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
def evaluate_design_tool(  # noqa: PLR0913
    *,
    design_matrix: list[dict[str, Any]],
    model: str | None = None,
    metric: str | list[str] = "d_efficiency",
    effect_size: float | None = None,
    alpha: float = 0.05,
    sigma: float | None = None,
) -> dict[str, Any]:
    """Evaluate design quality; see tool spec for details."""
    try:
        from process_improve.experiments.evaluate import evaluate_design  # noqa: PLC0415

        df = pd.DataFrame(design_matrix)
        result = evaluate_design(
            df,
            model=model,
            metric=metric,
            effect_size=effect_size,
            alpha=alpha,
            sigma=sigma,
        )
        return clean(result)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("evaluate_design")


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
        "Always returns a model summary with R², adj-R², pred-R², and adequate precision. "
        "The design_matrix should contain factor columns with coded values (-1/+1). "
        "The response can be in a separate column or included in design_matrix."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "design_matrix": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": {"type": "number"}},
                    "description": (
                        "List of dicts, one per run. Must contain factor columns and "
                        "optionally the response column. Example: "
                        "[{'A': -1, 'B': -1, 'y': 28}, {'A': 1, 'B': -1, 'y': 36}, ...]"
                    ),
                    "minItems": 2,
                },
                "response_column": {
                    "type": "string",
                    "description": "Name of the response column in the design_matrix.",
                },
                "model": {
                    "type": "string",
                    "enum": ["main_effects", "interactions", "quadratic"],
                    "description": (
                        "Model type. 'main_effects' = main effects only, "
                        "'interactions' = main effects + 2FI (default), "
                        "'quadratic' = interactions + squared terms."
                    ),
                },
                "analysis_type": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": [
                                "anova",
                                "effects",
                                "coefficients",
                                "significance",
                                "residual_diagnostics",
                                "lack_of_fit",
                                "curvature_test",
                                "model_selection",
                                "box_cox",
                                "lenth_method",
                                "confidence_intervals",
                                "prediction",
                                "confirmation_test",
                            ],
                        },
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "One or more analysis types to run. Default: 'anova'.",
                },
                "significance_level": {
                    "type": "number",
                    "description": "Significance level (default 0.05).",
                },
                "transform": {
                    "type": "string",
                    "enum": ["log", "sqrt", "inverse", "box_cox"],
                    "description": "Optional response transform before fitting.",
                },
                "new_points": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": {"type": "number"}},
                    "description": "New factor settings for prediction or confirmation.",
                },
                "observed_at_new": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observed values at new_points (for confirmation testing).",
                },
            },
            "required": ["design_matrix", "response_column"],
        }
    },
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
def analyze_experiment_tool(  # noqa: PLR0913
    *,
    design_matrix: list[dict[str, Any]],
    response_column: str,
    model: str | None = None,
    analysis_type: str | list[str] = "anova",
    significance_level: float = 0.05,
    transform: str | None = None,
    new_points: list[dict[str, Any]] | None = None,
    observed_at_new: list[float] | None = None,
) -> dict[str, Any]:
    """Analyze experimental data; see tool spec for details."""
    try:
        from process_improve.experiments.analysis import analyze_experiment  # noqa: PLC0415

        df = pd.DataFrame(design_matrix)
        np_df = pd.DataFrame(new_points) if new_points else None

        result = analyze_experiment(
            design_matrix=df,
            response_column=response_column,
            model=model,
            analysis_type=analysis_type,
            significance_level=significance_level,
            transform=transform,
            new_points=np_df,
            observed_at_new=observed_at_new,
        )
        return clean(result)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("analyze_experiment")


@tool_spec(
    name="optimize_responses",
    description=(
        "Find optimal factor settings for one or multiple responses from fitted experimental models. "
        "Supports several methods: 'desirability' (Derringer-Suich desirability functions for single or "
        "multi-response optimisation), 'steepest_ascent' / 'steepest_descent' (move along the gradient "
        "of a first-order model), 'stationary_point' (locate the optimum of a second-order model), "
        "'canonical_analysis' (eigenvalue decomposition to classify the response surface shape). "
        "Ridge analysis and Pareto front are planned but not yet implemented. "
        "Each fitted_model must include coefficients (as returned by analyze_experiment with "
        "analysis_type='coefficients'), factor_names, and response_name. "
        "For desirability, each goal specifies whether to maximize, minimize, or target a value."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "fitted_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "response_name": {
                                "type": "string",
                                "description": "Name of the response variable.",
                            },
                            "coefficients": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "term": {"type": "string"},
                                        "coefficient": {"type": "number"},
                                    },
                                    "required": ["term", "coefficient"],
                                },
                                "description": (
                                    "List of model coefficients as returned by "
                                    "analyze_experiment(analysis_type='coefficients'). "
                                    "Each entry has 'term' (e.g. 'Intercept', 'A', 'A:B', "
                                    "'I(A ** 2)') and 'coefficient' (float)."
                                ),
                            },
                            "factor_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of factor names.",
                            },
                            "mse_residual": {
                                "type": "number",
                                "description": "Mean squared error of the model (optional).",
                            },
                            "r_squared": {
                                "type": "number",
                                "description": "R-squared of the model (optional).",
                            },
                        },
                        "required": ["coefficients", "factor_names"],
                    },
                    "description": "One or more fitted models from analyze_experiment.",
                    "minItems": 1,
                },
                "goals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "Response name (must match a fitted_model).",
                            },
                            "goal": {
                                "type": "string",
                                "enum": ["maximize", "minimize", "target"],
                                "description": "Optimisation direction.",
                            },
                            "target": {
                                "type": "number",
                                "description": "Target value (required when goal='target').",
                            },
                            "low": {
                                "type": "number",
                                "description": "Lower acceptable bound for desirability.",
                            },
                            "high": {
                                "type": "number",
                                "description": "Upper acceptable bound for desirability.",
                            },
                            "weight": {
                                "type": "number",
                                "description": "Desirability shape parameter (default 1.0 = linear).",
                            },
                            "importance": {
                                "type": "number",
                                "description": "Relative importance for composite desirability.",
                            },
                        },
                        "required": ["response", "goal", "low", "high"],
                    },
                    "description": (
                        "Per-response optimisation goals. Required for 'desirability' method."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": [
                        "desirability",
                        "steepest_ascent",
                        "steepest_descent",
                        "stationary_point",
                        "canonical_analysis",
                        "ridge_analysis",
                        "pareto_front",
                    ],
                    "description": "Optimisation method (default: 'desirability').",
                },
                "factor_ranges": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                        },
                        "required": ["low", "high"],
                    },
                    "description": (
                        "Factor bounds in actual units, e.g. "
                        '{"Temperature": {"low": 150, "high": 200}}. '
                        "Used to convert coded settings to actual units in the output."
                    ),
                },
                "step_size": {
                    "type": "number",
                    "description": "Step size in coded units for steepest ascent/descent (default 0.5).",
                },
                "n_steps": {
                    "type": "integer",
                    "description": "Number of steps for steepest ascent/descent (default 10).",
                    "minimum": 1,
                },
                "desirability_weights": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Importance weights for composite desirability (overrides per-goal importance).",
                },
            },
            "required": ["fitted_models"],
        }
    },
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
def optimize_responses_tool(  # noqa: PLR0913
    *,
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]] | None = None,
    method: str = "desirability",
    factor_ranges: dict[str, dict[str, float]] | None = None,
    step_size: float = 0.5,
    n_steps: int = 10,
    desirability_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Optimize experimental responses; see tool spec for details."""
    try:
        from process_improve.experiments.optimization import optimize_responses  # noqa: PLC0415

        result = optimize_responses(
            fitted_models=fitted_models,
            goals=goals,
            method=method,
            factor_ranges=factor_ranges,
            step_size=step_size,
            n_steps=n_steps,
            desirability_weights=desirability_weights,
        )
        return clean(result)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("optimize_responses")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "existing_design": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "description": (
                        "Current design matrix as list of dicts with factor names as keys "
                        "and coded values (-1/+1) as values. "
                        "Example: [{'A': -1, 'B': -1}, {'A': 1, 'B': -1}, ...]"
                    ),
                    "minItems": 2,
                },
                "augmentation_type": {
                    "type": "string",
                    "enum": [
                        "foldover",
                        "semifold",
                        "add_center_points",
                        "add_axial_points",
                        "add_runs_optimal",
                        "upgrade_to_rsm",
                        "add_blocks",
                        "replicate",
                    ],
                    "description": "Type of augmentation to apply to the design.",
                },
                "target_model": {
                    "type": "string",
                    "enum": ["main_effects", "interactions", "quadratic"],
                    "description": (
                        "Desired model after augmentation. Used by 'add_runs_optimal' "
                        "and 'upgrade_to_rsm'. Default: 'interactions'."
                    ),
                },
                "n_additional_runs": {
                    "type": "integer",
                    "description": (
                        "Budget for additional runs. Interpretation depends on type: "
                        "number of center points, D-optimal runs, replicates, or blocks."
                    ),
                    "minimum": 1,
                },
                "fold_on": {
                    "type": "string",
                    "description": (
                        "Factor name to fold on (semifold only). "
                        "If omitted, the best factor is auto-selected."
                    ),
                },
                "alpha": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": ["rotatable", "face_centered", "orthogonal"],
                        },
                        {"type": "number"},
                    ],
                    "description": (
                        "Axial distance for add_axial_points or upgrade_to_rsm. "
                        "'rotatable', 'face_centered', 'orthogonal', or a numeric value."
                    ),
                },
                "generators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Generator strings from the original fractional factorial design "
                        "(e.g. ['D=ABC']). Needed for foldover/semifold alias analysis."
                    ),
                },
            },
            "required": ["existing_design", "augmentation_type"],
        }
    },
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
def augment_design_tool(  # noqa: PLR0913
    *,
    existing_design: list[dict[str, Any]],
    augmentation_type: str,
    target_model: str | None = None,
    n_additional_runs: int | None = None,
    fold_on: str | None = None,
    alpha: str | float | None = None,
    generators: list[str] | None = None,
) -> dict[str, Any]:
    """Augment an existing design; see tool spec for details."""
    try:
        from process_improve.experiments.augment import augment_design  # noqa: PLC0415

        df = pd.DataFrame(existing_design)
        result = augment_design(
            existing_design=df,
            augmentation_type=augmentation_type,
            target_model=target_model,
            n_additional_runs=n_additional_runs,
            fold_on=fold_on,
            alpha=alpha,
            generators=generators,
        )
        return clean(result)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("augment_design")


# ---------------------------------------------------------------------------
# Tool 6: Visualise DOE
# ---------------------------------------------------------------------------


@tool_spec(
    name="visualize_doe",
    description=(
        "Generate DOE visualisations from analysis results or design data. "
        "Supports 20 plot types: significance plots (pareto, half_normal, daniel), "
        "factor-effect plots (main_effects, interaction, perturbation), "
        "diagnostic plots (residuals_vs_fitted, normal_probability, residuals_vs_order, box_cox), "
        "response-surface plots (contour, surface_3d, prediction_variance), "
        "cube plot (cube_plot), "
        "optimisation plots (desirability_contour, overlay, ridge_trace, steepest_ascent_path), "
        "and design-quality plots (fds_plot, power_curve). "
        "Returns both Plotly and ECharts configurations for dual-backend rendering. "
        "Pass analysis_results from fit_linear_model or analyze_experiment, or raw design_data."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "pareto", "half_normal", "daniel",
                        "main_effects", "interaction", "perturbation",
                        "residuals_vs_fitted", "normal_probability",
                        "residuals_vs_order", "box_cox",
                        "contour", "surface_3d", "prediction_variance",
                        "cube_plot",
                        "desirability_contour", "overlay",
                        "ridge_trace", "steepest_ascent_path",
                        "fds_plot", "power_curve",
                    ],
                    "description": "Type of DOE plot to generate.",
                },
                "analysis_results": {
                    "type": "object",
                    "description": (
                        "Results from fit_linear_model or analyze_experiment. "
                        "Should contain keys like 'coefficients', 'effects', "
                        "'residual_diagnostics', 'lenth_method', 'model_summary'."
                    ),
                },
                "design_data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Raw design matrix as a list of dicts (one per run). "
                        "Factor columns should be coded -1/+1."
                    ),
                },
                "response_column": {
                    "type": "string",
                    "description": "Name of the response column in design_data.",
                },
                "factors_to_plot": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Which factors to plot (2 for contour/interaction, "
                        "3 for cube_plot). If omitted, inferred from data."
                    ),
                },
                "hold_values": {
                    "type": "object",
                    "description": (
                        "Coded values for factors not being plotted "
                        "(default 0 = centre). E.g. {'C': 0.5}."
                    ),
                },
                "highlight_significant": {
                    "type": "boolean",
                    "description": "Highlight significant effects on Pareto/half-normal plots.",
                    "default": True,
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for thresholds (default 0.95).",
                    "default": 0.95,
                },
                "backend": {
                    "type": "string",
                    "enum": ["both", "plotly", "echarts"],
                    "description": "Which rendering backend(s) to include in output.",
                    "default": "both",
                },
            },
            "required": ["plot_type"],
        }
    },
    examples="""
    # "Show me a Pareto chart of my effects"
        -> ``visualize_doe(plot_type="pareto",
                analysis_results={"effects": {"A": 5.2, "B": -3.1, "A:B": 1.0}})``

    # "Draw a contour plot of Temperature vs Pressure"
        -> ``visualize_doe(plot_type="contour",
                analysis_results={"coefficients": [...]},
                factors_to_plot=["Temperature", "Pressure"])``

    # "Show residuals vs fitted values"
        -> ``visualize_doe(plot_type="residuals_vs_fitted",
                analysis_results={"residual_diagnostics":
                    {"residuals": [...], "fitted_values": [...]}})``

    # "Create a cube plot for my 3-factor design"
        -> ``visualize_doe(plot_type="cube_plot",
                analysis_results={"coefficients": [...]},
                factors_to_plot=["A", "B", "C"])``
    """,
    category="experiments",
)
def visualize_doe_tool(  # noqa: PLR0913
    *,
    plot_type: str,
    analysis_results: dict[str, Any] | None = None,
    design_data: list[dict[str, Any]] | None = None,
    response_column: str | None = None,
    factors_to_plot: list[str] | None = None,
    hold_values: dict[str, float] | None = None,
    highlight_significant: bool = True,
    confidence_level: float = 0.95,
    backend: str = "both",
) -> dict[str, Any]:
    """Generate a DOE visualisation; see tool spec for details."""
    try:
        from process_improve.experiments.visualization import visualize_doe  # noqa: PLC0415

        result = visualize_doe(
            plot_type=plot_type,
            analysis_results=analysis_results,
            design_data=design_data,
            response_column=response_column,
            factors_to_plot=factors_to_plot,
            hold_values=hold_values,
            highlight_significant=highlight_significant,
            confidence_level=confidence_level,
            backend=backend,
        )
        return clean(result)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("visualize_doe")


# ---------------------------------------------------------------------------
# Tool 7 – doe_knowledge
# ---------------------------------------------------------------------------


@tool_spec(
    name="doe_knowledge",
    description=(
        "Retrieve DOE (Design of Experiments) domain knowledge: design-type descriptions, "
        "design-selection decision logic, statistical concept definitions, residual-diagnostic "
        "troubleshooting guides, interpretation guidance, and worked examples. "
        "Use this tool whenever the user asks a conceptual DOE question, needs help choosing "
        "a design, or wants to understand how to interpret DOE results."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query, e.g. 'What is design resolution?' or "
                        "'funnel shaped residuals'."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Restrict search to a topic: design_selection, design_properties, "
                        "design_types, analysis_methods, interpretation, troubleshooting, "
                        "diagnostics, optimization, statistical_concepts, screening, "
                        "response_surface, worked_examples.  Leave empty for broad search."
                    ),
                    "enum": [
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
                    ],
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Experimental context for design-selection queries. "
                        "Keys: n_factors (int), budget (int), goal ('screening'|'optimization'), "
                        "sequential (bool), curvature_important (bool), has_hard_to_change (bool)."
                    ),
                },
                "detail_level": {
                    "type": "string",
                    "description": "Depth of explanation: novice, intermediate, or expert.",
                    "enum": ["novice", "intermediate", "expert"],
                    "default": "intermediate",
                },
            },
            "required": [],
        }
    },
    examples="""
    # "Which design should I use for 7 screening factors with a budget of 15 runs?"
        -> ``doe_knowledge(query="screening 7 factors 15 runs",
                topic="design_selection",
                context={"n_factors": 7, "budget": 15, "goal": "screening"})``

    # "What is design resolution?"
        -> ``doe_knowledge(query="What is design resolution?",
                topic="statistical_concepts")``

    # "My residuals look like a funnel"
        -> ``doe_knowledge(query="funnel shaped residuals",
                topic="troubleshooting")``

    # "Compare Box-Behnken and CCD"
        -> ``doe_knowledge(query="Box-Behnken CCD comparison",
                topic="design_types")``
    """,
    category="experiments",
)
def doe_knowledge_tool(
    *,
    query: str = "",
    topic: str = "",
    context: dict[str, Any] | None = None,
    detail_level: str = "intermediate",
) -> dict[str, Any]:
    """Query the DOE knowledge graph; see tool spec for details."""
    try:
        from process_improve.experiments.knowledge import doe_knowledge  # noqa: PLC0415

        return clean(doe_knowledge(
            query=query,
            topic=topic,
            context=context,
            detail_level=detail_level,
        ))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


_register("doe_knowledge")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_experiments_tool_specs() -> list[dict]:
    """Return tool specs for all experiments tools registered in this module."""
    return get_tool_specs(names=_EXPERIMENTS_TOOL_NAMES)
