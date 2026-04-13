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


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_experiments_tool_specs() -> list[dict]:
    """Return tool specs for all experiments tools registered in this module."""
    return get_tool_specs(names=_EXPERIMENTS_TOOL_NAMES)
