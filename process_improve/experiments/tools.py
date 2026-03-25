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


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_experiments_tool_specs() -> list[dict]:
    """Return tool specs for all experiments tools registered in this module."""
    return get_tool_specs(names=_EXPERIMENTS_TOOL_NAMES)
