# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``analyze_omars``, the staged analysis of a DSD or OMARS design."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: Columns a design carries that are not factors.
_NOT_FACTORS = frozenset({"RunOrder", "Block"})


class AnalyzeOmarsInput(BaseModel):
    """Input contract for ``analyze_omars``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=3,
        description=(
            "One dict per run with the factor settings, coded or in actual units (each factor is coded from its "
            "own lowest and highest value), and the response. Example: [{'A': -1, 'B': 0, 'C': 1, 'y': 12.1}, ...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response column in design_matrix.")
    factor_names: list[str] | None = Field(
        None,
        description="The factor columns. Default: every column except the response, RunOrder and Block.",
    )
    alpha_main: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level of the main-effect t-tests.")
    alpha_second_overall: float = Field(
        0.20, gt=0.0, lt=1.0, description="Significance level of the overall F-test that opens the second-order search."
    )
    alpha_second_subset: float = Field(
        0.20, gt=0.0, lt=1.0, description="Significance level of the F-tests that end the second-order subset search."
    )
    quadratic_heredity: Literal["none", "strong"] = Field(
        "none", description="'strong' considers the square of a factor only when its main effect is active."
    )
    interaction_heredity: Literal["none", "weak", "strong"] = Field(
        "none",
        description=(
            "'strong' considers an interaction only when both its factors' main effects are active, 'weak' when "
            "at least one is."
        ),
    )
    effects_to_drop: list[str] | None = Field(
        None, description="Second-order terms to leave out of the search, e.g. ['A^2', 'B:C']."
    )
    force_main_effects: list[str] | None = Field(
        None, description="Factors to treat as active whatever their test says."
    )
    second_order: bool = Field(True, description="False stops after the main effects.")
    max_subset_terms: int | None = Field(
        None, ge=0, description="Most second-order terms the subset search may hold. Default: chosen from the design."
    )


def _model_formula(response: str, result: Any) -> str | None:  # noqa: ANN401
    """Write the selected model as an ``analyze_experiment`` formula, with the main effects of its terms' factors.

    None when a name is not a Python identifier, which a formula cannot hold bare.
    """
    squares = [term.removesuffix("^2") for term in result.active_quadratics]
    pairs = [term.split(":") for term in result.active_interactions]
    mains = list(dict.fromkeys([*result.active_main_effects, *squares, *(name for pair in pairs for name in pair)]))
    if not all(name.isidentifier() for name in [response, *mains]):
        return None
    terms = [*mains, *(":".join(pair) for pair in pairs), *(f"I({name} ** 2)" for name in squares)]
    return f"{response} ~ {' + '.join(terms)}" if terms else f"{response} ~ 1"


@tool_spec(
    name="analyze_omars",
    description=(
        "Analyse the results of a definitive screening design (DSD) or an OMARS design with the staged protocol "
        "these designs are built for (Jones and Nachtsheim 2017; Hameed, Nunez Ares and Goos 2023). Their main "
        "effects are orthogonal to every two-factor interaction and square, so the main effects are tested "
        "first, against an error estimate from what the full second-order model cannot fit; the inactive main "
        "effects are pooled into that estimate; then an F-test decides whether any second-order term is "
        "active, and a best-subset search picks the interactions and squares. Prefer it to analyze_experiment's "
        "stepwise search for these designs. Returns the active main effects with their p-values, the active "
        "interactions and squares, and 'model_formula', the selected model with its factors' main effects, to "
        "pass to analyze_experiment (analysis_type='coefficients', coding='coded') and then optimize_responses. "
        "success is False when the design leaves no degrees of freedom for error (a DSD of an even number of "
        "factors with one centre run): add centre runs."
    ),
    input_model=AnalyzeOmarsInput,
    examples="""
    # "Which factors matter in my 8-factor DSD?"
        -> ``analyze_omars(design_matrix=[{"A": -1, "B": 0, ..., "y": 101.2}, ...], response_column="y")``

    # "Same, but only consider squares and interactions of the active factors"
        -> ``analyze_omars(design_matrix=[...], response_column="y",
                quadratic_heredity="strong", interaction_heredity="strong")``
    """,
    category="experiments",
)
def analyze_omars_tool(spec: AnalyzeOmarsInput) -> dict[str, Any]:
    """Run the staged DSD / OMARS analysis."""
    try:
        from process_improve.experiments.omars import analyze_omars  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        if spec.response_column not in df.columns:
            raise ValueError(f"response_column {spec.response_column!r} is not a column of design_matrix.")
        names = spec.factor_names or [c for c in df.columns if c != spec.response_column and c not in _NOT_FACTORS]
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"factor_names {missing} are not columns of design_matrix.")
        result = analyze_omars(
            df[names],
            df[spec.response_column],
            alpha_main=spec.alpha_main,
            alpha_second_overall=spec.alpha_second_overall,
            alpha_second_subset=spec.alpha_second_subset,
            quadratic_heredity=spec.quadratic_heredity,
            interaction_heredity=spec.interaction_heredity,
            effects_to_drop=spec.effects_to_drop,
            force_main_effects=spec.force_main_effects,
            second_order=spec.second_order,
            max_subset_terms=spec.max_subset_terms,
        )
        output = {"factor_names": names, "response_name": spec.response_column, **dataclasses.asdict(result)}
        output["model_formula"] = _model_formula(spec.response_column, result) if result.success else None
        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool analyze_omars failed")
        return {"error": str(e)}


_register("analyze_omars")
