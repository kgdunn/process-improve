# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``recommend_next_step``, the next round of a study from its results."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

Action = Literal[
    "stop",
    "confirm",
    "add_center_points",
    "replicate",
    "widen_ranges",
    "de_alias",
    "augment_to_rsm",
    "add_runs_for_model",
    "move_region",
    "investigate",
]


class RecommendNextStepInput(BaseModel):
    """Input contract for ``recommend_next_step``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "The study's factors, as for generate_design: continuous ({'name', 'low', 'high', 'units'}) or "
            "categorical at two levels ({'name', 'type': 'categorical', 'levels': [a, b]}). low and high set "
            "the region the optimum is sought in. After widen_ranges or move_region, pass the new ranges."
        ),
    )
    runs: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "Every run of the study so far, one dict per run, with each factor in its own units (a categorical "
            "factor's level) and the response. Runs with no response are left out. Example: "
            "[{'Temperature': 150, 'Time': 30, 'Yield': 61.2}, ...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response in runs.")
    goal: Literal["maximize", "minimize", "target"] = Field("maximize", description="What the study wants.")
    target: float | None = Field(None, description="The value aimed at, for goal='target'.")
    confirmation_runs: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Runs made at the settings a 'confirm' step gave, with their responses, in the same form as runs. "
            "They are judged against the model the other runs give; leave them out of runs."
        ),
    )
    model_terms: list[str] | None = Field(
        None,
        description=(
            "The model to use, as terms ('A', 'A:B', 'I(A ** 2)'). Default: every term the runs can test is "
            "judged, and the active ones kept."
        ),
    )
    budget: int | None = Field(
        None, ge=0, description="Runs left. A step that needs more falls back to confirmation runs, or to stopping."
    )
    ranges_can_move: bool = Field(
        True,
        description=(
            "Whether the next round may set factors past their ranges (widen_ranges, move_region). False keeps "
            "every run inside them, as when a range is a hard limit of the process."
        ),
    )
    n_confirmation_runs: int = Field(3, ge=1, description="Runs made to confirm an optimum.")
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level of every test the decision reads.")
    random_state: int = Field(42, description="Seed for the search for the optimum and for runs chosen by exchange.")
    block: str | None = Field(
        None,
        description=(
            "The column of runs that holds each run's block, such as the round it was made in. Every model then "
            "fits a shift for each block, so a process that drifted between the rounds does not bias the effects; "
            "the next runs are predicted as the process was in the latest block."
        ),
    )
    history: list[Action] | None = Field(
        None,
        description=(
            "The actions of the earlier rounds, oldest first, as this tool returned them. Pass them: they stop a "
            "study in which nothing stands out from widening the ranges a second time, and a second missed "
            "confirmation from being repeated."
        ),
    )


@tool_spec(
    name="recommend_next_step",
    description=(
        "Decide the next round of a designed experiment from the results so far: call it after every round with "
        "every run made, and do what it says. Returns 'action' (stop, confirm, add_center_points, replicate, "
        "widen_ranges, de_alias, augment_to_rsm, add_runs_for_model, move_region or investigate), a 'headline', "
        "'reasons' with the numbers behind them, 'runs' (the next round, in the factors' own units; add the "
        "response column as each is made), 'after' (how to use those runs), the model it read, the evidence "
        "(active terms, curvature, lack of fit, aliasing, the predicted optimum with its interval) and "
        "'alternatives'. The loop: append the new runs to runs and call again; for 'confirm', pass them as "
        "confirmation_runs instead; for 'widen_ranges' and 'move_region', change the factors' ranges as the "
        "reply says and start the runs afresh; 'stop' and 'investigate' end it. Every test it reads is at alpha, "
        "including the test of whether anything stands out at all. Use it instead of reading analyze_experiment's "
        "output to choose the next design by hand."
    ),
    input_model=RecommendNextStepInput,
    examples="""
    # "I ran the screening design; what now?"
        -> ``recommend_next_step(factors=[...], runs=[{"A": -1, ..., "y": 101.2}, ...], response_column="y")``

    # "I made the three confirmation runs it asked for"
        -> ``recommend_next_step(factors=[...], runs=[...], response_column="y",
                confirmation_runs=[{"A": 1, ..., "y": 130.4}, ...], history=["confirm"])``

    # "The temperature range is a hard limit, and I have 10 runs left"
        -> ``recommend_next_step(..., ranges_can_move=False, budget=10)``
    """,
    category="experiments",
)
def recommend_next_step_tool(spec: RecommendNextStepInput) -> dict[str, Any]:
    """Recommend the next round of a study."""
    try:
        from process_improve.experiments.factor import Factor  # noqa: PLC0415
        from process_improve.experiments.next_step import recommend_next_step  # noqa: PLC0415

        confirmation = pd.DataFrame(spec.confirmation_runs) if spec.confirmation_runs else None
        result = recommend_next_step(
            [Factor(**f) for f in spec.factors],
            pd.DataFrame(spec.runs),
            spec.response_column,
            goal=spec.goal,
            target=spec.target,
            confirmation_runs=confirmation,
            model_terms=spec.model_terms,
            budget=spec.budget,
            ranges_can_move=spec.ranges_can_move,
            n_confirmation_runs=spec.n_confirmation_runs,
            alpha=spec.alpha,
            random_state=spec.random_state,
            history=list(spec.history or []),
            block=spec.block,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool recommend_next_step failed")
        return {"error": str(e)}


_register("recommend_next_step")
