# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``suggest_next_runs``, the next runs from a model of every run so far (#1374)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: The most runs the model is fitted to: its cost grows with the cube of the runs, about 3 s at 200.
MAX_TOOL_RUNS = 200

#: Below this improvement ratio little is left to gain, and confirmation runs are the next step (Jones et al., 1998).
LITTLE_TO_GAIN = 0.01


class SuggestNextRunsInput(BaseModel):
    """Input contract for ``suggest_next_runs``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "The study's numeric factors, as for generate_design: {'name', 'low', 'high', 'units'}. low and high "
            "bound where a run may be made."
        ),
    )
    runs: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            f"Every run so far, at most {MAX_TOOL_RUNS}, one dict per run: each factor in its own units and the "
            "response. Example: [{'Temperature': 150, 'Time': 30, 'Yield': 61.2}, ...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response in runs.")
    goal: Literal["maximize", "minimize", "target"] = Field(
        "maximize", description="Whether a larger or a smaller response is better, or one closer to `target`."
    )
    target: float | None = Field(None, description="The response to reach, with goal 'target' only.")
    constraints: list[str] | None = Field(
        None,
        description='Limits on the settings in actual units, as for generate_design, e.g. ["T + 25*P <= 250"].',
    )
    n_runs: int = Field(
        1,
        ge=1,
        le=50,
        description="How many runs to propose, to be made before any result is read: a batch for a day's work.",
    )
    noise: bool = Field(
        True,
        description=(
            "Fit a noise term, as a physical experiment's runs need. False for a computer experiment, which gives "
            "the same output at the same inputs."
        ),
    )
    random_state: int = Field(42, description="Seed for the candidate settings and the model's fit.")


@tool_spec(
    name="suggest_next_runs",
    description=(
        "Propose the next runs of a study from a model of every run so far: a Gaussian process, and the settings "
        "where the expected improvement over the best result so far is largest. It suits a response that no "
        "quadratic fits, a computer experiment, and runs made one at a time or in small batches; for a designed "
        "experiment in rounds, recommend_next_step decides the next design. Returns 'runs' (each factor in its "
        "own units, with the 'predicted' response, its 'std_error' and the 'expected_improvement'), 'best' (the "
        "best so far: the best predicted response at the runs made, with noise; for a target, the one closest to "
        "it), 'improvement_ratio' and "
        "'little_to_gain' (true below 1%: then make confirmation runs at the best settings instead), and the "
        "model's 'theta' and 'noise_ratio'. The loop: make the runs, append them with their results to runs, "
        "and call again. Every run keeps the factors' ranges and the constraints."
    ),
    input_model=SuggestNextRunsInput,
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 42},
    examples="""
    # "Where should I run next to raise the yield?"
        -> ``suggest_next_runs(factors=[{"name": "T", "low": 150, "high": 200}, ...],
                runs=[{"T": 160, "P": 1.5, "Yield": 61.2}, ...], response_column="Yield")``

    # "Four runs for tomorrow, keeping T + 25*P <= 250"
        -> ``suggest_next_runs(..., n_runs=4, constraints=["T + 25*P <= 250"])``

    # "Where next, to bring the viscosity to 1200?"
        -> ``suggest_next_runs(..., response_column="Viscosity", goal="target", target=1200)``
    """,
    category="experiments",
)
def suggest_next_runs_tool(spec: SuggestNextRunsInput) -> dict[str, Any]:
    """Propose the next runs from a Gaussian process of every run so far."""
    try:
        from process_improve.experiments.expected_improvement import suggest_next_runs  # noqa: PLC0415
        from process_improve.experiments.factor import Factor  # noqa: PLC0415
        from process_improve.experiments.region import DesignRegion  # noqa: PLC0415

        if len(spec.runs) > MAX_TOOL_RUNS:
            return {"error": f"runs has {len(spec.runs)} runs; the model takes at most {MAX_TOOL_RUNS}."}
        factors = [Factor(**f) for f in spec.factors]
        found = suggest_next_runs(
            factors,
            pd.DataFrame(spec.runs),
            spec.response_column,
            goal=spec.goal,
            target=spec.target,
            region=DesignRegion(factors, spec.constraints) if spec.constraints else None,
            n_runs=spec.n_runs,
            noise=spec.noise,
            random_state=spec.random_state,
        )
        return clean(
            {
                "runs": found.runs.to_dict("records"),
                "best": found.best,
                "improvement_ratio": found.improvement_ratio,
                "little_to_gain": found.improvement_ratio < LITTLE_TO_GAIN,
                "theta": dict(zip(found.fit.inputs, found.fit.theta, strict=True)),
                "noise_ratio": found.fit.noise_ratio,
            }
        )
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool suggest_next_runs failed")
        return {"error": str(e)}


_register("suggest_next_runs")
