# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``simulate_design_performance``, designs judged on a hypothesised process by simulation."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class SimulateDesignPerformanceInput(BaseModel):
    """Input contract for ``simulate_design_performance``."""

    model_config = ConfigDict(extra="forbid")

    designs: dict[str, list[dict[str, float]]] = Field(
        ...,
        min_length=1,
        max_length=6,
        description=(
            "The designs to simulate, by label: each design's runs in coded units (-1 to +1), one dict per run "
            "with a value per factor, as generate_design returns them under 'design_coded'. Every design has the "
            "same factors. Example: {'DSD': [{'A': -1, 'B': 0, ...}, ...], 'PB': [...]}"
        ),
    )
    true_coefficients: dict[str, float] = Field(
        ...,
        min_length=1,
        description=(
            "The process you expect, in coded units: each term and its coefficient. A term is 'Intercept', a "
            "factor ('A'), a two-factor interaction ('A:B') or a square ('A^2'). Factors no term names are inert. "
            "Example: {'Intercept': 100, 'A': 10, 'C': 7, 'G': -6, 'A:C': 8}"
        ),
    )
    sigma: float = Field(..., gt=0.0, description="The run-to-run noise standard deviation, in the response's units.")
    goal: Literal["learn", "maximize", "minimize", "target"] = Field(
        "learn",
        description=(
            "'learn' reports what the analysis finds. A goal also reports how far the best settings each study "
            "predicts fall short of the true best."
        ),
    )
    target: float | None = Field(None, description="The response wanted, for goal='target'.")
    n_simulations: int = Field(
        100,
        ge=1,
        le=1000,
        description=(
            "Simulated studies per design. Each takes about 0.02 to 0.1 s. A share from 100 studies is within "
            "about 0.1 of its limit (95%)."
        ),
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level of every test of the judging.")
    random_state: int = Field(42, ge=0, description="Seed for the noise. Default 42, for a reproducible report.")


@tool_spec(
    name="simulate_design_performance",
    description=(
        "Compare designs BEFORE running them, on the process you expect: each design is made many times on that "
        "process with fresh noise, and every simulated study is analysed as recommend_next_step analyses real "
        "runs. Unlike evaluate_design and compare_designs, which assume one fixed model, this shows what "
        "happens when interactions outside a screening model bias its effects, and when the analysis has to "
        "pick the model from noisy runs. Returns, per design: the share of studies that find each true term "
        "('found'), all of them ('found_all'), all and nothing else ('found_exactly'), the false alarms per "
        "study and which terms they were, how often nothing stood out, and with a goal, how far the predicted "
        "best settings fall short of the true best ('shortfall', also in noise standard deviations). A term "
        "found through an exact alias (D for A:B in a resolution III fraction, the centre-run curvature for a "
        "square) counts as found. Use it to choose between candidate designs, or to check that a design is "
        "large enough for the effects you need to see."
    ),
    input_model=SimulateDesignPerformanceInput,
    examples="""
    # "Would a 12-run Plackett-Burman design find A, C, G and the A:C interaction, or do I need a DSD?"
        -> ``simulate_design_performance(designs={"PB": [...], "DSD": [...]},
                true_coefficients={"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)``

    # "How close to the best settings would this response surface design get me?"
        -> ``simulate_design_performance(designs={"CCD": [...]}, goal="maximize",
                true_coefficients={"Intercept": 60, "A": 2, "A^2": -6, "B^2": -4}, sigma=0.7)``
    """,
    category="experiments",
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 42},
)
def simulate_design_performance_tool(spec: SimulateDesignPerformanceInput) -> dict[str, Any]:
    """Simulate studies with each design on the expected process."""
    try:
        from process_improve.experiments.simulate import simulate_design_performance  # noqa: PLC0415
        from process_improve.experiments.true_process import TrueProcess  # noqa: PLC0415

        designs = {label: pd.DataFrame(runs) for label, runs in spec.designs.items()}
        result = simulate_design_performance(
            designs,
            TrueProcess(spec.true_coefficients, sigma=spec.sigma),
            goal=spec.goal,
            target=spec.target,
            n_simulations=spec.n_simulations,
            alpha=spec.alpha,
            random_state=spec.random_state,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool simulate_design_performance failed")
        return {"error": str(e)}


_register("simulate_design_performance")
