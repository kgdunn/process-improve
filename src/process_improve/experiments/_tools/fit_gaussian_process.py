# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``fit_gaussian_process``, a model of a computer experiment, or of runs no quadratic fits."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: The most runs the model is fitted to: its cost grows with the cube of the runs, about 3 s at 200.
MAX_TOOL_RUNS = 200


class FitGaussianProcessInput(BaseModel):
    """Input contract for ``fit_gaussian_process``."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(
        ...,
        min_length=3,
        description=(
            f"The runs, at most {MAX_TOOL_RUNS}, one dict per run: each input in its own units and the response. "
            "Example: [{'x1': 0.1, 'x2': 3.0, 'y': 4.2}, ...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response in data.")
    inputs: list[str] | None = Field(None, description="The input columns; every other column by default.")
    noise: bool = Field(
        False,
        description=(
            "Fit a noise term, for runs whose response scatters, as a physical experiment's does; runs may then "
            "repeat a setting. False fits the model of a computer experiment, which passes through every run."
        ),
    )
    predict_at: list[dict[str, Any]] | None = Field(
        None,
        max_length=1000,
        description="Settings to predict at, one dict each with every input in its own units.",
    )
    random_state: int = Field(0, description="Seed for the random starts of the likelihood search.")


@tool_spec(
    name="fit_gaussian_process",
    description=(
        "Fit a Gaussian process (ordinary kriging) to runs by maximum likelihood: a model that passes through "
        "every run of a computer experiment (a simulation that gives the same answer each time), or, with "
        "noise=true, smooths a physical experiment's runs. Use it where no quadratic fits the response. "
        "Returns each input's correlation parameter 'theta' (large: the response changes quickly along it), "
        "the mean 'mu', the process variance 'sigma2', the 'noise_ratio' (noise variance over process "
        "variance), 'sensitivity' (each input's main-effect and total share of the prediction's variance over "
        "the runs' box), and, for predict_at, each prediction with its standard error. To choose the next runs "
        "from such a model, call suggest_next_runs."
    ),
    input_model=FitGaussianProcessInput,
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
    examples="""
    # "Fit a model to my simulation's 20 runs and say which input matters"
        -> ``fit_gaussian_process(data=[{"x1": 0.1, "x2": 3.0, "y": 4.2}, ...], response_column="y")``

    # "Predict at two new settings, the runs scattering"
        -> ``fit_gaussian_process(..., noise=True, predict_at=[{"x1": 0.5, "x2": 2.0}, ...])``
    """,
    category="experiments",
)
def fit_gaussian_process_tool(spec: FitGaussianProcessInput) -> dict[str, Any]:
    """Fit a Gaussian process to the runs."""
    try:
        from process_improve.experiments.gaussian_process import fit_gaussian_process  # noqa: PLC0415

        if len(spec.data) > MAX_TOOL_RUNS:
            return {"error": f"data has {len(spec.data)} runs; the model takes at most {MAX_TOOL_RUNS}."}
        fit = fit_gaussian_process(
            pd.DataFrame(spec.data),
            spec.response_column,
            spec.inputs,
            noise=spec.noise,
            random_state=spec.random_state,
        )
        result: dict[str, Any] = {
            "inputs": list(fit.inputs),
            "theta": dict(zip(fit.inputs, fit.theta, strict=True)),
            "mu": fit.mu,
            "sigma2": fit.sigma2,
            "noise_ratio": fit.noise_ratio,
            "minus_two_log_likelihood": fit.minus_two_log_likelihood,
            "sensitivity": fit.sensitivity().reset_index(names="input").to_dict("records"),
        }
        if spec.predict_at:
            points = pd.DataFrame(spec.predict_at)
            result["predictions"] = pd.concat([points, fit.predict(points)], axis=1).to_dict("records")
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool fit_gaussian_process failed")
        return {"error": str(e)}


_register("fit_gaussian_process")
