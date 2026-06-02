"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for the fake-data simulator.

Three tools are exposed:

- ``create_simulator`` - records a hidden response-surface model.
- ``simulate_process`` - evaluates the hidden model at given factor
  settings, with fresh Gaussian noise each call.
- ``reveal_simulator`` - returns the underlying coefficients, gated
  behind a ``confirmed`` flag enforced by the host.

The tools are intentionally stateless: the hidden model lives in a
``private_state`` dict that the host (e.g. the factorial web app)
persists and injects on each call. ``simulator_state`` and the reveal
``confirmed`` flag are not kwargs: the host supplies them out of band
through :func:`process_improve.simulation.context.simulator_host_context`,
which stores them in :class:`contextvars.ContextVar` slots. Keeping them
off the kwarg surface means a prompt-injected agent cannot re-introduce
them through the dispatch path to forge state or bypass the reveal gate
(SEC-15).

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument. ``create_simulator.seed`` is
declared as ``SkipJsonSchema[int | None]`` so the field is callable
from Python (tests, notebooks) but does NOT appear in the JSON Schema
exposed to the LLM.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from process_improve.simulation.context import (
    get_injected_simulator_state,
    get_reveal_confirmed,
)
from process_improve.simulation.model import (
    draw_initial_seed,
    materialize_model,
    simulate,
    validate_factors,
    validate_noise_level,
    validate_outputs,
)
from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_SIMULATION_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _SIMULATION_TOOL_NAMES.append(name)


def _public_from_private(private: dict[str, Any], process_description: str, created_at: str) -> dict[str, Any]:
    """Build the LLM-visible summary of a simulator from its private state."""
    return {
        "factors": [
            {
                "name": f["name"],
                "low": float(f["low"]),
                "high": float(f["high"]),
                "units": f.get("units"),
            }
            for f in private["factors"]
        ],
        "outputs": [
            {"name": o["name"], "units": o.get("units"), "direction": o.get("direction")}
            for o in private["outputs"]
        ],
        "noise_level": private["noise_level"],
        "time_drift": private["time_drift"],
        "process_description": process_description,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# create_simulator
# ---------------------------------------------------------------------------


class CreateSimulatorInput(BaseModel):
    """Input contract for ``create_simulator``."""

    model_config = ConfigDict(extra="forbid")

    process_description: str = Field(
        ...,
        description=(
            "One-sentence description of the process being simulated, "
            "e.g. 'nickel flotation vessel for recovery and grade'."
        ),
    )
    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Input variables the user can set. Choose plausible low/high "
            "bounds from your domain expertise; do not ask the user for "
            "ranges unless they insist."
        ),
    )
    outputs: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Response variables the user wants measured. Confirm with the "
            "user before calling the tool."
        ),
    )
    structural_hints: list[str] | None = Field(
        None,
        description=(
            "Free-text biases for the hidden model, e.g. 'negative "
            "interaction between pH and surfactant', 'flow has a "
            "quadratic effect on recovery'. Unparseable hints are "
            "silently ignored."
        ),
    )
    noise_level: Literal["low", "medium", "high"] = Field(
        "medium",
        description=(
            "Noise magnitude as a fraction of the output range "
            "(~1 %, ~5 %, ~15 %). Default 'medium'."
        ),
    )
    time_drift: bool = Field(
        False,
        description=(
            "When true, each output gets a slow linear drift applied "
            "whenever the user passes 'timestamp_offset_days' to "
            "'simulate_process'. Default false."
        ),
    )
    # SkipJsonSchema hides the field from the public JSON schema so the LLM
    # cannot pin a seed it will later try to reason about; Python callers
    # (tests, notebooks) can still pass it.
    seed: SkipJsonSchema[int | None] = Field(
        None,
        description="Internal: seed for reproducibility. Not exposed to the LLM.",
    )


@tool_spec(
    name="create_simulator",
    description=(
        "Create a hidden process-simulator that the user can query for synthetic "
        "response data. Use this when the user wants fake but realistic data to "
        "plan or demonstrate a designed experiment. "
        "Pick factor ranges silently from your domain knowledge; pass them in "
        "the 'factors' list. The 'outputs' list must come from the user - "
        "propose defaults if they are undecided but confirm before calling. "
        "The underlying model (intercepts, main effects, 2-factor interactions, "
        "quadratic terms) is generated internally and must NOT be disclosed to "
        "the user unless they explicitly ask to reveal it, in which case call "
        "'reveal_simulator'. "
        "Returns a 'sim_id' to pass to subsequent 'simulate_process' calls, plus "
        "a public summary of the declared factors, outputs, and noise level. "
        "The host application persists the hidden state - do NOT try to store "
        "or paraphrase it yourself."
    ),
    input_model=CreateSimulatorInput,
    examples="""
    # "Simulate a Ni flotation vessel with flow, pH, surfactant and the known
    #  negative pH*surfactant interaction; recovery and grade as outputs."
        -> ``create_simulator(
                process_description="Ni flotation vessel",
                factors=[
                    {"name": "flow", "low": 100, "high": 300, "units": "L/min"},
                    {"name": "pH", "low": 7.0, "high": 11.0},
                    {"name": "surfactant", "low": 10, "high": 80, "units": "ppm"},
                ],
                outputs=[
                    {"name": "recovery", "units": "%", "direction": "maximize"},
                    {"name": "grade", "units": "%", "direction": "maximize"},
                ],
                structural_hints=["negative interaction between pH and surfactant"],
                noise_level="medium",
            )``
    """,
    category="simulation",
)
def create_simulator(spec: CreateSimulatorInput) -> dict:
    """Create a simulator; see tool spec for parameter details."""
    validate_factors(spec.factors)
    validate_outputs(spec.outputs)
    validate_noise_level(spec.noise_level)
    if not isinstance(spec.process_description, str) or not spec.process_description.strip():
        raise ValueError("'process_description' must be a non-empty string.")

    seed_value = int(spec.seed) if spec.seed is not None else draw_initial_seed()
    sim_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    private_state: dict[str, Any] = {
        "seed": seed_value,
        "factors": [
            {
                "name": f["name"],
                "low": float(f["low"]),
                "high": float(f["high"]),
                "units": f.get("units"),
            }
            for f in spec.factors
        ],
        "outputs": [
            {"name": o["name"], "units": o.get("units"), "direction": o.get("direction")}
            for o in spec.outputs
        ],
        "structural_hints": list(spec.structural_hints or []),
        "noise_level": spec.noise_level,
        "time_drift": bool(spec.time_drift),
        "model_version": 1,
    }

    public = _public_from_private(private_state, spec.process_description, created_at)

    # ``_private`` uses a leading underscore so hosts can identify and
    # strip it before forwarding the tool result to the LLM.
    return clean({"sim_id": sim_id, "public": public, "_private": private_state})


_register("create_simulator")


# ---------------------------------------------------------------------------
# simulate_process
# ---------------------------------------------------------------------------


class SimulateProcessInput(BaseModel):
    """Input contract for ``simulate_process``."""

    model_config = ConfigDict(extra="forbid")

    sim_id: str = Field(
        ...,
        description="Simulator id returned by 'create_simulator'.",
    )
    settings: dict[str, float] = Field(
        ...,
        description=(
            "Mapping of factor-name to numeric value, in the factor's declared units."
        ),
    )
    timestamp_offset_days: float = Field(
        0.0,
        description=(
            "Optional time axis, in days since simulator creation. "
            "Only meaningful if the simulator was created with time_drift=true."
        ),
    )


@tool_spec(
    name="simulate_process",
    description=(
        "Evaluate a previously created simulator at specific factor settings. "
        "Returns the simulated output values (with fresh Gaussian noise per call, "
        "so identical settings yield similar but not identical outputs - this is "
        "intentional, matching real asset behaviour). "
        "Pass 'sim_id' from the 'create_simulator' response. Supply all declared "
        "factors in 'settings'; missing factors default to their mid-range value "
        "and out-of-range values are clipped to the declared bounds (both with a "
        "warning in the response). "
        "Use 'timestamp_offset_days' only when the simulator was created with "
        "time_drift=true."
    ),
    input_model=SimulateProcessInput,
    examples="""
    # "Run the simulator at flow=200, pH=9, surfactant=50"
        -> ``simulate_process(
                sim_id="<uuid>",
                settings={"flow": 200, "pH": 9, "surfactant": 50},
            )``
    """,
    category="simulation",
)
def simulate_process(spec: SimulateProcessInput) -> dict:
    """Evaluate a simulator at *settings*.

    ``simulator_state`` is not a parameter: the host injects it out of band
    via :func:`process_improve.simulation.context.simulator_host_context`,
    so it can never arrive as an LLM-supplied kwarg. If it is missing we
    return a structured error rather than dispatch with a guessed state.
    """
    simulator_state = get_injected_simulator_state()
    if simulator_state is None:
        return {
            "sim_id": spec.sim_id,
            "error": "simulator_state_missing",
            "message": (
                "The host did not inject 'simulator_state'. Ensure sim_id refers "
                "to a simulator created in this conversation."
            ),
        }
    if not isinstance(spec.settings, dict):
        raise TypeError("'settings' must be a dict of factor-name to numeric value.")

    result = simulate(
        simulator_state,
        spec.settings,
        timestamp_offset_days=float(spec.timestamp_offset_days),
    )
    result["sim_id"] = spec.sim_id
    return clean(result)


_register("simulate_process")


# ---------------------------------------------------------------------------
# reveal_simulator
# ---------------------------------------------------------------------------


class RevealSimulatorInput(BaseModel):
    """Input contract for ``reveal_simulator``."""

    model_config = ConfigDict(extra="forbid")

    sim_id: str = Field(
        ...,
        description="Simulator id returned by 'create_simulator'.",
    )


@tool_spec(
    name="reveal_simulator",
    description=(
        "Reveal the hidden model behind a simulator. The host application gates "
        "this call behind a double-confirmation: the first attempt returns a "
        "'confirmation_needed' status (surface it to the user verbatim), the "
        "second attempt after the user confirms returns the full coefficient "
        "set. Only use this when the user explicitly asks to see the underlying "
        "model."
    ),
    input_model=RevealSimulatorInput,
    examples="""
    # "Show me the hidden model behind the simulator"
        -> ``reveal_simulator(sim_id="<uuid>")``
    """,
    category="simulation",
)
def reveal_simulator(spec: RevealSimulatorInput) -> dict:
    """Return the materialised model.

    ``simulator_state`` and the ``confirmed`` flag are injected by the host
    out of band via
    :func:`process_improve.simulation.context.simulator_host_context`, never
    by the LLM. They are not parameters, so a tool call cannot forge state or
    pre-clear the double-confirmation gate.
    """
    if not get_reveal_confirmed():
        return {
            "sim_id": spec.sim_id,
            "status": "confirmation_needed",
            "message": (
                "Revealing the simulator will expose the hidden response model. "
                "Ask the user to confirm; call reveal_simulator again after they "
                "confirm."
            ),
        }
    simulator_state = get_injected_simulator_state()
    if simulator_state is None:
        return {
            "sim_id": spec.sim_id,
            "error": "simulator_state_missing",
            "message": (
                "The host did not inject 'simulator_state' even though the "
                "reveal was confirmed. Ensure sim_id refers to a simulator "
                "created in this conversation."
            ),
        }

    model = materialize_model(simulator_state)
    return clean(
        {
            "sim_id": spec.sim_id,
            "status": "revealed",
            "factors": simulator_state["factors"],
            "outputs": simulator_state["outputs"],
            "structural_hints": simulator_state.get("structural_hints", []),
            "noise_level": simulator_state["noise_level"],
            "time_drift": simulator_state["time_drift"],
            "model": model,
        }
    )


_register("reveal_simulator")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_simulation_tool_specs() -> list[dict]:
    """Return tool specs for all simulation tools registered in this module."""
    return get_tool_specs(names=_SIMULATION_TOOL_NAMES)
