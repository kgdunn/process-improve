"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for the fake-data simulator.

Three tools are exposed:

- ``create_simulator`` — records a hidden response-surface model.
- ``simulate_process`` — evaluates the hidden model at given factor
  settings, with fresh Gaussian noise each call.
- ``reveal_simulator`` — returns the underlying coefficients, gated
  behind a ``confirmed`` flag enforced by the host.

The tools are intentionally stateless: the hidden model lives in a
``private_state`` dict that the host (e.g. the factorial web app)
persists and injects on each call via a hidden ``simulator_state``
kwarg. The JSON schema advertised to the LLM does not mention
``simulator_state`` or ``confirmed`` — those are injected server-side
so the LLM cannot bypass the reveal policy or fabricate state.
"""

# ``dict[str, Any]`` is the tool-call contract shape; the real schema
# lives in the @tool_spec JSON blocks below, not in the Python types.
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from process_improve.simulation.model import (
    draw_initial_seed,
    materialize_model,
    simulate,
    validate_factors,
    validate_noise_level,
    validate_outputs,
)
from process_improve.tool_spec import clean, get_tool_specs, tool_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Tools
# ---------------------------------------------------------------------------


@tool_spec(
    name="create_simulator",
    description=(
        "Create a hidden process-simulator that the user can query for synthetic "
        "response data. Use this when the user wants fake but realistic data to "
        "plan or demonstrate a designed experiment. "
        "Pick factor ranges silently from your domain knowledge; pass them in "
        "the 'factors' list. The 'outputs' list must come from the user — "
        "propose defaults if they are undecided but confirm before calling. "
        "The underlying model (intercepts, main effects, 2-factor interactions, "
        "quadratic terms) is generated internally and must NOT be disclosed to "
        "the user unless they explicitly ask to reveal it, in which case call "
        "'reveal_simulator'. "
        "Returns a 'sim_id' to pass to subsequent 'simulate_process' calls, plus "
        "a public summary of the declared factors, outputs, and noise level. "
        "The host application persists the hidden state — do NOT try to store "
        "or paraphrase it yourself."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "process_description": {
                    "type": "string",
                    "description": (
                        "One-sentence description of the process being simulated, "
                        "e.g. 'nickel flotation vessel for recovery and grade'."
                    ),
                },
                "factors": {
                    "type": "array",
                    "description": (
                        "Input variables the user can set. Choose plausible low/high "
                        "bounds from your domain expertise; do not ask the user for "
                        "ranges unless they insist."
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                            "units": {"type": "string"},
                        },
                        "required": ["name", "low", "high"],
                    },
                },
                "outputs": {
                    "type": "array",
                    "description": (
                        "Response variables the user wants measured. Confirm with the "
                        "user before calling the tool."
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "units": {"type": "string"},
                            "direction": {
                                "type": "string",
                                "enum": ["maximize", "minimize", "target"],
                                "description": "Optional optimisation direction.",
                            },
                        },
                        "required": ["name"],
                    },
                },
                "structural_hints": {
                    "type": "array",
                    "description": (
                        "Free-text biases for the hidden model, e.g. 'negative "
                        "interaction between pH and surfactant', 'flow has a "
                        "quadratic effect on recovery'. Unparseable hints are "
                        "silently ignored."
                    ),
                    "items": {"type": "string"},
                },
                "noise_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": (
                        "Noise magnitude as a fraction of the output range "
                        "(~1 %, ~5 %, ~15 %). Default 'medium'."
                    ),
                },
                "time_drift": {
                    "type": "boolean",
                    "description": (
                        "When true, each output gets a slow linear drift applied "
                        "whenever the user passes 'timestamp_offset_days' to "
                        "'simulate_process'. Default false."
                    ),
                },
            },
            "required": ["process_description", "factors", "outputs"],
        }
    },
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
def create_simulator(  # noqa: PLR0913
    *,
    process_description: str,
    factors: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    structural_hints: list[str] | None = None,
    noise_level: str = "medium",
    time_drift: bool = False,
    seed: int | None = None,
) -> dict:
    """Create a simulator; see tool spec for parameter details.

    The ``seed`` kwarg is intentionally missing from the public JSON schema
    so the LLM cannot pin a seed it will later try to reason about; callers
    that need reproducibility (tests, notebooks) pass it in Python directly.
    """
    validate_factors(factors)
    validate_outputs(outputs)
    validate_noise_level(noise_level)
    if not isinstance(process_description, str) or not process_description.strip():
        raise ValueError("'process_description' must be a non-empty string.")

    seed_value = int(seed) if seed is not None else draw_initial_seed()
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
            for f in factors
        ],
        "outputs": [
            {"name": o["name"], "units": o.get("units"), "direction": o.get("direction")}
            for o in outputs
        ],
        "structural_hints": list(structural_hints or []),
        "noise_level": noise_level,
        "time_drift": bool(time_drift),
        "model_version": 1,
    }

    public = _public_from_private(private_state, process_description, created_at)

    # ``_private`` uses a leading underscore so hosts can identify and
    # strip it before forwarding the tool result to the LLM.
    return clean({"sim_id": sim_id, "public": public, "_private": private_state})


_register("create_simulator")


@tool_spec(
    name="simulate_process",
    description=(
        "Evaluate a previously created simulator at specific factor settings. "
        "Returns the simulated output values (with fresh Gaussian noise per call, "
        "so identical settings yield similar but not identical outputs — this is "
        "intentional, matching real asset behaviour). "
        "Pass 'sim_id' from the 'create_simulator' response. Supply all declared "
        "factors in 'settings'; missing factors default to their mid-range value "
        "and out-of-range values are clipped to the declared bounds (both with a "
        "warning in the response). "
        "Use 'timestamp_offset_days' only when the simulator was created with "
        "time_drift=true."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "sim_id": {
                    "type": "string",
                    "description": "Simulator id returned by 'create_simulator'.",
                },
                "settings": {
                    "type": "object",
                    "description": (
                        "Mapping of factor-name to numeric value, in the factor's "
                        "declared units."
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "timestamp_offset_days": {
                    "type": "number",
                    "description": (
                        "Optional time axis, in days since simulator creation. "
                        "Only meaningful if the simulator was created with "
                        "time_drift=true."
                    ),
                },
            },
            "required": ["sim_id", "settings"],
        }
    },
    examples="""
    # "Run the simulator at flow=200, pH=9, surfactant=50"
        -> ``simulate_process(
                sim_id="<uuid>",
                settings={"flow": 200, "pH": 9, "surfactant": 50},
            )``
    """,
    category="simulation",
)
def simulate_process(
    *,
    sim_id: str,
    settings: dict[str, float],
    timestamp_offset_days: float = 0.0,
    simulator_state: dict[str, Any] | None = None,
) -> dict:
    """Evaluate a simulator at *settings*; see tool spec for details.

    ``simulator_state`` is not in the JSON schema: the host injects it
    from its own persistence layer. If it is missing we return a
    structured error rather than dispatch with a guessed state.
    """
    if simulator_state is None:
        return {
            "sim_id": sim_id,
            "error": "simulator_state_missing",
            "message": (
                "The host did not inject 'simulator_state'. Ensure sim_id refers "
                "to a simulator created in this conversation."
            ),
        }
    if not isinstance(settings, dict):
        raise TypeError("'settings' must be a dict of factor-name to numeric value.")

    result = simulate(
        simulator_state,
        settings,
        timestamp_offset_days=float(timestamp_offset_days),
    )
    result["sim_id"] = sim_id
    return clean(result)


_register("simulate_process")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "sim_id": {
                    "type": "string",
                    "description": "Simulator id returned by 'create_simulator'.",
                },
            },
            "required": ["sim_id"],
        }
    },
    examples="""
    # "Show me the hidden model behind the simulator"
        -> ``reveal_simulator(sim_id="<uuid>")``
    """,
    category="simulation",
)
def reveal_simulator(
    *,
    sim_id: str,
    simulator_state: dict[str, Any] | None = None,
    confirmed: bool = False,
) -> dict:
    """Return the materialised model; see tool spec for details.

    ``simulator_state`` and ``confirmed`` are injected by the host, not
    by the LLM (their absence from the JSON schema is intentional).
    """
    if not confirmed:
        return {
            "sim_id": sim_id,
            "status": "confirmation_needed",
            "message": (
                "Revealing the simulator will expose the hidden response model. "
                "Ask the user to confirm; call reveal_simulator again after they "
                "confirm."
            ),
        }
    if simulator_state is None:
        return {
            "sim_id": sim_id,
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
            "sim_id": sim_id,
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
