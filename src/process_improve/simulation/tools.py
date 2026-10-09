"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for the simulation subpackage.

The tools exposed here:

- ``create_simulator`` - records a hidden response-surface model.
- ``simulate_process`` - evaluates the hidden model at given factor
  settings, with fresh Gaussian noise each call.
- ``reveal_simulator`` - returns the underlying coefficients, gated
  behind a ``confirmed`` flag enforced by the host.
- ``simulate_batch_campaign`` - runs a fed-batch bioreactor campaign
  under a named operating policy and reports the titer outcomes.
- ``decompose_batch_quality_variance`` - splits the replay-campaign
  titer variance into its disturbance sources.
- ``correct_batch_midcourse`` - corrects a handful of fresh batches at a
  decision point and reports the executed (re-simulated) outcomes.
- ``evaluate_batch_control_policy`` - the campaign-level executed
  comparison of replay vs mid-course correction (optionally with the
  feedforward and oracle ceilings).

The three DOE tools above are stateless in the ``private_state`` sense
described below; the four bioreactor tools have no hidden state at all
(their model is deliberately open, see
:mod:`process_improve.simulation.batch`) and are reproducible from
their ``random_state`` field alone.

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
            {"name": o["name"], "units": o.get("units"), "direction": o.get("direction")} for o in private["outputs"]
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
        min_length=1,
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
        description=("Response variables the user wants measured. Confirm with the user before calling the tool."),
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
        description=("Noise magnitude as a fraction of the output range (~1 %, ~5 %, ~15 %). Default 'medium'."),
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
    # process_description's str + min_length=1 contract is enforced by pydantic.
    # All-whitespace strings would slip through pydantic's min_length check, but
    # the downstream model is content-agnostic, so we accept them here.

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
            {"name": o["name"], "units": o.get("units"), "direction": o.get("direction")} for o in spec.outputs
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
        description=("Mapping of factor-name to numeric value, in the factor's declared units."),
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
    # ``settings`` is constrained to ``dict[str, float]`` by the pydantic model;
    # the previous isinstance defensive check is now unreachable and removed.

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
# Bioreactor (golden batch) tools
# ---------------------------------------------------------------------------


class SimulateBatchCampaignInput(BaseModel):
    """Input contract for ``simulate_batch_campaign``."""

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(..., ge=2, le=500, description="Number of batches to simulate in the campaign.")
    policy: Literal["replay", "historical"] = Field(
        "replay",
        description=(
            "Operating policy. 'replay' gives every batch the same nominal setpoint schedule (the "
            "golden-batch practice). 'historical' adds deliberate per-batch setpoint variation of size "
            "mv_variation, producing a history that carries information about how the controls affect quality."
        ),
    )
    mv_variation: float = Field(
        0.0,
        ge=0.0,
        le=3.0,
        description=(
            "Size of the deliberate setpoint variation for the 'historical' policy: the standard deviation "
            "in degC of each batch's random temperature offset and ramp (pH varies at one tenth of this)."
        ),
    )
    ic_scale: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description=(
            "Scale of the measured initial-condition (upstream Z block) disturbance channel; 0 gives every "
            "batch identical initial conditions."
        ),
    )
    within_batch_scale: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description=(
            "Scale of the unmeasured within-batch disturbance channel (slow metabolic drift and feed-rate "
            "drift); 0 switches it off."
        ),
    )
    noise_scale: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description="Scale of the control-loop and measurement noise channel; 0 switches it off.",
    )
    random_state: int = Field(0, ge=0, le=2**31 - 1, description="Seed; the same seed reproduces the campaign exactly.")


@tool_spec(
    name="simulate_batch_campaign",
    description=(
        "Simulate a campaign of fed-batch bioreactor runs under a named operating policy and report the "
        "final-titer outcomes. The simulator is a deterministic mechanistic model (Rosso cardinal "
        "temperature/pH growth kinetics, Luedeking-Piret production, an oxygen-transfer ceiling) with three "
        "independently tunable disturbance channels: measured initial conditions, an unmeasured within-batch "
        "disturbance, and instrument-scale noise. Use it to demonstrate that replaying a fixed 'golden batch' "
        "schedule does not reproduce its outcome, and to generate batch data for latent-variable modelling. "
        "Returns the per-batch titers with feed-class labels, summary statistics, and the disturbance-free "
        "reference titer of the same schedule for comparison."
    ),
    input_model=SimulateBatchCampaignInput,
    examples="""
    # "Replay our best batch's recipe 50 times and show me the spread"
        -> ``simulate_batch_campaign(n_batches=50, policy="replay", random_state=0)``
    # "Give me a historical campaign with deliberate setpoint variation for model building"
        -> ``simulate_batch_campaign(n_batches=80, policy="historical", mv_variation=1.0)``
    # "Same initial conditions for every batch; how much spread remains?"
        -> ``simulate_batch_campaign(n_batches=50, ic_scale=0.0)``
    """,
    category="simulation",
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
)
def simulate_batch_campaign(spec: SimulateBatchCampaignInput) -> dict[str, Any]:
    """Simulate a bioreactor campaign and summarise the final-quality outcomes."""
    import dataclasses  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from process_improve.simulation.batch import BioreactorConfig, BioreactorSimulator  # noqa: PLC0415

    try:
        config = dataclasses.replace(
            BioreactorConfig(),
            ic_scale=spec.ic_scale,
            within_batch_scale=spec.within_batch_scale,
            noise_scale=spec.noise_scale,
        )
        simulator = BioreactorSimulator(config)
        campaign = simulator.simulate_campaign(
            spec.n_batches,
            policy=spec.policy,
            mv_variation=spec.mv_variation,
            random_state=spec.random_state,
        )
        nominal = simulator.nominal_trajectory()
        reference = BioreactorSimulator(
            dataclasses.replace(config, ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0)
        ).simulate_batch(None, nominal)
        titer = campaign.quality["titer"]
        by_class = titer.groupby(campaign.classes).agg(["count", "mean", "std"])
        return clean(
            {
                "n_batches": spec.n_batches,
                "policy": spec.policy,
                "titer_g_L": {
                    "mean": float(titer.mean()),
                    "sd": float(titer.std(ddof=1)),
                    "cv_pct": float(100.0 * titer.std(ddof=1) / titer.mean()),
                    "min": float(titer.min()),
                    "max": float(titer.max()),
                },
                "reference_titer_g_L": {
                    "value": float(reference.titer),
                    "meaning": "the same nominal schedule with every disturbance channel switched off",
                    "fraction_of_batches_below": float(np.mean(titer.to_numpy() < reference.titer)),
                },
                "by_feed_class": {
                    str(label): {
                        "count": int(row["count"]),
                        "mean": float(row["mean"]),
                        "sd": float(row["std"]),
                    }
                    for label, row in by_class.iterrows()
                },
                "batches": [
                    {
                        "batch_id": int(batch_id),
                        "feed_class": str(campaign.classes.loc[batch_id]),
                        "titer_g_L": round(float(value), 4),
                    }
                    for batch_id, value in titer.items()
                ],
            }
        )
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("simulate_batch_campaign")


class DecomposeBatchQualityVarianceInput(BaseModel):
    """Input contract for ``decompose_batch_quality_variance``."""

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(
        150,
        ge=10,
        le=400,
        description="Batches per campaign; four campaigns are run (all channels on, then each channel alone).",
    )
    ic_scale: float = Field(
        1.0, ge=0.0, le=3.0, description="Scale of the measured initial-condition disturbance channel."
    )
    within_batch_scale: float = Field(
        1.0, ge=0.0, le=3.0, description="Scale of the unmeasured within-batch disturbance channel."
    )
    noise_scale: float = Field(
        1.0, ge=0.0, le=3.0, description="Scale of the control-loop and measurement noise channel."
    )
    random_state: int = Field(0, ge=0, le=2**31 - 1, description="Seed; the same seed reproduces the decomposition.")


@tool_spec(
    name="decompose_batch_quality_variance",
    description=(
        "Split the batch-to-batch final-quality variance of a replayed (golden batch) schedule into its "
        "sources: measured initial conditions, an unmeasured within-batch disturbance, and control plus "
        "measurement noise, with the interaction residual reported separately. This answers the question "
        "'we replicate our best batch's recipe exactly, why do the outcomes still differ?'. The "
        "initial-condition share is the part that adapting the schedule before the batch could address; "
        "the within-batch share is observable only while the batch runs, so it is the part a mid-course "
        "correction at decision points could address; the noise share is the floor."
    ),
    input_model=DecomposeBatchQualityVarianceInput,
    examples="""
    # "Why do our batches differ when we run the same recipe every time?"
        -> ``decompose_batch_quality_variance()``
    # "How much of the spread is left if incoming materials were perfectly consistent?"
        -> ``decompose_batch_quality_variance(ic_scale=0.0)``
    """,
    category="simulation",
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
)
def decompose_batch_quality_variance(spec: DecomposeBatchQualityVarianceInput) -> dict[str, Any]:
    """Decompose replay-campaign titer variance into its disturbance sources."""
    import dataclasses  # noqa: PLC0415

    from process_improve.simulation.batch import (  # noqa: PLC0415
        BioreactorConfig,
        BioreactorSimulator,
        variance_decomposition,
    )

    try:
        config = dataclasses.replace(
            BioreactorConfig(),
            ic_scale=spec.ic_scale,
            within_batch_scale=spec.within_batch_scale,
            noise_scale=spec.noise_scale,
        )
        simulator = BioreactorSimulator(config)
        frame = variance_decomposition(simulator, n_batches=spec.n_batches, random_state=spec.random_state)
        return clean(
            {
                "n_batches_per_campaign": spec.n_batches,
                "mean_titer_g_L": float(frame.attrs["mean_titer_g_L"]),
                "sources": [
                    {
                        "source": str(source),
                        "variance": float(row["variance"]),
                        "sd_g_L": float(row["sd"]),
                        "cv_pct": float(row["cv_pct"]),
                        "pct_of_total": float(row["pct_of_total"]),
                    }
                    for source, row in frame.iterrows()
                ],
                "reading_the_result": (
                    "'measured initial conditions' is addressable before the batch starts (feedforward "
                    "adaptation of the schedule); 'within-batch disturbance' is observable only through the "
                    "trajectories while the batch runs, so only a mid-course correction can address it; "
                    "'control and measurement noise' is the irreducible floor. The buckets need not sum to "
                    "the total because the process model is nonlinear; the difference is the interaction "
                    "residual."
                ),
            }
        )
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("decompose_batch_quality_variance")


# ---------------------------------------------------------------------------
# correct_batch_midcourse
# ---------------------------------------------------------------------------


class CorrectBatchMidcourseInput(BaseModel):
    """Input contract for the single-decision-point correction demonstration."""

    model_config = ConfigDict(extra="forbid")

    y_target: float = Field(
        8.0,
        gt=0.0,
        le=50.0,
        description="Final-titer target in g/L; batches predicted at or above it are left alone.",
    )
    n_batches: int = Field(5, ge=1, le=10, description="Number of fresh batches to run through the decision point.")
    decision_point: int = Field(
        8, ge=1, le=19, description="Sample index of the decision point (samples are 12 h apart)."
    )
    n_train: int = Field(
        200, ge=30, le=400, description="Historical (training) campaign size for the per-class models."
    )
    mv_variation: float = Field(
        2.5,
        ge=0.1,
        le=5.0,
        description=(
            "Deliberate setpoint variation of the historical campaign (degC per knot; pH scales at 10%). "
            "Corrections can only be justified inside the region this history explored."
        ),
    )
    dead_band: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description=(
            "No-correction dead band in half-widths of the prediction interval at the decision point: 1.0 corrects "
            "only when the whole interval falls short of the target (Yabuki and MacGregor's no-control region); "
            "0 corrects every below-target batch."
        ),
    )
    random_state: int = Field(0, ge=0, le=2**31 - 1, description="Seed; the same seed reproduces everything.")


@tool_spec(
    name="correct_batch_midcourse",
    description=(
        "Run fresh simulated bioreactor batches to a decision point, apply the latent-variable mid-course "
        "correction (per-feed-class PLS models fitted on a deliberately varied historical campaign; TSR "
        "batch-so-far projection; a QP over the remaining setpoint schedule with SPE/T2 caps and a "
        "no-correction dead band), and EXECUTE the corrected schedules by re-simulating each batch with the "
        "identical disturbances. Reports realised before/after titers, not model predictions, so the gain is "
        "a true same-batch counterfactual. Requires the 'control' extra (osqp)."
    ),
    input_model=CorrectBatchMidcourseInput,
    examples="""
    # "Correct a few batches at day 4 and show me what actually happened"
        -> ``correct_batch_midcourse(n_batches=5, decision_point=8, y_target=8.0)``
    # "Correct every below-target batch, no dead band"
        -> ``correct_batch_midcourse(dead_band=0.0)``
    """,
    category="simulation",
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
)
def correct_batch_midcourse(spec: CorrectBatchMidcourseInput) -> dict[str, Any]:
    """Correct fresh batches at one decision point and report the executed outcomes."""
    from process_improve.batch.control import evaluate_control_policies  # noqa: PLC0415
    from process_improve.simulation.batch import BioreactorSimulator  # noqa: PLC0415

    try:
        result = evaluate_control_policies(
            BioreactorSimulator(),
            y_target=spec.y_target,
            n_train=spec.n_train,
            n_test=spec.n_batches,
            mv_variation=spec.mv_variation,
            decision_points=(spec.decision_point,),
            dead_band=spec.dead_band,
            include_adapted=False,
            oracle="none",
            random_state=spec.random_state,
        )
        rows = []
        for batch_id, row in result.batches.iterrows():
            rows.append(
                {
                    "batch_id": str(batch_id),
                    "feed_class": str(row["class_assigned"]),
                    "replay_titer_g_L": round(float(row["replay"]), 3),
                    "corrected": bool(row["corrected"]),
                    "outcome": str(row["reason"]),
                    "executed_titer_g_L": round(float(row["midcourse"]), 3),
                    "predicted_titer_g_L": None if not row["corrected"] else round(float(row["y_hat_predicted"]), 3),
                }
            )
        return clean(
            {
                "decision_point": spec.decision_point,
                "y_target_g_L": spec.y_target,
                "n_corrected": int(result.n_corrected),
                "n_harmed": int(result.n_harmed),
                "batches": rows,
                "model_fit_r2_per_class": {str(k): round(float(v), 3) for k, v in result.models.fit_r2.items()},
                "note": (
                    "executed_titer_g_L re-simulates the batch with the corrected schedule and the identical "
                    "disturbance seed; the gain over replay_titer_g_L is realised, not predicted."
                ),
            }
        )
    except ImportError as exc:
        return {"error": str(exc)}
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("correct_batch_midcourse")


# ---------------------------------------------------------------------------
# evaluate_batch_control_policy
# ---------------------------------------------------------------------------


class EvaluateBatchControlPolicyInput(BaseModel):
    """Input contract for the campaign-level executed policy comparison."""

    model_config = ConfigDict(extra="forbid")

    y_target: float = Field(8.0, gt=0.0, le=50.0, description="Final-titer target in g/L for the corrector.")
    n_train: int = Field(200, ge=30, le=400, description="Historical (training) campaign size.")
    n_test: int = Field(40, ge=5, le=60, description="Fresh test batches, every policy executed per batch.")
    mv_variation: float = Field(
        2.5, ge=0.1, le=5.0, description="Deliberate setpoint variation of the historical campaign."
    )
    decision_point: int = Field(8, ge=1, le=19, description="Sample index of the decision point.")
    dead_band: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="No-correction dead band in half-widths of the prediction interval at the decision point.",
    )
    include_ceilings: bool = Field(
        default=False,
        description=(
            "Also compute the two ceilings: the per-batch perfect-feedforward optimum (adapted) and the "
            "oracle that optimises the remaining schedule against the simulator at the same decision point. "
            "Roughly 10 s per test batch slower."
        ),
    )
    random_state: int = Field(0, ge=0, le=2**31 - 1, description="Seed; the same seed reproduces everything.")


@tool_spec(
    name="evaluate_batch_control_policy",
    description=(
        "Executed campaign-level comparison of batch operating policies on the bioreactor simulator: replay "
        "(the nominal schedule; the floor), mid-course correction (per-feed-class PLS models, corrected at a "
        "decision point, every corrected schedule re-simulated with the identical disturbances), and "
        "optionally the perfect-feedforward ceiling and the oracle-from-the-decision-point ceiling. Reports "
        "realised mean/sd titers per policy and the per-batch outcomes, closing the gap the literature "
        "leaves: published mid-course gains are usually model predictions, never executed validations. "
        "Requires the 'control' extra (osqp)."
    ),
    input_model=EvaluateBatchControlPolicyInput,
    examples="""
    # "How much does mid-course correction actually recover over replay?"
        -> ``evaluate_batch_control_policy(n_test=40, decision_point=8)``
    # "Include the ceilings so I can see what is left on the table"
        -> ``evaluate_batch_control_policy(n_test=20, include_ceilings=True)``
    """,
    category="simulation",
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 0},
)
def evaluate_batch_control_policy(spec: EvaluateBatchControlPolicyInput) -> dict[str, Any]:
    """Compare replay vs executed mid-course correction (and optional ceilings) on a campaign."""
    from process_improve.batch.control import evaluate_control_policies  # noqa: PLC0415
    from process_improve.simulation.batch import BioreactorSimulator  # noqa: PLC0415

    try:
        result = evaluate_control_policies(
            BioreactorSimulator(),
            y_target=spec.y_target,
            n_train=spec.n_train,
            n_test=spec.n_test,
            mv_variation=spec.mv_variation,
            decision_points=(spec.decision_point,),
            dead_band=spec.dead_band,
            include_adapted=spec.include_ceilings,
            oracle="corrected" if spec.include_ceilings else "none",
            random_state=spec.random_state,
        )
        corrected = result.batches[result.batches["corrected"]]
        gains = (corrected["midcourse"] - corrected["replay"]) if len(corrected) else None
        return clean(
            {
                "summary_titer_g_L": {
                    str(policy): {k: round(float(v), 3) for k, v in row.items()}
                    for policy, row in result.summary.iterrows()
                },
                "n_test": spec.n_test,
                "n_corrected": int(result.n_corrected),
                "n_harmed": int(result.n_harmed),
                "corrected_batches": [
                    {
                        "batch_id": str(batch_id),
                        "feed_class": str(row["class_assigned"]),
                        "replay_titer_g_L": round(float(row["replay"]), 3),
                        "executed_titer_g_L": round(float(row["midcourse"]), 3),
                        "realised_gain_g_L": round(float(row["midcourse"] - row["replay"]), 3),
                        "predicted_titer_g_L": round(float(row["y_hat_predicted"]), 3),
                    }
                    for batch_id, row in corrected.iterrows()
                ],
                "mean_realised_gain_of_corrected_g_L": None if gains is None else round(float(gains.mean()), 3),
                "model_fit_r2_per_class": {str(k): round(float(v), 3) for k, v in result.models.fit_r2.items()},
            }
        )
    except ImportError as exc:
        return {"error": str(exc)}
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("evaluate_batch_control_policy")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_simulation_tool_specs() -> list[dict]:
    """Return tool specs for all simulation tools registered in this module."""
    return get_tool_specs(names=_SIMULATION_TOOL_NAMES)
