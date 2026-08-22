"""(c) Kevin Dunn, 2010-2026. MIT License.

Analysis recipes for the batch (bioreactor) simulator.

One guided workflow: establish, quantitatively, why replaying a golden
batch's schedule does not reproduce its outcome, and how the remaining
variance splits between what can be addressed before a batch and what can
only be addressed while it runs. Registered into the package-wide catalog
on import; see :mod:`process_improve.recipes` for the framework.
"""

from __future__ import annotations

from process_improve.recipes import AnalysisRecipe, RecipeStep, register_recipe

_DOMAIN = "simulation"

# Hoisted so the list elements below are single names: implicit string
# concatenation directly inside a list looks like a missing comma to static
# analysis (CodeQL py/implicit-string-concatenation-in-list).
_INPUT_SIMULATOR = (
    "nothing external: the workflow runs on the package's bioreactor simulator with its default "
    "configuration, so every number is reproducible from the stated seed"
)
_INPUT_SCALES = (
    "optionally, disturbance channel scales to match a scenario (for example ic_scale=0 for perfectly "
    "consistent raw materials)"
)


_GOLDEN_BATCH_BASELINE = AnalysisRecipe(
    key="golden_batch_baseline",
    title="Golden batch baseline: why replaying the best batch does not repeat it",
    summary=(
        "Quantify what happens when a batch process replays its best (golden) batch schedule open-loop. "
        "Simulate a replay campaign to measure the outcome spread, hold the measured initial conditions "
        "identical to show the spread that remains, then decompose the variance into the share addressable "
        "before the batch (feedforward adaptation), the share observable only while the batch runs "
        "(mid-course correction), and the noise floor. Use this to ground a discussion of batch trajectory "
        "adaptation in numbers rather than assertion."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "golden batch",
        "best batch",
        "replicate our best run",
        "replay the recipe",
        "batches do not repeat",
        "batches don't repeat",
        "batch to batch variation",
        "batch-to-batch variability",
        "same recipe different result",
        "why do batches differ",
        "mid-course correction",
        "trajectory adaptation",
        "batch consistency",
    ],
    inputs_needed=[_INPUT_SIMULATOR, _INPUT_SCALES],
    stages=[
        RecipeStep(
            order=1,
            directive=(
                "Run a replay campaign with all disturbance channels at their defaults: "
                "simulate_batch_campaign(n_batches=50, policy='replay', random_state=0). Report the titer "
                "mean, standard deviation and coefficient of variation, and compare the spread against the "
                "disturbance-free reference titer the tool returns: the schedule is identical for every "
                "batch, yet the outcomes are not."
            ),
            tools=["simulate_batch_campaign"],
        ),
        RecipeStep(
            order=2,
            directive=(
                "Repeat the campaign with identical measured initial conditions for every batch: "
                "simulate_batch_campaign(n_batches=50, policy='replay', ic_scale=0.0, random_state=0). The "
                "spread that remains cannot be removed by characterising incoming materials better; it "
                "arises during the batch."
            ),
            tools=["simulate_batch_campaign"],
        ),
        RecipeStep(
            order=3,
            directive=(
                "Decompose the variance: decompose_batch_quality_variance(random_state=0). Present the four "
                "sources (measured initial conditions, within-batch disturbance, noise, interaction "
                "residual) with their shares, and state which intervention reaches which share: schedule "
                "adaptation before the batch for the first, mid-course correction at decision points for "
                "the second, neither for the floor."
            ),
            tools=["decompose_batch_quality_variance"],
        ),
    ],
)

register_recipe(_GOLDEN_BATCH_BASELINE)


_MIDCOURSE_CORRECTION = AnalysisRecipe(
    key="midcourse_correction",
    title="Mid-course correction: steer a running batch back toward its quality target",
    summary=(
        "Demonstrate, with executed (re-simulated) counterfactuals rather than model predictions, how a "
        "latent-variable mid-course correction recovers quality on batches headed for a shortfall. Establish "
        "the replay baseline first, correct a handful of fresh batches at a decision point to see the "
        "workflow per batch (validity gate, dead band, corrected schedule, realised outcome), then run the "
        "campaign-level policy comparison, optionally against the perfect-feedforward and "
        "oracle-from-the-decision-point ceilings that bracket what any scheme could achieve."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "mid-course correction",
        "midcourse correction",
        "correct a running batch",
        "steer the batch",
        "batch is heading low",
        "save this batch",
        "adjust the rest of the batch",
        "advanced batch control",
        "batch quality control",
        "correct the schedule mid-batch",
    ],
    inputs_needed=[_INPUT_SIMULATOR, _INPUT_SCALES],
    stages=[
        RecipeStep(
            order=1,
            directive=(
                "Establish the do-nothing baseline: simulate_batch_campaign(n_batches=40, policy='replay', "
                "random_state=0). Note the titer mean and spread, and which feed classes sit below the "
                "quality target; those are the batches a correction could help."
            ),
            tools=["simulate_batch_campaign"],
        ),
        RecipeStep(
            order=2,
            directive=(
                "Walk through the decision point on a few batches: correct_batch_midcourse(n_batches=5, "
                "decision_point=8, y_target=8.0, random_state=0). For each batch report whether it was "
                "corrected, gated by the model-validity check, or left alone by the dead band, and compare "
                "the executed titer with the replay titer; the difference is a same-batch counterfactual, "
                "not a prediction."
            ),
            tools=["correct_batch_midcourse"],
        ),
        RecipeStep(
            order=3,
            directive=(
                "Run the campaign-level comparison: evaluate_batch_control_policy(n_test=40, "
                "decision_point=8, y_target=8.0, random_state=0). Report the realised mean and spread per "
                "policy and the mean realised gain of the corrected batches. If the discussion needs the "
                "ceilings (what a perfect feedforward or a mechanistic optimiser could still recover), "
                "re-run with include_ceilings=True and present the gap honestly: it is the price of an "
                "empirical model restricted to the region its history explored."
            ),
            tools=["evaluate_batch_control_policy"],
        ),
    ],
)

register_recipe(_MIDCOURSE_CORRECTION)
