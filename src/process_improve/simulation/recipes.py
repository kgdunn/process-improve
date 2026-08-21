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
