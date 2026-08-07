"""(c) Kevin Dunn, 2010-2026. MIT License.

Tests for :mod:`process_improve.sensory.screening`: carry-over balanced serving
orders, near-balanced incomplete blocks, the assembled panel serving plan, and
the minimum-detectable-difference sizing helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise

import pandas as pd
import pytest

from process_improve.sensory.screening import (
    ScreeningPlan,
    cyclic_block_design,
    detectable_difference,
    plan_diagnostics,
    required_panelists,
    sensory_screening_plan,
    williams_design,
)
from process_improve.sensory.tools import (
    _DetectableDifferenceInput,
    _ScreeningPlanInput,
    get_sensory_tool_specs,
)
from process_improve.sensory.tools import (
    sensory_detectable_difference as detectable_difference_tool,
)
from process_improve.sensory.tools import (
    sensory_screening_plan as screening_plan_tool,
)

# ---------------------------------------------------------------------------
# williams_design
# ---------------------------------------------------------------------------


def _ordered_pairs(sequences: list[list[int]]) -> Counter:
    """Count every ordered (predecessor, successor) pair across the sequences."""
    pairs: Counter = Counter()
    for seq in sequences:
        for before, after in pairwise(seq):
            pairs[before, after] += 1
    return pairs


def _sequences(design: pd.DataFrame) -> list[list[int]]:
    return [grp.sort_values("position")["treatment"].tolist() for _, grp in design.groupby("sequence", sort=True)]


@pytest.mark.parametrize("n_treatments", [2, 4, 6, 8])
def test_williams_design_even_is_carryover_balanced(n_treatments: int) -> None:
    """For an even number of treatments each ordered pair follows exactly once."""
    design = williams_design(n_treatments)
    sequences = _sequences(design)

    assert len(sequences) == n_treatments
    assert all(sorted(seq) == list(range(n_treatments)) for seq in sequences)

    pairs = _ordered_pairs(sequences)
    expected = {(i, j) for i in range(n_treatments) for j in range(n_treatments) if i != j}
    assert set(pairs) == expected
    assert set(pairs.values()) == {1}


@pytest.mark.parametrize("n_treatments", [3, 5, 7])
def test_williams_design_odd_uses_two_squares_and_balances(n_treatments: int) -> None:
    """An odd treatment count needs 2t sequences; each ordered pair then follows twice."""
    design = williams_design(n_treatments)
    sequences = _sequences(design)

    assert len(sequences) == 2 * n_treatments

    pairs = _ordered_pairs(sequences)
    expected = {(i, j) for i in range(n_treatments) for j in range(n_treatments) if i != j}
    assert set(pairs) == expected
    assert set(pairs.values()) == {2}


def test_williams_design_every_treatment_appears_once_per_position() -> None:
    """The design is a Latin square: each treatment occupies each position once."""
    design = williams_design(6)
    counts = design.pivot_table(index="treatment", columns="position", aggfunc="size", values="sequence")
    assert (counts.to_numpy() == 1).all()


def test_williams_design_n_subjects_cycles_the_sequences() -> None:
    """Asking for more subjects than sequences recycles them in order."""
    design = williams_design(4, n_subjects=10)
    assert design["sequence"].nunique() == 10
    assert len(design) == 40


def test_williams_design_rejects_too_few_treatments() -> None:
    """Fewer than two treatments has no ordering to balance."""
    with pytest.raises(ValueError, match="at least 2"):
        williams_design(1)


# ---------------------------------------------------------------------------
# cyclic_block_design
# ---------------------------------------------------------------------------


def test_cyclic_block_design_recovers_a_known_bibd() -> None:
    """t=7, k=3, b=7 is the Fano plane: r=3 and every pair concurs exactly once."""
    blocks = cyclic_block_design(7, block_size=3, n_blocks=7, seed=0)
    assert len(blocks) == 7
    assert all(len(b) == 3 for b in blocks)
    assert all(len(set(b)) == 3 for b in blocks)

    diag = plan_diagnostics(blocks, n_treatments=7)
    assert diag["replication"]["min"] == diag["replication"]["max"] == 3
    assert diag["concurrence"]["min"] == diag["concurrence"]["max"] == 1
    assert diag["balanced"] is True


def test_cyclic_block_design_keeps_replication_near_equal_when_not_a_bibd() -> None:
    """An arbitrary (t, k, b) has no exact BIBD; replication must still be near-equal."""
    blocks = cyclic_block_design(20, block_size=6, n_blocks=10, seed=1)
    diag = plan_diagnostics(blocks, n_treatments=20)

    assert sum(len(b) for b in blocks) == 60
    assert diag["replication"]["max"] - diag["replication"]["min"] <= 1
    assert diag["balanced"] is False


def test_cyclic_block_design_is_deterministic_for_a_seed() -> None:
    """The same seed reproduces the same blocks; a different seed may differ."""
    assert cyclic_block_design(12, block_size=4, n_blocks=9, seed=7) == cyclic_block_design(
        12, block_size=4, n_blocks=9, seed=7
    )


def test_cyclic_block_design_rejects_block_larger_than_treatments() -> None:
    """A block cannot hold more distinct treatments than exist."""
    with pytest.raises(ValueError, match="block_size"):
        cyclic_block_design(4, block_size=5, n_blocks=3)


# ---------------------------------------------------------------------------
# sensory_screening_plan
# ---------------------------------------------------------------------------

CANDIDATES = [f"C{i:02d}" for i in range(1, 22)]


def test_screening_plan_covers_every_candidate_at_least_the_requested_replicates() -> None:
    """Every candidate is served at least `replicates` times, and near-equally."""
    result = sensory_screening_plan(
        CANDIDATES,
        n_panelists=12,
        samples_per_session=6,
        control="Base",
        replicates=2,
        seed=0,
    )
    assert isinstance(result, ScreeningPlan)
    served = result.plan.loc[result.plan["role"] == "test", "product"].value_counts()
    assert set(served.index) == set(CANDIDATES)
    assert served.min() >= 2
    assert served.max() - served.min() <= 1
    assert result.diagnostics["replication"]["min"] == served.min()


def test_screening_plan_puts_the_control_in_every_block() -> None:
    """The reference anchors every panelist-session block, at the first position."""
    result = sensory_screening_plan(
        CANDIDATES,
        n_panelists=10,
        samples_per_session=6,
        control="Base",
        seed=0,
    )
    blocks = result.plan.groupby(["panelist_id", "session"])
    for _, block in blocks:
        assert (block["product"] == "Base").sum() == 1
        assert block.loc[block["product"] == "Base", "position"].iloc[0] == 1
    assert result.diagnostics["control_coverage"] == pytest.approx(1.0)


def test_screening_plan_respects_the_session_capacity() -> None:
    """No panelist ever sees more than `samples_per_session` samples in a session."""
    result = sensory_screening_plan(CANDIDATES, n_panelists=8, samples_per_session=5, control="Base", seed=2)
    sizes = result.plan.groupby(["panelist_id", "session"]).size()
    assert sizes.max() <= 5
    assert result.plan["position"].max() <= 5


def test_screening_plan_never_repeats_a_product_within_a_block() -> None:
    """A panelist does not taste the same sample twice in one session."""
    result = sensory_screening_plan(CANDIDATES, n_panelists=9, samples_per_session=7, control="Base", seed=3)
    for _, block in result.plan.groupby(["panelist_id", "session"]):
        assert block["product"].is_unique


def test_screening_plan_without_a_control_uses_the_whole_capacity() -> None:
    """With no control every slot in the block is a test sample."""
    result = sensory_screening_plan(CANDIDATES, n_panelists=7, samples_per_session=6, seed=4)
    assert (result.plan["role"] == "test").all()
    assert result.diagnostics["control_coverage"] == 0.0


def test_screening_plan_is_reproducible_for_a_seed() -> None:
    """The same seed gives an identical plan."""
    kwargs = {"n_panelists": 6, "samples_per_session": 5, "control": "Base", "seed": 11}
    first = sensory_screening_plan(CANDIDATES, **kwargs).plan
    second = sensory_screening_plan(CANDIDATES, **kwargs).plan
    pd.testing.assert_frame_equal(first, second)


def test_screening_plan_warns_when_the_panel_is_too_small_for_the_candidates() -> None:
    """A capacity shortfall is reported rather than silently dropping candidates."""
    result = sensory_screening_plan(
        CANDIDATES,
        n_panelists=2,
        samples_per_session=4,
        control="Base",
        n_sessions=1,
        seed=5,
    )
    assert result.warnings
    assert any("cannot cover" in w or "not every" in w.lower() for w in result.warnings)


def test_screening_plan_rejects_a_capacity_of_one_when_a_control_is_anchored() -> None:
    """A control that fills the only slot leaves no room for a test sample."""
    with pytest.raises(ValueError, match="samples_per_session"):
        sensory_screening_plan(CANDIDATES, n_panelists=5, samples_per_session=1, control="Base")


def test_screening_plan_rejects_duplicate_candidates() -> None:
    """Duplicated labels would silently double a candidate's replication."""
    with pytest.raises(ValueError, match="duplicate"):
        sensory_screening_plan(["A", "B", "A"], n_panelists=4, samples_per_session=3)


def test_screening_plan_carries_the_long_format_columns() -> None:
    """The plan is a serving sheet, ready to be joined to scores."""
    result = sensory_screening_plan(CANDIDATES, n_panelists=6, samples_per_session=6, control="Base", seed=6)
    assert list(result.plan.columns) == [
        "panelist_id",
        "session",
        "position",
        "product",
        "role",
        "block",
    ]
    assert result.config["n_panelists"] == 6


# ---------------------------------------------------------------------------
# detectable_difference / required_panelists
# ---------------------------------------------------------------------------


def test_detectable_difference_shrinks_as_the_panel_grows() -> None:
    """More assessors per product means a smaller difference is detectable."""
    small = detectable_difference(sd=1.5, n_per_product=8)["difference"]
    large = detectable_difference(sd=1.5, n_per_product=32)["difference"]
    assert large < small


def test_detectable_difference_grows_with_more_comparisons() -> None:
    """Correcting for a family of comparisons costs sensitivity."""
    one = detectable_difference(sd=1.5, n_per_product=16, n_comparisons=1)["difference"]
    many = detectable_difference(sd=1.5, n_per_product=16, n_comparisons=21)["difference"]
    assert many > one


def test_detectable_difference_scales_linearly_with_the_noise() -> None:
    """Doubling the residual SD doubles the minimum detectable difference."""
    base = detectable_difference(sd=1.0, n_per_product=20)["difference"]
    doubled = detectable_difference(sd=2.0, n_per_product=20)["difference"]
    assert doubled == pytest.approx(2 * base, rel=1e-9)


def test_detectable_difference_reports_its_inputs() -> None:
    """The result is self-describing so it can be quoted in a report."""
    out = detectable_difference(sd=1.2, n_per_product=12, alpha=0.05, power=0.8)
    assert out["sd"] == 1.2
    assert out["n_per_product"] == 12
    assert out["power"] == 0.8
    assert out["difference"] > 0


def test_detectable_difference_rejects_a_single_assessor() -> None:
    """With one assessor per product there are no error degrees of freedom."""
    with pytest.raises(ValueError, match="n_per_product"):
        detectable_difference(sd=1.0, n_per_product=1)


def test_required_panelists_delivers_the_requested_difference() -> None:
    """The returned n actually reaches the target difference, and n-1 does not."""
    target = 0.8
    n = required_panelists(sd=1.5, difference=target)["n_per_product"]
    assert detectable_difference(sd=1.5, n_per_product=n)["difference"] <= target
    assert detectable_difference(sd=1.5, n_per_product=n - 1)["difference"] > target


def test_required_panelists_rejects_a_non_positive_difference() -> None:
    """A zero or negative target difference is never reachable."""
    with pytest.raises(ValueError, match="difference"):
        required_panelists(sd=1.0, difference=0.0)


# ---------------------------------------------------------------------------
# Agent-callable tool wrappers
# ---------------------------------------------------------------------------


def test_screening_plan_tool_returns_json_serialisable_rows() -> None:
    """The tool hands back plain records the agent can pass straight on."""
    result = screening_plan_tool(
        _ScreeningPlanInput(products=CANDIDATES, n_panelists=12, samples_per_session=6, control="Base", seed=0)
    )
    assert result["ok"] is True
    assert isinstance(result["plan"], list)
    assert set(result["plan"][0]) == {"panelist_id", "session", "position", "product", "role", "block"}
    assert result["diagnostics"]["control_coverage"] == 1.0
    json.dumps(result)


def test_screening_plan_tool_reports_bad_input_instead_of_raising() -> None:
    """A design that cannot be built comes back as ok=False so the agent can relay it."""
    result = screening_plan_tool(
        _ScreeningPlanInput(products=CANDIDATES, n_panelists=5, samples_per_session=1, control="Base")
    )
    assert result["ok"] is False
    assert any("samples_per_session" in error for error in result["errors"])


def test_detectable_difference_tool_answers_both_questions() -> None:
    """One argument asks what the panel can see; the other asks how big it must be."""
    seeing = detectable_difference_tool(_DetectableDifferenceInput(sd=1.5, n_per_product=24))
    sizing = detectable_difference_tool(_DetectableDifferenceInput(sd=1.5, difference=1.0))
    assert seeing["ok"] is True
    assert seeing["difference"] > 0
    assert sizing["ok"] is True
    assert sizing["achieved_difference"] <= 1.0


def test_detectable_difference_tool_refuses_both_or_neither() -> None:
    """Giving both (or neither) is ambiguous and is rejected, not guessed at."""
    both = detectable_difference_tool(_DetectableDifferenceInput(sd=1.5, n_per_product=24, difference=1.0))
    neither = detectable_difference_tool(_DetectableDifferenceInput(sd=1.5))
    assert both["ok"] is False
    assert neither["ok"] is False


def test_screening_tools_are_registered_for_the_agent() -> None:
    """Both tools appear in the sensory tool specs so an LLM can call them."""
    names = {spec["name"] for spec in get_sensory_tool_specs()}
    assert {"sensory_screening_plan", "sensory_detectable_difference"} <= names


# ---------------------------------------------------------------------------
# Guard clauses
#
# Every one of these is a way to ask for a design that cannot exist. They are
# tested individually because the message is the deliverable: a scientist who
# gets these back needs to know which number to change.
# ---------------------------------------------------------------------------


def test_williams_design_rejects_a_non_positive_subject_count() -> None:
    """Zero subjects is not a design."""
    with pytest.raises(ValueError, match="n_subjects"):
        williams_design(4, n_subjects=0)


def test_cyclic_block_design_rejects_fewer_than_two_treatments() -> None:
    """There is nothing to block with a single treatment."""
    with pytest.raises(ValueError, match="at least 2 treatments"):
        cyclic_block_design(1, block_size=1, n_blocks=3)


def test_cyclic_block_design_rejects_an_empty_block() -> None:
    """A block of size zero would serve nothing."""
    with pytest.raises(ValueError, match="block_size"):
        cyclic_block_design(6, block_size=0, n_blocks=3)


def test_cyclic_block_design_rejects_a_non_positive_block_count() -> None:
    """Asking for no blocks is a caller error, not an empty design."""
    with pytest.raises(ValueError, match="n_blocks"):
        cyclic_block_design(6, block_size=3, n_blocks=0)


def test_screening_plan_rejects_an_empty_panel() -> None:
    """No assessors means no plan."""
    with pytest.raises(ValueError, match="n_panelists"):
        sensory_screening_plan(CANDIDATES, n_panelists=0, samples_per_session=5)


def test_screening_plan_rejects_non_positive_replicates() -> None:
    """Every candidate must be served at least once."""
    with pytest.raises(ValueError, match="replicates"):
        sensory_screening_plan(CANDIDATES, n_panelists=6, samples_per_session=5, replicates=0)


def test_screening_plan_rejects_a_non_positive_session_count() -> None:
    """Fixing the sessions at zero contradicts asking for a plan."""
    with pytest.raises(ValueError, match="n_sessions"):
        sensory_screening_plan(CANDIDATES, n_panelists=6, samples_per_session=5, n_sessions=0)


def test_screening_plan_rejects_a_single_candidate() -> None:
    """One candidate is not a screen."""
    with pytest.raises(ValueError, match="at least 2 candidates"):
        sensory_screening_plan(["Only one"], n_panelists=6, samples_per_session=5)


def test_screening_plan_rejects_a_control_that_is_also_a_candidate() -> None:
    """The reference cannot double as a treatment; its replication would be wrong."""
    with pytest.raises(ValueError, match="must not also appear"):
        sensory_screening_plan([*CANDIDATES, "Base"], n_panelists=6, samples_per_session=5, control="Base")


def test_screening_plan_handles_a_single_slot_per_block() -> None:
    """With one test slot per block there is no order to balance, and that is fine."""
    result = sensory_screening_plan(CANDIDATES, n_panelists=6, samples_per_session=2, control="Base", seed=12)

    for _, block in result.plan.groupby("block"):
        assert (block["role"] == "test").sum() == 1
    assert result.diagnostics["position_balance"] == 0.0


def test_detectable_difference_rejects_a_non_positive_sd() -> None:
    """A zero or negative noise estimate is not a noise estimate."""
    with pytest.raises(ValueError, match="sd"):
        detectable_difference(sd=0.0, n_per_product=10)


@pytest.mark.parametrize(("alpha", "power"), [(0.0, 0.8), (1.0, 0.8), (0.05, 0.0), (0.05, 1.0)])
def test_detectable_difference_rejects_out_of_range_rates(alpha: float, power: float) -> None:
    """Both alpha and power are probabilities strictly inside (0, 1)."""
    with pytest.raises(ValueError, match=r"alpha|power"):
        detectable_difference(sd=1.0, n_per_product=10, alpha=alpha, power=power)


def test_detectable_difference_rejects_an_empty_comparison_family() -> None:
    """There is always at least one comparison being made."""
    with pytest.raises(ValueError, match="n_comparisons"):
        detectable_difference(sd=1.0, n_per_product=10, n_comparisons=0)


def test_required_panelists_refuses_an_unreachable_target() -> None:
    """A target the search ceiling cannot reach raises rather than looping or lying."""
    with pytest.raises(ValueError, match="No n_per_product up to"):
        required_panelists(sd=10.0, difference=0.001, max_n=20)


def test_detectable_difference_tool_relays_an_unreachable_target() -> None:
    """A target no panel size can reach comes back as ok=False, not as an exception."""
    result = detectable_difference_tool(_DetectableDifferenceInput(sd=1e6, difference=1e-6))

    assert result["ok"] is False
    assert any("No n_per_product up to" in error for error in result["errors"])
