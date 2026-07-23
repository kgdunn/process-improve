"""Tests for the designed-mode comparison module (:mod:`process_improve.sensory.designed`).

A single synthetic randomized-complete-block scenario drives most tests: seven
panelists score five formulations (one named ``Control``) at three aging
conditions (``REF``, ``RT``, ``HB``) on two attributes. On attribute ``A`` the
formulation ``T4`` is planted well above the four others (which are identical),
aging lowers every score, and a formulation-by-condition interaction is planted
so that ``T2`` collapses only under ``HB``. Attribute ``B`` carries no
formulation signal (a negative control). Panelist offsets make the block matter.
Because the truth is known, the ANOVA F-tests, Tukey groupings,
Dunnett-vs-control flags and letter display can each be checked against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.sensory import (
    ComparisonResult,
    compare_products,
    dunnett_vs_control,
    factorial_anova,
    tukey_hsd,
)
from process_improve.sensory.designed import _compact_letter_display

PANELISTS = [f"P{i}" for i in range(7)]
FORMULATIONS = ["Control", "T1", "T2", "T3", "T4"]
CONDITIONS = ["REF", "RT", "HB"]
_FORM_EFFECT = {"Control": 0.0, "T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 3.0}
_COND_EFFECT = {"REF": 0.0, "RT": -0.5, "HB": -1.0}
_NOISE_SD = 0.35


def make_rcbd_panel(seed: int = 0) -> pd.DataFrame:
    """Return a descriptive-long panel with the planted effects described above."""
    rng = np.random.default_rng(seed)
    panelist_offset = {pid: rng.normal(0.0, 0.5) for pid in PANELISTS}
    rows: list[dict[str, object]] = []
    for pid in PANELISTS:
        for form in FORMULATIONS:
            for cond in CONDITIONS:
                interaction = -2.0 if (form == "T2" and cond == "HB") else 0.0
                mean_a = 5.0 + _FORM_EFFECT[form] + _COND_EFFECT[cond] + interaction + panelist_offset[pid]
                mean_b = 5.0 + _COND_EFFECT[cond] + panelist_offset[pid]  # no formulation signal
                for attribute, center in (("A", mean_a), ("B", mean_b)):
                    rows.append(
                        {
                            "panelist_id": pid,
                            "product": f"{form}-{cond}",
                            "formulation": form,
                            "condition": cond,
                            "attribute": attribute,
                            "replicate": 1,
                            "score": center + rng.normal(0.0, _NOISE_SD),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_rcbd_panel(seed=0)


def test_factorial_anova_detects_planted_effects(panel: pd.DataFrame) -> None:
    table = factorial_anova(panel, factors=["formulation", "condition"])
    attr_a = table[table["attribute"] == "A"].set_index("source")
    # Every planted term is strongly significant on attribute A.
    assert attr_a.loc["formulation", "p_value"] < 1e-6
    assert attr_a.loc["condition", "p_value"] < 1e-3
    assert attr_a.loc["formulation:condition", "p_value"] < 1e-3
    assert attr_a.loc["panelist_id", "p_value"] < 1e-3
    # Degrees of freedom: 5 formulations, 3 conditions, 7 panelists.
    assert attr_a.loc["formulation", "df"] == pytest.approx(4.0)
    assert attr_a.loc["condition", "df"] == pytest.approx(2.0)
    assert attr_a.loc["formulation:condition", "df"] == pytest.approx(8.0)


def test_factorial_anova_null_attribute_has_no_formulation_effect(panel: pd.DataFrame) -> None:
    table = factorial_anova(panel, factors=["formulation", "condition"])
    attr_b = table[table["attribute"] == "B"].set_index("source")
    # Attribute B carries no formulation signal, so its effect should not be significant.
    assert attr_b.loc["formulation", "p_value"] > 0.05
    assert attr_b.loc["formulation:condition", "p_value"] > 0.05
    # Aging is still present on B.
    assert attr_b.loc["condition", "p_value"] < 1e-3


def test_factorial_anova_missing_column_raises(panel: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="missing required column"):
        factorial_anova(panel, factors=["not_a_column"])


def test_tukey_only_high_formulation_separates_at_ref(panel: pd.DataFrame) -> None:
    ref = panel[panel["condition"] == "REF"]
    tukey = tukey_hsd(ref, factor="formulation")
    attr_a = tukey[tukey["attribute"] == "A"]

    def rejected(a: str, b: str) -> bool:
        row = attr_a[
            ((attr_a["group1"] == a) & (attr_a["group2"] == b)) | ((attr_a["group1"] == b) & (attr_a["group2"] == a))
        ]
        return bool(row["reject"].iloc[0])

    # T4 separates from all four others; none of the low four separate from each other.
    for other in ["Control", "T1", "T2", "T3"]:
        assert rejected("T4", other)
    for a, b in [("Control", "T1"), ("Control", "T2"), ("T1", "T3"), ("T2", "T3")]:
        assert not rejected(a, b)


def test_tukey_matches_scipy_on_balanced_oneway() -> None:
    """Without a block, the studentized-range maths must match scipy.stats.tukey_hsd."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    groups = {g: rng.normal(loc, 1.0, 8) for g, loc in zip("WXYZ", [0.0, 0.5, 2.0, 2.2], strict=True)}
    rows = [
        {"panelist_id": f"P{i}", "attribute": "A", "score": val, "grp": g, "replicate": 1}
        for g, arr in groups.items()
        for i, val in enumerate(arr)
    ]
    long = pd.DataFrame(rows)
    ours = tukey_hsd(long, factor="grp", block=None).set_index(["group1", "group2"])
    reference = scipy_stats.tukey_hsd(*groups.values())
    labels = list(groups.keys())
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                key = (a, b) if (a, b) in ours.index else (b, a)
                assert ours.loc[key, "p_value"] == pytest.approx(reference.pvalue[i, j], abs=1e-6)


def test_dunnett_flags_only_the_true_treatment(panel: pd.DataFrame) -> None:
    ref = panel[panel["condition"] == "REF"]
    dunnett = dunnett_vs_control(ref, factor="formulation", control="Control")
    attr_a = dunnett[dunnett["attribute"] == "A"].set_index("level")
    assert bool(attr_a.loc["T4", "reject"])
    assert attr_a.loc["T4", "meandiff"] > 2.0
    for other in ["T1", "T2", "T3"]:
        assert not bool(attr_a.loc[other, "reject"])


def test_compare_products_simple_effects_capture_interaction(panel: pd.DataFrame) -> None:
    result = compare_products(panel, factors=["formulation", "condition"], within="condition", control="Control")
    assert isinstance(result, ComparisonResult)
    assert set(result.letters["condition"]) == set(CONDITIONS)

    def letter(condition: str, formulation: str) -> str:
        row = result.letters.query("attribute == 'A' and condition == @condition and formulation == @formulation")
        return str(row["letters"].iloc[0])

    # Under HB, T2 collapses to its own group, distinct from the Control cluster.
    assert letter("HB", "T4") != letter("HB", "Control")
    assert letter("HB", "T2") != letter("HB", "Control")
    # Under REF, T2 sits with the Control cluster (no collapse there).
    assert letter("REF", "T2") == letter("REF", "Control")


def test_compare_products_without_control_has_empty_dunnett(panel: pd.DataFrame) -> None:
    result = compare_products(panel, factors=["formulation", "condition"], within="condition")
    assert result.dunnett.empty
    assert not result.tukey.empty
    assert not result.means.empty


def test_unbalanced_panel_is_handled(panel: pd.DataFrame) -> None:
    # Drop a whole panelist-by-formulation cell and a scatter of individual scores.
    unbalanced = panel[~((panel["panelist_id"] == "P0") & (panel["formulation"] == "T3"))].copy()
    rng = np.random.default_rng(11)
    drop_idx = rng.choice(unbalanced.index, size=15, replace=False)
    unbalanced = unbalanced.drop(index=drop_idx)

    table = factorial_anova(unbalanced, factors=["formulation", "condition"])
    assert not table.empty
    # Type III still recovers the dominant formulation effect on A despite the imbalance.
    attr_a = table[table["attribute"] == "A"].set_index("source")
    assert attr_a.loc["formulation", "p_value"] < 1e-4

    result = compare_products(unbalanced, factors=["formulation", "condition"], within="condition", control="Control")
    assert not result.tukey.empty
    assert not result.letters.empty


def test_compact_letter_display_handles_non_interval_pattern() -> None:
    # A can equal B and C, but B and C differ: A must carry both letters.
    levels = ["A", "B", "C"]  # ordered by mean
    sig = {frozenset(("B", "C"))}
    mapping = _compact_letter_display(levels, sig)
    assert mapping["B"] != mapping["C"]
    # A shares a letter with B and a letter with C.
    assert set(mapping["A"]) & set(mapping["B"])
    assert set(mapping["A"]) & set(mapping["C"])


def test_compact_letter_display_all_same_when_nothing_significant() -> None:
    mapping = _compact_letter_display(["A", "B", "C"], set())
    assert mapping["A"] == mapping["B"] == mapping["C"] == "a"
