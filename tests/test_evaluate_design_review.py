"""Regression tests for review findings on ``evaluate_design``.

Each test names the defect it pins: the statistics are checked against the textbook
definition, not against the implementation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design, generate_design
from process_improve.experiments.factor import DesignResult


def _two_level(k: int, names: str = "ABCDEFGH") -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product([-1, 1], repeat=k)), columns=list(names[:k]))


# ---------------------------------------------------------------------------
# Clear effects (Wu & Hamada, 2009, Sec. 5.2)
# ---------------------------------------------------------------------------


class TestClearEffects:
    def test_resolution_iii_has_no_clear_main_effect_whatever_the_names(self) -> None:
        """A main effect aliased with a 2FI is not clear, also for multi-letter names."""
        for names in (["Temp", "Press", "Time", "Conc", "Speed"], list("ABCDE")):
            fs = [Factor(name=n, low=-1, high=1) for n in names]
            r = generate_design(fs, design_type="fractional_factorial", resolution=3, n_center_points=0)
            clear = evaluate_design(r, model="main_effects", metric="clear_effects")["clear_effects"]
            assert clear["main_effects"] == [], names

    def test_multi_letter_two_factor_interactions_can_be_clear(self) -> None:
        """In a 2^(5-1)_V every main effect and 2FI is clear, whatever the names."""
        names = ["Temp", "Press", "Time", "Conc", "Speed"]
        fs = [Factor(name=n, low=-1, high=1) for n in names]
        r = generate_design(fs, design_type="fractional_factorial", generators=["Speed=TempPressTimeConc"])
        clear = evaluate_design(r, model="main_effects", metric="clear_effects")["clear_effects"]
        assert clear["main_effects"] == names
        assert len(clear["two_factor_interactions"]) == 10

    def test_full_factorial_effects_are_all_clear(self) -> None:
        """An effect aliased with nothing is clear."""
        r = generate_design([Factor(name=n, low=-1, high=1) for n in "ABC"], "full_factorial", n_center_points=0)
        clear = evaluate_design(r, model="interactions", metric="clear_effects")["clear_effects"]
        assert clear["main_effects"] == ["A", "B", "C"]
        assert clear["two_factor_interactions"] == ["A:B", "A:C", "B:C"]

    def test_resolution_v_dataframe_main_effects_are_clear(self) -> None:
        """Main effects aliased only with 4FIs are clear (correlation path, no generators)."""
        d = _two_level(4)
        d["E"] = d.A * d.B * d.C * d.D
        clear = evaluate_design(d, model="main_effects", metric="clear_effects")["clear_effects"]
        assert clear["main_effects"] == list("ABCDE")
        assert len(clear["two_factor_interactions"]) == 10

    def test_resolution_iii_dataframe(self) -> None:
        d = _two_level(2)
        d["C"] = d.A * d.B
        clear = evaluate_design(d, model="main_effects", metric="clear_effects")["clear_effects"]
        assert clear == {"main_effects": [], "two_factor_interactions": []}


# ---------------------------------------------------------------------------
# Signed generators
# ---------------------------------------------------------------------------


class TestNegativeGenerators:
    def test_defining_relation_and_aliases_keep_the_sign(self) -> None:
        r = generate_design(
            [Factor(name=n, low=-1, high=1) for n in "ABCD"],
            design_type="fractional_factorial",
            generators=["D=-ABC"],
            n_center_points=0,
        )
        d = r.design
        assert set((d.A * d.B * d.C * d.D).unique()) == {-1}
        out = evaluate_design(r, metric=["defining_relation", "alias_structure", "confounding"])
        assert out["defining_relation"] == ["I=-ABCD"]
        assert "A = -BCD" in out["alias_structure"]
        assert "AB = -CD" in out["alias_structure"]
        assert {"effect": "A", "confounded_with": ["BCD"]} in out["confounding"]

    def test_two_negative_generators_multiply_to_a_positive_word(self) -> None:
        names = list("ABCDE")
        fs = [Factor(name=n, low=-1, high=1) for n in names]
        r = generate_design(fs, "fractional_factorial", generators=["D=-AB", "E=-AC"], n_center_points=0)
        d = r.design
        relation = evaluate_design(r, metric="defining_relation")["defining_relation"]
        assert set(relation) == {"I=-ABD", "I=-ACE", "I=BCDE"}
        assert set((d.B * d.C * d.D * d.E).unique()) == {1}


# ---------------------------------------------------------------------------
# Resolution II
# ---------------------------------------------------------------------------


def test_resolution_ii_roman_numeral_and_wordlength_pattern() -> None:
    d = _two_level(3)
    d["D"] = d.A
    r = DesignResult(
        design=d,
        design_actual=d,
        run_order=list(range(1, 9)),
        design_type="fractional_factorial",
        n_runs=8,
        n_factors=4,
        factor_names=list("ABCD"),
        generators=["D=A"],
    )
    out = evaluate_design(r, model="main_effects", metric=["resolution", "minimum_aberration"])
    assert out["resolution"] == 2
    assert out["roman"] == "II"
    assert out["minimum_aberration"]["wordlength_pattern"] == [1]
    assert out["minimum_aberration"]["wordlength_pattern_labels"] == ["A_2"]


# ---------------------------------------------------------------------------
# Alias structure of a CCD built on a fractional cube
# ---------------------------------------------------------------------------


def test_ccd_alias_structure_uses_the_whole_design() -> None:
    """The axial runs break the cube's aliasing, so A = BCDE is not reported as full aliasing."""
    fs = [Factor(name=n, low=-1, high=1) for n in "ABCDE"]
    r = generate_design(fs, design_type="ccd", cube="fractional", n_center_points=2)
    assert r.generators
    out = evaluate_design(r, model="quadratic", metric=["alias_structure", "clear_effects"])
    assert not any(chain.startswith("A = ") for chain in out["alias_structure"])
    assert "cube" in out["notes"]["alias_structure"]
    assert out["clear_effects"]["main_effects"] == list("ABCDE")
    d = r.design
    assert abs(np.corrcoef(d.A, d.B * d.C * d.D * d.E)[0, 1]) < 0.995


@pytest.mark.parametrize("metric", ["alias_structure", "clear_effects"])
def test_fractional_factorial_keeps_generator_chains(metric: str) -> None:
    fs = [Factor(name=n, low=-1, high=1) for n in "ABCD"]
    r = generate_design(fs, design_type="fractional_factorial", generators=["D=ABC"], n_center_points=2)
    out = evaluate_design(r, metric=metric)
    assert "notes" not in out


def test_negative_generator_is_signed_in_the_design_metadata() -> None:
    r = generate_design(
        [Factor(name=n, low=-1, high=1) for n in "ABCD"],
        design_type="fractional_factorial",
        generators=["D=-ABC"],
        n_center_points=0,
    )
    assert r.defining_relation == ["I=-ABCD"]


def test_each_metric_keeps_its_own_note() -> None:
    """Notes are keyed by metric, so one metric's note does not replace another's."""
    d = _two_level(2)
    for order in (["d_efficiency", "resolution"], ["resolution", "d_efficiency"]):
        out = evaluate_design(d, model="quadratic", metric=order)
        assert "note" not in out
        assert "rank-deficient" in out["notes"]["d_efficiency"]
        assert "fractional factorial" in out["notes"]["resolution"]
