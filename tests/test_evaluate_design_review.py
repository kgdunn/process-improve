"""Regression tests for review findings on ``evaluate_design``.

Each test names the defect it pins: the statistics are checked against the textbook
definition, not against the implementation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from patsy import build_design_matrices, dmatrix

from process_improve.experiments import Factor, evaluate_all, evaluate_design, generate_design
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


# ---------------------------------------------------------------------------
# Categorical factors: alias matrix and the region
# ---------------------------------------------------------------------------


def _with_categorical(base: pd.DataFrame, levels: str = "xy") -> pd.DataFrame:
    return pd.concat([base.assign(C=lev) for lev in levels], ignore_index=True)


class TestCategoricalFactor:
    def test_alias_matrix_with_a_categorical_factor(self) -> None:
        d = _with_categorical(_two_level(2))
        out = evaluate_design(d, model="main_effects", metric="alias_matrix")["alias_matrix"]
        assert out["model_terms"] == ["Intercept", "C[T.y]", "A", "B"]
        assert out["alias_terms"] == ["A:B", "A:C[T.y]", "B:C[T.y]"]
        # Reference: the same matrix from an explicit 0/1 dummy for C (A:C[T.y] is half aliased with A).
        c = (d.C == "y").astype(float).to_numpy()
        x1 = np.column_stack([np.ones(len(d)), c, d.A, d.B])
        x2 = np.column_stack([d.A * d.B, d.A * c, d.B * c])
        expected = np.linalg.solve(x1.T @ x1, x1.T @ x2)
        np.testing.assert_allclose(out["matrix"], expected, atol=1e-12)

    def test_evaluate_all_runs_with_a_categorical_factor(self) -> None:
        d = _with_categorical(_two_level(2))
        out = evaluate_all(d, model="main_effects", n_samples=2000)
        assert out["alias_matrix"]["alias_terms"] == ["A:B", "A:C[T.y]", "B:C[T.y]"]

    def test_region_is_honoured_with_a_categorical_factor(self) -> None:
        base = pd.DataFrame(list(itertools.product([-1, 0, 1], repeat=2)), columns=list("AB"))
        d = _with_categorical(base)
        mod = "A + B + C + A:B + I(A**2) + I(B**2)"
        cube = evaluate_design(d, model=mod, metric="g_efficiency", region="cuboidal", n_samples=5000)
        ball = evaluate_design(d, model=mod, metric="g_efficiency", region="spherical", n_samples=5000)
        assert ball["max_prediction_variance"] > cube["max_prediction_variance"]
        with pytest.raises(ValueError, match="Unknown region"):
            evaluate_design(d, model=mod, metric="g_efficiency", region="bogus")

    def test_every_vertex_is_crossed_with_every_level(self) -> None:
        rows = [(a, b, c) for c in "xy" for a in (-1, 0, 1) for b in (-1, 0, 1)]
        rows += [(-1, -1, "z"), (1, 1, "z"), (1, -1, "z"), (0, 0, "z")]
        d = pd.DataFrame(rows, columns=["A", "B", "C"])
        mod = "(A+B+C)**2+I(A**2)+I(B**2)"
        out = evaluate_design(d, model=mod, metric="g_efficiency", n_samples=2000)
        dm = dmatrix(mod, d, return_type="dataframe")
        xtx_inv = np.linalg.inv(np.asarray(dm).T @ np.asarray(dm))
        corners = pd.DataFrame(
            [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in "xyz"],
            columns=["A", "B", "C"],
        )
        p = np.asarray(build_design_matrices([dm.design_info], corners)[0])
        assert out["max_prediction_variance"] == pytest.approx(np.max(np.sum(p @ xtx_inv * p, axis=1)))


# ---------------------------------------------------------------------------
# Region average is taken over the uniform sample only
# ---------------------------------------------------------------------------


def test_vertices_do_not_bias_the_average_prediction_variance() -> None:
    """The I-criterion is an integral over the region; the corners only serve the maximum."""
    d = _two_level(3)
    # Exact average of x'(X'X)^-1 x over [-1, 1]^3 for the main-effects model of a 2^3: (1 + 3/3) / 8.
    exact = (1 + 3 * (1 / 3)) / 8
    out = evaluate_design(d, model="main_effects", metric=["average_prediction_variance", "fds"], n_samples=50)
    with_vertices = out["average_prediction_variance"]
    without = evaluate_design(
        d, model="main_effects", metric="average_prediction_variance", n_samples=50, include_vertices=False
    )["average_prediction_variance"]
    assert with_vertices == pytest.approx(without)
    assert with_vertices == pytest.approx(exact, rel=0.15)
    assert out["fds"]["max_prediction_variance"] == pytest.approx(0.5)  # a corner: (1 + 3) / 8
    assert out["fds"]["average_prediction_variance"] == pytest.approx(with_vertices)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_factor_setting_raises(self) -> None:
        d = _two_level(3).astype(float)
        d.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match=r"missing.*'A'"):
            evaluate_design(d, model="main_effects", metric="d_efficiency")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
    def test_alpha_outside_unit_interval_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            evaluate_design(_two_level(3), model="main_effects", metric="power", alpha=alpha, effect_size=1)

    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_non_positive_sigma_raises(self, sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma"):
            evaluate_design(_two_level(3), model="main_effects", metric="power", sigma=sigma, effect_size=1)

    def test_zero_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            evaluate_design(_two_level(3), model="main_effects", metric="g_efficiency", n_samples=0)

    def test_actual_units_warn(self) -> None:
        with pytest.warns(UserWarning, match="coded units"):
            evaluate_design(_two_level(3) * 5 + 5, model="main_effects", metric="d_efficiency")

    def test_rotatable_ccd_does_not_warn(self, recwarn: pytest.WarningsRecorder) -> None:
        fs = [Factor(name=n, low=-1, high=1) for n in "ABCDEF"]
        r = generate_design(fs, design_type="ccd", alpha="rotatable", n_center_points=2)
        evaluate_design(r, model="quadratic", metric="d_efficiency")
        assert not [w for w in recwarn if "coded units" in str(w.message)]
