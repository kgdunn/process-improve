"""Regression tests for review findings on ``evaluate_design``.

Each test names the defect it pins: the statistics are checked against the textbook
definition, not against the implementation.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from patsy import build_design_matrices, dmatrix
from scipy import stats

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
        assert out["model_terms"] == ["Intercept", "C[S.y]", "A", "B"]
        assert out["alias_terms"] == ["A:B", "A:C[S.y]", "B:C[S.y]"]
        # Reference: the same matrix from an explicit effect-coded column for C (x = -1, y = +1).
        c = np.where(d.C == "y", 1.0, -1.0)
        x1 = np.column_stack([np.ones(len(d)), c, d.A, d.B])
        x2 = np.column_stack([d.A * d.B, d.A * c, d.B * c])
        expected = np.linalg.solve(x1.T @ x1, x1.T @ x2)
        np.testing.assert_allclose(out["matrix"], expected, atol=1e-12)
        # The design is balanced, so in effect coding no main effect is biased by an interaction
        # (a 0/1 dummy made A:C half aliased with A).
        assert np.abs(expected).max() < 1e-12

    def test_evaluate_all_runs_with_a_categorical_factor(self) -> None:
        d = _with_categorical(_two_level(2))
        out = evaluate_all(d, model="main_effects", n_samples=2000)
        assert out["alias_matrix"]["alias_terms"] == ["A:B", "A:C[S.y]", "B:C[S.y]"]

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


# ---------------------------------------------------------------------------
# Degrees of freedom from the rank of X
# ---------------------------------------------------------------------------


class TestDegreesOfFreedom:
    def test_rank_deficient_model_uses_the_rank(self) -> None:
        d = pd.DataFrame(list(itertools.product([-1, 1], repeat=2)) * 2, columns=list("AB"))
        out = evaluate_design(d, model="quadratic", metric="degrees_of_freedom")["degrees_of_freedom"]
        # Rank 4 (the squares equal the intercept): model 3, residual 4, all of it pure error.
        assert out == {"model": 3, "residual": 4, "total": 7, "pure_error": 4, "lack_of_fit": 0}

    def test_no_intercept_model(self) -> None:
        d = pd.DataFrame(list(itertools.product([-1, 1], repeat=2)) + [(0, 0)] * 3, columns=list("AB"))
        out = evaluate_design(d, model="A + B - 1", metric="degrees_of_freedom")["degrees_of_freedom"]
        assert out == {"model": 2, "residual": 5, "total": 7, "pure_error": 2, "lack_of_fit": 3}

    def test_scheffe_model_carries_the_mean_in_its_linear_terms(self) -> None:
        """The linear blending terms span the constant, so the ANOVA is corrected for the mean (Cornell)."""
        m = pd.DataFrame(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)] * 2,
            columns=["x1", "x2", "x3"],
        )
        out = evaluate_design(m, model="scheffe_quadratic", metric="degrees_of_freedom")["degrees_of_freedom"]
        assert out == {"model": 5, "residual": 8, "total": 13, "pure_error": 7, "lack_of_fit": 1}

    def test_unreplicated_design_reports_zero_pure_error(self) -> None:
        out = evaluate_design(_two_level(3), model="main_effects", metric="degrees_of_freedom")["degrees_of_freedom"]
        assert out == {"model": 3, "residual": 4, "total": 7, "pure_error": 0, "lack_of_fit": 4}


# ---------------------------------------------------------------------------
# VIF and power for Scheffe mixture models
# ---------------------------------------------------------------------------


_CENTROID = pd.DataFrame(
    [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)],
    columns=["x1", "x2", "x3"],
)


class TestScheffeModels:
    def test_vif_is_finite_and_defined_from_the_inverse(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vif = evaluate_design(_CENTROID, model="scheffe_quadratic", metric="vif")["vif"]
        x = np.asarray(dmatrix("-1 + (x1 + x2 + x3) ** 2", _CENTROID))
        c = np.diag(np.linalg.inv(x.T @ x))
        ss = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
        np.testing.assert_allclose(list(vif.values()), c * ss)
        assert max(vif.values()) < 100

    def test_vif_matches_the_classical_definition_with_an_intercept(self) -> None:
        rng = np.random.default_rng(1)
        d = pd.DataFrame(rng.uniform(-1, 1, (12, 3)), columns=list("ABC"))
        vif = evaluate_design(d, model="main_effects", metric="vif")["vif"]
        for name in "ABC":
            others = np.column_stack([np.ones(12), d.drop(columns=name)])
            fitted = others @ np.linalg.lstsq(others, d[name], rcond=None)[0]
            r2 = 1 - np.sum((d[name] - fitted) ** 2) / np.sum((d[name] - d[name].mean()) ** 2)
            assert vif[name] == pytest.approx(1 / (1 - r2))

    def test_power_skips_the_linear_blending_terms(self) -> None:
        out = evaluate_design(pd.concat([_CENTROID] * 2), model="scheffe_quadratic", metric="power", effect_size=2.0)
        assert set(out["power"]) == {"x1:x2", "x1:x3", "x2:x3"}
        assert "linear blending" in out["notes"]["power"]

    def test_process_model_names_map_to_scheffe_models_for_a_mixture(self) -> None:
        fs = [Factor(name=f"x{i}", type="mixture") for i in range(3)]
        r = generate_design(fs, design_type="mixture", model_type="special_cubic")
        assert r.metadata["model_type"] == "scheffe_special_cubic"
        by_alias = evaluate_design(r, model="special_cubic", metric="d_efficiency")
        by_name = evaluate_design(r, model="scheffe_special_cubic", metric="d_efficiency")
        default = evaluate_design(r, metric="d_efficiency")
        assert by_alias["d_efficiency"] == pytest.approx(by_name["d_efficiency"])
        assert default["d_efficiency"] == pytest.approx(by_name["d_efficiency"])
        quad = evaluate_design(r, model="interactions", metric="d_efficiency")
        assert quad["d_efficiency"] == pytest.approx(
            evaluate_design(r, model="scheffe_quadratic", metric="d_efficiency")["d_efficiency"]
        )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class TestBlocks:
    def _blocked(self) -> pd.DataFrame:
        d = _two_level(3)
        d["Block"] = np.where(d.A * d.B * d.C > 0, 1, 2)
        return d

    def test_formula_may_reference_the_block(self) -> None:
        out = evaluate_design(self._blocked(), model="A + B + C + Block", metric="degrees_of_freedom")
        assert out["degrees_of_freedom"]["model"] == 4
        assert out["degrees_of_freedom"]["residual"] == 3

    def test_ignored_blocks_are_reported(self) -> None:
        out = evaluate_design(self._blocked(), model="interactions", metric="degrees_of_freedom")
        assert out["degrees_of_freedom"]["residual"] == 1
        assert "+ Block" in out["notes"]["blocks"]

    def test_single_block_is_not_reported(self) -> None:
        d = _two_level(3).assign(Block=1)
        out = evaluate_design(d, model="interactions", metric="degrees_of_freedom")
        assert "notes" not in out


def test_power_is_for_an_anticipated_coefficient() -> None:
    """effect_size is a coefficient: a high-minus-low effect of 2 in a 2^3 is effect_size=1."""
    power = evaluate_design(_two_level(3), model="main_effects", metric="power", effect_size=1.0)["power"]["A"]
    # Coefficient 1 in a 2^3: c_jj = 1/8, so the noncentrality is 8; 4 residual df.
    expected = 1 - stats.ncf.cdf(stats.f.ppf(0.95, 1, 4), 1, 4, 8.0)
    assert power == pytest.approx(expected)
