"""Categorical factors in effect (sum-to-zero) coding, on the design side.

``evaluate_design`` and the built-in optimal-design exchange code a categorical factor
the same way. The coding-dependent metrics (A, E, K, VIF, power) are checked against an
explicit numpy model matrix, and the coding-invariant ones (prediction variance, G,
FDS, degrees of freedom) are pinned to the values measured before the change.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design, generate_design


def _fixed_design() -> pd.DataFrame:
    """Return a 2 x 2 x 3 factorial in A, B and a three-level categorical C, plus three unbalancing runs."""
    rows = [(a, b, c) for a, b, c in itertools.product([-1.0, 1.0], [-1.0, 1.0], ["lo", "mid", "hi"])]
    rows += [(1.0, -1.0, "lo"), (-1.0, 1.0, "hi"), (0.0, 0.0, "mid")]
    return pd.DataFrame(rows, columns=["A", "B", "C"])


def _effect_columns(labels: list) -> np.ndarray:
    """Effect coding of one categorical column, written out independently of the library.

    Levels in sorted order. Two levels: the first is -1 and the second +1. Three or more:
    level ``j`` is +1 in column ``j`` and the last level is -1 in every column.
    """
    levels = sorted(set(labels))
    index = np.array([levels.index(v) for v in labels])
    if len(levels) == 2:
        return np.where(index == 0, -1.0, 1.0)[:, None]
    return np.vstack([np.eye(len(levels) - 1), -np.ones((1, len(levels) - 1))])[index]


def _effect_model_matrix(design: pd.DataFrame) -> np.ndarray:
    """Intercept, A, B, C and the two-factor interactions, with C effect-coded."""
    a = design["A"].to_numpy(dtype=float)
    b = design["B"].to_numpy(dtype=float)
    c = _effect_columns(list(design["C"]))
    return np.column_stack([np.ones(len(a)), a, b, c, a * b, a[:, None] * c, b[:, None] * c])


def _factors(levels: list[str]) -> list[Factor]:
    return [
        Factor(name="A", low=0, high=1),
        Factor(name="B", low=0, high=1),
        Factor(name="C", type="categorical", levels=levels),
    ]


# ---------------------------------------------------------------------------
# Metrics that do not depend on the coding keep their values
# ---------------------------------------------------------------------------


class TestCodingInvariantMetrics:
    def test_prediction_variance_metrics_and_degrees_of_freedom_are_pinned(self) -> None:
        out = evaluate_design(
            _fixed_design(),
            model="interactions",
            metric=["average_prediction_variance", "g_efficiency", "fds", "degrees_of_freedom"],
            n_samples=5000,
        )
        assert out["average_prediction_variance"] == pytest.approx(0.3692822142386848, rel=1e-9)
        assert out["g_efficiency"] == pytest.approx(81.93384223918574, rel=1e-9)
        assert out["fds"]["max_prediction_variance"] == pytest.approx(0.813664596273292, rel=1e-9)
        assert out["degrees_of_freedom"] == {
            "model": 9,
            "residual": 5,
            "total": 14,
            "pure_error": 2,
            "lack_of_fit": 3,
        }

    def test_prediction_variance_d_ranking_and_dof_agree_between_codings(self) -> None:
        metrics = ["average_prediction_variance", "g_efficiency", "fds", "degrees_of_freedom"]
        designs = [_fixed_design(), _fixed_design().iloc[:-3], _fixed_design().iloc[2:]]
        effect = [evaluate_design(d, metric=[*metrics, "d_efficiency"], n_samples=3000) for d in designs]
        treatment = [
            evaluate_design(d, metric=[*metrics, "d_efficiency"], n_samples=3000, categorical_coding="treatment")
            for d in designs
        ]
        for e, t in zip(effect, treatment, strict=True):
            assert e["average_prediction_variance"] == pytest.approx(t["average_prediction_variance"], rel=1e-9)
            assert e["g_efficiency"] == pytest.approx(t["g_efficiency"], rel=1e-9)
            assert e["fds"]["quantiles"] == pytest.approx(t["fds"]["quantiles"], rel=1e-9)
            assert e["degrees_of_freedom"] == t["degrees_of_freedom"]
        # The D-efficiency value depends on the coding, but only by a factor common to every design.
        ratios = [e["d_efficiency"] / t["d_efficiency"] for e, t in zip(effect, treatment, strict=True)]
        assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-9)


# ---------------------------------------------------------------------------
# Effect coding: the coding-dependent metrics, checked against an explicit matrix
# ---------------------------------------------------------------------------


class TestEffectCoding:
    def test_default_is_effect_coding_with_sum_term_names(self) -> None:
        out = evaluate_design(_fixed_design(), model="main_effects", metric="vif")
        assert set(out["vif"]) == {"A", "B", "C[S.hi]", "C[S.lo]"}

    def test_a_e_and_vif_match_an_explicit_effect_coded_matrix(self) -> None:
        design = _fixed_design()
        out = evaluate_design(design, model="interactions", metric=["a_optimality", "e_optimality", "vif"])
        x = _effect_model_matrix(design)
        info = x.T @ x
        assert out["a_optimality"] == pytest.approx(np.trace(np.linalg.inv(info)), rel=1e-9)
        assert out["e_optimality"] == pytest.approx(np.linalg.eigvalsh(info).min(), rel=1e-9)
        # VIF of A: 1 / (1 - R^2) of A regressed on the other non-intercept columns.
        others = np.delete(x, 1, axis=1)
        a = x[:, 1]
        resid = a - others @ np.linalg.lstsq(others, a, rcond=None)[0]
        vif_a = np.sum((a - a.mean()) ** 2) / np.sum(resid**2)
        assert out["vif"]["A"] == pytest.approx(vif_a, rel=1e-9)

    def test_two_level_categorical_matches_the_same_factor_coded_continuous(self) -> None:
        base = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=list("ABD"))
        base = pd.concat([base, base.iloc[[0, 3, 6]]], ignore_index=True)
        labelled = base.assign(D=np.where(base["D"] > 0, "y", "x"))
        metrics = ["a_optimality", "e_optimality", "vif", "condition_number"]
        cont = evaluate_design(base, model="interactions", metric=metrics)
        cat = evaluate_design(labelled, model="interactions", metric=metrics)
        assert cat["a_optimality"] == pytest.approx(cont["a_optimality"], rel=1e-9)
        assert cat["e_optimality"] == pytest.approx(cont["e_optimality"], rel=1e-9)
        assert cat["condition_number"] == pytest.approx(cont["condition_number"], rel=1e-9)
        assert cat["vif"]["D[S.y]"] == pytest.approx(cont["vif"]["D"], rel=1e-9)

    def test_alias_matrix_terms_and_values_use_the_same_coding(self) -> None:
        design = _fixed_design().query("C != 'mid'").reset_index(drop=True)
        out = evaluate_design(design, model="main_effects", metric="alias_matrix")["alias_matrix"]
        assert out["model_terms"] == ["Intercept", "C[S.lo]", "A", "B"]
        assert out["alias_terms"] == ["A:B", "A:C[S.lo]", "B:C[S.lo]"]
        c = np.where(design["C"] == "hi", -1.0, 1.0)
        x1 = np.column_stack([np.ones(len(design)), c, design["A"], design["B"]])
        x2 = np.column_stack([design["A"] * design["B"], design["A"] * c, design["B"] * c])
        np.testing.assert_allclose(out["matrix"], np.linalg.solve(x1.T @ x1, x1.T @ x2), atol=1e-12)

    def test_a_formula_contrast_overrides_the_default(self) -> None:
        design = _fixed_design().rename(columns={"C": "cat"})
        out = evaluate_design(design, model="A + B + C(cat, Treatment)", metric="vif")
        assert set(out["vif"]) == {"A", "B", "C(cat, Treatment)[T.lo]", "C(cat, Treatment)[T.mid]"}
        bare = evaluate_design(design, model="A + B + cat", metric="vif")
        assert set(bare["vif"]) == {"A", "B", "cat[S.hi]", "cat[S.lo]"}

    def test_blocks_named_in_the_formula_are_sum_coded(self) -> None:
        design = _fixed_design().assign(Block=[1, 2] * 7 + [1])
        out = evaluate_design(design, model="A + B + C + Block", metric=["vif", "a_optimality"])
        assert "Block[S.1]" in out["vif"]
        block = np.where(design["Block"] == 1, 1.0, -1.0)
        x = np.column_stack([_effect_model_matrix(design)[:, :5], block])
        assert out["a_optimality"] == pytest.approx(np.trace(np.linalg.inv(x.T @ x)), rel=1e-9)

    def test_an_unknown_coding_is_refused(self) -> None:
        with pytest.raises(ValueError, match="categorical_coding"):
            evaluate_design(_fixed_design(), categorical_coding="helmert")

    def test_evaluate_all_and_the_tool_pass_the_coding_through(self) -> None:
        from process_improve.experiments import evaluate_all
        from process_improve.experiments._tools.evaluate_design import EvaluateDesignInput, evaluate_design_tool

        design = _fixed_design()
        treat = evaluate_all(design, n_samples=2000, categorical_coding="treatment")
        assert treat["a_optimality"] == pytest.approx(3.3692546583850933, rel=1e-9)
        rows = design.to_dict(orient="records")
        tool = evaluate_design_tool(EvaluateDesignInput(design_matrix=rows, metric="a_optimality"))
        assert tool["a_optimality"] == pytest.approx(
            np.trace(np.linalg.inv(_effect_model_matrix(design).T @ _effect_model_matrix(design)))
        )
        tool_t = evaluate_design_tool(
            EvaluateDesignInput(design_matrix=rows, metric="a_optimality", categorical_coding="treatment")
        )
        assert tool_t["a_optimality"] == pytest.approx(3.3692546583850933, rel=1e-9)


# ---------------------------------------------------------------------------
# Treatment coding: the values evaluate_design gave before effect coding
# ---------------------------------------------------------------------------


class TestTreatmentCoding:
    def test_treatment_coding_values_are_pinned(self) -> None:
        out = evaluate_design(
            _fixed_design(),
            model="interactions",
            metric=["d_efficiency", "a_optimality", "e_optimality", "condition_number", "vif", "power"],
            effect_size=1.0,
            categorical_coding="treatment",
        )
        assert out["d_efficiency"] == pytest.approx(34.35028327680051, rel=1e-9)
        assert out["a_optimality"] == pytest.approx(3.3692546583850933, rel=1e-9)
        assert out["e_optimality"] == pytest.approx(1.0717967697244946, rel=1e-9)
        assert out["condition_number"] == pytest.approx(4.42277022639572, rel=1e-9)
        assert out["vif"]["C[T.lo]"] == pytest.approx(1.4285714285714288, rel=1e-9)
        assert out["vif"]["A:C[T.mid]"] == pytest.approx(1.863354037267081, rel=1e-9)
        assert out["power"]["C[T.lo]"] == pytest.approx(0.23836843380200268, rel=1e-9)
        assert out["power"]["A"] == pytest.approx(0.4143624547309154, rel=1e-9)


# ---------------------------------------------------------------------------
# Quality of the built-in exchange's designs, judged in effect coding
# ---------------------------------------------------------------------------

#: Five-seed means (seeds 0-4, 16 runs, interactions model) measured on the treatment-coded
#: exchange, each design scored in effect coding. Effect coding changes which designs the
#: A-, E- and K-criteria prefer, so the new designs must be no worse on these numbers.
#: "lo mid hi" is declared out of sorted order, so the reference level is not the last declared.
_EXCHANGE_BASELINE = {
    ("lo mid hi", "a_optimal"): 1.0625,
    ("lo mid hi", "e_optimal"): 5.071796769724491,
    ("lo mid hi", "k_optimal"): 3.797273352909202,
    ("lo mid hi", "d_optimal"): 23.958672110309426,
    ("x y", "a_optimal"): 0.4375,
    ("x y", "e_optimal"): 16.0,
    ("x y", "k_optimal"): 1.0,
    ("x y", "d_optimal"): 19.408121055678468,
}

#: The prediction-variance criteria do not depend on the coding (same seeds and size).
_INVARIANT_BASELINE = {
    ("lo mid hi", "i_optimal"): ("trace_criterion", 0.345413968034522),
    ("lo mid hi", "g_optimal"): ("max_prediction_variance", 0.8045112781954888),
    ("x y", "i_optimal"): ("trace_criterion", 0.2153264006629351),
    ("x y", "g_optimal"): ("max_prediction_variance", 0.4375),
}

_SEEDS = range(5)


def _exchange_designs(levels: str, criterion: str) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            generate_design(_factors(levels.split()), design_type=criterion, budget=16, random_state=seed)
            for seed in _SEEDS
        ]


def _effect_score(design: pd.DataFrame, criterion: str) -> float:
    x = _effect_model_matrix(design)
    info = x.T @ x
    eig = np.linalg.eigvalsh(info)
    return {
        "a_optimal": float(np.trace(np.linalg.inv(info))),
        "e_optimal": float(eig.min()),
        "k_optimal": float(eig.max() / eig.min()),
        "d_optimal": float(np.linalg.slogdet(info)[1]),
    }[criterion]


@pytest.mark.parametrize(("levels", "criterion"), list(_EXCHANGE_BASELINE))
def test_exchange_designs_are_no_worse_in_effect_coding(levels: str, criterion: str) -> None:
    designs = _exchange_designs(levels, criterion)
    mean = float(np.mean([_effect_score(r.design[["A", "B", "C"]], criterion) for r in designs]))
    baseline = _EXCHANGE_BASELINE[(levels, criterion)]
    if criterion in ("a_optimal", "k_optimal"):
        assert mean <= baseline * (1 + 1e-9)
    else:
        assert mean >= baseline * (1 - 1e-9)


@pytest.mark.parametrize(("levels", "criterion"), list(_INVARIANT_BASELINE))
def test_exchange_prediction_variance_criteria_are_no_worse(levels: str, criterion: str) -> None:
    key, baseline = _INVARIANT_BASELINE[(levels, criterion)]
    mean = float(np.mean([r.metadata[key] for r in _exchange_designs(levels, criterion)]))
    assert mean <= baseline * (1 + 1e-9)


# ---------------------------------------------------------------------------
# The design builders report the criterion evaluate_design computes
# ---------------------------------------------------------------------------


def test_exchange_trace_criterion_equals_evaluate_design() -> None:
    """The levels are declared out of sorted order, so the shared reference level is checked too."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate_design(_factors(["lo", "mid", "hi"]), design_type="a_optimal", budget=14, random_state=1)
    a = evaluate_design(result, metric="a_optimality")["a_optimality"]
    assert result.metadata["trace_criterion"] == pytest.approx(a, rel=1e-9)
    assert a == pytest.approx(_effect_score(result.design[["A", "B", "C"]], "a_optimal"), rel=1e-9)


@pytest.mark.parametrize("levels", [["lo", "mid", "hi"], ["x", "y"]])
def test_pyoptex_a_optimal_trace_equals_the_effect_coded_recomputation(levels: list[str]) -> None:
    pytest.importorskip("pyoptex")
    factors = [*_factors(levels)[:2], Factor(name="cat", type="categorical", levels=levels)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate_design(factors, design_type="a_optimal", budget=14, random_state=1, backend="pyoptex")
    trace = -result.metadata["metric_value"]  # pyoptex reports -trace, in its own effect coding
    assert result.metadata["trace_criterion"] == pytest.approx(trace, rel=1e-9)
    assert evaluate_design(result, metric="a_optimality")["a_optimality"] == pytest.approx(trace, rel=1e-9)
