"""Regression tests for the 2026-08 repo-wide correctness audit: experiments/DOE.

Each test pins a specific defect found and fixed in the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyDOE3 import fullfact

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_optimal import _n_model_parameters, dispatch_d_optimal
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Factor
from process_improve.experiments.optimal import (
    index_to_replace_in_design_row,
    optimization_function,
    point_exchange,
)
from process_improve.experiments.structures import c, gather

pytest.importorskip("pyDOE3")


def _factors(names: list[str]) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


class TestClearEffects:
    """Wu & Hamada: clear = aliased only with third-order-or-higher effects."""

    @staticmethod
    def _clear(names: list[str], generators: list[str]) -> dict:
        result = generate_design(
            _factors(names),
            design_type="fractional_factorial",
            generators=generators,
            n_center_points=0,
        )
        evaluation = evaluate_design(result, metric="clear_effects")
        return evaluation["clear_effects"]

    def test_resolution_iii_has_no_clear_main_effects(self) -> None:
        """2^(3-1) with C=AB: every main effect is aliased with a 2FI.

        The pre-fix rule ("aliases of higher order than the effect") declared
        all three main effects clear.
        """
        clear = self._clear(["A", "B", "C"], ["C=AB"])
        assert clear["main_effects"] == []
        assert clear["two_factor_interactions"] == []

    def test_resolution_iv_mains_clear_2fis_not(self) -> None:
        """2^(4-1) with D=ABC: mains aliased with 3FIs (clear); 2FIs aliased
        with 2FIs (not clear).
        """
        clear = self._clear(["A", "B", "C", "D"], ["D=ABC"])
        assert set(clear["main_effects"]) == {"A", "B", "C", "D"}
        assert clear["two_factor_interactions"] == []

    def test_resolution_v_mains_and_2fis_clear(self) -> None:
        """2^(5-1) with E=ABCD: mains and all 2FIs are clear."""
        clear = self._clear(["A", "B", "C", "D", "E"], ["E=ABCD"])
        assert set(clear["main_effects"]) == {"A", "B", "C", "D", "E"}
        assert len(clear["two_factor_interactions"]) == 10


class TestFractionalFactorialGenerators:
    """Explicit generators: correct factor-column mapping, any factor names."""

    @staticmethod
    def _coded_frame(names: list[str], generators: list[str]) -> pd.DataFrame:
        result = generate_design(
            _factors(names),
            design_type="fractional_factorial",
            generators=generators,
            n_center_points=0,
        )
        return pd.DataFrame({name: np.asarray(result.design[name], dtype=float) for name in names})

    def test_generator_on_a_middle_factor(self) -> None:
        """B=AC previously swapped the B and C columns silently."""
        frame = self._coded_frame(["A", "B", "C"], ["B=AC"])
        np.testing.assert_allclose(frame["B"], frame["A"] * frame["C"])
        # And the base factors must form the full 2^2 factorial.
        assert len(set(zip(frame["A"], frame["C"], strict=True))) == 4

    def test_last_factor_generator_still_correct(self) -> None:
        frame = self._coded_frame(["A", "B", "C", "D"], ["D=ABC"])
        np.testing.assert_allclose(frame["D"], frame["A"] * frame["B"] * frame["C"])

    def test_multi_character_factor_names(self) -> None:
        """Real factor names were previously lower-cased into pyDOE3's
        single-letter notation and misread as products of letters.
        """
        frame = self._coded_frame(["Temp", "Press", "Flow"], ["Flow=TempPress"])
        np.testing.assert_allclose(frame["Flow"], frame["Temp"] * frame["Press"])

    def test_negative_generator(self) -> None:
        frame = self._coded_frame(["A", "B", "C"], ["C=-AB"])
        np.testing.assert_allclose(frame["C"], -frame["A"] * frame["B"])

    def test_unknown_factor_in_generator_raises(self) -> None:
        with pytest.raises(ValueError, match="not a factor name"):
            self._coded_frame(["A", "B", "C"], ["C=AZ"])


class TestColumnCoding:
    def test_explicit_zero_center_is_respected(self) -> None:
        """to_coded(center=0) previously fell back to pi_center (0 is falsy)."""
        col = c([4.0, 6.0], name="x", lo=4, hi=6)  # pi_center = 5
        coded = col.to_coded(center=0.0, range=(-10, 10))
        np.testing.assert_allclose(np.asarray(coded.values, dtype=float), [0.4, 0.6])

    def test_zero_width_range_raises(self) -> None:
        """Previously a zero-width range silently produced inf/NaN values."""
        col = c([1.0, 2.0], name="x", lo=3, hi=3)
        with pytest.raises(ValueError, match="zero width"):
            col.to_coded()


class TestGather:
    def test_positional_columns_are_kept(self) -> None:
        """gather(A, B, y=y) previously dropped A and B silently."""
        a = c([1.0, 2.0, 3.0], name="A")
        b = c([4.0, 5.0, 6.0], name="B")
        y = c([7.0, 8.0, 9.0], name="y")
        expt = gather(a, b, y=y, title="positional")
        assert set(expt.columns) == {"A", "B", "y"}

    def test_nameless_positional_raises(self) -> None:
        nameless = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="has no name"):
            gather(nameless, y=c([1.0, 2.0], name="y"))


class TestDOptimal:
    def test_replicates_change_the_score(self) -> None:
        """The scorer must NOT de-duplicate: n copies of a point carry more
        information than one copy.
        """
        base = pd.DataFrame([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        replicated = pd.DataFrame([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        assert optimization_function(replicated) < optimization_function(base)  # lower = better

    def test_rank_deficient_designs_are_rejected_not_preferred(self) -> None:
        """A design the model cannot be fitted to must score ``+inf``.

        The scorer used to invert ``X'X``. ``np.linalg.inv`` raises only on an
        exactly-zero LU pivot, which these rank-deficient +-1 designs slip past;
        the log-determinant of the resulting numerical noise came back as a
        large NEGATIVE number, so the minimising search treated an inestimable
        design as the best one available and actively selected it.
        """
        constant_factor = pd.DataFrame([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])
        aliased = pd.DataFrame([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]])
        too_few_runs = pd.DataFrame([[1.0, 1.0], [1.0, -1.0]])
        for degenerate in (constant_factor, aliased, too_few_runs):
            model_matrix = np.column_stack([np.ones(degenerate.shape[0]), degenerate])
            assert np.linalg.matrix_rank(model_matrix) < model_matrix.shape[1]
            assert optimization_function(degenerate) == float(np.inf)

    def test_swap_onto_row_label_zero_is_not_discarded(self) -> None:
        """A best-swap row with index LABEL 0 is falsy; the caller previously
        skipped the improving swap entirely.
        """
        design = pd.DataFrame([[-1.0, 0.05], [1.0, -1.0], [-1.0, 1.0]], index=[0, 1, 2])
        candidate = pd.DataFrame([[-1.0, -1.0]], index=[99])
        current = optimization_function(design)
        chosen = index_to_replace_in_design_row(design, candidate, current, optimization_function)
        assert chosen == 0
        assert chosen is not None

    def test_point_exchange_returns_the_requested_number_of_points(self) -> None:
        """The design size is a constraint, not something the search may trade away.

        The design used to be seeded with one row per factor and grown towards
        `number_points` only by additions that improved D-optimality, so when no
        addition improved it the caller silently received a short design. A
        design with fewer runs than model parameters cannot be fitted at all.
        """
        candidates = pd.DataFrame(fullfact([3] * 3) - 1.0, columns=["X1", "X2", "X3"])
        for number_points in (4, 5, 8, 12):
            sizes = {
                point_exchange(candidates, number_points=number_points, random_state=seed)[0].shape[0]
                for seed in range(25)
            }
            assert sizes == {number_points}, f"requested {number_points}, got sizes {sorted(sizes)}"

    def test_d_optimal_below_minimum_budget_is_estimable(self) -> None:
        """A budget below the model size yields a design the model can be fitted to."""
        from process_improve.experiments import designs_optimal as _designs_optimal

        original = _designs_optimal._PYOPTEX_AVAILABLE
        _designs_optimal._PYOPTEX_AVAILABLE = False
        try:
            factors = [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(3)]
            for _ in range(25):
                design, _meta = dispatch_d_optimal(factors, budget=2)
                assert design.shape[0] >= 4, "intercept plus 3 main effects needs at least 4 runs"
                # Estimability is the point: the model matrix must have full rank.
                model_matrix = np.column_stack([np.ones(design.shape[0]), design])
                assert np.linalg.matrix_rank(model_matrix) == 4
        finally:
            _designs_optimal._PYOPTEX_AVAILABLE = original

    def test_point_exchange_minimum_size_counts_the_intercept(self) -> None:
        """One run per factor is one short: the model this scores has an intercept.

        The bound used to be ``x.shape[1]``, so a request for exactly that many
        runs was accepted and then spent 1000 attempts failing to find a
        non-singular start, reported as a candidate-set problem.
        """
        candidates = pd.DataFrame(fullfact([3] * 3) - 1.0, columns=["X1", "X2", "X3"])
        with pytest.raises(ValueError, match="at least 4"):
            point_exchange(candidates, number_points=3, random_state=0)

    def test_point_exchange_will_not_silently_shrink_to_the_candidate_count(self) -> None:
        """More runs than unique candidates is an error, not a quiet clamp."""
        candidates = pd.DataFrame([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="at most 4"):
            point_exchange(candidates, number_points=5, random_state=0)

    @pytest.mark.parametrize(
        ("model_type", "n_parameters"),
        [("main_effects", 4), ("interactions", 7), ("quadratic", 10)],
    )
    def test_budget_floor_matches_the_declared_model(
        self, model_type: str, n_parameters: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An under-budget request is raised to the DECLARED model's size, not to one run per factor.

        The floor used to be ``n_factors + 1`` regardless of `model_type`, which
        is the size of a main-effects model only; an interactions or quadratic
        model was handed a design far too small to fit it.
        """
        from process_improve.experiments import designs_optimal as _designs_optimal

        monkeypatch.setattr(_designs_optimal, "_PYOPTEX_AVAILABLE", False)
        factors = [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(3)]
        assert _n_model_parameters(factors, model_type) == n_parameters
        with caplog.at_level("WARNING", logger="process_improve.experiments.designs_optimal"):
            design, _meta = dispatch_d_optimal(factors, budget=2, model_type=model_type)
        assert design.shape[0] == n_parameters
        assert "raising the budget" in caplog.text  # the clamp is announced, not silent

    def test_default_budget_supports_the_declared_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``2 * n_factors + 1`` default is itself below an interactions model from k=4 up."""
        from process_improve.experiments import designs_optimal as _designs_optimal

        monkeypatch.setattr(_designs_optimal, "_PYOPTEX_AVAILABLE", False)
        factors = [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(4)]
        design, _meta = dispatch_d_optimal(factors, model_type="interactions")  # default budget = 9
        assert design.shape[0] == _n_model_parameters(factors, "interactions") == 11

    def test_n_model_parameters_counts_categorical_levels(self) -> None:
        """A categorical factor with L levels is L - 1 columns, and has no square."""
        factors = [
            Factor(name="X1", low=0, high=10),
            Factor(name="X2", low=0, high=10),
            Factor(name="C", type="categorical", levels=["A", "B", "C"]),
        ]
        # 1 + (1 + 1 + 2) main effects = 5
        assert _n_model_parameters(factors, "main_effects") == 5
        # + interactions 1*1 + 1*2 + 1*2 = 5, so 10
        assert _n_model_parameters(factors, "interactions") == 10
        # + a square for each of the two continuous factors only, so 12
        assert _n_model_parameters(factors, "quadratic") == 12

    def test_point_exchange_rejects_a_duplicated_index(self) -> None:
        """A repeated index label made `.loc` return every row carrying it.

        These six candidate rows are all distinct by value, so de-duplication
        leaves them alone; the repeated label 0 made a request for 4 runs come
        back with all 6.
        """
        candidates = pd.DataFrame(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
            index=[0, 0, 1, 2, 3, 4],
        )
        with pytest.raises(ValueError, match="unique index"):
            point_exchange(candidates, number_points=4, random_state=0)

    def test_point_exchange_reports_a_candidate_set_it_cannot_start_from(self) -> None:
        """Collinear candidate columns make every subset singular, whatever the size."""
        repeated = [-1.0, -1.0, 1.0, 1.0, 0.0, 0.0]
        candidates = pd.DataFrame({"a": repeated, "b": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], "c": repeated})
        with pytest.raises(ValueError, match="non-singular starting design"):
            point_exchange(candidates, number_points=4, random_state=0)

    def test_point_exchange_is_reproducible_with_random_state(self) -> None:
        rng = np.random.default_rng(0)
        candidates = pd.DataFrame(rng.choice([-1.0, 0.0, 1.0], size=(40, 3)))
        d1, v1 = point_exchange(candidates, number_points=6, random_state=7)
        d2, v2 = point_exchange(candidates, number_points=6, random_state=7)
        assert v1 == v2
        pd.testing.assert_frame_equal(d1, d2)


class TestLackOfFit:
    def test_generated_design_with_replicates_is_testable(self) -> None:
        """RunOrder (unique per row) previously made every replicate group a
        singleton, so the test always reported 'No replicated points'.
        """
        rng = np.random.default_rng(1)
        base = pd.DataFrame(
            {
                "A": [-1, 1, -1, 1] * 2,
                "B": [-1, -1, 1, 1] * 2,
            }
        )
        base["RunOrder"] = np.arange(len(base)) + 1
        base["y"] = 5 + 2 * base["A"] - base["B"] + 0.5 * rng.standard_normal(len(base))
        result = analyze_experiment(
            base,
            response_column="y",
            model="main_effects",
            analysis_type="lack_of_fit",
        )
        lof = result["lack_of_fit"]
        assert "error" not in lof
        assert lof["df_pure_error"] == 4  # four pairs of replicated settings
        assert np.isfinite(lof["f_statistic"])
