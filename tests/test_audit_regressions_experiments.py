"""Regression tests for the 2026-08 repo-wide correctness audit: experiments/DOE.

Each test pins a specific defect found and fixed in the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.designs import generate_design
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
        base = pd.DataFrame([[1.0, 1.0], [1.0, -1.0]])
        replicated = pd.DataFrame([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        assert optimization_function(replicated) < optimization_function(base)  # lower = better

    def test_swap_onto_row_label_zero_is_not_discarded(self) -> None:
        """A best-swap row with index LABEL 0 is falsy; the caller previously
        skipped the improving swap entirely.
        """
        design = pd.DataFrame([[1.0, 0.05], [1.0, -1.0]], index=[0, 1])
        candidate = pd.DataFrame([[1.0, 1.0]], index=[99])
        current = optimization_function(design)
        chosen = index_to_replace_in_design_row(design, candidate, current, optimization_function)
        assert chosen == 0
        assert chosen is not None

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
