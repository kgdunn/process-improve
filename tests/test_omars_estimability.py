"""The OMARS foldover estimability frontier, and the guards that respect it.

A foldover OMARS design is ``[H; -H; 0]``. Every second-order term is an *even*
function, so the quadratic and interaction columns of ``H`` and ``-H`` are
identical: the even block has at most ``h + 1`` distinct rows against
``1 + k(k+1)/2`` columns. The main effects live in the odd block and contribute
at most ``k`` more, so for **every** half-design

    rank(X) <= k + min(h + 1, 1 + k(k+1)/2)                     (the bound)

with equality for half-designs in general position. The bound is what matters:
below ``N = k^2 + k + 1`` runs it is strictly less than the number of
second-order parameters, so no foldover of that size can fit the full
second-order model, however cleverly its runs are chosen.

Both directions are tested here: the bound holds universally, and it is
attained. None of it needs a solver, so unlike ``tests/test_omars_ilp.py`` this
module is not gated on CBC and runs everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import Factor
from process_improve.experiments.designs_omars_ilp import (
    _d_efficiency,
    _foldover,
    _full_second_order_params,
    _half_bounds,
    _half_pool,
    _min_half_runs,
    _min_runs,
    _model_matrix,
    _model_rank,
)

# The frontier, N = k^2 + k + 1, spelled out so a change to the formula has to
# disagree with a written-down table rather than with itself.
FRONTIER_FULL_SECOND_ORDER = {3: 13, 4: 21, 5: 31, 6: 43, 7: 57}


def rank_bound(n_factors: int, n_half: int) -> int:
    """Upper bound on the rank of a foldover's full second-order model matrix."""
    return n_factors + min(n_half + 1, 1 + n_factors * (n_factors + 1) // 2)


def leading_half(n_factors: int, n_half: int) -> np.ndarray:
    """Return the first *n_half* candidate half-runs.

    Deliberately degenerate: the pool is in lexicographic order, so it opens
    with sparse runs like ``[0, 0, 0, 1]`` that do not span. Useful for showing
    that the bound is an upper bound rather than an identity.
    """
    return _half_pool(n_factors)[:n_half]


def generic_half(n_factors: int, n_half: int, *, seed: int = 0, tries: int = 200) -> np.ndarray:
    """Return a half-design in general position: one attaining :func:`rank_bound`."""
    pool = _half_pool(n_factors)
    rng = np.random.default_rng(seed)
    target = rank_bound(n_factors, n_half)
    best = pool[:n_half]
    for _ in range(tries):
        candidate = pool[rng.choice(pool.shape[0], n_half, replace=False)]
        if _model_rank(_foldover(candidate)) == target:
            return candidate
    return best


class TestFrontier:
    @pytest.mark.parametrize(("k", "expected"), FRONTIER_FULL_SECOND_ORDER.items(), ids=str)
    def test_full_second_order_frontier(self, k: int, expected: int) -> None:
        assert _min_runs(k) == expected
        assert _min_runs(k) == k * k + k + 1

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_main_quadratic_frontier_is_the_dsd_size(self, k: int) -> None:
        """Dropping the interactions drops the frontier to the DSD's 2k + 1."""
        assert _min_runs(k, "main_quadratic") == 2 * k + 1

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_frontier_is_odd(self, k: int) -> None:
        """A foldover has 2h + 1 runs, so every frontier value is odd."""
        assert _min_runs(k) % 2 == 1

    @pytest.mark.parametrize("k", [4, 5, 6, 7])
    def test_frontier_exceeds_the_parameter_count(self, k: int) -> None:
        """N > p is necessary but not sufficient; that gap was the bug.

        At three factors the two coincide; from four up the frontier pulls away.
        """
        assert _min_runs(k) > _full_second_order_params(k)


class TestRankBoundHoldsUniversally:
    """The direction the fix depends on: no half-design can beat the bound."""

    @pytest.mark.parametrize("k", [3, 4, 5])
    @pytest.mark.parametrize("offset", [-2, -1, 0, 2])
    def test_bound_is_respected_by_a_degenerate_half(self, k: int, offset: int) -> None:
        n_half = max(1, min(_min_half_runs(k) + offset, _half_pool(k).shape[0]))
        assert _model_rank(_foldover(leading_half(k, n_half))) <= rank_bound(k, n_half)

    @pytest.mark.parametrize("k", [3, 4, 5])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_bound_is_respected_by_random_halves(self, k: int, seed: int) -> None:
        pool = _half_pool(k)
        rng = np.random.default_rng(seed)
        for n_half in range(1, min(pool.shape[0], _min_half_runs(k) + 3)):
            half = pool[rng.choice(pool.shape[0], n_half, replace=False)]
            assert _model_rank(_foldover(half)) <= rank_bound(k, n_half)

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_below_the_frontier_the_bound_forbids_full_rank(self, k: int) -> None:
        """The heart of it: one run-pair short, the ceiling is under the floor.

        This is pure arithmetic on the bound, so it holds for every design at
        that size without enumerating any of them.
        """
        n_half = _min_half_runs(k) - 1
        assert rank_bound(k, n_half) < _full_second_order_params(k)
        assert 2 * n_half + 1 == _min_runs(k) - 2


class TestRankBoundIsAttained:
    """The other direction: at the frontier, a good half-design reaches it."""

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_full_rank_at_the_frontier(self, k: int) -> None:
        n_half = _min_half_runs(k)
        design = _foldover(generic_half(k, n_half))
        assert design.shape[0] == _min_runs(k)
        assert _model_rank(design) == _full_second_order_params(k)

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_no_design_is_full_rank_one_pair_below(self, k: int) -> None:
        """Search hard for a counterexample below the frontier; find none."""
        n_half = _min_half_runs(k) - 1
        params = _full_second_order_params(k)
        assert _model_rank(_foldover(generic_half(k, n_half))) < params

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_a_degenerate_half_can_fall_short_of_the_bound(self, k: int) -> None:
        """The bound is not an identity, which is why the tests separate them."""
        if k < 4:  # the three-factor leading half already spans
            pytest.skip("the three-factor leading half already spans")
        n_half = _min_half_runs(k)
        assert _model_rank(_foldover(leading_half(k, n_half))) < rank_bound(k, n_half)


class TestDEfficiencyRankGuard:
    """Regression: ``_d_efficiency`` used to report a number for a singular X.

    ``slogdet`` returns a finite log-determinant for an exactly singular integer
    Gram matrix, because round-off moves it off zero. Before the guard, a
    nineteen-run four-factor design reported a D-efficiency near 3, which reads
    as "poor" rather than "cannot be fitted at all".
    """

    def test_zero_for_a_rank_deficient_nineteen_run_four_factor_design(self) -> None:
        design = _foldover(generic_half(4, 9))
        assert design.shape[0] == 19
        assert _model_rank(design) < _full_second_order_params(4)
        assert _d_efficiency(design) == 0.0

    def test_the_gram_matrix_really_is_singular(self) -> None:
        """Independent confirmation, not routed through the guard."""
        model_matrix = _model_matrix(_foldover(generic_half(4, 9)))
        gram = model_matrix.T @ model_matrix
        assert np.linalg.matrix_rank(gram) < gram.shape[0]
        assert min(np.linalg.svd(gram, compute_uv=False)) < 1e-6

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_positive_at_the_frontier(self, k: int) -> None:
        assert _d_efficiency(_foldover(generic_half(k, _min_half_runs(k)))) > 0.0

    @pytest.mark.parametrize(("k", "n_half"), [(4, 5), (4, 9), (5, 10), (5, 14)])
    def test_zero_everywhere_below_the_frontier(self, k: int, n_half: int) -> None:
        assert n_half < _min_half_runs(k)
        assert _d_efficiency(_foldover(generic_half(k, n_half))) == 0.0

    @pytest.mark.parametrize("sign", [0.0, -1.0])
    def test_a_non_positive_slogdet_sign_also_scores_zero(self, monkeypatch, sign: float) -> None:
        """The second guard, behind the rank one.

        ``matrix_rank`` decides rank against an SVD tolerance, so a Gram matrix
        can clear the rank guard and still be too ill-conditioned for
        ``slogdet`` to return a positive sign. Reaching that state with a real
        design would mean finding one on the knife edge of the tolerance, so
        the sign is forced instead.
        """
        design = _foldover(generic_half(4, _min_half_runs(4)))
        assert _d_efficiency(design) > 0.0  # the guard is what changes the answer

        monkeypatch.setattr(np.linalg, "slogdet", lambda _: (sign, 12.0))
        assert _d_efficiency(design) == 0.0


class TestHalfBounds:
    """The sizing window starts at whichever floor binds harder."""

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_auto_window_starts_at_the_frontier(self, k: int) -> None:
        """For the full second-order model, estimability is the binding floor."""
        min_half = _min_half_runs(k)
        params = _full_second_order_params(k)
        low, high = _half_bounds(None, params, half_pool_size=10_000, min_half=min_half)
        assert low == min_half
        assert 2 * low + 1 == _min_runs(k)
        assert high >= low

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_error_df_floor_binds_for_main_quadratic(self, k: int) -> None:
        """Dropping the interactions makes N > p the binding floor instead."""
        min_half = _min_half_runs(k, "main_quadratic")
        params = 1 + 2 * k
        low, _high = _half_bounds(None, params, half_pool_size=10_000, min_half=min_half)
        assert 2 * low + 1 > params
        assert low >= min_half

    def test_requested_window_is_lifted_to_the_frontier(self) -> None:
        """A caller asking below the frontier gets the frontier, not their floor."""
        low, high = _half_bounds((9, 41), _full_second_order_params(4), 10_000, _min_half_runs(4))
        assert 2 * low + 1 == _min_runs(4) == 21
        assert 2 * high + 1 <= 41

    def test_window_never_inverts_against_a_small_pool(self) -> None:
        low, high = _half_bounds(None, _full_second_order_params(4), half_pool_size=3, min_half=10)
        assert high >= low


class TestGenerateOmarsRefusesSubFrontierSizes:
    """The validation runs before any solver call, so this needs no CBC."""

    @pytest.mark.parametrize(("k", "n_runs"), [(4, 19), (4, 17), (5, 27), (5, 29), (6, 41)])
    def test_sub_frontier_n_runs_is_refused(self, k: int, n_runs: int) -> None:
        from process_improve.experiments.designs_omars_ilp import generate_omars

        factors = [Factor(name=chr(65 + i), low=-1, high=1) for i in range(k)]
        with pytest.raises(ValueError, match="cannot estimate the full_second_order model"):
            generate_omars(factors, n_runs=n_runs)

    def test_the_message_names_the_frontier_and_the_way_out(self) -> None:
        from process_improve.experiments.designs_omars_ilp import generate_omars

        factors = [Factor(name=chr(65 + i), low=-1, high=1) for i in range(5)]
        with pytest.raises(ValueError, match="cannot estimate") as excinfo:
            generate_omars(factors, n_runs=29)
        message = str(excinfo.value)
        assert "n_runs >= 31" in message
        assert "main_quadratic" in message

    def test_main_quadratic_accepts_what_the_full_model_refuses(self) -> None:
        """17 runs clears the parameter count (15) but not the frontier (21).

        Sizes that fail both gates report the error-df one first, so this picks
        a size where estimability is the only thing standing in the way. The
        same size is comfortably above the main-quadratic frontier of 9.
        """
        from process_improve.experiments.designs_omars_ilp import generate_omars

        assert _min_runs(4, "main_quadratic") <= 17 < _min_runs(4)
        assert _full_second_order_params(4) < 17
        factors = [Factor(name=chr(65 + i), low=-1, high=1) for i in range(4)]
        with pytest.raises(ValueError, match="cannot estimate"):
            generate_omars(factors, n_runs=17)
