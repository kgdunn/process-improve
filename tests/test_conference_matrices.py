"""Tests for the conference-matrix constructions behind the definitive screening design.

A DSD is only a DSD if the matrix it folds is a genuine conference matrix:
``C.T @ C == (m - 1) * I``.  If that fails, main effects come out correlated
with each other and the design silently stops doing the one thing it exists to
do.  These tests check the constructions, the finite-field arithmetic they rest
on, and the guard that refuses anything that does not satisfy the identity.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from process_improve.experiments.designs_response_surface import (
    _conference_matrix,
    _gf_multiplication_table,
    _irreducible_polynomial,
    _is_constructible_conference_order,
    _polynomial_remainder,
    _prime_power_factorization,
    _quadratic_character,
    _smallest_constructible_conference_order,
    _validate_conference_matrix,
    dsd_conference_order,
    dsd_run_count,
)

# Odd prime powers that matter for DSDs at practical factor counts.
_PRIME_POWERS = [3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 27, 29, 31, 37, 41, 43, 47, 49, 81, 121, 125]


class TestPrimePowerFactorization:
    """Recognising ``q = p ** n``."""

    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            (3, (3, 1)),
            (9, (3, 2)),
            (25, (5, 2)),
            (27, (3, 3)),
            (49, (7, 2)),
            (81, (3, 4)),
            (125, (5, 3)),
            (169, (13, 2)),
        ],
    )
    def test_prime_powers_recognised(self, q: int, expected: tuple[int, int]) -> None:
        assert _prime_power_factorization(q) == expected

    @pytest.mark.parametrize("q", [1, 0, -4, 15, 21, 33, 35, 45, 51, 55, 57])
    def test_non_prime_powers_rejected(self, q: int) -> None:
        """15, 21, 33, ... are the orders that force a table or a step up."""
        assert _prime_power_factorization(q) is None

    def test_agrees_with_brute_force(self) -> None:
        """Cross-check against an independent definition over a wide range."""
        primes = [p for p in range(2, 500) if all(p % d for d in range(2, int(p**0.5) + 1))]
        for q in range(2, 500):
            brute = None
            for p in primes:
                n, remaining = 0, q
                while remaining % p == 0:
                    remaining //= p
                    n += 1
                if remaining == 1 and n > 0:
                    brute = (p, n)
                    break
            assert _prime_power_factorization(q) == brute, f"disagreement at q={q}"


class TestFiniteFieldArithmetic:
    """GF(p**n) built from the first monic irreducible polynomial of degree n."""

    @pytest.mark.parametrize("q", _PRIME_POWERS)
    def test_multiplication_is_a_group_on_nonzero_elements(self, q: int) -> None:
        """Every non-zero row of the table permutes the non-zero elements.

        This is the property that fails if the modulus polynomial is reducible:
        the ring would then have zero divisors and rows would repeat values.
        """
        p, n = _prime_power_factorization(q)
        table = _gf_multiplication_table(p, n)
        assert table.shape == (q, q)
        assert np.all(table[0, :] == 0)
        assert np.all(table[:, 0] == 0)
        for row in range(1, q):
            assert sorted(table[row, 1:]) == list(range(1, q))

    @pytest.mark.parametrize("q", _PRIME_POWERS)
    def test_multiplication_is_commutative_and_associative(self, q: int) -> None:
        p, n = _prime_power_factorization(q)
        table = _gf_multiplication_table(p, n)
        assert np.array_equal(table, table.T)
        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            a, b, c = rng.integers(0, q, size=3)
            assert table[table[a, b], c] == table[a, table[b, c]]

    @pytest.mark.parametrize("q", _PRIME_POWERS)
    def test_quadratic_character_is_multiplicative(self, q: int) -> None:
        """chi(ab) == chi(a) chi(b), and exactly half the non-zero elements are squares."""
        p, n = _prime_power_factorization(q)
        table = _gf_multiplication_table(p, n)
        chi = _quadratic_character(p, n)

        assert chi[0] == 0
        assert set(np.unique(chi[1:])) == {-1, 1}
        assert (chi[1:] == 1).sum() == (q - 1) // 2

        for a in range(1, q):
            for b in range(1, q):
                assert chi[table[a, b]] == chi[a] * chi[b]

    def test_prime_field_matches_integer_arithmetic(self) -> None:
        """For n == 1 the table must be plain multiplication modulo p."""
        for p in (3, 5, 7, 11, 13):
            table = _gf_multiplication_table(p, 1)
            assert np.array_equal(table, np.outer(np.arange(p), np.arange(p)) % p)

    def test_legendre_symbol_matches_euler_criterion(self) -> None:
        """On a prime field, chi(a) must equal a**((p-1)/2) mod p, mapped to ±1."""
        for p in (3, 5, 7, 11, 13, 17, 19, 23):
            chi = _quadratic_character(p, 1)
            for a in range(1, p):
                euler = pow(a, (p - 1) // 2, p)
                assert chi[a] == (1 if euler == 1 else -1), f"p={p}, a={a}"


class TestPolynomialHelpers:
    """Polynomial remainder and irreducibility over GF(p)."""

    def test_remainder_of_known_division(self) -> None:
        """(x**2 + 1) mod (x + 1) over GF(3) is 2, since (-1)**2 + 1 = 2."""
        assert _polynomial_remainder([1, 0, 1], [1, 1], 3) == [2]

    def test_remainder_degree_is_bounded(self) -> None:
        rng = np.random.default_rng(seed=7)
        for p in (3, 5):
            for _ in range(50):
                dividend = [int(c) for c in rng.integers(0, p, size=6)]
                divisor = [*[int(c) for c in rng.integers(0, p, size=3)], 1]
                remainder = _polynomial_remainder(dividend, divisor, p)
                assert len(remainder) == len(divisor) - 1
                assert all(0 <= c < p for c in remainder)

    def test_remainder_of_a_multiple_is_zero(self) -> None:
        """Dividing the product f * g by g must leave no remainder."""
        p = 5
        g = [2, 3, 1]  # x**2 + 3x + 2
        f = [4, 1]  # x + 4
        product = [0] * (len(f) + len(g) - 1)
        for i, x in enumerate(f):
            for j, y in enumerate(g):
                product[i + j] = (product[i + j] + x * y) % p
        assert not any(_polynomial_remainder(product, g, p))

    @pytest.mark.parametrize(("p", "n"), [(3, 2), (3, 3), (3, 4), (5, 2), (5, 3), (7, 2), (11, 2)])
    def test_irreducible_polynomial_has_no_roots(self, p: int, n: int) -> None:
        """A necessary condition, and sufficient for degree 2 and 3."""
        polynomial = _irreducible_polynomial(p, n)
        assert len(polynomial) == n + 1
        assert polynomial[-1] == 1
        for x in range(p):
            value = sum(c * pow(x, i, p) for i, c in enumerate(polynomial)) % p
            assert value != 0, f"GF({p}**{n}) modulus has root x={x}"


class TestConferenceMatrices:
    """The constructions themselves."""

    @pytest.mark.parametrize("m", [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 38, 42, 44, 48, 50])
    def test_defining_property(self, m: int) -> None:
        """C.T @ C == (m - 1) I, zero diagonal, entries in {-1, 0, +1}."""
        matrix, _construction = _conference_matrix(m)
        assert matrix.shape == (m, m)
        assert np.all(np.diag(matrix) == 0)
        assert np.all(np.isin(matrix, (-1.0, 0.0, 1.0)))
        assert np.allclose(matrix.T @ matrix, (m - 1) * np.eye(m))
        assert np.allclose(matrix @ matrix.T, (m - 1) * np.eye(m))
        assert np.all((matrix == 0).sum(axis=0) == 1)
        assert np.all((matrix == 0).sum(axis=1) == 1)

    @pytest.mark.parametrize(("m", "expected"), [(10, "paley_q=9"), (16, "tabulated_order_16"), (26, "paley_q=25")])
    def test_construction_reported(self, m: int, expected: str) -> None:
        """The orders that the prime-only Paley construction used to miss."""
        _matrix, construction = _conference_matrix(m)
        assert construction == expected

    def test_order_16_table_is_exact(self) -> None:
        """The tabulated matrix is checked, not trusted.

        The upstream `definitive_screening_design` package carries an order-10
        table with a single mistyped entry, which leaves its 9- and 10-factor
        designs slightly non-orthogonal.  A table is only as good as its
        verification, so verify it.
        """
        matrix, _construction = _conference_matrix(16)
        assert np.array_equal(matrix.T @ matrix, 15 * np.eye(16))
        # Order 16 is skew-symmetric: C.T == -C away from the border convention.
        assert np.array_equal(matrix[1:, 1:], -matrix[1:, 1:].T)

    @pytest.mark.parametrize("m", [22, 34, 36, 40, 46])
    def test_unconstructible_orders_raise(self, m: int) -> None:
        """No approximation is ever returned.

        Order 22 is the first even order with no conference matrix at all: one
        exists for ``m = 2 (mod 4)`` only when ``m - 1`` is a sum of two
        squares, and 21 is not.  The others are orders this module has no
        construction for.  Both cases must raise so the caller steps up.
        """
        with pytest.raises(ValueError, match="No conference-matrix construction"):
            _conference_matrix(m)

    @pytest.mark.parametrize("m", [3, 5, 7, 15, 0, -2])
    def test_odd_orders_rejected(self, m: int) -> None:
        with pytest.raises(ValueError, match="even order"):
            _conference_matrix(m)

    def test_constructibility_predicate_agrees_with_the_constructor(self) -> None:
        """The cheap predicate and the actual construction must never disagree."""
        for m in range(2, 121):
            predicted = _is_constructible_conference_order(m)
            try:
                _conference_matrix(m)
                actual = True
            except ValueError:
                actual = False
            assert predicted == actual, f"disagreement at m={m}"

    def test_smallest_order_steps_up_only_when_it_must(self) -> None:
        assert _smallest_constructible_conference_order(6) == 6
        assert _smallest_constructible_conference_order(10) == 10
        assert _smallest_constructible_conference_order(16) == 16
        assert _smallest_constructible_conference_order(22) == 24
        assert _smallest_constructible_conference_order(28) == 28
        assert _smallest_constructible_conference_order(34) == 38


class TestConferenceMatrixValidation:
    """The guard that stops a bad matrix reaching a design."""

    def test_accepts_a_valid_matrix(self) -> None:
        matrix, _construction = _conference_matrix(6)
        _validate_conference_matrix(matrix, 6)  # must not raise

    def test_rejects_a_single_flipped_entry(self) -> None:
        """Exactly the upstream order-10 defect: one sign wrong out of 100."""
        matrix, _construction = _conference_matrix(10)
        matrix[7, 3] = -matrix[7, 3]
        with pytest.raises(ValueError, match=re.escape("fails C.T @ C")):
            _validate_conference_matrix(matrix, 10)

    def test_rejects_a_non_zero_diagonal(self) -> None:
        matrix, _construction = _conference_matrix(6)
        matrix[2, 2] = 1
        with pytest.raises(ValueError, match="non-zero diagonal"):
            _validate_conference_matrix(matrix, 6)

    def test_rejects_out_of_range_values(self) -> None:
        matrix, _construction = _conference_matrix(6)
        matrix[1, 2] = 2
        with pytest.raises(ValueError, match="outside"):
            _validate_conference_matrix(matrix, 6)

    def test_rejects_the_wrong_shape(self) -> None:
        matrix, _construction = _conference_matrix(6)
        with pytest.raises(ValueError, match="has shape"):
            _validate_conference_matrix(matrix[:, :4], 6)


class TestRunCountHelpers:
    """The public run-count helpers used by the strategy layer."""

    @pytest.mark.parametrize(
        ("k", "order", "runs"),
        [
            (3, 4, 9),
            (4, 4, 9),
            (5, 6, 13),
            (6, 6, 13),
            (9, 10, 21),
            (10, 10, 21),
            (15, 16, 33),
            (16, 16, 33),
            (21, 24, 49),  # order 22 does not exist, so 49 runs rather than 45
            (22, 24, 49),
            (25, 26, 53),
            (26, 26, 53),
            (27, 28, 57),
            (28, 28, 57),
        ],
    )
    def test_known_values(self, k: int, order: int, runs: int) -> None:
        assert dsd_conference_order(k) == order
        assert dsd_run_count(k) == runs

    def test_matches_the_generated_design(self) -> None:
        """The prediction must equal what the generator actually produces."""
        from process_improve.experiments.designs_response_surface import dispatch_dsd
        from process_improve.experiments.factor import Factor

        for k in range(3, 41):
            matrix, meta = dispatch_dsd([Factor(name=f"X{i}", low=-1, high=1) for i in range(k)])
            assert matrix.shape[0] == dsd_run_count(k), f"run count mismatch at k={k}"
            assert meta["conference_order"] == dsd_conference_order(k), f"order mismatch at k={k}"

    def test_usual_formula_holds_away_from_the_exceptions(self) -> None:
        """2k + 1 for even k and 2k + 3 for odd k, except where no matrix exists."""
        exceptions = {21, 22, 33, 34, 35, 36, 39, 40}
        for k in range(3, 41):
            if k in exceptions:
                assert dsd_run_count(k) > (2 * k + 1 if k % 2 == 0 else 2 * k + 3)
            else:
                assert dsd_run_count(k) == (2 * k + 1 if k % 2 == 0 else 2 * k + 3)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_too_few_factors(self, k: int) -> None:
        with pytest.raises(ValueError, match="at least 3 factors"):
            dsd_run_count(k)
