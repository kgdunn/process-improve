"""Conference and Hadamard matrices from finite fields (``experiments/_finite_fields.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import _finite_fields as ff
from process_improve.experiments.designs import generate_design
from process_improve.experiments.factor import Factor

#: Orders up to 100 with no conference matrix here: 22, 34, 58, 70, 78, 94 cannot exist
#: (Belevitch), the others need constructions that are not implemented.
_NOT_BUILT = {22, 34, 36, 46, 52, 58, 66, 70, 76, 78, 86, 92, 94, 100}


@pytest.mark.parametrize("q", [3, 4, 5, 7, 8, 9, 16, 25, 27, 49, 81, 121, 125])
def test_field_tables_are_a_field(q: int) -> None:
    """Every non-zero element has an inverse, and subtraction undoes addition."""
    multiply, subtract = ff._field_tables(q)
    assert all((multiply[a] == 1).sum() == 1 for a in range(1, q))
    assert np.array_equal(np.diag(subtract), np.zeros(q))
    # a - (a - b) == b
    assert all(subtract[a, subtract[a, b]] == b for a in range(q) for b in range(q))


@pytest.mark.parametrize("q", [9, 25, 27, 49, 81, 121, 125])
def test_quadratic_character_has_equal_halves(q: int) -> None:
    """Exactly half of the non-zero elements of GF(q) are squares."""
    character = ff.quadratic_character(q)
    assert (character == 1).sum() == (character == -1).sum() == (q - 1) // 2


@pytest.mark.parametrize("m", [m for m in range(4, 101, 2) if m not in _NOT_BUILT])
def test_conference_matrices(m: int) -> None:
    matrix, construction = ff.conference_matrix(m)
    assert ff.is_conference_matrix(matrix), construction


@pytest.mark.parametrize("m", sorted(_NOT_BUILT))
def test_missing_conference_orders_raise(m: int) -> None:
    with pytest.raises(ValueError, match=f"order {m}"):
        ff.conference_matrix(m)


def test_doubling_gives_order_16() -> None:
    _, construction = ff.conference_matrix(16)
    assert construction == "doubled(paley_q=7)"


@pytest.mark.parametrize(("m", "expected"), [(21, 24), (22, 24), (33, 38), (35, 38), (45, 48), (51, 54)])
def test_conference_order_steps_up(m: int, expected: int) -> None:
    assert ff.conference_order_at_least(m) == expected


@pytest.mark.parametrize("n", [n for n in range(4, 201, 4) if n not in {92, 116, 156, 172, 184, 188}])
def test_hadamard_matrices(n: int) -> None:
    found = ff.hadamard_matrix(n)
    assert found is not None
    matrix, construction = found
    assert ff.is_hadamard_matrix(matrix), construction
    assert (matrix[:, 0] == 1).all()


def test_prime_power() -> None:
    assert ff.prime_power(27) == (3, 3)
    assert ff.prime_power(49) == (7, 2)
    assert ff.prime_power(12) is None
    assert ff.prime_power(1) is None


#: Every field of order up to 32, prime and extension.
_FIELDS_UP_TO_32 = [q for q in range(2, 33) if ff.prime_power(q) is not None]


@pytest.mark.parametrize("q", _FIELDS_UP_TO_32)
def test_field_axioms_hold_exhaustively(q: int) -> None:
    """Every field axiom, checked over all pairs and triples of elements at once.

    Addition is recovered from the subtraction table as a + b = a - (0 - b).
    """
    multiply, subtract = ff._field_tables(q)
    add = subtract[:, subtract[0]]
    a, b, c = np.ix_(range(q), range(q), range(q))
    elements = np.arange(q)
    for table in (add, multiply):
        assert np.array_equal(table, table.T)  # commutative
        assert np.array_equal(table[table[a, b], c], table[a, table[b, c]])  # associative
    assert np.array_equal(add[0], elements)  # 0 is the additive identity
    assert np.array_equal(multiply[1], elements)  # 1 is the multiplicative identity
    assert np.array_equal(add[elements, subtract[0]], np.zeros(q, dtype=int))  # a + (0 - a) = 0
    assert ((multiply[1:, 1:] == 1).sum(axis=1) == 1).all()  # one inverse per non-zero element
    assert np.array_equal(multiply[a, add[b, c]], add[multiply[a, b], multiply[a, c]])  # distributive


@pytest.mark.parametrize("q", [q for q in _FIELDS_UP_TO_32 if q % 2])
def test_quadratic_character_is_multiplicative_and_sets_the_paley_symmetry(q: int) -> None:
    """chi(ab) = chi(a) chi(b) for all a, b; the Paley matrix is symmetric exactly when q = 1 (mod 4)."""
    multiply, _ = ff._field_tables(q)
    chi = ff.quadratic_character(q)
    assert np.array_equal(chi[multiply], np.outer(chi, chi))
    matrix = ff.paley_conference_matrix(q)
    assert np.array_equal(matrix, matrix.T) == (q % 4 == 1)
    assert np.array_equal(matrix, -matrix.T) == (q % 4 == 3)


@pytest.mark.parametrize("q", [2, 4, 8, 15])
def test_paley_needs_an_odd_prime_power(q: int) -> None:
    with pytest.raises(ValueError, match=f"^Paley's construction needs an odd prime power; got q={q}\\.$"):
        ff.paley_conference_matrix(q)


def test_an_odd_order_has_no_conference_matrix() -> None:
    with pytest.raises(ValueError, match=r"^No conference matrix of order 5 can be built here\.$"):
        ff.conference_matrix(5)


def test_no_buildable_conference_order_above_the_search_limit() -> None:
    with pytest.raises(ValueError, match=r"^No conference matrix of order between 201 and 200 can be built here\.$"):
        ff.conference_order_at_least(ff.MAX_ORDER + 1)


@pytest.mark.parametrize(("n", "construction"), [(1, "trivial"), (2, "sylvester"), (248, "sylvester(paley_ii_q=61)")])
def test_hadamard_matrices_outside_the_conference_constructions(n: int, construction: str) -> None:
    """Orders 1 and 2 are written down; 248 doubles the Paley II matrix of order 124 (Sylvester)."""
    found = ff.hadamard_matrix(n)
    assert found is not None
    matrix, name = found
    assert name == construction
    assert ff.is_hadamard_matrix(matrix)
    assert (matrix[:, 0] == 1).all()


@pytest.mark.parametrize(
    ("check", "build", "message"),
    [
        pytest.param(
            "is_conference_matrix",
            lambda: ff.conference_matrix(4),
            r"^The paley_q=3 construction did not give a conference matrix of order 4\.$",
            id="conference",
        ),
        pytest.param(
            "is_hadamard_matrix",
            lambda: ff.hadamard_matrix(4),
            r"^The I\+C\(paley_q=3\) construction did not give a Hadamard matrix of order 4\.$",
            id="hadamard",
        ),
    ],
)
def test_a_construction_that_fails_its_own_check_never_reaches_a_design(
    monkeypatch: pytest.MonkeyPatch, check: str, build, message: str
) -> None:
    """Every matrix is verified before it is returned; a failed verification is a RuntimeError, not a design."""
    monkeypatch.setattr(ff, check, lambda _matrix: False)
    with pytest.raises(RuntimeError, match=message):
        build()


@pytest.mark.parametrize("k", [24, 27, 33, 43, 51, 59, 89])
def test_plackett_burman_beyond_pydoe3(k: int) -> None:
    """pyDOE3 asserts at these sizes; the design is now built from a finite-field Hadamard matrix."""
    factors = [Factor(name=f"x{i}", low=0, high=1) for i in range(k)]
    result = generate_design(factors, "plackett_burman", n_center_points=0)
    x = result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)
    assert len(x) == 4 * (k // 4 + 1) if k < 88 else len(x) == 96
    np.testing.assert_array_equal(x.T @ x, len(x) * np.eye(k))
