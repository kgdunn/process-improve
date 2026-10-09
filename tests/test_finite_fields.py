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


@pytest.mark.parametrize("k", [24, 27, 33, 43, 51, 59, 89])
def test_plackett_burman_beyond_pydoe3(k: int) -> None:
    """pyDOE3 asserts at these sizes; the design is now built from a finite-field Hadamard matrix."""
    factors = [Factor(name=f"x{i}", low=0, high=1) for i in range(k)]
    result = generate_design(factors, "plackett_burman", n_center_points=0)
    x = result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)
    assert len(x) == 4 * (k // 4 + 1) if k < 88 else len(x) == 96
    np.testing.assert_array_equal(x.T @ x, len(x) * np.eye(k))
