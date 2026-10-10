"""Property tests for the finite-field arithmetic in :mod:`process_improve.experiments._finite_fields`.

GF(q) exists exactly when ``q`` is a prime power. ``prime_power`` decides that, and
``_field_tables`` must build the field for every prime power and refuse every other
integer, with a message naming it. The reference here factorises by plain trial
division, independently of the code under test.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from process_improve.experiments import _finite_fields as ff


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of ``n`` with multiplicity, by trial division."""
    factors, d = [], 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


@given(n=st.integers(min_value=2, max_value=2000))
@settings(max_examples=300, deadline=None)
def test_gf_exists_exactly_for_prime_powers(n: int) -> None:
    """A prime power decomposes as p**k with p prime; any other integer has no field, and the error says so."""
    factors = _prime_factors(n)
    found = ff.prime_power(n)
    if len(set(factors)) == 1:
        assert found == (factors[0], len(factors))
    else:
        assert found is None
        with pytest.raises(ValueError, match=f"^{re.escape(f'GF({n}) does not exist: {n} is not a prime power.')}$"):
            ff._field_tables(n)
