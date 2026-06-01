"""Property tests for :func:`process_improve._random.check_random_state`.

First example of the ENG-15 property-test pattern. Use as a template
when adding properties for PCA / PLS / scalers in follow-up PRs:

- Pick a property that must hold for *all* legal inputs, not a
  specific case.
- Use ``hypothesis.strategies`` to generate the legal inputs.
- Use ``np.testing.assert_array_equal`` / ``assert_allclose`` for
  numerical equality, never raw ``==`` on floats.
- Keep the test small (one property per function); hypothesis
  shrinks failing inputs better when each function tests one thing.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from process_improve._random import check_random_state


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_int_seed_is_deterministic(seed: int) -> None:
    """For any int seed, two calls draw bit-identical sequences.

    This is the load-bearing property of the helper: the entire
    reproducibility contract (`docs/development/reproducibility.rst`)
    rests on it.
    """
    draws_1 = check_random_state(seed).random(10)
    draws_2 = check_random_state(seed).random(10)
    np.testing.assert_array_equal(draws_1, draws_2)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_int_input_returns_a_generator(seed: int) -> None:
    """Whatever ``int`` we feed in, we get a ``Generator`` back."""
    rng = check_random_state(seed)
    assert isinstance(rng, np.random.Generator)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50)
def test_generator_input_is_returned_unchanged(seed: int) -> None:
    """A passed ``Generator`` is the *same* object on the way out.

    This is how the contract preserves caller-owned state: the
    caller's generator advances in place, so a subsequent call by
    the caller sees the advance.
    """
    g_in = np.random.default_rng(seed)
    g_out = check_random_state(g_in)
    assert g_out is g_in


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_draws=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=30)
def test_int_seed_advances_a_freshly_built_generator(seed: int, n_draws: int) -> None:
    """The ``Generator`` returned for an ``int`` seed behaves like a
    fresh ``default_rng(seed)``: same byte stream, same advance.

    This pins the "int -> default_rng(int)" rule against any future
    refactor that might add a transformation in between.
    """
    helper = check_random_state(seed).random(n_draws)
    reference = np.random.default_rng(seed).random(n_draws)
    np.testing.assert_array_equal(helper, reference)


@given(
    bad=st.one_of(
        st.text(),
        st.floats(),
        st.lists(st.integers()),
        st.dictionaries(keys=st.text(), values=st.integers()),
        st.booleans(),  # bool is excluded per docstring
    )
)
@settings(max_examples=50)
def test_unsupported_types_are_rejected(bad: object) -> None:
    """Anything outside ``int | Generator | None`` must raise ``TypeError``.

    No silent fall-through to "use it anyway" or "convert it
    somehow"; the rejection is the contract.
    """
    import pytest

    with pytest.raises(TypeError, match="random_state must be"):
        check_random_state(bad)  # type: ignore[arg-type]
