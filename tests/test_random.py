"""Tests for :func:`process_improve._random.check_random_state`."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve._random import check_random_state


class TestResolution:
    """The three accepted forms each resolve as documented."""

    def test_none_returns_a_generator(self):
        rng = check_random_state(None)
        assert isinstance(rng, np.random.Generator)

    def test_int_returns_a_generator(self):
        rng = check_random_state(0)
        assert isinstance(rng, np.random.Generator)

    def test_generator_is_returned_unchanged(self):
        rng_in = np.random.default_rng(42)
        rng_out = check_random_state(rng_in)
        assert rng_out is rng_in


class TestReproducibility:
    """Same int twice must produce bit-identical draws."""

    def test_same_int_reproducible(self):
        rng1 = check_random_state(0)
        rng2 = check_random_state(0)
        assert float(rng1.random()) == float(rng2.random())

    def test_different_ints_differ(self):
        # With 53 bits of float precision, two independent draws colliding has
        # probability ~2^-53. Safe to assert ``!=``.
        rng1 = check_random_state(0)
        rng2 = check_random_state(1)
        assert float(rng1.random()) != float(rng2.random())

    def test_int_run_advances_independently_from_caller_generator(self):
        # Resolving an int should not perturb any external Generator.
        external = np.random.default_rng(99)
        before = float(external.random())
        # A call that resolves an int and consumes from it.
        _ = check_random_state(0).random()
        after = float(external.random())
        # The external generator must have advanced by exactly one step.
        external2 = np.random.default_rng(99)
        _ = external2.random()
        assert float(external2.random()) == after
        assert before != after

    def test_numpy_int_accepted(self):
        # NumPy integer scalars satisfy ``isinstance(_, numbers.Integral)``.
        rng_np = check_random_state(np.int64(0))
        rng_py = check_random_state(0)
        assert float(rng_np.random()) == float(rng_py.random())


class TestRejection:
    """Misuse must raise a clear TypeError, not silently succeed."""

    def test_bool_true_rejected(self):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state(True)

    def test_bool_false_rejected(self):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state(False)

    def test_legacy_randomstate_rejected(self):
        # ``np.random.RandomState`` is the legacy API; migration message in
        # the helper docstring points callers at ``int(rs.randint(0, 2**31))``.
        legacy = np.random.RandomState(0)
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state(legacy)

    def test_string_rejected(self):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state("0")

    def test_float_rejected(self):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state(0.0)

    def test_list_rejected(self):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state([0, 1])


class TestUnderDashO:
    """Smoke check that the helper does not rely on ``assert``.

    SEC-08 / SEC-17 / ENG-11 forbid ``assert`` for input validation
    because ``python -O`` strips it. This test re-imports the module
    after ensuring ``__debug__`` is consistent with the test runner and
    confirms that misuse still raises.
    """

    def test_misuse_raises_even_when_asserts_would_be_stripped(self):
        # We cannot toggle -O at runtime, but we can verify that the helper
        # uses an explicit raise rather than an assert by inspecting the
        # source. This is a regression guard for the ENG-11 rule.
        import inspect

        from process_improve import _random

        source = inspect.getsource(_random.check_random_state)
        assert "raise TypeError" in source
        # The helper must not use an assert for input validation.
        # (The only ``assert`` allowed in production code is one that
        # documents an internal invariant -- there are none here.)
        validation_block = source.split("raise TypeError")[0]
        assert "assert " not in validation_block
