"""Deterministic cost-shape assertions for :func:`process_improve._random.check_random_state`.

The reproducibility contract makes this helper the first call of every
RNG-touching public function, so its cost shape matters: a passed
``Generator`` must be returned as-is (zero-copy pass-through, no wrapping or
re-seeding), and an ``int`` seed must resolve to a fresh, deterministic
generator each call.

These tests assert those properties directly instead of timing them (#511):
identity for the pass-through path, and bit-identical draws plus fresh state
for the int path. Wall-clock benchmarking remains the planned ENG-15 CI job
(see CONTRIBUTING.md, "Performance-regression policy").
"""

from __future__ import annotations

import numpy as np

from process_improve._random import check_random_state


def test_check_random_state_generator_is_passed_through_unchanged() -> None:
    """A pre-built ``Generator`` comes back as the identical object.

    Identity (not equality) is the assertion: any wrapping, copying, or
    re-seeding on this hot path would both cost time and silently take
    ownership of the caller's RNG state.
    """
    g = np.random.default_rng(42)
    assert check_random_state(g) is g


def test_check_random_state_int_returns_fresh_deterministic_generator() -> None:
    """An ``int`` seed resolves to a new generator with bit-identical draws.

    Two calls with the same seed must produce independent generators (no
    shared, advancing state) whose draw sequences match exactly.
    """
    rng1 = check_random_state(42)
    rng2 = check_random_state(42)
    assert isinstance(rng1, np.random.Generator)
    assert rng1 is not rng2, "each int resolution must return a fresh generator"
    np.testing.assert_array_equal(rng1.random(16), rng2.random(16))
