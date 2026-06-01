"""Perf baseline for :func:`process_improve._random.check_random_state`.

First example of the ENG-15 pattern. Use as a template when adding
baselines for PCA fit, PLS fit/predict, ``analyze_experiment``, and
``evaluate_design`` in follow-up PRs:

- One test per hot path.
- Use the smallest representative input that exercises the
  function; the baseline is about *change detection*, not real-world
  workloads.
- Name the test ``test_<function>_baseline`` so the regression-budget
  CI job (planned follow-up) can identify them.
- Mark anything that exceeds ~1 second per call as ``slow`` (see
  CONTRIBUTING.md / ENG-29) so contributors can skip it locally.
"""

from __future__ import annotations

import numpy as np

from process_improve._random import check_random_state


def test_check_random_state_int_baseline(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Resolving an ``int`` seed is the hot path inside any future
    iterative algorithm; track regressions here.
    """
    benchmark(check_random_state, 42)


def test_check_random_state_generator_passthrough_baseline(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Passing a pre-built ``Generator`` should be near-zero overhead."""
    g = np.random.default_rng(42)
    benchmark(check_random_state, g)
