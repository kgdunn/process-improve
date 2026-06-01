"""Performance baselines using ``pytest-benchmark``.

This directory hosts perf tests for hot paths so we notice silent
regressions. The CI job that enforces a regression budget lands
once we have at least two baselines and a stored reference run --
trying to enforce a budget against a single just-introduced
baseline would be meaningless.

Scaffolding landed in ENG-15. Follow-up PRs add baselines for PCA
fit, PLS fit/predict, ``analyze_experiment``, ``evaluate_design``,
and batch alignment (the public-API "hot paths" called out in the
performance-regression section of CONTRIBUTING.md).

Run locally with::

    pytest tests/perf -o "addopts=" --benchmark-only

The default ``addopts`` in ``pytest.ini`` include ``-n auto`` which
``pytest-benchmark`` does not support; the local run above strips it.
"""
