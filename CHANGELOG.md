# Changelog

All notable changes to `process-improve` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries before `1.21.3` predate this changelog; see the
[commit history](https://github.com/kgdunn/process-improve/commits/main) for
those changes.

## [Unreleased]

## [1.22.8] - 2026-05-29

### Fixed

- MBPLS / MBPCA NIPALS no longer silently produce `inf`/`nan` when a score or
  loading vector collapses (a degenerate or perfectly collinear block, or a
  fully-deflated component): the sum-of-squares and norm denominators are now
  floored away from zero (SEC-05). R-squared for a zero-variance block or
  column is reported as `NaN` (undefined) rather than a misleading `1.0` or a
  divide-by-zero warning.
- MBPLS / MBPCA record per-component convergence in `fitting_info_["converged"]`
  and emit a `SpecificationWarning` when `max_iter` is reached without
  converging (SEC-06); previously a non-converged solution was returned
  silently.
- `Resampler.fractional` re-validates `fraction_excluded` (must be in the open
  interval `(0, 1)`), so mutating it to `0` after construction raises a clear
  error instead of `ZeroDivisionError` (SEC-06).

## [1.22.7] - 2026-05-29

### Security

- `safe_execute_tool_call` now actually terminates a runaway worker on timeout
  (SEC-02). Previously it called `shutdown(wait=False)`, which does not
  interrupt a worker already running a task, so a CPU-bound or infinite-loop
  tool kept holding a core after `ToolTimeoutError` was raised. Workers are now
  force-terminated (`terminate()` then `kill()`).
- The module-managed worker pool is recycled after every call (SEC-03), so each
  call runs in a fresh worker with isolated process-global state and reclaimed
  memory. Caller-provided executors are left untouched.

## [1.22.6] - 2026-05-29

### Security

- Tool dispatch now enforces each tool's declared JSON `input_schema` on the
  untrusted path (`safe_execute_tool_call`): types, `minimum`/`maximum`,
  `minItems`/`maxItems`, `enum`, `required`, and rejection of unknown
  parameters (SEC-04). Previously the schema was advisory and inputs were
  passed straight through as `**kwargs`. New dependency-free
  `validate_against_schema` in `tool_safety.py`; the in-process
  `execute_tool_call` fast path is unchanged.

## [1.22.5] - 2026-05-29

### Security

- `fit_linear_model` tool (`experiments/tools.py`): the user-supplied `formula`
  was passed straight to statsmodels/patsy, which evaluates formula terms as
  Python expressions; a crafted formula could execute arbitrary code (SEC-01).
  Formulas are now validated against a strict Wilkinson-notation allowlist over
  the data columns (new `validate_formula_is_safe` / `UnsafeFormulaError` in
  `experiments/models.py`) before they reach patsy.

## [1.22.4] - 2026-05-29

### Added

- `SECURITY_AUDIT.md`: a structured security and robustness audit of the
  package and its MCP tool-dispatch surface, ranking each finding under both an
  untrusted and a local-trusted threat model. This is a planning artifact that
  seeds follow-up hardening issues; no behaviour changes in this release.
  
### Fixed

- Docstring corrections so they match the implementation:
  - `ttest_paired` (`univariate/metrics.py`): the returned dict's
    `"Standard deviation"` key actually holds the standard error of the
    mean difference, not the sample standard deviation; the docstring now
    documents every returned key and calls out the misnomer explicitly.
  - `calculate_cpk` (`monitoring/metrics.py`): the `trim_percentile`
    parameter is described accurately - it is used both as the percentile
    for estimating missing specification limits and as the toggle between
    classical and robust centre/spread.
  - `repeated_median_slope` (`regression/methods.py`): added missing
    Parameters and Returns sections documenting the signature and return
    type.

## [1.22.3] - 2026-05-27

### Changed

- CI now includes experimental, allow-failure jobs for free-threaded
  Python 3.13t and 3.14t on ubuntu, since numba >= 0.63 ships
  free-threaded wheels (#43).

## [1.22.2] - 2026-05-22

### Fixed

- `f_elbow` (batch features): the per-batch result was written through a
  chained-indexing assignment that is a no-op under modern pandas, so the
  function silently returned column sums instead of elbow values. It now
  writes results correctly, and batches with no detectable elbow record
  `np.nan` instead of the `np.isnan` function object.
- Docstring corrections so they match the implementation: the `TPLS` class
  and its `fit`/`predict` examples (the `X` input has the keys `F`, `Z`, `Y`;
  `D` is passed at construction), `robust_regression` return-value keys,
  `find_elbow_point` window-growth description, `f_count` (counts non-missing
  observations), `ControlChart.calculate_limits` (supports both variants),
  `_steepest_path` `n_steps`, `colours_per_batch_id` parameter name, and the
  `extract_batch_features` tool no longer advertises unavailable `mad` /
  `robust_mad` features.

## [1.22.1] - 2026-05-21

### Fixed

- Batch multitags animation: the Pause button now targets the currently
  running animation with Plotly's `[None]` sentinel instead of the malformed
  `[[None]]` placeholder, so pausing works.

## [1.22.0] - 2026-05-21

### Added

- A shared Plotly base theme. Four templates are registered with Plotly
  (`pi_tufte`, `pi_economist`, `pi_journal`, `pi_brand`) so every plot in
  the library shares a consistent, professional look. `pi_journal` is
  applied as the default; switch the global default with
  `process_improve.visualization.set_theme(...)`, or override a single
  plot through its `template` setting.

### Changed

- The latent-variable plots (score, loadings, SPE, Hotelling's T2,
  explained-variance, correlation-loadings, observed-vs-predicted,
  coefficient), the batch plots and the DOE plots now inherit their
  styling from the base theme instead of hard-coding margins, legends,
  axes and marker colours.

## [1.21.7] - 2026-05-18

### Fixed

- `OLS.fit()` no longer raises `ValueError: The indices for endog and exog are
  not aligned` when the input `X` has a non-default pandas index (for example
  a `DatetimeIndex`, or a row subset sliced out of a larger DataFrame). The
  feature matrix is now reset to a clean `RangeIndex` so it always aligns
  positionally with the target.

## [1.21.6] - 2026-05-17

### Fixed

- `calculate_cpk` no longer raises a `TypeError` under NumPy 2.x when a
  specification limit is passed as `None` (the limit is estimated from the
  data via `trim_percentile`).
- `c()` no longer raises an `AttributeError` when building a categorical
  factor column from non-numeric values without an explicit `levels`
  argument; the levels are now sorted correctly.

### Removed

- `f_mad` and `f_robust_mad` batch feature functions. Both were
  non-functional: `f_mad` relied on `DataFrameGroupBy.mad()` (removed in
  pandas 2.0, below this project's supported pandas) and `f_robust_mad`
  was an unfinished stub that always raised. The `"mad"` and
  `"robust_mad"` keys are also removed from the batch feature dispatch
  table.

## [1.21.4] - 2026-05-17

### Added

- `CHANGELOG.md` to track notable changes between releases.
- Issue forms for bug reports, feature requests, and questions, plus a pull
  request template, under `.github/`.

## [1.21.3] - 2026-05-17

### Added

- Per-observation diagnostics for PCA and PLS.
- Cross-validation for PLS: component selection and beta-coefficient error
  bars.
- Explained-variance, correlation-loadings, observed-vs-predicted, and
  coefficient plots for PCA and PLS.
- RV and modified RV (RV2) matrix-correlation coefficients.
- `CITATION.cff` so GitHub renders a "Cite this repository" button, and
  `CONTRIBUTING.md` with development setup, testing, and code-style notes.

### Changed

- Reworked the README with a sharper value proposition and a
  "Why not scikit-learn?" comparison table.

[Unreleased]: https://github.com/kgdunn/process-improve/compare/v1.22.8...HEAD
[1.22.8]: https://github.com/kgdunn/process-improve/compare/v1.22.7...v1.22.8
[1.22.7]: https://github.com/kgdunn/process-improve/compare/v1.22.6...v1.22.7
[1.22.6]: https://github.com/kgdunn/process-improve/compare/v1.22.5...v1.22.6
[1.22.5]: https://github.com/kgdunn/process-improve/compare/v1.22.4...v1.22.5
[1.22.4]: https://github.com/kgdunn/process-improve/compare/v1.22.3...v1.22.4
[1.22.3]: https://github.com/kgdunn/process-improve/compare/v1.22.2...v1.22.3
[1.22.2]: https://github.com/kgdunn/process-improve/compare/v1.22.1...v1.22.2
[1.22.1]: https://github.com/kgdunn/process-improve/compare/v1.22.0...v1.22.1
[1.22.0]: https://github.com/kgdunn/process-improve/compare/v1.21.7...v1.22.0
[1.21.7]: https://github.com/kgdunn/process-improve/compare/v1.21.6...v1.21.7
[1.21.6]: https://github.com/kgdunn/process-improve/compare/v1.21.4...v1.21.6
[1.21.4]: https://github.com/kgdunn/process-improve/compare/v1.21.3...v1.21.4
[1.21.3]: https://github.com/kgdunn/process-improve/releases/tag/v1.21.3
