# Changelog

All notable changes to `process-improve` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries before `1.21.3` predate this changelog; see the
[commit history](https://github.com/kgdunn/process-improve/commits/main) for
those changes.

## [Unreleased]

## [1.22.0] - 2026-05-21

### Added

- A shared Plotly base theme. Four templates are registered with Plotly
  (`pi_tufte`, `pi_economist`, `pi_journal`, `pi_brand`) so every plot in
  the library shares a consistent, professional look. `pi_brand` is
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

[Unreleased]: https://github.com/kgdunn/process-improve/compare/v1.22.0...HEAD
[1.22.0]: https://github.com/kgdunn/process-improve/compare/v1.21.7...v1.22.0
[1.21.7]: https://github.com/kgdunn/process-improve/compare/v1.21.6...v1.21.7
[1.21.6]: https://github.com/kgdunn/process-improve/compare/v1.21.4...v1.21.6
[1.21.4]: https://github.com/kgdunn/process-improve/compare/v1.21.3...v1.21.4
[1.21.3]: https://github.com/kgdunn/process-improve/releases/tag/v1.21.3
