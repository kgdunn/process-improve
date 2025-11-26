# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with installation instructions, features overview, and quick start examples
- CHANGELOG.md to track version history and changes
- CONTRIBUTING.md with development guidelines
- Package classifiers in pyproject.toml for better PyPI discoverability
- Modern CI/CD pipeline with GitHub Actions testing Python 3.10, 3.11, and 3.12
- Coverage reporting with Codecov integration

### Changed
- Updated GitHub Actions workflow to use modern action versions (v4, v5)
- Converted dependency-groups to project.optional-dependencies for pip compatibility
- CI/CD now uses uv package manager for faster installations

### Fixed
- Broken CI/CD pipeline that referenced non-existent requirements.txt and setup.py
- Python version mismatch between CI (3.9) and package requirements (3.10+)
- Removed pytest import from production code in multivariate/methods.py

## [0.9.96] - 2025-01-XX

### Added
- Variable importance metric to F and D spaces

### Changed
- Improved Resampler functionality
- Better error handling across modules
- Naming consistency for input arguments

### Fixed
- Fix glitch around PLS class; all tests pass again
- Whitespace and formatting improvements

## [0.9.95] - 2025-01-XX

### Added
- Bootstrap and jackknife resampler to TPLS
- Pre-processing options for various modules
- Type hints throughout codebase
- VIP calculations use case

### Changed
- Drop older Python version support
- Remove scipy constraint
- Make naming explicit on R² variable: it is fractional

### Fixed
- Fix percentages and typing issues
- Repository administration updates

## [0.9.x] - Historical Releases

### Added
- Fit intercept for simple robust regression (contributed by mars-hub)
- TPLS (Three-way Partial Least Squares) implementation
- PCA and PLS with missing data support
- Batch data analysis tools
- Design of Experiments (DOE) functionality
- Process monitoring and control charts
- Comprehensive visualization tools

## Version History Notes

This package has been in active development since 2010. For detailed commit history, please see the [GitHub repository](https://github.com/kgdunn/process_improve).

### Migration Notes for 1.0.0 (Upcoming)

When version 1.0.0 is released, the following breaking changes are planned:

1. **API Consistency**: Plot methods will be added to model objects (e.g., `model.plot_scores()`)
2. **Attribute Naming**: Shorter aliases for common attributes (e.g., `model.spe` for `model.squared_prediction_error`)
3. **Class Naming**: Snake_case class names will be converted to PascalCase (e.g., `PCA_missing_values` → `PCAMissingValues`)
4. **fit_transform**: All transformers will implement the `fit_transform()` method for sklearn compatibility

[Unreleased]: https://github.com/kgdunn/process_improve/compare/v0.9.96...HEAD
[0.9.96]: https://github.com/kgdunn/process_improve/releases/tag/v0.9.96
