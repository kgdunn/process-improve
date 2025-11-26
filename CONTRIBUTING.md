# Contributing to Process Improve

Thank you for your interest in contributing to the Process Improve package! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of professional conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/process_improve.git
cd process_improve

# Add the upstream repository
git remote add upstream https://github.com/kgdunn/process_improve.git

# Install uv (optional but recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv  # or: python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with dev dependencies
uv pip install -e ".[dev]"  # or: pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Push updates to your fork
git push origin main
```

## Making Changes

### Creating a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### Development Workflow

1. **Make your changes** - Edit the relevant files
2. **Add tests** - Include tests for new functionality
3. **Run tests** - Ensure all tests pass
4. **Check code style** - Run linters and formatters
5. **Commit changes** - Write clear commit messages
6. **Push to your fork** - Push your branch
7. **Create pull request** - Submit PR for review

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=process_improve --cov-report=html

# Run specific test file
pytest tests/test_multivariate.py

# Run specific test
pytest tests/test_multivariate.py::test_pca_basic

# Run tests in parallel (faster)
pytest -n auto

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Include docstrings for complex tests
- Aim for high test coverage (minimum 70%, target 85%+)

Example test:

```python
def test_pca_handles_missing_data():
    """Test that PCA correctly handles datasets with missing values."""
    X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    model = PCA(n_components=2, missing_data_settings={'method': 'NIPALS'})
    model.fit(X)
    assert model.has_missing_data is True
    assert model.n_components_ == 2
```

### Test Coverage

Check coverage with:

```bash
# Generate HTML coverage report
pytest --cov=process_improve --cov-report=html
# Open htmlcov/index.html in your browser

# Or use make command
make coverage
```

## Code Style

This project follows strict code quality standards:

### Style Guidelines

- **Line length**: 120 characters maximum
- **Docstrings**: Google-style or NumPy-style
- **Type hints**: Use type hints for all public APIs
- **Imports**: Organized with isort
- **Formatting**: Black and Ruff

### Running Code Quality Tools

```bash
# Format code with black
black process_improve tests

# Format code with ruff
ruff format .

# Check for issues with ruff
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Sort imports
isort process_improve tests

# Type checking
mypy process_improve

# Run all pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They check:
- Code formatting (black, ruff)
- Import sorting (isort)
- Type hints (mypy)
- Linting (flake8, ruff)
- File quality (trailing whitespace, etc.)
- Security (no private keys, credentials)

To run manually:

```bash
pre-commit run --all-files
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
Short (50 chars or less) summary

More detailed explanatory text, if necessary. Wrap it to about 72
characters. The blank line separating the summary from the body is
critical.

- Bullet points are okay
- Use imperative mood ("Add feature" not "Added feature")
- Reference issues and pull requests

Fixes #123
```

### Pull Request Process

1. **Update documentation** - Include relevant documentation updates
2. **Add tests** - Ensure new code is tested
3. **Update CHANGELOG** - Add entry to CHANGELOG.md under [Unreleased]
4. **Ensure tests pass** - All CI checks must pass
5. **Request review** - Tag maintainers for review

### Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass locally
- [ ] Code coverage maintained or improved

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Reporting Bugs

### Before Submitting a Bug Report

- Check existing issues to avoid duplicates
- Verify the bug with the latest version
- Collect information about your environment

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import '...'
2. Call function '...'
3. See error

**Expected behavior**
What you expected to happen.

**Code example**
```python
# Minimal reproducible example
from process_improve.multivariate import PCA
# ...
```

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- Package version: [e.g., 0.9.96]
- Installation method: [pip, uv, conda]

**Additional context**
Add any other context about the problem here.
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, screenshots, or examples.
```

## Development Guidelines

### Adding New Features

1. **Discuss first** - Open an issue to discuss major changes
2. **Follow existing patterns** - Match the style of existing code
3. **Document thoroughly** - Include docstrings and update documentation
4. **Add examples** - Provide usage examples
5. **Test comprehensively** - Cover edge cases

### API Design Principles

- **Consistency** - Follow sklearn API conventions where applicable
- **Clarity** - Use descriptive names, avoid abbreviations unless standard
- **Simplicity** - Simple interfaces, complex implementation
- **Documentation** - Every public function/class needs a docstring

### Documentation

- Use NumPy-style or Google-style docstrings
- Include parameter types and descriptions
- Provide examples in docstrings
- Update README.md for major features
- Add entries to CHANGELOG.md

Example docstring:

```python
def calculate_spe(self, X: np.ndarray, component: int = None) -> np.ndarray:
    """
    Calculate the Squared Prediction Error (SPE) for observations.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    component : int, optional
        Number of components to use. If None, uses all fitted components.

    Returns
    -------
    np.ndarray
        SPE values for each observation, shape (n_samples,)

    Examples
    --------
    >>> model = PCA(n_components=2)
    >>> model.fit(X_train)
    >>> spe = model.calculate_spe(X_test)
    >>> print(f"Max SPE: {spe.max():.3f}")
    """
```

## Questions?

If you have questions about contributing:

- Open a [GitHub Discussion](https://github.com/kgdunn/process_improve/discussions)
- Email: kgdunn@gmail.com
- Check existing documentation at [learnche.org/pid](https://learnche.org/pid)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Process Improve! 🎉
