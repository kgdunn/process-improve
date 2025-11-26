# Process Improvement (using Data)

[![Python Package](https://github.com/kgdunn/process_improve/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/kgdunn/process_improve/actions/workflows/pythonpackage.yml)
[![PyPI version](https://badge.fury.io/py/process-improve.svg)](https://badge.fury.io/py/process-improve)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for industrial process improvement and data analysis, accompanying the online book [Process Improvement using Data](https://learnche.org/pid).

## Features

### 🧪 Designed Experiments (DOE)
- **Factorial designs**: Full and fractional factorial experiments
- **Optimal designs**: D-optimal, custom criteria
- **Model fitting**: Linear models with interaction effects
- **Optimization**: Response surface methodology
- **Analysis tools**: ANOVA, effect plots, diagnostics

### 📊 Latent Variable Methods
- **PCA**: Principal Component Analysis with missing data support
- **PLS**: Partial Least Squares regression
- **TPLS**: Three-way Partial Least Squares for batch data
- **Missing data**: Advanced algorithms for incomplete datasets
- **Visualization**: Score plots, loading plots, biplots
- **Diagnostics**: SPE (Squared Prediction Error), Hotelling's T²

### 📈 Process Monitoring
- **Control charts**: Shewhart, EWMA, CUSUM
- **Multivariate monitoring**: T² and SPE charts
- **Metrics**: Process capability indices

### 🔄 Batch Data Analysis
- **Alignment**: Synchronize batch trajectories
- **Feature extraction**: Summary statistics from batch profiles
- **Multivariate analysis**: Batch-wise and variable-wise PCA/PLS
- **Visualization**: Trajectory plots, heatmaps

### 📉 Regression & Univariate Methods
- **Robust regression**: Repeated median slope
- **Univariate statistics**: Comprehensive statistical tests
- **Visualization**: Modern plotting with matplotlib, plotly, seaborn

## Installation

### Using pip

```bash
pip install process-improve
```

### Using uv (recommended for development)

```bash
uv pip install process-improve
```

### From source

```bash
git clone https://github.com/kgdunn/process_improve.git
cd process_improve
pip install -e .
```

## Quick Start

### Principal Component Analysis (PCA)

```python
from process_improve.multivariate import PCA
import pandas as pd

# Load your data
X = pd.read_csv("data.csv")

# Fit PCA model
model = PCA(n_components=3)
model.fit(X)

# Visualize results
model.plot_scores()
model.plot_loadings()

# Access results
scores = model.transform(X)
print(f"Explained variance: {model.explained_variance_ratio_}")
```

### PCA with Missing Data

```python
from process_improve.multivariate import PCA

# PCA automatically handles missing values (NaN)
model = PCA(
    n_components=2,
    missing_data_settings={
        'method': 'NIPALS',  # or 'TSR'
        'max_iter': 100
    }
)
model.fit(X_with_missing)
```

### Design of Experiments

```python
from process_improve.experiments import lm, c, factorial_design

# Create a factorial design
design = factorial_design(
    factors={'Temperature': [60, 80], 'Time': [30, 45], 'Catalyst': ['A', 'B']},
    replicates=2
)

# Fit a linear model
model = lm("Yield ~ Temperature * Time + Catalyst", data=design)
model.summary()
model.plot_effects()
```

### Partial Least Squares (PLS)

```python
from process_improve.multivariate import PLS

# Fit PLS model relating X to Y
model = PLS(n_components=3)
model.fit(X, Y)

# Make predictions
Y_pred = model.predict(X_new)

# Visualize
model.plot_scores()
model.plot_loadings()
```

### Process Monitoring

```python
from process_improve.monitoring import control_chart

# Create control chart
chart = control_chart(
    data=process_data,
    chart_type='xbar',
    subgroup_size=5
)
chart.plot()
```

## Documentation

- **Full book and tutorials**: [https://learnche.org/pid](https://learnche.org/pid)
- **API Reference**: Coming soon
- **Example notebooks**: See `examples/` directory (coming soon)

## Dependencies

Core dependencies:
- Python ≥ 3.10
- numpy ≥ 1.26.4
- pandas ≥ 2.3.1
- scikit-learn ≥ 1.7.0
- matplotlib ≥ 3.10.3
- statsmodels ≥ 0.14.5

For visualization:
- plotly, bokeh, seaborn

For performance:
- numba ≥ 0.61.2

See `pyproject.toml` for complete list.

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/kgdunn/process_improve.git
cd process_improve

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
uv pip install pytest pytest-cov ruff black mypy pre-commit
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=process_improve --cov-report=html

# Run specific test file
pytest tests/test_multivariate.py
```

### Code quality

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check .

# Format code
ruff format .
black .

# Type checking
mypy process_improve
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this package in your research, please cite:

```
Kevin Dunn (2025). Process Improvement using Data.
https://learnche.org/pid
```

## License

MIT License

Copyright (c) 2010-2025 Kevin Dunn

See [LICENSE](LICENSE) file for details.

## Support

- **Bug reports**: [GitHub Issues](https://github.com/kgdunn/process_improve/issues)
- **Questions**: [GitHub Discussions](https://github.com/kgdunn/process_improve/discussions)
- **Email**: kgdunn@gmail.com

## Acknowledgments

This package accompanies the online book [Process Improvement using Data](https://learnche.org/pid) and incorporates statistical methods for industrial process improvement, experimental design, and multivariate data analysis.
