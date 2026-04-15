# Process Improvement using Data

A Python package for multivariate data analysis, designed experiments, and process monitoring. Companion to the online textbook [Process Improvement using Data](https://learnche.org/pid). This package also powers the statistical engine behind [factori.al](https://factori.al).

## Installation

```bash
pip install process-improve
```

## Quick Start

### PCA — Principal Component Analysis

```python
import pandas as pd
from process_improve.multivariate.methods import PCA, MCUVScaler

# Load and scale your data
X = pd.read_csv("your_data.csv", index_col=0)
scaler = MCUVScaler().fit(X)
X_scaled = scaler.transform(X)

# Fit a PCA model
pca = PCA(n_components=3).fit(X_scaled)

# Inspect results
print(pca.scores_)  # Score matrix (N x A)
print(pca.loadings_)  # Loading matrix (K x A)
print(pca.r2_cumulative_)  # Cumulative R² per component

# Detect outliers
outliers = pca.detect_outliers(conf_level=0.95)

# Contribution analysis
contrib = pca.score_contributions(pca.scores_.iloc[0].values)

# Select number of components via cross-validation
result = PCA.select_n_components(X_scaled, max_components=10)
print(result.n_components)

# Built-in plots
pca.score_plot()
pca.spe_plot()
pca.t2_plot()
pca.loading_plot()
```

### PLS — Projection to Latent Structures

```python
from process_improve.multivariate.methods import PLS, MCUVScaler

# Scale X and Y separately
scaler_x = MCUVScaler().fit(X)
scaler_y = MCUVScaler().fit(Y)

# Fit a PLS model
pls = PLS(n_components=3).fit(scaler_x.transform(X), scaler_y.transform(Y))

# Inspect results
print(pls.scores_)  # X scores (N x A)
print(pls.beta_coefficients_)  # Regression coefficients (K x M)
print(pls.r2_cumulative_)  # Cumulative R² for Y

# Predict new observations
result = pls.predict(scaler_x.transform(X_new))
print(result.y_hat)  # Predicted Y values
print(result.spe)  # SPE for new data
print(result.hotellings_t2)  # Hotelling's T² for new data

# Detect outliers and analyze contributions
outliers = pls.detect_outliers(conf_level=0.95)
contrib = pls.score_contributions(pls.scores_.iloc[0].values)
```

### DOE — Experimental Strategy Recommendation

Plan a complete multi-stage experimental program before running any experiments:

```python
from process_improve.experiments.factor import Factor, Response
from process_improve.experiments.strategy import recommend_strategy

# Define factors for a fermentation optimization
factors = [
    Factor(name="Temperature", low=25, high=40, units="degC"),
    Factor(name="pH", low=5.0, high=7.5),
    Factor(name="Glucose", low=10, high=50, units="g/L"),
    Factor(name="Yeast extract", low=1, high=10, units="g/L"),
    Factor(name="Agitation", low=100, high=400, units="rpm"),
    Factor(name="Aeration", low=0.5, high=2.0, units="vvm"),
    Factor(name="Inoculum", low=2, high=10, units="%v/v"),
]
responses = [Response(name="Yield", goal="maximize", units="g/L")]

# Get a complete experimental plan
strategy = recommend_strategy(
    factors=factors,
    responses=responses,
    budget=40,
    domain="fermentation",
)

# Inspect the multi-stage strategy
for stage in strategy["stages"]:
    print(f"Stage {stage['stage_number']}: {stage['stage_name']}")
    print(f"  Design: {stage['design_type']}, Runs: {stage['estimated_runs']}")
    print(f"  Purpose: {stage['purpose']}")

# Review reasoning, risks, and alternatives
print(strategy["budget_allocation"])
print(strategy["reasoning"])
```

The engine applies ~50 deterministic rules (from Montgomery, NIST, Stat-Ease)
to recommend screening, optimization, and confirmation stages — with
budget-aware allocation and domain-specific advice for fermentation, cell
culture, pharma, and 5 other application domains.

## Features

- **PCA** with SVD, NIPALS, and missing data (TSR) algorithms
- **PLS** regression with sklearn-compatible API
- **TPLS** (Total PLS) for multi-block data
- **Missing data handling** via TSR and NIPALS algorithms
- **Outlier detection** combining Hotelling's T² and SPE with robust ESD test
- **Score contributions** for variable-level diagnostics
- **Cross-validation** for component selection (PRESS with Wold's criterion)
- **Interactive plots** (Plotly) for scores, loadings, SPE, and T²
- **Designed experiments** — full factorial, fractional factorial, response surface
- **DOE strategy recommender** — multi-stage experimental planning (screening, optimization, confirmation) with budget-aware allocation and 8 application domains
- **Process monitoring** — Shewhart, CUSUM, EWMA control charts
- **Batch data analysis** — alignment, feature extraction, multivariate batch monitoring

## API Design

Both PCA and PLS follow sklearn conventions:
- Fitted attributes end with `_` (e.g., `scores_`, `loadings_`, `spe_`)
- `fit()` returns `self`
- `predict()` returns a `Bunch` object with named fields
- `score()` is compatible with `sklearn.model_selection.cross_val_score`
- Works with `pandas.DataFrame` inputs (preserves index and column names)

## Documentation

Full documentation is available at **https://kgdunn.github.io/process-improve/**.

To build the documentation locally:

```bash
cd docs
make html
```

## License

MIT License. See [LICENSE](LICENSE) for details.
