# Process Improvement using Data

> A pragmatic Python toolkit for industrial process data — multivariate
> analysis, designed experiments, and process monitoring, in one place.

[![PyPI version](https://img.shields.io/pypi/v/process-improve.svg)](https://pypi.org/project/process-improve/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkgdunn%2Fprocess-improve%2Fmain%2Fpyproject.toml&label=python)](https://pypi.org/project/process-improve/)
[![License](https://img.shields.io/pypi/l/process-improve.svg)](LICENSE)
[![CI](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/kgdunn/process-improve/branch/main/graph/badge.svg)](https://codecov.io/gh/kgdunn/process-improve)
[![Docs](https://img.shields.io/badge/docs-kgdunn.github.io-blue.svg)](https://kgdunn.github.io/process-improve/)

## What is this?

`process-improve` is the companion package to the online textbook
[Process Improvement using Data](https://learnche.org/pid), and powers the
statistical engine behind [factori.al](https://factori.al). It bundles the
methods practitioners actually reach for on real plant and lab data —
PCA / PLS with proper missing-data handling, designed experiments with a
multi-stage strategy recommender, control charts, and batch-data tooling —
behind an API that is sklearn-compatible where it makes sense and
pandas-native throughout.

## Highlights

### 🧪 Designed Experiments

- Full-factorial, fractional-factorial, and response-surface designs (built on `pyDOE3`)
- A **DOE strategy recommender** that plans a complete multi-stage program — screening → optimization → confirmation — from ~50 deterministic rules, with budget-aware allocation and domain-specific advice for fermentation, cell culture, pharma, and 5 other domains
- ANOVA, main-effects plots, linear-model fitting, and response optimization

### 📊 Latent Variable Methods

- **PCA** with SVD and NIPALS algorithms, plus missing-data via Trimmed Score Regression
- **PLS** regression with a fully sklearn-compatible API
- **TPLS** — PLS for *T-shaped data structures*
- Diagnostics: Hotelling's T², SPE, score contributions, and ESD-based outlier detection
- Component selection via PRESS / Wold's criterion
- Interactive Plotly score, loading, SPE, and T² plots, bound directly to fitted models

### 📈 Process Monitoring

- Shewhart, CUSUM, and Holt-Winters control charts (regular and robust variants)
- Process-capability index `Cpk`

### 🔄 Batch Data Analysis

- DTW-based batch alignment, reference-batch selection, resampling
- 15+ batch feature extractors (mean, slope, area, elbow, rupture, crossings, robust variants, …)
- Format conversions between wide, melted, and dict-of-frames batch layouts

### 📐 Univariate & Robust Regression

- t-tests (paired, independent, plus DataFrame-aware helpers)
- ESD outliers, Sn estimator, MAD, normality tests, variance decomposition
- Robust regression: repeated-median slope and friends, for outlier-resistant fits

### 🎨 Visualization

- Plotly-backed plots that attach to fitted PCA / PLS models
- A backend-agnostic `ChartSpec` layer with Plotly and ECharts adapters
- DOE-specific plots: main-effects, design visualization

## Installation

```bash
pip install process-improve
```

Requires Python 3.10 or newer.

## Quick start

### PCA — Principal Component Analysis

```python
import pandas as pd
from process_improve.multivariate.methods import PCA, MCUVScaler

X = pd.read_csv("your_data.csv", index_col=0)
X_scaled = MCUVScaler().fit_transform(X)

pca = PCA(n_components=3).fit(X_scaled)
print(pca.r2_cumulative_)        # cumulative R² per component
pca.score_plot()                  # interactive Plotly plot
```

### PLS — Projection to Latent Structures

```python
from process_improve.multivariate.methods import PLS, MCUVScaler

X_s = MCUVScaler().fit_transform(X)
Y_s = MCUVScaler().fit_transform(Y)

pls = PLS(n_components=3).fit(X_s, Y_s)
result = pls.predict(X_s)
print(result.y_hat, result.spe, result.hotellings_t2)
```

### DOE — multi-stage experimental strategy

```python
from process_improve.experiments.factor import Factor, Response
from process_improve.experiments.strategy import recommend_strategy

factors = [
    Factor(name="Temperature", low=25, high=40, units="degC"),
    Factor(name="pH", low=5.0, high=7.5),
    Factor(name="Glucose", low=10, high=50, units="g/L"),
]
strategy = recommend_strategy(
    factors=factors,
    responses=[Response(name="Yield", goal="maximize", units="g/L")],
    budget=40,
    domain="fermentation",
)
for s in strategy["stages"]:
    print(s["stage_number"], s["design_type"], s["estimated_runs"])
```

Longer, fully-worked versions of each example live in the
[Quickstart guide](https://kgdunn.github.io/process-improve/quickstart.html)
and the `notebooks_examples/` folder.

## API design

PCA and PLS follow scikit-learn conventions: `fit()` returns `self`, fitted
attributes end with a trailing underscore (`scores_`, `loadings_`, `spe_`,
`hotellings_t2_`, `r2_cumulative_`, …), and `predict()` returns an
`sklearn.utils.Bunch` with named fields (`y_hat`, `spe`, `hotellings_t2`, …).
Inputs are accepted as `pandas.DataFrame`, and index/column labels are
preserved through `fit` and `transform`.

## Documentation & learning resources

- **API reference & user guide:** <https://kgdunn.github.io/process-improve/>
- **Companion textbook:** [Process Improvement using Data](https://learnche.org/pid)
- **Hosted experiment-design tool:** [factori.al](https://factori.al)
- **Local docs build:** `cd docs && make html`

## Contributing

Bug reports and feature requests are welcome on the
[issue tracker](https://github.com/kgdunn/process-improve/issues).

## License

MIT — see [LICENSE](LICENSE) for details.
