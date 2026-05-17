# process-improve

**Multivariate analysis, designed experiments, and process monitoring for Python.**
Built for chemometrics, manufacturing, and pharma data - the methods that scikit-learn skips.

[![PyPI version](https://img.shields.io/pypi/v/process-improve.svg)](https://pypi.org/project/process-improve/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkgdunn%2Fprocess-improve%2Fmain%2Fpyproject.toml&label=python)](https://pypi.org/project/process-improve/)
[![Downloads](https://static.pepy.tech/badge/process-improve)](https://pepy.tech/project/process-improve)
[![Downloads per month](https://static.pepy.tech/badge/process-improve/month)](https://pepy.tech/project/process-improve)
[![CI](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/kgdunn/process-improve/branch/main/graph/badge.svg)](https://codecov.io/gh/kgdunn/process-improve)
[![Docs](https://img.shields.io/badge/docs-kgdunn.github.io-blue.svg)](https://kgdunn.github.io/process-improve/)
[![License](https://img.shields.io/pypi/l/process-improve.svg)](LICENSE)

---

## What it does

`process-improve` provides production-grade implementations of the methods
practitioners actually use on real plant and lab data:

- **PCA** with SVD and NIPALS, plus native missing-value handling via Trimmed
  Score Regression
- **PLS** regression with a fully sklearn-compatible API, VIP scores, and
  cross-validated diagnostics
- **TPLS** - PLS for *T-shaped (multi-block) data structures*
- **Outlier detection** combining Hotelling's T² and SPE with an ESD-based test
- **Designed experiments** - full-factorial, fractional-factorial, and
  response-surface designs, plus a multi-stage DOE strategy recommender
- **Process monitoring** - Shewhart, CUSUM, and Holt-Winters control charts
- **Batch data analysis** - alignment, feature extraction, and multivariate
  batch monitoring (MBPCA / MBPLS)
- **Interactive Plotly diagnostics** bound directly to every fitted model

Outputs are `pandas`-native: scores, loadings, and predictions keep your row
and column labels.

It is the companion package to the online textbook
[Process Improvement using Data](https://learnche.org/pid), and powers the
statistical engine behind [factori.al](https://factori.al).

## Why not scikit-learn?

scikit-learn answers *"what fits the data?"* - `process-improve` answers
*"is this batch normal, which variable went off, and how confident am I in the
prediction?"* The two libraries are designed to be used together;
`process-improve` follows sklearn conventions (`fit`, `predict`, `score`, the
`_` suffix on fitted attributes) and drops into existing pipelines.

| Capability                                       | scikit-learn | process-improve |
| ------------------------------------------------ | :----------: | :-------------: |
| PCA, PLS with sklearn-style API                  |       ✓      |        ✓        |
| Missing-data fitting (NIPALS / TSR)              |       -      |        ✓        |
| Hotelling's T² + SPE outlier limits              |       -      |        ✓        |
| Variable-level score contributions               |       -      |        ✓        |
| Cross-validated coefficient confidence intervals |       -      |        ✓        |
| Multi-block models (TPLS)                         |       -      |        ✓        |
| Designed experiments (DoE)                        |       -      |        ✓        |
| Control charts (Shewhart / CUSUM / Holt-Winters)  |       -      |        ✓        |
| Batch process monitoring (MBPCA / MBPLS)          |       -      |        ✓        |
| Plotly diagnostics built in                       |       -      |        ✓        |
| Labeled `DataFrame` outputs                       |    partial   |        ✓        |

## Installation

```bash
pip install process-improve
```

Requires Python 3.10 or newer. Built on `numpy`, `pandas`, `scipy`,
`scikit-learn`, `statsmodels`, `plotly`, and `pyDOE3`.

## Quick start

### PCA - Principal Component Analysis

```python
import pandas as pd
from process_improve.multivariate.methods import PCA, MCUVScaler

X = pd.read_csv("your_data.csv", index_col=0)
X_scaled = MCUVScaler().fit_transform(X)

pca = PCA(n_components=3).fit(X_scaled)
print(pca.r2_cumulative_)         # cumulative R² per component
pca.score_plot()                  # interactive Plotly figure

# Flag outliers using combined T² and SPE limits at 95% confidence
outliers = pca.detect_outliers(conf_level=0.95)

# Which variables drove the first observation off?
contrib = pca.score_contributions(pca.scores_.iloc[0].values)
```

### PLS - Projection to Latent Structures

```python
from process_improve.multivariate.methods import PLS, MCUVScaler

# Scale X and Y separately
scaler_x = MCUVScaler().fit(X)
scaler_y = MCUVScaler().fit(Y)
X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

pls = PLS(n_components=3).fit(X_s, Y_s)
print(pls.beta_coefficients_)     # regression coefficients (K x M)
print(pls.r2_cumulative_)         # cumulative R² for Y
print(pls.vip())                  # VIP scores per X variable

# Predict new observations, with diagnostics on the prediction
result = pls.predict(scaler_x.transform(X_new))
result.y_hat                      # point predictions
result.spe                        # squared prediction error
result.hotellings_t2              # Hotelling's T² for new observations

# Cross-validated component selection
cv_select = PLS.select_n_components(X_s, Y_s, max_components=6)
print(cv_select.n_components)     # recommended number of components
print(cv_select.rmsecv)           # RMSECV per component count

# Cross-validation with beta-coefficient confidence intervals
cv = pls.cross_validate(X_s, Y_s, cv="loo")
print(cv.beta_ci_lower, cv.beta_ci_upper)   # 95% CI for each beta
print(cv.significant)                       # betas significantly != 0
print(cv.q_squared)                         # cross-validated R² (Q²)
```

### DOE - multi-stage experimental strategy

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
and the `process_improve/notebooks_examples/` folder.

New to designed experiments? The
[**Applied DoE tutorial**](https://kgdunn.github.io/process-improve/applied_doe/index.html)
is an eight-module worked-solution series that mirrors the
[12-week DoE short course](https://yint.org) and shows the same workflow in
Python with `process-improve` end to end.

## API design

PCA and PLS follow scikit-learn conventions: `fit()` returns `self`, fitted
attributes end with a trailing underscore (`scores_`, `loadings_`, `spe_`,
`hotellings_t2_`, `r2_cumulative_`, ...), and `predict()` returns an
`sklearn.utils.Bunch` with named fields (`y_hat`, `spe`, `hotellings_t2`, ...).
Inputs are accepted as `pandas.DataFrame`, and index/column labels are
preserved through `fit` and `transform`.

## Documentation & learning resources

- **API reference & user guide:** <https://kgdunn.github.io/process-improve/>
- **Applied DoE tutorial (8 modules):**
  <https://kgdunn.github.io/process-improve/applied_doe/index.html>
- **Companion textbook:** [Process Improvement using Data](https://learnche.org/pid)
- **Hosted experiment-design tool:** [factori.al](https://factori.al)
- **Local docs build:** `cd docs && make html`

## Citing process-improve

If you use this package in academic work, please cite it:

```bibtex
@software{dunn_process_improve,
  author  = {Dunn, Kevin G.},
  title   = {{process-improve: Multivariate Analysis for Process Improvement}},
  year    = {2026},
  version = {v1.20.1},
  url     = {https://github.com/kgdunn/process-improve}
}
```

A `CITATION.cff` file is included, so GitHub renders a *"Cite this
repository"* button in the sidebar.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code
style. Bugs and feature requests can be filed on the
[issue tracker](https://github.com/kgdunn/process-improve/issues).

## License

MIT - see [LICENSE](LICENSE) for details.
