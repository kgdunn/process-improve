# process-improve

**Multivariate analysis, designed experiments, and process monitoring for Python.**
Built for the chemometrics, manufacturing, and pharma workflows where you need to
know not just *what fits*, but *is this observation normal, which variable moved,
and how sure am I?*

[![PyPI version](https://img.shields.io/pypi/v/process-improve.svg)](https://pypi.org/project/process-improve/)
[![Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkgdunn%2Fprocess-improve%2Fmain%2Fpyproject.toml&label=python)](https://pypi.org/project/process-improve/)
[![Downloads](https://static.pepy.tech/badge/process-improve)](https://pepy.tech/project/process-improve)
[![Downloads per month](https://static.pepy.tech/badge/process-improve/month)](https://pepy.tech/project/process-improve)
[![CI](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/kgdunn/process-improve/actions/workflows/run-tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/kgdunn/process-improve/branch/main/graph/badge.svg)](https://codecov.io/gh/kgdunn/process-improve)
[![Docs](https://img.shields.io/badge/docs-kgdunn.github.io-blue.svg)](https://kgdunn.github.io/process-improve/)
[![License](https://img.shields.io/pypi/l/process-improve.svg)](LICENSE)

---

> **New here?** The [architecture overview](https://kgdunn.github.io/process-improve/architecture.html)
> ([source](docs/architecture.rst)) is the map of the codebase - package layout, the estimator
> stack, and the MCP tool layer.

## What's new

The last few releases extend `process-improve` from offline model-building into
end-to-end, on-line workflows. Highlights (full history in [CHANGELOG.md](CHANGELOG.md)):

- **Models that keep up with a drifting process.** `AdaptivePCA` and
  `AdaptivePLS` (v1.55) are recursive estimators for on-line monitoring and soft
  sensing: start from an initial fit, then stream one observation at a time.
  They track the operating point, re-learn the correlation structure, and tell
  you - in units of components - exactly how far the process has drifted from
  where it was trained.
- **A DOE engine that goes past textbook designs.** OMARS (orthogonal minimally
  aliased response surfaces), D-/I-/A-optimal designs, fractional-cube CCDs,
  design augmentation (`fixed_runs=`), and an `evaluate_design` suite that scores
  any design on D/I/G-efficiency, aliasing, and prediction variance.
- **Sensory & descriptive panel analysis** (`process_improve.sensory`): validate
  a panel, flag inconsistent assessors with the Mixed Assessor Model, and relate
  attributes to product covariates - with an honest genuine-vs-proxy separation.
- **Robust regression** (`process_improve.regression`): repeated-median and
  Theil-Sen estimators for data with outliers, plus `OLS` and `fit_robust_lm`.

## What it does

`process-improve` provides production-grade implementations of the methods
practitioners actually use on real plant and lab data:

- **PCA** with SVD and NIPALS, plus native missing-value handling via Trimmed
  Score Regression
- **PLS** regression with a fully sklearn-compatible API, VIP scores, and
  cross-validated diagnostics
- **TPLS** - PLS for *T-shaped (multi-block) data structures*
- **Adaptive PCA / PLS** - recursive, self-updating models for on-line process
  monitoring and soft sensing; they follow a drifting process one observation at
  a time and report how far it has moved
- **Outlier detection** combining Hotelling's T² and SPE with an ESD-based test
- **Designed experiments** - full-factorial, fractional-factorial, and
  response-surface designs; OMARS and D-/I-/A-optimal designs; a design-quality
  scorer (`evaluate_design`); and a multi-stage DOE strategy recommender
- **Process monitoring** - Shewhart, CUSUM, and Holt-Winters control charts
- **Batch data analysis** - alignment, feature extraction, and multivariate
  batch monitoring (MBPCA / MBPLS)
- **Sensory & descriptive panel analysis** - panel validation, the Mixed Assessor
  Model, and attribute-to-product relations with a genuine-vs-proxy separation
- **Robust regression** - repeated-median and Theil-Sen estimators for data with
  outliers
- **Interactive Plotly diagnostics** bound directly to every fitted model

Outputs are `pandas`-native: scores, loadings, and predictions keep your row
and column labels.

It is the companion package to the online textbook
[Process Improvement using Data](https://learnche.org/pid).

## Works alongside scikit-learn

`process-improve` is designed to sit *next to* scikit-learn, not replace it. It
follows the same conventions (`fit`, `predict`, `score`, the `_` suffix on fitted
attributes), so its estimators drop straight into `Pipeline`, `GridSearchCV`, and
`cross_val_score`. What it adds is the process-analytics layer on top: the
diagnostics that tell you whether a new observation is normal, which variable moved, and
how confident the prediction is.

| Capability                                        | scikit-learn | process-improve |
| ------------------------------------------------- | :----------: | :-------------: |
| PCA, PLS with sklearn-style API                   |       ✓      |        ✓        |
| Missing-data fitting (NIPALS / TSR)               |       -      |        ✓        |
| Hotelling's T² + SPE outlier limits               |       -      |        ✓        |
| Variable-level score contributions                |       -      |        ✓        |
| Cross-validated coefficient confidence intervals  |       -      |        ✓        |
| Multi-block models (TPLS)                          |       -      |        ✓        |
| On-line / adaptive monitoring (recursive PCA/PLS) |       -      |        ✓        |
| Designed experiments, incl. OMARS & optimal       |       -      |        ✓        |
| Control charts (Shewhart / CUSUM / Holt-Winters)  |       -      |        ✓        |
| Batch process monitoring (MBPCA / MBPLS)          |       -      |        ✓        |
| Plotly diagnostics built in                       |       -      |        ✓        |
| Labeled `DataFrame` outputs                       |    partial   |        ✓        |

## Installation

```bash
pip install process-improve                    # core (numpy, pandas, sklearn, statsmodels, patsy, pydantic, pyyaml, tqdm)
pip install 'process-improve[plotting]'        # adds matplotlib, plotly, seaborn, ridgeplot
pip install 'process-improve[expt]'            # adds pyDOE3 (designed experiments / DOE)
pip install 'process-improve[batch]'           # adds openpyxl, scikit-image (batch process data IO)
pip install 'process-improve[mcp]'             # adds the MCP server runtime
pip install 'process-improve[fast]'            # adds numba (JIT speedups for batch alignment)
pip install 'process-improve[all]'             # everything above (the pre-1.24.11 closure)
```

Requires Python 3.10 or newer. The core install pulls in `numpy`, `pandas`,
`scikit-learn`, `statsmodels`, `patsy`, `pydantic`, `pyyaml`, and
`tqdm` (`scipy` arrives transitively via scikit-learn and statsmodels).
Heavier optional surfaces (plotting, designed experiments, batch IO,
MCP server, numba JIT) live in extras so a caller who only needs, say,
`detect_multivariate_outliers` does not have to install Plotly or numba.

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

# Predict new observations (sklearn-compatible: returns just y_hat)
y_pred = pls.predict(scaler_x.transform(X_new))

# Predict with full per-row diagnostics (scores, T², SPE, plus y_hat)
result = pls.diagnose(scaler_x.transform(X_new))
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

### One-shot optimal & OMARS designs

Ask for a ready-to-run design table and score it, in two lines:

```python
from process_improve.experiments import Factor, generate_design, evaluate_design

factors = [
    Factor(name="A", low=-1, high=1),
    Factor(name="B", low=-1, high=1),
    Factor(name="C", low=-1, high=1),
]

# An OMARS design: main effects clear of every second-order term
design = generate_design(factors, design_type="omars")

# Or a run-budgeted D-optimal design, then grade its quality
d_opt = generate_design(factors, design_type="d_optimal", budget=14)
print(evaluate_design(d_opt, metric="all"))   # D/I/G-efficiency, aliasing, prediction variance
```

### On-line monitoring with Adaptive PCA

A static model goes stale the moment the process drifts. `AdaptivePCA` starts
from an initial fit, then keeps learning as data streams in - flagging faults and
reporting exactly how far the process has moved from where it was trained:

```python
from process_improve.multivariate import AdaptivePCA

# Seed on a block of known-good ("common cause") data. Pass RAW data:
# the adaptive model mean-centres and scales internally.
monitor = AdaptivePCA(n_components=3).fit(X_reference)

# Feed live observations one row at a time
for _, row in X_stream.iterrows():
    result = monitor.update(row.to_numpy())
    if not result.in_control:
        print(f"Out-of-control point: SPE={result.spe:.2f}, T²={result.hotellings_t2:.2f}")

# How far has the model drifted from its training subspace? (in units of components)
print(monitor.distance_.tail())
print(monitor.center_shift_.tail())   # operating-point migration, in training-SD units
```

`AdaptivePLS` does the same for regression and soft sensing, and handles
infrequently-sampled responses: the X-space model adapts every step while the
regression part waits for the next lab result.

Longer, fully-worked versions of each example live in the
[Quickstart guide](https://kgdunn.github.io/process-improve/quickstart.html)
and the [`examples/`](examples/) folder.

New to designed experiments? The
[**Applied DoE tutorial**](https://kgdunn.github.io/process-improve/applied_doe/index.html)
is an eight-module worked-solution series.

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
- **Local docs build:** `cd docs && make html`

## Citing process-improve

If you use this package in academic work, please cite it. The
[`CITATION.cff`](CITATION.cff) file carries the current version and
release date, and GitHub renders a *"Cite this repository"* button in
the sidebar with ready-made BibTeX and APA entries:

```bibtex
@software{dunn_process_improve,
  author  = {Dunn, Kevin G.},
  title   = {{process-improve: Multivariate Analysis for Process Improvement}},
  year    = {2026},
  url     = {https://github.com/kgdunn/process-improve}
}
```

Add the `version` field from `CITATION.cff` (or the release tag you
installed) when citing a specific version.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and code
style. Bugs and feature requests can be filed on the
[issue tracker](https://github.com/kgdunn/process-improve/issues).

## License

MIT - see [LICENSE](LICENSE) for details.
