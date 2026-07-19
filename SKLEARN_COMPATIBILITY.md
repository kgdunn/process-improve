# scikit-learn compatibility

This document tracks the state of `process_improve`'s interop with
scikit-learn as of v1.35.0. It is a living document — refresh the audit
numbers and re-read the "What doesn't work yet" section before each
release that touches the multivariate package.

## TL;DR

- **Cross-validation:** complete. Repeated K-fold, 1-SE rule, in-fold
  scaling, ekf for PCA, randomization test, stability selection, nested
  CV. Closed in PRs #376 → #390 across issues #377, #378, #379, #380,
  #381, #382, #383.
- **Pipeline / `cross_val_score` / `GridSearchCV` / `clone`:** the
  blocking PLS contract issue is fixed (PR #390): `PLS.predict(X)` returns
  the predicted Y as a `pd.DataFrame`, satisfying `RegressorMixin`; the
  rich diagnostic view moved to `PLS.diagnose(X)`. `MCUVScaler.fit` now
  accepts the `y` kwarg that Pipeline threads through, and `PLS.fit`
  accepts a 1-D Y.
- **Everything else:** see the `check_estimator` baseline below. Three
  estimators sit at 60–66% pass on the sklearn 1.6+ convention battery;
  40 distinct failures, scoped into five tracked issues.

## What works today

### Cross-validation

| Capability | Estimator | Reference |
|---|---|---|
| Repeated K-fold + 1-SE rule + in-fold scaling | `PLS.select_n_components` | #376 |
| Element-wise k-fold (ekf) CV (Bro 2008) | `PCA.select_n_components` | #384 closes #377 |
| ekf + repeated K-fold + in-fold scaling | `PCA.select_n_components` | #385 closes #382 |
| Minka MLE + Horn parallel analysis + consensus mode | `PCA.minka_mle`, `PCA.parallel_analysis`, `return_consensus=True` | #386 closes #378 |
| Van der Voet randomization test | `PLS.select_n_components(selection_rule="randomization")` | #387 closes #379 |
| Stability-selection signal across CV repeats | `PLS.select_n_components` `selection_distribution` / `selection_is_stable` | #388 closes #380 |
| Nested CV for honest RMSEP | `PLS.nested_cv` | #389 closes #381 |

### sklearn Pipeline composition

The following compositions are verified by tests in
`tests/test_multivariate.py` (search for `test_pipeline_*`):

| Pattern | Status | Reference |
|---|---|---|
| `Pipeline([MCUVScaler(), PCA(n_components=k)]).fit_transform(X)` | ✅ | #390 |
| `Pipeline([MCUVScaler(), PLS(n_components=k)]).fit(X, y).predict(X)` | ✅ | #390 |
| `cross_val_score(pipe, X, y, cv=RepeatedKFold(...))` | ✅ | #390 |
| `GridSearchCV(pipe, {"pls__n_components": [...]}).fit(X, y)` | ✅ | #390 |
| `clone(pipe).fit(X, y)` | ✅ | #390 |

### API conventions

- `MCUVScaler.fit(X, y=None)` and `.transform(X, y=None)` accept and
  ignore `y` — needed for Pipeline.
- `PLS.fit(X, Y)` accepts 1-D `Y` (the shape sklearn pipelines pass for
  single-target regression); promoted to `(N, 1)` internally.
- `PLS.predict(X)` returns a `pd.DataFrame` (sklearn-compatible).
- `PLS.diagnose(X)` returns the rich `Bunch(scores, hotellings_t2, spe,
  y_hat)` that `predict()` used to return before 1.35.0.
- `PLS.score(X, Y, sample_weight=...)` accepts `sample_weight` (forwards
  to `sklearn.metrics.r2_score`). Note `fit` does *not* accept
  `sample_weight` — tracked in #394.

## What doesn't work yet

### `check_estimator` baseline (v1.38.0)

Run `python tools/check_estimator_audit.py` to refresh (the log output is
not committed; `*.log` is gitignored). As of v1.38.0
(after `get_feature_names_out` added in #391):

| Estimator | Passing | Total | Pass % | Issues that close remaining |
|---|---|---|---|---|
| `MCUVScaler` | 44 | 46 | 96% | #391-leftover (transformer-dtype: ndarray transform) |
| `PCA(n_components=2)` | 38 | 46 | 83% | #391-leftover, #396 |
| `PLS(n_components=2)` | 24 | 28 | 86% | (small: `score()` validation, `dtype_object`) |

`get_feature_names_out` itself doesn't close audit failures (it's purely
additive ecosystem support), but it unblocks
`pipe.set_output(transform="pandas")`, `pipe.get_feature_names_out()`,
SHAP / eli5 introspection, and similar tooling that keys off output
column names. The two remaining transformer failures
(`check_transformer_data_not_an_array`,
`check_transformer_preserve_dtypes`) require `transform()` to return an
ndarray by default - that's a breaking change to the package's
DataFrame-everywhere idiom and is left for a future major-version
refactor; the scaffolding (`get_feature_names_out`) is in place so the
switch lands cleanly when chosen.

`TPLS` / `MBPLS` / `MBPCA` are not in this audit — they take
`dict[str, DataFrame]` for X, which is outside `check_estimator`'s
fixture scope. Pipeline interop for those classes is tracked in #395.

### Tracked issues by capability

#### High-leverage (close many `check_estimator` failures at once)

- **#401** — Route `fit` / `predict` / `transform` through
  `sklearn.utils.validation.validate_data`. Single biggest yield:
  ~22 of 40 audit failures across all three estimators. Closes the
  cluster around `n_features_in_`, sparse rejection, complex/object
  dtype, empty data, NaN/inf, and error-message wording.
- **#391** — `get_feature_names_out` on `MCUVScaler` / `PCA` / `PLS`,
  plus making `transform()` return `ndarray` so `set_output("pandas")`
  drives the column-label decoration. Closes 2 PCA failures
  (`check_transformer_data_not_an_array`,
  `check_transformer_preserve_dtypes`) on top of unlocking
  `set_output` and `pipe.get_feature_names_out()`.
- **#393** — `__sklearn_tags__` declarations and the `MCUVScaler` mixin
  inheritance order fix. Closes `check_estimator_sparse_tag` (×3),
  `check_mixin_order`, and the setup-time `transformer_tags`
  `RuntimeError`.
- **#396** — `PCA.predict` returning `Bunch` and mutating `self.__dict__`
  via the `_LazyFrame` cache. Closes
  `check_estimators_pickle` (×2), `check_fit_idempotent`, and
  `check_dict_unchanged`.

#### Other ecosystem gaps

- **#392** — Set `feature_names_in_` (with trailing underscore, sklearn
  convention) on every estimator. Unblocks SHAP, eli5, model-card tools.
- **#394** — Support `sample_weight` in `PLS.fit` and forward it through
  `cross_validate` / `nested_cv` / `select_n_components`.
- **#395** — `TPLS.predict` / `MBPLS.predict` / `MBPCA.predict` still
  return `Bunch`. Same architectural choice as PR #390 needs to be
  applied to the multiblock classes.
- **#397** — `TransformedTargetRegressor` composition with
  `MCUVScaler` + `PLS` Pipeline. Untested.
- **#398** — `HalvingGridSearchCV` / `HalvingRandomSearchCV` interop.
  Untested.
- **#399** — `make_column_transformer` with `MCUVScaler` + `OneHotEncoder`
  on disjoint column subsets. Untested.

### Outstanding unscoped finding

The PLS audit reports one `<setup>` `IndexError: list index out of range`
on `check_estimator`. The root cause is undiagnosed (likely a
`check_classifiers_classes` or `check_regressor_data_not_an_array` that
constructs a degenerate Y). Likely to resolve incidentally when #401
lands; revisit afterward.

## Refreshing the audit

```bash
python tools/check_estimator_audit.py | tee tools/check_estimator_audit_main.log
```

The output is grouped by reason so identical failures across multiple
checks are bucketed together. Compare against the previous log to see
which checks moved from FAIL to PASS. Keep the log local: `*.log` files
are gitignored and must not be committed.

## Suggested order to land the follow-ups

Pure ordering by yield-per-risk, not a dependency graph (all of these
land independently; #391 benefits from #393 landing first because tags
inform the `set_output` behaviour, but it's not a hard requirement):

1. **#401** (`validate_data` sweep). Biggest single yield (~22 audit
   failures). Mechanical. Smallest blast radius.
2. **#393** (`__sklearn_tags__` + mixin order). Unlocks the NaN-through-
   Pipeline path for the NIPALS-based PCA and the multi-output scorer
   dispatch for PLS.
3. **#391** (`get_feature_names_out` + ndarray `transform`). Closes the
   transformer cluster and unlocks `set_output("pandas")`.
4. **#396** (`PCA.predict` → ndarray, drop `__dict__` mutation). Closes
   the pickle / idempotent / dict-unchanged cluster on PCA. This was
   intentionally scoped out of PR #390 because PCA inherits
   `TransformerMixin` (not `RegressorMixin`) so the Bunch return isn't
   a Pipeline blocker.
5. **#392** (`feature_names_in_`). Single mechanical assignment per
   `fit`; downstream tools benefit.
6. **#395** (multiblock `predict` rename). Same architectural decision
   as PR #390, applied to `TPLS` / `MBPLS`. Lower priority than PCA/PLS
   because users multi-pipeline these less often.
7. **#394** (`sample_weight` in `PLS.fit`). Real new behaviour, not
   just a contract fix.
8. **#397** (`TransformedTargetRegressor`), **#398**
   (`HalvingGridSearchCV`), **#399** (`make_column_transformer`).
   Verification work: add tests, fix whatever falls out.

After (1)–(4) the audit should sit comfortably above 90% pass on every
estimator; re-run the audit and refine remaining issues at that point.
