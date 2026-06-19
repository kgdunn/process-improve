# Changelog

All notable changes to `process-improve` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries before `1.21.3` predate this changelog; see the
[commit history](https://github.com/kgdunn/process-improve/commits/main) for
those changes.

## [Unreleased]

## [1.43.0] - 2026-06-19

### Added

- `evaluate_design` gains four model-aware metrics and a convenience aggregate:
  `a_optimality` (`trace((XᵀX)⁻¹)`) and `e_optimality` (smallest eigenvalue of
  `XᵀX`), each with an optional normalised efficiency; `correlation`, the
  coding-invariant residualised maximum and mean absolute correlation among the
  second-order terms plus the full matrix; `alias_matrix`, the general bias
  matrix `A = (X1ᵀX1)⁻¹ X1ᵀX2` (default `X2` = the two-factor interactions
  outside the model) with its worst single bias, main-effect-row maximum, and
  Frobenius norm; and `fds`, the fraction-of-design-space distribution of the
  prediction variance over the design region with the region average (I) and
  maximum (G) in σ² units and the run-count-scaled SPV variants. `metric="all"`
  and a thin `evaluate_all(...)` wrapper return every metric in one call.
- Region-based metrics take `region` (`"cuboidal"` or `"spherical"`),
  `n_samples`, `include_vertices`, and `random_seed` parameters; all sampling is
  seeded and the region and sample size are echoed in the `fds` payload, and the
  `2**k` cube vertices are always included so the worst-case G value at a corner
  is represented.

### Fixed

- `i_efficiency` / `g_efficiency` no longer raise a matmul size mismatch (for
  example `size 11 is different from 21`) when evaluated against an explicit
  reduced formula such as `"A+B+C+D+E+I(A**2)+...+I(E**2)"`. The region
  evaluation grid is now expanded through the exact fitted model matrix instead
  of a re-inferred shorthand, and `i_efficiency` / `g_efficiency` derive from
  the same region machinery as `fds`.

## [1.42.1] - 2026-06-19

### Fixed

- Clarified PCA/PLS `spe_` attribute docstring (it stores the square root of
  the row sum-of-squared X-residuals, so it is on the residual scale, not the
  squared scale).
- Corrected the TPLS `d_matrix` docstring type from
  `dict[str, dict[str, pd.DataFrame]]` to `dict[str, pd.DataFrame]` so it
  matches the flat-dict shape that the constructor actually validates.
- Fixed a typo in the `center()` docstring ("but return" -> "must return")
  and added a NumPy-style Returns section to `scale()` describing the two
  return shapes (with and without `extra_output=True`).
- Documented `ControlChart` variant strings in lowercase (`'hw'`,
  `'xbar.no.subgroup'`, `'cusum'`) and noted that the comparison is
  case-insensitive, matching the `.strip().lower()` normalisation applied on
  assignment.

## [1.42.0] - 2026-06-19

### Added

- `generate_design(design_type="ccd", cube="fractional")` builds the cube
  (factorial) portion of a central composite design from a resolution-V (or
  higher) fractional factorial instead of the full 2^k factorial, keeping the
  run count practical for k >= 5 (for example, a five-factor CCD with a 16-run
  cube + 10 axial + 6 centre = 32 runs). An optional `generators=[...]` pins the
  cube fraction; the chosen generators, defining relation, and resolution are
  recorded on the returned `DesignResult`. `cube="full"` remains the default.

## [1.41.0] - 2026-06-15

### Added

- OMARS (Orthogonal Minimally Aliased Response Surface) designs are now a
  first-class design type: `generate_design(..., design_type="omars")`
  constructs the conference-matrix foldover family (the definitive screening
  design is its minimal member). New `omars_properties()` and `is_omars()`
  verifiers check the defining OMARS properties (three-level, balanced, main
  effects mutually orthogonal and orthogonal to all second-order terms) on any
  coded design matrix, so generated or externally supplied designs can be
  validated without network access. An `omars` knowledge node is included in
  the DOE knowledge base.

## [1.40.2] - 2026-06-12

### Fixed

- `ControlChart` Holt-Winters warm-up: when the warm-up residuals have
  genuine variability but a zero median-absolute-deviation (for example,
  values symmetrically distributed around the median), `sigma_0` now
  falls back to the standard deviation instead of staying at zero. This
  avoids spuriously raising the "zero-variance warm-up window" error (or
  producing undefined limits) for data that actually carries
  information. A genuinely constant warm-up window still raises, as
  before.

## [1.40.1] - 2026-06-11

### Fixed

- Importing `process_improve` (or `process_improve.visualization`) no
  longer changes Plotly's global default template. Previously the
  package set `plotly.io.templates.default = "pi_journal"` on import,
  which silently restyled every other Plotly figure created in the same
  process. The `pi_*` themes are still registered on import (additive,
  namespaced), and the library's own plots are unchanged because they
  request the `pi_journal` template explicitly. Call
  `process_improve.visualization.set_theme()` to opt a whole session
  into a process-improve theme as before.

## [1.40.0] - 2026-06-07

### Added

- **`PLS.fit(X, Y, sample_weight=...)`** for weighted PLS regression
  (#394). Implemented via the `sqrt(w)`-rescale identity at NIPALS
  entry: row-rescaling X and Y by `sqrt(w)` makes every cross-product
  in the NIPALS inner loop weighted (`X' W u`, `Y' W t`, `X' W X`),
  while loadings, weights, and beta fall out unchanged. Scores are
  rebuilt on the original sample scale via the `T = X @ R` direct-
  weights identity, which correctly handles zero-weight rows
  (`sample_weight=[1, 1, 0, 0, 1]` produces a fit identical to one on
  rows `[0, 1, 4]` only).

  Validation up front: non-negative, finite, length matches `X` / `Y`.

  - **`PLS.cross_validate(..., sample_weight=...)`** threads the
    weights through to each per-fold refit (the weights are subset by
    the training index of each fold, never globally normalised - so
    1-SE-style component selection stays well-defined). Length
    validation matches `fit()`.
  - **`PLS.score(X, Y, sample_weight=...)`** already forwarded
    `sample_weight` to `sklearn.metrics.r2_score`; this version's
    weighted fit makes the trio (fit + score + cross_validate)
    consistent end-to-end.

  Eight tests in `tests/test_multivariate_sample_weight.py` cover:
  - Identity (`sample_weight=ones(N)` reproduces the unweighted fit
    to `1e-10` on every fitted attribute).
  - Half-zero collapse (`sample_weight=[1..., 0...]` matches a fit on
    the surviving rows to `1e-9`).
  - Heteroscedastic recovery (`1/sigma_i**2` weights produce a beta
    meaningfully closer to a clean-data oracle than the unweighted
    full-data fit).
  - sklearn Pipeline routing
    (`pipe.fit(X, y, pls__sample_weight=w)` reaches the inner PLS).
  - Three rejection paths (negative, NaN/inf, wrong length).
  - `cross_validate` thread-through (ones-weighted CV matches the
    unweighted CV's beta_mean and q_squared).

  Not yet threaded (follow-up): `PLS.select_n_components` and
  `PLS.nested_cv` don't accept `sample_weight` in this release. The
  per-fold subsetting pattern that landed in `cross_validate` is the
  template; doing the same in those two entry points is mechanical
  but tedious and was left as a future scope.

## [1.39.0] - 2026-06-07

### Added

- **`q2_se` in the component selectors.** `PCA.select_n_components` and
  `PLS.select_n_components` now return a per-component `q2_se` series: the
  standard error on the Q2 scale (the half-width of a +/-1 SE band around the
  validated Q2 curve). For PCA it is the existing `se_press` rescaled by the
  constant null-model sum-of-squares; for PLS it is the per-fold total-PRESS
  standard error divided by the total Y sum-of-squares. This lets callers draw
  a Q2 uncertainty band without re-deriving the normalisation themselves.

## [1.38.4] - 2026-06-07

### Added

- **`diagnose(X)`** on `MBPLS`, `MBPCA`, and `TPLS` as the canonical
  name for the existing diagnostics `Bunch` (#395). Matches
  `PLS.diagnose` and `PCA.diagnose` (added in #406).
  - `MBPLS.diagnose(X)` → `Bunch(super_scores, block_scores,
    predictions, block_spe, hotellings_t2)`.
  - `MBPCA.diagnose(X)` → `Bunch(super_scores, block_scores,
    block_spe, hotellings_t2)`.
  - `TPLS.diagnose(X)` → `Bunch(hat, t_scores_super, spe,
    hotellings_t2)`.

### Deprecated

- **`MBPLS.predict(X)` / `MBPCA.predict(X)` / `TPLS.predict(X)`** are
  now thin shims that forward to `diagnose(X)` and emit a
  `DeprecationWarning`. Behaviour and return shape are unchanged; the
  rename clears the `predict` name for a future contract that matches
  its sklearn-convention meaning (regression-style prediction), and
  removes the asymmetry with `PLS.predict` / `PLS.diagnose`. Slated
  for removal in 2.0.0.

  `TPLS.score` was migrated to call `diagnose` internally so the
  package itself doesn't emit the new warning.

## [1.38.3] - 2026-06-07

### Added

- **sklearn-interop verification tests** for three sklearn compositions
  spun out of the audit in #383:
  - `TransformedTargetRegressor(regressor=Pipeline([MCUVScaler, PLS]),
    transformer=MCUVScaler())` round-trips Y through scaling and
    predicts on the original Y scale (#397).
  - `make_column_transformer((MCUVScaler, numeric_cols),
    (OneHotEncoder, cat_cols))` feeding into `Pipeline([..., PLS])`
    fits and predicts; `get_feature_names_out` composes correctly
    through the ColumnTransformer (#399).
  - `HalvingGridSearchCV` and `HalvingRandomSearchCV` over
    `Pipeline([MCUVScaler, PLS])` finish and pick a sensible
    `n_components` (#398).

  No source-code changes were required - the foundational work in
  #391 (`get_feature_names_out`), #392 (`feature_names_in_`), #402
  (`validate_data` routing) and #393 (`__sklearn_tags__`) was
  sufficient. These tests lock the working compositions in.

## [1.38.2] - 2026-06-07

### Added

- **`feature_names_in_` on `MBPCA` and `MBPLS`** (#392). Set on the
  fitted estimator as a flat ndarray that concatenates each X-block's
  column names in block-iteration order, matching the sklearn
  convention. Lets downstream tooling (SHAP, eli5, model-card
  libraries, ad-hoc introspection) treat a multiblock fit through the
  same surface as a single-block estimator.

  `n_features_in_` was already set; this commit pairs the public
  feature-name vector with it. The `block_widths_`, `block_names_`,
  and `_block_columns` per-block surfaces are unchanged - the flat
  vector is additional, not a replacement.

  `PCA`, `PLS`, and `MCUVScaler` already set `feature_names_in_` as a
  side-effect of `validate_data(reset=True)` (landed in #402). `TPLS`
  is deliberately not included: its nested input (`{"F", "Z", "Y"}`,
  each sub-keyed by group) has no clean single feature-name vector
  and it cannot be placed in a standard sklearn Pipeline.

## [1.38.1] - 2026-06-07

### Added

- **`PCA.diagnose(X)`** as the canonical name for the existing diagnostics
  Bunch (`scores`, `hotellings_t2`, `spe`). Matches `PLS.diagnose`.

### Deprecated

- **`PCA.predict(X)`** is now a thin shim that forwards to
  `PCA.diagnose(X)` and emits a `DeprecationWarning`. Behaviour and return
  shape are unchanged; the rename clears the `predict` name for a future
  contract that matches its sklearn-convention meaning (regression-style
  prediction), and removes the asymmetry with `PLS.predict` / `PLS.diagnose`
  (#396). Slated for removal in 2.0.0.

## [1.38.0] - 2026-06-07

### Added

- **`get_feature_names_out` on `MCUVScaler`, `PCA`, and `PLS`** for
  sklearn 1.2+ `set_output` and Pipeline introspection. Three small
  additions, no breaking changes:
  - `MCUVScaler.get_feature_names_out()` is column-preserving:
    returns the fit-time `feature_names_in_` (sklearn's standard
    fallback when an `input_features` argument is supplied).
  - `PCA.get_feature_names_out()` returns
    `np.array(["PC1", "PC2", ..., "PC{n_components}"])` - the score
    column names.
  - `PLS.get_feature_names_out()` returns
    `np.array(["T1", "T2", ..., "T{n_components}"])` - the X-score
    column names.

  These unblock:
  - `pipe.get_feature_names_out()` for any Pipeline containing one of
    these estimators (previously would raise `AttributeError`).
  - `pipe.set_output(transform="pandas")` correctly labels the
    resulting DataFrame's columns; sklearn's `_SetOutputMixin` wraps
    the ndarray with our component names.
  - SHAP / eli5 / model-card libraries that introspect output column
    names of pipeline components.

  Note: `transform()` itself still returns a `pd.DataFrame` by default
  (the chemometric idiom this package has used since v1.0). Users who
  prefer sklearn-canonical ndarray output can opt in via
  `pca.set_output(transform="default")` or work through a
  `Pipeline(...).set_output(transform="pandas")` and let sklearn drive
  the type. A future major-version revisit may flip the default; the
  scaffolding (`get_feature_names_out`) is in place so the switch is
  one-line when chosen.

## [1.37.0] - 2026-06-07

### Added

- **`__sklearn_tags__` declarations on `MCUVScaler`, `PCA`, and `PLS`**
  for sklearn 1.6+ capability dispatch:
  - `MCUVScaler`: `input_tags.allow_nan = True`. `fit` / `transform`
    use `np.nanmean` / `np.nanstd` and pass NaN through unchanged.
  - `PCA`: `input_tags.allow_nan = True`. The NIPALS / TSR algorithm
    paths handle missing data; the SVD path still raises explicitly.
    Inside a `Pipeline`, sklearn's `check_array` no longer rejects
    NaN before the algorithm has a chance to see it.
  - `PLS`: `input_tags.allow_nan = True` and
    `target_tags.multi_output = True`. The first unblocks the missing-
    data path through a Pipeline; the second tells multi-output scorers
    and cross-validators to dispatch correctly on multi-target Y (the
    default chemometric PLS case).

### Fixed

- **`PLS.fit` on data with missing values no longer raises
  `NotImplementedError`.** When NaN was detected in X or Y, the missing-
  data settings defaulted to `md_method="tsr"`, but TSR for PLS is
  raised as `NotImplementedError` in `_fit_nipals`. The default now
  routes to `md_method="nipals"`, which handles per-cell NaN directly
  via skipna sums inside the NIPALS iterations and was the intended
  behaviour all along. Surfaced by `check_estimator` when the new
  `allow_nan=True` tag let sklearn send NaN-containing data through to
  `PLS.fit`.

### Audit movement (`tools/check_estimator_audit.py`)

| Estimator | Before #393 | After #393 |
|---|---|---|
| `MCUVScaler` | 44/47 (94%) | 44/46 (96%) |
| `PCA(n_components=2)` | 38/47 (81%) | 38/46 (83%) |
| `PLS(n_components=2)` | 24/29 (83%) | 24/28 (86%) |

Total check counts drop by 1 each because `__sklearn_tags__` declares
`allow_nan=True`, so sklearn skips the "do you reject NaN?" check
(it's no longer the contract). Net behavioural change: the NaN-
through-Pipeline path that #391 / #393 unblock now actually works
on PLS too (not just MCUVScaler / PCA).

## [1.36.0] - 2026-06-07

### Changed

- **`MCUVScaler`, `PCA`, and `PLS` route input through
  `sklearn.utils.validation.validate_data`.** That helper, called from
  every `fit` / `predict` / `transform`, standardises sklearn-style
  input handling in one place:
  - Sets `n_features_in_` on `fit` and validates it on subsequent calls.
  - Captures `feature_names_in_` when the input is a DataFrame.
  - Rejects sparse / complex / object-dtype / empty input with the
    sklearn-standard error messages that `check_estimator` expects.
  - Preserves NaN tolerance (`ensure_all_finite="allow-nan"`) so the
    NIPALS path keeps working - the chemometric preprocessing contract
    threads missing data through the downstream estimator.
- **`MCUVScaler` mixin inheritance order flipped** from
  `(BaseEstimator, TransformerMixin)` to
  `(TransformerMixin, BaseEstimator)`. sklearn 1.6+ requires the more-
  specialised mixin first; without the flip the `check_estimator`
  audit aborts at setup with a `transformer_tags` `RuntimeError`. The
  fix belongs to #393 but is the only thing gating the audit from
  completing on `MCUVScaler` so it lands here.

### Audit movement (`tools/check_estimator_audit.py`)

| Estimator | Before | After |
|---|---|---|
| `MCUVScaler` | 18/29 (62%) | 44/47 (94%) |
| `PCA(n_components=2)` | 28/47 (60%) | 38/47 (81%) |
| `PLS(n_components=2)` | 19/29 (66%) | 24/29 (83%) |

Total checks rose because the mixin-order fix unlocked checks that
were previously skipped behind the setup error. Net new passes:
+26 on MCUVScaler, +10 on PCA, +5 on PLS.

Remaining failures are scoped to existing follow-up issues:
- `__sklearn_tags__` for `allow_nan` and `multi_output` declarations
  (#393).
- `PCA.predict` returning `Bunch` + `_LazyFrame` `__dict__` mutation
  (#396).
- `transform()` returning DataFrame instead of ndarray, blocking the
  transformer-dtype checks (#391).

## [1.35.0] - 2026-06-07

### Changed

- **`PLS.predict(X)` now returns the predicted Y as a `pd.DataFrame`**
  instead of a rich `Bunch(scores, hotellings_t2, spe, y_hat)`. This
  satisfies the scikit-learn `RegressorMixin` contract so a fitted
  `PLS` composes cleanly inside `Pipeline`, `cross_val_score`, and
  `GridSearchCV`. The rich diagnostic view moves to a new method:
  `PLS.diagnose(X) -> Bunch` (same fields as the pre-1.35.0
  `predict()` return).

  This is a **breaking change** for callers that read
  `predict(X).y_hat` / `predict(X).scores` / `predict(X).spe` /
  `predict(X).hotellings_t2`. The migration is mechanical:

  ```python
  # before
  result = pls.predict(X)
  y_hat = result.y_hat
  spe = result.spe

  # after
  y_hat = pls.predict(X)           # sklearn-compatible
  result = pls.diagnose(X)         # rich Bunch view
  spe = result.spe
  ```

  The internal `PLS.score(X, Y)` (sklearn's R² scorer) is unaffected.

### Added

- **sklearn `Pipeline` interop for `MCUVScaler` + `PCA` / `PLS`.**
  - `MCUVScaler.fit` and `MCUVScaler.transform` now accept (and ignore)
    an optional `y` keyword argument so the scaler can be a step in
    `sklearn.pipeline.Pipeline` (sklearn threads `y` through every
    step's `fit`).
  - `PLS.fit(X, Y)` accepts a 1-D `Y` (the shape sklearn passes for
    single-target regression) and promotes it to `(N, 1)` internally.
  - Together with the `predict()` shape change above, these unblock:
    `Pipeline([MCUVScaler(), PCA(n_components=k)]).fit_transform(X)`;
    `Pipeline([MCUVScaler(), PLS(n_components=k)]).fit(X, y).predict(X)`;
    `cross_val_score(pipe, X, y, cv=RepeatedKFold(...))`;
    `GridSearchCV(pipe, {"pls__n_components": ...}, cv=...).fit(X, y)`;
    `clone(pipe).fit(X, y)`.

## [1.34.0] - 2026-06-07

### Added

- **`PLS.nested_cv(X, Y, ...)`** classmethod: an honest, optimism-free
  performance estimate via nested cross-validation. The outer loop
  splits the data; the inner loop runs `PLS.select_n_components`
  (`inner_cv * n_inner_repeats` folds) on each outer-train to pick the
  component count for that fold; a final PLS is fit on the outer-train
  at that count and predicts the outer-test. Accumulated out-of-fold
  predictions give RMSEP that is *not* biased by the selection rule.
  - kwargs: `max_components`, `outer_cv` (default 5), `inner_cv`
    (default 5), `n_inner_repeats` (default 10), `selection_rule`
    (default `"1se"`), `scale_inside_folds`, `min_q2_increase`,
    `n_permutations`, `alpha`, `random_state`. The inner seed is
    offset per outer fold so every outer split sees a fresh inner
    shuffle.
  - Returned `Bunch`: `rmsep` (per-Y plus `"total"`),
    `q2y` (per-Y plus `"total"`), `cv_predictions` (on the original Y
    scale), `selected_components_per_fold` (list, one int per outer
    fold), `selected_components_distribution` (vote share, a stability
    signal across the outer folds).

## [1.33.0] - 2026-06-07

### Added

- **Stability-selection diagnostic on `PLS.select_n_components`**:
  when ``selection_rule`` is ``"1se"`` or ``"min"`` and ``n_repeats > 1``,
  the chosen rule is re-applied to each repeat's slice of per-fold RMSE
  values and the distribution of votes is reported. A concentrated
  distribution flags a confident recommendation; a flat or multi-modal
  one flags it for review (Meinshausen & Bühlmann 2010 stability
  selection, adapted from the variable-selection setting to the
  component-count setting).
  - Returned `Bunch` gains `selection_distribution` (per-component
    vote share, `pd.Series` indexed `1..A`), `selection_mode` (the
    most-voted count), and `selection_is_stable` (`True` iff the modal
    share meets ``stability_threshold``).
  - New ``stability_threshold`` kwarg (default `0.6`); fields are `None`
    when stability is not computed (single repeat, or rules that don't
    decompose cleanly per repeat: ``"q2_increment"`` /
    ``"randomization"``).

## [1.32.0] - 2026-06-07

### Added

- **`PLS.select_n_components(..., selection_rule="randomization")`**:
  Van der Voet's (1994, *Chemom. Intell. Lab. Syst.* 25(2):313-323)
  randomization test as a new fourth selection rule, joining `"1se"`,
  `"min"`, and `"q2_increment"`.
  - For each candidate model, the paired squared-residual differences
    against the argmin-RMSECV reference model are randomly sign-flipped
    `n_permutations` (default 999) times; the right-tail *p*-value is
    the share of permutation sums at least as large as observed (with
    the standard `+1`/`+1` correction so *p* is strictly positive).
  - The smallest component count whose *p*-value exceeds `alpha`
    (default 0.01, matching R's `pls::selectNcomp`) is recommended -
    statistically indistinguishable from the best, but more parsimonious.
  - Smaller `alpha` is more parsimonious; the reference model itself has
    `p = 1` by construction.
  - Returned `Bunch` gains `randomization_pvalues` (`pd.Series`,
    indexed 1..A) when the rule is selected, `None` otherwise.

## [1.31.0] - 2026-06-07

### Added

- **`PCA.minka_mle(X)`** classmethod: Minka (2000) automatic-dimensionality
  estimate via the PPCA evidence (Laplace approximation on the covariance
  eigenvalues). Cheap closed-form cross-check for the ekf-CV
  recommendation. Mean-centres `X` internally but does *not* unit-variance
  scale, because the MLE's eigenvalue model misreads scaled noise as
  additional signal.
- **`PCA.parallel_analysis(X)`** classmethod: Horn (1965) parallel
  analysis. Generates `n_simulations` random matrices of the same shape,
  retains every observed component whose eigenvalue exceeds the
  `quantile` (default 95th-percentile) of the null distribution at the
  same rank. Returns a Bunch with the observed eigenvalues and null
  threshold so callers can inspect the spectrum.
- **`PCA.select_n_components(..., return_consensus=True)`**: alongside
  the ekf recommendation, also reports Minka MLE and Horn parallel
  analysis counts on the returned Bunch as `minka_n_components`,
  `parallel_analysis_n_components`, `consensus_counts` (3-tuple) and
  `consensus` (`"agree"` when all three counts sit within 1 of each other,
  `"disagree"` otherwise). A clean low-rank dataset typically returns
  `consensus="agree"`; downstream tooling can use that as a high-
  confidence signal and route disagreements to human review.

## [1.30.0] - 2026-06-07

### Changed

- **`PCA.select_n_components` (under `cv_scheme="ekf"`) gains repeated-CV
  averaging and in-fold scaling**, finishing the port of PR #376's PLS CV
  upgrades to PCA.
  - New `n_repeats` kwarg (default `1`). Each repeat re-shuffles the
    element-fold permutation and contributes a fresh batch of
    per-fold PRESS columns; the pooled PRESS is averaged across repeats
    so it stays on the per-cell scale. `n_repeats > 1` narrows the
    per-component standard error (helpful on borderline 1-SE
    selections) at roughly linear extra runtime.
  - New `scale_inside_folds` kwarg (default `True`). Mean-centring and
    unit-variance scaling constants are now fit per-column on each
    fold's in-fold cells and applied to the whole matrix before EM,
    removing the centring/scaling leakage of the prior implementation.
    Set to `False` to reproduce the previous behaviour (the column mean
    is recomputed each EM iteration from the imputed matrix; no unit-
    variance scaling).
  - Callers who relied on the prior contract ("X must be pre-scaled,
    EM recomputes centring each iteration") can opt in via
    `scale_inside_folds=False`; the pre-existing scaled-input flow keeps
    working without code changes because per-fold MCUVScaler on already-
    centred-and-scaled data is approximately a no-op.

## [1.29.0] - 2026-06-07

### Changed

- **`PCA.select_n_components` rebuilt around the element-wise k-fold (ekf)
  cross-validation algorithm** of Bro, Kjeldahl, Smilde & Kiers (2008,
  *Anal. Bioanal. Chem.* 390:1241-1251). The legacy row-wise scheme it
  replaces suffered from the *trivial-fit* problem: holding out whole
  rows and projecting them back via `transform()` lets the held-out
  row's own values reach its own prediction, so PRESS shrunk
  monotonically with the component count and the recommendation tended
  to run to the maximum.
  - New `cv_scheme` kwarg: `"ekf"` (default) holds out individual cells
    of `X` and predicts them via EM-style imputation from a model that
    never sees their true values - the prediction-independence the
    row-wise scheme violates. `"row_wise"` is preserved for back-compat
    but emits a `SpecificationWarning`.
  - New `selection_rule` kwarg (`"min"` (default, GlobalMin per Bro
    2008), `"1se"`, `"q2_increment"`), mirroring `PLS.select_n_components`.
  - New `min_q2_increase`, `n_iter`, `tol`, `random_state` kwargs;
    `threshold` (the legacy Wold PRESS-ratio cutoff) is deprecated and
    ignored, emitting a `DeprecationWarning`.
  - The returned `Bunch` gains `per_fold_press`, `se_press`,
    `cv_scheme`, and `selection_rule` fields; the existing `press`,
    `press_ratio`, `q2`, `cv_scores` keys keep their semantics (under
    ekf, `cv_scores` aliases `per_fold_press`). The Q² normalisation
    switches from mean-cell SS to total SS so it stays comparable to
    `r2_cumulative_`.

  Callers who relied on the old recommendation can opt in via
  `cv_scheme="row_wise"` (which warns).

## [1.28.0] - 2026-06-07

### Changed

- **`PLS.select_n_components` rebuilt around the one-standard-error rule on
  repeated, shuffled K-fold CV with in-fold scaling**, replacing the
  pre-1.28 argmin-RMSECV rule (which routinely ran to the maximum
  component count because the validated error keeps drifting down by
  noise-level amounts past the systematic components).
  - New `selection_rule` kwarg: `"1se"` (default; Breiman/Friedman/
    Olshen/Stone 1984; Hastie/Tibshirani/Friedman *ESL* sec.7.10),
    `"min"` (the pre-1.28 default, preserved for explicit opt-in), and
    `"q2_increment"` (a Wold's-R-style cumulative-:math:`Q^2` threshold).
  - New `n_repeats` kwarg (default `10` when `cv` is an `int`) wires up
    `sklearn.model_selection.RepeatedKFold` so the 1-SE rule's standard
    error is estimated across `n_splits * n_repeats` fold errors;
    `random_state` makes that selection reproducible.
  - New `scale_inside_folds` kwarg (default `True`) fits `MCUVScaler` on
    each training fold and inverse-transforms predictions to the
    original Y scale, removing the centring/scaling leakage of the prior
    implementation. Passing `False` emits a `SpecificationWarning`.
  - The returned `Bunch` gains `per_fold_rmsecv`, `se_rmsecv` and
    `selection_rule` fields; the existing fields keep their semantics.

  Callers who relied on the old argmin behaviour can opt in via
  `selection_rule="min"`; callers who relied on the pre-1.28 leaky
  scaling can opt in via `scale_inside_folds=False` (which warns).

- **`PCA.select_n_components` docstring now flags the trivial-fit
  limitation** (Bro, Kjeldahl, Smilde & Kiers 2008, *Anal. Bioanal. Chem.*
  390:1241-1251): the row-wise CV scheme makes PRESS shrink monotonically
  with components, so the recommendation tends to over-select. A
  proper element-wise k-fold (ekf) rewrite is tracked as a follow-up; for
  now the docstring recommends cross-checking with parallel analysis or
  Minka's MLE.

## [1.27.1] - 2026-06-06

### Fixed

- **PyPI publish workflow (`publish.yml`).** The CycloneDX SBOM step called
  `cyclonedx-py environment --outfile ...`, but `cyclonedx-bom` 6+ renamed that
  flag, so the step failed (`unrecognized arguments: --outfile`) and blocked
  every release. It now uses `--output-file`. CI/tooling only; no library
  changes from 1.27.0.

## [1.27.0] - 2026-06-06

### Added

- Canonical per-variable MSPC contributions for PCA and PLS:
  `t2_contributions(model, X)` and `spe_contributions(model, X)`, also available
  as methods (`model.t2_contributions(X)` / `model.spe_contributions(X)`).
  `t2_contributions` decomposes each observation's Hotelling's T2 onto the
  variables (signed; row sums equal the observation's T2, with an optional
  1-based `components` subset), and `spe_contributions` returns the signed
  per-variable residuals whose squares sum to the observation's SPE. Both work
  for PCA (SVD and NIPALS) and PLS and reproduce the model's stored
  `hotellings_t2_` / `spe_` when passed the training data.

## [1.26.1] - 2026-06-06

### Added

- **`CODE_OF_CONDUCT.md`.** Adopts the Contributor Covenant 2.1, with the
  maintainer as the enforcement contact, and is referenced from
  `CONTRIBUTING.md`. Rounds out the community-health files alongside the new
  `SECURITY.md`.
- **PEP 561 `py.typed` marker (SEC-36).** The package is fully type-annotated
  and `mypy src/process_improve` runs as a blocking CI check, but the
  distribution shipped no `py.typed` marker, so downstream type-checkers
  (mypy / pyright) silently treated `process-improve` as untyped. The marker is
  now bundled in the wheel and consumers get the published annotations.
- **Security disclosure policy (`SECURITY.md`, SEC-37).** Documents the
  supported version, the private vulnerability-reporting channels (GitHub
  private advisories and email), the expected response timeline, and the
  threat-model scope. Complements the existing `SECURITY_AUDIT.md` catalogue.

### Fixed

- **Malformed `items_to_highlight` keys now raise a clear error in the
  multivariate plots (SEC-34).** `score_plot`, `spe_plot`, and `t2_plot` parsed
  each highlight key with a bare `json.loads`, so a non-JSON key surfaced as a
  confusing `json.JSONDecodeError` deep inside the trace loop. They now decode
  via a shared helper that raises a clear `ValueError` at the API surface,
  matching the SEC-32 guard already in `batch.plotting`.
- **`ControlChart.calculate_limits` no longer accepts arbitrary keyword
  arguments (SEC-35).** A blanket `setattr(self, key, val)` over `**kwargs` let
  a caller silently overwrite internal state (`s`, `target`, `train_samples`,
  even a bound method) and swallowed typos. Only the documented Holt-Winters
  smoothing lambdas (`ld_1`, `ld_2`) are accepted now; any other keyword raises
  a clear `ValueError`.

## [1.26.0] - 2026-06-06

### Added

- `scale()` (`multivariate`) now takes an explicit `ddof` argument. It is
  forwarded to `np.std` when the default scaling function is used, so callers
  can request the sample standard deviation with `scale(center(X), ddof=1)`
  (matching `MCUVScaler`) instead of relying on `**kwargs`.
- `PCA.select_n_components` now returns a `q2` field: the cross-validated R2 of
  X per component count (mean-cell PRESS normalised by the null-model
  sum-of-squares). This mirrors `PLS.select_n_components`'s `r2y_validated` and
  saves callers from normalising `press` by hand.
- New `NotEnoughVarianceError` exception (exported from
  `process_improve.multivariate`). It is raised by PCA / PLS NIPALS fitting when
  the data runs out of variance before the requested number of components is
  reached. It subclasses `RuntimeError`, so existing `except RuntimeError`
  handlers keep working while callers can catch the narrower type.

### Fixed

- `scale()` no longer produces `inf` / `NaN` for constant (zero-variance)
  columns; such columns are left unchanged, matching `MCUVScaler`.
- `scale()`'s docstring no longer claims it divides by N-1 by default. The
  default (`ddof=0`) divides by N; the docstring now documents this and points
  to `MCUVScaler` / `ddof=1` for the sample standard deviation.

### Changed

- The `cv` parameter of `PCA.select_n_components` and
  `PLS.select_n_components` is now type-annotated as
  `int | BaseCrossValidator`, matching the documented (and already working)
  behaviour of passing an sklearn splitter object.

## [1.25.1] - 2026-06-05

### Fixed

- **Docstring/implementation mismatches across the public API (docs-only).**
  - `univariate.metrics.median_absolute_deviation`: docstring claimed the
    default for `scale` was `1.0`, but the signature default is `"normal"`.
    The docstring now documents the actual default and what the normal
    consistency factor does.
  - `experiments.structures.c`: docstring said the helper returns a
    `DataFrame` and converts every entry to a float. It actually returns a
    `Column` (a `pandas.Series` subclass) and only coerces numeric entries;
    categorical entries (via `levels=` or when `float()` raises) are kept
    as-is.
  - `multivariate._mbpls.MBPLS.predict`: docstring listed only 3 of the 5
    fields returned in the result `Bunch`. The full field list now
    documents `block_spe` and the cumulative `hotellings_t2` series.
  - `multivariate._tpls.TPLS.predict`: docstring described the return as
    "dict / Returns an array of prediction objects. More details to come
    here later." The method actually returns a `sklearn.utils.Bunch` with
    four well-defined fields (`hat`, `t_scores_super`, `spe`,
    `hotellings_t2`); they are now described.
  - `experiments._lm.Model.summary`: docstring said "Side effect: prints to
    the screen", but the method only returns the statsmodels summary
    instance and never prints. Replaced with an accurate description that
    also notes the unused `alpha` / `print_to_screen` arguments are kept
    for backwards compatibility.

## [1.25.0] - 2026-06-03

### Changed

- **TPLS D-block scaling now matches Garcia-Munoz (2014), section 2.1 (#192).** Each
  D-block (material properties) is block-scaled by `1/sqrt(P_i * M_i)` (P_i = number of
  lots/rows, M_i = number of properties/columns) instead of the previous `1/sqrt(M_i)`.
  After column auto-scaling this makes `trace(X_i^T X_i) ~= 1` for every block, removing
  bias toward blocks that simply have more lots or properties. The previous factor left
  `trace = P_i - 1` (e.g. 161 vs 8 across blocks on the pyphi example), over-weighting
  large blocks. **This changes all fitted TPLS results** (scores, R2, SPE, Hotelling's
  T2, limits, predictions); affected reference values in the test-suite were re-baselined
  and an independent `trace ~= 1` invariant test was added.

### Added

- **TPLS `vip(method="deflated")` (#192).** Opt-in deflated direct-weights feature
  importance: VIP computed on the rotated weights `S(V^T S)^-1` (D-block) and
  `P(P^T P)^-1` (F-block), which account for the deflation across components (equation 7
  in the paper). The default `method="vip"` is unchanged and still matches the standard
  VIP the paper reports. Note that for the D-block the rotation is ~identity
  (`V^T S ~= I` by construction), so the deflated and default importances coincide there;
  they differ for the F-block.
- **TPLS `d_block_scaling_` accessor (#192).** The D-block block-scaling factor is now stored as a
  plain scalar (previously a one-element `pd.Series` accessed via `[0]`) and exposed through the
  read-only `d_block_scaling_` property, returning `{group: factor}`.

## [1.24.34] - 2026-06-03

### Changed (internal)

- **#192 (partial)**: TPLS housekeeping. Removed a stale `TODO:` scratch block
  that was leaking into `TPLS.help()` output (the attributes it referenced are
  now methods, per ENG-05) and added a `.vip()` line in its place. Added test
  coverage for the previously-untested `feature_importance` / `vip()` surface
  (structure, finite non-negative VIPs, feature-name indexing, the `vip()`
  accessor and its error cases). This pins the current VIP-based behaviour so the
  open question in #192 - whether the D/F feature importance should instead use
  the deflated matrices `S(VᵀS)⁻¹` / `P(PᵀP)⁻¹` - can be settled as a deliberate,
  reviewed change. The scaling-vector and deflated-matrix items of #192 remain
  open.

## [1.24.33] - 2026-06-03

### Fixed

- **#197**: the DTW batch-alignment averaging (`batch.preprocessing.align_with_path`)
  no longer contaminates the first synced row. When several batch samples map to
  the same reference index (a compression in the warping path), the synced value
  is the average of those batch samples; the running accumulator was previously
  seeded from an `initial_row` argument - a reference row in one caller, an
  out-of-space batch row in the other - which mixed an unrelated row into the
  row-0 average. The accumulator is now seeded from the first batch sample for
  the index (matching the value already written to that row), and the misleading
  `initial_row` parameter is removed. (The thesis page-181 generalisation of the
  averaging, plus the Sakoe-Chiba and user-`band` constraints, remain tracked on
  #197.)

## [1.24.32] - 2026-06-03

### Changed (internal)

- **#195**: PCA and PLS NIPALS now seed the iteration from the highest-variance
  (highest sum-of-squares) column instead of the arbitrary first column. NIPALS
  converges to the same component for any non-degenerate seed and a
  deterministic sign convention fixes the sign, so the fitted result is
  unchanged (the full multivariate suite, including the reference-dataset and
  property-based invariant tests, is byte-for-byte green). The benefit is purely
  numerical: the highest-variance column is the best-conditioned seed, needing
  fewer iterations and avoiding the near-degenerate start when the first column
  carries little or no variance. This resolves the last open code item in #195.

## [1.24.31] - 2026-06-03

### Fixed

- **#195**: PCA and PLS now validate feature-name consistency between `fit` and
  `transform` / `predict`, not just the column count. Previously a
  correctly-shaped frame with reordered or renamed columns was projected
  positionally (PCA) or silently label-aligned to all-`NaN` (PLS
  `X @ direct_weights_`), producing wrong results with no error. Now: columns
  supplied in a different order are realigned to the training order; renamed or
  unexpected columns raise a clear `ValueError`; and unnamed (ndarray) input is
  taken positionally and works (in particular `PLS.transform` on an ndarray no
  longer returns all-`NaN`). `PLS.transform` also gained the column-count check
  the other paths already had. Models fitted from unnamed (ndarray) data are
  unaffected (count-only validation, as before).

## [1.24.30] - 2026-06-03

### Changed (internal)

- **ENG-25 (#307)**: chip away the in-line complexity suppressions. Refactored
  `_select_screening_design` (`experiments/strategy/engine.py`) - which carried
  `# noqa: C901, PLR0912, PLR0915` - into four focused helpers
  (`_mixture_screening_stage`, `_factorial_screening_stage`,
  `_large_factor_screening_choice`, `_screening_design_params`) and removed a
  dead local `reasoning` list (the user-facing reasoning is built separately by
  `_build_reasoning`). The suppression is gone and the decision logic is
  unchanged; suppression count drops from 68 to 67.

## [1.24.29] - 2026-06-03

### Fixed

- **#343**: `DataFrameDict` (multivariate T-PLS container) now compares by value.
  It subclasses `dict` but stores all data in `self.datadict`, leaving the `dict`
  base empty, so the inherited `dict.__eq__` / `dict.__ne__` reported every
  instance as equal regardless of its contents (CodeQL `py/missing-equals`). It
  now defines value-based `__eq__` / `__ne__` (comparing block structure and each
  contained `DataFrame` via `DataFrame.equals`) and is explicitly unhashable.
  Surfaced by CodeQL during the ENG-01 module split (#342).

## [1.24.28] - 2026-06-03

### Added

- **ENG-15 (#297)**: an MCP-boundary fuzz suite (`tests/fuzz/test_mcp_boundary.py`).
  It derives a `hypothesis` strategy from every registered tool's `input_schema`
  and asserts that `execute_tool_call` never leaks an undocumented exception (the
  SEC-14..21 invariant); it is deterministic (`derandomize=True`) so it is a
  stable CI gate. It surfaced and **fixed two real boundary leaks**: `control_chart`
  raised `AssertionError` on constant/degenerate data (the `assert`s are now a
  `ValueError`, which also closes the `python -O` strip-the-assert gap from
  SEC-17), and the multivariate tools could leak `RuntimeError`/`IndexError` on
  degenerate input (their boundary `except` clauses now catch these and return an
  `{"error": ...}` response).

## [1.24.27] - 2026-06-03

### Changed (internal)

- **CI**: remove the free-threaded (`3.13t` / `3.14t`) test matrix rows. They
  only ever failed at install - several dependencies still lack free-threaded
  (cp31Xt) wheels and fall back to source builds that fail on the GitHub images
  (llvmlite needs LLVM 20; Rust-built deps fail their maturin/cargo builds) - so
  they produced a permanent non-blocking red check with no useful signal. They
  can be re-added when the free-threaded wheel ecosystem catches up (issue #43).

## [1.24.26] - 2026-06-03

### Changed (internal)

- **ENG-03 / ENG-20 (#285, #302)**: drive the mypy backlog to zero and make the
  CI ``typecheck`` job blocking. The historical ~328-error backlog across 36
  modules (multivariate, experiments, batch, monitoring, univariate,
  visualization, regression) was cleared with behaviour-preserving, type-only
  fixes - no ``Any`` annotations and only two documented pandas-stubs
  ``# type: ignore`` lines. ``mypy src/process_improve`` now reports zero errors,
  so the ``typecheck`` job dropped ``continue-on-error`` and a new type error
  now turns CI red. mypy is upper-bounded (``<2.2``) in the dev deps so a future
  mypy release cannot silently break the gate. No public API or runtime change;
  the full suite passes at 91% coverage.

## [1.24.24] - 2026-06-03

### Added

- **ENG-19 (#301)**: a "Scaling and memory" docs page (``docs/scaling.rst``)
  documenting the estimators' in-memory assumption, how to estimate RAM
  (~``rows x cols x 8`` bytes, with 2-4x headroom during ``fit``), what to do
  for larger-than-RAM data, and the out-of-core roadmap. The README's
  "production-grade" claim now carries an honest note about scale. The
  out-of-core code path itself is deferred (demand-driven) per the issue.

## [1.24.23] - 2026-06-03

### Changed (internal)

- **ENG-14 (#296)**: adopt a ``src/`` layout. The package moved from
  ``process_improve/`` to ``src/process_improve/`` (no import changes - it is
  still imported as ``process_improve``), so running Python from the repo root
  imports the installed package rather than accidentally shadowing it with the
  source tree. The ``notebooks_examples/`` folder moved out of the package to a
  top-level ``examples/`` directory and is no longer shipped in the wheel.
  Tooling (uv build backend, mypy/ruff excludes, coverage globs, CI mypy target,
  docs autodoc path, and repo-relative test dataset paths) was updated
  accordingly. No public API or behavioural change.

## [1.24.22] - 2026-06-03

### Changed

- **ENG-30 (#312)**: the MCP server no longer blocks its event loop on tool
  execution. The FastMCP handler now offloads the synchronous tool call (the
  ``ProcessPoolExecutor`` future in safe mode, or the in-process call otherwise)
  to a worker thread via ``loop.run_in_executor``, so a slow tool call no longer
  serialises other calls when the server is fronted by HTTP / SSE. Single-call
  behaviour is unchanged; a regression test demonstrates concurrent dispatch.

## [1.24.21] - 2026-06-03

### Added

- **ENG-26 (#308)**: documentation - an architecture overview
  (``docs/architecture.rst``: package layout, conventions, the multivariate
  estimator stack, and the MCP tool layer), linked from the top of
  ``README.md``, and a step-by-step tool-authoring guide
  (``docs/development/tool_authoring.rst``). Complements the existing
  error-handling, reproducibility, deprecation-policy, and logging policy pages.

## [1.24.20] - 2026-06-03

### Added

- **ENG-12 (#294)**: diagnostic logging for the modules that do real work.
  Module-level loggers (``logging.getLogger(__name__)``) were added to the
  multivariate estimators (PCA / PLS / TPLS / MBPLS / MBPCA and the shared
  NIPALS inner loop), the experiments ``analysis`` / ``evaluate`` /
  ``optimization`` modules, the batch DTW alignment, and the monitoring control
  charts. Long-running algorithms emit a ``logger.debug`` per major step (e.g.
  NIPALS per-component convergence, DTW per-iteration norm). The library emits
  records but never configures handlers, so output stays silent by default. A
  new docs page (``docs/development/logging.rst``) explains how to enable
  verbose logging.

## [1.24.19] - 2026-06-03

### Changed

- **ENG-18 (#300)**: PCA and PLS now store their hot-path fitted attributes
  (PCA ``scores_`` / ``loadings_`` / ``spe_``; PLS ``scores_`` / ``spe_`` /
  ``x_loadings_`` / ``x_weights_``) as private numpy ndarrays, exposing the
  public ``pd.DataFrame`` as a lazily-built, cached view (a ``_LazyFrame``
  descriptor on the shared base). Internal math reads the ndarrays directly,
  avoiding a ``DataFrame.values`` conversion on every ``transform`` / ``predict``
  / ``score_contributions`` call; the cache is excluded from pickling, so a
  pickled model stores only the ndarrays and rebuilds the views on load. The
  public DataFrame views are byte-identical (values, index, columns, dtype), and
  ``check_is_fitted`` / ``NotFittedError`` behaviour is unchanged. A perf
  baseline is added under ``tests/perf/``.

## [1.24.18] - 2026-06-03

### Changed (internal)

- **ENG-17 (#299)**: factor the duplicated latent-variable scaffolding into a
  shared base. New ``multivariate/_base.py`` provides ``_LatentVariableModel``
  (the ENG-05 convenience methods + ``ellipse_coordinates``), a
  ``_HotellingsT2LimitMixin`` (the ``hotellings_t2_limit`` shared by
  PCA/PLS/MBPLS/MBPCA), and a ``_RenameGetattrMixin`` (the migration
  ``__getattr__`` driven by a per-class ``_ATTRIBUTE_RENAMES`` map). PCA and PLS
  inherit ``_LatentVariableModel``; MBPLS/MBPCA inherit only the T2-limit mixin;
  TPLS is unchanged (it reads differently-named fitted attributes). No public
  API change: each class's public method set is byte-identical, and the fit /
  predict / transform algorithms are untouched.

## [1.24.17] - 2026-06-03

### Changed

- **ENG-07 (#289)**: enforce sklearn API compatibility without coupling to a
  concrete sklearn estimator. ``PLS`` keeps the sklearn mixins
  (``BaseEstimator`` / ``TransformerMixin`` / ``RegressorMixin``) but does not
  inherit ``sklearn.cross_decomposition.PLSRegression``; a regression test locks
  this in across PCA / PLS / TPLS / MBPLS / MBPCA, and numerical-consistency
  tests cross-check our PCA against ``sklearn.decomposition.PCA`` and our PLS
  against ``PLSRegression``. ``PLS.has_missing_data_`` is now set in ``fit()``
  rather than ``__init__`` (sklearn requires ``__init__`` to set only the
  constructor parameters). No public API change.

## [1.24.16] - 2026-06-03

### Changed

- **ENG-05 (#287)**: the multivariate estimators (PCA, PLS, TPLS, MBPLS, MBPCA)
  now expose their convenience callables (``score_plot``, ``spe_limit``,
  ``vip``, ``hotellings_t2_limit``, ``ellipse_coordinates``, ...) as real
  methods instead of ``functools.partial`` instances bound in ``fit``. As a
  result ``help(model.score_plot)`` shows the underlying docstring,
  ``inspect.signature`` reports the true parameters (minus ``model``), IDEs can
  autocomplete, the methods are overridable by subclasses, and fitted models
  pickle robustly. Call sites are unchanged. The standalone functions remain
  available for advanced callers, and TPLS's ``spe_limit`` dict-of-callables
  API is unchanged.

## [1.24.15] - 2026-06-03

### Changed (internal)

- **ENG-02 (#344)**: split the two large ``experiments`` god modules into
  focused submodules. ``analysis.py``'s 13 ``_run_*`` analyses (and the two
  ``_compute_*`` summary helpers) moved to ``experiments/_analyses/``
  (``ols_extractors``, ``diagnostics``, ``lack_of_fit``, ``curvature``,
  ``model_selection``, ``box_cox``, ``lenth``, ``prediction``, ``_shared``);
  ``analysis.py`` keeps ``build_formula``, ``AnalysisResult`` and the
  ``analyze_experiment`` dispatcher. ``tools.py``'s 10 MCP tool wrappers moved
  to ``experiments/_tools/<tool>.py`` with shared registration primitives in
  ``experiments/_tools/__init__.py``; ``tools.py`` is now a thin aggregator.
  No public API change: ``analyze_experiment``, ``build_formula`` and
  ``get_experiments_tool_specs`` remain importable from their original paths,
  and the MCP tool-spec output is byte-identical.

## [1.24.14] - 2026-06-02

### Changed (internal)

- **ENG-01 (#342)**: split the 6,210-line
  ``process_improve/multivariate/_pca_pls.py`` into focused submodules
  (``_common``, ``_preprocessing``, ``_nipals``, ``_limits``,
  ``_diagnostics``, ``_pca``, ``_pls``, ``_tpls``, ``_mbpls``, ``_mbpca``,
  ``_resampling``). ``methods.py`` is now the canonical re-exporter that
  aggregates these submodules, and ``_pca_pls.py`` is a thin
  backward-compatibility shim. No public API change: every name remains
  importable from ``process_improve.multivariate`` and
  ``process_improve.multivariate.methods`` as before. The ``Plot`` accessor
  moved to ``plots.py``, and several blocks of long-dead commented-out code
  were removed along the way.

## [1.24.13] - 2026-06-02

### Changed (internal)

- **ENG-25 (#307) chip-away**: two internal helpers now take a bundled
  options dataclass instead of nine / six loose arguments, which
  removes their ``noqa: PLR0913`` suppressions. Net: 2 noqa
  suppressions deleted, one new internal dataclass per helper. No
  public API change.
  - ``process_improve/experiments/evaluate.py::_build_context`` now
    takes a single ``_EvalRequest`` (9 args -> 1).
  - ``process_improve/experiments/designs_optimal.py::_run_pyoptex``
    now takes an optional ``_PyoptexOptions`` (6 args -> 4).

  Combined with PR #340's ``find_elbow_point`` slim-down (-1 noqa),
  the running ENG-25 count is now **65** suppressions, down from
  68 at the start of this release window. Per the issue policy
  ("each release lowers the count"), this is on track.

## [1.24.12] - 2026-06-02

### Changed (internal)

- **ENG-23 (#305)**: the four ambiguously-named ``methods.py`` /
  ``models.py`` files are renamed to domain-specific paths so
  filename-ranked tooling (Jump-to-File, fuzzy search, codecov
  reports) is no longer ambiguous about which file is being shown:
  - ``multivariate/methods.py``  → ``multivariate/_pca_pls.py``
  - ``regression/methods.py``    → ``regression/_robust_regression.py``
  - ``bivariate/methods.py``     → ``bivariate/_elbow_peak.py``
  - ``experiments/models.py``    → ``experiments/_lm.py``

  Each original path remains as a thin re-exporter that does
  ``from ._new_name import *``, so every public import keeps
  working (``from process_improve.multivariate.methods import PCA``
  etc.). Only one in-tree test imported a private underscore-
  prefixed helper from the old path (``_nz`` from
  ``multivariate.methods``); it is now imported from
  ``multivariate._pca_pls`` directly. External callers that
  reached into private names by name would need the same update.

## [1.24.11] - 2026-06-02

### Changed (breaking: install footprint)

- **ENG-13 (#295)**: the runtime dependency closure is trimmed from
  16 packages to 8. The core install (``pip install process-improve``)
  now pulls in only `numpy`, `pandas`, `scikit-learn`, `statsmodels`,
  `patsy`, `pydantic`, `pyyaml`, and `tqdm`. Heavier optional
  surfaces (plotting, designed experiments, batch IO, MCP server,
  numba JIT) live in extras. The new ``[all]`` meta extra reproduces
  the pre-1.24.11 install closure.

  | Extra        | Packages                                                       |
  |--------------|----------------------------------------------------------------|
  | `plotting`   | matplotlib, plotly, seaborn, ridgeplot                         |
  | `expt`       | pyDOE3                                                         |
  | `batch`      | openpyxl, scikit-image                                         |
  | `mcp`        | mcp                                                            |
  | `fast`       | numba                                                          |
  | `all`        | union of the above                                             |

  Every top-level import of an extra-only package is gated with a
  ``try / except ImportError`` that swaps in a ``_MissingExtra``
  stand-in (``process_improve._extras``); any actual attribute
  access then raises an ImportError whose message tells the user
  which extra to install. ``numba``'s ``@jit`` decorator falls back
  to a no-op so ``process_improve.batch.alignment_helpers`` still
  imports and runs (as plain Python) without the ``[fast]`` extra.

  CI keeps using ``uv sync --dev --all-extras`` so the test suite
  exercises the same dependency closure as before. Existing users
  who relied on the implicit fat install should run
  ``pip install 'process-improve[all]'`` to keep current behaviour.

### Documentation

- README installation section documents the extras matrix.

## [1.24.10] - 2026-06-02

### Changed (release infrastructure)

- **ENG-21 (#303)**: the PyPI publish workflow is now **manually
  gated**. It runs only when a tag matching `v*` is pushed, or when a
  maintainer triggers `workflow_dispatch` from the Actions tab. A typo
  in `pyproject.toml` on `main` can no longer ship a release on its
  own. The maintainer's release procedure is now: (1) bump `version`
  in a PR, merge it; (2) push the matching `v<version>` tag.
- The job verifies that the pushed tag matches `pyproject.toml`'s
  declared version before any build step runs; mismatches fail the
  workflow loudly.
- `release.yml` (the auto-tag-on-version-bump workflow) is retired;
  `publish.yml` is now the single source of truth for PyPI publishing
  and GitHub release creation.

### Security

- **ENG-21 (#303)**: PyPI uploads now ship a **sigstore attestation**
  (PEP 740 / "trusted publishing") via
  `pypa/gh-action-pypi-publish@release/v1` with `attestations: true`.
- A **CycloneDX SBOM** (`sbom-<version>.cdx.json`) is built from the
  release's runtime dependency closure and attached to the GitHub
  release page alongside the wheel / sdist links.

### Documentation

- The GitHub release page now mirrors the matching `CHANGELOG.md`
  section (extracted between the `## [X.Y.Z]` heading and the next
  `## [` heading); auto-generated notes are used as a fallback only.
- CLAUDE.md updated to document the new manual-gate release flow.

## [1.24.9] - 2026-06-02

### Tests

- **SEC-33 (#282)**: pin regression tests for the four numerical /
  correctness sub-items already fixed in code:
  - Sub-item 1 -- ``terminate_check`` uses ``>=`` so NIPALS runs
    exactly ``md_max_iter`` iterations.
  - Sub-item 2 -- ``spe_calculation`` guards the ``np.var(neg)``
    sign-flip check with ``neg.size > 0`` (no RuntimeWarning, no
    silent NaN).
  - Sub-item 5 -- ``_optimize_desirability`` accepts a public
    ``random_state`` parameter (the hard-coded seed-42 was moved
    onto the function signature, resolved via ``check_random_state``).
  - Sub-item 6 -- ``BasePlot._get_factor_names`` cross-references
    formula tokens against the supplied ``design_data`` instead of
    the previous static ``{"I", "np", "power"}`` blocklist; common
    statsmodels transforms (Q, center, standardize, ...) are now
    correctly filtered.

  Sub-items 3 (``norm(t)*norm(u)`` eps-snap) and 4 (``arccos`` clamp)
  are kept open as cross-references in #195 and #206 respectively, per
  the SEC-33 issue's own deferral plan; their code-level fixes are
  already in place and are exercised indirectly by the existing PLS
  permutation and find-elbow tests.

## [1.24.8] - 2026-06-02

### Security

- **SEC-25 (#274)**: `pca_predict` and `pls_predict` now validate every
  array-like sub-field of the incoming `model_params` dict against
  `settings.max_matrix_rows` / `settings.max_matrix_cols` **before** any
  `np.array(...)` allocation. Previously, the `model_params` schema was
  `{type: object}` with no inner shape; an attacker-controlled
  `loadings=[[0]*10**6]*10**6` would allocation-bomb numpy before any
  later check could fire.

  The new internal helper `_validate_model_params` caps the 2-D keys
  (loadings, weights, beta-coefficients) via the existing
  `_validate_matrix_shape`, length-caps the 1-D keys (means, stds,
  spe_values, scaling factors), and bounds the scalar `n_components` /
  `n_samples` against `settings.max_matrix_cols` /
  `settings.max_matrix_rows`. A non-integer or negative scalar is
  rejected outright.

### Tests

- `tests/test_sec19_dos_caps.py` adds a `TestModelParamsCaps` class
  with six regression tests covering each of: oversize loadings rows,
  oversize loadings columns, oversize 1-D arrays, oversize
  `n_components`, negative `n_samples`, and PLS-specific
  `x_loadings` caps.

## [1.24.7] - 2026-06-02

### Documentation

- **ENG-24 (#306)**: documented the `pi_` prefix used on `Expt` /
  `Column` metadata attributes. The prefix stands for "process-improve"
  and namespaces library-managed metadata so it cannot collide with
  caller-supplied DataFrame column names. CONTRIBUTING.md gains a
  "Naming conventions" subsection covering the prefix and the `Expt`
  abbreviation; the `Expt` class docstring now lists the pinned
  ``_metadata`` fields and points at CONTRIBUTING.md.

## [1.24.6] - 2026-06-02

### Fixed

- **SEC-13 (#261)**: `find_reference_batch` no longer enters an unbounded
  cutoff-relaxation loop when the caller requests more reference batches
  than candidates exist. The loop is now bounded at `conf_level <= 0.95`
  and the function validates `number_of_reference_batches` against the
  candidate count up front. The previous behaviour eventually tripped a
  `python -O`-strippable `assert conf_level < 1.0` inside
  `spe_calculation`; the fix raises a clear `ValueError` instead.

### Tests

- `tests/batch/test_preprocessing.py` adds regression guards for the two
  newly-validated error paths (`number_of_reference_batches > len(batches)`
  and `number_of_reference_batches < 1`).

## [1.24.5] - 2026-06-02

### Tests

- Add regression tests pinning the existing fixes for seven open audit
  findings that already shipped code-level mitigations but lacked dedicated
  tests:
  - **SEC-27 (#276)**: `_parse_term` accepts both `I(A ** 2)` and the
    `np.power(A, 2)` / `power(A, 2)` forms emitted by newer statsmodels.
  - **SEC-28 (#277)**: `draw_initial_seed` returns at least 63 bits of
    entropy (sampled via `secrets.randbits(63)`).
  - **SEC-29 (#278)**: the `_SIGNIFICANT_FACTOR_PATTERN` regex parses a
    50KB whitespace-heavy payload in well under a second.
  - **SEC-30 (#279)**: `engine._load_yaml` rejects a file larger than
    `_MAX_YAML_BYTES` before `yaml.safe_load` resolves anchors.
  - **SEC-32 (#281)**: a non-JSON `batches_to_highlight` key raises
    `ValueError` at the plotting API surface, not `JSONDecodeError`.
  - **SEC-31 (#280)**: `test_cpython_pool_still_exposes_processes_attribute`
    (already in `tests/test_tool_safety.py`) pins the assumption.
  - **ENG-27 (#309)**: `tests/test_config.py` already covers lazy env-var
    reads via the central `settings` module (PR #326).
- **SEC-20 (#269)** items 2-4 (oneOf, nested items / properties, non-object
  root, missing array enum) are structurally closed by PR #332 -- the
  bespoke JSON-schema walker was replaced by `BaseModel.model_validate`,
  which supports all four directly. Item 1 (the `_SCALAR_CAPS` string-vs-
  number gap in `validate_input`) is mitigated in practice: a string
  `"50000"` passes the numeric cap but the very next stage,
  `_validate_against_model`, rejects it with `ToolInputInvalidError` via
  the tool's typed pydantic field.

## [1.24.4] - 2026-06-02

### Changed (breaking, internal API)

- `@tool_spec` no longer accepts the legacy `input_schema=` dict
  parameter -- `input_model=` (a pydantic `BaseModel` subclass with
  `ConfigDict(extra="forbid")`) is now the only supported form.
  This is the cleanup follow-up to PR1-PR4 (#328, #329, #330, #331)
  of the ENG-04 / ENG-10 migration. Every in-tree `@tool_spec`
  already uses `input_model=`; external callers that still passed
  `input_schema={...}` must migrate.
- `process_improve.tool_safety` drops the bespoke JSON-schema walker:
  `validate_against_schema()`, `_validate_value()`, `_JSON_TYPE_CHECKS`,
  and `_lookup_input_schema()` are gone. `safe_execute_tool_call` now
  validates via `input_model.model_validate(...)` and translates the
  resulting `pydantic.ValidationError` to `ToolInputInvalidError`.
  The new internal helpers are `_lookup_input_model()` and
  `_validate_against_model()`.
- `process_improve.tool_spec._filter_to_declared_keys` was removed;
  its job (rejecting undeclared kwargs) is now done structurally by
  `ConfigDict(extra="forbid")` at the pydantic boundary.

### Security

- SEC-20 (#266) and SEC-25 (#274) are closed by this PR: the
  "implicit MCP boundary trust" and "silent unknown-key drop hides
  bugs" findings both relied on the bespoke schema walker as a
  defence-in-depth backstop. With every `@tool_spec` now built on a
  pydantic `BaseModel` carrying `extra="forbid"`, those guarantees
  are part of the single canonical validation path
  (`execute_tool_call` and `safe_execute_tool_call`).

### Tests

- `tests/test_tool_spec.py` and `tests/test_tool_safety.py`: the
  synthetic test-only tools and the validation tests are rebuilt
  on top of pydantic. The previous `TestFilterToDeclaredKeys` and
  `TestValidateAgainstSchema` classes are replaced by a single
  `TestValidateAgainstModel` class that pins the new
  `_validate_against_model` / `_lookup_input_model` surface.

## [1.24.3] - 2026-06-02

### Changed

- Final per-package roll-out of the pydantic `@tool_spec` contract
  (ENG-04 / ENG-10; PR4/N). Five tools migrate to ``input_model=``
  with ``ConfigDict(extra="forbid")``:
  - **batch**: `extract_batch_features`
  - **visualization**: `boxplot`
  - **simulation**: `create_simulator`, `simulate_process`,
    `reveal_simulator`
  After this release every `@tool_spec` in the codebase uses the
  pydantic input contract.

### Security

- The `extra="forbid"` boundary on the simulation tools is now the
  structural closure of SEC-15 for the simulator suite (#264). A
  prompt-injected agent can no longer smuggle `confirmed=True` or a
  forged `simulator_state` through the dispatch path: the kwarg is
  rejected by pydantic before the function runs, so the host-only
  context-var channel remains the sole way to clear the reveal gate.
- `create_simulator.seed` uses `SkipJsonSchema[int | None]` so it
  stays callable from Python (tests, notebooks) but is omitted from
  the public JSON Schema, preserving the existing intent of hiding
  the seed from the LLM.

### Tests

- `tests/test_simulation.py` builds the new pydantic input models
  explicitly; the SEC-15 kwarg-injection tests now assert
  `ToolInputInvalidError` rather than the old in-band drop-and-fall-
  through behaviour.
- `tests/test_viz_boxplot.py` and `tests/batch/test_tools.py`
  introduce a thin test-local helper that builds the pydantic input
  from kwargs, so the existing kwarg-style call sites work
  unchanged.

## [1.24.2] - 2026-06-02

### Changed

- Migrated the **experiments** (10 tools) and **multivariate**
  (6 tools) packages to the pydantic `@tool_spec` contract (ENG-04 /
  ENG-10; PR3/N of the per-package roll-out). Tools covered:
  `create_factorial_design`, `fit_linear_model`, `generate_design`,
  `evaluate_design`, `analyze_experiment`, `optimize_responses`,
  `augment_design`, `visualize_doe`, `doe_knowledge`,
  `recommend_strategy`, `fit_pca`, `fit_pls`, `scale_data`,
  `detect_multivariate_outliers`, `pca_predict`, `pls_predict`.
  Each tool now declares a `BaseModel` with
  `ConfigDict(extra="forbid")` and takes the parsed model as its
  single positional argument. Several previously string-typed
  inputs are now `Literal[...]` enums, so malformed values are
  rejected at the dispatch boundary as `ToolInputInvalidError`
  rather than reaching the underlying call.

### Security

- The pydantic `Literal` constraints on `evaluate_design.model`,
  `augment_design.target_model`, `augment_design.augmentation_type`,
  `optimize_responses.method`, and `visualize_doe.plot_type` give
  SEC-14 (#263) an additional defence-in-depth layer: a malicious
  formula string injected in those fields is now rejected by the
  pydantic boundary before patsy / the underlying executor sees it.
  The behaviour already validated by `validate_formula_is_safe` is
  unchanged for the few fields that legitimately accept free-form
  formula strings (e.g. `analyze_experiment.model`).

### Tests

- Call-site tests across `test_design_generation.py`,
  `test_evaluate_design.py`, `test_augment_design.py`,
  `test_optimization.py`, `test_sec19_dos_caps.py`,
  `test_experiments_tools.py`, `test_experiments_security_sec14.py`,
  `test_recommend_strategy.py`, and `test_tool_spec.py` now drive
  the tools through `execute_tool_call(name, payload)` and assert
  `ToolInputInvalidError` for inputs the pydantic contract rejects.

## [1.24.1] - 2026-06-02

### Changed

- Migrated the **monitoring**, **regression**, and **bivariate**
  packages to the pydantic `@tool_spec` contract introduced in
  1.24.0 (ENG-04 / ENG-10; PR2/N of the per-package roll-out).
  Five tools updated: `control_chart`, `process_capability`,
  `robust_regression`, `repeated_median`, `find_elbow`. Each pairs
  its decorator with a `BaseModel` carrying
  `ConfigDict(extra="forbid")`; functions now take the parsed
  model as their single positional argument. Unknown keys raise
  `ToolInputInvalidError` at the dispatch boundary. No
  behavioural change for valid inputs.

### Tests

- The tool-call tests in `tests/test_bivariate.py`,
  `tests/test_monitoring.py`, and `tests/test_regression.py` now
  exercise the dispatch boundary via `execute_tool_call(...)`
  rather than importing the tool functions directly, mirroring
  PR1's univariate pattern.

## [1.24.0] - 2026-06-02

### Changed (breaking)

- `@tool_spec` now requires an `input_model: type[BaseModel]`
  parameter as the single source of truth for both the function's
  call signature and the MCP JSON Schema (ENG-04 / ENG-10
  decision; PR #328 of N). The legacy `input_schema={...}` form
  is still accepted while the per-package migration completes;
  new tools must use `input_model=`. Migration of the **univariate
  package** (9 tools) lands in this release; subsequent packages
  follow in immediate successor PRs in the same release window.
- `execute_tool_call` now validates the input dict against
  `input_model` (when present) via
  `BaseModel.model_validate(...)` and passes the parsed pydantic
  model to the tool function as a single positional argument.
  Pre-pydantic tools still use the legacy `**kwargs` filter.
- Unknown keys in a tool input now raise
  `ToolInputInvalidError` instead of being silently dropped with a
  warning. This is the structural closure of SEC-15's
  `confirmed=True` kwarg-injection vector (`extra="forbid"` on
  every input model).

### Security

- The new pydantic boundary catches SEC-15 (#264) again at a
  deeper layer: even if the runtime filter regressed, every
  ``extra=`` key now produces `extra_forbidden` validation errors
  by pydantic's construction.

### Tests

- `tests/test_tool_spec.py::TestExecuteToolCall::test_rejects_keys_not_declared_in_pydantic_model`
  (renamed from `test_drops_keys_not_declared_in_schema`) pins
  the new strict behaviour.
- `tests/fuzz/test_robust_scale_sn_fuzz.py` accepts
  `ToolInputInvalidError` as a documented MCP-boundary exception.

## [1.23.2] - 2026-06-02

### Security

- Close the MCP DoS surface tracked in **SEC-19** (#268). Every
  algorithm input that drives cost is now bounded, with the new
  caps wired through the central `settings` module (ENG-09 / ENG-27,
  PR #326) so they are test-overridable and contributors have one
  place to look. Six new caps default to "comfortably above any
  legitimate use":

  - `settings.max_factors_combinatorial = 15`. Caps `k` in
    `full_factorial`, the d-optimal `fullfact([3]*k)` fallback,
    and the mixture-design generators (`_simplex_centroid`,
    `_simplex_lattice`). A `k = 40` request no longer asks for
    `2**40` rows. `_simplex_lattice` also caps
    `(degree + 1) ** k <= 1_000_000` iterations.
  - `settings.max_regression_points = 5_000`.
    `repeated_median_slope` rejects oversize `x` before its
    O(N^2) inner loop.
  - `settings.max_matrix_rows = 10_000`,
    `settings.max_matrix_cols = 500`. `fit_pca`, `fit_pls`,
    `scale_data`, `detect_multivariate_outliers`, `pca_predict`,
    and `pls_predict` validate the input matrix shape before
    allocation.
  - `settings.max_formula_chars = 4_096`,
    `settings.max_formula_terms = 100`. `fit_linear_model`
    rejects oversize `formula` strings up front; `lm()` rejects
    formulas that expand to more than `max_formula_terms`
    after patsy parses the RHS.
- `_SCALAR_CAPS` in `tool_safety` gains five new keys:
  `n_steps` (100), `n_additional_runs` (500), `center_points` (50),
  `replicates` (50), and `n_factors` (15). An attacker can no
  longer drive cost through any of these.

### Tests

- New `tests/test_sec19_dos_caps.py` (19 tests, 5 classes) pins
  each cap.

## [1.23.1] - 2026-06-02

### Added

- New `process_improve.config` module exporting a single `settings`
  object as the canonical home for every configurable knob
  (ENG-09 #291, ENG-27 #309). Knobs include the existing
  `tool_safety` env-var contract (timeout, max cells / strings /
  depth, memory cap) plus the MCP safe-mode flag. Values are read
  on first access (not at import time) so tests can override via
  `settings.tool_timeout = 5.0` or `monkeypatch.setenv(...)` plus
  `settings.reload()`. The env-var names (`PROCESS_IMPROVE_*`) and
  default values are unchanged.

### Changed

- `process_improve.tool_safety` reads its five knobs through
  `settings.*` rather than `os.environ` at import time. The
  function signatures of `validate_input`,
  `safe_execute_tool_call`, and `get_pool` now accept `None` as
  the sentinel that resolves from `settings` at call time
  (previously the import-time `DEFAULT_*` constants were frozen).
- `process_improve.mcp_server` reads `mcp_safe_mode` through
  `settings` so a test fixture can flip safe-mode on/off without
  re-importing the module.

### Deprecated

- The module-level `DEFAULT_TIMEOUT_S`, `DEFAULT_MAX_CELLS`,
  `DEFAULT_MAX_STRING`, `DEFAULT_MAX_DEPTH`, and
  `DEFAULT_MEMORY_MB` names in `tool_safety` are now thin
  forwarding shims that read from `settings`. Imports keep
  working; new code should reference `settings.*` directly.
  Scheduled for removal in v2.0 per the deprecation policy.

## [1.23.0] - 2026-06-02

### Added

- `PCA.score_limit()` / `PLS.score_limit()`: per-component confidence limits
  for the model scores, ported from the legacy `calc_limits` block.
- `OLS.prediction_interval()`: prediction intervals for linear regression at
  arbitrary new `x` values, including points outside the training range.
- `PLS.prediction_interval()`: prediction intervals for new observations,
  computed from the residual error variance and each observation's
  latent-space leverage.
- `Factor.from_data()`: build a coded `Factor` from a column of historical
  data, inferring the range (continuous) or levels (categorical).
- `visualization.raincloud()`: a raincloud plot combining a one-sided violin
  (the density cloud), a boxplot, and the jittered raw observations.
- Univariate statistics: `holm_bonferroni()` (Holm step-down
  multiple-comparisons correction), `biweight_midvariance()` (the
  Mosteller-Tukey robust scale), `tietjen_moore_test()` (a test for a
  specified number of outliers), and `distribution_fit()` (a
  Kolmogorov-Smirnov goodness-of-fit check).
- Batch feature functions: `f_robust_mad` (a normal-consistency-scaled median
  absolute deviation) and `f_agemin` / `f_agemax` (the index label at which a
  tag attained its minimum / maximum).
- A `mode` argument for the batch plots (`plot_all_batches_per_tag`,
  `plot_multitags`), so traces can show markers as well as lines.
- `dict_to_wide(..., group_by_batch=...)`: the wide output carries a named
  `(tag, sequence)` column index whose levels are swapped when
  `group_by_batch=True`.

### Changed

- `calculate_cpk()` now returns a `Bunch` with `cpk`, `center`, `spread` and
  `rsd` (relative standard deviation), instead of a bare float. Callers must
  read the `cpk` field.
- The `plot_multitags` settings dictionary is now a `MultiTagPlotSettings`
  pydantic model.
- `gather()` accepts multi-column `DataFrame` inputs (one output column per
  block column) and validates that every input has the same length.
- `forg()` formats numbers at any precision instead of raising
  `NotImplementedError` for a precision other than 3 or 4.
- `find_elbow_point()` selects the elbow against the median of every
  accumulated intersection point in both `x` and `y`, making it more robust
  to spurious window fits.

## [1.22.21] - 2026-06-02

### Security

- **SEC-24** (#273) -- `confidence_interval`, paired `t_value`, and
  `calculate_cpk` now reject degenerate sample sizes (n < 2 / zero
  spread) up front with a clear `ValueError` or `RuntimeWarning`,
  instead of silently returning `inf` / `NaN`.
- **SEC-26** (#275) -- `analyze_experiment(transform="inverse")`
  rejects a zero in the response column with a clear `ValueError`
  rather than producing `inf` and a downstream `LinAlgError` whose
  text was leaked via the broad `except` in the tool wrapper.
- **SEC-27** (#276) -- the quadratic-term parser in
  `experiments/optimization.py` and the surface-plot generator now
  accept both `I(A ** 2)` and `np.power(A, 2)` / `power(A, 2)`.
  Newer statsmodels emits the latter; the silent fall-through to
  the linear branch was producing wrong predictions / surfaces.
- **SEC-28** (#277) -- `draw_initial_seed` now uses
  `secrets.randbits(63)` instead of truncating `SeedSequence`
  entropy to 31 bits. Brute-forcing the simulator seed is no
  longer feasible.
- **SEC-29** (#278) -- `_SIGNIFICANT_FACTOR_PATTERN` is bounded
  (`\b(\w+(?:\s\w+){0,4})`) so it matches in linear time. A new
  4 KiB cap on the `prior_knowledge` text rejects multi-MB payloads
  at the strategy-tool boundary, closing the regex-DoS surface.
- **SEC-30** (#279) -- the knowledge YAML loader now refuses files
  larger than 1 MiB. Defends against a tampered file that ships a
  YAML anchor bomb (`safe_load` resolves anchors and merges, so a
  small file can expand to gigabytes during parse).
- **SEC-32** (#281) -- `batch.plotting` now decodes the JSON dict
  keys outside the comprehension and re-raises `JSONDecodeError`
  as a documented `ValueError` at the API surface, rather than
  letting the decoder error escape from inside a dict-merge.

### Fixed

- **SEC-31** (#280) -- `tests/test_tool_safety.py` now asserts that
  `concurrent.futures.ProcessPoolExecutor` exposes the
  `_processes` attribute that `_terminate_workers` depends on. If a
  future Python release renames it, CI now fails loudly instead of
  silently degrading the SEC-02 timeout guarantee.
- **SEC-33** (#282) -- a five-item misc cleanup:
  - `terminate_check` now uses `>=` against `md_max_iter` (was an
    off-by-one that ran one extra iteration).
  - `np.var` on an empty negative-only slice no longer silently
    skips a sign-flip in `internal_pls_nipals_fit_one_pc`.
  - Float `==` zero guard on `norm(t)*norm(u)` in the MBPLS
    permutation test is now `<= epsqrt` (sub-eps near-zero denoms
    were producing meaningless ratios).
  - `np.arccos` argument in `bivariate.methods` is now clamped to
    `[-1, 1]` so floating-point excursions yield the boundary
    angle instead of silent `NaN`.
  - `_optimize_desirability` accepts an optional `random_state`
    parameter (default `42` preserves prior deterministic
    behaviour) via the ENG-08 `check_random_state` helper.
  - The registry's factor-name extractor now cross-references
    names against actual design columns when available, instead
    of the incomplete static `{"I", "np", "power"}` blocklist.

## [1.22.20] - 2026-06-02

### Security

- Complete the SEC-21 sweep: nine numerical sub-items in the
  single-block PCA / PLS / TPLS / MBPLS / MBPCA paths that
  previously divided by quantities that can be zero on
  legitimate-but-degenerate inputs, silently poisoning fitted
  models with `inf` / `NaN` (#270):

  1. **TPLS inner-PLS `np.linalg.norm` divisions**
     (`multivariate/methods.py:~2865, ~2876, ~3725-3727, ~4173,
     ~4206-4209`) -- floor each denominator via `_nz(...)` so a
     fully-deflated component or all-zero starting vector does not
     produce NaN convergence ratios.
  2. **PCA NIPALS initial-guess slice** (`~853`) -- add a defensive
     `.copy()` to `t_a_guess = Xd[:, [0]].copy()`. Current numpy
     fancy-indexing returns a copy, so the bug as stated in the
     audit is not exploitable today; the explicit copy matches the
     PLS path (~`:1527`) and protects against any future numpy
     change.
  3. **`spe_calculation` divide-by-zero** (`~2741-2744`) -- on a
     perfect-fit training set (`variance_spe == 0`) or all-equal
     SPE values, return `sqrt(center_spe)` rather than a NaN limit.
  4. **`r2_per_variable_` divide-by-zero on constant columns**
     (`~821, ~891, ~1744`) -- emit NaN via `np.where(prior_ssx_col
     > 0, ..., np.nan)` so callers can distinguish "no signal" from
     a numeric R^2. Same treatment for `r2y_per_variable_` on the
     PLS path. `r2cum` now also short-circuits to NaN when
     `base_variance == 0`.
  5. **Score-contribution weighting divide-by-zero**
     (`~1237-1242, ~2111-2116, ~4868-4873, ~5717-5722`) -- clamp
     `sqrt(explained_variance_)` so a degenerate component
     contributes nothing rather than `inf` / `NaN`.
  6. **`explained_variance_` divide-by-`(N-1)`** (`~803, ~909,
     ~971, ~1678`) -- mirror the MBPLS / MBPCA `max(1, N-1)` idiom
     so a single-row fit no longer produces `inf` variance.
  7. **`quick_regress` un-normalised numerator**
     (`~2554-2572`) -- in the `np.abs(denom) <= epsqrt` branch,
     return `0.0` rather than the un-normalised `sum(x*y)`. The
     previous behaviour silently mixed two different quantities
     into `b`.
  8. **`np.argmin` on all-NaN RMSECV** (`~2028-2042`) -- raise
     `RuntimeError` when every CV fold produced NaN total-RMSECV,
     rather than silently recommending `n_components = 1`.
  9. **`Resampler.bootstrap` / `.fractional` honor `random_state`**
     (`~5754-5783, ~5826-5879`) -- the class now accepts
     `random_state: int | Generator | None`, resolves it once via
     `process_improve._random.check_random_state` (PR #318), and
     uses `self._rng` for every draw. Two `Resampler`s built with
     the same int seed now produce bit-identical bootstrap and
     fractional resamples.

### Tests

- New `tests/properties/test_sec21_pca_pls_safety.py` covers
  sub-items 2 and 4 with both example-based and hypothesis-based
  tests (`TestR2PerVariableConstantColumn`,
  `TestNipalsDoesNotMutateInput`).
- `tests/test_multivariate_resampler.py` gains a new
  `TestResamplerReproducibility` class covering sub-item 9
  (same-seed identity, different-seed sequences, Generator
  passthrough, fractional reproducibility).

## [1.22.19] - 2026-06-02

### Security

- All MCP tool wrappers in `bivariate/tools.py`, `batch/tools.py`,
  `regression/tools.py`, `monitoring/tools.py`,
  `visualization/tools.py`, and the ten wrappers in
  `experiments/tools.py` now narrow their `except` to a documented
  set (`ValueError`, `TypeError`, `KeyError`,
  `numpy.linalg.LinAlgError`, and `PatsyError` where applicable),
  matching the pattern already used by the PCA / PLS tools (SEC-18 #267).
  Anything outside that set propagates to
  `mcp_server._serialise_tool_error`, which logs the traceback
  server-side and returns a generic message to the caller. This
  closes the last reliable path by which `str(exc)` containing
  pandas / numpy / statsmodels filesystem paths and internals could
  reach an MCP caller.
- `process_improve.experiments.designs_factorial.full_factorial`
  now raises `ValueError` when `nfactors < 1`, instead of falling
  through to a deep `ZeroDivisionError` inside numpy's `split`.
  Regression tested in `tests/test_doe.py::test_full_factorial_rejects_non_positive_nfactors`.

### Internal

- `experiments/tools.py` gains a module-level
  `_TOOL_EXPECTED_EXCEPTIONS` tuple that the ten wrappers share, so
  the canonical exception set stays in sync.

## [1.22.18] - 2026-06-01

### Security

- Complete the SEC-17 sweep: every remaining validation `assert` in
  `multivariate/methods.py`, `multivariate/plots.py`,
  `bivariate/methods.py`, `univariate/tools.py`,
  `experiments/structures.py`, `batch/features.py`,
  `batch/plotting.py`, `batch/preprocessing.py`, and
  `monitoring/metrics.py` is now an explicit
  `if not X: raise ValueError(...)` (or `TypeError` / `KeyError` /
  `NotImplementedError` per the [error-handling style guide](docs/development/error_handling.rst)).
  Under `python -O` these checks no longer disappear (SEC-17 #266).
- The `test-under-dash-O` CI job is now a blocking check (was
  `continue-on-error: true` in v1.22.16; flipped in this release).
  A future regression that re-introduces a validation `assert`
  turns CI red.

### Changed

- `PCA.t2_plot` and `PCA.spe_plot` now raise `ValueError` (rather
  than `AssertionError`) for `with_a == 0` or `with_a >
  n_components`. Tests updated.
- `PCA.fit` raises `ValueError` (rather than `AssertionError`) on
  invalid `missing_data_settings["md_tol"]`. Tests updated.
- `process_improve.batch.features._make_phase_features` raises
  `NotImplementedError` when multiple columns are passed, replacing
  a silent `assert len(columns) == 1`.

### Internal

- Post-preprocessing invariants in `TPLS._learn_center_and_scaling_parameters`
  (cross-checks on the centered/scaled `z_mats`, `f_mats`, `d_mats`,
  `y_mats`) remain as `assert`s with explanatory inline comments,
  per the error-handling guide's "internal invariant" carve-out.
- Endpoint invariants in `batch.preprocessing` (`new_time_axis.min()
  == sequence.min()`) likewise remain.

## [1.22.17] - 2026-06-01

### Added

- New `typecheck` CI job runs `mypy process_improve` on every PR
  (ENG-20). Initially `continue-on-error: true` because the codebase
  currently surfaces ~333 mypy errors with the existing permissive
  config (`disallow_untyped_defs = false`); the job will flip to
  blocking once a staged ENG-03 (#285) follow-up has paid the debt
  down. `notebooks_examples/` is excluded from the type-check (the
  scripts are not production code).

## [1.22.16] - 2026-06-01

### Added

- Thin scaffolding for the ENG-15 testing rigour push: new
  `tests/properties/` (hypothesis property tests),
  `tests/perf/` (pytest-benchmark baselines), and `tests/fuzz/`
  (MCP boundary fuzz). Each directory ships one example test
  against `process_improve._random.check_random_state` /
  `robust_scale_sn` so the pattern is documented in code for
  contributors who add coverage later.
- New `test-under-dash-O` CI job runs the full suite with
  `python -O`. Initially `continue-on-error: true` because
  SEC-17 (#266) has not yet swept the remaining
  assert-as-validation sites; the job flips to blocking once
  SEC-17 lands.
- `pytest-benchmark>=4.0.0` added to dev deps.

## [1.22.15] - 2026-06-01

### Added

- Register four pytest marker tiers (`unit`, `integration`, `slow`,
  `dataset`) in `pytest.ini` (ENG-29). Contributors can now run
  `pytest -m "not dataset"` (or `-m "not slow"`) to skip the
  network / slow tests during local iteration; CI still runs the
  full suite. The convention is documented in `CONTRIBUTING.md`
  under "Test tiers". The network-backed dataset loaders in
  `tests/test_experiments_datasets.py` are the first tests tagged
  with the new `dataset` marker.

## [1.22.14] - 2026-06-01

### Added

- New helper `process_improve._random.check_random_state` resolves
  `int | numpy.random.Generator | None` into a `numpy.random.Generator`,
  matching the reproducibility contract documented in
  `docs/development/reproducibility.rst` (ENG-08). Pre-requisite for the
  Wave 3 RNG sweeps (SEC-21 sub-item 9, SEC-33 sub-item).

## [1.22.13] - 2026-06-01

### Fixed

- The Holt-Winters control chart now raises a clear `ValueError` when the
  warm-up window has zero (or non-finite) variance, instead of silently
  dividing by `sigma_0 = 0` and returning `0`/`NaN` control limits (SEC-22).
- `regression.OLS.predict` now validates that `X` has the same number of
  features as the fitted model and raises a clear `ValueError` otherwise,
  rather than relying on a confusing numpy error or silently broadcasting to a
  wrong-shaped result (SEC-23).

## [1.22.12] - 2026-06-01

### Security

- The `reveal_simulator` double-confirmation gate can no longer be bypassed by
  injecting `confirmed=True` or a fabricated `simulator_state` through a tool
  call (SEC-15). `execute_tool_call` now drops any input key not declared in the
  tool's `input_schema` before dispatch, and the simulator's host-only
  parameters (`simulator_state`, `confirmed`) have moved off the function
  signature into a `contextvars` side channel populated via the new
  `process_improve.simulation.context.simulator_host_context`. Hosts that
  persist simulator state must wrap their dispatch in that context manager.

## [1.22.11] - 2026-06-01

### Security

- Closed a patsy-formula code-execution vector (SEC-14, the same class as
  SEC-01) in the `analyze_experiment`, `evaluate_design` and `augment_design`
  paths and in the public `lm()` API. Patsy evaluates each formula term as a
  Python expression, so an untrusted `model` / `target_model` string,
  `response_column`, or `design_matrix` column name could run arbitrary code.
  `validate_formula_is_safe` now guards every formula before it reaches patsy
  and gained `allow_transforms` / `allow_numpy` flags (an AST-based check that
  permits `I()`/`Q()` and a curated allowlist of element-wise `np.<func>`
  transforms while rejecting attribute access, string literals, dunders, and any
  other call). A new `validate_identifier_is_safe` rejects column and response
  names that are not plain identifiers. Legitimate Wilkinson, `quadratic`
  shorthand, and `I(...)` / numpy-transform models are unaffected.

## [1.22.10] - 2026-06-01

### Fixed

- `TPLS.score` no longer raises `NameError` when the `Y` block (`X["Y"]`) is an
  empty dict (SEC-16). The method now raises a clear `ValueError` instead of
  crashing on a malformed test bundle, and the per-block averaging uses an
  explicit counter rather than the loop index.

## [1.22.9] - 2026-05-29

This release lands the second half of the `SECURITY_AUDIT.md` hardening series
(SEC-07 through SEC-12). Each fix shipped as its own code+tests pull request;
this entry records them together.

### Security

- The MCP server no longer returns the raw text of an unexpected exception to
  the caller (SEC-09). Unexpected errors may contain internal detail (file
  paths, library internals); the full traceback is logged server-side and a
  generic message is returned. Structured `ToolSafetyError`s keep their curated
  payload.

### Fixed

- Matrix inversions that could silently return overflow-driven garbage on a
  singular or ill-conditioned matrix are now guarded (SEC-07). New internal
  `process_improve._linalg` provides `safe_inverse` (raises a clear
  `LinAlgError`) and `is_singular`; applied to the PLS direct-weights inverse
  and the TPLS Hotelling's T2 covariance inverse, with the response-surface and
  design-quality plots falling back to the pseudo-inverse for ill-conditioned
  `X'X`. Well-conditioned inputs are unchanged.
- `discover_tools` no longer silently swallows every `ImportError` (SEC-11): a
  missing optional dependency is tolerated and logged, while any other
  `ImportError` propagates instead of dropping a whole tool category.
- The knowledge-base YAML loader (`experiments/knowledge/engine._load_yaml`)
  rejects filenames that escape its data directory (SEC-10).

### Changed

- Input/state validation that used bare `assert` statements is now done with
  explicit `if ...: raise ValueError`/`NotFittedError` (SEC-08), so the checks
  remain active under `python -O`. Two control-chart parameter checks now raise
  `ValueError` instead of `AssertionError`.
- The Breusch-Pagan diagnostic (`experiments/analysis.py`) and design-metric
  evaluation (`experiments/augment.py`) catch only expected failure types and
  log, instead of silently swallowing every exception (SEC-09).
- The remote sample-dataset loaders (`experiments/datasets`) wrap their network
  fetch and raise a clear `RuntimeError` on failure (SEC-10).
- `find_reference_batch` (`batch/preprocessing.py`) filters with boolean-mask
  indexing instead of a `DataFrame.query()` expression string (SEC-12).

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

[Unreleased]: https://github.com/kgdunn/process-improve/compare/v1.43.0...HEAD
[1.43.0]: https://github.com/kgdunn/process-improve/compare/v1.42.1...v1.43.0
[1.42.1]: https://github.com/kgdunn/process-improve/compare/v1.42.0...v1.42.1
[1.42.0]: https://github.com/kgdunn/process-improve/compare/v1.41.0...v1.42.0
[1.41.0]: https://github.com/kgdunn/process-improve/compare/v1.40.2...v1.41.0
[1.40.2]: https://github.com/kgdunn/process-improve/compare/v1.40.1...v1.40.2
[1.40.1]: https://github.com/kgdunn/process-improve/compare/v1.40.0...v1.40.1
[1.40.0]: https://github.com/kgdunn/process-improve/compare/v1.39.0...v1.40.0
[1.39.0]: https://github.com/kgdunn/process-improve/compare/v1.38.4...v1.39.0
[1.38.4]: https://github.com/kgdunn/process-improve/compare/v1.38.3...v1.38.4
[1.38.3]: https://github.com/kgdunn/process-improve/compare/v1.38.2...v1.38.3
[1.38.2]: https://github.com/kgdunn/process-improve/compare/v1.38.1...v1.38.2
[1.38.1]: https://github.com/kgdunn/process-improve/compare/v1.38.0...v1.38.1
[1.38.0]: https://github.com/kgdunn/process-improve/compare/v1.37.0...v1.38.0
[1.37.0]: https://github.com/kgdunn/process-improve/compare/v1.36.0...v1.37.0
[1.36.0]: https://github.com/kgdunn/process-improve/compare/v1.35.0...v1.36.0
[1.35.0]: https://github.com/kgdunn/process-improve/compare/v1.34.0...v1.35.0
[1.34.0]: https://github.com/kgdunn/process-improve/compare/v1.33.0...v1.34.0
[1.33.0]: https://github.com/kgdunn/process-improve/compare/v1.32.0...v1.33.0
[1.32.0]: https://github.com/kgdunn/process-improve/compare/v1.31.0...v1.32.0
[1.31.0]: https://github.com/kgdunn/process-improve/compare/v1.30.0...v1.31.0
[1.30.0]: https://github.com/kgdunn/process-improve/compare/v1.29.0...v1.30.0
[1.29.0]: https://github.com/kgdunn/process-improve/compare/v1.28.0...v1.29.0
[1.28.0]: https://github.com/kgdunn/process-improve/compare/v1.27.1...v1.28.0
[1.27.1]: https://github.com/kgdunn/process-improve/compare/v1.27.0...v1.27.1
[1.27.0]: https://github.com/kgdunn/process-improve/compare/v1.26.1...v1.27.0
[1.26.1]: https://github.com/kgdunn/process-improve/compare/v1.26.0...v1.26.1
[1.26.0]: https://github.com/kgdunn/process-improve/compare/v1.25.1...v1.26.0
[1.25.1]: https://github.com/kgdunn/process-improve/compare/v1.25.0...v1.25.1
[1.25.0]: https://github.com/kgdunn/process-improve/compare/v1.24.34...v1.25.0
[1.24.34]: https://github.com/kgdunn/process-improve/compare/v1.24.33...v1.24.34
[1.24.33]: https://github.com/kgdunn/process-improve/compare/v1.24.32...v1.24.33
[1.24.32]: https://github.com/kgdunn/process-improve/compare/v1.24.31...v1.24.32
[1.24.31]: https://github.com/kgdunn/process-improve/compare/v1.24.30...v1.24.31
[1.24.30]: https://github.com/kgdunn/process-improve/compare/v1.24.29...v1.24.30
[1.24.29]: https://github.com/kgdunn/process-improve/compare/v1.24.28...v1.24.29
[1.24.28]: https://github.com/kgdunn/process-improve/compare/v1.24.27...v1.24.28
[1.24.27]: https://github.com/kgdunn/process-improve/compare/v1.24.26...v1.24.27
[1.24.26]: https://github.com/kgdunn/process-improve/compare/v1.24.24...v1.24.26
[1.24.24]: https://github.com/kgdunn/process-improve/compare/v1.24.23...v1.24.24
[1.24.23]: https://github.com/kgdunn/process-improve/compare/v1.24.22...v1.24.23
[1.24.22]: https://github.com/kgdunn/process-improve/compare/v1.24.21...v1.24.22
[1.24.21]: https://github.com/kgdunn/process-improve/compare/v1.24.20...v1.24.21
[1.24.20]: https://github.com/kgdunn/process-improve/compare/v1.24.19...v1.24.20
[1.24.19]: https://github.com/kgdunn/process-improve/compare/v1.24.18...v1.24.19
[1.24.18]: https://github.com/kgdunn/process-improve/compare/v1.24.17...v1.24.18
[1.24.17]: https://github.com/kgdunn/process-improve/compare/v1.24.16...v1.24.17
[1.24.16]: https://github.com/kgdunn/process-improve/compare/v1.24.15...v1.24.16
[1.24.15]: https://github.com/kgdunn/process-improve/compare/v1.24.14...v1.24.15
[1.24.14]: https://github.com/kgdunn/process-improve/compare/v1.24.13...v1.24.14
[1.24.13]: https://github.com/kgdunn/process-improve/compare/v1.24.12...v1.24.13
[1.24.12]: https://github.com/kgdunn/process-improve/compare/v1.24.11...v1.24.12
[1.24.11]: https://github.com/kgdunn/process-improve/compare/v1.24.10...v1.24.11
[1.24.10]: https://github.com/kgdunn/process-improve/compare/v1.24.9...v1.24.10
[1.24.9]: https://github.com/kgdunn/process-improve/compare/v1.24.8...v1.24.9
[1.24.8]: https://github.com/kgdunn/process-improve/compare/v1.24.7...v1.24.8
[1.24.7]: https://github.com/kgdunn/process-improve/compare/v1.24.6...v1.24.7
[1.24.6]: https://github.com/kgdunn/process-improve/compare/v1.24.5...v1.24.6
[1.24.5]: https://github.com/kgdunn/process-improve/compare/v1.24.4...v1.24.5
[1.24.4]: https://github.com/kgdunn/process-improve/compare/v1.24.3...v1.24.4
[1.24.3]: https://github.com/kgdunn/process-improve/compare/v1.24.2...v1.24.3
[1.24.2]: https://github.com/kgdunn/process-improve/compare/v1.24.1...v1.24.2
[1.24.1]: https://github.com/kgdunn/process-improve/compare/v1.24.0...v1.24.1
[1.24.0]: https://github.com/kgdunn/process-improve/compare/v1.23.2...v1.24.0
[1.23.2]: https://github.com/kgdunn/process-improve/compare/v1.23.1...v1.23.2
[1.23.1]: https://github.com/kgdunn/process-improve/compare/v1.23.0...v1.23.1
[1.23.0]: https://github.com/kgdunn/process-improve/compare/v1.22.21...v1.23.0
[1.22.21]: https://github.com/kgdunn/process-improve/compare/v1.22.20...v1.22.21
[1.22.20]: https://github.com/kgdunn/process-improve/compare/v1.22.19...v1.22.20
[1.22.19]: https://github.com/kgdunn/process-improve/compare/v1.22.18...v1.22.19
[1.22.18]: https://github.com/kgdunn/process-improve/compare/v1.22.17...v1.22.18
[1.22.17]: https://github.com/kgdunn/process-improve/compare/v1.22.16...v1.22.17
[1.22.16]: https://github.com/kgdunn/process-improve/compare/v1.22.15...v1.22.16
[1.22.15]: https://github.com/kgdunn/process-improve/compare/v1.22.14...v1.22.15
[1.22.14]: https://github.com/kgdunn/process-improve/compare/v1.22.13...v1.22.14
[1.22.13]: https://github.com/kgdunn/process-improve/compare/v1.22.12...v1.22.13
[1.22.12]: https://github.com/kgdunn/process-improve/compare/v1.22.11...v1.22.12
[1.22.11]: https://github.com/kgdunn/process-improve/compare/v1.22.10...v1.22.11
[1.22.10]: https://github.com/kgdunn/process-improve/compare/v1.22.9...v1.22.10
[1.22.9]: https://github.com/kgdunn/process-improve/compare/v1.22.8...v1.22.9
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
