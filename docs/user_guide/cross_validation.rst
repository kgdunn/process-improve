Cross-Validation
=================

Cross-validation is used for two purposes in multivariate analysis:

1. **Component selection** - choosing the right number of components, for
   both PCA and PLS.
2. **Coefficient uncertainty** - obtaining error bars for PLS beta
   coefficients.

Selecting the Number of Components
-----------------------------------

Choosing the right number of components is critical. Too few components
underfit (miss important structure), too many overfit (model noise).

Element-wise Cross-Validation (PCA)
-----------------------------------

``PCA.select_n_components()`` evaluates every component count from 1 to
``max_components``, measures the Predicted Residual Error Sum of Squares
(PRESS) of each, and recommends one. The default scheme is the
**element-wise k-fold** (ekf) algorithm of Bro et al. (2008):

1. Split the individual *cells* of ``X`` into K folds, so that every cell is
   held out exactly once.
2. For each fold and each component count: mask that fold's cells, impute
   them EM-style from a model fitted on the cells that remain, and add the
   squared error of those predictions to PRESS.
3. Recommend a component count from the PRESS curve using
   ``selection_rule``.

Holding out individual cells, rather than whole rows, is what keeps a
prediction independent of the value being predicted. Under the legacy
``cv_scheme="row_wise"`` scheme a held-out row flows back through
``transform()`` into its own prediction, so PRESS shrinks monotonically and
the recommendation tends to run to ``max_components``. That scheme is kept
for backwards compatibility and emits a ``SpecificationWarning``.

.. code-block:: python

   from process_improve.multivariate.methods import PCA

   # Pass the raw, unscaled X: with the default ``scale_inside_folds=True``
   # the centring and scaling are fit inside each fold, so nothing about the
   # held-out cells leaks into the model that predicts them.
   result = PCA.select_n_components(
       X,
       max_components=10,
       cv=7,  # 7 element-folds
   )

   print(f"Recommended components: {result.n_components}")
   print(result.press)  # PRESS per component count
   print(result.q2)     # cross-validated R2 of X

The result is a ``Bunch`` with:

- ``n_components``: recommended number of components
- ``press``: PRESS for each number of components
- ``q2``: cross-validated :math:`R^2_X` per component count, on the same
  scale as the calibration ``r2_cumulative_`` of a fitted model
- ``per_fold_press``, ``se_press`` and ``q2_se``: the per-fold PRESS
  contributions and the standard error built from them, which is what the
  1-SE rule needs
- ``press_ratio``: the ratio ``PRESS_a / PRESS_{a-1}``, for inspection
- ``cv_scores``: per-fold scores (an alias of ``per_fold_press`` under ekf)
- ``cv_scheme`` and ``selection_rule``: which scheme and rule were used

``n_repeats`` runs the whole pass again with a fresh fold permutation.
Each repeat still covers every cell exactly once; more repeats narrow
``se_press``, which helps when the 1-SE rule sits on a borderline.

Selection Rules
----------------

``selection_rule`` decides which count is recommended from the error curve.
PCA defaults to ``"min"``; PLS defaults to ``"1se"``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Rule
     - Recommends
   * - ``"min"``
     - The component count with the lowest cross-validated error. This is
       the GlobalMin criterion that Bro et al. pair with ekf.
   * - ``"1se"``
     - The smallest count whose error is within one standard error of that
       minimum, so it is never less parsimonious than ``"min"``.
   * - ``"q2_increment"``
     - Keeps a component only while it lifts the cumulative :math:`Q^2` by
       at least ``min_q2_increase`` (default 0.01), and stops at the first
       one that does not. A Wold's-R-style heuristic: cheap, but the
       threshold is absolute and hand-tuned.
   * - ``"randomization"``
     - PLS only. Van der Voet's (1994) permutation test: the smallest model
       whose predictive ability is statistically indistinguishable from the
       lowest-RMSECV one, at significance level ``alpha``.

.. note::

   The original Wold PRESS-ratio cutoff, the ``threshold`` argument of
   ``PCA.select_n_components``, is deprecated: passing it emits a
   ``DeprecationWarning`` and the value is ignored. Use
   ``selection_rule="q2_increment"``, tuned with ``min_q2_increase``, for a
   comparable preference for parsimony.

PLS Component Selection
------------------------

``PLS.select_n_components()`` cross-validates a PLS model and reports how it
performs on unseen data, in contrast to the calibration statistics stored on a
fitted model (``rmse_``, ``r2_cumulative_``), which always improve as
components are added.

.. code-block:: python

   from process_improve.multivariate.methods import PLS

   # Raw, unscaled blocks: each training fold fits its own MCUVScaler and
   # RMSECV is reported on the original Y scale.
   result = PLS.select_n_components(X, Y, max_components=8, cv=5)

   print(f"Recommended components: {result.n_components}")
   print(result.rmsecv["total"])         # RMSECV per component count
   print(result.r2y_validated["total"])  # Validated R2 of Y

Do not scale the blocks yourself before calling either selector. In-fold
re-standardisation overwrites whatever scaling you applied, so two
deliberately different choices (autoscale versus Pareto, say) become the
same model and a comparison between them shows no difference. Both
selectors emit a ``SpecificationWarning`` when they receive an ``X`` that
is already centred and unit-variance scaled. If you must keep your own
scaling, pass ``scale_inside_folds=False``; the scaling then leaks from the
full dataset into every fold, and a warning says so.

The result is a ``Bunch`` with:

- ``n_components``: recommended count, chosen by ``selection_rule`` (the
  1-SE rule by default, not the lowest RMSECV; see `Selection Rules`_)
- ``rmsecv``: root-mean-square error of cross-validation, per Y variable and overall
- ``se_rmsecv`` / ``q2_se``: the standard error of that curve, on the RMSECV
  and the :math:`Q^2` scale respectively
- ``r2y_validated`` / ``r2x_validated``: validated explained variance, per variable and overall
- ``press``: overall Y prediction error sum of squares per component count
- ``cv_predictions``: out-of-fold Y predictions at the recommended count
- ``selection_rule``: the rule that produced ``n_components``

The ``cv`` argument accepts an integer (K-fold) or any scikit-learn splitter
object, such as ``KFold`` or ``LeaveOneOut``. When it is an integer, the
split is repeated ``n_repeats`` times (10 by default) with a fresh shuffle,
which is what gives the 1-SE rule a usable standard error. A splitter object
is used as-is and ``n_repeats`` is then ignored.

PLS Beta Coefficient Error Bars
--------------------------------

For PLS models, ``model.cross_validate()`` refits the model on data subsets
and computes confidence intervals for the regression coefficients. This answers
the question: *"How reliable is each beta coefficient?"*

Three resampling strategies are supported:

- **Jackknife** (``cv="loo"``, default) - leave-one-out resampling. Uses the
  jackknife variance formula with t-distribution critical values.
- **K-fold** (``cv=5``) - K-fold cross-validation. Faster for large datasets.
- **Bootstrap** (``n_bootstrap=200``) - resample with replacement. Uses
  percentile confidence intervals.

.. code-block:: python

   from process_improve.multivariate.methods import PLS, MCUVScaler

   scaler_x = MCUVScaler().fit(X)
   scaler_y = MCUVScaler().fit(Y)
   X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

   pls = PLS(n_components=2).fit(X_s, Y_s)

   # Jackknife (leave-one-out) cross-validation
   cv = pls.cross_validate(X_s, Y_s, cv="loo")

   print(cv.significant)      # Which betas have CIs excluding zero
   print(cv.beta_ci_lower)    # Lower 95% CI
   print(cv.beta_ci_upper)    # Upper 95% CI
   print(cv.q_squared)        # Cross-validated R² (Q²)
   print(cv.rmse_cv)          # Cross-validated RMSE

The result is a ``Bunch`` with:

- ``beta_mean``, ``beta_std``: mean and standard error of betas across
  resamples
- ``beta_ci_lower``, ``beta_ci_upper``: confidence interval bounds
- ``significant``: boolean mask - ``True`` where the CI excludes zero
- ``beta_samples``: raw betas from every resample (n_resamples × K × M)
- ``y_hat_cv``: out-of-fold Y predictions (jackknife / K-fold only)
- ``press``: Prediction Error Sum of Squares
- ``rmse_cv``: cross-validated RMSE per Y variable
- ``q_squared``: cross-validated R² (Q²) per Y variable

See :doc:`pls` for detailed documentation and additional examples.
