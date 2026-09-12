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
prediction independent of the value being predicted. Under the deprecated
``cv_scheme="row_wise"`` scheme a held-out row flows back through
``transform()`` into its own prediction, so PRESS shrinks monotonically and
reaches zero once the components equal the variables. It measures compression
rather than prediction and cannot select a component count. It is kept for one
more release cycle and emits a ``DeprecationWarning``; it will be removed in
2.0.

Choosing among the schemes
--------------------------

Four schemes keep the prediction independent of the value predicted. They
differ in what they hold out and what they cost.

.. list-table::
   :header-rows: 1
   :widths: 12 34 30 24

   * - ``cv_scheme``
     - What is held out
     - Cost
     - Use it when
   * - ``"ekf"``
     - Scattered cells, imputed by EM from a model that never saw them
     - ``n_folds * n_repeats * max_components`` decompositions
     - The default. The only one that takes a block with missing cells.
   * - ``"ek"``
     - Nothing directly: a score comes from a model without the cell's
       column, a loading from a model without its row
     - ``2 * n_folds`` decompositions
     - A number has to line up with Simca-P or with ``pcaMethods::Q2``.
   * - ``"sacv"``
     - Nothing. Each residual is inflated by the leverage of the cell that
       produced it, approximating leave-one-cell-out
     - One decomposition
     - The block is large and ``"ekf"`` is too slow.
   * - ``"gcv"``
     - Nothing. One averaged leverage instead of one per cell
     - One decomposition
     - Comparing against ``FactoMineR`` or ``missMDA``, whose default it is.

The three new schemes factorise the matrix directly, so they raise on a block
with missing cells, and they ignore ``scale_inside_folds``, ``n_repeats``,
``n_iter`` and ``tol``. They hold nothing out in folds that could disagree, so
they report no per-fold spread and ``selection_rule="1se"`` has nothing to work
with.

Both leverage schemes are first-order approximations that degrade as the
component count approaches the number of variables, because a column's leverage
approaches one and the divisor approaches zero with it. Read them well below
that ceiling; ``FactoMineR`` defaults to five components for the same reason.
At the ceiling itself no cell has a defined leave-one-out residual at all, and
both schemes report ``NaN`` rather than a number there.

They also lean on the residual having something left in it. On the LDPE data
of 54 rows and 19 variables, whose fit reaches 99.98% by eleven components,
``"ekf"`` and ``"ek"`` both turn over at two components, which is where Simca-P
turns over on the same data. Neither ``"sacv"`` nor ``"gcv"`` turns over at all
within the first five: their curves rise at every count, so the number they
return is the largest one evaluated rather than an optimum. Any scheme that
does that now says so, through a :class:`SpecificationWarning`. Treat the
warning as the answer: the criterion failed on that data, and a scheme that
holds values out should be used instead.

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
- ``press_input_units``: the same curve in the units of the matrix that was
  passed in, for comparing the prediction error against a known instrument
  error
- ``q2``: cross-validated :math:`R^2_X` per component count, on the same
  scale as the calibration ``r2_cumulative_`` of a fitted model
- ``q2_per_variable``: that same quantity for each column on its own
- ``per_fold_press``, ``se_press`` and ``q2_se``: the per-fold PRESS
  contributions and the standard error built from them, which is what the
  1-SE rule needs
- ``press_ratio``: the ratio ``PRESS_a / PRESS_{a-1}``, for inspection
- ``cv_scores``: per-fold scores (an alias of ``per_fold_press`` under ekf)
- ``cv_scheme`` and ``selection_rule``: which scheme and rule were used

What PRESS is measured in
--------------------------

With ``scale_inside_folds=True`` each fold centres and scales the matrix
before fitting it, and the error is measured in that same space. Every
variable therefore contributes to ``press`` in proportion to how much of its
own variation the model predicts, not in proportion to its units. This
matters whenever the columns are on different scales: on the LDPE data used
in the book, the ``Mw`` column carries 99.5% of the raw sum of squares, so a
PRESS accumulated in the raw units would be ``Mw``'s prediction error and
almost nothing else.

``q2`` is that PRESS divided by what a null model would have got wrong on the
same held-out cells, measured the same way. The null model predicts each
held-out cell by the mean of the cells that were not held out, so
:math:`Q^2 = 0` is "no better than the column mean" and :math:`Q^2 = 1` is
exact prediction, the same reading as ``r2_cumulative_``.

Two consequences are worth knowing. Re-expressing a column in different units
(kilograms instead of grams, say) leaves the whole curve unchanged. And
passing the raw block gives the same curve as passing a mean-centred,
unit-variance block, so the recommendation in the paragraph above costs
nothing.

When a single number in the original units is what you need, for instance to
compare the prediction error against a known instrument error, read
``press_input_units`` instead. To see whether one column is carrying the
pooled figure, read ``q2_per_variable``, which splits ``q2`` by variable.

``scale_inside_folds=False`` is the opt-out for callers who have scaled their
own block. There is then no in-fold scale, so ``press`` is in the units of
whatever matrix was passed and the two PRESS fields coincide.

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
- ``r2y_validated`` / ``r2x_validated``: validated explained variance, per
  variable and overall. ``r2y_validated`` carries two overall columns:
  ``"total"`` pools the targets on the original Y scale, and
  ``"scaled_total"`` gives every target equal weight (see
  `Comparing the fitted and the validated R2`_)
- ``press``: overall Y prediction error sum of squares per component count
- ``cv_predictions``: out-of-fold Y predictions at the recommended count
- ``selection_rule``: the rule that produced ``n_components``

The ``cv`` argument accepts an integer (K-fold) or any scikit-learn splitter
object, such as ``KFold`` or ``LeaveOneOut``. When it is an integer, the
split is repeated ``n_repeats`` times (10 by default) with a fresh shuffle,
which is what gives the 1-SE rule a usable standard error. A splitter object
is used as-is and ``n_repeats`` is then ignored.

Comparing the fitted and the validated R2
------------------------------------------

A fitted model's ``r2_y_cumulative_`` is computed on the scaled Y, where every
target carries equal weight. ``r2y_validated["total"]`` is computed on the
original Y scale, where a target with a wide range carries more weight than a
narrow one. Read side by side on targets of unequal spread, the two can differ
by tens of percent, or disagree in sign, without the model having changed.

``r2y_validated["scaled_total"]`` is the held-out number on the fitted model's
footing: every target weighted equally, which on autoscaled Y is the arithmetic
mean of the per-target values. Compare fitted against held-out with those two,
and keep ``"total"`` for the question it answers, which is how much of the Y
variation *in its own units* the model predicts.

.. code-block:: python

   fitted = PLS(n_components=result.n_components).fit(X, Y)
   print(fitted.r2_y_cumulative_)                        # fitted, equal weight
   print(result.r2y_validated["scaled_total"])           # held out, equal weight
   print(result.r2y_validated["total"])                  # held out, original Y scale

``MBPLS.select_n_components`` returns the same two columns, on the same footing.

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
