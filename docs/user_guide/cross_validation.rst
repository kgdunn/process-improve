Cross-Validation
=================

Cross-validation is used for two purposes in multivariate analysis:

1. **Component selection** — choosing the right number of components (PCA).
2. **Coefficient uncertainty** — obtaining error bars for PLS beta
   coefficients.

Selecting the Number of Components
-----------------------------------

Choosing the right number of components is critical. Too few components
underfit (miss important structure), too many overfit (model noise).

PRESS Cross-Validation
-----------------------

The ``PCA.select_n_components()`` class method uses Predicted Residual Error
Sum of Squares (PRESS) with K-fold cross-validation:

1. For each candidate number of components (1, 2, ..., max):
   - Split data into K folds
   - Fit PCA on K-1 folds, predict the held-out fold
   - Compute prediction error (PRESS)
2. Apply **Wold's criterion**: stop when adding a component does not
   meaningfully reduce PRESS (ratio > threshold)

.. code-block:: python

   from process_improve.multivariate.methods import PCA, MCUVScaler

   X_scaled = MCUVScaler().fit_transform(X)

   result = PCA.select_n_components(
       X_scaled,
       max_components=10,
       cv=7,  # 7-fold cross-validation
       threshold=0.95,  # Wold's criterion threshold
   )

   print(f"Recommended components: {result.n_components}")
   print(f"PRESS values: {result.press}")
   print(f"PRESS ratios: {result.press_ratio}")

The result is a ``Bunch`` with:

- ``n_components``: recommended number of components
- ``press``: PRESS value for each number of components
- ``press_ratio``: ratio ``PRESS_a / PRESS_{a-1}`` (values > threshold suggest overfitting)
- ``cv_scores``: raw cross-validation scores per fold

Wold's Criterion
-----------------

The default threshold of 0.95 means: stop adding components when the PRESS
ratio exceeds 0.95. A ratio close to 1.0 means the new component barely
improves prediction — it is likely fitting noise.

Lower thresholds (e.g., 0.90) are more conservative (fewer components).
Higher thresholds (e.g., 0.98) are more liberal (more components).

PLS Beta Coefficient Error Bars
--------------------------------

For PLS models, ``model.cross_validate()`` refits the model on data subsets
and computes confidence intervals for the regression coefficients. This answers
the question: *"How reliable is each beta coefficient?"*

Three resampling strategies are supported:

- **Jackknife** (``cv="loo"``, default) — leave-one-out resampling. Uses the
  jackknife variance formula with t-distribution critical values.
- **K-fold** (``cv=5``) — K-fold cross-validation. Faster for large datasets.
- **Bootstrap** (``n_bootstrap=200``) — resample with replacement. Uses
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
- ``significant``: boolean mask — ``True`` where the CI excludes zero
- ``beta_samples``: raw betas from every resample (n_resamples × K × M)
- ``y_hat_cv``: out-of-fold Y predictions (jackknife / K-fold only)
- ``press``: Prediction Error Sum of Squares
- ``rmse_cv``: cross-validated RMSE per Y variable
- ``q_squared``: cross-validated R² (Q²) per Y variable

See :doc:`pls` for detailed documentation and additional examples.
