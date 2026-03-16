Selecting the Number of Components
===================================

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
