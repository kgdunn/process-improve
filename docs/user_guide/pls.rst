Projection to Latent Structures (PLS)
======================================

PLS finds directions in **X** that are maximally correlated with **Y**. It
is the method of choice when you have predictor variables and response
variables that you want to relate to each other.

Mathematical Background
-----------------------

PLS simultaneously decomposes both the X and Y matrices:

.. math::

   \mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}_X

   \mathbf{Y} = \mathbf{U}\mathbf{Q}^T + \mathbf{E}_Y

where:

- **T** (N × A) — X-scores (projections of observations in X-space).
- **U** (N × A) — Y-scores (projections in Y-space).
- **P** (K × A) — X-loadings.
- **Q** (M × A) — Y-loadings.
- **E_X**, **E_Y** — residual matrices.

The algorithm pursues three objectives simultaneously: explain the variance
in X, explain the variance in Y, and maximize the *covariance* between the
two sets of scores. This is what distinguishes PLS from simply applying PCA
to X and then regressing on Y (which is PCR — Principal Components
Regression).

Why PLS Over Alternatives
-------------------------

**vs Multiple Linear Regression (MLR):**

- MLR requires more observations than variables (N > K) and fails with
  collinear X variables. PLS handles both situations naturally.
- PLS provides built-in noise reduction: by projecting onto a few latent
  variables it filters out measurement noise.
- PLS gives consistency checks through SPE and T², which MLR lacks.
- PLS handles missing values in X natively.

**vs Principal Components Regression (PCR):**

- PCR first applies PCA to X alone, then regresses on Y. The PCA step may
  choose directions that explain X variance but are *irrelevant* to Y.
- PLS uses both X and Y simultaneously, so it finds directions that are
  both well-represented in X and predictive of Y. This typically requires
  fewer components for the same predictive quality.

**Multiple Y columns:** PLS builds a single model for multiple correlated
response variables, using the correlation between Y columns to improve
predictions for each one.

Interpreting Scores
-------------------

PLS produces two sets of scores:

- **T** (``model.scores_``) — the X-scores, always available even for new
  observations where Y is unknown. These are the primary scores for
  interpretation and monitoring.
- **U** (``model.y_scores_``) — the Y-scores, available only when Y data
  exists. Useful for examining inner-model relationships.

A critical difference from PCA: *PCA scores explain only X variance; PLS
scores are calculated to also explain Y*. This means PLS components may not
capture the largest X variation — they capture the variation most relevant to
predicting Y.

Score interpretation otherwise follows PCA: look for clusters, outliers, and
time trends in the T-scores.

Interpreting Loadings and Weights
---------------------------------

PLS has several related vectors that describe variable importance:

- **X-weights** (``model.x_weights_``) — the raw weight vectors **w** used
  during the iterative algorithm. Each **w** is found on deflated data.
- **X-loadings** (``model.x_loadings_``) — the regression coefficients **p**
  relating X to T.
- **Direct weights** (``model.direct_weights_``) — also called **r** or
  **w*** in the literature. These show the effect of each *original*
  (undeflated) variable on the scores. Prefer these for interpretation:
  they account for all prior components and give a clearer picture of each
  variable's total contribution.
- **Y-loadings** (``model.y_loadings_``) — how each Y variable contributes
  to the latent structure.

A powerful visualization technique is to **overlay the X and Y loadings** on
the same plot. Since X and Y variables originate from the same physical
system (just artificially separated into cause and effect), their joint
loading plot reveals the interconnections between process conditions and
quality outcomes.

SPE, T², Contributions, and Outlier Detection
----------------------------------------------

These diagnostics work identically to PCA (see :doc:`pca`):

- ``model.spe_`` and ``model.spe_limit()`` — conformity to the X-space
  correlation structure.
- ``model.hotellings_t2_`` and ``model.hotellings_t2_limit()`` — extremity
  within the model.
- ``model.score_contributions()`` — decompose scores back to original
  variables.
- ``model.detect_outliers()`` — combined statistical + robust ESD detection.

Variable Importance in Projection (VIP)
---------------------------------------

VIP scores quantify how much each X variable contributes to the PLS model's
ability to explain Y. The formula weights each variable's loading by the
variance explained by each component:

.. math::

   \text{VIP}_j = \sqrt{K \cdot
       \frac{\sum_{a=1}^{A} r2_a \cdot w_{ja}^2}{\sum_{a=1}^{A} r2_a}}

where :math:`K` is the number of features, :math:`A` the number of
components, :math:`r2_a` the fraction of Y variance explained by component
:math:`a`, and :math:`w_{ja}` the PLS weight for feature :math:`j` in
component :math:`a`.

- Variables with **VIP > 1** are considered important (above average
  contribution).
- Variables with **VIP < 0.5** contribute very little and may be candidates
  for removal.

.. code-block:: python

   pls = PLS(n_components=3).fit(X_scaled, Y_scaled)
   vip_scores = pls.vip()
   print(vip_scores.sort_values(ascending=False))

   # Use fewer components for VIP calculation
   vip_2 = pls.vip(n_components=2)

VIP is also available for PCA models (using loadings instead of weights),
accessed the same way via ``pca.vip()``.

Predictions
-----------

After fitting, ``model.predict(X_new)`` returns a ``Bunch`` with:

- ``y_hat`` — the predicted Y values.
- ``scores`` — the X-scores for the new observations.
- ``spe`` — SPE values for the new observations.
- ``hotellings_t2`` — T² values for the new observations.

The underlying regression relationship is captured in
``model.beta_coefficients_``, which maps directly from (preprocessed) X to
predicted Y.

Model Selection
---------------

Choosing the number of components is even more critical for PLS than for PCA,
because PLS can overfit more aggressively: it will find directions that
correlate X with Y in the training data even if those correlations are
spurious. Cross-validation is essential.

The same PRESS / Wold's criterion approach described in
:doc:`cross_validation` applies. A practical check: if the training R² is
much higher than the test-set R² (gap > 0.15–0.20), overfitting is likely
and you should reduce the number of components.

Cross-Validation and Beta Coefficient Error Bars
-------------------------------------------------

PLS regression coefficients (``model.beta_coefficients_``) represent the best
point estimate from the training data, but they do not convey how certain
those estimates are. The ``cross_validate()`` method provides uncertainty
quantification by refitting the model on data subsets and computing confidence
intervals.

**Three resampling strategies are available:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Strategy
     - Parameter
     - Description
   * - Jackknife (LOO)
     - ``cv="loo"``
     - Leave-one-out: N resamples. Uses the jackknife variance formula
       :math:`\hat{\text{var}}(\beta_j) = \frac{N-1}{N} \sum_{i=1}^{N}
       (\beta_{j,i} - \bar{\beta}_j)^2` with a t-distribution CI. This is
       the default and most common choice in chemometrics.
   * - K-fold
     - ``cv=5``
     - K-fold cross-validation: K resamples. Faster than LOO for large
       datasets.
   * - Bootstrap
     - ``n_bootstrap=200``
     - Resample with replacement B times. CI from percentiles of the
       distribution.

**Basic usage:**

.. code-block:: python

   from process_improve.multivariate.methods import PLS, MCUVScaler

   scaler_x = MCUVScaler().fit(X)
   scaler_y = MCUVScaler().fit(Y)
   X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)

   pls = PLS(n_components=2).fit(X_s, Y_s)
   cv = pls.cross_validate(X_s, Y_s, cv="loo")

   # Beta coefficient uncertainty
   print(cv.beta_mean)        # Mean beta across resamples
   print(cv.beta_std)         # Standard error
   print(cv.beta_ci_lower)    # Lower 95% CI bound
   print(cv.beta_ci_upper)    # Upper 95% CI bound
   print(cv.significant)      # True where CI excludes zero

   # Prediction metrics
   print(cv.q_squared)        # Cross-validated R² (Q²) per Y variable
   print(cv.rmse_cv)          # Cross-validated RMSE per Y variable
   print(cv.press)            # Total PRESS

**Interpreting the results:**

- ``significant`` flags which beta coefficients have confidence intervals that
  do not contain zero — these are the X variables with a statistically
  meaningful relationship to Y at the chosen confidence level.
- ``q_squared`` (Q²) is the cross-validated R². A large gap between training
  R² (``model.r2_cumulative_``) and Q² indicates overfitting.
- ``beta_samples`` (shape: n_resamples × K × M) contains the raw beta
  coefficients from every resample, useful for custom analyses or plotting
  distributions.

**Bootstrap example with custom confidence level:**

.. code-block:: python

   cv = pls.cross_validate(
       X_s, Y_s,
       n_bootstrap=200,
       conf_level=0.99,
       random_state=42,
   )

Missing Data and Troubleshooting
--------------------------------

See the :doc:`pca` page — the same algorithms (TSR, NIPALS, SCP), settings,
and troubleshooting advice apply to PLS models.
