Principal Component Analysis (PCA)
===================================

PCA finds the directions of maximum variance in a single data matrix **X**.
It is the workhorse of multivariate exploration, process monitoring, and
outlier detection.

Mathematical Background
-----------------------

PCA decomposes an (N × K) data matrix into scores, loadings, and residuals:

.. math::

   \mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}

where:

- **T** (N × A) — scores: the coordinates of each observation in the
  reduced space.
- **P** (K × A) — loadings: the direction vectors that define the model
  plane.
- **E** (N × K) — residuals: what the model does not explain.
- A — the number of components retained.

Geometrically, PCA fits a best-fit hyperplane through the cloud of
observations in K-dimensional space. Each observation is projected
perpendicularly onto this plane; the projected coordinates are the scores.
Minimizing the perpendicular residual distance is equivalent to maximizing
the variance captured in the scores — the two formulations yield the same
solution.

Preprocessing
-------------

Preprocessing decisions have a large influence on the model.

**Centering** subtracts the column means so that the model explains
*deviations* from average behavior. Mean centering is the default; median
centering is a robust alternative that simultaneously highlights outliers.

**Scaling** equalizes the influence of variables measured in different units
or ranges. Unit-variance scaling (dividing by the standard deviation) is the
most common choice. When combined with mean centering this is called
*autoscaling*. Use :class:`~process_improve.multivariate.methods.MCUVScaler`
for sklearn-compatible autoscaling:

.. code-block:: python

   from process_improve.multivariate.methods import PCA, MCUVScaler

   scaler = MCUVScaler()
   X_scaled = scaler.fit_transform(X)
   model = PCA(n_components=3).fit(X_scaled)

**Transformations** (log, square root) can reduce the effect of extreme
measurements or improve linearity. Consider adding derived columns
(interactions, squared terms, ratios) to the raw data matrix — the model
will sort out which are informative.

Interpreting Scores
-------------------

Scores (``model.scores_``) are the coordinates of each observation in the
latent variable space. They reveal the dominant patterns in the data:

- **Clusters** — observations that are similar in the original K variables
  will be close together in score space.
- **Outliers** — observations far from the main cluster may represent unusual
  operating conditions or measurement errors.
- **Time trends** — when rows are ordered by time, gradual drifts or sudden
  shifts become visible as trajectories in the score plot.
- **Groups** — coloring scores by a categorical variable (grade, shift,
  equipment ID) can reveal systematic differences.

Use ``model.score_plot()`` for quick visualization.

Interpreting Loadings
---------------------

Loadings (``model.loadings_``) describe how each original variable
contributes to each component:

- Variables with **large absolute loadings** on a component are important for
  that component.
- Variables that are **positively correlated** cluster together in the
  loading plot; variables that are **negatively correlated** appear on
  opposite sides.
- The **sign of a loading** can flip between different runs or software — the
  sign of the component as a whole is arbitrary, so always interpret loading
  *patterns* rather than individual signs.

Use ``model.loading_plot()`` for quick visualization.

SPE (Squared Prediction Error)
------------------------------

SPE (``model.spe_``) measures how well each observation conforms to the
correlation structure captured by the model.  It is the sum of squared
residuals for that observation:

.. math::

   \text{SPE}_i = \mathbf{e}_i^T \mathbf{e}_i

- SPE = 0 means the observation lies exactly on the model plane.
- **High SPE** means the observation *breaks* the correlation structure — a
  new event or fault has introduced variation that the model's A components
  do not describe.
- Column-wise residuals give the per-variable R² (``model.r2_per_variable_``),
  showing which variables are well explained and which are not.

Use ``model.spe_plot()`` and ``model.spe_limit()`` for monitoring.

Hotelling's T²
--------------

Hotelling's T² (``model.hotellings_t2_``) measures how far each
observation's projection is from the center of the model plane:

.. math::

   T^2_i = \sum_{a=1}^{A} \frac{t_{i,a}^2}{s_a^2}

where :math:`s_a^2` is the variance of the *a*-th score column.

- T² is always non-negative and follows an F-distribution, so confidence
  limits (e.g., 95 %) can be computed analytically.
- **High T²** means the observation follows the correlation structure but
  with **unusual magnitude** — it is extreme *within* the model, not outside
  it.
- On a score plot of component *a* vs *b*, the T² limit traces an
  **elliptical boundary**. This single ellipse replaces examining
  A(A−1)/2 pairwise scatterplots.

Use ``model.t2_plot()`` and ``model.hotellings_t2_limit()`` for monitoring.

Score Contributions
-------------------

Score contributions decompose a score-space movement back into the original
variable space, answering: *"Which variables caused this observation to score
where it did?"*

Each score value :math:`t_{i,a}` can be written as a sum of K contributions
:math:`x_{i,k} \, p_{k,a}`, one per original variable. Plotting these as a
bar chart reveals the dominant drivers.

The reference point matters: contributions always measure the difference
*from* one point *to* another. The default reference is the model center
(the origin after centering), but you can compare any two points or groups:

.. code-block:: python

   # Why does observation 5 differ from the model center?
   contrib = model.score_contributions(model.scores_.iloc[5].values)

   # Why do observations 5 and 10 differ from each other?
   contrib = model.score_contributions(
       model.scores_.iloc[5].values,
       model.scores_.iloc[10].values,
   )

   # T²-weighted contributions (scale by 1/sqrt(eigenvalue))
   contrib = model.score_contributions(model.scores_.iloc[5].values, weighted=True)

Outlier Detection
-----------------

The ``detect_outliers()`` method combines two complementary approaches:

1. **Statistical limits** — SPE and T² limits at a specified confidence level
2. **Robust ESD test** — Generalized Extreme Studentized Deviate test using
   median and MAD, which is robust to masking effects where multiple outliers
   hide each other

Results are returned as a list of dicts sorted by severity (most severe
first):

.. code-block:: python

   outliers = model.detect_outliers(conf_level=0.95)
   for o in outliers:
       print(f"{o['observation']}: {o['outlier_types']} (severity={o['severity']})")

Model Selection
---------------

R² always increases as components are added, so it cannot tell you when to
stop. The cross-validated Q² metric *decreases* once the model begins
fitting noise rather than systematic structure. The point where Q² stops
improving indicates the useful number of components.

Use ``PCA.select_n_components()`` for automated selection via PRESS
cross-validation with Wold's criterion (see :doc:`cross_validation` for
details). Remember that the answer is never exact — examine one or two
components beyond and before the suggestion, and consider the interpretability
of each additional component in the context of your application.

Process Monitoring
------------------

PCA-based process monitoring is one of the most common industrial
applications. The workflow:

1. **Build a reference model** on data from normal, in-control operation.
2. **Preprocess new observations** using the *same* centering and scaling
   learned from the training data (e.g., the same ``MCUVScaler`` instance).
3. **Project new observations** into the model with ``model.predict(X_new)``.
4. **Monitor SPE and T²** — compare each new observation's SPE and T² against
   the control limits:

   .. code-block:: python

      result = model.predict(X_new)

      spe_lim = model.spe_limit(conf_level=0.95)
      t2_lim = model.hotellings_t2_limit(conf_level=0.95)

      spe_violations = result.spe_ > spe_lim
      t2_violations = result.hotellings_t2_ > t2_lim

5. **Investigate violations** using score contributions to identify which
   original variables are responsible.

A high SPE signals a *new type* of event (the correlation structure has
changed). A high T² signals an *extreme* version of normal behavior.
Both can occur simultaneously.

Missing Data
------------

Real process data often has missing values — sensor failures, skipped lab
analyses, or variables recorded at different frequencies. The PCA
implementation supports several iterative algorithms for fitting models in
the presence of missing data:

- **TSR** (Trimmed Scores Regression) — generally the best default choice;
  uses the available data to estimate scores, then reconstructs missing
  values.
- **NIPALS** — the classical algorithm; handles moderate amounts of missing
  data well.
- **SCP** — Single Component Projection; a simpler alternative.

Enable missing data handling by passing a settings dictionary:

.. code-block:: python

   model = PCA(
       n_components=3,
       missing_data_settings={
           "md_method": "tsr",
           "md_tol": 1e-6,  # convergence tolerance
           "md_max_iter": 200,  # maximum iterations
       },
   )
   model.fit(X_with_nans)

   # Check convergence
   print(model.fitting_info_["iterations"])
   print(model.fitting_info_["final_error"])

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Likely cause and remedy
   * - ``LinAlgError: Singular matrix``
     - One or more columns are constant (zero variance) or are exact linear
       combinations of other columns. Remove constant columns and check for
       duplicate or perfectly correlated variables.
   * - Missing-data algorithm does not converge
     - Increase ``md_max_iter`` or relax ``md_tol``. If the fraction of
       missing data is very large (>40 %), the algorithm may struggle — try
       removing rows or columns with excessive missingness first.
   * - Shape mismatch in ``predict()``
     - The new data must have the same columns (K) as the training data, in
       the same order. If you used ``MCUVScaler``, call
       ``scaler.transform(X_new)`` to apply the training centering/scaling.
   * - Very different results with different software
     - Check that the same preprocessing (centering, scaling) was applied.
       Also note that the sign of a component can flip — this is normal and
       does not affect interpretation.
   * - First component explains almost all variance
     - The data may not have been centered. Without centering, the first
       component captures the mean rather than variation around it.
