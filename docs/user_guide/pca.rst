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

- **T** (N × A) - scores: the coordinates of each observation in the
  reduced space.
- **P** (K × A) - loadings: the direction vectors that define the model
  plane.
- **E** (N × K) - residuals: what the model does not explain.
- A - the number of components retained.

Geometrically, PCA fits a best-fit hyperplane through the cloud of
observations in K-dimensional space. Each observation is projected
perpendicularly onto this plane; the projected coordinates are the scores.
Minimizing the perpendicular residual distance is equivalent to maximizing
the variance captured in the scores - the two formulations yield the same
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
(interactions, squared terms, ratios) to the raw data matrix - the model
will sort out which are informative.

Interpreting Scores
-------------------

Scores (``model.scores_``) are the coordinates of each observation in the
latent variable space. They reveal the dominant patterns in the data:

- **Clusters** - observations that are similar in the original K variables
  will be close together in score space.
- **Outliers** - observations far from the main cluster may represent unusual
  operating conditions or measurement errors.
- **Time trends** - when rows are ordered by time, gradual drifts or sudden
  shifts become visible as trajectories in the score plot.
- **Groups** - coloring scores by a categorical variable (grade, shift,
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
- The **sign of a loading** can flip between different runs or software - the
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
- **High SPE** means the observation *breaks* the correlation structure - a
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
  with **unusual magnitude** - it is extreme *within* the model, not outside
  it.
- On a score plot of component *a* vs *b*, the T² limit traces an
  **elliptical boundary**. This single ellipse replaces examining
  A(A−1)/2 pairwise scatterplots.

Use ``model.t2_plot()`` and ``model.hotellings_t2_limit()`` for monitoring.

Quality of Representation (cos²)
--------------------------------

The squared cosine, or *cos²*, reports how well each observation is represented
on each component. It is the squared score divided by that observation's total
variation budget:

.. math::

   \cos^2_{i,a} = \frac{t_{i,a}^2}{\sum_{a=1}^{A} t_{i,a}^2 + \text{SPE}_i^2}

- cos² lies between 0 and 1. A value close to 1 means the component captures
  most of that observation's variation.
- Summed across all components, the cos² values plus the residual fraction
  add up to 1.
- For PCA, where the loadings are orthonormal, the denominator equals the
  squared distance of the observation from the origin, so cos² matches the
  classical factor-analysis definition.

cos² completes the trio of per-observation diagnostics: **Hotelling's T²**
measures distance *within* the model plane, **SPE** measures distance *to* it,
and **cos²** says how much of an observation's total variation a given
component accounts for.

.. code-block:: python

   model.squared_cosine()                  # all components
   model.squared_cosine(n_components=2)     # first two components only

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

Observation Contributions
-------------------------

Observation contributions answer a different question: *"Which observations
shaped this component?"* The contribution of observation *i* to component *a*
is its squared score as a fraction of the total over all observations:

.. math::

   \text{contribution}_{i,a} = \frac{t_{i,a}^2}{\sum_{i=1}^{N} t_{i,a}^2}

- Each component column sums to 1, so a contribution well above the average
  :math:`1/N` flags an observation that strongly drives that component.
- Use it to find which observations a component is built on, and whether a
  single observation is dominating it.

This is **not** the same as *Score Contributions*, despite the similar name.
``score_contributions`` is *per-variable* and signed: it decomposes one
observation's position back onto the original variables.
``observation_contributions`` is *per-observation* and non-negative: it reports
each observation's share of a component's variation. The two are orthogonal
views of the same score matrix; one decomposes across variables, the other
across observations.

.. code-block:: python

   model.observation_contributions()              # all components
   model.observation_contributions(n_components=2)

Outlier Detection
-----------------

The ``detect_outliers()`` method combines two complementary approaches:

1. **Statistical limits** - SPE and T² limits at a specified confidence level
2. **Robust ESD test** - Generalized Extreme Studentized Deviate test using
   median and MAD, which is robust to masking effects where multiple outliers
   hide each other

Results are returned as a list of dicts sorted by severity (most severe
first):

.. code-block:: python

   outliers = model.detect_outliers(conf_level=0.95)
   for o in outliers:
       print(f"{o['observation']}: {o['outlier_types']} (severity={o['severity']})")

Variable Importance in Projection (VIP)
---------------------------------------

VIP scores quantify how much each variable contributes to the PCA model. For
PCA, the loadings matrix is used as the weight matrix:

.. math::

   \text{VIP}_j = \sqrt{K \cdot
       \frac{\sum_{a=1}^{A} r2_a \cdot p_{ja}^2}{\sum_{a=1}^{A} r2_a}}

where :math:`p_{ja}` is the loading for feature :math:`j` on component
:math:`a`, and :math:`r2_a` is the fraction of variance explained by that
component.

- Variables with **VIP > 1** contribute more than average to the model.
- Variables with **VIP < 0.5** contribute very little.

.. code-block:: python

   model = PCA(n_components=3).fit(X_scaled)
   vip_scores = model.vip()
   print(vip_scores.sort_values(ascending=False))

   # Compute VIP using only the first 2 components
   vip_2 = model.vip(n_components=2)

Supplementary Variables
-----------------------

A *supplementary* (or passive) variable is an extra column that did not take
part in fitting the model but was measured on the same observations. Projecting
it shows how it relates to the model without letting it influence the model.

Each supplementary variable is represented by its correlation with each
component's scores, the standard representation for passive quantitative
variables. This is the column-wise counterpart of ``transform()``, which
projects supplementary *rows* (new observations).

A common use is to build a PCA on process variables and then project a quality
or outcome variable onto it, to see which components, and therefore which
process patterns, line up with that outcome.

.. code-block:: python

   # passive_columns: a DataFrame with the same rows as the training data
   model.project_variables(passive_columns)

Eigenvalue Summary
------------------

``model.eigenvalue_summary()`` collects the per-component variance information
into one tidy table, with one row per component and three columns:

- ``eigenvalue`` - the variance of that component's score column.
- ``percent_variance`` - the share of variance the component explains.
- ``cumulative_percent`` - the running total of ``percent_variance``.

It gathers ``explained_variance_``, ``r2_per_component_`` and
``r2_cumulative_`` into a single view, and is the numeric companion of
``model.explained_variance_plot()``. The cumulative column is a quick input to
the component-count decision discussed next.

.. code-block:: python

   model.eigenvalue_summary()

Model Selection
---------------

R² always increases as components are added, so it cannot tell you when to
stop. The cross-validated Q² metric *decreases* once the model begins
fitting noise rather than systematic structure. The point where Q² stops
improving indicates the useful number of components.

Use ``PCA.select_n_components()`` for automated selection via PRESS
cross-validation with Wold's criterion (see :doc:`cross_validation` for
details). Remember that the answer is never exact - examine one or two
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
4. **Monitor SPE and T²** - compare each new observation's SPE and T² against
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

Real process data often has missing values - sensor failures, skipped lab
analyses, or variables recorded at different frequencies. The PCA
implementation supports several iterative algorithms for fitting models in
the presence of missing data:

- **TSR** (Trimmed Scores Regression) - generally the best default choice;
  uses the available data to estimate scores, then reconstructs missing
  values.
- **NIPALS** - the classical algorithm; handles moderate amounts of missing
  data well.
- **SCP** - Single Component Projection; a simpler alternative.

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
       missing data is very large (>40 %), the algorithm may struggle - try
       removing rows or columns with excessive missingness first.
   * - Shape mismatch in ``predict()``
     - The new data must have the same columns (K) as the training data, in
       the same order. If you used ``MCUVScaler``, call
       ``scaler.transform(X_new)`` to apply the training centering/scaling.
   * - Very different results with different software
     - Check that the same preprocessing (centering, scaling) was applied.
       Also note that the sign of a component can flip - this is normal and
       does not affect interpretation.
   * - First component explains almost all variance
     - The data may not have been centered. Without centering, the first
       component captures the mean rather than variation around it.
