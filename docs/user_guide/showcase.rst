Model Evaluation and Visualization
==================================

After fitting a PCA or PLS model, the next questions are practical: how many
components should the model keep, how well does it predict, and which
variables drive it? This page is a worked tour of the evaluation and plotting
tools, built around a PLS model that relates a set of process measurements
(``X``) to quality outcomes (``Y``). Each example assumes ``X`` and ``Y`` are
already scaled, for example with ``MCUVScaler``.

Choosing the Number of Components
---------------------------------

Calibration fit always improves as components are added, so it cannot tell
you when to stop. ``PLS.select_n_components`` cross-validates the model and
reports the root-mean-square error of cross-validation (RMSECV) together with
the validated explained variance:

.. code-block:: python

   from process_improve.multivariate import PLS

   result = PLS.select_n_components(X, Y, max_components=8, cv=5)
   print(result.n_components)        # recommended component count
   print(result.rmsecv["total"])     # RMSECV per component count

See :doc:`cross_validation` for the full description, including
``PLS.cross_validate`` for beta-coefficient error bars.

Explained Variance
------------------

Once the model is fitted, ``explained_variance_plot`` shows how much variance
each component captures, both per component and cumulatively:

.. code-block:: python

   model = PLS(n_components=result.n_components).fit(X, Y)
   model.explained_variance_plot()

For PCA the bars refer to variance in the X-block; for PLS they refer to the
Y-block. The same method is available on a fitted PCA model.

Correlation Loadings
--------------------

``correlation_loadings_plot`` places each variable by its correlation with
two components' scores. A variable's squared distance from the origin is the
fraction of its variance explained by those two components, so every variable
lies inside the unit circle. Concentric ellipses mark variance-explained
thresholds:

.. code-block:: python

   model.correlation_loadings_plot(pc_horiz=1, pc_vert=2)

For PLS the X- and Y-variables are overlaid, which reveals how process
variables relate to quality outcomes. The ellipse thresholds are
configurable. The 50% and 100% ellipses are the convention - the outer
ellipse is the unit circle, the inner one marks variables that are well
explained - but any fractions work:

.. code-block:: python

   model.correlation_loadings_plot(variance_ellipses=(0.75, 0.95))

Observed versus Predicted
-------------------------

``predictions_vs_observed_plot`` draws a parity plot of the calibration
predictions against the observed Y, with a ``y = x`` reference line and an
RMSE annotation:

.. code-block:: python

   model.predictions_vs_observed_plot(y_observed=Y, variable="quality")

Points close to the reference line indicate accurate predictions; systematic
departures from it point to model bias.

Regression Coefficients
-----------------------

``coefficient_plot`` shows the PLS regression coefficients as a bar chart,
one bar per X-variable, for a chosen Y-variable:

.. code-block:: python

   model.coefficient_plot(variable="quality")

Tall bars mark the X-variables that most strongly drive the prediction. To
see how *reliable* each coefficient is, pair this plot with the
cross-validated error bars from ``PLS.cross_validate`` (see
:doc:`cross_validation`).

Comparing Two Data Blocks
-------------------------

The RV coefficient and its modified form RV2 measure how much common
structure two matrices, measured on the same observations, share. They are a
multivariate generalization of a squared correlation:

.. code-block:: python

   from process_improve.multivariate import rv_coefficient, rv2_coefficient

   rv_coefficient(X, Y)     # in [0, 1]; 1 means identical configurations
   rv2_coefficient(X, Y)    # modified RV, unbiased for high-dimensional data

Use ``rv2_coefficient`` when the blocks have many more variables than
observations: the ordinary RV coefficient is biased upwards in that regime
and tends towards 1 even for unrelated blocks.
