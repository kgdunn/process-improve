Multivariate Analysis
=====================

.. module:: process_improve.multivariate.methods

Models
------

PCA
~~~

.. autoclass:: PCA
   :members: fit, transform, fit_transform, predict, score, select_n_components, score_contributions, group_contributions, detect_outliers
   :undoc-members:
   :show-inheritance:

PLS
~~~

.. autoclass:: PLS
   :members: fit, transform, fit_transform, predict, score, select_n_components, score_contributions, group_contributions, detect_outliers, cross_validate
   :undoc-members:
   :show-inheritance:

PLS-DA
~~~~~~

PLS discriminant analysis: PLS regression against a one-hot class indicator, with
the decision rule, the classifier diagnostics and the permutation test on top.
Everything :class:`PLS` offers is inherited, so a fitted ``PLSDA`` also has scores,
loadings, VIP, Hotelling's T2 and SPE.

.. autoclass:: PLSDA
   :members:
   :show-inheritance:

TPLS
~~~~

.. autoclass:: TPLS
   :members:
   :undoc-members:
   :show-inheritance:

A T-shaped model carries its response inside ``X["Y"]``, so
:func:`~sklearn.model_selection.cross_val_score` is called without a ``y`` and a
scorer *string* such as ``scoring="r2"`` cannot be honoured: sklearn's
``_Scorer`` needs a ``y_true`` it was never given, the call fails before TPLS is
reached, and every fold is recorded as ``NaN``. Build the scorer with
:func:`make_tpls_scorer` instead; sklearn passes a callable ``scoring=`` through
untouched.

.. autofunction:: make_tpls_scorer

ASCA
~~~~

ANOVA-Simultaneous Component Analysis: partition a response matrix by its design terms,
then give each term's effect matrix its own PCA. This is the bridge between
:mod:`process_improve.experiments` and the latent-variable models: it answers which
*factor* owns which direction of multivariate variation, whether that is more than
chance, and which variables carry it.

.. autoclass:: ASCA
   :members:
   :show-inheritance:

MBPLS
~~~~~

Multi-block PLS in the hierarchical / superblock formulation of
Westerhuis, Kourti & MacGregor (1998). Each X-block is preprocessed
independently and weighted by ``1/sqrt(K_b)`` before the inner NIPALS
loop, so blocks of unequal width contribute fairly to the consensus
super-score.

.. autoclass:: MBPLS
   :members: fit, transform, predict, select_n_components, spe_contributions,
             block_spe_limit, super_spe_limit, display_results,
             super_score_plot, super_weights_bar_plot,
             predictions_vs_observed_plot
   :undoc-members:
   :show-inheritance:

.. autofunction:: randomization_test_mbpls

MBPCA
~~~~~

Multi-block PCA / consensus-PCA. Same dict-of-DataFrames API as
:class:`MBPLS`; no Y-block.

.. autoclass:: MBPCA
   :members: fit, transform, predict, spe_contributions,
             block_spe_limit, super_spe_limit, display_results,
             super_score_plot, super_loadings_bar_plot
   :undoc-members:
   :show-inheritance:

Analysis
--------

.. autofunction:: rv_coefficient

.. autofunction:: rv2_coefficient

Containers
----------

.. autoclass:: BlockSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DataFrameDict
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

.. autoclass:: MCUVScaler
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: center

.. autofunction:: scale

Diagnostics
-----------

These functions work with fitted :class:`PCA` and :class:`PLS` models. Each is
also bound as a convenience method on the model after :meth:`fit`.

.. note::

   **Two different "contributions" diagnostics.** The library has two methods
   whose names both contain "contributions"; they are not interchangeable and
   answer different questions about the same fitted score matrix.

   * :meth:`PCA.score_contributions` (and :meth:`PLS.score_contributions`) is
     *per-variable* and signed. It splits each score into the K terms
     :math:`x_{ik} R_{ka}` that form it, answering "which **variables**
     explain why this observation sits where it does?". It takes the
     preprocessed data and returns a sample-by-variable table whose rows sum
     to the score being decomposed.

   * :func:`observation_contributions` is *per-observation* and non-negative.
     It reports each observation's share of a component's total inertia
     (:math:`t_{ia}^2 / \sum_i t_{ia}^2`), answering "which **observations**
     most strongly shape this component?". It returns a sample-by-component
     table whose columns each sum to 1, and it takes no input beyond the
     fitted model.

   In short, ``score_contributions`` decomposes *across variables* while
   ``observation_contributions`` decomposes *across observations*.

.. autofunction:: vip

.. autofunction:: squared_cosine

.. autofunction:: observation_contributions

.. autofunction:: score_contributions

.. autofunction:: group_contributions

.. autofunction:: eigenvalue_summary

.. autofunction:: project_variables

Warnings
--------

.. currentmodule:: process_improve.multivariate

Both classes are importable from ``process_improve.multivariate`` as well as
from :mod:`process_improve.multivariate.methods`, so a ``filterwarnings`` entry
never has to name a private module.

.. autoclass:: SpecificationWarning
   :show-inheritance:

.. autoclass:: UncentredDataWarning
   :show-inheritance:

Plots
-----

.. module:: process_improve.multivariate.plots

.. autofunction:: score_plot

.. autofunction:: loading_plot

.. autofunction:: spe_plot

.. autofunction:: t2_plot

.. autofunction:: explained_variance_plot

.. autofunction:: correlation_loadings_plot

.. autofunction:: predictions_vs_observed_plot

.. autofunction:: coefficient_plot

.. autofunction:: confusion_matrix_plot
.. autofunction:: effect_summary_plot
