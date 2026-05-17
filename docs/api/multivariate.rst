Multivariate Analysis
=====================

.. module:: process_improve.multivariate.methods

Models
------

PCA
~~~

.. autoclass:: PCA
   :members: fit, transform, fit_transform, predict, score, select_n_components, score_contributions, detect_outliers
   :undoc-members:
   :show-inheritance:

PLS
~~~

.. autoclass:: PLS
   :members: fit, transform, fit_transform, predict, score, select_n_components, score_contributions, detect_outliers, cross_validate
   :undoc-members:
   :show-inheritance:

TPLS
~~~~

.. autoclass:: TPLS
   :members:
   :undoc-members:
   :show-inheritance:

MBPLS
~~~~~

Multi-block PLS in the hierarchical / superblock formulation of
Westerhuis, Kourti & MacGregor (1998). Each X-block is preprocessed
independently and weighted by ``1/sqrt(K_b)`` before the inner NIPALS
loop, so blocks of unequal width contribute fairly to the consensus
super-score.

.. autoclass:: MBPLS
   :members: fit, transform, predict, spe_contributions,
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
     *per-variable* and signed. It decomposes a single observation's movement
     in score space back onto the original variables, answering "which
     **variables** explain why this observation sits where it does?". It
     returns one signed value per variable, and it takes an observation's
     score vector as input.

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

.. autofunction:: eigenvalue_summary

.. autofunction:: project_variables

Plots
-----

.. module:: process_improve.multivariate.plots

.. autofunction:: score_plot

.. autofunction:: loading_plot

.. autofunction:: spe_plot

.. autofunction:: t2_plot
