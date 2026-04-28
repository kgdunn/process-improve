Multivariate Analysis
=====================

.. module:: process_improve.multivariate.methods

Models
------

PCA
~~~

.. autoclass:: PCA
   :members: fit, predict, score, select_n_components, score_contributions, detect_outliers
   :undoc-members:
   :show-inheritance:

PLS
~~~

.. autoclass:: PLS
   :members: fit, predict, score_contributions, detect_outliers
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

Preprocessing
-------------

.. autoclass:: MCUVScaler
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: center

.. autofunction:: scale

Plots
-----

.. module:: process_improve.multivariate.plots

.. autofunction:: score_plot

.. autofunction:: loading_plot

.. autofunction:: spe_plot

.. autofunction:: t2_plot
