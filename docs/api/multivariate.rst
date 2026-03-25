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
   :members: fit, transform, fit_transform, predict, score, score_contributions, detect_outliers, cross_validate
   :undoc-members:
   :show-inheritance:

TPLS
~~~~

.. autoclass:: TPLS
   :members:
   :undoc-members:
   :show-inheritance:

Analysis
--------

.. autofunction:: vip

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
