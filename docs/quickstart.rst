Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install process-improve

PCA Example
-----------

.. code-block:: python

   import pandas as pd
   from process_improve.multivariate.methods import PCA, MCUVScaler

   # Load and scale data
   X = pd.read_csv("your_data.csv", index_col=0)
   scaler = MCUVScaler().fit(X)
   X_scaled = scaler.transform(X)

   # Fit model
   pca = PCA(n_components=3).fit(X_scaled)

   # Results
   pca.scores_            # Score matrix (N x A)
   pca.loadings_          # Loading matrix (K x A)
   pca.r2_cumulative_     # Cumulative R² per component

   # Diagnostics
   pca.detect_outliers()
   pca.score_contributions(pca.scores_.iloc[0].values)

   # Plots
   pca.score_plot()
   pca.loading_plot()
   pca.spe_plot()
   pca.t2_plot()

PLS Example
-----------

.. code-block:: python

   from process_improve.multivariate.methods import PLS, MCUVScaler

   # Scale X and Y separately
   scaler_x = MCUVScaler().fit(X)
   scaler_y = MCUVScaler().fit(Y)
   X_scaled = scaler_x.transform(X)
   Y_scaled = scaler_y.transform(Y)

   # Fit model
   pls = PLS(n_components=3).fit(X_scaled, Y_scaled)

   # Predict new observations
   result = pls.predict(scaler_x.transform(X_new))
   result.y_hat           # Predicted Y
   result.spe             # SPE diagnostics
   result.hotellings_t2   # Hotelling's T² diagnostics

   # Diagnostics
   pls.detect_outliers()
   pls.score_contributions(pls.scores_.iloc[0].values)

Component Selection
-------------------

Use cross-validation to select the number of PCA components:

.. code-block:: python

   result = PCA.select_n_components(X_scaled, max_components=10)
   print(f"Recommended: {result.n_components} components")
   print(f"PRESS ratios: {result.press_ratio}")
