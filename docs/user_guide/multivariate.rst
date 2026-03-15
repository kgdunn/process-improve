Multivariate Analysis
=====================

Latent variable methods summarize high-dimensional, correlated data into a
small number of underlying variables that capture the dominant structure.
A latent variable is an unobservable quantity — your overall health, for
example — that manifests through measurable indicators (blood pressure,
cholesterol, heart rate). In process data, a single underlying phenomenon
such as a feed quality change can shift dozens of correlated sensors
simultaneously. Latent variable models recover those phenomena from the
measurements.

When to Use These Methods
-------------------------

Latent variable methods are the right tool when your data has one or more of
these characteristics (common in process industries):

- **Many correlated variables** — traditional regression struggles with
  collinearity, but latent variable methods thrive on it.
- **More variables than observations** (K > N) — ordinary least squares
  cannot be computed, but PCA/PLS handle this naturally.
- **Missing values** — sensors fail, lab samples are skipped. The algorithms
  in this package handle incomplete data natively.
- **Low signal-to-noise ratio** — by separating systematic variation from
  noise, latent variable models act as multivariate filters.
- **Need for visualization** — score plots and loading plots reveal structure
  that is invisible in univariate views.

Five common applications drive their use:

1. **Process understanding** — confirm existing knowledge or discover
   unexpected variable relationships through score and loading plots.
2. **Troubleshooting** — after a problem occurs, screen variables to isolate
   the most relevant ones using contribution plots.
3. **Optimization** — move along favorable directions in the latent variable
   space to improve yield, quality, or throughput.
4. **Predictive modeling** — build inferential sensors that predict
   hard-to-measure quality variables from readily available process data.
5. **Process monitoring** — extend univariate control charts (Shewhart, CUSUM)
   to the multivariate case with SPE and Hotelling's T² charts.

Available Methods
-----------------

This package provides three multivariate methods, each suited to different
data structures and goals:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Method
     - Type
     - Use when ...
   * - :doc:`PCA <pca>`
     - Unsupervised
     - You have a single data matrix **X** and want to explore, monitor, or
       reduce dimensionality.
   * - :doc:`PLS <pls>`
     - Supervised
     - You have predictor variables **X** and response variables **Y** and
       want to build a predictive or explanatory model.
   * - :doc:`TPLS <tpls>`
     - Multi-block
     - Your data is naturally organized in T-shaped blocks (materials,
       formulations, conditions, quality) as in batch processes.

.. toctree::
   :maxdepth: 2
   :hidden:

   pca
   pls
   tpls
