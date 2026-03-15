Multivariate Analysis
=====================

PCA vs PLS
----------

**PCA** (Principal Component Analysis) finds directions of maximum variance in a
single data matrix **X**. Use PCA for:

- Exploratory data analysis and visualization
- Dimensionality reduction
- Outlier detection in process data
- Understanding variable relationships

**PLS** (Projection to Latent Structures) finds directions in **X** that are
maximally correlated with **Y**. Use PLS for:

- Predictive modeling when X has many correlated variables
- Understanding which X variables drive Y responses
- Process optimization (relating process settings to quality outcomes)

Interpreting Model Outputs
--------------------------

Scores
~~~~~~

Scores (``model.scores_``) are the coordinates of each observation in the
latent variable space. Points close together in score space have similar
characteristics in the original variables.

Loadings
~~~~~~~~

Loadings (``model.loadings_`` for PCA, ``model.x_loadings_`` for PLS) describe
how each original variable contributes to each component. Variables with large
absolute loadings on a component are important for that component.

SPE (Squared Prediction Error)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPE (``model.spe_``) measures how well each observation is described by the
model. High SPE indicates the observation does not conform to the correlation
structure captured by the model — it may represent a novel event or
measurement error.

Hotelling's T²
~~~~~~~~~~~~~~

Hotelling's T² (``model.hotellings_t2_``) measures how far each observation is
from the model center in the latent variable space. High T² indicates extreme
but in-model behavior — the observation follows the correlation structure but
with unusual magnitude.

Outlier Detection
-----------------

The ``detect_outliers()`` method combines two approaches:

1. **Statistical limits** — SPE and T² limits at the specified confidence level
2. **Robust ESD test** — Generalized Extreme Studentized Deviate test using
   median and MAD (robust to masking effects)

Results are returned as a list of dicts sorted by severity (most severe first):

.. code-block:: python

   outliers = model.detect_outliers(conf_level=0.95)
   for o in outliers:
       print(f"{o['observation']}: {o['outlier_types']} (severity={o['severity']})")

Score Contributions
-------------------

Score contributions decompose a score-space movement back into the original
variable space, answering: "Which variables caused this observation to score
where it did?"

.. code-block:: python

   # Why does observation 5 differ from the model center?
   contrib = model.score_contributions(model.scores_.iloc[5].values)

   # Why do observations 5 and 10 differ from each other?
   contrib = model.score_contributions(
       model.scores_.iloc[5].values,
       model.scores_.iloc[10].values,
   )

   # T²-weighted contributions (scale by 1/sqrt(eigenvalue))
   contrib = model.score_contributions(
       model.scores_.iloc[5].values, weighted=True
   )
