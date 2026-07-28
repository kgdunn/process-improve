Model inversion and the orthogonal space
=========================================

A fitted PLS model runs forwards: measurements go in, a predicted quality comes
out. Running it backwards asks the question a product developer actually has:
*which inputs would give me the quality I want?* Fixing the target and solving
for the inputs is called **model inversion** (`Jaeckle and MacGregor, 2000
<https://doi.org/10.1016/S0169-7439(99)00058-1>`_).

The answer is rarely a single recipe. A PLS model compresses several correlated
inputs into a few latent directions, and one target value can pin down only one
of them. What is left over is free, so inversion returns a flat set of designs,
every one of which the model says will hit the target. That set is the **null
space**, and its dimension is the number of components minus the rank of the
response.

.. note::

   This page is the API-level summary. The worked example it is drawn from,
   with the geometry, the uncertainty in the null-space direction, and the
   specification-region case, is in `Process Improvement using Data
   <https://learnche.org/pid/latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space>`_.

Inverting a PLS model
---------------------

:meth:`~process_improve.multivariate.methods.PLS.invert` takes the desired
response on its original scale and returns a
:class:`~sklearn.utils.Bunch`:

.. code-block:: python

    import pandas as pd
    from process_improve.multivariate import PLS

    cheese = pd.read_csv("https://openmv.net/file/cheddar-cheese.csv")
    X = cheese[["Acetic", "H2S", "Lactic"]].iloc[4:]
    y = cheese[["Taste"]].iloc[4:]

    pls = PLS(n_components=2).fit(X, y)
    result = pls.invert(y_desired=20.9)

    result.x_new                  # the proposed inputs, in original units
    result.null_space_dimension   # 1: a line of equally valid designs
    result.null_space_basis       # the direction along that line
    result.hotellings_t2          # how far the design sits from the data

The point returned is the **direct-inversion** solution, the one of smallest
score norm. Because a step along the null space is at right angles to it,
Pythagoras gives

.. math::

    \left\| \boldsymbol{\tau}_\text{DI} + s\,\mathbf{g} \right\|^2
      = \left\| \boldsymbol{\tau}_\text{DI} \right\|^2 + s^2

so the norm grows by :math:`s^2` and is least when no step is taken.

.. warning::

   Smallest score norm is not smallest Hotelling's :math:`T^2`. :math:`T^2`
   divides each score by that score's standard deviation before squaring, so it
   is a weighted sum of squares and its least value sits slightly off the
   direct-inversion point. For the cheese model above the two differ by a step
   of :math:`-0.103`. The gap widens as the score spreads become more unequal.

Walking the null space
----------------------

Pass ``null_space_coordinates`` to move along the free directions. The predicted
response does not change; the recipe does.

.. code-block:: python

    for step in (-1.0, 0.0, 1.0):
        moved = pls.invert(y_desired=20.9, null_space_coordinates=[step])
        print(moved.x_new.round(2).to_list(), moved.hotellings_t2.round(2))
    # [4.95, 6.1, 1.33]  1.63
    # [5.52, 5.56, 1.4]  0.06
    # [6.09, 5.02, 1.46] 2.44

That freedom is what you spend on a secondary goal: cost, supplier availability,
or staying inside a regulatory window. It is free in terms of the predicted
response, but not in terms of how much the data support the design, which is
what the rising :math:`T^2` records.

Choosing the number of components
---------------------------------

Inversion places a different demand on the model than prediction does. To
rebuild a full set of inputs from one target number, the model has to describe
the **X**-space well enough to map a score back into it, not merely describe the
response. A one-component model spans only a line in the input space, so it
returns exactly one recipe and the question of equivalent designs never arises.

Cross-validation chosen for predictive accuracy may therefore keep fewer
components than inversion needs. Choosing more is a deliberate decision, made on
different grounds, and worth stating as such.

The orthogonal space, and why it is the same set
------------------------------------------------

:class:`~process_improve.multivariate.methods.OPLS` separates **X** into a
single predictive component and one or more Y-orthogonal components:

.. math::

    \mathbf{X} = \mathbf{t}_\text{p} \mathbf{p}_\text{p}^T
      + \mathbf{T}_\text{o} \mathbf{P}_\text{o}^T + \mathbf{E},
    \qquad
    \mathbf{y} = q_\text{p} \mathbf{t}_\text{p} + \mathbf{f}

The orthogonal scores do not appear in the equation for **y**, so moving along
them changes **X** and leaves the prediction alone. That is the same property
that defines the null space of an inverted PLS model, and for a single response
the two subspaces are provably identical (`García-Carrión et al., 2025
<https://doi.org/10.1002/cem.70057>`_).

The practical consequence is that inverting an O-PLS model needs no linear
algebra. One equation in one unknown gives the predictive score by division:

.. code-block:: python

    from process_improve.multivariate import OPLS

    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    opls_result = opls.invert(y_desired=20.9)

    opls_result.predictive_score        # y_desired / q_p
    opls_result.orthogonal_space_basis  # the freedom, handed back as an axis

Both routes describe the same set of designs. They report different
representative points on it: PLS the point of smallest score norm, O-PLS the
point whose orthogonal score is zero.

.. note::

   The equivalence is proved for a single response. ``OPLS`` therefore accepts
   one response only. Multi-response inversion still works through
   ``PLS.invert``, which returns a null space of dimension
   :math:`A - \text{rank}(\mathbf{Y})`; extending the equivalence proof to
   several responses is open work.

How well is the direction determined?
-------------------------------------

The null space is exactly what the algebra says it is for a *given* model. That
is not the same as saying the data pin it down. Its direction depends on the
ratio of the y-loadings, and the later loadings are often the ones
cross-validation did not keep.

For the cheese model above, a bootstrap over the calibration set puts the
direct-inversion **point** on firm ground while leaving the **direction** poorly
determined: the resampled null space sits a median of 21 degrees away from the
reported one, and beyond 45 degrees in 15% of resamples. The asymmetry is worth
checking before acting on a design reached by a long walk along the null space:

.. code-block:: python

    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(2000):
        sample = X.join(y).sample(len(X), replace=True, random_state=rng.integers(1 << 32))
        refit = PLS(n_components=2).fit(sample[X.columns], sample[y.columns])
        ...  # compare refit.invert(20.9) against the model fitted on all the data

A design proposed at the direct-inversion solution rests on firmer ground than
one reached by stepping a long way along the null space.

Specification regions
---------------------

A product is usually accepted over a range rather than at a point. Sweeping the
target across that range sweeps its null space across the score plot, and the
swept lines fill out a **multivariate specification region**. Bounding it by the
:math:`T^2` limit keeps it inside the space the data support (`Paris et al.,
2021 <https://doi.org/10.3389/frans.2021.729732>`_).

.. warning::

   Reporting such a region as one range per input describes a *box*, and the box
   is not the region. The region is flat, since every point on it is rebuilt
   from the scores, while the box is a solid. For the cheese model, every one of
   the eight corners of that box satisfies all three ranges and none is an
   acceptable lot: six predict a response outside the window, and the other two
   predict an acceptable response from a recipe well beyond the :math:`T^2`
   limit. Treat per-input limits as a summary of the region, not as the
   specification.

See also
--------

- :class:`~process_improve.multivariate.methods.PLS`
- :class:`~process_improve.multivariate.methods.OPLS`
