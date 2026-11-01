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

    result.x_new  # the proposed inputs, in original units
    result.null_space_dimension  # 1: a line of equally valid designs
    result.null_space_basis  # the direction along that line
    result.hotellings_t2  # how far the design sits from the data

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

    opls_result.predictive_score  # y_desired / q_p
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

Inversion as an optimisation: cost, ingredients and safety limits
-----------------------------------------------------------------

A specification region says which inputs are acceptable; a product developer
then has to choose one. Jaeckle and MacGregor chose by hand, "the most
economical or most energy-efficient" point of the window.
:meth:`~process_improve.multivariate.methods.PLS.optimal_inversion` makes that
choice an optimisation, for PLS and TPLS models alike:

.. math::

    \begin{aligned}
    \min_{\mathbf{x}} \quad & \mathbf{c}^T \mathbf{x}
        \;\text{ or }\; \textstyle\sum_k |\hat{y}_k - y_k^*| / s_k \\
    \text{subject to} \quad & \text{each } \hat{y}_k \text{ within its window} \\
    & \text{linear rules on } \mathbf{x}
        \text{ (a mixture that sums to 1, a safety or legal limit)} \\
    & T^2(\mathbf{x}) \le T^2_{\text{lim}}, \quad
      \text{SPE}(\mathbf{x}) \le \text{SPE}_{\text{lim}} \\
    & \text{at most } k \text{ inputs of a group, each at a minimum amount when used.}
    \end{aligned}

The :math:`T^2` and SPE limits play the part of "inside the tested region": the
answer must look like the data the model was built from. A latent-variable
model's scores, predictions and residuals are affine in its inputs, so
:func:`~process_improve.multivariate.methods.affine_model` reads them once,
exactly, and checks them against the model's own ``diagnose``. Everything is
then linear except :math:`T^2`, a convex quadratic, and SPE, the length of an
affine map: without the counting rules the problem is convex and its optimum is
global. With them, outer approximation alternates a mixed-integer master, which
chooses the inputs used, with the convex search for each choice, until the
master's bound meets the best answer: the answer is *proven* optimal.

.. code-block:: python

    from process_improve.multivariate import InversionRules, LinearRule

    rules = InversionRules(
        costs={"Acetic": 1.0, "H2S": 3.0, "Lactic": 8.0},  # per unit of each input
        windows={"Taste": (20.0, 30.0)},  # an acceptable taste
        rules=(LinearRule("sulfide limit", {"H2S": 1.0}, upper=8.0),),
    )
    result = pls.optimal_inversion(rules)

    result.x_new  # the cheapest inputs the model vouches for
    result.objective  # their cost
    result.constraints  # every rule, with its shadow price

For a mixture, give the total as a rule (``LinearRule("mixture", {...: 1.0},
lower=1.0, upper=1.0)``) and each ingredient bounds of 0 to 1. The number of
ingredients is a counting rule: ``groups={"all": names}``,
``max_count={"all": 3}``, ``min_amount={"all": 0.05}``. A TPLS model's
materials are groups already, and each material's lot fractions sum to 1 unless
``closure=False``. With ``targets`` instead of ``costs`` the answer is the
closest point to the targets; equally close points fill the null space, and the
one returned is the most typical of them, the one of least :math:`T^2`.

Reading the model costs one ``diagnose`` call; to solve many problems on one
model, read it once with :func:`~process_improve.multivariate.methods.affine_model`
and pass the maps in place of the model. ``AffineModel.to_dict`` stores them as
plain JSON, and ``AffineModel.from_dict`` reads them back, so a search can run
later without the fitted model.

When no inputs meet the rules, ``result.message`` says why: either the windows
cannot be met within the inputs' bounds at all, or they can only far from the
data, and then by how many times each limit would be exceeded.

Alternatives: the next best answers, each with other inputs
-----------------------------------------------------------

A product developer rarely wants one answer. Alternatives that use different
inputs protect against an input that becomes unavailable or restricted, and
they make a more informative next round of experiments than several points near
one optimum. With counting rules, ``alternatives`` asks for them:

.. code-block:: python

    rules = InversionRules(
        costs=costs, windows=windows, rules=(mixture_total,),
        groups={"all": names}, max_count={"all": 3}, min_amount={"all": 0.05},
    )
    result = pls.optimal_inversion(rules, alternatives=2, leave_out=1)

    result.used  # the inputs of the best answer
    [r.used for r in result.alternatives]  # each alternative's inputs, best first
    [r.objective - result.objective for r in result.alternatives]  # what each one costs extra

Each alternative is found by a *no-good cut* on the master problem: if an answer
uses the counted inputs :math:`S`, every later answer must meet

.. math::

    \sum_{i \in S} u_i \le |S| - \ell,

where :math:`u_i` is 1 when input :math:`i` is used and :math:`\ell` is
``leave_out``: it leaves out at least :math:`\ell` of :math:`S`, and may add
others. The tangent planes found so far stay valid for any choice of inputs,
and every choice already solved keeps its best point, so each alternative takes
a few more rounds of the same search, and its ``proven_optimal`` means that no
choice of inputs the earlier answers allow does better. The list ends early,
with a result whose ``message`` says why, when no further choice meets the rules.

Give a ``min_amount``: it is what "used" means, so that a trace of an input
does not count as using it.

Sensitivity: what each rule costs
---------------------------------

The ``constraints`` table gives each rule's **shadow price**, read from the
optimality conditions at the answer: the change in the optimal objective per
unit increase of the bound it sits at, and per 1% of that bound. A rule that
does not bind has a price of 0. The prices answer the question a constrained
answer always raises: which rule is expensive, and what would relaxing it be
worth?

:func:`~process_improve.multivariate.methods.inversion_sensitivity` checks the
prices and goes further:

.. code-block:: python

    report = pls.inversion_sensitivity(rules, front=("T²", [4.0, 6.0, 8.0]), prices=["Lactic"])

    report.checked  # each binding rule moved 1% and solved again: forecast and actual change
    report.front  # the optimal cost as the T² limit moves
    report.counts  # one input more or fewer per counted group
    report.prices  # how far each cost can move before the answer changes shape

A shadow price is a slope: it holds while the same rules bind. ``front`` traces
the objective over a range of one limit, where the slope may change, and
``prices`` gives the range of each cost over which the answer keeps its shape,
the same inputs used and the same rules binding.

See also
--------

- :class:`~process_improve.multivariate.methods.PLS`
- :class:`~process_improve.multivariate.methods.OPLS`
- :func:`~process_improve.multivariate.methods.optimal_inversion`
- :func:`~process_improve.multivariate.methods.inversion_sensitivity`
