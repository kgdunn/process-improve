Evaluating Design Quality
=========================

Once you have a candidate design, the next question is: *"How good is it for
the model I intend to fit?"*  The :func:`~process_improve.experiments.evaluate_design`
function answers this by computing a complete, model-aware set of quality
metrics from the design matrix and a chosen model.  A single call can fully
characterise a response-surface design: its efficiency, where it predicts well
or badly, how its coefficients are correlated, and what bias it carries from
terms left out of the model.

Every metric is computed against the *model you specify*, not against the
design in isolation.  A design that is excellent for a main-effects model can
be poor for a full quadratic model, so the model always comes first.

When to Use This Tool
---------------------

- **Comparing candidate designs** - score Box-Behnken, central composite,
  definitive screening, OMARS, and optimal designs on the same footing before
  committing runs.
- **Checking a design against a reduced model** - verify that an explicit
  formula (for example main effects plus pure quadratics) is estimable and
  well-conditioned.
- **Diagnosing where a design predicts poorly** - the fraction-of-design-space
  (FDS) curve shows the spread of prediction variance across the whole region,
  not just at the design points.
- **Quantifying bias** - the alias matrix reports how much omitted two-factor
  interactions would bias the fitted coefficients.

Quick Start
-----------

.. code-block:: python

   from process_improve.experiments import generate_design, evaluate_design, Factor

   factors = [Factor(name=n, low=-1, high=1) for n in "ABCDE"]
   design = generate_design(factors, design_type="box_behnken", center_points=6)

   # A reduced model: main effects plus pure quadratics (no two-factor interactions).
   model = "A+B+C+D+E+I(A**2)+I(B**2)+I(C**2)+I(D**2)+I(E**2)"

   metrics = evaluate_design(
       design,
       model=model,
       metric=["d_efficiency", "a_optimality", "e_optimality", "fds"],
   )

To compute every available metric in one call, pass ``metric="all"`` or use the
:func:`~process_improve.experiments.evaluate_all` convenience wrapper:

.. code-block:: python

   from process_improve.experiments import evaluate_all

   everything = evaluate_all(design, model=model)

The Metrics
-----------

Optimality and efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These summarise the information matrix :math:`X^\top X` for the fitted model.

- ``d_efficiency`` - :math:`100 \cdot \det(X^\top X)^{1/p} / N`; overall
  information content (higher is better).
- ``a_optimality`` - :math:`\operatorname{trace}((X^\top X)^{-1})`, the average
  coefficient variance (**lower** is better).  Reported alongside an
  ``a_efficiency`` score for parity with ``d_efficiency``.
- ``e_optimality`` - the smallest eigenvalue of :math:`X^\top X`, the
  worst-estimated direction in parameter space (**higher** is better).
- ``vif`` - variance inflation factor per term; how much each coefficient
  variance is inflated by non-orthogonality.
- ``condition_number`` - conditioning of the model matrix.

Term correlation
~~~~~~~~~~~~~~~~~

``correlation`` summarises the pairwise correlation among the *second-order*
terms (pure quadratics and two-factor interactions).  Because the
:math:`x_i^2` columns have a non-zero mean, a naive Pearson correlation is
inflated by that shared offset and depends on the coding.  The metric instead
residualises each second-order column against the intercept-and-main-effect
block first, giving a coding-invariant measure.  It returns ``max_abs_r``,
``mean_abs_r``, and the full ``matrix``.

Alias (bias) matrix
~~~~~~~~~~~~~~~~~~~~~

``alias_matrix`` generalises the two-level alias structure to any design and
model.  Given the fitted model :math:`X_1` and a set of potential extra terms
:math:`X_2` (by default the two-factor interactions *not* already in the
model), it computes

.. math::

   A = (X_1^\top X_1)^{-1} X_1^\top X_2,

so the expected fitted coefficients are biased as
:math:`\mathbb{E}[b_1] = \beta_1 + A\,\beta_2`.  The result reports the matrix,
the worst single bias (``max_abs``), the maximum over the main-effect rows
(``max_abs_main_effect_rows``), and the Frobenius norm.

Prediction variance and the FDS curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prediction_variance`` is the leverage :math:`d(x) = x^\top (X^\top X)^{-1} x`
*at the design runs*.  ``fds`` instead samples :math:`d(x)` over the **whole
design region**, giving the fraction-of-design-space distribution.  From the
same region sample it reports:

- ``average_prediction_variance`` - the region average (I / V-optimality),
- ``max_prediction_variance`` - the region maximum (G-optimality), in
  :math:`\sigma^2` units,
- the run-count-scaled SPV variants (each multiplied by ``N``), and
- a coarse 11-point ``quantiles`` summary.

``i_efficiency`` and ``g_efficiency`` are derived from this same region
machinery, so they are consistent with the ``fds`` payload.

Power
~~~~~

``power`` reports the statistical power to detect each model term.  Pass
``effect_size`` for a single power value per term, or omit it for a power curve
over a range of effect sizes.

Fractional-factorial structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two-level fractional factorials the tool also reports ``alias_structure``,
``confounding``, ``resolution``, ``defining_relation``, ``clear_effects``, and
``minimum_aberration`` from the generators.

Region Sampling and Reproducibility
-----------------------------------

The region-integrated metrics (``i_efficiency``, ``g_efficiency``,
``average_prediction_variance``, ``max_prediction_variance``, and ``fds``) are
computed by Monte-Carlo sampling the design region.  The sampling is fully
controllable and seeded:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - Meaning
   * - ``region``
     - ``"cuboidal"``
     - ``"cuboidal"`` samples :math:`[-1, 1]^k`; ``"spherical"`` samples the
       ball of radius :math:`\sqrt{k}`.
   * - ``n_samples``
     - ``100_000``
     - Number of random points drawn over the region.
   * - ``include_vertices``
     - ``True``
     - Always append the :math:`2^k` cube corners, where the worst-case
       prediction variance usually sits.
   * - ``random_seed``
     - ``42``
     - Seed for the region sampler; fixing it makes the maximum reproducible.

The region **average** (I) is stable across seeds, but the region **maximum**
(G) is sensitive to the sample: the worst point is often in the interior, so a
denser sample finds higher worst-case values.  To tighten and reproduce the
G estimate, raise ``n_samples`` and fix ``random_seed``:

.. code-block:: python

   metrics = evaluate_design(
       design, model=model, metric="fds",
       n_samples=120_000, random_seed=1,
   )
   metrics["fds"]["max_prediction_variance"]  # reproducible run to run

The region, sample size, vertex flag, and seed actually used are echoed back in
the ``fds`` payload.

Tunable FDS Curve
-----------------

By default ``fds`` returns the coarse 11-point ``quantiles`` summary.  For a
smooth plot, set ``fds_resolution`` to the number of points you want.  The
payload then gains a ``curve`` sub-dict with ``fraction``,
``prediction_variance``, and ``scaled_prediction_variance`` (the run-count
scaled SPV) arrays of that length, evaluated on evenly spaced fractions in
:math:`[0, 1]`.  The arrays are monotonically non-decreasing, and their
endpoints equal the minimum and maximum prediction variance.

.. code-block:: python

   fds = evaluate_design(
       design, model=model, metric="fds",
       fds_resolution=200, random_seed=1,
   )["fds"]

   curve = fds["curve"]
   curve["fraction"]              # 200 evenly spaced fractions in [0, 1]
   curve["prediction_variance"]   # the FDS curve, sigma^2 units
   curve["scaled_prediction_variance"]  # the same, scaled by N (SPV)

Setting ``fds_resolution`` is fully backward compatible: when it is ``None``
(the default) the output is unchanged and the coarse quantile summary is still
present.

See Also
--------

- :func:`~process_improve.experiments.evaluate_design` - full API reference.
- :func:`~process_improve.experiments.evaluate_all` - compute every metric in
  one call.
- :doc:`doe_strategy` - choosing which design to generate in the first place.
