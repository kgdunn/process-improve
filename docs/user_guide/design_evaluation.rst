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
   design = generate_design(factors, design_type="box_behnken", n_center_points=6)

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
- ``scaled_condition_number`` - the same with each column scaled to unit length, so
  the factors' units do not count: 1 for orthogonal columns.
- ``information_matrix`` - :math:`X^\top X` itself, entry by entry, with what is
  read from it: its determinant, the D-criterion :math:`\det(X^\top X)^{1/p}`, the
  eigenvalues and the E-criterion, the rank of :math:`X` and the residual degrees of
  freedom :math:`N - \operatorname{rank}(X)`, and for an estimable model the inverse,
  whose entries are the coefficients' variances and covariances in units of
  :math:`\sigma^2`, its trace (the A-criterion) and that trace over :math:`p`. It needs
  no responses, so a design's residual degrees of freedom are known before it is run.
- ``estimable_interactions`` - how many of the two-factor interactions left out of the
  model the design can estimate beside it: the rank the omitted interaction columns add
  to :math:`X`.

Split-plot and two-way split-plot designs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a split-plot design the runs that share a setting of the hard-to-change factors (a
whole plot) are correlated, and in a two-way split-plot (strip-plot) design the runs that
share a row or a column are; ordinary least squares then misstates the variances. Pass
``random_effects``, mapping each column that labels a random effect's groups to its
variance over the run variance :math:`\eta_k`. Every metric then uses the generalized
least squares information :math:`X^\top V^{-1} X`, with
:math:`V = I + \sum_k \eta_k Z_k Z_k^\top` and :math:`Z_k` the indicators of effect
:math:`k`'s groups, in place of :math:`X^\top X`. The variances are in units of the run
variance, as JMP reports them.

.. code-block:: python

   import itertools

   import pandas as pd

   # A 2^3 twice, in four whole plots: A is hard to change, B and C change from run to run.
   split_plot = pd.DataFrame(
       [{"WholePlot": wp, "A": a, "B": b, "C": c}
        for wp, a in enumerate([-1, 1, -1, 1]) for b, c in itertools.product([-1, 1], repeat=2)]
   )
   evaluate_design(
       split_plot,
       model="A * B * C",
       metric=["information_matrix", "average_prediction_variance"],
       random_effects={"WholePlot": 1.0},
   )

   # A two-way split-plot design: x1 set on four rows, x2 on four columns, every row in every column.
   strip_plot = pd.DataFrame(
       [{"Row": r, "Column": c, "x1": x1, "x2": x2}
        for (r, x1), (c, x2) in itertools.product(enumerate([-1, 1, -1, 1]), enumerate([-1, -1, 1, 1]))]
   )
   evaluate_design(strip_plot, model="x1 * x2", metric="information_matrix",
                   random_effects={"Row": 1.0, "Column": 1.0})

Term correlation
~~~~~~~~~~~~~~~~~

``correlation`` summarises the pairwise correlation among the *second-order*
terms (pure quadratics and two-factor interactions).  Because the
:math:`x_i^2` columns have a non-zero mean, a naive Pearson correlation is
inflated by that shared offset and depends on the coding.  The metric instead
residualises each second-order column against the intercept-and-main-effect
block first, giving a coding-invariant measure.  It returns ``max_abs_r``,
``mean_abs_r``, the full ``matrix``, and ``by_pair``: the largest and the mean
:math:`|r|`, and the number of pairs, for each kind of pair Jones and Nachtsheim
(2011) tabulate for definitive screening designs: ``two_squares``,
``square_and_own_interaction``, ``square_and_other_interaction`` and
``two_interactions``. Under the full quadratic model of a DSD this is their Table 3.

Correlation of estimates
~~~~~~~~~~~~~~~~~~~~~~~~

``estimate_correlation`` compares what the fit *estimates* rather than the columns:
:math:`r_{ij} = c_{ij} / \sqrt{c_{ii} c_{jj}}` with :math:`c = (X^\top X)^{-1}`, for the
model's second-order terms (JMP's "correlation of estimates"). A large :math:`|r_{ij}|`
means the runs cannot tell the two effects apart, so an active square can show up as an
interaction. It returns ``terms``, ``matrix`` and ``by_pair``, the same summary by kind of pair
as ``correlation``. A definitive screening design cannot fit the full quadratic model in all its
factors, so its estimates are compared in the projections onto three factors, where it can (Jones
and Nachtsheim's Table 4):

.. code-block:: python

   from process_improve.experiments import Factor, evaluate_projections, generate_design

   dsd = generate_design([Factor(name=f"x{i}", low=-1, high=1) for i in range(1, 7)], "dsd")
   found = evaluate_projections(dsd, 3, model="quadratic", metric="estimate_correlation")
   first = found["projections"][0]["metrics"]["estimate_correlation"]
   first["by_pair"]["square_and_other_interaction"]["mean_abs_r"]  # 0.483

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

The ``average_prediction_variance`` and ``g_efficiency`` metrics are derived from
this same region machinery, so they are consistent with the ``fds`` payload.

Two options make the region average and maximum exact rather than sampled:

- ``integration="exact"`` works out the average prediction variance over the cube as
  :math:`\operatorname{trace}((X^\top X)^{-1} W)`, with :math:`W` the cube's moment
  matrix, by Gauss-Legendre quadrature. It is exact for a model of degree up to four in
  each factor, and independent of the random sample.
- ``max_search="optimize"`` climbs from the best sampled and corner points by a compass
  search inside the cube. The largest prediction variance of a design's projection often
  lies inside a face or an edge, where a sample can miss it and overstate the
  G-efficiency.

.. code-block:: python

   exact = evaluate_design(
       design,
       model=model,
       metric=["average_prediction_variance", "g_efficiency", "information_matrix"],
       integration="exact",
       max_search="optimize",
   )

Projections
~~~~~~~~~~~

A screening design is often judged by what it can do once most factors turn out not
to matter: the runs already made are then a design in the few that do.
:func:`~process_improve.experiments.evaluate_projections` evaluates a design on every
projection onto a given number of its factors, or on a given number from each group
of factors, and reports how many projections estimate the model (the projection
estimation capacity) and each metric's average over them:

.. code-block:: python

   from process_improve.experiments import evaluate_projections

   found = evaluate_projections(design, 3, model="quadratic", metric=["d_efficiency", "g_efficiency"])
   found["n_estimable"], found["n_projections"], found["averages"]["d_efficiency"]

Under ``model="quadratic"`` a projection's model squares only the factors it runs at
three or more levels, so a mixed-level design's two-level factors enter by their main
effects and interactions alone.

Power
~~~~~

``power`` reports the statistical power to detect each model term.  Pass
``effect_size`` for a single power value per term, or omit it for a power curve
over a range of effect sizes.  ``effect_size`` is the anticipated coefficient in
coded units, which for a two-level factor is half its high-minus-low effect.

``screening_power`` asks instead how likely a screening design is to find one active term of
each kind, in the model an analyst would fit to find it: Jones and Nachtsheim's (2011) six
hypotheses, a linear effect, a quadratic effect or an interaction, each tested alone among the
main effects (hypotheses 1 to 3) or in the full quadratic model of two factors (4 to 6). It
averages each test's power over every factor or pair of factors, at Table 7's coefficients of
1, 2 and 3 times sigma unless ``effect_size`` is given, and it ignores ``model``. A test that a
design cannot make is marked ``"aliased"`` (every square of a two-level design with centre runs
has the same column) or ``"not estimable"``:

.. code-block:: python

   from process_improve.experiments import evaluate_design

   study = evaluate_design(dsd, metric="screening_power")["screening_power"]
   [(h["hypothesis"], h["status"], h["average_power"]) for h in study["hypotheses"]]

Fractional-factorial structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two-level fractional factorials the tool also reports ``alias_structure``,
``confounding``, ``resolution``, ``defining_relation``, ``clear_effects``, and
``minimum_aberration`` from the generators.

Categorical factors
~~~~~~~~~~~~~~~~~~~

A categorical factor (a column of level labels) enters the model in effect
(sum-to-zero) coding, the usual convention in DoE software: a factor with
:math:`L` levels gives :math:`L - 1` columns that each sum to zero over the
levels. A two-level factor is coded -1 for its first level and +1 for its
second (in sorted order), like a coded continuous factor; with more levels, the
last level in sorted order is -1 in every column. The terms read ``C[S.lo]``.
The built-in optimal designs (``generate_design``) are built in the same coding,
so the criterion value a design reports is the one ``evaluate_design`` returns.

The coding matters for some metrics and not others. The A- and E-criteria, VIF,
the condition number, power and the alias matrix depend on it; the prediction
variance (``average_prediction_variance``, ``g_efficiency``, ``fds``), the
ranking of designs by D-efficiency and the degrees of freedom do not. Pass
``categorical_coding="treatment"`` for 0/1 dummy coding against the first
level (terms ``C[T.mid]``), which earlier releases used.
:func:`~process_improve.experiments.analyze_experiment` codes its coefficients'
categorical factors much the same way by default (``coding="auto"``): a two-level
factor at -1 and +1 and, in a named model, a factor with more levels in sum-to-zero
contrasts. With ``coding="actual"`` they keep treatment coding. Its effects already
run from the low level to the high level.

Region Sampling and Reproducibility
-----------------------------------

The region-integrated metrics (``average_prediction_variance``, ``g_efficiency``,
``max_prediction_variance`` and ``fds``) are computed by Monte-Carlo sampling the
design region.  The sampling is fully controllable and seeded:

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
     - Append the :math:`2^k` cube corners (crossed with every combination of
       categorical levels), where the worst-case prediction variance usually
       sits. They serve the maximum only; the average and the FDS distribution
       come from the uniform sample.
   * - ``random_state``
     - ``42``
     - Seed for the region sampler; fixing it makes the maximum reproducible.

The region **average** (I) is stable across seeds, but the region **maximum**
(G) is sensitive to the sample: the worst point is often in the interior, so a
denser sample finds higher worst-case values.  To tighten and reproduce the
G estimate, raise ``n_samples`` and fix ``random_state``:

.. code-block:: python

   metrics = evaluate_design(
       design,
       model=model,
       metric="fds",
       n_samples=120_000,
       random_state=1,
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
       design,
       model=model,
       metric="fds",
       fds_resolution=200,
       random_state=1,
   )["fds"]

   curve = fds["curve"]
   curve["fraction"]  # 200 evenly spaced fractions in [0, 1]
   curve["prediction_variance"]  # the FDS curve, sigma^2 units
   curve["scaled_prediction_variance"]  # the same, scaled by N (SPV)

Setting ``fds_resolution`` is fully backward compatible: when it is ``None``
(the default) the output is unchanged and the coarse quantile summary is still
present.

Comparing Designs on Nine Measures
----------------------------------

:func:`~process_improve.experiments.compare_designs` sets several designs side by
side for one model. The model is fixed first, so the designs differ only in where
they place their runs, and each design is scored at its own size: the table labels
it with its run count rather than dividing the measures by it. The measures are the
nine that *Process Improvement using Data* defines in `Nine measures down one column
<https://learnche.org/pid/design-analysis-experiments/omars-designs#doe-omars-metric-choice>`__,
after three rows that say whether a design can fit the model at all:

.. code-block:: python

   from process_improve.experiments import Factor, compare_designs, generate_design

   factors = [Factor(name=n, low=-1, high=1) for n in "ABCD"]
   comparison = compare_designs(
       {
           "Box-Behnken": generate_design(factors, "box_behnken", n_center_points=3),
           "CCD": generate_design(factors, "ccd", alpha="face_centered", n_center_points=3),
           "DSD": generate_design(factors, "dsd", n_center_points=3),
       },
       model="main_quadratic",
   )
   print(comparison.table.to_string())

.. code-block:: text

                                      Better Box-Behnken [n=27] CCD [n=27] DSD [n=11]
   Runs                                                      27         27         11
   Fits the model?                                          yes        yes        yes
   Residual degrees of freedom                               18         18          2
   A/p, average coefficient variance       ↓             0.1574     0.2126     0.3004
   I, average prediction variance          ↓             0.2333     0.2741     0.5012
   E, smallest eigenvalue of M             ↑              1.453      2.000      2.000
   G, largest prediction variance          ↓             0.8333     0.6543      1.000
   D, |M|^(1/p)                            ↑              9.706      9.052      4.352
   Max |r| between second-order terms      ↓             0.2000     0.6667     0.6055
   Power, main effect                      ↑             0.9056     0.9797     0.2909
   Power, interaction                      ↑
   Power, quadratic                        ↑             0.5891     0.3296     0.1527

The ``main_quadratic`` model has the intercept, the main effects and the pure
quadratics, :math:`p = 2k + 1` terms; every measure but max :math:`|r|` is read from
its information matrix :math:`\mathbf{M} = \mathbf{X}^T\mathbf{X}`:

- :math:`A/p` and :math:`I` are averages: of the coefficients' variances, and of the
  variance of a prediction over the cube :math:`[-1, 1]^k`, both in units of the
  noise variance :math:`\sigma^2`.
- :math:`E` and :math:`G` are the worst cases of the same two quantities: the
  worst-estimated combination of the coefficients has variance :math:`\sigma^2/E`,
  and :math:`G` is the prediction variance at the worst point of the cube.
- :math:`D = |\mathbf{M}|^{1/p}` is the joint precision of all the coefficients, and
  max :math:`|r|` the largest correlation between two second-order terms, which the
  correlation map below draws.
- Power is the probability of detecting a coefficient of ``effect_size`` noise
  standard deviations at ``significance_level``, for the least precisely estimated
  term of each type the model has. This model has no interactions, so that row is
  empty.

No design is best on every row. The two twenty-seven-run designs have the same size:
the Box-Behnken design has the smaller average variances, the smaller max
:math:`|r|` and more power for a quadratic, while the face-centred CCD has the
smaller worst cases and more power for a main effect. The eleven-run DSD spends fewer
runs, and every measure that improves with runs is weaker for it, with two degrees of
freedom left to estimate the error. Which row matters depends on what the experiment
is for: :math:`A/p` or :math:`E` for precise coefficients, :math:`I` or :math:`G` for
prediction over the region, max :math:`|r|` for keeping second-order effects apart,
and power for detecting an effect of a stated size.

A design that cannot fit the model reports its rank instead of the measures, and
says which terms it cannot separate: a two-level factorial, for instance, cannot
tell its squares apart from each other or from the intercept.
:attr:`~process_improve.experiments.compare.DesignComparison.values` holds every
number at full precision, and ``to_dict()`` a JSON-ready copy.

The correlation map
~~~~~~~~~~~~~~~~~~~

The ``correlation_map`` plot draws the absolute correlation between a design's main
effects, squares and two-factor interactions, in three blocks:

.. code-block:: python

   from process_improve.experiments.visualization.plots import create_plot

   dsd = generate_design(factors, "dsd", n_center_points=3)
   rows = dsd.design[["A", "B", "C", "D"]].to_dict("records")
   figure = create_plot("correlation_map", design_data=rows).to_plotly()

A definitive screening design keeps every main effect clear of the other terms, so
its first block row and column are empty off the diagonal; its second-order terms
are correlated with each other.

Testing an analysis on a known process
--------------------------------------

:class:`~process_improve.experiments.TrueProcess` holds a second-order model in coded
units and a noise standard deviation. Its runs can be simulated for any design and
analysed as if they had come from the plant, so the estimates can be set beside the
truth before any experiment is spent:

.. code-block:: python

   import pandas as pd

   from process_improve.experiments import TrueProcess, analyze_experiment

   design = generate_design(factors, "box_behnken", n_center_points=3, random_state=1)
   process = TrueProcess(
       {"Intercept": 62, "A": 4, "B": -3.1, "C": 1.75, "A:B": 1.5, "I(C ** 2)": -2.25},
       sigma=1.0,
   )
   y = process.simulate(design, random_state=2)
   fit = analyze_experiment(design.design[["A", "B", "C", "D"]], y, model="quadratic",
                            analysis_type="coefficients")
   estimated = pd.DataFrame(fit["coefficients"]).set_index("term")["coefficient"]
   print(pd.DataFrame({"true": pd.Series(dict(process.coefficients)), "estimated": estimated})
         .fillna(0.0).round(2).loc[["Intercept", "A", "B", "C", "D", "A:B", "I(C ** 2)"]])

.. code-block:: text

               true  estimated
   Intercept  62.00      62.67
   A           4.00       4.09
   B          -3.10      -2.66
   C           1.75       1.24
   D           0.00      -0.06
   A:B         1.50       1.28
   I(C ** 2)  -2.25      -2.99

Factor ``D`` is not in the model, so it has no effect, and its estimate differs from
zero by the noise alone. The terms are named as ``analyze_experiment`` names them;
``A*B``, ``B:A``, ``C**2`` and ``C^2`` are accepted too.

.. _simulating-designs:

Simulating the Studies a Design Would Give
------------------------------------------

The measures above take the model as known and complete. Two things they cannot see decide
how a screening study goes: a term outside the model (an interaction that a 12-run
Plackett-Burman design was not built to estimate) biases the effects inside it, and the
analysis has to choose the model from the same noisy runs.
:func:`~process_improve.experiments.simulate_design_performance` sees both. It makes each
design many times on a :class:`~process_improve.experiments.TrueProcess`, the process you
expect, and judges every simulated study the way
:func:`~process_improve.experiments.recommend_next_step` judges real runs (see
:doc:`next_step`).

.. code-block:: python

   from process_improve.experiments import Factor, TrueProcess, generate_design, simulate_design_performance

   factors = [Factor(name=n, low=-1, high=1) for n in "ABCDEFGH"]
   truth = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)
   designs = {
       "PB": generate_design(factors, "plackett_burman", n_center_points=0),
       "PB + 3 centre": generate_design(factors, "plackett_burman", n_center_points=3),
       "2^(8-4) + 3 centre": generate_design(factors, "fractional_factorial", resolution=4, n_center_points=3),
       "DSD + 5 centre": generate_design(factors, "dsd", n_center_points=5),
   }
   report = simulate_design_performance(designs, truth, n_simulations=200)
   for label, design in report["designs"].items():
       print(f"{label:28} {design['found_all']:.2f} {design['false_alarms']:.2f} {design['nothing_stood_out']:.2f}")

.. code-block:: text

   PB [n=12]                    0.53 0.21 0.41
   PB + 3 centre [n=15]         0.94 0.32 0.01
   2^(8-4) + 3 centre [n=19]    1.00 0.48 0.00
   DSD + 5 centre [n=21]        0.99 0.55 0.00

The columns are the share of studies that find every true term, the mean number of terms
found that are not true, and the share in which nothing stood out from the noise. A share
from 200 studies is within about 0.07 of its limit (95%). The 12-run Plackett-Burman design
finds the process in half of the studies: the ``A:C`` interaction biases every main effect
outside it, and with no runs to spare for error, nothing stands out in two studies of five.
Three centre runs give it the degrees of freedom to see the interaction. The false alarms are
the price of testing every term at 0.05: a design that tests more terms raises more of them. A true term counts as found when
the analysis finds it, or a term the design makes identical to it: ``D`` for ``A:B`` in a
resolution III fraction whose generator is ``D = AB``, or the curvature at the centre runs
for a square. The design can do no better than that, so it is no false alarm.

With ``goal="maximize"`` (or ``"minimize"``, or ``"target"`` with a ``target``), each study
also predicts its best settings, and the report says how far the true response there falls
short of the true best inside the ranges, in the response's units and in noise standard
deviations. ``report["true_optimum"]`` holds the true best. Each design's ``summary`` says the
main numbers in a sentence, and ``found`` gives each true term's share.

The simulation is also an MCP tool, ``simulate_design_performance``, which takes each
design's coded runs as ``generate_design`` returns them; ``compare_designs`` is one too.

See Also
--------

- :func:`~process_improve.experiments.compare_designs` - several designs, one model.
- :func:`~process_improve.experiments.simulate_design_performance` - several designs, by simulated studies.
- :func:`~process_improve.experiments.evaluate_design` - full API reference.
- :func:`~process_improve.experiments.evaluate_all` - compute every metric in
  one call.
- :doc:`doe_strategy` - choosing which design to generate in the first place.

Run order and time trends
-------------------------

When the runs are made one after another, a slow drift in the process adds a smooth function of
time to every response, and biases any effect whose column follows it.
:func:`~process_improve.experiments.run_order_trends` measures that exposure. For each model
term, it gives the cosine between the term's column and each orthogonal polynomial time trend:
0 when the column is orthogonal to the trend, 1 when it follows the trend exactly.
:func:`~process_improve.experiments.trend_robust_run_order` searches for an order that brings
those cosines to zero, starting with the main effects against the linear trend.

.. code-block:: python

   import itertools
   import numpy as np
   from process_improve.experiments import run_order_trends, trend_robust_run_order

   cube = np.array(list(itertools.product((-1, 1), repeat=3)))   # the 2^3 in standard order
   run_order_trends(cube, model="main_effects", degree=1)["summary"]["linear"]["main_effects"]
   found = trend_robust_run_order(cube, model="main_effects", degree=1, random_state=0)
   found["order"], found["trends"]["total_level_changes"]

The search is a heuristic: it does not prove that a better order does not exist.
