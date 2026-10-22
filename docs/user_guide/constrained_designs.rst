.. _constrained-designs:

Designs over a Constrained Region
=================================

Factor limits are rarely independent. A reactor may run hot, or run long, but not
both, because the heat load would exceed what the jacket can remove. The feasible
region is then the factor box with a corner cut off, and a classical design (a
factorial, a CCD, a Box-Behnken) places runs in that corner.

``generate_design`` accepts such limits as :class:`~process_improve.experiments.Constraint`
objects. When constraints are given it builds a D-optimal design, choosing every run
from the feasible region.

Worked example: a shared heat budget
------------------------------------

Temperature ``T`` ranges from 100 to 150 degC and dosing time ``D`` from 20 to 60 min.
The heat load, proportional to ``3*T + 5*D``, may not exceed 600. The corner at
(150 degC, 60 min) scores 750 and is out of reach, and so is every point above the
line from (150, 30) to (100, 60).

.. code-block:: python

   from process_improve.experiments import Constraint, Factor, generate_design

   factors = [
       Factor(name="T", low=100, high=150, units="degC"),
       Factor(name="D", low=20, high=60, units="min"),
   ]
   heat = Constraint(expression="3*T + 5*D <= 600")

   result = generate_design(factors, budget=10, constraints=[heat], model_type="quadratic")
   print(result.design_type)  # d_optimal, chosen because a constraint is given

   design = result.design_actual.sort_values(["T", "D"])
   design["heat"] = 3 * design["T"] + 5 * design["D"]
   print(design.to_string(index=False))

.. code-block:: text

    RunOrder     T    D  heat
           1 100.0 20.0 400.0
           7 100.0 20.0 400.0
           2 100.0 40.0 500.0
           3 100.0 40.0 500.0
           6 100.0 60.0 600.0
          10 100.0 60.0 600.0
           5 125.0 20.0 475.0
           8 125.0 45.0 600.0
           9 150.0 20.0 550.0
           4 150.0 30.0 600.0

Every run satisfies the constraint. The ten runs sit on the four vertices of the
feasible region, (100, 20), (150, 20), (150, 30) and (100, 60), and on the midpoints
of three of its edges, which is where a quadratic model is best estimated. Three
points are replicated, which leaves error degrees of freedom for testing lack of fit.

Three of those runs lie exactly on the constraint line. One of them, (125, 45), is not
on the 5-by-5 grid the candidate set starts from; see below for how it is found.

Shrinking the box so that its high corner satisfies the constraint is the usual
alternative. Holding ``T`` up to 150 degC forces ``D`` down to 30 min, which keeps 500
of the 1250 (degC min) of the feasible region, 40% of it, and the model cannot be
used for the long dosing times in the remaining 60%.

How the design is chosen
------------------------

1. **Candidate set.** A grid over the factor box (5 levels per continuous factor,
   fewer for many factors), plus the points where each constraint boundary crosses a
   grid edge, located by bisection. Infeasible points are dropped. The metadata
   reports the counts:

   .. code-block:: python

      m = result.metadata
      print(m["constraints_enforced"], m["n_grid_points"], m["n_boundary_points"], m["n_candidates"])
      # True 25 7 21

2. **Model matrix.** Each candidate is expanded into the columns of the requested
   ``model_type``: intercept, main effects, two-factor interactions and, for
   ``"quadratic"``, squared terms. A categorical factor with ``L`` levels
   contributes ``L - 1`` effect-coded (sum-to-zero) columns, the coding
   ``evaluate_design`` uses.

3. **Exchange.** A Fedorov exchange swaps one design run for one candidate at a
   time, taking the swap that increases the determinant of ``X'X`` most, until no
   swap improves it. Five random starts guard against a poor local optimum;
   ``random_state`` makes the result reproducible.

.. _constrained-writing-constraints:

Writing constraints
-------------------

* Use factor names and actual units: ``"3*T + 5*D <= 600"``.
* ``<``, ``<=``, ``>`` and ``>=`` are accepted; a chained comparison such as
  ``"400 <= 3*T + 5*D <= 600"`` gives a lower and an upper limit.
* Nonlinear limits are allowed, with ``+ - * / **`` and ``abs``, ``sqrt``, ``exp``,
  ``log``, ``log10``: ``Constraint(expression="T * D <= 6000", type="nonlinear")``.
* Equalities are refused. A region with ``A + B == 1`` has no volume to place runs
  in; proportions that sum to one are mixture factors.
* Constraints refer to continuous factors. Categorical factors may appear in the
  design and are crossed with the constrained continuous region.

The expression is parsed into an arithmetic tree and evaluated with numpy; it is never
passed to ``eval``, so an expression from an untrusted source can only compute a number.
:ref:`region-limits` shows how to check what a set of limits leaves, propose a limit from
the settings that cannot be run, and draw the region.

Combining with other options
----------------------------

* ``fixed_runs`` keeps runs already performed (continuous factors in coded units,
  categorical as labels) and fills the rest of the budget around them. A fixed run
  outside the region is kept and a warning is logged.
* ``hard_to_change`` (split-plot) is not available together with constraints; it is
  ignored and recorded as ``metadata["hard_to_change_ignored"]``.
* Other design types do not enforce constraints. They log a warning and set
  ``metadata["constraints_enforced"] = False``.

Judging the design over its own region
--------------------------------------

A constrained design should be judged on the settings it may visit. The average
prediction variance (the I-criterion, lower is better) averages it over the region, and
the G-efficiency is based on its worst case. Over the full box, both are dominated by the cut-off corner, where the
model has to extrapolate. ``generate_design`` records the region in
``result.metadata["region"]``, and ``evaluate_design`` uses it by default:

.. code-block:: python

   from process_improve.experiments import evaluate_design

   metrics = ["average_prediction_variance", "g_efficiency"]
   inside = evaluate_design(result, model="quadratic", metric=metrics)
   cube = evaluate_design(result, model="quadratic", metric=metrics, region="cuboidal")
   print(f"{inside['average_prediction_variance']:.2f} {inside['g_efficiency']:.0f}")  # 0.50 73
   print(f"{cube['average_prediction_variance']:.2f} {cube['g_efficiency']:.1f}")      # 1.44 3.5

Inside the region the average prediction variance is a third of its value over the
box, and the worst case is about a twentieth. The box figure describes settings the plant cannot run,
so it says nothing about the design's quality for this process.

The region is sampled uniformly by rejection, and its boundary points (the feasible
grid corners and the constraint crossings) are added, since that is where the worst
case of a second-order model sits. Pass ``region=DesignRegion(factors, constraints)``
to evaluate a design held as a plain DataFrame.

Choosing the criterion: D, I or A
---------------------------------

The exchange maximises one number, and the three criteria ask different questions:

* **D-optimal** (``design_type="d_optimal"``) maximises ``det(X'X)``: the model
  coefficients are estimated jointly as precisely as possible.
* **I-optimal** (``"i_optimal"``) minimises the average prediction variance over the
  region, which ``evaluate_design`` reports as ``average_prediction_variance``. It suits a study whose purpose is
  prediction or optimisation.
* **A-optimal** (``"a_optimal"``) minimises the summed variance of the coefficients.
* **E-optimal** (``"e_optimal"``) maximises the smallest eigenvalue of ``X'X``, so no
  combination of the coefficients is left poorly estimated.

On the heat-budget example, with ten runs for a quadratic model, each design is best on
its own measure:

.. list-table::
   :header-rows: 1

   * - Design
     - D-efficiency (higher is better)
     - Summed coefficient variance (lower is better)
     - Smallest eigenvalue (higher is better)
   * - D-optimal
     - 29.6
     - 5.67
     - 0.272
   * - A-optimal
     - 27.8
     - 4.95
     - 0.322
   * - E-optimal
     - 26.0
     - 5.09
     - 0.325

For the I-optimal design the average is taken over the constrained region itself,
sampled uniformly, so the design is chosen for the same region ``evaluate_design``
uses to judge it:

.. code-block:: python

   for design_type in ["d_optimal", "i_optimal"]:
       r = generate_design(factors, design_type=design_type, budget=10, constraints=[heat], model_type="quadratic")
       m = evaluate_design(r, model="quadratic", metric=["d_efficiency", "average_prediction_variance", "g_efficiency"])
       print(design_type, {k: round(m[k], 3) for k in ("d_efficiency", "average_prediction_variance", "g_efficiency")})
   # d_optimal {'d_efficiency': 29.57, 'average_prediction_variance': 0.499, 'g_efficiency': 73.341}
   # i_optimal {'d_efficiency': 27.351, 'average_prediction_variance': 0.356, 'g_efficiency': 62.677}

Each design is best on its own criterion. The I-optimal design places more runs in the
interior, so its average prediction variance is lower by a factor of 1.4, at the cost
of a slightly less precise joint estimate of the coefficients and a larger worst case
at the boundary.

These designs come from the built-in exchange, which is the default for every optimal
design, so the same call gives the same design whether or not the optional pyoptex
package is installed. A split-plot design (``hard_to_change``) comes from the built-in
coordinate exchange; pyoptex is used only when asked for with ``backend="pyoptex"``.

Choosing runs from a list: candidate sets
-----------------------------------------

Sometimes the region is not a formula but a list: the operating points in the plant
records, the discrete settings a piece of equipment offers, or the blends that can be
made up from stock. Pass them as ``candidates`` and the optimal design is chosen from
those rows only:

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(0)
   history = pd.DataFrame(
       {"T": rng.uniform(100, 150, 30).round(1), "D": rng.uniform(20, 60, 30).round(1)},
       index=[f"batch_{i:02d}" for i in range(30)],
   )
   result = generate_design(
       factors, budget=8, candidates=history, constraints=[heat], model_type="interactions"
   )
   m = result.metadata
   print(m["n_candidates_supplied"], m["n_candidates_infeasible"], m["n_candidates"])  # 30 13 17
   print(m["selected_candidates"])
   # {'batch_02': 2, 'batch_20': 2, 'batch_29': 2, 'batch_05': 1, 'batch_06': 1}

The 13 records that break the heat budget are dropped, and the eight runs are chosen
from the remaining 17. ``selected_candidates`` names the chosen rows by their index
label, so the design can be read as "rerun batches 2, 20 and 29 twice, and batches 5
and 6 once".

The candidates are given in actual units, one column per factor (labels for
categorical factors, proportions for mixtures). A candidate set replaces the generated
grid, so for an I-optimal design it is also the region the average prediction variance
is taken over. Without ``design_type``, a candidate set selects ``"d_optimal"``.
Candidates outside the factors' ``low``/``high`` range are kept with a warning, since
the coding extrapolates beyond plus or minus one.

Running the design in blocks
----------------------------

When the runs cannot all be made under the same conditions (on two days, from two lots
of raw material), pass ``n_blocks``. The analysis then fits a level for each block, so
what the design has to estimate well is the factor effects once the differences
between blocks are taken out. The D-, A- and I-optimal designs choose their runs for
exactly that model: each block gets its own copy of the candidates, a run is only ever
exchanged for a candidate of its own block, and the criterion is that of the factor
effects adjusted for the blocks.

.. code-block:: python

   result = generate_design(factors, budget=12, n_blocks=2, constraints=[heat], model_type="quadratic")
   print(result.metadata["blocking"]["method"])  # optimal_exchange
   print(result.design_actual.sort_values(["Block", "T", "D"]).to_string(index=False))

.. code-block:: text

    RunOrder     T    D  Block
           4 100.0 20.0      1
           6 100.0 40.0      1
           5 100.0 60.0      1
           2 125.0 20.0      1
           3 125.0 45.0      1
           1 150.0 30.0      1
          10 100.0 20.0      2
          11 100.0 40.0      2
           8 100.0 60.0      2
          12 125.0 20.0      2
           7 125.0 45.0      2
           9 150.0 20.0      2

Each day's six runs cover the region's vertices and edge midpoints, so each block is
nearly a quadratic design on its own, and a shift from one day to the next moves both
blocks' averages without biasing the factor effects. The run sheet takes the blocks in
turn, shuffled within each.

The most useful case is a follow-up experiment. Pass the runs already made as
``fixed_runs``: with ``n_blocks`` they are the first block, kept first and in their
order, and the new runs fill the other blocks, chosen knowing that the new day may
differ from the old one.

.. code-block:: python

   first_day = generate_design(factors, budget=6, constraints=[heat], model_type="interactions").design
   follow_up = generate_design(
       factors,
       budget=12,
       n_blocks=2,
       fixed_runs=first_day[["T", "D"]],
       constraints=[heat],
       model_type="quadratic",
   )
   print(follow_up.design["Block"].tolist())  # [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

For the analysis with a day effect, this follow-up is 3.7% more D-efficient than the
same augmentation chosen as if both days were one (13% for a 2x2 factorial followed by
five runs). Splitting an optimal design into blocks after choosing it, the only option
before, was within 1% of the joint design for equal blocks over a regular region; the
gain is in follow-ups, in many small blocks (2 to 5% for blocks of two runs), and it
cannot hurt.

``metadata["log_det_information"]`` (or ``trace_criterion``) is then the value for the
factor effects adjusted for the blocks, and ``metadata["blocking"]["method"]`` is
``"optimal_exchange"``. E-, G- and K-optimal designs, mixtures, split-plot designs and
replicated designs are still split into blocks after they are chosen (``"exchange"``),
and cannot be combined with ``fixed_runs``. Each block needs at least two new runs, and
the budget is raised, with a warning, when the block effects leave too few.

Optimising inside the region
----------------------------

An optimum is only useful if the process can run it. Passing the same region to
``optimize_responses`` turns each constraint into an SLSQP inequality constraint and
starts the search from feasible points. Here, a fitted quadratic model whose maximum
over the box is the forbidden corner:

.. code-block:: python

   from process_improve.experiments import DesignRegion, optimize_responses

   model = {
       "response_name": "y",
       "factor_names": ["T", "D"],
       "coefficients": [
           {"term": "Intercept", "coefficient": 60.0},
           {"term": "T", "coefficient": 8.0},
           {"term": "D", "coefficient": 6.0},
           {"term": "I(T ** 2)", "coefficient": -2.0},
           {"term": "I(D ** 2)", "coefficient": -3.0},
       ],
   }
   goal = [{"response": "y", "goal": "maximize", "low": 40, "high": 75}]
   ranges = {"T": {"low": 100, "high": 150}, "D": {"low": 20, "high": 60}}
   region = DesignRegion.from_dict(result.metadata["region"])

   best = optimize_responses([model], goal, factor_ranges=ranges, region=region)["desirability"]
   print(f"{best['optimal_actual']['T']:.1f} {best['optimal_actual']['D']:.1f}", best["within_region"])
   # 140.7 35.6 True: on the heat line 3*T + 5*D = 600

Without ``region`` the same call returns (150 degC, 60 min), with a heat load of 750.
``method="pareto_front"`` accepts ``region`` too; the other methods ignore it and log
a warning.

Constrained mixtures
--------------------

In a formulation the components are proportions that sum to one, and each is usually
bounded: at least 10% polymer, at most 30% filler. Bounds and linear constraints cut
the simplex down to a polygon (a polytope in more than three components), and the
design is built from its geometry.

.. code-block:: python

   factors = [
       Factor(name="polymer", type="mixture", low=0.10, high=0.50),
       Factor(name="solvent", type="mixture", low=0.10, high=0.70),
       Factor(name="filler", type="mixture", low=0.05, high=0.30),
   ]
   cap = Constraint(expression="polymer + solvent <= 0.85")

   ev = generate_design(factors, constraints=[cap])
   print(ev.metadata["method"], ev.n_runs, ev.metadata["n_vertices"])  # extreme_vertices 11 5

   dopt = generate_design(factors, budget=12, constraints=[cap])
   print(dopt.metadata["method"])  # d_optimal_extreme_vertices

* **Extreme vertices.** In ``q`` components, a vertex is a blend where ``q - 1``
  constraints hold with equality. Constraints implied by the others are dropped, every
  choice of ``q - 1`` of the rest is solved as one linear system, in batches, and the
  feasible solutions are the vertices.
* **Without a budget** the vertices and the centroid are returned, plus the edge
  midpoints for a quadratic model and the centroids of the 2-dimensional faces for a
  special cubic one. Those faces are the constrained counterpart of ternary blends,
  which the ``x_i x_j x_k`` terms need. A full cubic model takes the 2-face centroids
  and, in place of the midpoints, the points a third and two thirds of the way along
  each edge: along an edge the model is a cubic polynomial, which needs four points on
  that edge. When that is more than three runs per model term, as it soon is with many
  bounded components, a D-optimal design with five runs more than the model has terms
  is returned instead.
* **With a budget** a D-optimal subset is chosen from the vertices, edge midpoints,
  edge third-points, 2-face and facet centroids, centroid and axial check blends, by
  the same exchange as above. Blends can repeat, which gives replicates for a
  pure-error estimate.

Mixture constraints must be linear in the proportions, since the vertex enumeration
relies on flat faces. On the full simplex (no bounds, no constraints) the design is
sized to the model: the pure components and the overall centroid for a linear model,
plus the binary 50:50 blends for a quadratic one (the ``{q, 2}`` simplex lattice), plus
the ternary blends for a special cubic one. With three components that is the 7-run
simplex centroid. The full cubic model (``"scheffe_cubic"``, or ``"cubic"``) adds a term
``x_i x_j (x_i - x_j)`` for each pair of components, which is zero at every one of those
blends. Its design is the ``{q, 3}`` simplex lattice, every blend in thirds, plus the
overall centroid for four or more components. A budget below that size gives a D-optimal
subset instead.

Analyse the runs with a Scheffé model. It has no intercept, because the proportions
sum to one:

.. code-block:: python

   import numpy as np

   from process_improve.experiments import analyze_experiment

   blends = ev.design_actual[["polymer", "solvent", "filler"]]
   # A simulated response stands in for the measurements here.
   rng = np.random.default_rng(1)
   blends["y"] = (
       10 * blends["polymer"] + 6 * blends["solvent"] + 3 * blends["filler"]
       + 8 * blends["polymer"] * blends["solvent"] + rng.normal(0, 0.1, len(blends))
   )
   fit = analyze_experiment(
       blends, response_column="y", model="scheffe_quadratic", analysis_type=["coefficients", "anova"]
   )
   print(fit["model_summary"]["formula"])  # y ~ -1 + (polymer + solvent + filler) ** 2

``"scheffe_linear"``, ``"scheffe_quadratic"``, ``"scheffe_special_cubic"`` and
``"scheffe_cubic"`` are accepted by ``analyze_experiment`` and ``evaluate_design``. In a
formula the full cubic's terms are written ``I(polymer * solvent * (polymer - solvent))``.
statsmodels recognises that the proportions carry an implicit intercept, so R-squared is
centred and the model degrees of freedom are one less than the number of terms, as for a
model with an intercept.

A blending coefficient tested against zero asks whether a pure component gives a
response of zero, which is rarely the question. So for a Scheffé model the ANOVA tests
the linear blending terms jointly (do the components blend differently at all?) and
each non-linear blending term, such as ``polymer:solvent``, on its own. Effects are
reported in the Cox direction (each linear coefficient minus the mean of the others),
and Lenth's method is not applied.

``evaluate_design`` defaults to the Scheffé quadratic model for a mixture
design and samples its constrained simplex. ``optimize_responses`` with the mixture
region searches the polytope and returns a blend that sums to one.

Mixture components with process factors
---------------------------------------

A formulation is usually made under process conditions that also matter, such as a cure
temperature or a mixing speed. Give the components and the process factors together, and
``generate_design`` builds one optimal design over the product of the constrained simplex
and the process box.

.. code-block:: python

   factors = [
       Factor(name="mill1", type="mixture", low=0.1, high=0.8),
       Factor(name="mill2", type="mixture", low=0.1, high=0.8),
       Factor(name="mill3", type="mixture", low=0.1, high=0.8),
       Factor(name="spray", low=-1, high=1),
       Factor(name="oil", low=-1, high=1),
   ]
   mills = generate_design(factors, model_type="scheffe_quadratic + main_effects", budget=12)
   print(mills.design_type, mills.n_runs)  # d_optimal 12
   print(mills.metadata["formula"])
   # -1 + mill1 + mill2 + mill3 + mill1:mill2 + mill1:mill3 + mill2:mill3 + spray + oil

This is the rolling-mill study of Goos and Jones (2011, section 6.2). Three mills share the
reduction in thickness of an aluminium sheet, and two coolant settings are the process
factors. The D-optimal design runs each of the six blends of the ``{3, 2}`` lattice twice,
once at each level of each coolant factor, as their Table 6.2 does.

The candidate runs are every candidate blend (the vertices, edge points, face centroids and
centroid of the constrained simplex) at every candidate process setting (the grid, and the
points where a constraint crosses it). The runs are chosen from them by the same exchange as
the other optimal designs. A constraint limits either the components or the process factors;
a constraint that names both is refused.

The model sets how the two kinds of factor combine. The counts are for three components and
two process factors.

.. list-table::
   :header-rows: 1

   * - ``model_type``
     - Terms
     - Terms for 3 and 2
   * - ``"main_effects"``
     - ``x_i`` and ``z_k``
     - 5
   * - ``"interactions"``
     - ``x_i``, ``x_i x_j``, ``x_i z_k`` and ``z_k z_l``
     - 13
   * - ``"quadratic"`` (or ``"kcv"``)
     - Those and ``z_k^2``: the model of Kowalski, Cornell and Vining (2000)
     - 15
   * - ``"scheffe_quadratic + main_effects"``
     - A Scheffé model plus a process model without its intercept
     - 8
   * - ``"scheffe_quadratic * interactions"``
     - Each Scheffé term times each process term, the intercept included
     - 24

The first three are the polynomial of that order in all the factors, less the terms that
``sum(x) = 1`` makes redundant. The intercept is ``sum(x_i)``, a square ``x_i^2`` is ``x_i``
less its products with the other components, and a process main effect ``z_k`` is
``sum(x_i z_k)``. Any Scheffé model can go on the left of ``+`` or ``*``, and
``"main_effects"``, ``"interactions"`` or ``"quadratic"`` on the right.

Both ``design`` and ``design_actual`` hold the components as proportions. The process
factors are coded in the first and in their own units in the second. The design records its
model's formula and its region, and ``evaluate_design`` uses both:

.. code-block:: python

   quality = evaluate_design(mills, metric=["d_efficiency", "g_efficiency"])
   print(round(quality["max_prediction_variance"], 3))  # 0.667

The largest prediction variance over the region is ``p / N = 8 / 12``. No 12-run design for
this model can do better, since the variances at the runs average ``p / N``. Goos and Jones
give the same two-thirds.

Fit the runs with the recorded formula. ``analyze_experiment`` keeps the components as
proportions, so the blending coefficients read as those of a Scheffé model do:

.. code-block:: python

   runs = mills.design.drop(columns="RunOrder")
   # A simulated response stands in for the measurements here.
   rng = np.random.default_rng(2)
   runs["reflectivity"] = (
       4.6 * runs["mill1"] + 5.0 * runs["mill2"] + 6.3 * runs["mill3"] + rng.normal(0, 0.01, len(runs))
   )
   fit = analyze_experiment(
       runs, response_column="reflectivity", model=mills.metadata["formula"], analysis_type="coefficients"
   )
   print([round(row["coefficient"], 1) for row in fit["coefficients"][:3]])  # [4.6, 5.0, 6.3]

``optimize_responses`` with the design's region searches blends that sum to one at process
settings inside the region.

A process factor can also be categorical, such as a catalyst type or a supplier. It enters
the model in ``L - 1`` sum-to-zero columns, so each level's effect is measured from the
average over the levels:

.. code-block:: python

   resins = [
       Factor(name="resin", type="mixture", low=0.2, high=0.7),
       Factor(name="filler", type="mixture", low=0.05, high=0.5),
       Factor(name="additive", type="mixture", low=0.05, high=0.3),
       Factor(name="T", low=140, high=180, units="degC"),
       Factor(name="catalyst", type="categorical", levels=["A", "B", "C"]),
   ]
   cure = generate_design(resins, model_type="interactions", budget=24)
   print(cure.metadata["n_parameters"])  # 17

The 17 terms are the six Scheffé blending terms, ``x_i:T`` and ``x_i:catalyst`` for each
component (three and six columns), and ``T:catalyst`` (two columns). A categorical factor
has no square, so ``"quadratic"`` adds only ``T^2``.

A model of mixture components has no intercept, and patsy then gives a categorical factor
a column for every level wherever the term below it is absent from the formula:
``T:catalyst`` in this model, where ``T`` enters only through ``x_i:T``. Those columns add up
to ones the other terms already span, so the model could not be estimated.
``evaluate_design`` and ``analyze_experiment`` therefore code every categorical factor of a
mixture-process model in ``L - 1`` columns, and the recorded formula needs no change.
``optimize_responses`` does not take categorical factors.

Split-plot designs (``hard_to_change``), fixed runs, candidate sets, blocks, and
constraints that join components and process factors are not yet available for
mixture-process designs, and ``generate_design`` refuses them.
