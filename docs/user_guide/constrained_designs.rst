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
   ``"quadratic"``, squared terms. A categorical factor contributes indicator
   columns.

3. **Exchange.** A Fedorov exchange swaps one design run for one candidate at a
   time, taking the swap that increases the determinant of ``X'X`` most, until no
   swap improves it. Five random starts guard against a poor local optimum;
   ``random_state`` makes the result reproducible.

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

These designs come from the built-in exchange and do not need pyoptex. When pyoptex is
installed and no constraints are given, it is used instead; it is still needed for
split-plot designs (``hard_to_change``).

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
  constraints hold with equality. Every such choice of constraints is solved as one
  linear system, all of them in a single batched call, and the feasible solutions are
  the vertices.
* **Without a budget** the classical extreme-vertices design is returned: the
  vertices and the centroid, plus the edge midpoints for a quadratic model and the
  centroids of the 2-dimensional faces for a special cubic one. Those faces are the
  constrained counterpart of ternary blends, which the ``x_i x_j x_k`` terms need.
* **With a budget** a D-optimal subset is chosen from the vertices, edge midpoints,
  2-face and facet centroids, centroid and axial check blends, by the same exchange
  as above. Blends can repeat, which gives replicates for a pure-error estimate.

Mixture constraints must be linear in the proportions, since the vertex enumeration
relies on flat faces. On the full simplex (no bounds, no constraints) the classical
simplex-lattice and simplex-centroid designs are used, as before.

Analyse the runs with a Scheffé model. It has no intercept, because the proportions
sum to one:

.. code-block:: python

   from process_improve.experiments import analyze_experiment

   # x: the design's proportions; y: the measured response
   fit = analyze_experiment(x, y, model="scheffe_quadratic", analysis_type=["coefficients", "anova"])
   # formula: y ~ -1 + (polymer + solvent + filler) ** 2

``"scheffe_linear"``, ``"scheffe_quadratic"`` and ``"scheffe_special_cubic"`` are
accepted by ``analyze_experiment`` and ``evaluate_design``. statsmodels recognises
that the proportions carry an implicit intercept, so R-squared is centred and the
model degrees of freedom are one less than the number of terms, as for a model with an
intercept.

A blending coefficient tested against zero asks whether a pure component gives a
response of zero, which is rarely the question. So for a Scheffé model the ANOVA tests
the linear blending terms jointly (do the components blend differently at all?) and
each non-linear blending term, such as ``polymer:solvent``, on its own. Effects are
reported in the Cox direction (each linear coefficient minus the mean of the others),
and Lenth's method is not applied.

``evaluate_design`` defaults to the Scheffé quadratic model for a mixture
design and samples its constrained simplex. ``optimize_responses`` with the mixture
region searches the polytope and returns a blend that sums to one.
