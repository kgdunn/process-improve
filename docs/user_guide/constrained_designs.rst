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
   ``random_seed`` makes the result reproducible.

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
