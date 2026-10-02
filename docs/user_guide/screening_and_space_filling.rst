Supersaturated and Space-Filling Designs
========================================

Two families sit at opposite ends of the run budget. A supersaturated design screens
more factors than it has runs; a space-filling design spends its runs on covering the
region evenly, with no model in mind.

Supersaturated designs: more factors than runs
----------------------------------------------

With ``k`` factors, any design that estimates every main effect needs at least
``k + 1`` runs. A supersaturated design uses fewer, so the columns cannot all be
orthogonal. It works under *effect sparsity*: when only a few of the many factors
matter, a selection method (stepwise regression, the lasso) can still find them.

``generate_design`` chooses one automatically when the budget is below ``k + 1``:

.. code-block:: python

   from process_improve.experiments import Factor, generate_design

   factors = [Factor(name=f"X{i + 1}", low=0, high=1) for i in range(22)]
   result = generate_design(factors, budget=12)
   m = result.metadata
   print(result.design_type, result.n_runs)                      # supersaturated 12
   print(round(m["e_s2"], 2), round(m["e_s2_lower_bound"], 2))   # 6.86 6.86
   print(m["max_abs_s"], m["n_fully_aliased_pairs"])             # 4 0

The quality measure is ``E(s^2)``, the average over all pairs of columns of
``s_ij^2``, where ``s_ij = x_i' x_j`` is zero for orthogonal columns. This 12-run
design for 22 factors reaches the theoretical lower bound for balanced designs. The
largest ``|s_ij|`` is 4 out of a possible 12, and no pair of columns is fully aliased.

The construction is Lin's (1993): take a Hadamard matrix of order ``2n``, keep the
``n`` rows where one *branching* column is +1, and drop that column. Every branching
column is tried and the best half-fraction is kept. Hadamard matrices built by
Kronecker products (orders 16, 24, 32 in pyDOE3) leave identical columns in their
half-fractions, so Paley's construction is used wherever it exists (orders ``N`` with
``N - 1`` a prime congruent to 3 mod 4, including 12, 20, 24, 32, 44 and 60). When a
requested run count forces fully aliased pairs, a warning names the run count that
avoids them.

Space-filling designs: no model assumed
---------------------------------------

Space-filling designs suit computer experiments, Gaussian-process and machine-learning
surrogates, and exploration where the shape of the response is unknown. ``budget``
sets the number of runs; the default is ``10 * k``.

.. code-block:: python

   box = [Factor(name=n, low=0, high=10) for n in "ABC"]
   for design_type in ["latin_hypercube", "maximin_lhs", "uniform", "sobol", "halton", "maximin"]:
       m = generate_design(box, design_type=design_type, budget=20, random_seed=0).metadata
       print(design_type, round(m["min_distance"], 2), round(m["centered_l2_discrepancy"], 4))

.. list-table:: Twenty runs in three factors (coded distances)
   :header-rows: 1

   * - Design type
     - Smallest distance between runs (higher is more spread)
     - Centred L2 discrepancy (lower is more uniform)
   * - ``latin_hypercube``
     - 0.21
     - 0.0062
   * - ``maximin_lhs``
     - 0.70
     - 0.0043
   * - ``uniform``
     - 0.57
     - 0.0030
   * - ``sobol``
     - 0.35
     - 0.0051
   * - ``halton``
     - 0.40
     - 0.0039
   * - ``maximin``
     - 1.00
     - 0.1025

The two columns measure different things. A maximin design keeps runs as far apart as
possible, which pushes them to the faces and corners of the region; a uniform design
matches the fraction of runs in every sub-box to that sub-box's volume. The Latin
hypercube types also guarantee that each factor's range is cut into ``n`` slices with
one run in each, which the maximin and sequence designs do not.

Constrained regions and mixtures
--------------------------------

``"sobol"``, ``"halton"`` and ``"maximin"`` also work inside a constrained region or a
constrained mixture, through :class:`~process_improve.experiments.DesignRegion`:

.. code-block:: python

   from process_improve.experiments import Constraint

   factors = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
   heat = Constraint(expression="3*T + 5*D <= 600")
   result = generate_design(factors, design_type="maximin", budget=12, constraints=[heat])

* ``"sobol"`` and ``"halton"`` map the sequence onto the region (onto the simplex for
  a mixture, through a map that keeps uniform points uniform) and skip its infeasible
  points.
* ``"maximin"`` chooses the runs from a dense uniform sample of the region plus its
  boundary points, by a farthest-point build followed by exchanges.

The Latin hypercube types refuse a constrained or mixture region: their guarantee is
about each factor's range, which a constraint cuts into pieces of unequal length.
