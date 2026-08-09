Generating OMARS Designs
========================

OMARS designs (orthogonal minimally aliased response surface designs) are
three-level designs that sit between screening designs and full response
surface designs.  Every main effect is orthogonal to every other main effect
*and* to all second-order terms (the pure quadratics and the two-factor
interactions), so all aliasing is confined to the second-order block, where it
is kept *minimal*.

:func:`~process_improve.experiments.generate_omars` constructs such designs on
demand with an integer linear program (ILP).  Unlike the minimal
conference-foldover member produced by ``generate_design(design_type="omars")``
(which is saturated and leaves no error degrees of freedom), the designs from
``generate_omars`` are sized to support a full second-order analysis with
:func:`~process_improve.experiments.analyze_omars`.

.. note::

   ``generate_omars`` requires the optional ``ilp`` extra (PuLP, which bundles
   the CBC solver)::

       pip install 'process-improve[ilp]'

Quick start
-----------

.. code-block:: python

   from process_improve.experiments import Factor, generate_omars, analyze_omars

   factors = [Factor(name=n, low=-1, high=1) for n in "ABCDE"]

   # Smallest foldover OMARS design that still leaves error degrees of freedom.
   result = generate_omars(factors)
   print(result.metadata["n_runs_selected"], result.metadata["expected_error_df"])
   print(result.metadata["omars_verified"])   # True

   # The design is ready for the staged OMARS analysis.
   design = result.design[result.factor_names]
   # ... collect responses y, then:
   analysis = analyze_omars(design, y)         # analysis.success is True

You can pin an exact (odd) run size or search a window:

.. code-block:: python

   result = generate_omars(factors, n_runs=29)
   result = generate_omars(factors, n_runs_range=(27, 41))

The method
----------

Every design produced here is a **foldover** :math:`[H; -H; 0]`: a half-design
:math:`H`, its mirror image :math:`-H`, and a single centre run.  The foldover
structure makes three of the four OMARS-defining conditions hold automatically,
which is what keeps the construction tractable.

For a design coded to :math:`\{-1, 0, +1\}`:

- **Balance** is automatic: a run :math:`h` and its mirror :math:`-h` cancel,
  so every main-effect column sums to zero.
- **Main effects clear of the two-factor interactions** is automatic: the term
  :math:`x_i x_a x_b` is an *odd* function, so the contributions from :math:`h`
  and :math:`-h` cancel.
- **Main effects clear of the pure quadratics** is automatic: :math:`x_i x_j^2`
  is odd in :math:`x_i`, so those contributions cancel too.
- **Quadratics are estimable** because the centre run makes each :math:`x_i^2`
  column take the value 0 at least once (so it is not constant).

The only condition that is **not** automatic is the mutual orthogonality of the
main effects.  That condition is *linear* in the binary "include half-run"
variables :math:`s_r`: for every pair of factors :math:`i < j`,

.. math::

   \sum_r \left( x_{r,i}\, x_{r,j} \right) s_r = 0 ,

and the run count is :math:`N = 2\sum_r s_r + 1`.  The ILP therefore selects a
half-design from the :math:`(3^k - 1)/2` distinct non-mirror three-level runs
subject to only :math:`k(k-1)/2` equality constraints.  Because the
coefficients are integers, the equalities are exact (no numerical tolerance
enters the optimisation); a floating-point :func:`~process_improve.experiments.is_omars`
re-check guards every accepted design as a sanity check.

The estimability frontier
~~~~~~~~~~~~~~~~~~~~~~~~~

A foldover cannot fit a full second-order model at just any run size, and the
threshold is higher than simply counting parameters.

Every second-order term is an **even** function of the factors, so the
quadratic and interaction columns of :math:`H` and :math:`-H` are *identical*.
The even block of the model matrix therefore has at most :math:`h + 1` distinct
rows, against :math:`1 + k(k+1)/2` columns (an intercept, :math:`k` quadratics
and :math:`k(k-1)/2` interactions).  The main effects live in the odd block and
contribute at most :math:`k` more, so for every foldover

.. math::

   \mathrm{rank}(X) \le k + \min\!\left(h + 1,\; 1 + \tfrac{k(k+1)}{2}\right),

with equality for half-designs in general position.  The full second-order
model is therefore estimable only from

.. math::

   N = k^2 + k + 1

runs: 13, 21, 31, 43 and 57 runs for three to seven factors.  Note that this is
strictly more than the parameter count :math:`1 + 2k + k(k-1)/2` from four
factors up, so "more runs than parameters" is *not* a sufficient test.
Sizing starts at this frontier, and ``n_runs`` below it is refused.

Choosing ``model="main_quadratic"`` drops the interactions from the model being
sized for, which lowers the frontier to the definitive screening design's
:math:`2k + 1` runs.

Choosing the run size and the design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- When ``n_runs`` is given it is used directly. It must be odd, and it must
  reach the **estimability frontier** described above; a smaller value is
  rejected rather than silently producing a design that cannot fit the model.
- Otherwise the solver minimises the run count within a window to return the
  smallest feasible design that still leaves error degrees of freedom.
- Several distinct designs are then enumerated at that run size by adding
  *no-good cuts* (forbidding a previously found selection), and the winner is
  chosen by ``selection_criterion``:

  - ``"dominance"`` (default) keeps the Pareto front on D-efficiency (higher is
    better) and the maximum second-order correlation (lower is better), then
    prefers the smallest, most efficient design.
  - ``"d_efficiency"`` maximises the D-efficiency of the full second-order model.
  - ``"min_second_order_correlation"`` minimises the largest second-order
    correlation.

- Optionally, ``satisfice`` sets *acceptability thresholds* that are applied
  **before** the dominance/criterion step: a design is kept only if it clears
  every threshold.  Supported keys are ``"d_efficiency"`` (a minimum) and
  ``"max_second_order_correlation"`` (a maximum), e.g.
  ``satisfice={"d_efficiency": 5.0, "max_second_order_correlation": 0.7}``.
  Together these implement the **satisficing-and-dominance** multicriteria
  selection of Nunez Ares and Goos (2020): first discard designs that fail the
  minimum bars (satisficing), then drop dominated designs and choose from the
  Pareto front (dominance).  This deliberately avoids collapsing several
  criteria into a single weighted score, which would hide the trade-offs.  A
  ``ValueError`` is raised if no enumerated design meets the thresholds.

The returned :class:`~process_improve.experiments.DesignResult` records the
provenance and a search report under ``metadata`` (``family``, ``sparsity``,
``expected_error_df``, ``d_efficiency``, ``max_second_order_correlation`` and an
``omars_search`` report with the ILP iteration count and solver time).

Performance: iterations and timing by factor count
--------------------------------------------------

The table below reports, for the automatic smallest-size search at
``n_restarts=8``, the size of the candidate half-pool, the run size of the
smallest design found, the resulting error degrees of freedom, the number of ILP
solves performed (the *iteration count*: one minimise-size probe plus the
no-good-cut re-solves), and the cumulative CBC solver time.  The run size is the
estimability frontier :math:`k^2 + k + 1` described above, and the error degrees
of freedom follow as :math:`N - (1 + 2k + k(k-1)/2)`; both are exact.  Times were
measured single-threaded on an ``x86_64`` machine with CPython 3.11 and CBC (the
solver bundled with PuLP); they are indicative and will vary by machine, and
they scale with ``n_restarts``.

.. list-table::
   :header-rows: 1
   :widths: 8 14 8 10 12 12

   * - Factors :math:`k`
     - Half-pool size
     - Runs :math:`N`
     - Error df
     - ILP solves
     - Solver time (s)
   * - 3
     - 13
     - 13
     - 3
     - 10
     - 0.1
   * - 4
     - 40
     - 21
     - 6
     - 10
     - 0.4
   * - 5
     - 121
     - 31
     - 10
     - 10
     - 2.2
   * - 6
     - 364
     - 43
     - 15
     - 10
     - 29
   * - 7
     - 1093
     - 57
     - 21
     - 10
     - see note

The iteration count is fixed by ``n_restarts`` (each iteration is a full ILP
solve); the cost per iteration grows with the half-pool size :math:`(3^k - 1)/2`,
the number of orthogonality constraints :math:`k(k-1)/2`, and the run size the
frontier demands.  Three to five factors solve in seconds; six takes about half a
minute.

At seven factors the solves are large enough to run into the per-solve
``solver_options["time_limit"]`` rather than finishing on their own, so no single
timing characterises them; budget minutes, raise the time limit above its 60 s
default, and expect the result to depend on where CBC was cut off.  Beyond seven
factors, pin ``n_runs`` or use ``model="main_quadratic"``, whose frontier is only
:math:`2k + 1`.

Limitations
-----------

- **Foldover family only.**  ``generate_omars`` builds the (dominant) foldover
  OMARS family.  The rarer non-foldover members from the enumerated catalogue
  are a documented future extension.
- **Odd run counts.**  A foldover design has :math:`2h + 1` runs, so ``n_runs``
  must be odd.
- **Second-order aliasing remains.**  Reaching the estimability frontier makes
  the full second-order model *fittable*; it does not make the second-order
  block orthogonal.  The quadratics and interactions stay mutually aliased -
  that is the "minimally aliased" in OMARS, not a defect - which is why
  :func:`~process_improve.experiments.analyze_omars` resolves them in stages
  rather than fitting them all at once.  Ask for more than the frontier run
  size to buy error degrees of freedom and lower second-order correlations.

References
----------

- Nunez Ares, J. and Goos, P. (2020).  "Enumeration and multicriteria selection
  of orthogonal minimally aliased response surface designs."  *Technometrics*,
  62(1):21-36.
- Nunez Ares, J. and Goos, P. (2019).  "An integer linear programming approach
  to find trend-robust run orders of experimental designs."  *Journal of Quality
  Technology*.
