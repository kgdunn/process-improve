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

Choosing the run size and the design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- When ``n_runs`` is given it is used directly (it must be odd and larger than
  the number of second-order parameters :math:`1 + 2k + k(k-1)/2`).
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

The table below reports, for the default settings (the automatic smallest-size
search with ``max_candidates=6``), the size of the candidate half-pool, the run
size of the smallest design found, the resulting error degrees of freedom, the
number of ILP solves performed (the *iteration count*: one minimise-size probe
plus the no-good-cut re-solves), and the cumulative CBC solver time.  Times were
measured single-threaded on an ``x86_64`` machine with CPython 3.11 and CBC (the
solver bundled with PuLP); they are indicative and will vary by machine.

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
     - 6
     - 0.03
   * - 4
     - 40
     - 19
     - 4
     - 6
     - 0.05
   * - 5
     - 121
     - 27
     - 6
     - 6
     - 0.14
   * - 6
     - 364
     - 35
     - 7
     - 6
     - 5.6
   * - 7
     - 1093
     - 45
     - 9
     - 6
     - 39

The iteration count is fixed by ``max_candidates`` (each iteration is a full ILP
solve); the cost per iteration grows with the half-pool size :math:`(3^k - 1)/2`
and the number of orthogonality constraints :math:`k(k-1)/2`.  Three to six
factors solve in well under a second; seven factors take tens of seconds.  Beyond
seven factors the half-pool grows quickly, so a longer ``solver_options["time_limit"]``
(default 60 s) or a tighter ``n_runs`` is advisable.

Limitations
-----------

- **Foldover family only.**  ``generate_omars`` builds the (dominant) foldover
  OMARS family.  The rarer non-foldover members from the enumerated catalogue
  are a documented future extension.
- **Odd run counts.**  A foldover design has :math:`2h + 1` runs, so ``n_runs``
  must be odd.
- **Full second-order conditioning.**  The smallest OMARS designs are highly
  aliased in the second-order block by construction, so the D-efficiency of the
  *full* quadratic model is low; this is expected, and is exactly why
  :func:`~process_improve.experiments.analyze_omars` resolves the second-order
  terms in stages rather than fitting them all at once.  Request a larger
  ``n_runs`` for a better-conditioned design.

References
----------

- Nunez Ares, J. and Goos, P. (2020).  "Enumeration and multicriteria selection
  of orthogonal minimally aliased response surface designs."  *Technometrics*,
  62(1):21-36.
- Nunez Ares, J. and Goos, P. (2019).  "An integer linear programming approach
  to find trend-robust run orders of experimental designs."  *Journal of Quality
  Technology*.
