Design families and how each one is checked
===========================================

Every design below comes from one call, ``generate_design(factors, design_type, ...)``.
Each construction is checked against its published definition by a test class in
`tests/test_doe_conformance.py
<https://github.com/kgdunn/process-improve/blob/main/tests/test_doe_conformance.py>`_,
so a change that breaks the defining property fails the test suite.

.. list-table::
   :header-rows: 1
   :widths: 22 20 40 18

   * - Family
     - ``design_type``
     - Defining property checked
     - Test class
   * - Full factorial, two-level and mixed-level
     - ``"full_factorial"``
     - Every combination of levels exactly once, two-level and mixed-level
     - ``TestFullFactorial``
   * - Fractional factorial
     - ``"fractional_factorial"``
     - Resolution and run count of the minimum-aberration fraction
     - ``TestFractionalFactorial``
   * - Plackett-Burman
     - ``"plackett_burman"``
     - Columns of a Hadamard matrix: :math:`H'H = N I`
     - ``TestPlackettBurman``
   * - Definitive screening (DSD)
     - ``"dsd"``
     - Conference matrix :math:`C'C = (m-1) I`, foldover, main effects orthogonal
       to every second-order term; fake factors from a run budget
     - ``TestDefinitiveScreening``, ``TestDSDFakeFactors``
   * - Supersaturated
     - ``"supersaturated"``
     - Balanced columns, :math:`E(s^2)` at or near its lower bound
     - ``TestSupersaturated``
   * - Taguchi orthogonal array
     - ``"taguchi"``
     - Strength 2: every pair of columns balanced
     - ``TestTaguchi``
   * - Box-Behnken
     - ``"box_behnken"``
     - Published run counts (Box and Behnken 1960), including the 6- and
       7-factor designs
     - ``TestBoxBehnken``
   * - Central composite (CCD)
     - ``"ccd"``
     - Rotatable :math:`\alpha = F^{1/4}`; orthogonal :math:`\alpha` makes the
       quadratic columns orthogonal
     - ``TestCentralComposite``
   * - OMARS
     - ``"omars"``
     - Main effects orthogonal to each other and to every second-order term
     - ``TestOMARS``
   * - Mixture: simplex and extreme vertices
     - ``"mixture"``
     - Simplex-centroid size :math:`2^q - 1`; every blend sums to 1 and respects
       its bounds
     - ``TestMixture``
   * - D-, I-, A- and E-optimal
     - ``"d_optimal"``, ``"i_optimal"``, ``"a_optimal"``, ``"e_optimal"``
     - Runs satisfy the constraints; each criterion is best on its own measure
     - ``TestOptimal``
   * - G- and K-optimal
     - ``"g_optimal"``, ``"k_optimal"``
     - Best G-efficiency and smallest condition number; the :math:`2^k`
       factorial is G-optimal for a first-order model
     - ``TestGAndKOptimal``
   * - Space-filling
     - ``"latin_hypercube"``, ``"maximin_lhs"``, ``"maximin"``, ``"uniform"``,
       ``"sobol"``, ``"halton"``, ``"maxpro"``
     - One run per slice (Latin hypercube), spread, discrepancy, Sobol balance,
       MaxPro projections in constrained and mixture regions
     - ``TestSpaceFilling``, ``TestMaxPro``

Blocking (``n_blocks``) is checked separately in ``tests/test_blocking.py``: a
two-level factorial is blocked by confounding high-order interactions, so main
effects and unconfounded two-factor interactions keep their full information; a
central composite design is blocked by its cube and axial portions, orthogonally to
every term of the quadratic model with the default axial distance; other designs
are blocked by exchange.
