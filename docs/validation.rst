.. _validation:

Validation
==========

This page records how the numbers this package computes are checked against references that
do not depend on it: what a source prints, and what other software computes. Each comparison
names its reference, with its version and date, and gives its result and what it found. The
last sections say what the comparisons do not show, and how to repeat them.

.. list-table:: The comparisons
   :header-rows: 1
   :widths: 26 26 48

   * - What is checked
     - Against
     - How
   * - Designs, and the analysis of designed experiments
     - 22 books, handbooks and papers
     - The test suite recomputes every printed number on every change
       (:ref:`validation-sources`).
   * - How a design is built
     - Nine packages in R and Python
     - Each package builds the same designs, and one checker measures both
       (:ref:`validation-designs`).
   * - Mixed models, sensory analysis, latent variable models, control charts and regression
     - R packages, SIMCA-P, MATLAB, statsmodels and scikit-learn
     - The tests hold values computed in the other software (:ref:`validation-analysis`).
   * - The package as a reader uses it
     - The companion book
     - Every Python example of the book runs against the released package
       (:ref:`validation-book`).

Two rules hold for every comparison:

- A difference is never dropped. It stays in the record with its cause: an error in the
  source, another convention, a limit of this package, or a cause not known yet.
- A result is a record of one run: it names the versions, the date and the commit, and the
  command that repeats it.

.. _validation-sources:

Against printed sources
-----------------------

The reference catalogue (:mod:`process_improve.experiments.reference_catalogue`) holds the
designs and experiments of 22 sources: textbooks such as Box, Hunter and Hunter
(1978), Box and Draper (2007) and Montgomery (2013), the NIST/SEMATECH e-Handbook, and papers
on minimum aberration, definitive screening and OMARS designs. Each entry names the page it is
printed on and holds every number the source prints for it: a run count, an alias chain, an
effect, a sum of squares, a standard error.

The test suite recomputes those numbers on every change
(``tests/test_reference_catalogue.py``). A number agrees when it is within half a unit of the
last digit the source prints. A number that does not agree must record its cause, and so must
a number that is not recomputed; the tests fail without it.

.. list-table:: The catalogue on 22 November 2026 (commit ``37f81b6``)
   :header-rows: 1
   :widths: 60 20 20

   * - Printed numbers
     - Count
     - Share
   * - In the catalogue
     - 42,897
     -
   * - Recomputed by the library
     - 37,341
     - 87.0%
   * - Recomputed, and the same as the source prints
     - 34,761
     - 93.1% of those recomputed
   * - Recomputed, and different for a recorded cause
     - 2,580
     - 6.9% of those recomputed
   * - Not recomputed, for a recorded cause
     - 5,556
     - 13.0%

The causes of the numbers that differ:

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Cause
     - Numbers
     - Meaning
   * - ``source_error``
     - 909
     - The source misprints the number or works it out wrongly; the library's value follows
       from the source's own data.
   * - ``source_rounding``
     - 888
     - The source works from values it has already rounded, or prints fewer digits than it
       carries.
   * - ``method``
     - 415
     - The source uses another convention, definition, model or software.
   * - ``incorrect_option``
     - 92
     - An option of a quiz that its answer key marks incorrect, as the library finds it to be.
   * - ``library``
     - 1
     - A bug or a limit of this package.
   * - ``unexplained``
     - 275
     - The cause is not known yet.

Of the 5,556 numbers that are not recomputed, 2,219 have nothing to recompute (a page number, a
count the source states), 1,555 need a function the library does not have yet, 1,073 depend on
data the source does not print, and 709 are computed but not yet read by a check.

`Reproduced from the books <https://factori.al/catalog/reproduced/>`__ shows these counts as they
stand today, for the whole catalogue and for each source, with an example of every cause.
:ref:`design-catalog` describes the records, and
:func:`~process_improve.experiments.reference_catalogue.reproduce_reference_catalogue` returns
one row per number.

.. _validation-designs:

Against other software: how a design is built
---------------------------------------------

A printed number says what a design should be. Another package says what a second,
independent construction gives. Of the catalogue's numbers, 3,377 depend on how a design is
built: run counts, generators, defining relations, alias chains, resolutions, word length
patterns, axial distances and block arrangements.

``scripts/design_crosscheck/compare.py`` checks each of them twice with the catalogue's own
checker: on the design process-improve builds, and on the design the other package builds.
Only the construction changes, so a difference is a difference in the design.

.. csv-table:: Design construction against other packages, 22 November 2026 (commit ``37f81b6``)
   :file: validation/design_crosscheck.csv
   :header-rows: 1
   :widths: 30 38 8 8 8 8

How to read the table:

- **Checks** are the numbers of the kinds of design the package builds. **Agree** means both
  packages give the number the source prints. **Differ** means they give different answers.
  **Not built** means the package has no design for the request, or declines it.
- The numbers that neither package gives as printed are left out: the catalogue records their
  cause, most often a misprint in the source.
- No package here builds OMARS designs, so their 47 checks are in no row.
- A difference is examined before it is counted as a finding. Several first differences came
  from how a request was put to the other package, for example which centre runs go to which
  block.

**How independent is each package?** process-improve's two-level factorial, Plackett-Burman,
Box-Behnken, central composite and Taguchi arrays began as copies of pyDOE3 1.6.2's
(``experiments/_classical.py``), and a test still compares them element by element. Agreement
with pyDOE3 and pydoe on those arrays is therefore expected, and is not evidence. The evidence
from those two packages is where they differ, because each difference was traced to a printed
source. The R packages, dexpy, OApackage and definitive-screening-design share no code with
process-improve.

.. _validation-frf2:

FrF2 and DoE.base (R)
~~~~~~~~~~~~~~~~~~~~~

FrF2 2.3-5 built the two-level full and fractional factorials, the Plackett-Burman designs and
the blocked designs. DoE.base 1.2-5 built the factorials with more than two levels and the
orthogonal arrays.

- **Fractional factorials and Plackett-Burman designs.** No difference: FrF2 gives every number
  that process-improve gives as printed.
- **Blocks.** FrF2's default blocking confounds interactions of the same orders as the sources
  for every blocked full factorial, and the same alias classes for the blocked fractions of Box
  and Draper (2007), Table 5.14.
- **Minimum aberration.** For the 30 sizes from 8 to 128 runs that both packages build, the two
  designs have the same word length pattern. For 69 larger sizes FrF2's catalogue has a design
  and process-improve's search stops at its limit (A12 in `issue 788 <https://github.com/kgdunn/process-improve/issues/788>`__).
- **Orthogonal arrays.** DoE.base's arrays have the size and the levels of Taguchi's L8, L9,
  L12, L16 and L27, but not the layout of the printed tables, so the checks that compare runs
  differ. Their equivalence was not tested.
- **Not built.** A Plackett-Burman design of 136 runs (Schmidt and Launsby, 2000) and a
  factorial of one factor.

Two observations on FrF2:

1. Permitting two-factor interactions to be aliased with blocks gives a blocking that confounds
   one, where the default confounds none. The same holds for 64 runs in 4 blocks.

   .. code-block:: r

      library(FrF2)
      d1 <- FrF2(32, 5, blocks = 2, randomize = FALSE)
      d2 <- FrF2(32, 5, blocks = 2, randomize = FALSE,
                 alias.block.2fis = TRUE)
      design.info(d1)$aliased.with.blocks
      # "none": the blocks confound ABCDE
      design.info(d2)$aliased.with.blocks
      # "AE"

2. ``pb()`` stops with an internal error above 100 runs, where its help asks only for a
   multiple of 4.

   .. code-block:: r

      pb(104)
      # Error in pb.list[[2]][[which(pb.list[[1]] == nruns)]] :
      #   attempt to select less than one element in get1index

:download:`The checks on which DoE.base differs <validation/differences_DoE.base.csv>`.

.. _validation-rsm:

rsm (R)
~~~~~~~

rsm 2.10.6 built the central composite and Box-Behnken designs.

- **Box-Behnken designs.** No difference.
- **Central composite designs.** Run counts and axial distances agree: on full and fractional
  cubes, for a rotatable, a spherical or a given distance, and in blocks.
- **Centre runs in each block.** rsm takes them block by block (``n0``), and so gives axial
  distances of NIST 5.3.3.6.4 that process-improve does not, because it takes one total (A11 in
  `issue 788 <https://github.com/kgdunn/process-improve/issues/788>`__).
- **A rotatable distance on a small cube.** rsm declines ``alpha = "rotatable"`` when the cube is
  below resolution V, where no distance makes the design rotatable. Box and Draper (2007) print
  the fourth root of the cube's run count for such cubes, and process-improve gives that number.
- **The word "orthogonal".** rsm's ``alpha = "orthogonal"`` is the distance for orthogonal
  blocking. Davies (1956) uses the word for the distance that makes the quadratic terms
  orthogonal, which is the one process-improve gives for his designs.

:download:`The checks on which rsm differs <validation/differences_rsm.csv>`.

.. _validation-daewr:

daewr (R)
~~~~~~~~~

daewr 1.2-11 built the definitive screening designs, with ``DefScreen``.

- **4, 6, 8 and 10 factors.** No difference.
- **5, 7, 9, 11 and 12 factors.** ``DefScreen`` returns the designs of Jones and Nachtsheim
  (2011) run for run, which process-improve does not. process-improve builds these sizes from a
  conference matrix (Xiao, Lin and Bai, 2012): 2m + 3 runs for an odd number of factors m, the
  run count later sources print, and 25 runs for 12 factors. The checks of run counts, and of
  the measures that follow from them, therefore differ.
- **Not built.** Designs for 3 factors, and for more than 12.

:download:`The checks on which daewr differs <validation/differences_daewr.csv>`.

One observation on daewr: for 5, 7, 9, 11 and 12 factors the main effects of its designs are
correlated, where the conference-matrix designs are orthogonal.

.. code-block:: r

   library(daewr)
   for (m in 4:12) {
     d <- DefScreen(m = m, c = 0, center = 0, randomize = FALSE)
     r <- cor(as.matrix(d)); diag(r) <- 0
     cat(m, "factors:", nrow(d), "runs, largest |r| =",
         round(max(abs(r)), 3), "\n")
   }
   #  5 factors: 11 runs, largest |r| = 0.25
   #  7 factors: 15 runs, largest |r| = 0.167
   #  9 factors: 19 runs, largest |r| = 0.125
   # 11 factors: 23 runs, largest |r| = 0.3
   # 12 factors: 25 runs, largest |r| = 0.182
   # 4, 6, 8 and 10 factors: 0

.. _validation-pydoe:

pydoe and pyDOE3 (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~

pyDOE3 1.6.2 is the release process-improve's basic arrays were copied from; it is archived.
pydoe 1.6.0 continues it. Each difference below was traced to a printed source and reported to
pydoe with an example that reproduces it, and a fix is proposed for each.

.. list-table::
   :header-rows: 1
   :widths: 58 12 15 15

   * - Difference
     - Checks
     - Issue
     - Fix proposed
   * - ``bbdesign``: from 6 factors on, not the designs Box and Behnken published
     - 21
     - `186 <https://github.com/pydoe/pydoe/issues/186>`__
     - `190 <https://github.com/pydoe/pydoe/pull/190>`__
   * - ``small_composite_design``: from 5 factors on, the full quadratic model cannot be fitted
     - none
     - `187 <https://github.com/pydoe/pydoe/issues/187>`__
     - `191 <https://github.com/pydoe/pydoe/pull/191>`__
   * - ``ccdesign`` and ``star``: the axial distance on a fractional cube uses the run count of
       the full cube
     - 17
     - `188 <https://github.com/pydoe/pydoe/issues/188>`__
     - `189 <https://github.com/pydoe/pydoe/pull/189>`__
   * - ``pbdesign``: no design for 28, 36, 44 and other run counts, and a 24-run design that
       is the 12-run design doubled
     - 13
     - `192 <https://github.com/pydoe/pydoe/issues/192>`__
     - `196 <https://github.com/pydoe/pydoe/pull/196>`__
   * - Orthogonal arrays: two names that do not describe their arrays, an L64 that is not
       orthogonal, and no L18 with one two-level column
     - 1
     - `193 <https://github.com/pydoe/pydoe/issues/193>`__
     - `195 <https://github.com/pydoe/pydoe/pull/195>`__
   * - ``fracfact_by_res``: resolution V returned where the same runs give resolution VI or VII
     - 2
     - `194 <https://github.com/pydoe/pydoe/issues/194>`__
     - `197 <https://github.com/pydoe/pydoe/pull/197>`__

The checks counted are those of the full comparison; issue 187 was found by reading the code,
because the comparison builds a central composite design from its own cube.

- **The word "orthogonal".** Four more checks differ, on the axial distances of Davies (1956):
  ``alpha="orthogonal"`` is the distance for orthogonal blocking, as in rsm.
- **Not built by pydoe.** A blocked factorial without the block generators given, a blocked
  fraction, a spherical axial distance, and OMARS designs. ``fracfact_opt`` gave no answer in
  60 seconds for 18 checks.
- **pyDOE3 1.6.2** differs on 93 more checks than pydoe 1.6.0, on fractional factorials chosen
  by resolution or by aberration and on central composite designs with such a cube. pydoe gives
  the printed numbers for all of them.

:download:`The checks on which pydoe differs <validation/differences_pydoe.csv>`, and
:download:`those of pyDOE3 <validation/differences_pyDOE3.csv>`.

.. _validation-dexpy:

dexpy (Python)
~~~~~~~~~~~~~~

dexpy 0.12, from Stat-Ease, built two-level factorials, Box-Behnken designs and central
composite designs on a full cube.

- **What it builds agrees.** No difference.
- **Not built.** dexpy chooses a fraction from its run count: it takes no generators, no
  resolution and no blocks, so most of the catalogue's fractions cannot be asked of it. It has
  no central composite design on a fractional cube, and no factor with more than two levels.

.. _validation-oapackage:

OApackage (Python and C++)
~~~~~~~~~~~~~~~~~~~~~~~~~~

OApackage 2.7.21 built the definitive screening designs, from the conference designs it
enumerates, and the orthogonal arrays, by extending an array one column at a time.

- **Definitive screening designs.** No difference, for an even or an odd number of factors.
- **Orthogonal arrays.** The 12-run array has the size and the levels of Taguchi's L12 but not
  the layout of the printed table, so the one check that compares its runs differs. The others
  agree.
- **Not built.** The search for a 24-factor design did not end in 60 seconds.

:download:`The check on which OApackage differs <validation/differences_oapackage.csv>`.

.. _validation-dsd-package:

definitive-screening-design (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

definitive-screening-design 0.5.1 built the definitive screening designs. No difference, and
nothing not built.

Three packages therefore build these designs alike, from a conference matrix: process-improve,
OApackage and definitive-screening-design. daewr returns the earlier designs.

.. _validation-analysis:

Against other software: analysis
--------------------------------

Outside design construction, the tests hold values that other software computed, on data sets
kept with the tests. A change that moves one of them fails the test.

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Method
     - Reference
     - What is compared
   * - Split-plot analysis by REML
     - lme4 1.1-36, lmerTest 3.1-3 and pbkrtest 0.5-3 (R 4.4.2)
     - Variance components, estimates, standard errors, Satterthwaite and Kenward-Roger
       degrees of freedom, Type III F-tests
   * - Mixed assessor model of a sensory panel
     - SensMixed 2.1-0 (R 3.4.4)
     - Every Type I test of every attribute of the TVbo panel: F, sums of squares, degrees
       of freedom and p
   * - PCA with missing values (NIPALS)
     - mixOmics 6.34.0
     - Scores, loadings and eigenvalues
   * - Partial robust M-regression
     - chemometrics 1.4.4
     - Coefficients, intercept and observation weights
   * - PCA and PLS
     - SIMCA-P 14.1
     - Scores, loadings, R2, the squared prediction error (DModX) and Hotelling's T2, on the
       LDPE data among others
   * - PCA and PLS
     - scikit-learn
     - The same fit as ``PCA`` and ``PLSRegression``
   * - Batch PCA
     - A MATLAB latent-variable toolkit
     - Scores, loadings and R2 on simulated styrene-butadiene rubber batches
   * - Control charts
     - qcc (R)
     - Centre line, standard deviation and limits of the individuals chart
   * - Least squares, t-tests and analysis of variance
     - R (``lm``, ``t.test``), statsmodels, SciPy
     - Coefficients, standard errors, p-values, sums of squares, adjusted p-values
   * - Generalized linear models, Gaussian processes, REML and least squares means
     - JMP, Minitab, SAS and Design-Expert output, as textbooks print it
     - The tables of Montgomery (2013) and Goos and Jones (2011)

The tests, in the order of the table, are in ``tests/``:
``test_experiments_split_plot.py``,
``test_sensory_mam_reml.py``,
``test_pca_nipals_mixomics_crosscheck.py``,
``test_multivariate_prm.py``,
``test_multivariate.py``,
``test_sklearn_compat.py``,
``batch/test_sbr_reference.py``,
``test_monitoring.py``,
``test_regression.py``,
``test_univariate.py``,
``test_analysis.py``.
The last row is checked through the reference catalogue.

The R values are recomputed in a Docker image with dated snapshots of CRAN (``tools/r``), so a
fixture can be made again with the same versions.

.. _validation-book:

Against the companion book
--------------------------

`Process Improvement using Data <https://learnche.org/pid>`_ shows its computations with this
package. `Its repository <https://github.com/kgdunn/pid-book>`_ runs every Python example of
the book against the released package on each change, in toctree order, one chapter as one
script. A result the text echoes beside a ``print`` is compared with what the code prints, so a
number that stops reproducing fails the chapter. A second run against this repository's
``main`` branch shows a coming change before it is released.

.. _validation-findings:

What the comparisons found
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Where
     - Finding
     - Record
   * - The sources
     - 909 printed numbers that a source misprints or works out wrongly, each with the value
       its own data give.
     - `Reproduced from the books <https://factori.al/catalog/reproduced/>`__
   * - pydoe
     - Six differences from the printed sources, in ``bbdesign``, ``small_composite_design``,
       ``ccdesign``, ``pbdesign``, the orthogonal arrays and ``fracfact_by_res``.
     - :ref:`validation-pydoe`
   * - FrF2
     - A blocking that confounds a two-factor interaction where the default confounds none,
       and an internal error of ``pb()`` above 100 runs.
     - :ref:`validation-frf2`
   * - daewr
     - Definitive screening designs with correlated main effects for 5, 7, 9, 11 and 12
       factors.
     - :ref:`validation-daewr`
   * - process-improve
     - No way to give the centre runs of each block of a central composite design, and a
       minimum-aberration search that stops where published designs exist.
     - A11 and A12 in `issue 788 <https://github.com/kgdunn/process-improve/issues/788>`__
   * - process-improve
     - 1,555 printed numbers that need a function the library does not have yet, 275 that
       differ for a cause not known yet, and 1 that differs for a bug or limit of the library.
     - `The catalogue's tracking issue <https://github.com/kgdunn/process-improve/issues/804>`__

.. _validation-limits:

What this page does not show
----------------------------

- **Optimal designs are not cross-checked with other software.** A D- or I-optimal search
  starts from random designs, so two packages need not find the same design. The catalogue
  checks the criteria of the designs a source prints.
- **OMARS designs have no second implementation here.** They are checked against the papers
  that define them only.
- **Equal numbers, not equal layouts.** A design passes when it gives the printed numbers. Two
  orthogonal arrays that differ in the order of their rows and columns were not tested for
  equivalence.
- **Agreement with pyDOE3 on the basic arrays is by construction**, as said above.
- **A result is one run.** The versions and dates are those named in each table. A later
  release of a package can change its row.
- **The references have limits too.** The SIMCA-P values are from version 14.1, recorded in
  2020, and no later version was tried. SensMixed 2.1-0 was archived from CRAN in 2020 and runs
  only on R 3.4.4.

.. _validation-reproduce:

Repeating a comparison
----------------------

The catalogue:

.. code-block:: bash

   uv run pytest tests/test_reference_catalogue.py --no-cov

.. code-block:: python

   from process_improve.experiments.reference_catalogue import reproduce_reference_catalogue

   numbers = reproduce_reference_catalogue()  # one row per printed number; several minutes
   numbers.groupby(["checked", "agrees", "cause"], dropna=False).size()

The designs of another package, here the R packages (``scripts/design_crosscheck/README.md``
lists what each package needs):

.. code-block:: bash

   DOE_LIB=R DOE_OUT=results uv run python scripts/design_crosscheck/compare.py all
   DOE_OUT=results uv run python scripts/design_crosscheck/tabulate.py

The first command writes one row per check; the second writes the table of this page and the
lists of differences it links to.
