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
     - Eight packages in R and Python
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

The reference catalog (:mod:`process_improve.experiments.reference_catalogue`) holds the
designs and experiments of 22 sources: textbooks such as Box, Hunter and Hunter
(1978), Box and Draper (2007) and Montgomery (2013), the NIST/SEMATECH e-Handbook, and papers
on minimum aberration, definitive screening and OMARS designs. Each entry names the page it is
printed on and holds every number the source prints for it: a run count, an alias chain, an
effect, a sum of squares, a standard error.

The test suite recomputes those numbers on every change
(``tests/test_reference_catalogue.py``). A number agrees when it is within half a unit of the
last digit the source prints. A number that does not agree must record its cause, and so must
a number that is not recomputed; the tests fail without it.

.. list-table:: The catalog on 22 November 2026 (commit ``37f81b6``)
   :header-rows: 1
   :widths: 60 20 20

   * - Printed numbers
     - Count
     - Share
   * - In the catalog
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
stand today, for the whole catalog and for each source, with an example of every cause.
:ref:`design-catalog` describes the records, and
:func:`~process_improve.experiments.reference_catalogue.reproduce_reference_catalog` returns
one row per number.

.. _validation-designs:

Against other software: how a design is built
---------------------------------------------

A printed number says what a design should be. Another package says what a second,
independent construction gives. Of the catalog's numbers, 3,377 depend on how a design is
built: run counts, generators, defining relations, alias chains, resolutions, word length
patterns, axial distances and block arrangements.

``scripts/design_crosscheck/compare.py`` checks each of them twice with the catalog's own
checker: on the design process-improve builds, and on the design the other package builds.
Only the construction changes, so a difference is a difference in the design.

.. csv-table:: Design construction against other packages, 22 to 24 November 2026
   :file: validation/design_crosscheck.csv
   :header-rows: 1
   :widths: 30 42 8 12 8

How to read the table:

- **Checks** are the printed numbers measured on a design the package built. **As printed**
  means the package gives the number the source prints. **Differ** means it gives another.
  Each package is measured against the source, as process-improve is: a number that only
  process-improve misses is process-improve's finding (:ref:`validation-findings`).
- A request outside what a package offers, such as blocks or a kind of design it does not
  build, is not counted: it is the package's scope, not a fault. Each package's section below
  says what it does not offer.
- The numbers that neither package gives as printed are left out: the catalog records their
  cause, most often a misprint in the source.
- No package here builds OMARS designs, so their 47 checks are in no row.
- A difference is examined before it is counted as a finding. Several first differences came
  from how a request was put to the other package, for example which center runs go to which
  block.

**How independent is each package?** process-improve's two-level factorial, Plackett-Burman,
Box-Behnken, central composite and Taguchi arrays began as copies of those of pyDOE3 1.6.2,
the package that pydoe continues (``experiments/_classical.py``), and a test still compares
them element by element. Agreement with pydoe on those arrays is therefore expected, and is not
evidence. The R packages, dexpy, OApackage and definitive-screening-design share no code with
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
- **Minimum aberration.** For the 44 sizes from 8 to 128 runs that both packages build, the two
  designs have the same word length pattern: every size up to 32 runs and 25 factors, and 64
  and 128 runs up to 11 factors. For 55 larger sizes, in 64 and 128 runs, FrF2's catalog has a
  design and process-improve's search stops at its limit (A12 in `issue 788 <https://github.com/kgdunn/process-improve/issues/788>`__).
- **Orthogonal arrays.** DoE.base's arrays are Taguchi's L8, L9, L12, L16 and L27 in another
  layout: the same runs once the order of the runs and columns and the labels of each column's
  levels are changed. An array is the same array in any layout, so they agree.
- **Not counted.** A factorial of one factor, which FrF2 does not offer, and a Plackett-Burman
  design of 136 runs (Schmidt and Launsby, 2000), for the second reason below.

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

.. _validation-rsm:

rsm (R)
~~~~~~~

rsm 2.10.6 built the central composite and Box-Behnken designs.

- **Box-Behnken designs.** No difference.
- **Central composite designs.** No difference. Run counts and axial distances are as printed:
  on full and fractional cubes, for a rotatable, a spherical or a given distance, and in
  blocks.
- **Center runs in each block.** rsm takes them block by block (``n0``), as process-improve does
  (``n_center_points=[2, 2, 2]``), and both give the axial distances that NIST 5.3.3.6.4 prints.
- **Outside what rsm offers.** A rotatable distance on a cube below resolution V, where no
  distance makes the design rotatable: rsm declines it, and Box and Draper (2007) print the
  fourth root of the cube's run count. Davies' (1956) orthogonal design: rsm's
  ``alpha = "orthogonal"`` is the distance for orthogonal blocking, a different design.

.. _validation-daewr:

daewr (R)
~~~~~~~~~

daewr 1.2-11 built the definitive screening designs, with ``DefScreen``.

- **4, 6, 8 and 10 factors.** No difference.
- **5, 7, 9, 11 and 12 factors.** ``DefScreen`` returns the designs of Jones and Nachtsheim
  (2011) run for run, and gives the numbers that paper prints. Later sources print the designs
  built from a conference matrix (Xiao, Lin and Bai, 2012): 2m + 3 runs for an odd number of
  factors m, and 25 runs for 12 factors. process-improve builds those, and daewr differs on the
  20 checks of these later sources: run counts, and the measures that follow from them.
- **Outside what DefScreen offers.** Designs for 3 factors, and for more than 12.

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

pydoe (Python)
~~~~~~~~~~~~~~

pydoe 1.7.0 built every kind of design in the table. No difference.

- **Outside what pydoe offers.** A blocked factorial whose block generators the source does not
  give, a blocked fraction, and a spherical axial distance. Davies' (1956) orthogonal design:
  pydoe's ``alpha="orthogonal"`` is the distance for orthogonal blocking, as in rsm.
- **Slow searches.** ``fracfact_opt`` searches exhaustively, so a fraction of 10 or 11 factors
  takes minutes: the search for a 2\ :sup:`10-4` fraction took 8 minutes. The 14 checks that
  did not finish in 3 minutes are not counted.

.. _validation-dexpy:

dexpy (Python)
~~~~~~~~~~~~~~

dexpy 0.12, from Stat-Ease, built two-level factorials, Box-Behnken designs and central
composite designs on a full cube.

- **What it builds agrees.** No difference.
- **Outside what dexpy offers.** dexpy chooses a fraction from its run count: it takes no
  generators, no resolution and no blocks, so most of the catalog's fractions cannot be asked of
  it. It has no central composite design on a fractional cube, and no factor with more than two
  levels.

.. _validation-oapackage:

OApackage (Python and C++)
~~~~~~~~~~~~~~~~~~~~~~~~~~

OApackage 2.7.21 built the definitive screening designs, from the conference designs it
enumerates, and the orthogonal arrays, by extending an array one column at a time.

- **Definitive screening designs.** No difference, for an even or an odd number of factors.
- **Orthogonal arrays.** No difference. The 12-run array is Taguchi's L12 in another layout,
  as DoE.base's arrays are (:ref:`validation-frf2`).
- **Not counted.** The search for a 24-factor design did not end in 3 minutes.

.. _validation-dsd-package:

definitive-screening-design (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

definitive-screening-design 0.5.1 built every definitive screening design asked of it. No
difference.

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
     - Center line, standard deviation and limits of the individuals chart
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
The last row is checked through the reference catalog.

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
     - Six differences from the printed sources, each reported with a fix. pydoe 1.7.0, the
       release compared here, has all six fixes.
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
     - A minimum-aberration search that stops where published designs exist, in 64 and 128
       runs. The same gap in 32 runs, and a second one, no way to give the center runs of each
       block of a central composite design, are closed.
     - A12 in `issue 788 <https://github.com/kgdunn/process-improve/issues/788>`__
   * - process-improve
     - 1,555 printed numbers that need a function the library does not have yet, 275 that
       differ for a cause not known yet, and 1 that differs for a bug or limit of the library.
     - `The catalog's tracking issue <https://github.com/kgdunn/process-improve/issues/804>`__

.. _validation-limits:

What this page does not show
----------------------------

- **Optimal designs are not cross-checked with other software.** A D- or I-optimal search
  starts from random designs, so two packages need not find the same design. The catalog
  checks the criteria of the designs a source prints.
- **OMARS designs have no second implementation here.** They are checked against the papers
  that define them only.
- **Equal numbers, and equal arrays.** A design passes when it gives the printed numbers. An
  orthogonal array that another package lays out differently passes when a change of the order
  of its runs and columns and of its level labels gives the printed array.
- **Agreement with pydoe on the basic arrays is by construction**, as said above.
- **A result is one run.** The versions and dates are those named in each table. A later
  release of a package can change its row.
- **The references have limits too.** The SIMCA-P values are from version 14.1, recorded in
  2020, and no later version was tried. SensMixed 2.1-0 was archived from CRAN in 2020 and runs
  only on R 3.4.4.

.. _validation-reproduce:

Repeating a comparison
----------------------

The catalog:

.. code-block:: bash

   uv run pytest tests/test_reference_catalogue.py --no-cov

.. code-block:: python

   from process_improve.experiments.reference_catalogue import reproduce_reference_catalog

   numbers = reproduce_reference_catalog()  # one row per printed number; several minutes
   numbers.groupby(["checked", "agrees", "cause"], dropna=False).size()

The designs of another package, here the R packages (``scripts/design_crosscheck/README.md``
lists what each package needs):

.. code-block:: bash

   DOE_LIB=R DOE_OUT=results uv run python scripts/design_crosscheck/compare.py all
   DOE_OUT=results uv run python scripts/design_crosscheck/tabulate.py

The first command writes one row per check; the second writes the table of this page and the
lists of differences it links to.
