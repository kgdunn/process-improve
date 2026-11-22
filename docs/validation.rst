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

`Reproduced from the books <https://factori.al/catalog/reproduced/>`_ shows these counts as they
stand today, for the whole catalogue and for each source, with an example of every cause.
:ref:`design-catalog` describes the records, and
:func:`~process_improve.experiments.reference_catalogue.reproduce_reference_catalogue` returns
one row per number.

.. _validation-designs:

Against other software: how a design is built
---------------------------------------------

A printed number says what a design should be. Another package says what a second,
independent construction gives. Of the catalogue's numbers, @@N_DESIGN_CHECKS@@ depend on how a design is
built: run counts, generators, defining relations, alias chains, resolutions, word length
patterns, axial distances and block arrangements.

``scripts/design_crosscheck/compare.py`` checks each of them twice with the catalogue's own
checker: on the design process-improve builds, and on the design the other package builds.
Only the construction changes, so a difference is a difference in the design.

@@DESIGN_TABLE@@

How to read the table:

- **Checks** are the numbers of the kinds of design the package builds. **Agree** means both
  packages give the number the source prints. **Differ** means they give different answers.
  **Not built** means the package has no design for the request, or declines it.
- The numbers that neither package gives as printed are left out: the catalogue records their
  cause, most often a misprint in the source.
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

@@PACKAGE_SECTIONS@@

.. _validation-analysis:

Against other software: analysis
--------------------------------

Outside design construction, the tests hold values that other software computed, on data sets
kept with the tests. A change that moves one of them fails the test.

.. list-table::
   :header-rows: 1
   :widths: 24 28 30 18

   * - Method
     - Reference
     - What is compared
     - Test
   * - Split-plot analysis by REML
     - lme4 1.1-36, lmerTest 3.1-3 and pbkrtest 0.5-3 (R 4.4.2)
     - Variance components, estimates, standard errors, Satterthwaite and Kenward-Roger
       degrees of freedom, Type III F-tests
     - ``test_experiments_split_plot.py``
   * - Mixed assessor model of a sensory panel
     - SensMixed 2.1-0 (R 3.4.4)
     - Every Type I test of every attribute of the TVbo panel: F, sums of squares, degrees
       of freedom and p
     - ``test_sensory_mam_reml.py``
   * - PCA with missing values (NIPALS)
     - mixOmics 6.34.0
     - Scores, loadings and eigenvalues
     - ``test_pca_nipals_mixomics_crosscheck.py``
   * - Partial robust M-regression
     - chemometrics 1.4.4
     - Coefficients, intercept and observation weights
     - ``test_multivariate_prm.py``
   * - PCA and PLS
     - SIMCA-P 14.1
     - Scores, loadings, R2, the squared prediction error (DModX) and Hotelling's T2, on the
       LDPE data among others
     - ``test_multivariate.py``
   * - PCA and PLS
     - scikit-learn
     - The same fit as ``PCA`` and ``PLSRegression``
     - ``test_sklearn_compat.py``
   * - Batch PCA
     - A MATLAB latent-variable toolkit
     - Scores, loadings and R2 on simulated styrene-butadiene rubber batches
     - ``batch/test_sbr_reference.py``
   * - Control charts
     - qcc (R)
     - Centre line, standard deviation and limits of the individuals chart
     - ``test_monitoring.py``
   * - Least squares, t-tests and analysis of variance
     - R (``lm``, ``t.test``), statsmodels, SciPy
     - Coefficients, standard errors, p-values, sums of squares, adjusted p-values
     - ``test_regression.py``, ``test_univariate.py``, ``test_analysis.py``
   * - Generalized linear models, Gaussian processes, REML and least squares means
     - JMP, Minitab, SAS and Design-Expert output, as textbooks print it
     - The tables of Montgomery (2013) and Goos and Jones (2011)
     - The reference catalogue

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
@@FINDINGS_ROWS@@

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
- **The references have limits too.** SIMCA-P 14.1 and SensMixed 2.1-0 are the versions the
  fixtures were made with; later versions were not tried.

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
