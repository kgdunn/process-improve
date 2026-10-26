.. _design-catalog:

The Design Catalog
==================

The design catalog at `factori.al/catalog <https://factori.al/catalog/>`_ lists two kinds of
design. The published ones are the experiments and designs of the reference catalogue
(:mod:`process_improve.experiments.reference_catalogue`): textbooks, handbooks and papers, each
entry with its source, the page it is printed on, and every number the source prints for it. The
generated ones are the designs :func:`~process_improve.experiments.generate_design` makes, one for
each family and size in :data:`~process_improve.experiments.design_catalog.GENERATED`.

A reader filters the designs by family, application, size, the model they estimate, what they
withstand, their structure, their source and the year it appeared; sees each one on a map of runs
against factors; and opens a design in Factorial, the browser app, with a click. A design's page
links to the place its source prints it: for the book and the NIST e-Handbook, the section itself.
The same records are available in Python and, through the ``search_design_catalog`` tool, to an
agent.

Searching it in Python
----------------------

:func:`~process_improve.experiments.design_catalog.search_catalog` takes the filters the page
offers and returns the designs that pass every one, the fewest runs first:

.. code-block:: python

   from process_improve.experiments.design_catalog import search_catalog

   # Six factors, at most 17 runs, and a response surface in any three that turn out to matter.
   for design in search_catalog(min_factors=6, max_factors=6, max_runs=17, min_quadratic_projection=3):
       print(design["title"], design["n_runs"], design["source_short"], design["url"])

``topics=["food-beverages"]`` keeps the published designs of one application, as the page's
Application filter does; ``text="food"`` finds them by a word of its name.

The first search builds the catalog, which takes some seconds; later searches reuse it.
:func:`~process_improve.experiments.design_catalog.catalog_records` returns every record whole,
with its coded runs, factors and printed results.

What a record holds
-------------------

Each record describes one design by facets read off its coded runs, and, for a published design,
by what its source says of it.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - What it holds
   * - ``family``
     - The family, such as ``"dsd"`` or ``"fractional_factorial"``. A published design's family is
       read from the source's own words for it by
       :func:`~process_improve.experiments.design_catalog.normalise_family`.
   * - ``collection``
     - ``"published"`` or ``"generated"``.
   * - ``n_runs``, ``n_factors``
     - The size of the design.
   * - ``n_numeric``, ``n_categorical``, ``n_mixture``
     - How many factors of each kind.
   * - ``levels``, ``n_centre``
     - Each factor's number of distinct settings, and the runs with every numeric factor at its centre.
   * - ``models``
     - Whether the design estimates main effects, two-factor interactions and the full quadratic, each
       in full (see :func:`~process_improve.experiments.design_measures.estimable_models`).
   * - ``survives_run_loss``, ``critical_runs``, ``worst_d_ratio``
     - What losing one run costs the design, for the largest model it estimates (see below).
   * - ``quadratic_projection``, ``projectivity``
     - What the design's projections onto fewer factors hold (see below).
   * - ``design_class``
     - Records of one class hold one design, printed or made in more than one place.
   * - ``source``, ``year``, ``added``, ``where``
     - The published source, the year it appeared, the date the entry entered the catalogue, and
       the pages, tables and figures that print it. A figure is named by its caption or alt text
       (:func:`~process_improve.experiments.design_catalog.figure_name`), not by its file.
   * - ``where_url``
     - The address of the place a web source prints the design: a page of the book, at the
       section's anchor, or a page of the NIST e-Handbook
       (:func:`~process_improve.experiments.design_catalog.where_url`). ``None`` for a printed
       book, and for a generated design.
   * - ``topics``
     - What the design is applied to, such as ``"food-beverages"`` or ``"semiconductors"``: its
       subject-matter topics in the reference catalogue
       (:func:`~process_improve.experiments.design_catalog.catalog_topics`), ``"general"`` when the
       source states none. A generated design has none, so filtering by an application keeps the
       published designs only.
   * - ``checks``
     - The numbers the source prints for the design: how many are recomputed by the library, how
       many agree, and how many are known to differ, with the reason recorded.

What a design withstands
------------------------

Three measures describe a design before it is run: what losing a run costs, what a projection onto
fewer factors still estimates, and whether the design is printed elsewhere under other labels. They
are in :mod:`process_improve.experiments.design_measures`.

**Losing a run.** Let :math:`X` be the design's model matrix, with *p* columns, and
:math:`h_i = x_i^\top (X^\top X)^{-1} x_i` the leverage of run *i*. Deleting the run scales the
determinant of the information matrix by one less its leverage:

.. math::

   \det\left(X_{(i)}^\top X_{(i)}\right) = \det\left(X^\top X\right)\,(1 - h_i).

So the design still estimates the model after losing run *i* exactly when :math:`h_i < 1`, and keeps
:math:`(1 - h_i)^{1/p}` of its D-efficiency. One hat matrix answers this for every run, without
refitting. A saturated design, with as many runs as coefficients, has every :math:`h_i = 1`: the
12-run Plackett-Burman design for 11 factors cannot lose any of its runs, while a two-level factorial
with a center run can lose any one of them.

.. code-block:: python

   import numpy as np
   from process_improve.experiments.design_measures import run_loss

   square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0]], dtype=float)
   found = run_loss(square, ["numeric", "numeric"], "interactions")
   print(found["survives"], found["leverage"])  # True [0.95, 0.95, 0.95, 0.95, 0.2]

**Projections.** Projecting a design onto *p* of its factors keeps every run and drops the other
factors. :func:`~process_improve.experiments.design_measures.quadratic_projection` is the largest
*p* such that every projection onto *p* factors estimates the full quadratic in them; for a
six-factor definitive screening design it is 3, so if only three factors matter, their response
surface can be fitted from the runs already made. For a two-level design,
:func:`~process_improve.experiments.design_measures.projectivity` is the largest *p* such that every
projection onto *p* factors holds all :math:`2^p` combinations of their levels; it is one less than
the resolution of a regular fraction, and 3 for the 12-run Plackett-Burman design.

**The same design elsewhere.** Two designs are one design when a reordering of the runs, a
reordering of the factors and a swap of some factors' two levels make their runs the same.
:func:`~process_improve.experiments.design_measures.fingerprint` hashes what does not change under
those three; :func:`~process_improve.experiments.design_measures.same_design` confirms a match by
finding the relabelling itself. The catalog groups its designs so: the 12-run Plackett-Burman design
it makes is also the Taguchi L12 array printed by Schmidt and Launsby and the design of Table 3.18 of
the NIST e-Handbook.

Comparing designs
-----------------

Designs with the same number of factors can be set side by side on the measures of
:func:`~process_improve.experiments.compare.compare_designs`, for one model: the feasibility rows,
the averages :math:`A/p` and :math:`I`, the worst cases :math:`E` and :math:`G`, :math:`D`, the
largest correlation between second-order terms, and the power for each kind of term. Each design is
scored at its own size, so its values do not depend on the other designs it is set beside.
:func:`~process_improve.experiments.design_catalog.catalog_comparisons` scores every design of
numeric factors for each of the catalog's three models, with its fraction-of-design-space curve
from :func:`~process_improve.experiments.design_measures.fds_curve`: the prediction variance that
each fraction of the cube does not exceed.

.. code-block:: python

   from process_improve.experiments.design_catalog import catalog_comparisons, catalog_records

   chosen = [r for r in catalog_records("generated") if r["id"] in {"lib-dsd-k6", "lib-omars-k6"}]
   for name, scored in catalog_comparisons(chosen).items():
       print(name, scored["main_effects"]["i"], scored["main_effects"]["fds"][10])

On the catalog's page, a design's page also projects its runs onto any two or three of its factors
and says whether that projection estimates the full quadratic, draws its correlation map, and links
to the families it can be augmented to.

The errata
----------

:func:`~process_improve.experiments.design_catalog.catalog_errata` lists every printed number that
the library recomputes and does not reproduce, with the value as printed, where it is printed, the
source's words around it and the reason the catalogue records: the source's rounding, a misprint, or
a method that differs from the library's. Every other recomputed number agrees with its source, as
the test suite requires. The catalog's page lists them by source and by words, and
``catalog/feed.xml`` (:func:`~process_improve.experiments.design_catalog.catalog_feed`) is an Atom
feed with an entry for each source as it enters the catalogue.

How closely each case reproduces
--------------------------------

:func:`~process_improve.experiments.design_catalog.catalog_reproduction` puts every case of the
reference catalogue in one of four tiers, by the largest relative difference between a number its
source prints and the library's value (:data:`~process_improve.experiments.design_catalog.REPRODUCTION_TIERS`):
fully reproduced, when every recomputed number agrees to the source's last printed digit; within 1%;
within 5%; and unable to reproduce, when a number differs by more than 5% or a result that is not a
number, such as an alias chain, differs. A case is placed by its worst number.

The catalogue records the cause of every number that differs, one of
:data:`~process_improve.experiments.reference_catalogue.DISAGREEMENT_CAUSES`: an error in the source,
a quiz option its answer key marks incorrect, the source's rounding, another method or convention,
the library, or not known yet. In the first two the library's value is the right one
(:data:`~process_improve.experiments.design_catalog.LIBRARY_RIGHT`), so they do not count against a
case. A printed number that is not recomputed records its cause too, one of
:data:`~process_improve.experiments.reference_catalogue.UNCHECKED_CAUSES`, which tells a number the
library has no function for yet from one with nothing to recompute:

.. code-block:: python

   from process_improve.experiments.design_catalog import catalog_reproduction

   for case in catalog_reproduction("lenth-1989"):
       print(case["tier"], case["title"][:40], [n["cause"] for n in case["differs"]])

The catalog's page `Reproduced from the books <https://factori.al/catalog/reproduced/>`_ draws the
same classification for every source: a table of the tiers, by case and by number; the causes of the
differences and of the numbers not recomputed, for all the sources together; a table by source, with
the sources grouped by their form (:func:`~process_improve.experiments.design_catalog.source_form`:
books, journal papers, and handbooks, encyclopedias and courses) and each group's totals; the spread
of the differences; a list of the cases in each tier; and a list of what the library lacks, the
missing methods that would recompute the most printed numbers first. The tiers describe how far the
library reproduces a source, not the quality of the source, and the errors found in the sources are
counted for all of them together only.

The page opens with these in words, and keeps the tables behind *Show the numbers*. Each source also
has a page of its own, under `Books <https://factori.al/catalog/books/>`_: how closely the library
gives its answers, its cases in page order, the misprints found in it with the printed value beside
the value its own data give, and the printed numbers the library has no function for yet. Each
application has a page too, under `Fields <https://factori.al/catalog/fields/>`_: the typical size
of its published experiments and the kinds of design they use, drawn on a chart of factors and runs.
Factorial's planning panel shows the same chart beside the plan, for the field the reader chooses.

Opening a design, and limits on its settings
--------------------------------------------

A design whose type the browser app makes opens with the runs the app makes, which the build checks
are the source's runs. Any other design with a complete table of runs, of 2 to 12 numeric or
two-level factors and without blocks or whole plots, opens as *runs as given*: the source's runs, in
its order and its factors' own units, with its results. Factorial analyses them as it analyses a
design it made, with the largest model the runs estimate in full.

A limit on the settings, typed beside the list in coded units (``A + B <= 1``, with ``A`` a design's
first factor and ``B`` its second), keeps the designs whose every run meets it. A design's page marks
the runs outside the limit, and opens the design in Factorial with the limit written in the design's
own factors and units. There, the run sheet's notes name the runs outside the limit, and the
D-optimal and I-optimal designs make runs inside it.

Proposing a design or a case study
----------------------------------

A design printed in a book, a paper or a handbook can be proposed for the catalogue with the
`catalog entry form <https://github.com/kgdunn/process-improve/issues/new?template=catalog_entry.yml>`_.
It asks for the source, where the runs are printed, the factors, the runs and the numbers the source
prints for them. The entry enters the reference catalogue once the library recomputes those numbers,
each agreeing with the source or recorded with the reason it does not.

A designed experiment of your own can be shared with the
`case study form <https://github.com/kgdunn/process-improve/issues/new?template=case_study.yml>`_.
It asks for the least an analysis needs: the factors and the responses, the runs in the order they
were done, how the runs were grouped (in a random order, in blocks, or as a split plot), and the
licence the data are shared under (CC BY 4.0 or CC0 1.0). The numbers your own analysis found are
optional; each one you give becomes a check. The entry cites the issue as its source, so the catalog
links back to it.
