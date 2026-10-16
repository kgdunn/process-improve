.. _design-catalog:

The Design Catalog
==================

The design catalog at `factori.al/catalog <https://factori.al/catalog/>`_ lists two kinds of
design. The published ones are the experiments and designs of the reference catalogue
(:mod:`process_improve.experiments.reference_catalogue`): textbooks, handbooks and papers, each
entry with its source, the page it is printed on, and every number the source prints for it. The
generated ones are the designs :func:`~process_improve.experiments.generate_design` makes, one for
each family and size in :data:`~process_improve.experiments.design_catalog.GENERATED`.

A reader filters the designs, sees each one on a map of runs against factors, and opens a design
in Factorial, the browser app, with a click. The same records are available in Python and, through
the ``search_design_catalog`` tool, to an agent.

Searching it in Python
----------------------

:func:`~process_improve.experiments.design_catalog.search_catalog` takes the filters the page
offers and returns the designs that pass every one, the fewest runs first:

.. code-block:: python

   from process_improve.experiments.design_catalog import search_catalog

   # Six factors, at most 17 runs, and a response surface in any three that turn out to matter.
   for design in search_catalog(min_factors=6, max_factors=6, max_runs=17, min_quadratic_projection=3):
       print(design["title"], design["n_runs"], design["source_short"], design["url"])

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
       the pages, tables and figures that print it.
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
