Scaling and memory
==================

.. note::

   This page documents the in-memory assumption of the estimators and the
   practical data-size limits. Tracks
   `ENG-19 <https://github.com/kgdunn/process-improve/issues/301>`_.

The in-memory assumption
------------------------

The estimators in ``process-improve`` (PCA, PLS, TPLS, MBPCA, MBPLS, and the
batch feature pipeline) are **in-memory**: ``fit`` expects the (scaled) data
matrix to fit in RAM, and the iterative NIPALS / SVD paths hold a small number
of working copies on top of it. There is currently **no out-of-core or
streaming code path** - the whole matrix is materialised as a dense
``numpy``/``pandas`` array.

This is the right trade-off for the data sizes these methods are normally
applied to (process and lab data: thousands to low millions of rows, tens to a
few hundred columns), where keeping the pandas row/column labels and the full
diagnostic suite (SPE, Hotelling's T2, contributions) is worth far more than
streaming.

Estimating the memory you need
------------------------------

A dense ``float64`` matrix needs roughly:

.. code-block:: text

   bytes ~= n_rows x n_cols x 8

So a ``10,000,000 x 200`` matrix is about **16 GB** just for the data - before
any working copies. As a rule of thumb, budget **2-4x** the raw matrix size for
a fit:

- ``MCUVScaler`` / ``center`` / ``scale`` return a scaled copy.
- ``PCA.fit`` copies ``X`` once for the working array; NIPALS deflation works in
  place on that copy.
- ``PLS.fit`` similarly holds scaled ``X`` and ``Y`` plus deflation copies.

Example: a ``1,000,000 x 100`` matrix is ~0.8 GB raw, so expect ~2-3 GB
resident during ``fit`` - comfortable on a workstation, tight on a laptop.

When this is *not* the right tool
---------------------------------

If your matrix does not fit in RAM (with the 2-4x headroom above), this package
is not currently the right fit for a single ``fit`` call. Options today:

- **Down-sample or aggregate** to a representative subset for model building,
  then ``transform`` / ``predict`` the full data in chunks (``transform`` and
  ``predict`` are far cheaper than ``fit`` and can be applied batch-by-batch).
- **Reduce dtype** upstream (e.g. ``float32``) if the precision budget allows;
  this halves the footprint. Note the estimators promote to ``float64``
  internally, so this mainly helps the input/transform side.
- **Use an out-of-core PCA** from another library (e.g.
  ``sklearn.decomposition.IncrementalPCA`` with ``partial_fit`` over chunks, or
  a ``dask``-backed SVD) for the dimensionality-reduction step, then bring the
  reduced scores back into this package for the diagnostics.

Roadmap
-------

A first-class out-of-core path for PCA (an incremental / chunked fitter, likely
behind a ``[bigdata]`` optional-dependency extra) is tracked in
`ENG-19 <https://github.com/kgdunn/process-improve/issues/301>`_. It is demand-driven:
if you have a concrete larger-than-RAM use case, please comment on that issue.
