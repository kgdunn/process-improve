Logging
=======

.. note::

   This document describes the logging policy for ``process-improve``.
   Tracks `ENG-12 <https://github.com/kgdunn/process-improve/issues/294>`_.

Why logging?
------------

Diagnosing a surprising result (a PCA that loaded onto the wrong component, a
PLS fit that did not converge, a designed-experiment analysis that silently
dropped a term) should not require editing the library and re-running with a
debugger. The package emits diagnostic logging from the modules that do real
work so that you can turn on a verbose trace and see what the algorithm did.

Conventions
-----------

- **Module-level logger.** Every module that does real work defines
  ``logger = logging.getLogger(__name__)`` at module scope. The logger name is
  the dotted module path (e.g. ``process_improve.multivariate._pls``), so you
  can raise or lower verbosity per subpackage.
- **``logger.debug(...)`` per major step.** Long-running, iterative algorithms
  emit one ``debug`` record per major step. For example, the NIPALS fitters
  (PCA / PLS / TPLS / MBPLS / MBPCA) log the iteration count at which each
  component converged, and the batch DTW alignment logs its per-iteration
  weight-change norm.
- **``logger.info(...)`` / ``logger.warning(...)`` when a guard rail fires.** A
  rejected input, a clamped parameter, or a diagnostic that could not be
  computed is logged so the failure is not silent.
- The library **never** configures logging handlers or levels itself (no
  ``basicConfig`` at import time). It only emits records; the application
  chooses whether and how to surface them. By default Python's logging is
  silent below ``WARNING``, so importing the package adds no output.

Enabling verbose logging
-------------------------

Turn on debug logging for the whole package from your script or notebook:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.DEBUG)

   import numpy as np
   import pandas as pd
   from process_improve.multivariate.methods import PCA, MCUVScaler

   X = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 8)))
   X.iloc[0, 0] = np.nan  # force the iterative NIPALS path
   PCA(n_components=3, algorithm="nipals").fit(MCUVScaler().fit_transform(X))
   # DEBUG process_improve.multivariate._pca: PCA NIPALS: component 1 converged in 12 iterations (md_tol=1.49012e-08)
   # ...

To keep third-party libraries quiet and only see this package's records, scope
the level to the ``process_improve`` logger:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.WARNING)
   logging.getLogger("process_improve").setLevel(logging.DEBUG)

You can narrow further to a single subpackage, e.g.
``logging.getLogger("process_improve.multivariate").setLevel(logging.DEBUG)``.
