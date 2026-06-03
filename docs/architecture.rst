Architecture overview
=====================

This page is the map of the codebase: how ``process-improve`` is laid out, the
conventions every subpackage follows, and the two cross-cutting systems (the
estimator stack and the MCP tool layer) that most changes touch. Read it before
your first contribution; the per-topic policy pages under
:doc:`development/index` go deeper.

Package layout
--------------

.. code-block:: text

   process_improve/
       multivariate/    # PCA, PLS, TPLS, and multi-block (MBPCA / MBPLS)
       experiments/     # designed experiments: designs, analysis, optimisation
       monitoring/      # control charts (Shewhart / CUSUM / EWMA-style)
       batch/           # batch data alignment (DTW), features, preprocessing
       regression/      # robust regression (repeated median, Theil-Sen)
       bivariate/       # elbow / peak detection, area-under-curve
       univariate/      # robust summary statistics, outlier detection
       visualization/   # shared plotting themes and helpers
       simulation/      # process simulators
       datasets/        # sample datasets used by examples and tests
       tool_spec.py     # the MCP @tool_spec decorator + global tool registry
       config.py        # runtime settings (caps, limits)
       _linalg.py, _random.py, _extras.py   # small shared utilities

Each domain subpackage exposes its public API through a thin re-export module
(``methods.py`` for multivariate, package ``__init__`` elsewhere) so the import
path callers use is stable even when the implementation files move.

Conventions every subpackage follows
-------------------------------------

- **sklearn-compatible estimators.** Estimators inherit ``BaseEstimator`` plus
  the relevant mixin (``TransformerMixin`` / ``RegressorMixin``). They do *not*
  inherit a concrete sklearn estimator - the mixins give ``get_params`` /
  ``set_params`` / ``clone`` / Pipeline support without coupling to sklearn's
  private attribute layout (ENG-07). ``fit()`` returns ``self``; fitted
  attributes use the trailing-underscore convention (``scores_``, ``spe_``,
  ``hotellings_t2_``) and are set only in ``fit()``, never in ``__init__``.
- **Optional dependencies live in extras.** Plotting (plotly / ridgeplot),
  the experiments designed-experiment generators (pyDOE3 / pyoptex), the batch
  and MCP layers are installed via ``[plotting]`` / ``[expt]`` / ``[batch]`` /
  ``[mcp]`` extras (ENG-13). Modules import them through a ``_MissingExtra``
  stand-in so a missing optional dependency only fails when the feature is
  actually used, not at import time.
- **Diagnostic logging.** Modules that do real work define
  ``logger = logging.getLogger(__name__)`` and emit ``debug`` records at major
  algorithm steps; nothing configures handlers. See :doc:`development/logging`.
- **Error handling.** Tool wrappers narrow their ``except`` to a canonical set
  so unexpected errors propagate to the server and get redacted. See
  :doc:`development/error_handling`.
- **Reproducibility.** Randomised paths take an explicit ``random_state``. See
  :doc:`development/reproducibility`.

The multivariate estimator stack
---------------------------------

The latent-variable estimators are split into single-responsibility modules
under ``multivariate/`` and aggregated by ``methods.py`` (the stable public
import path; ``_pca_pls.py`` remains as a backward-compatibility shim):

.. code-block:: text

   _common.py        # DataMatrix alias, epsqrt, _nz, SpecificationWarning, _model_method
   _preprocessing.py # MCUVScaler, center, scale
   _nipals.py        # NIPALS / least-squares kernels (missing-data aware)
   _limits.py        # Hotelling's T2 / SPE / score limits, ellipse geometry
   _diagnostics.py   # VIP, squared cosine, contributions, RV coefficients
   _base.py          # _LatentVariableModel base + mixins (see below)
   _pca.py  _pls.py  _tpls.py  _mbpls.py  _mbpca.py   # the estimators
   _resampling.py    # jackknife / bootstrap resampling
   plots.py          # score / loading / SPE / T2 / coefficient plots + Plot accessor

Two base-class ideas tie the estimators together (``_base.py``):

- **``_LatentVariableModel``** owns the scaffolding PCA and PLS share: the
  convenience methods (``score_plot``, ``vip``, ``spe_limit``, ...) that forward
  to the standalone functions, ``ellipse_coordinates``, and the attribute-rename
  ``__getattr__`` (driven by a per-class ``_ATTRIBUTE_RENAMES`` map). The
  convenience methods are *real methods* built by the ``_model_method`` factory,
  so ``help`` / ``inspect.signature`` report the underlying function and the
  fitted model pickles and subclasses cleanly (ENG-05, ENG-17). MBPLS / MBPCA
  share only ``_HotellingsT2LimitMixin``.
- **Ndarray-backed fitted attributes.** Hot-path attributes (``scores_``,
  ``loadings_``, ``spe_``, ...) are stored as private numpy ndarrays; the public
  ``pd.DataFrame`` is a lazily-built, cached view via the ``_LazyFrame``
  descriptor. Internal math reads the ndarray (no per-call ``.values``
  conversion); the cache is excluded from pickling (ENG-18).

A typical fit therefore: validates input, runs the algorithm-specific
``_fit_*`` (SVD / NIPALS / TSR for PCA; NIPALS for PLS; hierarchical NIPALS for
the multi-block models), stores the ndarrays + index/column metadata, and
computes limits and R-squared bookkeeping. The numerical kernels live in
``_nipals.py`` and are shared, so a fix there benefits every estimator.

The MCP tool layer
------------------

Agent-callable tools are declared with the ``@tool_spec`` decorator
(``process_improve/tool_spec.py``). Each tool pairs a pydantic
``BaseModel`` input contract (``ConfigDict(extra="forbid")``) with a wrapper
function; the decorator registers the function in a global ``_TOOL_REGISTRY``
and attaches the JSON-schema spec. ``get_tool_specs()`` returns the specs in
registry (decorator-execution) order; ``discover_tools()`` imports each
subpackage's ``tools`` module so the decorators run.

In ``experiments/`` the tools and analyses are split one-per-module
(``_tools/<tool>.py``, ``_analyses/<analysis>.py``) with ``tools.py`` acting as
the ordered aggregator (ENG-02). To add a tool, see :doc:`development/tool_authoring`.

Where to make a change
----------------------

- A numerical fix to a latent-variable algorithm: the shared kernels in
  ``multivariate/_nipals.py`` / ``_limits.py`` / ``_common.py``, or the
  estimator's ``_fit_*`` method.
- A new estimator convenience method shared by PCA and PLS:
  ``multivariate/_base.py``.
- A new agent-callable tool: a new module under the subpackage's ``_tools/``
  (or ``tools.py``) - see :doc:`development/tool_authoring`.
- A new designed-experiment analysis: a module under
  ``experiments/_analyses/`` dispatched from ``experiments/analysis.py``.
