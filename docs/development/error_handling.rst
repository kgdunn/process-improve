Error-Handling Style Guide
==========================

.. note::

   This document is the canonical policy for how errors and warnings
   are surfaced in ``process-improve``. New code is expected to follow
   it; review feedback will reference it directly. Tracks
   `ENG-11 <https://github.com/kgdunn/process-improve/issues/293>`_.

Why one style?
--------------

A reader landing in a new module should be able to predict, without
running the code, what a failure will look like. Today, the codebase
mixes ``raise ValueError``, ``assert``, ``return None``,
``return {"error": ...}``, ``except Exception: pass``, silent
``NaN``, and bare ``warnings.warn`` calls. Each of those is fine in
isolation; the cost is the *spread*: a contributor patching one site
cannot tell whether the surrounding modules will react the same way,
and the audit work to verify a fix is multiplied.

This guide pins one pattern per situation. The patterns are not
novel - they follow the conventions used by NumPy, SciPy, and
scikit-learn - so a contributor familiar with the scientific Python
stack should not need to memorise anything new.

Decision tree
-------------

Pick the case that best matches the failure you are handling and
follow the linked section.

1. The failure is a violation of a contract the *caller* signed
   (wrong argument type, out-of-range value, mismatched shape).
   :ref:`raise <eng11_caller_error>`.
2. The failure is a violation of an invariant the *library*
   guarantees internally (a "this can't happen" branch).
   :ref:`assert or RuntimeError <eng11_internal_error>`.
3. The failure is a known degeneracy of the algorithm
   (singular design, non-convergence, missing data when not
   allowed). :ref:`dedicated exception class <eng11_algorithm_error>`.
4. The failure happens at the MCP boundary (the host's untrusted
   transport surfaced bad input or a tool died).
   :ref:`ToolSafetyError subclass <eng11_mcp_error>`.
5. The condition is recoverable but the caller probably wants to
   know about it. :ref:`warnings.warn <eng11_warning>`.
6. **None of the above.** Default to (1) and add a unit test
   asserting the exception text.

.. _eng11_caller_error:

1. User-input violations -> ``ValueError`` / ``TypeError``
----------------------------------------------------------

This is the most common case. The caller passed something the
function cannot work with: a wrong type, an out-of-range scalar, an
empty dataframe, a column name that does not exist.

Pattern::

    if X.shape[1] != self.n_features_in_:
        raise ValueError(
            f"Prediction data must have {self.n_features_in_} columns, "
            f"got {X.shape[1]}."
        )

Use ``ValueError`` for wrong *value*; use ``TypeError`` for wrong
*type*. The message always names:

- the offending parameter,
- the observed value or type,
- the expected value or constraint.

**Do not use** ``assert`` for these checks. Asserts are stripped
by ``python -O``; the check silently disappears in optimised
deployments, and the bug surfaces miles downstream as a confusing
NaN or shape mismatch.

**Do not return** ``None``, ``False``, an empty DataFrame, or a
"sentinel" value. The next caller cannot tell a successful empty
result from a swallowed error.

**Do not catch** ``Exception`` and re-raise; let the exception
propagate.

.. _eng11_internal_error:

2. Internal invariants -> ``assert`` (rare) or ``RuntimeError``
---------------------------------------------------------------

A genuinely impossible branch ("at this point ``A`` must be > 0
because the constructor checked it") may be guarded with
``assert``, but **must** carry a comment naming the invariant
that justifies the assertion.

Pattern::

    # Invariant: _fit_one_component sets self._loadings_np for a >= 0.
    assert self._loadings_np is not None, "internal: loadings missing after fit"  # noqa: S101

If the branch is *not* genuinely impossible (the input could
trigger it), use ``RuntimeError`` and let it propagate:

::

    raise RuntimeError(
        "internal: NIPALS reached max_iter without converging "
        "after fitting passed shape validation; this is a bug."
    )

Asserts on user input are forbidden (see SEC-08 / SEC-17 in
``SECURITY_AUDIT.md``).

.. _eng11_algorithm_error:

3. Algorithm-level degeneracies -> dedicated exception class
------------------------------------------------------------

When a known algorithmic edge case fires (rank-deficient design,
non-convergence, singular matrix), raise a *dedicated* exception
class. This lets callers handle the specific case without catching
``Exception``.

Existing classes (do not invent more for the same condition):

- ``numpy.linalg.LinAlgError`` for singular / ill-conditioned
  matrices. ``process_improve._linalg.safe_inverse`` already
  raises it; mirror its message style.
- ``process_improve.tool_safety.ToolSafetyError`` and its
  subclasses for MCP-layer failures (see
  :ref:`eng11_mcp_error`).

When a new algorithmic case appears (e.g. "PLS NIPALS did not
converge"), add a class in the relevant module:

::

    class NotConvergedError(RuntimeError):
        """Iterative algorithm hit max_iter without satisfying tol."""

Then raise it from the algorithm. The caller decides whether to
catch or propagate.

**Never** silently substitute ``np.nan`` or ``1.0`` for an
undefined statistic without recording the substitution in the
fitted model's ``fitting_info_`` dict and warning the caller
(see :ref:`eng11_warning`).

.. _eng11_mcp_error:

4. MCP-boundary failures -> ``ToolSafetyError`` subclass
--------------------------------------------------------

The MCP server is the only place in the package where an exception
becomes an attacker-visible string. ``mcp_server._serialise_tool_error``
redacts unhandled exceptions, but per-tool ``except`` clauses can
short-circuit that redaction. Follow the existing pattern in
``process_improve/tool_safety.py``:

- ``ToolInputTooLargeError`` - request exceeded a size limit.
- ``ToolInputInvalidError`` - schema / structural violation.
- ``ToolTimeoutError`` - wall-clock budget exceeded.
- ``ToolMemoryExceededError`` - worker process died.

Inside a tool wrapper:

- Narrow the ``except`` to the specific exception types the
  algorithm can raise: ``(ValueError, TypeError, KeyError,
  numpy.linalg.LinAlgError)`` is the canonical set for the
  multivariate tools.
- For anything *outside* that set, let the exception propagate
  to ``_serialise_tool_error``, which logs the traceback
  server-side and returns a generic message to the caller.

Forbidden pattern (information disclosure):

::

    # DO NOT do this. str(e) carries pandas / numpy paths and
    # statsmodels internals.
    try:
        ...
    except Exception as e:
        return {"error": str(e)}

Replace with:

::

    try:
        ...
    except (ValueError, KeyError, LinAlgError) as e:
        return {"error": str(e)}
    # Anything else propagates to _serialise_tool_error.

The existing PCA/PLS tools already follow this; new wrappers
should mirror them.

.. _eng11_warning:

5. Soft warnings -> ``warnings.warn``
-------------------------------------

For conditions that are recoverable but worth surfacing (a
non-converged solution returned anyway, a constant column
dropped, a degenerate fold during cross-validation), use
``warnings.warn`` with the right category and ``stacklevel=2``:

::

    import warnings

    warnings.warn(
        "PLS NIPALS hit max_iter without converging; results may "
        "be unstable. Pass max_iter=... to extend.",
        category=RuntimeWarning,
        stacklevel=2,
    )

Category guide:

- ``RuntimeWarning`` - the result was returned but the user
  should know it is approximate.
- ``UserWarning`` - the caller's input was unusual but the
  function adapted.
- ``DeprecationWarning`` - the caller used a deprecated path
  (see :doc:`deprecation_policy`).

**Never** use ``print`` for diagnostics. ``print`` writes to
``stdout``, which is the MCP server's protocol channel; a stray
``print`` would corrupt the MCP framing.

**Never** use bare ``warnings.warn("text")`` without a category;
the default category is ``UserWarning`` and callers cannot
filter the warning by class.

Logging vs. warning
-------------------

Use ``logger.debug`` / ``logger.info`` to describe what an
algorithm is doing (convergence iteration count, intermediate
shapes). Use ``warnings.warn`` to tell the user something they
should act on. Logging is for the developer; warnings are for
the user.

Every module that does real work should declare a module-level
logger (`ENG-12 <https://github.com/kgdunn/process-improve/issues/294>`_
covers the rollout)::

    import logging
    logger = logging.getLogger(__name__)

How to verify
-------------

Mechanical checks contributors should run locally before
submitting a PR:

- Tests pass under ``python -O -m pytest`` (catches stripped
  ``assert``-as-validation).
- ``ruff check .`` is clean (the codebase's existing rules
  cover the bare-``except`` and ``print`` cases).
- New code references one of the patterns above in its
  docstring or commit message.

Cross-references
----------------

- :doc:`reproducibility` -- contract for RNG seeding.
- :doc:`deprecation_policy` -- how to phase out a function safely.
- `SEC-08 <https://github.com/kgdunn/process-improve/pull/243>`_,
  `SEC-09 <https://github.com/kgdunn/process-improve/pull/244>`_,
  `SEC-17 <https://github.com/kgdunn/process-improve/issues/266>`_,
  `SEC-18 <https://github.com/kgdunn/process-improve/issues/267>`_ --
  the issues this guide retrospectively documents.
