Deprecation Policy
==================

.. note::

   This document is the canonical policy for retiring public API in
   ``process-improve``. Tracks
   `ENG-22 <https://github.com/kgdunn/process-improve/issues/304>`_.

Scope
-----

This policy covers anything documented as public:

- Top-level imports (``process_improve.PCA``, ``process_improve.lm``).
- Public methods and attributes on documented classes
  (``PCA.scores_``, ``PLS.predict``, ``MCUVScaler.fit``).
- The public signatures of those methods (kwarg names, defaults,
  keyword-only-ness).
- The MCP tool surface declared via ``@tool_spec``: tool names,
  schema keys, and the meaning of each field.

Private API (anything prefixed with ``_``, anything under
``process_improve._internal`` or ``process_improve._linalg``,
and anything not re-exported from a package ``__init__.py``) may
change without notice.

The contract
------------

For any breaking change to a public surface, follow this schedule:

============ ===================== =====================================
Phase        At least one MINOR    What contributors do
             release each
============ ===================== =====================================
Announce     `X.Y.0`               Add the new API. Emit a
                                   ``DeprecationWarning`` from the
                                   old API. Document the rename in
                                   the docstring. Add a CHANGELOG
                                   entry.
Warn         `X.(Y+1).*` and on    Old API still works; the warning
                                   still fires. CHANGELOG carries
                                   "Deprecated since X.Y; will be
                                   removed in (X+1).0".
Remove       `(X+1).0`             Old API is deleted. CHANGELOG
                                   entry under "Removed".
============ ===================== =====================================

In words: one MINOR cycle of "you can keep using this, but here is the
new name" plus a second cycle of "we mean it" before removal at the
next MAJOR.

Mechanism
---------

A renamed attribute or function uses the helper
``warnings.warn`` from the standard library with the
``DeprecationWarning`` category and ``stacklevel=2`` (consistent
with :doc:`error_handling`):

::

    import warnings

    def old_name(*args, **kwargs):
        warnings.warn(
            "process_improve.X.old_name is deprecated since 1.23.0 "
            "and will be removed in 2.0; use X.new_name instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return new_name(*args, **kwargs)

Required fields in the message:

- The fully qualified deprecated name.
- The version that announced the deprecation
  (``deprecated since X.Y.0``).
- The version that will remove it (``will be removed in
  (X+1).0``).
- The intended replacement (or "no replacement; the
  functionality is no longer supported").

For renamed *attributes* on PCA / PLS / TPLS, the existing
``__getattr__`` migration helper is the right mechanism; it
should be updated to emit a ``DeprecationWarning`` rather than
raising immediately. The helper is a single place to enforce
the message format.

When a deprecation is unavoidable mid-cycle
-------------------------------------------

If a bug or security finding forces a behaviour change inside a
PATCH release:

1. The CHANGELOG entry under "Changed" explicitly names the
   behaviour change.
2. If the public signature changes, the next MINOR release
   includes a deprecation shim under this policy. PATCH
   releases never break a signature.

If the change is forced by a downstream library (a sklearn or
statsmodels deprecation cascading through the package), the
CHANGELOG cross-references the upstream change.

What is *not* a breaking change
-------------------------------

- Adding a new optional kwarg with a default that preserves the
  previous behaviour.
- Adding a new public function.
- Fixing a documented bug (the fix is the breaking change of
  the bug, not of the API).
- A purely internal refactor that does not affect any public
  surface.

Edge case: ``Bunch`` return types. The fitted-model ``predict``
methods return ``sklearn.utils.Bunch`` with named fields. Adding
a field is **not** a breaking change. Renaming or removing one
**is**.

Tracking and rollout
--------------------

- The ``__getattr__`` migration helpers on PCA / PLS already
  exist; they currently raise rather than warn. Migrating them
  to emit ``DeprecationWarning`` is in scope for the next
  Wave-7 PR (`ENG-22 <https://github.com/kgdunn/process-improve/issues/304>`_).
- A ``tools/check_deprecations.py`` script will be added to
  list everything currently deprecated and its scheduled
  removal version. The script is run in CI; a deprecation
  past its removal version fails the build, so we cannot
  forget to delete the shim.

Cross-references
----------------

- :doc:`error_handling` -- warnings categories, ``stacklevel``,
  the ``warnings.warn`` pattern.
- :doc:`reproducibility` -- ``random_state`` migrations follow
  this policy.
- `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_ -
  the format the project's CHANGELOG follows.
