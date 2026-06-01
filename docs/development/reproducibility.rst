Reproducibility Contract (RNG Handling)
=======================================

.. note::

   This document is the canonical policy for randomness in
   ``process-improve``. Tracks
   `ENG-08 <https://github.com/kgdunn/process-improve/issues/290>`_.

Why one contract?
-----------------

Reproducibility of a fitted model, a bootstrap interval, or an
optimisation result is part of the package's value proposition. A
chemometrics or DoE user comparing two runs across a software
upgrade has to be able to distinguish "the algorithm changed" from
"the random seed changed". Today the codebase mixes:

- unseeded ``np.random.default_rng()`` calls inside production
  algorithms,
- hard-coded literal seeds (``np.random.default_rng(42)``,
  ``seed=0``) inside production algorithms,
- well-seeded paths that thread ``random_state`` through correctly,
- ``@tool_spec(rng={...})`` metadata that *describes* the contract
  but enforces nothing.

The result is that "this run is reproducible" is currently a vibe,
not a guarantee. This document pins the guarantee.

The contract
------------

1. **Every public function that touches an RNG MUST accept a
   ``random_state`` parameter.** The accepted types are
   ``int | numpy.random.Generator | None`` (matching the
   convention used by scikit-learn since 1.4).

   ::

       def bootstrap(
           self,
           X: pd.DataFrame,
           *,
           n_boot: int = 1000,
           random_state: int | np.random.Generator | None = None,
       ) -> ...:

2. **Resolve ``random_state`` once at function entry** using the
   helper :func:`process_improve._random.check_random_state`. Its
   resolution rules match sklearn's ``check_random_state`` but it
   returns a modern :class:`numpy.random.Generator`:

   - ``None`` -> a fresh, unseeded ``np.random.default_rng()``.
   - ``int`` -> ``np.random.default_rng(int)``.
   - ``Generator`` -> returned as-is.

   Use the resolved ``Generator`` for all draws inside the
   function; never call ``np.random.*`` directly.

3. **Hard-coded literal seeds in production code are forbidden.**
   If a function currently does

   ::

       rng = np.random.default_rng(42)  # don't

   the ``42`` must move to the public signature as the default,
   and the function must use ``check_random_state`` to resolve it:

   ::

       def find_optimum(
           ...,
           *,
           random_state: int | np.random.Generator | None = None,
       ):
           rng = check_random_state(random_state)

4. **Unseeded ``default_rng()`` is forbidden except where
   "fresh noise on every call" is the documented contract.**
   The only such case currently is
   ``simulation.model.simulate``'s noise term, which is part of
   the simulator's documented behaviour and is gated behind an
   explicit ``# Fresh noise: documented behaviour, not an
   accident`` comment.

5. **Every ``@tool_spec`` that touches an RNG MUST declare its
   contract via the ``rng=`` metadata**:

   ::

       @tool_spec(
           name="bootstrap_pca",
           ...,
           rng={
               "uses_rng": True,
               "seed_param": "random_state",
               "default_seed": 0,
           },
       )

   The ``seed_param`` field is the name of the kwarg the
   reproducibility-check harness will exercise. Deterministic
   tools declare ``{"uses_rng": False}``.

6. **A self-test harness enforces (5).** A test in
   ``tests/test_rng_contract.py`` will import every registered
   tool, run it twice with the declared ``default_seed``, and
   assert byte-equal outputs. The same test verifies that a
   tool declaring ``{"uses_rng": False}`` produces byte-equal
   outputs without any seeding.

How to migrate an existing function
-----------------------------------

Most production paths are one mechanical edit:

Before:

::

    def my_thing(...):
        rng = np.random.default_rng()        # or np.random.default_rng(42)
        ...

After:

::

    def my_thing(
        ...,
        *,
        random_state: int | np.random.Generator | None = None,
    ):
        rng = check_random_state(random_state)
        ...

The behaviour for callers that did not pass ``random_state`` is
preserved (``None`` falls through to an unseeded ``default_rng``).

For an algorithm whose documented public behaviour was "always
seeded with 42", set ``default_seed=42`` in the
``@tool_spec(rng=...)`` metadata and accept ``random_state=None``
as a synonym; this preserves byte-equivalence for existing
callers.

What is *not* covered
---------------------

- The unseeded noise in ``simulation.model.simulate`` is
  deliberately *not* reproducible; that is the simulator's
  contract.
- The hard-coded plot seeds in ``surfaces.py`` /
  ``design_quality.py`` (seed for plot-only jitter) are
  in-scope for the migration but are low priority -- a plot
  with shifted-by-one jitter is not a correctness failure.
- Numerical reproducibility across NumPy major versions is
  out of scope (NumPy ships RNG algorithm changes on a
  documented schedule).

Open work
---------

Aspects of the contract that are not yet implemented:

- The ``tests/test_rng_contract.py`` harness that exercises every
  ``@tool_spec(rng={"uses_rng": True})`` against its declared
  ``default_seed`` will land alongside the first sweep that
  migrates production callsites.
- Migration of the existing offenders is tracked in
  `SEC-21 sub-item 9 <https://github.com/kgdunn/process-improve/issues/270>`_
  (Resampler), the relevant SEC-33 sub-item
  `(#282) <https://github.com/kgdunn/process-improve/issues/282>`_
  (``optimization.py:564``), and
  `ENG-08 (#290) <https://github.com/kgdunn/process-improve/issues/290>`_
  itself for the remaining call sites.

Cross-references
----------------

- :doc:`error_handling` -- warnings vs. errors policy.
- :doc:`deprecation_policy` -- migrating existing callers.
- scikit-learn's
  `random-state convention <https://scikit-learn.org/stable/glossary.html#term-random_state>`_
  is the reference semantics for ``check_random_state``.
