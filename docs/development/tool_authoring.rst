Authoring an MCP tool
=====================

.. note::

   This is the step-by-step for adding a new agent-callable tool. Tracks
   `ENG-26 <https://github.com/kgdunn/process-improve/issues/308>`_. See
   :doc:`../architecture` for how the tool layer fits together.

Anatomy of a tool
-----------------

Every tool is three things in one module:

#. a pydantic **input contract** - a ``BaseModel`` with
   ``model_config = ConfigDict(extra="forbid")`` so unknown keys are rejected;
#. a **wrapper function** decorated with ``@tool_spec(...)`` that takes the
   parsed model as its single positional argument and returns a JSON-serialisable
   ``dict``;
#. a **registration** so discovery can find it.

The ``@tool_spec`` decorator (``process_improve/tool_spec.py``) attaches the
JSON-schema spec (derived from the ``input_model``) and registers the function
in the global ``_TOOL_REGISTRY``. ``get_tool_specs()`` returns specs in
registry order; ``discover_tools()`` imports each subpackage's ``tools`` module
so the decorators run.

Step by step
------------

1. **Pick the home.** Domain tools live in ``<subpackage>/tools.py`` (small
   subpackages) or, where the surface is large, one module per tool under
   ``<subpackage>/_tools/<tool_name>.py`` with ``tools.py`` as the aggregator -
   this is the pattern in ``experiments/`` (ENG-02).

2. **Define the input model.** Use ``Field(...)`` with descriptions and
   validation (``ge``/``le``, ``min_length``, ``Literal[...]``). The descriptions
   become the tool's JSON schema that the LLM reads, so write them for a caller.

   .. code-block:: python

      from pydantic import BaseModel, ConfigDict, Field

      class SummariseInput(BaseModel):
          model_config = ConfigDict(extra="forbid")
          data: list[float] = Field(..., min_length=1, description="The values to summarise.")

3. **Write the wrapper** and decorate it. Narrow the ``except`` to the canonical
   expected set (see :doc:`error_handling`) and pass the result through
   ``clean(...)`` so numpy / pandas types serialise:

   .. code-block:: python

      from process_improve.tool_spec import clean, tool_spec

      @tool_spec(
          name="summarise_values",
          description="Return the mean and standard deviation of a list of numbers.",
          input_model=SummariseInput,
          examples='# "summarise [1, 2, 3]" -> ``summarise_values(data=[1, 2, 3])``',
          category="univariate",
      )
      def summarise_values(spec: SummariseInput) -> dict:
          try:
              import numpy as np  # noqa: PLC0415 - keep heavy imports lazy

              arr = np.asarray(spec.data, dtype=float)
              return clean({"mean": arr.mean(), "std": arr.std(ddof=1)})
          except (ValueError, TypeError) as exc:
              logger.exception("Tool summarise_values failed")
              return {"error": str(exc)}

4. **Register it.** Importing the module must run the decorator. If you use the
   per-tool layout, the subpackage's ``tools.py`` imports each tool module **in
   a fixed order** (the order fixes the spec-emission order) and tracks the
   names; if you add tools inline in ``tools.py`` they register in source order.
   Do not reorder existing imports - the tool-spec output is asserted stable.

5. **Confirm discovery.** ``tool_spec.discover_tools()`` imports your
   subpackage's ``tools`` module. If the subpackage is new, add its dotted
   ``...tools`` path to the discovery list in ``tool_spec.py``.

Conventions
-----------

- Keep heavy imports (numpy, pandas, statsmodels, the domain algorithm)
  **inside** the wrapper function (``# noqa: PLC0415``) so importing the tools
  module stays cheap.
- Return ``{"error": "..."}`` for *expected* failures; let unexpected exceptions
  propagate (the server redacts them).
- Always wrap the payload in ``clean(...)``.

Verifying
---------

.. code-block:: python

   from process_improve.tool_spec import get_tool_specs

   specs = {s["name"]: s for s in get_tool_specs()}
   assert "summarise_values" in specs
   assert specs["summarise_values"]["input_schema"]["additionalProperties"] is False

Add a test under ``tests/`` that drives the tool through the same path the MCP
server uses, plus an assertion that the spec is present and well-formed (see the
existing ``tests/test_experiments_tools.py`` and ``tests/test_tool_spec.py``).
