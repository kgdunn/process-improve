Recipes and the Tool Layer
==========================

The machinery that exposes the package to an LLM agent: the analysis-recipe
framework, the ``@tool_spec`` decorator and its registry, the settings
singleton, the subprocess isolation used when a tool call arrives over an
untrusted transport, and the MCP server that publishes the registered tools.
The narrative overview is in :doc:`/architecture`.

Analysis recipes
----------------

.. automodule:: process_improve.recipes
   :members:
   :show-inheritance:

Tool specifications and registry
--------------------------------

.. automodule:: process_improve.tool_spec
   :members:
   :show-inheritance:

Configuration
-------------

.. automodule:: process_improve.config
   :members:
   :show-inheritance:

Tool-call safety
----------------

.. automodule:: process_improve.tool_safety
   :members:
   :show-inheritance:

MCP server
----------

Requires the optional ``mcp`` extra; the docs environment installs every extra,
so autodoc imports it directly rather than mocking it.

.. automodule:: process_improve.mcp_server
   :members:
   :show-inheritance:
