Sensory Panel Analysis
======================

Descriptive panel-data analysis: validate the data, identify and optionally
correct panel anomalies, then relate the panel attributes back to the product.
The narrative walkthrough is in :doc:`/user_guide/sensory_panel`.

The subpackage ``__init__`` re-exports every name below, so
``from process_improve.sensory import validate_descriptive`` works. Each object is
documented once, under the module that defines it.

.. automodule:: process_improve.sensory
   :no-members:

Validation and ingest
---------------------

.. automodule:: process_improve.sensory.validation
   :members:
   :show-inheritance:

.. automodule:: process_improve.sensory.ingest
   :members:
   :show-inheritance:

Panel diagnostics
-----------------

.. automodule:: process_improve.sensory.panel
   :members:
   :show-inheritance:

.. Canonical entry lives in :doc:`/user_guide/sensory_diagnostics`, which documents these
   three alongside the prose that explains when to reach for them. Repeated here so the API
   reference is complete, without claiming the index entry.

.. automodule:: process_improve.sensory.diagnostics
   :members:
   :show-inheritance:
   :no-index:

.. automodule:: process_improve.sensory.mam
   :members:
   :show-inheritance:

Relating the panel to the product
---------------------------------

.. automodule:: process_improve.sensory.analysis
   :members:
   :show-inheritance:

.. automodule:: process_improve.sensory.designed
   :members:
   :show-inheritance:

Recipes and agent-callable tools
--------------------------------

.. automodule:: process_improve.sensory.recipes
   :members:
   :show-inheritance:

.. automodule:: process_improve.sensory.tools
   :members:
   :show-inheritance:
