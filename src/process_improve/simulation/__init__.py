"""(c) Kevin Dunn, 2010-2026. MIT License.

Fake-data / process-simulator subpackage.

Provides three agent-callable tools used to demonstrate DOE workflows
against a synthetic (but deterministic) response surface:

- ``create_simulator`` - records a hidden model from a seed + factor
  specs + structural hints.
- ``simulate_process`` - evaluates the hidden surface at given factor
  settings, adding fresh Gaussian noise each call.
- ``reveal_simulator`` - returns the underlying coefficients, gated
  behind a ``confirmed`` flag that the host application enforces.

The math itself lives in :mod:`process_improve.simulation.model`; the
tools in :mod:`process_improve.simulation.tools` are the
JSON-schema-wrapped entry points registered with ``@tool_spec``.

A second, open simulator lives in :mod:`process_improve.simulation.batch`:
a deterministic fed-batch bioreactor with tunable disturbance channels,
the baseline for batch trajectory adaptation and mid-course correction.
Unlike the DOE simulator above, its parameters are deliberately visible.
"""

from process_improve.simulation.batch import (
    LATENT_FACTOR_NAMES,
    UPSTREAM_VARIABLE_NAMES,
    BioreactorConfig,
    BioreactorSimulator,
    cardinal_ph,
    cardinal_temperature,
    sample_initial_conditions,
    variance_decomposition,
)

__all__ = [
    "LATENT_FACTOR_NAMES",
    "UPSTREAM_VARIABLE_NAMES",
    "BioreactorConfig",
    "BioreactorSimulator",
    "cardinal_ph",
    "cardinal_temperature",
    "sample_initial_conditions",
    "variance_decomposition",
]
