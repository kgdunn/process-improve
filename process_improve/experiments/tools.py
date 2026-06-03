"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for designed experiments.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.

ENG-02: each tool now lives in its own module under ``_tools/``. This module
is the aggregator that ``process_improve.tool_spec`` discovery imports; importing
it pulls in every tool submodule (running their ``@tool_spec`` decorators and
``_register`` calls). The import order below is load-bearing: it fixes the
``@tool_spec`` decorator execution order, which in turn fixes the registry
insertion order and therefore the order tool specs are emitted in. Do not
reorder it - hence the file-level ``isort`` opt-out.
"""

# ruff: noqa: I001  -- import order below is significant (see module docstring)
from __future__ import annotations

from process_improve.experiments._tools import _EXPERIMENTS_TOOL_NAMES
from process_improve.tool_spec import get_tool_specs

# Tool submodules, imported in their original source order so the @tool_spec
# decorators register in the same order as before the ENG-02 split.
from process_improve.experiments._tools.create_factorial_design import create_factorial_design
from process_improve.experiments._tools.fit_linear_model import fit_linear_model
from process_improve.experiments._tools.generate_design import generate_design_tool
from process_improve.experiments._tools.evaluate_design import evaluate_design_tool
from process_improve.experiments._tools.analyze_experiment import analyze_experiment_tool
from process_improve.experiments._tools.optimize_responses import optimize_responses_tool
from process_improve.experiments._tools.augment_design import augment_design_tool
from process_improve.experiments._tools.visualize_doe import visualize_doe_tool
from process_improve.experiments._tools.doe_knowledge import doe_knowledge_tool
from process_improve.experiments._tools.recommend_strategy import recommend_strategy_tool

__all__ = [
    "analyze_experiment_tool",
    "augment_design_tool",
    "create_factorial_design",
    "doe_knowledge_tool",
    "evaluate_design_tool",
    "fit_linear_model",
    "generate_design_tool",
    "get_experiments_tool_specs",
    "optimize_responses_tool",
    "recommend_strategy_tool",
    "visualize_doe_tool",
]


def get_experiments_tool_specs() -> list[dict]:
    """Return tool specs for all experiments tools registered in this module."""
    return get_tool_specs(names=_EXPERIMENTS_TOOL_NAMES)
