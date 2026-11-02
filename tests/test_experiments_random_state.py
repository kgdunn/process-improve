"""``random_state`` is the experiments module's seed keyword; ``random_seed`` was removed in 2.0.0."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import evaluate_all, evaluate_design, generate_design, generate_omars
from process_improve.experiments.factor import Factor
from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import execute_tool_call

FACTORS = [Factor(name="A", low=0, high=1), Factor(name="B", low=0, high=1), Factor(name="C", low=0, high=1)]


@pytest.mark.parametrize("function", [generate_design, evaluate_design, evaluate_all, generate_omars])
def test_random_seed_is_an_unexpected_keyword(function: Callable[..., Any]) -> None:
    # Passed through a dict: a literal keyword the signature lacks fails static analysis (CodeQL).
    removed = {"random_seed": 7}
    with pytest.raises(TypeError, match="unexpected keyword argument 'random_seed'"):
        function(FACTORS, **removed)


def test_the_tool_refuses_random_seed_as_an_unknown_key() -> None:
    factors = [{"name": "A", "low": 0, "high": 1}, {"name": "B", "low": 0, "high": 1}]
    with pytest.raises(ToolInputInvalidError, match="random_seed"):
        execute_tool_call("generate_design", {"factors": factors, "random_seed": 7})


def test_the_default_seed_is_42() -> None:
    """The default design is the one ``random_state=42`` gives, so a script gets the same run order every time."""
    pd.testing.assert_frame_equal(
        generate_design(FACTORS, "ccd").design, generate_design(FACTORS, "ccd", random_state=42).design
    )


def test_a_generator_is_accepted() -> None:
    result = generate_design(FACTORS, "latin_hypercube", budget=10, random_state=np.random.default_rng(3))
    assert result.n_runs == 10


def test_evaluate_design_and_evaluate_all_take_random_state() -> None:
    design = generate_design(FACTORS, "ccd")
    first = evaluate_design(design, model="quadratic", metric="g_efficiency", random_state=3)
    assert first == evaluate_design(design, model="quadratic", metric="g_efficiency", random_state=3)
    assert evaluate_all(design, model="quadratic", random_state=3, n_samples=500)["fds"]["random_seed"] == 3


@pytest.mark.slow
def test_generate_omars_takes_random_state() -> None:
    pd.testing.assert_frame_equal(
        generate_omars(FACTORS, random_state=5).design, generate_omars(FACTORS, random_state=5).design
    )
