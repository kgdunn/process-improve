"""``random_seed`` is deprecated in favour of ``random_state`` (reproducibility contract)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import evaluate_all, evaluate_design, generate_design, generate_omars
from process_improve.experiments.factor import Factor

FACTORS = [Factor(name="A", low=0, high=1), Factor(name="B", low=0, high=1), Factor(name="C", low=0, high=1)]


def test_generate_design_random_seed_warns_and_matches_random_state() -> None:
    with pytest.warns(DeprecationWarning, match=r"generate_design\(random_seed=\.\.\.\) is deprecated since 1\.97\.0"):
        old = generate_design(FACTORS, "full_factorial", random_seed=7)
    new = generate_design(FACTORS, "full_factorial", random_state=7)
    pd.testing.assert_frame_equal(old.design, new.design)


def test_the_warning_points_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning, match="random_seed") as record:
        generate_design(FACTORS, "full_factorial", random_seed=7)
    assert record[0].filename == __file__


def test_passing_both_raises() -> None:
    with pytest.raises(ValueError, match="not both"), pytest.warns(DeprecationWarning, match="random_seed"):
        generate_design(FACTORS, "full_factorial", random_seed=7, random_state=8)


def test_default_seed_is_unchanged() -> None:
    """The default design is the one random_seed=42 gave, so existing scripts see the same run order."""
    default = generate_design(FACTORS, "ccd")
    with pytest.warns(DeprecationWarning, match="random_seed"):
        old = generate_design(FACTORS, "ccd", random_seed=42)
    pd.testing.assert_frame_equal(default.design, old.design)


def test_a_generator_is_accepted() -> None:
    result = generate_design(FACTORS, "latin_hypercube", budget=10, random_state=np.random.default_rng(3))
    assert result.n_runs == 10


def test_evaluate_design_and_evaluate_all_take_random_state() -> None:
    design = generate_design(FACTORS, "ccd")
    with pytest.warns(DeprecationWarning, match="evaluate_design"):
        old = evaluate_design(design, model="quadratic", metric="g_efficiency", random_seed=3)
    new = evaluate_design(design, model="quadratic", metric="g_efficiency", random_state=3)
    assert old == new
    with pytest.warns(DeprecationWarning, match="evaluate_all"):
        evaluate_all(design, model="quadratic", random_seed=3, n_samples=500)


@pytest.mark.slow
def test_generate_omars_takes_random_state() -> None:
    with pytest.warns(DeprecationWarning, match="generate_omars"):
        old = generate_omars(FACTORS, random_seed=5)
    new = generate_omars(FACTORS, random_state=5)
    pd.testing.assert_frame_equal(old.design, new.design)
