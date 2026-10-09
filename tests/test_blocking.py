"""Blocking in ``generate_design(n_blocks=...)`` and ``augment_design(..., "add_blocks")``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import augment_design, generate_design
from process_improve.experiments._blocking import confounding_blocks, exchange_blocks
from process_improve.experiments.factor import Factor


def _factors(k: int) -> list[Factor]:
    return [Factor(name=f"x{i}", low=0, high=10) for i in range(k)]


def _within_block_centred(x: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    out = x.copy()
    for b in np.unique(blocks):
        out[blocks == b] -= x[blocks == b].mean(axis=0)
    return out


@pytest.mark.parametrize(("k", "n_blocks"), [(3, 2), (4, 2), (4, 4), (5, 4), (6, 8), (7, 8)])
def test_factorial_blocks_are_orthogonal_to_main_effects_and_interactions(k: int, n_blocks: int) -> None:
    """With blocks from confounded words, main effects and 2FIs keep their full information."""
    result = generate_design(_factors(k), "full_factorial", n_blocks=n_blocks, n_center_points=0)
    x = result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)
    blocks = result.design["Block"].to_numpy()
    model = np.column_stack([x, *[x[:, i] * x[:, j] for i in range(k) for j in range(i + 1, k)]])
    centred = _within_block_centred(model, blocks)
    # Every main effect stays orthogonal to blocks; 2FIs too, except any the blocks confound.
    confounded = set(result.metadata["blocking"]["confounded_with"])
    assert np.allclose(centred[:, :k], x)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    for column, (i, j) in zip(centred.T[k:], pairs, strict=True):
        if f"x{i}x{j}" not in confounded:
            np.testing.assert_allclose(column, x[:, i] * x[:, j])
    assert np.all(np.bincount(blocks)[1:] == len(x) // n_blocks)


def test_four_blocks_of_a_two_to_the_four_confound_only_one_interaction() -> None:
    """Every 4-block scheme of a 2^4 confounds some two-factor interaction; never a main effect (#bug: D was)."""
    result = generate_design(_factors(4), "full_factorial", n_blocks=4, n_center_points=0)
    confounded = result.metadata["blocking"]["confounded_with"]
    assert all(len(word) >= 4 for word in confounded)  # "x2x3" or longer: no single factor
    assert sum(len(word) == 4 for word in confounded) == 1


def test_runs_are_randomised_within_blocks_and_blocks_run_in_turn() -> None:
    result = generate_design(_factors(4), "full_factorial", n_blocks=2, n_center_points=2, random_state=3)
    blocks = result.design["Block"].tolist()
    assert blocks == sorted(blocks)
    assert blocks.count(1) == blocks.count(2) == 9


@pytest.mark.parametrize(("design_type", "k"), [("ccd", 3), ("box_behnken", 4), ("dsd", 6)])
def test_other_designs_block_by_exchange(design_type: str, k: int) -> None:
    result = generate_design(_factors(k), design_type, n_blocks=2)
    assert result.metadata["blocking"]["method"] == "exchange"
    x = result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)
    blocks = result.design["Block"].to_numpy()
    # Main effects keep (nearly) all of their information once each block has its own mean.
    full = (x - x.mean(axis=0)).T @ (x - x.mean(axis=0))
    blocked = _within_block_centred(x, blocks).T @ _within_block_centred(x, blocks)
    assert np.linalg.det(blocked) >= 0.95 * np.linalg.det(full)


def test_blocks_must_be_a_power_of_two_for_a_factorial() -> None:
    with pytest.raises(ValueError, match="2, 4, 8"):
        confounding_blocks(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float), 3, ["a", "b"])


def test_too_many_blocks_for_the_runs_raises() -> None:
    with pytest.raises(ValueError, match="between 2 and half"):
        exchange_blocks(np.eye(4), 3, np.random.default_rng(1))


def test_blocks_with_fixed_runs_raise_unless_the_exchange_blocks_them() -> None:
    """E-optimal designs are blocked afterwards, which cannot move runs already made; D-optimal ones block them in."""
    fixed = pd.DataFrame({"x0": [0.0], "x1": [0.0]})
    with pytest.raises(ValueError, match="fixed_runs"):
        generate_design(_factors(2), "e_optimal", budget=8, fixed_runs=fixed, n_blocks=2)
    result = generate_design(_factors(2), "d_optimal", budget=8, fixed_runs=fixed, n_blocks=2)
    assert result.design["Block"].iloc[0] == 1
    assert (result.design["Block"].iloc[1:] == 2).all()


def test_add_blocks_never_confounds_a_main_effect() -> None:
    """augment_design used ABCD and ABC as generators for 4 blocks, whose product D is a main effect."""
    base = generate_design(_factors(4), "full_factorial", n_center_points=0)
    design = base.design[[f"x{i}" for i in range(4)]]
    result = augment_design(design, "add_blocks", n_additional_runs=4)
    blocks = np.array([row["Block"] for row in result["augmented_design"]])
    x = design.to_numpy(dtype=float)
    np.testing.assert_allclose(_within_block_centred(x, blocks), x)
    assert all(len(word) >= 4 for word in result["confounded_with"])
