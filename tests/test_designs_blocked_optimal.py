"""Tests for optimal designs whose blocks enter the exchange as fixed effects (#631).

A blocked design is analysed with the blocks as fixed effects, so what it estimates is
the factor information adjusted for the blocks: ``S = F'F - F'Z (Z'Z)^-1 Z'F`` for model
rows ``F`` and sum-coded block columns ``Z``. These tests compute ``S`` independently of
the exchange and compare the joint design with the old way, an unblocked optimum split
into blocks afterwards.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, analyze_experiment, generate_design
from process_improve.experiments._blocking import exchange_blocks
from process_improve.experiments.designs_constrained import _spread_over_blocks, block_columns


def _factors(k: int) -> list[Factor]:
    return [Factor(name=f"X{i}", low=-1, high=1) for i in range(k)]


def _quadratic_rows(x: np.ndarray) -> np.ndarray:
    """Return the full quadratic model rows: intercept, main effects, interactions, squares."""
    k = x.shape[1]
    columns = [np.ones(len(x)), *x.T]
    columns += [x[:, i] * x[:, j] for i, j in itertools.combinations(range(k), 2)]
    columns += [x[:, i] ** 2 for i in range(k)]
    return np.column_stack(columns)


def _adjusted_information(f: np.ndarray, labels: np.ndarray, n_blocks: int) -> np.ndarray:
    """Return ``S``, the information on the model coefficients once the (sum-coded) block effects are estimated."""
    z = np.vstack([np.eye(n_blocks - 1), -np.ones((1, n_blocks - 1))])[np.asarray(labels) - 1]
    return f.T @ f - f.T @ z @ np.linalg.solve(z.T @ z, z.T @ f)


def _adjusted_log_det(design: pd.DataFrame, n_blocks: int, labels: np.ndarray | None = None) -> float:
    """Return ``log det(S)`` for a quadratic model in the design's ``X`` columns."""
    names = [c for c in design.columns if c.startswith("X")]
    labels = design["Block"].to_numpy() if labels is None else labels
    return float(
        np.linalg.slogdet(_adjusted_information(_quadratic_rows(design[names].to_numpy(float)), labels, n_blocks))[1]
    )


def _blocked(k: int, budget: int, n_blocks: int, **kwargs) -> pd.DataFrame:
    """Return the design of a quadratic D-optimal design in ``n_blocks`` blocks, warnings silenced."""
    kwargs.setdefault("design_type", "d_optimal")
    kwargs.setdefault("model_type", "quadratic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_design(_factors(k), budget=budget, n_blocks=n_blocks, random_state=0, **kwargs)


def _blocked_afterwards(k: int, budget: int, n_blocks: int) -> float:
    """Return ``log det(S)`` of the unblocked quadratic D-optimum split into blocks by the old exchange."""
    unblocked = generate_design(_factors(k), "d_optimal", budget=budget, model_type="quadratic", random_state=0)
    x = unblocked.design[[f"X{i}" for i in range(k)]]
    labels = exchange_blocks(x.to_numpy(float), n_blocks, np.random.default_rng(0)).labels
    return _adjusted_log_det(x, n_blocks, labels)


class TestBetterThanBlockingAfterwards:
    """Choosing the runs for the blocked model is never worse, and sometimes better, than blocking an optimum."""

    def test_two_factors_two_blocks(self) -> None:
        """The issue's case: 2 factors, quadratic, 2 blocks. At 11 runs the joint design is 0.9% more D-efficient."""
        joint, afterwards = _adjusted_log_det(_blocked(2, 11, 2).design, 2), _blocked_afterwards(2, 11, 2)
        assert np.exp((joint - afterwards) / 6) > 1.008

    @pytest.mark.parametrize(("k", "budget", "n_blocks"), [(2, 8, 2), (2, 12, 2), (2, 16, 2), (2, 9, 3), (2, 16, 4)])
    def test_never_worse(self, k: int, budget: int, n_blocks: int) -> None:
        """Over a range of sizes the joint design matches or beats blocking afterwards."""
        joint = _adjusted_log_det(_blocked(k, budget, n_blocks).design, n_blocks)
        assert joint >= _blocked_afterwards(k, budget, n_blocks) - 1e-9

    def test_many_small_blocks(self) -> None:
        """Twelve runs in six pairs: 2.4% more D-efficient than pairing the unblocked optimum's runs."""
        joint, afterwards = _adjusted_log_det(_blocked(2, 12, 6).design, 6), _blocked_afterwards(2, 12, 6)
        assert np.exp((joint - afterwards) / 6) > 1.02


class TestAugmentationInANewBlock:
    """Runs already made are a block of their own: the follow-up runs are chosen knowing they come on another day."""

    @pytest.fixture(scope="class")
    def fixed(self) -> pd.DataFrame:
        """Return yesterday's 2^2 factorial."""
        return pd.DataFrame({"X0": [-1.0, 1.0, -1.0, 1.0], "X1": [-1.0, -1.0, 1.0, 1.0]})

    def test_fixed_runs_are_the_first_block_and_keep_their_order(self, fixed: pd.DataFrame) -> None:
        """The four runs already made come first, unshuffled, in block 1; the five new ones are block 2."""
        result = _blocked(2, 9, 2, fixed_runs=fixed)
        design = result.design
        assert design["Block"].tolist() == [1] * 4 + [2] * 5
        np.testing.assert_array_equal(design[["X0", "X1"]].to_numpy()[:4], fixed.to_numpy())
        assert design["RunOrder"].tolist()[:4] == [1, 2, 3, 4]
        assert result.metadata["blocking"]["method"] == "optimal_exchange"

    def test_beats_augmenting_without_the_block(self, fixed: pd.DataFrame) -> None:
        """Analysed with the day effect, the blocked augmentation is 13% more D-efficient than ignoring the day."""
        blocked = _adjusted_log_det(_blocked(2, 9, 2, fixed_runs=fixed).design, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unblocked = generate_design(
                _factors(2), "d_optimal", budget=9, model_type="quadratic", fixed_runs=fixed, random_state=0
            )
        ignoring = _adjusted_log_det(unblocked.design, 2, np.repeat([1, 2], [4, 5]))
        assert np.exp((blocked - ignoring) / 6) > 1.10


class TestTheBlockedDesign:
    """Structure and metadata of a blocked optimal design."""

    def test_metadata_reports_the_blocks_adjusted_log_det(self) -> None:
        """``log_det_information`` is ``log det(S)``, computed here independently; the labels are not left behind."""
        result = _blocked(2, 12, 3)
        assert result.metadata["log_det_information"] == pytest.approx(_adjusted_log_det(result.design, 3), rel=1e-10)
        assert result.metadata["blocking"] == {
            "method": "optimal_exchange",
            "generators": [],
            "confounded_with": [],
            "model": "quadratic",
        }
        assert "block_labels" not in result.metadata

    def test_blocks_are_near_equal_and_run_in_turn(self) -> None:
        """Eleven runs in three blocks: sizes 4, 4, 3, and the run sheet takes the blocks one after another."""
        design = _blocked(2, 11, 3).design
        assert sorted(design["Block"].value_counts().tolist()) == [3, 4, 4]
        assert design["Block"].is_monotonic_increasing

    def test_a_optimal_scores_the_adjusted_trace(self) -> None:
        """A-optimality reports ``trace(S^-1)``, the summed variance of the coefficients adjusted for blocks."""
        result = _blocked(2, 12, 2, design_type="a_optimal")
        s = _adjusted_information(
            _quadratic_rows(result.design[["X0", "X1"]].to_numpy(float)), result.design["Block"], 2
        )
        assert result.metadata["trace_criterion"] == pytest.approx(np.trace(np.linalg.inv(s)), rel=1e-9)
        assert result.metadata["blocking"]["method"] == "optimal_exchange"

    def test_i_optimal_is_blocked_in_the_exchange(self) -> None:
        """I-optimality blocks its runs too, in near-equal blocks."""
        result = _blocked(2, 12, 2, design_type="i_optimal")
        assert result.metadata["blocking"]["method"] == "optimal_exchange"
        assert result.design["Block"].value_counts().tolist() == [6, 6]

    def test_constraints_and_categorical_factors(self) -> None:
        """Every run satisfies the constraint, and a three-level categorical factor is blocked with the rest."""
        factors = [*_factors(2), Factor(name="Supplier", type="categorical", levels=["a", "b", "c"])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate_design(
                factors,
                "d_optimal",
                budget=16,
                n_blocks=2,
                model_type="interactions",
                constraints=["X0 + X1 <= 1"],
                random_state=0,
            )
        design = result.design_actual
        assert (design["X0"] + design["X1"] <= 1 + 1e-9).all()
        assert result.metadata["blocking"]["method"] == "optimal_exchange"
        assert design["Block"].value_counts().tolist() == [8, 8]

    def test_a_candidate_set(self) -> None:
        """Runs chosen from a list are blocked too, and each selection names a row of the list."""
        candidates = pd.DataFrame(
            {"X0": [-1, 0, 1, -1, 0, 1, -1, 0, 1], "X1": [-1, -1, -1, 0, 0, 0, 1, 1, 1]},
            index=[f"point_{i}" for i in range(9)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate_design(
                _factors(2), "d_optimal", budget=12, n_blocks=2, model_type="quadratic", candidates=candidates
            )
        selected = result.metadata["selected_candidates"]
        assert set(selected) <= set(candidates.index)
        assert sum(selected.values()) == 12
        assert result.design["Block"].value_counts().tolist() == [6, 6]

    def test_a_polished_design_keeps_its_blocks(self) -> None:
        """Six factors: the coarse grid is polished on a finer one, and the moves keep each run's block."""
        result = _blocked(6, 32, 2)
        assert result.metadata["polish_levels"] == 5
        assert result.design["Block"].value_counts().tolist() == [16, 16]
        assert result.metadata["log_det_information"] == pytest.approx(_adjusted_log_det(result.design, 2), rel=1e-10)

    def test_the_analysis_fits_the_blocks(self) -> None:
        """A shift between the blocks is estimated as the Block effect and leaves the factor effects alone."""
        design = _blocked(2, 12, 2, model_type="interactions").design
        x0, x1 = design["X0"], design["X1"]
        y = 10 + 3 * x0 - 2 * x1 + x0 * x1 + np.where(design["Block"] == 1, 5.0, -5.0)
        result = analyze_experiment(design.assign(y=y), response_column="y", analysis_type="coefficients")
        coefficients = {row["term"]: row["coefficient"] for row in result["coefficients"]}
        assert coefficients["Block1"] == pytest.approx(5.0)
        assert coefficients["X0"] == pytest.approx(3.0)
        assert coefficients["X0:X1"] == pytest.approx(1.0)


class TestWhenBlocksAreAssignedAfterwards:
    """Criteria and settings the exchange cannot block keep the old assignment."""

    def test_e_optimal(self) -> None:
        """E-optimality's search does not take blocks: the runs are split afterwards."""
        assert _blocked(2, 12, 2, design_type="e_optimal").metadata["blocking"]["method"] == "exchange"

    def test_replicated_designs(self) -> None:
        """A replicated design is blocked after it is replicated."""
        assert _blocked(2, 8, 2, n_replicates=2).metadata["blocking"]["method"] == "exchange"


class TestLimits:
    """Budgets the blocks do not fit in."""

    def test_the_budget_rises_for_the_block_effects(self, caplog: pytest.LogCaptureFixture) -> None:
        """A quadratic model in 2 factors has 6 coefficients; 3 blocks add 2, so 7 runs become 8."""
        with caplog.at_level("WARNING"):
            result = _blocked(2, 7, 3)
        assert len(result.design) == 8
        assert result.metadata["budget_requested"] == 7
        assert "adds 2 block effect(s)" in caplog.text

    def test_too_many_blocks_for_the_budget(self) -> None:
        """Four blocks need at least two runs each, more than a budget of 6 leaves."""
        with pytest.raises(ValueError, match="fewer blocks"):
            _blocked(2, 6, 4, model_type="main_effects")


def test_block_columns_are_sum_coded() -> None:
    """Block ``b`` has a 1 in column ``b``; the last block has -1 in every column, as analyze_experiment codes it."""
    np.testing.assert_array_equal(block_columns(np.array([0, 1, 2]), 3), [[1, 0], [0, 1], [-1, -1]])


def test_starts_are_dealt_out_to_the_blocks() -> None:
    """A start's runs go to the blocks in turn, each to the same point in its block's copy of the candidates."""
    segment = np.repeat([1, 2, 3], 10)  # three blocks, ten candidate points
    rows = _spread_over_blocks(np.array([13, 24, 5, 27]), segment)
    np.testing.assert_array_equal(rows, [3, 14, 25, 7])
    np.testing.assert_array_equal(segment[rows], [1, 2, 3, 1])
