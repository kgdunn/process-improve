# (c) Kevin Dunn, 2010-2026. MIT License.

"""Shared utilities for design generation: randomization, center points, blocking, coded/actual mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve.experiments.structures import Column, Expt, c, gather

if TYPE_CHECKING:
    from process_improve.experiments.factor import DesignResult, Factor


def matrix_to_columns(
    matrix: np.ndarray,
    factors: list[Factor],
    *,
    is_actual: bool = False,
) -> list[Column]:
    """Convert a numpy design matrix to a list of Column objects with factor metadata.

    Parameters
    ----------
    matrix : np.ndarray
        Design matrix of shape (n_runs, n_factors).
    factors : list[Factor]
        Factor specifications (must match column order of *matrix*).
    is_actual : bool
        If ``True``, the matrix values are already in actual (real-world)
        units (e.g. mixture proportions).  If ``False`` (default), values
        are in coded -1/+1 units.

    Returns
    -------
    list[Column]
        One ``Column`` per factor, with ``pi_*`` metadata set from the ``Factor``.
    """
    columns: list[Column] = []
    for i, factor in enumerate(factors):
        col = c(
            matrix[:, i].tolist(),
            name=factor.name,
            lo=factor.low,
            hi=factor.high,
            units=factor.units,
            coded=not is_actual,
        )
        columns.append(col)
    return columns


def columns_to_expt(columns: list[Column], title: str | None = None) -> Expt:
    """Assemble a list of Columns into an Expt dataframe.

    Parameters
    ----------
    columns : list[Column]
        Factor columns (all must have the same length).
    title : str or None
        Optional experiment title.

    Returns
    -------
    Expt
    """
    return gather(**{col.pi_name: col for col in columns}, title=title)


def coded_to_actual(columns: list[Column]) -> list[Column]:
    """Convert a list of coded Columns to real-world units.

    Parameters
    ----------
    columns : list[Column]
        Columns in coded units.

    Returns
    -------
    list[Column]
        Columns converted to actual (real-world) units via ``to_realworld()``.
    """
    return [col.to_realworld() for col in columns]


def add_center_points(matrix: np.ndarray, n_center: int) -> np.ndarray:
    """Append center point rows (all zeros) to a coded design matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Coded design matrix of shape (n_runs, n_factors).
    n_center : int
        Number of center point replicates to add.

    Returns
    -------
    np.ndarray
        Design matrix with center points appended.
    """
    if n_center <= 0:
        return matrix
    center_rows = np.zeros((n_center, matrix.shape[1]))
    return np.vstack([matrix, center_rows])


def replicate_design(matrix: np.ndarray, replicates: int) -> np.ndarray:
    """Replicate an entire design matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Design matrix of shape (n_runs, n_factors).
    replicates : int
        Number of full replicates (1 = no replication).

    Returns
    -------
    np.ndarray
        Vertically stacked matrix with ``replicates`` copies.
    """
    if replicates <= 1:
        return matrix
    return np.tile(matrix, (replicates, 1))


def assign_blocks(n_runs: int, n_blocks: int) -> list[int]:
    """Assign runs to blocks using round-robin.

    Parameters
    ----------
    n_runs : int
        Total number of runs.
    n_blocks : int
        Number of blocks.

    Returns
    -------
    list[int]
        Block assignment (1-based) for each run.
    """
    return [(i % n_blocks) + 1 for i in range(n_runs)]


def build_design_result(  # noqa: PLR0913
    coded_matrix: np.ndarray,
    factors: list[Factor],
    design_type: str,
    center_points: int = 0,
    replicates: int = 1,
    blocks: int | None = None,
    random_seed: int | None = 42,
    generators: list[str] | None = None,
    defining_relation: list[str] | None = None,
    resolution: int | None = None,
    alpha: float | None = None,
    metadata: dict | None = None,
    is_actual: bool = False,
) -> DesignResult:
    """Post-process a raw design matrix into a complete DesignResult.

    This is the common pipeline shared by all dispatch handlers:
    1. Add center points
    2. Replicate
    3. Randomize run order
    4. Convert to Column/Expt (coded + actual)
    5. Assign blocks if requested
    6. Build DesignResult

    Parameters
    ----------
    coded_matrix : np.ndarray
        Raw design matrix from a dispatch handler.  In coded -1/+1 units
        unless *is_actual* is ``True``.
    factors : list[Factor]
        Factor specifications.
    design_type : str
        Name of the design type.
    center_points : int
        Number of center point replicates to add.
    replicates : int
        Number of full replicates.
    blocks : int or None
        Number of blocks (None = no blocking).
    random_seed : int
        Seed for reproducible randomization.
    generators : list[str] or None
        Generator strings (fractional factorials).
    defining_relation : list[str] or None
        Defining relation words.
    resolution : int or None
        Design resolution.
    alpha : float or None
        Axial distance (CCD).
    metadata : dict or None
        Extra design-specific metadata.
    is_actual : bool
        If ``True``, *coded_matrix* contains actual-unit values (e.g.
        mixture proportions).  Both the coded and actual ``Expt`` will
        contain these values directly.

    Returns
    -------
    DesignResult
    """
    from process_improve.experiments.factor import DesignResult  # noqa: PLC0415

    # 1. Add center points (only for coded designs)
    matrix = add_center_points(coded_matrix, center_points) if not is_actual else coded_matrix

    # 2. Replicate
    matrix = replicate_design(matrix, replicates)

    n_runs = matrix.shape[0]

    # 3. Randomize: shuffle the row order of the design matrix.
    #    When random_seed is None the original order is preserved (used for
    #    optimal designs whose run order is part of the solution, e.g. split-plot).
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(n_runs)
        matrix_randomized = matrix[perm]
        run_order = (perm + 1).tolist()
    else:
        matrix_randomized = matrix
        run_order = list(range(1, n_runs + 1))

    # 4. Convert to Columns and Expt
    if is_actual:
        # Matrix is already in actual units (e.g. mixture proportions)
        actual_columns = matrix_to_columns(matrix_randomized, factors, is_actual=True)
        coded_columns = actual_columns  # same for mixture designs
        design_coded = columns_to_expt(coded_columns, title=f"{design_type} design (proportions)")
        design_actual = columns_to_expt(actual_columns, title=f"{design_type} design (proportions)")
    else:
        coded_columns = matrix_to_columns(matrix_randomized, factors)
        actual_columns = coded_to_actual(coded_columns)
        design_coded = columns_to_expt(coded_columns, title=f"{design_type} design (coded)")
        design_actual = columns_to_expt(actual_columns, title=f"{design_type} design (actual)")

    factor_names = [f.name for f in factors]

    # Add run order column (sequential 1..N for the experimenter)
    design_coded.insert(0, "RunOrder", list(range(1, n_runs + 1)))
    design_actual.insert(0, "RunOrder", list(range(1, n_runs + 1)))

    # Reset index to 1-based
    design_coded.index = pd.RangeIndex(1, n_runs + 1)
    design_actual.index = pd.RangeIndex(1, n_runs + 1)

    # 5. Blocks
    block_assignments = None
    if blocks is not None and blocks > 1:
        block_assignments = assign_blocks(n_runs, blocks)
        design_coded["Block"] = block_assignments
        design_actual["Block"] = block_assignments

    return DesignResult(
        design=design_coded,
        design_actual=design_actual,
        run_order=run_order,
        design_type=design_type,
        n_runs=n_runs,
        n_factors=len(factors),
        factor_names=factor_names,
        generators=generators,
        defining_relation=defining_relation,
        resolution=resolution,
        alpha=alpha,
        blocks=block_assignments,
        metadata=metadata or {},
    )
