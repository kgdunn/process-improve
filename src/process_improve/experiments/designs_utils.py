# (c) Kevin Dunn, 2010-2026. MIT License.

"""Shared utilities for design generation: randomization, center points, blocking, coded/actual mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from process_improve._random import check_random_state
from process_improve.experiments._blocking import Blocking, confounding_blocks, exchange_blocks, is_regular_two_level
from process_improve.experiments.structures import Column, Expt, c, gather

if TYPE_CHECKING:
    from process_improve.experiments.factor import DesignResult, Factor


def categorical_codes(n_levels: int) -> np.ndarray:
    """Coded values that stand for the levels of a categorical factor in a numeric design matrix.

    Level ``i`` of ``n_levels`` is ``np.linspace(-1, 1, n_levels)[i]``: -1 and +1 for a
    two-level factor, as in every two-level design, and -1, 0, +1 for three levels.
    """
    return np.linspace(-1.0, 1.0, n_levels)


def categorical_labels(values: np.ndarray, factor: Factor) -> list:
    """Map a categorical factor's column of a numeric design matrix to its level labels.

    Raises
    ------
    ValueError
        If a value is not one of :func:`categorical_codes`, such as the 0 of a centre
        point for a two-level factor.
    """
    levels = list(factor.levels or [])
    codes = categorical_codes(len(levels))
    index = np.abs(np.asarray(values, dtype=float)[:, None] - codes[None, :]).argmin(axis=1)
    bad = ~np.isclose(np.asarray(values, dtype=float), codes[index], atol=1e-9)
    if bad.any():
        raise ValueError(
            f"Categorical factor {factor.name!r} has {len(levels)} levels, coded {codes.tolist()}, but the "
            f"design asks for the setting {float(np.asarray(values, dtype=float)[bad][0])}; this design type "
            "cannot place a categorical factor there."
        )
    return [levels[i] for i in index]


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
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    columns: list[Column] = []
    for i, factor in enumerate(factors):
        values = matrix[:, i].tolist()
        if factor.type == FactorType.categorical:
            if matrix.dtype != object:
                # Numeric designs carry level codes (see categorical_codes); optimal designs carry labels.
                values = categorical_labels(matrix[:, i], factor)
            # A categorical factor carries labels, not coded numbers. Build it
            # from its levels and mark it not-coded so the coded<->actual affine
            # map (which assumes a numeric low/high range) is skipped and the
            # labels pass straight through to both the coded and actual designs.
            col = c(values, name=factor.name, levels=factor.levels, units=factor.units)
            col.pi_is_coded = False
        else:
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
    return gather(**cast("dict[str, Column]", {col.pi_name: col for col in columns}), title=title)


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


def add_center_points(matrix: np.ndarray, n_center: int, factors: list[Factor] | None = None) -> np.ndarray:
    """Append center point rows to a coded design matrix.

    Continuous factors sit at 0. A categorical factor has no centre, so its centre runs
    cycle through its levels (the usual practice of a centre point per level).

    Parameters
    ----------
    matrix : np.ndarray
        Coded design matrix of shape (n_runs, n_factors).
    n_center : int
        Number of center point replicates to add.
    factors : list[Factor] or None
        The factors, to find the categorical columns; ``None`` treats every column as continuous.

    Returns
    -------
    np.ndarray
        Design matrix with center points appended.
    """
    if n_center <= 0:
        return matrix
    center_rows = np.zeros((n_center, matrix.shape[1]))
    for j, factor in enumerate(factors or []):
        if factor.levels and factor.type.value == "categorical":
            codes = categorical_codes(len(factor.levels))
            center_rows[:, j] = codes[np.arange(n_center) % len(codes)]
    return np.vstack([matrix, center_rows])


def replicate_design(matrix: np.ndarray, n_replicates: int) -> np.ndarray:
    """Replicate an entire design matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Design matrix of shape (n_runs, n_factors).
    n_replicates : int
        Number of full replicates (1 = no replication).

    Returns
    -------
    np.ndarray
        Vertically stacked matrix with ``n_replicates`` copies.
    """
    if n_replicates <= 1:
        return matrix
    return np.tile(matrix, (n_replicates, 1))


def _numeric_codes(matrix: np.ndarray, factors: list[Factor]) -> np.ndarray:
    """Return the design as floats, categorical labels replaced by their level codes."""
    if matrix.dtype != object:
        return matrix.astype(float)
    out = np.empty(matrix.shape)
    for j, factor in enumerate(factors):
        if factor.type.value == "categorical" and factor.levels:
            codes = categorical_codes(len(factor.levels))
            out[:, j] = [codes[list(factor.levels).index(v)] if v in factor.levels else v for v in matrix[:, j]]
        else:
            out[:, j] = matrix[:, j].astype(float)
    return out


def _assign_blocks(
    matrix: np.ndarray,
    factors: list[Factor],
    design_type: str,
    n_blocks: int,
    rng: np.random.Generator | None,
) -> Blocking:
    """Confound interaction words with blocks in a regular two-level factorial; exchange otherwise."""
    numeric = _numeric_codes(matrix, factors)
    if design_type in ("full_factorial", "fractional_factorial") and is_regular_two_level(numeric):
        return confounding_blocks(numeric, n_blocks, [f.name for f in factors])
    return exchange_blocks(numeric, n_blocks, rng if rng is not None else np.random.default_rng())


def build_design_result(  # noqa: PLR0913
    coded_matrix: np.ndarray,
    factors: list[Factor],
    design_type: str,
    n_center_points: int = 0,
    n_replicates: int = 1,
    n_blocks: int | None = None,
    random_state: int | np.random.Generator | None = 42,
    generators: list[str] | None = None,
    defining_relation: list[str] | None = None,
    resolution: int | None = None,
    alpha: float | None = None,
    metadata: dict | None = None,
    is_actual: bool = False,
    n_leading_fixed: int = 0,
    randomize: bool = True,
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
    n_center_points : int
        Number of center point replicates to add.
    n_replicates : int
        Number of full replicates.
    n_blocks : int or None
        Number of blocks (None = no blocking).
    random_state : int, numpy.random.Generator or None
        Seed for the run-order randomisation (``None``: fresh entropy).
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
    n_leading_fixed : int
        Number of leading rows that are runs already performed (the fixed runs
        of an augmentation). They keep their place at the top of the run sheet;
        only the remaining rows are randomised.
    randomize : bool
        ``False`` keeps the run order of *coded_matrix* (designs whose run order
        is part of the solution, e.g. split-plot optimal designs).

    Returns
    -------
    DesignResult
    """
    from process_improve.experiments.factor import DesignResult  # noqa: PLC0415

    # 1. Add center points (only for coded designs)
    matrix = add_center_points(coded_matrix, n_center_points, factors) if not is_actual else coded_matrix

    # 2. Replicate
    matrix = replicate_design(matrix, n_replicates)

    n_runs = matrix.shape[0]

    # 3. Blocks, then randomise: the run order is shuffled within each block, blocks in turn.
    #    Without randomisation the original order is preserved (used for optimal designs
    #    whose run order is part of the solution, e.g. split-plot).
    rng = check_random_state(random_state) if randomize else None
    blocking = None
    if n_blocks is not None and n_blocks > 1:
        if n_leading_fixed:
            raise ValueError("n_blocks cannot be combined with fixed_runs: the fixed runs were already made.")
        blocking = _assign_blocks(matrix, factors, design_type, n_blocks, rng)
    if blocking is not None:
        groups = [np.flatnonzero(blocking.labels == b) for b in range(1, n_blocks + 1)]  # type: ignore[operator]
        perm = np.concatenate([rng.permutation(g) if rng is not None else g for g in groups])
    elif rng is not None:
        # Runs already performed (fixed runs of an augmentation) stay first, in their
        # given order; only the new runs are shuffled.
        n_keep = n_leading_fixed if n_replicates == 1 else 0
        perm = np.concatenate([np.arange(n_keep), n_keep + rng.permutation(n_runs - n_keep)])
    else:
        perm = np.arange(n_runs)
    matrix_randomized = matrix[perm]
    run_order = (perm + 1).tolist()

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
    if blocking is not None:
        block_assignments = blocking.labels[perm].tolist()
        design_coded["Block"] = block_assignments
        design_actual["Block"] = block_assignments
        metadata = dict(metadata or {})
        metadata["blocking"] = {
            "method": blocking.method,
            "generators": blocking.generators,
            "confounded_with": blocking.confounded_with,
            "model": blocking.model,
        }

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
