"""A residual for test data that gives a PLS model a component with no spread (#1026, #1086, #1124).

A component has no spread when its training score standard deviation is at most epsqrt (about
1.5e-8) times the largest (``components_with_spread``). PLS fits a further component only while
what is left of X is above epsqrt times the size of X before the first component (#1070). Both
rules are relative, so the residual must be above the one and below the other: large in total,
but spread so evenly over many directions that none of them is long.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from process_improve.batch.data_input import dict_to_wide
from process_improve.multivariate._common import epsqrt


def thin_residual(signal: np.ndarray, *, relative_size: float = 1.5, seed: int = 0) -> np.ndarray:
    """Return a residual for ``signal``: centred, orthogonal to it, and with all its singular values equal.

    Its rows are orthogonal to the column of ones and to the columns of ``signal``, and its
    columns to the rows of ``signal``, so a model of ``signal + residual`` fits ``signal`` first
    and the residual after. Its size is ``relative_size * epsqrt`` times that of ``signal``,
    above PLS's floor, and its longest direction is that divided by the square root of its
    rank. With ten or more directions to spread over, each is well below the rule for spread.

    Parameters
    ----------
    signal : np.ndarray of shape (n_samples, n_features)
        The data the residual is added to, centred.
    relative_size : float, default=1.5
        The residual's Frobenius norm over the signal's, in units of epsqrt.
    seed : int, default=0
        Seed for the directions the residual takes.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = signal.shape
    rank = np.linalg.matrix_rank(signal)
    taken_rows = np.linalg.qr(np.column_stack([np.ones(n_samples), signal]))[0][:, : rank + 1]
    taken_columns = np.linalg.qr(signal.T)[0][:, :rank]
    directions = min(n_samples - rank - 1, n_features - rank)
    rows = rng.standard_normal((n_samples, directions))
    rows = np.linalg.qr(rows - taken_rows @ (taken_rows.T @ rows))[0]
    columns = rng.standard_normal((n_features, directions))
    columns = np.linalg.qr(columns - taken_columns @ (taken_columns.T @ columns))[0]
    size = relative_size * epsqrt * np.linalg.norm(signal)
    return size / np.sqrt(directions) * rows @ columns.T


def batches_with_a_thin_residual() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return #1120's forty batches of two tags over ten samples, and their quality, with a thin residual.

    The unfolded trajectories hold two patterns, and the quality follows them. The thin residual
    is added to the unfolded rows, so a three-component batch PLS has a third component with no
    spread.
    """
    rng = np.random.default_rng(0)
    time = np.arange(10)
    signal, quality = {}, []
    for i in range(40):
        a, b = rng.standard_normal(2)
        u = a * np.sin(time / 3.0)
        r = b * np.cos(time / 4.0) + 0.5 * a * np.sin(time / 3.0)
        signal[f"b{i}"] = pd.DataFrame({"u": u, "r": r})
        quality.append(2.0 * a + b + 0.3 * rng.standard_normal())
    wide = dict_to_wide(signal)
    wide += thin_residual((wide - wide.mean()).to_numpy())
    batches = {name: pd.DataFrame({tag: wide.loc[name, tag].to_numpy() for tag in ("u", "r")}) for name in signal}
    return batches, pd.DataFrame({"q": quality}, index=list(signal))
