"""Prediction from marginal means: Taguchi's additive model."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _pairwise_balanced(settings: pd.DataFrame) -> bool:
    """Whether every factor's levels, and every pair of factors' level combinations, occur equally often."""
    for name in settings.columns:
        counts = settings[name].value_counts().to_numpy()
        if counts.min() != counts.max():
            return False
    for first, second in itertools.combinations(settings.columns, 2):
        cells = pd.crosstab(settings[first], settings[second]).to_numpy()
        if cells.min() != cells.max():  # a combination never run counts 0
            return False
    return True


def marginal_means_prediction(
    design_matrix: pd.DataFrame,
    responses: pd.Series | Sequence[float],
    at: pd.DataFrame | Mapping[str, Any],
    interactions: Sequence[tuple[str, str]] = (),
) -> pd.Series:
    """Predict a response from the marginal means of the factors' levels, as Taguchi's additive model does.

    The prediction at a setting is the grand mean ``T`` plus, for each factor, the mean of the runs
    at its level less ``T``; each pair named in *interactions* adds the mean of the runs at its two
    levels, less the two factors' means, plus ``T``. For one 2-level factor A and an interaction
    with B this is ``T + (A - T) + (B - T) + (AB - A - B + T)``.

    When every factor's levels, and every pair of factors' level combinations, occur equally often
    (as in an orthogonal array), the main-effect terms are the least-squares estimates of a model of
    main effects. Otherwise each marginal mean also carries the effects of the other factors' levels
    it happens to meet, and the prediction is not the least-squares one; a ``UserWarning`` says so.

    Parameters
    ----------
    design_matrix : pd.DataFrame
        The runs, with a column for each factor in *at*. Levels may be numbers or labels; a
        level in *at* is matched to the runs by equality.
    responses : pd.Series or sequence of float
        One response per run, in the order of *design_matrix*. A run whose response is missing
        is left out of every mean.
    at : pd.DataFrame or mapping
        The settings to predict: one row per setting (or one mapping), with a level for each
        factor in the model. The model's factors are the columns of *at*.
    interactions : sequence of (str, str), optional
        Pairs of factors whose two-factor means enter the prediction.

    Returns
    -------
    pd.Series
        The prediction at each setting, indexed as *at* (from 0 for a mapping), named
        ``"predicted"``.

    Raises
    ------
    ValueError
        If *responses* and *design_matrix* differ in length, a factor of *at* or of
        *interactions* is not a column of *design_matrix* (or not in *at*), or no run has a level
        or pair of levels a setting asks for.

    Examples
    --------
    The grand mean is 14, A's level 2 averages 17 and B's 16, so the additive model gives 19; the
    interaction adds the cell's 20 less 17 and 16 plus 14, which returns the cell's own mean.

    >>> import pandas as pd
    >>> runs, y = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 1, 2]}), [10.0, 12.0, 14.0, 20.0]
    >>> marginal_means_prediction(runs, y, {"A": 2, "B": 2}).tolist()
    [19.0]
    >>> marginal_means_prediction(runs, y, {"A": 2, "B": 2}, interactions=[("A", "B")]).tolist()
    [20.0]
    """
    points = pd.DataFrame([dict(at)]) if isinstance(at, Mapping) else at
    y = np.asarray(responses, dtype=float)
    if len(y) != len(design_matrix):
        raise ValueError(f"responses has {len(y)} values; design_matrix has {len(design_matrix)} runs.")
    factors = list(points.columns)
    missing = [name for name in factors if name not in design_matrix.columns]
    if missing:
        columns = list(design_matrix.columns)
        raise ValueError(f"The factors {missing} of `at` are not columns of design_matrix: {columns}.")
    for pair in interactions:
        if len(pair) != 2 or any(name not in factors for name in pair):
            raise ValueError(f"Each interaction must be a pair of the factors in `at` {factors}; got {pair!r}.")
    observed = ~np.isnan(y)
    settings = design_matrix.loc[observed, factors].reset_index(drop=True)
    y = y[observed]
    grand = float(y.mean())

    def mean_at(levels: dict[str, Any]) -> float:
        mask = np.logical_and.reduce([settings[name].to_numpy() == level for name, level in levels.items()])
        if not mask.any():
            raise ValueError(f"No run has {levels}; the marginal mean there is not defined.")
        return float(y[mask].mean())

    predicted = []
    for _, point in points.iterrows():
        main = {name: mean_at({name: point[name]}) for name in factors}
        value = grand + sum(m - grand for m in main.values())
        for first, second in interactions:
            value += mean_at({first: point[first], second: point[second]}) - main[first] - main[second] + grand
        predicted.append(value)
    if not _pairwise_balanced(settings):
        warnings.warn(
            f"The runs are not balanced in the factors {factors}: their levels, or pairs of levels, occur "
            "unequally often, so each marginal mean carries other factors' effects and the prediction is not "
            "the least-squares one.",
            category=UserWarning,
            stacklevel=2,
        )
    return pd.Series(predicted, index=points.index, name="predicted")
