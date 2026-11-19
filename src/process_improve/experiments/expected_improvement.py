# (c) Kevin Dunn, 2010-2026. MIT License.
r"""The next runs of a study from a model of every run so far: expected improvement on a Gaussian process.

After each round the question is which runs come next. Where the response is close to a quadratic
over the region, the polynomial route (:func:`~process_improve.experiments.recommend_next_step`) answers
it with fewer runs and with tests. Where it is not, the usual method (Jones, Schonlau and Welch,
1998) fits a Gaussian process to the runs and proposes the setting where the expected improvement
over the best result so far is largest:

.. math::

    \operatorname{EI}(x) = (b - \hat y(x))\,\Phi(z) + s(x)\,\phi(z), \qquad z = \frac{b - \hat y(x)}{s(x)},

for a minimum sought, with :math:`\hat y` the prediction, :math:`s` its standard error and :math:`b`
the best so far. A setting is proposed because it looks good (a low :math:`\hat y`), or because
little is known there (a large :math:`s`). With a noise term, the best so far is the best predicted
mean at the runs made, not the best single result, which may be a lucky one.

The settings come from the region, :class:`~process_improve.experiments.region.DesignRegion`, so a
proposed run keeps every limit on the settings with no code of its own for them. A batch of runs
to make before any result is read follows the "kriging believer" rule (Ginsbourger, Le Riche and
Carraro, 2010): each run chosen is taken to give its prediction, and the next is chosen with it
among the runs; the model's parameters are not estimated again inside the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from process_improve._random import check_random_state
from process_improve.experiments.factor import FactorType
from process_improve.experiments.gaussian_process import GaussianProcessFit, fit_gaussian_process
from process_improve.experiments.region import DesignRegion

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

#: What a study can aim for.
GOALS = ("maximize", "minimize")

#: The most runs one call proposes, and the fewest settings it draws from the region to choose among.
MAX_RUNS = 50
MIN_CANDIDATES = 100

#: Rounds of the local search around the best setting drawn, the settings each draws, and the first round's
#: half-width in coded units; each round halves it.
_POLISH_ROUNDS = 4
_POLISH_DRAWS = 400
_POLISH_WIDTH = 0.2


@dataclass(frozen=True)
class SuggestedRuns:
    """The runs :func:`suggest_next_runs` proposes, and what they were chosen from.

    Attributes
    ----------
    runs : pandas.DataFrame
        One row per run, in the order to make them: each factor in its own units, the
        ``predicted`` output there, its ``std_error`` and the ``expected_improvement``.
    best : float
        The best so far that each run is to improve on.
    improvement_ratio : float
        The first run's expected improvement over the size of the best so far (over the range
        of the outputs when the best is 0). Jones, Schonlau and Welch stop when it is below 1%:
        then confirmation runs at the best setting are the next step.
    fit : GaussianProcessFit
        The model of every run so far.
    """

    runs: pd.DataFrame
    best: float
    improvement_ratio: float
    fit: GaussianProcessFit = field(repr=False)


def suggest_next_runs(  # noqa: PLR0913 - the study, its goal, where to look and how many runs
    factors: list[Factor],
    runs: pd.DataFrame,
    response: str,
    *,
    goal: str = "maximize",
    region: DesignRegion | None = None,
    n_runs: int = 1,
    noise: bool = True,
    n_candidates: int = 4000,
    random_state: int | np.random.Generator | None = 42,
) -> SuggestedRuns:
    """Propose the next runs: where the expected improvement over the best result so far is largest.

    Parameters
    ----------
    factors : list[Factor]
        The study's numeric factors, with their ranges.
    runs : pandas.DataFrame
        Every run so far: a column per factor, in its own units, and the ``response``.
    response : str
        The output's column.
    goal : str
        ``"maximize"`` or ``"minimize"``.
    region : DesignRegion or None
        Where a run may be made: the factors' box with any limits on the settings. The box
        of the factors by default.
    n_runs : int
        How many runs to propose, 1 to :data:`MAX_RUNS`, to be made before any result is read.
    noise : bool
        Fit the model with a noise term, as a physical experiment's runs need; False for a
        computer experiment, which gives the same output at the same inputs.
    n_candidates : int
        Settings drawn from the region to choose each run among, before a local search around
        the best of them.
    random_state : int, numpy.random.Generator or None
        Seeds the draws and the model's fit.

    Returns
    -------
    SuggestedRuns
        The runs, the best so far, the improvement ratio and the model.

    Raises
    ------
    ValueError
        If a factor is not numeric, the goal is not one of :data:`GOALS`, a factor or the
        response is not a column of ``runs``, ``n_runs`` or ``n_candidates`` is out of range, or
        the model cannot be fitted (see :func:`fit_gaussian_process`).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from process_improve.experiments import Factor, suggest_next_runs
    >>> factors = [Factor(name="x", low=0, high=1)]
    >>> runs = pd.DataFrame({"x": [0.0, 0.3, 0.6, 1.0]}).assign(y=lambda d: np.sin(3 * d["x"]))
    >>> found = suggest_next_runs(factors, runs, "y", noise=False)
    >>> 0.3 < found.runs["x"].iloc[0] < 0.7  # the maximum of sin(3x) is at 0.52
    True
    """
    _check(factors, runs, response, goal, n_runs, n_candidates)
    names = [f.name for f in factors]
    rng = check_random_state(random_state)
    seed = int(rng.integers(2**31))
    region = region or DesignRegion(factors)
    fit = fit_gaussian_process(runs[[*names, response]], response, names, noise=noise, random_state=seed)
    sign = 1.0 if goal == "minimize" else -1.0  # the expected improvement is worked for a minimum
    made = runs[names]
    at_runs = fit.predict(made)["predicted"] if noise else runs[response]
    best = float(sign * np.min(sign * at_runs.to_numpy(dtype=float)))
    scale = abs(best) or float(np.ptp(runs[response].to_numpy(dtype=float)))
    chosen, model, target = [], fit, sign * best
    for _ in range(n_runs):
        setting, gain = _best_setting(model, factors, region, target, sign, n_candidates, rng)
        predicted = model.predict(setting)
        chosen.append(setting.assign(**predicted[["predicted", "std_error"]], expected_improvement=gain))
        # The kriging believer: the run is taken to give its prediction, which joins the runs and may be the new
        # best, and the next run is chosen with it; so no two runs of a batch go to one place.
        model = model.with_runs(setting, predicted["predicted"].to_numpy())
        target = min(target, sign * float(predicted["predicted"].iloc[0]))
    proposed = pd.concat(chosen, ignore_index=True)
    ratio = float(proposed["expected_improvement"].iloc[0] / scale) if scale > 0 else 0.0
    return SuggestedRuns(proposed, best, ratio, fit)


def _check(factors: list[Factor], runs: pd.DataFrame, response: str, goal: str, n_runs: int, n_candidates: int) -> None:
    """Refuse what :func:`suggest_next_runs` cannot work with, each in a sentence."""
    if any(f.type != FactorType.continuous for f in factors) or not factors:
        msg = "suggest_next_runs takes one or more numeric factors; a categorical factor or a blend is not taken yet."
        raise ValueError(msg)
    if goal not in GOALS:
        msg = f"goal={goal!r} is not one of {list(GOALS)}."
        raise ValueError(msg)
    missing = [name for name in [*(f.name for f in factors), response] if name not in runs.columns]
    if missing:
        msg = f"The runs have no column {missing[0]!r}: give each factor's settings and the response."
        raise ValueError(msg)
    if not (isinstance(n_runs, int) and 1 <= n_runs <= MAX_RUNS):
        msg = f"n_runs={n_runs!r}: propose 1 to {MAX_RUNS} runs."
        raise ValueError(msg)
    if not (isinstance(n_candidates, int) and n_candidates >= MIN_CANDIDATES):
        msg = f"n_candidates={n_candidates!r}: draw at least {MIN_CANDIDATES} settings to choose among."
        raise ValueError(msg)


def _expected_improvement(fit: GaussianProcessFit, points: pd.DataFrame, best: float, sign: float) -> np.ndarray:
    """Return the expected amount by which a run at each point improves on ``best``, on the scale of a minimum."""
    found = fit.predict(points)
    mean, spread = sign * found["predicted"].to_numpy(), found["std_error"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(spread > 0, (best - mean) / spread, 0.0)
    gain = (best - mean) * stats.norm.cdf(z) + spread * stats.norm.pdf(z)
    return np.where(spread > 0, np.clip(gain, 0.0, None), 0.0)


def _actual(coded: np.ndarray, factors: list[Factor]) -> pd.DataFrame:
    """Return coded settings in the factors' own units."""
    return pd.DataFrame(
        {f.name: f.low + (coded[:, j] + 1.0) / 2.0 * (f.high - f.low) for j, f in enumerate(factors)}  # type: ignore[operator]
    )


def _best_setting(
    fit: GaussianProcessFit,
    factors: list[Factor],
    region: DesignRegion,
    best: float,
    sign: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """Return the setting of largest expected improvement, as one row in the factors' units, and that improvement.

    Settings are drawn from the region, and the best is then searched around: each round draws settings in a
    box about it, half as wide as the round before, and keeps those the region allows.
    """
    coded = region.sample(n_candidates, rng)
    gains = _expected_improvement(fit, _actual(coded, factors), best, sign)
    at, gain = coded[int(np.argmax(gains))], float(np.max(gains))
    width = _POLISH_WIDTH
    for _ in range(_POLISH_ROUNDS):
        near = np.clip(at + rng.uniform(-width, width, (_POLISH_DRAWS, len(at))), -1.0, 1.0)
        near = near[region.feasible(near)]
        if len(near):
            gains = _expected_improvement(fit, _actual(near, factors), best, sign)
            if gains.max() > gain:
                at, gain = near[int(np.argmax(gains))], float(gains.max())
        width /= 2.0
    return _actual(at[None, :], factors), gain
