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
little is known there (a large :math:`s`). For a target :math:`t` the improvement is in the distance
from it, :math:`d` being the smallest so far: :math:`\operatorname{E}[\max(d - |Y - t|, 0)]` for
:math:`Y` normal with that mean and standard error, worked exactly over the band from :math:`t - d`
to :math:`t + d`. With a noise term, the best so far is the best predicted
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

#: What a study can aim for: a target takes its value as ``target``.
GOALS = ("maximize", "minimize", "target")

#: The most runs one call proposes, and the settings it draws from the region to choose each among.
MAX_RUNS = 50
N_CANDIDATES = 4000

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
        The best so far that each run is to improve on: for a target, the output closest to it.
    improvement_ratio : float
        The first run's expected improvement over the size of the best so far (over the range
        of the outputs when the best is 0); for a target, over the best's distance from it.
        Jones, Schonlau and Welch stop when it is below 1%: then confirmation runs at the best
        setting are the next step.
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
    target: float | None = None,
    region: DesignRegion | None = None,
    n_runs: int = 1,
    noise: bool = True,
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
        ``"maximize"``, ``"minimize"`` or ``"target"``.
    target : float or None
        The value to reach, for ``goal="target"`` only.
    region : DesignRegion or None
        Where a run may be made: the factors' box with any limits on the settings. The box
        of the factors by default.
    n_runs : int
        How many runs to propose, 1 to :data:`MAX_RUNS`, to be made before any result is read.
    noise : bool
        Fit the model with a noise term, as a physical experiment's runs need; False for a
        computer experiment, which gives the same output at the same inputs.
    random_state : int, numpy.random.Generator or None
        Seeds the draws and the model's fit.

    Returns
    -------
    SuggestedRuns
        The runs, the best so far, the improvement ratio and the model.

    Raises
    ------
    ValueError
        If a factor is not numeric, the goal is not one of :data:`GOALS` or a target goal has no
        finite ``target`` (or another goal has one), a factor or the
        response is not a column of ``runs``, ``n_runs`` is out of range, or the model cannot be
        fitted (see :func:`fit_gaussian_process`).

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
    _check(factors, runs, response, goal, n_runs, target)
    names = [f.name for f in factors]
    rng = check_random_state(random_state)
    seed = int(rng.integers(2**31))
    region = region or DesignRegion(factors)
    fit = fit_gaussian_process(runs[[*names, response]], response, names, noise=noise, random_state=seed)
    aim = _Aim(goal, target)
    made = runs[names]
    at_runs = (fit.predict(made)["predicted"] if noise else runs[response]).to_numpy(dtype=float)
    best = float(at_runs[int(np.argmin(aim.loss(at_runs)))])
    scale = float(aim.loss(np.array([best]))[0]) if goal == "target" else abs(best)
    scale = scale or float(np.ptp(runs[response].to_numpy(dtype=float)))
    chosen, model, least = [], fit, float(aim.loss(np.array([best]))[0])
    for _ in range(n_runs):
        setting, gain = _best_setting(model, factors, region, least, aim, rng)
        predicted = model.predict(setting)
        chosen.append(setting.assign(**predicted[["predicted", "std_error"]], expected_improvement=gain))
        # The kriging believer: the run is taken to give its prediction, which joins the runs and may be the new
        # best, and the next run is chosen with it; so no two runs of a batch go to one place.
        model = model.with_runs(setting, predicted["predicted"].to_numpy())
        least = min(least, float(aim.loss(predicted["predicted"].to_numpy(dtype=float))[0]))
    proposed = pd.concat(chosen, ignore_index=True)
    ratio = float(proposed["expected_improvement"].iloc[0] / scale) if scale > 0 else 0.0
    return SuggestedRuns(proposed, best, ratio, fit)


def _check(
    factors: list[Factor], runs: pd.DataFrame, response: str, goal: str, n_runs: int, target: float | None
) -> None:
    """Refuse what :func:`suggest_next_runs` cannot work with, each in a sentence."""
    if any(f.type != FactorType.continuous for f in factors) or not factors:
        msg = "suggest_next_runs takes one or more numeric factors; a categorical factor or a blend is not taken yet."
        raise ValueError(msg)
    if goal not in GOALS:
        msg = f"goal={goal!r} is not one of {list(GOALS)}."
        raise ValueError(msg)
    if (goal == "target") != (target is not None) or (target is not None and not np.isfinite(target)):
        msg = f"goal={goal!r} with target={target!r}: give a finite target with goal='target', and none otherwise."
        raise ValueError(msg)
    missing = [name for name in [*(f.name for f in factors), response] if name not in runs.columns]
    if missing:
        msg = f"The runs have no column {missing[0]!r}: give each factor's settings and the response."
        raise ValueError(msg)
    if not (isinstance(n_runs, int) and 1 <= n_runs <= MAX_RUNS):
        msg = f"n_runs={n_runs!r}: propose 1 to {MAX_RUNS} runs."
        raise ValueError(msg)


@dataclass(frozen=True)
class _Aim:
    """What a run is judged by: its loss, the smaller the better, and the expected improvement in it."""

    goal: str
    target: float | None = None

    def loss(self, outputs: np.ndarray) -> np.ndarray:
        """Return each output's loss: itself for a minimum, its negative for a maximum, its distance from a target."""
        if self.goal == "target":
            return np.abs(outputs - self._at)
        return outputs if self.goal == "minimize" else -outputs

    @property
    def _at(self) -> float:
        """The target, which _check has made sure a target goal has."""
        return 0.0 if self.target is None else self.target

    def improvement(self, mean: np.ndarray, spread: np.ndarray, least: float) -> np.ndarray:
        """Return the expected amount by which an output, normal with ``mean`` and ``spread``, lowers ``least``."""
        cdf, pdf = stats.norm.cdf, stats.norm.pdf
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.goal != "target":
                z = (least - self.loss(mean)) / spread
                gain = (least - self.loss(mean)) * cdf(z) + spread * pdf(z)
            else:
                # E[max(least - |Y - t|, 0)]: the band from t - least to t + least, each half a linear integral
                t = self._at
                low, aim, high = (((at - mean) / spread) for at in (t - least, t, t + least))
                gain = (mean - t + least) * (cdf(aim) - cdf(low)) + spread * (pdf(low) - pdf(aim))
                gain += (t + least - mean) * (cdf(high) - cdf(aim)) - spread * (pdf(aim) - pdf(high))
        return np.where(spread > 0, np.clip(np.nan_to_num(gain), 0.0, None), 0.0)


def _expected_improvement(fit: GaussianProcessFit, points: pd.DataFrame, least: float, aim: _Aim) -> np.ndarray:
    """Return the expected amount by which a run at each point lowers ``least``, the smallest loss so far."""
    found = fit.predict(points)
    return aim.improvement(found["predicted"].to_numpy(), found["std_error"].to_numpy(), least)


def _actual(coded: np.ndarray, factors: list[Factor]) -> pd.DataFrame:
    """Return coded settings in the factors' own units."""
    return pd.DataFrame(
        {f.name: f.low + (coded[:, j] + 1.0) / 2.0 * (f.high - f.low) for j, f in enumerate(factors)}  # type: ignore[operator]
    )


def _best_setting(
    fit: GaussianProcessFit,
    factors: list[Factor],
    region: DesignRegion,
    least: float,
    aim: _Aim,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    """Return the setting of largest expected improvement, as one row in the factors' units, and that improvement.

    Settings are drawn from the region, and the best is then searched around: each round draws settings in a
    box about it, half as wide as the round before, and keeps those the region allows.
    """
    coded = region.sample(N_CANDIDATES, rng)
    gains = _expected_improvement(fit, _actual(coded, factors), least, aim)
    at, gain = coded[int(np.argmax(gains))], float(np.max(gains))
    width = _POLISH_WIDTH
    for _ in range(_POLISH_ROUNDS):
        near = np.clip(at + rng.uniform(-width, width, (_POLISH_DRAWS, len(at))), -1.0, 1.0)
        near = near[region.feasible(near)]
        if len(near):
            gains = _expected_improvement(fit, _actual(near, factors), least, aim)
            if gains.max() > gain:
                at, gain = near[int(np.argmax(gains))], float(gains.max())
        width /= 2.0
    return _actual(at[None, :], factors), gain
