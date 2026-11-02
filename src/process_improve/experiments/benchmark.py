# (c) Kevin Dunn, 2010-2026. MIT License.

"""Whole studies on known processes, to see whether following the next step reaches the optimum, and at what cost.

:func:`~process_improve.experiments.simulate_design_performance` judges one design. A study
is a sequence of designs, each chosen from the results of the last, and its quality is where it
ends and how many runs it took to get there. :func:`run_study` follows
:func:`~process_improve.experiments.recommend_next_step` round by round on a process whose
truth is known, making each round's runs on that truth, until the recommendation stops the
study; :func:`run_benchmark` does so for a set of :class:`BenchmarkCase` over many noise
seeds, and summarises each case by the runs used and how far the final settings fall short of
the true best.

The cases in :data:`CASES` include the adhesion problem of the "AI agents in DoE" study (its
true model as the study published it), and hypothesised processes for its Problems 02
(pendulum hardness of a polyurethane clearcoat) and 03 (reverse impact of a coil coating),
written from the chemistry of those coatings: they are not the study's hidden models.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from process_improve._random import check_random_state
from process_improve.experiments.designs import generate_design
from process_improve.experiments.factor import Factor
from process_improve.experiments.next_step import recommend_next_step
from process_improve.experiments.simulate import _loss, _true_best
from process_improve.experiments.true_process import TrueProcess

__all__ = ["CASES", "BenchmarkCase", "run_benchmark", "run_study"]

Goal = Literal["maximize", "minimize", "target"]

#: The most rounds a study may take before the benchmark ends it.
_MAX_ROUNDS = 12


@dataclass(frozen=True)
class BenchmarkCase:
    """A known process, where a study of it starts, and what the study is for.

    Parameters
    ----------
    name : str
        A short label for the reports.
    factors : tuple of Factor
        The factors with the ranges the first design spans, in actual units.
    truth : TrueProcess
        The process, in coded units of those first ranges: -1 at each factor's first low value
        and +1 at its first high value. A study that moves or widens its ranges is still
        judged on this one process.
    first_design : mapping
        Keyword arguments to :func:`generate_design` for the first round.
    goal : {"maximize", "minimize", "target"}, default "maximize"
    target : float or None
        The response wanted, for ``goal="target"``.
    ranges_can_move : bool, default False
        Whether the study may step past the first ranges (as ``recommend_next_step`` takes it).
        The true best is then sought over ``span`` times the first ranges about their middles.
    span : float, default 1.0
        How far the process can go, in coded units of the first ranges, when the ranges may move.
    max_runs : int, default 150
        The runs a study may spend; the recommendation is told what is left of it each round.
    note : str
        Where the case comes from.
    """

    name: str
    factors: tuple[Factor, ...]
    truth: TrueProcess
    first_design: Mapping[str, Any]
    goal: Goal = "maximize"
    target: float | None = None
    ranges_can_move: bool = False
    span: float = 1.0
    max_runs: int = 150
    note: str = ""

    def coded(self, actual: pd.DataFrame) -> pd.DataFrame:
        """Return settings in actual units, coded on the first ranges, the scale the truth is written on."""
        out = actual[[f.name for f in self.factors]].astype(float).copy()
        for f in self.factors:
            mid, half = (f.high + f.low) / 2, (f.high - f.low) / 2  # type: ignore[operator]
            out[f.name] = (out[f.name] - mid) / half
        return out


@dataclass
class _Round:
    """The state of a study between rounds."""

    factors: list[Factor]
    runs: pd.DataFrame
    used: int
    history: list[str] = field(default_factory=list)
    confirmation: pd.DataFrame | None = None


def run_study(case: BenchmarkCase, *, random_state: int | np.random.Generator | None = 0) -> dict[str, Any]:
    """Follow ``recommend_next_step`` on ``case`` from its first design until the study stops.

    Every round's runs are made on the true process with fresh noise. Confirmation runs are
    passed as confirmation runs, and join the other runs after a miss; after ``widen_ranges``
    the study starts again on the new ranges; after ``move_region`` the runs along the path are
    made in order up to the first that does worse than the one before, and a new first design
    is centred on the best of them, with ranges as wide as before. The study ends at ``stop``
    or ``investigate``, when the runs are spent, or after twelve rounds.

    Parameters
    ----------
    case : BenchmarkCase
    random_state : int, numpy.random.Generator or None, default 0
        Seed for the noise and for the choices the recommendation makes at random.

    Returns
    -------
    dict
        ``actions`` (each round's action), ``runs_used``, ``ended`` (the last action),
        ``settings`` (the final best settings, actual units), ``true_value`` (the true
        response there), ``true_best``, ``shortfall`` and ``shortfall_in_sigma``.
    """
    rng = check_random_state(random_state)
    seed = int(rng.integers(2**31))
    names = [f.name for f in case.factors]
    state = _Round(list(case.factors), _first(case, list(case.factors), rng), 0)
    state.used = len(state.runs)
    step: dict[str, Any] = {}
    for _ in range(_MAX_ROUNDS):
        step = recommend_next_step(
            state.factors,
            state.runs,
            "y",
            goal=case.goal,
            target=case.target,
            confirmation_runs=state.confirmation,
            ranges_can_move=case.ranges_can_move,
            budget=case.max_runs - state.used,
            random_state=seed,
            history=state.history,
        )
        state.history.append(step["action"])
        if step["action"] in ("stop", "investigate") or not step["runs"]:
            break
        _make(case, state, step, rng)
    settings = pd.DataFrame([step["evidence"]["optimum"]["settings"]])[names].astype(float)
    true_value = float(case.truth.mean(case.coded(settings))[0])
    best = _true_best(case.truth, names, case.goal, case.target, span=case.span if case.ranges_can_move else 1.0)
    short = max(float(_loss(np.array([true_value]), case.goal, case.target)[0]) - best["loss"], 0.0)
    return {
        "actions": state.history,
        "runs_used": state.used,
        "ended": state.history[-1],
        "settings": settings.iloc[0].to_dict(),
        "true_value": true_value,
        "true_best": best["response"],
        "shortfall": short,
        "shortfall_in_sigma": short / case.truth.sigma,
    }


def _first(case: BenchmarkCase, factors: list[Factor], rng: np.random.Generator) -> pd.DataFrame:
    """Make the first design over ``factors`` on the truth."""
    names = [f.name for f in factors]
    runs = generate_design(factors, **case.first_design).design_actual[names].astype(float).reset_index(drop=True)
    return _measured(case, runs, rng)


def _measured(case: BenchmarkCase, runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return the runs with the true process's response, ``y``, made with fresh noise."""
    return runs.assign(y=case.truth.simulate(case.coded(runs), random_state=rng).to_numpy())


def _make(case: BenchmarkCase, state: _Round, step: dict[str, Any], rng: np.random.Generator) -> None:
    """Make a step's runs and carry the study into its next round."""
    names = [f.name for f in state.factors]
    new = _measured(case, pd.DataFrame(step["runs"])[names].astype(float), rng)
    state.used += len(new)
    if state.confirmation is not None:  # the confirmation missed: its runs are data now
        state.runs, state.confirmation = pd.concat([state.runs, state.confirmation], ignore_index=True), None
    if step["action"] == "confirm":
        state.confirmation = new
    elif step["action"] == "widen_ranges":
        state.factors = [Factor(name=n, low=low, high=high) for n, (low, high) in step["new_ranges"].items()]
        state.runs = new
    elif step["action"] == "move_region":
        state.factors = _centred(state.factors, _best_on_path(new, case.goal, case.target))
        state.runs = _first(case, state.factors, rng)
        state.used += len(state.runs)
    else:
        state.runs = pd.concat([state.runs, new], ignore_index=True)


def _best_on_path(path: pd.DataFrame, goal: Goal, target: float | None) -> pd.Series:
    """Return the best run along the path, up to the first that does worse than the one before."""
    y = path["y"].to_numpy(dtype=float)
    worth = -np.abs(y - float(target)) if goal == "target" else (y if goal == "maximize" else -y)  # type: ignore[arg-type]
    stop = next((i for i in range(1, len(worth)) if worth[i] < worth[i - 1]), len(worth))
    return path.iloc[int(np.argmax(worth[:stop]))]


def _centred(factors: Sequence[Factor], best: pd.Series) -> list[Factor]:
    """Return the factors with ranges as wide as before, centred on the best run."""
    return [
        Factor(name=f.name, low=best[f.name] - (f.high - f.low) / 2, high=best[f.name] + (f.high - f.low) / 2)  # type: ignore[operator]
        for f in factors
    ]


def run_benchmark(
    cases: Sequence[BenchmarkCase] | None = None, *, n_seeds: int = 20, random_state: int = 0
) -> pd.DataFrame:
    """Run every case ``n_seeds`` times and summarise each.

    Returns
    -------
    pandas.DataFrame
        One row per case: ``case``, ``shortfall [sd]`` (mean, in noise standard deviations),
        ``worst [sd]``, ``runs`` (mean), ``ends`` (how the studies ended) and ``commonest path``.
    """
    rows = []
    for i, case in enumerate(CASES if cases is None else cases):
        studies = [run_study(case, random_state=random_state + 1000 * i + s) for s in range(n_seeds)]
        paths = collections.Counter(" > ".join(s["actions"]) for s in studies)
        rows.append(
            {
                "case": case.name,
                "shortfall [sd]": float(np.mean([s["shortfall_in_sigma"] for s in studies])),
                "worst [sd]": float(np.max([s["shortfall_in_sigma"] for s in studies])),
                "runs": float(np.mean([s["runs_used"] for s in studies])),
                "ends": dict(collections.Counter(s["ended"] for s in studies)),
                "commonest path": paths.most_common(1)[0][0],
            }
        )
    return pd.DataFrame(rows)


def _unit(names: str) -> tuple[Factor, ...]:
    return tuple(Factor(name=n, low=-1, high=1) for n in names)


def _x(ranges: Sequence[tuple[float, float]]) -> tuple[Factor, ...]:
    return tuple(Factor(name=f"X{i:02d}", low=low, high=high) for i, (low, high) in enumerate(ranges, start=1))


#: Problem 02 of the "AI agents in DoE" study: pendulum hardness [s] of a 2K polyurethane clearcoat.
_HARDNESS_RANGES = [
    (0.9, 1.3),
    (140, 180),
    (0, 400),
    (0.1, 1.0),
    (0, 4),
    (10, 25),
    (5, 20),
    (60, 100),
    (20, 60),
    (30, 60),
    (30, 70),
    (18, 30),
]
#: Problem 03: reverse impact [in-lb] to first crack of a coil coating.
_IMPACT_RANGES = [(0, 40), (0, 8), (0, 5), (0, 1.5), (1, 10), (20, 40), (0.5, 1.5), (50, 90), (216, 241), (25, 45)]

CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "adhesion, DSD",
        _unit("ABCDEFGH"),
        TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3),
        {"design_type": "dsd", "n_center_points": 5},
        note="The adhesion problem of the 'AI agents in DoE' study, with the true model it published.",
    ),
    BenchmarkCase(
        "adhesion, Plackett-Burman",
        _unit("ABCDEFGH"),
        TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3),
        {"design_type": "plackett_burman", "n_center_points": 3},
    ),
    BenchmarkCase(
        "hardness (02), hypothesised",
        _x(_HARDNESS_RANGES),
        TrueProcess(
            {
                "Intercept": 120,
                "X08": 15,
                "I(X08 ** 2)": -5,
                "X09": 10,
                "X08:X09": -6,
                "X01": 12,
                "I(X01 ** 2)": -6,
                "X03": 6,
                "X10": -5,
                "X06": -4,
            },
            sigma=5,
        ),
        {"design_type": "dsd", "n_center_points": 5},
        note="Hypothesised from polyurethane chemistry; not the study's hidden model.",
    ),
    BenchmarkCase(
        "impact (03), hypothesised",
        _x(_IMPACT_RANGES),
        TrueProcess(
            {
                "Intercept": 80,
                "X01": 18,
                "I(X01 ** 2)": -4,
                "X02": 12,
                "X01:X02": -5,
                "X06": -15,
                "I(X06 ** 2)": -5,
                "X08": -10,
                "X09": 6,
                "I(X09 ** 2)": -10,
                "X10": 3,
                "X09:X10": -4,
                "X04": 5,
            },
            sigma=8,
        ),
        {"design_type": "dsd", "n_center_points": 5},
        note="Hypothesised from coil-coating practice; not the study's hidden model.",
    ),
    BenchmarkCase(
        "interior optimum, 2^3 + centre",
        _unit("ABC"),
        TrueProcess({"Intercept": 60, "A": 2, "B": -1.5, "I(A ** 2)": -6, "I(B ** 2)": -4, "A:B": 1}, sigma=0.7),
        {"design_type": "full_factorial", "n_center_points": 3},
    ),
    BenchmarkCase(
        "curvature in 8 factors, 2^(8-4) + centre",
        _unit("ABCDEFGH"),
        TrueProcess({"Intercept": 100, "A": 2, "I(A ** 2)": -6, "I(C ** 2)": -4}, sigma=1),
        {"design_type": "fractional_factorial", "resolution": 4, "n_center_points": 3},
    ),
    BenchmarkCase(
        "optimum outside the ranges",
        _unit("ABC"),
        TrueProcess({"Intercept": 80, "A": 12, "B": -9, "I(A ** 2)": -3, "I(B ** 2)": -3}, sigma=1),
        {"design_type": "full_factorial", "n_center_points": 3},
        ranges_can_move=True,
        span=4.0,
    ),
    BenchmarkCase(
        "noise only",
        _unit("ABCD"),
        TrueProcess({"Intercept": 20}, sigma=2),
        {"design_type": "full_factorial", "n_center_points": 3},
        ranges_can_move=True,
    ),
)
