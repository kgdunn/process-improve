"""suggest_next_runs: the next runs from a Gaussian process of every run so far, by expected improvement (#1374).

The test surface is Branin's function, whose minimum is 0.398 at three places, inside the limit x1 + x2 <= 18. It is
not a quadratic, which is the case this method is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, fit_gaussian_process, suggest_next_runs
from process_improve.experiments import expected_improvement as ei
from process_improve.experiments.region import DesignRegion
from process_improve.tool_spec import execute_tool_call

FACTORS = [Factor(name="x1", low=-5.0, high=10.0), Factor(name="x2", low=0.0, high=15.0)]
REGION = DesignRegion(FACTORS, ["x1 + x2 <= 18"])


def branin(x1: pd.Series, x2: pd.Series) -> pd.Series:
    """Return Branin's function: three equal minima of 0.398."""
    return (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


def first_runs(n: int, seed: int) -> pd.DataFrame:
    """Return ``n`` runs drawn from the region, with Branin's output."""
    coded = REGION.sample(n, np.random.default_rng(seed))
    runs = pd.DataFrame({f.name: f.low + (coded[:, j] + 1) / 2 * (f.high - f.low) for j, f in enumerate(FACTORS)})
    return runs.assign(y=branin(runs["x1"], runs["x2"]))


def inside(runs: pd.DataFrame) -> bool:
    """Say whether every run keeps the limit and the factors' ranges."""
    ranged = all(runs[f.name].between(f.low - 1e-9, f.high + 1e-9).all() for f in FACTORS)
    return ranged and bool((runs["x1"] + runs["x2"] <= 18 + 1e-9).all())


@pytest.mark.slow
def test_fifteen_runs_one_at_a_time_find_the_minimum_inside_the_limit():
    runs = first_runs(10, seed=1)
    for i in range(15):
        found = suggest_next_runs(FACTORS, runs, "y", goal="minimize", region=REGION, noise=False, random_state=i)
        pick = found.runs[["x1", "x2"]]
        runs = pd.concat([runs, pick.assign(y=branin(pick["x1"], pick["x2"]))], ignore_index=True)
    assert runs["y"].min() == pytest.approx(0.398, abs=0.05)
    assert inside(runs)


def test_a_batch_s_runs_differ_from_each_other_and_from_the_runs_made_and_keep_the_limit():
    runs = first_runs(12, seed=5)
    found = suggest_next_runs(FACTORS, runs, "y", goal="minimize", region=REGION, n_runs=4, noise=False)
    batch = found.runs
    assert list(batch.columns) == ["x1", "x2", "predicted", "std_error", "expected_improvement"]
    assert len(batch) == 4
    assert inside(batch)
    every = pd.concat([runs[["x1", "x2"]], batch[["x1", "x2"]]]).to_numpy()
    gaps = np.linalg.norm(every[:, None, :] - every[None, :, :], axis=2) + np.eye(len(every)) * 1e9
    assert gaps[len(runs) :].min() > 0.05, "no run of the batch is on top of another run"
    assert (batch["expected_improvement"] >= 0).all()


def test_the_same_seed_gives_the_same_runs():
    runs = first_runs(12, seed=2)
    one, two = (suggest_next_runs(FACTORS, runs, "y", n_runs=2, random_state=7) for _ in range(2))
    pd.testing.assert_frame_equal(one.runs, two.runs)


def test_the_best_so_far_with_noise_is_the_best_predicted_mean_at_the_runs_made():
    runs = first_runs(12, seed=3)
    lucky = runs.assign(y=runs["y"] + np.random.default_rng(3).normal(0.0, 5.0, len(runs)))
    found = suggest_next_runs(FACTORS, pd.concat([lucky, lucky.iloc[:4]]), "y", goal="minimize", random_state=1)
    means = found.fit.predict(lucky[["x1", "x2"]])["predicted"]
    assert found.best == pytest.approx(means.min())
    assert found.improvement_ratio == pytest.approx(found.runs["expected_improvement"].iloc[0] / abs(found.best))


def test_a_maximum_is_sought_by_default():
    factors = [Factor(name="x", low=0.0, high=1.0)]
    runs = pd.DataFrame({"x": [0.0, 0.3, 0.6, 1.0]}).assign(y=lambda d: np.sin(3 * d["x"]))
    up = suggest_next_runs(factors, runs, "y", noise=False).runs["x"].iloc[0]
    down = suggest_next_runs(factors, runs, "y", goal="minimize", noise=False).runs["x"].iloc[0]
    assert 0.3 < up < 0.7, "sin(3x) is largest at 0.52"
    assert down < 0.2, "and smallest at the foot of the range, sin(0) = 0"


def test_a_believed_run_joins_the_model_with_its_parameters_kept():
    runs = first_runs(10, seed=4)
    fit = fit_gaussian_process(runs, "y", ["x1", "x2"])
    point = pd.DataFrame({"x1": [2.0], "x2": [3.0]})
    grown = fit.with_runs(point, np.array([1.5]))
    np.testing.assert_array_equal(grown.theta, fit.theta)
    assert len(grown.runs) == len(runs) + 1
    assert grown.predict(point)["predicted"].iloc[0] == pytest.approx(1.5, abs=1e-6), "it passes through the run"
    np.testing.assert_allclose(grown.predict(runs[["x1", "x2"]])["predicted"], runs["y"], atol=1e-6)


def test_where_the_region_allows_no_setting_near_the_best_the_first_draw_s_best_stands(monkeypatch):
    """Each round searches near the best among the settings the region allows; where it allows none, nothing moves."""
    runs = first_runs(10, seed=5)
    fit = fit_gaussian_process(runs, "y", ["x1", "x2"])
    region = DesignRegion(FACTORS, ["x1 + x2 <= 18"])
    drawn = region.sample(64, np.random.default_rng(0))
    monkeypatch.setattr(region, "sample", lambda n, rng: drawn)  # noqa: ARG005 - the region's own signature
    monkeypatch.setattr(region, "feasible", lambda coded: np.zeros(len(coded), dtype=bool))
    best = float(runs["y"].min())
    setting, gain = ei._best_setting(fit, FACTORS, region, best, 1.0, np.random.default_rng(1))
    gains = ei._expected_improvement(fit, ei._actual(drawn, FACTORS), best, 1.0)
    assert gain == pytest.approx(gains.max())
    pd.testing.assert_frame_equal(setting, ei._actual(drawn[[int(np.argmax(gains))]], FACTORS))


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"factors": [Factor(name="c", type="categorical", levels=["a", "b"])]}, "takes one or more numeric factors"),
        ({"goal": "target"}, "goal='target' is not one of"),
        ({"response": "z"}, "The runs have no column 'z'"),
        ({"n_runs": 0}, "propose 1 to 50 runs"),
    ],
)
def test_what_it_cannot_work_with_is_refused_in_a_sentence(change, refusal):
    asked = {"factors": FACTORS, "runs": first_runs(8, seed=1), "response": "y", **change}
    with pytest.raises(ValueError, match=refusal):
        suggest_next_runs(**asked)


# --------------------------------------------------------------------------- the tool an agent calls
def _tool_input(**change: object) -> dict:
    """Return the tool's input for twelve of Branin's runs, the factors as an agent gives them."""
    runs = first_runs(12, seed=6).round(4)
    factors = [{"name": f.name, "low": f.low, "high": f.high} for f in FACTORS]
    return {"factors": factors, "runs": runs.to_dict("records"), "response_column": "y", "goal": "minimize", **change}


def test_the_tool_proposes_a_batch_inside_the_limit_with_why_each_run_is_there():
    found = execute_tool_call("suggest_next_runs", _tool_input(n_runs=3, constraints=["x1 + x2 <= 18"]))
    assert inside(pd.DataFrame(found["runs"]))
    assert len(found["runs"]) == 3
    assert set(found["runs"][0]) == {"x1", "x2", "predicted", "std_error", "expected_improvement"}
    assert set(found["theta"]) == {"x1", "x2"}
    assert found["little_to_gain"] is (found["improvement_ratio"] < 0.01)


def test_the_tool_says_when_little_is_left_to_gain():
    """Hundredths to gain on a response near 1000: confirmation runs at the best settings are the next step."""
    near = pd.DataFrame(_tool_input()["runs"]).assign(y=lambda d: 1000 + 0.01 * d["x1"]).to_dict("records")
    found = execute_tool_call("suggest_next_runs", _tool_input(runs=near, noise=False))
    assert found["improvement_ratio"] < 0.01
    assert found["little_to_gain"] is True


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"runs": [{"x1": 0.0, "x2": 0.0, "y": float(i)} for i in range(201)]}, "the model takes at most 200"),
        ({"factors": [{"name": "c", "type": "categorical", "levels": ["a", "b"]}]}, "takes one or more numeric"),
        ({"constraints": ["x1 + x2 <= -100"]}, "the constraints may leave no feasible settings"),
    ],
    ids=["too-many-runs", "categorical", "no-room"],
)
def test_the_tool_answers_what_it_cannot_do_in_a_sentence(change, refusal):
    assert refusal in execute_tool_call("suggest_next_runs", _tool_input(**change))["error"]
