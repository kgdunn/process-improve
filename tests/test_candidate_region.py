"""The prediction variance over a discrete region: the candidate points a design was chosen from (#790 C3).

NIST 5.5.2.1 chooses a 12-run D-optimal design from 20 candidate points (Table 5.1) and reports its
I-efficiency as N times d(x) averaged over those 20 points: 12 x 0.388542 = 4.6625. Over the cube the
library's average is 0.2555, a different region, so the printed value needs the candidates as the region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_all, evaluate_design, generate_design
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments

MODEL = "X1 + X2 + X3 + I(X1**2)"


def runs(name: str) -> pd.DataFrame:
    [entry] = [e for e in reference_designs() + reference_experiments() if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])[["X1", "X2", "X3"]]


@pytest.fixture(scope="module")
def nist() -> tuple[pd.DataFrame, pd.DataFrame]:
    """NIST's 12-run D-optimal design and the 20 candidate points it was chosen from."""
    return runs("nist-5-5-2-1-d-optimal-design"), runs("nist-5-5-2-1-candidate-set")


@pytest.mark.dataset
def test_nist_s_i_efficiency_is_the_average_over_its_candidates(nist):
    """I = N times d(x) averaged over the 20 candidates: 12 x 0.388542 = 4.6625."""
    design, candidates = nist
    found = evaluate_design(design, model=MODEL, metric="average_prediction_variance", region=candidates)
    assert found["average_prediction_variance"] == pytest.approx(0.388542, abs=5e-7)
    assert len(design) * found["average_prediction_variance"] == pytest.approx(4.6625)
    cube = evaluate_design(design, model=MODEL, metric="average_prediction_variance")
    assert cube["average_prediction_variance"] == pytest.approx(0.2555, abs=1e-4)  # another region, another value


@pytest.mark.dataset
def test_over_the_design_s_own_runs_the_average_is_p_over_n_and_g_is_nist_s_one(nist):
    """The leverages of the 12 runs average p/N = 5/12, and each is 5/12, so G is 100 (NIST's G = 1)."""
    design, _ = nist
    found = evaluate_design(design, model=MODEL, metric=["average_prediction_variance", "g_efficiency"], region=design)
    assert found["average_prediction_variance"] == pytest.approx(5 / 12)
    assert found["max_prediction_variance"] == pytest.approx(5 / 12)
    assert found["g_efficiency"] == pytest.approx(100.0)


@pytest.mark.dataset
def test_the_average_over_candidates_is_exact_whatever_the_sampling_options(nist):
    """Each point counts once: no sample, so the seed, the integration and the search change nothing."""
    design, candidates = nist
    metrics = ["average_prediction_variance", "g_efficiency"]
    first = evaluate_design(design, model=MODEL, metric=metrics, region=candidates, random_state=1)
    for options in ({"random_state": 2}, {"integration": "exact"}, {"max_search": "optimize"}, {"n_samples": 7}):
        again = evaluate_design(design, model=MODEL, metric=metrics, region=candidates, **options)
        assert again == pytest.approx(first)
        assert "notes" not in again


@pytest.mark.dataset
def test_the_fds_curve_over_candidates_is_read_from_every_point(nist):
    design, candidates = nist
    fds = evaluate_design(design, model=MODEL, metric="fds", region=candidates)["fds"]
    assert (fds["region"], fds["n_samples"]) == ("candidates", 20)
    assert fds["quantiles"]["1"] == pytest.approx(fds["max_prediction_variance"])
    assert fds["average_prediction_variance"] == pytest.approx(0.388542, abs=5e-7)


def test_a_design_with_a_categorical_factor_takes_candidates_with_its_labels():
    """Over its own distinct runs, each once, a full factorial's average leverage is p/N."""
    factors = [Factor(name="T", low=-1, high=1), Factor(name="Cat", type="categorical", levels=["a", "b", "c"])]
    design = generate_design(factors, design_type="full_factorial", n_center_points=0)
    candidates = pd.DataFrame({"T": [-1.0, 1.0] * 3, "Cat": ["a", "a", "b", "b", "c", "c"]})
    found = evaluate_design(design, model="T + Cat", metric="average_prediction_variance", region=candidates)
    assert found["average_prediction_variance"] == pytest.approx(4 / 6)


def test_evaluate_all_takes_a_candidate_region_for_every_metric():
    design = pd.DataFrame({"A": [-1.0, 1.0, -1.0, 1.0, 0.0], "B": [-1.0, -1.0, 1.0, 1.0, 0.0]})
    candidates = pd.DataFrame({"A": np.linspace(-1, 1, 5).repeat(5), "B": np.tile(np.linspace(-1, 1, 5), 5)})
    found = evaluate_all(design, model="A + B", region=candidates)
    assert found["fds"]["region"] == "candidates"
    assert found["average_prediction_variance"] == pytest.approx(found["fds"]["average_prediction_variance"])


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (pd.DataFrame({"A": [0.0], "C": [0.0]}), r"no column for \['B'\]"),
        (pd.DataFrame({"A": [], "B": []}), "it has no points"),
        (pd.DataFrame({"A": [0.0, np.nan], "B": [0.0, 1.0]}), "missing settings"),
    ],
)
def test_candidates_that_cannot_stand_for_the_region_are_refused(candidates, message):
    design = pd.DataFrame({"A": [-1.0, 1.0, -1.0, 1.0], "B": [-1.0, -1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match=message):
        evaluate_design(design, model="A + B", metric="average_prediction_variance", region=candidates)
