"""Split-plot designs from the built-in coordinate exchange on X'V^-1 X (#788 A2).

Goos and Jones (2011, chapter 10) print two split-plot designs the search must find again, each on
the generalised least squares information with the whole-plot variance equal to the run variance:
the wind tunnel's I-optimal design (50 runs in ten whole plots of five, I = 0.512) and a 2^4 in
eight whole plots of two runs. The search, without pyoptex, matches both to within 1% or betters them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design, generate_design
from process_improve.experiments.designs_split_plot import SplitPlotOptions, split_plot_design
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments

WIND = ["frh", "rrh", "yaw", "grille"]
QUADRATIC = (
    " + ".join(WIND)
    + " + "
    + " + ".join(f"{a}:{b}" for i, a in enumerate(WIND) for b in WIND[i + 1 :])
    + " + "
    + " + ".join(f"I({n} ** 2)" for n in WIND)
)


def catalogue(name: str) -> dict:
    [entry] = [
        e for e in reference_experiments("goos-jones-2011") + reference_designs("goos-jones-2011") if e["name"] == name
    ]
    return entry


def continuous(names) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def frame(design: np.ndarray, meta: dict, names) -> pd.DataFrame:
    """Return the searched design with its whole plots, as evaluate_design reads it."""
    runs = pd.DataFrame(design, columns=list(names))
    runs["plot"] = meta["whole_plot"]
    return runs


def gls(runs: pd.DataFrame, model: str, metric: str) -> float:
    """Return a metric on X'V^-1 X with the whole-plot variance equal to the run variance (eta = 1)."""
    found = evaluate_design(runs, model=model, metric=metric, random_effects={"plot": 1.0}, integration="exact")
    return found[metric]


@pytest.fixture(scope="module")
def wind_tunnel_search() -> tuple[np.ndarray, dict]:
    options = SplitPlotOptions(criterion="i_optimal", model_type="quadratic", n_whole_plots=10, variance_ratio=1.0)
    return split_plot_design(continuous(WIND), ["frh", "rrh"], 50, options, random_state=1)


# --------------------------------------------------------------------------- the sources' designs, found again
@pytest.mark.dataset
def test_the_wind_tunnel_i_optimal_design_is_found_again(wind_tunnel_search):
    """Goos and Jones's 50-run design averages 0.512; the search matches it to within 1%."""
    entry = catalogue("goos-jones-wind-tunnel")
    printed = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    ranges = {f["name"]: f["levels"] for f in entry["factors"]}
    coded = pd.DataFrame({n: (printed[n] - np.mean(ranges[n])) / (np.ptp(ranges[n]) / 2) for n in WIND})
    coded["plot"] = printed["setting"]
    theirs = gls(coded, QUADRATIC, "average_prediction_variance")
    assert theirs == pytest.approx(0.512, abs=5e-4)
    design, meta = wind_tunnel_search
    ours = gls(frame(design, meta, WIND), QUADRATIC, "average_prediction_variance")
    assert ours <= 1.01 * theirs
    assert meta["trace_criterion"] == pytest.approx(ours)  # the search's W is the cube's exact moments


@pytest.mark.dataset
def test_the_two_runs_per_whole_plot_design_is_found_again():
    """The 2^4 in eight whole plots of two: the D-optimal search reaches its determinant."""
    entry = catalogue("goos-jones-split-plot-two-runs-per-whole-plot")
    printed = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])[["plot", "w", "s1", "s2", "s3"]]
    model = "w + s1 + s2 + s3 + w:s1 + w:s2 + w:s3 + s1:s2 + s1:s3 + s2:s3"
    theirs = gls(printed, model, "information_matrix")["log_determinant"]
    options = SplitPlotOptions(criterion="d_optimal", model_type="interactions", n_whole_plots=8, variance_ratio=1.0)
    design, meta = split_plot_design(continuous(["w", "s1", "s2", "s3"]), ["w"], 16, options, random_state=3)
    ours = gls(frame(design, meta, ["w", "s1", "s2", "s3"]), model, "information_matrix")["log_determinant"]
    assert ours >= theirs - np.log(1.01) * 11  # within 1% on the per-parameter scale |M|^(1/p), p = 11
    assert meta["log_det_information"] == pytest.approx(ours)


# --------------------------------------------------------------------------- the structure
def test_hard_to_change_factors_are_constant_within_each_whole_plot(wind_tunnel_search):
    design, meta = wind_tunnel_search
    runs = frame(design, meta, WIND)
    assert runs.groupby("plot")[["frh", "rrh"]].nunique().max().tolist() == [1, 1]
    assert runs.groupby("plot")[["yaw", "grille"]].nunique().max().max() > 1
    assert (meta["n_whole_plots"], np.bincount(meta["whole_plot"]).tolist()) == (10, [5] * 10)
    assert meta["backend"] == "split_plot_exchange"
    assert meta["whole_plot_variance_ratio"] == 1.0


def test_the_last_whole_plot_takes_the_runs_left_over():
    options = SplitPlotOptions(n_whole_plots=4, model_type="main_effects")
    _, meta = split_plot_design(continuous("ABC"), ["A"], 13, options, random_state=0)
    assert np.bincount(meta["whole_plot"]).tolist() == [3, 3, 3, 4]


def test_the_same_seed_gives_the_same_design():
    options = SplitPlotOptions(criterion="a_optimal", n_whole_plots=4)
    first = split_plot_design(continuous("ABC"), ["A", "B"], 12, options, random_state=7)
    again = split_plot_design(continuous("ABC"), ["A", "B"], 12, options, random_state=7)
    np.testing.assert_array_equal(first[0], again[0])
    assert first[1] == again[1]


def test_each_start_the_search_climbs_from_estimates_the_model(monkeypatch):
    """A random start that cannot estimate the model is drawn again before the search climbs from it.

    From a singular start the search worked on M plus a small ridge, so ill-conditioned that the
    BLAS kernel's rounding chose its path: the case study's 48-run I-optimal design came out
    differently on AVX2 and on AVX-512 processors. A quarter of this case's random starts are singular.
    """
    from process_improve.experiments import designs_split_plot

    draws, climbed = [], []
    draw, climb = designs_split_plot._random_start, designs_split_plot._climb

    def recorded_draw(*args):
        draws.append(1)
        return draw(*args)

    def recorded_climb(model, levels, *args):
        climbed.append(np.linalg.matrix_rank(model.rows(levels)) == len(model.parts))
        return climb(model, levels, *args)

    monkeypatch.setattr(designs_split_plot, "_random_start", recorded_draw)
    monkeypatch.setattr(designs_split_plot, "_climb", recorded_climb)
    options = SplitPlotOptions(criterion="i_optimal", model_type="quadratic", n_whole_plots=4)
    split_plot_design(continuous("ABC"), ["A"], 12, options, random_state=0)
    assert len(climbed) == options.n_starts
    assert all(climbed)
    assert len(draws) > options.n_starts, "some random starts were singular, and drawn again"


def test_categorical_factors_take_their_labels_and_their_effect_coding():
    factors = [Factor(name="Oven", type="categorical", levels=["x", "y", "z"]), *continuous(["T", "P"])]
    options = SplitPlotOptions(criterion="d_optimal", model_type="interactions", n_whole_plots=6)
    design, meta = split_plot_design(factors, ["Oven"], 18, options, random_state=2)
    runs = pd.DataFrame(design, columns=["Oven", "T", "P"]).assign(plot=meta["whole_plot"])
    assert set(runs["Oven"]) <= {"x", "y", "z"}
    assert runs.groupby("plot")["Oven"].nunique().max() == 1
    model = "C(Oven) + T + P + C(Oven):T + C(Oven):P + T:P"
    found = evaluate_design(runs, model=model, metric="information_matrix", random_effects={"plot": 0.5})
    assert meta["log_det_information"] == pytest.approx(found["information_matrix"]["log_determinant"])


# --------------------------------------------------------------------------- through generate_design
def test_generate_design_builds_a_split_plot_design_without_pyoptex(monkeypatch):
    from process_improve.experiments import designs_optimal

    monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
    result = generate_design(continuous("ABC"), design_type="d_optimal", budget=12, hard_to_change=["A"])
    assert result.metadata["backend"] == "split_plot_exchange"
    assert result.design.groupby("WholePlot")["A"].nunique().max() == 1
    # A split-plot design keeps its order: each whole plot's runs together, as they are to be run.
    assert result.design["WholePlot"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("asked", "reason"),
    [
        ({"fixed_runs": pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0]})}, "fixed runs are given"),
        ({"n_blocks": 2}, "blocks are asked for"),
    ],
)
def test_what_the_search_cannot_honour_falls_back_with_a_warning(asked, reason):
    with pytest.warns(UserWarning, match=f"hard_to_change=.*is ignored.*{reason}"):
        result = generate_design(continuous("ABC"), design_type="d_optimal", budget=12, hard_to_change=["A"], **asked)
    assert result.metadata["hard_to_change_ignored"] == ["A"]


def test_e_optimality_falls_back_with_a_warning():
    with pytest.warns(UserWarning, match="has no E-optimality criterion"):
        generate_design(continuous("ABC"), design_type="e_optimal", budget=12, hard_to_change=["A"])


@pytest.mark.parametrize(
    ("options", "n_runs", "message"),
    [
        (SplitPlotOptions(criterion="e_optimal"), 12, "searched for on"),
        (SplitPlotOptions(variance_ratio=-1.0), 12, "finite number of at least 0"),
        (SplitPlotOptions(n_whole_plots=13), 12, "from 1 to 12 whole plots"),
        (SplitPlotOptions(n_starts=0), 12, "at least one start"),
        (SplitPlotOptions(model_type="quadratic", n_whole_plots=2), 6, "cannot estimate the 10 terms"),
    ],
)
def test_requests_the_search_cannot_meet_are_refused(options, n_runs, message):
    with pytest.raises(ValueError, match=message):
        split_plot_design(continuous("ABC"), ["A"], n_runs, options)


def test_too_few_whole_plots_for_the_whole_plot_terms_is_refused():
    """Two whole plots cannot estimate A, A**2 and the intercept, whatever the runs within them."""
    options = SplitPlotOptions(model_type="quadratic", n_whole_plots=2, n_starts=2)
    with pytest.raises(ValueError, match="estimates the quadratic model; give more runs or more whole plots"):
        split_plot_design(continuous("AB"), ["A"], 12, options, random_state=0)
