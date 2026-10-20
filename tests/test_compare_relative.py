"""Two designs set against each other: relative efficiencies, coefficient variances and FDS curves (#790 C1).

Goos and Jones (2011, Section 10.2) compare the wind tunnel's I-optimal split-plot design with Dr.
Cavendish's second option. Their FDS curves cross at a fraction of 0.20 of the design space. The book
prints the second option's relative D-efficiency as 89.2% and the I-optimal design's gain as 12.1%;
the designs it prints give 91.4% and 9.4%, which the reference catalogue records as disagreements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, compare_designs, evaluate_design, generate_design
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments
from process_improve.experiments.region import DesignRegion

WIND = ["frh", "rrh", "yaw", "grille"]


def wind_tunnel(name: str) -> pd.DataFrame:
    """Return a printed wind tunnel design, coded, its whole plots marked, the variances equal."""
    [entry] = [
        e for e in reference_experiments("goos-jones-2011") + reference_designs("goos-jones-2011") if e["name"] == name
    ]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    ranges = {f["name"]: f["levels"] for f in entry["factors"]}
    frame = pd.DataFrame({n: (runs[n] - np.mean(ranges[n])) / (np.ptp(ranges[n]) / 2) for n in WIND})
    frame["WholePlot"] = runs["setting"]
    frame.attrs["whole_plot_variance_ratio"] = 1.0
    return frame


@pytest.fixture(scope="module")
def tunnel():
    return compare_designs(
        {
            "I-optimal": wind_tunnel("goos-jones-wind-tunnel"),
            "second option": wind_tunnel("goos-jones-wind-tunnel-second-option"),
        },
        model="quadratic",
    )


# --------------------------------------------------------------------------- Goos and Jones, Section 10.2
@pytest.mark.dataset
def test_the_wind_tunnel_designs_fds_curves_cross_at_a_fifth_of_the_space(tunnel):
    found = tunnel.relative("second option", "I-optimal")
    [crossing] = found["fds_crossings"]
    assert crossing == pytest.approx(0.20, abs=0.01)
    # The second option is the lower below the crossing, the I-optimal design above it.
    assert found["fds_lower_share"] == pytest.approx(crossing, abs=0.002)


@pytest.mark.dataset
def test_the_relative_d_efficiency_is_the_determinants_ratio_on_x_v_inverse_x(tunnel):
    """91.4% and a gain of 9.4%, from the printed designs; the book prints 89.2% and 12.1%."""
    found = tunnel.relative("second option", "I-optimal")
    assert round(100 * found["d_efficiency"], 1) == 91.4
    assert round(100 * (1 / found["d_efficiency"] - 1), 1) == 9.4
    model = " + ".join(WIND) + " + " + " + ".join(f"{a}:{b}" for i, a in enumerate(WIND) for b in WIND[i + 1 :])
    model += " + " + " + ".join(f"I({n} ** 2)" for n in WIND)
    logs = [
        evaluate_design(wind_tunnel(name), model=model, metric="information_matrix", random_effects={"WholePlot": 1.0})[
            "information_matrix"
        ]["log_determinant"]
        for name in ("goos-jones-wind-tunnel-second-option", "goos-jones-wind-tunnel")
    ]
    assert found["d_efficiency"] == pytest.approx(np.exp((logs[0] - logs[1]) / 15))


@pytest.mark.dataset
def test_each_comparison_is_the_other_one_turned_round(tunnel):
    forward, backward = tunnel.relative("second option", "I-optimal"), tunnel.relative("I-optimal", "second option")
    for key in ("d_efficiency", "a_efficiency", "i_efficiency", "g_efficiency"):
        assert forward[key] * backward[key] == pytest.approx(1.0)
    assert forward["fds_crossings"] == backward["fds_crossings"]
    assert forward["fds_lower_share"] + backward["fds_lower_share"] == pytest.approx(1.0)
    for term, ratio in forward["variance_ratio"].items():
        assert ratio * backward["variance_ratio"][term] == pytest.approx(1.0)


# --------------------------------------------------------------------------- what the numbers mean
def test_a_design_against_itself_is_level_everywhere():
    factors = [Factor(name=n, low=-1, high=1) for n in "ABC"]
    design = generate_design(factors, "box_behnken", n_center_points=3)
    found = compare_designs({"first": design, "copy": design}, model="quadratic").relative("first", "copy")
    for key in ("d_efficiency", "a_efficiency", "i_efficiency", "g_efficiency"):
        assert found[key] == pytest.approx(1.0)
    assert set(np.round(list(found["variance_ratio"].values()), 12)) == {1.0}
    assert (found["fds_crossings"], found["fds_lower_share"]) == ([], 0.0)


def test_a_replicated_design_is_lower_over_the_whole_space():
    """Two copies of a factorial halve every variance: no crossing, and lower at every fraction."""
    factors = [Factor(name=n, low=-1, high=1) for n in "ABC"]
    single = generate_design(factors, "full_factorial", n_center_points=0)
    double = generate_design(factors, "full_factorial", n_center_points=0, n_replicates=2)
    found = compare_designs({"double": double, "single": single}, model="interactions").relative("double", "single")
    assert found["d_efficiency"] == pytest.approx(2.0)
    assert found["i_efficiency"] == pytest.approx(2.0)
    assert set(np.round(list(found["variance_ratio"].values()), 12)) == {0.5}
    assert (found["fds_crossings"], found["fds_lower_share"]) == ([], 1.0)


def test_over_a_region_the_curves_are_drawn_from_the_region():
    factors = [Factor(name=n, low=-1, high=1) for n in "AB"]
    region = DesignRegion(factors, ["A + B <= 1"])
    designs = {
        "factorial": generate_design(factors, "full_factorial", n_center_points=1),
        "ccd": generate_design(factors, "ccd", alpha="face_centered", n_center_points=1),
    }
    found = compare_designs(designs, model="interactions", region=region).relative("ccd", "factorial", n_points=5000)
    assert found["n_points"] == 5000
    assert 0.0 <= found["fds_lower_share"] <= 1.0


def test_the_same_seed_gives_the_same_curves():
    factors = [Factor(name=n, low=-1, high=1) for n in "AB"]
    designs = {
        "factorial": generate_design(factors, "full_factorial"),
        "ccd": generate_design(factors, "ccd", alpha="face_centered"),
    }
    comparison = compare_designs(designs, model="interactions")
    assert comparison.relative("ccd", "factorial", random_state=3) == comparison.relative(
        "ccd", "factorial", random_state=3
    )


# --------------------------------------------------------------------------- what is refused
def test_labels_that_name_no_design_or_a_design_that_cannot_fit_are_refused():
    factors = [Factor(name=n, low=-1, high=1) for n in "AB"]
    comparison = compare_designs(
        {
            "factorial": generate_design(factors, "full_factorial", n_center_points=0),
            "ccd": generate_design(factors, "ccd", alpha="face_centered"),
        },
        model="quadratic",
    )
    with pytest.raises(ValueError, match="is not one of the designs compared"):
        comparison.relative("dsd", "ccd")
    with pytest.raises(ValueError, match="cannot fit the quadratic model"):
        comparison.relative("factorial", "ccd")
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        comparison.relative("ccd", "ccd", n_points=1)
