"""L-pseudocomponents of a mixture with lower bounds on its components (#788 A10).

The pid-book's mixture section puts lower bounds of 0.2, 0.2 and 0 on three ingredients, so the
pure first pseudocomponent is the blend (0.8, 0.2, 0.0). Goos and Jones's rolling mill bounds each
of three mills by 0.1, and the reference catalogue codes its runs in pseudocomponents itself.
These tests hold the conversions to both, and to the geometry of the region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import from_pseudocomponents, to_pseudocomponents
from process_improve.experiments.reference_catalogue import reference_experiments


def test_the_pure_first_pseudocomponent_of_the_pid_book_s_region():
    """Lower bounds 0.2, 0.2 and 0, which add to 0.4: x_1 = 100% is a_1 = 0.8, a_2 = 0.2, a_3 = 0.0."""
    blend = from_pseudocomponents([1.0, 0.0, 0.0], [0.2, 0.2, 0.0])
    assert blend == pytest.approx([0.8, 0.2, 0.0])


def test_the_region_s_vertices_are_the_pure_pseudocomponents():
    """Each vertex of the bounded region has one component at 1 - (the others' bounds), the rest at theirs."""
    lower = np.array([0.1, 0.25, 0.05])
    vertices = np.array([[1 - 0.30, 0.25, 0.05], [0.1, 1 - 0.15, 0.05], [0.1, 0.25, 1 - 0.35]])
    assert to_pseudocomponents(vertices, lower) == pytest.approx(np.eye(3))


def test_the_rolling_mill_runs_are_coded_as_the_catalogue_codes_them():
    """Mills bounded by 0.1 each: (0.1, 0.8, 0.1) is pure mill 2, (0.1, 0.45, 0.45) the 50:50 blend of 2 and 3."""
    [entry] = [e for e in reference_experiments(source="goos-jones-2011") if e["name"] == "goos-jones-rolling-mill"]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    mills = runs[["mill1", "mill2", "mill3"]]
    bounds = {f["name"]: f["low"] for f in entry["factors"] if f.get("coding") == "pseudocomponent"}
    coded = to_pseudocomponents(mills, bounds)
    assert list(coded.columns) == ["mill1", "mill2", "mill3"]
    assert coded.index.equals(mills.index)
    assert coded.to_numpy() == pytest.approx((mills.to_numpy() - 0.1) / 0.7)
    assert coded.iloc[0].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert coded.iloc[2].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert coded.sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_the_conversions_undo_each_other_and_keep_the_total():
    rng = np.random.default_rng(4)
    lower = np.array([0.15, 0.05, 0.3, 0.0])
    pseudo = rng.dirichlet(np.ones(4), size=50)
    blends = from_pseudocomponents(pseudo, lower)
    assert blends.sum(axis=1) == pytest.approx(1.0)
    assert np.all(blends >= lower - 1e-15)
    assert to_pseudocomponents(blends, lower) == pytest.approx(pseudo)


def test_percentages_use_the_total_100():
    assert from_pseudocomponents([0.5, 0.5, 0.0], [20.0, 20.0, 0.0], total=100.0) == pytest.approx([50.0, 50.0, 0.0])


def test_a_component_a_mapping_does_not_name_has_the_bound_0():
    blends = pd.DataFrame({"a": [0.5], "b": [0.3], "c": [0.2]})
    coded = to_pseudocomponents(blends, {"a": 0.2, "b": 0.2})
    assert coded.iloc[0].tolist() == pytest.approx([0.5, 1 / 6, 1 / 3])


@pytest.mark.parametrize(
    ("blends", "lower", "message"),
    [
        ([0.5, 0.5], [0.6, 0.5], "leaves no room"),
        ([0.5, 0.5], [-0.1, 0.0], "0 or more"),
        ([0.5, 0.5], [0.1], "one bound per component, 2"),
        ([0.5, np.nan], [0.1, 0.1], "finite numbers"),
        (pd.DataFrame({"a": [0.5], "b": [0.5]}), {"z": 0.1}, "not columns of the blends"),
    ],
)
def test_bounds_that_cannot_be_used_are_refused(blends, lower, message):
    with pytest.raises(ValueError, match=message):
        to_pseudocomponents(blends, lower)


def test_bounds_by_name_need_a_data_frame():
    with pytest.raises(TypeError, match="need the blends as a DataFrame"):
        to_pseudocomponents([0.5, 0.5], {"a": 0.1})
