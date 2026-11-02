"""augment_design on a mixture design: its axial check blends, its centroid, and what it refuses (#788 A6).

Ullmann's axial simplex (Table 6) is the three vertices, the three axial check blends and the
centroid: the design for the linear Scheffe model with its axial points added. Montgomery's
augmented simplex lattice is the simplex centroid design with the same three axial check blends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, augment_design, generate_design
from process_improve.experiments.reference_catalogue import reference_designs


def printed(source: str, name: str, columns: list[str]) -> pd.DataFrame:
    [entry] = [e for e in reference_designs(source) if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])[columns]


def components(names: str | list[str], low: float = 0.0, high: float = 1.0) -> list[Factor]:
    return [Factor(name=n, type="mixture", low=low, high=high) for n in names]


def mixture(factors: list[Factor], model_type: str) -> pd.DataFrame:
    design = generate_design(factors, "mixture", model_type=model_type, random_state=1).design
    return pd.DataFrame(design)[[f.name for f in factors]]


def in_order(runs: np.ndarray) -> np.ndarray:
    """Sort the runs, so that two tables of the same runs in any order compare equal."""
    return runs[np.lexsort(np.round(runs, 3).T)]


# --------------------------------------------------------------------------- the sources
@pytest.mark.dataset
def test_ullmann_s_axial_simplex_is_the_linear_design_with_its_axial_points():
    linear = mixture(components("ABC"), "scheffe_linear")  # the vertices and the centroid
    result = augment_design(linear, "add_axial_points")
    assert (result["n_runs_before"], result["n_runs_after"]) == (4, 7)
    table = printed("ullmann-doe-2012", "ullmann-axial-simplex", list("ABC")).to_numpy(dtype=float)
    # Table 6 prints four decimals: 0.6666, 0.1667 and 0.3334.
    np.testing.assert_allclose(
        in_order(pd.DataFrame(result["augmented_design"]).to_numpy()), in_order(table), atol=1e-4
    )


@pytest.mark.dataset
def test_montgomery_s_augmented_simplex_lattice_is_the_simplex_centroid_with_its_axial_points():
    centroid_design = mixture(components(["x1", "x2", "x3"]), "scheffe_quadratic")
    result = augment_design(centroid_design, "add_axial_points")
    table = printed("montgomery-2013", "montgomery-augmented-simplex-lattice-3", ["x1", "x2", "x3"])
    np.testing.assert_allclose(
        in_order(pd.DataFrame(result["augmented_design"]).to_numpy()), in_order(table.to_numpy(dtype=float)), atol=1e-6
    )


# --------------------------------------------------------------------------- where the blends go
def test_each_axial_check_blend_is_midway_between_the_centroid_and_a_vertex():
    """With four components the blend on an axis is (5/8, 1/8, 1/8, 1/8)."""
    result = augment_design(mixture(components("ABCD"), "scheffe_linear"), "add_axial_points")
    blends = pd.DataFrame(result["new_runs"]).to_numpy()
    np.testing.assert_allclose(blends, (np.full((4, 4), 0.25) + np.eye(4)) / 2)
    np.testing.assert_allclose(np.diag(blends), 5 / 8)
    np.testing.assert_allclose(blends.sum(axis=1), 1.0)


def test_the_blends_of_an_l_pseudocomponent_simplex():
    """With every component at least 0.1, the simplex has vertices like (0.8, 0.1, 0.1)."""
    design = mixture(components("ABC", low=0.1), "scheffe_quadratic")
    blends = pd.DataFrame(augment_design(design, "add_axial_points")["new_runs"]).to_numpy()
    np.testing.assert_allclose(blends, 0.1 + 0.7 * (np.eye(3) + 1 / 3) / 2)
    np.testing.assert_allclose(blends[0], [0.56667, 0.21667, 0.21667], atol=1e-5)


def test_center_points_of_a_mixture_are_at_the_centroid():
    factors = [
        Factor(name="A", type="mixture", low=0.1, high=1),
        Factor(name="B", type="mixture", low=0.2, high=1),
        Factor(name="C", type="mixture", low=0.0, high=1),
    ]
    result = augment_design(mixture(factors, "scheffe_quadratic"), "add_center_points", n_additional_runs=2)
    centroid = np.array([0.1, 0.2, 0.0]) + 0.7 / 3
    np.testing.assert_allclose(pd.DataFrame(result["new_runs"]).to_numpy(), [centroid, centroid])
    assert "centroid of the simplex" in result["explanation"]


def test_the_explanation_names_the_blends_and_reports_no_aliases():
    result = augment_design(mixture(components("ABC"), "scheffe_linear"), "add_axial_points")
    assert "3 axial check blends" in result["explanation"]
    assert "(0.6667, 0.1667, 0.1667)" in result["explanation"]
    assert "Aliasing" not in result["explanation"]


def test_replicate_still_copies_the_blends():
    design = mixture(components("ABC"), "scheffe_quadratic")
    result = augment_design(design, "replicate")
    assert result["n_runs_after"] == 2 * len(design)


def test_a_design_in_coded_units_still_gets_star_points():
    design = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
    star = pd.DataFrame(augment_design(design, "add_axial_points")["new_runs"]).to_numpy()
    np.testing.assert_allclose(np.abs(star).max(axis=1), np.sqrt(2))


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize("kind", ["foldover", "semifold", "upgrade_to_rsm", "add_runs_optimal", "add_blocks"])
def test_augmentations_that_would_not_give_blends_are_refused(kind):
    with pytest.raises(ValueError, match="the runs are mixture blends"):
        augment_design(mixture(components("ABC"), "scheffe_quadratic"), kind, n_additional_runs=2)


@pytest.mark.dataset
def test_proportions_printed_to_four_decimals_are_still_blends():
    """Ullmann's 0.6666 + 0.1667 + 0.1667 and 0.3333 + 0.3333 + 0.3334 each sum to 1."""
    table = printed("ullmann-doe-2012", "ullmann-axial-simplex", list("ABC"))
    with pytest.raises(ValueError, match="the runs are mixture blends"):
        augment_design(table, "foldover")
    centre = augment_design(table, "add_center_points", n_additional_runs=1)["new_runs"]
    np.testing.assert_allclose(list(centre[0].values()), 1 / 3)


def test_alpha_is_refused_for_the_axial_points_of_blends():
    with pytest.raises(ValueError, match="leave alpha unset"):
        augment_design(mixture(components("ABC"), "scheffe_linear"), "add_axial_points", alpha="rotatable")


def test_a_region_that_is_not_a_simplex_is_refused():
    """With A at most 0.5 no run is the vertex richest in A, so the simplex's points are not known."""
    factors = [Factor(name="A", type="mixture", low=0, high=0.5), *components("BC")]
    design = mixture(factors, "scheffe_quadratic")
    for kind in ("add_axial_points", "add_center_points"):
        with pytest.raises(ValueError, match=r"none is the vertex richest in \['A'\]"):
            augment_design(design, kind)
