"""The correlation map of a study's own runs: how far each pair of its terms moves together.

The comparison of designs draws this map for each design it compares. ``api_correlations`` gives it for the runs the
page holds, a design made here or a table of the reader's own, and lists the pairs that matter to the model the
design estimates: two of its terms that move together, and a term of it aliased with one it leaves out.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import FACTORS, correlated_study, factors_named, held_of


def found(api, spec: dict) -> dict:
    """Return the map of the design a specification makes."""
    return api.ok("api_correlations", held_of(api.ok("api_make_design", spec)))


def listed(reply: dict) -> list[tuple[str, str, float, bool]]:
    return [(pair["a"], pair["b"], round(pair["r"], 2), pair["both"]) for pair in reply["pairs"]]


def with_a_confirmation_run(held: dict, at: dict) -> dict:
    """Return a study with one more run, marked as a confirmation run, at the settings ``at``."""
    columns = [*held["columns"], "Confirmation run"]
    extra = [{"Run order": len(held["rows"]) + 1, "Confirmation run": "yes", **at}.get(column) for column in columns]
    return {**held, "columns": columns, "rows": [[*row, None] for row in held["rows"]] + [extra]}


def test_the_map_is_of_the_main_effects_the_squares_and_the_interactions(api):
    reply = found(api, {"design_type": "box_behnken", "factors": FACTORS})
    assert reply["terms"] == ["T", "P", "F", "I(T ** 2)", "I(P ** 2)", "I(F ** 2)", "T:P", "T:F", "P:F"]
    assert reply["k"] == 3
    matrix = np.array(reply["matrix"], dtype=float)
    assert matrix.shape == (9, 9)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert reply["model"] == "quadratic"
    assert (reply["pairs"], reply["n_pairs"]) == ([], 0), "a Box-Behnken design tells its quadratic model's terms apart"


def test_a_resolution_iii_fraction_aliases_each_main_effect_with_an_interaction_the_model_leaves_out(api):
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*"ABC"), "resolution": 3}
    reply = found(api, {**spec, "n_center_points": 0})
    assert reply["model"] == "main_effects"
    assert listed(reply) == [("A", "B:C", 1.0, False), ("B", "A:C", 1.0, False), ("C", "A:B", 1.0, False)]
    assert reply["n_pairs"] == 3


def test_the_map_is_of_the_runs_a_model_is_fitted_to_and_not_of_a_confirmation_run(api):
    """A confirmation run off the fraction would make its aliases look partial: no fit uses the run to part them."""
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*"ABC"), "resolution": 3}
    held = held_of(api.ok("api_make_design", {**spec, "n_center_points": 0}))
    off_the_fraction = with_a_confirmation_run(held, {"A": 1, "B": 1, "C": 0})
    assert api.ok("api_correlations", off_the_fraction) == api.ok("api_correlations", held)
    assert listed(api.ok("api_correlations", held))[0] == ("A", "B:C", 1.0, False)


def test_a_term_of_the_model_confounded_with_the_blocks_is_listed_as_moving_with_them(api):
    """A 2^2 in two blocks gives the blocks the interaction: the fit holds the blocks, so it cannot hold T:P too."""
    spec = {"design_type": "full_factorial", "factors": factors_named("T", "P"), "n_center_points": 0}
    reply = found(api, {**spec, "n_blocks": 2})
    assert reply["model"] == "interactions"
    (pair,) = reply["pairs"]
    assert (pair["a"], pair["both"], pair["blocks"]) == ("T:P", True, True)
    assert pair["r"] == pytest.approx(1.0)
    assert reply["terms"] == ["T", "P", "I(T ** 2)", "I(P ** 2)", "T:P"], "the map draws the factors' terms only"
    assert found(api, spec)["pairs"] == [], "unblocked, the runs tell every term apart"
    # Center runs in each block change the columns, not the verdict: the design's own note still says T:P is
    # confounded with the blocks, and every fit leaves it out.
    with_centers = found(api, {**spec, "n_blocks": 2, "n_center_points": 2})
    assert [(p["a"], p["blocks"], p["r"]) for p in with_centers["pairs"] if p["blocks"]] == [("T:P", True, 1.0)]


def test_a_term_of_the_model_that_does_not_vary_moves_with_nothing_the_blocks_included(api):
    """Two levels and no center runs hold every square at one value: asked for with squares, the map lists none."""
    spec = {"design_type": "full_factorial", "factors": factors_named("T", "P"), "n_center_points": 0, "n_blocks": 2}
    held = held_of(api.ok("api_make_design", spec))
    reply = api.ok("api_correlations", {**held, "spec": {**held["spec"], "model": "quadratic"}})
    assert [(pair["a"], pair["blocks"]) for pair in reply["pairs"]] == [("T:P", True)], "the squares are in no pair"


def test_a_term_the_blocks_balance_does_not_move_with_them(api):
    """A 2^3 in two blocks gives up A:B:C, which no model of two-factor interactions holds."""
    spec = {"design_type": "full_factorial", "factors": factors_named(*"ABC"), "n_center_points": 0, "n_blocks": 2}
    assert found(api, spec)["pairs"] == []


def test_a_term_that_does_not_vary_has_no_correlation_and_is_in_no_pair(api):
    """A two-level design with no center runs holds every square at one value."""
    reply = found(api, {"design_type": "full_factorial", "factors": factors_named(*"AB"), "n_center_points": 0})
    squares = [reply["terms"].index(term) for term in ("I(A ** 2)", "I(B ** 2)")]
    assert all(value is None for i in squares for value in reply["matrix"][i])
    assert reply["pairs"] == []


def test_squares_that_move_as_one_say_nothing_of_a_model_that_leaves_them_out(api):
    """With center runs, a two-level design's squares are one column: its interactions model holds none of them."""
    reply = found(api, {"design_type": "full_factorial", "factors": FACTORS})
    at = {term: i for i, term in enumerate(reply["terms"])}
    assert reply["matrix"][at["I(T ** 2)"]][at["I(P ** 2)"]] == pytest.approx(1.0)
    assert reply["model"] == "interactions"
    assert reply["pairs"] == []


def test_factors_that_moved_together_in_the_reader_s_own_runs_are_listed_first(api, bootstrap):
    reply = found(api, correlated_study())
    first = reply["pairs"][0]
    assert (first["a"], first["b"], first["both"]) == ("A", "B", True)
    assert first["r"] == pytest.approx(0.94, abs=0.005)
    assert reply["listed_from"] == 0.3 < reply["together_from"] == 0.7 < first["r"] < reply["inseparable_from"] == 0.95
    assert not first["blocks"]
    assert all(pair["r"] >= bootstrap.PARTLY_TOGETHER for pair in reply["pairs"])
    sizes = [pair["r"] for pair in reply["pairs"]]
    assert sizes == sorted(sizes, reverse=True), "the largest first"


def test_only_the_largest_pairs_are_listed_and_all_are_counted(api, bootstrap):
    """A Plackett-Burman design in 11 factors: each main effect partly aliased with 45 interactions."""
    reply = found(api, {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFGHIJK")})
    assert reply["n_pairs"] == 11 * 45
    assert len(reply["pairs"]) == bootstrap.MOST_PAIRS
    assert {round(pair["r"], 3) for pair in reply["pairs"]} == {0.333}
    assert not any(pair["both"] for pair in reply["pairs"])
    assert listed(reply)[0][:2] == ("A", "B:C"), "equal pairs stay in the map's order"
