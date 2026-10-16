"""The limits on the settings, as the panel describes them while the reader types: ``api_limits``.

Each limit comes back in the canonical text a study stores, with what it does over the factors'
ranges, or with the sentence that says what to change in it. What it does is checked here against
what it means: each corner of the ranges against the limit written as Python, the share kept
against the library's region, and the narrowed ranges against a grid of settings.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from process_improve.experiments import DesignRegion, Factor
from process_improve.experiments.designs_constrained import _FUNCTIONS
from tests.web._support import FACTORS, factors_named

if TYPE_CHECKING:
    from collections.abc import Callable

#: The worked example's factors as the library holds them.
LIBRARY_FACTORS = [Factor(name=f["name"], low=f["low"], high=f["high"]) for f in FACTORS]

#: The heat limit of the corner examples: it cuts the corner where T and P are both high.
HEAT = "T + 25 * P <= 250"


def described(api, *limits: str | dict, factors: list[dict] = FACTORS, **request: object) -> dict:
    """Return what the panel shows for the limits typed, each a text or ``{expression, note}``."""
    sent = [{"expression": limit} if isinstance(limit, str) else limit for limit in limits]
    return api.ok("api_limits", {"factors": factors, "constraints": sent, **request})


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ("T+25*P<=250", HEAT),
        ("  400<=3*T+5*P<=600 ", "400 <= 3 * T + 5 * P <= 600"),
        ("T \u2264 190", "T <= 190"),
        ("T \u2212 25\u00b7P \u2265 100", "T - 25 * P >= 100"),
        ("T^2 / 100 + P <= 400", "T ** 2 / 100 + P <= 400"),
        ("250 >= T + 25*P", "250 >= T + 25 * P"),
    ],
)
def test_a_limit_comes_back_in_the_text_a_study_stores(api, typed, stored):
    """Spacing is the library's, and the signs a phone or a document types are read as the operators they stand for."""
    assert described(api, typed)["constraints"][0]["expression"] == stored


def test_a_limit_that_cannot_be_read_leaves_the_others_described(api):
    shown = described(api, "T + Z <= 300", HEAT)
    problem, heat = shown["constraints"]
    assert problem == {
        "expression": "T + Z <= 300",
        "note": "",
        "problem": "The limit 'T + Z <= 300' names Z, which is not one of your factors: T, P and F.",
    }
    assert heat["expression"] == HEAT
    assert heat["removes"] == [6, 7]
    assert shown["refusal"] == problem["problem"], "the brief is held back until the limit can be read"
    assert shown["probes"]["ruled_out_by"][6] == [1], "a corner names the limits by their place in the list"


def test_the_corners_come_in_the_page_s_order_then_the_centre(api):
    """The first factor changes slowest and each factor's low end comes first: the order a reader scans a table in."""
    probes = described(api, HEAT)["probes"]
    corners = [list(c) for c in itertools.product(*[(f["low"], f["high"]) for f in FACTORS])]
    assert probes["columns"] == ["T", "P", "F"]
    assert probes["rows"] == [*corners, [175, 2, 15]]
    assert probes["center"] == 8


#: Limits, and the same limits written as Python, to test each corner and the centre with.
AS_PYTHON = {
    HEAT: lambda t, p, _f: t + 25 * p <= 250,
    "T - 25 * P >= 100": lambda t, p, _f: t - 25 * p >= 100,
    "400 <= 3 * T + 5 * P + 2 * F <= 620": lambda t, p, f: 400 <= 3 * t + 5 * p + 2 * f <= 620,
    "T * F <= 3000": lambda t, _p, f: t * f <= 3000,
    "sqrt(T) + P <= 15": lambda t, p, _f: t**0.5 + p <= 15,
}


def test_each_corner_names_the_limits_that_rule_it_out(api):
    shown = described(api, *AS_PYTHON)
    rows = shown["probes"]["rows"]
    for i, (text, holds) in enumerate(AS_PYTHON.items()):
        ruled_out = [r for r, row in enumerate(rows) if not holds(*row)]
        assert shown["constraints"][i]["removes"] == ruled_out, text
    expected = [[i for i, holds in enumerate(AS_PYTHON.values()) if not holds(*row)] for row in rows]
    assert shown["probes"]["ruled_out_by"] == expected


def test_the_share_each_limit_rules_out_and_the_share_all_of_them_keep_are_the_library_s(api):
    shown = described(api, *AS_PYTHON)
    for entry in shown["constraints"]:
        alone = DesignRegion(LIBRARY_FACTORS, [entry["expression"]])
        assert entry["removed"] == pytest.approx(1 - alone.volume_fraction(), abs=1e-12)
    assert shown["kept"] == DesignRegion(LIBRARY_FACTORS, list(AS_PYTHON)).volume_fraction()
    # The heat limit cuts a triangle of an eighth of the T-P square off the box, whatever F is.
    assert shown["constraints"][0]["removed"] == pytest.approx(1 / 8, abs=0.005)


def test_the_ranges_the_limits_narrow_are_the_settings_left_at_their_ends(api):
    """``T + 25 P <= 215`` keeps T at or below 190 (P is at least 1) and P at or below 2.6 (T is at least 150)."""
    shown = described(api, "T + 25 * P <= 215")
    t, p = np.meshgrid(np.linspace(150, 200, 501), np.linspace(1, 3, 201))
    allowed = t + 25 * p <= 215 + 1e-9
    assert shown["bounds"] == [
        {"name": "T", "low": None, "high": pytest.approx(t[allowed].max())},
        {"name": "P", "low": None, "high": pytest.approx(p[allowed].max())},
    ]
    assert [b["high"] for b in shown["bounds"]] == [190.0, 2.6]


def test_a_range_narrowed_by_a_hair_is_not_reported(api):
    """Narrowing a range by 0.5% of it or less is no news: 199.9 degC is the top of the range to a reader."""
    assert described(api, "T <= 199.9")["bounds"] == []
    assert described(api, "T <= 199", "T >= 151")["bounds"] == [{"name": "T", "low": 151.0, "high": 199.0}]


def test_no_narrowed_ranges_are_given_with_a_curved_limit_or_with_nothing_left(api):
    """A curved limit's narrowing is not exact, so none is claimed; and an empty region has no ranges."""
    assert described(api, "T <= 190", "T * F <= 3000")["bounds"] is None
    assert described(api, "T >= 210")["bounds"] is None


def test_a_limit_that_rules_out_nothing_says_so_and_one_the_others_cover_too(api):
    shown = described(api, HEAT, "T + 25 * P <= 260", "T <= 250", "T + 25 * P <= 260")
    assert [entry["redundant"] for entry in shown["constraints"]] == [None, "others", "alone", "others"]
    assert shown["constraints"][2]["removed"] == 0
    assert shown["constraints"][2]["removes"] == []


def test_a_curved_limit_that_rules_out_nothing(api):
    shown = described(api, "T * F <= 4000", "T * F <= 3000")
    assert [entry["redundant"] for entry in shown["constraints"]] == ["alone", None]
    assert [entry["type"] for entry in shown["constraints"]] == ["nonlinear", "nonlinear"]


def test_limits_that_leave_nothing_to_run(api):
    shown = described(api, "T + P >= 300")
    assert shown["empty"] is True
    assert shown["kept"] == 0
    assert shown["center_allowed"] is False
    assert shown["refusal"] == "No setting in your ranges meets every limit: loosen a limit, or widen a range."


def test_whether_the_centre_of_the_ranges_can_still_be_run(api):
    assert described(api, HEAT)["center_allowed"] is True
    assert described(api, "T + 25 * P <= 220")["center_allowed"] is False, "175 + 50 is above 220"
    assert described(api)["center_allowed"] is True


def test_no_limits_keep_everything(api):
    shown = described(api)
    assert shown["constraints"] == []
    assert (shown["kept"], shown["empty"], shown["bounds"], shown["refusal"]) == (1.0, False, [], None)
    assert shown["probes"]["ruled_out_by"] == [[]] * 9


@pytest.mark.parametrize(
    ("limit", "linear", "kind"),
    [
        (HEAT, {"lhs": "T + 25 * P", "low": None, "high": 250.0}, "linear"),
        ("250 >= T + 25*P", {"lhs": "T + 25 * P", "low": None, "high": 250.0}, "linear"),
        ("400 <= 3*T + 5*P <= 600", {"lhs": "3 * T + 5 * P", "low": 400.0, "high": 600.0}, "linear"),
        ("T <= 2*P + 160", {"lhs": "T - 2 * P", "low": None, "high": 160.0}, "linear"),
        ("-T + 20*P >= -140", {"lhs": "-T + 20 * P", "low": -140.0, "high": None}, "linear"),
        ("0.04*T + F/2 <= 14.5", {"lhs": "0.04 * T + 0.5 * F", "low": None, "high": 14.5}, "linear"),
        ("P <= T / 100 <= 1.9", None, "linear"),
        ("T * F <= 3000", None, "nonlinear"),
    ],
)
def test_a_straight_limit_comes_with_its_bounds_to_edit(api, limit, linear, kind):
    """``low <= lhs <= high``, null on an open side; none for a curved limit, or a chain of two different sides."""
    entry = described(api, limit)["constraints"][0]
    assert entry["linear"] == linear
    assert entry["type"] == kind


def test_a_note_is_kept_on_one_line(api):
    entry = described(api, {"expression": HEAT, "note": "  chars the\tproduct\n "})["constraints"][0]
    assert entry["note"] == "chars the product"


def test_corners_are_listed_for_up_to_eight_numeric_factors(api):
    eight = described(api, "X0 + X1 <= 1", factors=factors_named(*(f"X{i}" for i in range(8))))
    assert len(eight["probes"]["rows"]) == 2**8 + 1
    nine = described(api, "X0 + X1 <= 1", factors=factors_named(*(f"X{i}" for i in range(9))))
    assert nine["probes"] is None
    assert nine["constraints"][0]["removes"] is None
    assert nine["constraints"][0]["redundant"] is None, "without corners, the library says whether it rules out nothing"


def test_a_categorical_factor_has_no_corners_and_holds_the_limits_back(api):
    shown = described(api, "T <= 190", factors=[*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}])
    assert shown["probes"]["columns"] == ["T", "P"]
    assert shown["refusal"].startswith("Limits on the settings work with numeric factors only")


def test_the_functions_a_limit_can_use_are_the_library_s(bootstrap):
    assert tuple(_FUNCTIONS) == bootstrap.LIMIT_FUNCTIONS


# --------------------------------------------------------------------------- the limits a specification stores
def test_a_specification_without_limits_stays_as_it_was(bootstrap, held):
    assert "constraints" not in held["spec"]
    assert "constraints" not in bootstrap._checked_spec(held["spec"])
    assert "constraints" not in bootstrap._checked_spec({**held["spec"], "constraints": []})


def test_a_stored_limit_is_checked_and_its_type_worked_out_again(bootstrap, held):
    """A workbook can arrive from anywhere: the type it claims for a limit is not read, and the text is re-read."""
    doctored = [{"expression": "T*F<=3000", "type": "linear", "note": "kept"}, {"expression": HEAT, "type": "curved"}]
    checked = bootstrap._checked_spec({**held["spec"], "constraints": doctored})
    assert checked["constraints"] == [
        {"expression": "T * F <= 3000", "type": "nonlinear", "note": "kept"},
        {"expression": HEAT, "type": "linear"},
    ]
    with pytest.raises(ValueError, match="names Z, which is not one of your factors"):
        bootstrap._checked_spec({**held["spec"], "constraints": [{"expression": "Z <= 3"}]})


# --------------------------------------------------------------------------- limits proposed from the corners marked
#: The worked example's corners, in the order of the table of corners; its centre is row 8.
CORNERS = [tuple(c) for c in itertools.product(*[(f["low"], f["high"]) for f in FACTORS])]


def corners_where(condition: Callable[[float, float, float], bool]) -> list[int]:
    """Return the rows of the table of corners whose settings meet ``condition(t, p, f)``."""
    return [row for row, corner in enumerate(CORNERS) if condition(*corner)]


@pytest.mark.parametrize(
    ("condition", "proposed"),
    [
        (lambda t, p, f: (t, p, f) == (200, 3, 20), "T + 25 * P + 5 * F <= 350"),
        (lambda t, p, _f: t == 200 and p == 3, "T + 25 * P <= 250"),
        (lambda t, p, _f: t == 200 and p == 1, "T - 25 * P <= 150"),
        (lambda t, _p, _f: t == 200, "T <= 190"),
    ],
)
def test_marking_corners_proposes_the_limit_that_rules_out_just_those(api, condition, proposed):
    """The cut lies halfway between the corners marked and those that can be run, written in round numbers."""
    marked = corners_where(condition)
    [proposal] = described(api, marked=marked)["proposals"]
    assert proposal["marked"] == marked
    assert proposal["expression"] == proposed
    assert proposal["removes"] == marked, "the centre and every other corner can still be run"
    assert proposal["linear"]["high"] is not None


def test_marking_the_centre_as_well_moves_the_limit_past_it(api):
    marked = [*corners_where(lambda t, _p, _f: t == 200), 8]
    [proposal] = described(api, marked=marked)["proposals"]
    assert (proposal["expression"], proposal["marked"]) == ("T <= 160", marked)


def test_corners_no_edge_joins_get_a_limit_each(api):
    """One straight cut removes corners that hang together along the edges of the box; these two need two cuts."""
    proposals = described(api, marked=[0, 7])["proposals"]
    assert [p["marked"] for p in proposals] == [[0], [7]]
    assert [p["expression"] for p in proposals] == ["T + 25 * P + 5 * F >= 250", "T + 25 * P + 5 * F <= 350"]


def test_a_corner_a_limit_rules_out_already_is_left_out_of_the_proposal(api):
    assert described(api, HEAT, marked=[6, 7])["proposals"] == []
    [proposal] = described(api, "T <= 190", marked=[3, 6, 7])["proposals"]
    assert proposal["marked"] == [3]
    assert proposal["expression"] == "5 * P + F <= 30", "the corners T <= 190 rules out count neither way"


#: Why no straight cut can be proposed, as the page says it beside the corners marked.
CENTRE_ALONE = (
    "The center of the ranges lies between settings you can run, so no straight cut rules it out alone. Type a curved "
    "limit, or mark more."
)
CENTRE_BETWEEN = (
    "The center of the ranges lies between the corners you marked, so a straight cut that rules them out rules it out "
    "too. Mark it as well, or type a limit."
)
INTERLEAVED = (
    "No single straight cut rules out these corners and keeps the others. Mark them in smaller groups, or type a limit."
)


@pytest.mark.parametrize(
    ("marked", "factors", "problem"),
    [
        (list(range(8)), FACTORS, None),
        ([8], FACTORS, CENTRE_ALONE),
        ([0, 1, 3], FACTORS[:2], CENTRE_BETWEEN),  # three corners of the T-P square: their triangle holds the centre
        ([0, 1, 2, 5], factors_named(*"ABCD"), INTERLEAVED),  # interleaved with corners that can be run
    ],
)
def test_when_no_straight_cut_can_be_proposed_the_page_says_why(api, bootstrap, marked, factors, problem):
    [proposal] = described(api, marked=marked, factors=factors)["proposals"]
    assert proposal == {"marked": marked, "problem": problem or bootstrap.EVERY_CORNER}


def test_a_proposal_says_where_it_cuts_each_factor_from_its_corner(api):
    """From the corner marked, each factor can go as far as the limit's boundary with the others held there."""
    [proposal] = described(api, marked=[7])["proposals"]
    assert proposal["corner"] == 7
    assert proposal["cuts"] == [
        {"name": "T", "at": 200, "cut": 175},
        {"name": "P", "at": 3, "cut": 2},
        {"name": "F", "at": 20, "cut": 15},
    ]
    [both] = described(api, marked=[6, 7])["proposals"]
    assert both["cuts"][2] == {"name": "F", "at": 10, "cut": None}, "a factor the limit does not depend on is not cut"


def test_a_proposal_whose_boundary_misses_an_edge_from_every_corner_has_no_cuts(api):
    """P high with T or F high: from each of the three corners the limit's boundary misses one edge of the box."""
    [proposal] = described(api, marked=[3, 6, 7])["proposals"]
    assert proposal["expression"] == "T + 50 * P + 5 * F <= 380"
    assert (proposal["corner"], proposal["cuts"]) == (None, None)


def test_moving_the_cuts_rebuilds_the_limit_through_them(api):
    shown = described(api, marked=[7], cuts={"marked": [7], "corner": 7, "values": [190, 2.5, None]})
    adjusted = shown["adjusted"]
    assert adjusted["expression"] == "T + 20 * P <= 250"
    assert adjusted["marked"] == [7]
    assert adjusted["removes"] == [6, 7], "with F not limiting it, the corner beside the one marked goes too"
    assert [cut["cut"] for cut in adjusted["cuts"]] == [190, 2.5, None]


def test_a_cut_no_limit_goes_through_is_a_problem_to_show(api):
    shown = described(api, marked=[7], cuts={"marked": [7], "corner": 7, "values": [200, None, None]})
    assert shown["adjusted"] == {
        "marked": [7],
        "problem": (
            "The cut of 'T' at 200 is at or beyond the corner's T = 200: it must lie between the corner and the other "
            "end of the range."
        ),
    }


# --------------------------------------------------------------------------- the picture of the region
def inside(point: list[float], polygon: list[list[float]]) -> bool:
    """Return whether a point lies in a convex polygon whose corners run counter-clockwise.

    A point on an edge counts as inside, to within the 12 significant figures the corners are given to.
    """
    corners = np.asarray(polygon, dtype=float)
    edges = np.roll(corners, -1, axis=0) - corners
    to_point = np.asarray(point, dtype=float) - corners
    distance = (edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0]) / np.linalg.norm(edges, axis=1)
    return bool(np.all(distance >= -1e-9 * np.abs(corners).max()))


def test_the_picture_shows_every_pair_of_up_to_four_factors(api):
    picture = described(api, HEAT)["picture"]
    assert [(pair["x"], pair["y"]) for pair in picture["pairs"]] == [("T", "P"), ("T", "F"), ("P", "F")]
    assert picture["shown"] == ["T", "P", "F"]
    assert picture["center"] == {"T": 175, "P": 2, "F": 15}


def test_a_shadow_holds_every_setting_that_can_be_run_and_no_more(api):
    """A setting of two factors is in their shadow when some setting of the others can be run with it."""
    limits = {
        "T + 25 * P + 5 * F <= 300": lambda t, p, f: t + 25 * p + 5 * f <= 300,
        "P - F / 10 >= 0": lambda _t, p, f: p - f / 10 >= 0,
    }
    pairs = described(api, *limits)["picture"]["pairs"]
    grid = itertools.product(np.linspace(150, 200, 11), np.linspace(1, 3, 11), np.linspace(10, 20, 11))
    allowed = [s for s in grid if all(holds(*s) for holds in limits.values())]
    assert allowed
    columns = {"T": 0, "P": 1, "F": 2}
    for pair in pairs:
        assert pair["shadow"]["kind"] == "polygon"
        x, y = columns[pair["x"]], columns[pair["y"]]
        assert all(inside([s[x], s[y]], pair["shadow"]["points"]) for s in allowed), (pair["x"], pair["y"])
    t_p = next(pair for pair in pairs if (pair["x"], pair["y"]) == ("T", "P"))
    assert t_p["shadow"]["points"] == [[150, 1], [200, 1], [200, 2], [175, 3], [150, 3]], "F at its lowest, 10"


def test_a_straight_limit_is_drawn_on_the_panels_of_the_factors_it_names(api):
    pairs = {(p["x"], p["y"]): p["lines"] for p in described(api, HEAT, "T <= 190")["picture"]["pairs"]}
    assert pairs[("T", "P")] == [
        {"limit": 0, "points": [[175, 3], [200, 2]]},
        {"limit": 1, "points": [[190, 1], [190, 3]]},
    ]
    assert pairs[("T", "F")] == [{"limit": 1, "points": [[190, 10], [190, 20]]}]
    assert pairs[("P", "F")] == []


def test_a_curved_region_is_drawn_from_a_mask(api):
    pairs = described(api, "T * F <= 3000")["picture"]["pairs"]
    shadow = pairs[1]["shadow"]
    assert (pairs[1]["x"], pairs[1]["y"], shadow["kind"]) == ("T", "F", "mask")
    assert len(shadow["x"]) == len(shadow["y"]) == 40
    for i, f in enumerate(shadow["y"]):
        for j, t in enumerate(shadow["x"]):
            if t * f <= 2900:  # well inside, so some sampled setting falls in the cell
                assert shadow["z"][i][j] == 1, (t, f)
            if t * f >= 3100:
                assert shadow["z"][i][j] == 0, (t, f)


def test_from_five_factors_the_picture_shows_the_pairs_of_factors_the_limits_name(api):
    five = factors_named(*"ABCDE")
    assert described(api, "A + D <= 1", factors=five)["picture"]["shown"] == ["A", "D"]
    every = described(api, "A + D <= 1", factors=five, every_pair=True)["picture"]
    assert (every["shown"], len(every["pairs"])) == (list("ABCDE"), 10)


def test_the_picture_holds_the_others_at_a_setting_that_can_be_run(api):
    """With the centre of the ranges ruled out, the page looks inside at a central setting that can be run."""
    center = described(api, "T + 25 * P <= 220")["picture"]["center"]
    assert center["T"] + 25 * center["P"] <= 220
    assert described(api, "T >= 210")["picture"]["center"] is None


def test_looking_inside_holds_the_other_factors_where_asked(api):
    """At F = 18 L/min, T + 25 P + 5 F <= 350 leaves T + 25 P <= 260."""
    shown = described(api, "T + 25 * P + 5 * F <= 350", slice={"x": "T", "y": "P", "at": {"F": 18}})["slice"]
    assert shown["region"] == {"kind": "polygon", "points": [[150, 1], [200, 1], [200, 2.4], [185, 3], [150, 3]]}
    assert shown["lines"] == [{"limit": 0, "points": [[185, 3], [200, 2.4]]}]
    assert (shown["x"], shown["y"], shown["at"]) == ("T", "P", {"F": 18})


def test_a_slice_through_a_curved_region_is_a_mask_that_matches_the_limit(api):
    shown = described(api, "T * F <= 3000", slice={"x": "T", "y": "P", "at": {"F": 16}})["slice"]["region"]
    assert shown["kind"] == "mask"
    for row, p in zip(shown["z"], shown["y"], strict=True):
        assert row == [int(t * 16 <= 3000 + 1e-9) for t in shown["x"]], p


def crossed(trace: dict) -> list[tuple[float, float]]:
    """Return the settings where a traced boundary crosses the lines of its grid."""
    xs, ys = trace["x"], trace["y"]
    return [
        (xs[j] + t * (xs[j + 1] - xs[j]), ys[i]) if along == 0 else (xs[j], ys[i] + t * (ys[i + 1] - ys[i]))
        for i, j, along, t in trace["crossings"]
    ]


def test_a_curved_limit_on_two_factors_is_traced_where_it_crosses_the_grid(api, bootstrap):
    """The page draws the curve through these points, smooth where a mask of the region is stepped."""
    [pair] = described(api, "T * P <= 450", factors=FACTORS[:2])["picture"]["pairs"]
    trace = pair["trace"]
    assert len(trace["x"]) == len(trace["y"]) == bootstrap.TRACE_POINTS
    on_the_curve = 1e-6  # T 500/3 degC and P 2.7 bar lie on it: the region's tolerance counts them in
    assert trace["z"] == [[int(t * p <= 450 + on_the_curve) for t in trace["x"]] for p in trace["y"]]
    points = crossed(trace)
    assert len(points) > bootstrap.TRACE_POINTS, "the curve crosses many lines of the grid"
    assert all(abs(t * p - 450) <= 1e-3 for t, p in points), "each crossing is on the curve"


def test_looking_inside_a_curved_region_traces_its_boundary_there(api):
    shown = described(api, "T * F <= 3000", slice={"x": "T", "y": "P", "at": {"F": 16}})["slice"]
    assert {round(t, 6) for t, _p in crossed(shown["trace"])} == {187.5}, "T * 16 <= 3000 is T <= 187.5"


def test_straight_limits_are_drawn_exactly_with_nothing_to_trace(api):
    shown = described(api, HEAT, slice={"x": "T", "y": "P", "at": {"F": 15}})
    assert shown["slice"]["trace"] is None
    assert [pair["trace"] for pair in shown["picture"]["pairs"]] == [None, None, None]
    [pair] = described(api, HEAT, factors=FACTORS[:2])["picture"]["pairs"]
    assert pair["trace"] is None
