"""The limits on the settings, as the panel describes them while the reader types: ``api_limits``.

Each limit comes back in the canonical text a study stores, with what it does over the factors'
ranges, or with the sentence that says what to change in it. What it does is checked here against
what it means: each corner of the ranges against the limit written as Python, the share kept
against the library's region, and the narrowed ranges against a grid of settings.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments import DesignRegion, Factor
from process_improve.experiments.designs_constrained import _FUNCTIONS
from tests.web._support import FACTORS, factors_named

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


def test_renaming_factors_renames_them_inside_the_limits(api):
    """A swap of two names is a swap: each limit keeps its meaning, and what it rules out stays as it was."""
    before = described(api, HEAT, "T * F <= 3000")
    swapped = [{**FACTORS[0], "name": "P"}, {**FACTORS[1], "name": "T"}, FACTORS[2]]
    after = described(api, HEAT, "T * F <= 3000", factors=swapped, rename={"T": "P", "P": "T"})
    assert [e["expression"] for e in after["constraints"]] == ["P + 25 * T <= 250", "P * F <= 3000"]
    assert [e["removes"] for e in after["constraints"]] == [e["removes"] for e in before["constraints"]]
    assert after["kept"] == before["kept"]


def test_a_limit_that_cannot_be_read_is_renamed_by_nobody(api):
    after = described(api, "T + (P <= 3", factors=[{**FACTORS[0], "name": "Temp"}, *FACTORS[1:]], rename={"T": "Temp"})
    assert after["constraints"][0]["expression"] == "T + (P <= 3"
    assert "cannot be read" in after["constraints"][0]["problem"]


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
