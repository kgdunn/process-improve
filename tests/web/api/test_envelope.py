"""The reply every entry point gives: ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": ...}``, with warnings.

The page reads each reply with ``JSON.parse`` and shows an error or a warning as it is, so
refusals come back as data, numbers the browser cannot read come back as ``null``, and what
the library warns about comes back with the result rather than in the browser's console.
"""

from __future__ import annotations

import json
import logging
import math
import warnings

import numpy as np
import pytest

from tests.web._support import FACTORS, factors_named


# --------------------------------------------------------------------------- the reply
def test_a_reply_is_a_result_or_an_error_and_always_its_warnings(api):
    done = api.reply("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    refused = api.reply("api_make_design", {"design_type": "rocket", "factors": FACTORS})
    assert set(done) == {"ok", "result", "warnings"}
    assert done["ok"] is True
    assert set(refused) == {"ok", "error", "warnings"}
    assert refused["ok"] is False
    assert isinstance(refused["error"], str)
    assert refused["warnings"] == []


def test_an_empty_payload_is_an_empty_request(api, bootstrap):
    """The worker sends ``"{}"`` when the page gives no payload; an empty string means the same."""
    expected = api.reply("api_catalogue", text="{}")
    assert api.reply("api_catalogue", text="") == expected
    assert json.loads(bootstrap.api_catalogue()) == expected


@pytest.mark.parametrize(
    ("fn", "payload"),
    [
        ("api_analyze", {}),  # KeyError: no workbook
        ("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "n_center_points": "x"}),  # ValueError
        ("api_make_design", {"design_type": "box_behnken", "factors": 5}),  # TypeError
    ],
    ids=["missing field", "text for a number", "wrong type"],
)
def test_a_request_the_page_could_not_have_sent_is_refused_not_raised(api, fn, payload):
    api.error(fn, payload)


@pytest.mark.parametrize("text", ["[1]", "3", '"design"', "null"])
def test_a_request_that_is_not_an_object_is_refused_in_a_sentence(api, text):
    assert api.reply("api_make_design", text=text) == {
        "ok": False,
        "error": "The page sent a request that is not a JSON object.",
        "warnings": [],
    }


def test_an_unexpected_error_comes_back_in_one_line(bootstrap):
    def breaks(_: dict) -> dict:
        message = "the solver gave up"
        raise RuntimeError(message)

    assert json.loads(bootstrap._reply(breaks, "{}")) == {
        "ok": False,
        "error": "Python stopped with RuntimeError: the solver gave up",
        "warnings": [],
    }


def test_a_refusal_keeps_what_was_heard_before_it(bootstrap):
    def warn_then_refuse(_: dict) -> dict:
        warnings.warn("heard first", UserWarning, stacklevel=1)
        message = "then refused"
        raise ValueError(message)

    assert json.loads(bootstrap._reply(warn_then_refuse, "{}")) == {
        "ok": False,
        "error": "then refused",
        "warnings": ["heard first"],
    }


# --------------------------------------------------------------------------- values the browser can read
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float64(1.5), 1.5),
        (np.int64(3), 3),
        (np.bool_(True), True),
        (math.nan, None),
        (np.float64(np.inf), None),
        (-math.inf, None),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        ((1, (2, np.nan)), [1, [2, None]]),
        ({1: np.float32(0.5), "a": [np.int8(1)]}, {"1": 0.5, "a": [1]}),
        ("text", "text"),
        (None, None),
    ],
    ids=["float", "int", "bool", "nan", "inf", "-inf", "array", "tuples", "dict keys", "text", "none"],
)
def test_results_are_made_of_values_json_parse_can_read(bootstrap, value, expected):
    cleaned = bootstrap._clean(value)
    assert cleaned == expected
    assert type(cleaned) is type(expected)
    json.dumps(cleaned, allow_nan=False)


# --------------------------------------------------------------------------- warnings
def test_library_warnings_reach_the_page(api):
    """An even-k DSD with only its own center run leaves the staged analysis no error df, and says so."""
    spec = {"design_type": "dsd", "factors": factors_named("A", "B", "C", "D"), "n_center_points": 1}
    assert any("degrees of freedom" in w for w in api.reply("api_make_design", spec)["warnings"])
    spec["n_center_points"] = 3
    assert api.reply("api_make_design", spec)["warnings"] == []


def test_what_is_heard_goes_to_the_reply_and_nowhere_else(bootstrap, caplog):
    """The library's logger and ``warnings.warn`` are both heard, and the logger is left as it was found."""
    logger = logging.getLogger("process_improve")
    before = (logger.handlers[:], logger.propagate)
    with caplog.at_level(logging.WARNING), bootstrap._listening() as heard:
        logger.warning("logged")
        warnings.warn("warned", UserWarning, stacklevel=1)
    assert heard == ["logged", "warned"]
    assert (logger.handlers, logger.propagate) == before
    assert "logged" not in caplog.text, "not also in the browser's console"
    logger.warning("after")
    assert "after" in caplog.text, "an enclosing log capture still works"


def test_warnings_heard_in_an_inner_block_are_passed_on_to_the_outer_one(bootstrap):
    """A fit heard on its own, to decide whether to refit, still reports what it heard to the call around it."""
    with bootstrap._listening() as outer:
        with bootstrap._listening() as inner:
            warnings.warn("heard inside", UserWarning, stacklevel=1)
        assert outer == [], "kept inside until passed on"
        bootstrap._relay(inner)
    assert outer == ["heard inside"]


def test_a_message_heard_twice_is_shown_once(bootstrap):
    """A library function can log a message with its name in front and warn the same text; the page shows it once."""

    def say_twice(_: dict) -> dict:
        logging.getLogger("process_improve").warning("analyze_experiment: no degrees of freedom left")
        warnings.warn("no degrees of freedom left", UserWarning, stacklevel=1)
        return {}

    assert json.loads(bootstrap._reply(say_twice, "{}"))["warnings"] == ["no degrees of freedom left"]


@pytest.mark.parametrize(
    ("filename", "foreign"),
    [
        ("/site-packages/numpy/linalg/_linalg.py", True),
        ("C:\\Python\\Lib\\site-packages\\scipy\\stats\\_stats_py.py", True),
        ("/site-packages/statsmodels/regression/linear_model.py", True),
        ("/site-packages/process_improve/experiments/analysis.py", False),
        ("/home/reader/numpy_notes.py", False),
    ],
)
def test_warnings_from_the_numerical_packages_are_left_out(bootstrap, filename, foreign):
    """A saturated fit's division by zero inside numpy is not news for the reader; the library's own warnings are."""
    assert bootstrap._foreign(filename) is foreign
