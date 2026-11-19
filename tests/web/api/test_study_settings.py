"""A study's settings in its workbook: ``api_workbook`` keeps them and ``api_read_workbook`` gives them back.

The settings are the study's own: its colour map, its decimal mark, its significance level. The study's link and
file carry them (web/campaign.js), and so does its workbook, checked as ``studySettings`` checks them.
"""

from __future__ import annotations

import pytest

SETTINGS = {"colour_map": "lajolla", "colour_map_flipped": True, "decimal_mark": "comma", "significance_level": 0.01}


def test_a_workbook_gives_back_the_settings_it_was_written_with(api, held, simulated):
    book = api.ok("api_workbook", {**held, "responses": simulated["responses"], "settings": SETTINGS})
    assert api.ok("api_read_workbook", book)["settings"] == SETTINGS


def test_a_study_with_no_settings_has_none_in_its_workbook(api, held, simulated):
    for none in ({}, {"settings": {}}, {"settings": None}):
        book = api.ok("api_workbook", {**held, "responses": simulated["responses"], **none})
        assert "settings" not in api.ok("api_read_workbook", book)


def test_a_setting_of_a_later_page_is_kept_as_it_came(api, held):
    later = {"digits_shown": 4, "a_look_of_a_later_page": "wide"}
    assert api.ok("api_read_workbook", api.ok("api_workbook", {**held, "settings": later}))["settings"] == later


@pytest.mark.parametrize(
    "settings",
    [
        ["lajolla"],
        {"Colour Map": "lajolla"},
        {"colour_map": {"name": "lajolla"}},
        {"colour_map": "x" * 65},
        {"digits": float("inf")},
        {f"s{'a' * i}": 1 for i in range(33)},
        {"significance_level": 0.02},
        {"significance_level": True},
    ],
)
def test_settings_the_page_does_not_read_are_refused(api, held, settings):
    refusal = api.error("api_workbook", {**held, "settings": settings})
    assert "setting" in refusal or "significance level" in refusal
