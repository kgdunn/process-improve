"""The reference catalogue, reproduced with this library.

``process_improve.experiments.reference_catalogue`` holds worked examples, exercises and case studies of
published sources, each with its runs as printed and every number the source prints for it, with
where it is printed. ``reproduce_reference_catalogue`` recomputes every number that carries a check; here
each one must agree with the printed text to within the source's own rounding. A number that is
not recomputed must say why. A change to the library that contradicts a published source fails
here, and so does a number copied wrongly into the catalogue.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import reference_catalogue
from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.reference_catalogue import (
    reference_designs,
    reference_experiments,
    reference_sources,
    reference_topics,
    reproduce_reference_catalogue,
)

CHECKS = list(reference_catalogue._each_check())
FITS = reference_catalogue._Fits()
SEARCHED = {"omars_ilp"}  # designs built by search take seconds each: check what they would show more cheaply


def _case(key: str, entry: dict, expected: dict, check: dict) -> pytest.param:
    name = f"{key}:{entry['name']}:{expected['what']}:{expected.get('term')}"[:120]
    return pytest.param(entry, expected, check, id=name)


@pytest.mark.parametrize(("entry", "expected", "check"), [_case(*c) for c in CHECKS if "kind" in c[3]])
def test_every_printed_number_is_reproduced(entry, expected, check):
    found, agrees = reference_catalogue._reproduce(entry, expected, check, FITS)
    printed = check.get("printed", expected["value"])
    if check.get("disagreement"):
        assert not agrees, f"{entry['name']}: it reproduces now ({found}); take away the disagreement it records"
        return
    assert agrees, f"{entry['name']}: {expected['what']} {expected.get('term')}: printed {printed}, found {found}"


def test_no_check_builds_a_design_by_search():
    """A run count of the OMARS frontier comes from omars_minimum_runs, not from building the design (seconds each)."""
    searched = [
        entry["name"] for *_, entry, _, check in CHECKS if (check.get("design") or {}).get("design_type") in SEARCHED
    ]
    assert not searched, f"these checks build a design by search; use a library_value check instead: {searched}"


def test_every_number_not_recomputed_says_why():
    reasons = [check.get("manual") for *_, check in CHECKS if "kind" not in check]
    assert all(reasons), "a number with no check says why it is not recomputed"
    assert "TO CHECK" not in reasons, "every number is given a check or the reason it has none"


def test_the_report_has_a_row_for_every_check():
    report = reproduce_reference_catalogue("ullmann-doe-2012")
    assert len(report) == sum(key == "ullmann-doe-2012" for key, *_ in CHECKS)
    assert report.loc[report["checked"], "agrees"].all()


def test_every_experiment_says_where_it_was_printed():
    sources = reference_sources()
    for entry in [*reference_experiments(), *reference_designs()]:
        name = entry["name"]
        assert entry["source"] in sources, name
        assert entry["where"], name
        assert entry["expected"] or entry.get("unanswered"), f"{name}: what the source prints is part of the entry"
        for expected in entry["expected"]:
            where = expected["where"]
            assert where.get("quote"), f"{name}: every number carries a quotation of where it is printed"
            assert where.get("line") or where.get("page") or where.get("question"), (
                f"{name}: every number carries its line, page or quiz question"
            )
            if where.get("option_id"):
                assert where.get("answer_key") in ("correct", "incorrect"), f"{name}: an option says how it is keyed"
    for experiment in reference_experiments():
        assert experiment.get("runs") or experiment.get("data_source"), f"{experiment['name']}: no runs, no source"


def test_each_name_is_one_entry():
    names = [e["name"] for e in [*reference_experiments(), *reference_designs()]]
    assert len(set(names)) == len(names)


def test_the_sources_are_cited_in_full():
    for key, source in reference_sources().items():
        assert source["authors"], key
        assert all(source.get(field) for field in ("title", "url", "year", "read", "rights")), key


def test_the_terms_of_the_sources_are_stated_beside_them():
    """The MIT licence covers the code; the notice beside the data says it does not cover the sources' contents."""
    notice = (reference_catalogue._CATALOGUE_DIR / "NOTICE.md").read_text(encoding="utf-8")
    assert "does not extend" in notice
    assert "source.rights" in notice


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="Unknown source 'nowhere'"):
        reference_experiments("nowhere")


def test_every_entry_says_what_it_is_about_in_the_vocabulary():
    """One or two topics an entry, the main one first, each from the vocabulary; the vocabulary stays 20 to 40 long."""
    topics = reference_topics()
    assert 20 <= len(topics) <= 40, "the vocabulary is kept between 20 and 40 topics"
    used = set()
    for entry in [*reference_experiments(), *reference_designs()]:
        listed = entry["topics"]
        assert 1 <= len(listed) <= 2, f"{entry['name']}: one or two topics"
        assert len(set(listed)) == len(listed), f"{entry['name']}: a topic is listed once"
        assert set(listed) <= set(topics), f"{entry['name']}: {set(listed) - set(topics)} is not in topics.json"
        used.update(listed)
    assert used == set(topics), f"no entry is about {set(topics) - used}: drop the topic or tag an entry with it"


def test_each_topic_names_its_classes_in_isic_and_ford():
    for key, topic in reference_topics().items():
        assert topic["label"], key
        assert topic["description"], key
        assert all(re.fullmatch(r"[A-U]|\d{2,3}", code) for code in topic["isic"]), f"{key}: {topic['isic']}"
        assert (topic["ford"] is None) == (topic["ford_field"] is None), f"{key}: FORD {topic['ford']} is named"


def test_entries_are_selected_by_topic():
    paper = reference_experiments(topic="paper")
    assert paper
    assert all("paper" in e["topics"] for e in paper)
    second = [e["name"] for e in reference_experiments(topic="pharmaceuticals") if e["topics"][0] != "pharmaceuticals"]
    assert "davies-penicillin-2-5" in second, "a second topic selects too"
    assert all(d["topics"] == ["general"] for d in reference_designs("goos-2025", topic="general"))
    with pytest.raises(ValueError, match="Unknown topic 'astrology'"):
        reference_designs(topic="astrology")


def test_a_wrong_number_is_reported_and_not_agreed(monkeypatch, tmp_path):
    """The report says so when a printed number does not reproduce: a check that cannot fail checks nothing."""
    from process_improve.experiments import reference_catalogue

    source = json.loads(next(reference_catalogue._CATALOGUE_DIR.glob("ullmann-*.json")).read_text(encoding="utf-8"))
    source["experiments"][0]["expected"][0]["value"] = "11"  # the effect of A is 10
    (tmp_path / "wrong.json").write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(reference_catalogue, "_CATALOGUE_DIR", tmp_path)
    report = reproduce_reference_catalogue()
    first = report.iloc[0]
    assert (first["printed"], first["found"], first["agrees"]) == ("11", "10", False)


def test_each_kind_of_check_reports_what_it_finds(monkeypatch, tmp_path):
    """A small source with one check of each remaining kind: ANOVA, bounds, a design's metrics, and errors."""
    from process_improve.experiments import reference_catalogue

    def item(value, check):
        return {"what": "x", "term": None, "value": value, "where": {"page": 1, "quote": "x"}, "check": check}

    fit = {"model": "A + B"}
    runs = {"columns": ["A", "B", "y"], "rows": [[-1, -1, 1], [1, -1, 3], [-1, 1, 2], [1, 1, 6]], "coded": True}
    bbd = {"design_type": "box_behnken", "k": 3, "options": {"n_center_points": 3}}
    expected = [
        item("9", {"kind": "anova_sum_sq", "term": "A", **fit}),  # 4 x 1.5 squared
        item("< 0.5", {"kind": "p_value", "term": "A", **fit}),  # t = 3 on 1 df: p = 0.205
        item("> 0.1", {"kind": "p_value", "term": "A", **fit}),
        item("1", {"kind": "design_metric", "metric": "degrees_of_freedom", "key": "residual", **fit}),
        item("1", {"kind": "design_metric", "metric": "vif", "key": "max", **fit}),
        item(
            "1",
            {
                "kind": "design_metric",
                "metric": "correlation",
                "key": "matrix.0.0",
                "model": "quadratic",
                "design": bbd,
            },
        ),
        item(
            "0.071",
            {"kind": "design_metric", "metric": "correlation", "key": "max_abs_r", "model": "quadratic", "design": bbd},
        ),
        item("21", {"kind": "library_value", "function": "omars_minimum_runs", "args": [4]}),
        item("2", {"kind": "no_such_kind"}),
        item("2", {"kind": "library_value", "function": "no_such_function"}),
        item("2", {"kind": "coefficient", "term": "C", **fit}),
        item("2 ** 3", {"kind": "coefficient", "term": "A", "expression": True, **fit}),
        item("about two", {"kind": "coefficient", "term": "A", **fit}),
    ]
    experiment = {
        "name": "small",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in "AB"],
        "responses": [{"name": "y"}],
        "runs": runs,
        "expected": expected,
    }
    source = {"key": "small", "authors": ["A"], "title": "t", "url": "u", "year": 2026, "read": "r"}
    (tmp_path / "small.json").write_text(json.dumps({"source": source, "experiments": [experiment]}), encoding="utf-8")
    monkeypatch.setattr(reference_catalogue, "_CATALOGUE_DIR", tmp_path)
    report = reproduce_reference_catalogue("small")
    assert report["agrees"].tolist() == [True] * 8 + [False] * 5
    assert report["found"].iloc[6] == "0.0714286"
    reasons = report["reason"].iloc[8:].str.split(":").str[0].tolist()
    assert reasons == ["KeyError", "KeyError", "KeyError", "ValueError", "ValueError"]


def test_a_one_sided_p_value_is_half_the_two_sided_on_the_side_of_the_estimate():
    """A source testing one direction quotes half the two-sided p-value, or one less that half."""
    from process_improve.experiments.reference_catalogue import _p_value

    row = {"coefficient": 0.65, "p_value": 0.39}
    assert _p_value(row, None) == 0.39
    assert _p_value(row, "greater") == pytest.approx(0.195)
    assert _p_value(row, "less") == pytest.approx(0.805)


def test_a_tiny_printed_number_is_held_to_its_own_digits():
    """The allowance for floating-point error never swamps the half unit of a number like 0.798e-8."""
    from process_improve.experiments.reference_catalogue import _agrees

    assert _agrees(7.977e-9, "0.798e-8")
    assert not _agrees(7.9e-9, "0.798e-8")


def test_a_trailing_zero_is_a_printed_digit():
    """0.40 is printed to hundredths: 0.37 does not round to it, though it rounds to 0.4."""
    from process_improve.experiments.reference_catalogue import _agrees

    assert not _agrees(0.37, "0.40")
    assert _agrees(0.37, "0.4")
    assert _agrees(0.404, "0.40")


def test_a_blocking_arrangement_is_compared_by_how_many_interactions_of_each_order_it_confounds():
    """The factors of a full factorial are interchangeable: 124, 134, 23 is as good as the library's ABC, ABD, CD."""
    from process_improve.experiments.reference_catalogue import _block_confounding

    check = {"design": {"design_type": "full_factorial", "k": 4, "options": {"n_blocks": 4}}}
    assert _block_confounding(check, "124, 134, 23")[1]
    assert not _block_confounding(check, "12, 34, 1234")[1]  # two two-factor interactions where one will do


def test_a_difference_of_two_rounded_averages_carries_both_roundings():
    """.695 - .567 = .128, from averages of .69450 and .56725: a half unit each, a whole unit in their difference."""
    from process_improve.experiments.reference_catalogue import _agrees

    assert not _agrees(0.12725, ".128")
    assert _agrees(0.12725, ".128", of_rounded=2)
    assert not _agrees(0.1262, ".128", of_rounded=2)


def test_a_difference_of_rounded_averages_worked_without_a_fit_carries_both_roundings_too():
    """Averages 9.35663 and 9.35987, printed 9.3566 and 9.3599, differ by .00324; the source prints .0033."""
    rows = [[1, 9.35663], [1, 9.35663], [2, 9.35987], [2, 9.35987]]
    runs = {"columns": ["level", "y"], "rows": rows}
    entry = {"name": "pair", "factors": [], "responses": [{"name": "y"}], "runs": runs}
    check = {"kind": "difference_of_means", "runs": [3, 4], "minus": [1, 2]}
    fits = reference_catalogue._Fits()
    assert not reference_catalogue._reproduce(entry, {"value": ".0033"}, check, fits)[1]
    assert reference_catalogue._reproduce(entry, {"value": ".0033"}, {**check, "of_rounded": 2}, fits)[1]


def test_a_regressor_marked_uncoded_is_fitted_in_its_own_units():
    """A source's slope is per unit of x: a factor with ``"coding": "none"`` is not rescaled to -1 and +1."""
    from process_improve.experiments.reference_catalogue import _coded_runs

    def factor(coding):
        return {"name": "x", "type": "numeric", "low": 10, "high": 30, **coding}

    entry = {
        "responses": [{"name": "y"}],
        "runs": {"columns": ["x", "y"], "rows": [[10, 1], [20, 2], [30, 3]], "coded": False},
    }
    coded, _ = _coded_runs({**entry, "factors": [factor({})]}, None)
    own, _ = _coded_runs({**entry, "factors": [factor({"coding": "none"})]}, None)
    assert coded["x"].tolist() == [-1.0, 0.0, 1.0]
    assert own["x"].tolist() == [10.0, 20.0, 30.0]


def _replicated_2x2(readings):
    """Return a 2^2 factorial in coded units, each run read twice: the columns as a source prints them."""
    rows = [[a, b, *r] for (a, b), r in zip([(-1, -1), (1, -1), (-1, 1), (1, 1)], readings, strict=True)]
    return {
        "name": "replicated",
        "factors": [{"name": n, "type": "numeric", "low": 10, "high": 20} for n in "AB"],
        "responses": [{"name": "y1"}, {"name": "y2"}],
        "runs": {"columns": ["A", "B", "y1", "y2"], "rows": rows, "coded": True},
    }


def test_a_run_read_several_times_is_analysed_through_its_mean_or_standard_deviation():
    """Each run's statistic of its readings is the response: its effects are those of the column a source works out."""
    from process_improve.experiments.reference_catalogue import _coded_runs

    entry = _replicated_2x2([[1, 3], [2, 2.5], [4, 4], [6, 9]])
    summary = {"of": ["y1", "y2"], "statistic": "std_dev"}
    _, s = _coded_runs(entry, None, summary)
    assert s["std_dev"].tolist() == pytest.approx([2**0.5, 0.125**0.5, 0, 4.5**0.5])
    _, rounded = _coded_runs(entry, [1, 2], {**summary, "round": 2})
    assert rounded["std_dev"].tolist() == [1.41, 0.35]
    fits = reference_catalogue._Fits()
    mean_check = {"kind": "effect", "term": "A", "model": "A * B", "summary": {**summary, "statistic": "mean"}}
    s_check = {"kind": "effect", "term": "A", "model": "A * B", "summary": summary}
    assert reference_catalogue._reproduce(entry, {"value": "1.875"}, mean_check, fits) == ("1.875", True)
    expected_s = (0.125**0.5 + 4.5**0.5 - 2**0.5) / 2
    found, agrees = reference_catalogue._reproduce(entry, {"value": f"{expected_s:.6f}"}, s_check, fits)
    assert agrees, found


def test_the_setting_that_reaches_a_target_is_solved_from_the_fitted_model():
    """The fit is 3 + A + 2B + AB; at A = +1 it is 4 + 3B, so 5.5 needs B = 0.5 coded, 17.5 in its own units."""
    entry = _replicated_2x2([[1, 1], [1, 1], [3, 3], [7, 7]])
    fits = reference_catalogue._Fits()
    check = {"kind": "target_setting", "model": "A * B", "response": "y1", "solve": "B", "target": 5.5}
    at_a_high = {**check, "point": {"A": 1}}
    assert reference_catalogue._reproduce(entry, {"value": "0.5"}, at_a_high, fits) == ("0.5", True)
    actual = {**at_a_high, "units": "actual"}
    assert reference_catalogue._reproduce(entry, {"value": "17.5"}, actual, fits) == ("17.5", True)
    flat = _replicated_2x2([[1, 1], [3, 3], [1, 1], [3, 3]])  # B has no effect: no setting of it reaches the target
    with pytest.raises(ValueError, match="does not change with B"):
        reference_catalogue._reproduce(flat, {"value": "0"}, check, reference_catalogue._Fits())


def test_on_a_parabola_the_setting_nearer_the_centre_is_taken():
    """The fit (1 + A)^2 reaches 2.25 at A = 0.5 and at A = -2.5; the first lies within the design."""
    entry = {
        "name": "parabola",
        "factors": [{"name": "A", "type": "numeric", "low": 10, "high": 20}],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["A", "y"], "rows": [[-1, 0], [0, 1], [1, 4]], "coded": True},
    }
    check = {"kind": "target_setting", "model": "A + I(A ** 2)", "solve": "A", "target": 2.25}
    assert reference_catalogue._reproduce(entry, {"value": "0.5"}, check, reference_catalogue._Fits()) == ("0.5", True)
    out_of_reach = {**check, "target": -1}
    with pytest.raises(ValueError, match="No setting of A reaches -1"):
        reference_catalogue._reproduce(entry, {"value": "0"}, out_of_reach, reference_catalogue._Fits())


def test_the_rounding_of_the_data_is_carried_into_a_coefficient_only():
    """A coefficient is a weighted sum of the responses, so their rounding bounds it; no other statistic gets it."""
    entry = _replicated_2x2([[1, 1], [1, 1], [3, 3], [7, 7]])
    check = {"kind": "r_squared", "model": "A + B", "response": "y1", "data_rounding": {"decimals": 1}}
    with pytest.raises(ValueError, match="carried into a coefficient, not a 'r_squared'"):
        reference_catalogue._reproduce(entry, {"value": "0.9"}, check, reference_catalogue._Fits())


def test_each_reading_of_a_replicated_run_can_be_an_observation_of_its_own():
    """Four runs read twice are eight observations, a blank reading left out."""
    from process_improve.experiments.reference_catalogue import _coded_runs

    entry = _replicated_2x2([[1, 3], [2, None], [4, 4], [6, 9]])
    settings, y = _coded_runs(entry, None, {"of": ["y1", "y2"], "statistic": "each"})
    assert y["each"].tolist() == [1, 3, 2, 4, 4, 6, 9]
    assert settings["A"].tolist() == [-1, -1, 1, -1, -1, 1, 1]


def test_a_capability_index_is_the_library_cpk_on_the_plain_mean_and_standard_deviation():
    """Mean 20 and standard deviation 1.298 against a lower specification of 17: Cpk = 3 / (3 x 1.298)."""
    from process_improve.experiments.reference_catalogue import _capability

    weeks = [21, 19, 20, 22, 18, 21, 20, 20, 18, 20, 19, 22, 20, 21, 19, 20, 19, 21, 18, 22]
    entry = {"factors": [], "responses": [{"name": "y"}], "runs": {"columns": ["y"], "rows": [[w] for w in weeks]}}
    sd = (32 / 19) ** 0.5
    assert _capability(entry, {"lower": 17}) == pytest.approx(3 / (3 * sd))
    assert _capability(entry, {"lower": 17, "upper": 22}) == pytest.approx(2 / (3 * sd))


def _printed_design(columns, rows):
    """Return a design entry holding a printed coded matrix, its factors named by its columns."""
    factors = [{"name": c, "type": "numeric", "low": -1, "high": 1} for c in columns]
    return {"name": "printed", "factors": factors, "responses": [], "runs": {"columns": columns, "rows": rows}}


HALF_FRACTION = [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]]  # C = AB, in another run order


def test_a_printed_matrix_is_compared_with_the_library_design_whatever_the_run_order():
    from process_improve.experiments.reference_catalogue import _same_runs

    entry = _printed_design(["A", "B", "C"], HALF_FRACTION)
    fraction = {
        "design": {
            "design_type": "fractional_factorial",
            "k": 3,
            "options": {"generators": ["C=AB"], "n_center_points": 0},
        }
    }
    assert _same_runs(entry, fraction, "4") == ("4 of 4 printed runs, of 4 the library's", True)
    assert not _same_runs(entry, fraction, "5")[1]  # a printed run count must be the one compared
    other = {"design": {**fraction["design"], "options": {"generators": ["C=-AB"], "n_center_points": 0}}}
    assert _same_runs(entry, other, "4") == ("0 of 4 printed runs, of 4 the library's", False)


def test_two_constructions_of_one_design_agree_once_columns_are_reordered_and_signs_changed():
    """With B and C swapped and a sign changed, the runs are the fraction C = -AB; a sign change makes it C = AB."""
    from process_improve.experiments.reference_catalogue import _same_runs

    swapped = [[a, -c, b] for a, b, c in HALF_FRACTION]
    entry = _printed_design(["A", "B", "C"], swapped)
    fraction = {
        "design": {
            "design_type": "fractional_factorial",
            "k": 3,
            "options": {"generators": ["C=AB"], "n_center_points": 0},
        }
    }
    assert not _same_runs(entry, fraction, "4")[1]
    assert not _same_runs(entry, {**fraction, "relabel": "columns"}, "4")[1]
    relabelled = _same_runs(entry, {**fraction, "relabel": "columns and signs"}, "4")
    assert relabelled == ("the library's columns 1, 2, -3", True)  # the first relabelling found; 1, -3, 2 also works


def test_a_printed_alpha_is_the_axial_distance_of_the_library_design():
    from process_improve.experiments.reference_catalogue import _same_runs

    cube = [[a, b] for a in (-1, 1) for b in (-1, 1)]
    alpha = "\u03b1"  # the axial distance, as a source prints it
    axial = [[alpha, 0], ["-" + alpha, 0], [0, alpha], [0, "-" + alpha]]
    entry = _printed_design(["A", "B"], [*cube, [0, 0], *axial])
    ccd = {"design": {"design_type": "ccd", "k": 2, "options": {"n_center_points": 1, "alpha": "rotatable"}}}
    assert _same_runs(entry, ccd, "9")[1]


def test_a_stored_array_can_have_a_column_written_as_two():
    """The six levels of the first column of L18(6^1 3^6) are a two-level and a three-level column."""
    from process_improve.experiments.reference_catalogue import _array_runs

    split = _array_runs({"array": "L18(6^1 3^6)", "split": {"0": [2, 3]}})
    assert split.shape == (18, 8)
    assert sorted(set(split[:, 0])) == [-1, 1]
    assert sorted(set(split[:, 1])) == [-1, 0, 1]
    assert len({tuple(row) for row in split[:, :2]}) == 6  # each pair of levels once in every three runs


def test_a_printed_matrix_obeys_its_generators_in_every_run_or_says_where_not():
    from process_improve.experiments.reference_catalogue import _generator_holds

    entry = _printed_design(["A", "B", "C"], HALF_FRACTION)
    assert _generator_holds(entry, {}, "C = AB") == ("every run", True)
    assert _generator_holds(entry, {}, "C = A \u00d7 B, C = -AB") == ("not in runs 1, 2, 3, 4", False)
    runs = {"runs": [1, 2]}
    flipped = _printed_design(["A", "B", "C"], [*HALF_FRACTION[:2], [1, -1, 1], [0, 0, 0]])
    assert _generator_holds(flipped, runs, "C = AB") == ("every run", True)
    assert _generator_holds(flipped, {}, "C = AB") == ("not in runs 3, 4", False)


def test_the_dot_product_of_two_columns_is_zero_when_they_are_orthogonal():
    from process_improve.experiments.reference_catalogue import _dot_product

    entry = _printed_design(["A", "B", "C"], [[1, 1, 1], [1, -1, -1], [-1, 1, 1], [-1, -1, -1]])
    assert _dot_product(entry, {"columns": ["B", "C"]}) == 4
    assert _dot_product(entry, {"columns": ["A", "B"]}) == 0


def test_a_design_of_three_level_factors_is_made_from_its_levels():
    from process_improve.experiments.reference_catalogue import _design_property

    spec = {"design_type": "full_factorial", "k": 3, "levels": 3, "options": {"n_center_points": 0}}
    assert _design_property({"property": "n_runs", "design": spec}) == 27
    mixed = {"design_type": "taguchi", "k": 4, "levels": [2, 3, 3, 3]}
    assert _design_property({"property": "n_runs", "design": mixed}) == 18


def _line(ys):
    """Return a regression entry: y against x in its own units, x at 1, 2, 3, 4."""
    rows = [[x, y] for x, y in zip((1, 2, 3, 4), ys, strict=True)]
    return {
        "name": "line",
        "factors": [{"name": "x", "type": "numeric", "low": 1, "high": 4, "coding": "none"}],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["x", "y"], "rows": rows, "coded": False},
    }


def test_a_prediction_interval_limit_and_the_multiple_correlation_are_read_from_the_fit():
    """The interval is symmetric about the prediction; R is the square root of R², as a source prints Multiple R."""
    entry = _line([1.0, 2.5, 2.5, 4.0])  # y = 1 x exactly at the ends, R² = 0.9
    fits = reference_catalogue._Fits()
    check = {"kind": "prediction", "model": "x", "point": {"x": 2.5}}
    out = fits(entry, check)
    middle, low, high = (
        reference_catalogue._prediction(out, {**check, "bound": b}, entry) for b in ("predicted", "pi_low", "pi_high")
    )
    assert middle == pytest.approx(2.5)
    assert low < middle
    assert middle - low == pytest.approx(high - middle)
    r = reference_catalogue._reproduce(entry, {"value": "0.949"}, {"kind": "multiple_r", "model": "x"}, fits)
    assert r == ("0.948683", True)


def test_the_squared_correlation_of_a_term_with_the_others_is_one_less_one_over_its_vif():
    """1, 2, 3 and their squares are 97.96% collinear; -1, 0, +1 and theirs not at all."""
    from process_improve.experiments.reference_catalogue import _design_metric

    def entry(levels, coding):
        factor = {"name": "A", "type": "numeric", "low": min(levels), "high": max(levels), **coding}
        rows = [[v] for v in levels for _ in range(2)]
        return {"factors": [factor], "responses": [], "runs": {"columns": ["A"], "rows": rows, "coded": False}}

    check = {"metric": "vif", "model": "A + I(A ** 2)", "key": "A", "reciprocal": True, "complement": True}
    assert _design_metric(entry([1, 2, 3], {"coding": "none"}), check) == pytest.approx(0.979592, abs=1e-6)
    assert _design_metric(entry([-1, 0, 1], {}), check) == pytest.approx(0, abs=1e-12)


def test_a_term_with_a_squared_factor_takes_the_library_name():
    from process_improve.experiments.reference_catalogue import _library_term

    assert _library_term("x1²·x2") == "I(x1 ** 2):x2"
    assert _library_term("A·B") == "A:B"
    assert _library_term("C²") == "I(C ** 2)"


def test_a_run_is_summarised_by_ln_s_and_by_taguchi_signal_to_noise_ratios():
    """Readings 20 and 30: ln s = 1.956, S/NL = 27.43 dB, S/NS = -28.13 dB and S/NN = 10 log10(625/50) = 10.97 dB."""
    from process_improve.experiments.reference_catalogue import _summarised

    table = pd.DataFrame({"y1": [20.0], "y2": [30.0]})
    found = {
        stat: float(_summarised(table, {"of": ["y1", "y2"], "statistic": stat}).iloc[0, 0])
        for stat in ("ln_std_dev", "sn_larger", "sn_smaller", "sn_nominal")
    }
    assert found["ln_std_dev"] == pytest.approx(math.log(50**0.5))
    assert found["sn_larger"] == pytest.approx(-10 * math.log10((1 / 400 + 1 / 900) / 2))
    assert found["sn_smaller"] == pytest.approx(-10 * math.log10((400 + 900) / 2))
    assert found["sn_nominal"] == pytest.approx(10 * math.log10(625 / 50))


def test_a_dispersion_effect_is_twice_the_log_ratio_of_the_spreads_at_the_two_levels():
    """Standard deviations 1 at A = -1 and 2 at A = +1: ln(s+² / s-²) = 2 ln 2."""
    from process_improve.experiments.reference_catalogue import _log_variance_ratio

    entry = _replicated_2x2([[0, 2**0.5], [0, 8**0.5], [0, 2**0.5], [0, 8**0.5]])  # s = 1, 2, 1, 2
    summary = {"of": ["y1", "y2"], "statistic": "std_dev"}
    assert _log_variance_ratio(entry, {"runs": [2, 4], "minus": [1, 3], "summary": summary}) == pytest.approx(
        2 * math.log(2)
    )


def test_the_spread_of_the_residuals_at_one_level_can_be_worked_from_rounded_residuals():
    """A source working from its printed residuals rounds them first; the ratio of two levels' spreads does too."""
    entry = _line([1.0, 2.1, 2.9, 4.0])
    fits = reference_catalogue._Fits()
    out = fits(entry, {"model": "x"})
    residuals = reference_catalogue._residuals(out)[1]
    exact = reference_catalogue._residual_std_dev(out, [1, 2])
    assert exact == pytest.approx(float(np.std(residuals[:2], ddof=1)))
    rounded = reference_catalogue._residual_std_dev(out, [1, 2], 1)
    assert rounded == pytest.approx(float(np.std(np.round(residuals[:2], 1), ddof=1)))
    ratio = reference_catalogue._FIT_KINDS["residual_log_variance_ratio"](out, {"at": [1, 2], "versus": [3, 4]}, entry)
    assert ratio == pytest.approx(2 * math.log(exact / reference_catalogue._residual_std_dev(out, [3, 4])))


def test_a_matrix_printed_in_the_factors_own_units_is_compared_in_coded_units():
    from process_improve.experiments.reference_catalogue import _same_runs

    rows = [[t, n] for t in (4, 12) for n in (10, 18)]
    entry = {
        "factors": [
            {"name": "T", "type": "numeric", "low": 4, "high": 12},
            {"name": "N", "type": "numeric", "low": 10, "high": 18},
        ],
        "responses": [],
        "runs": {"columns": ["T", "N"], "rows": rows, "coded": False},
    }
    full = {"design": {"design_type": "full_factorial", "k": 2, "options": {"n_center_points": 0}}}
    assert _same_runs(entry, full, "4") == ("4 of 4 printed runs, of 4 the library's", True)


def test_an_axial_distance_printed_rounded_matches_once_both_matrices_are_rounded():
    from process_improve.experiments.reference_catalogue import _same_runs

    cube = [[a, b] for a in (-1, 1) for b in (-1, 1)]
    axial = [[1.414, 0], [-1.414, 0], [0, 1.414], [0, -1.414]]  # the square root of 2, as a source prints it
    entry = _printed_design(["A", "B"], [*cube, [0, 0], *axial])
    ccd = {"design": {"design_type": "ccd", "k": 2, "options": {"n_center_points": 1, "alpha": "rotatable"}}}
    assert not _same_runs(entry, ccd, "9")[1]
    assert _same_runs(entry, {**ccd, "round": 3}, "9")[1]


def test_a_printed_design_may_use_only_some_columns_of_a_stored_array():
    """Two factors on an L4 take two of its three columns; the third is left for their interaction."""
    from process_improve.experiments.reference_catalogue import _same_runs

    l4 = {"array": "L4(2^3)", "relabel": "columns"}
    entry = _printed_design(["A", "B"], [[a, b] for a in (-1, 1) for b in (-1, 1)])
    assert _same_runs(entry, l4, "4") == ("the library's columns 1, 2", True)
    repeated = _printed_design(["A", "B"], [[-1, -1], [-1, -1], [1, -1], [1, 1]])
    assert not _same_runs(repeated, l4, "4")[1]


def _plane(a, b):
    """Return a 2^2 with a centre point, y = 10 + a x1 + b x2 exactly, x1 from 100 to 200 and x2 from 125 to 175."""
    runs = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
    return {
        "name": "plane",
        "factors": [
            {"name": "x1", "type": "numeric", "low": 100, "high": 200},
            {"name": "x2", "type": "numeric", "low": 125, "high": 175},
        ],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["x1", "x2", "y"], "rows": [[u, v, 10 + a * u + b * v] for u, v in runs], "coded": True},
    }


def test_a_step_on_the_path_of_steepest_ascent_can_be_read_in_coded_units():
    """A step of -25 in x2 is -1 coded; along the gradient (3, -1), x1 then moves +3 coded, 150 in its own units."""
    entry = _plane(3, -1)
    check = {"kind": "steepest_path", "model": "x1 + x2", "per": "x2", "change": -25, "steps": 1, "key": "x1"}
    out = reference_catalogue._Fits()(entry, check)
    assert reference_catalogue._steepest_path(out, {**check, "units": "coded"}, entry) == pytest.approx(3.0)
    assert reference_catalogue._steepest_path(out, check, entry) == pytest.approx(300.0)
    from_centre = {**check, "units": "coded", "from_centre": True, "steps": 2}
    assert reference_catalogue._steepest_path(out, from_centre, entry) == pytest.approx(6.0)


def test_a_factor_left_out_of_a_prediction_point_sits_at_its_centre():
    entry = _plane(2, 1)
    check = {"kind": "prediction", "model": "x1", "point": {"x1": 1}}
    out = reference_catalogue._Fits()(entry, check)
    assert reference_catalogue._prediction(out, check, entry) == pytest.approx(12.0)


def test_a_statistic_can_be_worked_in_other_units_than_the_readings_are_printed_in():
    """Readings printed in units of 10^17: ln s in the readings' own units is ln s printed plus ln 10^17."""
    from process_improve.experiments.reference_catalogue import _summarised

    table = pd.DataFrame({"y1": [0.70, 2.0], "y2": [0.66, 1.6]})
    plain = _summarised(table, {"of": ["y1", "y2"], "statistic": "ln_std_dev"})
    scaled = _summarised(table, {"of": ["y1", "y2"], "statistic": "ln_std_dev", "scale": 1e17})
    assert scaled.iloc[:, 0].to_numpy() == pytest.approx(plain.iloc[:, 0].to_numpy() + math.log(1e17))
    assert scaled.iloc[0, 0] == pytest.approx(35.578, abs=5e-4)  # as Schmidt and Launsby's GaAs case study prints it


def test_a_variance_component_is_read_from_the_library_reml_fit():
    """Balanced, REML gives the analysis-of-variance estimates: the pooled variance within groups, 1.25, and between."""
    from process_improve.experiments.reference_catalogue import _variance_component

    rows = [["a", 1], ["a", 3], ["b", 4], ["b", 6], ["c", 1], ["c", 2], ["d", 5], ["d", 6]]
    entry = {"responses": [{"name": "y"}], "runs": {"columns": ["group", "y"], "rows": rows}}
    check = {"groups": ["group"]}
    assert _variance_component(entry, {**check, "key": "residual"}) == pytest.approx(1.25)
    between = (sum((sum(r[1] for r in rows[i : i + 2]) / 2 - 3.5) ** 2 for i in range(0, 8, 2)) / 3) - 1.25 / 2
    assert _variance_component(entry, {**check, "key": "group"}) == pytest.approx(between, abs=1e-6)


def test_a_confidence_printed_as_one_less_the_p_value_is_checked_as_its_complement():
    """Design Cube prints 1 - p as its "Probability"; a fit's check with ``complement`` reads it that way."""
    entry = _replicated_2x2([(10, 12), (20, 21), (11, 13), (22, 20)])
    check = {"kind": "anova_p", "term": "A", "model": "A * B", "summary": {"of": ["y1", "y2"], "statistic": "each"}}
    fits = reference_catalogue._Fits()
    p = float(reference_catalogue._reproduce(entry, {"value": "0"}, check, fits)[0])
    confidence = {"value": f"{1 - p:.4f}"}
    found, agrees = reference_catalogue._reproduce(entry, confidence, {**check, "complement": True}, fits)
    assert agrees
    assert float(found) == pytest.approx(1 - p, abs=1e-6)


def test_the_total_of_a_runs_readings_is_what_yates_algorithm_takes():
    """Schmidt and Launsby's Karl Fischer case study prints each run's total of its two readings."""
    from process_improve.experiments.reference_catalogue import _summarised

    table = pd.DataFrame({"y1": [1.2402, 0.8562], "y2": [1.1005, 0.9602]})
    total = _summarised(table, {"of": ["y1", "y2"], "statistic": "total"})
    assert total.iloc[:, 0].tolist() == pytest.approx([2.3407, 1.8164])


def test_software_that_prints_more_digits_than_it_carries_is_held_to_fewer():
    """A fitted value printed -4.70511 by single-precision software, where the fit gives -4.70513."""
    from process_improve.experiments.reference_catalogue import _agrees

    assert not _agrees(-4.70513, "-4.70511")
    assert _agrees(-4.70513, "-4.70511", decimals=4)
    assert not _agrees(-4.7052, "-4.70511", decimals=4)


def test_a_design_for_more_factors_than_there_are_letters_is_made():
    """A Plackett-Burman design for 134 inputs, as Schmidt and Launsby's AIDS model case study screens, has 136 runs."""
    from process_improve.experiments.reference_catalogue import _design_property

    design = {"design_type": "plackett_burman", "k": 134, "options": {"n_center_points": 0}}
    assert _design_property({"property": "n_runs", "design": design}) == 136


def test_a_number_printed_in_e_notation_is_held_to_the_digits_it_prints():
    """1.25E+08 is rounded to the nearest million: 125,272,056 agrees with it, 125,600,000 does not."""
    from process_improve.experiments.reference_catalogue import _agrees

    assert _agrees(125_272_056, "1.25E+08")
    assert not _agrees(125_600_000, "1.25E+08")


def _plain_2x2(y):
    rows = [[a, b, v] for (a, b), v in zip([(-1, -1), (1, -1), (-1, 1), (1, 1)], y, strict=True)]
    return {
        "name": "plain",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in "AB"],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["A", "B", "y"], "rows": rows, "coded": True},
    }


def test_a_coefficient_fitted_to_responses_printed_rounded_may_carry_their_rounding():
    """Fitted to readings it does not print, a source's intercept is 25.04; its runs, printed rounded, give 25.00.

    Each printed run may be off by 0.05 and the intercept weighs each by 1/4, so it may move by 0.05:
    the printed 25.04 agrees, and a coefficient of the wrong sign still does not.
    """
    entry = _plain_2x2([10.0, 20.0, 30.0, 40.0])
    check = {"kind": "coefficient", "term": "Intercept", "model": "A + B"}
    fits = reference_catalogue._Fits()
    assert not reference_catalogue._reproduce(entry, {"value": "25.04"}, check, fits)[1]
    rounded = {**check, "data_rounding": {"decimals": 1}}
    assert reference_catalogue._reproduce(entry, {"value": "25.04"}, rounded, fits)[1]
    assert not reference_catalogue._reproduce(entry, {"value": "-25.04"}, rounded, fits)[1]
    assert reference_catalogue._carried_rounding(fits(entry, rounded), rounded) == pytest.approx(0.05)


def test_the_durbin_watson_statistic_is_read_from_the_library_fit():
    """The sum of the squared differences of successive residuals over the sum of the squared residuals."""
    x, y = np.arange(1.0, 7.0), np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    entry = {
        "name": "line",
        "factors": [{"name": "x", "type": "numeric", "low": 1, "high": 6, "coding": "none"}],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["x", "y"], "rows": [[a, b] for a, b in zip(x, y, strict=True)], "coded": False},
    }
    check = {"kind": "durbin_watson", "model": "x"}
    residuals = y - np.polyval(np.polyfit(x, y, 1), x)
    expected = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    found = reference_catalogue._FIT_KINDS["durbin_watson"](reference_catalogue._Fits()(entry, check), check, entry)
    assert found == pytest.approx(expected)


def _two_groups(a, b):
    """Return an entry of two groups, A in runs 1 to len(a) and B after them."""
    return {
        "name": "groups",
        "factors": [{"name": "group", "type": "categorical", "low": "A", "high": "B", "levels": ["A", "B"]}],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["group", "y"], "rows": [["A", y] for y in a] + [["B", y] for y in b], "coded": False},
    }


def test_two_groups_are_compared_with_the_library_t_tests():
    """B less A by the library's pooled, Welch and paired tests, each as scipy computes it."""
    from process_improve.experiments.reference_catalogue import _t_test

    a, b = [1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 7.0, 6.0]
    entry = _two_groups(a, b)
    groups = {"runs": [5, 6, 7, 8], "minus": [1, 2, 3, 4]}
    pooled, welch, paired = stats.ttest_ind(b, a), stats.ttest_ind(b, a, equal_var=False), stats.ttest_rel(b, a)
    assert _t_test(entry, {**groups, "key": "difference"}) == pytest.approx(2.0)
    assert _t_test(entry, {**groups, "key": "t"}) == pytest.approx(pooled.statistic)
    assert _t_test(entry, {**groups, "key": "p"}) == pytest.approx(pooled.pvalue)
    assert _t_test(entry, {**groups, "key": "p", "alternative": "greater"}) == pytest.approx(pooled.pvalue / 2)
    pooled_variance = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2  # equal groups: the average of the two
    assert _t_test(entry, {**groups, "key": "pooled_std_dev"}) == pytest.approx(np.sqrt(pooled_variance))
    for key, expected in (("t", welch.statistic), ("df", welch.df), ("p", welch.pvalue)):
        assert _t_test(entry, {**groups, "test": "welch", "key": key}) == pytest.approx(expected)
    differences = np.subtract(b, a)
    assert _t_test(entry, {**groups, "test": "paired", "key": "t"}) == pytest.approx(paired.statistic)
    assert _t_test(entry, {**groups, "test": "paired", "key": "std_dev"}) == pytest.approx(np.std(differences, ddof=1))
    half = stats.t.ppf(0.95, 3) * np.std(differences, ddof=1) / 2  # a 90% interval on 3 degrees of freedom
    assert _t_test(entry, {**groups, "test": "paired", "key": "half_width", "confidence": 0.90}) == pytest.approx(half)


def test_the_pooled_t_test_is_the_regression_on_the_two_groups_coded():
    """Student's t is the t of the group's coefficient, coded -1 and +1, and their intervals are the effect's."""
    entry = _two_groups([1.0, 2.0, 4.0, 5.0], [4.0, 3.0, 7.0, 6.0])
    fits = reference_catalogue._Fits()
    groups = {"kind": "t_test", "runs": [5, 6, 7, 8], "minus": [1, 2, 3, 4]}
    for key, kind in (("t", "t_value"), ("ci_low", "ci_low"), ("std_error", "effect_std_error")):
        by_test = reference_catalogue._reproduce(entry, {"value": "0"}, {**groups, "key": key}, fits)[0]
        regression = {"kind": kind, "term": "group", "model": "group", "scale": "effect"}
        assert reference_catalogue._reproduce(entry, {"value": by_test}, regression, fits)[1], key


def test_a_median_minimum_and_maximum_are_read_from_the_runs():
    entry = _two_groups([3.0, 1.0, 2.0], [10.0])
    fits = reference_catalogue._Fits()
    for kind, value in (("median", "2"), ("minimum", "1"), ("maximum", "3")):
        assert reference_catalogue._reproduce(entry, {"value": value}, {"kind": kind, "runs": [1, 2, 3]}, fits)[1]


def _three_groups():
    """Return an entry of three groups of three runs, a categorical factor fitted by name."""
    groups = {"a": [1.0, 2.0, 3.0], "b": [4.0, 6.0, 5.0], "c": [9.0, 7.0, 8.0]}
    rows = [[g, y] for g, values in groups.items() for y in values]
    return {
        "name": "three",
        "factors": [{"name": "g", "type": "categorical", "low": "a", "high": "b", "levels": ["a", "b", "c"]}],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["g", "y"], "rows": rows, "coded": False},
    }


def test_a_contrast_of_level_means_is_tested_on_the_residual_mean_square():
    """Means 2, 5 and 8, each of three runs, MSE = 1 on 6 df: a - b is -3 with standard error √(2/3)."""
    entry = _three_groups()
    fits = reference_catalogue._Fits()
    check = {"model": "0 + g", "weights": {"g[a]": 1, "g[b]": -1}}
    out = fits(entry, check)
    se = (2 / 3) ** 0.5
    found = {
        key: reference_catalogue._contrast(out, {**check, "key": key}, entry)
        for key in ("value", "std_error", "t", "f", "sum_sq", "p", "half_width")
    }
    assert found["value"] == pytest.approx(-3)
    assert found["std_error"] == pytest.approx(se)
    assert found["t"] == pytest.approx(-3 / se)
    assert found["sum_sq"] == pytest.approx(found["f"] * 1.0)  # the residual mean square is 1
    assert found["p"] == pytest.approx(2 * stats.t.sf(3 / se, 6))
    assert found["half_width"] == pytest.approx(stats.t.ppf(0.975, 6) * se)


def test_tukey_comparisons_are_the_library_tukey_hsd_on_the_one_way_model():
    """The interval is the difference plus or minus q(0.95; 3, 6) √(MSE/n); a level less another changes sign."""
    from process_improve.experiments.reference_catalogue import _tukey

    entry = _three_groups()
    check = {"factor": "g", "level": "b", "minus": "a"}
    q = stats.studentized_range.ppf(0.95, 3, 6)
    assert _tukey(entry, {**check, "key": "difference"}) == pytest.approx(3)
    assert _tukey(entry, {**check, "key": "q_critical"}) == pytest.approx(q)
    assert _tukey(entry, {**check, "key": "ci_low"}) == pytest.approx(3 - q * (1 / 3) ** 0.5)
    assert _tukey(entry, {**check, "level": "a", "minus": "b", "key": "difference"}) == pytest.approx(-3)


def test_a_transformed_fit_reads_its_response_on_the_transformed_scale():
    """The total sum of squares of a log fit is that of the logs, so the residual and model sums add up to it."""
    entry = _three_groups()
    fits = reference_catalogue._Fits()
    check = {"model": "g", "transform": "log"}
    out = fits(entry, check)
    logs = np.log([1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0])
    assert reference_catalogue._FIT_KINDS["ss_total"](out, check, entry) == pytest.approx(
        np.sum((logs - logs.mean()) ** 2)
    )
    assert reference_catalogue._FIT_KINDS["sum"](out, check, entry) == pytest.approx(logs.sum())


def test_each_run_case_statistic_is_read_from_the_library_diagnostics():
    entry = _three_groups()
    fits = reference_catalogue._Fits()
    out = fits(entry, {"model": "g"})
    for kind in ("leverage", "cooks_distance", "studentized_residual", "externally_studentized_residual"):
        assert math.isfinite(reference_catalogue._FIT_KINDS[kind](out, {"run": 1}, entry))
    assert reference_catalogue._FIT_KINDS["leverage"](out, {"run": 1}, entry) == pytest.approx(1 / 3)


def test_the_reml_fit_reads_its_variances_their_covariance_and_the_fixed_effects():
    """Three treatments in two random blocks, balanced: the treatment effects are their means less the grand mean."""
    from process_improve.experiments.reference_catalogue import _variance_component

    cells = [("a", "1", 1), ("a", "2", 3), ("b", "1", 4), ("b", "2", 7), ("c", "1", 2), ("c", "2", 4)]
    entry = {
        "responses": [{"name": "y"}],
        "runs": {"columns": ["treatment", "block", "y"], "rows": [list(c) for c in cells]},
    }
    check = {"groups": ["block"], "fixed": ["treatment"]}
    keys = (
        "block",
        "residual",
        "total",
        "std_error.block",
        "wald_low.block",
        "percent.residual",
        "ratio.block",
        "rmse",
    )
    found = {key: _variance_component(entry, {**check, "key": key}) for key in keys}
    assert found["total"] == pytest.approx(found["block"] + found["residual"])
    assert found["wald_low.block"] == pytest.approx(found["block"] - 1.959963984540054 * found["std_error.block"])
    assert found["percent.residual"] == pytest.approx(100 * found["residual"] / found["total"])
    assert found["ratio.block"] == pytest.approx(found["block"] / found["residual"])
    assert found["rmse"] == pytest.approx(found["residual"] ** 0.5)
    effects = [_variance_component(entry, {**check, "key": f"effect.treatment.{t}"}) for t in "abc"]
    assert effects == pytest.approx([2 - 3.5, 5.5 - 3.5, 3 - 3.5])  # c is the level the coding leaves out
    with pytest.raises(KeyError, match="no level 'd'"):
        _variance_component(entry, {**check, "key": "effect.treatment.d"})


def _two_way_single(table):
    """Return a two-factor layout with one observation per cell, rows r1.. and columns c1.."""
    rows = [[f"r{i}", f"c{j}", y] for i, values in enumerate(table, start=1) for j, y in enumerate(values, start=1)]
    return {
        "name": "single",
        "factors": [
            {"name": "row", "type": "categorical", "low": "r1", "high": "r2", "levels": [f"r{i}" for i in range(1, 4)]},
            {"name": "col", "type": "categorical", "low": "c1", "high": "c2", "levels": [f"c{j}" for j in range(1, 6)]},
        ],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["row", "col", "y"], "rows": rows, "coded": False},
    }


def test_tukey_nonadditivity_is_worked_from_the_row_and_column_effects():
    """SSN = (Σ yij ri cj)² / (Σ ri² Σ cj²), ri and cj the row and column effects, one df taken from the residual."""
    from process_improve.experiments.reference_catalogue import _nonadditivity

    table = np.array([[5, 4, 6, 3, 5], [3, 1, 4, 2, 3], [1, 1, 3, 1, 2]], dtype=float)  # Montgomery's Table 5.10
    r, c = table.mean(axis=1) - table.mean(), table.mean(axis=0) - table.mean()
    ssn = (r @ table @ c) ** 2 / ((r**2).sum() * (c**2).sum())
    residual = ((table - table.mean(axis=1, keepdims=True) - table.mean(axis=0) + table.mean()) ** 2).sum()
    entry = _two_way_single(table.tolist())
    check = {"model": "row + col"}
    found = {key: _nonadditivity(entry, {**check, "key": key}) for key in ("ss", "ss_error", "df_error", "f", "p")}
    assert found["ss"] == pytest.approx(ssn)
    assert found["ss_error"] == pytest.approx(residual - ssn)
    assert found["df_error"] == 7
    assert found["f"] == pytest.approx(ssn / ((residual - ssn) / 7))
    assert found["p"] == pytest.approx(stats.f.sf(found["f"], 1, 7))
    row_ms = 5 * (r**2).sum() / 2
    assert _nonadditivity(entry, {**check, "key": "f.row"}) == pytest.approx(row_ms / ((residual - ssn) / 7))


def test_a_reml_fit_can_hold_an_interaction_of_fixed_factors_and_report_its_criterion():
    """Two fixed factors and their interaction in random blocks: the interaction's F is that of the blocked ANOVA."""
    from process_improve.experiments.reference_catalogue import _variance_component

    cells = {("a", "x"): [5, 7], ("a", "y"): [6, 9], ("b", "x"): [8, 8], ("b", "y"): [12, 14]}
    rows = [[f, g, str(block + 1), y] for (f, g), values in cells.items() for block, y in enumerate(values)]
    entry = {"responses": [{"name": "y"}], "runs": {"columns": ["f", "g", "block", "y"], "rows": rows}}
    check = {"groups": ["block"], "fixed": ["f", "g", "f:g"]}
    table = pd.DataFrame(rows, columns=["f", "g", "block", "y"])
    anova = analyze_experiment(table[["f", "g", "block"]], table["y"].astype(float), model="f * g + block")
    interaction = next(r for r in anova["anova_table"] if r["source"] == "f:g")
    assert _variance_component(entry, {**check, "key": "f.f:g"}) == pytest.approx(interaction["F"])
    assert math.isfinite(_variance_component(entry, {**check, "key": "reml_criterion"}))


def test_a_prediction_can_be_read_as_the_half_width_of_its_interval():
    entry = _line([1.0, 2.5, 2.5, 4.0])
    fits = reference_catalogue._Fits()
    check = {"kind": "prediction", "model": "x", "point": {"x": 2.5}}
    out = fits(entry, check)
    low, high = (reference_catalogue._prediction(out, {**check, "bound": b}, entry) for b in ("ci_low", "ci_high"))
    half = reference_catalogue._prediction(out, {**check, "bound": "ci_half_width"}, entry)
    assert half == pytest.approx((high - low) / 2)


def _factorial(k, y, centre=()):
    """Return an unreplicated 2^k in standard order, factors A, B, ... coded -1 and +1, then any centre runs."""
    names = list("ABCDE"[:k])
    rows = [[1 if (i >> j) & 1 else -1 for j in range(k)] + [v] for i, v in enumerate(y)] + [
        [0] * k + [v] for v in centre
    ]
    return {
        "name": f"two-level {k}",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in names],
        "responses": [{"name": "y"}],
        "runs": {"columns": [*names, "y"], "rows": rows, "coded": True},
    }


FILTRATION = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]  # Montgomery's Example 6.2


def test_a_nested_test_of_one_term_is_that_term_s_f_test():
    """A model against the one with one term more: the F test of that term, as the analysis of variance prints it."""
    from process_improve.experiments.reference_catalogue import _nested_test

    entry = _factorial(4, FILTRATION)
    check = {"model": "A + C + D + A:C", "full": "A + C + D + A:C + A:D"}
    fuller = analyze_experiment(
        *reference_catalogue._coded_runs(entry, None), model=check["full"], analysis_type=["anova"]
    )
    row = next(r for r in fuller["anova_table"] if r["source"] == "A:D")
    assert _nested_test(entry, {**check, "key": "ss"}) == pytest.approx(row["sum_sq"])
    assert _nested_test(entry, {**check, "key": "df"}) == 1
    assert _nested_test(entry, {**check, "key": "f"}) == pytest.approx(row["F"])
    assert _nested_test(entry, {**check, "key": "p"}) == pytest.approx(row["p_value"])


def test_lenth_pseudo_t_is_an_effect_over_the_pse_on_a_third_as_many_degrees_of_freedom():
    """JMP's screening platform: Temp's estimate 10.8125 over the PSE 1.3125 is 8.24, judged on t with 15/3 df."""
    entry = _factorial(4, FILTRATION)
    fits = reference_catalogue._Fits()
    check = {"model": "A * B * C * D", "term": "A"}
    out = fits(entry, check)
    t = reference_catalogue._FIT_KINDS["lenth_t"](out, check, entry)
    assert t == pytest.approx(21.625 / 2.625)
    assert reference_catalogue._FIT_KINDS["lenth_p"](out, check, entry) == pytest.approx(2 * stats.t.sf(t, 5))


def test_a_percent_of_total_is_a_term_s_share_of_the_total_sum_of_squares():
    """Montgomery's percent contribution: a term's sum of squares as a percentage of the total, not Taguchi's."""
    entry = _factorial(4, FILTRATION)
    fits = reference_catalogue._Fits()
    check = {"model": "A * B * C * D - A:B:C:D", "term": "A"}
    out = fits(entry, check)
    total = np.sum((np.array(FILTRATION) - np.mean(FILTRATION)) ** 2)
    assert reference_catalogue._FIT_KINDS["percent_of_total"](out, check, entry) == pytest.approx(
        100 * 1870.5625 / total
    )


def test_centre_runs_give_the_curvature_test_and_the_largest_r_squared():
    """Four centre runs: pure error on 3 df, and 1 - SSpe/SST the best R² any model of these runs could reach."""
    entry = _factorial(4, FILTRATION, centre=[73, 75, 66, 69])
    fits = reference_catalogue._Fits()
    check = {"model": "A * B * C * D"}
    out = fits(entry, check)
    assert reference_catalogue._FIT_KINDS["curvature"](out, {"key": "ss_curvature"}, entry) == pytest.approx(
        16 * 4 * (70.0625 - 70.75) ** 2 / 20
    )
    y = np.array([*FILTRATION, 73, 75, 66, 69], dtype=float)
    pure = np.sum((y[16:] - y[16:].mean()) ** 2)
    found = reference_catalogue._FIT_KINDS["max_r_squared"](out, check, entry)
    assert found == pytest.approx(1 - pure / np.sum((y - y.mean()) ** 2))


def test_a_range_is_read_from_a_run_s_readings():
    entry = {**_factorial(1, [0, 0]), "responses": [{"name": "r1"}, {"name": "r2"}]}
    entry["runs"] = {"columns": ["A", "r1", "r2"], "rows": [[-1, 550, 604], [1, 669, 650]], "coded": True}
    fits = reference_catalogue._Fits()
    summary = {"of": ["r1", "r2"], "statistic": "range"}
    assert reference_catalogue._reproduce(
        entry, {"value": "54"}, {"kind": "observation", "run": 1, "summary": summary}, fits
    )[1]
    each = {"of": ["r1", "r2"], "statistic": "each"}
    assert reference_catalogue._reproduce(
        entry, {"value": "19"}, {"kind": "range", "runs": [2], "summary": each}, fits
    )[1]


def test_two_columns_of_a_two_level_factorial_are_correlated_once_a_run_is_missing():
    """Run cd left out of a 2^4: any two columns have the correlation -1/14, which a source prints as ±0.0714."""
    from process_improve.experiments.reference_catalogue import _column_correlation

    entry = _factorial(4, FILTRATION)
    kept = [i for i in range(1, 17) if i != 13]
    assert _column_correlation(entry, {"columns": ["A", "B"], "runs": kept}) == pytest.approx(-1 / 14)
    assert _column_correlation(entry, {"columns": ["A", "B"]}) == pytest.approx(0)


def test_blocks_confound_the_products_of_the_effects_chosen_to_generate_them():
    from process_improve.experiments.reference_catalogue import _generalized_interactions

    assert _generalized_interactions({}, {"of": "ADE, BCE"}, "ADE, BCE, ABCD")[1]
    assert _generalized_interactions({}, {"of": "ABEF, ABCD, ACE"}, "ABEF, ABCD, ACE, CDEF, BCF, BDE, ADF")[1]
    assert not _generalized_interactions({}, {"of": "ABCD, CDEF, ADFG"}, "ABC, DEF, AFG, ABCDEF, BCFG, ADEG, BCDEG")[1]


def test_the_signs_of_two_effects_pick_one_of_the_four_blocks_they_generate():
    from process_improve.experiments.reference_catalogue import _fraction_runs

    entry = _factorial(3, [0] * 8)
    assert _fraction_runs(entry, {"generator": "-AB, -BC"}) == [3, 6]  # b and ac
    assert _fraction_runs(entry, {"generator": "AB, -BC"}) == [4, 5]  # ab and c


def test_a_path_into_a_library_result_may_hold_keys_with_a_dot():
    assert reference_catalogue._read({"quantiles": {"0.5": 0.42}}, ["quantiles", "0.5"]) == pytest.approx(0.42)
    assert reference_catalogue._read({"q": [1.0, 3.0]}, "q.max") == pytest.approx(3.0)


def test_a_fraction_s_word_lengths_and_clear_effects_are_read_from_its_defining_relation():
    """Montgomery's design C for seven factors has the word lengths {4, 5, 5}; in I = BCDF, BC is aliased with DF."""
    from process_improve.experiments.reference_catalogue import _design_check

    design_c = {"kind": "word_lengths", "generators": ["F=ABCD", "G=ABDE"]}
    assert _design_check(design_c, "{4, 5, 5}", 7) == ("4, 5, 5", True)
    assert not _design_check(design_c, "{4, 4, 4}", 7)[1]
    folded = {"generators": ["F=BCD"]}
    assert _design_check({**folded, "kind": "clear", "term": "AB"}, "AB", 6)[1]
    assert _design_check({**folded, "kind": "clear", "term": "BC"}, "BC", 6) == ("DF", False)


def test_generators_are_compared_by_word_lengths_with_the_fraction_the_library_chooses():
    """Nine factors in 32 runs: the library's resolution IV fraction has six four-letter words; F = ABC, ... has 14."""
    from process_improve.experiments.reference_catalogue import _design_check

    table = {"kind": "minimum_aberration", "generators": ["F=BCDE", "G=ACDE", "H=ABDE", "J=ABCE"]}
    assert _design_check(table, "", 9) == ("F=BCDE, G=ACDE, H=ABDE, J=ABCE", True)
    assert not _design_check({**table, "generators": ["F=ABC", "G=ABD", "H=ACD", "J=BCD"]}, "", 9)[1]


def test_a_printed_column_of_signs_is_the_product_of_the_effects_it_is_headed_with():
    from process_improve.experiments.reference_catalogue import _product_column

    rows = [[a, b, c, a * b] for a, b, c in HALF_FRACTION]  # C = AB, so the column AB is also C's
    entry = _printed_design(["A", "B", "C", "AB"], rows)
    assert _product_column(entry, {"column": "AB"}, "AB = C") == ("every run", True)
    assert _product_column(entry, {"column": "AB"}, "AB = \u2212C") == ("not in runs 1, 2, 3, 4", False)


PB12_ROW = [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]  # Montgomery's Table 8.23, N = 12


def _cyclic(row):
    """Return the Plackett-Burman design of a generating row: the row as the first column, shifted down for the next."""
    k = len(row)
    return [[row[(i - j) % k] for j in range(k)] for i in range(k)] + [[-1] * k]


def test_a_plackett_burman_design_is_matched_by_its_projections_whatever_its_column_order_and_signs():
    from process_improve.experiments.reference_catalogue import _same_projections

    names = list("ABCDEFGHJKL")
    library = {"design": {"design_type": "plackett_burman", "k": 11, "options": {"n_center_points": 0}}}
    shuffled = [[-r[3], *r[:3], *r[4:]] for r in _cyclic(PB12_ROW)]  # a column moved first, its signs changed
    assert _same_projections(_printed_design(names, shuffled), library, "")[1]
    broken = [row[:] for row in shuffled]
    broken[0][0] = -broken[0][0]  # one sign miscopied: the columns are no longer balanced and orthogonal
    assert not _same_projections(_printed_design(names, broken), library, "")[1]


def test_a_projection_counts_the_different_runs_it_leaves():
    """The 12-run Plackett-Burman design covers the eight corners of any three of its columns, and 11 of 16 of four."""
    from process_improve.experiments.reference_catalogue import _distinct_runs

    entry = _printed_design(list("ABCDEFGHJKL"), _cyclic(PB12_ROW))
    assert _distinct_runs(entry, {"columns": ["A", "B", "C"]}) == 8
    assert _distinct_runs(entry, {"columns": ["A", "B", "C", "D"]}) == 11


def test_a_coefficient_s_standard_error_in_units_of_sigma_is_a_property_of_the_runs():
    from process_improve.experiments.reference_catalogue import _unit_std_error

    entry = _factorial(4, FILTRATION)
    assert _unit_std_error(entry, {"model": "A + B + A:B", "key": "max"}) == pytest.approx(0.25)
    missing = {"model": "A + B + A:B", "runs": list(range(1, 16))}
    assert _unit_std_error(entry, {**missing, "key": "min"}) > 0.25
    assert _unit_std_error(entry, {**missing, "exclude": ["Intercept", "A"], "key": "max"}) == pytest.approx(
        _unit_std_error(entry, {**missing, "exclude": ["Intercept", "A"], "key": "min"})
    )


def test_a_nested_test_of_responses_printed_rounded_may_carry_their_rounding():
    """JMP fitted responses the book prints to fewer digits: its sums of squares carry what that rounding moves."""
    from process_improve.experiments.reference_catalogue import _nested_test, _nested_test_rounding

    exact = np.array(FILTRATION) + 0.04 * np.sin(np.arange(16))
    printed = _factorial(4, list(np.round(exact, 1)))
    check = {"kind": "nested_test", "model": "A + C + D", "full": "A + C + D + A:C", "key": "ss"}
    truth = _nested_test(_factorial(4, list(exact)), check)
    rounded = {**check, "data_rounding": {"decimals": 1}}
    assert abs(_nested_test(printed, check) - truth) <= _nested_test_rounding(printed, rounded)
    shown = f"{truth:.4f}"
    assert not reference_catalogue._reproduce(printed, {"value": shown}, check, reference_catalogue._Fits())[1]
    assert reference_catalogue._reproduce(printed, {"value": shown}, rounded, reference_catalogue._Fits())[1]


def test_counts_and_proportions_are_transformed_run_by_run_as_a_source_prints_them():
    """Montgomery's Tables P8.13 and P8.14: arcsin sqrt(p), Freeman and Tukey's forms, and sqrt(c)."""
    entry = {
        "name": "transforms",
        "factors": [{"name": "A", "type": "numeric", "low": -1, "high": 1}],
        "responses": [{"name": "p"}, {"name": "c"}],
        "runs": {"columns": ["A", "p", "c"], "rows": [[-1, 0.958, 56], [1, 1.0, 0]], "coded": True},
    }
    fits = reference_catalogue._Fits()
    cases = [
        ("arcsin_sqrt", "p", 1, {}, "1.364"),
        ("freeman_tukey_arcsin", "p", 2, {"n": 1000}, "1.555"),
        ("sqrt", "c", 1, {}, "7.48"),
        ("freeman_tukey_sqrt", "c", 1, {}, "7.52"),
        ("freeman_tukey_sqrt", "c", 2, {}, "0.50"),
    ]
    for statistic, column, run, extra, printed in cases:
        check = {"kind": "observation", "run": run, "summary": {"of": [column], "statistic": statistic, **extra}}
        assert reference_catalogue._reproduce(entry, {"value": printed}, check, fits)[1], statistic


def test_an_alias_matrix_is_read_entry_by_entry_from_one_evaluation():
    """The 2^(3-1) with I = ABC: x1's coefficient is biased by the x2x3 interaction, with weight 1."""
    from process_improve.experiments.reference_catalogue import _design_metric

    entry = _printed_design(["A", "B", "C"], HALF_FRACTION)
    entry["runs"]["coded"] = True
    check = {"metric": "alias_matrix", "model": "A + B + C"}
    assert _design_metric(entry, {**check, "key": "matrix.1.2"}) == pytest.approx(1)
    assert _design_metric(entry, {**check, "key": "matrix.1.0"}) == pytest.approx(0)
    assert _design_metric(entry, {**check, "key": "matrix.3.0"}) == pytest.approx(1)


TOOL_LIFE = {  # Montgomery's Example 5.5: (angle, speed) -> the two readings of each cell
    (15, 125): [-2, -1],
    (15, 150): [-3, 0],
    (15, 175): [2, 3],
    (20, 125): [0, 2],
    (20, 150): [1, 3],
    (20, 175): [4, 6],
    (25, 125): [-1, 0],
    (25, 150): [5, 6],
    (25, 175): [0, -1],
}


def _three_level(columns, rows, responses=()):
    """Return an entry of a 3^k design, its factors' levels 0, 1 and 2 as a source writes them."""
    factors = [{"name": c, "type": "numeric", "low": 0, "high": 2, "levels": [0, 1, 2]} for c in columns]
    return {
        "name": "three-level",
        "factors": factors,
        "responses": [{"name": r} for r in responses],
        "runs": {"columns": [*columns, *responses], "rows": rows, "coded": False},
    }


def test_a_3k_component_groups_the_runs_by_its_defining_contrast_modulo_3():
    """The tool life data split the AB interaction into AB (x1 + x2) and AB² (x1 + 2x2): 33.33 and 28.00 of 61.33."""
    from process_improve.experiments.reference_catalogue import _component_ss

    level = {15: 0, 20: 1, 25: 2, 125: 0, 150: 1, 175: 2}
    rows = [[level[a], level[s], y] for (a, s), ys in TOOL_LIFE.items() for y in ys]
    entry = _three_level(["A", "B"], rows, ["y"])
    assert _component_ss(entry, {"word": "AB"}) == pytest.approx(100 / 3)
    assert _component_ss(entry, {"word": "AB²"}) == pytest.approx(28)
    assert [_component_ss(entry, {"word": "AB", "key": f"total.{u}"}) for u in range(3)] == [18, -2, 8]
    assert [_component_ss(entry, {"word": "AB2", "key": f"total.{u}"}) for u in range(3)] == [0, 18, 6]
    assert _component_ss(entry, {"word": "A2B", "key": "count.0"}) == 6  # A²B is AB², squared
    interaction = _component_ss(entry, {"word": "AB"}) + _component_ss(entry, {"word": "AB2"})
    assert interaction == pytest.approx(61 + 1 / 3)


def test_3k_words_multiply_modulo_3_into_aliases_and_a_defining_relation():
    from process_improve.experiments.reference_catalogue import _component_product, _component_relation

    assert _component_product({}, {"words": ["A", "AB2C2"]}, "ABC")[1]
    assert _component_product({}, {"words": ["A", "AB2C2"], "powers": [1, 2]}, "BC")[1]
    assert _component_product({}, {"words": ["ABC", "AB²D²"]}, "AC²D")[1]
    assert not _component_product({}, {"words": ["ABC", "AB²D²"]}, "ACD")[1]
    relation = {"words": ["AB2C", "BCD"]}
    assert _component_relation({}, relation, "I = AB²C = BCD = AC²D = ABD²")[1]
    assert not _component_relation({}, relation, "I = AB²C = BCD = AC²D")[1]
    assert _component_relation({}, {**relation, "key": "resolution"}, "III")[1]
    assert _component_relation({}, {"words": ["AB2CD"], "key": "resolution"}, "IV")[1]


def test_3k_blocks_and_fractions_are_the_runs_with_one_value_of_each_defining_contrast():
    """Montgomery's Figure 9.6: the 3^2 in three blocks with AB² confounded, and the 3^(4-2) of Table 9.7."""
    from process_improve.experiments.reference_catalogue import (
        _component_blocks,
        _component_fraction,
        _component_values,
        _mod3_sum,
    )

    blocks = {"00": 1, "11": 1, "22": 1, "10": 2, "21": 2, "02": 2, "01": 3, "12": 3, "20": 3}
    entry = _three_level(["A", "B"], [[int(r[0]), int(r[1])] for r in blocks], [])
    entry["runs"] = {
        "columns": ["A", "B", "block"],
        "rows": [[int(r[0]), int(r[1]), b] for r, b in blocks.items()],
        "coded": False,
    }
    entry["factors"].append({"name": "block", "type": "categorical", "levels": ["1", "2", "3"]})
    check = {"words": ["AB2"], "block": "block"}
    assert _component_blocks(entry, check, "")[1]
    assert _component_blocks(entry, {**check, "of": "2"}, "1")[1]
    assert not _component_blocks(entry, {**check, "words": ["AB"]}, "")[1]
    assert _component_values({}, {"word": "AB2"}, "01: 2, 21: 1, 22: 0")[1]
    assert not _component_values({}, {"word": "AB2"}, "01: 1")[1]
    assert _mod3_sum({}, {}, "101 + 021 = 122")[1]
    assert not _mod3_sum({}, {}, "101 + 021 = 120")[1]
    runs = ["0000", "0111", "0222", "1021", "1102", "1210", "2012", "2120", "2201"]
    fraction = _three_level(list("ABCD"), [[int(c) for c in r] for r in runs])
    assert _component_fraction(fraction, {"words": ["AB2C", "BCD"], "u": [0, 0]}, "")[1]
    assert not _component_fraction(fraction, {"words": ["ABC"]}, "")[1]


def test_a_design_printed_as_columns_of_another_is_a_projection_of_it(tmp_path, monkeypatch):
    from process_improve.experiments.reference_catalogue import _projection

    rows = _cyclic(PB12_ROW)
    whole = {**_printed_design(list("ABCDEFGHJKL"), rows), "name": "whole", "expected": []}
    (tmp_path / "whole.json").write_text(json.dumps({"experiments": [], "designs": [whole]}), encoding="utf-8")
    monkeypatch.setattr(reference_catalogue, "_CATALOGUE_DIR", tmp_path)
    part = _printed_design(["X", "Y"], [[r[2], r[5]] for r in rows])
    assert _projection(part, {"of": "whole", "columns": ["C", "F"]}, "")[1]
    found, same = _projection(part, {"of": "whole", "columns": ["C", "G"]}, "")
    assert not same
    assert found.startswith("not in runs")
    twice = _printed_design(["A", "B", "A again"], [[r[0], r[1], r[0]] for r in rows])
    assert _projection(twice, {"columns": ["A again"], "into": ["A"]}, "")[1]


def test_screening_designs_are_compared_on_their_main_effect_and_interaction_columns():
    """Montgomery's Table 9.23: the regular 2^(6-2) of resolution IV confounds nine pairs, and E(s²) = 9 * 16²/210."""
    from process_improve.experiments.reference_catalogue import _estimable_projections, _interaction_columns

    regular = {
        "design": {"design_type": "fractional_factorial", "k": 6, "options": {"resolution": 4, "n_center_points": 0}}
    }
    assert _interaction_columns({}, {**regular, "key": "confounded_pairs"}) == 9
    assert _interaction_columns({}, {**regular, "key": "e_s2"}) == pytest.approx(9 * 256 / 210)
    assert _interaction_columns({}, {**regular, "key": "max_abs_r_main_interaction"}) == pytest.approx(0)
    pb12 = _printed_design(list("ABCDEFGHJKL"), _cyclic(PB12_ROW))
    assert _interaction_columns(pb12, {"key": "max_abs_r_main_interaction"}) == pytest.approx(1 / 3)
    library_pb = {"design": {"design_type": "plackett_burman", "k": 11, "options": {"n_center_points": 0}}}
    assert _estimable_projections({}, {**library_pb, "order": 4}) == 0  # any four factors fit their 2FI model
    assert _estimable_projections({}, {**library_pb, "order": 5}) == 462  # 16 terms in 12 runs never do


def test_relative_variances_code_a_categorical_factor_as_jmp_does():
    """Montgomery's Tables 9.43 and 9.44: a 14-run D-optimal design, each three-level factor in two columns."""
    from process_improve.experiments.reference_catalogue import _run_count, _unit_std_error

    a = ["L1", "L3", "L1", "L1", "L3", "L2", "L1", "L3", "L2", "L3", "L2", "L2", "L1", "L3"]  # Table 9.43
    b = ["L3", "L2", "L1", "L2", "L3", "L3", "L1", "L1", "L3", "L3", "L2", "L1", "L2", "L1"]
    c = [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1]
    levels = ["L1", "L2", "L3"]
    entry = {
        "name": "jmp",
        "factors": [{"name": n, "type": "categorical", "low": "L1", "high": "L3", "levels": levels} for n in "AB"]
        + [{"name": "C", "type": "numeric", "low": -1, "high": 1, "levels": [-1, 1]}],
        "responses": [],
        "runs": {"columns": ["A", "B", "C"], "rows": [list(run) for run in zip(a, b, c, strict=True)], "coded": False},
    }
    model = "A1 + A2 + B1 + B2 + C + A1:B1 + A2:B1 + A1:B2 + A2:B2 + A1:C + A2:C + B1:C + B2:C"
    check = {"model": model, "jmp_coded": ["A", "B"], "variance": True}
    printed = {"Intercept": 0.340, "A1": 0.093, "A2": 0.179, "C": 0.167, "A2:B2": 0.136, "A2:C": 0.278}
    for term, value in printed.items():
        assert _unit_std_error(entry, {**check, "term": term}) == pytest.approx(value, abs=5e-4), term
    assert _unit_std_error(entry, {**check, "key": "len"}) == 14
    assert _run_count(entry, {"at": {"A": "L1", "C": 1}}) == 2


def test_the_normal_equations_are_read_entry_by_entry():
    """Montgomery's Examples 10.2 and 10.3: the 2^3 with four centre runs is orthogonal, and not with run 8 gone."""
    from process_improve.experiments.reference_catalogue import _normal_equations

    entry = _factorial(3, [32, 46, 57, 65, 36, 48, 57, 68], centre=[50, 44, 53, 56])
    model = "A + B + C"
    assert _normal_equations(entry, {"model": model, "key": "xtx.0.0"}) == 12
    assert _normal_equations(entry, {"model": model, "key": "xtx.1.2"}) == 0
    assert _normal_equations(entry, {"model": model, "key": "xty.1"}) == 45
    assert _normal_equations(entry, {"model": model, "key": "inverse.1.1"}) == pytest.approx(1 / 8)
    without = {"model": model, "runs": [r for r in range(1, 13) if r != 8]}
    assert _normal_equations(entry, {**without, "key": "xtx.1.2"}) == -1
    assert _normal_equations(entry, {**without, "key": "inverse.0.1"}) == pytest.approx(1 / 52)


def test_a_prediction_interval_is_read_as_its_half_width_and_the_point_s_leverage():
    """At a corner of the 2^4 fitted with A and B, x0'(X'X)^-1 x0 = 3/16; the interval is t s sqrt(1 + 3/16) wide."""
    from process_improve.experiments.reference_catalogue import _dot_product

    entry = _factorial(4, FILTRATION)
    fits = reference_catalogue._Fits()
    check = {"model": "A + B", "point": {"A": 1, "B": 1}}
    out = fits(entry, check)
    leverage = reference_catalogue._prediction(out, {**check, "bound": "leverage"}, entry)
    assert leverage == pytest.approx(3 / 16)
    half = reference_catalogue._prediction(out, {**check, "bound": "pi_half_width"}, entry)
    mse = out["model_summary"]["mse_residual"]
    assert half == pytest.approx(stats.t.ppf(0.975, 13) * math.sqrt(mse * (1 + 3 / 16)))
    assert _dot_product(entry, {"columns": ["A", "B"]}) == 0
    assert _dot_product(entry, {"columns": ["A"], "runs": [1, 2, 4]}) == 1


def _ccd(y=None, *, blocks=False):
    """Return Montgomery's Example 11.2 CCD in two factors: the 2^2, five centre runs, then the axial runs at 1.414."""
    runs = [(-1, -1), (-1, 1), (1, -1), (1, 1), *[(0, 0)] * 5, (1.414, 0), (-1.414, 0), (0, 1.414), (0, -1.414)]
    y = y or [76.5, 77.0, 78.0, 79.5, 79.9, 80.3, 80.0, 79.7, 79.8, 78.4, 75.6, 78.5, 77.0]
    block = [1] * 9 + [2] * 4
    return {
        "name": "ccd",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in ("A", "B")]
        + ([{"name": "block", "type": "categorical", "low": 1, "high": 2, "levels": [1, 2]}] if blocks else []),
        "responses": [{"name": "y"}],
        "runs": {
            "columns": ["A", "B", "block", "y"],
            "rows": [[a, b, k, v] for (a, b), k, v in zip(runs, block, y, strict=True)],
            "coded": True,
            "exact": {"1.414": math.sqrt(2)},
        },
    }


def test_a_design_is_compared_on_its_inverse_as_a_whole():
    """|(X'X)^-1|, its trace, x0'(X'X)^-1 x0 and the D-efficiency of one set of runs relative to another."""
    from process_improve.experiments.reference_catalogue import _normal_equations

    entry = _factorial(2, [1, 2, 3, 4], centre=[5])
    model = "A + B"
    assert _normal_equations(entry, {"model": model, "key": "det_inverse"}) == pytest.approx(1 / (5 * 4 * 4))
    assert _normal_equations(entry, {"model": model, "key": "trace_inverse"}) == pytest.approx(1 / 5 + 1 / 4 + 1 / 4)
    assert _normal_equations(entry, {"model": model, "key": "leverage", "point": {"A": 1, "B": 1}}) == pytest.approx(
        1 / 5 + 1 / 4 + 1 / 4
    )
    # the 2^2 alone against it with the centre run: (|X2'X2|^-1 / |X1'X1|^-1)^(1/3) = (64 / 80)^(1/3)
    relative = _normal_equations(
        entry, {"model": model, "key": "relative_d", "runs": [1, 2, 3, 4], "versus": [1, 2, 3, 4, 5]}
    )
    assert relative == pytest.approx((64 / 80) ** (1 / 3))


def test_settings_printed_rounded_are_fitted_exactly_unless_asked_as_printed():
    """A natural-unit column printed 92.07 for 85 + 5√2 is fitted at its exact value; as_printed keeps 92.07."""
    entry = _ccd()
    entry["factors"].append({"name": "xi", "type": "numeric", "low": 77.93, "high": 92.07, "coding": "none"})
    natural = [80, 80, 90, 90, 85, 85, 85, 85, 85, 92.07, 77.93, 85, 85]
    entry["runs"]["columns"].insert(2, "xi")
    entry["runs"]["rows"] = [[*row[:2], v, *row[2:]] for row, v in zip(entry["runs"]["rows"], natural, strict=True)]
    entry["runs"]["exact"] |= {"92.07": 85 + 5 * math.sqrt(2), "77.93": 85 - 5 * math.sqrt(2)}
    exact = reference_catalogue._coded_runs(entry, None)[0]["xi"]
    printed = reference_catalogue._coded_runs(entry, None, as_printed=True)[0]["xi"]
    assert exact[9] == pytest.approx(85 + 5 * math.sqrt(2))
    assert printed[9] == 92.07
    fits = reference_catalogue._Fits()
    slope = reference_catalogue._coefficient_row(fits(entry, {"model": "xi"}), "xi")["coefficient"]
    assert slope != pytest.approx(
        reference_catalogue._coefficient_row(fits(entry, {"model": "xi", "settings": "as_printed"}), "xi")[
            "coefficient"
        ]
    )


def test_mixture_components_are_fitted_as_pseudocomponents():
    """Pseudocomponents are (x - l)/(T - sum l): the lower bounds 5, 25 and 50 percent leave 20 percent to share."""
    entry = {
        "name": "paint",
        "factors": [
            {"name": n, "type": "numeric", "low": low, "high": high, "coding": "pseudocomponent"}
            for n, low, high in (("A", 5, 25), ("B", 25, 40), ("C", 50, 70))
        ],
        "responses": [{"name": "y"}],
        "runs": {
            "columns": ["A", "B", "C", "y"],
            "rows": [[25, 25, 50, 1], [5, 40, 55, 2], [5, 25, 70, 3], [18.13, 28.75, 53.13, 4]],
            "coded": False,
            "mixture_total": 100,
            "exact": {"18.13": 18.125, "53.13": 53.125},
        },
    }
    settings = reference_catalogue._coded_runs(entry, None)[0]
    np.testing.assert_allclose(settings.to_numpy()[:3], [[1, 0, 0], [0, 0.75, 0.25], [0, 0, 1]])
    np.testing.assert_allclose(settings.sum(axis=1), 1)
    assert settings["A"][3] == pytest.approx(13.125 / 20)


def test_a_nested_test_can_take_its_error_from_a_third_model():
    """Design-Expert's linear blending row: the linear model about the mean, over the quadratic model's residual."""
    from process_improve.experiments.reference_catalogue import _nested_test

    entry = _ccd()
    base = {"model": "1", "full": "A + B"}
    own = _nested_test(entry, {**base, "key": "f"})
    third = _nested_test(entry, {**base, "key": "f", "error": "A * B + I(A ** 2) + I(B ** 2)"})
    ss = _nested_test(entry, {**base, "key": "ss"})
    assert own == pytest.approx(ss / 2 / (18.7 / 10), rel=1e-2)
    assert third == pytest.approx(ss / 2 / 0.0708, rel=1e-2)


def test_the_canonical_analysis_names_only_the_factors_the_model_holds():
    """A block in the entry adds no flat direction: Example 11.2's two eigenvalues, their sum and their product."""
    entry = _ccd(blocks=True)
    fits = reference_catalogue._Fits()
    check = {"model": "A * B + I(A ** 2) + I(B ** 2)", "method": "canonical_analysis"}
    out = fits(entry, check)
    eigenvalues = [
        reference_catalogue._optimum(out, {**check, "key": f"canonical_analysis.eigenvalues.{i}"}, entry)
        for i in (0, 1)
    ]
    assert eigenvalues == pytest.approx([-1.4141, -0.9634], abs=1e-4)
    assert reference_catalogue._optimum(out, {**check, "key": "canonical_analysis.eigenvalues.len"}, entry) == 2
    assert reference_catalogue._optimum(
        out, {**check, "key": "canonical_analysis.eigenvalues.sum"}, entry
    ) == pytest.approx(-2.3775, abs=1e-4)
    assert reference_catalogue._optimum(
        out, {**check, "key": "canonical_analysis.eigenvalues.prod"}, entry
    ) == pytest.approx(1.3624, abs=1e-4)


def test_desirability_scores_each_fitted_response_at_a_point():
    """At the design centre the yield is 79.94: 0.994 of the way from 70 to 80; the composite is a geometric mean."""
    from process_improve.experiments.reference_catalogue import _desirability

    entry = _ccd()
    entry["responses"].append({"name": "z"})
    entry["runs"]["columns"].append("z")
    for row, z in zip(entry["runs"]["rows"], [62, 60, 66, 59, 72, 69, 68, 70, 71, 68, 71, 58, 57], strict=True):
        row.append(z)
    quadratic = "A * B + I(A ** 2) + I(B ** 2)"
    goals = [
        {"response": "y", "model": quadratic, "goal": "maximize", "low": 70, "high": 80},
        {"response": "z", "model": quadratic, "goal": "target", "low": 62, "target": 65, "high": 68},
    ]
    check = {"point": {"A": 0.0, "B": 0.0}, "goals": goals}
    first = _desirability(entry, {**check, "key": "d.0"})
    second = _desirability(entry, {**check, "key": "d.1"})
    assert first == pytest.approx(0.994, abs=1e-3)
    assert second == 0  # the fitted viscosity at the centre, 70, is above its upper limit 68
    assert _desirability(entry, check) == 0


def test_spherical_and_blocked_designs_are_read_from_their_runs():
    """Example 11.2's CCD: the axial runs lie at √2; the blocks are orthogonal with two centre runs in each."""
    from process_improve.experiments.reference_catalogue import _blocks_orthogonal, _run_radius

    entry = _ccd(blocks=True)
    assert _run_radius(entry, {"runs": [10, 11, 12, 13], "key": "min"}) == pytest.approx(math.sqrt(2))
    assert _run_radius(entry, {"key": "max"}) == pytest.approx(math.sqrt(2))
    assert (
        _blocks_orthogonal(entry, {"block": "block"}) == 0
    )  # five centre runs with the cube, none with the axial runs
    entry["runs"]["rows"] = [
        [*row[:2], 2 if i in (7, 8) else row[2], row[3]] for i, row in enumerate(entry["runs"]["rows"])
    ]
    entry["runs"]["rows"][4][2] = 1
    # three centre runs with the cube and two with the axial runs: still not the share Equation 11.18 needs
    assert _blocks_orthogonal(entry, {"block": "block"}) == 0
    half = [r for r in entry["runs"]["rows"] if r[0] or r[1]] + [
        [0, 0, 1, 80],
        [0, 0, 1, 80],
        [0, 0, 2, 80],
        [0, 0, 2, 80],
    ]
    entry["runs"]["rows"] = half
    assert (
        _blocks_orthogonal(entry, {"block": "block"}) == 1
    )  # Montgomery's p. 509 design: two centre runs in each block


def test_a_definitive_screening_design_is_fold_over_pairs_with_squares_orthogonal_to_main_effects():
    """Figure 11.32's design for four factors: four fold-over pairs and a centre run."""
    from process_improve.experiments.reference_catalogue import (
        _estimable_projections,
        _fold_over_pairs,
        _interaction_columns,
    )

    text = ["0+--", "0-++", "-0-+", "+0+-", "--0-", "++0+", "-++0", "+--0", "0000"]
    rows = [[{"+": 1, "-": -1, "0": 0}[c] for c in t] for t in text]
    names = ["x1", "x2", "x3", "x4"]
    entry = {
        "name": "dsd",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in names],
        "runs": {"columns": names, "rows": rows, "coded": True},
    }
    assert _fold_over_pairs(entry, {"pairs": 4}) == 0
    entry["runs"]["rows"] = [rows[1], rows[0], rows[3], rows[2], *rows[4:]]
    assert _fold_over_pairs(entry, {"pairs": 4}) == 0
    entry["runs"]["rows"] = [rows[0], rows[2], *rows[1:2], *rows[3:]]
    assert _fold_over_pairs(entry, {"pairs": 4}) == 2
    entry["runs"]["rows"] = rows
    assert _interaction_columns(entry, {"key": "max_abs_r_main_square"}) == 0
    assert _interaction_columns(entry, {"key": "max_abs_r_main_interaction"}) == 0
    # nine runs cannot hold the ten terms of a full quadratic in three of the four factors
    assert _estimable_projections(entry, {"order": 3, "squares": True}) == 4
    assert _estimable_projections(entry, {"order": 3}) == 0


def test_a_contrast_of_a_saturated_fit_is_read_without_a_standard_error():
    """One cycle of an EVOP: each condition run once, the effects are contrasts of the five readings."""
    entry = {
        "name": "evop",
        "factors": [{"name": "point", "type": "categorical", "low": 1, "high": 5, "levels": [1, 2, 3, 4, 5]}],
        "responses": [{"name": "y"}],
        "runs": {
            "columns": ["point", "y"],
            "rows": [[1, 84.5], [2, 84.2], [3, 84.9], [4, 84.5], [5, 84.3]],
            "coded": False,
        },
    }
    out = reference_catalogue._Fits()(entry, {"model": "0 + point"})
    weights = {"point[3]": 0.5, "point[4]": 0.5, "point[2]": -0.5, "point[5]": -0.5}
    assert reference_catalogue._contrast(out, {"weights": weights}, entry) == pytest.approx(0.45)


def test_a_library_value_or_design_property_may_be_squared():
    """Montgomery's p. 510: (nF)^1/2 = 2.83 for the rotatable CCD in three factors, and 64/22 = 2.91 by Eq. 11.19."""
    from process_improve.experiments.reference_catalogue import _design_property, _library_value

    rotatable = {
        "design": {"design_type": "ccd", "k": 3, "options": {"alpha": "rotatable"}},
        "property": "alpha",
        "squared": True,
    }
    assert _design_property(rotatable) == pytest.approx(math.sqrt(8))
    assert _library_value({"function": "blocking_alpha", "args": [8, 3, 3, 2], "squared": True}) == pytest.approx(
        64 / 22
    )


def test_a_response_model_transmits_the_variance_of_its_noise_factor():
    """Montgomery's Example 12.1: temperature (A) a noise factor of Example 6.2's filtration 2^4.

    The slope in A is 10.8125 - 9.0625 x2 + 8.3125 x3 (C and D), so with var(z) = 1 the variance model is
    its square plus the residual mean square, 19.51.
    """
    from process_improve.experiments.reference_catalogue import _transmitted_variance

    entry = _factorial(4, FILTRATION)
    check = {"model": "A + C + D + A:C + A:D", "noise": ["A"]}
    out = reference_catalogue._Fits()(entry, check)
    mse = out["model_summary"]["mse_residual"]
    assert mse == pytest.approx(19.5125)
    read = {
        key: _transmitted_variance(out, {**check, "key": key}, entry) for key in ("Intercept", "C", "D", "C²", "C·D")
    }
    assert read["Intercept"] == pytest.approx(10.8125**2 + mse)
    assert read["C"] == pytest.approx(-2 * 10.8125 * 9.0625)
    assert read["D"] == pytest.approx(2 * 10.8125 * 8.3125)
    assert read["C²"] == pytest.approx(9.0625**2)
    assert read["C·D"] == pytest.approx(-2 * 9.0625 * 8.3125)
    assert _transmitted_variance(out, {**check, "key": "Intercept", "error": False}, entry) == pytest.approx(10.8125**2)
    point = {"C": 1.0, "D": 0.0}
    poe = _transmitted_variance(out, {**check, "key": "poe", "point": point}, entry)
    assert poe == pytest.approx(math.sqrt((10.8125 - 9.0625) ** 2 + mse))
    halved = _transmitted_variance(out, {**check, "key": "variance", "point": point, "sigma_z": {"A": 0.5}}, entry)
    assert halved == pytest.approx(0.5 * (10.8125 - 9.0625) ** 2 + mse)


def test_a_model_of_ln_variance_is_read_back_as_a_variance():
    """A fit with transform log predicts on the log scale; exp: true returns the variance itself."""
    entry = _factorial(2, [0.090, 0.071, 0.001, 0.008])
    fits = reference_catalogue._Fits()
    check = {"kind": "prediction", "model": "B", "transform": "log", "point": {"B": 0.5}}
    expected = {"what": "variance", "value": "0.0", "where": {}}
    found, _ = reference_catalogue._reproduce(entry, expected, {**check, "exp": True}, fits)
    logs = np.log([0.090, 0.071, 0.001, 0.008])
    slope = (logs[2] + logs[3] - logs[0] - logs[1]) / 4
    assert float(found) == pytest.approx(math.exp(logs.mean() + 0.5 * slope), rel=1e-5)
