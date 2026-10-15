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

import pytest

from process_improve.experiments import reference_catalogue
from process_improve.experiments.reference_catalogue import (
    reference_designs,
    reference_experiments,
    reference_sources,
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


def test_a_variance_component_is_read_from_the_library_reml_fit():
    """Balanced, REML gives the analysis-of-variance estimates: the pooled variance within groups, 1.25, and between."""
    from process_improve.experiments.reference_catalogue import _variance_component

    rows = [["a", 1], ["a", 3], ["b", 4], ["b", 6], ["c", 1], ["c", 2], ["d", 5], ["d", 6]]
    entry = {"responses": [{"name": "y"}], "runs": {"columns": ["group", "y"], "rows": rows}}
    check = {"groups": ["group"]}
    assert _variance_component(entry, {**check, "key": "residual"}) == pytest.approx(1.25)
    between = (sum((sum(r[1] for r in rows[i : i + 2]) / 2 - 3.5) ** 2 for i in range(0, 8, 2)) / 3) - 1.25 / 2
    assert _variance_component(entry, {**check, "key": "group"}) == pytest.approx(between, abs=1e-6)
