"""The reference designed experiments, reproduced with this library.

``process_improve.experiments.reference`` holds worked examples, exercises and case studies of
published sources, each with its runs as printed and every number the source prints for it, with
where it is printed. ``reproduce_reference`` recomputes every number that carries a check; here
each one must agree with the printed text to within the source's own rounding. A number that is
not recomputed must say why. A change to the library that contradicts a published source fails
here, and so does a number copied wrongly into the catalogue.
"""

from __future__ import annotations

import json

import pytest

from process_improve.experiments import reference
from process_improve.experiments.reference import (
    reference_designs,
    reference_experiments,
    reference_sources,
    reproduce_reference,
)

CHECKS = list(reference._each_check())
FITS = reference._Fits()
SLOW_DESIGNS = {"omars_ilp", "dsd"}  # built by search: seconds each, so each is made on one worker, once


def _case(key: str, entry: dict, expected: dict, check: dict) -> pytest.param:
    marks = []
    design = check.get("design") or {}
    if design.get("design_type") in SLOW_DESIGNS:
        spec = json.dumps(design, sort_keys=True)
        marks = [pytest.mark.slow, pytest.mark.xdist_group(f"reference-design/{spec}")]
    name = f"{key}:{entry['name']}:{expected['what']}:{expected.get('term')}"[:120]
    return pytest.param(entry, expected, check, id=name, marks=marks)


@pytest.mark.parametrize(("entry", "expected", "check"), [_case(*c) for c in CHECKS if "kind" in c[3]])
def test_every_printed_number_is_reproduced(entry, expected, check):
    found, agrees = reference._reproduce(entry, expected, check, FITS)
    printed = check.get("printed", expected["value"])
    if check.get("disagreement"):
        assert not agrees, f"{entry['name']}: it reproduces now ({found}); take away the disagreement it records"
        return
    assert agrees, f"{entry['name']}: {expected['what']} {expected.get('term')}: printed {printed}, found {found}"


def test_every_number_not_recomputed_says_why():
    reasons = [check.get("manual") for *_, check in CHECKS if "kind" not in check]
    assert all(reasons), "a number with no check says why it is not recomputed"
    assert "TO CHECK" not in reasons, "every number is given a check or the reason it has none"


def test_the_report_has_a_row_for_every_check():
    report = reproduce_reference("ullmann-doe-2012")
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
        assert all(source.get(field) for field in ("title", "url", "year", "read")), key


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="Unknown source 'nowhere'"):
        reference_experiments("nowhere")


def test_a_wrong_number_is_reported_and_not_agreed(monkeypatch, tmp_path):
    """The report says so when a printed number does not reproduce: a check that cannot fail checks nothing."""
    from process_improve.experiments import reference

    source = json.loads(next(reference._REFERENCE_DIR.glob("ullmann-*.json")).read_text(encoding="utf-8"))
    source["experiments"][0]["expected"][0]["value"] = "11"  # the effect of A is 10
    (tmp_path / "wrong.json").write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(reference, "_REFERENCE_DIR", tmp_path)
    report = reproduce_reference()
    first = report.iloc[0]
    assert (first["printed"], first["found"], first["agrees"]) == ("11", "10", False)
