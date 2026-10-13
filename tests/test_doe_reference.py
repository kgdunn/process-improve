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

from process_improve.experiments.reference import (
    reference_designs,
    reference_experiments,
    reference_sources,
    reproduce_reference,
)

REPORT = reproduce_reference()
CHECKED = REPORT[REPORT["checked"]]


@pytest.mark.parametrize(
    "row",
    [row for _, row in CHECKED.iterrows()],
    ids=[f"{r.experiment}:{r.what}:{r.term}"[:120] for r in CHECKED.itertuples()],
)
def test_every_printed_number_is_reproduced(row):
    assert row["agrees"], (
        f"{row['experiment']}: {row['what']} {row['term']}: printed {row['printed']}, found {row['found']}"
        f"{'; ' + row['reason'] if row['reason'] else ''} (printed at {row['where']})"
    )


def test_every_number_not_recomputed_says_why():
    unchecked = REPORT[~REPORT["checked"]]
    assert unchecked["reason"].str.len().gt(0).all()
    assert not unchecked["reason"].eq("TO CHECK").any(), "every number is given a check or the reason it has none"


def test_every_experiment_says_where_it_was_printed():
    sources = reference_sources()
    for entry in [*reference_experiments(), *reference_designs()]:
        name = entry["name"]
        assert entry["source"] in sources, name
        assert entry["where"], name
        assert entry["expected"], f"{name}: what the source prints is part of the entry"
        for expected in entry["expected"]:
            where = expected["where"]
            assert where.get("quote"), f"{name}: every number carries a quotation of where it is printed"
            assert where.get("line") or where.get("page"), f"{name}: every number carries its line or page"
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
