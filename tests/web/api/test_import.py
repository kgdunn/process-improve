"""A table of the reader's own: ``api_read_table`` reads it, and ``api_import_study`` makes the study of it.

The table is pasted text or a .csv or .xlsx file, of a design run elsewhere or of a plant's records. Its
runs become a design of runs as given, so Learn does with them what it does with any design's; the
design's notes say what such runs can and cannot show.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest


def _yield(i: int, t: float, p: float, c: str) -> str:
    """Return a run's yield as a spreadsheet in a country of decimal commas writes it: 60,5."""
    return str(round(60 + 0.1 * (t - 175) - 3 * (p - 2) + (2 if c == "B" else 0) + np.sin(i), 2)).replace(".", ",")


#: A 2^3 factorial run twice, its lines split by semicolons.
DESIGNED = "Run;Temp (degC);Pressure [bar];Catalyst;Yield (%)\n" + "\n".join(
    f"{i + 1};{t};{p};{c};{_yield(i, t, p, c)}"
    for i, (t, p, c) in enumerate([(t, p, c) for t in (150, 200) for p in (1, 3) for c in "AB"] * 2)
)


def records(n: int = 40) -> str:
    """Return a plant's records: nothing set, T and P moving together, and the yield following T and F."""
    rng = np.random.default_rng(1)
    t = rng.normal(180, 5, n)
    p = 2 + 0.05 * (t - 180) + rng.normal(0, 0.02, n)
    f = rng.normal(15, 2, n)
    y = 50 + 0.5 * (t - 180) + 1.5 * (f - 15) + rng.normal(0, 0.5, n)
    lines = [f"2026-01-{i + 1:02d},{t[i]:.2f},{p[i]:.3f},{f[i]:.2f},{y[i]:.2f}" for i in range(n)]
    return "date,T,P,F,yield\n" + "\n".join(lines)


def roles(table: dict) -> dict[str, str]:
    """Index a read table's guessed roles by heading."""
    return {c["heading"]: c["role"] for c in table["columns"]}


def study(api, table: dict, **more: object) -> dict:
    """Make the study of a read table, with the roles as they stand."""
    return api.ok("api_import_study", {"columns": table["columns"], "rows": table["rows"], **more})


def analysed(api, campaign: dict) -> dict:
    """Analyse an imported study's first response, as Learn does when the study opens."""
    design = campaign["design"]
    request = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"], "model": campaign["model"]}
    return api.ok("api_analyze_runs", {**request, "responses": campaign["responses"]})


# --------------------------------------------------------------------------- reading
def test_a_designed_table_is_read_with_its_units_decimal_commas_and_roles(api):
    table = api.ok("api_read_table", {"text": DESIGNED})
    assert table["n_rows"] == 16
    assert roles(table) == {
        "Run": "ignore",
        "Temp (degC)": "factor",
        "Pressure [bar]": "factor",
        "Catalyst": "factor",
        "Yield (%)": "response",
    }
    temp, _, catalyst, y = table["columns"][1:]
    assert (temp["name"], temp["units"], temp["low"], temp["high"]) == ("Temp", "degC", 150, 200)
    assert (catalyst["kind"], catalyst["levels"]) == ("text", ["A", "B"])
    assert (y["name"], y["units"], y["kind"]) == ("Yield", "%", "number"), "60,5 is read as 60.5"


def test_in_records_where_nothing_was_set_the_last_measured_column_is_the_response(api):
    table = api.ok("api_read_table", {"text": records()})
    assert roles(table) == {"date": "ignore", "T": "factor", "P": "factor", "F": "factor", "yield": "response"}


def test_a_workbook_is_read_by_its_values_and_its_formulas_are_never_evaluated(api):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Trials"
    for row in (["A", "B", "y"], [1, 10, 5.5], [2, 10, 6.5], [1, 20, 7.5], [2, 20, "=SUM(A2:A4)"]):
        sheet.append(row)
    book.create_sheet("Notes").append(["nothing here"])
    held = io.BytesIO()
    book.save(held)
    table = api.ok("api_read_table", {"xlsx_base64": base64.b64encode(held.getvalue()).decode()})
    assert (table["sheets"], table["sheet"]) == (["Trials", "Notes"], "Trials")
    assert table["rows"][-1] == [2, 20, None], "a formula has no value until a spreadsheet works it out"


def test_blank_lines_and_columns_are_left_out_and_blanks_are_read_as_missing(api):
    text = "\n\nA,,B,y\n1,,10,5\n\n2,,10,NA\n1,,20,7\n2,,20,8\n"
    table = api.ok("api_read_table", {"text": text})
    assert [c["heading"] for c in table["columns"]] == ["A", "B", "y"]
    assert table["rows"][1] == [2, 10, None]


# --------------------------------------------------------------------------- the study
def test_a_designed_table_becomes_a_study_learn_analyses(api):
    made = study(api, api.ok("api_read_table", {"text": DESIGNED}), title="Trials in March")
    campaign = made["campaign"]
    spec = campaign["design"]["spec"]
    assert spec["design_type"] == "given"
    assert [f["name"] for f in spec["factors"]] == ["Temp", "Pressure", "Catalyst"]
    assert (spec["response"], spec["response_info"]["units"]) == ("Yield", "%")
    assert (campaign["title"], campaign["model"], made["left_out"]) == ("Trials in March", "", [])
    assert campaign["sources"] == ["entered"] * 16
    leads = analysed(api, campaign)["findings"][0]
    assert (leads["kind"], leads["factor"], leads["up"], leads["low"], leads["high"]) == (
        "leads",
        "Pressure",
        False,
        1,
        3,
    )


def test_records_say_what_they_cannot_show(api):
    made = study(api, api.ok("api_read_table", {"text": records()}))
    assert made["notes"][0] == (
        "T and P moved together in these runs (correlation above 0.99): their effects cannot be told apart."
    )
    assert made["notes"][1].startswith("These runs were not set out as an experiment: ")
    assert made["campaign"]["design"]["facts"]["notes"][-2:] == made["notes"], (
        "kept with the design, for Design to show"
    )
    assert made["campaign"]["model"] == "main_effects", "records seldom hold a surface's worth of settings"
    analysis = analysed(api, made["campaign"])
    vif = {c["term"]: c["vif"] for c in analysis["coefficients"] if c["term"] != "Intercept"}
    assert min(vif["T"], vif["P"]) > 10, "the table of coefficients shows how far each is inflated"
    assert vif["F"] < 2


def test_a_run_with_no_setting_for_a_factor_is_left_out_and_said(api):
    table = api.ok("api_read_table", {"text": "A,B,y\n1,10,5\n2,,6\n1,20,7\n2,20,8\n2,10,6.5\n"})
    made = study(api, table)
    assert made["left_out"] == [3], "by its line in the table"
    assert made["campaign"]["design"]["n_runs"] == 4
    assert made["notes"][-1] == "1 of the table's 5 runs had no setting for some factor, and were left out."


def test_a_run_with_no_result_is_kept_as_a_run_still_to_do(api):
    made = study(api, api.ok("api_read_table", {"text": "A,B,y\n1,10,5\n2,10,\n1,20,7\n2,20,8\n"}))
    assert made["campaign"]["responses"] == [5, None, 7, 8]
    assert made["campaign"]["sources"] == ["entered", None, "entered", "entered"]


def test_several_responses_are_each_a_response_of_the_study(api):
    text = "A,B,strength,cost\n" + "\n".join(
        f"{a},{b},{10 + a + 0.31 * i},{3 + b / 7 + 0.17 * i}"
        for i, (a, b) in enumerate([(1, 10), (2, 10), (1, 20), (2, 20)] * 2)
    )
    made = study(api, api.ok("api_read_table", {"text": text}))
    assert [r["name"] for r in made["campaign"]["design"]["spec"]["responses"]] == ["strength", "cost"]
    assert list(made["campaign"]["more_responses"]) == ["cost"]


def test_the_reader_s_choices_are_the_study_s(api):
    """A column guessed a factor is made the response, and a factor is given a wider range and a name."""
    table = api.ok("api_read_table", {"text": DESIGNED})
    by = {c["heading"]: c for c in table["columns"]}
    by["Temp (degC)"].update(name="T", low=100, high=250)
    by["Catalyst"]["role"] = "ignore"
    spec = study(api, table)["campaign"]["design"]["spec"]
    assert spec["factors"][0] == {"name": "T", "low": 100.0, "high": 250.0, "units": "degC"}
    assert [f["name"] for f in spec["factors"]] == ["T", "Pressure"]


# --------------------------------------------------------------------------- the edges of a table
def workbook(rows: list[list]) -> str:
    """Return a workbook of one sheet holding ``rows``, in base64 as the page sends a file."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    for row in rows:
        book.active.append(row)
    held = io.BytesIO()
    book.save(held)
    return base64.b64encode(held.getvalue()).decode()


def test_a_workbook_longer_than_a_study_is_read_no_further_than_it_needs_to_be_refused(api):
    rows = [["A", "y"], *[[i % 2, i] for i in range(2100)]]
    said = api.error("api_read_table", {"xlsx_base64": workbook(rows)})
    assert said == "The table has 2050 runs; a study holds 2000 at most.", "read just past what a study holds"


def test_a_file_that_is_not_base64_is_refused_as_no_workbook(api):
    assert api.error("api_read_table", {"xlsx_base64": "not base64!"}) == (
        "The file could not be read as an Excel workbook (.xlsx)."
    )


def test_more_measured_columns_than_a_study_s_responses_are_left_out(api):
    """Seven measured columns beside two that were set: the first five are responses, as a study has at most."""
    head = "A,B," + ",".join(f"y{j}" for j in range(7))
    lines = [
        f"{a},{b}," + ",".join(str(10 * j + a + b / 7 + 0.3 * i) for j in range(7))
        for i, (a, b) in enumerate([(1, 10), (2, 10), (1, 20), (2, 20)] * 2)
    ]
    guessed = roles(api.ok("api_read_table", {"text": head + "\n" + "\n".join(lines)}))
    assert [guessed[f"y{j}"] for j in range(7)] == ["response"] * 5 + ["ignore"] * 2


def test_more_columns_that_were_set_than_a_study_s_factors_are_left_out(api):
    head = ",".join(f"x{j}" for j in range(42)) + ",y"
    lines = [",".join(str((i >> (j % 5)) & 1) for j in range(42)) + f",{50 + 1.7 * i}" for i in range(32)]
    guessed = roles(api.ok("api_read_table", {"text": head + "\n" + "\n".join(lines)}))
    assert list(guessed.values()).count("factor") == 40, "a study has 40 factors at most"
    assert (guessed["x40"], guessed["x41"], guessed["y"]) == ("ignore", "ignore", "response")


def test_one_factor_has_no_other_to_move_with(api):
    made = study(api, api.ok("api_read_table", {"text": "A,y\n1,5\n2,6.1\n1,5.2\n2,6.3\n"}))
    assert made["notes"] == []


def test_many_pairs_that_moved_together_are_counted_past_the_strongest_five(api):
    """Four factors that all follow one another make six pairs: five are named, and the sixth counted."""
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 30)
    columns = [base + rng.normal(0, 0.05, 30) for _ in range(4)]
    y = base + rng.normal(0, 0.1, 30)
    text = "a,b,c,d,y\n" + "\n".join(",".join(f"{c[i]:.4f}" for c in [*columns, y]) for i in range(30))
    notes = study(api, api.ok("api_read_table", {"text": text}))["notes"]
    assert sum("moved together in these runs" in note for note in notes) == 5
    assert "1 more pairs of factors moved together." in notes


# --------------------------------------------------------------------------- what a careful reviewer tried
def test_time_is_a_factor_and_one_shift_column_does_not_make_records_a_design(api):
    """Time is among the commonest factors; and a plant's shift, at three settings, is not a design's doing."""
    timed = "Time (min),Temp,y\n" + "".join(
        f"{t},{c},{50 + t / 10 + c / 20 + 0.37 * i}\n"
        for i, (t, c) in enumerate([(10, 150), (30, 150), (10, 200), (30, 200)] * 2)
    )
    assert roles(api.ok("api_read_table", {"text": timed}))["Time (min)"] == "factor"
    lines = records().split("\n")
    shifted = "\n".join([f"{lines[0]},shift", *[f"{line},{1 + i % 3}" for i, line in enumerate(lines[1:])]])
    shifted = "\n".join(
        ",".join([*row.split(",")[:4], row.split(",")[5], row.split(",")[4]]) for row in shifted.split("\n")
    )
    guessed = roles(api.ok("api_read_table", {"text": shifted}))
    assert guessed == {
        "date": "ignore",
        "T": "factor",
        "P": "factor",
        "F": "factor",
        "shift": "factor",
        "yield": "response",
    }


def test_a_name_typed_with_a_space_after_it_is_the_name_without(api):
    table = api.ok("api_read_table", {"text": DESIGNED})
    table["columns"][-1]["name"] = "Yield "
    made = study(api, table)
    assert made["campaign"]["design"]["spec"]["response"] == "Yield"
    assert len(made["campaign"]["responses"]) == 16


def test_a_number_too_large_to_hold_is_a_blank(api):
    table = api.ok("api_read_table", {"text": "A,B,y\n1,10,5\n2,10,6\n1,20,7\n2,20,8\n"})
    table["rows"][1][2] = 10**400
    assert study(api, table)["campaign"]["responses"] == [5, None, 7, 8]


def test_a_quote_never_closed_is_refused_in_a_sentence(api):
    text = 'A,B,y\n1,"2,3\n' + "4,5,6\n" * 30_000
    assert api.error("api_read_table", {"text": text}) == (
        "The table could not be split into columns: look for a quote that is opened and never closed."
    )


def zipped(name: str, size: int) -> str:
    """Return a zip file of one part of ``size`` zero bytes, in base64: small as sent, whatever it unpacks to."""
    import zipfile

    held = io.BytesIO()
    with zipfile.ZipFile(held, "w", zipfile.ZIP_DEFLATED) as packed:
        packed.writestr(name, b"\0" * size)
    return base64.b64encode(held.getvalue()).decode()


def test_a_workbook_that_unpacks_to_far_more_than_it_weighs_is_refused_before_it_is_opened(api):
    said = api.error("api_read_table", {"xlsx_base64": zipped("xl/sharedStrings.xml", 60_000_000)})
    assert said == "The workbook unpacks to more than this page reads: 50 MB."


def test_a_zip_file_that_is_no_workbook_is_refused(api):
    assert api.error("api_read_table", {"xlsx_base64": zipped("notes.txt", 10)}) == (
        "The file could not be read as an Excel workbook (.xlsx)."
    )
