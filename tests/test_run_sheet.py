"""Tests for run sheets: a design written out for the lab, and the results read back in."""

from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import process_improve.experiments.tools  # noqa: F401 - registers the experiments tools
from process_improve.experiments import Factor, generate_design, read_run_sheet, run_sheet
from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import execute_tool_call

FACTORS = [
    Factor(name="NCO_OH", low=0.9, high=1.3),
    Factor(name="Catalyst", low=0, high=400, units="ppm"),
    Factor(name="Supplier", type="categorical", levels=["north", "south"]),
]
NAMES = [f.name for f in FACTORS]


@pytest.fixture(scope="module")
def design() -> object:
    return generate_design(FACTORS, "full_factorial", n_center_points=2, random_state=1)


class TestWriting:
    def test_the_sheet_keeps_the_run_order_and_leaves_the_responses_empty(self, design: object) -> None:
        sheet = run_sheet(design, FACTORS, responses=["Hardness", "Gloss"])  # type: ignore[arg-type]
        assert list(sheet.columns) == ["Run", *NAMES, "Hardness", "Gloss"]
        assert sheet["Run"].tolist() == list(range(1, len(sheet) + 1))
        actual = design.design_actual  # type: ignore[attr-defined]
        assert sheet[NAMES].to_numpy().tolist() == actual[NAMES].to_numpy().tolist()
        assert sheet[["Hardness", "Gloss"]].isna().all().all()
        assert sheet.attrs["columns"] == {n: n for n in NAMES}

    def test_coded_units_run_from_minus_one_to_plus_one_and_levels_stay_levels(self, design: object) -> None:
        sheet = run_sheet(design, FACTORS, units="coded", labels="X")  # type: ignore[arg-type]
        assert list(sheet.columns) == ["Run", "X01", "X02", "X03", "y"]
        assert set(sheet["X01"].round(9)) == {-1.0, 0.0, 1.0}
        assert set(sheet["X03"]) == {"north", "south"}

    def test_labels_can_be_given_one_by_one(self, design: object) -> None:
        sheet = run_sheet(design, FACTORS, labels={"Catalyst": "DBTDL [ppm]"}, responses=())  # type: ignore[arg-type]
        assert list(sheet.columns) == ["Run", "NCO_OH", "DBTDL [ppm]", "Supplier"]

    def test_a_hundred_factors_get_three_digit_labels(self) -> None:
        factors = [Factor(name=f"F{i}", low=0, high=1) for i in range(100)]
        runs = pd.DataFrame([dict.fromkeys((f.name for f in factors), 0.5)])
        sheet = run_sheet(runs, factors, labels="X", responses=())
        assert sheet.columns[1] == "X001"
        assert sheet.columns[-1] == "X100"

    def test_runs_from_the_next_step_keep_their_block(self) -> None:
        runs = [{"NCO_OH": 1.1, "Catalyst": 200.0, "Supplier": "north", "Block": 2}]
        assert run_sheet(runs, FACTORS)["Block"].tolist() == [2]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"units": "metric"}, "units must be"),
            ({"labels": "Y"}, "labels must be"),
            ({"labels": {"Pressure": "X09"}}, "not factors"),
            ({"labels": {"NCO_OH": "X01", "Catalyst": "X01"}}, "column of its own"),
            ({"labels": {"NCO_OH": "Run"}}, "column of its own"),
            ({"responses": ["y", "y"]}, "response needs a column"),
            ({"responses": ["NCO_OH"]}, "response needs a column"),
        ],
    )
    def test_bad_choices_are_refused(self, design: object, change: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            run_sheet(design, FACTORS, **change)  # type: ignore[arg-type]

    def test_a_missing_factor_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"no column for the factor\(s\) \['Supplier'\]"):
            run_sheet(pd.DataFrame({"NCO_OH": [1.0], "Catalyst": [0.0]}), FACTORS)


class TestReading:
    @pytest.mark.parametrize("units", ["actual", "coded"])
    @pytest.mark.parametrize("labels", [None, "X"])
    @pytest.mark.parametrize("through", ["csv", "xlsx", "frame"])
    def test_a_sheet_comes_back_in_actual_units(
        self, design: object, units: str, labels: str | None, through: str
    ) -> None:
        sheet = run_sheet(design, FACTORS, units=units, labels=labels, responses=["Hardness"])  # type: ignore[arg-type]
        sheet["Hardness"] = np.arange(len(sheet), dtype=float)
        if through == "csv":
            source: object = io.StringIO(sheet.to_csv(index=False))
        elif through == "xlsx":
            buf = io.BytesIO()
            sheet.to_excel(buf, index=False)
            source = buf.getvalue()
        else:
            source = sheet
        back = read_run_sheet(source, FACTORS, labels=labels)  # type: ignore[arg-type]
        actual = design.design_actual.reset_index(drop=True)  # type: ignore[attr-defined]
        assert back.attrs["units"] == units
        np.testing.assert_allclose(back[["NCO_OH", "Catalyst"]], actual[["NCO_OH", "Catalyst"]])
        assert back["Supplier"].tolist() == actual["Supplier"].tolist()
        assert back["Hardness"].tolist() == list(range(len(sheet)))
        assert list(back.columns) == [*NAMES, "Run", "Hardness"]

    def test_files_are_read_by_their_suffix(self, design: object, tmp_path: Path) -> None:
        sheet = run_sheet(design, FACTORS, labels="X")  # type: ignore[arg-type]
        sheet.to_csv(tmp_path / "runs.csv", index=False)
        sheet.to_excel(tmp_path / "runs.xlsx", index=False)
        for name in ("runs.csv", "runs.xlsx"):
            assert len(read_run_sheet(tmp_path / name, FACTORS, labels="X")) == len(sheet)
        (tmp_path / "runs.txt").write_text("x")
        with pytest.raises(ValueError, match="none of them"):
            read_run_sheet(tmp_path / "runs.txt", FACTORS)

    def test_the_sheet_with_every_factor_is_read_from_a_workbook(self, design: object) -> None:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf) as book:
            pd.DataFrame({"note": ["read me first"]}).to_excel(book, sheet_name="About", index=False)
            run_sheet(design, FACTORS, labels="X").to_excel(book, sheet_name="Runs", index=False)  # type: ignore[arg-type]
        assert len(read_run_sheet(buf.getvalue(), FACTORS, labels="X")) == len(design.design_actual)  # type: ignore[attr-defined]

    def test_columns_match_without_case_spaces_or_leading_zeros(self) -> None:
        table = pd.DataFrame({" x1 ": [1.1], "X2": [0.0], "x003": ["south"]})
        back = read_run_sheet(table, FACTORS, labels="X")
        assert back.attrs["columns"] == {"NCO_OH": " x1 ", "Catalyst": "X2", "Supplier": "x003"}

    def test_a_factor_is_also_found_under_its_own_name(self) -> None:
        table = pd.DataFrame({"nco_oh": [1.1], "Catalyst": [0.0], "Supplier": ["south"]})
        assert read_run_sheet(table, FACTORS, labels="X").attrs["columns"]["NCO_OH"] == "nco_oh"

    def test_blank_rows_are_left_out(self) -> None:
        table = pd.DataFrame({"NCO_OH": [1.1, None], "Catalyst": [0.0, None], "Supplier": ["south", None]})
        assert len(read_run_sheet(table, FACTORS)) == 1

    def test_settings_inside_both_scales_are_read_as_actual(self) -> None:
        """0 and 1 ppm of catalyst are actual settings; no coded value is out of a range here."""
        table = pd.DataFrame({"NCO_OH": [1.0, 1.0], "Catalyst": [0.0, 1.0], "Supplier": ["north", "south"]})
        assert read_run_sheet(table, FACTORS).attrs["units"] == "actual"
        assert read_run_sheet(table, FACTORS, units="coded")["Catalyst"].tolist() == [200.0, 400.0]

    def test_settings_outside_the_ranges_are_warned_about(self) -> None:
        table = pd.DataFrame({"NCO_OH": [0.8, 1.0], "Catalyst": [0.0, 500.0], "Supplier": ["north", "south"]})
        with pytest.warns(UserWarning, match=r"NCO_OH from 0.8 to 1 \(range 0.9 to 1.3\); Catalyst from 0 to 500"):
            read_run_sheet(table, FACTORS)

    def test_settings_inside_the_ranges_are_read_silently(self, design: object) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            read_run_sheet(run_sheet(design, FACTORS), FACTORS)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("table", "message"),
        [
            ({"NCO_OH": [1.0], "Catalyst": [0.0]}, r"no column for \['Supplier'\]"),
            ({"NCO_OH": ["high"], "Catalyst": [0.0], "Supplier": ["north"]}, "needs a number in every run"),
            ({"NCO_OH": [1.0, None], "Catalyst": [0.0, 1.0], "Supplier": ["north", "north"]}, "an empty cell"),
            ({"NCO_OH": [1.0], "Catalyst": [0.0], "Supplier": ["east"]}, "takes the levels"),
        ],
    )
    def test_bad_sheets_are_refused(self, table: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            read_run_sheet(pd.DataFrame(table), FACTORS)

    def test_a_file_that_is_not_a_workbook_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Excel workbook"):
            read_run_sheet(b"not a workbook", FACTORS)
        with pytest.raises(ValueError, match="units must be"):
            read_run_sheet(pd.DataFrame(), FACTORS, units="metric")  # type: ignore[arg-type]


class TestThroughTheTools:
    def _factors(self) -> list[dict]:
        return [f.model_dump(exclude_none=True) for f in FACTORS]

    def test_a_sheet_goes_out_and_comes_back(self, design: object) -> None:
        runs = design.design_actual.drop(columns="RunOrder").to_dict(orient="records")  # type: ignore[attr-defined]
        out = execute_tool_call(
            "write_run_sheet",
            {"runs": runs, "factors": self._factors(), "labels": "X", "units": "coded", "responses": ["Hardness"]},
        )
        assert out["columns"] == ["Run", "X01", "X02", "X03", "Hardness"]
        assert out["factor_columns"] == {"NCO_OH": "X01", "Catalyst": "X02", "Supplier": "X03"}
        filled = pd.read_csv(io.StringIO(out["csv"])).assign(Hardness=95.0)
        back = execute_tool_call(
            "read_run_sheet", {"csv_text": filled.to_csv(index=False), "factors": self._factors(), "labels": "X"}
        )
        json.dumps(back)
        assert back["units_read"] == "coded"
        assert [r["Catalyst"] for r in back["runs"]] == [r["Catalyst"] for r in runs]
        assert {r["Catalyst"] for r in back["runs_coded"]} == {-1.0, 0.0, 1.0}
        assert back["runs"][0]["Hardness"] == 95.0

    def test_a_workbook_goes_out_and_comes_back(self, design: object) -> None:
        runs = design.design_actual.drop(columns="RunOrder").to_dict(orient="records")  # type: ignore[attr-defined]
        out = execute_tool_call("write_run_sheet", {"runs": runs, "factors": self._factors(), "file_format": "xlsx"})
        back = execute_tool_call("read_run_sheet", {"xlsx_base64": out["xlsx_base64"], "factors": self._factors()})
        assert back["n_runs"] == len(runs)
        assert pd.read_excel(io.BytesIO(base64.b64decode(out["xlsx_base64"]))).shape == (len(runs), 5)

    def test_the_sheet_is_given_once(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="Give the run sheet once"):
            execute_tool_call("read_run_sheet", {"csv_text": "a\n1", "xlsx_base64": "", "factors": self._factors()})

    def test_errors_are_reported(self) -> None:
        result = execute_tool_call("read_run_sheet", {"csv_text": "a,b\n1,2", "factors": self._factors()})
        assert "no column" in result["error"]
        result = execute_tool_call("write_run_sheet", {"runs": [{"NCO_OH": 1.0}], "factors": self._factors()})
        assert "no column for the factor" in result["error"]
