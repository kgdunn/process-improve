# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrappers: ``write_run_sheet`` and ``read_run_sheet``, a design out to the lab and its results back."""

from __future__ import annotations

import base64
import datetime
import io
import zipfile
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

_FACTORS = (
    "The factors, as for generate_design: continuous ({'name', 'low', 'high', 'units'}) or categorical "
    "({'name', 'type': 'categorical', 'levels': [...]})."
)
_LABELS = (
    "How the factors' columns are named: omitted for the factors' own names, 'X' for X01, X02, ... in the order "
    "of factors, or a mapping from a factor's name to its column, e.g. {'Temperature': 'X01'}."
)
#: The one date a written workbook holds, in place of the time it was written: the earliest a zip archive can hold.
_WORKBOOK_DATE = datetime.datetime(1980, 1, 1, tzinfo=datetime.UTC)


class WriteRunSheetInput(BaseModel):
    """Input contract for ``write_run_sheet``."""

    model_config = ConfigDict(extra="forbid")

    runs: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "The runs in the factors' actual units, one dict per run: generate_design's 'design_actual', or the "
            "'runs' recommend_next_step returns. A 'Block' value is kept."
        ),
    )
    factors: list[dict[str, Any]] = Field(..., min_length=1, description=_FACTORS)
    units: Literal["actual", "coded"] = Field(
        "actual", description="Write the settings in actual units, or coded from -1 (low) to +1 (high)."
    )
    labels: Literal["X"] | dict[str, str] | None = Field(None, description=_LABELS)
    responses: list[str] = Field(["y"], description="Response columns, written empty for the results.")
    file_format: Literal["csv", "xlsx"] = Field(
        "csv", description="'csv' returns the sheet as CSV text; 'xlsx' as a base64-encoded Excel workbook."
    )


class ReadRunSheetInput(BaseModel):
    """Input contract for ``read_run_sheet``."""

    model_config = ConfigDict(extra="forbid")

    csv_text: str | None = Field(None, description="The run sheet as CSV text, with a header row.")
    xlsx_base64: str | None = Field(None, description="The run sheet as a base64-encoded .xlsx or .xls workbook.")
    factors: list[dict[str, Any]] = Field(..., min_length=1, description=_FACTORS)
    labels: Literal["X"] | dict[str, str] | None = Field(None, description=_LABELS)
    units: Literal["auto", "actual", "coded"] = Field(
        "auto",
        description=(
            "The units of the settings. 'auto' reads them as coded when every setting is between -1 and +1 and "
            "some cannot be an actual value, otherwise as actual; the reply says which."
        ),
    )

    @model_validator(mode="after")
    def _one_source(self) -> ReadRunSheetInput:
        if (self.csv_text is None) == (self.xlsx_base64 is None):
            raise ValueError("Give the run sheet once: as csv_text or as xlsx_base64.")
        return self


def _without_the_clock(workbook: bytes) -> bytes:
    """Return ``workbook`` with ``_WORKBOOK_DATE`` in place of the time openpyxl wrote into it.

    Parameters
    ----------
    workbook : bytes
        An .xlsx workbook as openpyxl writes it: the time of the call is the date of every zip member
        and the created and modified dates in ``docProps/core.xml``.

    Returns
    -------
    bytes
        The same workbook, its members in the same order and compression, with ``_WORKBOOK_DATE`` for each
        of those dates, so the same sheet gives the same bytes.
    """
    from openpyxl.packaging.core import DocumentProperties  # noqa: PLC0415
    from openpyxl.xml.functions import fromstring, tostring  # noqa: PLC0415

    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(workbook)) as source, zipfile.ZipFile(out, "w") as target:
        for member in source.infolist():
            content = source.read(member)
            if member.filename == "docProps/core.xml":
                properties = DocumentProperties.from_tree(fromstring(content))
                properties.created = properties.modified = _WORKBOOK_DATE
                content = tostring(properties.to_tree())
            member.date_time = _WORKBOOK_DATE.timetuple()[:6]
            target.writestr(member, content)
    return out.getvalue()


@tool_spec(
    name="write_run_sheet",
    description=(
        "Write runs as a run sheet for the people or the system that make them: one row per run in the order "
        "given, numbered in a 'Run' column, the settings in actual or coded units under the factors' names or "
        "labels such as X01..Xnn, and empty response columns to fill in. Returns the sheet as CSV text (or a "
        "base64 .xlsx), its columns, and which factor each column holds. Read the filled sheet back with "
        "read_run_sheet."
    ),
    input_model=WriteRunSheetInput,
    examples="""
    # "Give me the runs as a CSV with columns X01..X12 in real units, for the upload"
        -> ``write_run_sheet(runs=<design_actual>, factors=[...], labels="X", responses=["Hardness"])``
    """,
    category="experiments",
)
def write_run_sheet_tool(spec: WriteRunSheetInput) -> dict[str, Any]:
    """Write the runs as a run sheet."""
    try:
        from process_improve.experiments.factor import Factor  # noqa: PLC0415
        from process_improve.experiments.run_sheet import run_sheet  # noqa: PLC0415

        factors = [Factor(**f) for f in spec.factors]
        sheet = run_sheet(spec.runs, factors, units=spec.units, labels=spec.labels, responses=spec.responses)
        output: dict[str, Any] = {
            "columns": list(sheet.columns),
            "factor_columns": sheet.attrs["columns"],
            "n_runs": len(sheet),
        }
        if spec.file_format == "csv":
            output["csv"] = sheet.to_csv(index=False)
        else:
            buf = io.BytesIO()
            sheet.to_excel(buf, index=False, sheet_name="Runs")
            output["xlsx_base64"] = base64.b64encode(_without_the_clock(buf.getvalue())).decode("ascii")
        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool write_run_sheet failed")
        return {"error": str(e)}


@tool_spec(
    name="read_run_sheet",
    description=(
        "Read a filled run sheet (CSV text or a base64 Excel workbook) back into the factors' own names and "
        "actual units, whichever units and labels (X01..Xnn, or the names) it was written in. Returns 'runs' "
        "(the settings in actual units with the sheet's other columns, such as the responses), 'runs_coded', "
        "'units_read' and 'factor_columns'. Settings outside a factor's range come back with a warning under "
        "library_warnings. Pass 'runs' on to analyze_experiment or recommend_next_step."
    ),
    input_model=ReadRunSheetInput,
    examples="""
    # "Here are the results the site gave back; read them in"
        -> ``read_run_sheet(csv_text="X01,X02,...,Hardness\\n1.1,160,...,92.3\\n...", factors=[...], labels="X")``
    """,
    category="experiments",
)
def read_run_sheet_tool(spec: ReadRunSheetInput) -> dict[str, Any]:
    """Read a filled run sheet."""
    try:
        from process_improve.experiments.factor import Factor  # noqa: PLC0415
        from process_improve.experiments.run_sheet import read_run_sheet, run_sheet  # noqa: PLC0415

        factors = [Factor(**f) for f in spec.factors]
        source: io.StringIO | bytes = (
            io.StringIO(spec.csv_text) if spec.csv_text is not None else base64.b64decode(spec.xlsx_base64 or "")
        )
        runs = read_run_sheet(source, factors, labels=spec.labels, units=spec.units)
        names = [f.name for f in factors]
        coded = run_sheet(runs, factors, units="coded", responses=())[names]
        return clean(
            {
                "runs": runs.to_dict(orient="records"),
                "runs_coded": pd.concat([coded, runs.drop(columns=names)], axis=1).to_dict(orient="records"),
                "units_read": runs.attrs["units"],
                "factor_columns": runs.attrs["columns"],
                "n_runs": len(runs),
            }
        )
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool read_run_sheet failed")
        return {"error": str(e)}


_register("write_run_sheet")
_register("read_run_sheet")
