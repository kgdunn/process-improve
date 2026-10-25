"""The multivariate session tools, driven as the MCP server drives them: open, fit, look, change, fit again.

Each tool takes the session the last one returned and returns it with one step added. The tests
check that the session is the whole record: a model grown from another keeps its own steps, a
row left out is still projected, the script of a model rebuilds it, and a session that was
tampered with, or names a model it does not have, is refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.config import settings
from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import execute_tool_call, get_tool_specs
from process_improve.workbench import Variant

TOOLS = ("mva_open", "mva_fit", "mva_explain", "mva_predict", "mva_script")


def call(tool: str, **arguments: object) -> dict:
    """Call a tool as the MCP server does, and fail on an error result."""
    result = execute_tool_call(tool, arguments)
    assert "error" not in result, result
    return result


@pytest.fixture
def table() -> pd.DataFrame:
    """Return a small table with a skewed column, a response, and rows named by text."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3))
    frame = pd.DataFrame(x, columns=["a", "b", "c"], index=[f"r{i}" for i in range(30)])
    frame["d"] = np.exp(2 * rng.normal(size=30))  # skewed: the column a log10 transform is for
    frame["y"] = x @ [1.0, -0.5, 0.2] + 0.1 * rng.normal(size=30)
    frame.index.name = "id"
    return frame


def opened(frame: pd.DataFrame) -> dict:
    return call("mva_open", source={"csv": frame.to_csv(), "name": "plant.csv"})


def test_the_five_tools_are_registered_and_refuse_unknown_keys() -> None:
    specs = {spec["name"]: spec for spec in get_tool_specs()}
    for name in TOOLS:
        assert specs[name]["input_schema"]["additionalProperties"] is False
    with pytest.raises(ToolInputInvalidError):
        execute_tool_call("mva_open", {"source": {"dataset": "LDPE"}, "extra": 1})


def test_open_describes_each_column(table: pd.DataFrame) -> None:
    result = opened(table)
    assert result["table"]["n_rows"] == 30
    assert result["table"]["first_row_ids"][:2] == ["r0", "r1"]
    d = next(c for c in result["table"]["columns"] if c["name"] == "d")
    assert d["skewness"] == pytest.approx(table["d"].skew())
    assert result["session"]["models"] == []


def test_a_model_grown_from_another_keeps_its_parents_settings_and_only_adds(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    first = call("mva_fit", session=session, name="M1", kind="PCA", n_components=2, cross_validate=False)
    session = first["session"]
    second = call(
        "mva_fit",
        session=session,
        name="M2",
        parent="M1",
        exclude=["r3", "r7"],
        add_steps=[{"kind": "transform", "column": "d", "function": "log10"}],
        reason="d is skewed; r3 and r7 checked",
        cross_validate=False,
    )
    model = second["model"]
    assert model["changes"] == ["Transform d with log10", "leaves out row r3", "leaves out row r7"]
    assert model["n_rows_fitted"] == 28
    assert model["parent"] == "M1"
    # The later transform is M2's own: M1, refitted from the final session, is the model it was.
    again = call("mva_predict", session=second["session"], model="M1", rows=["r0"])
    before = call("mva_predict", session=session, model="M1", rows=["r0"])
    assert again["rows"] == before["rows"]


def test_the_fit_reports_what_the_workbench_computes(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    result = call("mva_fit", session=session, name="M1", kind="PCA", n_components=2, cross_validate=False)
    fit = Variant(kind="PCA", n_components=2).fit(table)
    assert result["model"]["r2x_per_component"] == pytest.approx(fit.r2x())
    limits = fit.limits(0.95)
    rows = fit.rows()
    outside = rows[(rows["T2"] > limits["T2"]) | (rows["SPE"] > limits["SPE"])]
    assert result["model"]["n_rows_outside"] == len(outside)
    assert {r["row"] for r in result["model"]["rows_outside"]} == set(outside.index)
    assert all(len(r["top_variables"]) == 3 for r in result["model"]["rows_outside"])


def test_a_row_left_out_is_projected_and_a_pls_predicts_new_rows(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    result = call(
        "mva_fit",
        session=session,
        name="P1",
        kind="PLS",
        x_columns=["a", "b", "c", "d"],
        y_columns=["y"],
        n_components=2,
        exclude=["r0"],
        cross_validate=False,
    )
    session = result["session"]
    fit = Variant(kind="PLS", n_components=2, x_columns=("a", "b", "c", "d"), y_columns=("y",), excluded=("r0",)).fit(
        table
    )
    left_out = call("mva_predict", session=session, model="P1", rows=["r0"])["rows"][0]
    assert left_out["T2"] == pytest.approx(fit.rows().loc["r0", "T2"])
    new = table.iloc[:3].copy()
    new.index = ["n1", "n2", "n3"]
    predicted = call("mva_predict", session=session, model="P1", csv=new.to_csv())["rows"]
    expected = fit.scalers[1].inverse_transform(fit.model.predict(fit.scalers[0].transform(new[["a", "b", "c", "d"]])))
    assert [r["row"] for r in predicted] == ["n1", "n2", "n3"]
    assert [r["predicted y"] for r in predicted] == pytest.approx(expected.iloc[:, 0].tolist())


def test_explain_names_the_variable_that_sets_a_row_apart(table: pd.DataFrame) -> None:
    table.loc["r5", "b"] = 12.0
    session = call(
        "mva_fit", session=opened(table)["session"], name="M1", kind="PCA", n_components=2, cross_validate=False
    )["session"]
    result = call("mva_explain", session=session, model="M1", rows=["r5"], top=1)
    assert result["top_variables"] == ["b"]
    assert result["means"]["b"]["group mean"] == pytest.approx(12.0)


def test_auto_takes_the_number_cross_validation_recommends(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    result = call("mva_fit", session=session, name="M1", kind="PCA", n_components="auto")
    model = result["model"]
    assert model["n_components"] == model["cross_validation"]["recommended_n_components"]
    assert result["session"]["models"][0]["variant"]["n_components"] == model["n_components"]


def test_a_script_rebuilds_its_model(table: pd.DataFrame, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = opened(table)["session"]
    session = call("mva_fit", session=session, name="M1", kind="PCA", n_components=2, cross_validate=False)["session"]
    result = call(
        "mva_fit",
        session=session,
        name="M2",
        parent="M1",
        exclude=["r1"],
        add_steps=[{"kind": "transform", "column": "d", "function": "log10"}],
        reason="d is skewed",
        cross_validate=False,
    )
    script = call("mva_script", session=result["session"], models=["M2"])["scripts"][0]["script"]
    assert script.startswith("# Model M2, grown from M1.\n# d is skewed\n")
    (tmp_path / "plant.csv").write_text(result["session"]["source"]["csv"])
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(compile(script, "M2.py", "exec"), namespace)  # noqa: S102 - the tool's own script, run to check it
    assert namespace["model"].r2_cumulative_.tolist() == pytest.approx(result["model"]["r2x_cumulative"])


@pytest.mark.dataset
def test_a_bundled_dataset_opens_and_fits() -> None:
    session = call("mva_open", source={"dataset": "LDPE"})["session"]
    result = call("mva_fit", session=session, name="M1", kind="PCA", n_components=3, cross_validate=False)
    assert result["model"]["n_rows_fitted"] == 54
    assert result["model"]["x_columns"][:2] == ["Tin", "Tmax1"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": "M1"}, "already has a model"),
        ({"parent": "M9"}, "has no model 'M9'"),
        ({"kind": "PLS", "parent": None, "name": "P"}, "needs at least one Y column"),
        ({"add_steps": [{"kind": "shell", "command": "ls"}]}, "Unknown recipe step"),
        ({"exclude": ["nope"]}, "not in the table"),
    ],
)
def test_requests_the_session_cannot_hold_are_refused(table: pd.DataFrame, change: dict, message: str) -> None:
    session = opened(table)["session"]
    session = call("mva_fit", session=session, name="M1", kind="PCA", n_components=2, cross_validate=False)["session"]
    request = {"session": session, "name": "M2", "parent": "M1", "cross_validate": False} | change
    result = execute_tool_call("mva_fit", request)
    assert message in result["error"]


def test_a_tampered_session_is_refused(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    session["source"]["csv"] = session["source"]["csv"].replace("r0,", "r0,9", 1)
    result = execute_tool_call("mva_fit", {"session": session, "name": "M1", "kind": "PCA", "n_components": 2})
    assert "not the file its recipe was written for" in result["error"]


def test_a_table_beyond_the_caps_is_refused(table: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_matrix_rows", 10)
    result = execute_tool_call("mva_open", {"source": {"csv": table.to_csv()}})
    assert "caps" in result["error"]


@pytest.mark.dataset
def test_a_session_passes_safe_mode_from_call_to_call() -> None:
    """Safe mode runs each call in a fresh process: the session alone carries the analysis from one to the next."""
    from process_improve.tool_safety import safe_execute_tool_call

    session = safe_execute_tool_call("mva_open", {"source": {"dataset": "LDPE"}}, timeout=60)["session"]
    request = {"session": session, "name": "M1", "kind": "PCA", "n_components": 2, "cross_validate": False}
    first = safe_execute_tool_call("mva_fit", request, timeout=60)
    request = {"session": first["session"], "name": "M2", "parent": "M1", "exclude": [1], "cross_validate": False}
    second = safe_execute_tool_call("mva_fit", request, timeout=60)
    assert second["model"]["n_rows_fitted"] == 53
    assert [m["name"] for m in second["session"]["models"]] == ["M1", "M2"]


def test_a_session_with_models_out_of_order_or_twice_is_refused(table: pd.DataFrame) -> None:
    session = opened(table)["session"]
    session = call("mva_fit", session=session, name="M1", kind="PCA", n_components=2, cross_validate=False)["session"]
    model = session["models"][0]
    for models, message in (
        ([model, model], "Two models are called 'M1'"),
        ([model | {"name": "M2", "parent": "M3"}], "comes later or not at all"),
    ):
        result = execute_tool_call("mva_script", {"session": session | {"models": models}})
        assert message in result["error"]


def test_other_requests_that_cannot_be_met_say_why(table: pd.DataFrame) -> None:
    session = opened(table.assign(site="north"))["session"]
    assert "needs a kind" in execute_tool_call("mva_fit", {"session": session, "name": "M1"})["error"]
    session = call(
        "mva_fit",
        session=session,
        name="M1",
        kind="PCA",
        x_columns=["a", "b", "c"],
        n_components=2,
        cross_validate=False,
    )["session"]
    both = execute_tool_call("mva_predict", {"session": session, "model": "M1", "rows": ["r0"], "csv": "id,a\n"})
    assert "not both" in both["error"]
    assert (
        "not in the table"
        in execute_tool_call("mva_explain", {"session": session, "model": "M1", "rows": ["x"]})["error"]
    )
    assert "has no model 'M2'" in execute_tool_call("mva_script", {"session": session, "models": ["M2"]})["error"]


def test_open_describes_a_text_column_without_numbers(table: pd.DataFrame) -> None:
    columns = opened(table.assign(site="north"))["table"]["columns"]
    site = next(c for c in columns if c["name"] == "site")
    assert site == {"name": "site", "numeric": False, "n_missing": 0}
