"""``web/mva/bootstrap.py``, asked what the multivariate page asks it, answered natively."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

GLUE = Path(__file__).parents[3] / "web" / "mva" / "bootstrap.py"
X = ["Tin", "Tmax1", "Tout1", "Tmax2", "Tout2", "Tcin1", "Tcin2", "z1", "z2", "Fi1", "Fi2", "Fs1", "Fs2", "Press"]


@pytest.fixture
def glue() -> ModuleType:
    """Load the glue afresh, as the worker does, with nothing open."""
    spec = importlib.util.spec_from_file_location("mva_bootstrap", GLUE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ask(glue: ModuleType, fn: str, payload: dict) -> dict:
    """Call an entry point with a request, as the page does, and return its answer read back."""
    reply = json.loads(getattr(glue, fn)(json.dumps(payload)))
    assert reply["ok"], reply.get("error")
    return reply


def opened(glue: ModuleType) -> dict:
    return ask(glue, "api_open", {"example": "ldpe"})


def variant(**overrides: object) -> dict:
    return {"kind": "PCA", "n_components": 3, "x_columns": X, "excluded": ["52", "53", "54"]} | overrides


@pytest.mark.dataset
def test_opening_an_example_returns_its_table_recipe_and_python(glue) -> None:
    table = opened(glue)
    assert table["rowIds"][:3] == ["1", "2", "3"]
    assert [c["id"] for c in table["columns"]][:3] == ["Tin", "Tmax1", "Tout1"]
    assert table["recipe"]["source"]["name"] == "LDPE.csv"
    assert "pd.read_csv(SOURCE, index_col=0)" in table["code"]


@pytest.mark.dataset
def test_a_recipe_step_from_the_page_is_applied_and_written(glue) -> None:
    recipe = opened(glue)["recipe"]
    recipe["steps"] = [{"kind": "add_column", "name": "dT", "expression": "Tout1 - Tin"}]
    table = ask(glue, "api_apply", {"recipe": recipe})
    assert table["columns"][-1]["id"] == "dT"
    assert table["steps"] == ["Add the column dT = Tout1 - Tin"]
    assert 'df["dT"] = (df["Tout1"] - df["Tin"])' in table["code"]


@pytest.mark.dataset
def test_a_fit_reports_every_row_the_excluded_ones_projected(glue) -> None:
    recipe = opened(glue)["recipe"]
    result = ask(glue, "api_fit", {"recipe": recipe, "variant": variant()})
    assert len(result["rowIds"]) == 54
    assert result["excluded"] == ["52", "53", "54"]
    assert len(result["scores"]) == 3
    assert len(result["ellipse"]["x"]) == 100
    assert result["SPE"][53] > result["limits"]["0.95"]["SPE"]
    assert "projected = model.project(" in result["code"]


@pytest.mark.dataset
def test_a_lasso_is_explained_with_its_evidence(glue) -> None:
    recipe = opened(glue)["recipe"]
    group = ["48", "49", "50", "51", "52", "53", "54"]
    result = ask(glue, "api_explain", {"recipe": recipe, "variant": variant(), "group": group, "components": [1, 2]})
    assert result["top"][0] == "z2"
    assert set(result["contributions"]) == {"PC1", "PC2"}
    assert result["evidence"]["group"] == group
    assert result["evidence"]["series"][0]["name"] == "z2"
    assert len(result["evidence"]["series"][0]["values"]) == 54


@pytest.mark.dataset
def test_a_project_saved_by_the_page_opens_again_and_reproduces(glue) -> None:
    recipe = opened(glue)["recipe"]
    saved = ask(glue, "api_save", {"recipe": recipe, "variants": [variant()], "selections": {"Upset": ["52", "53"]}})
    again = ask(glue, "api_load", {"content": saved["content"]})
    assert again["reproduces"]
    assert again["variants"][0]["excluded"] == ["52", "53", "54"]
    assert again["selections"] == {"Upset": ["52", "53"]}
    assert len(again["rowIds"]) == 54


def test_an_uploaded_file_opens_with_its_read_options(glue) -> None:
    content = base64.b64encode(b"id;a;b\nr1;1,5;2\nr2;3,5;4\n").decode()
    table = ask(
        glue,
        "api_open",
        {"name": "mine.csv", "content": content, "options": {"sep": ";", "decimal": ",", "index_col": 0}},
    )
    assert table["rowIds"] == ["r1", "r2"]
    assert table["columns"][0]["values"] == [1.5, 3.5]


@pytest.mark.parametrize(
    ("fn", "payload", "message"),
    [
        ("api_fit", {"recipe": {"source": {"name": "x.csv"}}, "variant": {}}, "Open a data file first"),
        ("api_open", {"example": "secret"}, "Unknown example"),
        ("api_open", {"name": "a.csv", "content": "not base64!"}, ""),
        ("api_open", {"name": "a.csv", "content": "", "options": {"engine": "python"}}, "Unsupported read options"),
    ],
)
def test_a_request_the_glue_cannot_answer_comes_back_as_an_error(glue, fn, payload, message) -> None:
    reply = json.loads(getattr(glue, fn)(json.dumps(payload)))
    assert reply["ok"] is False
    assert message in reply["error"]


@pytest.mark.dataset
def test_rows_named_by_the_page_must_be_in_the_table(glue) -> None:
    recipe = opened(glue)["recipe"]
    reply = json.loads(glue.api_explain(json.dumps({"recipe": recipe, "variant": variant(), "group": ["999"]})))
    assert reply["ok"] is False
    assert "not in the table" in reply["error"]


def test_a_request_that_is_not_an_object_is_refused(glue) -> None:
    assert json.loads(glue.api_version("[1, 2]"))["ok"] is False
    assert json.loads(glue.api_version("{}"))["version"]


def test_numbers_reach_the_page_as_json_holds_them(glue) -> None:
    assert glue._clean({"a": np.float64(np.inf), "b": [np.int64(3), (1.5,)]}) == {"a": None, "b": [3, [1.5]]}


def test_text_columns_and_missing_cells_reach_the_page(glue) -> None:
    content = base64.b64encode(b"id,grade,x\nr1,A,1\nr2,,2\n").decode()
    table = ask(glue, "api_open", {"name": "mine.csv", "content": content, "options": {"index_col": 0}})
    grade = next(c for c in table["columns"] if c["id"] == "grade")
    assert grade["values"] == ["A", None]
    assert grade["numeric"] is False


def test_rows_whose_ids_read_alike_as_text_are_refused(glue) -> None:
    with pytest.raises(ValueError, match="read the same as text"):
        glue._labels(pd.Index([1, "1"]))


@pytest.mark.parametrize(
    ("fn", "payload", "message"),
    [
        ("api_explain", {"variant": variant(), "group": "52"}, "must be a list of row ids"),
        ("api_save", {"variants": [], "selections": ["Upset"]}, "must be a mapping"),
    ],
)
@pytest.mark.dataset
def test_malformed_requests_after_opening_are_refused(glue, fn, payload, message) -> None:
    payload = {"recipe": opened(glue)["recipe"], **payload}
    reply = json.loads(getattr(glue, fn)(json.dumps(payload)))
    assert reply["ok"] is False
    assert message in reply["error"]


def test_a_file_too_large_or_with_options_that_are_not_a_mapping_is_refused(glue, monkeypatch) -> None:
    monkeypatch.setattr(glue, "MAX_SOURCE_BYTES", 4)
    content = base64.b64encode(b"a,b\n1,2\n").decode()
    assert "larger than" in json.loads(glue.api_open(json.dumps({"name": "a.csv", "content": content})))["error"]
    reply = json.loads(glue.api_open(json.dumps({"name": "a.csv", "content": content, "options": [1]})))
    assert "must be a mapping" in reply["error"]


@pytest.mark.dataset
def test_a_one_component_model_has_no_ellipse(glue) -> None:
    result = ask(glue, "api_fit", {"recipe": opened(glue)["recipe"], "variant": variant(n_components=1)})
    assert result["ellipse"] == {}
    assert len(result["scores"]) == 1


def test_the_version_is_named_even_in_a_checkout_that_was_never_installed(glue, monkeypatch) -> None:
    def missing(_name: str) -> str:
        raise glue.PackageNotFoundError

    monkeypatch.setattr(glue, "version", missing)
    assert json.loads(glue.api_version("{}"))["version"] == "(unreleased)"
