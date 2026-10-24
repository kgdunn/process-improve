"""Python glue between the multivariate page and ``process_improve.workbench``.

Pyodide loads this file into the page's worker (``web/mva/worker.js``). The page calls the
``api_*`` functions at the bottom; each takes and returns a JSON string, so only plain text
crosses into JavaScript. The worker keeps the source file's bytes, so the page sends only
the recipe and the variant with each call, and a fit is cached by both.

Security
--------
A recipe, a variant and a project file can come from someone else, so everything read from
the page is checked by the workbench's own ``from_dict`` methods, an expression is parsed and
never evaluated, and a project opens without executing anything from it. Row ids cross into
JavaScript as text and are mapped back to the table's own labels here.
"""

from __future__ import annotations

import base64
import functools
import json
import math
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.workbench import Project, Recipe, SourceFile, Variant

if TYPE_CHECKING:
    from collections.abc import Callable

#: The examples the page offers, by name: the file in the package and how to read it.
EXAMPLES: dict[str, tuple[str, dict[str, Any]]] = {
    "ldpe": ("multivariate/LDPE/LDPE.csv", {"index_col": 0}),
    "foods": ("multivariate/food-consumption.csv", {"index_col": 0}),
}

#: The largest source file the page opens, in bytes.
MAX_SOURCE_BYTES = 64 * 2**20

#: What the worker keeps between calls: the source file's bytes.
SOURCE: dict[str, bytes] = {}


# ---------------------------------------------------------------------------
# Plain JSON
# ---------------------------------------------------------------------------


def _clean(value: object) -> object:
    """Return ``value`` as JSON can hold it: NaN and infinity as null, numpy scalars as numbers."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _reply(fn: Callable[[dict], dict], payload: str) -> str:
    """Run ``fn`` on the page's request and return its answer, or the error, as JSON."""
    try:
        request = _request(payload)
        return json.dumps({"ok": True, **_clean(fn(request))}, allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        return json.dumps({"ok": False, "error": str(error)})


def _request(payload: str) -> dict:
    request = json.loads(payload or "{}")
    if not isinstance(request, dict):
        raise TypeError("A request is a JSON object.")
    return request


def _column(series: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return [None if not math.isfinite(v) else v for v in series.astype(float).tolist()]
    return [None if pd.isna(v) else str(v) for v in series.tolist()]


def _labels(index: pd.Index) -> dict[str, Any]:
    """Map each row id as the page holds it (text) to the table's own label."""
    labels = {str(label): label for label in index}
    if len(labels) != len(index):
        raise ValueError("Two rows have ids that read the same as text; give the rows unique ids first.")
    return labels


def _rows(ids: object, labels: dict[str, Any], what: str) -> list:
    if not isinstance(ids, list):
        raise TypeError(f"'{what}' must be a list of row ids.")
    unknown = [i for i in ids if str(i) not in labels]
    if unknown:
        raise ValueError(f"The {what} names rows that are not in the table: {', '.join(map(str, unknown[:5]))}.")
    return [labels[str(i)] for i in ids]


# ---------------------------------------------------------------------------
# The table, the fit and the explanation, cached by the JSON that names them
# ---------------------------------------------------------------------------


def _source() -> bytes:
    if "content" not in SOURCE:
        raise ValueError("Open a data file first.")
    return SOURCE["content"]


@functools.lru_cache(maxsize=8)
def _table(recipe_json: str) -> pd.DataFrame:
    return Recipe.from_dict(json.loads(recipe_json)).run(_source())


@functools.lru_cache(maxsize=8)
def _fit(recipe_json: str, variant_json: str) -> Any:  # noqa: ANN401 - a workbench Fit
    table = _table(recipe_json)
    raw = json.loads(variant_json)
    labels = _labels(table.index)
    raw["excluded"] = _rows(raw.get("excluded", []), labels, "excluded rows")
    return Variant.from_dict(raw).fit(table)


def _key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _table_payload(recipe: Recipe) -> dict[str, Any]:
    table = _table(_key(recipe.to_dict()))
    _labels(table.index)
    return {
        "recipe": recipe.to_dict(),
        "steps": recipe.describe(),
        "code": recipe.to_python(),
        "rowIds": [str(label) for label in table.index],
        # The labels themselves, numbers or text, for the recipe steps the page writes (a dropped row).
        "rowLabels": [
            label if isinstance(label, int | float | str | np.generic) else str(label) for label in table.index
        ],
        "columns": [
            {"id": str(c), "values": _column(table[c]), "numeric": bool(pd.api.types.is_numeric_dtype(table[c]))}
            for c in table.columns
        ],
    }


def _open(name: str, content: bytes, options: dict[str, Any]) -> dict[str, Any]:
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"The file is larger than {MAX_SOURCE_BYTES // 2**20} MB.")
    fmt = "excel" if name.lower().endswith((".xlsx", ".xls")) else "csv"
    recipe = Recipe(SourceFile.of(name, content, format=fmt, **options))
    SOURCE["content"] = content
    _table.cache_clear()
    _fit.cache_clear()
    return _table_payload(recipe)


def open_file(request: dict) -> dict:
    """Open a file the reader chose: its name, its content in base64, and read options."""
    if "example" in request:
        example = str(request["example"])
        if example not in EXAMPLES:
            raise ValueError(f"Unknown example {example!r}; choose from {', '.join(EXAMPLES)}.")
        path, options = EXAMPLES[example]
        content = (files("process_improve.datasets") / path).read_bytes()
        return _open(path.rsplit("/", 1)[-1], content, options)
    options = request.get("options", {})
    if not isinstance(options, dict):
        raise TypeError("'options' must be a mapping.")
    return _open(str(request["name"]), base64.b64decode(request["content"], validate=True), options)


def apply_recipe(request: dict) -> dict:
    """Return the table a recipe makes from the open file, with its steps and its Python."""
    return _table_payload(Recipe.from_dict(request["recipe"]))


def fit(request: dict) -> dict:
    """Fit a variant to the recipe's table: every row's scores, T² and SPE, the loadings, limits and ellipse."""
    recipe = Recipe.from_dict(request["recipe"])
    result = _fit(_key(recipe.to_dict()), _key(request["variant"]))
    rows = result.rows()
    a = result.variant.n_components
    loadings = result.model.loadings_ if result.variant.kind == "PCA" else result.model.x_loadings_
    limits = {str(level): result.limits(level) for level in (0.95, 0.99)}
    ellipse = {}
    if a >= 2:
        x, y = result.model.ellipse_coordinates(score_horiz=1, score_vert=2, conf_level=0.95)
        ellipse = {"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist()}
    return {
        "rowIds": [str(label) for label in rows.index],
        "scores": rows.iloc[:, :a].to_numpy().T.tolist(),
        "T2": rows["T2"].tolist(),
        "SPE": rows["SPE"].tolist(),
        "excluded": [str(label) for label in rows.index[rows["excluded"].to_numpy(dtype=bool)]],
        "variables": [str(c) for c in loadings.index],
        "loadings": loadings.to_numpy().T.tolist(),
        "r2x": result.r2x(),
        "limits": limits,
        "ellipse": ellipse,
        "code": recipe.to_python() + "\n" + result.variant.to_python(),
    }


def explain(request: dict) -> dict:
    """Explain why a group of rows differs from the rest, or from a reference, per component."""
    recipe = Recipe.from_dict(request["recipe"])
    result = _fit(_key(recipe.to_dict()), _key(request["variant"]))
    labels = _labels(result.table.index)
    group = _rows(request.get("group"), labels, "group")
    reference = request.get("reference")
    components = request.get("components")
    explanation = result.explain_group(
        group,
        reference=None if reference is None else _rows(reference, labels, "reference"),
        components=None if components is None else [int(c) for c in components],
        top=int(request.get("top", 5)),
    )
    evidence = explanation.evidence
    return {
        "variables": [str(v) for v in explanation.contributions.index],
        "contributions": {str(c): explanation.contributions[c].tolist() for c in explanation.contributions},
        "scoreDifference": {str(k): float(v) for k, v in explanation.score_difference().items()},
        "top": [str(v) for v in explanation.top],
        "summary": {str(column): explanation.summary[column].tolist() for column in explanation.summary.columns},
        "evidence": {
            "rowIds": [str(label) for label in evidence.index],
            "series": [{"name": str(v), "values": _column(evidence[v])} for v in explanation.top],
            "group": [str(label) for label in evidence.index[evidence["in_group"].to_numpy(dtype=bool)]],
        },
        "reference": [str(r) for r in explanation.reference],
    }


def save(request: dict) -> dict:
    """Return the project file, in base64: the open file, its recipe, the variants and the saved selections."""
    recipe = Recipe.from_dict(request["recipe"])
    table = _table(_key(recipe.to_dict()))
    labels = _labels(table.index)
    variants = []
    for given in request.get("variants", []):
        raw = {**given, "excluded": _rows(given.get("excluded", []), labels, "excluded rows")}
        variants.append(Variant.from_dict(raw))
    selections = request.get("selections", {})
    if not isinstance(selections, dict):
        raise TypeError("'selections' must be a mapping of names to row ids.")
    project = Project(
        recipe=recipe,
        source=_source(),
        variants=tuple(variants),
        selections={str(k): _rows(v, labels, f"selection {k}") for k, v in selections.items()},
    )
    return {"content": base64.b64encode(project.to_bytes()).decode("ascii")}


def load(request: dict) -> dict:
    """Open a project file (base64): its table, variants and selections, and whether its fits reproduce."""
    opened = Project.open(base64.b64decode(request["content"], validate=True))
    SOURCE["content"] = opened.project.source
    _table.cache_clear()
    _fit.cache_clear()
    payload = _table_payload(opened.project.recipe)
    return {
        **payload,
        "variants": [{**v.to_dict(), "excluded": [str(r) for r in v.excluded]} for v in opened.project.variants],
        "selections": {k: [str(r) for r in v] for k, v in opened.project.selections.items()},
        "reproduces": opened.reproduces,
        "checks": [
            {"variant": c.variant, "array": c.array, "maxDifference": c.max_difference, "ok": c.ok}
            for c in opened.checks
        ],
        "library": opened.library,
    }


def _release() -> str:
    try:
        return version("process-improve")
    except PackageNotFoundError:
        return "(unreleased)"


# ---------------------------------------------------------------------------
# The page's entry points: JSON text in, JSON text out
# ---------------------------------------------------------------------------


def api_open(payload: str) -> str:
    """Open a data file, or one of the examples."""
    return _reply(open_file, payload)


def api_apply(payload: str) -> str:
    """Apply a recipe to the open file."""
    return _reply(apply_recipe, payload)


def api_fit(payload: str) -> str:
    """Fit a variant."""
    return _reply(fit, payload)


def api_explain(payload: str) -> str:
    """Explain a group of rows."""
    return _reply(explain, payload)


def api_save(payload: str) -> str:
    """Write the project file."""
    return _reply(save, payload)


def api_load(payload: str) -> str:
    """Open a project file."""
    return _reply(load, payload)


def api_version(payload: str) -> str:
    """Name the library the page runs."""
    return _reply(lambda _request: {"version": _release()}, payload)
