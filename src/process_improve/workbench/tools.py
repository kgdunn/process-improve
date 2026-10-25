# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tools for a multivariate analysis that proceeds step by step: fit, look, change, fit again.

A multivariate analysis is rarely one fit. In the way Eriksson et al. (Multi- and Megavariate
Data Analysis, Umetrics) work their examples, the analyst fits a model, finds rows outside
the T² or SPE limits, asks which variables put them there, transforms a skewed column,
leaves rows out and fits again. Each tool here takes the analysis so far, a *session*, and
returns it with the new step added. The session is the whole record and the tools keep no
state, which safe mode needs anyway: it runs every call in its own process.

A session is JSON: the source (a dataset bundled with the package, or CSV text), the
recipe that reads it, and every model, each with the model it grew from, the reason, its
own recipe steps and its workbench :class:`~process_improve.workbench.Variant`. A model
keeps all of its steps, so a log transform added to a later model leaves an earlier one
as it was.

=================  ===========================================================================
Tool               Does
=================  ===========================================================================
``mva_open``       start a session; describe each column (missing cells, range, skewness)
``mva_fit``        add a model, new or grown from another; report its fit, outliers and columns
``mva_explain``    say which variables set some rows apart from the others, per component
``mva_predict``    project rows of the table, or new rows, through a model
``mva_script``     write the Python that rebuilds each model, for a record outside MCP
=================  ===========================================================================
"""

from __future__ import annotations

import dataclasses
import hashlib
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.config import settings
from process_improve.tool_spec import clean, tool_spec

from .recipe import Recipe, SourceFile, _comment, step_from_dict
from .variant import Variant

if TYPE_CHECKING:
    from .variant import CrossValidation, Fit

#: The bundled datasets a session can open by name, and the file of each.
DATASETS: dict[str, str] = {
    "LDPE": "LDPE/LDPE.csv",
    "cheddar-cheese": "cheddar-cheese.csv",
    "solvents": "solvents.csv",
}

#: Rows outside a limit that a fit reports in full; the count is always reported.
MAX_REPORTED_ROWS = 20

#: What a tool turns into ``{"error": ...}``: the data or the request is wrong, not the tool.
EXPECTED = (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError)

RowIdValue = int | float | str


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class DatasetSource(BaseModel):
    """A dataset bundled with process-improve, by name."""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["LDPE", "cheddar-cheese", "solvents"] = Field(
        ..., description="LDPE (54 runs of a polyethylene reactor), cheddar-cheese (30 cheeses) or solvents (103)."
    )


class CsvText(BaseModel):
    """A table given as CSV text."""

    model_config = ConfigDict(extra="forbid")

    csv: str = Field(..., min_length=1, description="The table as CSV text: column names on the first line.")


class CsvSource(CsvText):
    """A table given as CSV text, with what the session needs to read it."""

    name: str = Field("data.csv", min_length=1, max_length=200, description="A file name, used in the script.")
    index_column: int | None = Field(
        0, ge=0, description="The position of the column that names the rows; null numbers them from 0."
    )


class Model(BaseModel):
    """One model of a session: what it grew from and why, its recipe steps, and its workbench Variant."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, description="The model's name, unique in the session.")
    parent: str | None = Field(None, description="The model it grew from; null for a first model.")
    reason: str = Field("", max_length=2000, description="Why it was made: what was seen, and what was changed.")
    steps: list[dict[str, Any]] = Field(
        default_factory=list, max_length=500, description="Every recipe step of its table, after the session's."
    )
    variant: dict[str, Any] = Field(..., description="Its Variant: kind, n_components, columns, excluded rows.")


class Session(BaseModel):
    """A multivariate analysis so far: pass it back unchanged to the next tool call."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["process-improve-mva-session"] = "process-improve-mva-session"
    version: Literal[1] = 1
    source: DatasetSource | CsvText
    recipe: dict[str, Any] = Field(..., description="How the source is read: its file, options and digest.")
    models: list[Model] = Field(default_factory=list, max_length=100)


@dataclasses.dataclass(frozen=True)
class _Loaded:
    """A session checked and ready: the source's bytes, its recipe, and its models by name."""

    session: Session
    content: bytes
    recipe: Recipe
    models: dict[str, Model]

    def recipe_of(self, model: Model) -> Recipe:
        """Return the recipe of a model's table: the session's, then the model's own steps."""
        return self.recipe.then(*(step_from_dict(step) for step in model.steps))

    def model(self, name: str) -> Model:
        """Return a model by name."""
        if name not in self.models:
            raise ValueError(f"The session has no model {name!r}; it has {', '.join(self.models) or 'none'}.")
        return self.models[name]

    def fit(self, name: str) -> Fit:
        """Fit a model to its table."""
        model = self.model(name)
        return Variant.from_dict(model.variant).fit(self.recipe_of(model).run(self.content))


def _content(source: DatasetSource | CsvText) -> bytes:
    if isinstance(source, DatasetSource):
        return resources.files("process_improve.datasets.multivariate").joinpath(DATASETS[source.dataset]).read_bytes()
    return source.csv.encode()


def _check_size(table: pd.DataFrame) -> None:
    """Refuse a table larger than the multivariate tools take (SEC-19)."""
    rows, columns = table.shape
    if rows > settings.max_matrix_rows or columns > settings.max_matrix_cols:
        raise ValueError(
            f"The table is {rows} by {columns}; the caps are settings.max_matrix_rows={settings.max_matrix_rows} "
            f"and settings.max_matrix_cols={settings.max_matrix_cols}."
        )


def _load(session: Session) -> _Loaded:
    """Check a session: its source is the recipe's, its table within the caps, its models well formed."""
    content = _content(session.source)
    recipe = Recipe.from_dict(session.recipe)
    if recipe.source.sha256 != hashlib.sha256(content).hexdigest():
        raise ValueError("The session's source is not the file its recipe was written for.")
    _check_size(recipe.run(content))
    models: dict[str, Model] = {}
    for model in session.models:
        if model.name in models:
            raise ValueError(f"Two models are called {model.name!r}.")
        if model.parent is not None and model.parent not in models:
            raise ValueError(f"Model {model.name!r} grows from {model.parent!r}, which comes later or not at all.")
        Variant.from_dict(model.variant)
        for step in model.steps:
            step_from_dict(step)
        models[model.name] = model
    return _Loaded(session, content, recipe, models)


# ---------------------------------------------------------------------------
# mva_open
# ---------------------------------------------------------------------------


class MvaOpenInput(BaseModel):
    """Input contract for ``mva_open``."""

    model_config = ConfigDict(extra="forbid")

    source: DatasetSource | CsvSource = Field(..., description="A bundled dataset by name, or a table as CSV text.")


def _describe(table: pd.DataFrame) -> dict[str, Any]:
    """Describe each column as the first look at the data does: missing cells, range, mean, spread, skewness."""
    columns = []
    for name in table.columns:
        values = table[name]
        entry: dict[str, Any] = {"name": str(name), "numeric": pd.api.types.is_numeric_dtype(values)}
        entry["n_missing"] = int(values.isna().sum())
        if entry["numeric"]:
            entry |= {
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "std": values.std(),
                "skewness": values.skew(),
            }
        columns.append(entry)
    rows = [r.item() if hasattr(r, "item") else r for r in table.index[:10]]
    return {"n_rows": len(table), "n_columns": table.shape[1], "first_row_ids": rows, "columns": columns}


@tool_spec(
    name="mva_open",
    description=(
        "Start a multivariate analysis session on a table: a dataset bundled with process-improve, or CSV text. "
        "Returns the session, which every other mva_ tool takes and returns, and a description of each column "
        "(missing cells, min, max, mean, standard deviation, skewness) to decide what to transform before the "
        "first model. Pass the session back unchanged: it is the whole record of the analysis."
    ),
    input_model=MvaOpenInput,
    examples="""
    # "Explore the LDPE reactor data"
        -> ``mva_open(source={"dataset": "LDPE"})``
    # "Here is my data as CSV"
        -> ``mva_open(source={"csv": "id,a,b\\n1,2.0,3.1\\n...", "name": "plant.csv"})``
    """,
    category="multivariate",
)
def mva_open(spec: MvaOpenInput) -> dict[str, Any]:
    """Start a session on a table, and describe its columns."""
    try:
        content = _content(spec.source)
        if isinstance(spec.source, CsvSource):
            index = spec.source.index_column
            name, options = spec.source.name, ({} if index is None else {"index_col": index})
            source: DatasetSource | CsvText = CsvText(csv=spec.source.csv)
        else:
            name, options, source = DATASETS[spec.source.dataset], {"index_col": 0}, spec.source
        recipe = Recipe(SourceFile.of(name.rsplit("/", 1)[-1], content, "csv", **options))
        table = recipe.run(content)
        _check_size(table)
        session = Session(source=source, recipe=recipe.to_dict())
        return clean({"session": session.model_dump(mode="json"), "table": _describe(table)})
    except EXPECTED as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# mva_fit
# ---------------------------------------------------------------------------


class MvaFitInput(BaseModel):
    """Input contract for ``mva_fit``."""

    model_config = ConfigDict(extra="forbid")

    session: Session = Field(..., description="The session, as the last mva_ tool returned it.")
    name: str = Field(..., min_length=1, max_length=100, description="A name for the new model, such as 'M2'.")
    parent: str | None = Field(None, description="The model to grow this one from; it keeps all of its settings.")
    reason: str = Field("", max_length=2000, description="Why: what the last model showed, and what this changes.")
    kind: Literal["PCA", "PLS"] | None = Field(None, description="PCA or PLS; needed for a first model.")
    n_components: int | Literal["auto"] | None = Field(
        None, description="A number of components, or 'auto' for the number cross-validation recommends."
    )
    x_columns: list[str] | None = Field(None, description="The X columns; an empty list takes every numeric one.")
    y_columns: list[str] | None = Field(None, description="The Y columns, for PLS.")
    add_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Recipe steps added to this model's table, for example "
            '{"kind": "transform", "column": "Mw", "function": "log10"} or '
            '{"kind": "drop_rows", "rows": [17], "reason": "a broken sensor"}.'
        ),
    )
    exclude: list[RowIdValue] = Field(
        default_factory=list, max_length=10_000, description="Rows to leave out of the fit; they are still projected."
    )
    include: list[RowIdValue] = Field(
        default_factory=list, max_length=10_000, description="Rows the parent left out, to fit again."
    )
    cross_validate: bool = Field(True, description="Report Q² per number of components by cross-validation.")
    conf_level: float = Field(0.95, gt=0.5, lt=1.0, description="The confidence level of the T² and SPE limits.")


def _grown(loaded: _Loaded, spec: MvaFitInput) -> Model:
    """Return the new model: the parent's settings with this request's changes, or a first model."""
    if spec.name in loaded.models:
        raise ValueError(f"The session already has a model {spec.name!r}.")
    if spec.parent is None and spec.kind is None:
        raise ValueError("A first model needs a kind, PCA or PLS.")
    parent = loaded.model(spec.parent) if spec.parent is not None else None
    variant: dict[str, Any] = dict(parent.variant) if parent else {"kind": spec.kind, "n_components": 2}
    changes = {"kind": spec.kind, "x_columns": spec.x_columns, "y_columns": spec.y_columns}
    variant |= {key: value for key, value in changes.items() if value is not None}
    if isinstance(spec.n_components, int):
        variant["n_components"] = spec.n_components
    excluded = [r for r in variant.get("excluded", []) if r not in spec.include]
    variant["excluded"] = excluded + [r for r in spec.exclude if r not in excluded]
    variant["name"] = spec.name
    for step in spec.add_steps:
        step_from_dict(step)
    steps = [*(parent.steps if parent else []), *spec.add_steps]
    return Model(name=spec.name, parent=spec.parent, reason=spec.reason, steps=steps, variant=variant)


def _changes(parent: Model | None, model: Model) -> list[str]:
    """Say in words how a model differs from its parent."""
    if parent is None:
        return ["a first model"]
    out = [step_from_dict(step).describe() for step in model.steps[len(parent.steps) :]]
    was, now = parent.variant, model.variant
    out += [f"leaves out row {r}" for r in now["excluded"] if r not in was.get("excluded", [])]
    out += [f"fits row {r} again" for r in was.get("excluded", []) if r not in now["excluded"]]
    for key in ("kind", "n_components", "x_columns", "y_columns"):
        if now.get(key) != was.get(key):
            out.append(f"{key} {now.get(key)} (was {was.get(key)})")
    return out


def _outside(fit: Fit, conf_level: float) -> dict[str, Any]:
    """Return the limits and the rows beyond either, the farthest first, each with the variables that set it apart."""
    limits = fit.limits(conf_level)
    rows = fit.rows()
    ratio = pd.concat([rows["T2"] / limits["T2"], rows["SPE"] / limits["SPE"]], axis=1).max(axis=1)
    beyond = ratio[ratio > 1].sort_values(ascending=False)
    reported = [
        {
            "row": row,
            "T2": rows.loc[row, "T2"],
            "SPE": rows.loc[row, "SPE"],
            "beyond": [name for name in ("T2", "SPE") if rows.loc[row, name] > limits[name]],
            "excluded": bool(rows.loc[row, "excluded"]),
            "top_variables": fit.explain_group([row], top=3).top,
        }
        for row in beyond.index[:MAX_REPORTED_ROWS]
    ]
    return {"conf_level": conf_level, "limits": limits, "n_rows_outside": len(beyond), "rows_outside": reported}


def _report(loaded: _Loaded, model: Model, fit: Fit, cv: CrossValidation | None, conf_level: float) -> dict[str, Any]:
    """Return what the book reads off a fit: R² and Q² by component, and the rows outside the limits."""
    r2x = fit.r2x()
    report: dict[str, Any] = {
        "name": model.name,
        "parent": model.parent,
        "reason": model.reason,
        "changes": _changes(loaded.models.get(model.parent or ""), model),
        "kind": fit.variant.kind,
        "n_components": fit.variant.n_components,
        "n_rows_fitted": int(fit.model.n_samples_),
        "excluded": list(fit.variant.excluded),
        "steps": loaded.recipe_of(model).describe(),
        "x_columns": list(fit.x_scaled.columns),
        "r2x_per_component": r2x,
        "r2x_cumulative": np.cumsum(r2x),
    }
    # The work set statistics: did the model's steps make a skewed column symmetric?
    x, y = fit.variant.blocks(fit.table)
    work_set = x if y is None else pd.concat([x, y], axis=1)
    report["columns"] = _describe(work_set.loc[fit.variant.included(fit.table)])["columns"]
    if fit.variant.kind == "PLS":
        report["y_columns"] = list(fit.variant.y_columns)
        report["r2y_per_component"] = fit.model.r2_per_component_
        report["r2y_cumulative"] = fit.model.r2_cumulative_
    if cv is not None:
        report["cross_validation"] = {
            "q2_cumulative": cv.q2,
            "q2_standard_error": cv.q2_se,
            "recommended_n_components": cv.n_components,
            "rule": cv.rule,
            "held_out": cv.held_out,
        }
    return report | _outside(fit, conf_level)


@tool_spec(
    name="mva_fit",
    description=(
        "Add a PCA or PLS model to a multivariate session and report its fit: R² (and Q² by cross-validation) "
        "per component, the T² and SPE limits, and the rows beyond them, each with the variables that set it apart; "
        "and each of its columns as fitted (after its steps, on its rows), to check that a transform worked. "
        "Grow a model from an earlier one with `parent` to change one thing at a time: leave rows out (they are "
        "still projected), fit rows again, add recipe steps such as a log10 transform, or change the components. "
        "Say why in `reason`. Returns the session with the model added; pass it to the next mva_ call."
    ),
    input_model=MvaFitInput,
    examples="""
    # "Fit a PCA, and let cross-validation choose the components"
        -> ``mva_fit(session=..., name="M1", kind="PCA", n_components="auto")``
    # "Rows 52 and 53 are outliers caused by a fouled sensor: refit without them, with Mw on a log scale"
        -> ``mva_fit(session=..., name="M2", parent="M1", exclude=[52, 53],
                     add_steps=[{"kind": "transform", "column": "Mw", "function": "log10"}],
                     reason="52 and 53 beyond T2, high Tout2; Mw skewed")``
    """,
    category="multivariate",
)
def mva_fit(spec: MvaFitInput) -> dict[str, Any]:
    """Add a model to the session, fit it, and report what to look at next."""
    try:
        loaded = _load(spec.session)
        model = _grown(loaded, spec)
        table = loaded.recipe_of(model).run(loaded.content)
        variant = Variant.from_dict(model.variant)
        cv = variant.cross_validate(table) if spec.cross_validate or spec.n_components == "auto" else None
        if spec.n_components == "auto" and cv is not None:
            variant = dataclasses.replace(variant, n_components=cv.n_components)
            model.variant = variant.to_dict()
        fit = variant.fit(table)
        session = spec.session.model_copy(update={"models": [*spec.session.models, model]})
        return clean(
            {"session": session.model_dump(mode="json"), "model": _report(loaded, model, fit, cv, spec.conf_level)}
        )
    except EXPECTED as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# mva_explain
# ---------------------------------------------------------------------------


class MvaExplainInput(BaseModel):
    """Input contract for ``mva_explain``."""

    model_config = ConfigDict(extra="forbid")

    session: Session = Field(..., description="The session, as the last mva_ tool returned it.")
    model: str = Field(..., description="The model to explain the rows in.")
    rows: list[RowIdValue] = Field(..., min_length=1, max_length=10_000, description="The rows to explain.")
    reference: list[RowIdValue] | None = Field(
        None, max_length=10_000, description="The rows to compare against; default every other fitted row."
    )
    components: list[int] | None = Field(None, description="1-based components to explain; default all.")
    top: int = Field(5, ge=1, le=50, description="How many variables to report.")


@tool_spec(
    name="mva_explain",
    description=(
        "Explain why some rows sit apart in a model of a multivariate session: each variable's contribution to "
        "the difference between their average score and the reference rows', per component, with the two groups' "
        "means in the table's own units. Use it on the rows mva_fit reports beyond a limit, or on a cluster."
    ),
    input_model=MvaExplainInput,
    examples="""
    # "Why are rows 52 and 53 outside the T2 limit of M1?"
        -> ``mva_explain(session=..., model="M1", rows=[52, 53])``
    """,
    category="multivariate",
)
def mva_explain(spec: MvaExplainInput) -> dict[str, Any]:
    """Return the variables that set the rows apart, and by how much in their own units."""
    try:
        fit = _load(spec.session).fit(spec.model)
        found = fit.explain_group(spec.rows, spec.reference, spec.components, spec.top)
        return clean(
            {
                "rows": found.group,
                "n_reference_rows": len(found.reference),
                "score_difference": found.score_difference().to_dict(),
                "top_variables": found.top,
                "contributions": found.contributions.loc[found.top].to_dict(orient="index"),
                "means": found.summary.loc[found.top].to_dict(orient="index"),
            }
        )
    except EXPECTED as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# mva_predict
# ---------------------------------------------------------------------------


class MvaPredictInput(BaseModel):
    """Input contract for ``mva_predict``."""

    model_config = ConfigDict(extra="forbid")

    session: Session = Field(..., description="The session, as the last mva_ tool returned it.")
    model: str = Field(..., description="The model to project through.")
    rows: list[RowIdValue] | None = Field(None, max_length=10_000, description="Rows of the table; default all.")
    csv: str | None = Field(
        None, description="New rows as CSV text, in the source's layout; the model's recipe steps are applied."
    )


def _new_table(loaded: _Loaded, model: Model, text: str) -> pd.DataFrame:
    """Read new rows as the source is read, then the model's steps."""
    source = dataclasses.replace(loaded.recipe.source, name="new rows", sha256=None)
    table = dataclasses.replace(loaded.recipe_of(model), source=source).run(text.encode())
    _check_size(table)
    return table


@tool_spec(
    name="mva_predict",
    description=(
        "Project rows through a model of a multivariate session: rows of its table (the ones it left out, say) "
        "or new rows as CSV text in the source's layout. Returns each row's scores, T², SPE and, for PLS, "
        "predicted Y in its own units, with the model's limits."
    ),
    input_model=MvaPredictInput,
    examples="""
    # "Where do the rows M2 left out fall?"
        -> ``mva_predict(session=..., model="M2", rows=[52, 53])``
    """,
    category="multivariate",
)
def mva_predict(spec: MvaPredictInput) -> dict[str, Any]:
    """Return each row's scores, T², SPE and prediction under the model."""
    try:
        loaded = _load(spec.session)
        fit = loaded.fit(spec.model)
        if spec.csv is not None and spec.rows is not None:
            raise ValueError("Give rows of the table or new rows as CSV, not both.")
        if spec.csv is not None:
            x, _ = fit.variant.blocks(_new_table(loaded, loaded.model(spec.model), spec.csv))
            scaled = fit.scalers[0].transform(x)
        else:
            scaled = fit.x_scaled if spec.rows is None else fit.x_scaled.loc[list(spec.rows)]
        projected = fit.model.project(scaled)
        scores = pd.DataFrame(np.asarray(projected.scores), index=scaled.index, columns=fit.model.scores_.columns)
        out = scores.assign(T2=np.asarray(projected.hotellings_t2), SPE=np.asarray(projected.spe))
        if fit.variant.kind == "PLS":
            y_scaled = pd.DataFrame(scores.to_numpy() @ np.asarray(fit.model.y_loadings_).T, index=scaled.index)
            y_hat = fit.scalers[1].inverse_transform(y_scaled)  # type: ignore[union-attr]  # PLS has a Y scaler
            out = out.join(y_hat.set_axis([f"predicted {c}" for c in fit.variant.y_columns], axis=1))
        rows = [{"row": row, **values} for row, values in out.to_dict(orient="index").items()]
        return clean({"limits": fit.limits(), "rows": rows})
    except EXPECTED as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# mva_script
# ---------------------------------------------------------------------------


class MvaScriptInput(BaseModel):
    """Input contract for ``mva_script``."""

    model_config = ConfigDict(extra="forbid")

    session: Session = Field(..., description="The session, as the last mva_ tool returned it.")
    models: list[str] | None = Field(None, description="The models to write; default all, in the order made.")


@tool_spec(
    name="mva_script",
    description=(
        "Write the Python that rebuilds each model of a multivariate session with pandas and process-improve, "
        "one script per model, headed by what it grew from and why. Run next to the source file, a script "
        "repeats the model exactly: the record of the analysis outside MCP."
    ),
    input_model=MvaScriptInput,
    examples="""
    # "Give me the code for the final model"
        -> ``mva_script(session=..., models=["M3"])``
    """,
    category="multivariate",
)
def mva_script(spec: MvaScriptInput) -> dict[str, Any]:
    """Return one script per model, each rebuilding its table and fitting it."""
    try:
        loaded = _load(spec.session)
        scripts = []
        for name in spec.models or list(loaded.models):
            model = loaded.model(name)
            grown = f", grown from {model.parent}" if model.parent else ""
            header = [f"# Model {_comment(model.name)}{grown}."]
            header += [f"# {_comment(model.reason)}"] if model.reason else []
            body = loaded.recipe_of(model).to_python() + "\n" + Variant.from_dict(model.variant).to_python()
            scripts.append({"model": model.name, "script": "\n".join(header) + "\n" + body})
        return clean({"scripts": scripts})
    except EXPECTED as exc:
        return {"error": str(exc)}
