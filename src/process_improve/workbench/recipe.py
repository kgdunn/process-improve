# (c) Kevin Dunn, 2010-2026. MIT License.
"""A recipe: the steps that refine a source table, and the Python that repeats them.

A :class:`Recipe` is a source (a file and how to read it) and a list of steps. It is
the source of truth for the table an analysis uses: :meth:`Recipe.apply` runs the
steps, :meth:`Recipe.to_python` writes a script that does the same with pandas and
numpy alone, and :meth:`Recipe.to_dict` / :meth:`Recipe.from_dict` round-trip it
through JSON. Each step's ``apply`` and ``code`` perform the same operations in the
same order, so the script rebuilds the table bit for bit.

A recipe can arrive in someone else's project file, so :meth:`Recipe.from_dict` checks
every field, an expression is parsed and never evaluated as Python, and every name,
row id and reason reaches the script only as a quoted literal or a one-line comment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from ._expressions import Expression, python_literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The read options a source may carry, for CSV and Excel files.
READ_OPTIONS: dict[str, frozenset[str]] = {
    "csv": frozenset({"sep", "decimal", "index_col", "header", "skiprows", "na_values", "encoding"}),
    "excel": frozenset({"sheet_name", "index_col", "header", "skiprows", "na_values"}),
}

#: The transforms :class:`Transform` offers, as expressions of the column ``x`` and the constant ``c``.
TRANSFORMS: dict[str, tuple[str, float | None]] = {
    "log": ("log({x} + {c})", 0.0),
    "log10": ("log10({x} + {c})", 0.0),
    "sqrt": ("sqrt({x})", None),
    "reciprocal": ("1 / {x}", None),
    "logit": ("log({x} / (1 - {x}))", None),
    "neglog": ("sign({x}) * log1p(abs({x}))", None),
    "power": ("sign({x}) * abs({x}) ** {c}", 0.5),
}

RowId = int | float | str


def _comment(text: str) -> str:
    """Return ``text`` on one line, safe to write after a ``#``."""
    return re.sub("[\\x00-\\x1f\\x7f\\u2028\\u2029]+", " ", text).strip()


def _names(columns: Sequence[str]) -> str:
    return ", ".join(columns[:6]) + (f" and {len(columns) - 6} more" if len(columns) > 6 else "")


# ---------------------------------------------------------------------------
# Checking fields read from JSON
# ---------------------------------------------------------------------------


def _text(data: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be text, got {value!r}.")
    return value


def _texts(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list | tuple) or not all(isinstance(v, str) for v in value):
        raise TypeError(f"{key!r} must be a list of text, got {value!r}.")
    return tuple(value)


def _row_ids(data: Mapping[str, Any], key: str) -> tuple[RowId, ...]:
    value = data.get(key)
    if not isinstance(value, list | tuple) or not all(_is_row_id(v) for v in value):
        raise TypeError(f"{key!r} must be a list of row ids (numbers or text), got {value!r}.")
    return tuple(value)


def _is_row_id(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, int | str) and not isinstance(value, bool)


def _optional_number(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{key!r} must be a number, got {value!r}.")
    return float(value)


def _has_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"The table has no column {_names(missing)}.")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One step of a recipe. Subclasses set ``kind`` and implement the four methods."""

    kind: ClassVar[str] = ""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the table after this step; ``df`` is never modified."""
        raise NotImplementedError

    def code(self) -> list[str]:
        """Return the lines of Python that perform this step on the table ``df``."""
        raise NotImplementedError

    def describe(self) -> str:
        """Return the step in a sentence, for the recipe's history."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return the step as JSON-ready data, with its ``kind``."""
        out: dict[str, Any] = {"kind": self.kind}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Rebuild a step from :meth:`to_dict`'s output, checking every field."""
        raise NotImplementedError


@dataclass(frozen=True)
class DropColumns(Step):
    """Remove columns."""

    columns: tuple[str, ...]
    kind: ClassVar[str] = "drop_columns"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, self.columns)
        return df.drop(columns=list(self.columns))

    def code(self) -> list[str]:  # noqa: D102
        return [f"df = df.drop(columns={python_literal(list(self.columns))})"]

    def describe(self) -> str:  # noqa: D102
        return f"Drop the columns {_names(self.columns)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DropColumns:  # noqa: D102
        return cls(columns=_texts(data, "columns"))


@dataclass(frozen=True)
class KeepColumns(Step):
    """Keep only these columns, in this order."""

    columns: tuple[str, ...]
    kind: ClassVar[str] = "keep_columns"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, self.columns)
        return df[list(self.columns)].copy()

    def code(self) -> list[str]:  # noqa: D102
        return [f"df = df[{python_literal(list(self.columns))}].copy()"]

    def describe(self) -> str:  # noqa: D102
        return f"Keep the columns {_names(self.columns)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KeepColumns:  # noqa: D102
        return cls(columns=_texts(data, "columns"))


@dataclass(frozen=True)
class RenameColumns(Step):
    """Rename columns, given as ``(old, new)`` pairs."""

    pairs: tuple[tuple[str, str], ...]
    kind: ClassVar[str] = "rename_columns"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, [old for old, _ in self.pairs])
        renamed = [dict(self.pairs).get(c, c) for c in df.columns]
        if len(set(renamed)) < len(renamed):
            raise ValueError("Renaming would give two columns the same name.")
        return df.rename(columns=dict(self.pairs))

    def code(self) -> list[str]:  # noqa: D102
        return [f"df = df.rename(columns={python_literal(dict(self.pairs))})"]

    def describe(self) -> str:  # noqa: D102
        return "Rename " + ", ".join(f"{old} to {new}" for old, new in self.pairs)

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {"kind": self.kind, "pairs": [list(p) for p in self.pairs]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenameColumns:  # noqa: D102
        pairs = data.get("pairs")
        if not isinstance(pairs, list | tuple) or not all(
            isinstance(p, list | tuple) and len(p) == 2 and all(isinstance(n, str) for n in p) for p in pairs
        ):
            raise TypeError(f"'pairs' must be a list of [old, new] names, got {pairs!r}.")
        return cls(pairs=tuple((p[0], p[1]) for p in pairs))


@dataclass(frozen=True)
class DropRows(Step):
    """Remove rows by their id (index label), with the reason."""

    rows: tuple[RowId, ...]
    reason: str = ""
    kind: ClassVar[str] = "drop_rows"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        missing = [r for r in self.rows if r not in df.index]
        if missing:
            raise ValueError(f"The table has no row {_names([str(r) for r in missing])}.")
        return df.drop(index=list(self.rows))

    def code(self) -> list[str]:  # noqa: D102
        reason = f"  # {_comment(self.reason)}" if _comment(self.reason) else ""
        return [f"df = df.drop(index={python_literal(list(self.rows))}){reason}"]

    def describe(self) -> str:  # noqa: D102
        rows = _names([str(r) for r in self.rows])
        return f"Drop the rows {rows}" + (f": {_comment(self.reason)}" if _comment(self.reason) else "")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DropRows:  # noqa: D102
        return cls(rows=_row_ids(data, "rows"), reason=_text(data, "reason", default=""))


@dataclass(frozen=True)
class FilterRows(Step):
    """Keep the rows where an expression is true, for example ``Tmax1 > 290``."""

    expression: str
    kind: ClassVar[str] = "filter_rows"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        mask = Expression(self.expression, df.columns).evaluate(df)
        if not isinstance(mask, pd.Series) or not pd.api.types.is_bool_dtype(mask):
            raise ValueError(f"{self.expression!r} is not a condition: it must be true or false on each row.")
        return df[mask].copy()

    def code(self) -> list[str]:  # noqa: D102
        # Parsed against no columns only to write it: apply() has already checked the names.
        return [f"df = df[{_unchecked(self.expression).to_code()}].copy()"]

    def describe(self) -> str:  # noqa: D102
        return f"Keep the rows where {_comment(self.expression)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FilterRows:  # noqa: D102
        step = cls(expression=_text(data, "expression"))
        _unchecked(step.expression)
        return step


@dataclass(frozen=True)
class AddColumn(Step):
    """Add (or replace) a column computed from an expression, for example ``Tout1 - Tin``."""

    name: str
    expression: str
    kind: ClassVar[str] = "add_column"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        value = Expression(self.expression, df.columns).evaluate(df)
        out = df.copy()
        out[self.name] = value
        return out

    def code(self) -> list[str]:  # noqa: D102
        return [f"df[{python_literal(self.name)}] = {_unchecked(self.expression).to_code()}"]

    def describe(self) -> str:  # noqa: D102
        return f"Add the column {_comment(self.name)} = {_comment(self.expression)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AddColumn:  # noqa: D102
        step = cls(name=_text(data, "name"), expression=_text(data, "expression"))
        _unchecked(step.expression)
        return step


@dataclass(frozen=True)
class Transform(Step):
    """Transform a column, in place or into a new column: one of :data:`TRANSFORMS`.

    ``constant`` is the offset of ``log`` and ``log10`` (``log(x + c)``, default 0) and
    the exponent of ``power`` (default 0.5, a signed square root).
    """

    column: str
    function: str
    constant: float | None = None
    new_name: str | None = None
    kind: ClassVar[str] = "transform"

    def __post_init__(self) -> None:
        """Refuse a transform that is not offered, and a column name a back-tick cannot quote."""
        if self.function not in TRANSFORMS:
            raise ValueError(f"Unknown transform {self.function!r}; choose from {', '.join(TRANSFORMS)}.")
        if "`" in self.column:
            raise ValueError("A column whose name holds a back-tick cannot be transformed; rename it first.")

    @property
    def expression(self) -> str:
        """The transform as an expression of the column, which is what it applies and writes."""
        template, default = TRANSFORMS[self.function]
        constant = self.constant if self.constant is not None else default
        if constant == 0.0 and " + {c}" in template:
            template = template.replace(" + {c}", "")
        c = python_literal(float(constant)) if constant is not None else ""
        return template.format(x=f"`{self.column}`", c=c)

    def _target(self) -> AddColumn:
        return AddColumn(name=self.new_name or self.column, expression=self.expression)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, [self.column])
        return self._target().apply(df)

    def code(self) -> list[str]:  # noqa: D102
        return self._target().code()

    def describe(self) -> str:  # noqa: D102
        into = f" into {_comment(self.new_name)}" if self.new_name else ""
        return f"Transform {_comment(self.column)} with {self.function}{into}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Transform:  # noqa: D102
        new_name = data.get("new_name")
        if new_name is not None and not isinstance(new_name, str):
            raise TypeError(f"'new_name' must be text, got {new_name!r}.")
        return cls(
            column=_text(data, "column"),
            function=_text(data, "function"),
            constant=_optional_number(data, "constant"),
            new_name=new_name,
        )


@dataclass(frozen=True)
class ToNumeric(Step):
    """Read text columns as numbers; a value that is not a number becomes missing."""

    columns: tuple[str, ...]
    kind: ClassVar[str] = "to_numeric"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, self.columns)
        out = df.copy()
        for column in self.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out

    def code(self) -> list[str]:  # noqa: D102
        return [
            f'df[{python_literal(c)}] = pd.to_numeric(df[{python_literal(c)}], errors="coerce")' for c in self.columns
        ]

    def describe(self) -> str:  # noqa: D102
        return f"Read {_names(self.columns)} as numbers"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToNumeric:  # noqa: D102
        return cls(columns=_texts(data, "columns"))


@dataclass(frozen=True)
class SetIndex(Step):
    """Make a column the row ids."""

    column: str
    kind: ClassVar[str] = "set_index"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        _has_columns(df, [self.column])
        if df[self.column].duplicated().any():
            raise ValueError(f"{self.column!r} has repeated values, so it cannot identify the rows.")
        return df.set_index(self.column)

    def code(self) -> list[str]:  # noqa: D102
        return [f"df = df.set_index({python_literal(self.column)})"]

    def describe(self) -> str:  # noqa: D102
        return f"Identify the rows by {_comment(self.column)}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SetIndex:  # noqa: D102
        return cls(column=_text(data, "column"))


STEPS: dict[str, type[Step]] = {
    cls.kind: cls
    for cls in (
        DropColumns,
        KeepColumns,
        RenameColumns,
        DropRows,
        FilterRows,
        AddColumn,
        Transform,
        ToNumeric,
        SetIndex,
    )
}


def step_from_dict(data: Mapping[str, Any]) -> Step:
    """Rebuild any step from its :meth:`Step.to_dict` output."""
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind not in STEPS:
        raise ValueError(f"Unknown recipe step {kind!r}; the steps are {', '.join(STEPS)}.")
    return STEPS[kind].from_dict(data)


def _unchecked(expression: str) -> Expression:
    """Parse an expression to write its code; ``apply`` checks its names against the table."""
    return Expression(expression, None)


# ---------------------------------------------------------------------------
# Source file and recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    """Where a recipe's table comes from: a file's name, format, read options and digest.

    Parameters
    ----------
    name : str
        The file name, as the script will look for it next to itself.
    format : {"csv", "excel"}
    options : dict
        Options for :func:`pandas.read_csv` or :func:`pandas.read_excel`, from
        :data:`READ_OPTIONS`.
    sha256 : str, optional
        The file's SHA-256 digest. The script checks it before reading, so it refuses
        a different file of the same name.
    """

    name: str
    format: str = "csv"
    options: Mapping[str, Any] = field(default_factory=dict)
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown format, a read option that is not offered, and a malformed digest."""
        if self.format not in READ_OPTIONS:
            raise ValueError(f"Unknown source format {self.format!r}; choose from {', '.join(READ_OPTIONS)}.")
        unknown = set(self.options) - READ_OPTIONS[self.format]
        if unknown:
            raise ValueError(f"Unsupported read options {sorted(unknown)} for a {self.format} source.")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("'sha256' must be 64 lower-case hexadecimal characters.")

    @classmethod
    def of(cls, name: str, content: bytes, format: str = "csv", **options: Any) -> SourceFile:  # noqa: A002, ANN401
        """Describe a file from its name and content, recording its digest."""
        return cls(name=name, format=format, options=options, sha256=hashlib.sha256(content).hexdigest())

    def read(self, content: bytes | str | Path) -> pd.DataFrame:
        """Read the table from the file's content (bytes) or from a path."""
        handle = io.BytesIO(content) if isinstance(content, bytes) else content
        if self.format == "excel":
            return pd.read_excel(handle, **self.options)
        return pd.read_csv(handle, **self.options)

    def code(self) -> list[str]:
        """Return the lines that check and read the file into ``df``."""
        reader = "pd.read_excel" if self.format == "excel" else "pd.read_csv"
        options = "".join(f", {k}={python_literal(v)}" for k, v in self.options.items())
        lines = [f"SOURCE = Path({python_literal(self.name)})"]
        if self.sha256:
            lines += [
                f"if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != {python_literal(self.sha256)}:",
                f"    raise ValueError({python_literal(f'{self.name} is not the file this recipe was written for.')})",
            ]
        return [*lines, f"df = {reader}(SOURCE{options})"]

    def to_dict(self) -> dict[str, Any]:
        """Return the source as JSON-ready data."""
        return {"name": self.name, "format": self.format, "options": dict(self.options), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceFile:
        """Rebuild a source from :meth:`to_dict`'s output, checking every field."""
        options = data.get("options", {})
        if not isinstance(options, dict):
            raise TypeError(f"'options' must be a mapping, got {options!r}.")
        sha = data.get("sha256")
        if sha is not None and not isinstance(sha, str):
            raise TypeError(f"'sha256' must be text, got {sha!r}.")
        return cls(name=_text(data, "name"), format=_text(data, "format", default="csv"), options=options, sha256=sha)


@dataclass(frozen=True)
class Recipe:
    """A source and the steps that refine it into the table an analysis uses.

    Recipes are immutable: :meth:`then` returns a longer recipe, so the previous one is
    the undo.

    Examples
    --------
    >>> recipe = Recipe(SourceFile("LDPE.csv", options={"index_col": 0}))
    >>> recipe = recipe.then(DropColumns(("Tcin1", "Tcin2")), AddColumn("dT", "Tout1 - Tin"))
    >>> table = recipe.apply(pd.read_csv("LDPE.csv", index_col=0))  # doctest: +SKIP
    >>> print(recipe.to_python())  # doctest: +SKIP
    """

    source: SourceFile
    steps: tuple[Step, ...] = ()

    def then(self, *steps: Step) -> Recipe:
        """Return this recipe with ``steps`` added at the end."""
        return dataclasses.replace(self, steps=(*self.steps, *steps))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every step on ``df`` (the source's table), which is not modified.

        Raises
        ------
        ValueError
            Naming the step that failed and why.
        """
        for number, step in enumerate(self.steps, start=1):
            try:
                df = step.apply(df)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Step {number} ({step.describe()}) failed: {error}") from error
        return df

    def run(self, content: bytes | str | Path) -> pd.DataFrame:
        """Read the source from its content or path, then run every step."""
        return self.apply(self.source.read(content))

    def describe(self) -> list[str]:
        """Return each step in a sentence, in order."""
        return [step.describe() for step in self.steps]

    def to_python(self) -> str:
        """Return a script that rebuilds the table with pandas and numpy alone, as ``df``."""
        lines = [
            f'"""The recipe for {_comment(self.source.name)}, as Python: rebuilds the table as `df`.',
            "",
            f"Written by process-improve {_release()}. Run it next to {_comment(self.source.name)}.",
            '"""',
            "",
        ]
        if self.source.sha256:
            lines.append("import hashlib")
        lines += ["from pathlib import Path", "", "import numpy as np", "import pandas as pd", ""]
        lines += self.source.code()
        for number, step in enumerate(self.steps, start=1):
            lines += ["", f"# {number}. {_comment(step.describe())}", *step.code()]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe as JSON-ready data."""
        return {"source": self.source.to_dict(), "steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recipe:
        """Rebuild a recipe from :meth:`to_dict`'s output, checking every field."""
        source = data.get("source")
        steps = data.get("steps", [])
        if not isinstance(source, dict) or not isinstance(steps, list):
            raise TypeError("A recipe needs a 'source' mapping and a 'steps' list.")
        return cls(source=SourceFile.from_dict(source), steps=tuple(step_from_dict(s) for s in steps))


def _release() -> str:
    try:
        return version("process-improve")
    except PackageNotFoundError:  # pragma: no cover - only in a checkout that was never installed
        return "(unreleased)"
