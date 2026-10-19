# (c) Kevin Dunn, 2010-2026. MIT License.

"""Run sheets: a design written out for the lab, and the results read back in.

A run sheet is the table that leaves the analysis and comes back filled in: a spreadsheet for
the people making the runs, the upload an online study asks for, a CSV from a laboratory system.
:func:`run_sheet` writes one, in actual or coded units, under the factors' own names or labels
such as ``X01`` to ``Xnn``, with empty columns for the responses. :func:`read_run_sheet` reads
one back into the factors' own names and units, whichever units and labels it was written in.
"""

from __future__ import annotations

import io
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from process_improve.experiments.factor import DesignResult, Factor, FactorType

__all__ = ["read_run_sheet", "run_sheet"]

Units = Literal["actual", "coded"]
Labels = Mapping[str, str] | Literal["X"] | None

#: The column that numbers the runs in the order they are made.
RUN = "Run"
#: A setting this close to the end of a range, in coded units, counts as on it.
_TOLERANCE = 1e-9
_X_LABEL = re.compile(r"x0*(\d+)")


def run_sheet(
    runs: DesignResult | pd.DataFrame | Sequence[Mapping[str, Any]],
    factors: Sequence[Factor],
    *,
    units: Units = "actual",
    labels: Labels = None,
    responses: Sequence[str] = ("y",),
) -> pd.DataFrame:
    """Write runs as a run sheet: one row per run, in the order given, with empty response columns.

    Parameters
    ----------
    runs : DesignResult, DataFrame or list of dicts
        The runs in the factors' actual units: a :class:`DesignResult` (its ``design_actual``,
        in its randomised run order), or a table with a column per factor, such as the
        ``runs`` that :func:`~process_improve.experiments.recommend_next_step` returns. A
        ``Block`` column is kept.
    factors : sequence of Factor
        The factors, which give the ranges for coding and the order of the columns.
    units : {"actual", "coded"}, default "actual"
        The units the settings are written in. Coded units run from -1 at a factor's low
        value to +1 at its high value; a categorical factor keeps its levels either way.
    labels : mapping, "X" or None
        The column names: the factors' own names (None), ``X01``, ``X02``, ... in the order
        of ``factors`` (``"X"``; three digits from 100 factors), or a mapping from a factor's
        name to its label, with unmapped factors under their own names.
    responses : sequence of str, default ("y",)
        The response columns, written empty for the results.

    Returns
    -------
    pandas.DataFrame
        Columns ``Run`` (1, 2, ...), ``Block`` when the runs have one, a column per factor
        and a column per response. Save it with ``to_csv`` or ``to_excel``. ``attrs["columns"]``
        maps each factor to its column.

    Raises
    ------
    ValueError
        If a factor is missing from the runs, a label is given twice or clashes with another
        column, a response is named twice, or ``units`` is unknown.

    Examples
    --------
    >>> from process_improve.experiments import Factor, generate_design
    >>> factors = [Factor(name="Temperature", low=60, high=100), Factor(name="Time", low=20, high=60)]
    >>> sheet = run_sheet(generate_design(factors, "full_factorial"), factors, labels="X")
    >>> list(sheet.columns)
    ['Run', 'X01', 'X02', 'y']
    """
    if units not in ("actual", "coded"):
        raise ValueError(f"units must be 'actual' or 'coded', not {units!r}.")
    frame = _runs_frame(runs)
    names = [f.name for f in factors]
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise ValueError(
            f"The runs have no column for the factor(s) {missing}; their columns are {list(frame.columns)}."
        )
    columns = _labels(factors, labels)
    settings = frame[names] if units == "actual" else _coded(frame[names], factors)
    parts: dict[str, Any] = {RUN: np.arange(1, len(frame) + 1)}
    if "Block" in frame.columns:
        parts["Block"] = frame["Block"].to_numpy()
    parts |= {columns[name]: settings[name].to_numpy() for name in names}
    clash = [r for r in responses if r in parts]
    if clash or len(set(responses)) != len(responses):
        raise ValueError(f"Each response needs a column of its own; {list(responses)} repeats one or names {clash}.")
    sheet = pd.DataFrame(parts | {r: np.full(len(frame), np.nan) for r in responses})
    sheet.attrs["columns"] = columns
    return sheet


def read_run_sheet(
    source: str | Path | bytes | io.IOBase | pd.DataFrame,
    factors: Sequence[Factor],
    *,
    labels: Labels = None,
    units: Units | Literal["auto"] = "auto",
) -> pd.DataFrame:
    """Read a run sheet back into the factors' own names and actual units.

    Parameters
    ----------
    source : path, bytes, file object or DataFrame
        The sheet: a path to a ``.csv``, ``.xlsx`` or ``.xls`` file, the bytes of an Excel
        file, a text or binary file object holding CSV, or a DataFrame. In a workbook, the
        first sheet that has a column for every factor is read.
    factors : sequence of Factor
        The factors the sheet sets.
    labels : mapping, "X" or None
        How the factors' columns are named, as for :func:`run_sheet`. A factor's column is
        also found under its own name, and matching ignores case, spaces and the leading
        zeros of an ``X`` label (``x1`` is ``X01``).
    units : {"auto", "actual", "coded"}, default "auto"
        The units of the settings. ``"auto"`` reads them as coded when every setting is
        between -1 and +1 and at least one of them cannot be an actual value (it lies
        outside its factor's range); otherwise as actual.

    Returns
    -------
    pandas.DataFrame
        A column per factor, under its name and in its actual units, then the sheet's other
        columns (responses, run numbers) as they were. Rows with no setting at all are left
        out. ``attrs["units"]`` says which units the settings were read in, and
        ``attrs["columns"]`` maps each factor to the column it was read from.

    Raises
    ------
    ValueError
        If the source cannot be read, a factor has no column, a setting is not a number
        (or not one of a categorical factor's levels), or ``units`` is unknown.

    Warns
    -----
    UserWarning
        For a setting outside its factor's range, which the runs did not keep to.
    """
    if units not in ("auto", "actual", "coded"):
        raise ValueError(f"units must be 'auto', 'actual' or 'coded', not {units!r}.")
    table = _table(source, factors, labels)
    found = _find_columns(table, factors, labels)
    settings = table[list(found.values())].set_axis(list(found), axis=1)
    keep = settings.notna().any(axis=1)
    settings, table = settings[keep].reset_index(drop=True), table[keep].reset_index(drop=True)
    numeric = [f for f in factors if f.type != FactorType.categorical]
    values = {f.name: _numbers(settings[f.name], f.name) for f in numeric}
    read_as = _units_of(values, numeric) if units == "auto" else units
    actual: dict[str, Any] = {}
    for f in factors:
        if f.type == FactorType.categorical:
            actual[f.name] = _levels(settings[f.name], f)
        else:
            low, high = float(f.low), float(f.high)  # type: ignore[arg-type]
            column = values[f.name]
            actual[f.name] = (low + high) / 2 + column * (high - low) / 2 if read_as == "coded" else column
    out = pd.DataFrame(actual)
    _warn_outside(out, numeric)
    others = [c for c in table.columns if c not in found.values()]
    out = pd.concat([out, table[others]], axis=1)
    out.attrs["units"] = read_as
    out.attrs["columns"] = dict(found)
    return out


# --------------------------------------------------------------------------- writing
def _runs_frame(runs: DesignResult | pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(runs, DesignResult):
        return pd.DataFrame(runs.design_actual).reset_index(drop=True)
    return pd.DataFrame(runs).reset_index(drop=True)


def _labels(factors: Sequence[Factor], labels: Labels) -> dict[str, str]:
    """Return each factor's column name, refusing a label given twice."""
    names = [f.name for f in factors]
    if labels is None:
        columns = {n: n for n in names}
    elif isinstance(labels, str):
        if labels != "X":
            raise ValueError(f"labels must be a mapping, 'X' or None, not {labels!r}.")
        width = max(2, len(str(len(names))))
        columns = {n: f"X{i:0{width}d}" for i, n in enumerate(names, start=1)}
    else:
        unknown = sorted(set(labels) - set(names))
        if unknown:
            raise ValueError(f"labels names {unknown}, which are not factors: {names}.")
        columns = {n: str(labels.get(n, n)) for n in names}
    if len(set(columns.values())) != len(columns) or {RUN, "Block"} & set(columns.values()):
        raise ValueError(f"Each factor needs a column of its own, not 'Run' or 'Block'; the labels are {columns}.")
    return columns


def _coded(frame: pd.DataFrame, factors: Sequence[Factor]) -> pd.DataFrame:
    coded = frame.copy()
    for f in factors:
        if f.type != FactorType.categorical:
            low, high = float(f.low), float(f.high)  # type: ignore[arg-type]
            coded[f.name] = (frame[f.name].astype(float) - (low + high) / 2) / ((high - low) / 2)
    return coded


# --------------------------------------------------------------------------- reading
def _table(
    source: str | Path | bytes | io.IOBase | pd.DataFrame, factors: Sequence[Factor], labels: Labels
) -> pd.DataFrame:
    """Return the sheet as a DataFrame, from whichever form it came in."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, bytes):
        return _workbook(io.BytesIO(source), factors, labels)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
            return _workbook(path, factors, labels)
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Read a .csv, .xlsx or .xls file; {path.name!r} is none of them.")
        return pd.read_csv(path)
    try:
        return pd.read_csv(source)  # type: ignore[call-overload]
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"The run sheet could not be read as CSV: {exc}") from exc


def _workbook(source: Path | io.BytesIO, factors: Sequence[Factor], labels: Labels) -> pd.DataFrame:
    """Return the first sheet of a workbook that has a column for every factor."""
    try:
        sheets = pd.read_excel(source, sheet_name=None)
    except Exception as exc:  # the Excel readers raise several unrelated types on a bad file
        raise ValueError(f"The run sheet could not be read as an Excel workbook: {exc}") from exc
    for sheet in sheets.values():
        try:
            _find_columns(sheet, factors, labels)
        except ValueError:
            continue
        return sheet
    raise ValueError(f"No sheet of the workbook ({', '.join(map(str, sheets))}) has a column for every factor.")


def _key(text: object) -> str:
    """Write a column name the way columns are matched: lower case, no spaces, X labels without leading zeros."""
    key = re.sub(r"\s+", "", str(text)).lower()
    match = _X_LABEL.fullmatch(key)
    return f"x{match.group(1)}" if match else key


def _find_columns(table: pd.DataFrame, factors: Sequence[Factor], labels: Labels) -> dict[str, Any]:
    """Map each factor to its column: its label first, then its own name."""
    by_key: dict[str, Any] = {}
    for column in table.columns:
        by_key.setdefault(_key(column), column)
    columns = _labels(factors, labels)
    found: dict[str, Any] = {}
    missing: list[str] = []
    for f in factors:
        match = by_key.get(_key(columns[f.name]), by_key.get(_key(f.name)))
        if match is None:
            missing.append(columns[f.name])
        else:
            found[f.name] = match
    if missing:
        raise ValueError(f"The run sheet has no column for {missing}; its columns are {list(table.columns)}.")
    return found


def _numbers(column: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(column, errors="coerce")
    bad = column[values.isna() & column.notna()]
    if len(bad) or values.isna().any():
        shown = bad.iloc[0] if len(bad) else "an empty cell"
        raise ValueError(f"Factor {name!r} needs a number in every run; the run sheet has {shown!r}.")
    return values.to_numpy(dtype=float)


def _levels(column: pd.Series, factor: Factor) -> pd.Series:
    """Return a categorical factor's settings as its levels, matching a level written as text."""
    levels = {str(level).strip(): level for level in factor.levels or []}
    read = column.map(lambda v: levels.get(str(v).strip()))
    if read.isna().any():
        bad = column[read.isna()].iloc[0]
        raise ValueError(f"Factor {factor.name!r} takes the levels {factor.levels}; the run sheet has {bad!r}.")
    return read


def _units_of(values: dict[str, np.ndarray], factors: Sequence[Factor]) -> Units:
    """Read the sheet as coded when every setting fits -1 to +1 and some cannot be an actual value."""
    if not factors:
        return "actual"
    fits_coded = all(np.all(np.abs(values[f.name]) <= 1 + _TOLERANCE) for f in factors)
    impossible = any(not np.all(_inside(values[f.name], f)) for f in factors)
    return "coded" if fits_coded and impossible else "actual"


def _inside(values: np.ndarray, factor: Factor) -> np.ndarray:
    low, high = float(factor.low), float(factor.high)  # type: ignore[arg-type]
    slack = _TOLERANCE * max(1.0, abs(high - low))
    return (values >= low - slack) & (values <= high + slack)


def _warn_outside(actual: pd.DataFrame, factors: Sequence[Factor]) -> None:
    """Warn about settings outside the factors' ranges, naming each factor and its extremes."""
    outside = []
    for f in factors:
        values = actual[f.name].to_numpy(dtype=float)
        if not np.all(_inside(values, f)):
            outside.append(
                f"{f.name} from {values.min():.4g} to {values.max():.4g} (range {f.low:.4g} to {f.high:.4g})"
            )
    if outside:
        warnings.warn(
            f"The run sheet sets factors outside their ranges: {'; '.join(outside)}.",
            UserWarning,
            stacklevel=3,
        )
