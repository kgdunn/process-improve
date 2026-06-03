# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import ClassVar, cast

import numpy as np
import pandas as pd


class Column(pd.Series):
    """Create a column. Can be used as a factor, or a response vector."""

    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    # Temporary properties
    _internal_names: ClassVar[list[str]] = [*pd.DataFrame._internal_names, "not_used_for_now"]  # type: ignore[attr-defined]  # _internal_names exists at runtime but is missing from pandas stubs
    _internal_names_set: ClassVar[set[str]] = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata: ClassVar[list[str]] = [
        "pi_index",  # might be used later if the user provides their own index
        "pi_numeric",  # if numeric indicator
        "pi_lo",  # if numeric: low level (-1)
        "pi_hi",  # if numeric: high level (+1)
        "pi_range",  # if numeric: range: distance from low to high
        "pi_center",  # if numeric: midway between low and high (0)
        "pi_is_coded",  # is it a coded variables, or in real-world units
        "pi_units",  # string variable, containing the units
        "pi_name",  # name of the column
    ]

    # Declared for static typing only. These are populated at runtime via the
    # pandas ``_metadata`` mechanism (bare annotations create no class-level
    # attribute, so the pandas attribute machinery is untouched).
    pi_index: bool
    pi_numeric: bool
    pi_lo: float | None
    pi_hi: float | None
    pi_range: tuple | None
    pi_center: float | None
    pi_is_coded: bool
    pi_units: str | None
    pi_name: str | None
    pi_levels: dict

    @property
    def _constructor(self) -> type[Column]:
        return Column

    def to_coded(self, center: float | None = None, range: tuple | None = None) -> Column:  # noqa: A002
        """Convert the column vector to coded units."""
        out = self.copy(deep=True)
        if self.pi_is_coded:
            return out

        x_center = center or self.pi_center
        x_range = range or self.pi_range

        # Simply override the values and the `pi_is_coded` flag, but all the
        # rest remains as is.
        out.iloc[:] = (np.asarray(self.values) - x_center) / (0.5 * np.diff(np.asarray(x_range))[0])
        out.pi_is_coded = True
        if out.pi_name:
            out.name = f"{out.pi_name} [coded]"

        return out

    def to_realworld(self, center: float | None = None, range: tuple | None = None) -> Column:  # noqa: A002
        """Convert the column vector to real-world units."""
        out = self.copy(deep=True)
        if not self.pi_is_coded:
            return out

        x_center = center or self.pi_center
        x_range = range or self.pi_range
        # Simply override the values and the `pi_is_coded` flag, but all the
        # rest remains as is.
        out.iloc[:] = np.asarray(self.values) * (0.5 * np.diff(np.asarray(x_range))[0]) + x_center
        out.pi_is_coded = False
        if out.pi_name and out.pi_units:
            out.name = f"{out.pi_name} [{out.pi_units}]"

        return out

    def copy(self, deep: bool = True) -> Column:  # type: ignore[misc]  # pandas marks Series.copy @final; subclassing is intentional here
        """Create a copy of this Column, preserving the name."""
        out = pd.Series.copy(self, deep=deep)
        out.name = self.name
        return cast("Column", out)

    def extend(self, values: list) -> Column:
        """Extend the column with the list of new values."""
        if not isinstance(values, list):
            raise TypeError(
                f"'values' must be a list; got {type(values).__name__}."
            )
        prior_n = self.index[-1]
        index = list(range(prior_n + 1, prior_n + len(values) + 1))
        new = pd.Series(data=values, index=index)
        intermediate = self.copy(deep=True)
        intermediate = pd.concat([intermediate, new])

        # Carry the meta data over. pd.concat does not propagate custom
        # metadata from subclassed Series.
        for key in self._metadata:
            setattr(intermediate, key, getattr(self, key))
        intermediate.name = self.name
        return intermediate


class Expt(pd.DataFrame):
    """Dataframe carrying experimental data plus process-improve metadata.

    ``Expt`` (short for "Experiment") is a :class:`pandas.DataFrame` subclass
    that adds library-managed metadata fields prefixed with ``pi_`` -- short
    for "process-improve". The prefix is what keeps these reserved attribute
    names from colliding with column names from a caller-supplied DataFrame.

    Pinned metadata (preserved across subsetting via ``_metadata``):

    - ``pi_title``  -- short human-readable name for the dataset
    - ``pi_source`` -- provenance string (file path, URL, ...)
    - ``pi_units``  -- units string for the numeric columns

    Other ``pi_*`` attributes (``pi_range``, ``pi_lo``, ``pi_hi``,
    ``pi_center``, ``pi_name``) are set by the experiments factory helpers
    in this module; see :func:`expt` / :func:`create_names`.

    The ``pi_`` prefix is documented in ``CONTRIBUTING.md`` and is part of
    the package's public API surface; new metadata fields should follow the
    same prefix.
    """

    # Temporary properties
    _internal_names: ClassVar[list[str]] = [*pd.DataFrame._internal_names, "not_used_for_now"]  # type: ignore[attr-defined]  # _internal_names exists at runtime but is missing from pandas stubs
    _internal_names_set: ClassVar[set[str]] = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata: ClassVar[list[str]] = ["pi_source", "pi_title", "pi_units"]

    # Declared for static typing only. These are populated at runtime via the
    # pandas ``_metadata`` mechanism (bare annotations create no class-level
    # attribute, so the pandas attribute machinery is untouched).
    pi_source: dict | None
    pi_title: str | None
    pi_units: dict | None

    @property
    def _constructor(self) -> type[Expt]:
        return Expt

    def __repr__(self) -> str:
        """Return a string representation of the experiment."""
        title = f"Name: {self.pi_title}"
        dimensions = f"Size: {self.shape[0]} experiments; {self.shape[1]} columns."
        return "\n".join([pd.DataFrame.__repr__(self), title, dimensions])

    def get_title(self) -> str:
        """Return the experiment title, or empty string if not set."""
        return self.pi_title or ""


def create_names(n: int, letters: bool = True, prefix: str = "X", start_at: int = 1, padded: bool = True) -> list[str]:
    """
    Return default factor names, for a given number of `n` [integer] factors.
    The factor name "I" is never used.

    If `letters` is True (default), then at most 25 factors can be returned.

    If `letters` is False, then the prefix is used to construct names which are
    the combination of the prefix and numbers, starting at `start_at`.

    Example:
        >>> create_names(5)
            ["A", "B", "C", "D", "E"]

        >>> create_names(3, letters=False)
            ["X1", "X2", "X3"]

        >>> create_names(3, letters=False, prefix='Q', start_at=9,
                             padded=True)
            ["Q09", "Q10", "Q11"]
    """
    if letters and n <= 25:
        out = [chr(65 + i) for i in range(n)]
        if "I" in out:
            out.remove("I")
            out.append(chr(65 + n))

    else:
        longest = 0
        if padded:
            longest = len(str(start_at + n - 1))

        out = [f"{prefix!s}{str(i).rjust(longest, '0')}" for i in range(start_at, n + start_at)]

    return out


def c(*args, **kwargs) -> Column:  # noqa: C901, PLR0912, PLR0915
    """
    Perform the equivalent of the R function "c(...)", to combine data elements
    into a DataFrame. Convert every entry into a floating point object.

    Inputs
    ------

    index: a list of names for the entries in `args`

    name:  a name for the column

    Usage
    -----
    # All equivalent ways of creating a factor, "A"

    A = c(-1, 0, +1, -1, +1)

    A = c(-1, 0, +1, -1, +1, index=['lo', 'cp', 'hi', 'lo', 'hi'])
    A = c( 4, 5,  6,  4,  6, range=(4, 6))
    A = c( 4, 5,  6,  4,  6, center=5, range=(4, 6))  # more explicit
    A = c( 4, 5,  6,  4,  6, lo=4, hi=6)
    A = c( 4, 5,  6,  4,  6, lo=4, hi=6, name = 'A')
    A = c([4, 5,  6,  4,  6], lo=4, hi=6, name = 'A')
    A = c([4, 5,  6,  4,  6], lo=4, hi=6, name = 'A')

    # By default, the assumption is the variable levels supplied are coded
    # units. But if any one of the following: `lo`, `hi`, `center`, `range` OR
    # `units` are specified, then immediately it is assumed that the variable
    # values are not coded.
    # So, to force the specification, you may supply the optional input of
    # `coded` as True or False
    A = c([4, 5,  6,  4,  6], lo=1, hi=3, coded=True)
    A = c([4, 5,  6,  4,  6], lo=1, hi=3, coded=False, units="g/mL")

    # Categorical variables
    B = c(0, 1, 0, 1, 0, 2, levels =(0, 1, 2))
    M = c("Dry", "Wet", "Dry", "Wet", levels = ("Dry", "Wet"))

    """
    sanitize: list | pd.Series = []
    numeric = True
    override_coded = kwargs.get("coded")

    if "levels" in kwargs:
        numeric = False

    for j in args:
        if isinstance(j, Iterable):
            if isinstance(j, np.ndarray):
                sanitize = j.ravel().tolist()

            if isinstance(j, pd.Series):
                sanitize = j.copy()
                if "index" not in kwargs:
                    kwargs["index"] = sanitize.index

            if isinstance(j, list):
                sanitize = j.copy()

            try:
                sanitize = [float(j) for j in sanitize]
            except ValueError:
                numeric = False

        else:
            try:
                sanitize.append(float(j))
            except ValueError:
                numeric = False
                sanitize.append(j)

    # Index creation
    default_idx = list(range(1, len(sanitize) + 1))
    index = kwargs.get("index", default_idx)
    if len(index) != len(sanitize):
        raise IndexError('Length of "index" must match the number of numeric inputs.')

    out = Column(data=sanitize, index=index, name=None)
    # Use sensible defaults, if not provided
    out.pi_index = True
    out.pi_lo = None
    out.pi_hi = None
    out.pi_range = None
    out.pi_center = None
    out.pi_numeric = numeric
    out.pi_units = None
    out.pi_name = None
    out.pi_is_coded = True

    out.pi_name = kwargs.get("name", "Unnamed")
    out.name = out.pi_name
    if numeric:
        # If any of 'lo', 'hi', 'center', or 'range' are specified, then it
        # is assumed that the variable is NOT coded
        try:
            out.pi_lo = kwargs["lo"]
            out.pi_is_coded = False
        except KeyError:
            out.pi_lo = out.min()

        try:
            out.pi_hi = kwargs["hi"]
            out.pi_is_coded = False
        except KeyError:
            out.pi_hi = out.max()

        try:
            out.pi_range = kwargs["range"]
            out.pi_is_coded = False
        except KeyError:
            out.pi_range = (out.pi_lo, out.pi_hi)

        try:
            _ = (e for e in out.pi_range)
        except TypeError as err:
            raise TypeError("The `range` input must be an iterable, with 2 values.") from err
        if len(out.pi_range) != 2:
            raise ValueError(
                f"The `range` variable must be a tuple with 2 values; "
                f"got {len(out.pi_range)} value(s)."
            )
        out.pi_range = tuple(out.pi_range)

        try:
            out.pi_center = kwargs["center"]
            out.pi_is_coded = False
        except KeyError:
            out.pi_center = np.mean(out.pi_range)

        try:
            out.pi_units = kwargs["units"]
            out.pi_is_coded = False
        except KeyError:
            out.pi_units = ""

        # Finally, the user might have over-ridden the coding flag:
        if override_coded is not None:
            out.pi_is_coded = override_coded

    elif "levels" in kwargs:
        levels = kwargs.get("levels")
        if not isinstance(levels, Iterable):
            raise TypeError("Levels must be list or tuple of the unique level names.")
        levels_list = list(levels)
        raw_values: list = []
        for arg in args:
            if isinstance(arg, str) or not isinstance(arg, Iterable):
                raw_values.append(arg)
            else:
                raw_values.extend(list(arg))
        extras = {v for v in raw_values if not pd.isna(v)} - set(levels_list)
        if extras:
            raise ValueError(
                f"All values must be present in `levels`. "
                f"Found value(s) not in levels: {sorted(extras, key=str)}."
            )
        out.pi_levels = {out.pi_name: levels_list}
    else:
        # np.sort handles both ndarray (numeric columns) and pandas
        # extension arrays (e.g. StringArray for categorical columns).
        levels = np.sort(out.unique())
        out.pi_levels = {out.pi_name: levels.tolist()}  # for use with Patsy

    units = kwargs.get("units", "")
    if units and not (out.pi_is_coded):
        out.name = f"{out.name} [{units}]"
    if out.pi_is_coded:
        out.name = f"{out.name} [coded]"

    return out


def expand_grid(**kwargs: Column) -> list[Column]:
    """Create the expanded grid here."""
    n_col = len(kwargs)
    itrs = [v.values for v in kwargs.values()]
    product = list(itertools.product(*itrs))
    vals = np.fliplr(np.array(product).reshape(len(product), n_col))
    out = []
    for name, values in zip(kwargs.keys(), np.split(vals, n_col, axis=1), strict=False):
        out.append(c(values, name=name))

    return out


def supplement(x: Column, **kwargs: object) -> Column:
    """Supplement an existing column with additional metadata (name, units, lo, hi, etc.)."""
    return c(x.values, **kwargs)
    # (A, name = 'Feed rate', units='g/min', lo = 5, high = 8.0)
    # B = supplement(B, name = 'Initial inoculate amount', units = 'g', lo = 300,
    # hi = 400)
    # C = supplement(C, name = 'Feed substrate concentration', units = 'g/L',
    # lo = 40, hi = 60)
    # D = supplement(D, name = 'Dissolved oxygen set-point', units = 'mg/L',
    # lo = 4, hi = 5)


def gather(*args: Column, title: str | None = None, **kwargs: Column | list) -> Expt:
    """
    Gathers the named inputs together as columns for a data frame.

    Removes any rows that have ANY missing values. If even 1 value in a row
    is missing, then that row is removed.

    Usage
    -----

    expt = gather(A=A, B=B, y=y, title='My experiment in factors A and B')

    A multi-column input (a ``pandas.DataFrame``, e.g. a categorical factor
    expanded into several indicator columns) is gathered column by column.

    """
    out = Expt(data=None, index=None, columns=None, dtype=None)
    out.pi_source = defaultdict(str)
    out.pi_units = defaultdict(str)

    # Every input is merged positionally (row i with row i), so they must all
    # contribute the same number of rows.
    lengths = {len(value) for value in kwargs.values()}
    if len(lengths) > 1:
        msg = f"All inputs to gather() must have the same length; got lengths {sorted(lengths)}."
        raise ValueError(msg)

    for key, value in kwargs.items():
        if isinstance(value, list):
            out[key] = value
        elif isinstance(value, pd.DataFrame):
            # A block of two or more columns: gather each column separately.
            # A single-column frame keeps the original key as its name.
            for col_name in value.columns:
                sub_key = str(key) if value.shape[1] == 1 else f"{key}_{col_name}"
                out[sub_key] = value[col_name].to_numpy()
                out.pi_source[sub_key] = col_name
                out.pi_units[sub_key] = getattr(value, "pi_units", "")
        elif isinstance(value, pd.Series):
            out[key] = value.values
            out.pi_source[key] = value.name
            out.pi_units[key] = value.pi_units if hasattr(value, "pi_units") else ""

    # Drop any missing values:
    out = out.dropna(axis=0, how="any")

    # Set the title, if one was provided
    out.pi_title = title
    return out
