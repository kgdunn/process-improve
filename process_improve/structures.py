# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.

import itertools
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import pandas as pd


class Column(pd.Series):
    """
    Creates a column. Can be used as a factor, or a response vector.
    """

    # https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    # Temporary properties
    _internal_names = pd.DataFrame._internal_names + ["not_used_for_now"]
    _internal_names_set = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata = [
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

    @property
    def _constructor(self):
        return Column

    def to_coded(self, center=None, range=None):
        """
        Converts the column vector to coded units.
        """
        out = self.copy(deep=True)
        if self.pi_is_coded:
            return out

        x_center = center or self.pi_center
        x_range = range or self.pi_range

        # Simply override the values and the `pi_is_coded` flag, but all the
        # rest remains as is.
        out.iloc[:] = (self.values - x_center) / (0.5 * np.diff(x_range)[0])
        out.pi_is_coded = True
        if out.pi_name:
            out.name = f"{out.pi_name} [coded]"

        return out

    def to_realworld(self, center=None, range=None):
        """
        Converts the column vector to real-world units.
        """
        out = self.copy(deep=True)
        if not self.pi_is_coded:
            return out

        x_center = center or self.pi_center
        x_range = range or self.pi_range
        # Simply override the values and the `pi_is_coded` flag, but all the
        # rest remains as is.
        out.iloc[:] = self.values * (0.5 * np.diff(x_range)[0]) + x_center
        out.pi_is_coded = False
        if out.pi_name and out.pi_units:
            out.name = f"{out.pi_name} [{out.pi_units}]"

        return out

    def copy(self, deep=True):
        out = pd.Series.copy(self, deep=deep)
        out.name = self.name
        return out

    def extend(self, values):
        """
        Extends the column with the list of new values.
        """
        assert isinstance(values, list), "The 'values' must be in a list [...]"
        prior_n = self.index[-1]
        index = list(range(prior_n + 1, prior_n + len(values) + 1))
        new = pd.Series(data=values, index=index)
        intermediate = self.copy(deep=True)
        intermediate = intermediate.append(new)

        # Carry the meta data over. For some reason the `pd.Series.append`
        # function does not do this (yet?)
        for key in self._metadata:
            setattr(intermediate, key, getattr(self, key))
        intermediate.name = self.name
        return intermediate


class Expt(pd.DataFrame):
    """
    Dataframe object with experimental data. Builds on the Pandas dataframe,
    but with some extra attributes.
    """

    # Temporary properties
    _internal_names = pd.DataFrame._internal_names + ["not_used_for_now"]
    _internal_names_set = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata = ["pi_source", "pi_title", "pi_units"]

    @property
    def _constructor(self):
        return Expt

    def __repr__(self):
        title = f"Name: {self.pi_title}"
        dimensions = f"Size: {self.shape[0]} experiments; " f"{self.shape[1]} columns."
        return "\n".join([pd.DataFrame.__repr__(self), title, dimensions])

    def get_title(self):
        return self.pi_title or ""


def create_names(n: int, letters=True, prefix="X", start_at=1, padded=True):
    """
    Returns default factor names, for a given number of `n` [integer] factors.
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

        out = [
            f'{str(prefix)}{str(i).rjust(longest, "0")}'
            for i in range(start_at, n + start_at)
        ]

    return out


def c(*args, **kwargs) -> Column:  # noqa: C901
    """
    Performs the equivalent of the R function "c(...)", to combine data elements
    into a DataFrame. Converts every entry into a floating point object.

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
    sanitize = []
    numeric = True
    override_coded = kwargs.get("coded", None)

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
        raise IndexError(
            ('Length of "index" must match the ' "number of numeric inputs.")
        )

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
        except TypeError:
            assert False, "The `range` input must be an iterable, with " "2 values."
        assert len(out.pi_range) == 2, (
            "The `range` variable must be a tuple, " "with 2 values."
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

    else:
        if "levels" in kwargs:
            msg = "Levels must be list or tuple of the unique level names."
            # TODO: Check that all entries in the level list are accounted for.
            assert isinstance(kwargs.get("levels"), Iterable), msg
            out.pi_levels = {out.pi_name: list(kwargs.get("levels", []))}
        else:
            levels = out.unique()
            levels.sort()
            out.pi_levels = {out.pi_name: levels.tolist()}  # for use with Patsy

    units = kwargs.get("units", "")
    if units and not (out.pi_is_coded):
        out.name = f"{out.name} [{units}]"
    if out.pi_is_coded:
        out.name = f"{out.name} [coded]"

    return out


def expand_grid(**kwargs):
    """
    Create the expanded grid here.
    """
    n_col = len(kwargs)
    itrs = [v.values for v in kwargs.values()]
    product = list(itertools.product(*itrs))
    vals = np.fliplr(np.array(product).reshape(len(product), n_col))
    out = []
    for name, values in zip(kwargs.keys(), np.split(vals, n_col, axis=1)):
        out.append(c(values, name=name))

    return out


def supplement(x, **kwargs):
    return c(x.values, **kwargs)
    # (A, name = 'Feed rate', units='g/min', lo = 5, high = 8.0)
    # B = supplement(B, name = 'Initial inoculate amount', units = 'g', lo = 300,
    # hi = 400)
    # C = supplement(C, name = 'Feed substrate concentration', units = 'g/L',
    # lo = 40, hi = 60)
    # D = supplement(D, name = 'Dissolved oxygen set-point', units = 'mg/L',
    # lo = 4, hi = 5)


def gather(*args, title=None, **kwargs) -> Expt:
    """
    Gathers the named inputs together as columns for a data frame.

    Removes any rows that have ANY missing values. If even 1 value in a row
    is missing, then that row is removed.

    Usage
    -----

    expt = gather(A=A, B=B, y=y, title='My experiment in factors A and B')

    """
    # TODO : handle the case where the shape of an input >= 2 columns:  category

    out = Expt(data=None, index=None, columns=None, dtype=None)
    out.pi_source = defaultdict(str)
    out.pi_units = defaultdict(str)

    _ = [len(value) for value in kwargs.values()]
    index = []
    for key, value in kwargs.items():
        if isinstance(value, list):
            out[key] = value
        elif isinstance(value, pd.Series):
            out[key] = value.values
            out.pi_source[key] = value.name
            out.pi_units[key] = value.pi_units if hasattr(value, "pi_units") else ""

            if hasattr(value, "pi_index"):
                index.append(value.index)

        elif isinstance(value, pd.DataFrame):
            assert False, "Handle this case still"

    # TODO : check that all indexes are common, to merge. Or use the pandas
    #        functionality of merging series with the same index
    if index:
        out.index = index[0]

    # Drop any missing values:
    out.dropna(axis=0, how="any", inplace=True)

    # Set the title, if one was provided
    out.pi_title = title
    return out
