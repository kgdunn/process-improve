# (c) Kevin Dunn, 2019. MIT License.
import itertools
from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import pandas as pd

class Column(pd.Series):
    """
    Creates a column. Can be used as a factor, or a response vector.
    """
    # Temporary properties
    _internal_names = pd.DataFrame._internal_names + ['not_used_for_now']
    _internal_names_set = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata = ['pi_index',   # might be used later if the user provides their own index
                 'pi_numeric', # if numeric indicator
                 'pi_lo',      # if numeric: low level (-1)
                 'pi_hi',      # if numeric: high level (+1)
                 'pi_range',   # if numeric: range: distance from low to high
                 'pi_center',  # if numeric: midway between low and high (0)
                ]

    @property
    def _constructor(self):
        return Column


class Expt(pd.DataFrame):
    """
    Dataframe object with experimental data. Builds on the Pandas dataframe,
    but with some extra attributes.
    """
    # Temporary properties
    _internal_names = pd.DataFrame._internal_names + ['not_used_for_now']
    _internal_names_set = set(_internal_names)

    # Properties which survive subsetting, etc
    _metadata = ['pi_source', 'pi_title']

    @property
    def _constructor(self):
        return Expt

    def __repr__(self):
        title = f'Name: {self.pi_title}'
        dimensions = (f'Size: {self.shape[0]} experiments; '
                      f'{self.shape[1]} columns.')
        return '\n'.join([pd.DataFrame.__repr__(self), title, dimensions])


    def get_title(self):
        return self.pi_title or ''


def create_names(n: int, letters=True, prefix='X', start_at=1, padded=True):
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
        out = [chr(65+i) for i in range(n)]
        if 'I' in out:
            out.remove('I')
            out.append(chr(65+n))

    else:
        longest = 0
        if padded:
            longest = len(str(start_at + n - 1))

        out = [f'{str(prefix)}{str(i).rjust(longest, "0")}' for i in \
                                        range(start_at, n + start_at)]

    return out



def c(*args, **kwargs) -> Column:
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


    B = c(0, 1, 0, 1, 0, 2, levels =(0, 1, 2))

    """
    sanitize = []
    numeric = True
    if 'levels' in kwargs:
        numeric = False

    for j in args:
        if isinstance(j, Iterable):
            if isinstance(j, np.ndarray):
                sanitize = j.ravel().tolist()

            if isinstance(j, pd.Series):
                sanitize = j.copy()
                if 'index' not in kwargs:
                    kwargs['index'] = sanitize.index

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
    default_idx = list(range(1, len(sanitize)+1))
    index = kwargs.get('index', default_idx)
    if len(index) != len(sanitize):
        raise IndexError(('Length of "index" must match the '
                          'number of numeric inputs.'))

    name = kwargs.get('name', 'Unnamed')
    units = kwargs.get('units', '')
    if units:
        name = f'{name} [{units}]'

    out = Column(data=sanitize, index=index, name=name)
    #out = pd.Series(data=sanitize, index=index, name=name)

    # Use sensible defaults, if not provided
    out.pi_index = True
    out.pi_lo = None
    out.pi_hi = None
    out.pi_range = None
    out.pi_center = None
    out.pi_numeric = numeric
    if numeric:
        out.pi_lo = kwargs.get('lo', out.min())
        out.pi_hi = kwargs.get('hi', out.max())
        out.pi_range = kwargs.get('range', (out.pi_lo, out.pi_hi))
        out.pi_center = kwargs.get('center', np.mean(out.pi_range))
    else:
        if 'levels' in kwargs:
            msg = "Levels must be list or tuple of the unique level names."
            # TODO: Check that all entries in the level list are accounted for.
            assert isinstance(kwargs.get('levels'), Iterable), msg
            out.pi_levels = {name: list(kwargs.get('levels'))}
        else:
            levels = out.unique()
            levels.sort()
            out.pi_levels = {name: levels.tolist()} # for use with Patsy

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
    #(A, name = 'Feed rate', units='g/min', lo = 5, high = 8.0)
        #B = supplement(B, name = 'Initial inoculate amount', units = 'g', lo = 300,
                       #hi = 400)
        #C = supplement(C, name = 'Feed substrate concentration', units = 'g/L',
                       #lo = 40, hi = 60)
        #D = supplement(D, name = 'Dissolved oxygen set-point', units = 'mg/L',
                       #lo = 4, hi = 5)

def gather(*args, **kwargs):
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

    lens = [len(value) for value in kwargs.values()]
    avg_count = pd.np.median(lens)
    index = []
    for key, value in kwargs.items():
        #assert len(value) == avg_count, (f"Column {key} must have length "
        #                                 f"{avg_count}.")
        if isinstance(value, list):
            out[key] = value
        elif isinstance(value, pd.Series):
            out[key] = value.values
            out.pi_source[key] = value.name

            if hasattr(value, 'pi_index'):
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
    out.pi_title = kwargs.get('title', None)
    return out
