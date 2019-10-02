# (c) Kevin Dunn, 2019. MIT License.
import itertools
from collections.abc import Iterable
import numpy as np
import pandas as pd

class Column(pd.Series):
    """
    Creates a column. Can be used as a factor, or a response vector.
    """
    def __init__(self):
        pass

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

            try:
                [float(j) for j in sanitize]
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
    out = pd.Series(data=sanitize, index=index, name=name)
    out._pi_index = True

    # Use sensible defaults, if not provided
    out._pi_lo = None
    out._pi_hi = None
    out._pi_range = None
    out._pi_center = None
    out._pi_numeric = numeric
    if numeric:
        out._pi_lo = kwargs.get('lo', out.min())
        out._pi_hi = kwargs.get('hi', out.max())
        out._pi_range = kwargs.get('range', (out._pi_lo, out._pi_hi))
        out._pi_center = kwargs.get('center', np.mean(out._pi_range))
    else:
        if 'levels' in kwargs:
            msg = "Levels must be list or tuple of the unique level names."
            # TODO: Check that all entries in the level list are accounted for.
            assert isinstance(kwargs.get('levels'), Iterable), msg
            out._pi_levels = {name: list(kwargs.get('levels'))}
        else:
            levels = out.unique()
            levels.sort()
            out._pi_levels = {name: levels.tolist()} # for use with Patsy

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



def gather(*args, **kwargs):
    """
    Gathers the named inputs together as columns for a data frame.

    """
    # TODO : handle the case where the shape of an input >= 2 columns

    out = pd.DataFrame(data=None, index=None, columns=None, dtype=None)
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

            if hasattr(value, '_pi_index'):
                index.append(value.index)

        elif isinstance(value, pd.DataFrame):
            assert False, "Handle this case still"

    # TODO : check that all indexes are common, to merge. Or use the pandas
    #        functionality of merging series with the same index


    if kwargs.get('title', False):
        out._pi_title = kwargs.get('title')
    return out