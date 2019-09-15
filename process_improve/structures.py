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
    A = c( 4, 5,  6,  4,  6, lo=4, hi=6, name = 'A)
    A = c([4, 5,  6,  4,  6], lo=4, hi=6, name = 'A)

    """
    #TODO: handle the case when this is a list, tuple, or iterable
    sanitize = []
    for j in args:
        if isinstance(j, Iterable):
            if isinstance(j, np.ndarray):
                sanitize = j.ravel().tolist()

        else:
            sanitize.append(float(j))

    default_idx = list(range(1, len(sanitize)+1))
    index = kwargs.get('index', default_idx)
    assert len(index) == len(sanitize), ('Length of "index" must match the '
                                         'number of numeric inputs.')

    name = kwargs.get('name', 'Unnamed')
    out = pd.DataFrame(data={name: sanitize}, index=index)
    out._pi_index = True
    return out


class expand(object):
    pass

    def grid(**kwargs):
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
        elif isinstance(value, pd.DataFrame):
            out[key] = value.values.ravel()

            if hasattr(value, '_pi_index'):
                index.append(value.index)

    # TODO : check that all indexes are common, to merge. Or use the pandas
    #        functionality of merging series with the same index

    return out