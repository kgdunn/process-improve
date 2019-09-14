# (c) Kevin Dunn, 2019. MIT License.

import pandas as pd
def c(*args, **kwargs) -> list:
    """
    Performs the equivalent of the R function "c(...)", to combine data elements
    into a DataFrame. Converts every entry into a floating point object.

    Inputs
    ------

    index: a list of names for the entries in `args`

    name:  a name for the column

    Usage
    -----

    A = c(-1, 0, +1, -1, +1)
    A = c(-1, 0, +1, -1, +1, index=['lo', 'cp', 'hi', 'lo', 'hi'])
    """
    sanitize = [float(j) for j in args]
    default_idx = list(range(1, len(sanitize)+1))
    index = kwargs.get('names', default_idx)
    assert len(index) == len(sanitize), ('Length of "names" must match the '
                                         'number of numeric inputs.')

    out = pd.DataFrame(data={'Unnamed': sanitize}, index=index)
    out._pi_index = True
    return out
