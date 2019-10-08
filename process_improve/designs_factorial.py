"""
Various factorial designs
"""
from typing import Optional
import itertools


try:
    from .structures import create_names
except ImportError:
    from structures import create_names


def full_factorial(nfactors: int, names:Optional[list] = None):
    """
    Creates a full factorial (2^k) design for the case when there are
    `nfactors` [integer] number of factors.

    The optional list of `names` can be provided. The entries in the list
    should be strings. If not provided, the names will be created.
    """
    nfactors = int(nfactors)
    if names is None:
        names = create_names(nfactors)



    #n_col = len(kwargs)
    #itrs = [v.values for v in kwargs.values()]
    #product = list(itertools.product(*itrs))
    #vals = np.fliplr(np.array(product).reshape(len(product), n_col))
    #out = []
    #for name, values in zip(kwargs.keys(), np.split(vals, n_col, axis=1)):
        #out.append(c(values, name=name))

    #return out

    #A = B = C = D = c(-1, +1)
    #A, B, C, D = expand_grid(A=A, B=B, C=C, D=D)




