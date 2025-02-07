# (c) Kevin Dunn, 2010-2025. MIT License.

"""Various factorial designs"""

from typing import Optional

from .structures import c, create_names, expand_grid


def full_factorial(nfactors: int, names: Optional[list] = None):
    """Create a full factorial (2^k) design for the case when there are `nfactors` [integer] number of factors.

    The optional list of `names` can be provided. The entries in the list
    should be strings. If not provided, the names will be created.
    """
    nfactors = int(nfactors)
    if names is None:
        names = create_names(nfactors)

    # Expand the full factorial out into variables
    return expand_grid(**dict(zip(names, [c(-1, +1)] * len(names))))
