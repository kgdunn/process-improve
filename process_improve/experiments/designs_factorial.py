# (c) Kevin Dunn, 2010-2026. MIT License.

"""Various factorial designs."""

from process_improve.config import settings

from .structures import c, create_names, expand_grid


def full_factorial(nfactors: int, names: list | None = None) -> list:
    """Create a full factorial (2^k) design for the case when there are `nfactors` [integer] number of factors.

    The optional list of `names` can be provided. The entries in the list
    should be strings. If not provided, the names will be created.

    Raises
    ------
    ValueError
        If ``nfactors < 1`` (the empty design is undefined) or
        ``nfactors > settings.max_factors_combinatorial`` (SEC-19 #268:
        a request for 2**40 rows is a memory-exhaustion attack, not a
        legitimate design).
    """
    nfactors = int(nfactors)
    if nfactors < 1:
        raise ValueError(f"nfactors must be >= 1; got {nfactors}.")
    cap = settings.max_factors_combinatorial
    if nfactors > cap:
        raise ValueError(
            f"nfactors={nfactors} exceeds the SEC-19 combinatorial cap of {cap}; "
            f"a full factorial would require 2**{nfactors} rows. "
            "Increase settings.max_factors_combinatorial if intentional."
        )
    if names is None:
        names = create_names(nfactors)

    # Expand the full factorial out into variables
    return expand_grid(**dict(zip(names, [c(-1, +1)] * len(names), strict=False)))
