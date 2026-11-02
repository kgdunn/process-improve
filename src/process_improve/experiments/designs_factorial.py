# (c) Kevin Dunn, 2010-2026. MIT License.

"""Various factorial designs."""

from process_improve.config import settings

from .structures import c, create_names, expand_grid


def full_factorial(nfactors: int, names: list | None = None) -> list:
    """Create a two-level full factorial (2^k) design in coded units.

    Parameters
    ----------
    nfactors : int
        Number of factors, k. The design has ``2**k`` runs.
    names : list of str or None
        One name per factor. If ``None``, names are created (``A``, ``B``, ...).

    Returns
    -------
    list[Column]
        One ``Column`` per factor, each of length ``2**k``, holding the coded
        settings -1 and +1 in standard (Yates) order.

    Raises
    ------
    ValueError
        If ``nfactors < 1`` (the empty design is undefined),
        ``nfactors > settings.max_factors_combinatorial`` (SEC-19 #268:
        a request for 2**40 rows is a memory-exhaustion attack, not a
        legitimate design), or *names* does not hold ``nfactors`` distinct names.

    Examples
    --------
    >>> from process_improve.experiments.designs_factorial import full_factorial
    >>> columns = full_factorial(2, names=["Temp", "Pressure"])
    >>> len(columns), len(columns[0])
    (2, 4)
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
    names = list(names)
    if len(names) != nfactors or len(set(names)) != len(names):
        raise ValueError(f"names must hold nfactors={nfactors} distinct names; got {names}.")

    # Expand the full factorial out into variables
    return expand_grid(**{name: c(-1, +1) for name in names})
