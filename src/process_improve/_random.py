"""(c) Kevin Dunn, 2010-2026. MIT License.

Shared ``random_state`` resolver used by every public function in the
package that touches an RNG.

The :doc:`reproducibility contract </development/reproducibility>` pins
one rule: every public function accepts ``random_state: int |
np.random.Generator | None`` and resolves it once at the top of the
function via this helper. Callers that pass the same ``int`` twice get
bit-identical draws; callers that pass a ``Generator`` get the
generator they passed (so they own the state).

Reference semantics: scikit-learn's
:func:`sklearn.utils.check_random_state` since 1.4, but returning a
modern ``numpy.random.Generator`` rather than the legacy
``RandomState``.
"""

from __future__ import annotations

import numbers

import numpy as np


def check_random_state(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Resolve ``random_state`` into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    random_state : int, np.random.Generator, or None
        - ``None`` -- a fresh, unseeded ``np.random.default_rng()``.
        - ``int`` (including ``numpy`` integer scalars) --
          ``np.random.default_rng(int(random_state))``. Callers passing
          the same int twice get bit-identical draws.
        - ``np.random.Generator`` -- returned as-is so the caller owns
          and advances its own state.

    Returns
    -------
    np.random.Generator
        A generator suitable for any of NumPy's RNG methods.

    Raises
    ------
    TypeError
        If *random_state* is none of the three accepted forms. In
        particular the legacy :class:`numpy.random.RandomState` is
        rejected; pass ``int(rs.randint(0, 2**31))`` to migrate.

    Examples
    --------
    Same int twice produces the same draws:

    >>> rng1 = check_random_state(0)
    >>> rng2 = check_random_state(0)
    >>> float(rng1.random()) == float(rng2.random())
    True

    A passed generator is returned unchanged:

    >>> g = np.random.default_rng(42)
    >>> check_random_state(g) is g
    True

    ``None`` is the documented opt-out for callers that explicitly want
    fresh entropy:

    >>> rng = check_random_state(None)
    >>> isinstance(rng, np.random.Generator)
    True

    Notes
    -----
    Booleans are rejected. Python treats ``True`` / ``False`` as
    integer subclasses, but ``check_random_state(True)`` is almost
    always a caller mistake (forgetting to pass an actual seed) and
    silently seeding with ``1`` would hide the bug.

    See Also
    --------
    Reproducibility contract: :doc:`/development/reproducibility`.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, numbers.Integral) and not isinstance(random_state, bool):
        return np.random.default_rng(int(random_state))
    raise TypeError(
        f"random_state must be int | np.random.Generator | None, "
        f"got {type(random_state).__name__}."
    )
