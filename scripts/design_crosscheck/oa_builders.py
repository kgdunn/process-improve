"""Build a conference design and an orthogonal array with OApackage: depth-first, first extension first."""

import itertools
import math
import time

import numpy as np
import oapackage


def conference_design(n_rows: int, k: int, deadline: float) -> np.ndarray:
    """Return a conference design with n_rows rows and k columns: OApackage's extensions, first found first."""
    ctype = oapackage.conference_t(n_rows, k, 0)

    def dfs(design):  # noqa: ANN202 - an OApackage array, or None
        if design.n_columns >= k:
            return design
        if time.monotonic() > deadline:
            raise TimeoutError
        for ext in oapackage.extend_conference([design], ctype, 0):
            found = dfs(ext)
            if found is not None:
                return found
        return None

    found = dfs(ctype.create_root())
    if found is None:
        raise ValueError(f"no conference design with {n_rows} rows and {k} columns")
    return np.array(found)


def _extend(arr, ad, n_columns: int, deadline: float):  # noqa: ANN202 - an OApackage array, or None
    """Extend an array to n_columns columns, depth-first; None when no extension reaches them."""
    if arr.n_columns >= n_columns:
        return arr
    if time.monotonic() > deadline:
        raise TimeoutError
    for ext in oapackage.extend_array(arr, ad):
        found = _extend(ext, ad, n_columns, deadline)
        if found is not None:
            return found
    return None


def orthogonal_array(levels: list[int], deadline: float, max_runs: int = 200) -> np.ndarray:
    """Return the smallest strength-2 orthogonal array with these level counts that OApackage finds."""
    order = sorted(range(len(levels)), key=lambda j: -levels[j])  # OApackage wants levels in non-increasing order
    s = [levels[j] for j in order]
    step = math.lcm(s[0], *(a * b for a, b in itertools.combinations(s, 2)))  # every pair of columns balanced
    rao = 1 + sum(v - 1 for v in s)  # Rao's bound for strength 2
    n = step * max(1, math.ceil(rao / step))
    while n <= max_runs:
        ad = oapackage.arraydata_t(s, n, 2, len(s))

        found = _extend(ad.create_root(), ad, len(s), deadline)
        if found is not None:
            arr = np.array(found)
            out = np.empty_like(arr)
            out[:, order] = arr
            return out
        n += step
    raise ValueError(f"no orthogonal array for levels {levels} in {max_runs} runs or fewer")
