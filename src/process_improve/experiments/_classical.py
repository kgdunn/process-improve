"""Classical two-level and response-surface arrays, built without optional dependencies.

These functions reproduce, element for element, the arrays of pyDOE3 1.6.2's
``ff2n``, ``fracfact``, ``pbdesign``, ``bbdesign`` and ``ccdesign``, so the full and
fractional factorial, Plackett-Burman, Box-Behnken and central composite designs no
longer need the ``expt`` extra. ``tests/test_classical_arrays.py`` checks the
equivalence against pyDOE3 when it is installed.

pyDOE3 is distributed under the BSD 3-Clause licence. Its code descends from the
Scilab ``scidoe`` toolbox (Copyright (C) 2009-2013 Yann Collette, CEA / Jean-Marc
Martinez, INRIA / Michael Baudin, Maria Christopoulou), converted to Python by
Abraham Lee and maintained as pyDOE2 / pyDOE3.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
from scipy.linalg import hankel, toeplitz

#: First column and row of the cyclic 12-run Plackett-Burman core (pyDOE3's construction).
_PB12_COLUMN = [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1]
_PB12_ROW = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
#: First column and last row of the 20-run Plackett-Burman Hankel core.
_PB20_COLUMN = [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1]
_PB20_ROW = [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1]
#: Centre points pyDOE3's ``bbdesign`` adds by default, indexed by the number of factors.
_BB_DEFAULT_CENTRE = [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]


def ff2n(n_factors: int) -> np.ndarray:
    """Return the 2^k full factorial in coded units, rows in standard (binary) order.

    Parameters
    ----------
    n_factors : int
        Number of two-level factors, ``k``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2**k, k)`` with entries -1 and +1; the last column changes fastest.

    Examples
    --------
    >>> ff2n(2)
    array([[-1., -1.],
           [-1.,  1.],
           [ 1., -1.],
           [ 1.,  1.]])
    """
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n_factors))).reshape(-1, n_factors)


def fracfact(gen: str) -> np.ndarray:
    """Return a two-level fractional factorial from a generator string such as ``"a b c ab -ac"``.

    Single letters are the basic factors and must be the first letters of the alphabet;
    a word is the product of its letters' columns, negated by a leading ``-``.

    Parameters
    ----------
    gen : str
        Space-separated generator words; case is ignored.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2**b, len(words))`` for ``b`` basic factors, columns in the order of ``gen``.

    Raises
    ------
    ValueError
        If the generator has no basic factor, repeats one, or uses a letter that is not a basic factor.

    Examples
    --------
    >>> fracfact("a b ab")
    array([[-1., -1.,  1.],
           [-1.,  1., -1.],
           [ 1., -1., -1.],
           [ 1.,  1.,  1.]])
    """
    tokens = gen.lower().split(" ")
    words = [item for item in re.split(r"\-|\s|\+", gen.lower()) if item]
    if len(words) != len(tokens):
        raise ValueError("Generator does not match the number of factors.")
    basic = [word for word in words if len(word) == 1]
    if not basic:
        raise ValueError("At least one unconfounded main factor is needed.")
    if len(set(basic)) != len(basic):
        raise ValueError("Main factors are confounded with each other.")
    letters = "abcdefghijklmnopqrstuvwxyz"[: len(basic)]
    if "".join(sorted(basic)) != letters:
        raise ValueError(f"Use the letters `{' '.join(letters)}` for the main factors.")
    if len(set(words)) != len(words):
        raise ValueError("Generators are not unique.")
    if not all(set(word) <= set(basic) for word in words):
        raise ValueError("Generators are not valid.")

    base = ff2n(len(basic))
    design = np.column_stack([np.prod(base[:, [ord(c) - ord("a") for c in word]], axis=1) for word in words])
    negative = [i for i, token in enumerate(tokens) if token.startswith("-")]
    design[:, negative] *= -1
    return design


def pbdesign(n_factors: int) -> np.ndarray:
    """Return a Plackett-Burman design for ``n_factors`` factors, as pyDOE3 builds it.

    The run count is the next multiple of 4 above ``n_factors``. Orders of the form
    ``2^e``, ``12 * 2^e`` and ``20 * 2^e`` are built from the 1-, 12- or 20-run core by
    Sylvester doubling; any other order raises ``ValueError``.

    Parameters
    ----------
    n_factors : int
        Number of factors, at least 1.

    Returns
    -------
    numpy.ndarray
        Coded design of shape ``(N, n_factors)``.

    Raises
    ------
    ValueError
        If ``n_factors`` is not positive or the run count has no 1-, 12- or 20-run core.
    """
    if n_factors <= 0:
        raise ValueError("Number of factors must be a positive integer")
    n_runs = 4 * (n_factors // 4 + 1)
    # The core is the first of 1, 12 and 20 that leaves a power of 2 to double up to n_runs.
    core_size = next((s for s in (1, 12, 20) if n_runs % s == 0 and (n_runs // s) & (n_runs // s - 1) == 0), None)
    if core_size is None:
        raise ValueError(f"No Plackett-Burman construction for {n_runs} runs (not 2^e, 12*2^e or 20*2^e).")
    matrix = np.ones((1, 1))
    if core_size == 12:
        matrix = np.vstack([np.ones((1, 12)), np.hstack([np.ones((11, 1)), toeplitz(_PB12_COLUMN, _PB12_ROW)])])
    elif core_size == 20:
        matrix = np.vstack([np.ones((1, 20)), np.hstack([np.ones((19, 1)), hankel(_PB20_COLUMN, _PB20_ROW)])])
    for _ in range(int(np.log2(n_runs // core_size))):
        matrix = np.vstack([np.hstack([matrix, matrix]), np.hstack([matrix, -matrix])])
    return np.flipud(matrix[:, 1 : n_factors + 1]).astype(float)


def bbdesign(n_factors: int, center: int | None = None) -> np.ndarray:
    """Return the Box-Behnken design pyDOE3 builds: a 2^2 factorial in every pair of factors, then centre runs.

    Parameters
    ----------
    n_factors : int
        Number of factors, at least 3.
    center : int or None
        Number of centre runs; ``None`` uses pyDOE3's default for ``n_factors``.

    Returns
    -------
    numpy.ndarray
        Coded design with entries -1, 0 and +1.
    """
    if n_factors < 3:
        raise ValueError("Number of variables must be at least 3")
    square = ff2n(2)
    blocks = []
    for i, j in itertools.combinations(range(n_factors), 2):
        block = np.zeros((4, n_factors))
        block[:, i], block[:, j] = square[:, 0], square[:, 1]
        blocks.append(block)
    if center is None:
        center = _BB_DEFAULT_CENTRE[n_factors] if n_factors <= 16 else n_factors
    return np.vstack([*blocks, np.zeros((center, n_factors))])


def _star(n_factors: int, alpha: float) -> np.ndarray:
    """Axial points: a pair ``-alpha, +alpha`` on each axis, axis by axis."""
    points = np.zeros((2 * n_factors, n_factors))
    for i in range(n_factors):
        points[2 * i : 2 * i + 2, i] = [-1.0, 1.0]
    return points * alpha


def ccdesign(
    n_factors: int, center: tuple[int, int] = (4, 4), alpha: str = "orthogonal", face: str = "circumscribed"
) -> np.ndarray:
    """Return a central composite design laid out as pyDOE3's ``ccdesign`` lays it out.

    Rows are the factorial points, the centre runs of the cube portion, the axial points,
    then the centre runs of the axial portion.

    Parameters
    ----------
    n_factors : int
        Number of factors, at least 2.
    center : tuple[int, int]
        Centre runs added to the cube portion and to the axial portion.
    alpha : {"orthogonal", "o", "rotatable", "r"}
        How the axial distance is chosen for a circumscribed design.
    face : {"circumscribed", "ccc", "inscribed", "cci", "faced", "ccf"}
        Where the axial points sit relative to the cube.

    Returns
    -------
    numpy.ndarray
        Coded design matrix.
    """
    if n_factors < 2:
        raise ValueError('"n" must be an integer greater than 1.')
    alpha, face = alpha.lower(), face.lower()
    if alpha not in ("orthogonal", "o", "rotatable", "r"):
        raise ValueError(f'Invalid value for "alpha": {alpha}')
    if face not in ("circumscribed", "ccc", "inscribed", "cci", "faced", "ccf"):
        raise ValueError(f'Invalid value for "face": {face}')
    if len(center) != 2:
        raise ValueError(f'Invalid number of values for "center" (expected 2, but got {len(center)})')

    n_cube = 2**n_factors
    if alpha in ("orthogonal", "o"):
        distance = (n_factors * (1 + center[1] / (2.0 * n_factors)) / (1 + center[0] / float(n_cube))) ** 0.5
    else:
        distance = n_cube**0.25
    cube = ff2n(n_factors)
    axial = _star(n_factors, distance)
    if face in ("inscribed", "cci"):
        cube, axial = cube / distance, _star(n_factors, 1.0)
    elif face in ("faced", "ccf"):
        axial = _star(n_factors, 1.0)
    return np.vstack([cube, np.zeros((center[0], n_factors)), axial, np.zeros((center[1], n_factors))])
