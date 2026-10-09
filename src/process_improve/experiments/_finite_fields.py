# (c) Kevin Dunn, 2010-2026. MIT License.

"""Conference and Hadamard matrices built exactly from finite fields.

Definitive screening designs need a conference matrix ``C`` (zero diagonal, +/-1
elsewhere, ``C'C = (m - 1) I``); Plackett-Burman and supersaturated designs need a
Hadamard matrix ``H`` (+/-1 entries, ``H'H = n I``). Both come from the same few
constructions, collected here so that every design family uses matrices that are
checked against their defining property before use:

- **Paley.** For an odd prime power ``q``, the quadratic character of GF(q) gives
  the Jacobsthal matrix ``Q[a, b] = chi(b - a)`` and a conference matrix of order
  ``q + 1``: antisymmetric when ``q = 3 (mod 4)`` (type I), symmetric when
  ``q = 1 (mod 4)`` (type II). Prime powers such as 9, 25, 27 and 49 need the
  arithmetic of GF(p^n), built here from an irreducible polynomial.
- **Doubling an antisymmetric conference matrix.** If ``C`` is antisymmetric of
  order ``n``, then ``[[C, C + I], [C - I, -C]]`` is antisymmetric of order ``2n``
  (its Gram matrix is ``(2n - 1) I``), which reaches orders such as 16, 40 and 56.
- **Hadamard matrices.** ``I + C`` for an antisymmetric conference matrix (Paley I
  and its doublings), ``[[C + I, C - I], [C - I, -C - I]]`` for a symmetric one
  (Paley II), and Sylvester doubling ``[[H, H], [H, -H]]``.

No conference matrix exists at orders 22, 34, 58, ... (Belevitch: ``m - 1`` must be
a sum of two squares when ``m = 2 (mod 4)``), and a few other orders need
constructions not implemented here (36, 46, 52, ...). Callers step up to the next
order that can be built.

References
----------
Paley, R. E. A. C. (1933). On orthogonal matrices. *Journal of Mathematics and
Physics*, 12, 311-320.

Belevitch, V. (1950). Theorem of 2n-terminal networks with application to
conference telephony. *Electrical Communication*, 27, 231-244.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np

#: Largest matrix order searched when stepping up to a buildable order.
MAX_ORDER = 200


def prime_power(q: int) -> tuple[int, int] | None:
    """Return ``(p, n)`` with ``p`` prime and ``p ** n == q``, or ``None`` if ``q`` is not a prime power."""
    if q < 2:
        return None
    p = next(d for d in range(2, q + 1) if q % d == 0)
    n, rest = 0, q
    while rest % p == 0:
        rest //= p
        n += 1
    return (p, n) if rest == 1 else None


def _irreducible_polynomial(p: int, n: int) -> tuple[int, ...]:
    """First monic irreducible polynomial of degree ``n`` over GF(p), coefficients in ascending order."""
    for low in itertools.product(range(p), repeat=n):
        candidate = (*low, 1)
        if candidate[0] == 0:
            continue  # divisible by x
        # Irreducible iff it has no monic factor of degree 1 .. n // 2.
        if not any(
            not any(_poly_mod(candidate, (*divisor, 1), p))
            for degree in range(1, n // 2 + 1)
            for divisor in itertools.product(range(p), repeat=degree)
        ):
            return candidate
    raise ValueError(f"No irreducible polynomial of degree {n} over GF({p}).")  # unreachable: one always exists


def _poly_mod(dividend: tuple[int, ...], divisor: tuple[int, ...], p: int) -> list[int]:
    """Remainder of ``dividend`` by the monic ``divisor`` over GF(p), padded to ``len(divisor) - 1`` terms."""
    rest = [c % p for c in dividend]
    degree = len(divisor) - 1
    for i in range(len(rest) - 1, degree - 1, -1):
        if coefficient := rest[i]:
            for j in range(degree + 1):
                rest[i - degree + j] = (rest[i - degree + j] - coefficient * divisor[j]) % p
    rest = rest[:degree]
    return rest + [0] * (degree - len(rest))


@cache
def _field_tables(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Multiplication and subtraction tables of GF(q), elements labelled ``0 .. q-1`` by their base-p digits."""
    factors = prime_power(q)
    if factors is None:
        raise ValueError(f"GF({q}) does not exist: {q} is not a prime power.")
    p, n = factors
    if n == 1:
        values = np.arange(q)
        return np.outer(values, values) % q, (values[:, None] - values[None, :]) % q
    digits = np.array([[(e // p**i) % p for i in range(n)] for e in range(q)])
    weights = p ** np.arange(n)
    subtract = ((digits[:, None, :] - digits[None, :, :]) % p) @ weights
    modulus = _irreducible_polynomial(p, n)
    multiply = np.zeros((q, q), dtype=int)
    for a, b in itertools.combinations_with_replacement(range(q), 2):
        product = np.convolve(digits[a], digits[b]) % p
        multiply[a, b] = multiply[b, a] = int(np.dot(_poly_mod(tuple(product), modulus, p), weights))
    return multiply, subtract


def quadratic_character(q: int) -> np.ndarray:
    """Return the quadratic character of GF(q) for odd ``q``: 0 at zero, +1 on non-zero squares, -1 otherwise."""
    multiply, _ = _field_tables(q)
    character = np.full(q, -1, dtype=int)
    character[0] = 0
    character[np.unique(np.diag(multiply)[1:])] = 1
    return character


def paley_conference_matrix(q: int) -> np.ndarray:
    """Paley conference matrix of order ``q + 1`` for an odd prime power ``q``.

    The first row is ``[0, 1, ..., 1]``; the first column is ``+1`` below the diagonal
    for ``q = 1 (mod 4)`` (symmetric) and ``-1`` for ``q = 3 (mod 4)``
    (antisymmetric). The rest is the Jacobsthal matrix ``Q[a, b] = chi(b - a)``.
    """
    if q < 3 or q % 2 == 0 or prime_power(q) is None:
        raise ValueError(f"Paley's construction needs an odd prime power; got q={q}.")
    _, subtract = _field_tables(q)
    matrix = np.zeros((q + 1, q + 1), dtype=int)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 1 if q % 4 == 1 else -1
    matrix[1:, 1:] = quadratic_character(q)[subtract.T]
    return matrix


def is_conference_matrix(matrix: np.ndarray) -> bool:
    """Whether ``matrix`` is square with a zero diagonal, +/-1 elsewhere, and ``C'C = (m - 1) I``."""
    m = matrix.shape[0]
    off_diagonal = matrix[~np.eye(m, dtype=bool)]
    return (
        matrix.shape == (m, m)
        and not np.diag(matrix).any()
        and bool(np.all(np.abs(off_diagonal) == 1))
        and np.array_equal(matrix.T @ matrix, (m - 1) * np.eye(m, dtype=matrix.dtype))
    )


def is_hadamard_matrix(matrix: np.ndarray) -> bool:
    """Whether ``matrix`` is square with +/-1 entries and ``H'H = n I``."""
    n = matrix.shape[0]
    return (
        matrix.shape == (n, n)
        and bool(np.all(np.abs(matrix) == 1))
        and np.array_equal(matrix.T @ matrix, n * np.eye(n, dtype=matrix.dtype))
    )


@cache
def _conference(m: int) -> tuple[np.ndarray, str] | None:
    """Build a conference matrix of order ``m``; return it with its construction name, or ``None``."""
    if m < 2 or m % 2:
        return None
    q = m - 1
    if q >= 3 and prime_power(q) is not None:
        return paley_conference_matrix(q), f"paley_q={q}"
    half = m // 2
    if half % 2 == 0 and (inner := _conference(half)) is not None and np.array_equal(inner[0].T, -inner[0]):
        c, eye = inner[0], np.eye(half, dtype=int)
        return np.block([[c, c + eye], [c - eye, -c]]), f"doubled({inner[1]})"
    return None


def conference_matrix(m: int) -> tuple[np.ndarray, str]:
    """Return a verified conference matrix of order ``m`` and the name of its construction.

    Raises
    ------
    ValueError
        If ``m`` is odd or no construction here reaches order ``m``. An approximate
        matrix is never returned.
    """
    found = _conference(m)
    if found is None:
        raise ValueError(f"No conference matrix of order {m} can be built here.")
    matrix, construction = found
    if not is_conference_matrix(matrix):  # a construction bug must never reach a design
        raise RuntimeError(f"The {construction} construction did not give a conference matrix of order {m}.")
    return matrix.copy(), construction


def conference_order_at_least(m: int) -> int:
    """Smallest even order ``>= m`` at which :func:`conference_matrix` succeeds."""
    for order in range(m + m % 2, MAX_ORDER + 1, 2):
        if _conference(order) is not None:
            return order
    raise ValueError(f"No conference matrix of order between {m} and {MAX_ORDER} can be built here.")


@cache
def _hadamard(n: int) -> tuple[np.ndarray, str] | None:
    """Build a Hadamard matrix of order ``n``; return it with its construction name, or ``None``."""
    if n in (1, 2):
        return (np.array([[1]]), "trivial") if n == 1 else (np.array([[1, 1], [1, -1]]), "sylvester")
    if n % 4:
        return None
    conference = _conference(n)
    if conference is not None and np.array_equal(conference[0].T, -conference[0]):
        return np.eye(n, dtype=int) + conference[0], f"I+C({conference[1]})"
    half = n // 2
    q = half - 1
    if q >= 5 and q % 4 == 1 and prime_power(q) is not None:
        c, eye = paley_conference_matrix(q), np.eye(half, dtype=int)
        return np.block([[c + eye, c - eye], [c - eye, -c - eye]]), f"paley_ii_q={q}"
    if (inner := _hadamard(half)) is not None:
        h = inner[0]
        return np.block([[h, h], [h, -h]]), f"sylvester({inner[1]})"
    return None


def hadamard_matrix(n: int) -> tuple[np.ndarray, str] | None:
    """Return a verified, normalised Hadamard matrix of order ``n`` (first column all +1), or ``None``.

    The second element names the construction.
    """
    found = _hadamard(n)
    if found is None:
        return None
    matrix, construction = found
    if not is_hadamard_matrix(matrix):  # a construction bug must never reach a design
        raise RuntimeError(f"The {construction} construction did not give a Hadamard matrix of order {n}.")
    return matrix * matrix[:, [0]], construction
