# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

"""Teaching simulators used in the *Process Improvement using Data* course.

Each function here is a small, deliberately opaque process that a student
drives with a designed experiment. They are ports of the simulators in the
companion R package (``pid``): :func:`popcorn`, :func:`grocery` and
:func:`manufacture`.

All three add random noise to the response, so that repeated runs at the same
settings do not give an identical answer. Pass ``random_state`` to make a run
reproducible; leave it at ``None`` (the default) for fresh noise on every call,
which is what the classroom exercise expects.

Every simulator refuses vector input on purpose: the point of the exercise is
sequential experimentation, one run at a time, with the fewest number of runs.
"""

from __future__ import annotations

import numpy as np

from process_improve._random import check_random_state

# Below this cooking time nothing pops, so the simulator refuses to answer.
_POPCORN_MINIMUM_TIME = 77.0


def popcorn(
    t: float = 120,
    T: float | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
) -> int:
    """
    Simulate stovetop popcorn cooking.

    Returns the number of popped kernels after cooking a fixed set of kernels,
    at the same heat setting on the stove, for `t` seconds. There is only one
    factor: the cooking time.

    Parameters
    ----------
    t : float, default 120
        Number of seconds the pot is left on the stove. Cooking times less
        than 77 seconds are not supported: nothing has popped yet. A vector
        (list or array) of time values is not permitted, since the goal is to
        perform sequential experimentation to determine the optimum time, with
        the fewest number of function calls.
    T : float or None, default None
        Alias for `t`, matching the ``popcorn(T=...)`` spelling used in the R
        package. When given, it overrides `t`.
    random_state : int, np.random.Generator, or None, default None
        Seed or generator for the noise term. ``None`` (the default) draws
        fresh noise on every call, which is the intended behaviour for the
        classroom exercise; pass an ``int`` to make a call reproducible.

    Returns
    -------
    int
        The number of popped kernels, with random noise added for realism.
        Never negative.

    Raises
    ------
    ValueError
        If `t` is not a single finite number, or is below 77 seconds.

    Examples
    --------
    >>> popcorn(t=135, random_state=13)  # doctest: +SKIP
    94

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2026,
    https://learnche.org/pid

    Also see
    --------
    grocery
    manufacture
    """
    time_taken = t if T is None else T

    if np.ndim(time_taken) > 0:
        raise ValueError("Cooking popcorn batches in parallel is (intentionally) not allowed.")
    if not np.isfinite(time_taken):
        raise ValueError("Please provide finite numeric values as inputs.")
    if time_taken < _POPCORN_MINIMUM_TIME:
        raise ValueError("No popcorn was made: please cook for a longer time.")

    rng = check_random_state(random_state)
    coded = (float(time_taken) - 135.0) / 15.0
    y = coded * 15 - 2.4 * coded * coded + 93 + rng.uniform(0.0, 1.0) * 6 - 3.0
    return max(0, round(y))


def grocery(
    p: float = 3.46,
    h: float = 150,
    P: float | None = None,
    H: float | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
) -> int:
    """
    Simulate grocery store profits for a single product.

    The hourly profit made when selling the product at price `p` and the product
    is displayed at height `h` [cm up from the ground] on the shelf.

    Simulates a grocery store profit function where there are 2 factors:
    * `p` = selling price of the product, measured in dollars and cents
    * `h` = height of the product on the shelf, measured in centimeters above
          the ground.

    Typical values are p = $3.50 and h = 150cm
    The outcome is: profit made per hour [dollars/hour], with random noise
                    added, for realism.

    Parameters
    ----------
    p : float, default 3.46
        Selling price of the product [dollars].
    h : float, default 150
        Height of the product on the shelf [cm above the ground].
    P : float or None, default None
        Alias for `p`, matching the ``grocery(P=...)`` spelling used in the R
        package. When given, it overrides `p`.
    H : float or None, default None
        Alias for `h`. When given, it overrides `h`.
    random_state : int, np.random.Generator, or None, default None
        Seed or generator for the noise term. ``None`` (the default) draws
        fresh noise on every call; pass an ``int`` to make a call reproducible.

    Returns
    -------
    int
        Profit made per hour [dollars/hour], with random noise added.

    Raises
    ------
    ValueError
        If either input is a vector, is not finite, or is negative.

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2026,
    https://learnche.org/pid

    Also see
    --------
    popcorn
    manufacture
    """
    if P is None:
        P = p
    if H is None:
        H = h

    if np.ndim(P) > 0 or np.ndim(H) > 0:
        raise ValueError("Running the grocery store experiments in parallel is (intentionally) not allowed.")

    if not np.isfinite(P) or not np.isfinite(H):
        raise ValueError("All function inputs must be finite numbers.")
    if P < 0:
        raise ValueError("Please provide a positive sales price, P.")
    if H < 0:
        raise ValueError("The height of the shelving, H, must be a positive value.")

    rng = check_random_state(random_state)
    a_coded = (P - 3.2) / 0.2
    b_coded = (H - 50) / 100
    return round(
        (18 * a_coded + 12 * b_coded - 7 * a_coded * a_coded - 6.0 * b_coded * b_coded - 8.5 * a_coded * b_coded + 60)
        * 10.0
        + rng.normal(0, 1) * 2
    )


def manufacture(
    p: float = 0.75,
    t: float = 325,
    P: float | None = None,
    T: float | None = None,
    *,
    random_state: int | np.random.Generator | None = None,
) -> int:
    """
    Simulate the hourly profit of a manufacturing facility.

    Two factors affect the outcome:
    * `p` = selling price of the product, measured in dollars and cents
    * `t` = throughput (production rate) of the process, in parts per hour

    Typical values are p = $0.75 and t = 325 parts per hour. The outcome is the
    profit made per hour [dollars/hour], with random noise added for realism.
    The aim of the exercise is to maximize that profit.

    Parameters
    ----------
    p : float, default 0.75
        Selling price of the product [dollars].
    t : float, default 325
        Throughput (production rate) of the process [parts per hour].
    P : float or None, default None
        Alias for `p`, matching the ``manufacture(P=...)`` spelling used in the
        R package. When given, it overrides `p`.
    T : float or None, default None
        Alias for `t`. When given, it overrides `t`.
    random_state : int, np.random.Generator, or None, default None
        Seed or generator for the noise term. ``None`` (the default) draws
        fresh noise on every call; pass an ``int`` to make a call reproducible.

    Returns
    -------
    int
        Profit made per hour [dollars/hour], with random noise added.

    Raises
    ------
    ValueError
        If either input is a vector, is not finite, or is negative.

    Examples
    --------
    >>> manufacture(p=1.5, t=320, random_state=42)  # doctest: +SKIP
    601

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2026,
    https://learnche.org/pid

    Also see
    --------
    grocery
    popcorn
    """
    if P is None:
        P = p
    if T is None:
        T = t

    if np.ndim(P) > 0 or np.ndim(T) > 0:
        raise ValueError("Running the manufacturing experiments in parallel is (intentionally) not allowed.")

    if not np.isfinite(P) or not np.isfinite(T):
        raise ValueError("All function inputs must be finite numbers.")
    if P < 0:
        raise ValueError("Please provide a positive sales price, P.")
    if T < 0:
        raise ValueError("The throughput (parts per hour) must be a positive value.")

    rng = check_random_state(random_state)
    p_coded = (P - 1.5) / 1.0
    t_coded = (T - 320.0) / 20.0
    y = (
        18.0 * t_coded
        + 10 * p_coded
        - 5 * t_coded * p_coded
        - 7 * t_coded * t_coded
        - 24 * p_coded * p_coded
        + 50
    ) * 12 + 2 * np.sin(T) + 2 * np.cos(P) + rng.normal(0, 1) * 2
    return round(y)
