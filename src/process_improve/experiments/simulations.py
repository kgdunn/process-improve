# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import numpy as np


def popcorn(t: float = 120, T: float | None = None) -> None:
    """
    Simulate stovetop popcorn cooking (unimplemented stub).

    Placeholder for a planned textbook simulation. When implemented, it will
    return the number of edible popcorn kernels after cooking a fixed set of
    kernels at the same temperature for `t` seconds. The current body is
    empty, so this function returns ``None`` regardless of its inputs, and
    the return type annotation reflects that.

    Parameters
    ----------
    t : float, default 120
        Planned: number of seconds the pot is left on the stove. Time
        durations less than 77 seconds are not supported. A vector (list) of
        time values is not permitted, since the goal is to perform sequential
        experimentation to determine the optimum time, with the fewest number
        of function calls. Currently unused.
    T : float or None, default None
        Reserved for a future temperature parameter. Currently unused.

    Returns
    -------
    None
        Placeholder. A future implementation will return the number of
        edible kernels (with random noise added for realism).

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2026,
    https://learnche.org/pid

    Also see
    --------
    grocery
    manufacter
    """


def grocery(
    p: float = 3.46,
    h: float = 150,
    P: float | None = None,
    H: float | None = None,
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

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2026,
    https://learnche.org/pid

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

    a_coded = (P - 3.2) / 0.2
    b_coded = (H - 50) / 100
    return round(
        (18 * a_coded + 12 * b_coded - 7 * a_coded * a_coded - 6.0 * b_coded * b_coded - 8.5 * a_coded * b_coded + 60)
        * 10.0
        + np.random.default_rng().normal(0, 1) * 2
    )
