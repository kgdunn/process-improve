# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.

from numpy.random import normal
import pandas as pd


def popcorn(t=120, T=None):
    """
    Simulation of stovetop popcorn cooking.

    A fixed number of popcorn kernels are cooked are heated at the same
    temperature. This simulation returns the number of kernels that are edible
    when the pot is left on the stove for a given number of `t` seconds.


    Parameters
    ----------

    `t` is the number of seconds that the pot is left on the stove. The
    default amount, if not provided, is 120 seconds.

    Time durations less than 77 seconds are not supported. A vector (list) of
    time values is not permitted, since the goal is to perform sequential
    experimentation to determine the optimum time, with the fewest number of
    function calls.

    Returns
    -------
    The number of edible popcorn kernels. Random noise is added for realism.

    Source
    ------
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2021,
    https://learnche.org/pid


    Examples
    --------
    >>> popcorn(t=55)    # will fail
    >>> popcorn(t=120)
    >>> popcorn(t=152.2)

    >>> # What happens if we leave the pot on the stove for too long?
    >>> popcorn(t=500)


    Also see
    grocery
    manufacter

    """
    pass


def grocery(p=3.46, h=150, P=None, H=None):
    """
    Simulation of grocery store profits for a single product.

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
    Kevin Dunn, Process Improvement using Data, Chapter 5, 2010 to 2021,
    https://learnche.org/pid

    """
    if P is None:
        P = p
    if H is None:
        H = h

    if (len(P) > 1) | (len(H) > 1):
        assert (
            False
        ), "Running the grocery store experiments in parallel is (intentionally) not allowed."

    if pd.isna(P) or pd.isna(H):
        assert False, "All function inputs must be finite numbers."
    elif P < 0:
        assert False, "Please provide a positive sales price, P."
    elif H < 0:
        assert False, "The height of the shelving, H, must be a positive value."

    a_coded = (P - 3.2) / 0.2
    b_coded = (H - 50) / 100
    y = round(
        (
            18 * a_coded
            + 12 * b_coded
            - 7 * a_coded * a_coded
            - 6.0 * b_coded * b_coded
            - 8.5 * a_coded * b_coded
            + 60
        )
        * 10.0
        + normal(0, 1) * 2
    )
    return y
