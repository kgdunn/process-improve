from collections.abc import Callable

import numpy as np
import pandas as pd


def optimization_function(x: pd.DataFrame) -> float:
    """Score a design for the point-exchange D-optimal search (lower is better).

    Returns ``log|det((X'X)^-1)|`` which is equivalent to the negative
    log of the standard D-criterion ``log|det(X'X)|``. The point-exchange
    routine in this module uses this convention and selects swaps that
    decrease the returned value, i.e. that increase ``|det(X'X)|``.

    Returns ``+inf`` when ``X'X`` is singular (no improvement possible).
    """
    try:
        # Do NOT de-duplicate here: replicated runs carry real information
        # (|X'X| for n copies of a point differs from one copy), so dropping
        # them would score a different design from the one being evaluated.
        x = pd.DataFrame(x)
        xtx_i = np.linalg.inv(np.dot(np.transpose(x), x))
    except np.linalg.LinAlgError:
        return float(np.inf)
    log_det_inverse = float(np.linalg.slogdet(xtx_i)[1])
    if not np.isfinite(log_det_inverse):
        # A numerically singular X'X can pass np.linalg.inv without raising and
        # yield a garbage inverse whose determinant is exactly zero, scoring
        # log|det| = -inf: under the lower-is-better convention that would rank
        # the singular design as unbeatable, freezing the exchange loop (no
        # replacement or addition can improve on -inf). Score it unacceptable
        # instead, like the exactly singular case above.
        return float(np.inf)
    return log_det_inverse


def index_to_replace_in_design_row(
    design: pd.DataFrame,
    candidate_point: pd.DataFrame,
    current_optimum: float,
    optimization_function: Callable,
) -> int | None:
    """Find the row index in ``design`` to replace with ``candidate_point``.

    Iterates over each row of ``design`` and evaluates the D-optimality of the
    design that results from swapping that row with ``candidate_point``.

    Returns
    -------
    int | None
        The index of the row whose replacement yields the best (lowest)
        optimality score. If no row produces an improvement over
        ``current_optimum``, ``None`` is returned. (An earlier version
        returned an empty list here and the caller tested the result for
        truthiness, which also discarded a legitimate improving swap onto
        the row whose index LABEL is 0 - 0 is falsy.)
    """
    design_index = []
    new_optimimum = []
    design_to_consider = design.copy()
    for row_idx in design_to_consider.index:
        updated_design = pd.DataFrame(
            np.vstack([design_to_consider.drop(index=row_idx).values, candidate_point.values])
        )
        candidate_optimum = optimization_function(updated_design)
        if current_optimum > candidate_optimum:
            design_index.append(row_idx)
            new_optimimum.append(candidate_optimum)
    if not design_index:
        return None
    return design_index[int(np.argmin(new_optimimum))]


def point_exchange(  # noqa: C901
    x: pd.DataFrame, number_points: int = 10, random_state: int | None = None
) -> tuple[pd.DataFrame, float]:
    """
    Return a design that is optimal in terms of D-optimality.

    Start with a random rows from X.
    For each row, for each factor, try alternate a row from the remaining rows in X.
    If the D-optimality of the new design is better, keep the new design.

    When do you swap a row? E.g. you request 2 points, and the 2 it selected are (-1,-1) and (-1, 1).
    While the optimum should be the opposite ends, right?

    Returns
    -------
    tuple[pd.DataFrame, float]
        The selected design (sorted by the original row index) and its
        D-optimality value (log-determinant of ``(X'X)^-1``).
    """
    if number_points < x.shape[1]:
        raise ValueError(f"`number_points` must be at least {x.shape[1]} (the number of columns in `x`).")
    if number_points > x.shape[0]:
        raise ValueError(f"`number_points` must be at most {x.shape[0]} (the number of rows in `x`).")
    x = pd.DataFrame(x).drop_duplicates()

    number_points = min(number_points, x.shape[0])
    # Continually try to pick rows from x, until it is not singular.
    # A seedable Generator (rather than the global numpy RNG) makes the
    # point-exchange result reproducible; see docs/development/reproducibility.rst.
    rng = np.random.default_rng(random_state)
    max_attempts = 1000
    xtx_i = None
    for _attempt in range(max_attempts):
        try:
            x = x.sample(frac=1, random_state=rng)
            design = x.iloc[0 : x.shape[1]]
            xtx_i = np.linalg.inv(np.dot(np.transpose(design), design))
            break
        except np.linalg.LinAlgError:
            pass
    else:
        msg = (
            f"Could not find a non-singular starting design after {max_attempts} "
            "attempts. The candidate set may contain collinear columns."
        )
        raise ValueError(msg)

    _, d_optimality_i = np.linalg.slogdet(xtx_i)

    for i in range(x.shape[1], x.shape[0]):  # we've already considered the first `x.shape[1]` rows to start
        candidate_point = x.iloc[[i]]

        # Try to replace the candidate point with each row in the current design
        design_row_to_replace = index_to_replace_in_design_row(
            design,
            candidate_point,
            current_optimum=d_optimality_i,
            optimization_function=optimization_function,
        )
        if design_row_to_replace is not None:
            design_index = design.index.tolist()
            # Replace the row in `design` which as index of `design_row_to_replace`:
            design_index[design_index.index(design_row_to_replace)] = candidate_point.index[0]
            design = x.loc[design_index]
            d_optimality_i = optimization_function(design)
            # print(f"New D-optimality at {i=} (replc): {d_optimality_i}")
            continue

        # Now do the additionsm if there is room.
        if design.shape[0] < number_points:
            potential_design = pd.concat([design, candidate_point])
            d_optimality_i_potential = optimization_function(potential_design)
            if d_optimality_i > d_optimality_i_potential:
                design = potential_design
                d_optimality_i = d_optimality_i_potential
                # print(f"New D-optimality at {i=} (merge): {d_optimality_i}")

    # The loop above grows the design only when an addition improves the
    # criterion, and a candidate consumed by a replacement is never considered
    # for addition, so an unlucky shuffle can end below number_points (about
    # 1 to 2% of unseeded runs for a 27-candidate, 4-point request). The
    # contract is to return number_points rows: complete the design greedily
    # with the remaining candidates that best preserve D-optimality.
    while design.shape[0] < number_points:
        remaining = x.loc[~x.index.isin(design.index)]
        if remaining.empty:  # pragma: no cover - number_points is capped at len(x) above
            break
        scores = [optimization_function(pd.concat([design, remaining.iloc[[j]]])) for j in range(remaining.shape[0])]
        best_j = int(np.argmin(scores))
        design = pd.concat([design, remaining.iloc[[best_j]]])
        d_optimality_i = scores[best_j]

    return design.sort_index(), d_optimality_i
