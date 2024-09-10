from typing import Callable

import numpy as np
import pandas as pd


def optimization_function(x: pd.DataFrame) -> float:
    """Return the D-optimality of the design."""
    try:
        x = pd.DataFrame(x).drop_duplicates()
        xtx_i = np.linalg.inv(np.dot(np.transpose(x), x))
    except np.linalg.LinAlgError:
        return float(np.inf)
    return float(np.linalg.slogdet(xtx_i)[1])


def index_to_replace_in_design_row(
    design: pd.DataFrame,
    candidate_point: pd.DataFrame,
    current_optimum: float,
    optimization_function: Callable,
) -> pd.DataFrame:
    """Find index to replace in design with the candidate point if the D-optimality of the new design is better."""
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
    try:
        return design_index[np.argmin(new_optimimum)]
    except ValueError:
        return design_index


def point_exchange(x: pd.DataFrame, number_points: int = 10) -> np.ndarray:
    """
    Return a design that is optimal in terms of D-optimality.

    Start with a random rows from X.
    For each row, for each factor, try alternate a row from the remaining rows in X.
    If the D-optimality of the new design is better, keep the new design.

    When do you swap a row? E.g. you request 2 points, and the 2 it selected are (-1,-1) and (-1, 1).
    While the optimum should be the opposite ends, right?
    """
    assert number_points >= x.shape[1], f"`number_point`s must be at least {x.shape[1]} (the number of columns in `x`)"
    assert number_points <= x.shape[0], f"`number_point`s must be at most {x.shape[0]} (the number of rows in `x`)"
    x = pd.DataFrame(x).drop_duplicates()

    number_points = min(number_points, x.shape[0])
    # Continually try to pick rows from x, until it is not singular
    is_singular = True
    while is_singular:  # TODO: add a limit to the number of repeats here
        try:
            x = x.sample(frac=1)
            design = x.iloc[0 : x.shape[1]]
            xtx_i = np.linalg.inv(np.dot(np.transpose(design), design))
            is_singular = False
        except np.linalg.LinAlgError:  # noqa: PERF203
            pass

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
        if design_row_to_replace:
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

    return design.sort_index(), d_optimality_i
