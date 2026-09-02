from collections.abc import Callable

import numpy as np
import pandas as pd


def _model_matrix(x: pd.DataFrame) -> np.ndarray:
    """Return the model matrix ``[1 | X]`` for a first-order model with intercept.

    The D-criterion is defined on the model matrix, not on the raw factor
    settings. Scoring ``X`` alone (the previous behaviour) leaves the intercept
    out of the design's information matrix, so a candidate that holds a factor
    at a single level scores perfectly well even though that factor is then
    completely aliased with the intercept and its effect cannot be estimated.
    """
    values = np.asarray(pd.DataFrame(x), dtype=float)
    return np.column_stack([np.ones(values.shape[0]), values])


def optimization_function(x: pd.DataFrame) -> float:
    """Score a design for the point-exchange D-optimal search (lower is better).

    Returns ``-log|det(X'X)|``, the negative log of the standard D-criterion,
    where ``X`` is the model matrix ``[1 | factors]`` for a first-order model
    with an intercept, which is what this package fits. The point-exchange
    routine in this module uses this convention and selects swaps that decrease
    the returned value, i.e. that increase ``|det(X'X)|``.

    Returns ``+inf`` for any design the model cannot be fitted to: fewer runs
    than parameters, or a model matrix that is rank deficient. That covers a
    design holding a factor at a constant level (aliased with the intercept)
    and one whose factor columns are linearly dependent.

    The score is computed from ``slogdet(X'X)`` and guarded by an explicit rank
    check, never by inverting ``X'X``. Inversion cannot be used to detect
    singularity here: ``np.linalg.inv`` raises only when the LU factorisation
    hits an exactly-zero pivot, which a rank-deficient design of +-1 levels
    frequently avoids. The inverse then comes back as numerical noise and its
    log-determinant is a large *negative* number, so a search minimising this
    value treated inestimable designs as the best available and actively
    selected them.
    """
    # Do NOT de-duplicate here: replicated runs carry real information
    # (|X'X| for n copies of a point differs from one copy), so dropping
    # them would score a different design from the one being evaluated.
    model_matrix = _model_matrix(x)
    n_runs, n_parameters = model_matrix.shape
    sign, log_abs_det = np.linalg.slogdet(np.dot(np.transpose(model_matrix), model_matrix))
    # Both tests are needed. ``slogdet`` reports an exactly singular ``X'X``
    # (sign 0, log -inf), which catches a factor held at a constant level; a
    # design whose columns are only linearly dependent in exact arithmetic
    # slips past it with a finite, very negative log, and is caught by the rank
    # check. Without the second test that design would score as the best one
    # available, since the search minimises this value.
    if (
        n_runs < n_parameters
        or sign <= 0
        or not np.isfinite(log_abs_det)
        or np.linalg.matrix_rank(model_matrix) < n_parameters
    ):
        return float(np.inf)
    return float(-log_abs_det)


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


def point_exchange(
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
        D-optimality value, ``-log|det(X'X)|`` on the model matrix
        ``[1 | factors]``.
    """
    x = pd.DataFrame(x).drop_duplicates()
    # Intercept plus one coefficient per factor. Asking for fewer runs than
    # that cannot produce an estimable design, so it is rejected here rather
    # than left to fail later as an exhausted search for a non-singular start.
    # The bound used to be ``x.shape[1]``, one short of the model it scores.
    n_parameters = x.shape[1] + 1
    if number_points < n_parameters:
        raise ValueError(
            f"`number_points` must be at least {n_parameters} (an intercept plus "
            f"one coefficient per column of `x`); {number_points} were requested."
        )
    if number_points > x.shape[0]:
        # Checked after de-duplication: the search picks distinct candidate
        # rows, so duplicates in `x` cannot make up the shortfall. This used to
        # silently clamp the size down to the number of candidates available.
        raise ValueError(f"`number_points` must be at most {x.shape[0]} (the number of unique rows in `x`).")
    if not x.index.is_unique:
        # The search tracks the chosen rows by index LABEL, so a repeated label
        # makes the `.loc` lookup below return every row carrying it: asking for
        # 4 runs from a 6-row candidate set with one repeated label returned all
        # 6. Rows that are duplicated by VALUE are fine, and dropped above.
        raise ValueError("`x` must have a unique index; the point-exchange search selects rows by index label.")

    # Continually try to pick rows from x, until it is not singular.
    # A seedable Generator (rather than the global numpy RNG) makes the
    # point-exchange result reproducible; see docs/development/reproducibility.rst.
    rng = np.random.default_rng(random_state)
    max_attempts = 1000
    # `d_optimality_i` is deliberately NOT pre-initialised: the loop below runs
    # at least once and always assigns it, and its `else` raises, so a seed
    # value would only be dead (CodeQL flags it as such).
    for _attempt in range(max_attempts):
        x = x.sample(frac=1, random_state=rng)
        # Seed the design at the REQUESTED size. It used to start with
        # only ``x.shape[1]`` (one row per factor) and grow towards
        # `number_points` through the addition branch below, which
        # accepted a row only when it improved D-optimality. When no
        # addition improved it the design was returned short: callers
        # asking for `number_points` runs silently received fewer, and a
        # design with fewer runs than model parameters cannot be fitted
        # at all. The size is a constraint, not something to optimise.
        design = x.iloc[0:number_points]
        # Score the seed with the SAME criterion used for every comparison
        # below, so the search cannot be misled by an inconsistent start, and
        # so that a rank-deficient seed is rejected here rather than carried
        # through to the returned design.
        d_optimality_i = optimization_function(design)
        if np.isfinite(d_optimality_i):
            break
    else:
        msg = (
            f"Could not find a non-singular starting design after {max_attempts} "
            "attempts. The candidate set may contain collinear columns."
        )
        raise ValueError(msg)

    for i in range(number_points, x.shape[0]):  # the first `number_points` rows are the starting design
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

    # Every exchange above swaps one row for another, onto a unique index and
    # only ever onto a design scoring better than the (finite, full-rank)
    # current one, so the returned design keeps both the size and the
    # estimability it was seeded with.
    return design.sort_index(), d_optimality_i
