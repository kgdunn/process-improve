# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Mid-course correction of a running batch with a latent-variable model.

At a decision point during a batch, the initial conditions and the
trajectories observed so far are known, the future responses are not, and the
future manipulated-variable (MV) columns are ours to choose. This module
solves that choice as a quadratic program in the scaled space of a fitted
:class:`process_improve.batch.BatchPLS` model:

- the score vector of the candidate row is an affine function of the future
  MV columns, ``t = b + A_F u``, built from the fixed-missingness-pattern
  projection operator (:meth:`~process_improve.batch.BatchPLS.projection_matrix`,
  trimmed score regression by default), with the future *response* columns
  treated as missing;
- the objective trades off quality tracking (or maximisation) against
  movement from the nominal remaining schedule, plus soft SPE and Hotelling's
  T2 penalties that keep the correction where the model has data;
- box bounds, rate-of-change limits between consecutive samples (including
  the seam to the last implemented sample) and optional hard SPE / T2 caps
  complete the program.

With the quadratic caps the problem is a convex quadratically-constrained QP.
The workhorse is the penalty-form pure QP solved with `osqp
<https://osqp.org>`_ (the ``control`` extra); the hard-cap mode wraps the
same QP in an outer scalar iteration on the two penalty multipliers, which is
exact for this convex problem and converges in a handful of inner solves at
this size (a few dozen decision variables).

The formulation follows the latent-variable batch control literature:
Flores-Cerrillo and MacGregor (2004) for the quality-tracking objective, the
soft T2 term, and the reconstruction of the remaining trajectories
consistently with the realised past; Garcia-Munoz, Kourti and MacGregor
(2004) for the per-decision-point score covariance and limits; Yabuki and
MacGregor (1997) for the no-correction dead band; Golshan et al. (2010) for
the LV-MPC lineage and the practical trimmed-score-regression form. Two
departures from those papers are deliberate: the decision variables are the
future MV columns themselves (not a score correction), so bounds and rate
limits apply exactly in engineering units; and the SPE of the *candidate*
row is penalised and capped, not only checked on the measurements so far.

A practical caveat: models of this kind are identified on *recorded* (noisy,
realised) trajectories, while the corrector outputs *setpoints*. That is the
standard identification practice, and it attenuates the apparent gain
slightly (the regression sees the control error as input noise); the
executed-policy evaluation in :func:`evaluate_control_policies`
(:mod:`process_improve.simulation`) measures the realised effect rather than
trusting the model's own prediction.

References
----------
Flores-Cerrillo, J. and MacGregor, J.F., "Control of batch product quality
by trajectory manipulation using latent variable models", Journal of Process
Control, 14, 539-553, 2004.

Garcia-Munoz, S., Kourti, T. and MacGregor, J.F., "Model Predictive
Monitoring for Batch Processes", Industrial & Engineering Chemistry
Research, 43, 5929-5941, 2004.

Yabuki, Y. and MacGregor, J.F., "Product quality control in semibatch
reactors using midcourse correction policies", Industrial & Engineering
Chemistry Research, 36, 1268-1275, 1997.

Golshan, M., MacGregor, J.F., Bruwer, M.-J. and Mhaskar, P., "Latent
Variable Model Predictive Control (LV-MPC) for trajectory tracking in batch
processes", Journal of Process Control, 20, 538-550, 2010.
"""

from __future__ import annotations

import itertools
import typing

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist
from sklearn.utils import Bunch

from ..multivariate._limits import spe_calculation
from ..multivariate._projection import project_rows

if typing.TYPE_CHECKING:
    from ._batch_pls import BatchPLS

_MODES = ("target", "maximize")
_DEFAULT_WEIGHTS = {"target": 1.0, "movement": 0.1, "spe": 0.0, "t2": 0.0}
_CAP_REL_TOL = 0.01


def _solve_qp(
    H: np.ndarray,
    f: np.ndarray,
    A_con: np.ndarray | None,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> np.ndarray:
    """Solve ``min 0.5 x'Hx + f'x  s.t.  lower <= A_con x <= upper`` with osqp.

    The unconstrained case has the closed-form stationary solution
    ``H x = -f`` and skips the solver entirely.
    """
    if A_con is None:
        from .._linalg import safe_inverse  # noqa: PLC0415

        return safe_inverse(H, what="the QP Hessian") @ (-f)
    try:
        import osqp  # noqa: PLC0415 - deferred so the module imports without the extra
        from scipy import sparse  # noqa: PLC0415
    except ImportError as exc:
        from .._extras import require_extra  # noqa: PLC0415

        raise require_extra("osqp", "control") from exc

    problem = osqp.OSQP()
    problem.setup(
        sparse.csc_matrix(H),
        f,
        sparse.csc_matrix(A_con),
        lower,
        upper,
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=100_000,
        polish=True,
    )
    result = problem.solve()
    status = str(result.info.status)
    if "solved" not in status.lower():
        raise RuntimeError(
            f"The mid-course QP did not solve: osqp status {status!r}. "
            "Check that the bounds and rate limits admit any feasible schedule."
        )
    return np.asarray(result.x, dtype=float)


def _knot_matrix(n_free: int, n_knots: int) -> np.ndarray:
    """Linear-interpolation matrix mapping ``n_knots`` values to ``n_free`` samples."""
    if n_knots < 2 or n_knots > n_free:
        raise ValueError(f"n_knots must lie in [2, {n_free}] for {n_free} free samples; got {n_knots}.")
    positions = np.linspace(0, n_free - 1, n_knots)
    B = np.zeros((n_free, n_knots))
    for i in range(n_free):
        j = int(np.searchsorted(positions, i, side="right") - 1)
        j = min(j, n_knots - 2)
        left, right = positions[j], positions[j + 1]
        w = (i - left) / (right - left)
        B[i, j] = 1.0 - w
        B[i, j + 1] = w
    return B


def _check_columns(model: BatchPLS, labels: list, argument: str) -> None:
    known = set(model.feature_columns_)
    unknown = [label for label in labels if label not in known]
    if unknown:
        raise ValueError(f"{argument} contains labels that are not model features: {unknown[:5]}.")


def midcourse_correction(  # noqa: PLR0913, PLR0912, PLR0915, C901
    model: BatchPLS,
    *,
    observed: pd.Series,
    free_columns: list,
    mode: str = "target",
    y_target: pd.Series | dict | float | None = None,
    weights: dict | None = None,
    bounds: dict | None = None,
    rate_limits: dict | None = None,
    seam: dict | None = None,
    nominal_remaining: pd.Series | None = None,
    spe_cap: float | None = None,
    t2_cap: float | None = None,
    score_covariance: np.ndarray | pd.DataFrame | None = None,
    method: str = "tsr",
    ridge: float = 0.0,
    n_knots: int | None = None,
) -> Bunch:
    """Optimise the remaining manipulated-variable columns of one batch.

    This is the pure optimisation: everything is explicit and nothing is
    gated (no dead band, no validity check; use
    :class:`MidCourseCorrector` for the full decision-point workflow). The
    unfolded row of the model splits three ways: ``observed`` columns carry
    known values; ``free_columns`` are the decision variables (the future MV
    columns); every other column is a missing future response, imputed by the
    projection operator.

    Parameters
    ----------
    model : BatchPLS
        A fitted :class:`process_improve.batch.BatchPLS` model.
    observed : pd.Series
        Known values in engineering units, indexed by unfolded column labels:
        the initial conditions ``(name, "")`` and the past trajectory columns
        ``(tag, sample)``.
    free_columns : list
        Unfolded column labels of the decision variables, e.g.
        ``[("temperature", 12), ("temperature", 13), ...]``. Must be disjoint
        from ``observed``.
    mode : {"target", "maximize"}, default="target"
        ``"target"`` tracks ``y_target`` with a quadratic penalty (the
        Yabuki-MacGregor use case). ``"maximize"`` pushes the predicted
        quality up with a linear term; the quadratic movement penalty keeps
        the program bounded, which is the correct form for quality
        maximisation (an unreachable setpoint inside a quadratic is
        deliberately not used).
    y_target : Series, dict, or float, optional
        The quality target in original units; required for ``mode="target"``
        (a bare float is accepted for a single-target model).
    weights : dict, optional
        Keys (all optional): ``"target"`` (scalar or per-target array;
        tracking weight, or the linear reward in ``maximize`` mode),
        ``"movement"`` (scalar or per-free-column array; penalty on the
        scaled deviation from ``nominal_remaining``; must be positive in
        ``maximize`` mode), ``"spe"`` and ``"t2"`` (soft penalties on the
        candidate row's SPE and Hotelling's T2; the
        manufacturing-vs-development exploration dial). Defaults:
        ``{"target": 1.0, "movement": 0.1, "spe": 0.0, "t2": 0.0}``.
    bounds : dict, optional
        Per-tag box bounds in engineering units, ``{tag: (low, high)}``,
        applied to every free column of that tag. Tighten the box inward by
        roughly two control-error standard deviations, so optimised setpoints
        do not sit on the actuator rails where clipping biases the realised
        mean.
    rate_limits : dict, optional
        Per-tag limit on the change between consecutive samples, in
        engineering units, ``{tag: max_step}``. Applied between consecutive
        free samples of the tag and, when ``seam`` provides the last
        implemented value, across the seam as well.
    seam : dict, optional
        ``{tag: last_implemented_value}`` in engineering units, for the seam
        rate constraint.
    nominal_remaining : pd.Series, optional
        The nominal remaining schedule in engineering units, indexed by
        ``free_columns``; the movement penalty is measured from it. Default:
        the training average (the model's centring) of those columns.
    spe_cap : float, optional
        Hard cap on the candidate row's SPE (on the square-root scale used
        throughout the package, so the quadratic constraint bounds
        ``SPE**2``). Enforced by the outer multiplier iteration.
    t2_cap : float, optional
        Hard cap on the candidate row's Hotelling's T2.
    score_covariance : array-like of shape (A, A), optional
        Covariance used in the T2 quadratic. Default: the diagonal of the
        training score variances. Pass the per-decision-point covariance of
        the score *estimates* (Garcia-Munoz et al., 2004) for a reference
        that matches the pattern; :meth:`MidCourseCorrector.limits_at` builds
        it.
    method : {"tsr", "scp", "pmp"}, default="tsr"
        Score-estimation method for the projection operator.
    ridge : float, default=0.0
        Regularisation for the operator; see
        :meth:`~process_improve.batch.BatchPLS.projection_matrix`.
    n_knots : int, optional
        Parameterise each tag's free samples by ``n_knots`` linearly
        interpolated knot values. Shrinks the decision space (useful early in
        the batch) and smooths the schedule; the problem stays a QP.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``mv`` (Series, the optimised free columns in engineering
        units, in model-feature order), ``y_hat`` and ``y_hat_no_change``
        (Series, original quality units), ``scores`` (Series), ``spe`` and
        ``t2`` (floats for the candidate row over the observed-plus-free
        pattern), ``active_constraints`` (dict with keys ``bounds``,
        ``rate``, ``spe_cap``, ``t2_cap``), ``solver`` (Bunch with ``status``,
        ``n_solves``, ``spe_multiplier``, ``t2_multiplier``) and
        ``operator_condition_number``.
    """
    from sklearn.utils.validation import check_is_fitted  # noqa: PLC0415

    check_is_fitted(model, "x_weights_")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}; got {mode!r}.")
    if not free_columns:
        raise ValueError("free_columns is empty: there is nothing to optimise.")
    if not isinstance(observed, pd.Series):
        raise TypeError(
            f"observed must be a pandas Series indexed by unfolded column labels; got {type(observed).__name__}."
        )
    overlap = set(observed.index) & set(free_columns)
    if overlap:
        raise ValueError(f"observed and free_columns overlap: {sorted(overlap, key=str)[:5]}.")
    _check_columns(model, list(observed.index), "observed")
    _check_columns(model, list(free_columns), "free_columns")

    weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
    features = pd.Index(model.feature_columns_)
    observed_mask = features.isin(observed.index)
    free_mask = features.isin(set(free_columns))
    pattern_mask = observed_mask | free_mask

    A = int(model.n_components)
    n_targets = len(model.target_names_)
    center = model.center_.to_numpy(dtype=float)
    scale = model.scale_.to_numpy(dtype=float)

    # The operator over the observed-plus-free pattern; split by position.
    op = model.projection_matrix(pattern_mask, method=method, ridge=ridge)
    matrix = op.matrix.to_numpy(dtype=float)
    pattern_positions = np.flatnonzero(pattern_mask)
    in_free = free_mask[pattern_positions]
    M_free = matrix[:, in_free]
    M_obs = matrix[:, ~in_free]

    observed_positions = np.flatnonzero(observed_mask)
    free_positions = np.flatnonzero(free_mask)
    z_obs = (
        observed.reindex(pd.Index(features[observed_positions])).to_numpy(dtype=float) - center[observed_positions]
    ) / scale[observed_positions]
    if np.isnan(z_obs).any():
        raise ValueError("observed contains NaN values; every observed column needs a value.")

    b = M_obs @ z_obs  # scores of the batch-so-far with u = 0 (the training average)
    A_F = M_free

    # Quality map in scaled-Y space: y_s = C (b + A_F u).
    C = np.asarray(model.y_loadings_, dtype=float)  # (n_targets, A)
    G_y = C @ A_F
    y0_scaled = C @ b

    # SPE of the candidate row: residuals over observed and free positions.
    P = model.x_loadings_.to_numpy(dtype=float)
    P_obs = P[observed_positions, :]
    P_free = P[free_positions, :]
    D_obs = -P_obs @ A_F
    c_obs = z_obs - P_obs @ b
    n_free = len(free_positions)
    D_free = np.eye(n_free) - P_free @ A_F
    c_free = -P_free @ b

    # T2 quadratic: t' S^{-1} t with t = b + A_F u.
    if score_covariance is None:
        s_inv = np.diag(1.0 / np.asarray(model.explained_variance_, dtype=float))
    else:
        s_arr = np.asarray(score_covariance, dtype=float)
        if s_arr.shape != (A, A):
            raise ValueError(f"score_covariance must have shape ({A}, {A}); got {s_arr.shape}.")
        from .._linalg import safe_inverse  # noqa: PLC0415

        s_inv = safe_inverse(s_arr, what="score_covariance")

    # Nominal remaining schedule, scaled.
    if nominal_remaining is None:
        u_nom = np.zeros(n_free)
    else:
        nominal_values = nominal_remaining.reindex(pd.Index(features[free_positions])).to_numpy(dtype=float)
        if np.isnan(nominal_values).any():
            raise ValueError("nominal_remaining must cover every free column with a finite value.")
        u_nom = (nominal_values - center[free_positions]) / scale[free_positions]

    # Movement weights per free column.
    movement = np.asarray(weights["movement"], dtype=float)
    movement = np.full(n_free, float(movement)) if movement.ndim == 0 else movement
    if movement.shape != (n_free,):
        raise ValueError(f"weights['movement'] must be a scalar or length-{n_free} array; got shape {movement.shape}.")
    if mode == "maximize" and not np.all(movement > 0):
        raise ValueError("weights['movement'] must be strictly positive in maximize mode (it bounds the program).")

    # Target handling.
    w_target = np.asarray(weights["target"], dtype=float)
    w_target = np.full(n_targets, float(w_target)) if w_target.ndim == 0 else w_target
    if w_target.shape != (n_targets,):
        raise ValueError(f"weights['target'] must be a scalar or length-{n_targets} array; got shape {w_target.shape}.")
    y_scale = model.y_scale_.to_numpy(dtype=float)
    y_center = model.y_center_.to_numpy(dtype=float)
    if mode == "target":
        if y_target is None:
            raise ValueError("y_target is required when mode='target'.")
        if isinstance(y_target, (int, float)) and n_targets == 1:
            target_values = np.array([float(y_target)])
        else:
            target_series = pd.Series(y_target)
            target_values = target_series.reindex(model.target_names_).to_numpy(dtype=float)
            if np.isnan(target_values).any():
                raise ValueError(f"y_target must supply a value for every target in {model.target_names_}.")
        y_target_scaled = (target_values - y_center) / y_scale
    elif y_target is not None:
        raise ValueError("y_target only applies to mode='target'; in maximize mode use weights['target'].")

    # --- Quadratic assembly on u ------------------------------------------
    H_move = 2.0 * np.diag(movement)
    f_move = -2.0 * movement * u_nom
    H_spe = 2.0 * (D_obs.T @ D_obs + D_free.T @ D_free)
    f_spe = 2.0 * (D_obs.T @ c_obs + D_free.T @ c_free)
    spe_const = float(c_obs @ c_obs + c_free @ c_free)
    H_t2 = 2.0 * (A_F.T @ s_inv @ A_F)
    f_t2 = 2.0 * (A_F.T @ s_inv @ b)

    if mode == "target":
        W1 = np.diag(w_target)
        H_track = 2.0 * (G_y.T @ W1 @ G_y)
        f_track = 2.0 * (G_y.T @ W1 @ (y0_scaled - y_target_scaled))
    else:
        H_track = np.zeros((n_free, n_free))
        f_track = -(G_y.T @ w_target)

    # --- Constraints on u --------------------------------------------------
    free_labels = list(features[free_positions])
    free_tags = [label[0] for label in free_labels]
    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []
    row_names: list[str] = []

    if bounds:
        for tag, (low, high) in bounds.items():
            if low >= high:
                raise ValueError(f"bounds for {tag!r} must satisfy low < high; got ({low}, {high}).")
        for j, (label, tag) in enumerate(zip(free_labels, free_tags, strict=True)):
            if tag in bounds:
                low, high = bounds[tag]
                row = np.zeros(n_free)
                row[j] = 1.0
                rows.append(row)
                lows.append((low - center[free_positions[j]]) / scale[free_positions[j]])
                highs.append((high - center[free_positions[j]]) / scale[free_positions[j]])
                row_names.append(f"bound:{label}")

    if rate_limits:
        by_tag: dict[object, list[int]] = {}
        for j, tag in enumerate(free_tags):
            by_tag.setdefault(tag, []).append(j)
        for tag, positions_in_free in by_tag.items():
            if tag not in rate_limits:
                continue
            max_step = float(rate_limits[tag])
            if max_step <= 0:
                raise ValueError(f"rate_limits for {tag!r} must be positive; got {max_step}.")
            ordered = sorted(positions_in_free, key=lambda j: free_labels[j][1])
            for j_prev, j_next in itertools.pairwise(ordered):
                row = np.zeros(n_free)
                row[j_next] = scale[free_positions[j_next]]
                row[j_prev] = -scale[free_positions[j_prev]]
                offset = center[free_positions[j_next]] - center[free_positions[j_prev]]
                rows.append(row)
                lows.append(-max_step - offset)
                highs.append(max_step - offset)
                row_names.append(f"rate:{free_labels[j_next]}")
            if seam and tag in seam:
                j0 = ordered[0]
                row = np.zeros(n_free)
                row[j0] = scale[free_positions[j0]]
                offset = center[free_positions[j0]] - float(seam[tag])
                rows.append(row)
                lows.append(-max_step - offset)
                highs.append(max_step - offset)
                row_names.append(f"seam:{free_labels[j0]}")

    A_con = np.vstack(rows) if rows else None
    lower = np.asarray(lows, dtype=float) if rows else None
    upper = np.asarray(highs, dtype=float) if rows else None

    # --- Optional knot parameterisation ------------------------------------
    # The knots live in ENGINEERING units (the whole point is a smooth
    # schedule), so the substitution into the scaled decision space is
    # affine: u = W v + d with W = S^{-1} B_eng and d = -S^{-1} c, where S
    # and c are the per-column scale and centre of the free columns.
    if n_knots is not None:
        by_tag = {}
        for j, tag in enumerate(free_tags):
            by_tag.setdefault(tag, []).append(j)
        blocks = []
        order: list[int] = []
        for positions_in_free in by_tag.values():
            ordered = sorted(positions_in_free, key=lambda j: free_labels[j][1])
            order.extend(ordered)
            blocks.append(_knot_matrix(len(ordered), min(n_knots, len(ordered))))
        permute = np.zeros((n_free, n_free))
        for row_pos, j in enumerate(order):
            permute[j, row_pos] = 1.0
        from scipy.linalg import block_diag  # noqa: PLC0415

        scale_free = scale[free_positions]
        center_free = center[free_positions]
        W_sub = (permute @ block_diag(*blocks)) / scale_free[:, None]
        d_sub = -center_free / scale_free
    else:
        W_sub = None
        d_sub = None

    def _solve(mu_spe: float, mu_t2: float) -> np.ndarray:
        H = H_track + H_move + (weights["spe"] + mu_spe) * H_spe + (weights["t2"] + mu_t2) * H_t2
        f = f_track + f_move + (weights["spe"] + mu_spe) * f_spe + (weights["t2"] + mu_t2) * f_t2
        if W_sub is None:
            return _solve_qp(H, f, A_con, lower, upper)
        H_v = W_sub.T @ H @ W_sub
        f_v = W_sub.T @ (f + H @ d_sub)
        if A_con is None or lower is None or upper is None:
            A_v, low_v, up_v = None, None, None
        else:
            shift = A_con @ d_sub
            A_v, low_v, up_v = A_con @ W_sub, lower - shift, upper - shift
        v = _solve_qp(H_v, f_v, A_v, low_v, up_v)
        return W_sub @ v + d_sub

    def _statistics(u: np.ndarray) -> tuple[float, float]:
        t = b + A_F @ u
        ssr = float(np.sum((c_obs + D_obs @ u) ** 2) + np.sum((c_free + D_free @ u) ** 2))
        t2_value = float(t @ s_inv @ t)
        return np.sqrt(ssr), t2_value

    # --- Solve, with the outer multiplier iteration for hard caps ----------
    # Escalate whichever multiplier's cap is violated until both hold, then
    # bisect each active multiplier down so the achieved statistic lands just
    # inside its cap instead of far below it (the escalation overshoots).
    # Each statistic is non-increasing in its own multiplier for this convex
    # problem, so the per-coordinate bisection is well posed.
    mu_spe = mu_t2 = 0.0
    n_solves = 0
    u_star = _solve(mu_spe, mu_t2)
    n_solves += 1
    spe_value, t2_value = _statistics(u_star)
    spe_unconstrained, t2_unconstrained = spe_value, t2_value
    spe_cap_value = float("inf") if spe_cap is None else float(spe_cap)
    t2_cap_value = float("inf") if t2_cap is None else float(t2_cap)
    cap_status = "ok"
    for _ in range(30):
        spe_bad = spe_value > spe_cap_value * (1 + _CAP_REL_TOL)
        t2_bad = t2_value > t2_cap_value * (1 + _CAP_REL_TOL)
        if not spe_bad and not t2_bad:
            break
        if spe_bad:
            mu_spe = max(mu_spe * 4.0, 1e-3)
        if t2_bad:
            mu_t2 = max(mu_t2 * 4.0, 1e-3)
        u_star = _solve(mu_spe, mu_t2)
        n_solves += 1
        spe_value, t2_value = _statistics(u_star)
    else:
        cap_status = "cap_not_met"

    if cap_status == "ok" and (mu_spe > 0 or mu_t2 > 0):
        for _round in range(2):
            if mu_spe > 0 and spe_value < spe_cap_value * (1 - _CAP_REL_TOL):
                low_mu, high_mu = 0.0, mu_spe
                for _ in range(12):
                    mid = 0.5 * (low_mu + high_mu)
                    u_try = _solve(mid, mu_t2)
                    n_solves += 1
                    spe_try, _t2_try = _statistics(u_try)
                    if spe_try > spe_cap_value * (1 + _CAP_REL_TOL):
                        low_mu = mid
                    else:
                        high_mu = mid
                        u_star, spe_value, t2_value = u_try, spe_try, _t2_try
                        if spe_try >= spe_cap_value * (1 - _CAP_REL_TOL):
                            break
                mu_spe = high_mu
            if mu_t2 > 0 and t2_value < t2_cap_value * (1 - _CAP_REL_TOL):
                low_mu, high_mu = 0.0, mu_t2
                for _ in range(12):
                    mid = 0.5 * (low_mu + high_mu)
                    u_try = _solve(mu_spe, mid)
                    n_solves += 1
                    spe_try, t2_try = _statistics(u_try)
                    if t2_try > t2_cap_value * (1 + _CAP_REL_TOL):
                        low_mu = mid
                    else:
                        high_mu = mid
                        u_star, spe_value, t2_value = u_try, spe_try, t2_try
                        if t2_try >= t2_cap_value * (1 - _CAP_REL_TOL):
                            break
                mu_t2 = high_mu
            spe_ok = spe_value <= spe_cap_value * (1 + _CAP_REL_TOL)
            t2_ok = t2_value <= t2_cap_value * (1 + _CAP_REL_TOL)
            if spe_ok and t2_ok:
                break

    scores = b + A_F @ u_star
    y_hat_scaled = C @ scores
    y_hat = pd.Series(y_center + y_scale * y_hat_scaled, index=model.target_names_, name="y_hat")
    t_nominal = b + A_F @ u_nom
    y_no_change = pd.Series(y_center + y_scale * (C @ t_nominal), index=model.target_names_, name="y_hat_no_change")
    mv = pd.Series(
        center[free_positions] + scale[free_positions] * u_star,
        index=pd.Index(free_labels),
        name="mv",
    )

    active: dict[str, object] = {"bounds": [], "rate": [], "spe_cap": False, "t2_cap": False}
    if rows and A_con is not None and lower is not None and upper is not None:
        values = A_con @ u_star
        tol = 1e-6
        for name, value, low, high in zip(row_names, values, lower, upper, strict=True):
            if value <= low + tol or value >= high - tol:
                kind = name.split(":", 1)[0]
                key = "bounds" if kind == "bound" else "rate"
                typing.cast("list", active[key]).append(name.split(":", 1)[1])
    if spe_cap is not None:
        active["spe_cap"] = bool(mu_spe > 0 or spe_value >= spe_cap_value * (1 - _CAP_REL_TOL))
    if t2_cap is not None:
        active["t2_cap"] = bool(mu_t2 > 0 or t2_value >= t2_cap_value * (1 - _CAP_REL_TOL))

    return Bunch(
        mv=mv,
        y_hat=y_hat,
        y_hat_no_change=y_no_change,
        scores=pd.Series(scores, index=model.scores_.columns, name="scores"),
        spe=spe_value,
        t2=t2_value,
        active_constraints=active,
        solver=Bunch(
            status=cap_status,
            n_solves=n_solves,
            spe_multiplier=mu_spe,
            t2_multiplier=mu_t2,
            spe_unconstrained=spe_unconstrained,
            t2_unconstrained=t2_unconstrained,
        ),
        operator_condition_number=float(op.condition_number),
        spe_offset=spe_const,
    )


class MidCourseCorrector:
    """Decision-point workflow around :func:`midcourse_correction`.

    Holds the model, the nominal schedule and the tuning, and at each decision
    point: checks the batch-so-far against the model (the SPE validity gate of
    Flores-Cerrillo and MacGregor, 2004), applies the no-correction dead band
    (Yabuki and MacGregor, 1997) in target mode, builds the per-decision-point
    reference limits (Garcia-Munoz et al., 2004), solves the QP, and returns
    the full corrected schedule ready to implement (or to hand to
    :meth:`process_improve.simulation.BioreactorSimulator.simulate_batch`).

    Parameters
    ----------
    model : BatchPLS
        Fitted model whose X block unfolds recorded tag trajectories (and
        optionally initial conditions). Must be fitted with the default
        column layout (``group_by_batch=False``).
    nominal_schedule : pd.DataFrame
        The nominal setpoint schedule: ``n_timesteps_`` rows (positionally
        aligned with the tag samples), one column per manipulated tag.
    mv_tags : list
        The manipulated tags (a subset of the model's tag names); every other
        tag is a response, treated as missing after the decision point.
    mode : {"target", "maximize"}, default="target"
    y_target : Series, dict, or float, optional
        Required for ``mode="target"``.
    weights, bounds, rate_limits, method, ridge, n_knots
        Passed through to :func:`midcourse_correction`.
    spe_cap, t2_cap : float, "limit", or None, default="limit"
        Hard caps for the QP. ``"limit"`` resolves, per decision point, to
        the training-based limit for the same missingness pattern at
        ``conf_level`` (see :meth:`limits_at`); a float is used as given;
        None disables the cap.
    conf_level : float, default=0.95
        Confidence level for the per-decision-point limits and the dead-band
        prediction interval.
    dead_band : float, default=1.0
        Multiplier on the prediction-interval half-width: in target mode the
        correction is skipped while the no-change prediction lies within
        ``dead_band`` half-widths of the target for every quality variable.
        Set to 0.0 to correct at every decision point. Ignored in maximize
        mode.
    target_side : {"both", "below", "above"}, default="both"
        Which deviations from the target warrant a correction. ``"below"``
        treats the target as a floor (a more-is-better quality): batches
        predicted at or above it are left alone, whatever the dead band
        says. ``"above"`` is the mirror (a ceiling); ``"both"`` corrects
        deviations in either direction (an on-target specification).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: BatchPLS,
        nominal_schedule: pd.DataFrame,
        *,
        mv_tags: list,
        mode: str = "target",
        y_target: pd.Series | dict | float | None = None,
        weights: dict | None = None,
        bounds: dict | None = None,
        rate_limits: dict | None = None,
        spe_cap: float | str | None = "limit",
        t2_cap: float | str | None = "limit",
        conf_level: float = 0.95,
        dead_band: float = 1.0,
        target_side: str = "both",
        method: str = "tsr",
        ridge: float = 0.0,
        n_knots: int | None = None,
    ) -> None:
        from sklearn.utils.validation import check_is_fitted  # noqa: PLC0415

        check_is_fitted(model, "x_weights_")
        if model.group_by_batch:
            raise ValueError("MidCourseCorrector requires a model fitted with group_by_batch=False.")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {list(_MODES)}; got {mode!r}.")
        if mode == "target" and y_target is None:
            raise ValueError("y_target is required when mode='target'.")
        unknown_tags = [t for t in mv_tags if t not in model.tag_names_]
        if unknown_tags:
            raise ValueError(f"mv_tags contains tags the model does not carry: {unknown_tags}.")
        if not isinstance(nominal_schedule, pd.DataFrame):
            raise TypeError(f"nominal_schedule must be a DataFrame; got {type(nominal_schedule).__name__}.")
        if nominal_schedule.shape[0] != model.n_timesteps_:
            raise ValueError(
                f"nominal_schedule must have {model.n_timesteps_} rows (one per aligned sample); "
                f"got {nominal_schedule.shape[0]}."
            )
        missing_columns = [t for t in mv_tags if t not in nominal_schedule.columns]
        if missing_columns:
            raise ValueError(f"nominal_schedule is missing columns for mv_tags: {missing_columns}.")
        self.model = model
        self.nominal_schedule = nominal_schedule
        self.mv_tags = list(mv_tags)
        self.mode = mode
        self.y_target = y_target
        self.weights = weights
        self.bounds = bounds
        self.rate_limits = rate_limits
        self.spe_cap = spe_cap
        self.t2_cap = t2_cap
        if target_side not in ("both", "below", "above"):
            raise ValueError(f"target_side must be 'both', 'below' or 'above'; got {target_side!r}.")
        self.conf_level = conf_level
        self.dead_band = dead_band
        self.target_side = target_side
        self.method = method
        self.ridge = ridge
        self.n_knots = n_knots
        self._limit_cache: dict[int, Bunch] = {}

    # ------------------------------------------------------------------ #

    def _masks_at(self, k: int) -> Bunch:
        """Boolean masks over the unfolded features for decision point ``k``."""
        features = self.model.feature_columns_
        if not isinstance(features, pd.MultiIndex):
            raise TypeError("The model's feature columns must carry the 2-level (tag, sequence) index.")
        sequence = features.get_level_values("sequence")
        tags = features.get_level_values("tag")
        is_z = np.array([s == "" for s in sequence])
        seq_num = np.array([-1 if z else int(s) for s, z in zip(sequence, is_z, strict=True)])
        past = ~is_z & (seq_num < k)
        future = ~is_z & (seq_num >= k)
        is_mv = np.array([t in set(self.mv_tags) for t in tags])
        observed = is_z | past
        free = future & is_mv
        return Bunch(observed=observed, free=free, missing=future & ~is_mv)

    def limits_at(self, k: int) -> Bunch:
        """Per-decision-point reference limits from the training batches.

        The training rows are re-projected under decision point ``k``'s two
        patterns: the *monitoring* pattern (initial conditions plus every tag
        up to ``k``; the future entirely missing) for the SPE validity gate,
        and the *candidate* pattern (monitoring plus the future MV columns,
        which the optimiser treats as observed) for the QP's SPE cap and the
        score covariance behind its T2 term. Limits: the g-chi-squared SPE
        limit of Nomikos and MacGregor on each pattern's training SPE values,
        and the F-distribution T2 limit on the candidate-pattern score
        estimates with their own covariance (Garcia-Munoz et al., 2004).

        Results are cached per ``k``.
        """
        if k in self._limit_cache:
            return self._limit_cache[k]
        if not 1 <= k <= self.model.n_timesteps_:
            raise ValueError(f"k must lie in [1, {self.model.n_timesteps_}]; got {k}.")
        masks = self._masks_at(k)
        training = self.model._x_scaled_training.to_numpy(dtype=float)
        loadings = self.model.x_loadings_.to_numpy(dtype=float)
        guide = self.model.direct_weights_.to_numpy(dtype=float)
        variances = np.asarray(self.model.explained_variance_, dtype=float)

        monitor_rows = training.copy()
        monitor_rows[:, ~masks.observed] = np.nan
        monitor = project_rows(loadings, guide, variances, monitor_rows, method=self.method, ridge=self.ridge)

        candidate_mask = masks.observed | masks.free
        candidate_rows = training.copy()
        candidate_rows[:, ~candidate_mask] = np.nan
        candidate = project_rows(loadings, guide, variances, candidate_rows, method=self.method, ridge=self.ridge)

        n = training.shape[0]
        A = int(self.model.n_components)
        score_cov = np.cov(candidate.scores, rowvar=False, ddof=1)
        score_cov = np.atleast_2d(score_cov)
        t2_limit = float((A * (n**2 - 1)) / (n * (n - A)) * f_dist.ppf(self.conf_level, A, n - A))
        result = Bunch(
            spe_limit_monitor=float(spe_calculation(monitor.spe, conf_level=self.conf_level)),
            spe_limit_candidate=float(spe_calculation(candidate.spe, conf_level=self.conf_level)),
            t2_limit=t2_limit,
            score_covariance=score_cov,
        )
        self._limit_cache[k] = result
        return result

    def _observed_series(
        self,
        batch_so_far: pd.DataFrame,
        initial_conditions: pd.Series | pd.DataFrame | None,
        k: int,
    ) -> pd.Series:
        """Build the engineering-unit observed Series for decision point ``k``."""
        model = self.model
        if list(batch_so_far.columns) != list(model.tag_names_):
            raise ValueError(
                f"batch_so_far must carry exactly the training tags {model.tag_names_}; "
                f"got {list(batch_so_far.columns)}."
            )
        entries: dict = {}
        if model.n_initial_conditions_:
            if initial_conditions is None:
                raise ValueError("The model was fitted with initial conditions; they are required here.")
            z_row = initial_conditions.iloc[0] if isinstance(initial_conditions, pd.DataFrame) else initial_conditions
            for name in model.initial_condition_names_:
                if name not in z_row.index:
                    raise ValueError(f"initial_conditions is missing {name!r}.")
                entries[(name, "")] = float(z_row[name])
        elif initial_conditions is not None:
            raise ValueError("The model was fitted without initial conditions; do not pass any.")
        for s in range(k):
            for tag in model.tag_names_:
                entries[(tag, s)] = float(batch_so_far.iloc[s][tag])
        return pd.Series(entries)

    def correct(  # noqa: C901, PLR0912, PLR0915 - the decision-point workflow is one narrative
        self,
        batch_so_far: pd.DataFrame,
        *,
        initial_conditions: pd.Series | pd.DataFrame | None = None,
        implemented_schedule: pd.DataFrame | None = None,
        k: int | None = None,
    ) -> Bunch:
        """Decide and (when warranted) compute the correction at one decision point.

        Parameters
        ----------
        batch_so_far : pd.DataFrame
            The recorded tag trajectories up to the decision point: the first
            ``k`` samples, columns = the model's tags.
        initial_conditions : pd.Series or pd.DataFrame, optional
            The batch's Z values; required if the model was fitted with a Z
            block.
        implemented_schedule : pd.DataFrame, optional
            The setpoint schedule actually implemented so far (same layout as
            ``nominal_schedule``); its first ``k`` rows are carried into the
            returned schedule verbatim and its row ``k - 1`` anchors the seam
            rate constraint. Defaults to the nominal schedule.
        k : int, optional
            The decision point (number of completed samples). Defaults to
            ``len(batch_so_far)``.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``schedule`` (the full setpoint DataFrame: implemented
            past plus the decided remainder), ``corrected`` (bool),
            ``reason`` (``"corrected"``, ``"spe_gate"``, ``"dead_band"`` or
            ``"batch_complete"``), ``k``, ``spe_so_far`` and
            ``spe_limit_monitor`` (the validity gate), ``y_hat_no_change``
            and, in target mode, ``dead_band_margin`` (Series; deviation of
            the no-change prediction from the target in units of the
            prediction-interval half-width), plus ``correction`` (the full
            :func:`midcourse_correction` Bunch) when a correction was
            computed.
        """
        model = self.model
        if k is None:
            k = len(batch_so_far)
        if not 1 <= k <= model.n_timesteps_:
            raise ValueError(f"k must lie in [1, {model.n_timesteps_}]; got {k}.")
        if len(batch_so_far) < k:
            raise ValueError(f"batch_so_far has {len(batch_so_far)} samples but k={k} were requested.")
        schedule = (implemented_schedule if implemented_schedule is not None else self.nominal_schedule).copy()
        if schedule.shape[0] != model.n_timesteps_:
            raise ValueError(f"implemented_schedule must have {model.n_timesteps_} rows; got {schedule.shape[0]}.")
        if k == model.n_timesteps_:
            return Bunch(schedule=schedule, corrected=False, reason="batch_complete", k=k)

        limits = self.limits_at(k)
        masks = self._masks_at(k)
        observed = self._observed_series(batch_so_far, initial_conditions, k)

        # --- SPE validity gate on the batch so far -------------------------
        features = pd.Index(model.feature_columns_)
        center = model.center_.to_numpy(dtype=float)
        scale = model.scale_.to_numpy(dtype=float)
        row = np.full(len(features), np.nan)
        positions = features.get_indexer(observed.index)
        row[positions] = (observed.to_numpy(dtype=float) - center[positions]) / scale[positions]
        so_far = project_rows(
            model.x_loadings_.to_numpy(dtype=float),
            model.direct_weights_.to_numpy(dtype=float),
            np.asarray(model.explained_variance_, dtype=float),
            row[None, :],
            method=self.method,
            ridge=self.ridge,
        )
        spe_so_far = float(so_far.spe[0])
        if spe_so_far > limits.spe_limit_monitor:
            return Bunch(
                schedule=schedule,
                corrected=False,
                reason="spe_gate",
                k=k,
                spe_so_far=spe_so_far,
                spe_limit_monitor=limits.spe_limit_monitor,
            )

        # --- Assemble the QP inputs ---------------------------------------
        free_labels = list(features[masks.free])
        nominal_remaining = pd.Series({(tag, s): float(self.nominal_schedule.iloc[s][tag]) for (tag, s) in free_labels})
        seam = {tag: float(schedule.iloc[k - 1][tag]) for tag in self.mv_tags} if k > 0 else None
        caps: dict[str, float | None] = {}
        for name, setting, resolved in (
            ("spe_cap", self.spe_cap, limits.spe_limit_candidate),
            ("t2_cap", self.t2_cap, limits.t2_limit),
        ):
            if setting == "limit":
                caps[name] = float(resolved)
            elif setting is None:
                caps[name] = None
            else:
                caps[name] = float(typing.cast("float", setting))

        # --- Dead band (target mode): correct only when the projected
        # deviation is significant against the prediction interval. ---------
        dead_band_margin = None
        if self.mode == "target" and (self.dead_band > 0 or self.target_side != "both"):
            probe = midcourse_correction(
                model,
                observed=observed,
                free_columns=free_labels,
                mode="target",
                y_target=self.y_target,
                weights={"target": 0.0, "movement": 1.0},
                nominal_remaining=nominal_remaining,
                score_covariance=limits.score_covariance,
                method=self.method,
                ridge=self.ridge,
            )
            y0 = probe.y_hat_no_change
            n = model.n_samples_
            df = max(n - int(model.n_components) - 1, 1)
            t_crit = t_dist.ppf(1 - (1 - self.conf_level) / 2, df)
            leverage = 1.0 / n + probe.t2 / (n - 1)
            error_std = model.rmse_.iloc[:, -1].to_numpy(dtype=float)
            half_width = t_crit * np.sqrt(1.0 + leverage) * error_std
            target = pd.Series(self.y_target) if not isinstance(self.y_target, (int, float)) else None
            target_values = (
                target.reindex(model.target_names_).to_numpy(dtype=float)
                if target is not None
                else np.array([float(typing.cast("float", self.y_target))])
            )
            signed = y0.to_numpy(dtype=float) - target_values
            if self.target_side == "below":
                deviation = np.maximum(-signed, 0.0)
            elif self.target_side == "above":
                deviation = np.maximum(signed, 0.0)
            else:
                deviation = np.abs(signed)
            dead_band_margin = pd.Series(deviation / half_width, index=model.target_names_, name="dead_band_margin")
            if bool((deviation <= self.dead_band * half_width).all()):
                return Bunch(
                    schedule=schedule,
                    corrected=False,
                    reason="dead_band",
                    k=k,
                    spe_so_far=spe_so_far,
                    spe_limit_monitor=limits.spe_limit_monitor,
                    y_hat_no_change=y0,
                    dead_band_margin=dead_band_margin,
                )

        result = midcourse_correction(
            model,
            observed=observed,
            free_columns=free_labels,
            mode=self.mode,
            y_target=self.y_target if self.mode == "target" else None,
            weights=self.weights,
            bounds=self.bounds,
            rate_limits=self.rate_limits,
            seam=seam,
            nominal_remaining=nominal_remaining,
            spe_cap=caps["spe_cap"],
            t2_cap=caps["t2_cap"],
            score_covariance=limits.score_covariance,
            method=self.method,
            ridge=self.ridge,
            n_knots=self.n_knots,
        )

        for (tag, s), value in result.mv.items():
            schedule.iloc[s, typing.cast("int", schedule.columns.get_loc(tag))] = float(value)
        return Bunch(
            schedule=schedule,
            corrected=True,
            reason="corrected",
            k=k,
            spe_so_far=spe_so_far,
            spe_limit_monitor=limits.spe_limit_monitor,
            spe_limit_candidate=limits.spe_limit_candidate,
            t2_limit=limits.t2_limit,
            y_hat=result.y_hat,
            y_hat_no_change=result.y_hat_no_change,
            dead_band_margin=dead_band_margin,
            correction=result,
        )
