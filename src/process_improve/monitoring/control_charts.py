"""Class for ControlChart: robust control charts with a balance between CUSUM and Shewhart properties."""

import logging
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..regression.methods import repeated_median_slope
from ..univariate.metrics import median_absolute_deviation

logger = logging.getLogger(__name__)


#: Consistency constant for the bounded biweight rho with cutoff k = 2.52,
#: c_k = 1 / E[rho_norm(Z)] for Z ~ N(0, 1), where rho_norm is the biweight
#: rho normalised to a maximum of 1. Computed via numerical integration
#: (scipy.integrate.quad of rho_norm(z) * phi(z); E = 0.3061160, so
#: c_k = 3.266736). This makes E[rho(Z)] = 1, the condition for the scale
#: estimates built from rho to be consistent for sigma on Gaussian data.
#:
#: Departure from the paper, kept deliberately: equation (13) of Gelper, Fried and
#: Croux (2010, p 289) pairs the cutoff k = 2 with c_k = 2.52 ("for the common choice
#: of k = 2 we have c_k = 2.52"; numerically 1 / E[rho_norm(Z)] = 2.5153 for k = 2).
#: This module uses 2.52 as the cutoff instead, with the matching c_k above. Both pairs
#: give consistent scale estimates on Gaussian data, but this biweight saturates later
#: (at ``|x| > 2.52`` rather than ``|x| > 2``), so it down-weights large forecast errors
#: less than the paper's. The constants are left unchanged because existing results
#: and callers depend on them.
BIWEIGHT_RHO_CONSISTENCY = 3.266736


def rho(x: float, k: float = 2.52) -> float:
    """
    Bi-weight rho function.

    The multiplier is the consistency constant c_k (chosen so that
    ``E[rho(Z)] = 1`` for standard-normal Z), NOT the cutoff k. The paper
    treats the two as separate constants; an earlier version of this code
    conflated them and used k = 2.52 as the multiplier, which made every
    scale estimate derived from rho a factor ``sqrt(2.52 * 0.30612) = 0.878``
    too small, i.e. +/-3S control limits that were really +/-2.63 sigma
    (a ~3x inflation of the false-alarm rate).

    Parameters
    ----------
    x : float
        Value at which to evaluate the bi-weight rho function.
    k : float, optional
        Bi-weight cutoff. The default of 2.52 is kept for compatibility. In the
        paper (equation 13, p 289) 2.52 is instead the consistency constant c_k
        for the cutoff k = 2; see the note on ``BIWEIGHT_RHO_CONSISTENCY``.

    Returns
    -------
    float
        ``c_k * (1 - (1 - (x / k)**2)**3)`` for ``|x| <= k``; for ``|x| > k``
        the function saturates at the consistency constant ``c_k``.

    References
    ----------
    https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
    """
    c_k = BIWEIGHT_RHO_CONSISTENCY
    return c_k if np.abs(x) > k else c_k * (1 - np.power(1 - np.power(x / k, 2), 3))


def _rho_array(x: np.ndarray | pd.Series, k: float = 2.52) -> np.ndarray:
    """
    Evaluate the bi-weight :func:`rho` element-wise over an array.

    Replaces ``np.vectorize(rho)``, which calls the scalar function once per element. The
    polynomial branch is evaluated on ``x`` clipped to ``[-k, k]``: ``np.where`` evaluates both
    branches, and an unclipped huge ``|x|`` would overflow in ``np.power`` and raise a
    ``RuntimeWarning`` from values that are then discarded. Saturated entries take ``c_k``
    directly and NaN propagates, exactly as in :func:`rho`.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        Values at which to evaluate the bi-weight rho function.
    k : float, optional
        Bi-weight cutoff, as in :func:`rho`.

    Returns
    -------
    np.ndarray
        ``rho(x)`` for every element of ``x``, as float64.
    """
    x = np.asarray(x, dtype=float)
    c_k = BIWEIGHT_RHO_CONSISTENCY
    inside = c_k * (1 - np.power(1 - np.power(np.clip(x, -k, k) / k, 2), 3))
    return np.where(np.abs(x) > k, c_k, inside)


def _finite(values: pd.Series | np.ndarray) -> bool:
    """Return ``True`` if at least one entry of ``values`` is finite."""
    return bool(np.isfinite(np.asarray(values, dtype=float)).any())


def _training_error_radicand(future_errors: pd.Series) -> float:
    """
    Return tau squared for the training-sample errors, or NaN when it is undefined.

    Equation 16 of the Holt-Winters paper. This is the shared computation behind the
    lambda grid search, which treats an undefined cell as a non-contender, and
    :func:`_tau_from_training_errors`, which rejects it. Returning NaN rather than
    routing an all-NaN slice through ``np.nanmedian`` / ``np.nanmean`` keeps a
    ``RuntimeWarning`` from escaping ``calculate_limits``. (#557)

    Parameters
    ----------
    future_errors : pd.Series
        One-step-ahead errors over the training samples (those after the warm-up).

    Returns
    -------
    float
        The radicand, or ``np.nan`` when the errors hold nothing finite or carry no
        usable spread.
    """
    if not _finite(future_errors):
        return float("nan")

    # s_T = 1.48 Med|r_t|, as printed below equation (16) of the paper: 1.48 is the Gaussian
    # consistency factor of the median absolute error, 1 / Phi^-1(3/4) = 1.4826, rounded. The
    # robust Shewhart chart below uses the unrounded value.
    s_t_median_error = 1.48 * future_errors.abs().median()
    if not np.isfinite(s_t_median_error) or s_t_median_error <= 0:
        return float("nan")

    return float(np.power(s_t_median_error, 2) * np.nanmean(_rho_array(future_errors / s_t_median_error)))


def _tau_from_training_errors(future_errors: pd.Series) -> float:
    """
    Return the robust scale estimate (tau) for the training-sample errors.

    Equation 16 of the Holt-Winters paper, with each way the estimate can come out
    undefined reported rather than silently reduced to zero. The previous
    ``np.sqrt(max(0.0, resids))`` was written to stop a negative radicand reaching
    ``sqrt``, but ``max(0.0, nan)`` returns ``0.0``: ``nan > 0.0`` is ``False``, so
    ``max`` keeps its first argument. A NaN therefore became a zero scale, which the
    caller carried outward as control limits of zero width. (#557)

    Parameters
    ----------
    future_errors : pd.Series
        One-step-ahead errors over the training samples (those after the warm-up).

    Returns
    -------
    float
        The scale estimate, strictly positive.

    Raises
    ------
    ValueError
        If no training error is finite, or if the resulting radicand is non-finite
        or not positive (the errors carry no usable spread).
    """
    if not _finite(future_errors):
        # Every training error is NaN when the training period holds no finite observation
        # to forecast against, or when the warm-up statistics themselves are not finite.
        raise ValueError(
            f"The control-chart scale estimate is undefined: none of the {future_errors.size} "
            "training-sample errors is finite, so the series has no finite observation in its "
            "training period to forecast against. Supply a series with more finite values."
        )

    resids = _training_error_radicand(future_errors)
    if not np.isfinite(resids) or resids <= 0:
        raise ValueError(
            f"The control-chart scale estimate is undefined (tau^2 = {resids}), so the control "
            "limits would have zero width. Supply more representative data, or pass an "
            "explicit positive 's'."
        )

    return float(np.sqrt(resids))


def psi(x: float, k: float = 2.0) -> float:
    """
    Pre-clean based on the Huber psi function.

    Can be interpreted as replacing unexpected high or low values by a more likely value.

    Parameters
    ----------
    x : float
        Value to clean.
    k : float, optional
        Huber tuning constant: the threshold beyond which `x` is clipped to
        ``k * sign(x)``. The default of 2.0 follows p 288 of the referenced
        paper.

    Returns
    -------
    float
        `x` itself when ``|x| < k``; otherwise ``k * sign(x)``.

    References
    ----------
    https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
    """
    return x if abs(x) < k else k * np.sign(x)


def _holt_winters_recursion(
    y: np.ndarray, warm_up: Mapping[str, Any], ld_1: float, ld_2: float, ld_s: float
) -> dict[str, np.ndarray]:
    """
    Run the robust Holt-Winters recursion over ``y``, one time step per row.

    Row ``i`` needs row ``i - 1``, so the recursion stays a loop, but over plain numpy arrays
    that are read and filled in place, instead of several ``DataFrame`` lookups per row and a
    ``df.loc`` write of every row. The arithmetic is kept expression for expression, in the
    same order, on the same ``np.float64`` scalars and through the same scalar :func:`rho` and
    :func:`psi`, so the results are bit-for-bit those of the per-row pandas implementation it
    replaces (pinned by ``tests/test_control_chart_hw_golden.py``).

    Parameters
    ----------
    y : np.ndarray
        The observations, as stored in ``ControlChart.df["y"]``.
    warm_up : Mapping[str, Any]
        ``ControlChart.warm_up``. Row 0 is set from ``y_zero_robust`` (median of the warm-up
        window) and the initial level ``alpha_0``, trend ``beta_0`` and strictly positive
        scale ``sigma_0`` (p 290 of the paper).
    ld_1 : float
        Smoothing parameter for the level, lambda_1.
    ld_2 : float
        Smoothing parameter for the trend, lambda_2.
    ld_s : float
        Smoothing parameter for the scale, lambda_s.

    Returns
    -------
    dict[str, np.ndarray]
        Float64 arrays of length ``len(y)``, keyed by their ``ControlChart.df`` column:
        ``psi_input``, ``rho_input``, ``y_star``, ``alpha_hat``, ``beta_hat``, ``sigma_hat``
        and ``error``. ``error[0]`` is NaN: row 0 has no one-step-ahead prediction.
    """
    y_zero, alpha_0, beta_0, sigma_0 = (warm_up[key] for key in ("y_zero_robust", "alpha_0", "beta_0", "sigma_0"))
    n = len(y)
    psi_input, rho_input, y_star, alpha_hat, beta_hat, sigma_hat, error = (np.full(n, np.nan) for _ in range(7))

    rho_input[0] = (y_zero - alpha_0 - beta_0) / sigma_0
    psi_input[0] = rho_input[0]
    y_star[0] = y_zero
    alpha_hat[0] = alpha_0
    beta_hat[0] = beta_0
    sigma_hat[0] = sigma_0

    for i in range(1, n):
        # Cover the warm-up period, and the rest of the data set. We need that for the residual
        # calculation later anyway. The previous state is read back from the float64 arrays,
        # which yields the same np.float64 scalars the per-row DataFrame lookups returned.
        alpha_prev, beta_prev, sigma_prev = alpha_hat[i - 1], beta_hat[i - 1], sigma_hat[i - 1]

        # Error = observed - predicted. Predicted = one-step-ahead prediction
        error_i = y[i] - (alpha_prev + beta_prev)
        if np.isnan(error_i):
            # If there is an error, replace it with the median of the last 10 error estimates
            # or as many points as available, filtered to the finite ones so an all-NaN slice
            # never reaches a median (which returns NaN but emits "Mean of empty slice"). (#557)
            #
            # Known limitation, kept for now: the imputed error is a median of ABSOLUTE errors,
            # so it is never negative. Each missing row is treated as an observation above its
            # prediction, which nudges the level up, and over a long gap the trend term
            # compounds that into a drift (with lambdas 0.4 / 0.7, a 40-row gap in N(50, 2) noise
            # took the level from about 49 to 550). Carrying the forecast forward for every
            # missing row, as below, would remove the bias; later gaps keep the imputation so
            # that their results do not change.
            recent_errors = np.abs(error[max(i - 10, 0) : i])
            recent_errors = recent_errors[np.isfinite(recent_errors)]
            if not recent_errors.size:
                # No finite error to impute from: a gap that starts at index 1, the first row
                # with a one-step-ahead error. Imputing NaN used to spread through every later
                # row and make the whole fit fail. Carry the forecast forward instead (the level
                # follows the trend; trend and scale are held) and leave this row's error and
                # cleaned value NaN, so nothing invented reaches the target or the scale.
                alpha_hat[i], beta_hat[i], sigma_hat[i] = alpha_prev + beta_prev, beta_prev, sigma_prev
                continue
            error_i = float(np.median(recent_errors))
        rho_i = error_i / sigma_prev
        prior_variance = np.power(sigma_prev, 2)
        sigma_i = np.sqrt(rho(rho_i) * ld_s * prior_variance + (1.0 - ld_s) * prior_variance)
        psi_i = error_i / sigma_i
        y_star_i = psi(psi_i) * sigma_i + alpha_prev + beta_prev
        alpha_i = ld_1 * y_star_i + (1 - ld_1) * (alpha_prev + beta_prev)
        beta_i = ld_2 * (alpha_i - alpha_prev) + (1 - ld_2) * beta_prev

        psi_input[i], rho_input[i], y_star[i] = psi_i, rho_i, y_star_i
        alpha_hat[i], beta_hat[i], sigma_hat[i], error[i] = alpha_i, beta_i, sigma_i, error_i

    return {
        "psi_input": psi_input,
        "rho_input": rho_input,
        "y_star": y_star,
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "sigma_hat": sigma_hat,
        "error": error,
    }


def _one_sided_cusum(
    deviations: np.ndarray, reference: float, interval: float
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Run a one-sided tabular CUSUM, ``C_t = max(0, C_(t-1) + d_t - reference)``, from ``C_0 = 0``.

    The sum restarts from zero after each alarm (``C_t > interval``), and a missing deviation
    holds it. Returns the sum at every sample, and for each alarm a tuple of the alarm's
    position, the position where its run of non-zero sums started, and the estimated shift,
    ``reference + C_t / N`` for a run of ``N`` observations.
    """
    sums = np.empty_like(deviations)
    alarms = []
    total, start, n_run = 0.0, 0, 0
    for t, deviation in enumerate(deviations):
        if not np.isnan(deviation):
            total = max(0.0, total + deviation - reference)
            start = t if n_run == 0 else start
            n_run = n_run + 1 if total > 0.0 else 0
        sums[t] = total
        if total > interval:
            alarms.append((t, start, reference + total / n_run))
            total, n_run = 0.0, 0
    return sums, alarms


class ControlChart:
    """Create control chart instance objects."""

    #: Tuning parameters, declared but not assigned here: each exists on an instance only once
    #: a caller pins it in ``calculate_limits`` or a fit sets it, which ``hasattr`` relies on.
    #: A Holt-Winters weight left as None is searched for.
    ld_1: float | None
    ld_2: float | None
    k: float
    h: float

    def __init__(self, style: str = "robust", variant: str = "HW") -> None:
        """
        Create/initialize a control chart.


        Args: style (str, optional): Which style control chart to calculate. Defaults to "robust".
            Other choice is 'regular' (i.e. not-robust) calculations. User should then ensure that
            no outliers are present in the data.

            variant (str, optional): Four variants are accepted: ``'hw'`` (the default),
                ``'xbar.no.subgroup'``, ``'ewma'`` and ``'cusum'``. Any other value raises
                ``ValueError`` at construction time.
                The variant string is compared case-insensitively (it is normalised
                via ``.strip().lower()`` on assignment), so ``'HW'``, ``'hw'``, and
                ``'Hw'`` are all equivalent.

                The default is a Holt-Winters (`'hw'`) chart, with automatic determination of
                control chart parameters. This chart is a blend of infinite history (CUSUM)
                charts, and an instantaneous (no history taken into account) Shewhart chart. The
                exact blend is specified by parameters `ld_1` (lambda 1) and `ld_2` (lambda 2).

                The other accepted variants are:

                'xbar.no.subgroup' [Shewhart chart, with no subgroups]. In other words, each
                observation is independently plotted on the control chart.

                'ewma' [exponentially weighted moving average chart]. The charted statistic is
                ``z_t = ld_1 * y_t + (1 - ld_1) * z_{t-1}``, started at the target, and it is
                compared with its exact time-varying limits,
                ``target +/- 3 s sqrt(ld_1 / (2 - ld_1) * (1 - (1 - ld_1) ** (2 t)))``. Pass the
                weight ``ld_1`` (lambda, default 0.2) to ``calculate_limits``. Averaging over
                recent samples makes it detect a small sustained shift that a Shewhart chart
                misses; ``ld_1 = 1`` gives the Shewhart individuals chart.

                'cusum' [tabular cumulative sum chart]. Two one-sided sums,
                ``C+_t = max(0, C+_(t-1) + (y_t - target) - k s)`` and
                ``C-_t = max(0, C-_(t-1) - (y_t - target) - k s)``, start at zero, and an alarm
                is raised when either exceeds the decision interval ``h s``; that sum then
                restarts from zero. Pass ``k`` (the reference value: half the shift to detect, in
                units of ``s``, default 0.5) and ``h`` (default 5) to ``calculate_limits``. Each
                alarm also estimates when the shift started and the new mean (``cusum_alarms``).
        """
        self.style = style.strip()
        self.variant = variant.strip().lower()
        # An unknown variant previously slipped through every fit branch and
        # surfaced much later as a misleading "input is likely constant or too
        # short" error from calculate_limits. Reject it up front instead.
        if self.variant not in self._TUNING_KWARGS:
            raise ValueError(
                f"Control chart variant {variant!r} is not implemented; "
                f"supported variants are {sorted(self._TUNING_KWARGS)}."
            )

        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        """
        Forget everything an earlier fit set; only the constructor arguments survive.

        ``calculate_limits`` starts here, so a chart reused for a new series gives exactly
        what a fresh chart would. It used to keep the previous series' ``target`` and ``s``
        and treat them as given, keep its fitted lambdas instead of searching again, and
        fail outright on a series of a different length.
        """
        constructor_arguments = {"style": self.style, "variant": self.variant}
        self.__dict__.clear()
        self.__dict__.update(constructor_arguments)

        # Will be calculated by the self.calculate_limits() function
        self.target: float | None = None
        self._given_target: float | None = None
        self._given_s: float | None = None
        self.s: float | None = None
        # index of elements which are found to be outside +/- 3S
        self.idx_outside_3S: list[int] = []
        self.warm_up: dict[str, float | np.ndarray | pd.Series] = {}
        self.warm_up_M: int = 0

        columns = [
            "y",
            "psi_input",
            "rho_input",
            "y_star",
            "alpha_hat",
            "beta_hat",
            "sigma_hat",
            "error",
        ]
        self.df = pd.DataFrame(columns=columns, dtype=np.float64)

    def calculate_limits(
        self,
        y: np.ndarray | pd.Series,
        target: float | None = None,
        s: float | None = None,
        **kwargs,
    ) -> None:
        """
        Find for a given vector `y`, the control chart target and limits.

        Works for the Holt-Winters ('hw'), 'xbar.no.subgroup' and 'ewma' variants. For 'ewma',
        ``idx_outside_3S`` lists the samples where the EWMA statistic is outside its limits,
        and ``df`` gains the columns ``ewma``, ``ewma_ucl`` and ``ewma_lcl``.

        For the Holt-Winters variant, when there are fewer than

            min(20, max(10, np.ceil(0.10 * N)))

        measurements (where N is the length of the input vector), the target and
        standard deviation are estimated directly from the data and any provided
        `target` / `s` are ignored for that small-sample case.
        Otherwise, if `target` and `s` are numeric, those values are used;
        if not, they are estimated.

        Missing values (NaN) in `y` are bridged by imputing a one-step-ahead error equal
        to the median absolute error of the previous 10 rows. That error is never negative,
        so a gap pulls the Holt-Winters level upward, increasingly so for long gaps; fill or
        drop long runs of missing values before fitting. A gap that starts at index 1, the
        first row with a one-step-ahead error, has no earlier error to impute from: those
        rows carry the forecast forward instead (the level follows the trend, the trend and
        scale are held), and their error and cleaned value stay NaN.

        Every call starts afresh: nothing fitted by an earlier call on the same chart
        (target, ``s``, lambdas, warm-up statistics) is carried into this one.
        """
        self._reset_fit_state()
        self._given_target = target
        self._given_s = s
        logger.debug(
            "ControlChart.calculate_limits: variant=%s, style=%s, given target=%s, s=%s",
            self.variant,
            self.style,
            target,
            s,
        )

        if s is not None:
            self.s = float(s)
            if not 0.0 < s < 1e300:
                raise ValueError(
                    f"The given standard deviation must be positive and not excessively large (0 < s < 1e300); got {s}."
                )

        if target is not None:
            self.target = float(target)

        self.df["y"] = y.ravel() if isinstance(y, np.ndarray) else pd.Series(y).values.ravel()
        self.N = self.df.shape[0]

        # Between M = 10 and 20 samples required to warm-up (calculate summary statistics).
        # The paper fixes its startup period at m = 10 (p 291); here it grows with N, to 10%
        # of the series, capped at 20. Kept as is: existing results depend on it.
        self.warm_up["M"] = self.warm_up_M = int(min(20, max(10, np.ceil(0.10 * self.N))))

        if (self.warm_up_M > self.N) and self.variant.strip().lower() == "hw":
            # TO CHECK: Completely handle the case with very few samples. Is everything filled in?
            # Also check case when some of these samples are NAN, you might have even fewer still.
            self.target = self._target_calculated_best = self.df["y"].median()
            self.s = self._tau = self.df["y"].std()
            self.df["y_star"] = self.df["y"].values
            self.df["alpha_hat"] = self.target
            self.df["beta_hat"] = 0.0
            self.df["sigma_hat"] = np.nan
            self.df["error"] = np.nan
            return

        # Check if there are enough training samples:
        if 2 * self.warm_up_M > self.N:
            self.train_samples: list[int] = [int(i) for i in np.arange(0, self.N)]
        else:
            self.train_samples = [int(i) for i in np.arange(self.warm_up_M, self.N)]

        self._apply_tuning_kwargs(kwargs)
        fits = {
            "hw": self._holt_winters_fit,
            "xbar.no.subgroup": self._xbar_no_subgroup_fit,
            "ewma": self._ewma_fit,
            "cusum": self._cusum_fit,
        }
        fits[self.variant]()

        # After whichever fit is completed, check which are outside +/- 3S.
        # Explicit validation (not assert) so the guard survives `python -O`
        # and surfaces as a documented ValueError at the tool boundary (SEC-17).
        if self.target is None or self.s is None:
            raise ValueError(
                "Control chart limits could not be estimated; the input is likely "
                "constant or too short to fit the chosen variant."
            )
        self.idx_outside_3S = self._samples_outside_limits(self.target, self.s)

    def _samples_outside_limits(self, target: float, s: float) -> list[int]:
        """Return the positions of the samples outside the chart's limits (the CUSUM chart's alarms)."""
        if self.variant == "cusum":
            return sorted(set(self.cusum_alarms.index.tolist()))
        if self.variant == "ewma":
            # The statistic is compared with its own limits; a missing observation is never flagged.
            half_width = self.df["ewma_ucl"] - target
            outside = ((self.df["ewma"] - target).abs() > half_width) & self.df["y"].notna()
        else:
            outside = (self.df["y"] - target).abs() > 3.0 * s
        return np.nonzero(outside.to_numpy())[0].tolist()

    #: The variants, and the tuning parameters a caller may pin for each via
    #: ``calculate_limits(**kwargs)``: the Holt-Winters smoothing weights, the EWMA weight, and
    #: the CUSUM reference value and decision interval. Everything else is internal state.
    _TUNING_KWARGS: ClassVar[dict[str, frozenset[str]]] = {
        "hw": frozenset({"ld_1", "ld_2"}),
        "xbar.no.subgroup": frozenset({"ld_1", "ld_2"}),  # accepted and unused, as before
        "ewma": frozenset({"ld_1"}),
        "cusum": frozenset({"k", "h"}),
    }

    def _apply_tuning_kwargs(self, kwargs: dict[str, object]) -> None:
        """Set caller-pinned tuning parameters, rejecting anything off the allowlist.

        A blanket ``setattr(self, key, val)`` over ``**kwargs`` would let a caller
        silently overwrite internal state (``self.s``, ``self.target``,
        ``self.train_samples``, even a bound method) and would swallow typos. We
        therefore accept only the documented parameters of the chart's variant and raise a
        clear ``ValueError`` otherwise.
        """
        accepted = self._TUNING_KWARGS[self.variant]
        unknown = set(kwargs) - accepted
        if unknown:
            raise ValueError(
                f"calculate_limits() got unexpected keyword argument(s) {sorted(unknown)}; "
                f"the {self.variant!r} chart accepts {sorted(accepted)}."
            )
        for key, val in kwargs.items():
            setattr(self, key, val)

    #: Default EWMA weight, inside the 0.05 to 0.25 range usually recommended for small shifts.
    EWMA_DEFAULT_WEIGHT: ClassVar[float] = 0.2

    def _ewma_fit(self) -> None:
        """
        Fit an EWMA chart: the statistic and its exact time-varying 3-sigma limits.

        A given ``target`` and ``s`` are used as they are; whichever is missing is estimated
        from the data as for the Shewhart chart (median and 1.4826 times the MAD for the robust
        style, mean and standard deviation for the regular style).

        After ``t`` observations the statistic has variance ``lambda / (2 - lambda) * (1 - (1 -
        lambda) ** (2 t))`` times ``s**2``, which grows to its steady-state value, so the first
        limits are narrower and a shift present from the start is not missed. A missing
        observation carries the statistic forward and does not count towards ``t``.
        """
        given_weight = getattr(self, "ld_1", None)
        weight = self.EWMA_DEFAULT_WEIGHT if given_weight is None else float(given_weight)
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"The EWMA weight ld_1 must satisfy 0 < ld_1 <= 1; got {weight}.")
        self.ld_1 = weight

        self._estimate_missing_target_and_s()
        if self.target is None or self.s is None:
            return  # an unknown style estimates nothing; calculate_limits raises for it

        y = self.df["y"].to_numpy(dtype=float)
        statistic = np.empty_like(y)
        previous = self.target
        for t, value in enumerate(y):
            previous = previous if np.isnan(value) else weight * value + (1.0 - weight) * previous
            statistic[t] = previous
        n_observed = np.cumsum(~np.isnan(y))
        half_width = 3.0 * self.s * np.sqrt(weight / (2.0 - weight) * (1.0 - (1.0 - weight) ** (2 * n_observed)))
        self.df["ewma"] = statistic
        self.df["ewma_ucl"] = self.target + half_width
        self.df["ewma_lcl"] = self.target - half_width

    #: Default CUSUM reference value and decision interval, in units of ``s``: ``k = 0.5`` tunes
    #: the chart to a one-sigma shift, and ``h = 5`` gives an in-control average run length of 465.
    CUSUM_DEFAULT_K: ClassVar[float] = 0.5
    CUSUM_DEFAULT_H: ClassVar[float] = 5.0

    def _cusum_fit(self) -> None:
        """
        Fit a tabular CUSUM chart: the upper and lower cumulative sums and the alarms they raise.

        The reference value is ``K = k s`` and the decision interval ``H = h s``. When a sum
        exceeds ``H`` it raises an alarm and restarts from zero, as it would once the cause has
        been found, so each alarm is a separate detection. A missing observation holds both sums.

        For each alarm, ``cusum_alarms`` gives its direction, where the shift is estimated to
        have started (the first sample of the run of non-zero sums that raised it), and the
        estimated new mean, ``target + K + C+/N`` upward or ``target - K - C-/N`` downward, where
        ``N`` is the number of observations in that run.
        """
        given_k, given_h = getattr(self, "k", None), getattr(self, "h", None)
        k = self.CUSUM_DEFAULT_K if given_k is None else float(given_k)
        h = self.CUSUM_DEFAULT_H if given_h is None else float(given_h)
        if not (0.0 <= k < np.inf and 0.0 < h < np.inf):
            raise ValueError(f"The CUSUM parameters must satisfy k >= 0 and h > 0, both finite; got k={k}, h={h}.")
        self.k, self.h = k, h

        self._estimate_missing_target_and_s()
        if self.target is None or self.s is None:
            return  # an unknown style estimates nothing; calculate_limits raises for it

        deviation = self.df["y"].to_numpy(dtype=float) - self.target
        reference, interval = k * self.s, h * self.s
        self.df["cusum_upper"], upward = _one_sided_cusum(deviation, reference, interval)
        self.df["cusum_lower"], downward = _one_sided_cusum(-deviation, reference, interval)
        alarms = [(t, "up", start, self.target + shift) for t, start, shift in upward]
        alarms += [(t, "down", start, self.target - shift) for t, start, shift in downward]
        columns = ["sample", "direction", "shift_start", "new_mean"]
        self.cusum_alarms = pd.DataFrame(sorted(alarms), columns=columns).set_index("sample")

    def _estimate_missing_target_and_s(self) -> None:
        """Estimate whichever of ``target`` and ``s`` was not given, as the Shewhart chart does.

        That is the median and 1.4826 times the MAD for the robust style, and the mean and
        standard deviation for the regular style; a given value is kept as it is.
        """
        if self.target is None or self.s is None:
            given_target, given_s = self.target, self.s
            self._xbar_no_subgroup_fit()
            self.target = given_target if given_target is not None else self.target
            self.s = given_s if given_s is not None else self.s

    def _holt_winters_fit(self) -> None:
        """Fit the Holt-Winters chart, searching for whichever smoothing weight was not pinned."""
        if not hasattr(self, "ld_1"):
            self.ld_1 = None
        if not hasattr(self, "ld_2"):
            self.ld_2 = None
        self._holt_winters_parameter_fit()

    def _xbar_no_subgroup_fit(self) -> None:
        """
        Fit the control chart from the data samples, assuming each sample is its own subgroup.
        The `style` attribute ('regular' | 'robust') switches how the average and standard
        deviation are calculated. A ``target`` or ``s`` given to ``calculate_limits`` is kept,
        and only a missing one is estimated. The robust ``s`` is the MAD about the data's own
        median, so a given target away from the centre of the data does not inflate it.

        Control chart limits assume the data are normally distributed and independent. In
        particular, this last assumption can have consequences if not actually met. Limits may be
        too wide, or too narrow.

        """
        y = self.df["y"]
        if self.style == "regular":
            centre, spread = y.mean(), y.std()
        elif self.style == "robust":
            centre = y.median()
            spread = (y - centre).abs().median() * 1.4826
        else:
            return  # an unknown style estimates nothing; calculate_limits raises for it
        self.target = centre if self.target is None else self.target
        self.s = spread if self.s is None else self.s

    def _holt_winters_parameter_fit(self) -> None:
        """
        Recommended in the paper: not to fit the lambda_s value, but to use a grid search for the
        lambda_1 and lambda_2 values. This is done in a 5x5 grid in the code below.
        """
        self.ld_s = ld_s = 0.2

        # ``is not None``: an explicit ld_1=0.0 (or ld_2=0.0) is a legitimate
        # user choice, but 0.0 is falsy and a plain truthiness test silently
        # discarded it and ran the grid search instead.
        if self.ld_1 is not None and self.ld_2 is not None:
            # User has provided their own lambda_1 and lambda_2 values.
            for _name, _val in (("Lambda_1", self.ld_1), ("Lambda_2", self.ld_2), ("Lambda_s", self.ld_s)):
                if _val < 0.0:
                    raise ValueError(f"{_name} must be greater than or equal to zero.")
                if _val > 1.0:
                    raise ValueError(f"{_name} must be less than or equal to 1.0.")
            self._holt_winters_warmup_fit(ld_1=self.ld_1, ld_2=self.ld_2, ld_s=self.ld_s)

        else:
            # User wants to find an value for ld_1 and ld_2 that best fits the data

            ld_1_index = np.linspace(0.1, 0.9, num=5, endpoint=True)
            ld_2_index = np.linspace(0.1, 0.9, num=5, endpoint=True)
            residuals, _ = np.meshgrid(ld_1_index, ld_2_index)

            # The warm-up statistics do not depend on the lambdas: estimate them once, not per cell.
            self._holt_winters_warm_up_statistics()
            for i, ld_1 in enumerate(ld_1_index):
                for j, ld_2 in enumerate(ld_2_index):
                    self._holt_winters_smooth(ld_1=ld_1, ld_2=ld_2, ld_s=ld_s)

                    # Apply equation 16 from the paper to the residuals in the 'training' period,
                    # that is the samples after the warm-up period. NaN-aware
                    # statistics are required: row 0 never receives an "error"
                    # value, and for small samples (2 * warm_up_M > N) the
                    # training window includes row 0. With plain
                    # np.median/np.average every grid cell became NaN and the
                    # search silently "chose" (0.1, 0.1) via argmin-of-NaN.
                    future_errors = self.df["error"].iloc[np.asarray(self.train_samples, dtype=int)]

                    # An unusable cell records NaN and cannot win the search; the
                    # `np.all(np.isnan(residuals))` check below still catches the case
                    # where every cell is unusable. (#557)
                    residuals[i, j] = _training_error_radicand(future_errors)

            if np.all(np.isnan(residuals)):
                raise ValueError(
                    "The Holt-Winters lambda grid search produced no usable residuals; "
                    "the input is likely constant, too short, or entirely missing."
                )
            min_idx = np.nanargmin(residuals)
            best_ld_1 = ld_1_index[np.unravel_index(min_idx, residuals.shape)[0]]
            best_ld_2 = ld_2_index[np.unravel_index(min_idx, residuals.shape)[1]]
            # Store the parameters that were calculated, even if a `target` or `s` were provided.
            self.ld_1 = best_ld_1
            self.ld_2 = best_ld_2
            self._residuals_HW = residuals
            self._holt_winters_smooth(ld_1=best_ld_1, ld_2=best_ld_2, ld_s=ld_s)

        # Common code for both branches of if-else above
        future_errors = self.df["error"].iloc[np.asarray(self.train_samples, dtype=int)]
        self._tau = _tau_from_training_errors(future_errors)
        if self.target is None:
            # Estimate the target as the median of the y-star (cleaned) y-values
            self.target = self.df["y_star"].median()
        else:
            self._target_calculated_best = self.df["y_star"].median()

        if self.s is None:
            self.s = self._tau  # or an alternative: self.df["sigma_hat"] is approximately OK

        # The "delta" emphasizes that it is the deviation from the target.
        self._delta_UCL_3sigma = +3.0 * self._tau
        self._delta_LCL_3sigma = -3.0 * self._tau

    def _holt_winters_warmup_fit(self, ld_1: float = 0.5, ld_2: float = 0.8, ld_s: float = 0.2) -> None:
        """
        See paper: https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125.

        Calculates the Holt-Winters fitting and control chart parameters, for given values of the
        smoothing parameters lambda_1 (how much local history for the level is used, with values
        approaching 1.0 implying that less history is used), and lambda_2 (history for the trend
        that is used, with lambda_2 approaching 1.0 implying that historical data is less
        interesting), and lambda_s, a similar parameter for the moving variance of the sequence.

        lambda_1 = ld_1 = 0.5 (default): value must be between 0 <= ld_1 <= 1.0
        lambda_2 = ld_2 = 0.8 (default): value must be between 0 <= ld_2 <= 1.0
        lambda_s = ld_s = 0.2 (default): value must be between 0 <= ld_s <= 1.0, based on values
                                         used in the paper, recommended on page 291.

        The ideal lambda values (ld_1, ld_2, ld_s) can be found from a grid search.
        """
        self._holt_winters_warm_up_statistics()
        self._holt_winters_smooth(ld_1=ld_1, ld_2=ld_2, ld_s=ld_s)

    def _holt_winters_warm_up_statistics(self) -> None:
        """
        Estimate the warm-up level, trend and scale (p 290 of the paper) into ``self.warm_up``.

        They depend only on the warm-up window and on a given ``target`` / ``s``, not on the
        smoothing lambdas, so the lambda grid search estimates them once instead of per cell.

        Raises
        ------
        ValueError
            If the warm-up window has zero (or non-finite) variance.
        """
        y_warm_up = self.df["y"].iloc[0 : self.warm_up_M]
        self.warm_up["y_zero_robust"] = y_warm_up.median()

        if isinstance(self.target, float):
            self.warm_up["alpha_0"] = self.target
            self.warm_up["beta_0"] = 0.0
        else:
            # p 290 of the paper, https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
            self.warm_up["beta_0"] = repeated_median_slope(np.arange(self.warm_up_M), y_warm_up.to_numpy())
            self.warm_up["alpha_0"] = np.nanmedian(y_warm_up - self.warm_up["beta_0"] * np.arange(self.warm_up_M))

        if isinstance(self.s, float):
            self.warm_up["sigma_0"] = self.s

        else:
            # p 290 of the paper, https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
            # The residual is y_t - alpha_0 - beta_0 * t (alpha_0 above is the
            # median of exactly that de-trended series). Subtracting beta_0 as
            # a constant, as an earlier version did, leaves the whole warm-up
            # trend inside the residuals and inflates sigma_0 whenever the
            # window drifts - precisely the situation this chart is for.
            warm_up_residuals = y_warm_up - self.warm_up["alpha_0"] - self.warm_up["beta_0"] * np.arange(self.warm_up_M)

            # Some other method that does not rely on SciPy for 1 function.
            self.warm_up["sigma_0"] = median_absolute_deviation(np.asarray(warm_up_residuals), nan_policy="omit")
            self.warm_up["residuals"] = warm_up_residuals

            # A constant (zero-variance) warm-up window gives sigma_0 = MAD = 0, which
            # would make rho/psi infinite and silently poison every downstream control
            # limit with 0/NaN. Fail loudly instead of returning meaningless limits.
            _residuals_array = warm_up_residuals.dropna().to_numpy()
            _is_constant = _residuals_array.shape[0] == 0 or (_residuals_array[0] == _residuals_array).all()
            if not _is_constant and self.warm_up["sigma_0"] == 0:
                # Corner case: if there are multiple unique values in the warm-up residuals, but the MAD is zero,
                # then sigma_0 is set to regular standard deviation instead, which is non-zero.
                # This can happen when the warm-up residuals are symmetrically distributed around the median,
                # leading to a MAD of zero, but still have variability that can be captured by the standard deviation.
                self.warm_up["sigma_0"] = warm_up_residuals.std()

        # A constant (zero-variance) warm-up window gives sigma_0 = MAD = 0, which
        # would make rho/psi infinite and silently poison every downstream control
        # limit with 0/NaN. Fail loudly instead of returning meaningless limits.
        if not np.isfinite(self.warm_up["sigma_0"]) or self.warm_up["sigma_0"] <= 0:
            raise ValueError(
                "The Holt-Winters warm-up window has zero (or non-finite) variance "
                "(sigma_0 = 0), so the control-chart limits would be undefined. "
                "Supply more representative warm-up data, or pass a positive 's'."
            )

    def _holt_winters_smooth(self, ld_1: float, ld_2: float, ld_s: float) -> None:
        """Run the Holt-Winters recursion from ``self.warm_up`` and write its columns into ``self.df``."""
        df = self.df
        columns = _holt_winters_recursion(df["y"].to_numpy(), self.warm_up, ld_1=ld_1, ld_2=ld_2, ld_s=ld_s)
        # One write per column per fit; the "y" column, the column order and the index are untouched.
        for name, values in columns.items():
            df[name] = values
