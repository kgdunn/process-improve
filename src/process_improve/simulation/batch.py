# (c) Kevin Dunn, 2010-2026. MIT License.
"""A deterministic fed-batch bioreactor simulator with tunable disturbance channels.

Purpose
-------
This module is the quantitative baseline for batch trajectory adaptation and,
later, batch mid-course correction. It makes three claims demonstrable, with
numbers a reader can reproduce exactly from a seed:

1. Replaying a "golden batch" trajectory open-loop does not reproduce the
   golden outcome, because each batch carries its own disturbances.
2. The spread persists even when the measured initial conditions are held
   identical, because disturbances also arise *during* the batch.
3. The quality variance under replay therefore decomposes into a bucket that
   pre-batch (feedforward) adaptation can address, a bucket that only
   mid-course (feedback) correction can address, and a noise floor.

The model
---------
A 10-day fed-batch bioreactor with biomass ``X``, substrate ``S``, product
(titer) ``P`` and working volume ``V``. The specific growth rate follows the
gamma-concept model of Rosso et al. (1995): a cardinal temperature model with
inflection (CTMI) multiplied by a cardinal pH model (CPM), each equal to 1 at
its optimum and exactly 0 outside its cardinal range, further multiplied by
Monod substrate limitation, a within-batch disturbance ``phi(t)``, an
initial-condition growth-inhibition factor and an oxygen-limitation factor::

    mu_pot = mu_opt * gamma_T(T) * gamma_pH(pH) * S / (K_S + S) * phi * inh
    f_O2   = smooth_min(1, our_capacity / ((o2_yield * mu_pot + o2_m) * X))
    mu     = mu_pot * f_O2

    k_d = k_d0 * (1 + exp((T - temp_death) / width_death))
          + k_d_starv * k_s_starv / (k_s_starv + S)
          + k_d_hyp * (1 - f_O2)
    q_P = (alpha_lp * mu + beta_lp * gamma_q(T) * gamma_pH(pH) * phi * f_O2)
          * S / (k_sp + S)

    dX/dt = mu * X - k_d * X - (F/V) * X
    dS/dt = -(mu / yield_xs + maintenance) * X - q_P * X / yield_ps
            + (F/V) * (feed_substrate - S)
    dP/dt = q_P * X - (F/V) * P
    dV/dt = F

The couplings are chosen so the industrially standard biphasic temperature
shift (a warm growth phase, then a mild-hypothermia production phase) is a
genuine optimum rather than decoration, and so the *right* schedule depends
on the batch's own conditions:

- The oxygen-transfer capacity caps the biomass pile the reactor can sustain,
  and the cap is temperature-dependent because warmer cells each demand more
  oxygen. Overshooting the cap is punished by hypoxic death, an irreversible
  loss, so the optimal pile size is interior.
- Strong hypothermic growth arrest below about 28 degC (the reason biphasic
  mammalian-cell culture works) freezes the pile once the batch shifts cold,
  so the *timing* of the shift decides the production capacity, and the right
  timing depends on the inoculum and the growth rate the raw-material lot
  supports.
- Production consumes substrate (``yield_ps``) and stalls when it runs out
  (``k_sp``); starvation kills cells (``k_d_starv``). Residual growth at
  too-warm production temperatures burns the feed that production needs,
  which is what places the production hold at an interior optimum
  (``temp_production``) below the isolated productivity optimum
  ``temp_q_opt``.
- The growth-inhibition factor ``inh`` (from the raw-material impurity latent
  factor) acts on growth only, so an inhibited lot needs a longer warm growth
  phase, not a scaled-down copy of the same batch.

The gamma hypothesis (temperature and pH acting independently on growth) is a
modelling choice, not an established fact: published work finds the cardinal
temperatures themselves shift with pH. It is adopted here because it keeps the
true optimum interpretable in closed form.

Sensitivity by construction
---------------------------
A simulator whose quality output moves visibly when an input moves by less
than an instrument can resolve is not believable. Three structural properties
keep this model insensitive to meaningless input changes while still
responsive to real ones:

- Quality is a time integral of bounded, smooth rates over 240 integration
  steps, so high-frequency input noise averages out and only sustained
  deviations accumulate.
- The nominal recipe sits at stationary points of the response: pH is held at
  its cardinal optimum, and the production-phase temperature hold is the
  interior optimum of the hold-temperature response. Instrument-scale
  deviations around a stationary point are second order.
- Sustained multi-degree deviations, by contrast, cross real mechanisms
  (hypothermic growth arrest on the cold side, feed-burning residual growth
  and hypoxia on the warm side), so they cost percent-level titer, as they
  should.

:meth:`BioreactorSimulator.sensitivity_budget` computes the resulting budget
from the live configuration, so the realism claim can be checked rather than
taken on trust. Indicative values for the default configuration: zero-mean
control-loop noise at instrument scale (sd 0.15 degC, 0.02 pH) moves the
final titer by under 0.3% (standard deviation), a sustained 0.1 degC bias
costs about 0.2%, a sustained 0.02 pH bias is invisible (about 0.01%), while
a sustained 1 degC bias costs 7 to 11% and a sustained 2 degC bias about
20%. Around the operating point overheating costs more than the same
excursion undercooling (feed burn plus hypoxia); far from it undercooling
costs more (full growth arrest).

Disturbance channels
--------------------
Three independently tunable channels, each with a scale that can be set to
zero (``ic_scale``, ``within_batch_scale``, ``noise_scale``):

1. Measured initial conditions: an 11-variable upstream ``Z`` block generated
   from three latent factors (seed viability, medium richness, inhibitor
   level) with three cluster centres, the feed classes A, B and C. Only the
   three latent factors drive the process, so ``Z`` also carries directions
   that do not matter, as real upstream data do.
2. Within-batch, unmeasured but observable: an Ornstein-Uhlenbeck process on
   ``log phi(t)`` with a correlation time comparable to the batch length,
   plus a per-batch feed-rate drift. It is not predictable from ``Z``, is
   continuously visible in the oxygen and offgas trajectories, and its
   present partly predicts its own future, which is exactly why observing
   the running batch helps.
3. Control-loop and measurement noise: autocorrelated zero-mean deviation of
   the realised trajectory from its setpoints, plus independent measurement
   noise on every recorded tag, at instrument scale.

All random draws are made on every call and multiplied by their channel
scale, so setting a scale to zero removes that channel without changing the
draw sequence of the others: the same seed with a channel switched off is a
true counterfactual for the same batch.

References
----------
Rosso, L., Lobry, J. R., Bajard, S. and Flandrois, J. P. (1995), "Convenient
model to describe the combined effects of temperature and pH on microbial
growth", *Applied and Environmental Microbiology*, **61** (2), 610-616.

Luedeking, R. and Piret, E. L. (1959), "A kinetic study of the lactic acid
fermentation", *Journal of Biochemical and Microbiological Technology and
Engineering*, **1** (4), 393-412.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from process_improve._random import check_random_state

logger = logging.getLogger(__name__)

# Names of the 11 upstream (initial-condition) variables in the Z block.
UPSTREAM_VARIABLE_NAMES: tuple[str, ...] = (
    "seed_viability_pct",
    "seed_age_h",
    "inoculum_density_e6_per_ml",
    "media_glucose_g_L",
    "media_glutamine_mM",
    "media_osmolality_mOsm_kg",
    "media_lot_age_d",
    "trace_metal_index",
    "impurity_index",
    "water_conductivity_uS_cm",
    "supplier_lot_score",
)

# Latent factor names, in order: they are the only directions in Z that
# actually drive the process.
LATENT_FACTOR_NAMES: tuple[str, ...] = ("seed_viability", "medium_richness", "inhibitor_level")

# Fixed loading matrix (3 latent factors x 11 standardized upstream
# variables). These are constants, not draws, so the Z-to-process mapping is
# identical across sessions, platforms and package versions.
_Z_LOADINGS: np.ndarray = np.array(
    [
        # viability: seed quality and inoculum variables
        [0.90, -0.70, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30],
        # richness: medium composition variables
        [0.00, 0.00, 0.00, 0.90, 0.85, 0.50, -0.60, 0.55, 0.00, 0.00, 0.00],
        # inhibitor: impurity-related variables
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.90, 0.60, -0.50],
    ]
)

# Means and standard deviations that place the standardized Z variables on
# plausible physical scales (units are in the variable names).
_Z_MEANS: np.ndarray = np.array([92.0, 48.0, 2.5, 6.0, 4.0, 300.0, 30.0, 1.0, 1.0, 1.2, 8.0])
_Z_SDS: np.ndarray = np.array([4.0, 10.0, 0.5, 0.5, 0.5, 15.0, 12.0, 0.2, 0.4, 0.3, 1.0])

# Residual (non-latent) standardized noise on each Z variable.
_Z_RESIDUAL_SD: float = 0.35

# Cluster centres of the three feed-disturbance classes in latent space, and
# the within-cluster standard deviation.
_CLASS_CENTRES: dict[str, tuple[float, float, float]] = {
    "A": (0.8, 0.6, -0.8),
    "B": (0.0, 0.0, 0.0),
    "C": (-0.8, -0.6, 1.0),
}
_CLASS_SD: float = 0.55

_DEFAULT_CLASS_PROPORTIONS: dict[str, float] = {"A": 0.40, "B": 0.35, "C": 0.25}

_TAG_COLUMNS: tuple[str, ...] = ("pH", "temperature", "dissolved_oxygen", "offgas_co2", "volume")
_TRAJECTORY_COLUMNS: tuple[str, ...] = ("pH", "temperature")
_POLICIES: tuple[str, ...] = ("replay", "historical", "adapted")


def cardinal_temperature(temperature: float | np.ndarray, t_min: float, t_opt: float, t_max: float) -> np.ndarray:
    """Cardinal temperature model with inflection (CTMI) of Rosso et al. (1995).

    Equal to 1 at ``t_opt``, exactly 0 at and beyond the cardinal bounds, and
    smoothly curved in between with an inflection in the suboptimal range.

    Parameters
    ----------
    temperature : float or np.ndarray
        Temperature(s) [degC].
    t_min, t_opt, t_max : float
        Cardinal temperatures [degC]: no growth at or below ``t_min``, maximum
        growth at ``t_opt``, no growth at or above ``t_max``.

    Returns
    -------
    np.ndarray
        The dimensionless growth factor gamma_T in [0, 1], with the same shape
        as ``temperature``.
    """
    temp = np.asarray(temperature, dtype=float)
    numerator = (temp - t_max) * (temp - t_min) ** 2
    denominator = (t_opt - t_min) * ((t_opt - t_min) * (temp - t_opt) - (t_opt - t_max) * (t_opt + t_min - 2.0 * temp))
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where((temp <= t_min) | (temp >= t_max), 0.0, numerator / denominator)
    return np.clip(gamma, 0.0, 1.0)


def cardinal_ph(ph: float | np.ndarray, ph_min: float, ph_opt: float, ph_max: float) -> np.ndarray:
    """Cardinal pH model (CPM) of Rosso et al. (1995).

    Equal to 1 at ``ph_opt`` and exactly 0 at and beyond the cardinal bounds.

    Parameters
    ----------
    ph : float or np.ndarray
        pH value(s) [-].
    ph_min, ph_opt, ph_max : float
        Cardinal pH values: no growth at or below ``ph_min``, maximum growth
        at ``ph_opt``, no growth at or above ``ph_max``.

    Returns
    -------
    np.ndarray
        The dimensionless growth factor gamma_pH in [0, 1], with the same
        shape as ``ph``.
    """
    ph_arr = np.asarray(ph, dtype=float)
    numerator = (ph_arr - ph_min) * (ph_arr - ph_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(
            (ph_arr <= ph_min) | (ph_arr >= ph_max),
            0.0,
            numerator / (numerator - (ph_arr - ph_opt) ** 2),
        )
    return np.clip(gamma, 0.0, 1.0)


def _cardinal_temperature_f(temp: float, t_min: float, t_opt: float, t_max: float) -> float:
    """Scalar-float CTMI for the integration hot loop (see :func:`cardinal_temperature`)."""
    if temp <= t_min or temp >= t_max:
        return 0.0
    numerator = (temp - t_max) * (temp - t_min) ** 2
    denominator = (t_opt - t_min) * ((t_opt - t_min) * (temp - t_opt) - (t_opt - t_max) * (t_opt + t_min - 2.0 * temp))
    return min(max(numerator / denominator, 0.0), 1.0)


def _cardinal_ph_f(ph: float, ph_min: float, ph_opt: float, ph_max: float) -> float:
    """Scalar-float CPM for the integration hot loop (see :func:`cardinal_ph`)."""
    if ph <= ph_min or ph >= ph_max:
        return 0.0
    numerator = (ph - ph_min) * (ph - ph_max)
    return min(max(numerator / (numerator - (ph - ph_opt) ** 2), 0.0), 1.0)


@dataclasses.dataclass(frozen=True)
class BioreactorConfig:
    """Kinetic, operating and disturbance parameters of the simulated bioreactor.

    All parameters have defaults chosen so the nominal batch lands in a
    physiologically plausible range for a pilot-scale mammalian-cell fed-batch
    process (peak biomass a few g/L, final titer of order 10 g/L). Units are
    given per field. The dataclass is frozen; derive variants with
    :func:`dataclasses.replace`.

    Parameters
    ----------
    mu_opt : float
        Maximum specific growth rate at the cardinal optima [1/day].
    k_s : float
        Monod half-saturation constant for the substrate [g/L].
    yield_xs : float
        Biomass yield on substrate [g biomass / g substrate].
    maintenance : float
        Maintenance substrate consumption [g substrate / g biomass / day].
    alpha_lp, beta_lp : float
        Luedeking-Piret coefficients: growth-associated product yield
        [g product / g biomass] and non-growth-associated specific
        productivity [g product / g biomass / day].
    yield_ps : float
        Product yield on substrate [g product / g substrate]: production
        consumes substrate, so a large biomass pile competes with its own
        productivity for feed.
    k_sp : float
        Half-saturation substrate concentration for product formation [g/L]:
        production stalls when the substrate runs out.
    k_d_starv : float
        Additional death rate under full starvation [1/day].
    k_s_starv : float
        Substrate concentration at which the starvation death rate reaches
        half its maximum [g/L].
    temp_min, temp_opt, temp_max : float
        Cardinal temperatures for growth [degC].
    k_d0 : float
        Baseline first-order death rate [1/day].
    temp_death, width_death : float
        Onset temperature [degC] and width [degC] of the exponential rise in
        the death rate.
    temp_q_min, temp_q_opt, temp_q_max : float
        Cardinal temperatures for the non-growth-associated productivity
        [degC]; ``temp_q_opt`` sits below ``temp_opt``, which is what makes
        the biphasic temperature shift a real trade-off.
    ph_min, ph_opt, ph_max : float
        Cardinal pH values, shared by growth and productivity.
    batch_days : float
        Batch duration [day].
    samples_per_batch : int
        Number of recorded samples (and setpoint intervals) per batch.
    steps_per_day : int
        Integration steps per day for the fixed-step RK4 integrator.
    feed_rate : float
        Nominal constant feed rate [L/day].
    feed_substrate : float
        Substrate concentration in the feed [g/L].
    volume_initial : float
        Working volume at inoculation [L].
    biomass_initial, substrate_initial : float
        Nominal initial biomass and substrate concentrations [g/L]; the
        measured initial-condition channel perturbs these per batch.
    temp_bounds, ph_bounds : tuple[float, float]
        Recipe-allowed operating window for the setpoints; requested and
        realised trajectories are validated or clipped against these.
    shift_start_day, shift_end_day : float
        Start and end [day] of the nominal biphasic temperature ramp from
        ``temp_opt`` down to ``temp_production``.
    temp_production : float
        The production-phase hold temperature of the nominal recipe [degC].
        It sits below the isolated productivity optimum ``temp_q_opt``
        because residual growth at warmer temperatures consumes the feed
        that production needs; cold enough to arrest growth is what a
        sensible recipe holds.
    ic_scale : float
        Scale of the measured initial-condition channel; 0 switches it off.
    within_batch_scale : float
        Scale of the unmeasured within-batch channel; 0 switches it off.
    noise_scale : float
        Scale of the control-loop and measurement noise channel; 0 switches
        it off.
    ou_tau_days : float
        Correlation time of the Ornstein-Uhlenbeck disturbance on
        ``log phi(t)`` [day].
    ou_sd : float
        Stationary standard deviation of ``log phi(t)`` [-].
    feed_drift_sd : float
        Per-batch fractional standard deviation of the realised feed rate [-].
    control_sd_temp, control_sd_ph : float
        Stationary standard deviation of the realised-minus-setpoint control
        error for temperature [degC] and pH [-].
    control_tau_h : float
        Correlation time of the control error [hour].
    meas_sd_temp, meas_sd_ph, meas_sd_do, meas_sd_co2, meas_sd_volume : float
        Measurement noise standard deviations on the recorded tags
        [degC, -, % saturation, % offgas, L].
    o2_yield : float
        Oxygen demand per unit growth [g O2 / g biomass].
    o2_maintenance : float
        Maintenance oxygen uptake [g O2 / g biomass / day].
    our_capacity : float
        Maximum oxygen transfer the sparger and agitation can supply
        [g O2 / L / day]. This is the binding constraint of the process: it
        caps the biomass pile the reactor can sustain, and the cap is
        temperature-dependent because warmer cells demand more oxygen each.
    k_d_hyp : float
        Additional death rate under full oxygen limitation [1/day]:
        overshooting the sustainable biomass pile is an irreversible loss,
        which is what gives the optimal schedule curvature on both sides.
    rq : float
        Respiratory quotient, the CO2 evolved per O2 consumed [g/g].
    co2_gain : float
        Offgas CO2 percentage per unit CO2 evolution rate [% / (g/L/day)].
    co2_inlet_pct : float
        CO2 percentage of the inlet gas [%].
    """

    # Kinetics
    mu_opt: float = 0.80
    k_s: float = 0.20
    yield_xs: float = 0.40
    maintenance: float = 0.02
    alpha_lp: float = 0.30
    beta_lp: float = 0.35
    yield_ps: float = 0.90
    k_sp: float = 0.10
    k_d_starv: float = 0.08
    k_s_starv: float = 0.12
    # Cardinal temperatures for growth. temp_min models the strong hypothermic
    # growth arrest that biphasic mammalian-cell culture relies on: growth is
    # essentially zero at 28 degC while productivity remains high.
    temp_min: float = 27.5
    temp_opt: float = 36.8
    temp_max: float = 41.5
    # Thermal death
    k_d0: float = 0.015
    temp_death: float = 39.0
    width_death: float = 0.9
    # Cardinal temperatures for non-growth-associated productivity
    temp_q_min: float = 22.0
    temp_q_opt: float = 31.5
    temp_q_max: float = 40.5
    # Cardinal pH
    ph_min: float = 6.3
    ph_opt: float = 7.10
    ph_max: float = 7.9
    # Operation
    batch_days: float = 10.0
    samples_per_batch: int = 20
    steps_per_day: int = 24
    feed_rate: float = 0.055
    feed_substrate: float = 50.0
    volume_initial: float = 1.0
    biomass_initial: float = 0.30
    substrate_initial: float = 5.0
    temp_bounds: tuple[float, float] = (28.0, 39.0)
    ph_bounds: tuple[float, float] = (6.6, 7.6)
    shift_start_day: float = 3.0
    shift_end_day: float = 4.5
    temp_production: float = 29.05
    # Disturbance channel scales
    ic_scale: float = 1.0
    within_batch_scale: float = 1.0
    noise_scale: float = 1.0
    # Within-batch channel
    ou_tau_days: float = 6.0
    ou_sd: float = 0.15
    feed_drift_sd: float = 0.06
    # Control-loop error
    control_sd_temp: float = 0.15
    control_sd_ph: float = 0.02
    control_tau_h: float = 6.0
    # Measurement noise
    meas_sd_temp: float = 0.05
    meas_sd_ph: float = 0.01
    meas_sd_do: float = 1.0
    meas_sd_co2: float = 0.08
    meas_sd_volume: float = 0.005
    # Oxygen transfer and gas-phase model
    o2_yield: float = 0.55
    o2_maintenance: float = 0.02
    our_capacity: float = 1.85
    k_d_hyp: float = 0.40
    rq: float = 1.0
    co2_gain: float = 1.0
    co2_inlet_pct: float = 0.04

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate parameter relationships that the model depends on."""
        for name in (
            "mu_opt",
            "k_s",
            "yield_xs",
            "yield_ps",
            "k_sp",
            "k_s_starv",
            "batch_days",
            "feed_substrate",
            "volume_initial",
            "biomass_initial",
            "substrate_initial",
            "ou_tau_days",
            "control_tau_h",
            "our_capacity",
            "width_death",
        ):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite positive number; got {value!r}.")
        for name in (
            "maintenance",
            "alpha_lp",
            "beta_lp",
            "k_d0",
            "k_d_starv",
            "k_d_hyp",
            "feed_rate",
            "ic_scale",
            "within_batch_scale",
            "noise_scale",
            "ou_sd",
            "feed_drift_sd",
            "control_sd_temp",
            "control_sd_ph",
            "meas_sd_temp",
            "meas_sd_ph",
            "meas_sd_do",
            "meas_sd_co2",
            "meas_sd_volume",
        ):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite non-negative number; got {value!r}.")
        if not self.temp_min < self.temp_opt < self.temp_max:
            raise ValueError(
                "Cardinal temperatures must satisfy temp_min < temp_opt < temp_max; got "
                f"({self.temp_min}, {self.temp_opt}, {self.temp_max})."
            )
        if not self.temp_q_min < self.temp_q_opt < self.temp_q_max:
            raise ValueError(
                "Productivity cardinal temperatures must satisfy temp_q_min < temp_q_opt < temp_q_max; got "
                f"({self.temp_q_min}, {self.temp_q_opt}, {self.temp_q_max})."
            )
        if not self.ph_min < self.ph_opt < self.ph_max:
            raise ValueError(
                f"Cardinal pH values must satisfy ph_min < ph_opt < ph_max; got "
                f"({self.ph_min}, {self.ph_opt}, {self.ph_max})."
            )
        if self.samples_per_batch < 2:
            raise ValueError(f"samples_per_batch must be at least 2; got {self.samples_per_batch}.")
        if self.steps_per_day < 1:
            raise ValueError(f"steps_per_day must be at least 1; got {self.steps_per_day}.")
        n_steps = self.batch_days * self.steps_per_day
        if abs(n_steps / self.samples_per_batch - round(n_steps / self.samples_per_batch)) > 1e-9:
            raise ValueError(
                "batch_days * steps_per_day must be an integer multiple of samples_per_batch so every "
                f"sample falls on an integration step; got {n_steps} steps for {self.samples_per_batch} samples."
            )
        for name, (low, high) in (("temp_bounds", self.temp_bounds), ("ph_bounds", self.ph_bounds)):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"{name} must be a finite (low, high) pair with low < high; got ({low}, {high}).")
        if not self.temp_min < self.temp_bounds[0] < self.temp_bounds[1] < self.temp_max:
            raise ValueError(
                f"temp_bounds {self.temp_bounds} must lie strictly inside the cardinal window "
                f"({self.temp_min}, {self.temp_max})."
            )
        if not self.ph_min < self.ph_bounds[0] < self.ph_bounds[1] < self.ph_max:
            raise ValueError(
                f"ph_bounds {self.ph_bounds} must lie strictly inside the cardinal window "
                f"({self.ph_min}, {self.ph_max})."
            )
        if not 0 <= self.shift_start_day <= self.shift_end_day <= self.batch_days:
            raise ValueError(
                "The nominal temperature shift must satisfy 0 <= shift_start_day <= shift_end_day <= "
                f"batch_days; got ({self.shift_start_day}, {self.shift_end_day}, {self.batch_days})."
            )
        if not self.temp_bounds[0] <= self.temp_production <= self.temp_bounds[1]:
            raise ValueError(
                f"temp_production must lie within temp_bounds {self.temp_bounds}; got {self.temp_production}."
            )

    @property
    def n_steps(self) -> int:
        """Total number of RK4 integration steps per batch."""
        return round(self.batch_days * self.steps_per_day)

    @property
    def interval_days(self) -> float:
        """Duration of one setpoint interval (one recorded sample) [day]."""
        return self.batch_days / self.samples_per_batch

    @property
    def sample_days(self) -> np.ndarray:
        """Recording times, the end of each setpoint interval [day]."""
        return np.linspace(self.interval_days, self.batch_days, self.samples_per_batch)

    @property
    def interval_start_days(self) -> np.ndarray:
        """Setpoint interval start times, the index of a trajectory frame [day]."""
        return np.linspace(0.0, self.batch_days - self.interval_days, self.samples_per_batch)


def _latent_effects(config: BioreactorConfig, latent: np.ndarray) -> tuple[float, float, float]:
    """Map the three latent initial-condition factors to their process effects.

    Parameters
    ----------
    config : BioreactorConfig
        Configuration supplying the nominal values and ``ic_scale``.
    latent : np.ndarray
        The three latent factor values (seed viability, medium richness,
        inhibitor level) in standardized units.

    Returns
    -------
    tuple[float, float, float]
        ``(biomass_initial, substrate_initial, inhibition)`` for this batch:
        the perturbed initial concentrations [g/L] and the multiplicative
        growth-inhibition factor in (0, 1]. The inhibition acts on growth
        only, which is why an inhibited lot calls for a different schedule
        (a longer warm growth phase) rather than a scaled-down copy.
    """
    scale = config.ic_scale
    viability, richness, inhibitor = (float(v) for v in latent)
    x0 = config.biomass_initial * min(max(1.0 + 0.18 * scale * viability, 0.40), 1.80)
    s0 = config.substrate_initial * min(max(1.0 + 0.20 * scale * richness, 0.50), 1.60)
    inhibition = min(max(1.0 - 0.12 * scale * max(inhibitor, 0.0), 0.55), 1.0)
    return x0, s0, inhibition


def _z_to_latent(z_row: np.ndarray) -> np.ndarray:
    """Recover the three latent factors from an 11-variable Z row.

    The Z block is generated as ``latent @ loadings`` in standardized space
    plus residual noise, so projecting a standardized row onto the pseudo-
    inverse of the (fixed, known) loading matrix recovers the latent factors.
    This makes the simulator a pure function of any user-supplied Z row, not
    only rows produced by :func:`sample_initial_conditions`.
    """
    z_std = (z_row - _Z_MEANS) / _Z_SDS
    return z_std @ np.linalg.pinv(_Z_LOADINGS)


def _coerce_z_row(initial_conditions: pd.Series | None) -> np.ndarray:
    """Validate a single batch's initial conditions and return the raw Z row.

    ``None`` means the nominal batch: every upstream variable at its mean, so
    all three latent factors are zero.
    """
    if initial_conditions is None:
        return _Z_MEANS.copy()
    if not isinstance(initial_conditions, pd.Series):
        raise TypeError(
            f"initial_conditions must be a pandas Series with the {len(UPSTREAM_VARIABLE_NAMES)} upstream "
            f"variables as its index, or None for the nominal batch; got {type(initial_conditions).__name__}."
        )
    missing = [name for name in UPSTREAM_VARIABLE_NAMES if name not in initial_conditions.index]
    if missing:
        raise ValueError(f"initial_conditions is missing upstream variables {missing}.")
    values = initial_conditions.reindex(list(UPSTREAM_VARIABLE_NAMES)).to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_conditions contains non-finite values; every upstream variable must be a number.")
    return values


def sample_initial_conditions(
    n_batches: int,
    *,
    proportions: dict[str, float] | None = None,
    ic_scale: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> Bunch:
    """Draw upstream (initial-condition) data for a campaign of batches.

    Each batch belongs to one of three feed-disturbance classes, A, B or C,
    with a cluster centre in the three-dimensional latent space (seed
    viability, medium richness, inhibitor level). The observed ``Z`` block has
    11 variables generated from those latent factors through a fixed loading
    matrix plus residual noise, so a PCA of ``Z`` recovers about three
    meaningful components and shows the three classes as clusters.

    Parameters
    ----------
    n_batches : int
        Number of batches to draw.
    proportions : dict[str, float], optional
        Expected class proportions keyed by ``"A"``, ``"B"``, ``"C"``. They
        must be non-negative and sum to a positive number (they are
        normalised). Default ``{"A": 0.40, "B": 0.35, "C": 0.25}``.
    ic_scale : float, default=1.0
        Scale of the initial-condition variation. 0 collapses every batch to
        the nominal upstream values while keeping the class labels.
    random_state : int, np.random.Generator, or None
        Seed or generator; see ``process_improve._random.check_random_state``.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``z`` (DataFrame, ``n_batches`` rows by 11 upstream
        variables, integer batch ids as index), ``classes`` (Series of "A" /
        "B" / "C" labels, same index) and ``latent`` (DataFrame of the three
        latent factor values, same index).
    """
    if not isinstance(n_batches, (int, np.integer)) or isinstance(n_batches, bool) or n_batches < 1:
        raise ValueError(f"n_batches must be a positive integer; got {n_batches!r}.")
    if not (isinstance(ic_scale, (int, float)) and math.isfinite(ic_scale) and ic_scale >= 0):
        raise ValueError(f"ic_scale must be a finite non-negative number; got {ic_scale!r}.")
    props = dict(_DEFAULT_CLASS_PROPORTIONS) if proportions is None else dict(proportions)
    unknown = sorted(set(props) - set(_CLASS_CENTRES))
    if unknown:
        raise ValueError(f"proportions has unknown class labels {unknown}; valid labels are ['A', 'B', 'C'].")
    weights = np.array([float(props.get(label, 0.0)) for label in sorted(_CLASS_CENTRES)])
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError(f"proportions must be non-negative with a positive sum; got {props!r}.")
    weights = weights / weights.sum()

    rng = check_random_state(random_state)
    labels_pool = sorted(_CLASS_CENTRES)
    labels = rng.choice(labels_pool, size=int(n_batches), p=weights)
    centres = np.array([_CLASS_CENTRES[label] for label in labels])
    latent = ic_scale * (centres + _CLASS_SD * rng.standard_normal((int(n_batches), 3)))
    z_std = latent @ _Z_LOADINGS + _Z_RESIDUAL_SD * ic_scale * rng.standard_normal((int(n_batches), len(_Z_MEANS)))
    z_values = _Z_MEANS + _Z_SDS * z_std

    index = pd.RangeIndex(1, int(n_batches) + 1, name="batch_id")
    return Bunch(
        z=pd.DataFrame(z_values, index=index, columns=list(UPSTREAM_VARIABLE_NAMES)),
        classes=pd.Series(labels, index=index, name="feed_class"),
        latent=pd.DataFrame(latent, index=index, columns=list(LATENT_FACTOR_NAMES)),
    )


class BioreactorSimulator:
    """A deterministic fed-batch bioreactor with tunable disturbance channels.

    See the module docstring for the model, its citations, and the design
    constraints. Given the same configuration, inputs and ``random_state``,
    every method reproduces its results exactly.

    Parameters
    ----------
    config : BioreactorConfig, optional
        The full parameter set; defaults to ``BioreactorConfig()``.

    Examples
    --------
    >>> from process_improve.simulation import BioreactorSimulator
    >>> sim = BioreactorSimulator()
    >>> golden = sim.golden_trajectory()
    >>> campaign = sim.simulate_campaign(50, policy="replay", trajectory=golden.trajectory, random_state=42)
    >>> float(campaign.quality["titer"].std()) > 0.0
    True
    """

    def __init__(self, config: BioreactorConfig | None = None) -> None:
        if config is None:
            config = BioreactorConfig()
        if not isinstance(config, BioreactorConfig):
            raise TypeError(f"config must be a BioreactorConfig or None; got {type(config).__name__}.")
        self.config = config

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------
    def nominal_trajectory(self) -> pd.DataFrame:
        """Return the nominal (recipe) setpoint schedule: biphasic temperature, pH held.

        Temperature holds the growth optimum until ``shift_start_day``, ramps
        linearly to the production hold ``temp_production`` by
        ``shift_end_day``, and holds it for the rest of the batch. pH is held
        at its cardinal optimum throughout, matching the industrial practice
        of holding pH and shifting temperature.

        Returns
        -------
        pd.DataFrame
            ``samples_per_batch`` rows indexed by setpoint interval start time [day],
            columns ``["pH", "temperature"]``. Each row is the setpoint held
            over the following interval (zero-order hold).
        """
        cfg = self.config
        days = cfg.interval_start_days
        if cfg.shift_end_day > cfg.shift_start_day:
            fraction = np.clip((days - cfg.shift_start_day) / (cfg.shift_end_day - cfg.shift_start_day), 0.0, 1.0)
        else:
            fraction = (days >= cfg.shift_start_day).astype(float)
        temperature = cfg.temp_opt - (cfg.temp_opt - cfg.temp_production) * fraction
        temperature = np.clip(temperature, cfg.temp_bounds[0], cfg.temp_bounds[1])
        ph = np.clip(np.full_like(days, cfg.ph_opt), cfg.ph_bounds[0], cfg.ph_bounds[1])
        return pd.DataFrame({"pH": ph, "temperature": temperature}, index=pd.Index(days, name="day"))

    def _validate_trajectory(self, trajectory: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Check a setpoint trajectory and return (ph, temperature) arrays."""
        cfg = self.config
        if not isinstance(trajectory, pd.DataFrame):
            raise TypeError(f"trajectory must be a pandas DataFrame; got {type(trajectory).__name__}.")
        missing = [c for c in _TRAJECTORY_COLUMNS if c not in trajectory.columns]
        if missing:
            raise ValueError(f"trajectory is missing columns {missing}; it needs {list(_TRAJECTORY_COLUMNS)}.")
        if len(trajectory) != cfg.samples_per_batch:
            raise ValueError(
                f"trajectory must have samples_per_batch = {cfg.samples_per_batch} rows; got {len(trajectory)}."
            )
        ph = trajectory["pH"].to_numpy(dtype=float)
        temperature = trajectory["temperature"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(ph)) and np.all(np.isfinite(temperature))):
            raise ValueError("trajectory contains non-finite values.")
        t_lo, t_hi = cfg.temp_bounds
        p_lo, p_hi = cfg.ph_bounds
        if np.any(temperature < t_lo - 1e-9) or np.any(temperature > t_hi + 1e-9):
            raise ValueError(f"trajectory temperature must lie within temp_bounds {cfg.temp_bounds} degC.")
        if np.any(ph < p_lo - 1e-9) or np.any(ph > p_hi + 1e-9):
            raise ValueError(f"trajectory pH must lie within ph_bounds {cfg.ph_bounds}.")
        return ph, temperature

    def _setpoints_hourly(self, ph: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand per-interval setpoints to the integration grid (zero-order hold).

        Returns arrays of length ``n_steps + 1``; entry ``i`` is the setpoint
        in force during step ``i`` (the final entry repeats the last setpoint).
        """
        steps_per_interval = self.config.n_steps // self.config.samples_per_batch
        ph_hourly = np.repeat(ph, steps_per_interval)
        temp_hourly = np.repeat(temperature, steps_per_interval)
        return (
            np.append(ph_hourly, ph_hourly[-1]),
            np.append(temp_hourly, temp_hourly[-1]),
        )

    # ------------------------------------------------------------------
    # Disturbance channels
    # ------------------------------------------------------------------
    def _draw_disturbances(self, rng: np.random.Generator) -> Bunch:
        """Draw one batch's worth of disturbance realisations.

        Every random quantity is drawn on every call, in a fixed order, and
        multiplied by its channel scale afterwards. Setting a scale to zero
        therefore removes the channel without altering the draws of any other
        channel: the same seed with one channel off is a true counterfactual.
        """
        cfg = self.config
        n_steps = cfg.n_steps

        # 1. Per-batch feed-rate drift (within-batch channel).
        feed_shock = float(rng.standard_normal())
        feed_scale = max(1.0 + cfg.feed_drift_sd * cfg.within_batch_scale * feed_shock, 0.50)

        # 2. Ornstein-Uhlenbeck path for log phi (within-batch channel).
        ou_draws = rng.standard_normal(n_steps + 1)
        dt = 1.0 / cfg.steps_per_day
        rho_ou = math.exp(-dt / cfg.ou_tau_days)
        innovation_sd = cfg.ou_sd * math.sqrt(1.0 - rho_ou * rho_ou)
        log_phi = np.empty(n_steps + 1)
        log_phi[0] = 0.0
        for i in range(1, n_steps + 1):
            log_phi[i] = rho_ou * log_phi[i - 1] + innovation_sd * ou_draws[i]
        phi = np.exp(cfg.within_batch_scale * log_phi)

        # 3. Control-loop error paths for temperature and pH (noise channel).
        rho_ctrl = math.exp(-1.0 / (cfg.control_tau_h * cfg.steps_per_day / 24.0))
        ctrl_sd_factor = math.sqrt(1.0 - rho_ctrl * rho_ctrl)
        temp_draws = rng.standard_normal(n_steps + 1)
        ph_draws = rng.standard_normal(n_steps + 1)
        temp_err = np.empty(n_steps + 1)
        ph_err = np.empty(n_steps + 1)
        temp_err[0] = 0.0
        ph_err[0] = 0.0
        for i in range(1, n_steps + 1):
            temp_err[i] = rho_ctrl * temp_err[i - 1] + ctrl_sd_factor * temp_draws[i]
            ph_err[i] = rho_ctrl * ph_err[i - 1] + ctrl_sd_factor * ph_draws[i]
        temp_err *= cfg.control_sd_temp * cfg.noise_scale
        ph_err *= cfg.control_sd_ph * cfg.noise_scale

        # 4. Measurement noise on the recorded tags (noise channel).
        meas = rng.standard_normal((cfg.samples_per_batch, len(_TAG_COLUMNS))) * cfg.noise_scale

        return Bunch(feed_scale=feed_scale, phi=phi, temp_err=temp_err, ph_err=ph_err, meas=meas)

    # ------------------------------------------------------------------
    # Core integration
    # ------------------------------------------------------------------
    def _integrate(  # noqa: PLR0913, PLR0915
        self,
        ph_hourly: np.ndarray,
        temp_hourly: np.ndarray,
        phi_hourly: np.ndarray,
        x0: float,
        s0: float,
        inhibition: float,
        feed_scale: float,
    ) -> np.ndarray:
        """Fixed-step RK4 integration of the four-state model.

        The inner loop runs on plain Python floats: on states this small that
        is roughly two orders of magnitude faster than numpy scalar
        arithmetic, and it keeps results bit-identical across processes.

        Returns
        -------
        np.ndarray
            States at every grid point, shape ``(n_steps + 1, 4)`` with
            columns biomass, substrate, titer, volume.
        """
        cfg = self.config
        n_steps = cfg.n_steps
        h = 1.0 / cfg.steps_per_day

        mu_opt = cfg.mu_opt
        k_s = cfg.k_s
        inv_yield = 1.0 / cfg.yield_xs
        inv_yield_ps = 1.0 / cfg.yield_ps
        maintenance = cfg.maintenance
        alpha_lp = cfg.alpha_lp
        beta_lp = cfg.beta_lp
        k_sp = cfg.k_sp
        k_d_starv = cfg.k_d_starv
        k_s_starv = cfg.k_s_starv
        o2_yield = cfg.o2_yield
        o2_maintenance = cfg.o2_maintenance
        our_capacity = cfg.our_capacity
        k_d_hyp = cfg.k_d_hyp
        t_min, t_opt, t_max = cfg.temp_min, cfg.temp_opt, cfg.temp_max
        tq_min, tq_opt, tq_max = cfg.temp_q_min, cfg.temp_q_opt, cfg.temp_q_max
        p_min, p_opt, p_max = cfg.ph_min, cfg.ph_opt, cfg.ph_max
        k_d0 = cfg.k_d0
        temp_death = cfg.temp_death
        width_death = cfg.width_death
        feed = cfg.feed_rate * feed_scale
        s_feed = cfg.feed_substrate

        def rhs(x: float, s: float, p: float, v: float, temp: float, ph: float, phi: float) -> tuple:  # noqa: PLR0913
            gamma_ph = _cardinal_ph_f(ph, p_min, p_opt, p_max)
            mu_pot = mu_opt * _cardinal_temperature_f(temp, t_min, t_opt, t_max) * gamma_ph
            # The impurity-driven inhibition factor acts on growth only; the
            # within-batch disturbance phi acts on the whole metabolism.
            mu_pot *= s / (k_s + s) * phi * inhibition
            # Oxygen limitation: the pile's demand against the reactor's
            # transfer capacity. f is a smooth min(1, supply/demand), so
            # metabolism throttles as the pile approaches the ceiling, and
            # hypoxia kills cells beyond it. This mirrors _gas_tags.
            demand = (o2_yield * mu_pot + o2_maintenance) * x
            a = our_capacity / max(demand, 1e-12)
            f_o2 = a / (1.0 + a**6) ** (1.0 / 6.0)
            mu = mu_pot * f_o2
            k_d = (
                k_d0 * (1.0 + math.exp((temp - temp_death) / width_death))
                + k_d_starv * k_s_starv / (k_s_starv + s)
                + k_d_hyp * (1.0 - f_o2)
            )
            # Production consumes substrate and stalls when it runs out; it is
            # oxygen-limited like growth (the growth-associated term already
            # carries phi and f_o2 inside mu).
            q_p = (
                alpha_lp * mu + beta_lp * _cardinal_temperature_f(temp, tq_min, tq_opt, tq_max) * gamma_ph * phi * f_o2
            ) * (s / (k_sp + s))
            dilution = feed / v
            dx = mu * x - k_d * x - dilution * x
            ds = -(mu * inv_yield + maintenance) * x - q_p * x * inv_yield_ps + dilution * (s_feed - s)
            dp = q_p * x - dilution * p
            dv = feed
            return dx, ds, dp, dv

        out = np.empty((n_steps + 1, 4))
        x, s, p, v = x0, s0, 0.0, cfg.volume_initial
        out[0] = (x, s, p, v)
        hh = 0.5 * h
        sixth_h = h / 6.0
        for i in range(n_steps):
            # Every input (setpoint schedule, control error, phi) is held
            # constant over its integration step, so setpoint jumps land
            # exactly on step boundaries and RK4 keeps its full order.
            # Blending across the boundary instead degrades the whole
            # integration to first order.
            temp = temp_hourly[i]
            ph = ph_hourly[i]
            phi = phi_hourly[i]

            k1 = rhs(x, s, p, v, temp, ph, phi)
            k2 = rhs(x + hh * k1[0], s + hh * k1[1], p + hh * k1[2], v + hh * k1[3], temp, ph, phi)
            k3 = rhs(x + hh * k2[0], s + hh * k2[1], p + hh * k2[2], v + hh * k2[3], temp, ph, phi)
            k4 = rhs(x + h * k3[0], s + h * k3[1], p + h * k3[2], v + h * k3[3], temp, ph, phi)

            x += sixth_h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            s += sixth_h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            p += sixth_h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            v += sixth_h * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

            # Physical floors: concentrations cannot go negative; volume
            # cannot vanish. The floors only bind in pathological corners of
            # parameter space (the RK4 step is far smaller than any dynamic
            # timescale at the defaults).
            x = max(x, 1e-9)
            s = max(s, 0.0)
            p = max(p, 0.0)
            v = max(v, 1e-9)
            out[i + 1] = (x, s, p, v)
        return out

    def _gas_tags(
        self, states: np.ndarray, temp: np.ndarray, ph: np.ndarray, phi: np.ndarray, inhibition: float
    ) -> Bunch:
        """Dissolved oxygen [%] and offgas CO2 [%] at every grid point.

        Uses the same specific growth rate as the integrator (including the
        within-batch disturbance and the growth inhibition factor), so the
        gas trajectories reflect what the cells are actually doing. This is
        what makes the unmeasured disturbance channel observable.
        """
        cfg = self.config
        biomass = states[:, 0]
        substrate = states[:, 1]
        gamma_t = cardinal_temperature(temp, cfg.temp_min, cfg.temp_opt, cfg.temp_max)
        gamma_p = cardinal_ph(ph, cfg.ph_min, cfg.ph_opt, cfg.ph_max)
        monod = substrate / (cfg.k_s + substrate)
        # Mirrors the oxygen-limitation calculation in _integrate's rhs.
        mu_pot = cfg.mu_opt * gamma_t * gamma_p * monod * phi * inhibition
        demand = (cfg.o2_yield * mu_pot + cfg.o2_maintenance) * biomass
        a = cfg.our_capacity / np.maximum(demand, 1e-12)
        f_o2 = a / (1.0 + a**6) ** (1.0 / 6.0)
        our = (cfg.o2_yield * mu_pot * f_o2 + cfg.o2_maintenance) * biomass
        dissolved_oxygen = 100.0 * np.clip(1.0 - our / cfg.our_capacity, 0.03, 1.0)
        offgas_co2 = cfg.co2_inlet_pct + cfg.co2_gain * cfg.rq * our
        return Bunch(dissolved_oxygen=dissolved_oxygen, offgas_co2=offgas_co2)

    # ------------------------------------------------------------------
    # Public simulation entry points
    # ------------------------------------------------------------------
    def simulate_batch(
        self,
        initial_conditions: pd.Series | None = None,
        trajectory: pd.DataFrame | None = None,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> Bunch:
        """Simulate one batch under given initial conditions and setpoints.

        Parameters
        ----------
        initial_conditions : pd.Series, optional
            One row of the upstream ``Z`` block (index: the 11 upstream
            variable names). ``None`` runs the nominal batch (all upstream
            variables at their means).
        trajectory : pd.DataFrame, optional
            Setpoint schedule with ``samples_per_batch`` rows and columns
            ``["pH", "temperature"]``; each row is held over its interval
            (zero-order hold). ``None`` uses :meth:`nominal_trajectory`.
        random_state : int, np.random.Generator, or None
            Seed or generator for the disturbance and noise draws.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``tags``: DataFrame, ``samples_per_batch`` rows indexed by sample time
              [day], columns ``["pH", "temperature", "dissolved_oxygen",
              "offgas_co2", "volume"]``: what the historian records,
              including measurement noise.
            - ``titer``: float, the final product concentration [g/L].
            - ``states``: DataFrame on the integration grid (indexed by day)
              with columns ``["biomass", "substrate", "titer", "volume",
              "phi", "temperature", "pH"]``: the noise-free god view,
              including the unmeasured disturbance ``phi``.
            - ``realised_trajectory``: DataFrame like ``tags`` but only
              ``["pH", "temperature"]`` and without measurement noise: what
              the control loops actually delivered.
            - ``initial_conditions``: Series, the ``Z`` row used.
        """
        cfg = self.config
        z_row = _coerce_z_row(initial_conditions)
        if trajectory is None:
            trajectory = self.nominal_trajectory()
        ph_set, temp_set = self._validate_trajectory(trajectory)
        rng = check_random_state(random_state)

        latent = _z_to_latent(z_row)
        x0, s0, inhibition = _latent_effects(cfg, latent)
        disturbances = self._draw_disturbances(rng)

        ph_hourly, temp_hourly = self._setpoints_hourly(ph_set, temp_set)
        temp_real = np.clip(temp_hourly + disturbances.temp_err, cfg.temp_bounds[0], cfg.temp_bounds[1])
        ph_real = np.clip(ph_hourly + disturbances.ph_err, cfg.ph_bounds[0], cfg.ph_bounds[1])

        states = self._integrate(ph_real, temp_real, disturbances.phi, x0, s0, inhibition, disturbances.feed_scale)
        gas = self._gas_tags(states, temp_real, ph_real, disturbances.phi, inhibition)

        grid_days = np.linspace(0.0, cfg.batch_days, cfg.n_steps + 1)
        sample_idx = np.round(cfg.sample_days * cfg.steps_per_day).astype(int)

        realised = pd.DataFrame(
            {"pH": ph_real[sample_idx], "temperature": temp_real[sample_idx]},
            index=pd.Index(cfg.sample_days, name="day"),
        )
        meas = disturbances.meas
        tags = pd.DataFrame(
            {
                "pH": ph_real[sample_idx] + cfg.meas_sd_ph * meas[:, 0],
                "temperature": temp_real[sample_idx] + cfg.meas_sd_temp * meas[:, 1],
                "dissolved_oxygen": gas.dissolved_oxygen[sample_idx] + cfg.meas_sd_do * meas[:, 2],
                "offgas_co2": gas.offgas_co2[sample_idx] + cfg.meas_sd_co2 * meas[:, 3],
                "volume": states[sample_idx, 3] + cfg.meas_sd_volume * meas[:, 4],
            },
            index=pd.Index(cfg.sample_days, name="day"),
        )
        states_frame = pd.DataFrame(
            {
                "biomass": states[:, 0],
                "substrate": states[:, 1],
                "titer": states[:, 2],
                "volume": states[:, 3],
                "phi": disturbances.phi,
                "temperature": temp_real,
                "pH": ph_real,
            },
            index=pd.Index(grid_days, name="day"),
        )
        return Bunch(
            tags=tags,
            titer=float(states[-1, 2]),
            states=states_frame,
            realised_trajectory=realised,
            initial_conditions=pd.Series(z_row, index=list(UPSTREAM_VARIABLE_NAMES), name="initial_conditions"),
        )

    def _deterministic_titer(self, latent: np.ndarray, ph_set: np.ndarray, temp_set: np.ndarray) -> float:
        """Return the final titer with every disturbance and noise channel off (the god view)."""
        cfg = self.config
        x0, s0, inhibition = _latent_effects(cfg, latent)
        ph_hourly, temp_hourly = self._setpoints_hourly(ph_set, temp_set)
        phi = np.ones(cfg.n_steps + 1)
        states = self._integrate(ph_hourly, temp_hourly, phi, x0, s0, inhibition, 1.0)
        return float(states[-1, 2])

    def optimal_trajectory(
        self,
        initial_conditions: pd.Series | None = None,
        *,
        n_knots: int = 4,
        n_starts: int = 5,
        random_state: int | np.random.Generator | None = 0,
    ) -> Bunch:
        """Find the true optimal setpoint schedule for a batch's initial conditions.

        Maximises the deterministic (disturbance-free) final titer over pH and
        temperature schedules parameterised by ``n_knots`` values each,
        linearly interpolated across the batch and held per interval. Because
        the optimiser queries the simulator's own model, the result is the
        *true* optimum for these initial conditions: the ceiling any
        data-driven scheme can be scored against.

        Parameters
        ----------
        initial_conditions : pd.Series, optional
            One upstream ``Z`` row; ``None`` for the nominal batch.
        n_knots : int, default=4
            Number of knots per manipulated variable.
        n_starts : int, default=5
            Multi-start count: the nominal trajectory plus ``n_starts - 1``
            random starts inside the operating bounds.
        random_state : int, np.random.Generator, or None, default=0
            Seed for the random starts; the default makes the search
            reproducible without an argument.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``trajectory`` (DataFrame in the same layout as
            :meth:`nominal_trajectory`), ``titer`` (float, the deterministic
            titer of that trajectory) and ``optimizer_success`` (bool).
        """
        from scipy import optimize  # noqa: PLC0415  (scipy is not a declared runtime dependency)

        cfg = self.config
        if not isinstance(n_knots, (int, np.integer)) or isinstance(n_knots, bool) or n_knots < 2:
            raise ValueError(f"n_knots must be an integer of at least 2; got {n_knots!r}.")
        if not isinstance(n_starts, (int, np.integer)) or isinstance(n_starts, bool) or n_starts < 1:
            raise ValueError(f"n_starts must be a positive integer; got {n_starts!r}.")
        z_row = _coerce_z_row(initial_conditions)
        latent = _z_to_latent(z_row)
        rng = check_random_state(random_state)

        knot_days = np.linspace(0.0, cfg.batch_days, int(n_knots))
        interval_days = cfg.interval_start_days

        def expand(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ph_knots, temp_knots = knots[: int(n_knots)], knots[int(n_knots) :]
            ph_set = np.interp(interval_days, knot_days, ph_knots)
            temp_set = np.interp(interval_days, knot_days, temp_knots)
            return ph_set, temp_set

        def negative_titer(knots: np.ndarray) -> float:
            ph_set, temp_set = expand(knots)
            return -self._deterministic_titer(latent, ph_set, temp_set)

        p_lo, p_hi = cfg.ph_bounds
        t_lo, t_hi = cfg.temp_bounds
        bounds = [(p_lo, p_hi)] * int(n_knots) + [(t_lo, t_hi)] * int(n_knots)

        nominal = self.nominal_trajectory()
        start_nominal = np.concatenate(
            [
                np.interp(knot_days, interval_days, nominal["pH"].to_numpy()),
                np.interp(knot_days, interval_days, nominal["temperature"].to_numpy()),
            ]
        )
        starts = [start_nominal]
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        starts.extend(rng.uniform(lows, highs) for _ in range(int(n_starts) - 1))

        best_result = None
        for start in starts:
            result = optimize.minimize(negative_titer, start, method="SLSQP", bounds=bounds)
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        if best_result is None:  # pragma: no cover - n_starts >= 1 guarantees a result
            raise RuntimeError("internal: the optimiser produced no result; this is a bug.")

        ph_set, temp_set = expand(best_result.x)
        trajectory = pd.DataFrame(
            {"pH": ph_set, "temperature": temp_set},
            index=pd.Index(interval_days, name="day"),
        )
        logger.debug(
            "optimal_trajectory: best titer %.4f g/L after %d starts (success=%s)",
            -best_result.fun,
            len(starts),
            best_result.success,
        )
        return Bunch(trajectory=trajectory, titer=float(-best_result.fun), optimizer_success=bool(best_result.success))

    def golden_trajectory(
        self,
        *,
        n_knots: int = 4,
        n_starts: int = 5,
        random_state: int | np.random.Generator | None = 0,
    ) -> Bunch:
        """Find the golden batch: the optimal schedule for the *nominal* initial conditions.

        This is what golden-batch practice enshrines as the recipe. It is the
        true optimum only for the conditions under which it was found; the
        whole point of the baseline is what happens when it is replayed under
        other conditions. Parameters and return value match
        :meth:`optimal_trajectory` with ``initial_conditions=None``.
        """
        return self.optimal_trajectory(None, n_knots=n_knots, n_starts=n_starts, random_state=random_state)

    def simulate_campaign(  # noqa: C901, PLR0913
        self,
        n_batches: int,
        *,
        policy: str = "replay",
        trajectory: pd.DataFrame | None = None,
        initial_conditions: pd.DataFrame | None = None,
        mv_variation: float = 0.0,
        n_knots: int = 4,
        n_starts: int = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> Bunch:
        """Simulate a campaign of batches under a named operating policy.

        Parameters
        ----------
        n_batches : int
            Number of batches in the campaign.
        policy : {"replay", "historical", "adapted"}, default="replay"
            - ``"replay"``: every batch is given the same setpoint schedule,
              the golden-batch practice.
            - ``"historical"``: the same schedule plus deliberate per-batch
              setpoint variation of size ``mv_variation`` (see below). A
              perfectly consistent history contains no information about how
              the controls affect quality; this policy produces a history
              that does.
            - ``"adapted"``: each batch runs the *true* optimal schedule for
              its own initial conditions, computed from the simulator's model
              via :meth:`optimal_trajectory`. This is the ceiling a perfect
              feedforward scheme could reach, not an implementable policy.
        trajectory : pd.DataFrame, optional
            The schedule used by ``"replay"`` and ``"historical"``; defaults
            to :meth:`nominal_trajectory`. Ignored by ``"adapted"``.
        initial_conditions : pd.DataFrame, optional
            The upstream ``Z`` block, one row per batch (the 11 upstream
            variables as columns). ``None`` draws it with
            :func:`sample_initial_conditions` using the configuration's
            ``ic_scale``.
        mv_variation : float, default=0.0
            Size of the deliberate variation for ``"historical"``: each batch
            draws a random constant offset and a random start-to-end ramp for
            temperature (standard deviation ``mv_variation`` degC each) and
            for pH (standard deviation ``0.1 * mv_variation`` each), clipped
            to the operating bounds.
        n_knots, n_starts : int
            Passed to :meth:`optimal_trajectory` for the ``"adapted"``
            policy, which runs the optimiser once per batch; lower values
            trade optimality for speed. Ignored by the other policies.
        random_state : int, np.random.Generator, or None
            Seed or generator; child seeds are spawned per batch, so a
            campaign is reproducible end to end.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``batches``: ``dict[int, pd.DataFrame]``, the recorded tags per
              batch in the package's standard batch-dictionary format.
            - ``quality``: DataFrame indexed by batch id with the single
              column ``titer`` [g/L].
            - ``initial_conditions``: DataFrame, the ``Z`` block used.
            - ``classes``: Series of feed-class labels ("A"/"B"/"C"), or
              ``"?"`` when ``initial_conditions`` was supplied by the caller.
            - ``trajectories``: ``dict[int, pd.DataFrame]``, the *requested*
              setpoint schedule per batch.
        """
        cfg = self.config
        if not isinstance(n_batches, (int, np.integer)) or isinstance(n_batches, bool) or n_batches < 1:
            raise ValueError(f"n_batches must be a positive integer; got {n_batches!r}.")
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {list(_POLICIES)}; got {policy!r}.")
        if not (isinstance(mv_variation, (int, float)) and math.isfinite(mv_variation) and mv_variation >= 0):
            raise ValueError(f"mv_variation must be a finite non-negative number; got {mv_variation!r}.")
        rng = check_random_state(random_state)

        if initial_conditions is None:
            drawn = sample_initial_conditions(int(n_batches), ic_scale=cfg.ic_scale, random_state=rng)
            z_block, classes = drawn.z, drawn.classes
        else:
            if not isinstance(initial_conditions, pd.DataFrame):
                raise TypeError(
                    f"initial_conditions must be a DataFrame with one row per batch or None; "
                    f"got {type(initial_conditions).__name__}."
                )
            if len(initial_conditions) != int(n_batches):
                raise ValueError(
                    f"initial_conditions must have n_batches = {n_batches} rows; got {len(initial_conditions)}."
                )
            missing = [name for name in UPSTREAM_VARIABLE_NAMES if name not in initial_conditions.columns]
            if missing:
                raise ValueError(f"initial_conditions is missing upstream variables {missing}.")
            z_block = initial_conditions
            classes = pd.Series("?", index=z_block.index, name="feed_class")

        base_trajectory = self.nominal_trajectory() if trajectory is None else trajectory
        self._validate_trajectory(base_trajectory)

        batch_ids = list(z_block.index)
        child_rngs = rng.spawn(int(n_batches))
        batches: dict = {}
        trajectories: dict = {}
        titers = np.empty(int(n_batches))
        interval_fraction = np.linspace(0.0, 1.0, cfg.samples_per_batch)

        for row, (batch_id, child) in enumerate(zip(batch_ids, child_rngs, strict=True)):
            if policy == "adapted":
                requested = self.optimal_trajectory(
                    z_block.loc[batch_id], n_knots=n_knots, n_starts=n_starts, random_state=0
                ).trajectory
            elif policy == "historical" and mv_variation > 0:
                offsets = rng.standard_normal(4)
                temp_delta = mv_variation * (offsets[0] + offsets[1] * interval_fraction)
                ph_delta = 0.1 * mv_variation * (offsets[2] + offsets[3] * interval_fraction)
                requested = base_trajectory.copy()
                requested["temperature"] = np.clip(
                    requested["temperature"].to_numpy() + temp_delta, cfg.temp_bounds[0], cfg.temp_bounds[1]
                )
                requested["pH"] = np.clip(requested["pH"].to_numpy() + ph_delta, cfg.ph_bounds[0], cfg.ph_bounds[1])
            else:
                requested = base_trajectory

            result = self.simulate_batch(z_block.loc[batch_id], requested, random_state=child)
            batches[batch_id] = result.tags
            trajectories[batch_id] = requested
            titers[row] = result.titer

        quality = pd.DataFrame({"titer": titers}, index=pd.Index(batch_ids, name="batch_id"))
        logger.debug(
            "simulate_campaign: %d batches under %r; titer mean %.3f g/L, sd %.3f g/L",
            n_batches,
            policy,
            float(quality["titer"].mean()),
            float(quality["titer"].std(ddof=1)) if n_batches > 1 else float("nan"),
        )
        return Bunch(
            batches=batches,
            quality=quality,
            initial_conditions=z_block,
            classes=classes,
            trajectories=trajectories,
        )

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    def sensitivity_budget(
        self,
        *,
        n_noise_replicates: int = 100,
        random_state: int | np.random.Generator | None = 0,
    ) -> pd.DataFrame:
        """How much the final titer moves under standard input perturbations.

        This is the realism check: a credible process model must not respond
        visibly to input changes smaller than an instrument can resolve, and
        must respond clearly to sustained multi-degree deviations. The rows
        cover zero-mean control-loop noise at instrument scale, sustained
        setpoint biases of increasing size, and a single-sample excursion.
        All effects are measured on the deterministic model (disturbance
        channels off) except the control-noise row, which uses the
        configured control-error model.

        Parameters
        ----------
        n_noise_replicates : int, default=100
            Number of batches used for the control-loop-noise row.
        random_state : int, np.random.Generator, or None, default=0
            Seed for the control-noise replicates.

        Returns
        -------
        pd.DataFrame
            One row per perturbation with columns ``perturbation`` (index),
            ``titer_g_L`` and ``effect_pct`` (percentage change of the final
            titer against the unperturbed nominal batch; for the noise row,
            the standard deviation across replicates).
        """
        if not isinstance(n_noise_replicates, (int, np.integer)) or n_noise_replicates < 2:
            raise ValueError(f"n_noise_replicates must be an integer of at least 2; got {n_noise_replicates!r}.")
        cfg = self.config
        rng = check_random_state(random_state)
        nominal = self.nominal_trajectory()
        latent0 = np.zeros(3)
        ph_set = nominal["pH"].to_numpy()
        temp_set = nominal["temperature"].to_numpy()
        base = self._deterministic_titer(latent0, ph_set, temp_set)

        rows: list[tuple[str, float, float]] = []

        # Zero-mean control-loop noise at the configured instrument scale.
        noise_config = dataclasses.replace(cfg, ic_scale=0.0, within_batch_scale=0.0, noise_scale=1.0)
        noise_sim = BioreactorSimulator(noise_config)
        children = rng.spawn(int(n_noise_replicates))
        titers = np.array([noise_sim.simulate_batch(None, nominal, random_state=child).titer for child in children])
        rows.append(
            (
                f"control-loop noise (sd {cfg.control_sd_temp} degC, {cfg.control_sd_ph} pH)",
                float(titers.mean()),
                float(100.0 * titers.std(ddof=1) / base),
            )
        )

        def clipped(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
            return np.clip(values, bounds[0], bounds[1])

        for bias in (0.1, 0.5, 1.0, 2.0, 3.0):
            for sign in (1.0, -1.0):
                titer = self._deterministic_titer(latent0, ph_set, clipped(temp_set + sign * bias, cfg.temp_bounds))
                label = f"sustained temperature bias {sign * bias:+.1f} degC"
                rows.append((label, titer, 100.0 * (titer - base) / base))
        for bias in (0.02, 0.1, 0.2):
            for sign in (1.0, -1.0):
                titer = self._deterministic_titer(latent0, clipped(ph_set + sign * bias, cfg.ph_bounds), temp_set)
                rows.append((f"sustained pH bias {sign * bias:+.2f}", titer, 100.0 * (titer - base) / base))

        one_sample = temp_set.copy()
        mid = cfg.samples_per_batch // 2
        one_sample[mid] = min(one_sample[mid] + 0.5, cfg.temp_bounds[1])
        titer = self._deterministic_titer(latent0, ph_set, one_sample)
        rows.append(("single-sample temperature excursion +0.5 degC", titer, 100.0 * (titer - base) / base))

        frame = pd.DataFrame(rows, columns=["perturbation", "titer_g_L", "effect_pct"]).set_index("perturbation")
        frame.attrs["nominal_titer_g_L"] = base
        return frame


def variance_decomposition(
    simulator: BioreactorSimulator,
    n_batches: int = 200,
    *,
    trajectory: pd.DataFrame | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Split the replay-policy titer variance into its three sources.

    Runs four replay campaigns with the same size and schedule: all channels
    on (the total), then each channel alone. The initial-condition bucket is
    what pre-batch (feedforward) trajectory adaptation could remove; the
    within-batch bucket is what only mid-course (feedback) correction can
    reach; the noise bucket is the floor. Because the model is nonlinear the
    three buckets do not sum exactly to the total; the difference is reported
    as the interaction residual rather than hidden.

    Parameters
    ----------
    simulator : BioreactorSimulator
        The simulator whose configuration (including channel scales) defines
        the "all channels on" case.
    n_batches : int, default=200
        Batches per campaign.
    trajectory : pd.DataFrame, optional
        The replayed schedule; defaults to the simulator's nominal.
    random_state : int, np.random.Generator, or None
        Seed or generator; each campaign draws its own child seed.

    Returns
    -------
    pd.DataFrame
        Rows ``["measured initial conditions", "within-batch disturbance",
        "control and measurement noise", "interaction residual", "total"]``
        with columns ``variance`` [g^2/L^2], ``sd`` [g/L], ``cv_pct`` (sd as
        a percentage of the all-channels mean titer) and ``pct_of_total``
        (share of the total variance; the residual's share can be negative).
    """
    if not isinstance(simulator, BioreactorSimulator):
        raise TypeError(f"simulator must be a BioreactorSimulator; got {type(simulator).__name__}.")
    if not isinstance(n_batches, (int, np.integer)) or isinstance(n_batches, bool) or n_batches < 2:
        raise ValueError(f"n_batches must be an integer of at least 2; got {n_batches!r}.")
    rng = check_random_state(random_state)
    cfg = simulator.config
    if trajectory is None:
        trajectory = simulator.nominal_trajectory()

    channel_configs = {
        "total": cfg,
        "measured initial conditions": dataclasses.replace(cfg, within_batch_scale=0.0, noise_scale=0.0),
        "within-batch disturbance": dataclasses.replace(cfg, ic_scale=0.0, noise_scale=0.0),
        "control and measurement noise": dataclasses.replace(cfg, ic_scale=0.0, within_batch_scale=0.0),
    }
    variances: dict[str, float] = {}
    mean_total = float("nan")
    for label, channel_config in channel_configs.items():
        campaign = BioreactorSimulator(channel_config).simulate_campaign(
            int(n_batches), policy="replay", trajectory=trajectory, random_state=rng.spawn(1)[0]
        )
        titer = campaign.quality["titer"]
        variances[label] = float(titer.var(ddof=1))
        if label == "total":
            mean_total = float(titer.mean())

    residual = variances["total"] - sum(v for k, v in variances.items() if k != "total")
    order = [
        "measured initial conditions",
        "within-batch disturbance",
        "control and measurement noise",
        "interaction residual",
        "total",
    ]
    values = {**variances, "interaction residual": residual}
    frame = pd.DataFrame(
        {
            "variance": [values[k] for k in order],
            "sd": [math.sqrt(values[k]) if values[k] >= 0 else float("nan") for k in order],
        },
        index=pd.Index(order, name="source"),
    )
    frame["cv_pct"] = 100.0 * frame["sd"] / mean_total
    frame["pct_of_total"] = 100.0 * frame["variance"] / values["total"]
    frame.attrs["mean_titer_g_L"] = mean_total
    return frame
