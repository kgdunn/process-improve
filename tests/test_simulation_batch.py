"""Tests for the fed-batch bioreactor simulator (``process_improve.simulation.batch``).

Grouped by intent:

- Realism acceptance tests: the sensitivity budget must show instrument-scale
  input noise to be immaterial and sustained multi-degree deviations to be
  material. These pin the property the simulator was designed around.
- Determinism: bitwise reproducibility from a seed, including across
  processes.
- Channel isolation: each disturbance channel can be switched off without
  altering the draws of the others, and the within-batch channel alone
  spreads the titer (the load-bearing claim for mid-course correction).
- Physical invariants (property-based, hypothesis).
- Optimiser behaviour and the variance decomposition (slow tier).
- Round-trips into the package's batch tooling (integration tier).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from process_improve.batch.data_input import check_valid_batch_dict, dict_to_wide
from process_improve.batch.preprocessing import resample_to_reference
from process_improve.multivariate import PCA, MCUVScaler
from process_improve.simulation import (
    UPSTREAM_VARIABLE_NAMES,
    BioreactorConfig,
    BioreactorSimulator,
    cardinal_ph,
    cardinal_temperature,
    sample_initial_conditions,
    variance_decomposition,
)


@pytest.fixture(scope="module")
def sim() -> BioreactorSimulator:
    """Return a simulator on the default configuration (stateless, safe to share)."""
    return BioreactorSimulator()


@pytest.fixture(scope="module")
def nominal(sim: BioreactorSimulator) -> pd.DataFrame:
    return sim.nominal_trajectory()


@pytest.fixture(scope="module")
def budget(sim: BioreactorSimulator) -> pd.DataFrame:
    """Compute the sensitivity budget once for all realism tests."""
    return sim.sensitivity_budget(n_noise_replicates=80, random_state=0)


def _effect(budget: pd.DataFrame, label_fragment: str) -> float:
    rows = [label for label in budget.index if label_fragment in label]
    assert len(rows) == 1, f"expected exactly one budget row matching {label_fragment!r}, found {rows}"
    return float(budget.loc[rows[0], "effect_pct"])


# ---------------------------------------------------------------------------
# Realism acceptance tests
# ---------------------------------------------------------------------------


def test_control_loop_noise_is_immaterial(budget: pd.DataFrame) -> None:
    """Zero-mean control noise at instrument scale moves titer by well under 1%."""
    assert 0.0 < _effect(budget, "control-loop noise") < 0.5


def test_instrument_resolution_bias_is_immaterial(budget: pd.DataFrame) -> None:
    """A sustained bias at instrument resolution (0.1 degC, 0.02 pH) is second order."""
    assert abs(_effect(budget, "temperature bias +0.1")) < 0.30
    assert abs(_effect(budget, "temperature bias -0.1")) < 0.30
    assert abs(_effect(budget, "pH bias +0.02")) < 0.05
    assert abs(_effect(budget, "pH bias -0.02")) < 0.05


def test_single_sample_excursion_is_immaterial(budget: pd.DataFrame) -> None:
    """One 12-hour 0.5 degC excursion costs well under 1%."""
    assert abs(_effect(budget, "single-sample temperature excursion")) < 0.5


def test_two_degree_bias_is_material(budget: pd.DataFrame) -> None:
    """A sustained 2 degC bias costs real titer, so the model is not inert."""
    assert _effect(budget, "temperature bias +2.0") < -5.0
    assert _effect(budget, "temperature bias -2.0") < -5.0


def test_ph_bias_is_material(budget: pd.DataFrame) -> None:
    """A sustained 0.2 pH bias is visible even though 0.02 is not."""
    assert _effect(budget, "pH bias +0.20") < -0.3
    assert _effect(budget, "pH bias -0.20") < -0.3


def test_overheating_costs_more_than_undercooling(budget: pd.DataFrame) -> None:
    """Near the operating point a warm bias costs more than the same cool bias.

    Warm burns feed through residual growth and approaches hypoxia; cool at
    this scale only slows the ramp. (Far from the operating point the
    asymmetry reverses: full growth arrest is catastrophic.)
    """
    assert _effect(budget, "temperature bias +1.0") < _effect(budget, "temperature bias -1.0") < 0.0


def test_perturbations_never_beat_the_stationary_nominal(budget: pd.DataFrame) -> None:
    """The nominal recipe sits at a stationary maximum: every probed deterministic
    perturbation should cost titer, not gain it.
    """
    deterministic = budget.drop(index=[label for label in budget.index if "noise" in label])
    assert (deterministic["effect_pct"] < 0.05).all()


# ---------------------------------------------------------------------------
# Cardinal functions
# ---------------------------------------------------------------------------


def test_cardinal_functions_peak_at_one_and_vanish_at_bounds() -> None:
    assert cardinal_temperature(36.8, 27.5, 36.8, 41.5) == pytest.approx(1.0)
    assert cardinal_temperature(27.5, 27.5, 36.8, 41.5) == 0.0
    assert cardinal_temperature(41.5, 27.5, 36.8, 41.5) == 0.0
    assert cardinal_temperature(20.0, 27.5, 36.8, 41.5) == 0.0
    assert cardinal_ph(7.10, 6.3, 7.10, 7.9) == pytest.approx(1.0)
    assert cardinal_ph(6.3, 6.3, 7.10, 7.9) == 0.0
    assert cardinal_ph(8.5, 6.3, 7.10, 7.9) == 0.0
    grid = np.linspace(20.0, 45.0, 301)
    values = cardinal_temperature(grid, 27.5, 36.8, 41.5)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_is_bitwise_identical(sim: BioreactorSimulator) -> None:
    one = sim.simulate_batch(random_state=42)
    two = sim.simulate_batch(random_state=42)
    assert one.titer == two.titer
    pd.testing.assert_frame_equal(one.tags, two.tags, check_exact=True)
    pd.testing.assert_frame_equal(one.states, two.states, check_exact=True)


def test_different_seeds_differ(sim: BioreactorSimulator) -> None:
    assert sim.simulate_batch(random_state=1).titer != sim.simulate_batch(random_state=2).titer


def test_campaign_is_reproducible(sim: BioreactorSimulator) -> None:
    one = sim.simulate_campaign(8, policy="replay", random_state=5)
    two = sim.simulate_campaign(8, policy="replay", random_state=5)
    pd.testing.assert_frame_equal(one.quality, two.quality, check_exact=True)
    pd.testing.assert_frame_equal(one.initial_conditions, two.initial_conditions, check_exact=True)


@pytest.mark.slow
def test_cross_process_determinism(sim: BioreactorSimulator) -> None:
    """A separate interpreter process reproduces the same titer bit for bit."""
    code = (
        "from process_improve.simulation import BioreactorSimulator; "
        "print(repr(BioreactorSimulator().simulate_batch(random_state=42).titer))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    assert float(result.stdout.strip()) == sim.simulate_batch(random_state=42).titer


# ---------------------------------------------------------------------------
# Channel isolation
# ---------------------------------------------------------------------------


def _config(**overrides: float) -> BioreactorConfig:
    return dataclasses.replace(BioreactorConfig(), **overrides)


def test_all_channels_zero_gives_identical_batches() -> None:
    quiet = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0))
    one = quiet.simulate_batch(random_state=1)
    two = quiet.simulate_batch(random_state=999)
    assert one.titer == two.titer
    pd.testing.assert_frame_equal(one.tags, two.tags, check_exact=True)


def test_replay_repeats_requested_but_not_realised(sim: BioreactorSimulator, nominal: pd.DataFrame) -> None:
    campaign = sim.simulate_campaign(4, policy="replay", trajectory=nominal, random_state=3)
    requested = list(campaign.trajectories.values())
    for other in requested[1:]:
        pd.testing.assert_frame_equal(requested[0], other, check_exact=True)
    realised_temp = np.array([campaign.batches[b]["temperature"].to_numpy() for b in campaign.batches])
    assert np.ptp(realised_temp, axis=0).max() > 0.0


def test_within_batch_channel_alone_spreads_titer() -> None:
    """The load-bearing claim: identical measured initial conditions for every
    batch, no measurement or control noise, and the titer still spreads far
    beyond the noise floor. This is the part of the variance only mid-course
    correction can reach.
    """
    wb_only = BioreactorSimulator(_config(ic_scale=0.0, noise_scale=0.0))
    campaign = wb_only.simulate_campaign(40, policy="replay", random_state=11)
    z = campaign.initial_conditions
    assert (z == z.iloc[0]).all().all(), "every batch must have identical initial conditions"
    titer = campaign.quality["titer"]
    assert 100.0 * titer.std(ddof=1) / titer.mean() > 3.0


def test_noise_channel_alone_is_the_floor() -> None:
    noise_only = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0))
    campaign = noise_only.simulate_campaign(40, policy="replay", random_state=11)
    titer = campaign.quality["titer"]
    assert 100.0 * titer.std(ddof=1) / titer.mean() < 1.0


def test_replay_spread_exceeds_zero_disturbance_spread(sim: BioreactorSimulator) -> None:
    campaign = sim.simulate_campaign(40, policy="replay", random_state=11)
    titer = campaign.quality["titer"]
    assert 100.0 * titer.std(ddof=1) / titer.mean() > 5.0


def test_channel_toggle_is_a_true_counterfactual() -> None:
    """Switching one channel off must not change the draws of the others: the
    same seed with within-batch off keeps the identical realised trajectory.
    """
    with_wb = BioreactorSimulator(_config(ic_scale=0.0, noise_scale=1.0, within_batch_scale=1.0))
    without_wb = BioreactorSimulator(_config(ic_scale=0.0, noise_scale=1.0, within_batch_scale=0.0))
    one = with_wb.simulate_batch(random_state=7)
    two = without_wb.simulate_batch(random_state=7)
    pd.testing.assert_frame_equal(one.realised_trajectory, two.realised_trajectory, check_exact=True)
    assert one.titer != two.titer


def test_unmeasured_disturbance_is_observable_in_the_gas_tags() -> None:
    """The within-batch disturbance must show up in the oxygen trajectory even
    though it is not a recorded variable: that observability is the premise of
    mid-course correction.
    """
    with_wb = BioreactorSimulator(_config(ic_scale=0.0, noise_scale=0.0, within_batch_scale=1.0))
    without_wb = BioreactorSimulator(_config(ic_scale=0.0, noise_scale=0.0, within_batch_scale=0.0))
    one = with_wb.simulate_batch(random_state=7)
    two = without_wb.simulate_batch(random_state=7)
    assert not np.allclose(one.tags["dissolved_oxygen"], two.tags["dissolved_oxygen"])


def test_historical_policy_varies_the_requested_trajectories(sim: BioreactorSimulator) -> None:
    campaign = sim.simulate_campaign(5, policy="historical", mv_variation=1.0, random_state=3)
    temps = np.array([t["temperature"].to_numpy() for t in campaign.trajectories.values()])
    assert np.ptp(temps, axis=0).max() > 0.1
    cfg = sim.config
    assert temps.min() >= cfg.temp_bounds[0]
    assert temps.max() <= cfg.temp_bounds[1]


# ---------------------------------------------------------------------------
# The upstream Z block
# ---------------------------------------------------------------------------


def test_z_block_layout() -> None:
    drawn = sample_initial_conditions(12, random_state=1)
    assert drawn.z.shape == (12, len(UPSTREAM_VARIABLE_NAMES))
    assert list(drawn.z.columns) == list(UPSTREAM_VARIABLE_NAMES)
    assert drawn.classes.isin(["A", "B", "C"]).all()
    assert drawn.latent.shape == (12, 3)
    assert (drawn.z.index == drawn.classes.index).all()


def test_pca_of_z_recovers_about_three_components() -> None:
    drawn = sample_initial_conditions(200, random_state=2)
    scaled = MCUVScaler().fit_transform(drawn.z)
    model = PCA(n_components=5).fit(scaled)
    r2 = model.r2_cumulative_.to_numpy().ravel()
    assert r2[2] > 0.55, "three components should capture the bulk of the upstream variation"
    assert r2[3] - r2[2] < 0.10, "a fourth component should add little"


def test_feed_classes_separate_in_z() -> None:
    drawn = sample_initial_conditions(300, random_state=3)
    grouped = drawn.z.groupby(drawn.classes)
    assert grouped["seed_viability_pct"].mean()["A"] > grouped["seed_viability_pct"].mean()["C"]
    assert grouped["impurity_index"].mean()["C"] > grouped["impurity_index"].mean()["A"]


def test_class_proportions_are_respected() -> None:
    drawn = sample_initial_conditions(50, proportions={"A": 1.0, "B": 0.0, "C": 0.0}, random_state=4)
    assert (drawn.classes == "A").all()


def test_ic_scale_zero_collapses_z_to_nominal() -> None:
    drawn = sample_initial_conditions(6, ic_scale=0.0, random_state=5)
    assert (drawn.z == drawn.z.iloc[0]).all().all()
    assert np.allclose(drawn.latent.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Structure and physical plausibility
# ---------------------------------------------------------------------------


def test_output_shapes_and_layout(sim: BioreactorSimulator) -> None:
    result = sim.simulate_batch(random_state=0)
    cfg = sim.config
    assert result.tags.shape == (cfg.samples_per_batch, 5)
    assert list(result.tags.columns) == ["pH", "temperature", "dissolved_oxygen", "offgas_co2", "volume"]
    assert result.tags.index.name == "day"
    assert result.tags.index[-1] == pytest.approx(cfg.batch_days)
    assert result.states.shape[0] == cfg.n_steps + 1
    assert list(result.realised_trajectory.columns) == ["pH", "temperature"]
    assert list(result.initial_conditions.index) == list(UPSTREAM_VARIABLE_NAMES)


def test_nominal_batch_is_physiologically_plausible(sim: BioreactorSimulator, nominal: pd.DataFrame) -> None:
    quiet = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0))
    result = quiet.simulate_batch(None, nominal)
    assert 4.0 < result.titer < 12.0, "final titer should sit in a modern fed-batch range [g/L]"
    assert 2.0 < result.states["biomass"].max() < 8.0, "peak biomass [g/L]"
    assert (result.tags["dissolved_oxygen"] > 3.0).all()
    assert (result.tags["dissolved_oxygen"] <= 100.0).all()
    expected_volume = sim.config.volume_initial + sim.config.feed_rate * sim.config.batch_days
    assert result.states["volume"].iloc[-1] == pytest.approx(expected_volume)


def test_oxygen_falls_as_the_pile_grows() -> None:
    quiet = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0))
    tags = quiet.simulate_batch(None).tags
    assert tags["dissolved_oxygen"].iloc[5] < tags["dissolved_oxygen"].iloc[0]


def test_rk4_step_halving_changes_little() -> None:
    coarse = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0))
    fine = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0, steps_per_day=48))
    t_coarse = coarse.simulate_batch(None).titer
    t_fine = fine.simulate_batch(None).titer
    assert abs(t_fine - t_coarse) / t_coarse < 1e-5  # inputs are held per step, so RK4 keeps its order


def test_nominal_trajectory_is_biphasic(sim: BioreactorSimulator, nominal: pd.DataFrame) -> None:
    cfg = sim.config
    assert nominal["temperature"].iloc[0] == pytest.approx(cfg.temp_opt)
    assert nominal["temperature"].iloc[-1] == pytest.approx(cfg.temp_production)
    assert np.allclose(nominal["pH"], cfg.ph_opt)
    assert nominal["temperature"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Physical invariants (property-based)
# ---------------------------------------------------------------------------


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    ic_scale=st.floats(min_value=0.0, max_value=2.0),
    within_batch_scale=st.floats(min_value=0.0, max_value=2.0),
    noise_scale=st.floats(min_value=0.0, max_value=2.0),
)
def test_physical_invariants_hold_for_any_input(
    seed: int, ic_scale: float, within_batch_scale: float, noise_scale: float
) -> None:
    """States stay finite and non-negative, the realised trajectory stays inside
    the actuator bounds, volume grows linearly, and the product mass never
    exceeds what the substrate fed in can support.
    """
    cfg = _config(ic_scale=ic_scale, within_batch_scale=within_batch_scale, noise_scale=noise_scale)
    simulator = BioreactorSimulator(cfg)
    rng = np.random.default_rng(seed)
    trajectory = pd.DataFrame(
        {
            "pH": rng.uniform(cfg.ph_bounds[0], cfg.ph_bounds[1], cfg.samples_per_batch),
            "temperature": rng.uniform(cfg.temp_bounds[0], cfg.temp_bounds[1], cfg.samples_per_batch),
        }
    )
    z = pd.Series(rng.normal(0.0, 3.0, len(UPSTREAM_VARIABLE_NAMES)), index=list(UPSTREAM_VARIABLE_NAMES))
    z = z * 10.0 + 50.0  # arbitrary but finite user-supplied upstream values
    result = simulator.simulate_batch(z, trajectory, random_state=seed)

    states = result.states
    assert np.isfinite(states.to_numpy()).all()
    assert (states["biomass"] >= 0.0).all()
    assert (states["substrate"] >= 0.0).all()
    assert (states["titer"] >= 0.0).all()
    assert (states["volume"] > 0.0).all()

    assert (result.realised_trajectory["temperature"] >= cfg.temp_bounds[0]).all()
    assert (result.realised_trajectory["temperature"] <= cfg.temp_bounds[1]).all()
    assert (result.realised_trajectory["pH"] >= cfg.ph_bounds[0]).all()
    assert (result.realised_trajectory["pH"] <= cfg.ph_bounds[1]).all()

    volume = states["volume"].to_numpy()
    increments = np.diff(volume)
    assert increments.min() > 0.0
    assert np.allclose(increments, increments[0], rtol=1e-9)

    # Mass balance: product mass cannot exceed the yield on all substrate
    # that entered the reactor (initial charge plus feed).
    feed_rate = (volume[-1] - volume[0]) / cfg.batch_days
    substrate_in = states["substrate"].iloc[0] * volume[0] + feed_rate * cfg.batch_days * cfg.feed_substrate
    product_mass = states["titer"].iloc[-1] * volume[-1]
    assert product_mass <= cfg.yield_ps * substrate_in * 1.001


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_config_rejects_bad_cardinal_order() -> None:
    with pytest.raises(ValueError, match="temp_min < temp_opt < temp_max"):
        BioreactorConfig(temp_min=40.0, temp_opt=36.8, temp_max=41.5)


def test_config_rejects_bad_sampling_grid() -> None:
    with pytest.raises(ValueError, match="integer multiple of samples_per_batch"):
        BioreactorConfig(samples_per_batch=7)


def test_config_rejects_negative_kinetics() -> None:
    with pytest.raises(ValueError, match="mu_opt"):
        BioreactorConfig(mu_opt=-1.0)


def test_config_rejects_bounds_outside_cardinal_window() -> None:
    with pytest.raises(ValueError, match="temp_bounds"):
        BioreactorConfig(temp_bounds=(20.0, 39.0))


def test_config_rejects_hold_outside_bounds() -> None:
    with pytest.raises(ValueError, match="temp_production"):
        BioreactorConfig(temp_production=45.0)


def test_simulator_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="BioreactorConfig"):
        BioreactorSimulator(config="not a config")  # type: ignore[arg-type]


def test_simulate_batch_rejects_bad_trajectory(sim: BioreactorSimulator, nominal: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        sim.simulate_batch(None, nominal[["pH"]])
    with pytest.raises(ValueError, match="rows"):
        sim.simulate_batch(None, nominal.iloc[:5])
    too_hot = nominal.copy()
    too_hot["temperature"] = 45.0
    with pytest.raises(ValueError, match="temp_bounds"):
        sim.simulate_batch(None, too_hot)
    not_finite = nominal.copy()
    not_finite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sim.simulate_batch(None, not_finite)
    with pytest.raises(TypeError, match="DataFrame"):
        sim.simulate_batch(None, "not a frame")  # type: ignore[arg-type]


def test_simulate_batch_rejects_bad_initial_conditions(sim: BioreactorSimulator) -> None:
    with pytest.raises(TypeError, match="Series"):
        sim.simulate_batch({"a": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing upstream variables"):
        sim.simulate_batch(pd.Series({"seed_viability_pct": 90.0}))
    bad = pd.Series(np.nan, index=list(UPSTREAM_VARIABLE_NAMES))
    with pytest.raises(ValueError, match="non-finite"):
        sim.simulate_batch(bad)


def test_campaign_rejects_bad_arguments(sim: BioreactorSimulator) -> None:
    with pytest.raises(ValueError, match="n_batches"):
        sim.simulate_campaign(0)
    with pytest.raises(ValueError, match="policy"):
        sim.simulate_campaign(3, policy="whatever")
    with pytest.raises(ValueError, match="mv_variation"):
        sim.simulate_campaign(3, policy="historical", mv_variation=-1.0)
    with pytest.raises(TypeError, match="DataFrame"):
        sim.simulate_campaign(3, initial_conditions="nope")  # type: ignore[arg-type]
    z = sample_initial_conditions(2, random_state=0).z
    with pytest.raises(ValueError, match="rows"):
        sim.simulate_campaign(3, initial_conditions=z)
    with pytest.raises(ValueError, match="missing upstream variables"):
        sim.simulate_campaign(2, initial_conditions=z[["seed_viability_pct"]])


def test_sample_initial_conditions_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="n_batches"):
        sample_initial_conditions(0)
    with pytest.raises(ValueError, match="ic_scale"):
        sample_initial_conditions(3, ic_scale=-1.0)
    with pytest.raises(ValueError, match="unknown class labels"):
        sample_initial_conditions(3, proportions={"D": 1.0})
    with pytest.raises(ValueError, match="positive sum"):
        sample_initial_conditions(3, proportions={"A": 0.0, "B": 0.0, "C": 0.0})


def test_optimal_trajectory_rejects_bad_arguments(sim: BioreactorSimulator) -> None:
    with pytest.raises(ValueError, match="n_knots"):
        sim.optimal_trajectory(n_knots=1)
    with pytest.raises(ValueError, match="n_starts"):
        sim.optimal_trajectory(n_starts=0)


def test_sensitivity_budget_rejects_bad_arguments(sim: BioreactorSimulator) -> None:
    with pytest.raises(ValueError, match="n_noise_replicates"):
        sim.sensitivity_budget(n_noise_replicates=1)


def test_variance_decomposition_rejects_bad_arguments(sim: BioreactorSimulator) -> None:
    with pytest.raises(TypeError, match="BioreactorSimulator"):
        variance_decomposition("not a simulator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_batches"):
        variance_decomposition(sim, n_batches=1)


# ---------------------------------------------------------------------------
# Optimiser and variance decomposition (slow tier)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_golden_trajectory_is_biphasic_and_beats_the_recipe(sim: BioreactorSimulator, nominal: pd.DataFrame) -> None:
    golden = sim.golden_trajectory(n_knots=3, n_starts=2)
    assert golden.optimizer_success
    temperature = golden.trajectory["temperature"].to_numpy()
    assert temperature[0] > 34.0, "the growth phase should be warm"
    assert temperature[-1] < 31.0, "the production phase should be cool"
    nominal_titer = sim._deterministic_titer(np.zeros(3), nominal["pH"].to_numpy(), nominal["temperature"].to_numpy())
    assert golden.titer > nominal_titer


@pytest.mark.slow
def test_adaptation_beats_replaying_the_golden_batch(sim: BioreactorSimulator) -> None:
    """For a poor feed lot, the true optimum for its own conditions beats
    replaying the golden trajectory by a material margin.
    """
    from process_improve.simulation.batch import _coerce_z_row, _z_to_latent

    golden = sim.golden_trajectory(n_knots=3, n_starts=2)
    drawn = sample_initial_conditions(30, random_state=3)
    # The batch with the strongest inhibitor level is the worst lot.
    worst = drawn.latent["inhibitor_level"].idxmax()
    z_row = drawn.z.loc[worst]
    latent = _z_to_latent(_coerce_z_row(z_row))
    replay_titer = sim._deterministic_titer(
        latent, golden.trajectory["pH"].to_numpy(), golden.trajectory["temperature"].to_numpy()
    )
    adapted = sim.optimal_trajectory(z_row, n_knots=3, n_starts=2)
    assert adapted.titer > replay_titer * 1.01, "adaptation should recover at least 1% for a poor lot"


@pytest.mark.slow
def test_variance_decomposition_buckets(sim: BioreactorSimulator) -> None:
    frame = variance_decomposition(sim, n_batches=60, random_state=11)
    assert list(frame.index) == [
        "measured initial conditions",
        "within-batch disturbance",
        "control and measurement noise",
        "interaction residual",
        "total",
    ]
    noise = frame.loc["control and measurement noise"]
    assert noise["cv_pct"] < 1.0
    assert frame.loc["measured initial conditions", "cv_pct"] > 3.0
    assert frame.loc["within-batch disturbance", "cv_pct"] > 3.0
    assert frame.loc["total", "cv_pct"] > 8.0
    parts = frame.loc[
        ["measured initial conditions", "within-batch disturbance", "control and measurement noise"], "variance"
    ].sum()
    residual = frame.loc["interaction residual", "variance"]
    assert frame.loc["total", "variance"] == pytest.approx(parts + residual)


# ---------------------------------------------------------------------------
# Integration with the batch tooling
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_campaign_output_is_a_valid_batch_dict(sim: BioreactorSimulator) -> None:
    campaign = sim.simulate_campaign(6, policy="replay", random_state=1)
    assert check_valid_batch_dict(campaign.batches)
    wide = dict_to_wide(campaign.batches)
    assert wide.shape == (6, sim.config.samples_per_batch * 5)


@pytest.mark.integration
def test_campaign_output_feeds_the_alignment_tooling(sim: BioreactorSimulator) -> None:
    campaign = sim.simulate_campaign(4, policy="replay", random_state=2)
    tags = list(next(iter(campaign.batches.values())).columns)
    first_id = next(iter(campaign.batches))
    aligned = resample_to_reference(
        campaign.batches, columns_to_align=tags, reference_batch=first_id, settings={"show_progress": False}
    )
    assert set(aligned.keys()) == set(campaign.batches.keys())


# ---------------------------------------------------------------------------
# Agent tools and recipe
# ---------------------------------------------------------------------------


def test_batch_tool_schemas_and_rng_metadata() -> None:
    from process_improve.tool_spec import get_tool_specs

    specs = {s["name"]: s for s in get_tool_specs(category="simulation")}
    for name in ("simulate_batch_campaign", "decompose_batch_quality_variance"):
        assert name in specs
        assert specs[name]["input_schema"]["additionalProperties"] is False
        assert specs[name]["rng"] == {"uses_rng": True, "seed_param": "random_state", "default_seed": 0}


def test_simulate_batch_campaign_tool_runs_and_reproduces() -> None:
    from process_improve.tool_spec import execute_tool_call

    payload = {"n_batches": 10, "policy": "replay", "random_state": 0}
    one = execute_tool_call("simulate_batch_campaign", payload)
    two = execute_tool_call("simulate_batch_campaign", payload)
    assert "error" not in one
    assert one == two
    assert len(one["batches"]) == 10
    assert one["titer_g_L"]["sd"] > 0.0
    assert one["reference_titer_g_L"]["value"] > 0.0


def test_simulate_batch_campaign_tool_rejects_bad_input() -> None:
    from process_improve.tool_safety import ToolInputInvalidError
    from process_improve.tool_spec import execute_tool_call

    with pytest.raises(ToolInputInvalidError):
        execute_tool_call("simulate_batch_campaign", {"n_batches": 1})
    with pytest.raises(ToolInputInvalidError):
        execute_tool_call("simulate_batch_campaign", {"n_batches": 10, "unexpected": True})


def test_decompose_batch_quality_variance_tool_runs() -> None:
    from process_improve.tool_spec import execute_tool_call

    out = execute_tool_call("decompose_batch_quality_variance", {"n_batches": 20, "random_state": 0})
    assert "error" not in out
    sources = {row["source"]: row for row in out["sources"]}
    assert set(sources) == {
        "measured initial conditions",
        "within-batch disturbance",
        "control and measurement noise",
        "interaction residual",
        "total",
    }
    assert sources["control and measurement noise"]["cv_pct"] < 1.0


def test_golden_batch_recipe_is_registered_and_matches() -> None:
    from process_improve.recipes import discover_recipes, get_recipe, select_recipe

    discover_recipes()
    assert get_recipe("golden_batch_baseline") is not None
    match = select_recipe("our golden batch does not repeat; batches differ with the same recipe")
    assert match is not None
    assert match.key == "golden_batch_baseline"


@pytest.mark.slow
def test_adapted_policy_runs_and_differs_per_batch(sim: BioreactorSimulator) -> None:
    campaign = sim.simulate_campaign(2, policy="adapted", n_knots=3, n_starts=1, random_state=4)
    ids = list(campaign.trajectories)
    assert not campaign.trajectories[ids[0]].equals(campaign.trajectories[ids[1]])
    assert (campaign.quality["titer"] > 0).all()


# ---------------------------------------------------------------------------
# Regression tests from the adversarial review
# ---------------------------------------------------------------------------


def test_ic_scale_acts_linearly_and_only_at_the_draw() -> None:
    """The initial-condition channel scales linearly (it was once applied twice,
    making it quadratic), and a caller-supplied Z row is used at face value,
    independent of ic_scale.
    """
    from process_improve.simulation.batch import _coerce_z_row, _latent_effects, _z_to_latent

    full = sample_initial_conditions(60, ic_scale=1.0, random_state=9)
    half = sample_initial_conditions(60, ic_scale=0.5, random_state=9)
    np.testing.assert_allclose(half.latent.to_numpy(), 0.5 * full.latent.to_numpy())

    cfg = BioreactorConfig()
    x0_full = np.array([_latent_effects(cfg, _z_to_latent(_coerce_z_row(row)))[0] for _, row in full.z.iterrows()])
    x0_half = np.array([_latent_effects(cfg, _z_to_latent(_coerce_z_row(row)))[0] for _, row in half.z.iterrows()])
    deviation_full = x0_full / cfg.biomass_initial - 1.0
    deviation_half = x0_half / cfg.biomass_initial - 1.0
    unclipped = (np.abs(deviation_full) < 0.35) & (np.abs(deviation_half) < 0.35)
    assert unclipped.sum() > 30
    np.testing.assert_allclose(deviation_half[unclipped], 0.5 * deviation_full[unclipped], rtol=1e-9)

    z_row = full.z.iloc[0]
    quiet = dict(within_batch_scale=0.0, noise_scale=0.0)
    titer_a = BioreactorSimulator(_config(ic_scale=0.25, **quiet)).simulate_batch(z_row).titer
    titer_b = BioreactorSimulator(_config(ic_scale=1.0, **quiet)).simulate_batch(z_row).titer
    assert titer_a == titer_b, "a supplied Z row must mean the same thing at every ic_scale"


def test_ctmi_rejects_ill_posed_cardinals() -> None:
    """t_opt below the window midpoint puts a pole inside the cardinal window."""
    with pytest.raises(ValueError, match="t_opt >= \\(t_min \\+ t_max\\) / 2"):
        cardinal_temperature(30.0, 27.5, 33.0, 41.5)
    with pytest.raises(ValueError, match="t_opt >= \\(t_min \\+ t_max\\) / 2"):
        BioreactorConfig(temp_opt=33.0)
    with pytest.raises(ValueError, match="t_opt >= \\(t_min \\+ t_max\\) / 2"):
        BioreactorConfig(temp_q_opt=31.0)


def test_realised_trajectory_matches_requested_when_noise_off(nominal: pd.DataFrame) -> None:
    """With control noise off, each realised row equals the requested setpoint of
    the interval that ends at its timestamp (it was once shifted one interval
    ahead, reporting the next interval's setpoint).
    """
    quiet = BioreactorSimulator(_config(ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0))
    result = quiet.simulate_batch(None, nominal)
    np.testing.assert_allclose(result.realised_trajectory["temperature"].to_numpy(), nominal["temperature"].to_numpy())
    np.testing.assert_allclose(result.realised_trajectory["pH"].to_numpy(), nominal["pH"].to_numpy())
    np.testing.assert_allclose(result.tags["temperature"].to_numpy(), nominal["temperature"].to_numpy())


def test_duplicate_batch_ids_are_rejected(sim: BioreactorSimulator) -> None:
    z = sample_initial_conditions(2, random_state=0).z
    doubled = pd.concat([z, z])
    with pytest.raises(ValueError, match="unique batch ids"):
        sim.simulate_campaign(4, initial_conditions=doubled)


def test_supplied_z_block_happy_path(sim: BioreactorSimulator) -> None:
    z = sample_initial_conditions(3, random_state=1).z
    campaign = sim.simulate_campaign(3, initial_conditions=z, random_state=2)
    assert list(campaign.quality.index) == list(z.index)
    assert (campaign.classes == "?").all()
