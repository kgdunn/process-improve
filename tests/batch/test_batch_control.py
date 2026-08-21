"""Tests for the mid-course correction QP and the decision-point workflow."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_pls import BatchPLS
from process_improve.batch.control import MidCourseCorrector, midcourse_correction

pytest.importorskip("osqp")


def _synthetic_batches(n_batches: int = 60, n_samples: int = 10, seed: int = 0):
    """Batches with one MV tag ("u") and one response tag ("r").

    Quality is driven by a batch-specific level (the disturbance, visible in
    the response) plus the late-batch average of the MV, so a correction of
    the future MV columns has a genuine, identified effect.
    """
    rng = np.random.default_rng(seed)
    batches, quality = {}, []
    for i in range(n_batches):
        level = rng.uniform(-1.0, 1.0)
        u = 0.2 * rng.standard_normal(n_samples)  # deliberate MV excitation
        r = level + 0.3 * u + 0.02 * rng.standard_normal(n_samples)
        batches[f"b{i}"] = pd.DataFrame({"u": u, "r": r})
        quality.append(2.0 * level + 3.0 * u[5:].mean() + 0.01 * rng.standard_normal())
    y = pd.DataFrame({"q": quality}, index=list(batches.keys()))
    return batches, y


@pytest.fixture(scope="module")
def fitted() -> BatchPLS:
    batches, y = _synthetic_batches()
    return BatchPLS(n_components=3).fit(batches, y)


def _observed_series(model: BatchPLS, batch: pd.DataFrame, k: int) -> pd.Series:
    return pd.Series({(tag, s): float(batch.iloc[s][tag]) for s in range(k) for tag in model.tag_names_})


def _free_columns(model: BatchPLS, k: int) -> list:
    return [("u", s) for s in range(k, model.n_timesteps_)]


def test_target_mode_reaches_reachable_target(fitted: BatchPLS) -> None:
    """With a mild movement penalty, the predicted quality lands on the target."""
    batches, _ = _synthetic_batches()
    batch = batches["b3"]
    k = 5
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        mode="target",
        y_target=0.5,
        weights={"target": 10.0, "movement": 1e-4},
    )
    assert abs(float(result.y_hat.iloc[0]) - 0.5) < 0.02
    assert result.solver.status == "ok"


def test_huge_movement_penalty_pins_to_nominal(fitted: BatchPLS) -> None:
    """An overwhelming movement penalty returns the nominal remaining schedule."""
    batches, _ = _synthetic_batches()
    batch = batches["b3"]
    k = 5
    free = _free_columns(fitted, k)
    nominal = pd.Series(0.05, index=pd.Index(free))
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batch, k),
        free_columns=free,
        mode="target",
        y_target=0.5,
        weights={"target": 1.0, "movement": 1e6},
        nominal_remaining=nominal,
    )
    np.testing.assert_allclose(result.mv.to_numpy(), 0.05, atol=1e-4)
    # And the no-change prediction is reported for that same nominal schedule.
    assert np.isclose(float(result.y_hat.iloc[0]), float(result.y_hat_no_change.iloc[0]), atol=1e-3)


def test_bounds_and_rate_limits_respected(fitted: BatchPLS) -> None:
    """Box bounds and rate limits (including the seam) hold on the returned MVs."""
    batches, _ = _synthetic_batches()
    batch = batches["b7"]
    k = 4
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        mode="target",
        y_target=2.0,  # far target: pushes into the constraints
        weights={"target": 10.0, "movement": 1e-3},
        bounds={"u": (-0.15, 0.15)},
        rate_limits={"u": 0.05},
        seam={"u": 0.0},
    )
    mv = result.mv.to_numpy()
    assert (mv >= -0.15 - 1e-6).all()
    assert (mv <= 0.15 + 1e-6).all()
    steps = np.abs(np.diff(np.concatenate([[0.0], mv])))
    assert (steps <= 0.05 + 1e-6).all()
    assert len(result.active_constraints["bounds"]) + len(result.active_constraints["rate"]) > 0


def test_t2_cap_binds_and_is_reported(fitted: BatchPLS) -> None:
    """A tight T2 cap is enforced by the multiplier iteration and flagged active."""
    batches, _ = _synthetic_batches()
    batch = batches["b7"]
    k = 4
    kwargs = dict(
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        mode="target",
        y_target=2.0,
        weights={"target": 10.0, "movement": 1e-3},
    )
    unconstrained = midcourse_correction(fitted, **kwargs)
    cap = 0.25 * unconstrained.t2
    capped = midcourse_correction(fitted, t2_cap=cap, **kwargs)
    assert capped.t2 <= cap * 1.02
    assert capped.active_constraints["t2_cap"]
    assert capped.solver.n_solves > 1


def test_inactive_caps_leave_solution_unchanged(fitted: BatchPLS) -> None:
    """Caps far above the achieved statistics do not perturb the solution."""
    batches, _ = _synthetic_batches()
    batch = batches["b2"]
    k = 5
    kwargs = dict(
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        mode="target",
        y_target=0.3,
        weights={"target": 1.0, "movement": 0.1},
    )
    plain = midcourse_correction(fitted, **kwargs)
    capped = midcourse_correction(fitted, spe_cap=plain.spe * 50, t2_cap=plain.t2 * 50 + 1.0, **kwargs)
    np.testing.assert_allclose(plain.mv.to_numpy(), capped.mv.to_numpy(), atol=1e-8)
    assert not capped.active_constraints["spe_cap"]
    assert not capped.active_constraints["t2_cap"]


def test_capped_solution_matches_scipy_slsqp(fitted: BatchPLS) -> None:
    """The multiplier iteration agrees with a direct SLSQP solve of the QCQP."""
    scipy_optimize = pytest.importorskip("scipy.optimize")
    batches, _ = _synthetic_batches()
    batch = batches["b7"]
    k = 6
    observed = _observed_series(fitted, batch, k)
    free = _free_columns(fitted, k)
    kwargs = dict(
        observed=observed,
        free_columns=free,
        mode="target",
        y_target=2.0,
        weights={"target": 10.0, "movement": 1e-2},
    )
    unconstrained = midcourse_correction(fitted, **kwargs)
    cap = 0.5 * unconstrained.t2
    ours = midcourse_correction(fitted, t2_cap=cap, **kwargs)

    # Rebuild the same objective pieces through the public operator API and
    # hand the capped problem to SLSQP as an independent check.
    features = pd.Index(fitted.feature_columns_)
    observed_mask = features.isin(observed.index)
    free_mask = features.isin(set(free))
    op = fitted.projection_matrix(observed_mask | free_mask)
    matrix = op.matrix.to_numpy()
    in_free = free_mask[np.flatnonzero(observed_mask | free_mask)]
    M_free, M_obs = matrix[:, in_free], matrix[:, ~in_free]
    center = fitted.center_.to_numpy()
    scale = fitted.scale_.to_numpy()
    # Align the observed values to model-feature order before scaling.
    z_obs = (observed.reindex(features[observed_mask]).to_numpy() - center[observed_mask]) / scale[observed_mask]
    b = M_obs @ z_obs
    C = fitted.y_loadings_.to_numpy()
    y_target_scaled = (2.0 - fitted.y_center_.to_numpy()) / fitted.y_scale_.to_numpy()
    s_inv = np.diag(1.0 / np.asarray(fitted.explained_variance_))

    def objective(u: np.ndarray) -> float:
        t = b + M_free @ u
        return float(10.0 * np.sum((C @ t - y_target_scaled) ** 2) + 1e-2 * np.sum(u**2))

    def t2_of(u: np.ndarray) -> float:
        t = b + M_free @ u
        return float(t @ s_inv @ t)

    reference = scipy_optimize.minimize(
        objective,
        np.zeros(len(free)),
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda u: cap - t2_of(u)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    u_ours = (ours.mv.to_numpy() - center[free_mask]) / scale[free_mask]
    assert objective(u_ours) <= objective(reference.x) * 1.02 + 1e-9
    assert t2_of(u_ours) <= cap * 1.02


def test_knots_give_piecewise_linear_schedule(fitted: BatchPLS) -> None:
    """With two knots the free samples of the tag lie on a straight line."""
    batches, _ = _synthetic_batches()
    batch = batches["b4"]
    k = 4
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        mode="target",
        y_target=0.8,
        weights={"target": 5.0, "movement": 1e-3},
        n_knots=2,
    )
    mv = result.mv.to_numpy()
    second_differences = np.diff(mv, n=2)
    np.testing.assert_allclose(second_differences, 0.0, atol=1e-8)


def test_empty_observed_reproduces_model_inversion(fitted: BatchPLS) -> None:
    """With nothing observed the QP is a model inversion: both hit the target.

    The QP picks the minimum-movement input, PLS.invert the model-plane
    input; they need not coincide, but both must predict the requested
    quality (the Jaeckle-MacGregor nothing-fixed special case).
    """
    all_columns = list(fitted.feature_columns_)
    result = midcourse_correction(
        fitted,
        observed=pd.Series(dtype=float),
        free_columns=all_columns,
        mode="target",
        y_target=0.75,
        weights={"target": 100.0, "movement": 1e-6},
    )
    assert abs(float(result.y_hat.iloc[0]) - 0.75) < 1e-3
    inverted = fitted._pls.invert(0.75)
    x_row = pd.DataFrame([inverted.x_new.to_numpy().ravel()], columns=fitted._pls.x_loadings_.index)
    y_check = fitted._pls.predict(x_row)
    assert abs(float(y_check.iloc[0, 0]) - 0.75) < 1e-6


def test_error_branches(fitted: BatchPLS) -> None:
    """Bad arguments are rejected with actionable messages."""
    batches, _ = _synthetic_batches()
    observed = _observed_series(fitted, batches["b0"], 5)
    free = _free_columns(fitted, 5)
    with pytest.raises(ValueError, match="mode must be one of"):
        midcourse_correction(fitted, observed=observed, free_columns=free, mode="nonsense")
    with pytest.raises(ValueError, match="free_columns is empty"):
        midcourse_correction(fitted, observed=observed, free_columns=[])
    with pytest.raises(ValueError, match="overlap"):
        midcourse_correction(fitted, observed=observed, free_columns=[("u", 0)])
    with pytest.raises(ValueError, match="y_target is required"):
        midcourse_correction(fitted, observed=observed, free_columns=free, mode="target")
    with pytest.raises(ValueError, match="strictly positive in maximize mode"):
        midcourse_correction(fitted, observed=observed, free_columns=free, mode="maximize", weights={"movement": 0.0})
    with pytest.raises(ValueError, match="not model features"):
        midcourse_correction(fitted, observed=observed, free_columns=[("nope", 1)])
    bad = observed.copy()
    bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="observed contains NaN"):
        midcourse_correction(fitted, observed=bad, free_columns=free, y_target=0.0)


# --------------------------------------------------------------------------- #
# MidCourseCorrector


@pytest.fixture(scope="module")
def corrector(fitted: BatchPLS) -> MidCourseCorrector:
    nominal = pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)})
    return MidCourseCorrector(
        fitted,
        nominal,
        mv_tags=["u"],
        mode="target",
        y_target=0.5,
        weights={"target": 5.0, "movement": 1e-3},
        dead_band=0.0,
    )


def test_corrector_schedule_layout(corrector: MidCourseCorrector) -> None:
    """Past rows come from the implemented schedule; only future MV rows change."""
    batches, _ = _synthetic_batches()
    batch = batches["b5"]
    k = 4
    implemented = pd.DataFrame({"u": np.full(corrector.model.n_timesteps_, 0.11)})
    out = corrector.correct(batch.iloc[:k], implemented_schedule=implemented, k=k)
    assert out.corrected
    np.testing.assert_allclose(out.schedule["u"].iloc[:k].to_numpy(), 0.11)
    assert not np.allclose(out.schedule["u"].iloc[k:].to_numpy(), 0.11)


def test_corrector_batch_complete(corrector: MidCourseCorrector) -> None:
    """At the final sample there is nothing to decide."""
    batches, _ = _synthetic_batches()
    out = corrector.correct(batches["b5"], k=corrector.model.n_timesteps_)
    assert not out.corrected
    assert out.reason == "batch_complete"


def test_corrector_spe_gate(corrector: MidCourseCorrector) -> None:
    """A batch-so-far far outside the model is not corrected."""
    batches, _ = _synthetic_batches()
    garbage = batches["b5"].iloc[:4] + 40.0
    out = corrector.correct(garbage, k=4)
    assert not out.corrected
    assert out.reason == "spe_gate"
    assert out.spe_so_far > out.spe_limit_monitor


def test_corrector_dead_band_below_side(fitted: BatchPLS) -> None:
    """target_side='below' leaves batches predicted at or above the target alone."""
    batches, y = _synthetic_batches()
    # Pick a batch whose quality is clearly above the target of 0.0.
    bid = y["q"].idxmax()
    nominal = pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)})
    one_sided = MidCourseCorrector(
        fitted,
        nominal,
        mv_tags=["u"],
        mode="target",
        y_target=0.0,
        target_side="below",
        dead_band=0.0,
        weights={"target": 5.0, "movement": 1e-3},
    )
    out = one_sided.correct(batches[bid].iloc[:5], k=5)
    assert not out.corrected
    assert out.reason == "dead_band"
    assert float(out.dead_band_margin.iloc[0]) == 0.0


def test_corrector_limits_cached_and_shaped(corrector: MidCourseCorrector) -> None:
    """Per-decision-point limits are positive, shaped, and cached."""
    limits = corrector.limits_at(4)
    assert limits.spe_limit_monitor > 0
    assert limits.spe_limit_candidate > 0
    assert limits.t2_limit > 0
    A = int(corrector.model.n_components)
    assert limits.score_covariance.shape == (A, A)
    assert corrector.limits_at(4) is limits


def test_corrector_deterministic(corrector: MidCourseCorrector) -> None:
    """The same inputs give the identical schedule."""
    batches, _ = _synthetic_batches()
    batch = batches["b9"]
    one = corrector.correct(batch.iloc[:4], k=4)
    two = corrector.correct(batch.iloc[:4], k=4)
    pd.testing.assert_frame_equal(one.schedule, two.schedule)


def test_corrector_validation_errors(fitted: BatchPLS) -> None:
    """Constructor and correct() validate their inputs."""
    nominal = pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)})
    with pytest.raises(ValueError, match="mv_tags contains tags"):
        MidCourseCorrector(fitted, nominal, mv_tags=["nope"], y_target=0.0)
    with pytest.raises(ValueError, match="y_target is required"):
        MidCourseCorrector(fitted, nominal, mv_tags=["u"])
    with pytest.raises(ValueError, match="target_side must be"):
        MidCourseCorrector(fitted, nominal, mv_tags=["u"], y_target=0.0, target_side="sideways")
    with pytest.raises(ValueError, match=r"rows \(one per aligned sample\)"):
        MidCourseCorrector(fitted, nominal.iloc[:3], mv_tags=["u"], y_target=0.0)
    corrector = MidCourseCorrector(fitted, nominal, mv_tags=["u"], y_target=0.0)
    batches, _ = _synthetic_batches()
    with pytest.raises(ValueError, match="k must lie in"):
        corrector.correct(batches["b0"], k=0)


@pytest.mark.integration
@pytest.mark.slow
def test_executed_correction_gains_on_simulator() -> None:
    """The locked demo recipe yields positive executed gains for the poor class.

    Trains per-class models on a knot-excited historical campaign and
    corrects fresh replay batches at day 4. Measured on these seeds the
    corrected batches (all in the poorest feed class) gain +0.62 g/L on
    average and none is harmed; the assertions sit well inside that.
    """
    from process_improve.simulation import BioreactorSimulator

    sim = BioreactorSimulator()
    nominal = sim.nominal_trajectory().reset_index(drop=True)
    train = sim.simulate_campaign(200, policy="historical", mv_variation=2.5, random_state=0)
    z_train = train.initial_conditions
    classes = np.array(train.classes)
    mu, sd = z_train.mean(), z_train.std(ddof=1)
    centers = {c: ((z_train - mu) / sd)[classes == c].mean() for c in set(classes)}

    correctors = {}
    for c in set(classes):
        ids = [bid for bid, cc in zip(train.batches, classes, strict=True) if cc == c]
        model = BatchPLS(n_components=4).fit(
            {i: train.batches[i] for i in ids}, train.quality.loc[ids], initial_conditions=z_train.loc[ids]
        )
        correctors[c] = MidCourseCorrector(
            model,
            nominal,
            mv_tags=["pH", "temperature"],
            mode="target",
            y_target=8.0,
            weights={"target": 1.0, "movement": 0.1},
            bounds={"temperature": (28.3, 38.7), "pH": (6.64, 7.56)},
            rate_limits={"temperature": 3.0, "pH": 0.5},
            spe_cap="limit",
            t2_cap="limit",
            dead_band=2.5,
            target_side="below",
            n_knots=4,
        )

    fresh = sim.simulate_campaign(40, policy="replay", random_state=100)
    z_fresh = fresh.initial_conditions
    gains = []
    for bid in list(fresh.batches):
        seed = 2000 + bid
        base = sim.simulate_batch(z_fresh.loc[bid], random_state=seed)
        zq = (z_fresh.loc[bid] - mu) / sd
        c = min(centers, key=lambda cc: ((zq - centers[cc]) ** 2).sum())
        out = correctors[c].correct(base.tags.iloc[:8].reset_index(drop=True), initial_conditions=z_fresh.loc[bid], k=8)
        if out.corrected:
            trajectory = out.schedule.copy()
            trajectory.index = sim.nominal_trajectory().index
            redo = sim.simulate_batch(z_fresh.loc[bid], trajectory, random_state=seed)
            gains.append(redo.titer - base.titer)
    assert len(gains) >= 3
    assert min(gains) > 0.1
    assert float(np.mean(gains)) > 0.3


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_control_policies_structure() -> None:
    """The executed policy comparison returns a coherent, reproducible result."""
    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    result = evaluate_control_policies(
        BioreactorSimulator(),
        y_target=8.0,
        n_train=60,
        n_test=8,
        include_adapted=False,
        oracle="none",
        random_state=0,
    )
    assert set(result.summary.index) == {"replay", "midcourse"}
    assert list(result.summary.columns) == ["mean", "sd", "min", "max"]
    assert len(result.batches) == 8
    # Non-corrected batches carry the replay titer unchanged.
    untouched = result.batches[~result.batches["corrected"]]
    np.testing.assert_allclose(untouched["replay"], untouched["midcourse"])
    assert set(result.batches["reason"]) <= {"corrected", "dead_band", "spe_gate", "batch_complete"}
    # Reproducible end to end.
    again = evaluate_control_policies(
        BioreactorSimulator(),
        y_target=8.0,
        n_train=60,
        n_test=8,
        include_adapted=False,
        oracle="none",
        random_state=0,
    )
    pd.testing.assert_frame_equal(result.batches, again.batches)
