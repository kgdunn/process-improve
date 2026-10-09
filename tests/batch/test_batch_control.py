"""Tests for the mid-course correction QP and the decision-point workflow."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

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


@pytest.mark.parametrize(
    ("overrides", "exc", "match"),
    [
        (
            {"observed": {("u", 0): 0.0}},
            TypeError,
            r"observed must be a pandas Series indexed by unfolded column labels; got dict\.",
        ),
        ({"score_covariance": np.eye(2)}, ValueError, r"score_covariance must have shape \(3, 3\); got \(2, 2\)\."),
        (
            {"nominal_remaining": pd.Series(0.0, index=pd.Index([("u", s) for s in range(5, 9)]))},
            ValueError,
            r"nominal_remaining must cover every free column with a finite value\.",
        ),
        (
            {"weights": {"movement": [1.0, 2.0]}},
            ValueError,
            r"weights\['movement'\] must be a scalar or length-5 array; got shape \(2,\)\.",
        ),
        (
            {"weights": {"target": [1.0, 2.0]}},
            ValueError,
            r"weights\['target'\] must be a scalar or length-1 array; got shape \(2,\)\.",
        ),
        ({"y_target": {"other": 0.5}}, ValueError, r"y_target must supply a value for every target in \['q'\]\."),
        (
            {"mode": "maximize"},
            ValueError,
            r"y_target only applies to mode='target'; in maximize mode use weights\['target'\]\.",
        ),
        ({"bounds": {"u": (0.2, 0.1)}}, ValueError, r"bounds for 'u' must satisfy low < high; got \(0\.2, 0\.1\)\."),
        ({"rate_limits": {"u": 0.0}}, ValueError, r"rate_limits for 'u' must be positive; got 0\.0\."),
        ({"n_knots": 1}, ValueError, r"n_knots must lie in \[2, 5\] for 5 free samples; got 1\."),
    ],
    ids=[
        "observed-not-a-series",
        "score-covariance-wrong-shape",
        "nominal-remaining-misses-a-free-column",
        "movement-weight-wrong-length",
        "target-weight-wrong-length",
        "y-target-misses-a-target",
        "y-target-in-maximize-mode",
        "bounds-low-not-below-high",
        "rate-limit-not-positive",
        "fewer-than-two-knots",
    ],
)
def test_midcourse_correction_rejects_bad_arguments(fitted: BatchPLS, overrides: dict, exc: type, match: str) -> None:
    """Each malformed argument is rejected with a message that names it."""
    batches, _ = _synthetic_batches()
    kwargs = {
        "observed": _observed_series(fitted, batches["b0"], 5),
        "free_columns": _free_columns(fitted, 5),
        "y_target": 0.5,
    }
    with pytest.raises(exc, match=match):
        midcourse_correction(fitted, **{**kwargs, **overrides})


def test_infeasible_constraints_raise(fitted: BatchPLS) -> None:
    """A box the seam rate limit cannot reach leaves osqp without a solution, which is an error."""
    batches, _ = _synthetic_batches()
    with pytest.raises(RuntimeError, match=r"The mid-course QP did not solve: osqp status '[^']*infeasible"):
        midcourse_correction(
            fitted,
            observed=_observed_series(fitted, batches["b0"], 5),
            free_columns=_free_columns(fitted, 5),
            y_target=0.5,
            bounds={"u": (0.5, 0.6)},
            rate_limits={"u": 0.05},
            seam={"u": 0.0},  # 0.5 is ten rate-limited steps away; there are five free samples
        )


def test_maximize_mode_raises_the_prediction_by_less_as_movement_costs_more(fitted: BatchPLS) -> None:
    """In maximize mode the predicted quality rises above no-change; a dearer movement buys a smaller rise."""
    batches, _ = _synthetic_batches()
    kwargs = {
        "observed": _observed_series(fitted, batches["b0"], 5),
        "free_columns": _free_columns(fitted, 5),
        "mode": "maximize",
    }
    cheap = midcourse_correction(fitted, weights={"movement": 0.1}, **kwargs)
    dear = midcourse_correction(fitted, weights={"movement": 1.0}, **kwargs)
    gain_cheap = float(cheap.y_hat.iloc[0] - cheap.y_hat_no_change.iloc[0])
    gain_dear = float(dear.y_hat.iloc[0] - dear.y_hat_no_change.iloc[0])
    assert gain_cheap > gain_dear > 0.0
    assert cheap.solver.status == "ok"


def test_y_target_as_mapping_matches_the_scalar(fitted: BatchPLS) -> None:
    """A target given as {name: value} solves the same problem as the bare float."""
    batches, _ = _synthetic_batches()
    kwargs = {"observed": _observed_series(fitted, batches["b0"], 5), "free_columns": _free_columns(fitted, 5)}
    as_float = midcourse_correction(fitted, y_target=0.5, **kwargs)
    as_mapping = midcourse_correction(fitted, y_target={"q": 0.5}, **kwargs)
    pd.testing.assert_series_equal(as_float.mv, as_mapping.mv)


def test_limits_on_a_tag_without_free_columns_are_ignored(fitted: BatchPLS) -> None:
    """Bounds and rate limits on the response tag constrain nothing: the solution is the unconstrained one."""
    batches, _ = _synthetic_batches()
    kwargs = {
        "observed": _observed_series(fitted, batches["b0"], 5),
        "free_columns": _free_columns(fitted, 5),
        "y_target": 0.5,
    }
    plain = midcourse_correction(fitted, **kwargs)
    limited = midcourse_correction(fitted, bounds={"r": (-5.0, 5.0)}, rate_limits={"r": 0.1}, **kwargs)
    pd.testing.assert_series_equal(plain.mv, limited.mv)
    assert limited.active_constraints["bounds"] == []
    assert limited.active_constraints["rate"] == []


def test_rate_limits_without_seam_bind_only_between_free_samples(fitted: BatchPLS) -> None:
    """Without a seam value the first free sample may jump; the steps after it stay within the limit."""
    batches, _ = _synthetic_batches()
    batch = batches["b0"]
    k = 5
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batch, k),
        free_columns=_free_columns(fitted, k),
        y_target=2.0,
        weights={"target": 10.0, "movement": 1e-3},
        rate_limits={"u": 0.05},
    )
    mv = result.mv.to_numpy()
    assert (np.abs(np.diff(mv)) <= 0.05 + 1e-6).all()
    assert abs(mv[0] - float(batch["u"].iloc[k - 1])) > 0.05
    assert result.active_constraints["rate"]
    assert str(("u", k)) not in result.active_constraints["rate"]  # a seam row would carry the first free label


def test_unreachable_spe_cap_reports_cap_not_met(fitted: BatchPLS) -> None:
    """An SPE cap no schedule can meet exhausts the multiplier escalation and says so."""
    batches, _ = _synthetic_batches()
    result = midcourse_correction(
        fitted,
        observed=_observed_series(fitted, batches["b0"], 5),
        free_columns=_free_columns(fitted, 5),
        y_target=0.5,
        spe_cap=1e-9,
    )
    assert result.solver.status == "cap_not_met"
    assert result.solver.n_solves == 31  # the unconstrained solve plus 30 escalations
    assert result.spe > 1e-9
    assert result.active_constraints["spe_cap"]


@pytest.mark.parametrize(
    ("batch_id", "k", "spe_factor", "t2_factor", "slack", "binding"),
    [("b3", 3, 2.0, 0.05, "spe", "t2"), ("b0", 6, 0.2, 0.9, "t2", "spe")],
    ids=["spe-cap-ends-slack", "t2-cap-ends-slack"],
)
def test_multiplier_of_a_slack_cap_is_bisected_away(  # noqa: PLR0913
    fitted: BatchPLS, batch_id: str, k: int, spe_factor: float, t2_factor: float, slack: str, binding: str
) -> None:
    """A multiplier escalated alongside the other is halved towards zero once its own cap is slack.

    The escalation raises both multipliers while both caps are violated; here
    meeting the binding cap also satisfies the other, so the polishing
    bisection drives the slack cap's multiplier far below the first
    escalation step, and the schedule is the one-cap schedule.
    """
    batches, _ = _synthetic_batches()
    kwargs = {
        "observed": _observed_series(fitted, batches[batch_id], k),
        "free_columns": _free_columns(fitted, k),
        "y_target": -2.0,
        "weights": {"target": 10.0, "movement": 1e-3},
    }
    unconstrained = midcourse_correction(fitted, **kwargs)
    caps = {"spe": spe_factor * unconstrained.spe, "t2": t2_factor * unconstrained.t2}
    both = midcourse_correction(fitted, spe_cap=caps["spe"], t2_cap=caps["t2"], **kwargs)
    alone = midcourse_correction(fitted, **{f"{binding}_cap": caps[binding]}, **kwargs)
    assert both.solver.status == "ok"
    assert both.solver[f"{slack}_multiplier"] < 1e-4  # the escalation starts at 1e-3
    assert both[slack] < 0.99 * caps[slack]
    assert 0.99 * caps[binding] <= both[binding] <= 1.01 * caps[binding]
    np.testing.assert_allclose(both.mv.to_numpy(), alone.mv.to_numpy(), atol=1e-4)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#681: cap_status reads 'ok' while a cap is violated",
)
def test_ok_status_means_both_binding_caps_hold(fitted: BatchPLS) -> None:
    """When both caps bind, an 'ok' status must mean both statistics sit within the 1% cap tolerance.

    The polishing loop bisects the SPE multiplier checking only SPE, and the
    T2 multiplier checking only T2. Here lowering the T2 multiplier pushes SPE
    to about 1.27 times its cap; neither bisection can raise a multiplier
    again, both polishing rounds end with the cap violated, and the status
    stays 'ok'.
    """
    batches, _ = _synthetic_batches()
    k = 5
    kwargs = {
        "observed": _observed_series(fitted, batches["b0"], k),
        "free_columns": _free_columns(fitted, k),
        "y_target": -2.0,
        "weights": {"target": 10.0, "movement": 1e-3},
    }
    unconstrained = midcourse_correction(fitted, **kwargs)
    spe_cap, t2_cap = 0.5 * unconstrained.spe, 0.1 * unconstrained.t2
    capped = midcourse_correction(fitted, spe_cap=spe_cap, t2_cap=t2_cap, **kwargs)
    within_caps = capped.spe <= spe_cap * 1.01 and capped.t2 <= t2_cap * 1.01
    assert within_caps or capped.solver.status == "cap_not_met"


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
    batches, _ = _synthetic_batches()
    # No target: the monitoring question is answered, the decision is refused.
    watcher = MidCourseCorrector(fitted, nominal, mv_tags=["u"])
    assert float(watcher.predict(batches["b0"].iloc[:4], k=4).half_width.iloc[0]) > 0
    with pytest.raises(ValueError, match="y_target is required"):
        watcher.correct(batches["b0"].iloc[:4], k=4)
    with pytest.raises(ValueError, match="target_side must be"):
        MidCourseCorrector(fitted, nominal, mv_tags=["u"], y_target=0.0, target_side="sideways")
    with pytest.raises(ValueError, match=r"rows \(one per aligned sample\)"):
        MidCourseCorrector(fitted, nominal.iloc[:3], mv_tags=["u"], y_target=0.0)
    corrector = MidCourseCorrector(fitted, nominal, mv_tags=["u"], y_target=0.0)
    with pytest.raises(ValueError, match="k must lie in"):
        corrector.correct(batches["b0"], k=0)


@pytest.mark.parametrize(
    ("overrides", "exc", "match"),
    [
        ({"mode": "x"}, ValueError, r"mode must be one of \['target', 'maximize'\]; got 'x'\."),
        ({"nominal_schedule": np.zeros((10, 1))}, TypeError, r"nominal_schedule must be a DataFrame; got ndarray\."),
        (
            {"nominal_schedule": pd.DataFrame({"v": np.zeros(10)})},
            ValueError,
            r"nominal_schedule is missing columns for mv_tags: \['u'\]\.",
        ),
    ],
    ids=["unknown-mode", "schedule-not-a-frame", "schedule-without-the-mv-column"],
)
def test_corrector_rejects_bad_settings(fitted: BatchPLS, overrides: dict, exc: type, match: str) -> None:
    """The constructor names the setting it cannot use."""
    kwargs = {"nominal_schedule": pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)}), "y_target": 0.0}
    with pytest.raises(exc, match=match):
        MidCourseCorrector(fitted, mv_tags=["u"], **{**kwargs, **overrides})


def test_corrector_requires_the_default_column_layout() -> None:
    """A model unfolded with group_by_batch=True does not carry the (tag, sample) layout the corrector needs."""
    batches, y = _synthetic_batches()
    grouped = BatchPLS(n_components=3, group_by_batch=True).fit(batches, y)
    with pytest.raises(ValueError, match=r"MidCourseCorrector requires a model fitted with group_by_batch=False\."):
        MidCourseCorrector(grouped, pd.DataFrame({"u": np.zeros(grouped.n_timesteps_)}), mv_tags=["u"], y_target=0.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda c, _batch: c.limits_at(0), r"k must lie in \[1, 10\]; got 0\."),
        (lambda c, batch: c.correct(batch.iloc[:3], k=4), r"batch_so_far has 3 samples but k=4 were requested\."),
        (
            lambda c, batch: c.correct(batch.iloc[:4], implemented_schedule=c.nominal_schedule.iloc[:3]),
            r"implemented_schedule must have 10 rows; got 3\.",
        ),
    ],
    ids=["limits-before-the-first-sample", "fewer-samples-than-k", "short-implemented-schedule"],
)
def test_decision_point_inputs_are_checked(corrector: MidCourseCorrector, call: object, match: str) -> None:
    """A decision point outside the batch, or inputs too short for it, are rejected."""
    batches, _ = _synthetic_batches()
    with pytest.raises(ValueError, match=match):
        call(corrector, batches["b0"])  # type: ignore[operator]


def test_correct_defaults_k_to_the_samples_given(corrector: MidCourseCorrector) -> None:
    """Without k, the decision point is the number of samples recorded so far."""
    batches, _ = _synthetic_batches()
    batch = batches["b5"].iloc[:4]
    implicit = corrector.correct(batch)
    explicit = corrector.correct(batch, k=4)
    assert implicit.k == 4
    pd.testing.assert_frame_equal(implicit.schedule, explicit.schedule)


def test_corrector_float_cap_is_used_as_given_and_none_disables(fitted: BatchPLS) -> None:
    """A float T2 cap is enforced as given; spe_cap=None leaves SPE above the limit 'limit' would impose."""
    batches, _ = _synthetic_batches()
    capped = MidCourseCorrector(
        fitted,
        pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)}),
        mv_tags=["u"],
        y_target=2.0,
        weights={"target": 10.0, "movement": 1e-3},
        dead_band=0.0,
        spe_cap=None,
        t2_cap=5.0,
    )
    out = capped.correct(batches["b7"].iloc[:4], k=4)
    assert out.corrected
    assert out.correction.t2 <= 5.0 * 1.01
    assert out.correction.active_constraints["t2_cap"]
    assert not out.correction.active_constraints["spe_cap"]
    assert out.correction.spe > out.spe_limit_candidate


def test_corrector_dead_band_above_side(fitted: BatchPLS) -> None:
    """target_side='above' leaves batches predicted at or below the target alone and corrects those above it."""
    batches, y = _synthetic_batches()
    one_sided = MidCourseCorrector(
        fitted,
        pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)}),
        mv_tags=["u"],
        y_target=0.0,
        target_side="above",
        dead_band=0.0,
        weights={"target": 5.0, "movement": 1e-3},
    )
    low = one_sided.correct(batches[y["q"].idxmin()].iloc[:5], k=5)
    high = one_sided.correct(batches[y["q"].idxmax()].iloc[:5], k=5)
    assert low.reason == "dead_band"
    assert float(low.dead_band_margin.iloc[0]) == 0.0
    assert high.reason == "corrected"
    assert float(high.dead_band_margin.iloc[0]) > 0.0


def test_corrector_dead_band_both_sides(fitted: BatchPLS) -> None:
    """The default two-sided band leaves a deviation within one half-width alone and corrects a larger one."""
    batches, _ = _synthetic_batches()
    batch = batches["b3"].iloc[:5]
    nominal = pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)})
    prediction = MidCourseCorrector(fitted, nominal, mv_tags=["u"]).predict(batch, k=5)
    y_hat, half_width = float(prediction.y_hat.iloc[0]), float(prediction.half_width.iloc[0])

    def correct_towards(target: float) -> object:
        corrector = MidCourseCorrector(
            fitted, nominal, mv_tags=["u"], y_target=target, weights={"target": 5.0, "movement": 1e-3}
        )
        return corrector.correct(batch, k=5)

    inside = correct_towards(y_hat + 0.5 * half_width)  # predicted below the target
    outside = correct_towards(y_hat - 2.0 * half_width)  # predicted above the target
    assert inside.reason == "dead_band"
    assert float(inside.dead_band_margin.iloc[0]) == pytest.approx(0.5)
    assert outside.reason == "corrected"
    assert float(outside.dead_band_margin.iloc[0]) == pytest.approx(2.0)


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
            dead_band=1.0,
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


@pytest.fixture(scope="module")
def bioreactor() -> Bunch:
    """Return a simulator and a corrector fitted on a small varied campaign, with no dead band."""
    from process_improve.simulation import BioreactorSimulator, sample_initial_conditions

    sim = BioreactorSimulator()
    train = sim.simulate_campaign(30, policy="historical", mv_variation=2.5, random_state=0)
    model = BatchPLS(n_components=4).fit(train.batches, train.quality, initial_conditions=train.initial_conditions)
    corrector = MidCourseCorrector(
        model,
        sim.nominal_trajectory().reset_index(drop=True),
        mv_tags=["pH", "temperature"],
        y_target=8.0,
        weights={"target": 1.0, "movement": 0.1},
        dead_band=0.0,
    )
    z_row = sample_initial_conditions(1, random_state=3).z.iloc[0]
    return Bunch(sim=sim, corrector=corrector, z_row=z_row)


def test_evaluate_control_policies_rejects_unknown_oracle() -> None:
    """An unknown oracle mode is rejected before the simulator is touched."""
    from process_improve.batch.control import evaluate_control_policies

    with pytest.raises(ValueError, match=r"oracle must be 'corrected' or 'none'; got 'x'\."):
        evaluate_control_policies(object(), y_target=8.0, oracle="x")


@pytest.mark.integration
def test_evaluate_control_policies_passes_explicit_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit bounds and rate limits reach the corrector in place of the simulator-derived defaults."""
    from process_improve.batch import control
    from process_improve.simulation import BioreactorSimulator

    settings = []

    class RecordingCorrector(control.MidCourseCorrector):
        def __init__(self, model: BatchPLS, nominal_schedule: pd.DataFrame, **kwargs: object) -> None:
            settings.append(kwargs)
            super().__init__(model, nominal_schedule, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(control, "MidCourseCorrector", RecordingCorrector)
    bounds = {"temperature": (29.0, 38.0), "pH": (6.7, 7.5)}
    rate_limits = {"temperature": 1.0, "pH": 0.1}
    control.evaluate_control_policies(
        BioreactorSimulator(),
        y_target=8.0,
        n_train=30,
        n_test=1,
        per_class=False,
        bounds=bounds,
        rate_limits=rate_limits,
        include_adapted=False,
        oracle="none",
        random_state=0,
    )
    assert [s["bounds"] for s in settings] == [bounds]
    assert [s["rate_limits"] for s in settings] == [rate_limits]


@pytest.mark.integration
def test_evaluate_control_policies_adds_adapted_and_oracle_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapted row runs each batch's optimal schedule; the oracle row scores corrected batches from k."""
    from process_improve.batch import control
    from process_improve.simulation import BioreactorSimulator

    sim = BioreactorSimulator()
    optimiser_calls, oracle_calls = [], []

    def nominal_is_optimal(z_row: pd.Series, **kwargs: object) -> Bunch:
        optimiser_calls.append(kwargs)
        return Bunch(trajectory=sim.nominal_trajectory())

    def oracle(simulator: object, z_row: pd.Series, seed: int, k: int) -> float:
        oracle_calls.append(k)
        return 100.0 + k

    monkeypatch.setattr(sim, "optimal_trajectory", nominal_is_optimal)
    monkeypatch.setattr(control, "_oracle_remaining", oracle)
    result = control.evaluate_control_policies(
        sim,
        y_target=7.0,  # with the default target_side="below", one of the three test batches is predicted above it
        n_train=30,
        n_test=3,
        per_class=False,
        dead_band=0.0,
        adapted_n_knots=3,
        adapted_n_starts=2,
        random_state=0,
    )
    batches = result.batches
    assert list(result.summary.index) == ["replay", "midcourse", "oracle_from_k", "adapted"]
    # The stand-in optimum is the nominal schedule, so the adapted batch replays the nominal one.
    np.testing.assert_array_equal(batches["adapted"], batches["replay"])
    assert optimiser_calls == [{"n_knots": 3, "n_starts": 2, "random_state": 0}] * 3
    # Only the corrected batches go to the oracle; the others count at their mid-course titer.
    corrected = batches["corrected"].to_numpy(dtype=bool)
    assert corrected.any()
    assert not corrected.all()
    assert oracle_calls == [8] * int(corrected.sum())
    np.testing.assert_array_equal(batches["oracle_from_k"].to_numpy()[corrected], 108.0)
    assert batches["oracle_from_k"][~corrected].isna().all()
    expected = batches["oracle_from_k"].fillna(batches["midcourse"]).mean()
    assert result.summary.loc["oracle_from_k", "mean"] == pytest.approx(expected)


@pytest.mark.integration
def test_oracle_is_never_below_the_nominal_remainder(bioreactor: Bunch) -> None:
    """The oracle's search starts on the nominal remainder, so its titer cannot fall below the replay titer.

    From sample 10 the nominal schedule holds the production setpoints, so two
    knots per tag reproduce it exactly and the starting point is the replay.
    """
    from process_improve.batch.control import _oracle_remaining

    replay = bioreactor.sim.simulate_batch(bioreactor.z_row, random_state=7).titer
    oracle = _oracle_remaining(bioreactor.sim, bioreactor.z_row, 7, 10, n_knots=2, max_evaluations=10)
    assert oracle >= replay


@pytest.mark.integration
def test_later_decision_points_keep_the_first_correction_on_record(bioreactor: Bunch) -> None:
    """Corrected at both points, the record keeps the first point and its predictions, and the titer moves on."""
    from process_improve.batch.control import _run_decision_points

    sim, corrector, z_row = bioreactor.sim, bioreactor.corrector, bioreactor.z_row
    twice = _run_decision_points(sim, corrector, z_row, (8, 12), 7)
    once = _run_decision_points(sim, corrector, z_row, (8,), 7)
    first = corrector.correct(
        sim.simulate_batch(z_row, random_state=7).tags.iloc[:8].reset_index(drop=True), initial_conditions=z_row, k=8
    )
    assert first.corrected
    assert twice["decision_point"] == 8
    assert twice["reason"] == "corrected"
    assert twice["y_hat_predicted"] == float(first.y_hat.iloc[0])
    assert twice["y_hat_no_change"] == float(first.y_hat_no_change.iloc[0])
    assert twice["half_width"] == float(first.half_width.iloc[0])
    assert twice["replay"] == once["replay"]
    assert twice["midcourse"] != once["midcourse"]  # the second correction was executed too


def test_limits_carry_decision_point_error_and_conditioning(corrector: MidCourseCorrector) -> None:
    """rmse_k is the model's error under the decision point's pattern; at the full row it is the training RMSE."""
    model = corrector.model
    T = model.n_timesteps_
    # k=1 observes two columns for three components: the estimator is singular
    # there, by construction. Two samples are the earliest usable point.
    early, full = corrector.limits_at(2), corrector.limits_at(T)
    assert np.all(np.isfinite(early.rmse_k))
    assert float(early.rmse_k.iloc[0]) > 0
    assert early.condition_number_candidate >= 1.0
    assert early.condition_number_monitor >= 1.0
    # With everything observed the candidate pattern is the complete row, so the
    # error equals the full-row training RMSE on N - A - 1 degrees of freedom.
    n, A = model.n_samples_, int(model.n_components)
    expected = float(model.rmse_.iloc[0, -1]) * np.sqrt(n / (n - A - 1))
    assert abs(float(full.rmse_k.iloc[0]) - expected) < 1e-8
    # Less of the batch observed cannot make the model's own fit better.
    assert float(early.rmse_k.iloc[0]) > float(full.rmse_k.iloc[0])


def test_predict_is_the_no_change_prediction_and_narrows(corrector: MidCourseCorrector) -> None:
    """predict() matches correct()'s no-change prediction, widens early, and equals predict() at the end."""
    batches, _ = _synthetic_batches()
    bid = "b7"
    batch = batches[bid]
    model = corrector.model
    T = model.n_timesteps_
    early = corrector.predict(batch.iloc[:2], k=2)
    late = corrector.predict(batch.iloc[:T], k=T)
    assert float(early.half_width.iloc[0]) > float(late.half_width.iloc[0])
    assert float(early.lower.iloc[0]) < float(early.y_hat.iloc[0]) < float(early.upper.iloc[0])
    assert early.in_control
    assert early.condition_number >= 1.0
    # The complete row: the same prediction as the model's ordinary predict().
    ordinary = model.predict({bid: batch})
    assert abs(float(late.y_hat.iloc[0]) - float(ordinary.y_hat.iloc[0, 0])) < 1e-8
    # correct() reports predict()'s numbers, whatever it decides.
    out = corrector.correct(batch.iloc[:4], k=4)
    same = corrector.predict(batch.iloc[:4], k=4)
    assert abs(float(out.y_hat_no_change.iloc[0]) - float(same.y_hat.iloc[0])) < 1e-10
    assert abs(float(out.half_width.iloc[0]) - float(same.half_width.iloc[0])) < 1e-10
    assert out.condition_number == same.condition_number
    # k defaults to the number of samples handed in.
    default_k = corrector.predict(batch.iloc[:4])
    assert default_k.k == 4
    assert abs(float(default_k.y_hat.iloc[0]) - float(same.y_hat.iloc[0])) < 1e-12
    with pytest.raises(ValueError, match="k must lie in"):
        corrector.predict(batch, k=T + 1)
    with pytest.raises(ValueError, match="batch_so_far has 3 samples"):
        corrector.predict(batch.iloc[:3], k=4)
    with pytest.raises(ValueError, match="schedule must have"):
        corrector.predict(batch.iloc[:4], schedule=corrector.nominal_schedule.iloc[:3], k=4)


def test_predict_uses_the_planned_schedule(corrector: MidCourseCorrector) -> None:
    """A different remaining schedule changes the prediction; the past rows do not matter to it."""
    batches, _ = _synthetic_batches()
    batch = batches["b7"]
    k = 4
    nominal = corrector.predict(batch.iloc[:k], k=k)
    moved = corrector.nominal_schedule.copy()
    moved.iloc[k:, 0] = 0.5
    with_move = corrector.predict(batch.iloc[:k], schedule=moved, k=k)
    assert abs(float(with_move.y_hat.iloc[0]) - float(nominal.y_hat.iloc[0])) > 1e-3
    past_only = moved.copy()
    past_only.iloc[k:, 0] = corrector.nominal_schedule.iloc[k:, 0]
    past_only.iloc[:k, 0] = 9.0
    unchanged = corrector.predict(batch.iloc[:k], schedule=past_only, k=k)
    assert abs(float(unchanged.y_hat.iloc[0]) - float(nominal.y_hat.iloc[0])) < 1e-12


def test_correct_at_penultimate_sample_with_knots(fitted: BatchPLS) -> None:
    """One remaining free sample cannot carry a knot basis; the correction still runs."""
    nominal = pd.DataFrame({"u": np.zeros(fitted.n_timesteps_)})
    knotted = MidCourseCorrector(
        fitted,
        nominal,
        mv_tags=["u"],
        mode="target",
        y_target=0.5,
        dead_band=0.0,
        n_knots=4,
        weights={"target": 5.0, "movement": 1e-3},
    )
    batches, _ = _synthetic_batches()
    T = fitted.n_timesteps_
    out = knotted.correct(batches["b5"].iloc[: T - 1], k=T - 1)
    assert out.reason in ("corrected", "dead_band")
    assert out.schedule.shape[0] == T
