# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Contribution plots, checked against the paper that defines them.

Source
------
Miller, P., Swanson, R.E. and Heckler, C.E. (1994). "Contribution plots: a
missing link in multivariate quality control." Presented at the ASQC/ASA Fall
Technical Conference; later in *Applied Mathematics and Computer Science*,
8(4), 775-792.

https://literature.learnche.org/item/78/contribution-plots-a-missing-link-in-multivariate-quality-control

Every test below names the equation, page or figure of that paper it encodes.
Page numbers refer to the article as paginated in the PDF at the link above
("Contribution Plots Article - Page N" in its footer).

Why this file exists
--------------------
Until version 1.61.0 ``score_contributions`` back-projected a score-space
difference through the loadings, ``(t_end - t_start) @ P``. That expression
never receives the observation's data: with one component it reduces exactly to
``-t_1 * p_1``, so it returned the loading vector rescaled by a constant and
gave every observation in a data set the same ranking of variables. The tests
that existed asserted only that the output equalled ``dt @ P.T``, which
re-derives the implementation and therefore held whatever it computed.

These tests are written against the paper instead of against the code, so they
constrain the behaviour rather than describe it. The load-bearing ones are
:func:`test_eq3_contributions_sum_to_the_score` (the defining property) and
:func:`test_contributions_are_not_the_loadings`, which is the failure the paper
was written about and the regression this file exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA, PLS, MCUVScaler

PAPER = "Miller, Swanson and Heckler (1994)"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emulsion() -> pd.DataFrame:
    """Correlated process data in the shape of the paper's application.

    The paper monitors a photographic emulsion process: 230 batches of 27
    correlated measurements (flows, pressures, temperatures, pH, mixer speeds),
    of which only a handful of independent phenomena drive the variation
    (pages 3 and 5). This is that structure at a size a test can run: 230 rows
    and 27 variables generated from six latent factors plus noise.
    """
    rng = np.random.default_rng(208)
    n_rows, n_vars, n_factors = 230, 27, 6
    factors = rng.standard_normal((n_rows, n_factors))
    weights = rng.standard_normal((n_factors, n_vars))
    values = factors @ weights + 0.35 * rng.standard_normal((n_rows, n_vars))
    frame = pd.DataFrame(values, columns=[f"v{k:02d}" for k in range(1, n_vars + 1)])
    # Batch numbers, so that a test using labels cannot silently be reading
    # positions: the paper refers to batches 101, 208 and 74 by number.
    frame.index = range(1, n_rows + 1)
    return frame


@pytest.fixture
def emulsion_model(emulsion: pd.DataFrame) -> tuple[PCA, pd.DataFrame]:
    """Fit a 10-component PCA on autoscaled data, as in Figure 1 of the paper."""
    scaled = MCUVScaler().fit_transform(emulsion)
    return PCA(n_components=10).fit(scaled), scaled


# ---------------------------------------------------------------------------
# Equations 1 and 2, page 4: T-squared, from the covariance form and the scores
# ---------------------------------------------------------------------------


def test_eq1_and_eq2_agree_when_every_component_is_kept(emulsion: pd.DataFrame) -> None:
    """T-squared from the covariance form equals T-squared from the scores.

    Equation (1), ``T2_i = x_i S^-1 x_i'``, and equation (2),
    ``T2_i = sum_a t_ia^2 / lambda_a``, are the same quantity when the PCA
    keeps every dimension. The paper notes this equivalence ("It is well known
    that T2 can be computed from the PCA scores") before choosing to truncate
    at ``a`` dimensions.
    """
    scaled = MCUVScaler().fit_transform(emulsion)
    n_vars = scaled.shape[1]
    model = PCA(n_components=n_vars).fit(scaled)

    values = scaled.to_numpy()
    covariance = np.cov(values, rowvar=False)
    from_covariance = np.einsum("ij,jk,ik->i", values, np.linalg.inv(covariance), values)

    scores = model.scores_.to_numpy()
    eigenvalues = scores.var(axis=0, ddof=1)
    from_scores = ((scores**2) / eigenvalues).sum(axis=1)

    assert from_scores == pytest.approx(from_covariance, rel=1e-8)


def test_t2_contributions_sum_to_t2(emulsion_model: tuple[PCA, pd.DataFrame]) -> None:
    """Contributions to T-squared add up to T-squared.

    The T-squared decomposition is the pooled-over-components counterpart of
    equation (3); ``t2_contributions`` already satisfied it before 1.61.0 and
    must keep doing so.
    """
    model, scaled = emulsion_model
    contributions = model.t2_contributions(scaled)
    assert contributions.sum(axis=1).to_numpy() == pytest.approx(
        model.hotellings_t2_.iloc[:, -1].to_numpy(), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Equation 3, pages 6 and 7: the contribution to a score
# ---------------------------------------------------------------------------


def test_eq3_contributions_sum_to_the_score(emulsion_model: tuple[PCA, pd.DataFrame]) -> None:
    """The k contributions to a score add up to that score.

    Equation (3) writes the score as a weighted sum of the data,
    ``t_id = sum_j x_ij p_jd``, and page 7 decomposes it into "k terms
    ``x_ij p_jd`` for j = 1,...,k. These k terms are the contributions to the
    score ``t_id``." Adding up to the score is what makes them contributions
    rather than merely a set of numbers per variable, and it is the property
    the pre-1.61.0 implementation did not have.
    """
    model, scaled = emulsion_model
    for component in range(1, model.n_components + 1):
        contributions = model.score_contributions(scaled, component=component)
        assert contributions.sum(axis=1).to_numpy() == pytest.approx(
            model.scores_[component].to_numpy(), abs=1e-10
        )


def test_eq3_each_term_is_the_datum_times_its_weight(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Each term is literally ``x_ij * p_jd``, not just something that sums right."""
    model, scaled = emulsion_model
    for component in (1, 4, 10):
        contributions = model.score_contributions(scaled, component=component)
        expected = scaled.to_numpy() * model.loadings_.to_numpy()[:, component - 1]
        assert contributions.to_numpy() == pytest.approx(expected, abs=1e-12)


def test_eq3_holds_for_pls_using_the_score_generating_weights(
    emulsion: pd.DataFrame,
) -> None:
    """For PLS the weights are the ones that generate the scores.

    Page 21 notes that contributions apply to "other dimension reduction
    techniques", naming PLS. Westerhuis, Gurden and Smilde (2000) give the
    generalisation: the decomposition uses whatever matrix ``R`` satisfies
    ``T = XR``, which for PLS is ``direct_weights_`` and not the X-loadings.
    Using the loadings here would break the sum in equation (3).
    """
    scaled = MCUVScaler().fit_transform(emulsion)
    y = pd.DataFrame(
        {"quality": scaled.to_numpy() @ np.linspace(1.0, -1.0, scaled.shape[1])}
    )
    model = PLS(n_components=3).fit(scaled, MCUVScaler().fit_transform(y))

    for component in (1, 2, 3):
        contributions = model.score_contributions(scaled, component=component)
        assert contributions.sum(axis=1).to_numpy() == pytest.approx(
            model.scores_[component].to_numpy(), abs=1e-10
        )
        expected = scaled.to_numpy() * model.direct_weights_.to_numpy()[:, component - 1]
        assert contributions.to_numpy() == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Pages 7 and 8, Figure 2: a contribution is not a loading
# ---------------------------------------------------------------------------


def test_contributions_are_not_the_loadings() -> None:
    """A large loading contributes nothing when the observation sits at its mean.

    This is the point of the paper. Page 7: "A practical difference between
    contributions and loadings occurs when some of the process variables have a
    value close to zero, even though those same variables may have large
    loadings." Page 8 gives the case: for batch 208 the loadings pick out
    variables 3, 5, 19 and 25, while the contributions pick out 12, 13, 16 and
    17; the pressures were the real fault, confirmed by univariate charts and
    traced to a replaced valve. "Thus interpreting the loadings would
    potentially detect a different process problem for this batch than actually
    occurred."

    Constructed to make that unambiguous: ``big_loading`` dominates the
    component, but the observation under test sits exactly at its mean, so it
    can have contributed nothing to that observation's score.
    """
    rng = np.random.default_rng(0)
    n_rows = 120
    driver = rng.standard_normal(n_rows)
    correlated = ["press_a", "press_b", "press_c"]
    frame = pd.DataFrame(
        {
            # Three variables moving together: after autoscaling, a variable
            # earns a large loading on component 1 by correlating with others.
            "press_a": driver + 0.05 * rng.standard_normal(n_rows),
            "press_b": driver + 0.05 * rng.standard_normal(n_rows),
            "press_c": driver + 0.05 * rng.standard_normal(n_rows),
            # Nearly independent of them, so its loading is small.
            "silver": rng.standard_normal(n_rows),
        }
    )
    scaled = MCUVScaler().fit_transform(frame)

    # The batch under investigation: every large-loading variable sits exactly
    # at its mean, and the small-loading one is the only thing that moved.
    culprit = 7
    scaled.loc[culprit, correlated] = 0.0
    scaled.loc[culprit, "silver"] = 6.0

    model = PCA(n_components=2).fit(scaled)
    loadings = model.loadings_[1].abs()
    contributions = model.score_contributions(scaled, component=1).loc[culprit]

    # Reading the loadings points at the pressures ...
    assert loadings.idxmax() in correlated
    assert loadings["silver"] < loadings[correlated].min()
    # ... but they contributed nothing to this batch, and the contributions
    # point at the variable that actually moved.
    assert contributions[correlated].to_numpy() == pytest.approx(np.zeros(3), abs=1e-12)
    assert contributions.abs().idxmax() == "silver"
    assert loadings.idxmax() != contributions.abs().idxmax()


def test_contributions_distinguish_observations(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Different observations get different answers.

    The regression guard for the pre-1.61.0 defect. Back-projecting the score
    through the loadings gives a result proportional to the loading vector, so
    the ranking of variables is the same for every observation and the plot
    carries no per-observation information at all. Contributions are per-batch
    by construction (page 7: "Contributions represent the particular process
    variables that were unusual *for a given batch*").
    """
    model, scaled = emulsion_model
    contributions = model.score_contributions(scaled, component=1)

    rankings = {
        tuple(np.argsort(-contributions.iloc[i].abs().to_numpy())) for i in range(len(scaled))
    }
    assert len(rankings) > len(scaled) // 2

    blamed = {contributions.iloc[i].abs().idxmax() for i in range(len(scaled))}
    assert len(blamed) > 1

    # Not proportional to the loadings: that is the defect's signature.
    loadings = model.loadings_.to_numpy()[:, 0]
    for i in range(0, len(scaled), 40):
        row = contributions.iloc[i].to_numpy()
        if np.allclose(row, 0.0):
            continue
        ratios = row / np.where(np.abs(loadings) > 1e-12, loadings, np.nan)
        assert not np.allclose(ratios, ratios[0], equal_nan=True)


# ---------------------------------------------------------------------------
# Equation 4, pages 10 and 12: contributions to Q (the residual, SPE)
# ---------------------------------------------------------------------------


def test_eq4_q_contributions_are_squared_residuals_that_sum_to_q(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Q splits into the squares of the k residual elements.

    Equation (4) defines ``Q = (x_i - x_hat_i)(x_i - x_hat_i)'`` and page 12
    says: "There are k elements in ``(x_i - x_hat_i)``, and the squares of these
    k values are plotted as bars in a contribution plot for Q." Squares, so
    unlike the score contributions these are non-negative, as in Figure 4 where
    every bar points upwards.
    """
    model, scaled = emulsion_model
    residuals = model.spe_contributions(scaled)

    fitted = model.scores_.to_numpy() @ model.loadings_.to_numpy().T
    assert residuals.to_numpy() == pytest.approx(scaled.to_numpy() - fitted, abs=1e-10)

    q_contributions = residuals**2
    assert (q_contributions.to_numpy() >= 0).all()
    q_statistic = (scaled.to_numpy() - fitted) ** 2
    assert q_contributions.sum(axis=1).to_numpy() == pytest.approx(
        q_statistic.sum(axis=1), rel=1e-10
    )
    assert q_contributions.sum(axis=1).to_numpy() == pytest.approx(
        model.spe_.iloc[:, -1].to_numpy() ** 2, rel=1e-8
    )


# ---------------------------------------------------------------------------
# Page 14: contributions for a group of batches
# ---------------------------------------------------------------------------


def test_page14_group_contributions_use_the_group_average(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """A cluster of batches is summarised by averaging the data, then weighting.

    Page 14: "When we compute the contributions, we can replace ``x_ij`` with a
    suitably chosen average, or other linear combination of the data. In this
    case, we would use the average value ``x_bar_.j``, where the averaging is
    over the batches of interest." The paper's example is a cluster of five
    non-sequential batches (31, 142, 147, 220, 221) seen on a score plot.
    """
    model, scaled = emulsion_model
    cluster = [31, 142, 147, 220, 221]

    contributions = model.group_contributions(scaled, group=cluster, component=1)

    mean_row = scaled.loc[cluster].to_numpy().mean(axis=0)
    expected = mean_row * model.loadings_.to_numpy()[:, 0]
    assert contributions.to_numpy() == pytest.approx(expected, abs=1e-12)
    assert contributions.sum() == pytest.approx(model.scores_.loc[cluster, 1].mean(), abs=1e-10)


def test_page14_the_comparison_is_against_the_centre_which_is_zero(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Averaging over every batch gives nothing, because the data are centred.

    Page 14: "This in effect compares the average value of the batches of
    interest to the mean of all the process variables, which is zero for each
    (mean centered) process variable." Taking the whole data set as the group
    is that statement's limiting case.
    """
    model, scaled = emulsion_model
    everything = model.group_contributions(scaled, group=list(scaled.index), component=1)
    assert everything.to_numpy() == pytest.approx(np.zeros(scaled.shape[1]), abs=1e-10)


def test_single_observation_group_is_the_single_observation_contribution(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """A group of one reduces to equation (3) for that batch."""
    model, scaled = emulsion_model
    one = model.group_contributions(scaled, group=[208], component=2)
    direct = model.score_contributions(scaled, component=2).loc[208]
    assert one.to_numpy() == pytest.approx(direct.to_numpy(), abs=1e-12)


# ---------------------------------------------------------------------------
# Page 18, Figures 8 and 9: a level shift between two periods
# ---------------------------------------------------------------------------


def test_page18_level_shift_uses_plus_and_minus_one_tenth_weights(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """The shift diagnostic: ten batches before against ten batches after.

    Page 18 works the case where score 3 steps at batch 74: "the linear
    combination of the data has weights of +0.1 for batches 64 through 73, -0.1
    for batches 74 through 83 and 0 elsewhere. The contributions are the k
    values of ``(sum_{i=64..73} x_ij - sum_{i=74..83} x_ij) p_j3 / 10``."

    Both spellings must agree: the explicit weight vector, and the
    group/reference form that is sugar for it.
    """
    model, scaled = emulsion_model
    before, after = list(range(64, 74)), list(range(74, 84))

    weights = np.zeros(len(scaled))
    positions = scaled.index.get_indexer_for(before)
    weights[positions] = 0.1
    weights[scaled.index.get_indexer_for(after)] = -0.1

    by_weights = model.group_contributions(scaled, weights=weights, component=3)
    by_groups = model.group_contributions(scaled, group=before, reference=after, component=3)
    assert by_weights.to_numpy() == pytest.approx(by_groups.to_numpy(), abs=1e-12)

    # Written out exactly as the paper prints it.
    literal = (
        scaled.loc[before].to_numpy().sum(axis=0) - scaled.loc[after].to_numpy().sum(axis=0)
    ) * model.loadings_.to_numpy()[:, 2] / 10.0
    assert by_weights.to_numpy() == pytest.approx(literal, abs=1e-12)

    scores = model.scores_[3]
    assert by_weights.sum() == pytest.approx(
        scores.loc[before].mean() - scores.loc[after].mean(), abs=1e-10
    )


def test_page18_a_drift_can_be_weighted_by_an_orthogonal_polynomial(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Any linear combination is allowed, including a slope.

    Page 18: "Sometimes, a drift upwards or downwards will be seen in the time
    sequence plot of the scores. In that situation, a linear combination of the
    data that estimates a slope will be useful for determining which process
    variables were drifting... we could use the first order orthogonal
    polynomial for n data points as the weights."
    """
    model, scaled = emulsion_model
    run = list(range(100, 120))
    weights = np.zeros(len(scaled))
    # First-order orthogonal polynomial over n points: centred, evenly spaced.
    weights[scaled.index.get_indexer_for(run)] = np.arange(len(run)) - (len(run) - 1) / 2.0
    assert weights.sum() == pytest.approx(0.0)

    contributions = model.group_contributions(scaled, weights=weights, component=2)

    expected = (weights @ scaled.to_numpy()) * model.loadings_.to_numpy()[:, 1]
    assert contributions.to_numpy() == pytest.approx(expected, abs=1e-12)
    assert contributions.sum() == pytest.approx(
        float(weights @ model.scores_[2].to_numpy()), abs=1e-9
    )


def test_weights_and_groups_are_mutually_exclusive(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """The two spellings of the same idea cannot be combined."""
    model, scaled = emulsion_model
    with pytest.raises(ValueError, match="not both"):
        model.group_contributions(scaled, group=[1, 2], weights=np.zeros(len(scaled)))
    with pytest.raises(ValueError, match="or weights"):
        model.group_contributions(scaled)
    with pytest.raises(ValueError, match="one entry per row"):
        model.group_contributions(scaled, weights=np.zeros(3))


# ---------------------------------------------------------------------------
# Pages 20 and 21: the two scalings
# ---------------------------------------------------------------------------


def test_page20_maximum_contribution_scaling(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Method 1: divide by the largest absolute contribution in the data set.

    Page 20: "for dimension d, we plot ``x_ij p_jd / max_ij |x_ij p_jd|``... we
    compare the contributions for batch i to the maximum, in absolute value, of
    the contributions for all of the batches. If the contribution for batch i is
    +/-1, then this represents the worst deviation from the mean of all of the
    batches over all variables."
    """
    model, scaled = emulsion_model
    raw = model.score_contributions(scaled, component=1)
    scaled_bars = model.score_contributions(scaled, component=1, scaling="maximum")

    assert np.abs(scaled_bars.to_numpy()).max() == pytest.approx(1.0)
    assert scaled_bars.to_numpy() == pytest.approx(
        raw.to_numpy() / np.abs(raw.to_numpy()).max(), abs=1e-12
    )
    # Exactly one variable-batch pair attains it: the worst deviation.
    assert int((np.abs(scaled_bars.to_numpy()) > 1.0 - 1e-12).sum()) == 1


def test_page21_within_batch_scaling(emulsion_model: tuple[PCA, pd.DataFrame]) -> None:
    """Method 2: divide by the total absolute contribution for that batch.

    Page 21: "for dimension d, we plot ``x_ij p_jd / sum_j |x_ij p_jd|``... The
    biggest bars in this method are truly the ones which contribute most to the
    score for this particular batch and the height of the bar is roughly the
    proportion of the variable's contribution."
    """
    model, scaled = emulsion_model
    scaled_bars = model.score_contributions(scaled, component=1, scaling="within")
    totals = np.abs(scaled_bars.to_numpy()).sum(axis=1)
    assert totals == pytest.approx(np.ones(len(scaled)), abs=1e-10)


def test_page21_within_batch_scaling_is_the_exact_proportion_when_signs_agree() -> None:
    """Method 2 gives the exact proportion when every contribution shares a sign.

    Page 21 qualifies method 2 with that parenthesis. Constructed so the
    qualification is met: with every ``x_ij p_j1`` of one sign, each scaled bar
    is exactly that variable's share of the score.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.uniform(1.0, 2.0, size=(40, 4)) + np.arange(4), columns=list("abcd")
    )
    scaled = MCUVScaler().fit_transform(frame)
    model = PCA(n_components=2).fit(scaled)

    raw = model.score_contributions(scaled, component=1)
    within = model.score_contributions(scaled, component=1, scaling="within")

    same_sign = raw.index[
        [bool(np.all(np.sign(row) == np.sign(row[0])) and row[0] != 0) for row in raw.to_numpy()]
    ]
    assert len(same_sign) > 0, "fixture no longer exercises the same-sign case"

    for label in same_sign:
        row = raw.loc[label].to_numpy()
        share = row / row.sum()
        # "The height of the bar is roughly the proportion of the variable's
        # contribution": heights are magnitudes. The scaled value carries the
        # sign of the contribution, so it equals the proportion up to the sign
        # of the score, which is what the division by a sum of absolute values
        # does when the common sign is negative.
        assert np.abs(within.loc[label].to_numpy()) == pytest.approx(np.abs(share), abs=1e-10)
        assert np.sign(within.loc[label].to_numpy()) == pytest.approx(np.sign(row))


def test_page20_scalings_leave_the_pattern_of_bar_heights_unchanged(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """Both scalings zoom, they do not reshape.

    Page 20: "The two methods either 'zoom in' or 'zoom out' on the plot, but
    leave the pattern of bar heights unchanged." So within a batch every bar is
    multiplied by the same positive number: signs are preserved and the ranking
    of variables is untouched.
    """
    model, scaled = emulsion_model
    raw = model.score_contributions(scaled, component=1)

    for scaling in ("maximum", "within"):
        bars = model.score_contributions(scaled, component=1, scaling=scaling)
        for i in (0, 73, 207):
            row_raw = raw.iloc[i].to_numpy()
            row_scaled = bars.iloc[i].to_numpy()
            ratio = row_scaled / row_raw
            assert ratio == pytest.approx(np.full(len(ratio), ratio[0]), rel=1e-10)
            assert ratio[0] > 0
            assert np.array_equal(np.argsort(-np.abs(row_scaled)), np.argsort(-np.abs(row_raw)))


def test_default_scaling_preserves_the_sum_to_the_score(
    emulsion_model: tuple[PCA, pd.DataFrame],
) -> None:
    """The scalings are for presentation; unscaled is the quantity in equation (3).

    Page 20 introduces both methods as ways "to make the variables with the
    biggest contributions stand out visually". Neither preserves the sum, so
    neither is the default here.
    """
    model, scaled = emulsion_model
    unscaled = model.score_contributions(scaled, component=1)
    assert unscaled.sum(axis=1).to_numpy() == pytest.approx(
        model.scores_[1].to_numpy(), abs=1e-10
    )
    for scaling in ("maximum", "within"):
        bars = model.score_contributions(scaled, component=1, scaling=scaling)
        assert not np.allclose(bars.sum(axis=1).to_numpy(), model.scores_[1].to_numpy())


# ---------------------------------------------------------------------------
# The pre-1.61.0 API must not come back
# ---------------------------------------------------------------------------


def test_the_back_projection_api_is_gone(emulsion_model: tuple[PCA, pd.DataFrame]) -> None:
    """Calling the old way raises rather than returning a different number.

    A score vector cannot carry the information equation (3) needs, so the old
    signature cannot be supported alongside the correct one.
    """
    model, scaled = emulsion_model
    with pytest.raises(TypeError, match="not a score vector"):
        model.score_contributions(model.scores_.iloc[0])
    with pytest.raises(TypeError, match="t2_contributions"):
        model.score_contributions(scaled, weighted=True)


def test_contributions_are_not_proportional_to_the_loadings_on_real_data() -> None:
    """The same guard, on the LDPE data set rather than a generated one.

    Real process data, so the test cannot be satisfied by a quirk of the
    generator. Under the pre-1.61.0 calculation every one of these observations
    produced the identical ranking of variables.
    """
    import pathlib

    folder = (
        pathlib.Path(__file__).parents[1]
        / "src"
        / "process_improve"
        / "datasets"
        / "multivariate"
        / "LDPE"
    )
    values = pd.read_csv(folder / "LDPE.csv", index_col=0).select_dtypes("number")
    scaled = MCUVScaler().fit_transform(values)
    model = PCA(n_components=3).fit(scaled)

    contributions = model.score_contributions(scaled, component=1)
    assert contributions.sum(axis=1).to_numpy() == pytest.approx(
        model.scores_[1].to_numpy(), abs=1e-9
    )

    blamed = {contributions.iloc[i].abs().idxmax() for i in range(len(scaled))}
    assert len(blamed) > 1, f"{PAPER}: contributions must vary between observations"


# ---------------------------------------------------------------------------
# Page 21: "contributions can be used with other dimension reduction
# techniques" - here the multi-block models
# ---------------------------------------------------------------------------


@pytest.fixture
def two_blocks(emulsion: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the emulsion variables into two blocks, as a plant would group them."""
    return {"phase1": emulsion.iloc[:, :14], "phase2": emulsion.iloc[:, 14:]}


def test_eq3_holds_for_the_super_score_of_a_multiblock_model(
    two_blocks: dict[str, pd.DataFrame],
) -> None:
    """Summed over every block and variable, the contributions give the super score.

    Equation (3) applied to a super score. The wrinkle is that the super score
    at component ``a`` is formed from the block data deflated through the
    earlier components, so the decomposition has to start from that same
    deflated data rather than from the raw preprocessed blocks.
    """
    from process_improve.multivariate.methods import MBPCA

    model = MBPCA(n_components=3).fit(two_blocks)
    for component in (1, 2, 3):
        contributions = model.score_contributions(two_blocks, component=component)
        total = sum(frame.sum(axis=1) for frame in contributions.values())
        assert total.to_numpy() == pytest.approx(
            model.super_scores_[component].to_numpy(), abs=1e-9
        )


def test_page14_group_contributions_for_a_multiblock_model(
    two_blocks: dict[str, pd.DataFrame],
) -> None:
    """The group and two-period forms carry over to the super score."""
    from process_improve.multivariate.methods import MBPCA

    model = MBPCA(n_components=2).fit(two_blocks)
    index = two_blocks["phase1"].index
    before, after = index[63:73], index[73:83]

    shift = model.group_contributions(two_blocks, group=before, reference=after)
    total = sum(float(series.sum()) for series in shift.values())
    scores = model.super_scores_[1]
    assert total == pytest.approx(
        scores.loc[before].mean() - scores.loc[after].mean(), abs=1e-9
    )


def test_eq4_q_contributions_for_a_multiblock_model(
    two_blocks: dict[str, pd.DataFrame],
) -> None:
    """Equation (4) per block: squared residuals summing to the block SPE.

    The multi-block form of the Q contribution plot of page 12 and Figure 4.
    Squares, so every bar is non-negative, and they sum across a block's
    variables to that block's squared residual distance.
    """
    from process_improve.multivariate.methods import MBPLS

    y = pd.DataFrame(
        {"quality": two_blocks["phase1"].to_numpy() @ np.linspace(1.0, -1.0, 14)},
        index=two_blocks["phase1"].index,
    )
    model = MBPLS(n_components=2).fit(two_blocks, y)

    contributions = model.spe_contributions(two_blocks)
    for name, frame in contributions.items():
        assert (frame.to_numpy() >= 0).all()
        assert frame.sum(axis=1).to_numpy() == pytest.approx(
            model.block_spe_[name].iloc[:, -1].to_numpy() ** 2, rel=1e-8
        )

    # The same input guards as the score-contribution side.
    with pytest.raises(TypeError, match="dict"):
        model.spe_contributions(two_blocks["phase1"])
    with pytest.raises(ValueError, match="Missing X-blocks"):
        model.spe_contributions({"phase1": two_blocks["phase1"]})
    as_arrays = {name: block.to_numpy() for name, block in two_blocks.items()}
    from_arrays = model.spe_contributions(as_arrays)
    for name, frame in from_arrays.items():
        assert frame.to_numpy() == pytest.approx(contributions[name].to_numpy(), abs=1e-12)
