"""PLS fitted to data with missing cells by imputation: ``md_method`` ``"tsr"`` and ``"pmp"`` (#189).

The reason to offer these beside the default NIPALS path is accuracy: they should
reach a model closer to the one the complete data would have given. So the tests
that matter here measure that distance, against NIPALS, rather than only checking
that a fit comes back. The rest pin the exact properties the method is built on,
and the convergence defect that forced it to be rebuilt.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._impute import impute_low_rank
from process_improve.multivariate._pls import _held_out_scores
from process_improve.multivariate.methods import PLS

N_ROWS, N_COLS, RANK = 80, 12, 3

#: Loose enough to keep the statistical tests quick: the gap between converging to
#: this and to the default ``epsqrt`` is far below anything these tests resolve.
QUICK = 1e-6


def _dataset(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a rank-3 X with a linear y and a little noise on both.

    Clean low-rank structure is the case imputation is for: the observed cells of a
    row say a lot about its missing ones, because both come from the same few scores.
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(N_ROWS, RANK))
    X = scores @ rng.normal(size=(N_COLS, RANK)).T + rng.normal(scale=0.3, size=(N_ROWS, N_COLS))
    y = scores @ rng.normal(size=RANK) + rng.normal(scale=0.3, size=N_ROWS)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(N_COLS)]), pd.DataFrame({"y": y})


def _with_gaps(X: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Blank out cells completely at random."""
    return X.mask(np.random.default_rng(seed + 1000).random(X.shape) < fraction)


def _fit(X: pd.DataFrame, Y: pd.DataFrame, method: str = "nipals", **params: float) -> PLS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        return PLS(n_components=RANK, missing_data_settings={"md_method": method}, **params).fit(X, Y)


def _beta(model: PLS) -> np.ndarray:
    return model.beta_coefficients_.to_numpy(dtype=float).ravel()


class TestImputationIsWorthIt:
    """The property that justifies offering these methods at all."""

    @pytest.mark.slow
    @pytest.mark.parametrize("fraction", [0.15, 0.30])
    def test_tsr_lands_closer_to_the_complete_data_model_than_nipals(self, fraction: float) -> None:
        """Measured as ``||beta - beta_complete||`` over eight seeds.

        Stated as a ratio against NIPALS on the same gaps, so the threshold does not
        encode this fixture's noise level. The measured gain here is about 2x to 3x;
        the bound is looser so that it pins the ordering, not the fixture.
        """
        nipals, tsr = [], []
        for seed in range(8):
            X, Y = _dataset(seed)
            reference = _beta(PLS(n_components=RANK).fit(X, Y))
            gapped = _with_gaps(X, fraction, seed)
            nipals.append(np.linalg.norm(_beta(_fit(gapped, Y)) - reference))
            tsr.append(np.linalg.norm(_beta(_fit(gapped, Y, "tsr", tol=QUICK)) - reference))
        assert float(np.median(nipals)) > 1.5 * float(np.median(tsr))

    @pytest.mark.slow
    def test_rescaling_every_round_is_what_buys_the_accuracy(self) -> None:
        """Scaling once, from the incomplete data, fixes the coordinate system at its worst guess.

        The completed data's centre is a better estimate of the complete data's centre
        than the observed cells' mean is, because it borrows from the correlated
        columns. That is why the loop re-estimates the scaling every round, and it is
        checked here where it can be seen directly: on the fitted centre.
        """
        closer = 0
        for seed in range(8):
            X, Y = _dataset(seed)
            gapped = _with_gaps(X, 0.30, seed)
            fitted_centre = _fit(gapped, Y, "tsr", tol=QUICK)._x_scaler.center_.to_numpy()
            observed_mean = gapped.mean().to_numpy()
            true_centre = X.mean().to_numpy()
            closer += np.linalg.norm(fitted_centre - true_centre) < np.linalg.norm(observed_mean - true_centre)
        assert closer >= 6, f"the completed-data centre was the better estimate on only {closer} of 8 seeds"


class TestConvergence:
    """The first version of this feature did not converge; this pins the fix."""

    @pytest.mark.parametrize(("fraction", "seed"), [(0.30, 2), (0.30, 8), (0.30, 9), (0.20, 10), (0.20, 16)])
    def test_the_fixtures_that_used_to_cycle_now_converge(self, fraction: float, seed: int) -> None:
        """Each of these ran to the iteration cap under the PLS-based loop, still moving.

        Rebuilding the missing cells from the PLS model is not an EM, because PLS picks
        its components for their covariance with Y rather than to reproduce X; about
        one fit in seven cycled indefinitely. Imputing from a model of X and Y jointly
        is a genuine EM. These are the exact cases that failed.
        """
        X, Y = _dataset(seed)
        model = _fit(_with_gaps(X, fraction, seed), Y, "tsr", tol=QUICK)
        assert model.fitting_info_["md_converged"]
        assert model.fitting_info_["md_rounds"] < 200


class TestExactProperties:
    """Things that hold exactly, so a regression shows up as a failure rather than a drift."""

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_complete_data_gives_exactly_the_nipals_model(self, method: str) -> None:
        """With nothing to impute there is no loop: the fit is ordinary NIPALS."""
        X, Y = _dataset(0)
        imputed = _fit(X, Y, method)
        np.testing.assert_allclose(_beta(imputed), _beta(_fit(X, Y)), rtol=0, atol=1e-12)
        assert imputed.fitting_info_["md_rounds"] == 0
        assert imputed.fitting_info_["md_converged"]

    def test_the_scaling_is_re_estimated_from_the_completed_data(self) -> None:
        """Scaling once from the gapped data would leave the centre at the observed mean, exactly."""
        X, Y = _dataset(0)
        gapped = _with_gaps(X, 0.30, 0)
        centre = _fit(gapped, Y, "tsr", tol=QUICK)._x_scaler.center_.to_numpy()
        assert not np.allclose(centre, gapped.mean().to_numpy())

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_a_row_with_no_observed_x_is_still_fittable(self, method: str) -> None:
        """NIPALS fits such a row, so switching md_method must not turn the same data into an error.

        Its X is estimated from its y, through how X and Y co-vary in the joint
        model: more than the column mean would say, though a single y can only
        inform one direction of X's variation (it was the closer estimate on 8 of 12
        seeds). A row with nothing observed at all goes to the mean; see
        ``TestImputeLowRank``.
        """
        X, Y = _dataset(0)
        X.iloc[5] = np.nan
        model = _fit(X, Y, method)
        assert model.fitting_info_["md_converged"]
        assert np.all(np.isfinite(model.scores_.to_numpy()))
        _fit(X, Y)  # and NIPALS agrees the data is fittable

    def test_a_row_observing_fewer_columns_than_components_is_well_posed(self) -> None:
        """Estimated with as many components as it has observed columns, not a singular inverse.

        One observed column cannot pin down three scores. The first version of this
        feature reached for a ridge to force the inverse through; trimming the row's
        components to what it can support is exact rather than regularised.
        """
        X, Y = _dataset(0)
        X.iloc[3, 1:] = np.nan  # one observed column, three components
        model = _fit(X, Y, "tsr")
        assert model.fitting_info_["md_converged"]
        assert np.all(np.isfinite(model.scores_.to_numpy()))

    def test_a_missing_training_cell_leaves_no_residual(self) -> None:
        """The residual block ``project`` regresses with must not count imputed cells as fitted.

        An imputed cell is fitted by construction; if it carried its (zero-by-design)
        residual as though observed, projection would be told the model fits better
        than it does.
        """
        X, Y = _dataset(0)
        gapped = _with_gaps(X, 0.20, 0)
        residuals = _fit(gapped, Y, "tsr")._x_residuals
        np.testing.assert_array_equal(residuals[gapped.isna().to_numpy()], 0.0)


class TestWhatIsImputed:
    @pytest.mark.slow
    def test_imputing_a_missing_y_beats_dropping_its_row(self) -> None:
        """A row with no y still has an X, and that X is information about the model.

        Dropping the row throws it away; imputing the y from the joint model keeps it.
        Measured over twelve seeds with 8 of 80 targets missing, the imputed fit sat
        about twice as close to the complete-data model as the dropped-rows fit.
        """
        imputed, dropped = [], []
        for seed in range(12):
            X, Y = _dataset(seed)
            reference = _beta(PLS(n_components=RANK).fit(X, Y))
            gapped = Y.copy()
            gapped.iloc[np.random.default_rng(seed).choice(N_ROWS, 8, replace=False), 0] = np.nan
            imputed.append(np.linalg.norm(_beta(_fit(X, gapped, "tsr", tol=QUICK)) - reference))
            kept = gapped["y"].notna()
            dropped.append(np.linalg.norm(_beta(PLS(n_components=RANK).fit(X[kept], gapped[kept])) - reference))
        assert float(np.median(imputed)) < float(np.median(dropped))

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_gaps_in_both_blocks(self, method: str) -> None:
        X, Y = _dataset(0)
        gapped_y = Y.copy()
        gapped_y.iloc[[2, 7], 0] = np.nan
        model = _fit(_with_gaps(X, 0.20, 0), gapped_y, method)
        assert model.fitting_info_["md_converged"]
        assert np.all(np.isfinite(model.predict(X).to_numpy()))


class TestTheSurroundingApi:
    def test_scale_false(self) -> None:
        X, Y = _dataset(0)
        scaled_x, scaled_y = (X - X.mean()) / X.std(), (Y - Y.mean()) / Y.std()
        model = PLS(n_components=RANK, scale=False, missing_data_settings={"md_method": "tsr"}).fit(
            _with_gaps(scaled_x, 0.20, 0), scaled_y
        )
        assert model.fitting_info_["md_converged"]

    def test_sample_weight_passes_through(self) -> None:
        X, Y = _dataset(0)
        weights = np.random.default_rng(0).uniform(0.2, 1.0, N_ROWS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecificationWarning)
            model = PLS(n_components=RANK, tol=QUICK, missing_data_settings={"md_method": "tsr"}).fit(
                _with_gaps(X, 0.20, 0), Y, sample_weight=weights
            )
        assert model.fitting_info_["md_converged"]

    def test_predict_and_project_after_an_imputed_fit(self) -> None:
        X, Y = _dataset(0)
        gapped = _with_gaps(X, 0.20, 0)
        model = _fit(gapped, Y, "tsr")
        assert model.predict(X).shape == (N_ROWS, 1)
        assert model.project(gapped.iloc[:5]).scores.shape == (5, RANK)

    def test_non_convergence_warns_and_says_what_to_change(self) -> None:
        X, Y = _dataset(0)
        with pytest.warns(SpecificationWarning, match="still moving.*Raise max_iter"):
            model = PLS(n_components=RANK, max_iter=2, missing_data_settings={"md_method": "tsr"}).fit(
                _with_gaps(X, 0.20, 0), Y
            )
        assert not model.fitting_info_["md_converged"]
        assert model.fitting_info_["md_rounds"] == 2

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_a_setpoint_column_leaves_the_imputation_as_it_was(self, method: str) -> None:
        """A constant column kept the imputation moving to the round limit, and the fit warned."""
        X, Y = _dataset(0)
        gapped = _with_gaps(X, 0.20, 0)
        with_setpoint = gapped.assign(setpoint=0.1)
        with_setpoint.loc[3, "setpoint"] = np.nan
        alone = _fit(gapped, Y, method)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SpecificationWarning)
            both = PLS(n_components=RANK, missing_data_settings={"md_method": method}).fit(with_setpoint, Y)
        assert both.fitting_info_["md_converged"]
        assert both.fitting_info_["md_rounds"] == alone.fitting_info_["md_rounds"]

    def test_scp_is_still_refused(self) -> None:
        """It is a projection-time method; at fit time it is simply what NIPALS already does."""
        X, Y = _dataset(0)
        with pytest.raises(ValueError, match="md_method must be one of"):
            _fit(_with_gaps(X, 0.20, 0), Y, "scp")

    def test_select_n_components_predicts_every_held_out_row(self) -> None:
        """#676: a held-out row with a missing X cell still gets a cross-validated prediction.

        Each fold's model copes with missing cells, but the held-out rows were scored by a
        plain product with the direct weights, so one gap made that row's prediction NaN:
        21 of these 40 rows.
        """
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((40, 6)), columns=[f"x{i}" for i in range(6)])
        y = pd.DataFrame({"y": X.to_numpy() @ rng.normal(size=6) + 0.5 * rng.standard_normal(40)})
        gappy = X.mask(rng.random(X.shape) < 0.15)

        result = PLS.select_n_components(
            gappy, y, cv=5, n_repeats=1, max_components=2, n_permutations=19, random_state=0
        )
        assert np.isfinite(result.cv_predictions.to_numpy()).all()


class TestHeldOutRowsWithGaps:
    """#676: cross-validation scores a held-out row as the model with that many components would."""

    @pytest.fixture
    def fold(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X, Y = _dataset(3)
        gappy = _with_gaps(X, 0.15, 3)
        x_train, y_train = gappy.iloc[:40], Y.iloc[:40]
        x_test = gappy.iloc[40:]
        assert x_test.isna().any(axis=1).sum() > 0, "some held-out rows have a gap"
        return x_train, y_train, x_test

    @pytest.mark.parametrize("n_components", [1, 2, 3])
    def test_the_scores_are_those_of_a_model_with_that_many_components(self, fold, n_components: int) -> None:
        """The fold model has five components; the first ``n_components`` give the smaller model's scores.

        With gaps in the training block, the direct weights of the first components change with
        the number fitted, so the first columns of the fold model's were not the smaller model's.
        """
        x_train, y_train, x_test = fold
        large = PLS(n_components=5, scale=False).fit(x_train, y_train)
        small = PLS(n_components=n_components, scale=False).fit(x_train, y_train)
        got = _held_out_scores(large, x_train.to_numpy(), x_test.to_numpy(), n_components)
        np.testing.assert_allclose(got, small.project(x_test).scores.to_numpy(), atol=1e-8)

    @pytest.mark.parametrize("rule", ["1se", "min", "q2_increment", "randomization"])
    def test_a_count_some_row_cannot_be_scored_at_is_not_recommended(self, rule: str) -> None:
        """Four columns, every row missing one: no row determines four scores, so that count has no RMSECV."""
        rng = np.random.default_rng(5)
        X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
        y = pd.DataFrame({"y": X.to_numpy() @ [1.0, -2.0, 0.5, 1.5] + 0.3 * rng.normal(size=30)})
        gappy = X.mask(np.eye(4, dtype=bool)[np.arange(30) % 4])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecificationWarning)  # NIPALS on these gaps reaches its cap once
            result = PLS.select_n_components(
                gappy, y, cv=5, n_repeats=1, selection_rule=rule, n_permutations=19, random_state=0
            )
        assert np.isfinite(result.rmsecv["total"].loc[1:3]).all()
        assert np.isnan(result.rmsecv["total"].loc[4])
        assert result.n_components < 4
        assert np.isfinite(result.cv_predictions.to_numpy()).all()
        if rule == "randomization":
            assert np.isnan(result.randomization_pvalues.loc[4]), "a count left out is not judged"
            assert np.isfinite(result.randomization_pvalues.loc[1:3]).all()

    def test_nested_cross_validation_predicts_every_row(self) -> None:
        X, Y = _dataset(4)
        result = PLS.nested_cv(
            _with_gaps(X, 0.15, 4), Y, max_components=3, outer_cv=3, inner_cv=3, n_inner_repeats=1, random_state=0
        )
        assert np.isfinite(result.cv_predictions.to_numpy()).all()


class TestImputeLowRank:
    """The imputation primitive on its own, where its properties can be stated exactly."""

    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_recovers_an_exactly_low_rank_matrix(self, method: str) -> None:
        """With no noise, the observed cells fully determine the missing ones.

        A rank-2 matrix with a fifth of its cells removed, imputed with two
        components, comes back to within 1e-9 of the truth: the strongest
        correctness check available, because nothing about it is statistical.
        """
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 8)) + rng.normal(size=8) * 5
        gapped = truth.copy()
        mask = rng.random(truth.shape) < 0.2
        gapped[mask] = np.nan

        result = impute_low_rank(gapped, 2, method=method, tol=1e-12, max_iter=5000)
        assert result.converged
        np.testing.assert_allclose(result.completed[mask], truth[mask], atol=1e-9)
        np.testing.assert_array_equal(result.completed[~mask], truth[~mask])

    @pytest.mark.parametrize("value", [0.00871, 0.1, 3.7, 63.696])
    @pytest.mark.parametrize("method", ["tsr", "pmp"])
    def test_a_constant_column_changes_no_other_imputation(self, method: str, value: float) -> None:
        """A constant column carries no information, so appending one changes no other imputed cell.

        Its standard deviation is more often rounding noise (1e-17) than 0, and standardising
        by that noise made it a column of +-0.98, a direction the low-rank model then fitted.
        Its own gap is filled with its value.
        """
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 8)) + 0.3 * rng.normal(size=(40, 8))
        gapped = np.where(rng.random(truth.shape) < 0.2, np.nan, truth)
        with_constant = np.column_stack([gapped, np.full(40, value)])
        with_constant[3, -1] = np.nan

        settings = {"method": method, "tol": 1e-12, "max_iter": 5000}
        alone = impute_low_rank(gapped, 2, **settings).completed
        both = impute_low_rank(with_constant, 2, **settings).completed
        np.testing.assert_allclose(both[:, :-1], alone, atol=1e-8)
        np.testing.assert_array_equal(both[:, -1], value)

    def test_complete_data_is_returned_untouched(self) -> None:
        data = np.random.default_rng(0).normal(size=(10, 4))
        result = impute_low_rank(data, 2)
        assert result.rounds == 0
        assert result.converged
        np.testing.assert_array_equal(result.completed, data)

    def test_a_row_with_nothing_observed_goes_to_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(30, 4))
        data[5] = np.nan
        completed = impute_low_rank(data, 2).completed
        np.testing.assert_allclose(completed[5], completed.mean(axis=0), atol=1e-8)

    def test_a_column_with_nothing_observed_is_refused(self) -> None:
        data = np.random.default_rng(0).normal(size=(10, 4))
        data[:, 2] = np.nan
        with pytest.raises(ValueError, match=r"Columns at positions \[2\] have no observed values"):
            impute_low_rank(data, 2)

    @pytest.mark.parametrize(
        ("kwargs", "message"), [({"method": "scp"}, "method must be one of"), ({"n_components": 0}, "at least 1")]
    )
    def test_bad_arguments_are_refused(self, kwargs: dict, message: str) -> None:
        data = np.random.default_rng(0).normal(size=(10, 4))
        settings = {"n_components": 2, **kwargs}
        with pytest.raises(ValueError, match=message):
            impute_low_rank(data, **settings)
