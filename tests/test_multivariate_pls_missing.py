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


def _fit(X: pd.DataFrame, Y: pd.DataFrame, method: str = "nipals", **settings: float) -> PLS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        return PLS(n_components=RANK, missing_data_settings={"md_method": method, **settings}).fit(X, Y)


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
            tsr.append(np.linalg.norm(_beta(_fit(gapped, Y, "tsr", md_tol=QUICK)) - reference))
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
            fitted_centre = _fit(gapped, Y, "tsr", md_tol=QUICK)._x_scaler.center_.to_numpy()
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
        model = _fit(_with_gaps(X, fraction, seed), Y, "tsr", md_tol=QUICK)
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
        centre = _fit(gapped, Y, "tsr", md_tol=QUICK)._x_scaler.center_.to_numpy()
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
            imputed.append(np.linalg.norm(_beta(_fit(X, gapped, "tsr", md_tol=QUICK)) - reference))
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
            model = PLS(n_components=RANK, missing_data_settings={"md_method": "tsr", "md_tol": QUICK}).fit(
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
        with pytest.warns(SpecificationWarning, match="still moving.*Raise md_max_iter"):
            model = PLS(n_components=RANK, missing_data_settings={"md_method": "tsr", "md_max_iter": 2}).fit(
                _with_gaps(X, 0.20, 0), Y
            )
        assert not model.fitting_info_["md_converged"]
        assert model.fitting_info_["md_rounds"] == 2

    def test_scp_is_still_refused(self) -> None:
        """It is a projection-time method; at fit time it is simply what NIPALS already does."""
        X, Y = _dataset(0)
        with pytest.raises(ValueError, match="md_method must be one of"):
            _fit(_with_gaps(X, 0.20, 0), Y, "scp")


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
