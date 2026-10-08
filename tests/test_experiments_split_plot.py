"""Tests for the split-plot REML analysis (#630).

The reference is Box, Hunter and Hunter's corrosion experiment (chapter 9): a balanced
split plot, for which REML and Satterthwaite reproduce the classical split-plot ANOVA
exactly. Unbalanced designs are checked against statsmodels' general ``MixedLM``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from patsy import dmatrices

from process_improve.experiments import datasets
from process_improve.experiments._analyses.split_plot import _deviance, _score, fit_reml


def _corrosion() -> pd.DataFrame:
    """Return the corrosion data, with temperature as a three-level category, as the book analyses it."""
    return datasets.corrosion().astype({"Temperature": str})


def _classical_anova(df: pd.DataFrame) -> dict[str, float]:
    """Return the sums of squares of the corrosion split plot, from group means, as in BHH Table 9.2."""
    y = df["Resistance"]

    def between(*by: str) -> float:
        return float(((df.groupby(list(by))["Resistance"].transform("mean") - y.mean()) ** 2).sum())

    ss = {"T": between("Temperature"), "C": between("Coating")}
    ss["wp_error"] = between("Heat") - ss["T"]
    ss["TC"] = between("Temperature", "Coating") - ss["T"] - ss["C"]
    ss["sp_error"] = float(((y - y.mean()) ** 2).sum()) - between("Heat") - ss["C"] - ss["TC"]
    return ss


def _terms(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the contrast matrix selecting each model term's columns, the intercept left out."""
    eye = np.eye(X.shape[1])
    return {term: eye[cols] for term, cols in X.design_info.term_name_slices.items() if term != "Intercept"}


@pytest.fixture(scope="module")
def corrosion_fit():
    """REML fit of the full temperature x coating model to the corrosion data."""
    df = _corrosion()
    y, X = dmatrices("Resistance ~ C(Temperature, Sum) * C(Coating, Sum)", df, return_type="dataframe")
    return df, X, fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["Heat"].to_numpy())


class TestCorrosionTextbook:
    """Box, Hunter and Hunter, chapter 9: REML equals the classical split-plot ANOVA."""

    def test_the_data_give_the_published_sums_of_squares(self) -> None:
        """BHH Table 9.2: temperature 26,519, heats 14,440, coatings 4,289, interaction 3,270, error 1,121."""
        ss = _classical_anova(_corrosion())
        assert {key: round(value) for key, value in ss.items()} == {
            "T": 26519,
            "wp_error": 14440,
            "C": 4289,
            "TC": 3270,
            "sp_error": 1121,
        }

    def test_variance_components_are_the_anova_estimates(self, corrosion_fit) -> None:
        """Residual variance = subplot error MS; whole-plot variance = (heats MS - error MS) / 4 bars."""
        df, _, fit = corrosion_fit
        ss = _classical_anova(df)
        ms_wp, ms_sp = ss["wp_error"] / 3, ss["sp_error"] / 9
        assert fit.sigma2 == pytest.approx(ms_sp, rel=1e-12)
        assert fit.sigma2_wp == pytest.approx((ms_wp - ms_sp) / 4, rel=1e-12)

    def test_temperature_is_tested_against_the_heats(self, corrosion_fit) -> None:
        """The whole-plot factor's F is MS_T / MS_heats on (2, 3) df: 2.75, not significant."""
        df, X, fit = corrosion_fit
        ss = _classical_anova(df)
        f_value, df_den = fit.wald_test(_terms(X)["C(Temperature, Sum)"])
        assert f_value == pytest.approx((ss["T"] / 2) / (ss["wp_error"] / 3), rel=1e-10)
        assert round(f_value, 2) == 2.75
        assert df_den == pytest.approx(3.0, rel=1e-10)

    @pytest.mark.parametrize(
        ("term", "key", "df_num"), [("C(Coating, Sum)", "C", 3), ("C(Temperature, Sum):C(Coating, Sum)", "TC", 6)]
    )
    def test_subplot_terms_are_tested_against_the_bars(self, corrosion_fit, term: str, key: str, df_num: int) -> None:
        """Coating and the interaction are tested against the subplot error, on 9 df."""
        df, X, fit = corrosion_fit
        ss = _classical_anova(df)
        f_value, df_den = fit.wald_test(_terms(X)[term])
        assert f_value == pytest.approx((ss[key] / df_num) / (ss["sp_error"] / 9), rel=1e-10)
        assert df_den == pytest.approx(9.0, rel=1e-10)

    def test_a_contrast_across_both_strata_gets_the_classical_satterthwaite_df(self, corrosion_fit) -> None:
        """Coating C1 at 360 against C1 at 370 mixes both errors: Satterthwaite's 1946 formula, exactly.

        Each cell mean is over two heats, so the difference has variance
        ``s2_wp + s2 = MS_heats / 4 + 3 MS_error / 4``, with ``(a + b)^2 / (a^2 / 3 + b^2 / 9)`` df.
        """
        df, X, fit = corrosion_fit
        ss = _classical_anova(df)
        a, b = ss["wp_error"] / 3 / 4, 3 * ss["sp_error"] / 9 / 4
        cells = X.groupby([df["Temperature"], df["Coating"]]).mean()
        contrast = (cells.loc[("360", "C1")] - cells.loc[("370", "C1")]).to_numpy()
        assert contrast @ fit.cov_beta @ contrast == pytest.approx(a + b, rel=1e-10)
        assert fit.satterthwaite_df(contrast) == pytest.approx((a + b) ** 2 / (a**2 / 3 + b**2 / 9), rel=1e-10)


def _unbalanced_split_plot(seed: int = 7) -> pd.DataFrame:
    """Seven whole plots of 2 to 5 runs: whole-plot factor A, subplot factor B, s2_wp = 2, s2 = 1."""
    rng = np.random.default_rng(seed)
    sizes = [2, 3, 4, 5, 3, 4, 2]
    plot = np.repeat(np.arange(len(sizes)), sizes)
    a = np.repeat(np.tile([-1.0, 1.0], 4)[: len(sizes)], sizes)
    b = rng.uniform(-1, 1, len(plot))
    u = rng.normal(0, np.sqrt(2.0), len(sizes))[plot]
    y = 10 + 3 * a - 2 * b + 1.5 * a * b + u + rng.normal(0, 1.0, len(plot))
    return pd.DataFrame({"plot": plot, "A": a, "B": b, "y": y})


class TestAgainstMixedLM:
    """On an unbalanced split plot, the estimates are statsmodels' REML estimates."""

    def test_estimates_match_mixedlm(self) -> None:
        """Fixed effects and both variance components agree with MixedLM(reml=True)."""
        df = _unbalanced_split_plot()
        y, X = dmatrices("y ~ A * B", df, return_type="dataframe")
        fit = fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = smf.mixedlm("y ~ A * B", df, groups=df["plot"]).fit(reml=True)
        np.testing.assert_allclose(fit.beta, reference.fe_params.to_numpy(), rtol=1e-4)
        assert fit.sigma2 == pytest.approx(reference.scale, rel=1e-3)
        assert fit.sigma2_wp == pytest.approx(float(reference.cov_re.iloc[0, 0]), rel=1e-3)

    def test_the_estimate_is_a_stationary_point_no_worse_than_mixedlm(self) -> None:
        """The REML score is zero at the estimate, and the deviance is not above MixedLM's."""
        df = _unbalanced_split_plot()
        y, X = dmatrices("y ~ A * B", df, return_type="dataframe")
        y_, X_ = y.to_numpy().ravel(), X.to_numpy()
        fit = fit_reml(y_, X_, df["plot"].to_numpy())
        Z = pd.get_dummies(df["plot"]).to_numpy(dtype=float)
        eta = fit.sigma2_wp / fit.sigma2
        assert abs(_score(eta, y_, X_, Z)) < 1e-9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = smf.mixedlm("y ~ A * B", df, groups=df["plot"]).fit(reml=True)
        eta_reference = float(reference.cov_re.iloc[0, 0]) / reference.scale
        assert _deviance(eta, y_, X_, Z) <= _deviance(eta_reference, y_, X_, Z) + 1e-12

    def test_whole_plot_effects_get_fewer_df_than_subplot_effects(self) -> None:
        """A's coefficient is judged on about the whole-plot error df; B's on about the subplot df."""
        df = _unbalanced_split_plot()
        y, X = dmatrices("y ~ A * B", df, return_type="dataframe")
        fit = fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())
        eye = np.eye(X.shape[1])
        df_a, df_b = (fit.satterthwaite_df(eye[X.columns.get_loc(name)]) for name in ("A", "B"))
        # 7 whole plots - intercept - A = 5 whole-plot error df; 23 runs - 7 plots - B - A:B = 14 subplot df.
        assert 4.0 < df_a < 6.5
        assert 12.0 < df_b < 16.0


class TestBoundary:
    """A whole-plot variance estimated as zero gives exactly the OLS fit."""

    def test_no_whole_plot_variation_gives_ols(self) -> None:
        """With the noise centred within every whole plot, eta is 0 and the fit is ordinary least squares."""
        df = _unbalanced_split_plot()
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 1.0, len(df))
        noise -= pd.Series(noise).groupby(df["plot"]).transform("mean").to_numpy()
        df["y"] = 10 + 3 * df["A"] - 2 * df["B"] + noise
        y, X = dmatrices("y ~ A * B", df, return_type="dataframe")
        fit = fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())
        ols = smf.ols("y ~ A * B", df).fit()
        assert fit.sigma2_wp == 0.0
        np.testing.assert_allclose(fit.beta, ols.params.to_numpy(), rtol=1e-12)
        assert fit.sigma2 == pytest.approx(ols.mse_resid, rel=1e-12)


class TestEstimability:
    """The fit refuses models whose terms or variance components cannot be separated."""

    def test_aliased_terms_are_refused(self) -> None:
        """A column that duplicates another makes the model matrix rank deficient."""
        df = _unbalanced_split_plot().assign(A2=lambda d: 2 * d["A"])
        y, X = dmatrices("y ~ A + A2 + B", df, return_type="dataframe")
        with pytest.raises(ValueError, match="rank"):
            fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())

    def test_no_whole_plot_error_df_is_refused(self) -> None:
        """A whole-plot factor with a different level in every whole plot leaves nothing to estimate s2_wp."""
        df = _unbalanced_split_plot().assign(W=lambda d: d["plot"].astype(str))
        y, X = dmatrices("y ~ W + B", df, return_type="dataframe")
        with pytest.raises(ValueError, match="whole-plot error"):
            fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())

    def test_no_subplot_error_df_is_refused(self) -> None:
        """One run per whole plot: the two variances cannot be told apart."""
        df = _unbalanced_split_plot()
        y, X = dmatrices("y ~ A + B", df, return_type="dataframe")
        with pytest.raises(ValueError, match="subplot error"):
            fit_reml(y.to_numpy().ravel(), X.to_numpy(), np.arange(len(df)))
