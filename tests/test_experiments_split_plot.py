"""Tests for the split-plot REML analysis (#630).

The reference is Box, Hunter and Hunter's corrosion experiment (chapter 9): a balanced
split plot, for which REML and Satterthwaite reproduce the classical split-plot ANOVA
exactly. Unbalanced designs are checked against statsmodels' general ``MixedLM``.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from patsy import dmatrices

from process_improve.experiments import DesignResult, Factor, datasets, evaluate_design, generate_design
from process_improve.experiments._analyses.split_plot import _deviance, _score, fit_reml
from process_improve.experiments._moment_aberration import moment_aberration
from process_improve.experiments.analysis import analyze_experiment


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


#: An unbalanced split plot (seven whole plots of 2 to 5 runs), at the precision the R
#: reference values below were computed from.
_UNBALANCED_CSV = """plot,A,B,y
0,-1,0.2502,4.9778
0,-1,0.7944,4.1254
1,1,0.5514,11.9675
1,1,-0.5496,12.6876
1,1,-0.3997,14.4824
2,-1,0.7471,3.3132
2,-1,-0.9895,10.1663
2,-1,0.6425,5.3714
2,-1,0.5941,4.0726
3,1,-0.0641,9.3611
3,1,-0.3939,9.7482
3,1,-0.4431,9.7261
3,1,-0.4903,8.4608
3,1,-0.1098,9.5718
4,-1,0.0091,7.5652
4,-1,0.1070,4.3165
4,-1,0.9910,3.6291
5,1,0.5853,12.7581
5,1,0.2444,12.1678
5,1,0.9779,14.4429
5,1,-0.5694,13.9784
6,-1,-0.6796,8.3395
6,-1,0.2251,6.4470
"""


class TestAgainstLmerTest:
    """Reference values from R 4.4.2, lme4 1.1.36, lmerTest 3.1.3 and pbkrtest 0.5.3.

    Computed with::

        u <- lmer(y ~ A * B + (1 | plot), data = unb, REML = TRUE)
        as.data.frame(VarCorr(u))
        coef(summary(u, ddf = "Satterthwaite")); coef(summary(u, ddf = "Kenward-Roger"))

    The estimates, standard errors and variance components are lme4's. The
    Satterthwaite df here use the expected information, as Kenward and Roger (1997)
    do, so they equal pbkrtest's Kenward-Roger df. lmerTest's Satterthwaite df use
    the observed information instead and are a little larger in an unbalanced design
    (5.25 and 15.52); in a balanced one all three agree.
    """

    @pytest.fixture(scope="class")
    def result(self) -> dict:
        """Return the split-plot analysis of the unbalanced data."""
        df = pd.read_csv(io.StringIO(_UNBALANCED_CSV))
        return analyze_experiment(
            df, response_column="y", model="y ~ A * B", whole_plot="plot", analysis_type="split_plot"
        )["split_plot"]

    def test_variance_components_are_lme4s(self, result: dict) -> None:
        """REML whole-plot and residual variances: 1.919521490 and 1.061817805."""
        components = result["variance_components"]
        assert components["whole_plot"] == pytest.approx(1.919521490, rel=1e-6)
        assert components["residual"] == pytest.approx(1.061817805, rel=1e-6)

    def test_estimates_and_standard_errors_are_lme4s(self, result: dict) -> None:
        """Fixed effects and their (unadjusted) standard errors."""
        rows = {row["term"]: row for row in result["coefficients"]}
        expected = {
            "Intercept": (9.158297227, 0.5773092658),
            "A": (2.720163358, 0.5773092658),
            "B": (-1.642715118, 0.4434929460),
            "A:B": (1.538946058, 0.4434929460),
        }
        for term, (estimate, std_error) in expected.items():
            assert rows[term]["coefficient"] == pytest.approx(estimate, rel=1e-6)
            assert rows[term]["std_error"] == pytest.approx(std_error, rel=1e-6)

    def test_df_are_kenward_rogers(self, result: dict) -> None:
        """Expected-information Satterthwaite df equal pbkrtest's: 4.863386965 and 15.211193371."""
        rows = {row["term"]: row for row in result["coefficients"]}
        for term, dof in {"Intercept": 4.863386965, "A": 4.863386965, "B": 15.211193371, "A:B": 15.211193371}.items():
            assert rows[term]["df"] == pytest.approx(dof, rel=1e-6)


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

    def test_no_run_to_run_variation_stops_at_the_largest_ratio(self) -> None:
        """With no error inside the whole plots the deviance falls without end; the search stops at its edge."""
        df = _unbalanced_split_plot()
        whole_plot_error = np.random.default_rng(5).normal(0, 2.0, df["plot"].max() + 1)[df["plot"]]
        df["y"] = 10 + 3 * df["A"] - 2 * df["B"] + whole_plot_error
        y, X = dmatrices("y ~ A + B", df, return_type="dataframe")
        fit = fit_reml(y.to_numpy().ravel(), X.to_numpy(), df["plot"].to_numpy())
        assert fit.sigma2_wp / fit.sigma2 == pytest.approx(1e8)
        np.testing.assert_allclose(fit.beta[2], -2.0, rtol=1e-6)


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


def _analyse(df: pd.DataFrame, **kwargs) -> dict:
    """Run ``analyze_experiment`` on the corrosion response, ``split_plot`` by default."""
    kwargs.setdefault("analysis_type", "split_plot")
    return analyze_experiment(df, response_column="Resistance", **kwargs)


def _tests_by_source(result: dict) -> dict[str, dict]:
    """Return the split-plot F-tests keyed by term."""
    return {row["source"]: row for row in result["split_plot"]["tests"]}


class TestAnalyzeExperiment:
    """``analyze_experiment(..., analysis_type="split_plot")`` on the corrosion data."""

    @pytest.fixture
    def corrosion(self) -> pd.DataFrame:
        """Return the corrosion data with the heats as the default ``WholePlot`` column."""
        return _corrosion().drop(columns="Position").rename(columns={"Heat": "WholePlot"})

    def test_reproduces_the_textbook_split_plot_anova(self, corrosion: pd.DataFrame) -> None:
        """Temperature is tested against the heats and is not significant; coating and the interaction are."""
        result = _analyse(corrosion)["split_plot"]
        tests = _tests_by_source({"split_plot": result})
        assert list(tests) == ["Temperature", "Coating", "Temperature:Coating"]
        assert [tests[s]["stratum"] for s in tests] == ["whole_plot", "subplot", "subplot"]
        assert [tests[s]["df"] for s in tests] == [2, 3, 6]
        np.testing.assert_allclose([tests[s]["df_denominator"] for s in tests], [3, 9, 9], rtol=1e-10)
        assert round(tests["Temperature"]["F"], 2) == 2.75
        assert tests["Temperature"]["p_value"] == pytest.approx(0.2093, abs=1e-4)
        assert result["significant_terms"] == ["Coating", "Temperature:Coating"]
        assert result["not_significant_terms"] == ["Temperature"]
        assert result["error_df"] == {"whole_plot": 3, "subplot": 9}
        assert result["n_whole_plots"] == 6
        assert result["variance_components"]["eta"] == pytest.approx(1172.1666666666667 / 124.54166666666667)

    def test_ordinary_least_squares_reaches_the_opposite_conclusions(self, corrosion: pd.DataFrame) -> None:
        """OLS finds temperature significant and coating not: the false positive and negative split plots risk."""
        with pytest.warns(UserWarning, match="ordinary least squares"):
            ols = _analyse(corrosion, analysis_type="anova")
        ols_p = {row["source"]: row["p_value"] for row in ols["anova_table"] if row["p_value"] is not None}
        reml_p = {source: row["p_value"] for source, row in _tests_by_source(_analyse(corrosion)).items()}
        assert ols_p["Temperature"] < 0.05 < reml_p["Temperature"]
        assert reml_p["Coating"] < 0.05 < ols_p["Coating"]

    def test_coefficients_carry_their_stratum_and_df(self, corrosion: pd.DataFrame) -> None:
        """Each coefficient is labelled in sum coding, with its own df: 3 for temperature, 9 for coating."""
        coefficients = {row["term"]: row for row in _analyse(corrosion)["split_plot"]["coefficients"]}
        assert coefficients["Temperature[S.360]"]["stratum"] == "whole_plot"
        assert coefficients["Temperature[S.360]"]["df"] == pytest.approx(3.0)
        assert coefficients["Coating[S.C1]"]["stratum"] == "subplot"
        assert coefficients["Coating[S.C1]"]["df"] == pytest.approx(9.0)
        row = coefficients["Coating[S.C1]"]
        assert row["ci_low"] < row["coefficient"] < row["ci_high"]
        # The intercept is the grand mean in sum coding.
        assert coefficients["Intercept"]["coefficient"] == pytest.approx(corrosion["Resistance"].mean())

    def test_a_named_column_gives_the_same_analysis(self, corrosion: pd.DataFrame) -> None:
        """``whole_plot="Heat"`` analyses exactly as the default ``WholePlot`` column does, and is not a factor."""
        named = _analyse(corrosion.rename(columns={"WholePlot": "Heat"}), whole_plot="Heat")
        default = _analyse(corrosion)
        assert named["factor_names"] == default["factor_names"] == ["Temperature", "Coating"]
        assert named["split_plot"]["tests"] == default["split_plot"]["tests"]
        assert named["split_plot"]["whole_plot_column"] == "Heat"

    def test_the_run_order_does_not_matter(self, corrosion: pd.DataFrame) -> None:
        """Shuffling the runs changes nothing: the whole plots come from the labels, not the row order."""
        shuffled = corrosion.sample(frac=1.0, random_state=1)
        expected = _tests_by_source(_analyse(corrosion))
        for source, row in _tests_by_source(_analyse(shuffled)).items():
            assert row["F"] == pytest.approx(expected[source]["F"], rel=1e-9)

    def test_a_repeated_index_is_read_by_position(self, corrosion: pd.DataFrame) -> None:
        """Two frames stacked without a new index: the whole plots still line up with their runs."""
        stacked = corrosion.set_axis(list(range(12)) * 2)
        expected = _tests_by_source(_analyse(corrosion))
        for source, row in _tests_by_source(_analyse(stacked)).items():
            assert row["F"] == pytest.approx(expected[source]["F"], rel=1e-9)

    def test_factors_named_like_patsy_functions(self, corrosion: pd.DataFrame) -> None:
        """Factors called ``C`` and ``Sum`` are sum-coded like any other; the analysis is unchanged."""
        renamed = corrosion.rename(columns={"Temperature": "Sum", "Coating": "C"})
        tests = _tests_by_source(_analyse(renamed))
        assert list(tests) == ["Sum", "C", "Sum:C"]
        assert tests["Sum"]["F"] == pytest.approx(_tests_by_source(_analyse(corrosion))["Temperature"]["F"])

    def test_blocks_are_tested_together_in_the_whole_plot_stratum(self, corrosion: pd.DataFrame) -> None:
        """Heats 1-3 and 4-6 as two blocks: one ``Block`` test, against the whole-plot error, not a factor term."""
        blocked = corrosion.assign(Block=np.where(corrosion["WholePlot"] <= 3, 1, 2))
        result = _analyse(blocked)["split_plot"]
        tests = _tests_by_source({"split_plot": result})
        assert tests["Block"]["df"] == 1
        assert tests["Block"]["stratum"] == "whole_plot"
        assert "Block" not in result["significant_terms"] + result["not_significant_terms"]
        assert result["error_df"]["whole_plot"] == 2

    def test_a_transform_is_analysed_on_its_scale(self, corrosion: pd.DataFrame) -> None:
        """``transform="log"`` gives the analysis of the logged response."""
        logged = corrosion.assign(Resistance=np.log(corrosion["Resistance"]))
        expected = _tests_by_source(_analyse(logged))
        for source, row in _tests_by_source(_analyse(corrosion, transform="log")).items():
            assert row["F"] == pytest.approx(expected[source]["F"], rel=1e-9)

    def test_a_run_with_no_whole_plot_label_is_left_out(self, corrosion: pd.DataFrame) -> None:
        """A missing label drops that run, with the usual warning."""
        corrosion = corrosion.astype({"WholePlot": float})
        corrosion.loc[1, "WholePlot"] = np.nan
        with pytest.warns(UserWarning, match="1 run"):
            result = _analyse(corrosion)
        assert result["model_summary"]["n_obs"] == 23

    def test_a_zero_whole_plot_variance_is_noted(self) -> None:
        """When REML puts the whole-plot variance at zero, a note says the estimates are OLS's."""
        df = _unbalanced_split_plot().rename(columns={"plot": "WholePlot"})
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 1.0, len(df))
        df["y"] = 10 + 3 * df["A"] - 2 * df["B"] + noise - pd.Series(noise).groupby(df["WholePlot"]).transform("mean")
        result = analyze_experiment(df, response_column="y", analysis_type="split_plot")
        assert result["split_plot"]["variance_components"]["whole_plot"] == 0.0
        assert "estimated as zero" in result["split_plot_note"]


class TestWholePlotArguments:
    """Where the whole plots come from, and the warning when they are ignored."""

    def test_split_plot_without_whole_plots_is_refused(self) -> None:
        """No ``WholePlot`` column and no ``whole_plot`` argument: say how to give one."""
        with pytest.raises(ValueError, match="needs each run's whole plot"):
            _analyse(_corrosion().drop(columns="Position"))

    def test_an_unknown_column_is_refused(self) -> None:
        """``whole_plot`` must name a column of the data."""
        with pytest.raises(ValueError, match="not a column"):
            _analyse(_corrosion(), whole_plot="Furnace")

    def test_ordinary_least_squares_on_whole_plots_warns(self) -> None:
        """Any analysis without ``split_plot`` warns when the runs are grouped into whole plots."""
        with pytest.warns(UserWarning, match="Add 'split_plot' to analysis_type"):
            _analyse(_corrosion().drop(columns="Position"), whole_plot="Heat", analysis_type="coefficients")

    def test_no_warning_with_split_plot_or_without_whole_plots(self) -> None:
        """Neither the split-plot analysis nor a design without whole plots warns about them."""
        corrosion = _corrosion().drop(columns="Position")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _analyse(corrosion, whole_plot="Heat", analysis_type=["split_plot", "anova"])
            _analyse(corrosion.drop(columns="Heat"), analysis_type="anova")


_FACTORS = [
    Factor(name="T", low=360, high=380),
    Factor(name="P", low=1, high=3),
    Factor(name="R", low=0, high=1),
]


def _split_plot_design(**kwargs) -> DesignResult:
    """Return a D-optimal split-plot design with ``T`` hard to change (pyoptex)."""
    pytest.importorskip("pyoptex")
    kwargs.setdefault("budget", 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_design(
            _FACTORS, design_type="d_optimal", hard_to_change=["T"], model_type="interactions", **kwargs
        )


def _assert_hard_to_change_within_whole_plots(design: pd.DataFrame) -> None:
    """``T`` takes one setting inside every whole plot."""
    assert (design.groupby("WholePlot")["T"].agg(["min", "max"]).diff(axis=1)["max"] == 0).all()


class TestSplitPlotDesigns:
    """``generate_design(hard_to_change=...)`` labels each run's whole plot in a ``WholePlot`` column."""

    def test_the_column_is_the_metadata_from_one(self) -> None:
        """``WholePlot`` is ``metadata["whole_plot"] + 1``, and the hard-to-change factor is constant within it."""
        result = _split_plot_design()
        assert (result.design["WholePlot"] - 1).tolist() == result.metadata["whole_plot"]
        assert result.design_actual["WholePlot"].tolist() == result.design["WholePlot"].tolist()
        assert result.metadata["n_whole_plots"] == result.design["WholePlot"].nunique()
        _assert_hard_to_change_within_whole_plots(result.design)

    def test_each_replicate_has_its_own_whole_plots(self) -> None:
        """A replicate is run again from scratch: twice the whole plots, and the metadata covers every run."""
        single, double = _split_plot_design(budget=8), _split_plot_design(budget=8, n_replicates=2)
        n_plots = single.metadata["n_whole_plots"]
        assert double.metadata["n_whole_plots"] == 2 * n_plots
        assert double.design["WholePlot"].tolist() == [
            *single.design["WholePlot"].tolist(),
            *(single.design["WholePlot"] + n_plots).tolist(),
        ]
        assert len(double.metadata["whole_plot"]) == len(double.design)
        _assert_hard_to_change_within_whole_plots(double.design)

    def test_other_designs_have_no_whole_plots(self) -> None:
        """Only a split-plot design gets the column."""
        result = generate_design(_FACTORS, design_type="full_factorial")
        assert "WholePlot" not in result.design.columns

    def test_a_factor_cannot_be_named_wholeplot(self) -> None:
        """``WholePlot`` is reserved, like ``RunOrder`` and ``Block``."""
        with pytest.raises(ValueError, match="reserved"):
            generate_design([*_FACTORS[:2], Factor(name="WholePlot", low=0, high=1)], design_type="full_factorial")

    def test_the_design_is_analysed_without_naming_its_whole_plots(self) -> None:
        """Responses joined onto the design go straight into ``split_plot``; ``T`` is in the whole-plot stratum."""
        result = _split_plot_design()
        design = result.design
        rng = np.random.default_rng(0)
        whole_plot_error = rng.normal(0, 3.0, design["WholePlot"].max() + 1)[design["WholePlot"]]
        y = 50 + 4 * design["T"] + 2 * design["P"] - design["R"] + whole_plot_error + rng.normal(0, 1.0, len(design))
        analysis = analyze_experiment(design.assign(y=y), response_column="y", analysis_type="split_plot")
        assert analysis["factor_names"] == ["T", "P", "R"]
        strata = {row["source"]: row["stratum"] for row in analysis["split_plot"]["tests"]}
        assert strata == {s: "whole_plot" if s == "T" else "subplot" for s in ["T", "P", "R", "T:P", "T:R", "P:R"]}
        assert analysis["split_plot"]["n_whole_plots"] == result.metadata["n_whole_plots"]

    def test_design_metrics_ignore_the_column(self) -> None:
        """``evaluate_design`` and ``moment_aberration`` do not treat ``WholePlot`` as a factor."""
        result = _split_plot_design()
        without = result.design.drop(columns="WholePlot")
        assert evaluate_design(result.design, model="interactions") == evaluate_design(without, model="interactions")
        coded = result.design.drop(columns="RunOrder")
        assert moment_aberration(coded).pattern == moment_aberration(coded.drop(columns="WholePlot")).pattern
