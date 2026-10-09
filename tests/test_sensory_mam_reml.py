"""The random-effects Mixed Assessor Model (#431) and its REML engine.

``TestAgainstSensMixed`` checks ``mixed_assessor_model_reml`` against the R package
SensMixed on the TVbo panel; ``tests/fixtures/sensmixed_mam/README.md`` describes the four
cases and how the reference values were made. The other tests are synthetic.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from process_improve.regression._variance_components import (
    _MAX_ITERATIONS,
    VarianceComponentFit,
    _minimise,
    fit_variance_components,
    independent_columns,
    type1_hypotheses,
)
from process_improve.sensory import MAMRemlResult, mixed_assessor_model, mixed_assessor_model_reml

_FIXTURE = Path(__file__).parent / "fixtures" / "sensmixed_mam"
_FACTORS = ["Assessor", "TVset", "Repeat", "Picture"]

#: SensMixed's names for the terms this package labels differently.
_RENAME = {"Assessor": "panelist", "Product": "product", "Scaling": "scaling"}

#: Attributes where SensMixed and this package disagree on which terms are at a zero
#: variance (see the fixture README); elsewhere the random-term selection matches exactly.
_BOUNDARY = {
    "one_way": {"Lightlevel", "Flickeringstationary", "Flickeringmovement"},
    "factorial": {"Flickeringstationary", "Dimglasseffect"},
    "unbalanced": {"Flickeringstationary", "Flickeringmovement", "Cutting"},
    "no_replicate": set(),
}

#: The cases, as (product factors, replicate column); the slow ones take over 2 s each.
_CASES = {
    "one_way": (("product",), "Repeat"),
    "factorial": (("TVset", "Picture"), "Repeat"),
    "unbalanced": (("TVset", "Picture"), "Repeat"),
    "no_replicate": (("TVset", "Picture"), None),
}
_CASE_PARAMS = [pytest.param(case, marks=[] if case == "no_replicate" else [pytest.mark.slow]) for case in _CASES]


def _name(term: str) -> str:
    """Translate a SensMixed term label into this package's."""
    return ":".join(_RENAME.get(part, part) for part in term.split(":"))


def _reference() -> dict:
    return json.loads((_FIXTURE / "reference.json").read_text())


def _tvbo_panel(case: str) -> pd.DataFrame:
    """Return the TVbo panel in long format, as the case uses it."""
    wide = pd.read_csv(_FIXTURE / "tvbo.csv", dtype=dict.fromkeys(_FACTORS, str))
    attributes = list(wide.columns[len(_FACTORS) :])
    if case == "unbalanced":
        wide = wide.drop(index=[row - 1 for row in _reference()["dropped_rows"]])
    elif case == "no_replicate":
        wide = wide.groupby(["Assessor", "TVset", "Picture"], as_index=False)[attributes].mean()
    ids = [column for column in _FACTORS if column in wide.columns]
    panel = wide.melt(id_vars=ids, value_vars=attributes, var_name="attribute", value_name="score")
    panel = panel.rename(columns={"Assessor": "panelist_id"})
    panel["product"] = panel["TVset"] + "_" + panel["Picture"]
    return panel


@cache
def _fit(case: str) -> MAMRemlResult:
    """Fit a case once per test session."""
    factors, replication = _CASES[case]
    return mixed_assessor_model_reml(_tvbo_panel(case), product_factors=factors, replication=replication)


@pytest.fixture(scope="module")
def reference() -> dict:
    """Return the R results written by the fixture's ``reference.R``."""
    return _reference()


@pytest.mark.dataset
class TestAgainstSensMixed:
    """Agreement with SensMixed 2.1 (lme4 1.1-15, lmerTest 2.0-36) on the TVbo panel."""

    @pytest.mark.parametrize("case", _CASE_PARAMS)
    def test_f_tests_are_sensmixeds(self, case: str, reference: dict) -> None:
        """Every Type I test of every attribute: F, sums of squares, df and p.

        The df are SensMixed's to 2e-4: lmerTest 2.0 differentiates numerically, and a test
        on several df magnifies that error (``test_df_are_lmertest3s`` is the tight check).
        """
        anova = _fit(case).anova.set_index(["attribute", "term"])
        for attribute, expected in reference["cases"][case].items():
            for term, row in expected["anova"].items():
                ours = anova.loc[(attribute, _name(term))]
                assert ours["num_df"] == row["num_df"]
                assert ours["f_value"] == pytest.approx(row["f_value"], rel=1e-5)
                assert ours["sum_sq"] == pytest.approx(row["sum_sq"], rel=1e-5)
                assert ours["den_df"] == pytest.approx(row["den_df"], rel=5e-4)
                assert ours["p_value"] == pytest.approx(row["p_value"], rel=5e-3, abs=1e-12)

    @pytest.mark.parametrize("case", _CASE_PARAMS)
    def test_variance_components_are_sensmixeds(self, case: str, reference: dict) -> None:
        """The final model's variances; a term only one side keeps is at zero."""
        components = _fit(case).variance_components
        for attribute, expected in reference["cases"][case].items():
            ours = components[components["attribute"] == attribute].set_index("term")["variance"]
            theirs = {_name(term): value for term, value in expected["variance_components"].items()}
            for term in set(ours.index) | set(theirs):
                assert ours.get(term, 0.0) == pytest.approx(theirs.get(term, 0.0), rel=1e-4, abs=1e-6), (
                    attribute,
                    term,
                )

    @pytest.mark.parametrize("case", _CASE_PARAMS)
    def test_scaling_coefficients_are_sensmixeds(self, case: str, reference: dict) -> None:
        """Each assessor's ``beta``, from SensMixed's ``Assessor:x`` coefficients."""
        scaling = _fit(case).scaling.set_index(["attribute", "panelist_id"])["beta"]
        for attribute, expected in reference["cases"][case].items():
            for panelist, beta in expected["beta"].items():
                assert scaling[(attribute, panelist)] == pytest.approx(beta, abs=1e-6)

    @pytest.mark.parametrize("case", _CASE_PARAMS)
    def test_random_selection_is_sensmixeds(self, case: str, reference: dict) -> None:
        """Statuses, elimination order and likelihood-ratio statistics, away from the boundary."""
        table = _fit(case).random_effects
        compared = 0
        for attribute, expected in reference["cases"][case].items():
            if attribute in _BOUNDARY[case]:
                continue
            ours = table[table["attribute"] == attribute].set_index("term")
            assert set(ours.index[ours["status"] == "zero_variance"]) == {_name(t) for t in expected["zero_variance"]}
            for term, row in expected["random"].items():
                mine = ours.loc[_name(term)]
                assert (0 if pd.isna(mine["step"]) else mine["step"]) == row["step"]
                assert mine["status"] == ("kept" if row["step"] == 0 else "eliminated")
                assert mine["chi_sq"] == pytest.approx(row["chi_sq"], abs=1e-6)
                assert mine["p_value"] == pytest.approx(row["p_value"], rel=1e-4, abs=1e-9)
                compared += 1
        assert compared >= len(reference["cases"][case]) - len(_BOUNDARY[case])  # not vacuous

    @pytest.mark.parametrize("case", _CASE_PARAMS)
    def test_boundary_differences_are_zero_variances(self, case: str, reference: dict) -> None:
        """Where the selections differ, each disputed term is at zero on both sides.

        lme4 stops near the boundary, not on it, so SensMixed's 1e-7 cut-off sorts boundary
        terms by where its search stopped. A disputed term is dropped for zero variance,
        eliminated with a zero statistic, or kept at zero variance, on either side.
        """
        table = _fit(case).random_effects
        components = _fit(case).variance_components
        for attribute in _BOUNDARY[case]:
            expected = reference["cases"][case][attribute]
            ours = table[table["attribute"] == attribute].set_index("term")
            our_zero = set(ours.index[ours["status"] == "zero_variance"])
            their_zero = {_name(term) for term in expected["zero_variance"]}
            assert our_zero != their_zero  # the attribute is listed for a reason
            our_final = components[components["attribute"] == attribute].set_index("term")["variance"]
            their_random = {_name(term): row for term, row in expected["random"].items()}
            their_final = {_name(term): value for term, value in expected["variance_components"].items()}
            for term in our_zero ^ their_zero:
                assert term in our_zero or ours.loc[term, "chi_sq"] < 1e-6 or our_final.get(term, 1.0) < 1e-8
                assert term in their_zero or their_random[term]["chi_sq"] < 1e-6 or their_final.get(term, 1.0) < 1e-8

    @pytest.mark.slow
    def test_df_are_lmertest3s(self) -> None:
        """On the unbalanced panel, lmerTest 3.1's df (accurate derivatives) equal the exact ones."""
        lmertest3 = json.loads((_FIXTURE / "reference_lmertest3.json").read_text())["unbalanced"]
        anova = _fit("unbalanced").anova.set_index(["attribute", "term"])
        for attribute, tests in lmertest3.items():
            for term, row in tests.items():
                ours = anova.loc[(attribute, _name(term))]
                assert ours["den_df"] == pytest.approx(row["den_df"], rel=1e-5)
                assert ours["f_value"] == pytest.approx(row["f_value"], rel=1e-5)

    def test_closed_form_beta_is_sensmixeds(self, reference: dict) -> None:
        """The issue's cross-check: the closed-form ``beta`` of ``mixed_assessor_model`` equals R's.

        In a balanced one-way panel the random-effects scaling coefficients reduce to the
        closed form's per-assessor regressions on the consensus.
        """
        scaling = mixed_assessor_model(_tvbo_panel("one_way")).scaling.set_index(["attribute", "panelist_id"])["beta"]
        for attribute, expected in reference["cases"]["one_way"].items():
            for panelist, beta in expected["beta"].items():
                assert scaling[(attribute, panelist)] == pytest.approx(beta, abs=1e-9)


# ---------------------------------------------------------------------------
# The REML engine
# ---------------------------------------------------------------------------


def _one_way_random(*, seed: int = 0, groups: int = 6, per_group: int = 4, sd_group: float = 1.0):
    """Balanced one-way random-effects data: ``y = 3 + u_group + e``."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(groups), per_group)
    y = 3.0 + rng.normal(0, sd_group, groups)[codes] + rng.normal(0, 1.0, groups * per_group)
    return y, codes


def test_balanced_one_way_reml_is_the_anova_estimate() -> None:
    """With a positive estimate, REML equals the ANOVA estimators exactly."""
    y, codes = _one_way_random()
    groups, per_group = codes.max() + 1, np.bincount(codes)[0]
    means = np.bincount(codes, y) / per_group
    ms_between = per_group * np.sum((means - y.mean()) ** 2) / (groups - 1)
    ms_within = np.sum((y - means[codes]) ** 2) / (len(y) - groups)
    fit = fit_variance_components(y, np.ones((len(y), 1)), [codes])
    assert fit.sigma2 == pytest.approx(ms_within, rel=1e-10)
    assert fit.variances[0] == pytest.approx((ms_between - ms_within) / per_group, rel=1e-10)
    # The mean is tested on the between-group df, exactly.
    assert fit.satterthwaite_df(np.array([1.0])) == pytest.approx(groups - 1, rel=1e-8)


def test_reml_criterion_is_minimal() -> None:
    """Any other variance ratio gives a larger REML criterion."""
    y, codes = _one_way_random(seed=3)
    X = np.ones((len(y), 1))
    best = fit_variance_components(y, X, [codes])
    ratio = best.variances[0] / best.sigma2
    for factor in (0.9, 1.1):
        other = fit_variance_components(y, X, [codes], start=np.array([ratio * factor]))
        assert other.reml_criterion == pytest.approx(best.reml_criterion, abs=1e-9)


def test_zero_variance_gives_the_ordinary_least_squares_fit() -> None:
    """When groups explain less than nothing, the estimate is exactly zero and the fit is OLS."""
    rng = np.random.default_rng(1)
    codes = np.repeat(np.arange(6), 4)
    y = rng.normal(0, 1, 24)
    y -= (np.bincount(codes, y) / 4)[codes] * 0.9  # shrink the group means: MS_between < MS_within
    X = np.column_stack([np.ones(24), np.linspace(-1, 1, 24)])
    fit = fit_variance_components(y, X, [codes])
    ols_beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert fit.variances[0] == 0.0
    np.testing.assert_allclose(fit.beta, ols_beta, rtol=1e-12)
    assert fit.sigma2 == pytest.approx(np.sum((y - X @ ols_beta) ** 2) / 22, rel=1e-12)


def test_no_random_terms_reproduces_sequential_anova() -> None:
    """Without random terms, the Type I tests are ordinary least squares' sequential ANOVA."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"a": np.repeat(["p", "q", "r"], 8), "x": rng.normal(size=24)})
    frame["y"] = (frame["a"] == "q") * 1.5 + 0.8 * frame["x"] + rng.normal(size=24)
    X = np.column_stack([np.ones(24), frame["a"] == "p", frame["a"] == "q", frame["x"]]).astype(float)
    assign = np.array([-1, 0, 0, 1])
    fit = fit_variance_components(frame["y"].to_numpy(), X, [])
    table = anova_lm(ols("y ~ C(a) + x", frame).fit(), typ=1)
    for term, (label, num_df) in enumerate([("C(a)", 2), ("x", 1)]):
        f_value, den_df = fit.f_test(type1_hypotheses(X, assign)[term])
        assert f_value == pytest.approx(table.loc[label, "F"], rel=1e-10)
        assert den_df == pytest.approx(20, rel=1e-8)
        assert num_df == type1_hypotheses(X, assign)[term].shape[0]


def _two_direction_fit(nus: tuple[float, float]) -> VarianceComponentFit:
    """Return a fit whose two estimates are independent, with Satterthwaite df ``nus``.

    With unit variances and one variance parameter of unit variance, a direction's df is
    ``2 / d**2``, ``d`` the derivative of its variance.
    """
    derivatives = np.sqrt(2.0 / np.asarray(nus))
    return VarianceComponentFit(
        beta=np.array([1.0, 2.0]),
        cov_beta=np.eye(2),
        variances=np.array([1.0]),
        sigma2=1.0,
        reml_criterion=0.0,
        vc_cov=np.eye(1),
        dcov=np.diag(derivatives)[None, :, :],
    )


def test_multi_df_test_combines_directions_as_fai_and_cornelius() -> None:
    """Unequal directions combine through E = sum(nu / (nu - 2)), df = 2E / (E - q)."""
    f_value, den_df = _two_direction_fit((10.0, 30.0)).f_test(np.eye(2))
    expected = 10 / 8 + 30 / 28
    assert f_value == pytest.approx((1.0 + 4.0) / 2)
    assert den_df == pytest.approx(2 * expected / (expected - 2))


def test_multi_df_test_with_a_direction_below_two_df() -> None:
    """A direction with 2 df or fewer has no finite F mean to match; the df floor at 2."""
    assert _two_direction_fit((1.5, 30.0)).f_test(np.eye(2))[1] == 2.0


class _ConstantSlope:
    """A criterion with score -1 and unit curvature everywhere, whose trial values are fixed."""

    def __init__(self, trial_criterion: float) -> None:
        self.trial_criterion = trial_criterion

    def evaluate(self, _psi: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(criterion=0.0)

    def gradient(self, _state: SimpleNamespace) -> np.ndarray:
        return np.array([-1.0])

    def hessian(self, _state: SimpleNamespace) -> np.ndarray:
        return np.array([[1.0]])

    def criterion(self, _psi: np.ndarray) -> float:
        return self.trial_criterion


@pytest.mark.parametrize(("trial_criterion", "steps"), [(1.0, 0), (-1e9, _MAX_ITERATIONS)])
def test_search_always_ends(trial_criterion: float, steps: int) -> None:
    """With no downhill step the search stays put; with endless ones it stops at its cap."""
    assert _minimise(_ConstantSlope(trial_criterion), np.array([0.5]))[0] == 0.5 + steps


def test_independent_columns_drops_the_later_aliased_column() -> None:
    """A column that is a combination of earlier ones goes; the earlier ones stay."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=10), rng.normal(size=10)
    keep = independent_columns(np.column_stack([np.ones(10), a, b, a + 2 * b, np.zeros(10)]))
    np.testing.assert_array_equal(keep, [True, True, True, False, False])


def test_fit_needs_residual_degrees_of_freedom() -> None:
    """As many fixed columns as observations leaves nothing to estimate error from."""
    with pytest.raises(ValueError, match="none are left for error"):
        fit_variance_components(np.arange(3.0), np.eye(3), [])


# ---------------------------------------------------------------------------
# mixed_assessor_model_reml
# ---------------------------------------------------------------------------


def _panel(*, seed: int = 0, replicates: int = 2, products: int = 6, panelists: int = 6) -> pd.DataFrame:
    """Return a balanced panel with scaling differences, panelist offsets and replicate noise."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, 2, products)
    rows = []
    for panelist in range(panelists):
        slope, offset = 0.5 + panelist * 0.2, rng.normal(0, 1)
        for product in range(products):
            disagreement = rng.normal(0, 0.4)
            rows.extend(
                {
                    "panelist_id": f"J{panelist}",
                    "product": f"p{product}",
                    "attribute": "sweet",
                    "replicate": replicate,
                    "session": replicate,
                    "score": 5 + offset + slope * effect[product] + disagreement + rng.normal(0, 0.3),
                }
                for replicate in range(1, replicates + 1)
            )
    return pd.DataFrame(rows)


def test_without_replicates_the_product_test_is_the_closed_forms() -> None:
    """On a balanced one-way panel the REML product F equals the closed-form MAM F."""
    panel = _panel(replicates=1)
    reml = mixed_assessor_model_reml(panel, replication=None)
    closed = mixed_assessor_model(panel)
    product = reml.anova.set_index("term").loc["product"]
    assert product["f_value"] == pytest.approx(closed.ftests.loc[0, "f_product_mam"], rel=1e-9)
    assert product["den_df"] == pytest.approx(closed.ftests.loc[0, "df_disagreement"], rel=1e-8)
    np.testing.assert_allclose(
        reml.scaling.set_index("panelist_id")["beta"], closed.scaling.set_index("panelist_id")["beta"], atol=1e-9
    )


def test_result_tables() -> None:
    """The four tables have the documented columns, and every candidate term is accounted for."""
    result = mixed_assessor_model_reml(_panel())
    assert list(result.anova.columns) == [
        "attribute",
        "term",
        "sum_sq",
        "mean_sq",
        "num_df",
        "den_df",
        "f_value",
        "p_value",
    ]
    assert list(result.anova["term"]) == ["product", "scaling"]
    assert list(result.random_effects.columns) == ["attribute", "term", "chi_sq", "chi_df", "p_value", "status", "step"]
    assert result.random_effects["step"].dtype == "Int64"
    assert set(result.random_effects["term"]) == {
        "product:panelist",
        "product:replicate",
        "panelist",
        "replicate",
        "panelist:replicate",
    }
    assert set(result.random_effects["status"]) <= {"kept", "eliminated", "zero_variance"}
    assert list(result.variance_components.columns) == ["attribute", "term", "variance", "std_dev"]
    assert result.variance_components["term"].iloc[-1] == "Residual"
    assert result.scaling["beta"].mean() == pytest.approx(1.0)


def test_scaling_is_detected() -> None:
    """Panelists with slopes from 0.5 to 1.5 make the scaling term significant, in order."""
    result = mixed_assessor_model_reml(_panel(products=8, panelists=8))
    assert result.anova.set_index("term").loc["scaling", "p_value"] < 0.01
    beta = result.scaling.set_index("panelist_id")["beta"]
    assert beta["J0"] < beta["J4"] < beta["J7"]


def test_panelist_terms_are_always_kept() -> None:
    """The panelist and product-by-panelist terms are tested but never eliminated."""
    table = mixed_assessor_model_reml(_panel(seed=5), alpha_random=0.0).random_effects.set_index("term")
    assert table.loc["panelist", "status"] == "kept"
    assert table.loc["product:panelist", "status"] == "kept"
    assert (table["status"] != "kept").sum() >= 1  # with alpha 0, everything else goes


def test_alpha_one_eliminates_nothing() -> None:
    """``alpha_random=1`` keeps every term with a nonzero variance."""
    table = mixed_assessor_model_reml(_panel(), alpha_random=1.0).random_effects
    assert "eliminated" not in set(table["status"])
    assert table["step"].isna().all()


def test_single_level_replicate_is_ignored() -> None:
    """A replicate column with one level adds no replicate terms."""
    table = mixed_assessor_model_reml(_panel(replicates=1)).random_effects
    assert not table["term"].str.contains("replicate").any()


def test_factorial_products_are_tested_term_by_term() -> None:
    """Products from a factorial design get one test per main effect and interaction."""
    panel = _panel(products=6)
    levels = {f"p{i}": (f"a{i % 2}", f"b{i % 3}") for i in range(6)}
    panel["A"], panel["B"] = zip(*panel["product"].map(levels), strict=True)
    result = mixed_assessor_model_reml(panel, product_factors=("A", "B"))
    assert list(result.anova["term"]) == ["A", "B", "A:B", "scaling"]
    assert list(result.anova["num_df"]) == [1, 2, 2, 5]
    assert "A:B:panelist" in set(result.random_effects["term"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.drop(columns="replicate"), "replication=None"),
        (lambda p: p.drop(columns="panelist_id"), "missing"),
        (lambda p: p.iloc[:0], "no rows"),
        (lambda p: p[p["product"].isin(["p0", "p1"])], "at least 3 products"),
    ],
)
def test_unusable_panels_raise(change, message: str) -> None:
    """Missing columns, an empty panel and too few products each get a clear message."""
    with pytest.raises(ValueError, match=message):
        mixed_assessor_model_reml(change(_panel()))
