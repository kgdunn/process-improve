"""Aliased terms in the model summary, the effects, Lenth's method and the Pareto chart (#16).

The design used throughout is the 2^(4-1) half fraction with D = ABC, whose
two-factor interactions are aliased in pairs: the A:B column *is* the C:D column,
likewise A:C with B:D and A:D with B:C. The response has a main effect of A and an
A:B interaction of known size, so every output can be checked against the truth.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from process_improve.experiments._analyses.aliasing import alias_chains, estimable_effects
from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.models import lm, summary
from process_improve.experiments.structures import c, gather
from process_improve.experiments.visualization.plots.significance import ParetoPlot

A = np.array([-1, 1, -1, 1, -1, 1, -1, 1.0])
B = np.array([-1, -1, 1, 1, -1, -1, 1, 1.0])
C = np.array([-1, -1, -1, -1, 1, 1, 1, 1.0])
#: Effects (twice the coded coefficients) of the response's two real terms.
TRUE_A, TRUE_AB = 10.0, 6.0


def _half_fraction(generator_sign: float = 1.0) -> pd.DataFrame:
    """Return the half fraction with D = +ABC (or -ABC), and y = 10 + 5 A + 3 AB + small noise."""
    noise = np.random.default_rng(0).normal(scale=0.1, size=8)
    y = 10 + TRUE_A / 2 * A + TRUE_AB / 2 * A * B + noise
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": generator_sign * A * B * C, "y": y})


def _analyse(frame: pd.DataFrame, model: str = "interactions", analyses: tuple = ("effects", "lenth_method")) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the rank-deficiency warning is the point of these designs
        return analyze_experiment(frame, response_column="y", model=model, analysis_type=list(analyses))


class TestTheSummary:
    """The ``lm`` path: the aliasing pattern must list every term's aliases, interactions included."""

    @pytest.fixture
    def model(self) -> object:
        frame = _half_fraction()
        expt = gather(**{name: c(*frame[name]) for name in "ABCD"}, y=c(*frame["y"]), title="half fraction")
        return lm("y ~ A*B*C*D", expt)

    def test_interactions_show_their_aliases(self, model: object) -> None:
        aliases = model.get_aliases(3, drop_intercept=False)  # type: ignore[attr-defined]
        assert "A:B + C:D" in aliases
        assert "A:C + B:D" in aliases
        assert "B:C + A:D" in aliases

    def test_the_printed_summary_carries_them(self, model: object) -> None:
        text = str(summary(model, show=False, aliasing_up_to_level=2))
        assert "A:B + C:D" in text

    def test_looking_up_aliases_does_not_change_the_model(self, model: object) -> None:
        """The alias map is a defaultdict; indexing it with a missing key used to add that key."""
        before = dict(model.aliasing)  # type: ignore[attr-defined]
        model.get_aliases(3, drop_intercept=False)  # type: ignore[attr-defined]
        assert dict(model.aliasing) == before  # type: ignore[attr-defined]


class TestTheEffects:
    """The ``analyze_experiment`` path, which feeds the Pareto chart."""

    def test_each_alias_chain_is_one_effect_of_its_full_size(self) -> None:
        """The pseudo-inverse fit gives A:B and C:D half the chain's effect each; report the chain."""
        results = _analyse(_half_fraction())
        effects = results["effects"]
        assert set(effects) == {"A", "B", "C", "D", "A:B + C:D", "A:C + B:D", "A:D + B:C"}
        assert effects["A:B + C:D"] == pytest.approx(TRUE_AB, abs=0.1)
        assert effects["A"] == pytest.approx(TRUE_A, abs=0.1)
        assert results["alias_chains"]["A:B + C:D"] == ["A:B", "C:D"]

    def test_an_anti_alias_is_named_with_its_sign(self) -> None:
        """With D = -ABC the C:D column is minus the A:B column, and the chain is A:B - C:D."""
        effects = _analyse(_half_fraction(generator_sign=-1.0))["effects"]
        assert effects["A:B - C:D"] == pytest.approx(TRUE_AB, abs=0.1)

    def test_a_term_aliased_with_the_mean_has_no_effect(self) -> None:
        """In the 2^(3-1) with C = AB, A:B:C is the constant column: it is the intercept."""
        frame = pd.DataFrame({"A": A[:4], "B": B[:4], "C": A[:4] * B[:4], "y": [41, 27, 35, 20.0]})
        results = _analyse(frame, model="A*B*C", analyses=("effects",))
        assert results["confounded_with_mean"] == ["A:B:C"]
        assert set(results["effects"]) == {"A + B:C", "B + A:C", "C + A:B"}
        assert results["effects"]["A + B:C"] == pytest.approx(-14.5)

    def test_without_aliasing_nothing_changes(self) -> None:
        """A replicated full factorial: effects and errors are exactly twice the fitted ones, and no new keys."""
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"A": np.tile(A, 2), "B": np.tile(B, 2), "C": np.tile(C, 2)})
        frame["y"] = 3 + 2 * frame["A"] - frame["B"] * frame["C"] + rng.normal(scale=0.2, size=16)
        results = _analyse(frame, analyses=("effects",))
        fit = smf.ols("y ~ (A + B + C)**2", data=frame).fit()
        expected = (2 * fit.params.drop("Intercept")).to_dict()
        assert results["effects"] == pytest.approx(expected)
        assert results["effect_std_errors"] == pytest.approx((2 * fit.bse.drop("Intercept")).to_dict())
        assert "alias_chains" not in results
        assert "confounded_with_mean" not in results

    def test_a_chains_standard_error_is_that_of_the_estimable_sum(self) -> None:
        """Replicate the half fraction so there is error to estimate; the chain's error is its column's."""
        frame = pd.concat([_half_fraction(), _half_fraction()], ignore_index=True)
        frame["y"] += np.random.default_rng(2).normal(scale=0.3, size=len(frame))
        results = _analyse(frame, analyses=("effects",))
        reduced = smf.ols("y ~ A + B + C + D + A:B + A:C + A:D", data=frame).fit()
        assert results["effects"]["A:B + C:D"] == pytest.approx(2 * reduced.params["A:B"])
        assert results["effect_std_errors"]["A:B + C:D"] == pytest.approx(2 * reduced.bse["A:B"])


class TestLenthAndPareto:
    def test_lenth_sees_each_chain_once(self) -> None:
        lenth = _analyse(_half_fraction())["lenth_method"]
        terms = [item["term"] for item in lenth["effects"]]
        assert len(terms) == 7
        assert "A:B + C:D" in terms

    def test_lenth_flags_the_real_chain(self) -> None:
        lenth = _analyse(_half_fraction())["lenth_method"]
        active = {item["term"] for item in lenth["effects"] if item["active_ME"]}
        assert active == {"A", "A:B + C:D"}

    def test_the_pareto_chart_draws_one_full_bar_per_chain(self) -> None:
        spec = ParetoPlot(analysis_results=_analyse(_half_fraction())).to_spec()
        bars = spec.panels[0].layers[0].data
        assert [bar["name"] for bar in bars][:2] == ["A", "A:B + C:D"]
        assert len(bars) == 7
        assert bars[1]["abs_effect"] == pytest.approx(TRUE_AB, abs=0.1)


class TestAliasChains:
    def test_every_column_lands_in_exactly_one_chain(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("y ~ (A + B + C + D)**2", data=_half_fraction()).fit()
        chains = alias_chains(fit)
        members = [term for chain in chains for term, _ in chain]
        assert sorted(members) == sorted(fit.model.exog_names)

    def test_the_shortest_term_leads_its_chain(self) -> None:
        """A 2^(3-1) with C = AB: the main effect leads, not the interaction aliased with it."""
        frame = pd.DataFrame({"A": A[:4], "B": B[:4], "C": A[:4] * B[:4], "y": [41, 27, 35, 20.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("y ~ A*B*C", data=frame).fit()
        leaders = [chain[0][0] for chain in alias_chains(fit)]
        assert leaders == ["Intercept", "A", "B", "C"]

    def test_the_chain_sum_does_not_depend_on_how_the_fit_split_it(self) -> None:
        """Only the sum is estimable: moving weight between aliased terms leaves it unchanged."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("y ~ (A + B + C + D)**2", data=_half_fraction()).fit()
        before = estimable_effects(fit).coefficients["A:B + C:D"]
        fit.params["A:B"] += 1.0
        fit.params["C:D"] -= 1.0
        assert estimable_effects(fit).coefficients["A:B + C:D"] == pytest.approx(before)
