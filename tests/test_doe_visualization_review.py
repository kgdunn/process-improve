"""Regression tests for review findings on the DoE design-quality and significance plots."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import Factor, analyze_experiment, evaluate_design, generate_design
from process_improve.experiments.visualization import visualize_doe
from process_improve.experiments.visualization.plots.registry import create_plot


def _factorial_rows(k: int) -> list[dict[str, float]]:
    return [dict(zip("ABCDEF", p, strict=False)) for p in itertools.product([-1.0, 1.0], repeat=k)]


def _ccd_rows() -> list[dict[str, float]]:
    r = generate_design([Factor(name=n, low=0, high=10) for n in "AB"], design_type="ccd", n_center_points=3)
    return r.design.to_dict("records")  # carries RunOrder


# ---------------------------------------------------------------------------
# Power curve
# ---------------------------------------------------------------------------


class TestPowerCurvePlot:
    def test_two_level_factorial_uses_the_estimable_model(self) -> None:
        """A 2^3 under its main-effects model: power at delta/sigma = 2 (coefficient 1) is about 0.57."""
        spec = create_plot("power_curve", design_data=_factorial_rows(3), model="main_effects").to_spec()
        assert spec.metadata["n_terms"] == 4
        assert spec.metadata["residual_df"] == 4
        layer = spec.panels[0].layers[0]
        point = min(layer.data, key=lambda p: abs(p["sn_ratio"] - 2.0))
        expected = 1 - stats.ncf.cdf(stats.f.ppf(0.95, 1, 4), 1, 4, 8 * (point["sn_ratio"] / 2) ** 2)
        assert point["power"] == pytest.approx(expected)
        assert point["power"] == pytest.approx(0.57, abs=0.02)

    def test_agrees_with_evaluate_design(self) -> None:
        rows = _ccd_rows()
        spec = create_plot("power_curve", design_data=rows, model="quadratic").to_spec()
        design = pd.DataFrame(rows).drop(columns=["RunOrder"])
        ev = evaluate_design(design, model="quadratic", metric="power", effect_size=0.5)["power"]
        for layer in spec.panels[0].layers:
            point = next(p for p in layer.data if p["sn_ratio"] == pytest.approx(1.0))
            for term in layer.name.split(", "):
                assert point["power"] == pytest.approx(ev[term])

    def test_run_order_is_not_a_factor(self) -> None:
        spec = create_plot("power_curve", design_data=_ccd_rows()).to_spec()
        assert spec.metadata["model"] == "quadratic"
        assert spec.metadata["n_terms"] == 6
        assert spec.metadata["residual_df"] == 5

    def test_saturated_model_raises(self) -> None:
        with pytest.raises(ValueError, match="residual degrees of freedom"):
            create_plot("power_curve", design_data=_factorial_rows(2), model="interactions").to_spec()

    def test_non_estimable_model_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot estimate"):
            create_plot("power_curve", design_data=_factorial_rows(3), model="quadratic").to_spec()


# ---------------------------------------------------------------------------
# FDS plot
# ---------------------------------------------------------------------------


class TestFDSPlot:
    def test_matches_evaluate_design(self) -> None:
        rows = _factorial_rows(3)
        spec = create_plot("fds_plot", design_data=rows).to_spec()
        assert spec.metadata["model"] == "interactions"
        expected = evaluate_design(pd.DataFrame(rows), model="interactions", metric="fds")["fds"]
        assert spec.metadata["max_spv"] == pytest.approx(expected["scaled_max_prediction_variance"])

    def test_run_order_is_not_a_factor(self) -> None:
        spec = create_plot("fds_plot", design_data=_ccd_rows()).to_spec()
        assert spec.metadata["n_factors"] == 2
        assert spec.metadata["max_spv"] < 20

    def test_non_estimable_model_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot estimate"):
            create_plot("fds_plot", design_data=_factorial_rows(3), model="quadratic").to_spec()

    def test_random_state_is_the_callers(self) -> None:
        rows = _ccd_rows()
        a = create_plot("fds_plot", design_data=rows, random_state=1, n_samples=500).to_spec()
        b = create_plot("fds_plot", design_data=rows, random_state=1, n_samples=500).to_spec()
        c = create_plot("fds_plot", design_data=rows, random_state=2, n_samples=500).to_spec()
        assert a.metadata["median_spv"] == b.metadata["median_spv"] != c.metadata["median_spv"]


# ---------------------------------------------------------------------------
# Prediction-variance contour
# ---------------------------------------------------------------------------


def _box_rows() -> list[dict[str, float]]:
    pts = [list(p) for p in itertools.product([-1, 1], repeat=3)]
    for i in range(3):
        for s in (-1, 1):
            p = [0, 0, 0]
            p[i] = s
            pts.append(p)
    pts += [[0, 0, 0]] * 3
    return [{"A": a, "B": b, "C": c, "y": 0.0} for a, b, c in pts]


def _quadratic_row(m: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(m))] + [m[:, i] for i in range(3)]
    cols += [m[:, i] * m[:, j] for i in range(3) for j in range(i + 1, 3)]
    cols += [m[:, i] ** 2 for i in range(3)]
    return np.column_stack(cols)


class TestPredictionVariancePlot:
    def test_non_plotted_factors_stay_in_the_model(self) -> None:
        rows = _box_rows()
        spec = create_plot(
            "prediction_variance", design_data=rows, response_column="y", factors_to_plot=["A", "B"]
        ).to_spec()
        z = np.array(spec.panels[0].layers[0].style["z_matrix"])
        x = np.array([[r["A"], r["B"], r["C"]] for r in rows], dtype=float)
        f = _quadratic_row(x)
        corner = _quadratic_row(np.array([[1.0, 1.0, 0.0]]))
        expected = len(x) * (corner @ np.linalg.inv(f.T @ f) @ corner.T).item()
        assert z[-1, -1] == pytest.approx(expected)

    def test_hold_value_moves_the_surface(self) -> None:
        rows = _box_rows()
        kw = {"design_data": rows, "response_column": "y", "factors_to_plot": ["A", "B"]}
        z0 = np.array(create_plot("prediction_variance", **kw).to_spec().panels[0].layers[0].style["z_matrix"])
        z1 = np.array(
            create_plot("prediction_variance", hold_values={"C": 1.0}, **kw)
            .to_spec()
            .panels[0]
            .layers[0]
            .style["z_matrix"]
        )
        assert not np.allclose(z0, z1)

    def test_non_estimable_model_raises(self) -> None:
        rows = [{"A": a, "B": b, "y": 0.0} for a, b in itertools.product([-1, 1], repeat=2)]
        rows += [{"A": 0, "B": 0, "y": 0.0}] * 3
        with pytest.raises(ValueError, match="cannot estimate"):
            create_plot("prediction_variance", design_data=rows, response_column="y", model="quadratic").to_spec()


# ---------------------------------------------------------------------------
# Lenth thresholds on the Pareto and half-normal plots
# ---------------------------------------------------------------------------


def _sme_quantile(alpha: float, m: int) -> float:
    """Lenth's simultaneous quantile, ``t(gamma; m/3)`` with ``gamma = (1 + (1 - alpha)**(1/m)) / 2``."""
    return stats.t.ppf((1 + (1 - alpha) ** (1 / m)) / 2, m / 3)


def _lenth_analysis() -> dict:
    effects = {"A": 6.0, "B": -4.0, "C": 0.3, "D": -0.2, "A:B": 0.25, "A:C": 3.0, "B:C": -0.1}
    pse = 0.3
    m = len(effects)
    return {
        "effects": effects,
        "lenth_method": {
            "PSE": pse,
            "ME": stats.t.ppf(0.975, m / 3) * pse,
            "SME": _sme_quantile(0.05, m) * pse,
            "significance_level": 0.05,
            "effects": [{"term": k, "effect": v, "active_ME": abs(v) > 1} for k, v in effects.items()],
        },
    }


@pytest.mark.parametrize("plot_type", ["pareto", "half_normal"])
def test_lenth_thresholds_follow_the_confidence_level(plot_type: str) -> None:
    """Recomputed at another level, SME uses Lenth's gamma, as ``analyze_experiment`` does, not Bonferroni."""
    analysis = _lenth_analysis()
    m = len(analysis["effects"])
    spec = create_plot(plot_type, analysis_results=analysis, confidence_level=0.99).to_spec()
    by_label = {a.label: a.value for a in spec.panels[0].annotations}
    assert by_label["ME (α=0.01)"] == pytest.approx(stats.t.ppf(0.995, m / 3) * 0.3)  # noqa: RUF001
    if plot_type == "pareto":
        assert by_label["SME (α=0.01)"] == pytest.approx(_sme_quantile(0.01, m) * 0.3)  # noqa: RUF001


def _unreplicated_analysis(significance_level: float) -> dict:
    """Effects and Lenth's method for a 2^3 factorial with one real effect, from ``analyze_experiment``."""
    design = pd.DataFrame(list(itertools.product([-1, 1], repeat=3)), columns=["A", "B", "C"])
    y = 10 + 3 * design["A"] + np.array([0.1, -0.2, 0.05, 0.15, -0.1, 0.2, -0.05, 0.0])
    return analyze_experiment(
        design,
        y.rename("y"),
        model="interactions",
        analysis_type=["effects", "lenth_method"],
        significance_level=significance_level,
    )


def test_lenth_margins_drawn_at_the_analysis_level_are_the_analysis_margins() -> None:
    """At the level the analysis used, the plot draws its ME and SME as reported, labelled with that level.

    The plot looked for the level under ``alpha``, but the analysis stores it as
    ``significance_level``, so an analysis at 0.10 was taken to be at 0.05.
    """
    analysis = _unreplicated_analysis(0.10)
    spec = create_plot("pareto", analysis_results=analysis, confidence_level=0.90).to_spec()
    by_label = {a.label: a.value for a in spec.panels[0].annotations}
    assert by_label["ME (α=0.1)"] == pytest.approx(analysis["lenth_method"]["ME"])  # noqa: RUF001
    assert by_label["SME (α=0.1)"] == pytest.approx(analysis["lenth_method"]["SME"])  # noqa: RUF001


def test_lenth_margins_recomputed_at_another_level_match_an_analysis_there() -> None:
    """Margins the plot recomputes at 0.05 equal those of an analysis run at 0.05."""
    spec = create_plot("pareto", analysis_results=_unreplicated_analysis(0.10), confidence_level=0.95).to_spec()
    by_label = {a.label: a.value for a in spec.panels[0].annotations}
    reference = _unreplicated_analysis(0.05)["lenth_method"]
    assert by_label["ME (α=0.05)"] == pytest.approx(reference["ME"])  # noqa: RUF001
    assert by_label["SME (α=0.05)"] == pytest.approx(reference["SME"])  # noqa: RUF001


def test_default_confidence_labels_are_formatted() -> None:
    spec = create_plot("pareto", analysis_results=_lenth_analysis()).to_spec()
    labels = sorted(a.label for a in spec.panels[0].annotations)
    assert labels == ["ME (α=0.05)", "SME (α=0.05)"]  # noqa: RUF001


# ---------------------------------------------------------------------------
# visualize_doe backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["Plotly", "matplotlib", ""])
def test_unknown_backend_raises(backend: str) -> None:
    with pytest.raises(ValueError, match="backend"):
        visualize_doe(plot_type="pareto", analysis_results={"effects": {"A": 5.0, "B": -3.0}}, backend=backend)


def test_visualize_doe_forwards_the_model() -> None:
    out = visualize_doe(plot_type="power_curve", design_data=_factorial_rows(3), model="main_effects", backend="plotly")
    assert [layer["name"] for layer in out["data"]["panels"][0]["layers"]] == ["A, B, C"]


def test_default_model_is_the_richest_the_design_estimates() -> None:
    """A 2^2 with centre points has three levels per factor, but cannot estimate both squares."""
    rows = [{"A": a, "B": b, "y": 0.0} for a, b in itertools.product([-1, 1], repeat=2)]
    rows += [{"A": 0, "B": 0, "y": 0.0}] * 3
    spec = create_plot("prediction_variance", design_data=rows, response_column="y").to_spec()
    assert spec.metadata["model"] == "interactions"
    assert create_plot("fds_plot", design_data=_ccd_rows()).to_spec().metadata["model"] == "quadratic"
