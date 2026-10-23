"""What if: ``api_predictors``, each response's fitted model as numbers, and ``predictAt`` in ``web/best.js``.

Learn's what-if panel predicts every response at settings the reader moves, in the page's own
JavaScript, from the fit Python sends once per analysis. These tests hold the JavaScript's numbers
to two others: statsmodels, fitted here to the run sheet with formulas of its own, whose
predictions, intervals and leverage do not depend on how the page codes its columns; and the best
settings and the balance, which predict with ``_FittedModel.at``, so the panel opened at them shows
what they show. ``api_outside_limits`` names the limits some settings break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols

from tests.web._support import FACTORS, held_of, javascript, run_sheet, source

T, P = FACTORS[0], FACTORS[1]
CATALYST = {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""}
SUPPLIER = {"name": "Supplier", "type": "categorical", "levels": ["1", "2", "3"], "units": ""}
BATCH = {"name": "Batch", "type": "categorical", "levels": ["a", "b", "c", "d"], "units": ""}
TWO = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize"},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
]


# --------------------------------------------------------------------------- helpers
def made(api, design_type: str, factors: list[dict], **options: object) -> dict:
    """Make a design as the page holds it."""
    return held_of(api.ok("api_make_design", {"design_type": design_type, "factors": factors, **options}))


def noisy(sheet: pd.DataFrame, seed: int = 1) -> np.ndarray:
    """Return results that curve in T, rise with P, and differ by catalyst, supplier and batch, with noise."""
    rng = np.random.default_rng(seed)
    y = 60 + rng.normal(0, 0.5, len(sheet))
    if "T" in sheet:
        t = (sheet["T"].to_numpy(float) - 175) / 25
        y += 4 * t - 3 * t**2
    if "P" in sheet:
        y += 2 * (sheet["P"].to_numpy(float) - 2)
    for name, size in (("Catalyst", 1.5), ("Supplier", 1.0), ("Batch", 0.7)):
        if name in sheet:
            y += size * pd.Categorical(sheet[name].astype(str)).codes
    return y.round(3)


def predicted_in_js(node: str, reply: dict, settings: list[dict]) -> list[list[dict]]:
    """Return what ``predictAt`` gives for each response with a fit at each of the settings, as the page computes it."""
    module = source("best.js")
    columns = javascript(node, module, [("columnsAt", [reply["factors"], s]) for s in settings])
    fits = [r for r in reply["responses"] if "terms" in r]
    out = javascript(node, module, [("predictAt", [fit, c]) for c in columns for fit in fits])
    return [out[i * len(fits) : (i + 1) * len(fits)] for i in range(len(settings))]


def random_settings(factors: list[dict], count: int, seed: int = 7) -> list[dict]:
    """Draw settings inside the factors' ranges, a level of each categorical factor, as the reader might set them."""
    rng = np.random.default_rng(seed)

    def one(f: dict) -> str | float:
        if f.get("type") == "categorical":
            return str(rng.choice(f["levels"]))
        return float(rng.uniform(f["low"], f["high"]))

    return [{f["name"]: one(f) for f in factors} for _ in range(count)]


# --------------------------------------------------------------------------- the predictions, against statsmodels
#: Studies and the statsmodels formula over the run sheet that spans the same columns as the page's model.
STUDIES = {
    "a quadratic in T and P": ("ccd", [T, P], {"alpha": "face_centered"}, "quadratic", "y ~ T * P + I(T**2) + I(P**2)"),
    "interactions in T and P": ("ccd", [T, P], {"alpha": "face_centered"}, "interactions", "y ~ T * P"),
    "a catalyst of three levels with T": (
        "full_factorial",
        [CATALYST, T],
        {"n_replicates": 2, "n_center_points": 0},
        "interactions",
        "y ~ C(Catalyst) * T",
    ),
    "batch nested in supplier": (
        "full_factorial",
        [SUPPLIER, BATCH],
        {"n_replicates": 2, "n_center_points": 0},
        {"terms": [["Supplier"], ["Supplier", "Batch"]]},
        "y ~ C(Supplier) + C(Supplier):C(Batch)",
    ),
}


@pytest.mark.parametrize("study", STUDIES, ids=list(STUDIES))
def test_the_page_predicts_as_statsmodels_does_at_any_settings(api, node, study):
    design_type, factors, options, model, formula = STUDIES[study]
    held = made(api, design_type, factors, **options)
    sheet = run_sheet(held)
    y = noisy(sheet)
    reply = api.ok("api_predictors", {**held, "responses": y.tolist(), "models": {"y": model}})
    settings = random_settings(factors, 5)
    frame = sheet.assign(y=y)
    for name in ("Catalyst", "Supplier", "Batch"):
        if name in frame:
            frame[name] = frame[name].astype(str)
    fit = ols(formula, frame).fit()
    for at, (page,) in zip(settings, predicted_in_js(node, reply, settings), strict=True):
        known = fit.get_prediction(pd.DataFrame([at])).summary_frame(alpha=0.05).iloc[0]
        assert page["predicted"] == pytest.approx(known["mean"], rel=1e-9)
        assert page["confidence"] == pytest.approx([known["mean_ci_lower"], known["mean_ci_upper"]], rel=1e-9)
        assert page["prediction"] == pytest.approx([known["obs_ci_lower"], known["obs_ci_upper"]], rel=1e-9)
        assert page["leverage"] == pytest.approx(known["mean_se"] ** 2 / fit.mse_resid, rel=1e-7)


def test_each_response_is_fitted_with_its_own_model(api, node):
    held = made(api, "ccd", [T, P], alpha="face_centered", responses=TWO)
    y = noisy(run_sheet(held))
    request = {**held, "responses": y.tolist(), "more_responses": {"Cost": (y / 3).tolist()}}
    reply = api.ok("api_predictors", {**request, "models": {"Yield": "quadratic", "Cost": "main_effects"}})
    assert [r["name"] for r in reply["responses"]] == ["Yield", "Cost"]
    assert [len(r["terms"]) for r in reply["responses"]] == [6, 3], "the quadratic's six terms, the main effects' three"
    assert [(r["label"], r["units"]) for r in reply["responses"]] == [("Yield", "%"), ("Cost", "$/kg")]


# --------------------------------------------------------------------------- the panel opened at the best settings
#: Studies whose best settings the panel opens at: a design, its factors, its options and the model fitted.
AIMED = {
    "ccd": ("ccd", [T, P], {"alpha": "face_centered"}, "quadratic"),
    "blocked box-behnken": ("box_behnken", FACTORS, {"n_blocks": 2, "n_center_points": 4}, "quadratic"),
    "catalyst": ("full_factorial", [CATALYST, T], {"n_replicates": 2, "n_center_points": 0}, "interactions"),
}


@pytest.mark.parametrize("study", AIMED, ids=list(AIMED))
def test_at_the_best_settings_the_page_predicts_what_the_best_settings_show(api, node, study):
    design_type, factors, options, model = AIMED[study]
    held = made(api, design_type, factors, **options)
    held["spec"]["response_info"] = {"goal": "maximize"}
    request = {**held, "responses": noisy(run_sheet(held)).tolist()}
    best = api.ok("api_best_settings", {**request, "model": model})
    reply = api.ok("api_predictors", {**request, "models": {"y": model}})
    settings = {s["name"]: s["value"] for s in best["settings"]}
    [(page,)] = predicted_in_js(node, reply, [settings])
    assert page["predicted"] == pytest.approx(best["predicted"], rel=1e-12)
    assert page["confidence"] == pytest.approx(best["confidence"], rel=1e-12)
    assert page["prediction"] == pytest.approx(best["prediction"], rel=1e-12)
    assert page["leverage"] == pytest.approx(best["leverage"], rel=1e-9)
    assert (page["max_leverage"], page["extrapolating"]) == (best["max_leverage"], best["extrapolating"])


def test_at_the_balanced_settings_each_response_is_as_the_balance_shows_it(api, node):
    held = made(api, "ccd", [T, P], alpha="face_centered", responses=TWO)
    y = noisy(run_sheet(held))
    request = {**held, "responses": y.tolist(), "more_responses": {"Cost": (20 + y / 4).tolist()}}
    models = {"Yield": "quadratic", "Cost": "interactions"}
    balanced = api.ok("api_balance", {**request, "models": models, "goals": {}})
    reply = api.ok("api_predictors", {**request, "models": models})
    [page] = predicted_in_js(node, reply, [{s["name"]: s["value"] for s in balanced["settings"]}])
    for at, shown in zip(page, balanced["responses"], strict=True):
        assert at["predicted"] == pytest.approx(shown["predicted"], rel=1e-12)
        assert at["prediction"] == pytest.approx(shown["prediction"], rel=1e-12)


# --------------------------------------------------------------------------- what cannot be predicted
def test_a_model_with_no_runs_left_over_predicts_with_no_interval(api, node):
    held = made(api, "full_factorial", [T, P], n_center_points=0)
    reply = api.ok("api_predictors", {**held, "responses": [1.0, 2.0, 4.0, 3.5], "models": {"y": "interactions"}})
    [fit] = reply["responses"]
    assert (fit["df"], fit["mse"], fit["t"]) == (0, None, None)
    [(page,)] = predicted_in_js(node, reply, [{"T": 175.0, "P": 2.0}])
    assert page["predicted"] == pytest.approx(2.625)
    assert (page["confidence"], page["prediction"]) == (None, None)


def test_a_response_without_results_waits_and_one_whose_model_cannot_be_fitted_says_why(api):
    held = made(api, "full_factorial", [SUPPLIER, BATCH], n_replicates=2, n_center_points=0, responses=TWO)
    n = len(held["rows"])
    request = {**held, "responses": [float(i % 5) for i in range(n)], "more_responses": {"Cost": [None] * n}}
    reply = api.ok("api_predictors", {**request, "models": {"Yield": {"terms": [["Supplier", "Batch"]]}}})
    yields, cost = reply["responses"]
    assert "needs" in yields["error"], "a term of two factors of several levels without either alone is refused"
    assert cost == {"name": "Cost", "label": "Cost", "units": "$/kg", "waiting": True}


def test_a_mixture_is_not_predicted_yet(api):
    held = made(api, "simplex_lattice", [{"name": n, "low": 0, "high": 1} for n in "ABC"])
    reply = api.ok("api_predictors", {**held, "responses": [1.0] * len(held["rows"])})
    assert reply == {"mixture": True, "factors": [], "responses": []}


def test_a_categorical_factor_s_levels_are_the_columns_its_model_is_fitted_on(api):
    held = made(api, "full_factorial", [CATALYST, T], n_replicates=2, n_center_points=0)
    reply = api.ok("api_predictors", {**held, "responses": noisy(run_sheet(held)).tolist()})
    catalyst = reply["factors"][0]
    assert catalyst["levels"] == ["A", "B", "C"]
    assert [{k: v for k, v in c.items() if "_L" in k} for c in catalyst["columns"]] == [
        {"Catalyst_L1": 1.0, "Catalyst_L2": 0.0},
        {"Catalyst_L1": 0.0, "Catalyst_L2": 1.0},
        {"Catalyst_L1": -1.0, "Catalyst_L2": -1.0},
    ], "sum-to-zero contrasts, the last level -1 in each"


# --------------------------------------------------------------------------- the limits the settings break
def test_the_limits_some_settings_break_are_named_by_their_place(api):
    limits = [{"expression": "T + 25*P <= 215"}, {"expression": "P >= 1.2"}]
    held = made(api, "d_optimal", [T, P], constraints=limits)
    spec = held["spec"]
    assert api.ok("api_outside_limits", {"spec": spec, "settings": {"T": 160.0, "P": 1.5}}) == {"broken": []}
    broken = api.ok("api_outside_limits", {"spec": spec, "settings": {"T": 200.0, "P": 1.1}})["broken"]
    assert [b["index"] for b in broken] == [0, 1]
    assert broken[0]["expression"] == spec["constraints"][0]["expression"]


@pytest.mark.parametrize(
    ("settings", "words"),
    [
        ({"T": 210.0, "P": 2.0}, "Set T to a number from 150 to 200."),
        ({"T": "hot", "P": 2.0}, "Set T to a number from 150 to 200."),
        ([175.0, 2.0], "The page sent settings this page did not write."),
    ],
)
def test_settings_the_page_would_not_send_are_refused(api, settings, words):
    held = made(api, "ccd", [T, P], alpha="face_centered")
    assert api.error("api_outside_limits", {"spec": held["spec"], "settings": settings}) == words
