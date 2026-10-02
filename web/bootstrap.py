"""Python glue between the browser page and ``process_improve.experiments``.

Pyodide loads this file into the web worker (``worker.js``). The page calls the
public functions at the bottom; each takes and returns a JSON string, so only
plain text crosses the JavaScript boundary and no ``PyProxy`` has to be kept
alive or destroyed on the JavaScript side.

The page has no state of its own. Everything needed to analyse a campaign
travels inside the ``.xlsx`` workbook the reader holds: the run sheet, and a
hidden sheet with the design specification (factors, ranges, design type).

Security
--------
A workbook can arrive by e-mail, so its contents are data and are never
executed. The specification is read with ``json.loads``. The model is **not**
read from the file as a formula, because patsy evaluates formula terms as
Python: only the names in :data:`MODELS` are accepted, and each maps to a
formula that ``analyze_experiment`` builds itself.
"""

from __future__ import annotations

import base64
import io
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.experiments import Factor, analyze_experiment, generate_design

if TYPE_CHECKING:
    from collections.abc import Callable

SPEC_SHEET = "_pi_design"
RUN_SHEET = "Runs"
SPEC_VERSION = 1

#: Design types the page offers. Each one is pure numpy / scipy, so it runs
#: under WebAssembly. Excluded: ``omars_ilp`` (its HiGHS multistart needs
#: scipy >= 1.15, newer than Pyodide's, and takes too long for a page) and
#: ``i_optimal`` (needs pyoptex).
DESIGNS: dict[str, dict[str, Any]] = {
    "full_factorial": {
        "label": "Full factorial (2-level)",
        "hint": "Every combination of low and high. 2^k runs, plus center points.",
        "model": "interactions",
        "options": ["n_center_points"],
    },
    "fractional_factorial": {
        "label": "Fractional factorial (2-level)",
        "hint": "A fraction of the full factorial; main effects are aliased with interactions.",
        "model": "main_effects",
        "options": ["n_center_points"],
    },
    "plackett_burman": {
        "label": "Plackett-Burman screening",
        "hint": "Main effects only, in a multiple of 4 runs.",
        "model": "main_effects",
        "options": [],
    },
    "dsd": {
        "label": "Definitive screening (DSD)",
        "hint": "Three levels, 2k+1 runs; main effects clear of two-factor interactions.",
        "model": "main_effects",
        "options": [],
    },
    "box_behnken": {
        "label": "Box-Behnken",
        "hint": "Response surface with every run inside the factor ranges (3 or more factors).",
        "model": "quadratic",
        "options": ["n_center_points"],
    },
    "ccd": {
        "label": "Central composite (CCD)",
        "hint": "Response surface: factorial corners, axial points and center points.",
        "model": "quadratic",
        "options": ["n_center_points", "alpha"],
    },
    "d_optimal": {
        "label": "D-optimal",
        "hint": "Best estimates of the chosen model's coefficients for a fixed number of runs.",
        "model": "quadratic",
        "options": ["budget", "model_type"],
    },
}

#: The only models an uploaded workbook may request. Mapped by name, never
#: parsed from the file, so a workbook cannot smuggle a patsy formula in.
MODELS = ("main_effects", "interactions", "quadratic")

ALPHAS = ("face_centered", "rotatable", "orthogonal")


# --------------------------------------------------------------------------- helpers
def _clean(value: object) -> object:
    """Turn numpy scalars, NaN and inf into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def _reply(fn: Callable[[dict], dict], payload: str) -> str:
    """Run ``fn`` on the decoded payload; report errors as data, not exceptions."""
    try:
        return json.dumps({"ok": True, "result": _clean(fn(json.loads(payload or "{}")))})
    except (ValueError, KeyError, TypeError) as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def _factors(spec: dict) -> list[Factor]:
    """Build and check the ``Factor`` list from the specification."""
    raw = spec.get("factors") or []
    if len(raw) < 2:
        raise ValueError("Enter at least two factors.")
    names = [str(f.get("name", "")).strip() for f in raw]
    if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
        raise ValueError("Factor names must be unique and look like variable names (letters, digits, _).")
    factors = []
    for f, name in zip(raw, names, strict=True):
        low, high = float(f["low"]), float(f["high"])
        if not low < high:
            raise ValueError(f"Factor {name}: the low value must be below the high value.")
        factors.append(Factor(name=name, low=low, high=high, units=str(f.get("units", ""))))
    return factors


# --------------------------------------------------------------------------- public API
def catalogue(_: dict) -> dict:
    """List what the page offers: design types, models and CCD axial choices."""
    return {"designs": DESIGNS, "models": list(MODELS), "alphas": list(ALPHAS)}


def make_design(spec: dict) -> dict:
    """Generate a design and return its runs plus a ready-to-download workbook.

    The workbook lists the runs in randomised order, in the factors' own units,
    with an empty response column for the reader to fill in.
    """
    design_type = spec.get("design_type", "")
    if design_type not in DESIGNS:
        raise ValueError(f"Unknown design type {design_type!r}.")
    factors = _factors(spec)
    response = str(spec.get("response") or "y").strip()
    if not response.isidentifier() or response in {f.name for f in factors}:
        raise ValueError("The response name must be a variable name different from every factor.")

    kwargs = _design_options(spec, design_type)
    result = generate_design(factors, design_type=design_type, **kwargs)

    names = [f.name for f in factors]
    runs = pd.DataFrame(result.design_actual[names].to_numpy(), columns=names)
    runs.insert(0, "Run order", result.run_order)
    runs.insert(0, "Std order", range(1, len(runs) + 1))
    runs = runs.sort_values("Run order").reset_index(drop=True)
    runs[response] = np.nan

    stored = {
        "version": SPEC_VERSION,
        "design_type": design_type,
        "factors": [{"name": f.name, "low": f.low, "high": f.high, "units": f.units} for f in factors],
        "response": response,
        "model": kwargs.get("model_type", DESIGNS[design_type]["model"]),
        "options": kwargs,
    }
    return {
        "n_runs": len(runs),
        "columns": list(runs.columns),
        "rows": [list(row) for row in runs.itertuples(index=False)],
        "outside_range": _outside_range(runs, factors),
        "model": stored["model"],
        "xlsx_base64": _to_workbook(runs, stored),
    }


def _design_options(spec: dict, design_type: str) -> dict[str, Any]:
    """Pick the ``generate_design`` keyword arguments this design type accepts."""
    kwargs: dict[str, Any] = {"random_seed": int(spec.get("random_seed", 42))}
    allowed = DESIGNS[design_type]["options"]
    if "n_center_points" in allowed:
        kwargs["n_center_points"] = int(spec.get("n_center_points", 3))
    if "alpha" in allowed:
        alpha = spec.get("alpha", "face_centered")
        if alpha not in ALPHAS:
            raise ValueError(f"Unknown axial distance {alpha!r}.")
        kwargs["alpha"] = alpha
    if "budget" in allowed and spec.get("budget"):
        kwargs["budget"] = int(spec["budget"])
    if "model_type" in allowed:
        model_type = spec.get("model_type", "quadratic")
        if model_type not in MODELS:
            raise ValueError(f"Unknown model {model_type!r}.")
        kwargs["model_type"] = model_type
    return kwargs


def _outside_range(runs: pd.DataFrame, factors: list[Factor]) -> list[str]:
    """Factors with runs beyond the ranges entered (CCD axial points can be)."""
    return [f.name for f in factors if runs[f.name].min() < f.low - 1e-9 or runs[f.name].max() > f.high + 1e-9]


def _to_workbook(runs: pd.DataFrame, spec: dict) -> str:
    """Write the run sheet and the hidden specification sheet, base64-encoded."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        runs.to_excel(xl, sheet_name=RUN_SHEET, index=False)
        pd.DataFrame({"spec": [json.dumps(spec)]}).to_excel(xl, sheet_name=SPEC_SHEET, index=False)
        xl.book[SPEC_SHEET].sheet_state = "hidden"
        xl.book[RUN_SHEET].freeze_panes = "A2"
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _read_workbook(data: bytes) -> tuple[pd.DataFrame, dict]:
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl")
    except Exception as exc:  # openpyxl raises several unrelated types on a bad file
        raise ValueError("That file could not be read as an .xlsx workbook.") from exc
    if RUN_SHEET not in sheets or SPEC_SHEET not in sheets:
        raise ValueError(f"This workbook was not made by this page: it needs the {RUN_SHEET!r} sheet and its design.")
    spec = json.loads(str(sheets[SPEC_SHEET]["spec"].iloc[0]))
    if spec.get("version") != SPEC_VERSION:
        raise ValueError("This workbook was made by a different version of the page.")
    return sheets[RUN_SHEET], spec


def analyze(request: dict) -> dict:
    """Fit the model to the completed runs of an uploaded workbook.

    Factors are coded to -1 / +1 from the ranges stored in the workbook, so the
    coefficients are comparable with each other whatever the factors' units.
    Rows with an empty response are left out and reported, so a partly filled
    workbook can be analysed as the runs come in.
    """
    runs, spec = _read_workbook(base64.b64decode(request["xlsx_base64"]))
    model = request.get("model") or spec.get("model", "interactions")
    if model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    response = spec["response"]
    factors = spec["factors"]
    names = [f["name"] for f in factors]
    missing_cols = [c for c in [*names, response] if c not in runs.columns]
    if missing_cols:
        raise ValueError(f"The Runs sheet is missing column(s): {', '.join(missing_cols)}.")

    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna()
    pending = runs.loc[~done, "Run order"].astype(int).tolist() if "Run order" in runs else []

    coded = pd.DataFrame(
        {f["name"]: (runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2) for f in factors}
    )[done].reset_index(drop=True)
    n_terms = _n_terms(len(names), model)
    if done.sum() <= n_terms:
        raise ValueError(
            f"The {model.replace('_', ' ')} model has {n_terms} coefficients, so it needs more than {n_terms} "
            f"completed runs; this workbook has {int(done.sum())}. Pending run(s): {pending}."
        )

    out = analyze_experiment(
        coded,
        y[done].reset_index(drop=True).rename(response),
        model=model,
        analysis_type=["anova", "coefficients", "lack_of_fit"],
    )
    return {
        "spec": spec,
        "model": model,
        "n_completed": int(done.sum()),
        "pending_runs": pending,
        "summary": out["model_summary"],
        "coefficients": out.get("coefficients", []),
        "anova": out.get("anova_table", []),
        "lack_of_fit": out.get("lack_of_fit"),
        "equation": _equation(response, out.get("coefficients", [])),
    }


def _n_terms(k: int, model: str) -> int:
    """Count the coefficients, intercept included, in each named model."""
    interactions = k * (k - 1) // 2
    return 1 + k + {"main_effects": 0, "interactions": interactions, "quadratic": interactions + k}[model]


def _equation(response: str, coefficients: list[dict]) -> str:
    """Write the fitted model, in coded units, out as one line."""
    parts = []
    for c in coefficients:
        term = c["term"].replace("I(", "").replace(" ** 2)", "²").replace(":", "·")
        value = c["coefficient"]
        text = f"{abs(value):.4g}" if term == "Intercept" else f"{abs(value):.4g}·{term}"
        sign = "-" if value < 0 else "+"
        parts.append(text if not parts and value >= 0 else f"{sign} {text}")
    return f"{response} = " + " ".join(parts)


# --------------------------------------------------------------------------- JSON entry points
def api_catalogue(payload: str = "{}") -> str:
    """JSON wrapper around :func:`catalogue`."""
    return _reply(catalogue, payload)


def api_make_design(payload: str) -> str:
    """JSON wrapper around :func:`make_design`."""
    return _reply(make_design, payload)


def api_analyze(payload: str) -> str:
    """JSON wrapper around :func:`analyze`."""
    return _reply(analyze, payload)
