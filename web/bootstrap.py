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
import itertools
import json
import math
import re
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
#: scipy >= 1.15, newer than Pyodide's, and takes too long for a page).
#:
#: ``min_factors`` is what the construction needs. ``max_factors`` keeps the run
#: sheet readable and the page responsive: a full factorial in 8 factors has 256
#: runs, a Box-Behnken design jumps from 59 to 115 runs at 8 factors, and an
#: I-optimal design for a quadratic in 6 factors already takes about 10 s under
#: WebAssembly.
DESIGNS: dict[str, dict[str, Any]] = {
    "full_factorial": {
        "label": "Full factorial (2-level)",
        "hint": "Every combination of low and high. 2^k runs, plus center points.",
        "model": "interactions",
        "options": ["n_center_points"],
        "min_factors": 2,
        "max_factors": 7,
    },
    "fractional_factorial": {
        "label": "Fractional factorial (2-level)",
        "hint": (
            "A fraction of the full factorial. Its resolution says what is aliased: at III, main effects "
            "with two-factor interactions; at IV, two-factor interactions with each other; at V, neither."
        ),
        "model": "main_effects",
        "options": ["n_center_points", "resolution"],
        "min_factors": 3,
        "max_factors": 11,
    },
    "plackett_burman": {
        "label": "Plackett-Burman screening",
        "hint": (
            "Main effects only, in a multiple of 4 runs plus center points; each main effect is partly "
            "or fully aliased with two-factor interactions."
        ),
        "model": "main_effects",
        "options": ["n_center_points"],
        "min_factors": 2,
        "max_factors": 11,
    },
    "dsd": {
        "label": "Definitive screening (DSD)",
        "hint": "Three levels, 2k+1 runs (2k+3 for odd k); main effects clear of interactions and quadratics.",
        "model": "main_effects",
        "options": [],
        "min_factors": 3,
        "max_factors": 12,
    },
    "box_behnken": {
        "label": "Box-Behnken",
        "hint": "Response surface with every run inside the factor ranges and none at the corners.",
        "model": "quadratic",
        "options": ["n_center_points"],
        "min_factors": 3,
        "max_factors": 7,
    },
    "ccd": {
        "label": "Central composite (CCD)",
        "hint": "Response surface: factorial corners, axial points and center points.",
        "model": "quadratic",
        "options": ["n_center_points", "alpha"],
        "min_factors": 2,
        "max_factors": 7,
    },
    "d_optimal": {
        "label": "D-optimal",
        "hint": "Best estimates of the chosen model's coefficients for a fixed number of runs.",
        "model": "quadratic",
        "options": ["budget", "model_type"],
        "min_factors": 2,
        "max_factors": 8,
    },
    "i_optimal": {
        "label": "I-optimal",
        "hint": "Most precise predictions, on average over the factor space, for a fixed number of runs.",
        "model": "quadratic",
        "options": ["budget", "model_type"],
        "min_factors": 2,
        "max_factors": 6,
    },
}

#: Designs whose non-centre runs all lie on one sphere (Box-Behnken at any size, a
#: rotatable CCD at 2 or 4 factors), so without a center run the squared terms sum
#: to a constant and the quadratic model cannot be fitted.
NEEDS_CENTER_RUN = {"box_behnken": None, "ccd": "rotatable"}

#: Runs an optimal design gets beyond its model's coefficients, when no budget is
#: given, so the fit keeps degrees of freedom to estimate the error.
SPARE_RUNS = 4

#: Fractional factorial resolutions offered on the page, and the default.
RESOLUTIONS = (3, 4, 5)
DEFAULT_RESOLUTION = 4

#: From this resolution a fraction keeps two-factor interactions clear of main
#: effects and of each other, so it can be analysed with them.
INTERACTIONS_CLEAR_FROM = 5

#: Terms whose centred columns have a correlation this close to 1 are fully aliased.
ALIAS_TOLERANCE = 1e-9

#: How many alias groups a summary sentence names before saying how many more.
ALIASES_SHOWN = 4

#: From this many factors a CCD uses a resolution V half fraction of the cube, the
#: switch ``process_improve.experiments.strategy.budget`` assumes.
CCD_HALF_FRACTION_FROM = 6

ROMAN = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}

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
    _check_factor_count(design_type, len(factors))
    response = str(spec.get("response") or "y").strip()
    if not response.isidentifier() or response in {f.name for f in factors}:
        raise ValueError("The response name must be a variable name different from every factor.")

    kwargs = _design_options(spec, design_type)
    result = generate_design(factors, design_type=design_type, **kwargs)

    # The library returns the runs already randomised, and ``run_order[i]`` is the
    # standard-order position of its i-th run: the sheet keeps the rows as they come.
    names = [f.name for f in factors]
    runs = pd.DataFrame(result.design_actual[names].to_numpy(), columns=names)
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", result.run_order)
    runs[response] = np.nan

    stored = {
        "version": SPEC_VERSION,
        "design_type": design_type,
        "factors": [{"name": f.name, "low": f.low, "high": f.high, "units": f.units} for f in factors],
        "response": response,
        "model": kwargs.get("model_type") or _designed_model(design_type, result.resolution),
        "options": kwargs,
    }
    return {
        "n_runs": len(runs),
        "columns": list(runs.columns),
        "rows": [list(row) for row in runs.itertuples(index=False)],
        "outside_range": _outside_range(runs, factors),
        "model": stored["model"],
        "facts": _design_facts(result, names, kwargs),
        "xlsx_base64": _to_workbook(runs, stored),
    }


def _check_factor_count(design_type: str, n_factors: int) -> None:
    """Refuse a design the factor count rules out, saying which counts it takes."""
    info = DESIGNS[design_type]
    low, high = info["min_factors"], info["max_factors"]
    if not low <= n_factors <= high:
        raise ValueError(
            f"The {info['label']} design takes {low} to {high} factors on this page; you have {n_factors}."
        )


def _designed_model(design_type: str, resolution: int | None) -> str:
    """Name the model a design is analysed with when the reader does not choose one.

    A fraction keeps two-factor interactions clear of main effects and of each other
    only from resolution V, so below that it is analysed for its main effects.
    """
    if design_type == "fractional_factorial" and (resolution or 0) >= INTERACTIONS_CLEAR_FROM:
        return "interactions"
    return DESIGNS[design_type]["model"]


def _design_options(spec: dict, design_type: str) -> dict[str, Any]:
    """Pick the ``generate_design`` keyword arguments this design type accepts."""
    kwargs: dict[str, Any] = {"random_state": int(spec.get("random_state", spec.get("random_seed", 42)))}
    allowed = DESIGNS[design_type]["options"]
    n_factors = len(spec.get("factors") or [])
    if "n_center_points" in allowed:
        kwargs["n_center_points"] = _center_points(spec, design_type)
    if "alpha" in allowed:
        alpha = spec.get("alpha", "face_centered")
        if alpha not in ALPHAS:
            raise ValueError(f"Unknown axial distance {alpha!r}.")
        kwargs["alpha"] = alpha
    if design_type == "ccd" and n_factors >= CCD_HALF_FRACTION_FROM:
        kwargs["cube"] = "fractional"
    if "resolution" in allowed:
        kwargs["resolution"] = _resolution(spec, n_factors)
    if "model_type" in allowed:
        model_type = spec.get("model_type", "quadratic")
        if model_type not in MODELS:
            raise ValueError(f"Unknown model {model_type!r}.")
        kwargs["model_type"] = model_type
        budget = spec.get("budget")
        kwargs["budget"] = int(budget) if budget else _n_terms(n_factors, model_type) + SPARE_RUNS
    return kwargs


def _center_points(spec: dict, design_type: str) -> int:
    """Read the number of center runs, refusing none where the quadratic model would be singular."""
    n_center = int(spec.get("n_center_points", 3))
    if n_center < 0:
        raise ValueError("The number of center points cannot be negative.")
    if design_type in NEEDS_CENTER_RUN and n_center == 0:
        alpha = NEEDS_CENTER_RUN[design_type]
        if alpha is None or spec.get("alpha", "face_centered") == alpha:
            raise ValueError(
                "This design needs at least one center point: without one every other run lies on one sphere "
                "around the center, so the squared terms cannot be told apart from the intercept."
            )
    return n_center


def _resolution(spec: dict, n_factors: int) -> int:
    """Read the resolution asked for, which can be at most the number of factors."""
    highest = min(max(RESOLUTIONS), n_factors)
    resolution = int(spec.get("resolution") or min(DEFAULT_RESOLUTION, highest))
    if not min(RESOLUTIONS) <= resolution <= highest:
        allowed = " to ".join(dict.fromkeys(ROMAN[r] for r in (min(RESOLUTIONS), highest)))
        raise ValueError(
            f"A fractional factorial in {n_factors} factors can have resolution {allowed} "
            f"on this page, not {ROMAN.get(resolution, resolution)}."
        )
    return resolution


def _design_facts(result: Any, names: list[str], kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """State what the generated design is, read off the design rather than its type.

    A fraction's resolution and generators come from the library; which main effects and
    two-factor interactions are fully aliased is read from the coded runs, so it holds for
    every design type and every naming of the factors. ``notes`` are the sentences the
    page shows.
    """
    coded = result.design[names].to_numpy(dtype=float)
    groups = _aliased_groups(coded, names)
    notes = []
    if result.resolution:
        generators = [_spaced(g, names) for g in result.generators or []]
        label = "generator" if len(generators) == 1 else "generators"
        notes.append(f"Resolution {ROMAN.get(result.resolution, result.resolution)}, {label} {', '.join(generators)}.")
    notes.append(_aliasing_sentence(groups))
    asked = kwargs.get("budget")
    if asked and len(coded) > asked:
        model = kwargs["model_type"]
        notes.append(
            f"You asked for {asked} runs; the {model.replace('_', ' ')} model in {len(names)} factors has "
            f"{_n_terms(len(names), model)} coefficients, so the design has {len(coded)} runs."
        )
    return {
        "resolution": result.resolution,
        "generators": [_spaced(g, names) for g in result.generators or []],
        "defining_relation": [_spaced(w, names) for w in result.defining_relation or []],
        "aliases": groups,
        "center_runs": int(np.all(np.abs(coded) < ALIAS_TOLERANCE, axis=1).sum()),
        "notes": notes,
    }


def _aliased_groups(coded: np.ndarray, names: list[str]) -> list[list[str]]:
    """Group the main effects and two-factor interactions the design cannot tell apart.

    Two terms are fully aliased when their centred columns are parallel, so the data
    determine only their sum (or difference). A partial correlation, such as the 1/3
    between a main effect and an interaction in a 12-run Plackett-Burman design, is not
    aliasing in this sense. Each group lists its main effect first.
    """
    pairs = list(itertools.combinations(range(len(names)), 2))
    labels = list(names) + [f"{names[i]}·{names[j]}" for i, j in pairs]
    columns = np.column_stack([coded, *[coded[:, i] * coded[:, j] for i, j in pairs]])
    centred = columns - columns.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    varying = np.flatnonzero(norms > ALIAS_TOLERANCE)
    unit = centred[:, varying] / norms[varying]
    parallel = np.abs(unit.T @ unit) > 1 - ALIAS_TOLERANCE
    groups, placed = [], set()
    for i in range(len(varying)):
        if i in placed:
            continue
        members = np.flatnonzero(parallel[i]).tolist()
        placed.update(members)
        if len(members) > 1:
            groups.append([labels[varying[m]] for m in members])
    return groups


def _aliasing_sentence(groups: list[list[str]]) -> str:
    """Say in one sentence what the alias groups mean for the analysis."""
    if not groups:
        return "No main effect or two-factor interaction is fully aliased with another."
    mixed = [g for g in groups if any("·" not in term for term in g)]
    shown = mixed or groups
    listed = "; ".join(" = ".join(g) for g in shown[:ALIASES_SHOWN])
    more = f"; and {len(shown) - ALIASES_SHOWN} more" if len(shown) > ALIASES_SHOWN else ""
    if mixed:
        return f"Main effects are aliased with two-factor interactions: {listed}{more}."
    return f"Main effects are clear of two-factor interactions, which are aliased with each other: {listed}{more}."


def _spaced(word: str, names: list[str]) -> str:
    """Write a generator or defining word with its factors apart: ``D=ABC`` as ``D = A·B·C``.

    The library writes the factors of a word side by side, which is ambiguous to read
    when the names are longer than a letter. A word that does not split into the factor
    names is left as the library wrote it.
    """
    pattern = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)))

    def split(part: str) -> str:
        sign, body = ("-", part[1:]) if part.startswith("-") else ("", part.lstrip("+"))
        found = pattern.findall(body)
        return sign + "·".join(found) if found and "".join(found) == body else part

    left, _, right = (side.strip() for side in word.partition("="))
    return f"{left if left == 'I' else split(left)} = {split(right)}"


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
