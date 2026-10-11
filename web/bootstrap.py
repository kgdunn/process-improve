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
import contextlib
import io
import itertools
import json
import logging
import math
import re
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.experiments import Factor, TrueProcess, analyze_experiment, compare_designs, generate_design
from process_improve.experiments.compare import MODELS as COMPARISON_MODELS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SPEC_SHEET = "_pi_design"
RUN_SHEET = "Runs"
SPEC_VERSION = 1

#: The run sheet's column marking simulated results. Its space keeps it apart from every factor
#: and response, which must be variable names; workbooks written before used ``Source``.
MARKS = "Result source"

#: Design types the page offers. Each one is pure numpy / scipy, so it runs
#: under WebAssembly. Excluded: ``omars_ilp`` (its HiGHS multistart needs
#: scipy >= 1.15, newer than Pyodide's, and takes too long for a page).
#:
#: ``min_factors`` is what the construction needs. ``max_factors`` keeps the run
#: sheet readable and the page responsive: a full factorial in 8 factors has 256
#: runs, a Box-Behnken design jumps from 59 to 115 runs at 8 factors, and an
#: I-optimal design for a quadratic in 6 factors already takes up to 20 s under
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
        "hint": (
            "Three levels in 2k+1 runs (2k+3 for odd k), one of them at the center, plus any further center "
            "runs; main effects clear of interactions and quadratics."
        ),
        "model": "main_effects",
        "options": ["n_center_points"],
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
        "hint": (
            "Most precise predictions, on average over the factor space, for a fixed number of runs. "
            "The search takes longest of the designs here: up to 20 s at six factors."
        ),
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

#: Below this many effects, Lenth's pseudo standard error rests on fewer than about two
#: degrees of freedom (one per three effects), so its margins are very wide.
LENTH_FEW_EFFECTS = 7

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

#: Processes the page can simulate results from, in coded units. A term is a tuple of
#: factor positions: ``()`` the intercept, ``(0,)`` the first factor the reader
#: entered, ``(0, 1)`` the interaction of the first two, ``(2, 2)`` the square of the
#: third. A term naming a factor the reader does not have is left out, and a factor
#: no term names has no effect. ``None`` draws the process at random from the seed.
SCENARIOS: dict[str, dict[str, Any]] = {
    "reactor": {
        "label": "Reactor yield",
        "hint": "Two strong main effects, an interaction between them, and curvature in the third factor.",
        "terms": {(): 62.0, (0,): 4.0, (1,): -3.1, (2,): 1.75, (0, 1): 1.5, (2, 2): -2.25},
        "sigma": 1.0,
    },
    "main_effects": {
        "label": "Main effects only",
        "hint": "Each of the first three factors changes the response on its own, along a straight line.",
        "terms": {(): 50.0, (0,): 3.0, (1,): -2.0, (2,): 1.5},
        "sigma": 1.0,
    },
    "interaction": {
        "label": "An interaction",
        "hint": "The effect of the first factor depends on the level of the second.",
        "terms": {(): 40.0, (0,): 1.0, (1,): 0.5, (0, 1): 3.0},
        "sigma": 1.0,
    },
    "optimum": {
        "label": "A maximum inside the ranges",
        "hint": "A curved surface in the first two factors, highest at a point inside their ranges.",
        "terms": {(): 80.0, (0,): 1.2, (1,): -0.8, (0, 1): 1.0, (0, 0): -3.0, (1, 1): -2.5},
        "sigma": 1.0,
    },
    "nothing": {
        "label": "Nothing active",
        "hint": "No factor changes the response: whatever the analysis finds is noise.",
        "terms": {(): 50.0},
        "sigma": 1.0,
    },
    "random": {
        "label": "A random process",
        "hint": (
            "Two or three active factors drawn at random, with an interaction or curvature; "
            "new noise draws a new process."
        ),
        "terms": None,
        "sigma": 1.0,
    },
}

#: The worked example the page opens with, and that "Run the worked example" runs live.
DEMO: dict[str, Any] = {
    "design": {
        "design_type": "box_behnken",
        "factors": [
            {"name": "T", "low": 150, "high": 200, "units": "degC"},
            {"name": "P", "low": 1, "high": 3, "units": "bar"},
            {"name": "F", "low": 10, "high": 20, "units": "L/min"},
        ],
        "response": "y",
        "n_center_points": 3,
    },
    "scenario": "reactor",
    "sigma": 1.0,
    "seed": 1,
    "model": "",
}


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
    """Run ``fn`` on the decoded payload; report errors as data, not exceptions.

    Whatever the library warned about during the call travels with the reply as
    ``warnings``, so the page can show it rather than leave it in the console. An error
    nothing here expected comes back in one line, its type and message: the page shows
    the error as it is, and a traceback is no sentence for a reader.
    """
    with _listening() as heard:
        try:
            reply: dict[str, Any] = {"ok": True, "result": _clean(fn(_request(payload)))}
        except (ValueError, KeyError, TypeError) as exc:
            reply = {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - anything else still reaches the reader as one line
            reply = {"ok": False, "error": f"Python stopped with {type(exc).__name__}: {exc}"}
    reply["warnings"] = list(dict.fromkeys(_LOG_PREFIX.sub("", message) for message in heard))
    return json.dumps(reply)


def _request(payload: str) -> dict:
    """Decode a request: a JSON object, or an empty one for no payload."""
    request = json.loads(payload or "{}")
    if not isinstance(request, dict):
        raise ValueError("The page sent a request that is not a JSON object.")  # noqa: TRY004 - a refusal, like the rest
    return request


#: Packages whose own warnings are numerical noise on this page, such as a saturated
#: fit's division by zero; the library's warnings are kept.
_FOREIGN = ("numpy", "scipy", "statsmodels", "pandas", "patsy", "sklearn")

#: The function name some library log messages start with, which a warning of the
#: same text does not carry.
_LOG_PREFIX = re.compile(r"^\w+: ")


class _Heard(logging.Handler):
    """Keep the library's logged warnings in a list, for the reply, instead of the console."""

    def __init__(self, heard: list[str]) -> None:
        super().__init__(logging.WARNING)
        self.heard = heard

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record's message."""
        self.heard.append(record.getMessage())


@contextlib.contextmanager
def _listening() -> Iterator[list[str]]:
    """Collect what the library warns about inside the block, and send it nowhere else.

    Both channels are heard: ``warnings.warn``, and the ``process_improve`` logger, whose
    records would otherwise reach the browser console as errors. The logger's handlers are
    swapped for the duration and restored after, so an enclosing block (or a test's log
    capture) is not affected. Warnings raised inside numpy, scipy, statsmodels or pandas
    are left out.
    """
    heard: list[str] = []
    logger = logging.getLogger("process_improve")
    saved = logger.handlers[:], logger.propagate
    logger.handlers, logger.propagate = [_Heard(heard)], False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield heard
    finally:
        logger.handlers, logger.propagate = saved
    heard.extend(str(w.message) for w in caught if not _foreign(w.filename))


def _foreign(filename: str) -> bool:
    """Say whether a warning was raised inside one of the numerical packages."""
    path = filename.replace("\\", "/")
    return any(f"/{package}/" in path for package in _FOREIGN)


def _relay(heard: list[str]) -> None:
    """Pass warnings heard in an inner block on to the enclosing one."""
    for message in heard:
        logging.getLogger("process_improve").warning(message)


def _number(value: object) -> float | None:
    """Return a finite number, or None for a blank (the page sends a blank box as null) or anything else."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        low, high = _number(f.get("low")), _number(f.get("high"))
        if low is None or high is None:
            raise ValueError(f"Factor {name}: enter a number for both the low and the high value.")
        if not low < high:
            raise ValueError(f"Factor {name}: the low value must be below the high value.")
        factors.append(Factor(name=name, low=low, high=high, units=str(f.get("units", ""))))
    return factors


# --------------------------------------------------------------------------- public API
def catalogue(_: dict) -> dict:
    """List what the page offers: design types, models, CCD axial choices, processes, and the worked example."""
    scenarios = {key: {"label": s["label"], "hint": s["hint"], "sigma": s["sigma"]} for key, s in SCENARIOS.items()}
    return {"designs": DESIGNS, "models": list(MODELS), "alphas": list(ALPHAS), "scenarios": scenarios, "demo": DEMO}


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
        "spec": stored,
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
    return _analysis(runs, spec, request.get("model") or spec.get("model", "interactions"))


def _analysis(runs: pd.DataFrame, spec: dict, model: str, truth: list[dict] | None = None) -> dict:
    """Fit the model to the completed runs of a run sheet, from a workbook or from the page.

    With the true process the results were simulated from, each coefficient also
    carries the true value of what it estimates (see :func:`_add_true_values`).
    """
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
    if pending and done.sum() < n_terms:
        raise ValueError(
            f"The {model.replace('_', ' ')} model has {n_terms} coefficients, so it needs at least {n_terms} "
            f"completed runs; this workbook has {int(done.sum())}. Pending run(s): {pending}."
        )

    fit = _fit_estimable(coded, y[done].reset_index(drop=True).rename(response), model, names)
    out = fit["out"]
    coefficients = [
        {**c, "label": fit["labels"].get(c["term"], _pretty(c["term"])), "aliases": fit["chains"].get(c["term"], [])}
        for c in out.get("coefficients", [])
    ]
    if truth:
        _add_true_values(coefficients, coded, fit["formula"], _true_coefficients(truth, names))
    anova = [{**a, "source": fit["labels"].get(a["source"], _pretty(a["source"]))} for a in out.get("anova_table", [])]
    return {
        "spec": spec,
        "model": model,
        "n_completed": int(done.sum()),
        "pending_runs": pending,
        "summary": out["model_summary"],
        "coefficients": coefficients,
        "anova": anova,
        "lack_of_fit": out.get("lack_of_fit"),
        "lenth": fit["lenth"],
        "not_estimable": [_pretty(t) for t in fit["not_estimable"]],
        "notes": fit["notes"],
        "equation": _equation(response, coefficients),
    }


#: What the page asks of every fit; the effects carry the alias chains of a design
#: that cannot tell some terms apart.
ANALYSES = ["anova", "coefficients", "lack_of_fit", "effects"]


def _fit_estimable(coded: pd.DataFrame, y: pd.Series, model: str, names: list[str]) -> dict[str, Any]:
    """Fit the model, estimating each group of terms the design cannot tell apart once.

    A pseudo-inverse fit of exactly aliased terms spreads their joint estimate evenly
    over them, so ``T`` aliased with ``P:F`` would read half of what the data say about
    their sum. The library reports those groups as alias chains and the terms that move
    with the intercept as confounded with the mean; the model is then refitted with one
    term per chain, the chain's leader, labelled with the whole chain. A design that
    still cannot fit the model is refused. With no runs to spare for the error, the
    effects of an orthogonal two-level design are judged with Lenth's method.
    """
    with _listening() as heard:
        out = analyze_experiment(coded, y, model=model, analysis_type=ANALYSES)
    chains: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    not_estimable: list[str] = []
    formula = model
    if out["model_summary"]["rank_deficient"]:
        # Keyed by the chain's leader, the term kept for it; the library's label keeps each member's sign.
        for label, members in (out.get("alias_chains") or {}).items():
            chains[members[0]] = members
            labels[members[0]] = _pretty(label)
        not_estimable = list(out.get("confounded_with_mean") or [])
        dropped = set(not_estimable) | {m for members in chains.values() for m in members[1:]}
        kept = [c["term"] for c in out["coefficients"] if c["term"] != "Intercept" and c["term"] not in dropped]
        if dropped and kept:
            formula = " + ".join(_checked_term(term, names) for term in kept)
            out = analyze_experiment(coded, y, model=formula, analysis_type=ANALYSES)
    else:
        _relay(heard)
    summary = out["model_summary"]
    if summary["rank_deficient"]:
        raise ValueError(
            f"This design cannot fit the {model.replace('_', ' ')} model: only {summary['model_rank']} of its "
            f"{summary['n_terms']} coefficients can be estimated from these {summary['n_obs']} runs. Choose a "
            "smaller model."
        )
    notes = _analysis_notes(list(labels.values()), not_estimable)
    lenth = None
    if summary["df_residual"] == 0:
        terms = [c["term"] for c in out["coefficients"] if c["term"] != "Intercept"]
        if _orthogonal_two_level(coded, terms):
            lenth = analyze_experiment(coded, y, model=formula, analysis_type="lenth_method")["lenth_method"]
            lenth["effects"] = [{**e, "label": labels.get(e["term"], _pretty(e["term"]))} for e in lenth["effects"]]
            notes.append(
                "The model uses every run, so no degrees of freedom are left to estimate the error. Each effect is "
                f"judged with Lenth's method instead: it is likely real beyond the margin of error ME = "
                f"{lenth['ME']:.4g}, and still so, allowing for the number of effects tested, beyond SME = "
                f"{lenth['SME']:.4g}."
            )
            if len(lenth["effects"]) < LENTH_FEW_EFFECTS:
                notes.append(
                    f"With only {len(lenth['effects'])} effects to compare, these margins are wide. Center points "
                    "or more runs leave degrees of freedom for an ordinary test."
                )
        else:
            notes.append(
                "The model uses every run, so no degrees of freedom are left to estimate the error and nothing can "
                "be tested. Add runs, or fit fewer terms."
            )
    return {
        "out": out,
        "formula": formula,
        "chains": chains,
        "labels": labels,
        "not_estimable": not_estimable,
        "lenth": lenth,
        "notes": notes,
    }


#: The model terms a refit may name: a factor, a product of factors, or a factor squared.
_TERM = re.compile(r"I\((\w+) \*\* 2\)|\w+(?::\w+)*")


def _checked_term(term: str, names: list[str]) -> str:
    """Pass a model term into a formula only if it is built from the factors' own names.

    The terms come from the library's fit, not from the workbook, but a formula is
    evaluated as Python, so each one is checked before it is used.
    """
    match = _TERM.fullmatch(term)
    factors = [match.group(1)] if match and match.group(1) else term.split(":")
    if match is None or not all(f in names for f in factors):
        raise ValueError(f"Unexpected model term {term!r}.")
    return term


def _pretty(term: str) -> str:
    """Write a model term as the page shows it: ``I(T ** 2)`` as ``T²`` and ``T:P`` as ``T·P``."""
    return re.sub(r"I\((\w+) \*\* 2\)", r"\1²", term).replace(":", "·")


def _orthogonal_two_level(coded: pd.DataFrame, terms: list[str]) -> bool:
    """Say whether the runs are at two levels only and the terms' columns are orthogonal.

    Lenth's method treats the effects as independent estimates of equal variance, which
    holds for the main effects and interactions of a two-level factorial or
    Plackett-Burman design, not for a saturated optimal design at three levels.
    """
    values = coded.to_numpy(dtype=float)
    if not np.all(np.isclose(np.abs(values), 1.0)) or any(t.startswith("I(") for t in terms):
        return False
    columns = np.column_stack([np.prod(coded[t.split(":")].to_numpy(dtype=float), axis=1) for t in terms])
    gram = columns.T @ columns
    return bool(np.allclose(gram - np.diag(np.diag(gram)), 0.0, atol=1e-9))


def _analysis_notes(chain_labels: list[str], not_estimable: list[str]) -> list[str]:
    """Say which terms the design could not estimate separately, or at all."""
    notes = []
    if chain_labels:
        groups = "; ".join(chain_labels)
        notes.append(
            f"This design cannot tell these terms apart, so each group is estimated once, as its sum: {groups}. "
            "A large estimate may belong to any term in its group."
        )
    if not_estimable:
        terms = ", ".join(_pretty(t) for t in not_estimable)
        notes.append(
            f"{terms} cannot be estimated: they take the same value at every run, like the intercept. "
            "Center points would separate them."
        )
    return notes


def _n_terms(k: int, model: str) -> int:
    """Count the coefficients, intercept included, in each named model."""
    interactions = k * (k - 1) // 2
    return 1 + k + {"main_effects": 0, "interactions": interactions, "quadratic": interactions + k}[model]


def _equation(response: str, coefficients: list[dict]) -> str:
    """Write the fitted model, in coded units, out as one line; an alias chain is one bracketed term."""
    parts = []
    for c in coefficients:
        label = c["label"]
        value = c["coefficient"]
        if " + " in label or " - " in label:
            label = f"({label})"
        text = f"{abs(value):.4g}" if label == "Intercept" else f"{abs(value):.4g}·{label}"
        sign = "-" if value < 0 else "+"
        parts.append(text if not parts and value >= 0 else f"{sign} {text}")
    return f"{response} = " + " ".join(parts)


# --------------------------------------------------------------------------- runs held by the page
def _checked_spec(spec: dict) -> dict[str, Any]:
    """Check a design specification sent back by the page, as ``make_design`` stored it.

    The specification travels through the page, and a workbook can carry one in, so it
    is checked as data: the design type and model must be ones the page offers, and the
    factor and response names must be variable names.
    """
    if not isinstance(spec, dict) or spec.get("design_type") not in DESIGNS:
        raise ValueError("The page sent a design this page did not make.")
    factors = _factors(spec)
    response = str(spec.get("response") or "")
    if not response.isidentifier() or response in {f.name for f in factors}:
        raise ValueError("The response name must be a variable name different from every factor.")
    model = spec.get("model") or DESIGNS[spec["design_type"]]["model"]
    if model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    return {
        **spec,
        "factors": [{"name": f.name, "low": f.low, "high": f.high, "units": f.units} for f in factors],
        "response": response,
        "model": model,
    }


def _runs_frame(request: dict, spec: dict) -> pd.DataFrame:
    """Rebuild the run sheet the page shows, with the responses it holds in the response column."""
    columns = [str(c) for c in request.get("columns") or []]
    rows = request.get("rows") or []
    names = [f["name"] for f in spec["factors"]]
    missing = [n for n in names if n not in columns]
    if missing or not rows:
        raise ValueError(f"The run sheet is missing the factor column(s) {missing}." if missing else "No runs.")
    runs = pd.DataFrame(rows, columns=columns)
    runs[names] = runs[names].apply(pd.to_numeric, errors="raise")
    responses = request.get("responses")
    if responses is not None:
        if len(responses) != len(runs):
            raise ValueError(f"{len(responses)} responses were sent for {len(runs)} runs.")
        runs[spec["response"]] = pd.to_numeric(pd.Series(responses, dtype=object), errors="coerce").to_numpy()
    return runs


def _coded(runs: pd.DataFrame, factors: list[dict]) -> pd.DataFrame:
    """Code each factor's settings to -1 at its low value and +1 at its high value."""
    return pd.DataFrame(
        {f["name"]: (runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2) for f in factors}
    )


def _term_name(term: tuple[int, ...], names: list[str]) -> str:
    """Name a scenario's term, given as factor positions, as ``analyze_experiment`` names it."""
    if not term:
        return "Intercept"
    if len(term) == 2 and term[0] == term[1]:
        return f"I({names[term[0]]} ** 2)"
    return ":".join(names[i] for i in term)


def _random_terms(k: int, rng: np.random.Generator) -> dict[tuple[int, ...], float]:
    """Draw a process: two or three active factors, with an interaction or curvature among them."""

    def size() -> float:
        return round(float(rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.0)), 2)

    active = sorted(int(i) for i in rng.choice(k, size=min(k, int(rng.integers(2, 4))), replace=False))
    terms: dict[tuple[int, ...], float] = {(): round(float(rng.uniform(40.0, 80.0)), 1)}
    terms.update({(i,): size() for i in active})
    i, j = (int(v) for v in sorted(rng.choice(active, size=2, replace=False)))
    terms[(i, j) if rng.random() < 0.6 else (i, i)] = size()
    return terms


def _scenario_process(key: str, names: list[str], request: dict) -> TrueProcess:
    """Return the scenario's process over the reader's factors, with the noise the reader chose."""
    if key not in SCENARIOS:
        raise ValueError(f"Unknown process {key!r}.")
    scenario = SCENARIOS[key]
    terms = scenario["terms"]
    if terms is None:
        terms = _random_terms(len(names), np.random.default_rng(int(request.get("seed", 1))))
    coefficients = {_term_name(t, names): c for t, c in terms.items() if all(i < len(names) for i in t)}
    # A noise of 0 is refused, not read as "not given": the reader would see results with noise they did not ask for.
    sigma = scenario["sigma"] if request.get("sigma") is None else _number(request["sigma"])
    if sigma is None or sigma <= 0:
        raise ValueError("The noise standard deviation must be a number above 0.")
    return TrueProcess(coefficients, sigma=sigma)


def _truth(process: TrueProcess) -> list[dict[str, Any]]:
    """List a process's terms as the analysis lists its coefficients."""
    return [{"term": t, "label": _pretty(t), "coefficient": c} for t, c in process.coefficients.items()]


def simulate(request: dict) -> dict:
    """Simulate the results of the page's runs from a known process, in coded units, plus noise.

    The responses are rounded to the noise's precision, two significant decimals below
    its standard deviation, as a measurement would be.
    """
    spec = _checked_spec(request.get("spec") or {})
    runs = _runs_frame(request, spec)
    names = [f["name"] for f in spec["factors"]]
    key = str(request.get("scenario") or "reactor")
    process = _scenario_process(key, names, request)
    seed = int(request.get("seed", 1))
    values = process.simulate(_coded(runs, spec["factors"]), random_state=seed)
    decimals = max(0, 2 - math.floor(math.log10(process.sigma)))
    truth = _truth(process)
    return {
        "scenario": key,
        "label": SCENARIOS[key]["label"],
        "sigma": process.sigma,
        "seed": seed,
        "responses": [round(float(v), decimals) for v in values],
        "truth": truth,
        "equation": _equation(spec["response"], truth),
        "inactive": [name for name in names if name not in process.factors],  # factors the process leaves out
    }


def analyze_runs(request: dict) -> dict:
    """Fit the model to the runs the page holds, with the responses typed or simulated there.

    ``truth``, the process the results were simulated from, adds each coefficient's true value.
    """
    spec = _checked_spec(request.get("spec") or {})
    model = request.get("model") or spec["model"]
    return _analysis(_runs_frame(request, spec), spec, model, request.get("truth"))


def _true_coefficients(truth: list[dict], names: list[str]) -> dict[str, float]:
    """Read a simulated process's terms, refusing any that are not terms of the reader's factors."""
    process = TrueProcess({str(t["term"]): float(t["coefficient"]) for t in truth}, sigma=1.0)
    missing = [f for f in process.factors if f not in names]
    if missing:
        raise ValueError(f"The true process names factors the design does not have: {missing}.")
    return dict(process.coefficients)


def _add_true_values(coefficients: list[dict], coded: pd.DataFrame, formula: str, truth: dict[str, float]) -> None:
    """Give each coefficient the value the same fit gives on the true process's results without noise.

    That is the true value of what the coefficient estimates: the term's own coefficient
    when the design separates it from the others, the signed sum over its chain when the
    design cannot, and the intercept together with any square a two-level design cannot
    tell apart from it. A term the process has but the model leaves out shifts the
    estimates it is correlated with, and the true values show by how much.
    """
    mean = TrueProcess(truth, sigma=1.0).mean(coded)
    with _listening():
        noiseless = analyze_experiment(coded, pd.Series(mean, name="y"), model=formula, analysis_type="coefficients")
    # Rounded to drop the rounding error of a fit with no noise (a true zero reads 1e-14).
    expected = {c["term"]: round(c["coefficient"], 9) + 0.0 for c in noiseless["coefficients"]}
    for c in coefficients:
        c["true_value"] = expected.get(c["term"])


def read_workbook(request: dict) -> dict:
    """Read an uploaded workbook into the runs, responses and sources the page holds."""
    runs, stored = _read_workbook(base64.b64decode(request["xlsx_base64"]))
    spec = _checked_spec(stored)
    response = spec["response"]
    if response not in runs.columns:
        raise ValueError(f"The Runs sheet has no {response!r} column for the results.")
    values = pd.to_numeric(runs[response], errors="coerce")
    column = _marks_column(runs, spec)
    marks = runs[column].fillna("").astype(str) if column else pd.Series("", index=runs.index)
    sheet = runs.drop(columns=[column] if column else [])
    sheet[response] = np.nan
    return {
        "spec": spec,
        "columns": list(sheet.columns),
        "rows": [list(row) for row in sheet.itertuples(index=False)],
        "responses": [None if pd.isna(v) else float(v) for v in values],
        "sources": [
            None if pd.isna(v) else ("simulated" if mark == "simulated" else "entered")
            for v, mark in zip(values, marks, strict=True)
        ],
    }


def _marks_column(runs: pd.DataFrame, spec: dict) -> str | None:
    """Name the run sheet's column of marks, if it has one: an older workbook's ``Source``, unless that is data."""
    data = {spec["response"], *(f["name"] for f in spec["factors"])}
    return next((c for c in (MARKS, "Source") if c in runs.columns and c not in data), None)


def workbook(request: dict) -> dict:
    """Write the page's runs and responses to a workbook the page can analyse again.

    Simulated responses are marked, in the ``MARKS`` column and in the stored
    specification, so the workbook says which values were measured.
    """
    spec = _checked_spec(request.get("spec") or {})
    runs = _runs_frame(request, spec)
    sources = request.get("sources")
    if sources and any(sources):
        if len(sources) != len(runs):
            raise ValueError(f"{len(sources)} sources were sent for {len(runs)} runs.")
        runs[MARKS] = ["simulated" if s == "simulated" else "" for s in sources]
    simulated = request.get("simulated")
    if simulated:
        spec = {**spec, "simulated": {k: simulated.get(k) for k in ("scenario", "sigma", "seed")}}
    return {"xlsx_base64": _to_workbook(runs, spec)}


def _contour_pair(coefficients: list[dict], names: list[str]) -> list[str]:
    """Pick the two factors with the largest main effects, for the response surface."""
    size = {c["term"]: abs(c["coefficient"] or 0.0) for c in coefficients}
    return sorted(names, key=lambda n: -size.get(n, 0.0))[:2]


def plots(request: dict) -> dict:
    """Draw the analysis of the page's runs with the library's plots, as Plotly figures.

    The response surface of the fitted model over the two factors with the largest
    main effects, the others held at their centre; the true surface beside it when the
    results were simulated; the residuals against the fitted values; and, for a
    saturated two-level design, the half-normal plot of the effects with Lenth's
    margins. Each figure is JSON text, for Plotly.js at the version ``plotly_js`` names.
    """
    from plotly.offline import get_plotlyjs_version  # noqa: PLC0415 - plotly is installed for the first chart
    from plotly.utils import PlotlyJSONEncoder  # noqa: PLC0415

    from process_improve.experiments.visualization.plots import create_plot  # noqa: PLC0415

    spec = _checked_spec(request.get("spec") or {})
    model = request.get("model") or spec["model"]
    if model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    runs = _runs_frame(request, spec)
    names = [f["name"] for f in spec["factors"]]
    response = spec["response"]
    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna().to_numpy()
    coded = _coded(runs, spec["factors"])[done].reset_index(drop=True)
    y = y[done].reset_index(drop=True).rename(response)
    with _listening():
        fit = _fit_estimable(coded, y, model, names)
        out = analyze_experiment(coded, y, model=fit["formula"], analysis_type=["coefficients", "residual_diagnostics"])
    rows = coded.assign(**{response: y}).to_dict("records")
    pair = _contour_pair(out["coefficients"], names)

    def draw(kind: str, results: dict, **options: Any) -> str:  # noqa: ANN401
        figure = create_plot(kind, analysis_results=results, design_data=rows, response_column=response, **options)
        return json.dumps(figure.to_plotly(), cls=PlotlyJSONEncoder)

    figures = {"contour": draw("contour", out, factors_to_plot=pair)}
    truth = request.get("truth")
    if truth:
        process = TrueProcess({t["term"]: t["coefficient"] for t in truth}, sigma=1.0)
        missing = [f for f in process.factors if f not in names]
        if missing:
            raise ValueError(f"The true process names factors the design does not have: {missing}.")
        figures["true_contour"] = draw("contour", {"coefficients": _truth(process)}, factors_to_plot=pair)
    if fit["out"]["model_summary"]["df_residual"] > 0:
        figures["residuals"] = draw("residuals_vs_fitted", out)
    if fit["lenth"]:
        effects = analyze_experiment(coded, y, model=fit["formula"], analysis_type="effects")["effects"]
        figures["half_normal"] = draw("half_normal", {"effects": effects, "lenth_method": fit["lenth"]})
    return {"figures": figures, "contour_factors": pair, "plotly_js": get_plotlyjs_version()}


# --------------------------------------------------------------------------- comparing designs
#: The options each design is compared with: the page's own defaults, so a design in the
#: comparison is the one step 1 would generate. The CCD is face-centred, to keep its runs
#: inside the ranges the other designs are scored over.
COMPARE_OPTIONS: dict[str, dict[str, Any]] = {
    "full_factorial": {"n_center_points": 3},
    "fractional_factorial": {"n_center_points": 3},
    "plackett_burman": {"n_center_points": 3},
    "dsd": {"n_center_points": 3},
    "box_behnken": {"n_center_points": 3},
    "ccd": {"n_center_points": 3, "alpha": "face_centered"},
    "d_optimal": {},
    "i_optimal": {},
}

#: The model an optimal design is built for, in a comparison on each model: the smallest
#: model the optimal designs take that contains it.
OPTIMAL_MODEL = {
    "main_effects": "main_effects",
    "interactions": "interactions",
    "main_quadratic": "quadratic",
    "quadratic": "quadratic",
}

#: Each design's name as a comparison's column heading, short as the book writes them.
COMPARE_LABELS = {
    "full_factorial": "Full factorial",
    "fractional_factorial": "Fractional factorial",
    "plackett_burman": "Plackett-Burman",
    "dsd": "DSD",
    "box_behnken": "Box-Behnken",
    "ccd": "CCD",
    "d_optimal": "D-optimal",
    "i_optimal": "I-optimal",
}

#: Points the FDS curves are computed from, and points along each curve.
FDS_SAMPLES = 4096
FDS_POINTS = 101


def compare_plan(request: dict) -> dict:
    """List the designs a comparison for these factors includes, each as ``make_design`` takes it.

    Every design the factor count allows is included, with the page's default options;
    the optimal designs are built for the comparison's model (see :data:`OPTIMAL_MODEL`).
    """
    model = request.get("model") or "main_quadratic"
    if model not in COMPARISON_MODELS:
        raise ValueError(f"Model must be one of {', '.join(COMPARISON_MODELS)}.")
    factors = _factors(request)
    k = len(factors)
    plan = []
    for design_type, info in DESIGNS.items():
        if not info["min_factors"] <= k <= info["max_factors"]:
            continue
        options = dict(COMPARE_OPTIONS[design_type])
        if "model_type" in info["options"]:
            options["model_type"] = OPTIMAL_MODEL[model]
        spec = {"design_type": design_type, "factors": request["factors"], "response": "y", **options}
        plan.append({"design_type": design_type, "label": COMPARE_LABELS[design_type], "spec": spec})
    notes = []
    if model != OPTIMAL_MODEL[model] and any("model_type" in DESIGNS[d["design_type"]]["options"] for d in plan):
        notes.append(
            f"The D- and I-optimal designs are built for the {OPTIMAL_MODEL[model]} model, the smallest model "
            f"they can be built for that contains this one."
        )
    return {"model": model, "designs": plan, "notes": notes}


def _comparison_formula(model: str, names: list[str]) -> str:
    """Write the comparison's model as ``evaluate_design`` takes it."""
    if model == "main_quadratic":
        return " + ".join([*names, *(f"I({n} ** 2)" for n in names)])
    return model


def compare(request: dict) -> dict:
    """Score the generated designs for one model on the nine measures, with their FDS curves.

    Each design arrives as the page holds it (its stored specification, columns and rows)
    and is coded from its own ranges. The FDS curve is each design's prediction variance,
    in units of the noise variance, sorted over points drawn uniformly from the cube.
    """
    from process_improve.experiments import evaluate_design  # noqa: PLC0415

    model = request.get("model") or "main_quadratic"
    if model not in COMPARISON_MODELS:
        raise ValueError(f"Model must be one of {', '.join(COMPARISON_MODELS)}.")
    designs = {}
    for item in request.get("designs") or []:
        spec = _checked_spec(item.get("spec") or {})
        designs[str(item["label"])] = _coded(_runs_frame(item, spec), spec["factors"])
    with _listening():
        comparison = compare_designs(designs, model)
    result = comparison.to_dict()
    table = comparison.table
    result["display"] = {label: table[label].tolist() for label in comparison.designs}
    names = list(next(iter(designs.values())).columns)
    formula = _comparison_formula(model, names)
    fds = {}
    for coded, sized in zip(designs.values(), comparison.designs, strict=True):
        if not comparison.values[sized]["fits"]:
            continue
        with _listening():
            curve = evaluate_design(
                coded,
                model=formula,
                metric="fds",
                n_samples=FDS_SAMPLES,
                fds_resolution=FDS_POINTS,
                random_state=1,
            )["fds"]["curve"]
        fds[sized] = {"fraction": curve["fraction"], "variance": curve["prediction_variance"]}
    result["fds"] = fds
    return result


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


def api_simulate(payload: str) -> str:
    """JSON wrapper around :func:`simulate`."""
    return _reply(simulate, payload)


def api_analyze_runs(payload: str) -> str:
    """JSON wrapper around :func:`analyze_runs`."""
    return _reply(analyze_runs, payload)


def api_workbook(payload: str) -> str:
    """JSON wrapper around :func:`workbook`."""
    return _reply(workbook, payload)


def api_plots(payload: str) -> str:
    """JSON wrapper around :func:`plots`."""
    return _reply(plots, payload)


def api_read_workbook(payload: str) -> str:
    """JSON wrapper around :func:`read_workbook`."""
    return _reply(read_workbook, payload)


def api_compare_plan(payload: str) -> str:
    """JSON wrapper around :func:`compare_plan`."""
    return _reply(compare_plan, payload)


def api_compare(payload: str) -> str:
    """JSON wrapper around :func:`compare`."""
    return _reply(compare, payload)
