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

import ast
import base64
import contextlib
import io
import itertools
import json
import logging
import math
import re
import unicodedata
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from process_improve.experiments import (
    DesignRegion,
    Factor,
    NotSeparableError,
    TrueProcess,
    analyze_experiment,
    compare_designs,
    constraint_through_cuts,
    generate_design,
    rename_factors,
    separating_constraint,
)
from process_improve.experiments._orthogonal_arrays import orthogonal_array
from process_improve.experiments.compare import MODELS as COMPARISON_MODELS
from process_improve.experiments.design_measures import estimable_models
from process_improve.experiments.designs_constrained import MAX_EXPRESSION_LENGTH, parse_constraint

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
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 7,
    },
    "fractional_factorial": {
        "label": "Fractional factorial (2-level)",
        "hint": "A fraction of the full factorial; its resolution says what is aliased with what.",
        "model": "main_effects",
        "options": ["n_center_points", "resolution", "n_replicates", "n_blocks"],
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
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 11,
    },
    "taguchi": {
        "label": "Taguchi orthogonal array",
        "hint": (
            "Main effects in a standard array of 4 to 36 runs, and at three levels each numeric factor's square; "
            "interactions are aliased with them. A categorical factor stays at its two levels."
        ),
        "model": "main_effects",
        "options": ["n_levels", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 12,
    },
    "dsd": {
        "label": "Definitive screening (DSD)",
        "hint": "Three levels in 2k+1 runs (2k+3 for odd k); main effects clear of interactions and quadratics.",
        "model": "main_effects",
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 3,
        "max_factors": 12,
    },
    "box_behnken": {
        "label": "Box-Behnken",
        "hint": "Response surface with every run inside the factor ranges and none at the corners.",
        "model": "quadratic",
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 3,
        "max_factors": 7,
    },
    "ccd": {
        "label": "Central composite (CCD)",
        "hint": "Response surface: factorial corners, axial points and center points.",
        "model": "quadratic",
        "options": ["n_center_points", "alpha", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 7,
    },
    "d_optimal": {
        "label": "D-optimal",
        "hint": "Best estimates of the chosen model's coefficients for a fixed number of runs.",
        "model": "quadratic",
        "options": ["budget", "model_type", "n_replicates", "n_blocks"],
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
        "options": ["budget", "model_type", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 6,
    },
}

#: The design the page holds but does not make: runs taken as they are given, such as a design a book
#: prints, opened from the design catalog. It is not in :data:`DESIGNS`, so the comparison never tries
#: it; the design list (index.html) holds its option hidden, and shows it once a run sheet of given
#: runs is open, so the reader can make those runs again after changing the brief.
GIVEN = "given"
GIVEN_INFO: dict[str, Any] = {
    "label": "Runs as given",
    "hint": "The runs of a design you have already, such as one a book prints: the page takes them as they are.",
    "min_factors": 2,
    "max_factors": 12,
}

#: The most runs a design of given runs may have: as many as a study the page holds (campaign.js).
MAX_GIVEN_RUNS = 2000

#: How far a given run may sit past a limit on the settings, in the limit's own units, and still meet it.
LIMIT_TOLERANCE = 1e-9

#: What a design of given runs says of itself. It names no factor, so a renaming keeps it.
GIVEN_NOTE = "These runs are taken as they were given, in that order: the page made none of them."

#: Designs whose non-centre runs all lie on one sphere (Box-Behnken at any size, a
#: rotatable CCD at 2 or 4 factors), so without a center run the squared terms sum
#: to a constant and the quadratic model cannot be fitted.
NEEDS_CENTER_RUN = {"box_behnken": None, "ccd": "rotatable"}

#: Runs an optimal design gets beyond its model's coefficients, when no budget is
#: given, so the fit keeps degrees of freedom to estimate the error.
SPARE_RUNS = 4

#: The most times the page makes each run of a design, center runs excepted.
MAX_REPLICATES = 10

#: The most blocks the page splits a design's runs into.
MAX_BLOCKS = 8

#: The designs that take a categorical factor, one whose two levels are labels (supplier A and B):
#: the two-level designs, which need only the two levels of each factor, and the Taguchi arrays,
#: which give a categorical factor a two-level column beside the numeric factors' columns.
CATEGORICAL_DESIGNS = ("full_factorial", "fractional_factorial", "plackett_burman", "taguchi")

#: The levels a Taguchi array can run each numeric factor at: low and high, or low, centre and high.
#: The array is the smallest standard one with a column for every factor (``dispatch_taguchi``).
TAGUCHI_LEVELS = (2, 3)

#: The model a Taguchi array at three levels is made for: the main effects and each numeric factor's
#: square, Taguchi's additive model at three levels. The comparison names it so too; the page fits it as
#: the list of its terms, and the reader does not choose it by name.
MAIN_QUADRATIC = "main_quadratic"

#: What the analysis of a Taguchi array says once, unless the model fitted holds interactions: what the array
#: estimates, the main effects or the main effects and squares, and that the interactions are aliased with them.
TAGUCHI_NOTE = (
    "An orthogonal array estimates the {estimates}; the interactions are aliased with them, in full or in part, "
    "so a large effect may belong to an interaction."
)

#: The most limits a study can have on its settings. The optimal designs take the corners of the region the
#: limits leave as candidate runs, and with 8 limits in 8 factors those are still quick to find in the browser.
MAX_CONSTRAINTS = 8

#: The designs that keep every run inside limits on the settings: they choose their runs among candidates inside.
REGION_DESIGNS = ("d_optimal", "i_optimal")

#: The most characters of a limit's note, which the page shows beside it and never reads.
NOTE_LENGTH = 80

#: The functions a limit can use, as the library reads them.
LIMIT_FUNCTIONS = ("abs", "sqrt", "exp", "log", "log10")

#: The most numeric factors whose corners the page lists for marking: 2**8 = 256 settings, and the centre.
PROBE_FACTORS = 8

#: The share of a factor's range the limits must take off one end before the page says the range is narrower.
NARROWED = 0.005

#: Text a spreadsheet reads back as a blank cell (pandas' defaults), so no level's label can be it.
BLANK_LABELS = frozenset(
    {"#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA"}
    | {"NULL", "NaN", "None", "n/a", "nan", "null"}
)

#: The run sheet's column naming each run's block, as ``generate_design`` names it. No factor may
#: take the name, which the library refuses, and no response either.
BLOCK = "Block"

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

#: A coefficient within this fraction of the largest result is round-off, not an estimate: an
#: orthogonal design whose results make a term exactly zero estimates it as 1e-16, and the page
#: would show that. A least squares fit on coded factors is far more accurate than this.
ROUND_OFF = 1e-12

#: From this many factors a CCD uses a resolution V half fraction of the cube, the
#: switch ``process_improve.experiments.strategy.budget`` assumes.
CCD_HALF_FRACTION_FROM = 6

ROMAN = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}

#: The only models an uploaded workbook may request. Mapped by name, never
#: parsed from the file, so a workbook cannot smuggle a patsy formula in.
MODELS = ("main_effects", "interactions", "quadratic")

#: What the reader wants of the response, as the library's ``ResponseGoal`` names it, or ``learn``:
#: to see how the factors act on it, with no setting to aim for.
GOALS = ("maximize", "minimize", "target", "learn")

#: A model of terms the reader chose, each a list of factor names (see :func:`_chosen_terms`),
#: is named this in a reply. It too is read as data and never as a formula.
CHOSEN = "chosen"

#: The most factors in a chosen interaction.
MAX_ORDER = 3

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

#: The worked example, which "Explore a worked example" shows and "Run it again in your browser" runs live. Its
#: response is the reactor's yield, near 62 % with a run-to-run noise of 1 %: the reactor process's own
#: intercept and noise, so the results are those the process gives in coded units, and a change of 2 % is a
#: coefficient of one noise standard deviation, the size a comparison scores power at by default.
DEMO: dict[str, Any] = {
    "design": {
        "design_type": "box_behnken",
        "factors": [
            {"name": "T", "low": 150, "high": 200, "units": "degC"},
            {"name": "P", "low": 1, "high": 3, "units": "bar"},
            {"name": "F", "low": 10, "high": 20, "units": "L/min"},
        ],
        "response": "Yield",
        "response_info": {
            "label": "Yield",
            "units": "%",
            "goal": "maximize",
            "typical": 62.0,
            "noise": 1.0,
            "smallest_change": 2.0,
        },
        "aim": "main_quadratic",
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
        labels = _label(f.get("low")), _label(f.get("high"))
        if None not in labels:
            factors.append(_categorical_factor(name, *labels))  # type: ignore[arg-type]
            continue
        low, high = _number(f.get("low")), _number(f.get("high"))
        if low is None or high is None:
            raise ValueError(
                f"Factor {name}: enter a number for both the low and the high value, or a label for both, as for "
                "a supplier A and B."
            )
        if not low < high:
            raise ValueError(f"Factor {name}: the low value must be below the high value.")
        factors.append(Factor(name=name, low=low, high=high, units=str(f.get("units", ""))))
    return factors


def _label(value: object) -> str | None:
    """Return a level typed as a label, such as ``old``; None for a number, a blank or anything else."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        float(value)  # a number, or inf and nan, which are no labels either
    except ValueError:
        return value.strip()
    return None


def _categorical_factor(name: str, low: str, high: str) -> Factor:
    """Make a categorical factor from its two labels: the first is coded -1, the second +1."""
    if low == high:
        raise ValueError(f"Factor {name}: its two labels must differ.")
    if blank := [label for label in (low, high) if label in BLANK_LABELS]:
        raise ValueError(f"Factor {name}: a spreadsheet reads {blank[0]!r} back as a blank cell; choose another label.")
    return Factor(name=name, type="categorical", levels=[low, high])


def _factor_spec(f: Factor) -> dict[str, Any]:
    """Write a factor as the page holds it: a categorical factor's labels stand for its low and high values."""
    if f.type == "categorical":
        return {"name": f.name, "low": f.levels[0], "high": f.levels[1], "units": ""}  # type: ignore[index]
    return {"name": f.name, "low": f.low, "high": f.high, "units": f.units}


def _categorical(factor: dict) -> bool:
    """Say whether a factor the page holds is categorical: its levels are labels, not numbers."""
    return isinstance(factor["low"], str)


def _categorical_names(spec: dict) -> tuple[str, ...]:
    """Name the categorical factors of a checked specification."""
    return tuple(f["name"] for f in spec["factors"] if _categorical(f))


# --------------------------------------------------------------------------- limits on the settings
#: What a reader may type or paste in a limit for an operator, and the operator a limit is written with: the
#: signs for at most, at least and not equal, the minus sign and the en dash, the multiplication sign and the
#: middle dot, the division sign, and ^ for a power, as a spreadsheet writes it.
_SYMBOLS = str.maketrans(
    {
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u2212": "-",
        "\u2013": "-",
        "\u00d7": "*",
        "\u00b7": "*",
        "\u00f7": "/",
        "^": "**",
    }
)

#: A number written against a name, as in 2T or 2 T for two times T.
_UNWRITTEN_PRODUCT = re.compile(r"(?<![\w.])(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*([A-Za-z_]\w*)")

#: An equals sign on its own, or a comparison written backwards, as =< or =>.
_EQUALS = re.compile(r"=<|=>|(?<![<>=!])=(?!=)")

#: How each comparison a limit cannot make is written.
_NOT_A_LIMIT = {ast.Eq: "==", ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}


def _limit_items(raw: object) -> list[tuple[str, str]]:
    """Read the limits the page sent as ``(text, note)`` pairs, refusing a list this page did not write.

    A note is the reader's own words beside a limit: its spaces are joined into one, as on one line.
    """
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(i, dict) and isinstance(i.get("expression"), str) for i in raw):
        raise ValueError("The page sent limits this page did not write.")
    if len(raw) > MAX_CONSTRAINTS:
        raise ValueError(f"A study can have at most {MAX_CONSTRAINTS} limits on its settings.")
    items = []
    for item in raw:
        note = item.get("note") or ""
        note = " ".join(note.split()) if isinstance(note, str) else None
        if note is None or len(note) > NOTE_LENGTH or any(unicodedata.category(ch) == "Cc" for ch in note):
            raise ValueError(f"A limit's note is at most {NOTE_LENGTH} characters, on one line.")
        items.append((item["expression"].strip().translate(_SYMBOLS), note))
    return items


def _read_limit(text: str, factors: list[Factor]) -> tuple[str, list[str]]:
    """Return a limit's canonical text and the factors it names, in the factors' order; refuse it saying what to change.

    The text is parsed and checked as the library parses it, and never evaluated. The canonical
    text is the library's (``ast.unparse``), so a limit written in two ways is stored one way.
    """
    numeric = [f.name for f in factors if f.type != "categorical"]
    if not text:
        raise ValueError(f"Write each limit as an inequality of your factors, such as {_example(numeric)}.")
    _short_enough(text)
    tree, hint = _parsed(text)
    if hint:
        raise ValueError(f"The limit {text!r} cannot be read: {hint}")
    _one_comparison(text, tree, numeric)
    named = _factors_named(text, tree, factors)
    try:
        parse_constraint(text, set(numeric))
    except ValueError as exc:
        found = re.fullmatch(r"Unsupported syntax in constraint: (.+)\.", str(exc), re.DOTALL)
        raise ValueError(
            f"The limit {text!r} cannot use {found[1] if found else exc}: a limit is written with numbers, your "
            f"factors, + - * / ** and the functions {_listed(list(LIMIT_FUNCTIONS))}."
        ) from exc
    canonical = ast.unparse(tree)
    _short_enough(canonical)
    return canonical, [name for name in numeric if name in named]


def _one_comparison(text: str, tree: ast.expr, numeric: list[str]) -> None:
    """Refuse a limit that is not one comparison with <, <=, > or >=, a chain such as ``a <= b <= c`` included."""
    if not isinstance(tree, ast.Compare):
        if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or):
            raise ValueError(  # noqa: TRY004 - a refusal, like the rest
                f"The limit {text!r} joins comparisons with or, which a limit cannot: each limit must hold."
            )
        parts = tree.values if isinstance(tree, ast.BoolOp) else tree.elts if isinstance(tree, ast.Tuple) else []
        if len(parts) > 1 and all(isinstance(part, ast.Compare) for part in parts):
            raise ValueError(f"The limit {text!r} holds {len(parts)} comparisons: type each as a limit of its own.")
        raise ValueError(  # noqa: TRY004 - a refusal, like the rest
            f"The limit {text!r} is not an inequality: write it as one, such as {_example(numeric)}."
        )
    if wrong := [_NOT_A_LIMIT[type(op)] for op in tree.ops if type(op) in _NOT_A_LIMIT]:
        raise ValueError(f"The limit {text!r} must compare with <, <=, > or >=, not with {wrong[0]}.")


def _factors_named(text: str, tree: ast.expr, factors: list[Factor]) -> list[str]:
    """Return the names a limit uses for factors, refusing a function, a factor or a name it cannot use."""
    names = [f.name for f in factors]
    numeric = [f.name for f in factors if f.type != "categorical"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in LIMIT_FUNCTIONS):
            raise ValueError(
                f"The limit {text!r} uses {ast.unparse(node.func)}, which a limit cannot: the functions it can use "
                f"are {_listed(list(LIMIT_FUNCTIONS))}."
            )
    named = _named(tree)
    for name in named:
        if name in names and name not in numeric:
            raise ValueError(
                f"The limit {text!r} names {name}, which is categorical: limits are on numeric factors only."
            )
        if name not in numeric:
            raise ValueError(f"The limit {text!r} names {name}, which is not one of your factors: {_listed(names)}.")
    if not named:
        raise ValueError(f"The limit {text!r} names none of your factors.")
    return named


def _short_enough(text: str) -> None:
    """Refuse a limit longer than the library reads: as typed, and as the page writes it out."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"The limit {text[:24] + '...'!r} is longer than {MAX_EXPRESSION_LENGTH} characters.")


def _parsed(text: str) -> tuple[ast.expr | None, str | None]:
    """Parse a limit's text: return its expression, or None and what to change for Python to read it."""
    try:
        tree = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError) as err:  # ValueError: a null byte, before Python 3.12
        return None, _misread(text, err)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Constant):  # 2(T + P), for 2 times (T + P)
            number, inner = ast.unparse(node.func), ", ".join(ast.unparse(a) for a in node.args)
            return None, f"write {number}*({inner}) for {number} times ({inner})."
    if isinstance(tree, ast.Tuple) and re.search(r"\d,\d", text):
        return None, "write a decimal number with a point, as in 0.5, not 0,5."
    return tree, None


def _misread(text: str, error: Exception) -> str:
    """Say what to change in a limit Python cannot read, for the slips a reader makes most."""
    if found := _UNWRITTEN_PRODUCT.search(text):
        return f"write {found[1]}*{found[2]} for {found[1]} times {found[2]}."
    if _EQUALS.search(text):
        return "write <= for at most, and >= for at least."
    message = str(getattr(error, "msg", error))
    return (
        "check it for a missing number, factor or bracket." if message.startswith("invalid syntax") else f"{message}."
    )


def _named(tree: ast.AST) -> list[str]:
    """Return the names a parsed limit uses for factors, every name but a function's, in the order they are written."""
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and id(node) not in called]
    return list(dict.fromkeys(node.id for node in sorted(found, key=lambda node: node.col_offset)))


def _example(numeric: list[str]) -> str:
    """Write an example limit in the reader's own factors, as a refusal shows it."""
    if len(numeric) == 1:
        return f"2*{numeric[0]} <= 10"
    first, second = numeric[:2] if numeric else ("A", "B")
    return f"2*{first} + {second} <= 10"


def _checked_constraints(raw: object, factors: list[Factor]) -> list[dict[str, Any]]:
    """Check the limits a specification carries, and return them as a study stores them.

    Each limit is its canonical text (see :func:`_read_limit`), its ``type``, which is worked out
    here and never read, and the reader's ``note`` when there is one, which is shown and never
    read either. A specification without limits stores none, so it stays as it was before the
    page had them.
    """
    limits = [(_read_limit(text, factors)[0], note) for text, note in _limit_items(raw)]
    curved = set(_region(factors, [text for text, _ in limits]).curved_constraints()) if limits else set()
    return [
        {"expression": text, "type": "nonlinear" if i in curved else "linear", **({"note": note} if note else {})}
        for i, (text, note) in enumerate(limits)
    ]


def _region(factors: list[Factor], expressions: list[str]) -> DesignRegion:
    """Return the region of settings that meets every limit, over the numeric factors."""
    return DesignRegion([f for f in factors if f.type != "categorical"], expressions)


# --------------------------------------------------------------------------- public API
def _release() -> str | None:
    """Name the process-improve release the page runs, or None when it is not installed as a package."""
    try:
        return version("process-improve")
    except PackageNotFoundError:
        return None


def catalogue(_: dict) -> dict:
    """List what the page offers: design types, models, CCD axial choices, processes, and the worked example.

    ``version`` is the process-improve release the page runs, which a notebook it writes installs.
    """
    scenarios = {key: {"label": s["label"], "hint": s["hint"], "sigma": s["sigma"]} for key, s in SCENARIOS.items()}
    return {
        "designs": DESIGNS,
        "models": list(MODELS),
        "alphas": list(ALPHAS),
        "scenarios": scenarios,
        "demo": DEMO,
        "version": _release(),
    }


def make_design(spec: dict) -> dict:
    """Generate a design and return its runs plus a ready-to-download workbook.

    The workbook lists the runs in randomised order, in the factors' own units,
    with an empty response column for the reader to fill in.
    """
    design_type = spec.get("design_type", "")
    if design_type == GIVEN:
        return _given_design(spec)
    if design_type not in DESIGNS:
        raise ValueError(f"Unknown design type {design_type!r}.")
    factors = _factors(spec)
    _check_factor_count(design_type, len(factors))
    limits = _checked_constraints(spec.get("constraints"), factors)
    expressions = [limit["expression"] for limit in limits]
    region = _region_refusal(factors, _region(factors, expressions), design_type) if limits else None
    if (labelled := [f.name for f in factors if f.type == "categorical"]) and design_type not in CATEGORICAL_DESIGNS:
        verb = "is" if len(labelled) == 1 else "are"
        raise ValueError(
            f"The {DESIGNS[design_type]['label']} design needs numeric factors, and {_listed(labelled)} {verb} "
            "categorical: choose a full factorial, fractional factorial, Plackett-Burman or Taguchi design."
        )
    response = _response_name(str(spec.get("response") or "y").strip(), factors)
    described = _described(spec)

    kwargs = _design_options(spec, design_type)
    made, asked = _library_arguments(factors, kwargs)
    # The limits are passed apart from the options, which hold one design's settings, as the comparison reuses them.
    result = generate_design(made, design_type=design_type, **asked, **({"constraints": expressions} if limits else {}))

    # The library returns the runs already randomised, and ``run_order[i]`` is the
    # standard-order position of its i-th run: the sheet keeps the rows as they come.
    names = [f.name for f in factors]
    runs = pd.DataFrame(result.design_actual[names].to_numpy(), columns=names)
    if BLOCK in result.design_actual.columns:  # the runs are listed block by block
        runs.insert(0, BLOCK, result.design_actual[BLOCK].to_numpy())
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", result.run_order)
    runs[response] = np.nan

    stored = {
        "version": SPEC_VERSION,
        "design_type": design_type,
        "factors": [_factor_spec(f) for f in factors],
        "response": response,
        "model": kwargs.get("model_type") or _designed_model(design_type, result.resolution, made),
        "options": kwargs,
        **described,
        **({"constraints": limits} if limits else {}),
    }
    coded = _coded(runs, stored["factors"])
    facts = _design_facts(result, coded.to_numpy(dtype=float), names, kwargs)
    if array := (result.metadata or {}).get("orthogonal_array"):
        facts["notes"].insert(0, _array_sentence(array, stored["model"]))
    if region is not None:
        if not region.feasible(coded[region.names].to_numpy(dtype=float)).all():
            raise RuntimeError("The design has a run outside the limits on the settings, which it must not have.")
        facts["notes"].append(INSIDE_LIMITS)
    return {
        "n_runs": len(runs),
        "columns": list(runs.columns),
        "rows": [list(row) for row in runs.itertuples(index=False)],
        "outside_range": _outside_range(runs, factors),
        "model": stored["model"],
        "facts": facts,
        "spec": stored,
        "xlsx_base64": _to_workbook(runs, stored),
    }


#: What a design made with limits on the settings says of its runs. It names no factor, so a renaming keeps it.
INSIDE_LIMITS = "Every run is inside your limits on the settings."


def _given_design(spec: dict) -> dict:
    """Take a design's runs as they are given, in the factors' own units and in the order given.

    ``runs`` holds one list per run, a setting for each factor in the order of ``factors``: a number,
    or one of a categorical factor's two labels. The run sheet keeps the runs in that order. The
    design's model is the largest of :data:`MODELS` its runs estimate in full. With limits on the
    settings, the runs stay as they are, and the design's notes say which of them the limits rule out.
    """
    factors = _factors(spec)
    low, high = GIVEN_INFO["min_factors"], GIVEN_INFO["max_factors"]
    if not low <= len(factors) <= high:
        raise ValueError(f"Runs as given take {low} to {high} factors on this page; you have {len(factors)}.")
    names = [f.name for f in factors]
    rows = spec.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The runs as given are not there: choose a design to make, or open the runs again.")
    if len(rows) > MAX_GIVEN_RUNS:
        raise ValueError(f"Runs as given hold at most {MAX_GIVEN_RUNS} runs; these are {len(rows)}.")
    if bad := next((i for i, row in enumerate(rows, 1) if not isinstance(row, list) or len(row) != len(names)), None):
        raise ValueError(f"Run {bad} does not give one setting for each of the {len(names)} factors.")
    runs = pd.DataFrame(rows, columns=names, dtype=object)
    for f in factors:
        if f.type == "categorical":
            continue
        values = [_number(v) for v in runs[f.name]]
        if None in values:
            raise ValueError(f"Factor {f.name}: each run's setting must be a number.")
        runs[f.name] = values
    response = _response_name(str(spec.get("response") or "y").strip(), factors)
    limits = _checked_constraints(spec.get("constraints"), factors)
    stored: dict[str, Any] = {
        "version": SPEC_VERSION,
        "design_type": GIVEN,
        "factors": [_factor_spec(f) for f in factors],
        "response": response,
        "options": {},
        **_described(spec),
        **({"constraints": limits} if limits else {}),
    }
    coded = _coded(runs, stored["factors"]).to_numpy(dtype=float)
    kinds = ["categorical" if f.type == "categorical" else "numeric" for f in factors]
    fits = estimable_models(coded, kinds)
    stored["model"] = next((m for m in reversed(MODELS) if fits.get(m)), MODELS[0])
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", range(1, len(runs) + 1))
    runs[response] = np.nan
    groups = _aliased_groups(coded, names)
    notes = [GIVEN_NOTE, _aliasing_sentence(groups)]
    if limits:
        # The limits alone: a given run may lie beyond a factor's range, as an axial run does, and still meet them.
        region = _region(factors, [limit["expression"] for limit in limits])
        points = coded[:, [names.index(name) for name in region.names]]
        meets = np.all([g(points) <= LIMIT_TOLERANCE for g in region.inequalities], axis=0)
        outside = np.flatnonzero(~meets).tolist()
        notes.append(_outside_limits_sentence(outside) if outside else INSIDE_LIMITS)
    facts = {
        "resolution": None,
        "generators": [],
        "defining_relation": [],
        "aliases": groups,
        "center_runs": int(np.all(np.abs(coded) < ALIAS_TOLERANCE, axis=1).sum()),
        "notes": notes,
    }
    return {
        "n_runs": len(runs),
        "columns": list(runs.columns),
        "rows": [list(row) for row in runs.itertuples(index=False)],
        "outside_range": _outside_range(runs, factors),
        "model": stored["model"],
        "facts": facts,
        "spec": stored,
        "xlsx_base64": _to_workbook(runs, stored),
    }


def _outside_limits_sentence(outside: list[int]) -> str:
    """Say which given runs the limits on the settings rule out, by their place in the run sheet."""
    shown = [str(i + 1) for i in outside]
    which = f"Run {shown[0]} is" if len(shown) == 1 else f"Runs {_listed(shown)} are"
    return (
        f"{which} outside your limits on the settings: choose the D-optimal or I-optimal design to make runs "
        "inside them."
    )


def _region_refusal(factors: list[Factor], region: DesignRegion, design_type: str | None = None) -> DesignRegion:
    """Refuse limits on the settings that no design on this page can keep to, and return the region they leave.

    The optimal designs are the only ones that place their runs inside limits, and on this page
    they take numeric factors only; so the refusals come in that order, which names the real
    conflict: a full factorial and a D-optimal design would otherwise each send the reader to
    the other. ``design_type`` is the design asked for, or None when none is asked for yet.
    """
    if labelled := [f.name for f in factors if f.type == "categorical"]:
        verb, which = ("is", "that factor") if len(labelled) == 1 else ("are", "those factors")
        raise ValueError(
            f"Limits on the settings work with numeric factors only on this page, and {_listed(labelled)} {verb} "
            f"categorical: remove the limits, or {which}."
        )
    if design_type is not None and design_type not in REGION_DESIGNS:
        raise ValueError(
            f"The {DESIGNS[design_type]['label']} design cannot keep its runs inside limits on the settings: choose "
            "the D-optimal or I-optimal design, or remove the limits."
        )
    most = max(DESIGNS[d]["max_factors"] for d in REGION_DESIGNS)
    if len(factors) > most:
        raise ValueError(
            f"With limits on the settings, the page makes designs in up to {most} factors; you have {len(factors)}."
        )
    if region.is_empty():
        raise ValueError("No setting in your ranges meets every limit: loosen a limit, or widen a range.")
    return region


def _response_name(response: str, factors: list[Factor]) -> str:
    """Check the response's name: a variable name, unlike every factor's and the run sheet's blocks column."""
    if not response.isidentifier() or response in {f.name for f in factors}:
        raise ValueError("The response name must be a variable name different from every factor.")
    if response == BLOCK:
        raise ValueError(f"The response cannot be named {BLOCK}: a blocked design's run sheet has a {BLOCK} column.")
    return response


def _response_info(raw: object) -> dict[str, Any]:
    """Clean what the page says about the response besides its column's name.

    The reader's own words for it (``label``) and its ``units``; the ``goal``, one of
    :data:`GOALS`, with its ``target`` when the goal is a value; and what the reader knows of
    it: its ``typical`` value now, its run-to-run ``noise`` (a standard deviation) and the
    ``smallest_change`` worth finding, all in its units. Every field is optional and an
    empty one is left out; a goal this page does not know is left out too, so a later page
    can add one. A field of the wrong kind is refused, as any other part of a specification.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("The page sent a description of the response this page did not write.")  # noqa: TRY004
    info: dict[str, Any] = {}
    for key, most in (("label", 64), ("units", 32)):
        if text := _described_text(raw.get(key), most):
            info[key] = text
    if raw.get("goal") in GOALS:
        info["goal"] = raw["goal"]
    numbers = (
        ("target", "target", False),
        ("typical", "typical value", False),
        ("noise", "run-to-run noise", True),
        ("smallest_change", "smallest change worth finding", True),
    )
    for key, what, positive in numbers:
        if (number := _described_number(raw.get(key), what, positive=positive)) is not None:
            info[key] = number
    return info


def _described_text(text: object, most: int) -> str:
    """Return a part of the response's description in words, stripped; refuse anything else."""
    if text is None:
        return ""
    if not isinstance(text, str) or len(text) > most:
        raise ValueError("The page sent a description of the response this page did not write.")
    return text.strip()


def _described_number(value: object, what: str, *, positive: bool) -> float | None:
    """Return a number of the response's description, None when not given; refuse one that is no number."""
    if value is None:
        return None
    number = _number(value)
    if positive and (number is None or number <= 0):
        raise ValueError(f"The {what} must be a number above 0.")
    if number is None:
        raise ValueError(f"The response's {what} must be a number.")
    return number


def _described(spec: dict) -> dict[str, Any]:
    """Return the specification's description of the response and the model it aims to learn, cleaned.

    Both are optional: a design made before the page asked for them has neither, and keeps
    working. ``aim`` is one of the comparison's models; one this page does not know is left out.
    """
    out: dict[str, Any] = {}
    if info := _response_info(spec.get("response_info")):
        out["response_info"] = info
    if spec.get("aim") in COMPARISON_MODELS:
        out["aim"] = spec["aim"]
    return out


def _check_factor_count(design_type: str, n_factors: int) -> None:
    """Refuse a design the factor count rules out, saying which counts it takes."""
    info = DESIGNS[design_type]
    low, high = info["min_factors"], info["max_factors"]
    if not low <= n_factors <= high:
        raise ValueError(
            f"The {info['label']} design takes {low} to {high} factors on this page; you have {n_factors}."
        )


def _designed_model(design_type: str, resolution: int | None, factors: list[Factor]) -> str:
    """Name the model a design is analysed with when the reader does not choose one.

    A fraction keeps two-factor interactions clear of main effects and of each other
    only from resolution V, so below that it is analysed for its main effects. A Taguchi
    array estimates the main effects, and with a numeric factor at three levels (``factors``
    as the library made the design) each numeric factor's square too.
    """
    if design_type == "fractional_factorial" and (resolution or 0) >= INTERACTIONS_CLEAR_FROM:
        return "interactions"
    if design_type == "taguchi" and any(f.type != "categorical" and f.levels for f in factors):
        return MAIN_QUADRATIC
    return DESIGNS[design_type]["model"]


def _library_arguments(factors: list[Factor], kwargs: dict[str, Any]) -> tuple[list[Factor], dict[str, Any]]:
    """Return the factors and keyword arguments ``generate_design`` takes for a design's page options.

    The options are what the specification stores. Each one is a keyword argument of the
    library but ``n_levels``, a Taguchi array's levels: at three, each numeric factor is given
    its low, centre and high values as its ``levels``, which the library reads as a
    three-level factor, and a categorical factor keeps its two labels.
    """
    asked = {key: value for key, value in kwargs.items() if key != "n_levels"}
    if kwargs.get("n_levels", TAGUCHI_LEVELS[0]) == TAGUCHI_LEVELS[0]:  # the low and the high, as every design here
        return factors, asked
    made = []
    for f in factors:
        if f.type == "categorical":
            made.append(f)
            continue
        low, high = float(f.low), float(f.high)  # type: ignore[arg-type]
        made.append(Factor(name=f.name, low=low, high=high, units=f.units, levels=[low, (low + high) / 2, high]))
    return made, asked


def _array_sentence(name: str, model: str) -> str:
    """Say which orthogonal array a Taguchi design is, its runs, and what it estimates: ``Taguchi L12 (2^11): ...``.

    The array is written from its columns' level counts, as Taguchi writes it: the library keeps
    names that do not always describe the array (its L27 is all three-level), and its six-level
    column holds a two-level and a three-level column, as Taguchi writes the L18 as ``2^1 3^7``.
    """
    array = orthogonal_array(name)
    counts: dict[int, int] = {}
    for column in range(array.shape[1]):
        n = int(array[:, column].max()) + 1
        for part in (2, 3) if n == 6 else (n,):  # a six-level column is a 2-level and a 3-level one
            counts[part] = counts.get(part, 0) + 1
    columns = " ".join(f"{n}^{c}" for n, c in sorted(counts.items()))
    estimates = "main effects and squares" if model == MAIN_QUADRATIC else "main effects"
    return f"Taguchi L{len(array)} ({columns}): {len(array)} runs, {estimates}."


def _design_options(spec: dict, design_type: str) -> dict[str, Any]:
    """Pick the ``generate_design`` keyword arguments this design type accepts."""
    kwargs: dict[str, Any] = {"random_state": int(spec.get("random_state", spec.get("random_seed", 42)))}
    allowed = DESIGNS[design_type]["options"]
    n_factors = len(spec.get("factors") or [])
    alpha = _alpha(spec) if "alpha" in allowed else None
    if "n_center_points" in allowed:
        kwargs["n_center_points"] = _center_points(spec, design_type, alpha, n_factors)
    if alpha is not None:
        kwargs["alpha"] = alpha
    if design_type == "ccd" and n_factors >= CCD_HALF_FRACTION_FROM:
        kwargs["cube"] = "fractional"
    if "resolution" in allowed:
        kwargs["resolution"] = _resolution(spec, n_factors)
    if "n_levels" in allowed and (levels := _levels(spec)) != TAGUCHI_LEVELS[0]:
        kwargs["n_levels"] = levels
    if "model_type" in allowed:
        model_type = spec.get("model_type", "quadratic")
        if model_type not in MODELS:
            raise ValueError(f"Unknown model {model_type!r}.")
        kwargs["model_type"] = model_type
        budget = spec.get("budget")
        kwargs["budget"] = int(budget) if budget else _n_terms(n_factors, model_type) + SPARE_RUNS
    if "n_replicates" in allowed and (replicates := _replicates(spec)) > 1:
        kwargs["n_replicates"] = replicates
    if "n_blocks" in allowed and (blocks := _blocks(spec)) > 1:
        kwargs["n_blocks"] = blocks
    return kwargs


def _alpha(spec: dict) -> str | float:
    """Read a CCD's axial distance: one of :data:`ALPHAS`, or a distance in coded units, such as 1.4."""
    alpha = spec.get("alpha", "face_centered")
    if alpha in ALPHAS:
        return str(alpha)
    if alpha is None:  # a blank box
        raise ValueError("Enter the axial distance in coded units: a number above 0, such as 1.4.")
    distance = None if isinstance(alpha, bool) else _number(alpha)
    if distance is None:
        raise ValueError(f"Unknown axial distance {alpha!r}.")
    if distance <= 0:
        raise ValueError(f"The axial distance must be above 0 in coded units, not {distance:g}.")
    return distance


def _replicates(spec: dict) -> int:
    """Read how many times each run is made, center runs excepted: 1 (no repeats) to :data:`MAX_REPLICATES`."""
    replicates = _number(spec.get("n_replicates", 1))
    if replicates is None or replicates != int(replicates) or not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"Replicates must be a whole number from 1 to {MAX_REPLICATES}.")
    return int(replicates)


def _levels(spec: dict) -> int:
    """Read the levels a Taguchi array runs each numeric factor at: one of :data:`TAGUCHI_LEVELS`."""
    levels = _number(spec.get("n_levels", TAGUCHI_LEVELS[0]))
    if levels not in TAGUCHI_LEVELS:
        raise ValueError("A Taguchi array runs each numeric factor at 2 levels or at 3 on this page.")
    return int(levels)  # type: ignore[arg-type]


def _blocks(spec: dict) -> int:
    """Read how many blocks the runs are made in: 1 (no blocks) to :data:`MAX_BLOCKS`."""
    blocks = _number(spec.get("n_blocks", 1))
    if blocks is None or blocks != int(blocks) or not 1 <= blocks <= MAX_BLOCKS:
        raise ValueError(f"Blocks must be a whole number from 1 to {MAX_BLOCKS}.")
    return int(blocks)


def _center_points(spec: dict, design_type: str, alpha: str | float | None, n_factors: int) -> int:
    """Read the number of center runs, refusing none where the quadratic model would be singular."""
    n_center = int(spec.get("n_center_points", 3))
    if n_center < 0:
        raise ValueError("The number of center points cannot be negative.")
    if n_center == 0 and _needs_center_run(design_type, alpha, n_factors):
        raise ValueError(
            "This design needs at least one center point: without one every other run lies on one sphere "
            "around the center, so the squared terms cannot be told apart from the intercept."
        )
    return n_center


def _needs_center_run(design_type: str, alpha: str | float | None, n_factors: int) -> bool:
    """Say whether a design's runs other than the center runs all lie on one sphere around the center.

    A Box-Behnken design's always do; a CCD's do when its axial distance is the cube corners'
    distance from the center, the square root of the number of factors, as the rotatable
    distance is at two or four factors (the page asks for a center run at any rotatable CCD).
    """
    if design_type not in NEEDS_CENTER_RUN:
        return False
    named = NEEDS_CENTER_RUN[design_type]
    if named is None or alpha == named:
        return True
    return isinstance(alpha, float) and math.isclose(alpha**2, n_factors)


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


def _design_facts(result: Any, coded: np.ndarray, names: list[str], kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """State what the generated design is, read off the design rather than its type.

    A fraction's resolution and generators come from the library; which main effects and
    two-factor interactions are fully aliased is read from the coded runs, so it holds for
    every design type and every naming of the factors. ``notes`` are the sentences the
    page shows.
    """
    groups = _aliased_groups(coded, names)
    notes = []
    if result.resolution:
        notes.append(_resolution_sentence(result.resolution, [_spaced(g, names) for g in result.generators or []]))
    notes.append(_aliasing_sentence(groups))
    replicates = kwargs.get("n_replicates", 1)
    if replicates > 1:
        notes.append(
            f"Each run other than the center runs appears {replicates} times, so the repeats estimate the pure error."
        )
    if BLOCK in result.design_actual.columns:
        sizes = result.design_actual[BLOCK].value_counts().sort_index().tolist()
        blocking = (result.metadata or {}).get("blocking", {})
        words = [_factor_word(w, names) for w in blocking.get("confounded_with") or []]
        # A word confounded with the blocks takes its aliases with it (in a 2^(4-1), C·D goes with A·B).
        confounded = list(dict.fromkeys(t for w in words for t in next((g for g in groups if w in g), [w])))
        notes.append(_blocking_sentence(sizes, confounded))
    asked = kwargs.get("budget")
    if asked and len(coded) > asked * replicates:
        model = kwargs["model_type"]
        made = len(coded) // replicates
        notes.append(
            f"You asked for {asked} runs; the {model.replace('_', ' ')} model in {len(names)} factors has "
            f"{_n_terms(len(names), model)} coefficients, so the design has {made} runs"
            f"{' before it is replicated' if replicates > 1 else ''}."
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


def _resolution_sentence(resolution: int, generators: list[str]) -> str:
    """Say a fraction's resolution and the generators that make it, each written with its factors apart."""
    label = "generator" if len(generators) == 1 else "generators"
    return f"Resolution {ROMAN.get(resolution, resolution)}, {label} {', '.join(generators)}."


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
    """Write a generator or defining word with its factors apart: ``D=ABC`` as ``D = A·B·C``."""
    left, _, right = (side.strip() for side in word.partition("="))
    return f"{left if left == 'I' else _factor_word(left, names)} = {_factor_word(right, names)}"


def _factor_word(part: str, names: list[str]) -> str:
    """Write a word of factors with the factors apart: ``ABC`` as ``A·B·C``, and ``-AB`` as ``-A·B``.

    The library writes the factors of a word side by side, which is ambiguous to read
    when the names are longer than a letter. A word that does not split into the factor
    names is left as the library wrote it.
    """
    pattern = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
    sign, body = ("-", part[1:]) if part.startswith("-") else ("", part.lstrip("+"))
    found = pattern.findall(body)
    return sign + "·".join(found) if found and "".join(found) == body else part


def _listed(items: list[str]) -> str:
    """Join items as a sentence does: ``A``, ``A and B``, ``A, B and C``."""
    return f"{', '.join(items[:-1])} and {items[-1]}" if len(items) > 1 else "".join(items)


def _blocking_sentence(sizes: list[int], confounded: list[str]) -> str:
    """Say how the runs are split into blocks, and which effects the split confounds with them."""
    of = f"{sizes[0]} runs each" if len(set(sizes)) == 1 else f"{_listed([str(n) for n in sizes])} runs"
    text = (
        f"The runs are made in {len(sizes)} blocks of {of}, one block after another, as the {BLOCK} column says; "
        "the analysis fits the blocks, so a shift between them does not inflate the error."
    )
    if confounded:
        verb, pronoun = ("is", "it") if len(confounded) == 1 else ("are", "them")
        text += f" {_listed(confounded)} {verb} confounded with the blocks, so the analysis leaves {pronoun} out."
    return text


def _outside_range(runs: pd.DataFrame, factors: list[Factor]) -> list[str]:
    """Factors with runs beyond the ranges entered (CCD axial points can be); a category has no range."""
    numeric = [f for f in factors if f.type != "categorical"]
    return [f.name for f in numeric if runs[f.name].min() < f.low - 1e-9 or runs[f.name].max() > f.high + 1e-9]


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
    runs, stored = _read_workbook(base64.b64decode(request["xlsx_base64"]))
    spec = _checked_spec(stored)  # a workbook can arrive from anywhere: its specification is checked as data
    return _analysis(runs, spec, request.get("model") or spec.get("analysis_model") or spec["model"])


def _analysis(runs: pd.DataFrame, spec: dict, model: str, truth: list[dict] | None = None) -> dict:
    """Fit the model to the completed runs of a run sheet, from a workbook or from the page.

    With the true process the results were simulated from, each coefficient also
    carries the true value of what it estimates (see :func:`_add_true_values`).
    """
    response = spec["response"]
    factors = spec["factors"]
    names = [f["name"] for f in factors]
    categorical = _categorical_names(spec)
    name = _model_name(model)
    model = _checked_model(model, names, categorical)
    blocked = _blocked(spec)
    missing_cols = [c for c in [*names, *([BLOCK] if blocked else []), response] if c not in runs.columns]
    if missing_cols:
        raise ValueError(f"The Runs sheet is missing column(s): {', '.join(missing_cols)}.")

    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna()
    pending = runs.loc[~done, "Run order"].astype(int).tolist() if "Run order" in runs else []

    coded = _design_frame(runs, spec)[done].reset_index(drop=True)
    n_terms = _n_terms(len(names), model)
    if pending and done.sum() < n_terms:
        complete = int(done.sum())
        raise ValueError(
            f"The {name} model has {n_terms} coefficients, so it needs at least {n_terms} "
            f"completed runs, and {complete} {'is' if complete == 1 else 'are'} complete. "
            f"Still to run: {', '.join(map(str, pending))}."
        )

    fit = _fit_estimable(coded, y[done].reset_index(drop=True).rename(response), model, names, categorical)
    out = fit["out"]
    zero = ROUND_OFF * float(y[done].abs().max())
    coefficients = [
        {
            **c,
            "coefficient": _snapped(c["coefficient"], zero),
            "label": fit["labels"].get(c["term"], _pretty(c["term"])),
            "aliases": fit["chains"].get(c["term"], []),
            "block": _block_contrast(c["term"], names),
        }
        for c in out.get("coefficients", [])
    ]
    lenth = fit["lenth"]
    if lenth:  # an effect is twice its coefficient at most, so twice the coefficients' round-off
        lenth["effects"] = [{**e, "effect": _snapped(e["effect"], 2 * zero)} for e in lenth["effects"]]
    if truth:
        _add_true_values(coefficients, coded, fit["formula"], _true_coefficients(truth, names))
    anova = [{**a, "source": fit["labels"].get(a["source"], _pretty(a["source"]))} for a in out.get("anova_table", [])]
    notes = fit["notes"]
    terms = _model_terms(model, names, categorical)
    if spec["design_type"] == "taguchi" and not any(":" in term for term in terms):
        # A model with interactions has the alias chains' note instead.
        estimates = "main effects and squares" if any(term.startswith("I(") for term in terms) else "main effects"
        notes = [TAGUCHI_NOTE.format(estimates=estimates), *notes]
    return {
        "spec": spec,
        "model": model if isinstance(model, str) else CHOSEN,
        "formula": fit["formula"],  # what was fitted: the model, less what the design cannot estimate
        "n_completed": int(done.sum()),
        "pending_runs": pending,
        "summary": out["model_summary"],
        "coefficients": coefficients,
        "anova": anova,
        "lack_of_fit": out.get("lack_of_fit"),
        "lenth": lenth,
        "not_estimable": [_pretty(t) for t in fit["not_estimable"]],
        "notes": notes,
        "equation": _equation(_response_words(spec)[0], [c for c in coefficients if not c["block"]]),
    }


def _snapped(value: float | None, zero: float) -> float | None:
    """Return 0 for a value no larger than ``zero``, the fit's round-off, and the value otherwise."""
    return 0.0 if value is not None and abs(value) <= zero else value


#: What the page asks of every fit; the effects carry the alias chains of a design
#: that cannot tell some terms apart.
ANALYSES = ["anova", "coefficients", "lack_of_fit", "effects"]


def _fit_estimable(
    coded: pd.DataFrame, y: pd.Series, model: str | list[str], names: list[str], categorical: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Fit the model, estimating each group of terms the design cannot tell apart once.

    A pseudo-inverse fit of exactly aliased terms spreads their joint estimate evenly
    over them, so ``T`` aliased with ``P:F`` would read half of what the data say about
    their sum. The library reports those groups as alias chains and the terms that move
    with the intercept as confounded with the mean; the model is then refitted with one
    term per chain, the chain's leader, labelled with the whole chain. A design that
    still cannot fit the model is refused. With no runs to spare for the error, the
    effects of an orthogonal two-level design are judged with Lenth's method.

    The runs of a blocked design carry their ``Block``, and the blocks are fitted as a
    fixed effect. A term the design confounds with them is left out, with a note: its
    effect cannot be told apart from the differences between blocks.
    """
    blocked = BLOCK in coded.columns and len(set(coded[BLOCK])) > 1
    formula, early = _first_formula(coded, model, names, categorical, blocked=blocked)
    with _listening() as heard:
        out = analyze_experiment(coded, y, model=formula, analysis_type=ANALYSES)
    chains: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    not_estimable: list[str] = []
    if out["model_summary"]["rank_deficient"]:
        # Keyed by the chain's leader, the term kept for it; the library's label keeps each member's sign.
        for label, members in (out.get("alias_chains") or {}).items():
            chains[members[0]] = members
            labels[members[0]] = _pretty(label)
        not_estimable = list(out.get("confounded_with_mean") or [])
        dropped = set(not_estimable) | {m for members in chains.values() for m in members[1:]}
        kept = [term for term in _fitted_terms(out, names) if term not in dropped]
        if dropped and kept:
            formula = _formula(kept, names, blocked=blocked)
            out = analyze_experiment(coded, y, model=formula, analysis_type=ANALYSES)
    else:
        _relay(heard)
    summary = out["model_summary"]
    if summary["rank_deficient"]:
        raise ValueError(
            f"This design cannot fit the {_model_name(model)} model: only {summary['model_rank']} of its "
            f"{summary['n_terms']} coefficients can be estimated from these {summary['n_obs']} runs. Choose a "
            "smaller model."
        )
    notes = [*early, *_analysis_notes(list(labels.values()), not_estimable)]
    if blocked:
        notes.append(
            "The blocks are fitted as a fixed effect, so a shift between blocks is not counted as error: each "
            "Block coefficient is how far that block lies from the average block, and the equation is for the "
            "average block."
        )
    labels.update(_block_labels(out, names))
    lenth = None
    if summary["df_residual"] == 0:
        lenth, said = _saturated(coded, y, formula, _fitted_terms(out, names), labels, names)
        notes += said
    return {
        "out": out,
        "formula": formula,
        "chains": chains,
        "labels": labels,
        "not_estimable": not_estimable,
        "lenth": lenth,
        "notes": notes,
    }


def _first_formula(
    coded: pd.DataFrame, model: str | list[str], names: list[str], categorical: tuple[str, ...], *, blocked: bool
) -> tuple[str, list[str]]:
    """Return the model to fit first, and a note for each term it leaves out before fitting.

    An interaction that is zero in every run cannot be estimated at all, and in a blocked
    design one that takes one value in each block is confounded with the blocks; both are
    left out. A named model the design can fit as it is goes to the library by name, which
    fits any blocks itself; the quadratic model has no square of a categorical factor.
    """
    terms = _model_terms(model, names, categorical)
    early: list[str] = []
    if inert := [t for t in terms if not t.startswith("I(") and np.allclose(coded[t.split(":")].prod(axis=1), 0)]:
        early.append(_inert_sentence(inert))
        terms = [t for t in terms if t not in inert]
    if blocked and (confounded := _confounded_with_blocks(coded, terms, categorical)):
        early.append(_confounded_sentence(confounded))
        terms = [t for t in terms if t not in confounded]
    if not terms and not blocked:
        raise ValueError("None of the model's terms can be estimated from this design.")
    whole = isinstance(model, str) and terms == _model_terms(model, names)
    return (model if whole else _formula(terms, names, blocked=blocked)), early


def _saturated(  # noqa: PLR0913 - the fit, and how the page names its terms
    coded: pd.DataFrame, y: pd.Series, formula: str, terms: list[str], labels: dict[str, str], names: list[str]
) -> tuple[dict | None, list[str]]:
    """Judge the effects of a fit with no runs to spare: by Lenth's method if the design allows, else say why not."""
    if not _orthogonal_two_level(coded[names], terms):
        return None, [
            (
                "The model uses every run, so no degrees of freedom are left to estimate the error and nothing can "
                "be tested. Add runs, or fit fewer terms."
            )
        ]
    lenth = analyze_experiment(coded, y, model=formula, analysis_type="lenth_method")["lenth_method"]
    lenth["effects"] = [{**e, "label": labels.get(e["term"], _pretty(e["term"]))} for e in lenth["effects"]]
    notes = [
        (
            "The model uses every run, so no degrees of freedom are left to estimate the error. Each effect, "
            "twice its coefficient (the change from the factor's low level to its high level), is judged with "
            f"Lenth's method instead: it is likely real beyond the margin of error ME = "
            f"{lenth['ME']:.4g}, and still so, allowing for the number of effects tested, beyond SME = "
            f"{lenth['SME']:.4g}."
        )
    ]
    if len(lenth["effects"]) < LENTH_FEW_EFFECTS:
        notes.append(
            f"With only {len(lenth['effects'])} effects to compare, these margins are wide. Center points "
            "or more runs leave degrees of freedom for an ordinary test."
        )
    return lenth, notes


def _checked_model(model: object, names: list[str], categorical: tuple[str, ...] = ()) -> str | list[str]:
    """Read the model to fit: the name of one in :data:`MODELS`, or ``{"terms": [...]}``, the terms a reader chose.

    A design can also be made for :data:`MAIN_QUADRATIC`, which the reader does not choose by name: it
    is fitted as the list of its terms, the main effects and each numeric factor's square.
    """
    if isinstance(model, dict) and set(model) == {"terms"}:
        return _chosen_terms(model["terms"], names, categorical)
    if model == MAIN_QUADRATIC:
        return _model_terms(MAIN_QUADRATIC, names, categorical)
    if model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    return str(model)


def _chosen_terms(chosen: object, names: list[str], categorical: tuple[str, ...] = ()) -> list[str]:
    """Read the terms a reader chose, and name them as ``analyze_experiment`` does.

    Each term is a list of factor names: one name is the factor's main effect, two or three
    different names their interaction, and one name twice its square. A model of chosen terms
    reaches the library as a formula, which is evaluated as Python, so the formula is written
    here from these names, each checked against the design's factors, and never read in.
    """
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("Choose at least one term for the model.")
    terms = []
    for term in chosen:
        if not isinstance(term, list) or not term or not all(isinstance(n, str) and n in names for n in term):
            raise ValueError(f"A model term can name only the design's factors, {_listed(names)}, not {term!r}.")
        if len(term) == 2 and term[0] == term[1]:  # a factor named twice is its square
            if term[0] in categorical:
                raise ValueError(f"{term[0]} is categorical, so it has no square: it is +1 at both its levels.")
            terms.append(f"I({term[0]} ** 2)")
        elif len(term) <= MAX_ORDER and len(set(term)) == len(term):
            terms.append(":".join(sorted(term, key=names.index)))
        else:
            raise ValueError(
                "A model term is a factor, an interaction of two or three different factors, or a factor squared."
            )
    return list(dict.fromkeys(terms))


def _model_name(model: object) -> str:
    """Name a model in a sentence: ``main effects``, ``quadratic``, or ``chosen`` for terms a reader chose."""
    if model == MAIN_QUADRATIC:
        return "main effects and squares"
    return model.replace("_", " ") if isinstance(model, str) else CHOSEN


def _model_terms(model: str | list[str], names: list[str], categorical: tuple[str, ...] = ()) -> list[str]:
    """List a model's terms as ``analyze_experiment`` names them: a named model's, or the ones chosen.

    A categorical factor has no square: it is +1 at both its levels, so the quadratic model leaves it out.
    """
    if not isinstance(model, str):
        return list(model)
    terms = list(names)
    if model in ("interactions", "quadratic"):
        terms += [":".join(pair) for pair in itertools.combinations(names, 2)]
    if model in ("quadratic", MAIN_QUADRATIC):
        terms += [f"I({name} ** 2)" for name in names if name not in categorical]
    return terms


def _fitted_terms(out: dict[str, Any], names: list[str]) -> list[str]:
    """List the model terms a fit estimated: its coefficients, less the intercept and any block contrasts."""
    terms = [c["term"] for c in out["coefficients"] if c["term"] != "Intercept"]
    return [term for term in terms if not _block_contrast(term, names)]


def _formula(terms: list[str], names: list[str], *, blocked: bool) -> str:
    """Write a model of these terms as a formula, each checked first; a blocked design's blocks come first."""
    return " + ".join([*([BLOCK] if blocked else []), *(_checked_term(term, names) for term in terms)])


def _block_contrast(term: str, names: list[str]) -> bool:
    """Say whether a fitted term is one of the library's block contrasts: ``Block1``, ``Block2``, ..."""
    return term not in names and re.fullmatch(rf"{BLOCK}\d+_*", term) is not None


def _block_labels(out: dict[str, Any], names: list[str]) -> dict[str, str]:
    """Name each block contrast as the page shows it: ``Block1`` as ``Block 1``."""
    contrasts = [c["term"] for c in out["coefficients"] if _block_contrast(c["term"], names)]
    return {term: f"{BLOCK} {term[len(BLOCK) :].rstrip('_')}" for term in contrasts}


def _confounded_with_blocks(coded: pd.DataFrame, terms: list[str], categorical: tuple[str, ...] = ()) -> list[str]:
    """List the main effects and interactions that a blocked design confounds with its blocks.

    A blocked two-level factorial puts each block's runs at one level of the interactions it
    confounds, so their effects cannot be told apart from the differences between blocks. Such
    a term takes one value within every block, away from the center runs, which a block may
    hold at zero. A square is not confounded this way; what it shares with the intercept, the
    fit reports.
    """
    numeric = [c for c in coded.columns if c != BLOCK and c not in categorical]  # a center run's category is +-1
    away = coded[(coded[numeric].abs() > ALIAS_TOLERANCE).any(axis=1)] if numeric else coded
    confounded = []
    for term in terms:
        if term.startswith("I("):
            continue
        column = away[term.split(":")].prod(axis=1)
        spread = column.groupby(away[BLOCK].to_numpy()).agg(lambda v: v.max() - v.min())
        if spread.max() < ALIAS_TOLERANCE < column.max() - column.min():
            confounded.append(term)
    return confounded


def _inert_sentence(terms: list[str]) -> str:
    """Say which interactions are zero in every run, so the design cannot estimate them."""
    if len(terms) == 1:
        return (
            f"{_pretty(terms[0])} cannot be estimated: it is zero in every run, since no run sets all its factors "
            "away from their centers at once, and the model leaves it out."
        )
    return (
        f"{_listed([_pretty(t) for t in terms])} cannot be estimated: each is zero in every run, since no run "
        "sets all its factors away from their centers at once, and the model leaves them out."
    )


def _confounded_sentence(terms: list[str]) -> str:
    """Say which terms the blocks absorb, and that the model leaves them out."""
    if len(terms) == 1:
        return (
            f"{_pretty(terms[0])} is confounded with the blocks: it takes one value in each block, so its effect "
            "cannot be told apart from the differences between blocks, and the model leaves it out."
        )
    return (
        f"{_listed([_pretty(t) for t in terms])} are confounded with the blocks: each takes one value in each "
        "block, so their effects cannot be told apart from the differences between blocks, and the model leaves "
        "them out."
    )


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


def _n_terms(k: int, model: str | list[str]) -> int:
    """Count the coefficients, intercept included, in a named model or a model of chosen terms."""
    if not isinstance(model, str):
        return 1 + len(model)
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
    is checked as data: the design type and model must be ones the page offers, the
    factor and response names must be variable names, the blocks a whole number, and the
    limits on the settings ones the page reads, each stored in its canonical text.
    """
    if not isinstance(spec, dict) or spec.get("design_type") not in (*DESIGNS, GIVEN):
        raise ValueError("The page sent a design this page did not make.")
    factors = _factors(spec)
    response = _response_name(str(spec.get("response") or ""), factors)
    model = spec.get("model") or DESIGNS.get(spec["design_type"], {}).get("model", MODELS[0])
    if model not in (*MODELS, MAIN_QUADRATIC):  # the second, a model a design is made for: see _checked_model
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    options = spec.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("The page sent a design this page did not make.")  # noqa: TRY004 - a refusal, like the rest
    if "analysis_model" in spec:  # the model the reader last analysed with, which a workbook keeps
        labelled = tuple(f.name for f in factors if f.type == "categorical")
        _checked_model(spec["analysis_model"], [f.name for f in factors], labelled)
    checked = {key: value for key, value in spec.items() if key not in ("response_info", "aim", "constraints")}
    limits = _checked_constraints(spec.get("constraints"), factors)
    return {
        **checked,
        "factors": [_factor_spec(f) for f in factors],
        "response": response,
        "model": model,
        "options": {**options, **({"n_blocks": blocks} if (blocks := _blocks(options)) > 1 else {})},
        **_described(spec),
        **({"constraints": limits} if limits else {}),
    }


def _response_words(spec: dict) -> tuple[str, str]:
    """Return the response's name as the reader wrote it, else its column's, and its units, else ''."""
    info = spec.get("response_info") or {}
    return info.get("label") or spec["response"], info.get("units") or ""


def _with_units(name: str, units: str) -> str:
    """Write a quantity's name with its units, as an axis title: ``T (degC)``, or the name alone."""
    return f"{name} ({units})" if units else name


def _blocked(spec: dict) -> bool:
    """Say whether a checked specification's runs are made in blocks, so its run sheet has a Block column."""
    return spec.get("options", {}).get("n_blocks", 1) > 1


def _runs_frame(request: dict, spec: dict) -> pd.DataFrame:
    """Rebuild the run sheet the page shows, with the responses it holds in the response column."""
    columns = [str(c) for c in request.get("columns") or []]
    rows = request.get("rows") or []
    names = [f["name"] for f in spec["factors"]]
    missing = [n for n in names if n not in columns]
    if missing or not rows:
        raise ValueError(f"The run sheet is missing the factor column(s) {missing}." if missing else "No runs.")
    runs = pd.DataFrame(rows, columns=columns)
    numeric = [f["name"] for f in spec["factors"] if not _categorical(f)]  # a category's labels stay text
    runs[numeric] = runs[numeric].apply(pd.to_numeric, errors="raise")
    responses = request.get("responses")
    if responses is not None:
        if len(responses) != len(runs):
            raise ValueError(f"{len(responses)} responses were sent for {len(runs)} runs.")
        runs[spec["response"]] = pd.to_numeric(pd.Series(responses, dtype=object), errors="coerce").to_numpy()
    return runs


def _coded(runs: pd.DataFrame, factors: list[dict]) -> pd.DataFrame:
    """Code each factor's settings to -1 at its low value and +1 at its high value, a category's labels too."""
    return pd.DataFrame({f["name"]: _coded_column(runs[f["name"]], f) for f in factors})


def _coded_column(column: pd.Series, factor: dict) -> pd.Series:
    """Code one factor's settings; a categorical factor's runs must each be at one of its two labels."""
    if not _categorical(factor):
        return (column - (factor["high"] + factor["low"]) / 2) / ((factor["high"] - factor["low"]) / 2)
    codes = column.astype(str).str.strip().map({factor["low"]: -1.0, factor["high"]: 1.0})
    if codes.isna().any():
        other = column[codes.isna()].iloc[0]
        levels = f"{factor['low']} or {factor['high']}"
        raise ValueError(f"Factor {factor['name']}: each run must be at {levels}, not {other!r}.")
    return codes


def _design_frame(runs: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Return the runs as the analysis fits them: the coded settings, and each run's block if it has one."""
    frame = _coded(runs, spec["factors"])
    if _blocked(spec):
        frame[BLOCK] = runs[BLOCK].to_numpy()
    return frame


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


def _scenario_process(  # noqa: PLR0913 - the scenario, the factors, the request, and the reader's scale
    key: str,
    names: list[str],
    request: dict,
    categorical: tuple[str, ...] = (),
    typical: float | None = None,
    noise: float | None = None,
) -> TrueProcess:
    """Return the scenario's process over the reader's factors, with the noise the reader chose.

    A term over a factor the reader does not have is left out, and so is the square of a categorical
    factor, which takes one value, +1, at both of its levels. A scenario's effects are sizes in
    noise standard deviations; given the response's ``typical`` value and run-to-run ``noise``,
    the process is centred on the one and its effects scaled by the other, so practice results
    look like the reader's own. Without them the scenario's own centre and a noise of 1 are kept.
    """
    if key not in SCENARIOS:
        raise ValueError(f"Unknown process {key!r}.")
    scenario = SCENARIOS[key]
    terms = scenario["terms"]
    if terms is None:
        terms = _random_terms(len(names), np.random.default_rng(int(request.get("seed", 1))))
    kept = {t: c for t, c in terms.items() if all(i < len(names) for i in t)}
    squares = {t for t in kept if len(t) == 2 and t[0] == t[1] and names[t[0]] in categorical}
    scale = 1.0 if noise is None else noise
    coefficients = {
        _term_name(t, names): (c if typical is None else typical) if not t else c * scale
        for t, c in kept.items()
        if t not in squares
    }
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
    info = spec.get("response_info") or {}
    process = _scenario_process(
        key, names, request, _categorical_names(spec), typical=info.get("typical"), noise=info.get("noise")
    )
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
        "equation": _equation(_response_words(spec)[0], truth),
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
    specification, so the workbook says which values were measured. The model the reader
    analyses with, ``model``, is kept as the specification's ``analysis_model``, so the
    workbook is analysed the same way when it comes back.
    """
    spec = _checked_spec(request.get("spec") or {})
    if request.get("model"):
        _checked_model(request["model"], [f["name"] for f in spec["factors"]], _categorical_names(spec))
        spec = {**spec, "analysis_model": request["model"]}
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

    With limits on the settings, the surfaces are ``shaded`` where a setting cannot be run;
    when the centre of the ranges cannot be run either, the other factors are held at a
    central setting that can, which ``held`` lists in their own units.
    """
    from plotly.offline import get_plotlyjs_version  # noqa: PLC0415 - plotly is installed for the first chart
    from plotly.utils import PlotlyJSONEncoder  # noqa: PLC0415

    from process_improve.experiments.visualization.plots import create_plot  # noqa: PLC0415

    spec = _checked_spec(request.get("spec") or {})
    names = [f["name"] for f in spec["factors"]]
    categorical = _categorical_names(spec)
    model = _checked_model(request.get("model") or spec["model"], names, categorical)
    runs = _runs_frame(request, spec)
    response = spec["response"]
    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna().to_numpy()
    coded = _design_frame(runs, spec)[done].reset_index(drop=True)
    y = y[done].reset_index(drop=True).rename(response)
    with _listening():
        fit = _fit_estimable(coded, y, model, names, categorical)
        out = analyze_experiment(coded, y, model=fit["formula"], analysis_type=["coefficients", "residual_diagnostics"])
    rows = coded.assign(**{response: y}).to_dict("records")
    pair = _contour_pair(out["coefficients"], names)
    shading, held = _shading(spec, pair)

    def draw(kind: str, results: dict, **options: Any) -> str:  # noqa: ANN401
        figure = create_plot(kind, analysis_results=results, design_data=rows, response_column=response, **options)
        return json.dumps(_dressed(figure.to_plotly(), kind, spec, pair), cls=PlotlyJSONEncoder)

    figures = {"contour": draw("contour", out, factors_to_plot=pair, **shading)}
    truth = request.get("truth")
    if truth:
        process = TrueProcess({t["term"]: t["coefficient"] for t in truth}, sigma=1.0)
        missing = [f for f in process.factors if f not in names]
        if missing:
            raise ValueError(f"The true process names factors the design does not have: {missing}.")
        figures["true_contour"] = draw("contour", {"coefficients": _truth(process)}, factors_to_plot=pair, **shading)
    if fit["out"]["model_summary"]["df_residual"] > 0:
        figures["residuals"] = draw("residuals_vs_fitted", out)
    if fit["lenth"]:
        effects = analyze_experiment(coded, y, model=fit["formula"], analysis_type="effects")["effects"]
        figures["half_normal"] = draw("half_normal", {"effects": effects, "lenth_method": fit["lenth"]})
    return {
        "figures": figures,
        "contour_factors": pair,
        "shaded": bool(shading),
        "held": held,
        "plotly_js": get_plotlyjs_version(),
    }


def _shading(spec: dict, pair: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return what a response surface needs to shade the settings that cannot be run, and where it holds the others.

    Without limits on the settings, nothing. With them, the region, and the coded values the
    factors not plotted are held at: their centre when the centre of the ranges can be run, as
    without limits, else a central setting that can be (see ``DesignRegion.centroid``), which is
    listed in the factors' own units for the caption to say.
    """
    if not spec.get("constraints"):
        return {}, []
    factors = _factors(spec)
    region = _region(factors, [limit["expression"] for limit in spec["constraints"]])
    if region.feasible(np.zeros((1, len(region.names))))[0]:
        return {"region": region}, []
    centroid = region.centroid() or {}
    others = [f for f in factors if f.type != "categorical" and f.name not in pair]
    hold = {f.name: (centroid[f.name] - (f.low + f.high) / 2) / ((f.high - f.low) / 2) for f in others}
    return {"region": region, "hold_values": hold}, [{"name": f.name, "value": centroid[f.name]} for f in others]


def _ticks(factor: dict) -> dict[str, Any]:
    """Mark a coded axis at -1, 0 and +1 with the factor's own settings, or a category's two labels."""
    if _categorical(factor):
        return {"tickvals": [-1, 1], "ticktext": [factor["low"], factor["high"]]}
    low, high = factor["low"], factor["high"]
    return {"tickvals": [-1, 0, 1], "ticktext": [f"{v:.4g}" for v in (low, (low + high) / 2, high)]}


def _dressed(figure: dict, kind: str, spec: dict, pair: list[str]) -> dict:
    """Title a library figure's axes in the reader's words and units, as the page shows it.

    The library draws in coded units and names the axes for any reader: the factors by their
    names, the fitted values as "Fitted Values". Here a response surface's axes are the two
    factors with their units, ticked at their own low, middle and high settings, its colours
    are the response with its units, and the residuals and effects are in the response's units.
    """
    label, units = _response_words(spec)
    layout = figure.setdefault("layout", {})

    def title(axis: str, text: str, **more: Any) -> None:  # noqa: ANN401
        layout[axis] = {**layout.get(axis, {}), "title": {"text": text}, **more}

    if kind == "contour":
        factors = {f["name"]: f for f in spec["factors"]}
        for axis, name in zip(("xaxis", "yaxis"), pair, strict=False):
            title(axis, _with_units(name, factors[name]["units"]), **_ticks(factors[name]))
        for trace in figure.get("data", []):  # the response's contours; the shading of a region's limits has no scale
            if trace.get("type") == "contour" and trace.get("contours", {}).get("type") != "constraint":
                trace["colorbar"] = {**trace.get("colorbar", {}), "title": {"text": _with_units(label, units)}}
    elif kind == "residuals_vs_fitted":
        title("xaxis", _with_units(f"Fitted {label}", units))
        title("yaxis", _with_units("Residual", units))
    elif kind == "half_normal":
        title("yaxis", _with_units("|Effect|", units))
    return figure


# --------------------------------------------------------------------------- renaming the factors
def relabel(request: dict) -> dict:
    """Rename a run sheet's factors and their units, keeping its runs and their results.

    A factor's name and units are labels, so the runs stay as they were made. ``factors`` are
    the design's factors in the same places with the same levels; only their names and units may
    differ, and each new name must be a variable name the run sheet does not use already.
    Everything this module wrote with the old names is written again with the new: the run
    sheet's columns, the limits on the settings, the design's facts and notes, the factors
    outside their ranges and, when
    ``simulation`` is given, the process the results were simulated from, with its equation and
    the factors it leaves out. Renaming a design this way gives what making it under the new
    names would have, which ``tests/web/api/test_relabel.py`` checks for each kind of note.
    """
    design = request.get("design") or {}
    spec = _checked_spec(design.get("spec") or {})
    renamed = _factors({"factors": request.get("factors")})
    before = spec["factors"]
    after = [_factor_spec(f) for f in renamed]
    levels = [(f["low"], f["high"]) for f in before]
    if [(f["low"], f["high"]) for f in after] != levels:
        raise ValueError("Only the factors' names and units can change without making the run sheet again.")
    columns = [str(c) for c in design.get("columns") or []]
    for taken in (spec["response"], *([BLOCK] if BLOCK in columns else [])):
        if taken in {f.name for f in renamed}:
            raise ValueError(f"A factor cannot be named {taken}: the run sheet has a {taken} column.")
    names = {old["name"]: new["name"] for old, new in zip(before, after, strict=True)}
    spec = {**spec, "factors": after}
    if limits := spec.get("constraints"):  # the limits on the settings, written with the new names, all at once
        spec["constraints"] = [{**limit, "expression": rename_factors(limit["expression"], names)} for limit in limits]
    if isinstance(chosen := spec.get("analysis_model"), dict):  # the terms a workbook was last analysed with
        spec["analysis_model"] = {"terms": [[names.get(n, n) for n in term] for term in chosen["terms"]]}
    reply: dict[str, Any] = {
        "spec": spec,
        "columns": [names.get(c, c) for c in columns],
        "outside_range": [names.get(n, n) for n in design.get("outside_range") or []],
    }
    if design.get("facts"):
        coded = _coded(_runs_frame({**design, "columns": reply["columns"]}, spec), after).to_numpy(dtype=float)
        reply["facts"] = _relabelled_facts(design["facts"], coded, names)
    if simulation := request.get("simulation"):
        truth = [
            {"term": term, "label": _pretty(term), "coefficient": t["coefficient"]}
            for t in simulation.get("truth") or []
            for term in [_renamed_term(str(t["term"]), names)]
        ]
        reply["simulation"] = {
            **simulation,
            "truth": truth,
            "equation": _equation(_response_words(spec)[0], truth),
            "inactive": [names.get(n, n) for n in simulation.get("inactive") or []],
        }
    return reply


def _renamed_term(term: str, names: dict[str, str]) -> str:
    """Rename the factors of a term as the analysis names it: ``T``, ``T:P`` or ``I(T ** 2)``."""
    if square := re.fullmatch(r"I\((\w+) \*\* 2\)", term):
        return f"I({names.get(square[1], square[1])} ** 2)"
    return ":".join(names.get(part, part) for part in term.split(":"))


def _renamed_word(word: str, names: dict[str, str]) -> str:
    """Rename the factors of a word written with its factors apart: ``A·B``, or ``-A·B``."""
    sign, body = ("-", word[1:]) if word.startswith("-") else ("", word)
    return sign + "·".join(names.get(part, part) for part in body.split("·"))


def _relabelled_facts(facts: dict, coded: np.ndarray, names: dict[str, str]) -> dict[str, Any]:
    """Write a design's facts again under new factor names, from the same runs.

    The alias groups are found again in the renamed runs, so they come out as the design would
    have found them; the generators and the defining relation, written with their factors
    apart, are renamed word by word. Each note that names factors is written again from its
    parts; the others name none, and stay as they are.
    """

    def spaced(word: str) -> str:
        left, _, right = word.partition(" = ")
        return f"{left if left == 'I' else _renamed_word(left, names)} = {_renamed_word(right, names)}"

    generators = [spaced(g) for g in facts.get("generators") or []]
    groups = _aliased_groups(coded, list(names.values()))
    notes = []
    for note in facts.get("notes") or []:
        if note.startswith("Resolution ") and facts.get("resolution"):
            notes.append(_resolution_sentence(facts["resolution"], generators))
        elif note.startswith(("No main effect", "Main effects are")):
            notes.append(_aliasing_sentence(groups))
        elif note.startswith("The runs are made in "):
            notes.append(_relabelled_blocking(note, names))
        else:
            notes.append(note)
    return {
        **facts,
        "generators": generators,
        "defining_relation": [spaced(w) for w in facts.get("defining_relation") or []],
        "aliases": groups,
        "notes": notes,
    }


def _relabelled_blocking(note: str, names: dict[str, str]) -> str:
    """Rename the effects a blocking sentence says are confounded with the blocks."""
    head, said, tail = note.partition("does not inflate the error. ")
    found = re.fullmatch(r"(.+?) (is|are) (confounded with the blocks.*)", tail)
    if not said or not found:
        return note
    listed, verb, rest = found.groups()
    words = [_renamed_word(word, names) for word in re.split(r", | and ", listed)]
    return f"{head}{said}{_listed(words)} {verb} {rest}"


# --------------------------------------------------------------------------- comparing designs
#: The options each design is compared with: the page's own defaults, so a design in the
#: comparison is the one Design would make. The CCD is face-centred, to keep its runs
#: inside the ranges the other designs are scored over.
COMPARE_OPTIONS: dict[str, dict[str, Any]] = {
    "full_factorial": {"n_center_points": 3},
    "fractional_factorial": {"n_center_points": 3},
    "plackett_burman": {"n_center_points": 3},
    "taguchi": {},
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

#: The models a Taguchi array is compared at three levels for: those with squares, which two levels cannot estimate.
#: A model without squares is compared on the array at two levels, the page's default.
TAGUCHI_THREE_LEVELS = ("main_quadratic", "quadratic")

#: Each design's name as a comparison's column heading, short as the book writes them.
COMPARE_LABELS = {
    "full_factorial": "Full factorial",
    "fractional_factorial": "Fractional factorial",
    "plackett_burman": "Plackett-Burman",
    "taguchi": "Taguchi",
    "dsd": "DSD",
    "box_behnken": "Box-Behnken",
    "ccd": "CCD",
    "d_optimal": "D-optimal",
    "i_optimal": "I-optimal",
}

#: What a comparison with limits on the settings says of the designs it lists.
LIMITED_DESIGNS = "With limits on the settings, only the D-optimal and I-optimal designs keep every run inside them."

#: Points the FDS curves are computed from, and points along each curve.
FDS_SAMPLES = 4096
FDS_POINTS = 101


def compare_plan(request: dict) -> dict:
    """List the designs a comparison for these factors includes, each as ``make_design`` takes it.

    Every design the factor count allows is included, with the page's default options;
    the optimal designs are built for the comparison's model (see :data:`OPTIMAL_MODEL`), and
    a Taguchi array runs its numeric factors at three levels for a model with squares.
    The response is the reader's, so a design taken into the plan keeps it. A comparison has
    no results, so when that name is blank or a factor's, the first of y, y1, y2, ... that
    no factor takes stands in. With limits on the settings (``constraints``), only the designs
    that keep their runs inside them are listed, each made with the limits.
    """
    model = request.get("model") or "main_quadratic"
    if model not in COMPARISON_MODELS:
        raise ValueError(f"Model must be one of {', '.join(COMPARISON_MODELS)}.")
    factors = _factors(request)
    limits = _checked_constraints(request.get("constraints"), factors)
    if limits:
        _region_refusal(factors, _region(factors, [limit["expression"] for limit in limits]))
    k = len(factors)
    names = {f.name for f in factors}
    candidates = [str(request.get("response") or "").strip(), "y", *(f"y{i}" for i in range(1, k + 1))]
    response = next(name for name in candidates if name.isidentifier() and name not in names)
    plan = []
    labelled = any(f.type == "categorical" for f in factors)
    # A design taken from the comparison keeps the reader's description of the response, and aims to learn the
    # model it was compared on.
    described = _described({"response_info": request.get("response_info"), "aim": model})
    for design_type, info in DESIGNS.items():
        if not info["min_factors"] <= k <= info["max_factors"] or (labelled and design_type not in CATEGORICAL_DESIGNS):
            continue
        if limits and design_type not in REGION_DESIGNS:
            continue
        options = dict(COMPARE_OPTIONS[design_type])
        if "model_type" in info["options"]:
            options["model_type"] = OPTIMAL_MODEL[model]
        if "n_levels" in info["options"] and model in TAGUCHI_THREE_LEVELS:
            options["n_levels"] = TAGUCHI_LEVELS[-1]
        spec = {"design_type": design_type, "factors": request["factors"], "response": response, **options, **described}
        if limits:
            spec["constraints"] = limits
        plan.append({"design_type": design_type, "label": COMPARE_LABELS[design_type], "spec": spec})
    notes = [LIMITED_DESIGNS] if limits else []
    if model != OPTIMAL_MODEL[model] and any("model_type" in DESIGNS[d["design_type"]]["options"] for d in plan):
        notes.append(
            f"The D- and I-optimal designs are built for the {OPTIMAL_MODEL[model]} model, the smallest model "
            f"they can be built for that contains this one."
        )
    return {"model": model, "designs": plan, "notes": notes}


def _effect_size(request: dict) -> float:
    """Return the coefficient, in noise standard deviations, a comparison scores power at.

    The reader's smallest change worth finding is a change from a factor's low setting to its
    high one, and a two-level factor's coefficient is half that change: so with the run-to-run
    noise it is ``smallest_change / (2 * noise)``. Without both, one noise standard deviation.
    """
    info = _response_info({key: request.get(key) for key in ("noise", "smallest_change")})
    if "noise" in info and "smallest_change" in info:
        return info["smallest_change"] / (2 * info["noise"])
    return 1.0


def _comparison_formula(model: str, names: list[str]) -> str:
    """Write the comparison's model as ``evaluate_design`` takes it."""
    if model == "main_quadratic":
        return " + ".join([*names, *(f"I({n} ** 2)" for n in names)])
    return model


def compare(request: dict) -> dict:
    """Score the generated designs for one model on the nine measures, with their FDS curves.

    Each design arrives as the page holds it (its stored specification, columns and rows)
    and is coded from its own ranges. The FDS curve is each design's prediction variance,
    in units of the noise variance, sorted over points drawn uniformly from the cube; with
    limits on the settings, the first design's, from the settings that meet them, and the
    measures that average or search the prediction variance do so over those settings too.
    """
    from process_improve.experiments import evaluate_design  # noqa: PLC0415

    model = request.get("model") or "main_quadratic"
    if model not in COMPARISON_MODELS:
        raise ValueError(f"Model must be one of {', '.join(COMPARISON_MODELS)}.")
    designs = {}
    region = None
    for item in request.get("designs") or []:
        spec = _checked_spec(item.get("spec") or {})
        designs[str(item["label"])] = _coded(_runs_frame(item, spec), spec["factors"])
        if region is None and spec.get("constraints"):
            region = _region(_factors(spec), [limit["expression"] for limit in spec["constraints"]])
    with _listening():
        comparison = compare_designs(designs, model, effect_size=_effect_size(request), region=region)
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
                region=region,
            )["fds"]["curve"]
        fds[sized] = {"fraction": curve["fraction"], "variance": curve["prediction_variance"]}
    result["fds"] = fds
    return result


#: From this many factors, the optimal designs' search for a model beyond main effects takes seconds
#: to tens of seconds (an I-optimal design in six factors, about 20 s in the browser), so the designs
#: listed as the reader types leave them out until the reader asks for the search.
SEARCH_FROM = 5


def candidates(request: dict) -> dict:
    """Make and score, in one call, every design the page offers for these factors and this aim.

    The designs are :func:`compare_plan`'s, each made as :func:`make_design` makes it, with
    the library's warnings for it kept beside it, and all of them scored by :func:`compare` on
    the model the reader aims to learn, at the power for the response's smallest change worth
    finding. A design that cannot be made is left out, with a note saying why. With
    :data:`SEARCH_FROM` factors or more and a model beyond main effects, the optimal designs
    are left out unless ``search`` is true, and listed in ``waiting``; with limits on the
    settings they are the only designs, so then ``made`` can be empty and ``result`` null.
    """
    plan = compare_plan(request)
    k = len(request.get("factors") or [])
    slow = k >= SEARCH_FROM and plan["model"] != "main_effects" and not request.get("search")
    made, waiting, notes = [], [], list(plan["notes"])
    for item in plan["designs"]:
        if slow and "model_type" in DESIGNS[item["design_type"]]["options"]:
            waiting.append({"design_type": item["design_type"], "label": item["label"]})
            continue
        with _listening() as heard:
            try:
                design = make_design(item["spec"])
            except Exception as exc:  # noqa: BLE001 - one design that cannot be made leaves the others to compare
                notes.append(f"{item['label']} could not be made: {exc}")
                continue
        design.pop("xlsx_base64", None)
        made.append({**item, "design": design, "warnings": list(dict.fromkeys(heard))})
    if not made and not waiting:
        raise ValueError("None of the designs this page offers can be made for these factors.")
    info = _response_info(request.get("response_info"))
    scored = [{"label": m["label"], **{key: m["design"][key] for key in ("spec", "columns", "rows")}} for m in made]
    asked = {"model": plan["model"], "designs": scored, **{k: info.get(k) for k in ("noise", "smallest_change")}}
    result = compare(asked) if made else None
    return {"model": plan["model"], "made": made, "waiting": waiting, "notes": notes, "result": result}


# --------------------------------------------------------------------------- limits, as the reader types them
def limits(request: dict) -> dict:
    """Describe the limits on the settings the reader is typing, and propose limits from the settings marked.

    ``factors`` are the brief's and ``constraints`` its limits, each ``{expression, note}`` in the
    factors' own units. Each limit comes back in its canonical text with what it does, or with
    the sentence that says what to change, so a limit that cannot be read leaves the others
    described. What a limit does over the factors' ranges:

    - ``removed``, the share of the ranges it rules out on its own, and ``removes``, the rows of
      ``probes`` it rules out;
    - ``redundant``: ``"alone"`` when it rules out nothing within the ranges, ``"others"`` when
      the other limits rule out all it does, else null;
    - ``linear``, its straight form ``low <= lhs <= high`` for a bound box to edit, else null.

    Over every limit that can be read: ``kept``, the share of the ranges left; ``empty``;
    ``center_allowed``, whether the centre of the ranges can still be run; ``bounds``, each factor
    whose range they narrow at an end, by more than :data:`NARROWED` of it (null with a curved
    limit, where the narrowing is not exact, or with nothing left); ``refusal``, the sentence
    making a design would refuse them with, or null; and ``picture``, the region on each pair of
    numeric factors (see :func:`_picture`; ``every_pair`` asks for every pair).

    ``probes`` are the corners of the numeric factors' ranges, then their centre, with the limits
    that rule out each (see :func:`_probes`). ``marked`` lists the rows the reader marked "cannot
    run", and ``proposals`` the straight limit proposed for each group of them (see
    :func:`_proposals`); ``cuts`` asks for a proposal rebuilt through a cut on each factor, which
    comes back as ``adjusted`` (see :func:`_adjusted`); and ``slice`` asks to look inside the
    region at given values of the other factors (see :func:`_sliced`).
    """
    factors = _factors(request)
    numeric = [f for f in factors if f.type != "categorical"]
    entries = _limit_entries(request, factors)
    read = [i for i, entry in enumerate(entries) if "problem" not in entry]
    region = _region(factors, [entries[i]["expression"] for i in read])
    probes = _probes(numeric)
    alone = _describe_each(entries, read, region, factors, probes)
    marked = _marked(request.get("marked"), probes)
    table = None if probes is None else _probe_table(probes[1], numeric, entries, read)
    empty = region.is_empty()
    curved = any(entries[i]["type"] == "nonlinear" for i in read)
    reply = {
        "constraints": entries,
        "probes": table,
        "kept": region.volume_fraction(),
        "empty": empty,
        "center_allowed": not read or bool(region.feasible(np.zeros((1, len(numeric))))[0]),
        "bounds": None if curved or empty else _narrowed(region.implied_bounds(), numeric),
        "refusal": _limits_refusal(entries, read, factors, region),
        "proposals": [],
        "picture": _picture(region, factors, entries, alone, every_pair=request.get("every_pair") is True),
    }
    if marked and probes is not None and table is not None:
        reply["proposals"] = _proposals(marked, probes, numeric, [bool(by) for by in table["ruled_out_by"]])
    if "cuts" in request:
        reply["adjusted"] = _adjusted(request["cuts"], probes, numeric)
    if "slice" in request:
        reply["slice"] = _sliced(request["slice"], factors, region, entries, alone)
    return reply


def _limit_entries(request: dict, factors: list[Factor]) -> list[dict[str, Any]]:
    """Read each limit the panel holds: its canonical text and the factors it names, or its problem."""
    return [_limit_entry(text, note, factors) for text, note in _limit_items(request.get("constraints"))]


def _limit_entry(text: str, note: str, factors: list[Factor]) -> dict[str, Any]:
    """Read one limit: its canonical text and the factors it names, or the sentence that says what to change."""
    try:
        canonical, named = _read_limit(text, factors)
    except ValueError as exc:
        return {"expression": text, "note": note, "problem": str(exc)}
    return {"expression": canonical, "note": note, "factors": named}


def _describe_each(
    entries: list[dict[str, Any]],
    read: list[int],
    region: DesignRegion,
    factors: list[Factor],
    probes: tuple[np.ndarray, np.ndarray] | None,
) -> dict[int, DesignRegion]:
    """Add to each limit that can be read what it does over the factors' ranges; return each one's own region."""
    names = [f.name for f in factors if f.type != "categorical"]
    curved = {read[j] for j in region.curved_constraints()}
    covered = {read[j] for j in region.redundant_constraints()}
    alone = {}
    for i in read:
        alone[i] = own = _region(factors, [entries[i]["expression"]])
        removed = 1.0 - own.volume_fraction()
        removes = None if probes is None else np.flatnonzero(~own.feasible(probes[0])).tolist()
        # A straight limit that keeps every corner keeps the whole box, so the corners decide it exactly; a curved
        # one is judged by them and the sample, as the library judges it.
        nothing = bool(own.redundant_constraints()) if removes is None else not removes and removed == 0
        entries[i].update(
            type="nonlinear" if i in curved else "linear",
            removed=removed,
            removes=removes,
            redundant="alone" if nothing else "others" if i in covered else None,
            linear=_linear_entry(own.linear_forms()[0], names),
        )
    return alone


def _limits_refusal(entries: list[dict], read: list[int], factors: list[Factor], region: DesignRegion) -> str | None:
    """Return the sentence making a design would refuse the limits with: the first limit's problem, or the region's."""
    if problem := next((entry["problem"] for entry in entries if "problem" in entry), None):
        return problem
    if not read:
        return None
    try:
        _region_refusal(factors, region)
    except ValueError as exc:
        return str(exc)
    return None


def _probes(numeric: list[Factor]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the settings the page lists for marking, coded and in actual units: the corners, then the centre.

    The corners of the numeric factors' ranges come in ``itertools.product`` order, the first factor
    changing slowest and each factor's low end before its high end; the centre of the ranges is
    last. None without numeric factors, or with more than :data:`PROBE_FACTORS`.
    """
    k = len(numeric)
    if not 1 <= k <= PROBE_FACTORS:
        return None
    coded = np.vstack([np.array(list(itertools.product((-1.0, 1.0), repeat=k))), np.zeros((1, k))])
    low = np.array([f.low for f in numeric], dtype=float)
    high = np.array([f.high for f in numeric], dtype=float)
    return coded, np.where(coded < 0, low, np.where(coded > 0, high, (low + high) / 2))


def _probe_table(actual: np.ndarray, numeric: list[Factor], entries: list[dict], read: list[int]) -> dict[str, Any]:
    """Return the probes as the page's table of corners: each row's settings, and the limits that rule it out."""
    removes = {i: set(entries[i]["removes"]) for i in read}
    return {
        "columns": [f.name for f in numeric],
        "rows": actual.tolist(),
        "center": len(actual) - 1,
        "ruled_out_by": [[i for i in read if row in removes[i]] for row in range(len(actual))],
    }


def _linear_entry(form: tuple[dict[str, float], float | None, float | None] | None, names: list[str]) -> dict | None:
    """Return a straight limit as ``{lhs, low, high}``, ``low <= lhs <= high`` with null for an open side, or None."""
    if form is None:
        return None
    coefficients, low, high = form
    terms: list[str] = []
    for name in names:
        if not (c := coefficients.get(name, 0.0)):
            continue
        size = str(int(abs(c))) if float(c).is_integer() and abs(c) < 1e15 else repr(abs(c))
        term = name if size == "1" else f"{size} * {name}"
        terms.append((f"-{term}" if c < 0 else term) if not terms else f"{'-' if c < 0 else '+'} {term}")
    return {"lhs": ast.unparse(ast.parse(" ".join(terms), mode="eval")), "low": low, "high": high}


def _narrowed(implied: dict[str, tuple[float, float]] | None, numeric: list[Factor]) -> list[dict[str, Any]] | None:
    """List the factors whose range the limits narrow, with each new end, or null for an end left where it was."""
    if implied is None:
        return None
    narrowed = []
    for f in numeric:
        low, high = implied[f.name]
        span = f.high - f.low
        ends = (low if low - f.low > NARROWED * span else None, high if f.high - high > NARROWED * span else None)
        if ends != (None, None):
            narrowed.append({"name": f.name, "low": ends[0], "high": ends[1]})
    return narrowed


# --------------------------------------------------------------------------- limits proposed from the settings marked
#: What the page says when every corner is marked or ruled out. A straight cut that rules out every corner of the
#: box rules out all of it, since the box is the convex hull of its corners.
EVERY_CORNER = (
    "Every corner is marked or ruled out: a straight cut that rules them all out leaves no settings. Narrow the "
    "ranges instead, or type a curved limit."
)


def _marked(raw: object, probes: tuple[np.ndarray, np.ndarray] | None) -> list[int]:
    """Check the rows of the table of corners the reader marked "cannot run"; return each once, in order."""
    if raw is None:
        return []
    count = 0 if probes is None else len(probes[0])
    if not isinstance(raw, list) or not all(type(row) is int and 0 <= row < count for row in raw):
        raise ValueError("The page sent marked settings this page did not write.")
    return sorted(set(raw))


def _proposals(
    marked: list[int], probes: tuple[np.ndarray, np.ndarray], numeric: list[Factor], ruled_out: list[bool]
) -> list[dict[str, Any]]:
    """Propose one straight limit for each group of marked settings that no limit rules out yet.

    One straight cut removes corners that hang together along the edges of the box, so the
    marked corners are grouped by those edges and each group gets its own limit: the cut of
    :func:`~process_improve.experiments.separating_constraint` between the group and the
    settings neither marked nor ruled out, the other groups counting neither way. The centre,
    which no edge joins to a corner, goes with the first group that one cut can rule out
    together with it, or else is a group of its own.
    """
    coded = probes[0]
    centre = len(coded) - 1
    left = [row for row in marked if not ruled_out[row]]
    if not left:
        return []
    if all(row in marked or ruled_out[row] for row in range(centre)):
        return [{"marked": left, "problem": EVERY_CORNER}]
    runnable = [row for row in range(len(coded)) if row not in marked and not ruled_out[row]]
    proposals = []
    centre_left = centre in left
    for group in _joined([row for row in left if row != centre], coded):
        proposal = _proposal([*group, centre], runnable, probes, numeric) if centre_left else None
        if proposal is None or "problem" in proposal:
            proposal = _proposal(group, runnable, probes, numeric)
        if centre_left and centre in proposal.get("removes", ()):
            proposal["marked"], centre_left = [*group, centre], False
        proposals.append(proposal)
    if centre_left:
        proposals.append(_proposal([centre], runnable, probes, numeric))
    return proposals


def _joined(rows: list[int], coded: np.ndarray) -> list[list[int]]:
    """Group corners joined by the edges of the box, two corners being joined when they differ in one factor."""
    left, groups = sorted(rows), []
    while left:
        group, reach = [], [left.pop(0)]
        while reach:
            row = reach.pop()
            group.append(row)
            joined = [r for r in left if np.count_nonzero(coded[r] != coded[row]) == 1]
            left = [r for r in left if r not in joined]
            reach += joined
        groups.append(sorted(group))
    return groups


def _proposal(
    group: list[int], runnable: list[int], probes: tuple[np.ndarray, np.ndarray], numeric: list[Factor]
) -> dict[str, Any]:
    """Propose the straight limit that rules out the rows of ``group`` and keeps ``runnable``, or say why none can."""
    actual = probes[1]
    try:
        constraint = separating_constraint(numeric, actual[group], actual[runnable])
    except NotSeparableError as exc:
        return {"marked": group, "problem": _inseparable(exc, group, runnable, len(actual) - 1)}
    return {"marked": group, **_limit_proposed(constraint.expression, group, probes, numeric)}


def _inseparable(exc: NotSeparableError, group: list[int], runnable: list[int], centre: int) -> str:
    """Say why no straight cut rules out a group of marked settings and keeps the others.

    A corner of the box lies between no other settings of the box, so the setting to blame, when
    there is one, is the centre: marked, between corners that can be run, or runnable, between
    marked corners.
    """
    if centre in [group[i] for i in exc.inside_can]:
        return (
            "The center of the ranges lies between settings you can run, so no straight cut rules it out alone. Type "
            "a curved limit, or mark more."
        )
    if centre in [runnable[i] for i in exc.inside_cannot]:
        return (
            "The center of the ranges lies between the corners you marked, so a straight cut that rules them out "
            "rules it out too. Mark it as well, or type a limit."
        )
    return (
        "No single straight cut rules out these corners and keeps the others. Mark them in smaller groups, or type a "
        "limit."
    )


def _limit_proposed(
    expression: str, group: list[int], probes: tuple[np.ndarray, np.ndarray], numeric: list[Factor]
) -> dict[str, Any]:
    """Describe a proposed limit as a typed one is described, with where it cuts each factor from one corner.

    The corner is the marked corner deepest beyond the limit. Each factor's cut is where the
    limit's boundary crosses the edge from that corner to the factor's other end, the others held
    at the corner, so the limit is the plane through the cuts and the reader can move them one at
    a time (see :func:`_adjusted`). When the boundary misses the edge of a factor the limit
    depends on, no question about that edge gives this limit, and ``corner`` and ``cuts`` are null.
    """
    coded, actual = probes
    alone = DesignRegion(numeric, [expression])
    form = alone.linear_forms()[0]
    if form is None:  # a limit proposed is straight, with one bound
        raise RuntimeError(f"The limit proposed, {expression!r}, is not a straight limit with one bound.")
    coefficients, low, high = form
    names = [f.name for f in numeric]
    a = np.array([coefficients.get(name, 0.0) for name in names])
    bound = high if high is not None else low
    corners = [row for row in group if row != len(coded) - 1]
    depth = [abs(float(a @ actual[row]) - bound) for row in corners]
    corner = corners[int(np.argmax(depth))] if corners else None
    cuts = None if corner is None else _cuts(a, bound, actual[corner], numeric)
    return {
        "expression": expression,
        "type": "linear",
        "linear": _linear_entry(form, names),
        "removes": np.flatnonzero(~alone.feasible(coded)).tolist(),
        "removed": 1.0 - alone.volume_fraction(),
        "corner": None if cuts is None else corner,
        "cuts": cuts,
    }


def _cuts(a: np.ndarray, bound: float, corner: np.ndarray, numeric: list[Factor]) -> list[dict[str, Any]] | None:
    """Return where a limit's boundary crosses each edge of the box from a corner, or None when it misses one.

    A factor the limit does not depend on has no cut: moving it alone never reaches the boundary.
    """
    cuts = []
    for weight, at, factor in zip(a, corner, numeric, strict=True):
        if weight == 0:
            cuts.append({"name": factor.name, "at": float(at), "cut": None})
            continue
        cut = float(f"{at + (bound - float(a @ corner)) / weight:.12g}")
        far = factor.low if at == factor.high else factor.high
        if cut == at or not min(at, far) <= cut <= max(at, far):
            return None
        cuts.append({"name": factor.name, "at": float(at), "cut": cut})
    return cuts


def _adjusted(raw: object, probes: tuple[np.ndarray, np.ndarray] | None, numeric: list[Factor]) -> dict[str, Any]:
    """Rebuild a proposed limit through the cut the reader gave each factor, from the corner it was proposed at.

    ``raw`` is ``{marked, corner, values}``: the marked rows the limit is for, the row of its
    corner, and per numeric factor the value it can reach from the corner with the others held
    there, or null where the factor does not limit it. A cut no limit can be drawn through
    (outside the range, or at or beyond the corner) comes back as the problem to show.
    """
    values = raw.get("values") if isinstance(raw, dict) else None
    if probes is None or not (
        isinstance(values, list)
        and len(values) == len(numeric)
        and all(v is None or _number(v) is not None for v in values)
        and type(raw.get("corner")) is int
        and 0 <= raw["corner"] < len(probes[0]) - 1
    ):
        raise ValueError("The page sent marked settings this page did not write.")
    marked = _marked(raw.get("marked"), probes)
    corner = probes[1][raw["corner"]]
    cuts = {f.name: None if v is None else _number(v) for f, v in zip(numeric, values, strict=True)}
    try:
        constraint = constraint_through_cuts(numeric, corner.tolist(), cuts)
    except ValueError as exc:
        return {"marked": marked, "problem": str(exc)}
    return {"marked": marked, **_limit_proposed(constraint.expression, [raw["corner"]], probes, numeric)}


# --------------------------------------------------------------------------- pictures of the region
#: Up to this many numeric factors, the picture of the region shows every pair of them.
PICTURE_EVERY_PAIR = 4

#: Points along each axis of the grid a curved limit's boundary is traced on, as many as a slice's mask has.
TRACE_POINTS = 61

#: How many times the step between two points of that grid is halved to find where the boundary crosses it.
TRACE_HALVINGS = 20


def _picture(
    region: DesignRegion,
    factors: list[Factor],
    entries: list[dict[str, Any]],
    alone: dict[int, DesignRegion],
    *,
    every_pair: bool,
) -> dict[str, Any]:
    """Return the region on each pair of numeric factors, for the panels the page draws.

    Each pair's ``shadow`` is the settings of the two that some setting of the others can run
    (see :meth:`~process_improve.experiments.DesignRegion.shadow`), and ``lines`` the boundary of
    each straight limit that names no other factor, numbered by its place in the list. With
    curved limits that name no factor but the pair's, ``trace`` traces the shadow exactly (see
    :func:`_trace`), and ``curves`` the boundary of each, numbered alike; else ``trace`` is null,
    and the shadow is a mask. Every pair
    is shown up to :data:`PICTURE_EVERY_PAIR` numeric factors, or with ``every_pair``; from more,
    the pairs of the factors the limits name. ``center`` is where the page holds the other factors
    when it looks inside: the centre of the ranges when it can be run, else a central setting
    that can be, and null when none can.
    """
    numeric = [f for f in factors if f.type != "categorical"]
    names = [f.name for f in numeric]
    named = {name for i in alone for name in entries[i]["factors"]}
    shown = names if every_pair or len(names) <= PICTURE_EVERY_PAIR else [n for n in names if n in named]
    curved = [i for i in alone if entries[i]["type"] == "nonlinear"]
    straight = DesignRegion(numeric, [entries[i]["expression"] for i in alone if entries[i]["type"] == "linear"])
    within = DesignRegion(numeric, [entries[i]["expression"] for i in curved])
    ends = {f.name: f for f in numeric}
    pairs = []
    for x, y in itertools.combinations(shown, 2):
        lines = [
            {"limit": i, "points": line["points"]}
            for i, own in alone.items()
            if entries[i]["type"] == "linear" and set(entries[i]["factors"]) <= {x, y}
            for line in own.slice_lines(x, y)
        ]
        on_pair = [i for i in curved if set(entries[i]["factors"]) <= {x, y}]
        trace = None
        if curved and len(on_pair) == len(curved):
            # The curved limits name no other factor, so the shadow is the straight limits' shadow cut by them.
            inside = _inside_polygon(straight.shadow(x, y)["points"], ends[x], ends[y])
            meets = _runnable_at(within, numeric, x, y)
            trace = _trace(lambda u, v, inside=inside, meets=meets: inside(u, v) & meets(u, v), numeric, x, y)
        curves = _curves(alone, on_pair, lambda own, x=x, y=y: _trace(_runnable_at(own, numeric, x, y), numeric, x, y))
        pairs.append({"x": x, "y": y, "shadow": region.shadow(x, y), "lines": lines, "trace": trace, "curves": curves})
    middle = {f.name: (f.low + f.high) / 2 for f in numeric}
    center = middle if region.feasible(np.zeros((1, len(numeric))))[0] else region.centroid()
    return {"pairs": pairs, "shown": shown, "center": center}


def _sliced(
    raw: object,
    factors: list[Factor],
    region: DesignRegion,
    entries: list[dict[str, Any]],
    alone: dict[int, DesignRegion],
) -> dict[str, Any]:
    """Return the slice of the region the page looks inside: two factors, the others held at ``at``.

    ``region`` is the settings of ``x`` and ``y`` that can be run there (see
    :meth:`~process_improve.experiments.DesignRegion.slice_polygon`), ``lines`` each straight
    limit's boundary there, numbered by its place in the list. With a curved limit, ``trace``
    traces the settings that can be run (see :func:`_trace`), else it is null, and ``curves`` is
    the boundary of each curved limit there (see :func:`_curves`).
    """
    names = [f.name for f in factors if f.type != "categorical"]
    at = raw.get("at") if isinstance(raw, dict) else None
    if not (
        isinstance(at, dict)
        and raw.get("x") in names
        and raw.get("y") in names
        and raw["x"] != raw["y"]
        and all(name in names and _number(value) is not None for name, value in at.items())
    ):
        raise ValueError("The page sent a slice of the settings this page did not write.")
    x, y, held = raw["x"], raw["y"], {name: _number(value) for name, value in at.items()}
    lines = [
        {"limit": i, "points": line["points"]}
        for i, own in alone.items()
        if entries[i]["type"] == "linear"
        for line in own.slice_lines(x, y, held)
    ]
    numeric = [f for f in factors if f.type != "categorical"]
    curved = [i for i in alone if entries[i]["type"] == "nonlinear"]
    return {
        "x": x,
        "y": y,
        "at": held,
        "region": region.slice_polygon(x, y, held),
        "lines": lines,
        "trace": _trace(_runnable_at(region, numeric, x, y, held), numeric, x, y) if curved else None,
        "curves": _curves(alone, curved, lambda own: _trace(_runnable_at(own, numeric, x, y, held), numeric, x, y)),
    }


def _curves(
    alone: dict[int, DesignRegion], limits: list[int], trace: Callable[[DesignRegion], dict]
) -> list[dict[str, Any]]:
    """Return the boundary of each of these curved limits, as ``trace`` traces the limit on its own.

    Each is ``{"limit": i, "trace": ...}`` (see :func:`_trace`), numbered by its place in the list;
    a limit whose boundary misses the rectangle of the two factors' ranges has none.
    """
    curves = [{"limit": i, "trace": trace(alone[i])} for i in limits]
    return [curve for curve in curves if curve["trace"]["crossings"]]


def _trace(runnable: Callable[[np.ndarray, np.ndarray], np.ndarray], numeric: list[Factor], x: str, y: str) -> dict:
    """Return which settings of two factors can be run on a grid, and where the boundary crosses the grid's lines.

    ``runnable`` takes coded values of ``x`` and ``y``, two arrays of one shape, and says which of
    those settings can be run. In the result, ``x`` and ``y`` are :data:`TRACE_POINTS` values from
    each factor's low end to its high end, and ``z[i][j]`` is 1 where ``(x[j], y[i])`` can be run.
    Each crossing is ``[i, j, along, t]``: on the grid line from point ``(i, j)`` to ``(i, j + 1)``
    (``along`` 0) or to ``(i + 1, j)`` (``along`` 1), the share ``t`` of the way along it where the
    setting changes between can and cannot be run, found by halving the step
    :data:`TRACE_HALVINGS` times. The page draws a smooth boundary through the crossings, where a
    mask of the settings that can be run would be stepped.
    """
    grid = np.linspace(-1.0, 1.0, TRACE_POINTS)
    step = grid[1] - grid[0]
    gx, gy = np.meshgrid(grid, grid)
    can = runnable(gx, gy)
    crossings = []
    for along, (di, dj) in enumerate([(0, 1), (1, 0)]):
        first = can[: TRACE_POINTS - di, : TRACE_POINTS - dj]
        rows, cols = np.nonzero(first != can[di:, dj:])
        lo, hi = np.zeros(len(rows)), np.ones(len(rows))  # the share of the step, from the end that can be run
        from_first = first[rows, cols]
        for _ in range(TRACE_HALVINGS):
            middle = (lo + hi) / 2
            share = np.where(from_first, middle, 1.0 - middle)
            ok = runnable(grid[cols] + share * step * dj, grid[rows] + share * step * di)
            lo, hi = np.where(ok, middle, lo), np.where(ok, hi, middle)
        share = np.where(from_first, (lo + hi) / 2, 1.0 - (lo + hi) / 2)
        crossings += [[int(i), int(j), along, round(float(t), 6)] for i, j, t in zip(rows, cols, share, strict=True)]
    ends = {f.name: f for f in numeric}

    def actual(factor: Factor) -> list[float]:
        return [float(f"{(factor.low + factor.high) / 2 + c * (factor.high - factor.low) / 2:.12g}") for c in grid]

    return {"x": actual(ends[x]), "y": actual(ends[y]), "z": can.astype(int).tolist(), "crossings": crossings}


def _runnable_at(
    region: DesignRegion, numeric: list[Factor], x: str, y: str, held: dict[str, float] | None = None
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return the region's test of settings of ``x`` and ``y``, the others held at ``held`` (else at their centre)."""
    names = [f.name for f in numeric]
    ix, iy = names.index(x), names.index(y)
    given = held or {}
    base = np.array([_coded_value(f, given[f.name]) if f.name in given else 0.0 for f in numeric])

    def runnable(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        points = np.tile(base, (u.size, 1))
        points[:, ix], points[:, iy] = u.ravel(), v.ravel()
        return region.feasible(points).reshape(u.shape)

    return runnable


def _inside_polygon(
    polygon: list[list[float]], fx: Factor, fy: Factor
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return the test of settings of two factors, in coded units, against a convex polygon in actual units.

    The polygon's corners run counter-clockwise, as :meth:`~process_improve.experiments.DesignRegion.shadow`
    gives them; a setting on an edge is inside. A segment or a point holds no setting.
    """
    corners = np.array([[_coded_value(fx, px), _coded_value(fy, py)] for px, py in polygon]).reshape(-1, 2)

    def inside(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        held = np.full(u.shape, len(corners) >= 3)
        for (ax, ay), (bx, by) in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            held &= (bx - ax) * (v - ay) - (by - ay) * (u - ax) >= -1e-9
        return held

    return inside


def _coded_value(factor: Factor, value: float) -> float:
    """Return a numeric factor's value in coded units: -1 at its low end, +1 at its high end."""
    return (value - (factor.low + factor.high) / 2) / ((factor.high - factor.low) / 2)


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


def api_candidates(payload: str) -> str:
    """JSON wrapper around :func:`candidates`."""
    return _reply(candidates, payload)


def api_relabel(payload: str) -> str:
    """JSON wrapper around :func:`relabel`."""
    return _reply(relabel, payload)


def api_limits(payload: str) -> str:
    """JSON wrapper around :func:`limits`."""
    return _reply(limits, payload)
