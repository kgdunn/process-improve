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
import functools
import io
import itertools
import json
import logging
import math
import operator
import platform
import re
import sys
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
    augment_design,
    compare_designs,
    constraint_through_cuts,
    evaluate_design,
    generate_design,
    rename_factors,
    separating_constraint,
)
from process_improve.experiments._desirability import composite_desirability, individual_desirability
from process_improve.experiments._orthogonal_arrays import orthogonal_array
from process_improve.experiments.compare import MODELS as COMPARISON_MODELS
from process_improve.experiments.design_measures import estimable_models, model_matrix
from process_improve.experiments.designs_constrained import MAX_EXPRESSION_LENGTH, factor_lookup, parse_constraint
from process_improve.univariate import (
    dunnett_value,
    least_favourable_means,
    one_way_power,
    one_way_sample_size,
    randomization_interval,
    randomization_test,
    rank_sum_test,
    studentized_range_value,
    ttest_independent,
    ttest_paired,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator

SPEC_SHEET = "_pi_design"
RUN_SHEET = "Runs"
SPEC_VERSION = 1

#: The run sheet's column marking simulated results. Its space keeps it apart from every factor
#: and response, which must be variable names; workbooks written before used ``Source``.
MARKS = "Result source"

#: The run sheet's column marking a confirmation run: a run made at the best settings, to test the model
#: there. Its space keeps it apart from every factor and response, as for :data:`MARKS`. A run sheet has it
#: once the reader adds confirmation runs, and the model is never fitted to the runs it marks.
CONFIRMATION = "Confirmation run"

#: What that column holds for a confirmation run; a run of the design leaves it blank.
CONFIRMS = "yes"

#: The most factors a study has on this page: a screening design's, made to find the few factors that matter among
#: many, such as Plackett-Burman's 44 runs in 40 factors, or the catalog's largest, 40 factors in 81 runs.
#: logic.js and campaign.js mirror it.
MAX_FACTORS = 40

#: The most factors the page models the interactions of: 12 factors have 66 two-factor interactions, and 23 have
#: 253, more than a screening design has runs. With more, a study aims to learn the main effects, a fraction is
#: made at resolution III or IV, and Learn fits the main effects, or the terms the reader chooses. logic.js mirrors it.
MAX_INTERACTION_FACTORS = 12

#: The most factors the page compares designs for on every measure. G, the worst prediction, is searched for at
#: every corner of the ranges, 2^k of them: a million at 20 factors. With more, the designs are scored as for a
#: factor of several levels: their runs, their spare runs and their power (see :func:`_powers`).
MAX_COMPARED_FACTORS = 12

#: The most factors the page works out the next step for. The step weighs runs for every factor whose effect
#: stands out, and in a screening design of many factors and few spare runs many do: natively it takes 7 s at
#: 24 factors, and from 30 s to 3 minutes at 40, and the browser is slower again.
MAX_NEXT_STEP_FACTORS = 24

#: Design types the page offers. Each one is pure numpy / scipy, so it runs
#: under WebAssembly. Excluded: ``omars_ilp`` (its HiGHS multistart needs
#: scipy >= 1.15, newer than Pyodide's, and takes too long for a page).
#:
#: ``min_factors`` is what the construction needs. ``max_factors`` keeps the run
#: sheet readable and the page responsive: a full factorial in 8 factors has 256
#: runs, a Box-Behnken design jumps from 59 to 115 runs at 8 factors, and an
#: I-optimal design for a quadratic in 6 factors already takes up to 20 s under
#: WebAssembly. The screening designs whose construction is quick at any size take
#: :data:`MAX_FACTORS`; a fraction stops at 24, since past it the library's search for
#: its generators takes from seconds to minutes.
DESIGNS: dict[str, dict[str, Any]] = {
    "full_factorial": {
        "label": "Full factorial (2-level)",
        "hint": "Every combination of low and high. 2^k runs, plus center points.",
        "model": "interactions",
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 1,
        "max_factors": 7,
    },
    "fractional_factorial": {
        "label": "Fractional factorial (2-level)",
        "hint": "A fraction of the full factorial; its resolution says what is aliased with what.",
        "model": "main_effects",
        "options": ["n_center_points", "resolution", "n_replicates", "n_blocks"],
        "min_factors": 3,
        "max_factors": 24,
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
        "max_factors": MAX_FACTORS,
    },
    "taguchi": {
        "label": "Taguchi orthogonal array",
        "hint": (
            "Main effects in a standard array of 4 to 81 runs, and at three levels each numeric factor's square; "
            "interactions are aliased with them. A categorical factor stays at its two levels."
        ),
        "model": "main_effects",
        "options": ["n_levels", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": MAX_FACTORS,
    },
    "dsd": {
        "label": "Definitive screening (DSD)",
        "hint": "Three levels in 2k+1 runs (2k+3 for odd k); main effects clear of interactions and quadratics.",
        "model": "main_effects",
        "options": ["n_center_points", "n_replicates", "n_blocks"],
        "min_factors": 3,
        "max_factors": MAX_FACTORS,
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
    "maximin_lhs": {
        "label": "Space-filling (Latin hypercube)",
        "hint": (
            "Runs spread evenly through the ranges, made for no model: each factor takes as many different settings "
            "as there are runs. For mapping a response whose shape is unknown, or for a machine-learning model."
        ),
        "model": "quadratic",
        "options": ["budget", "n_replicates", "n_blocks"],
        "min_factors": 2,
        "max_factors": 8,
    },
}

#: The space-filling designs (``designs_space_filling.py``): their runs spread through the ranges for no model in
#: particular, so the comparison of designs for a model leaves them out, and Learn analyses one with the model the
#: study aims to learn. A maximin Latin hypercube keeps each factor's settings apart as well as the runs.
SPACE_FILLING = ("maximin_lhs",)

#: Runs a factor in a space-filling design, unless the reader gives a number: the usual start for a model fitted
#: to a computer experiment (Loeppky, Sacks and Welch 2009), and the library's default (logic.js mirrors it).
SPACE_FILLING_RUNS = 10

#: The most runs a space-filling design has on this page, as the Runs box allows.
MAX_SPACE_FILLING_RUNS = 200

#: The design the page holds but does not make: runs taken as they are given, such as a design a book
#: prints, opened from the design catalog. It is not in :data:`DESIGNS`, so the comparison never tries
#: it; the design list (index.html) holds its option hidden, and shows it once a run sheet of given
#: runs is open, so the reader can make those runs again after changing the brief.
GIVEN = "given"
GIVEN_INFO: dict[str, Any] = {
    "label": "Runs as given",
    "hint": "The runs of a design you have already, such as one a book prints: the page takes them as they are.",
    "min_factors": 1,
    "max_factors": MAX_FACTORS,
}

#: The most runs a design of given runs may have: as many as a study the page holds (campaign.js).
MAX_GIVEN_RUNS = 2000

#: How far a given run may sit past a limit on the settings, in the limit's own units, and still meet it.
LIMIT_TOLERANCE = 1e-9

#: What a design of given runs says of itself. It names no factor, so a renaming keeps it.
GIVEN_NOTE = "These runs are taken as they were given, in that order: the page made none of them."

#: Mixture designs: the factors are the components of a blend, each a proportion between its lower and upper
#: bounds, and the proportions of every run sum to 1. Each is ``generate_design(design_type="mixture")``, built
#: for the Scheffé model its options name. They are not in :data:`DESIGNS`, as the runs as given are not: a
#: mixture has no ranges to code, no center runs and no blocks, so the comparison of designs never tries one.
#: The simplex centroid stops at four components, since from five the library's design for the special cubic
#: model blends at most three of them, which is not the whole simplex centroid.
MIXTURES: dict[str, dict[str, Any]] = {
    "simplex_lattice": {
        "label": "Simplex lattice (mixture)",
        "hint": (
            "The pure components and, at degree 2, every 50:50 blend of two, with the overall centroid to check "
            "the fit."
        ),
        "options": ["lattice_degree", "n_replicates"],
        "min_factors": 3,
        "max_factors": 6,
    },
    "simplex_centroid": {
        "label": "Simplex centroid (mixture)",
        "hint": (
            "Every blend of equal parts: the pure components, the 50:50 blends, the blends in thirds, and so on to "
            "the overall centroid."
        ),
        "options": ["n_replicates"],
        "min_factors": 3,
        "max_factors": 4,
    },
    "extreme_vertices": {
        "label": "Extreme vertices (mixture with bounds)",
        "hint": (
            "The corners of the blends your lower and upper proportions allow, the midpoints of the edges between "
            "them for the quadratic model, and their centroid."
        ),
        "options": ["mixture_model", "n_replicates"],
        "min_factors": 3,
        "max_factors": 6,
    },
}

#: The library's design type for every mixture design.
MIXTURE = "mixture"

#: The models a mixture is analysed with, as ``analyze_experiment`` names them: the Scheffé models, which have no
#: intercept, since the proportions sum to 1. Each is named in a sentence by its words.
MIXTURE_MODELS = ("scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic")
MIXTURE_WORDS = {
    "scheffe_linear": "linear blending",
    "scheffe_quadratic": "quadratic blending",
    "scheffe_special_cubic": "special cubic blending",
}

#: A simplex lattice's degree, and the Scheffé model its runs are built for: the pure components alone (with the
#: centroid) for the linear model, and with every 50:50 blend for the quadratic one.
LATTICE_DEGREES = {1: "scheffe_linear", 2: "scheffe_quadratic"}
DEFAULT_LATTICE_DEGREE = 2

#: The models an extreme vertices design is built for; the first is the page's default.
VERTICES_MODELS = ("scheffe_quadratic", "scheffe_linear")

#: How far a proportion may sit past 0 or 1, or a sum of bounds from 1, and still be read as it.
PROPORTION_TOLERANCE = 1e-9

#: Significant digits of a proportion on the run sheet: a third reads 0.333333333333, and an edge's midpoint
#: 0.15 rather than the 0.15000000000000002 that the arithmetic leaves.
PROPORTION_DIGITS = 12

#: The number of components of a blend of equal parts, in words.
PARTS = {2: "two", 3: "three", 4: "four", 5: "five"}

#: The seed a design's runs are put in random order with, unless its specification names one. The page names one
#: for each study it begins, the moment it began, so this is the seed of the worked example, the book's examples and
#: the studies saved before then. A study's report names its seed (report.js's RUN_ORDER_SEED).
RUN_ORDER_SEED = 42

#: Designs whose non-centre runs all lie on one sphere (Box-Behnken at any size, a
#: rotatable CCD at 2 or 4 factors), so without a center run the squared terms sum
#: to a constant and the quadratic model cannot be fitted.
NEEDS_CENTER_RUN = {"box_behnken": None, "ccd": "rotatable"}

#: Runs an optimal design gets beyond its model's coefficients, when no budget is
#: given, so the fit keeps degrees of freedom to estimate the error.
SPARE_RUNS = 4

#: The most times the page makes each run of a design, center runs excepted.
MAX_REPLICATES = 10

#: The most runs at each level of a design in one factor, which are its replicates: a comparison of two levels can
#: need many more runs at each than a design of more factors repeats each of its runs.
MAX_LEVEL_RUNS = 50

#: The most blocks the page splits a design's runs into.
MAX_BLOCKS = 8

#: The most blocks runs as given may come in: a source's paired comparison of ten pairs, or a design in 27 blocks.
MAX_GIVEN_BLOCKS = 32

#: The designs that take a categorical factor, one whose levels are labels (supplier A and B):
#: the two-level designs, which need only the two levels of each factor, and the Taguchi arrays,
#: which give a categorical factor a column of its own beside the numeric factors' columns.
CATEGORICAL_DESIGNS = ("full_factorial", "fractional_factorial", "plackett_burman", "taguchi")

#: Of those, the designs that take a categorical factor of three levels or more: the full factorial,
#: which runs every combination of the levels, and the Taguchi arrays, which have mixed-level columns.
SEVERAL_LEVEL_DESIGNS = ("full_factorial", "taguchi")

#: The most levels a categorical factor takes on this page; each level but one is a column of the model.
MAX_LEVELS = 8

#: The longest label a level of a categorical factor may have.
LABEL_MOST = 64

#: What joins a factor's name to the number of a level in the name of its column of the model, for a
#: categorical factor of three levels or more: ``Catalyst_L1`` is +1 at the first level, -1 at the last.
#: (The library refuses a double underscore in a name.)
LEVEL_JOIN = "_L"

#: What joins a factor's name to the number of a level in the name of its indicator column, 1 at that level and 0
#: elsewhere: ``Supplier_I1``. A model takes a factor's indicators where another factor is nested in it, as
#: Batch is in ``Supplier + Supplier:Batch``, so the term holds Batch's effect within each supplier.
INDICATOR_JOIN = "_I"

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

#: Above this variance inflation factor, a term is correlated with the others enough for a note: its standard
#: error is more than twice (the square root of 5) what runs that kept the terms uncorrelated would give.
VIF_HIGH = 5.0

#: How many alias groups a summary sentence names before saying how many more.
ALIASES_SHOWN = 4
#: The words of a defining relation the design's note lists before it counts the rest: a 2^(k-3) fraction's whole one.
RELATION_SHOWN = 7
#: The most words of the defining relation of runs as given that the design's facts list, as the library lists a
#: fraction's: ten generators' worth. A relation with more, as of a 2^(15-11) fraction, has its shortest words named.
RELATION_LISTED = 1023

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

#: The models of those with every two-factor interaction, which a study of more than
#: :data:`MAX_INTERACTION_FACTORS` factors is not fitted with.
INTERACTION_MODELS = ("interactions", "quadratic")

#: What the reader wants of the response, as the library's ``ResponseGoal`` names it, or ``learn``:
#: to see how the factors act on it, with no setting to aim for.
GOALS = ("maximize", "minimize", "target", "learn")

#: The most responses a study measures. Each has a column of its own on the run sheet, its own model and its own
#: analysis; the first one sizes the design, as the brief's typical value, noise and smallest change are its own.
MAX_RESPONSES = 5

#: How far apart the seeds of two responses' simulated noise are: the response in place j (0 for the first) draws
#: its noise with the seed plus j times this, so each response has noise of its own and the first keeps the seed's.
SEED_STEP = 1000

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

#: Processes a mixture's results can be simulated from, as a Scheffé quadratic model in the proportions. ``pure``
#: is the response of each pure component, the components taken in the order the reader entered them, as a size
#: in noise standard deviations from the scenario's ``centre``; a component it does not name gives the centre.
#: ``blends`` are the two-component blending terms, in noise standard deviations: positive where a blend gives
#: more than the average of its two pure components (synergy), negative where it gives less (antagonism).
MIXTURE_SCENARIOS: dict[str, dict[str, Any]] = {
    "blend": {
        "label": "A synergy and an antagonism",
        "hint": "The first two components blend in synergy, and the second and third against each other.",
        "centre": 50.0,
        "pure": {0: 4.0, 1: -2.0, 2: 1.0},
        "blends": {(0, 1): 12.0, (1, 2): -10.0},
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


def _factor_names(raw: list) -> list[str]:
    """Read and check the factors' names: one at least, each a variable name, none twice."""
    if not raw:
        raise ValueError("Enter at least one factor.")
    names = [str(f.get("name", "")).strip() for f in raw]
    if len(set(names)) != len(names) or not all(n.isidentifier() for n in names):
        raise ValueError("Factor names must be unique and look like variable names (letters, digits, _).")
    return names


def _factors(spec: dict) -> list[Factor]:
    """Build and check the ``Factor`` list from the specification."""
    if spec.get("design_type") in MIXTURES:
        return _components(spec)
    raw = spec.get("factors") or []
    names = _factor_names(raw)
    factors = []
    for f, name in zip(raw, names, strict=True):
        if f.get("type") == "categorical" or "levels" in f:  # a type the page says, and its levels, any text
            factors.append(_categorical_factor(name, f.get("levels")))
            continue
        labels = _label(f.get("low")), _label(f.get("high"))
        if f.get("type") != "numeric" and None not in labels:  # two labels, as the page has always sent them
            factors.append(_categorical_factor(name, list(labels)))
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
    columns = {
        f"{f.name}{join}{j}"
        for f in factors
        if f.type == "categorical"
        for join, most in ((LEVEL_JOIN, len(f.levels or []) - 1), (INDICATOR_JOIN, len(f.levels or [])))
        for j in range(1, most + 1)
    }
    if taken := sorted(set(names) & columns):
        raise ValueError(f"Factor {taken[0]}: rename it, as the page names a column of another factor's levels so.")
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


def _categorical_factor(name: str, raw: object) -> Factor:
    """Make a categorical factor from its levels, in their order: 2 to :data:`MAX_LEVELS` labels, any text.

    A label may read as a number, such as lot 1, 2 and 3: the factor is categorical because the page
    says so, not because of its labels. Two labels may not read as the same number (1 and 1.0), as a
    spreadsheet would make them one; nor may a label read as a blank cell there.
    """
    if not isinstance(raw, list) or not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in raw):
        raise ValueError(f"Factor {name}: list its levels, each a label.")
    levels = [_level_text(v) for v in raw]
    if not 2 <= len(levels) <= MAX_LEVELS:  # a factor varies: two levels at least
        raise ValueError(f"Factor {name}: give it 2 to {MAX_LEVELS} levels; it has {len(levels)}.")
    if "" in levels:
        raise ValueError(f"Factor {name}: a level is blank; give each level a label.")
    if long := next((v for v in levels if len(v) > LABEL_MOST), None):
        raise ValueError(f"Factor {name}: the level {long[:20]!r}... is longer than {LABEL_MOST} characters.")
    if blank := next((v for v in levels if v in BLANK_LABELS), None):
        raise ValueError(f"Factor {name}: a spreadsheet reads {blank!r} back as a blank cell; choose another label.")
    seen: dict[object, str] = {}
    for level in levels:
        key = _number(level) if _number(level) is not None else level
        if key in seen:
            same = "is listed twice" if seen[key] == level else f"and {seen[key]} read as the same number"
            raise ValueError(f"Factor {name}: {level} {same}; its levels must differ.")
        seen[key] = level
    return Factor(name=name, type="categorical", levels=levels)


def _level_text(value: object) -> str:
    """Write a level as the label it is: a number a spreadsheet read back, such as 2.0, as the 2 it was."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _factor_spec(f: Factor) -> dict[str, Any]:
    """Write a factor as the page holds it.

    A categorical factor of two labels that read as no number keeps them as its low and high values,
    as every study before categorical factors of more levels. Any other says its type and lists its
    levels; one of two levels keeps them as its low and high too, as every two-level design reads them.
    """
    if f.type == "categorical":
        levels = list(f.levels or [])
        two = {"low": levels[0], "high": levels[1]} if len(levels) == 2 else {}
        if two and all(_label(v) is not None for v in levels):
            return {"name": f.name, **two, "units": ""}
        return {"name": f.name, "type": "categorical", "levels": levels, **two, "units": ""}
    return {"name": f.name, "low": f.low, "high": f.high, "units": f.units}


def _categorical(factor: dict) -> bool:
    """Say whether a factor the page holds is categorical: its type says so, or its two levels are labels."""
    return factor.get("type") == "categorical" or isinstance(factor.get("low"), str)


def _levels_of(factor: dict) -> list[str]:
    """Return a categorical factor's levels, in their order: its list of levels, or its two labels."""
    return list(factor["levels"]) if "levels" in factor else [factor["low"], factor["high"]]


def _several(factor: dict) -> bool:
    """Say whether a factor is categorical with three levels or more: a column of the model per level but one."""
    return _categorical(factor) and len(_levels_of(factor)) > 2  # a two-level factor is one column


def _level_index(value: object, levels: list[str]) -> int | None:
    """Return which of a factor's levels a setting is: its label, or the number a spreadsheet read it as."""
    text = _level_text(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None
    if text is None:
        return None
    if text in levels:
        return levels.index(text)
    number = _number(text)
    matches = [i for i, level in enumerate(levels) if number is not None and _number(level) == number]
    return matches[0] if len(matches) == 1 else None


def _level_at(factor: dict, coded: float) -> str:
    """Return the level of a categorical factor nearest its coded setting: -1 its first, +1 its last."""
    levels = _levels_of(factor)
    return levels[min(len(levels) - 1, max(0, round((coded + 1) / 2 * (len(levels) - 1))))]


def _held_columns(hold: dict[str, float], spec: dict) -> dict[str, float]:
    """Write coded settings over the columns the model is fitted on: a factor of several levels as its contrasts.

    A categorical factor's indicators are written too, 1 at its level and 0 at the others, for a model with a factor
    nested in it; a model without them leaves them unused.
    """
    out: dict[str, float] = {}
    for f in spec["factors"]:
        if f["name"] not in hold:
            continue
        if _categorical(f):
            levels = _levels_of(f)
            at = levels.index(_level_at(f, hold[f["name"]]))
            out.update({name: float(j == at) for j, name in enumerate(_indicator_names(f))})
        if not _several(f):
            out[f["name"]] = hold[f["name"]]
            continue
        levels = _levels_of(f)
        index = np.array([levels.index(_level_at(f, hold[f["name"]]))])
        out.update(zip(_contrast_names(f), _contrasts(index, len(levels))[0], strict=True))
    return out


def _indicator_names(factor: dict) -> list[str]:
    """Name a categorical factor's indicator columns, one for each of its levels: ``Supplier_I1``, ``Supplier_I2``."""
    return [f"{factor['name']}{INDICATOR_JOIN}{j}" for j in range(1, len(_levels_of(factor)) + 1)]


def _indicators(index: np.ndarray, k: int) -> np.ndarray:
    """Return the indicator columns of a factor of ``k`` levels: column j is 1 at level j and 0 elsewhere."""
    return (index[:, None] == np.arange(k)[None, :]).astype(float)


def _nested(terms: list[str], spec: dict) -> dict[str, str]:
    """Find the terms in which a categorical factor takes its indicators, as patsy codes them, and name it for each.

    In a term with a categorical factor of three levels or more, a categorical factor takes its indicators, not
    its contrasts, when the model lacks the term without it: in ``Supplier + Supplier:Batch`` the term less
    Supplier is Batch, which the model lacks, so Supplier takes its indicators and the term holds Batch's effect
    within each supplier. A term in which two factors would need them is refused: the model cannot tell it from
    the terms it lacks. A term of two-level and numeric factors stays the product of their -1/+1 columns, the
    effect a two-level design estimates, with the terms below it in the model or not.
    """
    categorical = set(_categorical_names(spec))
    several = {f["name"] for f in spec["factors"] if _several(f)}
    present = {frozenset()} | {frozenset(t.split(":")) for t in terms if not t.startswith("I(")}
    out = {}
    for term in terms:
        parts = [] if term.startswith("I(") or not several & set(term.split(":")) else term.split(":")
        full = [p for p in parts if p in categorical and len(parts) > 1 and frozenset(parts) - {p} not in present]
        if len(full) > 1:
            lacking = _listed([_pretty(":".join(q for q in parts if q != p)) for p in reversed(full)], last="or")
            raise ValueError(
                f"The term {_pretty(term)} needs {lacking} in the model too: add one of them, or take it out."
            )
        if full:
            out[term] = full[0]
    return out


def _contrast_names(factor: dict) -> list[str]:
    """Name the columns of the model of a categorical factor of three levels or more: one per level but the last."""
    return [f"{factor['name']}{LEVEL_JOIN}{j}" for j in range(1, len(_levels_of(factor)))]


def _contrasts(index: np.ndarray, k: int) -> np.ndarray:
    """Return the sum-to-zero columns of a factor of ``k`` levels: column j is +1 at level j, -1 at the last, else 0.

    Each column's coefficient is then how far that level lies from the average of the levels, and the last
    level's is minus the sum of the others', as the blocks' are (``Block1``, ``Block2``, ...).
    """
    columns = np.zeros((len(index), k - 1))
    for j in range(k - 1):
        columns[:, j] = (index == j).astype(float) - (index == k - 1).astype(float)
    return columns


def _model_columns(spec: dict) -> list[str]:
    """Name the columns the model is fitted on: each factor's own, or a factor of several levels' contrasts."""
    return [c for f in spec["factors"] for c in (_contrast_names(f) if _several(f) else [f["name"]])]


def _expanded_terms(terms: list[str], spec: dict, model: list[str] | None = None) -> list[str]:
    """Write a model's terms, named over the factors, over the columns the model is fitted on.

    A categorical factor of three levels or more is its contrasts: its main effect is a column per
    level but the last, and its interaction with another factor is each of those columns times the
    other factor's. A factor another is nested in is its indicators in that term (:func:`_nested`),
    judged in ``model``, the whole model, when ``terms`` are only some of its terms. A square is a
    numeric factor's, and stays as it is.
    """
    columns = {f["name"]: _contrast_names(f) if _several(f) else [f["name"]] for f in spec["factors"]}
    indicators = {f["name"]: _indicator_names(f) for f in spec["factors"] if _categorical(f)}
    nested = _nested(terms if model is None else model, spec)
    out = []
    for term in terms:
        if term.startswith("I("):
            out.append(term)
            continue
        each = [indicators[name] if nested.get(term) == name else columns[name] for name in term.split(":")]
        out += [":".join(product) for product in itertools.product(*each)]
    return out


def _fitted_model(model: str | list[str], spec: dict) -> tuple[str | list[str], list[str]]:
    """Return the model as it is fitted, and the columns it is fitted on.

    Without a categorical factor of three levels or more, or one another is nested in, these are the
    model itself and the factors' names. Otherwise its terms written over the columns
    (:func:`_expanded_terms`), since a named model would multiply a factor's own contrasts together,
    and the columns with the indicators of each factor another is nested in.
    """
    names = [f["name"] for f in spec["factors"]]
    terms = _model_terms(model, names, _categorical_names(spec))
    nested = _nested(terms, spec)
    if not any(_several(f) for f in spec["factors"]) and not nested:
        return model, names
    factors = {f["name"]: f for f in spec["factors"]}
    indicators = [c for name in dict.fromkeys(nested.values()) for c in _indicator_names(factors[name])]
    return _expanded_terms(terms, spec), [*_model_columns(spec), *indicators]


def _level_words(text: str, spec: dict) -> str:
    """Write a fitted column of a categorical factor as the page shows it: ``Catalyst_L1`` as ``Catalyst[A]``.

    A contrast column and an indicator column of the same level read alike: the term each is in says which it is.
    """
    for f in spec["factors"]:
        if _categorical(f):
            levels = _levels_of(f)
            for j, column in enumerate(_indicator_names(f)):
                text = re.sub(rf"\b{re.escape(column)}\b", f"{f['name']}[{levels[j]}]", text)
        if _several(f):
            levels = _levels_of(f)
            for j, column in enumerate(_contrast_names(f)):
                text = re.sub(rf"\b{re.escape(column)}\b", f"{f['name']}[{levels[j]}]", text)
    return text


def _categorical_names(spec: dict) -> tuple[str, ...]:
    """Name the categorical factors of a checked specification."""
    return tuple(f["name"] for f in spec["factors"] if _categorical(f))


# --------------------------------------------------------------------------- a mixture's components
def _mixture(spec: dict) -> bool:
    """Say whether a specification is a mixture's: its factors are the components of a blend."""
    return spec.get("design_type") in MIXTURES


def _components(spec: dict) -> list[Factor]:
    """Build and check a mixture's components: each a proportion between its lower and upper bounds.

    A component's low and high are its lower and upper proportions, 0 and 1 when left blank (the
    page sends a blank box as null). A proportion lies between 0 and 1, so a factor with a range
    beyond them is a process variable, which a mixture on this page does not take; nor does it take
    a categorical factor. The lower proportions must sum to less than 1, and the upper ones to more
    than 1, or no blend meets them, or only one. The design type decides how many components there
    may be, and whether they may have bounds (see :func:`_check_bounds`).
    """
    design_type = spec["design_type"]
    raw = spec.get("factors") or []
    names = _factor_names(raw)
    components = []
    for f, name in zip(raw, names, strict=True):
        if _categorical(f) or (_label(f.get("low")) is not None and _label(f.get("high")) is not None):
            raise ValueError(
                f"A mixture's factors are its components, each a proportion, and {name} is categorical: remove it, "
                "or choose a design for factors with ranges."
            )
        low, high = _proportion(f.get("low"), 0.0), _proportion(f.get("high"), 1.0)
        if low is None or high is None:
            raise ValueError(
                f"Component {name}: enter its lower and upper proportions as numbers, or leave them blank for 0 and 1."
            )
        if low < -PROPORTION_TOLERANCE or high > 1 + PROPORTION_TOLERANCE:
            raise ValueError(
                f"Factor {name} runs from {low:g} to {high:g}, which is no proportion: each component of a mixture "
                "lies between 0 and 1, and a mixture on this page takes no process variables, such as a temperature."
            )
        if not low < high:
            raise ValueError(f"Component {name}: its lower proportion must be below its upper proportion.")
        units = str(f.get("units", ""))
        components.append(Factor(name=name, type="mixture", low=max(low, 0.0), high=min(high, 1.0), units=units))
    info = MIXTURES[design_type]
    if not info["min_factors"] <= len(components) <= info["max_factors"]:
        raise ValueError(
            f"The {info['label']} design takes {info['min_factors']} to {info['max_factors']} components on this "
            f"page; you have {len(components)}."
        )
    lows, highs = sum(c.low for c in components), sum(c.high for c in components)  # type: ignore[misc]
    if lows > 1 - PROPORTION_TOLERANCE:
        raise ValueError(
            f"The lower proportions sum to {lows:.4g}, and must sum to less than 1: above 1 no blend gives every "
            "component its lower proportion, and at 1 only one blend does. Lower some of them."
        )
    if highs < 1 + PROPORTION_TOLERANCE:
        raise ValueError(
            f"The upper proportions sum to {highs:.4g}, and must sum to more than 1: below 1 no blend of the "
            "components reaches 1, and at 1 only one does. Raise some of them."
        )
    _check_bounds(design_type, components)
    return components


def _proportion(value: object, blank: float) -> float | None:
    """Read a component's bound: ``blank`` for a blank box, the number typed, or None for anything else."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return blank
    return _number(value)


def _check_bounds(design_type: str, components: list[Factor]) -> None:
    """Refuse bounds the design type cannot take: extreme vertices need one, the simplex designs none."""
    bounded = [
        c.name
        for c in components
        if c.low > PROPORTION_TOLERANCE or c.high < 1 - PROPORTION_TOLERANCE  # type: ignore[operator]
    ]
    if design_type == "extreme_vertices" and not bounded:
        raise ValueError(
            "Extreme vertices are for a mixture with bounds: give a component a lower or upper proportion, or "
            "choose the simplex lattice or simplex centroid."
        )
    if design_type != "extreme_vertices" and bounded:
        verb = "has" if len(bounded) == 1 else "have"
        raise ValueError(
            f"The {MIXTURES[design_type]['label']} design covers every blend, each component from 0 to 1, and "
            f"{_listed(bounded)} {verb} bounds: choose extreme vertices for a mixture with bounds."
        )


def _mixture_model(model: object) -> str:
    """Read the model a mixture is analysed with: one of :data:`MIXTURE_MODELS`."""
    if model not in MIXTURE_MODELS:
        raise ValueError(
            f"A mixture is analysed with the linear, quadratic or special cubic blending model, not {model!r}."
        )
    return str(model)


def _mixture_terms(q: int, model: str) -> int:
    """Count a Scheffé model's coefficients in ``q`` components: no intercept, as the proportions sum to 1."""
    return q + math.comb(q, 2) * (model != "scheffe_linear") + math.comb(q, 3) * (model == "scheffe_special_cubic")


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
    # The library's canonical text, with each factor written as it is named: the parsed text holds a name's NFKC
    # form, the plain T of a full-width one (#808).
    canonical = rename_factors(text, {name: name for name in named})
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
    """Return the factors a limit uses, by their names, refusing a function, a factor or a name it cannot use.

    The parsed limit holds each name in its NFKC form, so a name is matched through that form (#808).
    """
    names = [f.name for f in factors]
    numeric = [f.name for f in factors if f.type != "categorical"]
    lookup = factor_lookup(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in LIMIT_FUNCTIONS):
            raise ValueError(
                f"The limit {text!r} uses {ast.unparse(node.func)}, which a limit cannot: the functions it can use "
                f"are {_listed(list(LIMIT_FUNCTIONS))}."
            )
    named = [lookup.get(name, name) for name in _named(tree)]
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


#: The packages whose releases decide an analysis's numbers, which a study's report names beside process-improve.
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels")


def _runtime() -> dict[str, Any]:
    """Name the Python the page runs, each of :data:`RUNTIME_PACKAGES` installed, and Pyodide's release.

    Pyodide is the one that has already imported its own ``pyodide`` package, so outside a browser it is None.
    """
    packages = {}
    for name in RUNTIME_PACKAGES:
        with contextlib.suppress(PackageNotFoundError):
            packages[name] = version(name)
    pyodide = sys.modules.get("pyodide")
    return {"python": platform.python_version(), "packages": packages, "pyodide": getattr(pyodide, "__version__", None)}


def catalogue(_: dict) -> dict:
    """List what the page offers: design types, mixtures, models, CCD axial choices, processes, and the worked example.

    ``version`` is the process-improve release the page runs, which a notebook it writes installs, and ``runtime``
    the Python, packages and Pyodide it runs on (see :func:`_runtime`), which a study's report names.
    """
    scenarios = {key: {"label": s["label"], "hint": s["hint"], "sigma": s["sigma"]} for key, s in SCENARIOS.items()}
    mixture_scenarios = {
        key: {"label": s["label"], "hint": s["hint"], "sigma": s["sigma"]} for key, s in MIXTURE_SCENARIOS.items()
    }
    return {
        "designs": DESIGNS,
        "mixtures": MIXTURES,
        "models": list(MODELS),
        "mixture_models": list(MIXTURE_MODELS),
        "alphas": list(ALPHAS),
        "scenarios": scenarios,
        "mixture_scenarios": mixture_scenarios,
        "demo": DEMO,
        "version": _release(),
        "runtime": _runtime(),
    }


def make_design(spec: dict) -> dict:
    """Generate a design and return its runs plus a ready-to-download workbook.

    The workbook lists the runs in randomised order, in the factors' own units,
    with an empty response column for the reader to fill in.
    """
    design_type = spec.get("design_type", "")
    if design_type == GIVEN:
        return _given_design(spec)
    if design_type in MIXTURES:
        return _mixture_design(spec)
    if design_type not in DESIGNS:
        raise ValueError(f"Unknown design type {design_type!r}.")
    factors = _factors(spec)
    _check_factor_count(design_type, len(factors))
    limits = _checked_constraints(spec.get("constraints"), factors)
    expressions = [limit["expression"] for limit in limits]
    region = _region_refusal(factors, _region(factors, expressions), design_type) if limits else None
    _check_categorical(design_type, factors)
    responses = _checked_responses(spec, factors, blank="y")
    described = {**_stored_responses(responses), **_aimed(spec)}

    kwargs = _design_options(spec, design_type)
    made, asked = _library_arguments(factors, kwargs)
    # The limits are passed apart from the options, which hold one design's settings, as the comparison reuses them.
    result = generate_design(made, design_type=design_type, **asked, **({"constraints": expressions} if limits else {}))

    runs = _made_runs(result, factors, design_type, kwargs)
    for response in responses:  # a column for each response's results, in the brief's order
        runs[response["name"]] = np.nan

    stored = {
        "version": SPEC_VERSION,
        "design_type": design_type,
        "factors": [_factor_spec(f) for f in factors],
        "response": responses[0]["name"],
        "model": kwargs.get("model_type") or _designed_model(design_type, result.resolution, made, spec.get("aim")),
        "options": kwargs,
        **described,
        **({"constraints": limits} if limits else {}),
    }
    coded = _coded(runs, stored["factors"])
    facts = _design_facts(result, coded.to_numpy(dtype=float), stored["factors"], kwargs)
    if array := (result.metadata or {}).get("orthogonal_array"):
        facts["notes"].insert(0, _array_sentence(array, stored["model"]))
    if design_type in SPACE_FILLING:
        facts["notes"].insert(0, _spread_sentence(kwargs["budget"]))
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


def _made_runs(result: Any, factors: list[Factor], design_type: str, kwargs: dict[str, Any]) -> pd.DataFrame:  # noqa: ANN401
    """Lay out the runs the library made as the run sheet lists them, before the responses' columns.

    The library returns the runs already randomised, and ``run_order[i]`` is the standard-order
    position of its i-th run: the sheet keeps the rows as they come, under their "Std order" and
    "Run order", with their block first when they come in blocks. A space-filling design's settings
    are rounded to what can be set (:func:`_round_spread`). A categorical factor's settings are its
    labels, as the reader typed them: the library reads labels such as 1 and 2 as numbers.
    """
    names = [f.name for f in factors]
    runs = pd.DataFrame(result.design_actual[names].to_numpy(), columns=names)
    for f in factors:
        if f.type == "categorical":
            levels = [str(v) for v in f.levels or []]
            runs[f.name] = [levels[i] for i in _level_indices(runs[f.name], {"name": f.name, "levels": levels})]
    if BLOCK in result.design_actual.columns:  # the runs are listed block by block
        runs.insert(0, BLOCK, result.design_actual[BLOCK].to_numpy())
    if design_type in SPACE_FILLING:
        _round_spread(runs, factors, kwargs["budget"])
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", result.run_order)
    return runs


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
    runs = _given_settings(rows, factors)
    responses = _checked_responses(spec, factors, blank="y")
    limits = _checked_constraints(spec.get("constraints"), factors)
    blocks = _given_blocks(spec.get("blocks"), len(rows))
    stored: dict[str, Any] = {
        "version": SPEC_VERSION,
        "design_type": GIVEN,
        "factors": [_factor_spec(f) for f in factors],
        "response": responses[0]["name"],
        "options": {
            **({"n_blocks": len(set(blocks))} if blocks else {}),
            **({"random_state": int(spec["random_state"])} if "random_state" in spec else {}),
        },
        **_stored_responses(responses),
        **_aimed(spec),
        **({"constraints": limits} if limits else {}),
    }
    coded = _coded(runs, stored["factors"]).to_numpy(dtype=float)
    kinds = ["categorical" if f.type == "categorical" else "numeric" for f in factors]
    stored["model"] = _largest_model(coded, kinds, blocks)
    if blocks:
        runs.insert(0, BLOCK, blocks)
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", range(1, len(runs) + 1))
    for response in responses:
        runs[response["name"]] = np.nan
    groups = _aliased_groups(coded, names, [f["name"] for f in stored["factors"] if _several(f)])
    said = (
        _levels_sentence(stored["factors"][0], coded[:, 0], given=True)
        if len(names) == 1
        else _aliasing_sentence(groups)
    )
    relation, words = _relation_of_runs(coded, names) if len(names) > 1 else ([], 0)
    notes = [
        GIVEN_NOTE,
        *([_relation_sentence(relation, words)] if relation else []),
        said,
        *([_blocks_sentence(coded, blocks, stored)] if blocks else []),
    ]
    if limits:
        notes.append(_given_limits_sentence(factors, limits, coded))
    facts = {
        "resolution": min(w.lstrip("-").count("·") + 1 for w in relation) if relation else None,
        "generators": [],
        "defining_relation": [f"I = {w}" for w in relation] if len(relation) == words else [],
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


def _given_settings(rows: list[list], factors: list[Factor]) -> pd.DataFrame:
    """Read the runs as given, a setting per factor: each numeric factor's a number, a categorical one's a label.

    A categorical factor's setting is written as its level's label, so a level 2 given as the number
    2.0, as a spreadsheet reads it, is the label 2. A setting at none of its levels is refused.
    """
    runs = pd.DataFrame(rows, columns=[f.name for f in factors], dtype=object)
    for f in factors:
        if f.type == "categorical":
            levels = [str(v) for v in f.levels or []]
            runs[f.name] = [levels[i] for i in _level_indices(runs[f.name], {"name": f.name, "levels": levels})]
            continue
        values = [_number(v) for v in runs[f.name]]
        if None in values:
            raise ValueError(f"Factor {f.name}: each run's setting must be a number.")
        runs[f.name] = values
    return runs


def _given_limits_sentence(factors: list[Factor], limits: list[dict], coded: np.ndarray) -> str:
    """Say which runs as given, coded, the limits on the settings rule out, or that every run meets them.

    The limits alone are checked: a given run may lie beyond a factor's range, as an axial run does, and still
    meet them.
    """
    names = [f.name for f in factors]
    region = _region(factors, [limit["expression"] for limit in limits])
    points = coded[:, [names.index(name) for name in region.names]]
    meets = np.all([g(points) <= LIMIT_TOLERANCE for g in region.inequalities], axis=0)
    outside = np.flatnonzero(~meets).tolist()
    return _outside_limits_sentence(outside) if outside else INSIDE_LIMITS


def _blocks_sentence(coded: np.ndarray, blocks: list, spec: dict) -> str:
    """Say how a design's runs, coded, fall into their blocks, and which terms of its model the blocks confound."""
    names = [f["name"] for f in spec["factors"]]
    categorical = _categorical_names(spec)
    frame = pd.DataFrame(coded, columns=names).assign(**{BLOCK: list(blocks)})
    terms = _model_terms(spec["model"], names, categorical)
    sizes = pd.Series(list(blocks)).value_counts().sort_index().tolist()
    return _blocking_sentence(sizes, [_pretty(t) for t in _confounded_with_blocks(frame, terms, categorical)])


def _given_blocks(raw: object, n_runs: int) -> list[int]:
    """Read the block of each run as given, a whole number from 1 per run; none (an empty list) for runs in no blocks.

    Runs taken as they are given can come in blocks: a study Learn added runs to holds its later runs as a
    block of their own, and keeps them when its runs are made again. One block is no blocking.
    """
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) != n_runs:
        raise ValueError("The runs as given need one block for each run, or none at all.")
    blocks = [_number(b) for b in raw]
    if not all(b is not None and b >= 1 and b == int(b) for b in blocks):
        raise ValueError("Each run's block must be a whole number from 1.")
    blocks = [int(b) for b in blocks]  # type: ignore[arg-type]
    if len(set(blocks)) > MAX_GIVEN_BLOCKS:
        raise ValueError(f"Runs as given come in at most {MAX_GIVEN_BLOCKS} blocks on this page.")
    if n_runs > 1 and len(set(blocks)) == n_runs:
        raise ValueError(
            "Each run is in a block of its own, so the blocks leave nothing to compare: give the runs in fewer blocks."
        )
    return blocks if len(set(blocks)) > 1 else []


def _mixture_design(spec: dict) -> dict:
    """Make a mixture design: its runs are blends, in proportions that sum to 1, in randomised order.

    The run sheet lists each component's proportion as it is to be weighed out. The design's
    model is the Scheffé model it supports: linear for a lattice of degree 1, else quadratic, and
    for extreme vertices the model they were built for. A mixture takes no limits on the settings:
    its bounds are each component's lower and upper proportions.
    """
    design_type = spec["design_type"]
    if spec.get("constraints"):
        raise ValueError(
            "A mixture takes no limits on the settings on this page: give each component its lower and upper "
            "proportions instead, or remove the limits."
        )
    components = _components(spec)
    responses = _checked_responses(spec, components, blank="y")
    kwargs = _mixture_options(spec, design_type)
    result = generate_design(components, design_type=MIXTURE, **kwargs)
    names = [c.name for c in components]
    blends = np.array([[_proportion_shown(v) for v in row] for row in result.design_actual[names].to_numpy(float)])
    runs = pd.DataFrame(blends, columns=names)
    runs.insert(0, "Run order", range(1, len(runs) + 1))
    runs.insert(0, "Std order", result.run_order)
    for response in responses:  # a column for each response's results, in the brief's order
        runs[response["name"]] = np.nan
    model_type = kwargs["model_type"]
    stored = {
        "version": SPEC_VERSION,
        "design_type": design_type,
        "factors": [_factor_spec(c) for c in components],
        "response": responses[0]["name"],
        "model": "scheffe_quadratic" if design_type == "simplex_centroid" else model_type,
        "options": kwargs,
        **_stored_responses(responses),
        **_aimed(spec),
    }
    facts = {
        "resolution": None,
        "generators": [],
        "defining_relation": [],
        "aliases": [],
        "center_runs": 0,
        "notes": _mixture_notes(design_type, blends, result.metadata or {}, kwargs),
    }
    return {
        "n_runs": len(runs),
        "columns": list(runs.columns),
        "rows": [list(row) for row in runs.itertuples(index=False)],
        "outside_range": [],
        "model": stored["model"],
        "facts": facts,
        "spec": stored,
        "xlsx_base64": _to_workbook(runs, stored),
    }


def _proportion_shown(value: float) -> float:
    """Round a proportion as the run sheet lists it, to :data:`PROPORTION_DIGITS` significant digits; -0 reads 0."""
    return float(f"{value:.{PROPORTION_DIGITS}g}") + 0.0


def _mixture_options(spec: dict, design_type: str) -> dict[str, Any]:
    """Pick the ``generate_design`` keyword arguments of a mixture design: its seed, its model and its replicates.

    The simplex centroid is the library's design for the special cubic model, which in three or
    four components is every blend of equal parts.
    """
    kwargs: dict[str, Any] = {"random_state": int(spec.get("random_state", spec.get("random_seed", RUN_ORDER_SEED)))}
    if design_type == "simplex_lattice":
        degree = _number(spec.get("lattice_degree", DEFAULT_LATTICE_DEGREE))
        if degree not in LATTICE_DEGREES:
            raise ValueError("A simplex lattice's degree is 1 or 2 on this page.")
        kwargs["model_type"] = LATTICE_DEGREES[int(degree)]  # type: ignore[arg-type]
    elif design_type == "simplex_centroid":
        kwargs["model_type"] = "scheffe_special_cubic"
    else:
        model = spec.get("mixture_model", VERTICES_MODELS[0])
        if model not in VERTICES_MODELS:
            raise ValueError(f"Extreme vertices are built for the linear or quadratic blending model, not {model!r}.")
        kwargs["model_type"] = model
    if (replicates := _replicates(spec)) > 1:
        kwargs["n_replicates"] = replicates
    return kwargs


def _mixture_notes(design_type: str, blends: np.ndarray, metadata: dict, kwargs: dict[str, Any]) -> list[str]:
    """Say what a mixture design's runs are: the kinds of blend, by how many components each holds.

    No note names a component, so a renaming keeps them all.
    """
    q = blends.shape[1]
    distinct = np.unique(blends.round(9), axis=0)
    notes = ["Each run is a blend: its proportions sum to 1."]
    if design_type == "simplex_lattice":
        degree = next(d for d, m in LATTICE_DEGREES.items() if m == kwargs["model_type"])
        lattice = distinct[(distinct > PROPORTION_TOLERANCE).sum(axis=1) < q]
        notes.append(
            f"The {{{q}, {degree}}} simplex lattice: {_listed(_blend_kinds(lattice))}. The overall centroid, 1/{q} "
            "of each component, is added to check the fit."
        )
    elif design_type == "simplex_centroid":
        notes.append(f"The simplex centroid: {_listed(_blend_kinds(distinct))}.")
    elif metadata.get("method") == "extreme_vertices":
        vertices = int(metadata.get("n_vertices", 0))
        edges = len(distinct) - vertices - 1
        between = f", the {edges} midpoints of the edges between them" if edges > 0 else ""
        notes.append(f"The {vertices} extreme vertices of the blends your bounds allow{between}, and their centroid.")
    else:
        notes.append(
            f"Your bounds leave {metadata.get('n_vertices')} extreme vertices, more than the model needs: the runs "
            "are a D-optimal choice among those vertices, the midpoints of the edges between them and their centroid."
        )
    replicates = kwargs.get("n_replicates", 1)
    if replicates > 1:
        notes.append(f"Each blend appears {replicates} times, so the repeats estimate the pure error.")
    return notes


def _blend_kinds(blends: np.ndarray) -> list[str]:
    """Name the kinds of blend of equal parts among distinct runs: the pure components, the blends of two, ..."""
    q = blends.shape[1]
    sizes = (blends > PROPORTION_TOLERANCE).sum(axis=1)
    kinds = []
    for size in sorted(set(sizes.tolist())):
        n = int((sizes == size).sum())
        if size == 1:
            kinds.append(f"the {n} pure components")
        elif size == q:
            kinds.append(f"the overall centroid, 1/{q} of each component")
        else:
            kinds.append(f"the {n} blends of {PARTS[size]} at 1/{size} each")
    return kinds


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
    they take numeric factors only, and two of them or more; so the refusals come in that order,
    which names the real conflict: a full factorial and a D-optimal design would otherwise each send
    the reader to the other. ``design_type`` is the design asked for, or None when none is asked for yet.
    """
    if labelled := [f.name for f in factors if f.type == "categorical"]:
        verb, which = ("is", "that factor") if len(labelled) == 1 else ("are", "those factors")
        raise ValueError(
            f"Limits on the settings work with numeric factors only on this page, and {_listed(labelled)} {verb} "
            f"categorical: remove the limits, or {which}."
        )
    if len(factors) == 1:  # the optimal designs, which keep to limits, take two factors or more
        raise ValueError(
            f"A limit on {factors[0].name} alone narrows its range: set its low and high to the settings you can "
            "run, and remove the limit."
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


def _aimed(spec: dict) -> dict[str, Any]:
    """Return the model the study aims to learn, cleaned: one of the comparison's models, or nothing.

    It is optional: a design made before the page asked for it has none, and keeps working. One
    this page does not know is left out.
    """
    return {"aim": spec["aim"]} if spec.get("aim") in COMPARISON_MODELS else {}


def _described(spec: dict) -> dict[str, Any]:
    """Return the specification's description of the first response and the model it aims to learn, cleaned."""
    info = _response_info(spec.get("response_info"))
    return {**({"response_info": info} if info else {}), **_aimed(spec)}


def _checked_responses(spec: dict, factors: list[Factor], *, blank: str = "") -> list[dict[str, Any]]:
    """Check the responses a specification names, and return each as its column's ``name`` and its description.

    A study made before the page measured several responses names one, ``response``, described by
    ``response_info``. One with several lists them all in ``responses``, up to :data:`MAX_RESPONSES`, each
    ``{name, ...}`` with its description beside its name, and the first of them is ``response``. Each name
    is a variable name unlike every factor's, the blocks column's and every other response's. ``blank`` is
    the name a blank first response takes when a design is made, as it always has; elsewhere it is refused.
    """
    raw = spec.get("responses")
    if raw is None:
        name = _response_name(str(spec.get("response") or blank).strip(), factors)
        info = _response_info(spec.get("response_info"))
        return [{"name": name, **info}]
    if not isinstance(raw, list) or not raw or not all(isinstance(r, dict) for r in raw):
        raise ValueError("The page sent a list of responses this page did not write.")
    if len(raw) > MAX_RESPONSES:
        raise ValueError(f"A study measures at most {MAX_RESPONSES} responses; this one has {len(raw)}.")
    responses: list[dict[str, Any]] = []
    for r in raw:
        name = _response_name(str(r.get("name") or "").strip(), factors)
        if any(name == other["name"] for other in responses):
            raise ValueError(f"Two responses cannot share the name {name}.")
        responses.append({"name": name, **_response_info({key: v for key, v in r.items() if key != "name"})})
    if spec.get("response") not in (None, "", responses[0]["name"]):
        raise ValueError("The page sent a list of responses this page did not write.")
    return responses


def _stored_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Return what a specification keeps of its responses, besides the first one's name (``response``).

    The first response's description is its ``response_info``, as a study of one response has always
    kept it, so a study of one response is stored as before. With several, ``responses`` lists them all.
    """
    info = {key: value for key, value in responses[0].items() if key != "name"}
    return {**({"response_info": info} if info else {}), **({"responses": responses} if len(responses) > 1 else {})}


def _responses(spec: dict) -> list[dict[str, Any]]:
    """Return the responses of a checked specification, the first first, each its column's ``name`` and description."""
    return spec.get("responses") or [{"name": spec["response"], **(spec.get("response_info") or {})}]


def _for_response(spec: dict, name: object) -> dict[str, Any]:
    """Return a checked specification as one response's analysis reads it: with that response as its ``response``.

    The analysis, its charts and the best settings read the response from ``response`` and
    ``response_info``. ``name`` is the column of the response the page asks about; None asks about the
    first. A workbook keeps the model each response was last analysed with, the first's as
    ``analysis_model`` and the others' in ``analysis_models``: the response's own is its ``analysis_model``.
    """
    if name is None or name == spec["response"]:
        return spec
    found = next((r for r in _responses(spec) if r["name"] == name), None)
    if found is None:
        raise ValueError(f"The study measures no response named {name!r}.")
    info = {key: value for key, value in found.items() if key != "name"}
    view = {key: value for key, value in spec.items() if key not in ("response_info", "analysis_model")}
    if (model := (spec.get("analysis_models") or {}).get(name)) is not None:
        view["analysis_model"] = model
    return {**view, "response": name, **({"response_info": info} if info else {})}


def _check_factor_count(design_type: str, n_factors: int) -> None:
    """Refuse a design the factor count rules out, saying which counts it takes."""
    info = DESIGNS[design_type]
    low, high = info["min_factors"], info["max_factors"]
    if not low <= n_factors <= high:
        raise ValueError(
            f"The {info['label']} design takes {low} to {high} factors on this page; you have {n_factors}."
        )


def _designed_model(design_type: str, resolution: int | None, factors: list[Factor], aim: str | None = None) -> str:
    """Name the model a design is analysed with when the reader does not choose one.

    A fraction keeps two-factor interactions clear of main effects and of each other
    only from resolution V, so below that it is analysed for its main effects. A Taguchi
    array estimates the main effects, and with a numeric factor at three levels (``factors``
    as the library made the design) each numeric factor's square too. A space-filling design
    is made for no model, so it is analysed with the smallest model that holds the study's
    ``aim``, as an optimal design is made for. One factor has no interaction, so its design is
    analysed for its main effect.
    """
    if len(factors) == 1:
        return "main_effects"
    if design_type in SPACE_FILLING and aim in OPTIMAL_MODEL:
        return OPTIMAL_MODEL[aim]
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


def _check_categorical(design_type: str, factors: list[Factor]) -> None:
    """Refuse a design that cannot hold the brief's categorical factors, saying which designs can."""
    if (labelled := [f.name for f in factors if f.type == "categorical"]) and design_type not in CATEGORICAL_DESIGNS:
        verb = "is" if len(labelled) == 1 else "are"
        raise ValueError(
            f"The {DESIGNS[design_type]['label']} design needs numeric factors, and {_listed(labelled)} {verb} "
            "categorical: choose a full factorial, fractional factorial, Plackett-Burman or Taguchi design."
        )
    if (several := _several_names(factors)) and design_type not in SEVERAL_LEVEL_DESIGNS:
        verb = "has" if len(several) == 1 else "have"
        raise ValueError(
            f"The {DESIGNS[design_type]['label']} design runs each categorical factor at two levels, and "
            f"{_listed(several)} {verb} more: choose a full factorial or a Taguchi design."
        )


def _several_names(factors: list[Factor]) -> list[str]:
    """Name the categorical factors of three levels or more."""
    return [f.name for f in factors if f.type == "categorical" and len(f.levels or []) > 2]


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
    kwargs: dict[str, Any] = {"random_state": int(spec.get("random_state", spec.get("random_seed", RUN_ORDER_SEED)))}
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
        kwargs["model_type"] = _model_type(spec)
    if "budget" in allowed:
        kwargs["budget"] = _budget(spec, design_type, n_factors, kwargs.get("model_type"))
    if "n_replicates" in allowed and (replicates := _replicates(spec)) > 1:
        kwargs["n_replicates"] = replicates
    if "n_blocks" in allowed and (blocks := _blocks(spec)) > 1:
        kwargs["n_blocks"] = blocks
    return kwargs


def _model_type(spec: dict) -> str:
    """Read the model an optimal design is made for: one of :data:`MODELS`, the quadratic unless named."""
    model_type = spec.get("model_type", "quadratic")
    if model_type not in MODELS:
        raise ValueError(f"Unknown model {model_type!r}.")
    return model_type


def _budget(spec: dict, design_type: str, k: int, model_type: str | None) -> int:
    """Read the runs of a design that takes a number of them, or give its own (logic.js's ``defaultBudget``).

    A space-filling design has :func:`_space_filling_runs`; an optimal one the reader's number, else the
    coefficients of the model it is made for and :data:`SPARE_RUNS` more.
    """
    if design_type in SPACE_FILLING:
        return _space_filling_runs(spec, k)
    budget = spec.get("budget")
    return int(budget) if budget else _n_terms(k, model_type) + SPARE_RUNS


def _space_filling_runs(spec: dict, k: int) -> int:
    """Read a space-filling design's runs: :data:`SPACE_FILLING_RUNS` a factor, unless the reader gives a number.

    The fewest is one more than the factors, which the main effects take to fit.
    """
    budget = spec.get("budget")
    if budget is None or budget in ("", 0):
        return SPACE_FILLING_RUNS * k
    runs = _number(budget)
    if runs is None or runs != int(runs) or not k + 1 <= runs <= MAX_SPACE_FILLING_RUNS:
        raise ValueError(f"A space-filling design in {k} factors has {k + 1} to {MAX_SPACE_FILLING_RUNS} runs.")
    return int(runs)


def _round_spread(runs: pd.DataFrame, factors: list[Factor], budget: int) -> None:
    """Round a space-filling design's settings, in place, to what can be set: to a tenth of their step, or finer.

    A Latin hypercube cuts each factor's range into ``budget`` equal steps and sets one run in the middle of each,
    such as 12.833... L/min. Rounded to the decimal place of a tenth of a step, each setting moves by at most a
    twentieth of a step, so it stays in its own step, and the settings keep their order and stay apart.
    """
    for f in factors:
        step = (f.high - f.low) / budget
        decimals = max(0, math.ceil(1 - math.log10(step)))
        runs[f.name] = runs[f.name].astype(float).round(decimals) + 0.0  # -0.0 reads 0


def _spread_sentence(runs: int) -> str:
    """Say how a maximin Latin hypercube spreads its runs: every factor's settings, and the runs among themselves."""
    return (
        f"Each factor takes {runs} different settings, evenly spaced across its range, and the runs are placed so "
        "the nearest two are as far apart as the search could make them."
    )


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
    """Read how many times each run is made, center runs excepted: 1 (no repeats) to :data:`MAX_REPLICATES`.

    A design in one factor makes its two levels that many times each, up to :data:`MAX_LEVEL_RUNS`.
    """
    most = MAX_LEVEL_RUNS if len(spec.get("factors") or []) == 1 else MAX_REPLICATES
    replicates = _number(spec.get("n_replicates", 1))
    if replicates is None or replicates != int(replicates) or not 1 <= replicates <= most:
        raise ValueError(f"Replicates must be a whole number from 1 to {most}.")
    return int(replicates)


def _levels(spec: dict) -> int:
    """Read the levels a Taguchi array runs each numeric factor at: one of :data:`TAGUCHI_LEVELS`."""
    levels = _number(spec.get("n_levels", TAGUCHI_LEVELS[0]))
    if levels not in TAGUCHI_LEVELS:
        raise ValueError("A Taguchi array runs each numeric factor at 2 levels or at 3 on this page.")
    return int(levels)  # type: ignore[arg-type]


def _blocks(spec: dict, most: int = MAX_BLOCKS) -> int:
    """Read how many blocks the runs are made in: 1 (no blocks) to ``most``, :data:`MAX_BLOCKS` unless said."""
    blocks = _number(spec.get("n_blocks", 1))
    if blocks is None or blocks != int(blocks) or not 1 <= blocks <= most:
        raise ValueError(f"Blocks must be a whole number from 1 to {most}.")
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
    """Read the resolution asked for, which can be at most the number of factors.

    Past :data:`MAX_INTERACTION_FACTORS` factors it is at most IV: the page models no interactions there, and a
    fraction of resolution V is made to estimate them.
    """
    highest = min(max(RESOLUTIONS), n_factors)
    if n_factors > MAX_INTERACTION_FACTORS:
        highest = min(highest, INTERACTIONS_CLEAR_FROM - 1)
    resolution = int(spec.get("resolution") or min(DEFAULT_RESOLUTION, highest))
    if not min(RESOLUTIONS) <= resolution <= highest:
        allowed = " to ".join(dict.fromkeys(ROMAN[r] for r in (min(RESOLUTIONS), highest)))
        raise ValueError(
            f"A fractional factorial in {n_factors} factors can have resolution {allowed} "
            f"on this page, not {ROMAN.get(resolution, resolution)}."
        )
    return resolution


def _design_facts(result: Any, coded: np.ndarray, factors: list[dict], kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """State what the generated design is, read off the design rather than its type.

    A fraction's resolution and generators come from the library; which main effects and
    two-factor interactions are fully aliased is read from the coded runs, so it holds for
    every design type and every naming of the factors. A design in one factor has nothing to
    alias, and says instead how many runs it makes at each level. ``notes`` are the sentences
    the page shows.
    """
    names = [f["name"] for f in factors]
    groups = _aliased_groups(coded, names, [f["name"] for f in factors if _several(f)])
    notes = []
    if result.resolution:
        notes.append(_resolution_sentence(result.resolution, [_spaced(g, names) for g in result.generators or []]))
    replicates = kwargs.get("n_replicates", 1)
    notes.append(_levels_sentence(factors[0], coded[:, 0]) if len(names) == 1 else _aliasing_sentence(groups))
    if replicates > 1 and len(names) > 1:
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


def _aliased_groups(coded: np.ndarray, names: list[str], several: Collection[str] = ()) -> list[list[str]]:
    """Group the main effects and two-factor interactions the design cannot tell apart.

    Two terms are fully aliased when their centred columns are parallel, so the data
    determine only their sum (or difference). A partial correlation, such as the 1/3
    between a main effect and an interaction in a 12-run Plackett-Burman design, is not
    aliasing in this sense. Each group lists its main effect first. A categorical factor of
    three levels or more (``several``) is no one column, and is left out; the designs that
    hold one, the full factorial and the Taguchi arrays, keep its main effect clear.
    """
    keep = [j for j, n in enumerate(names) if n not in several]
    coded, names = coded[:, keep], [names[j] for j in keep]
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


def _levels_sentence(factor: dict, coded: np.ndarray, *, given: bool = False) -> str:
    """Say how a design in one factor compares its levels: how many runs at its low, at its high and at its center.

    ``coded`` is the factor's coded setting in each run. A design the page makes runs in random order;
    runs as given (``given``) keep their order, so the sentence does not claim one. A categorical
    factor of three levels or more is counted at each of its levels.
    """
    order = "" if given else ", in random order"
    if _several(factor):
        levels = _levels_of(factor)
        codes = np.linspace(-1.0, 1.0, len(levels))
        counts = [int(np.isclose(coded, c, atol=ALIAS_TOLERANCE).sum()) for c in codes]
        each = (
            f"{counts[0]} runs at each"
            if len(set(counts)) == 1
            else _listed(
                [f"{n} run{'s' if n != 1 else ''} at {level}" for n, level in zip(counts, levels, strict=True)]
            )
        )
        return f"The runs compare {factor['name']} at {_listed(levels)}: {each}{order}."
    low, high = factor["low"], factor["high"]
    categorical = _categorical(factor)
    units = "" if categorical or not factor.get("units") else f" {factor['units']}"
    at = {level: int(np.isclose(coded, level, atol=ALIAS_TOLERANCE).sum()) for level in (-1, 0, 1)}
    shown = (low, high) if categorical else (f"{low:.4g}", f"{high:.4g}")
    if at[-1] + at[0] + at[1] < len(coded):  # runs as given at other settings: a curve, not a comparison
        levels = len(np.unique(np.round(coded, 6)))
        return f"The runs set {factor['name']} at {levels} levels, from {shown[0]} to {shown[1]}{units}."
    each = f"{at[-1]} runs at each" if at[-1] == at[1] else f"{at[-1]} runs at {shown[0]} and {at[1]} at {shown[1]}"
    centre = ""
    if at[0]:
        middle = f"{(low + high) / 2:.4g}{units}"
        centre = f", and {at[0]} run{'s' if at[0] != 1 else ''} at its center, {middle}"
    return f"The runs compare {factor['name']} at {shown[0]} and at {shown[1]}{units}: {each}{centre}{order}."


def _relation_of_runs(coded: np.ndarray, names: list[str]) -> tuple[list[str], int]:
    """Return the defining relation of runs that make a regular two-level fraction, and its number of words.

    Each run must be a corner of the cube or its centre. The corners make a regular fraction 2^(k-p) when they
    are 2^(k-p) distinct runs on which the product of some of the factors' columns is the same run for run:
    each such product is a word of the relation, ``A·B·C·D``, or ``-A·B·C·D`` where it is -1 on every run, and
    there are 2^p - 1 of them. A full factorial has none, and nor has a design that is no such fraction, such
    as a Plackett-Burman design: for those, ``([], 0)``.

    The words come shortest first, and in the order of their factors: all of them, up to :data:`RELATION_LISTED`,
    else the :data:`RELATION_SHOWN` shortest. With a bit per factor, set where a run is at -1, a word is a set of
    factors with an even number of bits set in each run's difference from the first: the relation is the null space
    of those differences over GF(2), read by elimination, since the 2^k sets of factors are too many to try one by one.
    """
    corner = np.all(np.isclose(np.abs(coded), 1.0), axis=1)
    centre = np.all(np.abs(coded) < ALIAS_TOLERANCE, axis=1)
    k = len(names)
    if k < 3 or not np.all(corner | centre):
        return [], 0
    distinct = np.unique(np.sign(coded[corner]), axis=0)
    n = len(distinct)
    if n < 4 or n >= 2**k or n & (n - 1):
        return [], 0
    low = [sum(1 << j for j in range(k) if run[j] < 0) for run in distinct]
    rows = _reduced(v ^ low[0] for v in low)
    if 2 ** len(rows) != n:  # the runs are no coset of a subspace: not a regular fraction
        return [], 0
    count = 2 ** (k - len(rows)) - 1
    masks = _null_space(rows, k) if count <= RELATION_LISTED else _shortest_words(rows, k, RELATION_SHOWN)
    factors = {mask: [j for j in range(k) if mask >> j & 1] for mask in masks}
    words = [
        ("-" if (mask & low[0]).bit_count() % 2 else "") + "·".join(names[j] for j in factors[mask])
        for mask in sorted(masks, key=lambda mask: (mask.bit_count(), factors[mask]))
    ]
    return words, count


def _reduced(vectors: Iterable[int]) -> list[tuple[int, int]]:
    """Reduce vectors over GF(2), each an integer with a bit per factor, to the rows of their reduced echelon form.

    Each row comes with its pivot, the bit that is set in that row alone.
    """
    rows: list[tuple[int, int]] = []
    for given in vectors:
        vector = given
        for pivot, row in rows:
            if vector >> pivot & 1:
                vector ^= row
        if vector:
            pivot = vector.bit_length() - 1
            rows = [(p, row ^ vector if row >> pivot & 1 else row) for p, row in rows]
            rows.append((pivot, vector))
    return rows


def _null_space(rows: list[tuple[int, int]], k: int) -> list[int]:
    """Return every nonzero vector over GF(2) of ``k`` bits orthogonal to the rows of a reduced echelon form."""
    pivots = {pivot for pivot, _ in rows}
    basis = [
        1 << free | sum(1 << pivot for pivot, row in rows if row >> free & 1) for free in range(k) if free not in pivots
    ]
    vectors = [0]
    for vector in basis:
        vectors += [v ^ vector for v in vectors]
    return vectors[1:]


def _shortest_words(rows: list[tuple[int, int]], k: int, most: int) -> list[int]:
    """Return the ``most`` shortest nonzero vectors orthogonal to the rows, in the order of their bits.

    A set of factors is orthogonal when each row has an even number of its bits: when the factors' columns of the
    rows, each an integer with a bit per row, add to 0.
    """
    columns = [sum((row >> j & 1) << i for i, (_, row) in enumerate(rows)) for j in range(k)]
    found: list[int] = []
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            if not functools.reduce(operator.xor, (columns[j] for j in subset)):
                found.append(sum(1 << j for j in subset))
                if len(found) == most:
                    return found
    return found


def _relation_sentence(words: list[str], count: int) -> str:
    """Say the resolution and the defining relation of runs given as a regular fraction, its first words listed."""
    resolution = min(w.lstrip("-").count("·") + 1 for w in words)
    shown = " = ".join(words[:RELATION_SHOWN])
    more = f", and {count - RELATION_SHOWN} more words" if count > RELATION_SHOWN else ""
    return f"Resolution {ROMAN.get(resolution, resolution)}: the runs make the defining relation I = {shown}{more}."


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


def _listed(items: list[str], last: str = "and") -> str:
    """Join items as a sentence does: ``A``, ``A and B``, ``A, B and C``, or with ``or`` before the last."""
    return f"{', '.join(items[:-1])} {last} {items[-1]}" if len(items) > 1 else "".join(items)


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
    # A workbook can arrive from anywhere: its specification is checked as data. ``response`` names the response to
    # analyse, by its column; the first without it.
    spec = _for_response(_checked_spec(stored), request.get("response"))
    return _analysis(runs, spec, request.get("model") or spec.get("analysis_model") or spec["model"])


def _analysis(
    runs: pd.DataFrame, spec: dict, model: str, truth: list[dict] | None = None, *, control: object = None
) -> dict:
    """Fit the model to the completed runs of a run sheet, from a workbook or from the page.

    With the true process the results were simulated from, each coefficient also
    carries the true value of what it estimates (see :func:`_add_true_values`). A study of
    one categorical factor of three levels or more compares each level with ``control``,
    one of its levels, the first unless named (see :func:`_comparison`).
    """
    runs = _design_runs(runs)
    if _mixture(spec):
        return _mixture_analysis(runs, spec, model, truth)
    response = spec["response"]
    factors = spec["factors"]
    names = [f["name"] for f in factors]
    categorical = _categorical_names(spec)
    name = _model_name(model)
    model = _checked_model(model, names, categorical)
    fitted, columns = _fitted_model(model, spec)
    blocked = _blocked(spec)
    missing_cols = [c for c in [*names, *([BLOCK] if blocked else []), response] if c not in runs.columns]
    if missing_cols:
        raise ValueError(f"The Runs sheet is missing column(s): {', '.join(missing_cols)}.")

    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna()
    pending = runs.loc[~done, "Run order"].astype(int).tolist() if "Run order" in runs else []

    coded = _design_frame(runs, spec, columns)[done].reset_index(drop=True)
    n_terms = _n_terms(len(columns), fitted)
    if pending and done.sum() < n_terms:
        complete = int(done.sum())
        raise ValueError(
            f"The {name} model has {n_terms} coefficients, so it needs at least {n_terms} "
            f"completed runs, and {complete} {'is' if complete == 1 else 'are'} complete. "
            f"Still to run: {', '.join(map(str, pending))}."
        )

    y_done = y[done].reset_index(drop=True).rename(response)
    fit = _fit_estimable(coded, y_done, fitted, columns, categorical)
    out = fit["out"]
    zero = ROUND_OFF * float(y[done].abs().max())
    coefficients = [
        {
            **c,
            "coefficient": _snapped(c["coefficient"], zero),
            "label": _level_words(fit["labels"].get(c["term"], _pretty(c["term"])), spec),
            "aliases": fit["chains"].get(c["term"], []),
            "block": _block_contrast(c["term"], columns),
            "vif": fit["vifs"].get(c["term"]),  # none for the intercept
        }
        for c in out.get("coefficients", [])
    ]
    lenth = fit["lenth"]
    if lenth:  # an effect is twice its coefficient at most, so twice the coefficients' round-off
        lenth["effects"] = [{**e, "effect": _snapped(e["effect"], 2 * zero)} for e in lenth["effects"]]
    if truth:
        _add_true_values(coefficients, coded, fit["formula"], _true_coefficients(truth, columns))
    anova = [{**a, "source": fit["labels"].get(a["source"], _pretty(a["source"]))} for a in out.get("anova_table", [])]
    if columns != names:  # a factor of several levels is tested as one term, over its contrasts together
        anova = _grouped_anova(anova, coded, y_done, fit["formula"], spec)
    anova = [{**a, "source": _level_words(a["source"], spec)} for a in anova]
    notes = [_level_words(note, spec) for note in fit["notes"]]
    order = runs.loc[done, "Run order"].astype(int).tolist() if "Run order" in runs else None
    comparison = _comparison(
        _coded(runs, factors)[done].reset_index(drop=True),
        y[done].to_numpy(dtype=float),
        spec,
        order,
        control,
        runs.loc[done, BLOCK].tolist() if blocked else None,
    )
    if comparison and comparison["difference"] and truth:  # a two-level factor's coefficient is half its difference
        true = next((c.get("true_value") for c in coefficients if c["term"] == names[0]), None)
        comparison["difference"]["true"] = None if true is None else 2 * true
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
        "not_estimable": [_level_words(_pretty(t), spec) for t in fit["not_estimable"]],
        "notes": notes,
        "equation": _equation(_response_words(spec)[0], [c for c in coefficients if not c["block"]]),
        "comparison": comparison,
    }


def _grouped_anova(rows: list[dict], frame: pd.DataFrame, y: pd.Series, formula: str, spec: dict) -> list[dict]:
    """Test each term of a factor of several levels as one source, its contrasts together.

    The fit lists each contrast as a source of one degree of freedom. A factor of k levels is one
    source of k - 1, and so is its interaction with another factor. Its sum of squares is of Type II,
    as the fit's other rows are: how much the residual sum of squares grows when its columns are
    left out of the model without the terms that contain it (the catalyst's interaction with T,
    for the catalyst). Its F is that over its degrees of freedom, against the residual mean square
    of the whole model.
    """
    terms = [t.strip() for t in formula.rsplit("~", maxsplit=1)[-1].split("+") if t.strip() and t.strip() != BLOCK]
    sources: dict[str, list[str]] = {}
    for term in terms:
        sources.setdefault(_source_of(term, spec), []).append(term)
    grouped = {source: members for source, members in sources.items() if len(members) > 1}
    # A source within a grouped one, as Supplier is within Supplier:Batch when Batch is nested in it, is
    # worked out here too: the fit cannot tell that Supplier_I1:Batch contains Supplier.
    inner = {s for s in sources if any(set(g.split(":")) > set(s.split(":")) for g in grouped)}
    grouped |= {source: sources[source] for source in inner}
    if not grouped:
        return rows
    full = _sse(frame, y, terms)
    residual = next((r for r in rows if r["source"] == "Residual"), None)
    df_resid = int(residual["df"]) if residual else len(y) - len(terms) - 1
    mse = full / df_resid if df_resid > 0 else math.nan
    from scipy import stats  # noqa: PLC0415 - scipy is loaded with the library

    out, done = [], set()
    for row in rows:
        source = "Residual" if row["source"] == "Residual" else _source_of(str(row["source"]), spec)
        members = grouped.get(source)
        if not members:
            out.append(row)
            continue
        if source in done:
            continue
        done.add(source)
        held = set(source.split(":"))
        kept = [t for t in terms if not set(_source_of(t, spec).split(":")) > held]  # the terms that contain it out
        ss = _sse(frame, y, [t for t in kept if t not in members]) - _sse(frame, y, kept)
        mean_sq = ss / len(members)
        f_value = mean_sq / mse if mse > 0 else math.nan
        p_value = float(stats.f.sf(f_value, len(members), df_resid)) if math.isfinite(f_value) else None
        out.append(
            {
                **row,
                "source": source,
                "df": len(members),
                "sum_sq": ss,
                "mean_sq": mean_sq,
                "F": f_value,
                "p_value": p_value,
            }
        )
    return out


def _source_of(term: str, spec: dict) -> str:
    """Name the source a fitted term belongs to, over the factors: ``Catalyst_L1:T`` is part of ``Catalyst:T``."""
    several = {f["name"] for f in spec["factors"] if _several(f)}
    categorical = set(_categorical_names(spec))

    def factor(part: str) -> str:
        stem, join, j = part.rpartition(LEVEL_JOIN)
        if join and stem in several and j.isdigit():
            return stem
        stem, join, j = part.rpartition(INDICATOR_JOIN)
        return stem if join and stem in categorical and j.isdigit() else part

    return ":".join(factor(part) for part in term.replace("·", ":").split(":"))


def _sse(frame: pd.DataFrame, y: pd.Series, terms: list[str]) -> float:
    """Return the residual sum of squares of a least-squares fit of these terms, an intercept and any blocks."""
    columns = [np.ones(len(frame))]
    for term in terms:
        square = re.fullmatch(r"I\((\w+) \*\* 2\)", term)
        values = frame[square.group(1)] ** 2 if square else frame[term.split(":")].prod(axis=1)
        columns.append(values.to_numpy(float))
    if BLOCK in frame.columns:
        columns += list(pd.get_dummies(frame[BLOCK], drop_first=True, dtype=float).to_numpy().T)
    x = np.column_stack(columns)
    fitted = x @ np.linalg.lstsq(x, y.to_numpy(float), rcond=None)[0]
    return float(((y.to_numpy(float) - fitted) ** 2).sum())


#: The confidence of each interval the comparison of one factor's levels gives, as of the coefficients'.
CONFIDENCE = 0.95

#: The most arrangements of the runs the randomization test counts: every one up to this many, else this many
#: drawn at random. Its interval, which counts every arrangement, is given only then.
ARRANGEMENTS = 200_000


def _comparison(  # noqa: PLR0913 - the runs, their results and blocks, the study, its run order, and the control
    coded: pd.DataFrame,
    y: np.ndarray,
    spec: dict,
    order: list[int] | None,
    control: object = None,
    blocks: list | None = None,
) -> dict | None:
    """Compare one factor's levels: each level's results, and the differences between them.

    Each level lists its runs (their run order and result), their average, standard deviation and the
    interval for the average from those runs alone. With the runs at the low and the high only, two
    or more at each, ``difference`` is the high's average less the low's, tested four ways: the
    t-test with one variance for both levels (the model's own test), Welch's t-test with each level's
    own variance, the randomization test, which counts the arrangements of the runs the random order
    could have made, and Wilcoxon's rank-sum test. A categorical factor run at three levels or more
    has ``several`` instead (:func:`_several_levels`). A study of more factors has none.

    Runs in ``blocks`` of two, one at each level, are pairs: the difference is taken within each pair, which
    leaves the blocks' effects out of it (see :func:`_paired_difference`). Runs in other blocks have no
    comparison: the model's analysis of variance, with its blocks, compares the levels.
    """
    factors = spec["factors"]
    if len(factors) != 1:
        return None
    factor = factors[0]
    x = np.round(coded[factor["name"]].to_numpy(dtype=float), 6)
    pairs = _pairs(x, y, blocks) if blocks is not None else None
    if blocks is not None and pairs is None:
        return None
    runs = order if order is not None else list(range(1, len(y) + 1))
    levels = [
        _level(factor, setting, y[x == setting], [r for r, at in zip(runs, x, strict=True) if at == setting])
        for setting in np.unique(x)
    ]
    out: dict[str, Any] = {"factor": factor["name"], "levels": levels, "difference": None, "several": None}
    if len(levels) > 2 and _categorical(factor):  # two levels have their difference
        out["several"] = _several_levels(levels, control)
        return out
    seed = spec.get("options", {}).get("random_state", 0)
    if pairs is not None:
        out["difference"] = _paired_difference(*pairs, factor, seed) if len(pairs[0]) >= 2 else None
        return out
    low, high = y[x == -1], y[x == 1]
    if len(levels) == 2 and min(len(low), len(high)) >= 2:  # two levels, two runs each, for a variance
        out["difference"] = _difference(low, high, factor, seed)
    return out


def _pairs(x: np.ndarray, y: np.ndarray, blocks: list) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the low and the high result of each block that is a pair, in the blocks' order; None for other blocks.

    A pair is a block of two runs, one at each of the two levels. A block with one result in, its other run still
    to do, waits; any other block, such as two runs at one level, means the runs are not in pairs.
    """
    if not set(np.unique(x)) <= {-1.0, 1.0}:
        return None
    low, high = [], []
    for block in sorted(set(blocks)):  # in the blocks' order, as the code the page shows takes them
        at = [i for i, b in enumerate(blocks) if b == block]
        if len(at) == 1:
            continue
        if len(at) != 2 or sorted(x[at]) != [-1.0, 1.0]:  # a pair is two runs
            return None
        first, second = at if x[at[0]] < 0 else at[::-1]
        low.append(y[first])
        high.append(y[second])
    return np.array(low, dtype=float), np.array(high, dtype=float)


def _paired_difference(low: np.ndarray, high: np.ndarray, factor: dict, seed: int) -> dict[str, Any]:
    """Test the difference between a factor's high and low levels within each pair, three ways.

    Each pair's difference, its high result less its low, leaves out the block both runs share. The paired
    t-test asks whether the differences average zero, from their spread; the randomization test counts
    every pattern of signs the random choice, within each pair, of which run got which level could have
    made; Wilcoxon's signed-rank test does the same with the differences' ranks, ties at their average,
    so it is exact with ties too.
    """
    differences = high - low
    paired = ttest_paired(pd.Series(differences), conflevel=CONFIDENCE)
    tests: list[dict[str, Any]] = [
        {
            "test": "paired",
            "statistic": float(paired["t value"]),
            "df": int(paired["Degrees of freedom"]),
            "ci_low": float(paired["ConfInt: Lo"]),
            "ci_high": float(paired["ConfInt: Hi"]),
            "p_value": float(paired["p value"]),
        }
    ]
    found = randomization_test(differences, max_arrangements=ARRANGEMENTS, random_state=seed)
    interval: dict[str, Any] = {"ci_low": None, "ci_high": None, "confidence": None}
    if found["exact"]:
        with contextlib.suppress(ValueError):  # too few patterns of signs for a 95% interval: 5 pairs make 32
            made = randomization_interval(differences, confidence=CONFIDENCE)
            interval = {k: made[k] for k in interval}
    tests.append(
        {
            "test": "sign_flip",
            "p_value": found["p_value_two_sided"],
            "n_arrangements": found["n_arrangements"],
            "exact": found["exact"],
            **interval,
        }
    )
    moved = differences[differences != 0]  # a pair that does not differ has no sign, and drops out
    ranks = pd.Series(np.abs(moved)).rank().to_numpy() * np.sign(moved)
    signed = randomization_test(ranks, max_arrangements=ARRANGEMENTS, random_state=seed) if len(ranks) else None
    tests.append(
        {
            "test": "signed_rank",
            "statistic": float(ranks[ranks > 0].sum()),
            "p_value": signed["p_value_two_sided"] if signed else 1.0,
            "exact": signed["exact"] if signed else True,
        }
    )
    labels = (factor["low"], factor["high"]) if _categorical(factor) else (_actual(factor, -1), _actual(factor, 1))
    return {
        "estimate": float(differences.mean()),
        "from": labels[0],
        "to": labels[1],
        "true": None,
        "paired": len(differences),
        "tests": tests,
    }


def _several_levels(levels: list[dict], control: object = None) -> dict[str, Any] | None:
    """Compare three levels or more: the one-way analysis of variance, Tukey's pairs, and Dunnett's against a control.

    Every test rests on the pooled standard deviation of the runs about their level's average. The
    F test asks whether the levels' averages differ at all. Tukey's intervals (``pairs``), one for
    each pair of levels, hold every difference between pairs together 95% of the time, from the
    studentized range of the k averages (Tukey and Kramer's, for levels of unequal runs). Dunnett's
    intervals (``versus``) compare each level with the control alone, so there are fewer of them and
    each is narrower; the control is the level named, the first unless another is. None when the
    runs leave no degree of freedom for the error.
    """
    groups = [np.array([r["value"] for r in level["runs"]]) for level in levels]
    names = [level["level"] for level in levels]
    n = np.array([len(g) for g in groups])
    k, df = len(groups), int(n.sum()) - len(groups)
    if df < 1:
        return None
    from scipy import stats  # noqa: PLC0415 - scipy is loaded with the library

    means = np.array([g.mean() for g in groups])
    mse = sum(float(((g - g.mean()) ** 2).sum()) for g in groups) / df
    s = math.sqrt(mse)
    grand = float(np.concatenate(groups).mean())
    between = float((n * (means - grand) ** 2).sum()) / (k - 1)
    f_value = between / mse if mse > 0 else math.inf
    test = {
        "f_value": f_value,
        "df_between": k - 1,
        "df_within": df,
        "p_value": float(stats.f.sf(f_value, k - 1, df)) if math.isfinite(f_value) else 0.0,
        "std_dev": s,
    }
    q = studentized_range_value(CONFIDENCE, k, df)
    pairs = []
    for i, j in itertools.combinations(range(k), 2):
        se = s * math.sqrt((1 / n[i] + 1 / n[j]) / 2)
        difference = float(means[j] - means[i])
        pairs.append(
            {
                "from": names[i],
                "to": names[j],
                "estimate": difference,
                "ci_low": difference - q * se,
                "ci_high": difference + q * se,
                "p_value": float(stats.studentized_range.sf(abs(difference) / se, k, df)) if se > 0 else 0.0,
            }
        )
    c = next((i for i, name in enumerate(names) if str(name) == str(control)), 0)
    others = [i for i in range(k) if i != c]
    d = _dunnett_critical(n[others], n[c], df)
    versus = []
    for i in others:
        se = s * math.sqrt(1 / n[i] + 1 / n[c])
        difference = float(means[i] - means[c])
        versus.append(
            {
                "level": names[i],
                "estimate": difference,
                "ci_low": difference - d * se,
                "ci_high": difference + d * se,
                "differs": bool(abs(difference) > d * se),
            }
        )
    return {"test": test, "pairs": pairs, "control": names[c], "versus": versus, "q": q, "d": d}


def _dunnett_critical(runs: np.ndarray, control_runs: int, df: int) -> float:
    """Return Dunnett's critical value for levels of these runs against a control of ``control_runs``.

    Two comparisons with the control share its average, so they are correlated: level i's and level j's by
    sqrt(c_i c_j), with c_i = n_i / (n_i + n_0), which is one half when every level has as many runs as the
    control. The library's ``dunnett_value`` takes each level's c_i, so the value is exact for unequal runs too.
    """
    return dunnett_value(CONFIDENCE, len(runs), df, correlation=(runs / (runs + control_runs)).tolist())


def _level(factor: dict, setting: float, values: np.ndarray, runs: list[int]) -> dict[str, Any]:
    """Summarise one level's results: its runs, their average, standard deviation and the interval for the average."""
    name = _level_at(factor, setting) if _categorical(factor) else _actual(factor, setting)
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(_critical_t(n - 1) * sd / math.sqrt(n)) if sd is not None else None
    mean = float(values.mean())
    return {
        "level": name,
        "coded": float(setting),
        "n": n,
        "mean": mean,
        "std_dev": sd,
        "ci_low": None if half is None else mean - half,
        "ci_high": None if half is None else mean + half,
        "runs": [{"run": r, "value": float(v)} for r, v in zip(runs, values, strict=True)],
    }


def _actual(factor: dict, coded: float) -> float:
    """Return a numeric factor's setting in its own units from its coded value."""
    return float(f"{(factor['low'] + factor['high']) / 2 + coded * (factor['high'] - factor['low']) / 2:.6g}")


def _critical_t(df: int) -> float:
    """Return the two-sided critical value of Student's t at :data:`CONFIDENCE` on ``df`` degrees of freedom."""
    from scipy import stats  # noqa: PLC0415 - scipy is loaded with the library

    return float(stats.t.ppf(1 - (1 - CONFIDENCE) / 2, df))


def _difference(low: np.ndarray, high: np.ndarray, factor: dict, seed: int) -> dict[str, Any]:
    """Test the difference between a factor's high and low levels four ways; see :func:`_comparison`."""
    tests = []
    for name, equal in (("pooled", True), ("welch", False)):
        found = ttest_independent(pd.Series(low), pd.Series(high), conflevel=CONFIDENCE, equal_var=equal)
        tests.append(
            {
                "test": name,
                "statistic": found["t value"],
                "df": found["Degrees of freedom"],
                "ci_low": found["ConfInt: Lo"],
                "ci_high": found["ConfInt: Hi"],
                "p_value": found["p value"],
            }
        )
    found = randomization_test(low, high, max_arrangements=ARRANGEMENTS, random_state=seed)
    interval: dict[str, Any] = {"ci_low": None, "ci_high": None, "confidence": None}
    if found["exact"]:
        with contextlib.suppress(ValueError):  # too few arrangements for a 95% interval: 3 runs at each make 20
            made = randomization_interval(low, high, confidence=CONFIDENCE)
            interval = {k: made[k] for k in interval}
    tests.append(
        {
            "test": "randomization",
            "p_value": found["p_value_two_sided"],
            "n_arrangements": found["n_arrangements"],
            "exact": found["exact"],
            **interval,
        }
    )
    ranks = rank_sum_test(low, high, max_arrangements=ARRANGEMENTS)
    exact = "exact_p_value_less" in ranks
    p = min(1.0, 2 * min(ranks["exact_p_value_less"], ranks["exact_p_value_greater"])) if exact else ranks["p_value"]
    tests.append({"test": "rank_sum", "statistic": ranks["rank_sum_y"], "p_value": p, "exact": exact})
    labels = (factor["low"], factor["high"]) if _categorical(factor) else (_actual(factor, -1), _actual(factor, 1))
    return {
        "estimate": float(high.mean() - low.mean()),
        "from": labels[0],
        "to": labels[1],
        "true": None,
        "tests": tests,
    }


def _snapped(value: float | None, zero: float) -> float | None:
    """Return 0 for a value no larger than ``zero``, the fit's round-off, and the value otherwise."""
    return 0.0 if value is not None and abs(value) <= zero else value


# --------------------------------------------------------------------------- the analysis of a mixture
#: What the page asks of a mixture's fit: the library's mixture ANOVA, which tests the linear blending terms
#: together, the coefficients, the lack of fit, and each component's effect along its Cox direction.
MIXTURE_ANALYSES = ["anova", "coefficients", "lack_of_fit", "effects"]

#: What every analysis of a mixture says of its model, and of the two-level methods it does without.
MIXTURE_NOTES = [
    (
        "A mixture's proportions sum to 1, so its model has no intercept: each component's coefficient is the "
        "response to that component alone, and a blend's coefficient is positive where the blend gives more than "
        "its components on average, negative where it gives less."
    ),
    (
        "Lenth's method and the effects of two-level factors do not apply to a mixture: its components are "
        "proportions, which move together, so each component's effect is taken along its Cox direction instead."
    ),
]


def _mixture_analysis(runs: pd.DataFrame, spec: dict, model: object, truth: list[dict] | None = None) -> dict:
    """Fit a Scheffé model to a mixture's completed runs, in their proportions, which are not coded.

    The mixture ANOVA tests the linear blending terms together, whether the components blend
    alike, and each blending term on its own. Each component's effect is the change in the fitted
    response along its Cox direction, through the centroid of the design's blends (see
    :func:`_cox_effects`). A model the design cannot fit is refused, saying how many of its
    coefficients the runs can estimate.
    """
    response = spec["response"]
    names = [f["name"] for f in spec["factors"]]
    model = _mixture_model(model)
    missing_cols = [c for c in [*names, response] if c not in runs.columns]
    if missing_cols:
        raise ValueError(f"The Runs sheet is missing column(s): {', '.join(missing_cols)}.")
    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna()
    pending = runs.loc[~done, "Run order"].astype(int).tolist() if "Run order" in runs else []
    n_terms = _mixture_terms(len(names), model)
    if pending and done.sum() < n_terms:
        complete = int(done.sum())
        raise ValueError(
            f"The {_model_name(model)} model has {n_terms} coefficients, so it needs at least {n_terms} "
            f"completed runs, and {complete} {'is' if complete == 1 else 'are'} complete. "
            f"Still to run: {', '.join(map(str, pending))}."
        )
    blends = runs.loc[done, names].astype(float).reset_index(drop=True)
    out = _mixture_fit(blends, y[done].reset_index(drop=True).rename(response), model)
    zero = ROUND_OFF * float(y[done].abs().max())
    coefficients = [
        {
            **c,
            "coefficient": _snapped(c["coefficient"], zero),
            "label": _pretty(c["term"]),
            "aliases": [],
            "block": False,
        }
        for c in out["coefficients"]
    ]
    if truth:
        _add_true_values(coefficients, blends, model, _true_coefficients(truth, names))
    anova = [{**a, "source": _mixture_source(a["source"])} for a in out.get("anova_table", [])]
    notes = list(MIXTURE_NOTES)
    if out["model_summary"]["df_residual"] == 0:
        notes.append(
            "The model uses every run, so no degrees of freedom are left to estimate the error and nothing can "
            "be tested. Add runs, or fit fewer terms."
        )
    return {
        "spec": spec,
        "model": model,
        "formula": model,  # the model's name is what the library fits: a Scheffé model of every component
        "n_completed": int(done.sum()),
        "pending_runs": pending,
        "summary": out["model_summary"],
        "coefficients": coefficients,
        "anova": anova,
        "lack_of_fit": out.get("lack_of_fit"),
        "lenth": None,
        "not_estimable": [],
        "notes": notes,
        "equation": _equation(_response_words(spec)[0], coefficients),
        "effects": _cox_effects(out, names),
    }


def _mixture_fit(blends: pd.DataFrame, y: pd.Series, model: str, analyses: list[str] | None = None) -> dict:
    """Fit a Scheffé model to blends, refusing one the runs cannot estimate in full."""
    with _listening() as heard:
        out = analyze_experiment(blends, y, model=model, analysis_type=analyses or MIXTURE_ANALYSES, coding="actual")
    summary = out["model_summary"]
    if summary["rank_deficient"]:
        raise ValueError(
            f"This design cannot fit the {_model_name(model)} model: only {summary['model_rank']} of its "
            f"{summary['n_terms']} coefficients can be estimated from these {summary['n_obs']} runs. Choose a "
            "smaller model."
        )
    _relay(heard)
    return out


def _mixture_source(source: str) -> str:
    """Name a row of the mixture ANOVA as the page shows it: the linear blending terms, or a blend as ``A·B``."""
    return "Linear blending" if source == "Linear mixture" else _pretty(source)


def _cox_effects(out: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """List each component's effect along its Cox direction, with the span of its proportion the effect is over.

    On a component's Cox direction the others keep their proportions to each other, so only the
    blend's share of that component changes. The effect is the fitted response at the end of the
    span, less the response at its start, from the reference blend: the centroid of the design's
    blends (``reference``, the same for every component).
    """
    effects, spans, reference = (
        out.get("effects") or {},
        out.get("effect_range") or {},
        out.get("effect_reference") or {},
    )
    return [
        {"component": n, "effect": effects[n], "low": spans[n][0], "high": spans[n][1], "reference": reference[n]}
        for n in names
        if n in effects
    ]


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

    Each term fitted gets its variance inflation factor (see :func:`_vifs`), and a note
    names the terms whose VIF is above :data:`VIF_HIGH`: runs as given, or runs left
    blank, can leave the terms correlated, so that no single term looks active.
    """
    blocked = BLOCK in coded.columns and len(set(coded[BLOCK])) > 1
    formula, early = _first_formula(coded, model, names, categorical, blocked=blocked)
    with _listening() as heard:
        out = analyze_experiment(coded, y, model=formula, analysis_type=ANALYSES, coding="actual")
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
            out = analyze_experiment(coded, y, model=formula, analysis_type=ANALYSES, coding="actual")
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
    vifs = _vifs(coded, out, names)
    notes += _correlated_sentences(vifs, labels, names)
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
        "vifs": vifs,
    }


def _vifs(coded: pd.DataFrame, out: dict[str, Any], names: list[str]) -> dict[str, float]:
    """Return the variance inflation factor (VIF) of each term a fit estimated, its block contrasts included.

    A term's VIF is how many times larger the variance of its coefficient is than it would be
    were the term's column uncorrelated with the others over the runs fitted: 1 for the terms of
    an orthogonal design. It is ``evaluate_design``'s, on the same runs and the same terms: the
    diagonal of (X'X)^-1 times each column's centred sum of squares, the intercept left out.
    There a block contrast is named ``Block[S.1]``, ``Block[S.2]``, ..., where the fit names it
    ``Block1``, ``Block2``, ...; both take the blocks in the same order. A model the runs cannot
    fit by ``evaluate_design``'s reckoning of the rank has no VIF.
    """
    blocked = BLOCK in coded.columns and len(set(coded[BLOCK])) > 1
    formula = _formula(_fitted_terms(out, names), names, blocked=blocked)
    with _listening():  # a given run far beyond its range is warned about as uncoded, which it is not
        found: dict[str, float] = evaluate_design(coded, model=formula, metric="vif").get("vif") or {}
    if not found:
        return {}
    contrasts = [c["term"] for c in out["coefficients"] if _block_contrast(c["term"], names)]
    blocks = [value for key, value in found.items() if key.startswith(f"{BLOCK}[")]
    terms = {key: value for key, value in found.items() if not key.startswith(f"{BLOCK}[")}
    return {**terms, **dict(zip(contrasts, blocks, strict=True))}


def _correlated_sentences(vifs: dict[str, float], labels: dict[str, str], names: list[str]) -> list[str]:
    """Name the terms whose VIF is above :data:`VIF_HIGH`, say what that does to them, and what to do; or nothing.

    A block contrast is left out: its VIF says how unevenly the runs fill the blocks, which
    no term of the model can change.
    """
    high = [
        f"{labels.get(term, _pretty(term))} (VIF {_two_figures(vif)})"
        for term, vif in vifs.items()
        if vif > VIF_HIGH and not _block_contrast(term, names)
    ]
    if not high:
        return []
    subject = f"{_listed(high)} {'is' if len(high) == 1 else 'are'}"
    return [
        (
            f"{subject} correlated with other terms in these runs. A term's variance inflation factor, VIF, is how "
            "many times larger the variance of its coefficient is than with uncorrelated runs, so its standard "
            "error is about √VIF times as large. A term can then look inactive even while the model explains the "
            "response. Add runs that separate the correlated terms, or drop one of them from the model."
        )
    ]


def _two_figures(value: float) -> str:
    """Write a positive number to two significant figures, with no exponent: ``5.0``, ``9.8``, ``12``, ``1200``."""
    rounded = value
    for _ in range(2):  # again from the rounded value, which may have reached the next power of ten
        decimals = 1 - math.floor(math.log10(abs(rounded)))
        rounded = round(value, decimals)
    return f"{rounded:.{max(decimals, 0)}f}"


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
    is fitted as the list of its terms, the main effects and each numeric factor's square. Past
    :data:`MAX_INTERACTION_FACTORS` factors, a model with every two-factor interaction is refused.
    """
    if isinstance(model, dict) and set(model) == {"terms"}:
        return _chosen_terms(model["terms"], names, categorical)
    if model == MAIN_QUADRATIC:
        return _model_terms(MAIN_QUADRATIC, names, categorical)
    if model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    if model in INTERACTION_MODELS and (k := len(names)) > MAX_INTERACTION_FACTORS:
        raise ValueError(
            f"Past {MAX_INTERACTION_FACTORS} factors, the page fits the main effects, or the terms you choose: "
            f"{k} factors have {k * (k - 1) // 2} two-factor interactions."
        )
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
    """Name a model in a sentence: ``main effects``, ``quadratic blending``, or ``chosen`` for terms a reader chose."""
    if model == MAIN_QUADRATIC:
        return "main effects and squares"
    return MIXTURE_WORDS.get(model, model.replace("_", " ")) if isinstance(model, str) else CHOSEN


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
    if not isinstance(spec, dict) or spec.get("design_type") not in (*DESIGNS, GIVEN, *MIXTURES):
        raise ValueError("The page sent a design this page did not make.")
    factors = _factors(spec)
    responses = _checked_responses(spec, factors)
    if _mixture(spec):
        model = _mixture_model(spec.get("model") or MIXTURE_MODELS[1])
        if spec.get("constraints"):
            raise ValueError(
                "A mixture takes no limits on the settings on this page: give each component its lower and upper "
                "proportions instead, or remove the limits."
            )
    else:
        model = spec.get("model") or DESIGNS.get(spec["design_type"], {}).get("model", MODELS[0])
        if model not in (*MODELS, MAIN_QUADRATIC):  # the second, a model a design is made for: see _checked_model
            raise ValueError(f"Model must be one of {', '.join(MODELS)}.")
    options = spec.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("The page sent a design this page did not make.")  # noqa: TRY004 - a refusal, like the rest
    most = MAX_GIVEN_BLOCKS if spec["design_type"] == GIVEN else MAX_BLOCKS
    names = [f.name for f in factors]
    labelled = tuple(f.name for f in factors if f.type == "categorical")
    chosen = [spec["analysis_model"]] if "analysis_model" in spec else []  # last analysed with, as a workbook keeps
    if "analysis_models" in spec:  # and those of the other responses, by their columns
        others = spec["analysis_models"]
        if not isinstance(others, dict) or not set(others) <= {r["name"] for r in responses[1:]}:
            raise ValueError("The page sent a design this page did not make.")
        chosen += list(others.values())
    for one in chosen:
        if _mixture(spec):
            _mixture_model(one)
        else:
            _checked_model(one, names, labelled)
    dropped = ("response_info", "responses", "aim", "constraints", "what_if")  # read_workbook checks what_if itself
    checked = {key: value for key, value in spec.items() if key not in dropped}
    limits = _checked_constraints(spec.get("constraints"), factors)
    return {
        **checked,
        "factors": [_factor_spec(f) for f in factors],
        "response": responses[0]["name"],
        "model": model,
        "options": {**options, **({"n_blocks": blocks} if (blocks := _blocks(options, most)) > 1 else {})},
        **_stored_responses(responses),
        **_aimed(spec),
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
    """Rebuild the run sheet the page shows, with the results it holds in the responses' columns.

    ``responses`` holds the first response's results, one per run, as the page has always sent them;
    ``more_responses`` the others', a list per run for each of their columns.
    """
    columns = [str(c) for c in request.get("columns") or []]
    rows = request.get("rows") or []
    names = [f["name"] for f in spec["factors"]]
    missing = [n for n in names if n not in columns]
    if missing or not rows:
        raise ValueError(f"The run sheet is missing the factor column(s) {missing}." if missing else "No runs.")
    runs = pd.DataFrame(rows, columns=columns)
    numeric = [f["name"] for f in spec["factors"] if not _categorical(f)]  # a category's labels stay text
    runs[numeric] = runs[numeric].apply(pd.to_numeric, errors="raise")
    first, *others = [r["name"] for r in _responses(spec)]
    results = {first: request.get("responses")} if request.get("responses") is not None else {}
    more = request.get("more_responses") or {}
    if not isinstance(more, dict) or not set(more) <= set(others):
        raise ValueError("The page sent results for a response the study does not measure.")
    for name, values in {**results, **more}.items():
        if not isinstance(values, list) or len(values) != len(runs):
            sent = len(values) if isinstance(values, list) else 0
            raise ValueError(f"{sent} responses were sent for {len(runs)} runs.")
        runs[name] = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy()
    return runs


def _confirming(runs: pd.DataFrame) -> pd.Series:
    """Say which runs of a run sheet are confirmation runs: those its :data:`CONFIRMATION` column marks."""
    if CONFIRMATION not in runs.columns:
        return pd.Series(False, index=runs.index)
    return runs[CONFIRMATION].map(lambda mark: isinstance(mark, str) and mark.strip().lower() == CONFIRMS)


def _design_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the runs the model is fitted to: the run sheet less its confirmation runs, which test the model.

    A confirmation run is made at the settings the model recommends, to see whether the process
    gives what the model predicts there. Fitted to it, the model would be judged on a run it was
    made to fit, so every fit leaves the confirmation runs out, as if they were not on the sheet.
    """
    confirming = _confirming(runs)
    if confirming.all():
        raise ValueError(
            "Every run on the run sheet is marked as a confirmation run, and the model is fitted to the design's runs."
        )
    return runs.loc[~confirming].drop(columns=[CONFIRMATION], errors="ignore").reset_index(drop=True)


def _coded(runs: pd.DataFrame, factors: list[dict]) -> pd.DataFrame:
    """Code each factor's settings to -1 at its low value and +1 at its high value, a category's labels too."""
    return pd.DataFrame({f["name"]: _coded_column(runs[f["name"]], f) for f in factors})


def _coded_column(column: pd.Series, factor: dict) -> pd.Series:
    """Code one factor's settings; a categorical factor's runs must each be at one of its levels.

    A categorical factor of k levels is coded as the library codes it, its levels spread evenly from
    -1 to +1 in their order (``categorical_codes``): -1 and +1 for two, -1, 0 and +1 for three.
    """
    if not _categorical(factor):
        return (column - (factor["high"] + factor["low"]) / 2) / ((factor["high"] - factor["low"]) / 2)
    codes = np.linspace(-1.0, 1.0, len(_levels_of(factor)))
    return pd.Series(codes[_level_indices(column, factor)], index=column.index)


def _level_indices(column: pd.Series, factor: dict) -> np.ndarray:
    """Return which level each run of a categorical factor is at, refusing a run at none of them."""
    levels = _levels_of(factor)
    found = [_level_index(v, levels) for v in column]
    if None in found:
        other = column.iloc[found.index(None)]
        raise ValueError(f"Factor {factor['name']}: each run must be at {_listed(levels, 'or')}, not {other!r}.")
    return np.array(found, dtype=int)


def _design_frame(runs: pd.DataFrame, spec: dict, columns: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    """Return the runs as the analysis fits them: the coded settings, and each run's block if it has one.

    A categorical factor of three levels or more is fitted as its contrasts (:func:`_contrasts`), a
    column per level but the last, in place of its one coded column. A factor whose indicators the
    model's ``columns`` name has them too (:func:`_nested`).
    """
    frame = _coded(runs, spec["factors"])
    for f in spec["factors"]:
        if _several(f):
            index = _level_indices(runs[f["name"]], f)
            at = frame.columns.get_loc(f["name"])
            frame = frame.drop(columns=f["name"])
            for j, name in enumerate(_contrast_names(f)):
                frame.insert(at + j, name, _contrasts(index, len(_levels_of(f)))[:, j])
        if _categorical(f) and set(_indicator_names(f)) & set(columns):
            index = _level_indices(runs[f["name"]], f)
            for j, name in enumerate(_indicator_names(f)):
                frame[name] = _indicators(index, len(_levels_of(f)))[:, j]
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


def _scenario_process(  # noqa: PLR0913 - the scenario, the factors, the request, the reader's scale and the place
    key: str,
    names: list[str],
    request: dict,
    categorical: tuple[str, ...] = (),
    typical: float | None = None,
    noise: float | None = None,
    place: int = 0,
) -> TrueProcess:
    """Return the scenario's process over the reader's factors, with the noise the reader chose.

    A term over a factor the reader does not have is left out, and so is the square of a categorical
    factor, which takes one value, +1, at both of its levels. A scenario's effects are sizes in
    noise standard deviations; given the response's ``typical`` value and run-to-run ``noise``,
    the process is centred on the one and its effects scaled by the other, so practice results
    look like the reader's own. Without them the scenario's own centre and a noise of 1 are kept.

    ``place`` is the response's place in the brief, 0 for the first. The first response follows the
    scenario, with the noise the reader chose (``sigma``); a later one follows it moved to other factors
    (see :func:`_moved_terms`), with its own run-to-run noise, else the scenario's.
    """
    if key in MIXTURE_SCENARIOS:
        raise ValueError(
            f"The process {MIXTURE_SCENARIOS[key]['label']!r} is for a mixture's components: choose another for "
            "factors with ranges."
        )
    if key not in SCENARIOS:
        raise ValueError(f"Unknown process {key!r}.")
    scenario = SCENARIOS[key]
    terms = scenario["terms"]
    if terms is None:
        terms = _random_terms(len(names), np.random.default_rng(int(request.get("seed", 1))))
    kept = _moved_terms({t: c for t, c in terms.items() if all(i < len(names) for i in t)}, place, len(names))
    squares = {t for t in kept if len(t) == 2 and t[0] == t[1] and names[t[0]] in categorical}
    scale = 1.0 if noise is None else noise
    coefficients = {
        _term_name(t, names): (c if typical is None else typical) if not t else c * scale
        for t, c in kept.items()
        if t not in squares
    }
    if place:
        return TrueProcess(coefficients, sigma=scenario["sigma"] if noise is None else noise)
    return TrueProcess(coefficients, sigma=_noise(request, scenario["sigma"]))


def _noise(request: dict, default: float) -> float:
    """Read the noise standard deviation a simulation asks for, or ``default`` when it names none."""
    # A noise of 0 is refused, not read as "not given": the reader would see results with noise they did not ask for.
    sigma = default if request.get("sigma") is None else _number(request["sigma"])
    if sigma is None or sigma <= 0:
        raise ValueError("The noise standard deviation must be a number above 0.")
    return sigma


def _mixture_process(  # noqa: PLR0913 - as _scenario_process
    key: str,
    names: list[str],
    request: dict,
    typical: float | None = None,
    noise: float | None = None,
    place: int = 0,
) -> TrueProcess:
    """Return a mixture scenario's process over the reader's components, a Scheffé quadratic model in proportions.

    As for a process over factors with ranges (see :func:`_scenario_process`), the scenario's sizes
    are in noise standard deviations: given the response's ``typical`` value and run-to-run
    ``noise``, each pure component's response sits around the one, at the scenario's distances
    scaled by the other. A blend of a component the reader does not have is left out.

    ``place`` is the response's place in the brief, 0 for the first. A later response follows the
    scenario turned ``place`` components along, so a different component is best for it, with its own
    run-to-run noise, else the scenario's.
    """
    if key in SCENARIOS:
        labels = _listed([repr(s["label"]) for s in MIXTURE_SCENARIOS.values()])
        raise ValueError(
            f"The process {SCENARIOS[key]['label']!r} is for factors with ranges: a mixture's results are simulated "
            f"from {labels}."
        )
    if key not in MIXTURE_SCENARIOS:
        raise ValueError(f"Unknown process {key!r}.")
    scenario = MIXTURE_SCENARIOS[key]
    scale = 1.0 if noise is None else noise
    centre = scenario["centre"] if typical is None else typical
    k = len(names)

    def turned(i: int) -> int:
        return (i + place) % k

    coefficients = {name: centre + scenario["pure"].get((i - place) % k, 0.0) * scale for i, name in enumerate(names)}
    for (i, j), size in scenario["blends"].items():
        if j < k:
            first, second = sorted((turned(i), turned(j)))
            coefficients[f"{names[first]}:{names[second]}"] = size * scale
    if place:
        return TrueProcess(coefficients, sigma=scenario["sigma"] if noise is None else noise)
    return TrueProcess(coefficients, sigma=_noise(request, scenario["sigma"]))


def _moved_terms(terms: dict[tuple[int, ...], float], place: int, k: int) -> dict[tuple[int, ...], float]:
    """Move a scenario's terms to other factors, for the response in ``place`` (0 for the first) of ``k`` factors.

    The first response follows the scenario as it is. A later one follows it moved, so its best settings lie
    elsewhere and the responses have to be balanced: what the scenario puts on factor ``i`` acts on factor
    ``(i + place) mod k``, and for an odd ``place`` every factor's setting is reflected, high for low, which
    flips the sign of each term of odd order (a main effect, a three-factor interaction) and keeps the others
    (the intercept, a two-factor interaction, a square). Moving the factors round is one to one, so no two
    terms land on one.
    """
    if not place:
        return terms
    sign = -1.0 if place % 2 else 1.0
    return {tuple(sorted((i + place) % k for i in t)): c * sign ** len(t) for t, c in terms.items()}


def _truth(process: TrueProcess) -> list[dict[str, Any]]:
    """List a process's terms as the analysis lists its coefficients."""
    return [{"term": t, "label": _pretty(t), "coefficient": c} for t, c in process.coefficients.items()]


def simulate(request: dict) -> dict:
    """Simulate the results of the page's runs from a known process, in coded units, plus noise.

    The responses are rounded to the noise's precision, two significant decimals below
    its standard deviation, as a measurement would be.

    Each response has a process of its own (see :func:`_scenario_process`), and its noise is drawn
    with the seed plus :data:`SEED_STEP` times its place. The first response's results, process and
    noise are the reply's own keys, as for a study of one response; with several, ``more`` holds
    each later response's, by its column, with its ``seed``.
    """
    spec = _checked_spec(request.get("spec") or {})
    runs = _runs_frame(request, spec)
    names = [f["name"] for f in spec["factors"]]
    mixture = _mixture(spec)
    key = str(request.get("scenario") or ("blend" if mixture else "reactor"))
    seed = int(request.get("seed", 1))
    # A mixture's process is in the proportions themselves, which are not coded.
    settings = runs[names].astype(float) if mixture else _coded(runs, spec["factors"])
    simulated = []
    for place, response in enumerate(_responses(spec)):
        known = {"typical": response.get("typical"), "noise": response.get("noise"), "place": place}
        if mixture:
            process = _mixture_process(key, names, request, **known)
        else:
            process = _scenario_process(key, names, request, _categorical_names(spec), **known)
        values = process.simulate(settings, random_state=seed + SEED_STEP * place)
        decimals = max(0, 2 - math.floor(math.log10(process.sigma)))
        truth = _contrast_truth(_truth(process), spec)
        simulated.append(
            {
                "sigma": process.sigma,
                "seed": seed + SEED_STEP * place,
                "responses": [round(float(v), decimals) for v in values],
                "truth": truth,
                "equation": _equation(response.get("label") or response["name"], truth),
                "inactive": [name for name in names if name not in process.factors],  # factors the process leaves out
            }
        )
    first, *more = simulated
    label = (MIXTURE_SCENARIOS if mixture else SCENARIOS)[key]["label"]
    reply = {"scenario": key, "label": label, **first, "seed": seed}
    if more:
        reply["more"] = {r["name"]: one for r, one in zip(_responses(spec)[1:], more, strict=True)}
    return reply


def _contrast_truth(truth: list[dict], spec: dict) -> list[dict]:
    """Write a simulated process over the columns the model is fitted on.

    The process moves the response by its coefficient times a factor's coded setting, so a
    categorical factor of k levels moves it evenly across its levels, from -c at the first to +c at
    the last. Over the factor's contrasts (:func:`_contrasts`) that is the coefficient c times each
    level's code, for each level but the last, and the same for each of its interactions: a term's
    coefficient is spread over the products of its factors' contrasts, each times their levels' codes.
    """
    several = {f["name"]: f for f in spec["factors"] if _several(f)}
    if not several:
        return truth
    out: dict[str, float] = {}
    for t in truth:
        term, value = str(t["term"]), float(t["coefficient"])
        if term == "Intercept" or term.startswith("I("):
            out[term] = out.get(term, 0.0) + value
            continue
        choices = []
        for name in term.split(":"):
            if name in several:
                codes = np.linspace(-1.0, 1.0, len(_levels_of(several[name])))
                choices.append(list(zip(_contrast_names(several[name]), codes[:-1], strict=True)))
            else:
                choices.append([(name, 1.0)])
        for product in itertools.product(*choices):
            column = ":".join(name for name, _ in product)
            out[column] = out.get(column, 0.0) + value * math.prod(code for _, code in product)
    return [
        {"term": term, "label": _level_words(_pretty(term), spec), "coefficient": round(value, 12) + 0.0}
        for term, value in out.items()
    ]


def analyze_runs(request: dict) -> dict:
    """Fit the model to the runs the page holds, with the responses typed or simulated there.

    ``truth``, the process the results were simulated from, adds each coefficient's true value.
    ``response`` names the response to analyse, by its column; the first without it. Each response
    is fitted to the runs that have its result, so a run with some of its results in counts for those.
    """
    spec = _checked_spec(request.get("spec") or {})
    runs = _runs_frame(request, spec)
    spec = _for_response(spec, request.get("response"))
    model = request.get("model") or spec["model"]
    return _analysis(runs, spec, model, request.get("truth"), control=request.get("control"))


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
        noiseless = analyze_experiment(
            coded, pd.Series(mean, name="y"), model=formula, analysis_type="coefficients", coding="actual"
        )
    # Rounded to drop the rounding error of a fit with no noise (a true zero reads 1e-14).
    expected = {c["term"]: round(c["coefficient"], 9) + 0.0 for c in noiseless["coefficients"]}
    for c in coefficients:
        c["true_value"] = expected.get(c["term"])


def read_workbook(request: dict) -> dict:
    """Read an uploaded workbook into the runs, responses and sources the page holds.

    The first response's results and their sources are ``responses`` and ``sources``, as for a study of
    one response; with several, ``more_responses`` and ``more_sources`` hold each later one's, by its column.
    A run the workbook marks as simulated has every result it holds marked so. ``what_if`` holds the settings
    the reader kept from the what-if panel, when the workbook keeps any (see :func:`_kept_settings`).
    """
    runs, stored = _read_workbook(base64.b64decode(request["xlsx_base64"]))
    spec = _checked_spec(stored)
    names = [r["name"] for r in _responses(spec)]
    if missing := next((name for name in names if name not in runs.columns), None):
        raise ValueError(f"The Runs sheet has no {missing!r} column for the results.")
    column = _marks_column(runs, spec)
    marks = runs[column].fillna("").astype(str) if column else pd.Series("", index=runs.index)
    sheet = runs.drop(columns=[column] if column else [])
    results = {}
    for name in names:
        values = pd.to_numeric(runs[name], errors="coerce")
        sheet[name] = np.nan
        results[name] = (
            [None if pd.isna(v) else float(v) for v in values],
            [
                None if pd.isna(v) else ("simulated" if mark == "simulated" else "entered")
                for v, mark in zip(values, marks, strict=True)
            ],
        )
    (responses, sources), *_ = results.values()
    reply = {
        "spec": spec,
        "columns": list(sheet.columns),
        "rows": [list(row) for row in sheet.itertuples(index=False)],
        "responses": responses,
        "sources": sources,
    }
    if len(names) > 1:
        reply["more_responses"] = {name: results[name][0] for name in names[1:]}
        reply["more_sources"] = {name: results[name][1] for name in names[1:]}
    if stored.get("what_if"):
        reply["what_if"] = _kept_settings(stored["what_if"])
    return reply


#: The most settings a study keeps from the what-if panel, and the most characters of the name each is kept under:
#: as ``SCENARIOS_HELD`` in web/campaign.js holds them (tests/web/page/test_parity.py keeps the two in step).
SCENARIOS_HELD = {"count": 20, "name": 80}


def _kept_settings(value: object) -> list[dict[str, Any]]:
    """Check the settings the reader kept from the what-if panel, each under a name, as ``checkCampaign`` checks them.

    Only their shape is checked: a name of one line, each name once, and each setting a number or a
    level, by its factor's name. Whether they still fit the factors is judged where the page lists
    them, so a workbook edited since still opens; what they predict is worked out from its results.
    """
    if (
        not isinstance(value, list)
        or len(value) > SCENARIOS_HELD["count"]
        or not all(map(_kept_one, value))
        or len({one["name"] for one in value}) < len(value)  # each name once
    ):
        raise ValueError("The page sent kept settings this page did not write.")
    return [{"name": one["name"], "settings": dict(one["settings"])} for one in value]


def _kept_one(one: object) -> bool:
    """Say whether one kept setting is a name of one line and a number or a level for each factor it names."""
    if not isinstance(one, dict) or not isinstance(one.get("name"), str) or not isinstance(one.get("settings"), dict):
        return False
    name, settings = one["name"], one["settings"]
    return (
        bool(name.strip())
        and len(name) <= SCENARIOS_HELD["name"]
        and not any(unicodedata.category(ch) == "Cc" for ch in name)
        and len(settings) <= MAX_FACTORS
        and all(isinstance(k, str) and len(k) <= LABEL_MOST and _kept_value(v) for k, v in settings.items())
    )


def _kept_value(value: object) -> bool:
    """Say whether a kept setting is a finite number or a level's label."""
    if isinstance(value, str):
        return len(value) <= LABEL_MOST
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _marks_column(runs: pd.DataFrame, spec: dict) -> str | None:
    """Name the run sheet's column of marks, if it has one: an older workbook's ``Source``, unless that is data."""
    data = {*(r["name"] for r in _responses(spec)), *(f["name"] for f in spec["factors"])}
    return next((c for c in (MARKS, "Source") if c in runs.columns and c not in data), None)


def workbook(request: dict) -> dict:
    """Write the page's runs and responses to a workbook the page can analyse again.

    Simulated responses are marked, in the ``MARKS`` column and in the stored
    specification, so the workbook says which values were measured. The model the reader
    analyses with, ``model``, is kept as the specification's ``analysis_model``, so the
    workbook is analysed the same way when it comes back; with several responses, ``models``
    holds each later response's, by its column, kept as ``analysis_models``.

    A run is marked simulated when every result it holds is: ``sources`` holds the first
    response's sources, and ``more_sources`` each later response's, by its column. The settings the
    reader kept from the what-if panel, ``what_if``, are kept in the stored specification too.
    """
    spec = _checked_spec(request.get("spec") or {})
    if request.get("model"):
        if _mixture(spec):
            _mixture_model(request["model"])
        else:
            _checked_model(request["model"], [f["name"] for f in spec["factors"]], _categorical_names(spec))
        spec = {**spec, "analysis_model": request["model"]}
    if models := request.get("models"):  # each later response's, but one analysed as designed, as for the first
        chosen = {name: m for name, m in models.items() if m} if isinstance(models, dict) else models
        spec = _checked_spec({**spec, "analysis_models": chosen})
    runs = _runs_frame(request, spec)
    marks = [request.get("sources"), *(request.get("more_sources") or {}).values()]
    marks = [m for m in marks if m and any(m)]
    if marks:
        if bad := next((m for m in marks if not isinstance(m, list) or len(m) != len(runs)), None):
            raise ValueError(f"{len(bad)} sources were sent for {len(runs)} runs.")
        runs[MARKS] = [
            "simulated" if "simulated" in run and all(s in (None, "simulated") for s in run) else ""
            for run in zip(*marks, strict=True)
        ]
    simulated = request.get("simulated")
    if simulated:
        spec = {**spec, "simulated": {k: simulated.get(k) for k in ("scenario", "sigma", "seed")}}
    if request.get("what_if"):
        spec = {**spec, "what_if": _kept_settings(request["what_if"])}
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
    every_run = _runs_frame(request, spec)
    spec = _for_response(spec, request.get("response"))  # the response drawn: the first, or the one named
    if _mixture(spec):
        return _mixture_plots(request, spec)
    names = [f["name"] for f in spec["factors"]]
    categorical = _categorical_names(spec)
    model = _checked_model(request.get("model") or spec["model"], names, categorical)
    fitted, columns = _fitted_model(model, spec)
    runs = _design_runs(every_run)
    response = spec["response"]
    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna().to_numpy()
    coded = _design_frame(runs, spec, columns)[done].reset_index(drop=True)
    y = y[done].reset_index(drop=True).rename(response)
    with _listening():
        fit = _fit_estimable(coded, y, fitted, columns, categorical)
        out = analyze_experiment(
            coded, y, model=fit["formula"], analysis_type=["coefficients", "residual_diagnostics"], coding="actual"
        )
    rows = coded.assign(**{response: y}).to_dict("records")
    # A factor of several levels is no axis: the surface holds it at its average level, its contrasts at 0.
    axes = [f["name"] for f in spec["factors"] if not _several(f)]
    pair = _contour_pair(out["coefficients"], axes)
    shading, held = _shading(spec, pair)
    if best := _held_at_best(request.get("best"), spec, pair):
        shading, held = {**shading, "hold_values": best[0]}, best[1]

    def draw(kind: str, results: dict, **options: Any) -> str:  # noqa: ANN401
        figure = create_plot(kind, analysis_results=results, design_data=rows, response_column=response, **options)
        return json.dumps(_dressed(figure.to_plotly(), kind, spec, pair), cls=PlotlyJSONEncoder)

    # One factor has no surface to draw: the page draws its levels, from the analysis's comparison.
    surface = len(pair) > 1
    figures = {"contour": draw("contour", out, factors_to_plot=pair, **shading)} if surface else {}
    truth = request.get("truth")
    if truth and surface:
        process = TrueProcess({t["term"]: t["coefficient"] for t in truth}, sigma=1.0)
        missing = [f for f in process.factors if f not in columns]
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
        "shaded": "region" in shading,
        "held": held,
        "held_best": best is not None,
        "plotly_js": get_plotlyjs_version(),
    }


def _held_at_best(raw: object, spec: dict, pair: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]] | None:
    """Return where the response surface holds the factors it does not draw when the page has the best settings.

    ``raw`` is the best settings as :func:`best_settings` gives them, every factor coded, or None.
    The surface is then a slice through the best settings, so the star the page puts on it sits on
    the surface it marks. Returns the coded values to hold, and each held factor in its own units,
    or its level, for the caption to say; None without best settings.
    """
    if raw is None:
        return None
    factors = spec["factors"]
    if not (
        isinstance(raw, dict)
        and set(raw) == {f["name"] for f in factors}
        and all(_number(v) is not None and -1 - AT_EDGE <= _number(v) <= 1 + AT_EDGE for v in raw.values())  # type: ignore[operator]
    ):
        raise ValueError("The page sent best settings this page did not write.")
    hold: dict[str, float] = {}
    held = []
    for f in factors:
        if f["name"] in pair:
            continue
        coded = float(raw[f["name"]])
        hold[f["name"]] = coded
        value = (
            _level_at(f, coded) if _categorical(f) else (f["low"] + f["high"]) / 2 + coded * (f["high"] - f["low"]) / 2
        )
        held.append({"name": f["name"], "value": value})
    return _held_columns(hold, spec), held


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


# --------------------------------------------------------------------------- the pictures of a mixture
#: Steps along each side of the triangle the fitted response is drawn on, for three components: 861 blends.
TERNARY_STEPS = 40

#: Blends along each component's Cox direction, for the trace plot of four components or more.
COX_POINTS = 41


def _mixture_plots(request: dict, spec: dict) -> dict:
    """Draw the analysis of a mixture's runs: its fitted response over the blends, and the residuals.

    For three components, ``ternary`` holds the fitted response on a grid of blends over the
    triangle of the blends the bounds allow, with the runs and their results, for the page to draw
    as a ternary plot (see ``ternaryFigure`` in ``charts.js``). For four components or more,
    ``cox`` holds the fitted response along each component's Cox direction through the reference
    blend, for a trace plot (``coxFigure``). With the true process the results were simulated from,
    each also holds the true response, on the same blends.

    With the best blend (``best``, as :func:`_best_blend` gives it), ``held_best`` is true: the page
    stars the blend on the triangle, and the Cox directions go through the best blend instead (see
    :func:`_cox_through`), so they cross at its prediction, where the page puts the star.
    """
    from plotly.offline import get_plotlyjs_version  # noqa: PLC0415 - plotly is installed for the first chart
    from plotly.utils import PlotlyJSONEncoder  # noqa: PLC0415

    from process_improve.experiments.visualization.plots import create_plot  # noqa: PLC0415

    names = [f["name"] for f in spec["factors"]]
    model = _mixture_model(request.get("model") or spec["model"])
    best = _blend_sent(request.get("best"), spec)
    runs = _design_runs(_runs_frame(request, spec))
    response = spec["response"]
    y = pd.to_numeric(runs[response], errors="coerce")
    done = y.notna().to_numpy()
    blends = runs.loc[done, names].astype(float).reset_index(drop=True)
    y = y[done].reset_index(drop=True).rename(response)
    with _listening():
        out = _mixture_fit(blends, y, model, ["coefficients", "residual_diagnostics", "effects"])
    fitted = {c["term"]: c["coefficient"] for c in out["coefficients"]}
    truth = request.get("truth")
    true = _true_coefficients(truth, names) if truth else None
    figures = {}
    if out["model_summary"]["df_residual"] > 0:
        rows = blends.assign(**{response: y}).to_dict("records")
        figure = create_plot("residuals_vs_fitted", analysis_results=out, design_data=rows, response_column=response)
        dressed = _dressed(figure.to_plotly(), "residuals_vs_fitted", spec, [])
        figures["residuals"] = json.dumps(dressed, cls=PlotlyJSONEncoder)
    reply: dict[str, Any] = {
        "figures": figures,
        "contour_factors": [],
        "shaded": False,
        "held": [],
        "held_best": best is not None,
        "plotly_js": get_plotlyjs_version(),
    }
    if len(names) == 3:
        reply["ternary"] = _ternary(spec, blends, y, fitted, true)
    elif best is None:
        reply["cox"] = _cox_traces(names, out, fitted, true)
    else:
        reply["cox"] = _cox_through(spec, best, fitted, true)
    return reply


def _blend_sent(raw: object, spec: dict) -> np.ndarray | None:
    """Return the best blend the page sends with its charts, checked as a blend of the components; None without one.

    ``raw`` is the blend as :func:`_best_blend` gives it, ``coded``: each component's proportion,
    within its bounds, the proportions summing to 1.
    """
    if raw is None:
        return None
    names = [f["name"] for f in spec["factors"]]
    low, high = _bounds(spec)
    values = [_number(raw.get(n)) for n in names] if isinstance(raw, dict) and set(raw) == set(names) else [None]
    blend = np.array(values, dtype=float) if None not in values else np.full(len(names), np.nan)
    if not (
        np.all(blend >= low - PROPORTION_TOLERANCE)
        and np.all(blend <= high + PROPORTION_TOLERANCE)
        and abs(blend.sum() - 1.0) <= 1e-6
    ):
        raise ValueError("The page sent a best blend this page did not write.")
    return blend


def _blended(coefficients: dict[str, float], blends: pd.DataFrame) -> list[float]:
    """Return a Scheffé model's response at each blend: each term's coefficient times the product of its components.

    The terms are the fit's (``A``, ``A:B``, ``A:B:C``) or a checked true process's, so each names
    columns of ``blends``; the formula is never evaluated as text.
    """
    total = np.zeros(len(blends))
    for term, value in coefficients.items():
        total += value * blends[term.split(":")].prod(axis=1).to_numpy(dtype=float)
    return [float(f"{v:.6g}") for v in total]


def _ternary(
    spec: dict, blends: pd.DataFrame, y: pd.Series, fitted: dict[str, float], true: dict[str, float] | None
) -> dict:
    """Return the fitted response over a grid of blends of three components, and the runs, for a ternary plot.

    The grid is :func:`_ternary_grid`'s, so the response is not drawn where the design has no runs
    to support it.
    """
    names = [f["name"] for f in spec["factors"]]
    grid, _ = _ternary_grid(spec)
    frame = pd.DataFrame(grid, columns=names)
    return {
        "components": names,
        "grid": {
            "a": grid[:, 0].tolist(),
            "b": grid[:, 1].tolist(),
            "c": grid[:, 2].tolist(),
            "fitted": _blended(fitted, frame),
            "true": None if true is None else _blended(true, frame),
        },
        "runs": {
            "a": blends[names[0]].tolist(),
            "b": blends[names[1]].tolist(),
            "c": blends[names[2]].tolist(),
            "y": y.tolist(),
        },
    }


def _ternary_grid(spec: dict) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return the blends of three components a triangle is drawn over, and each one's place ``(i, j)`` on its lattice.

    The grid covers the smallest triangle holding the blends the lower proportions allow (the whole
    triangle without lower bounds), in :data:`TERNARY_STEPS` steps a side: the blend at ``(i, j)`` has
    ``i`` and ``j`` steps of the first two components. It keeps the blends the upper proportions
    allow too.
    """
    low, high = _bounds(spec)
    n = TERNARY_STEPS
    places = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
    steps = np.array([(i, j, n - i - j) for i, j in places], dtype=float) / n
    grid = low + (1.0 - low.sum()) * steps
    kept = np.all(grid <= high + PROPORTION_TOLERANCE, axis=1)
    return grid[kept].round(9), [place for place, keep in zip(places, kept, strict=True) if keep]


def _cox_traces(names: list[str], out: dict, fitted: dict[str, float], true: dict[str, float] | None) -> dict:
    """Return the fitted response along each component's Cox direction through the reference blend.

    On component ``i``'s Cox direction the others keep their proportions to each other, so a
    blend there is the reference blend ``s`` with ``x_i`` in place of ``s_i`` and every other
    proportion scaled by ``(1 - x_i) / (1 - s_i)``. Each trace runs over the span of ``x_i`` the
    library's effects are taken over, within the range each component takes in the design, and is
    drawn against ``x_i - s_i``, the change from the reference blend, so every trace crosses the
    others at the reference blend's response.
    """
    reference = np.array([out["effect_reference"][n] for n in names])
    traces = []
    for i, name in enumerate(names):
        start, stop = out["effect_range"][name]
        share = np.linspace(start, stop, COX_POINTS)
        points = reference[None, :] * ((1.0 - share) / (1.0 - reference[i]))[:, None]
        points[:, i] = share
        frame = pd.DataFrame(points, columns=names)
        traces.append(
            {
                "component": name,
                "change": [float(f"{v:.6g}") for v in share - reference[i]],
                "fitted": _blended(fitted, frame),
                "true": None if true is None else _blended(true, frame),
            }
        )
    return {"components": names, "reference": dict(zip(names, reference.tolist(), strict=True)), "traces": traces}


def _cox_through(spec: dict, blend: np.ndarray, fitted: dict[str, float], true: dict[str, float] | None) -> dict:
    """Return the fitted response along each component's Cox direction through the best blend.

    As :func:`_cox_traces`, with the best blend in place of the reference blend: each trace moves one
    component from the best blend, the others keeping their proportions to each other (see
    :func:`_cox_moved`), over the span every component's bounds allow, and holds the best blend
    itself, a change of 0. So the traces cross at the best blend's prediction. ``through`` says
    which blend they go through.
    """
    names = [f["name"] for f in spec["factors"]]
    low, high = _bounds(spec)
    traces = []
    for i, name in enumerate(names):
        direction = _cox_moved(blend, i, 1.0) - blend  # a change of 1 in component i, the others moving with it
        start, stop = _span_along(blend, direction, low, high)
        change = np.unique(np.append(np.linspace(start, stop, COX_POINTS), 0.0))
        frame = pd.DataFrame(blend[None, :] + change[:, None] * direction[None, :], columns=names)
        traces.append(
            {
                "component": name,
                "change": [float(f"{v:.6g}") for v in change],
                "fitted": _blended(fitted, frame),
                "true": None if true is None else _blended(true, frame),
            }
        )
    reference = dict(zip(names, blend.tolist(), strict=True))
    return {"components": names, "reference": reference, "through": "best", "traces": traces}


def _span_along(blend: np.ndarray, direction: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Return how far a blend can move along a direction, either way, with every component within its bounds."""
    start, stop = -np.inf, np.inf
    for d, x, lo, hi in zip(direction, blend, low, high, strict=True):
        if d > 0:
            start, stop = max(start, (lo - x) / d), min(stop, (hi - x) / d)
        elif d < 0:
            start, stop = max(start, (hi - x) / d), min(stop, (lo - x) / d)
    return min(start, 0.0), max(stop, 0.0)  # the blend itself is inside, to rounding


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
    renamed = _factors({"factors": request.get("factors"), "design_type": spec["design_type"]})
    before = spec["factors"]
    after = [_factor_spec(f) for f in renamed]

    def settings(f: dict) -> object:
        return ("labels", _levels_of(f)) if _categorical(f) else ("range", f["low"], f["high"])

    if [settings(f) for f in after] != [settings(f) for f in before]:
        raise ValueError("Only the factors' names and units can change without making the run sheet again.")
    columns = [str(c) for c in design.get("columns") or []]
    for taken in (*(r["name"] for r in _responses(spec)), *([BLOCK] if BLOCK in columns else [])):
        if taken in {f.name for f in renamed}:
            raise ValueError(f"A factor cannot be named {taken}: the run sheet has a {taken} column.")
    names = {old["name"]: new["name"] for old, new in zip(before, after, strict=True)}
    spec = _relabelled_spec(spec, after, names)
    reply: dict[str, Any] = {
        "spec": spec,
        "columns": [names.get(c, c) for c in columns],
        "outside_range": [names.get(n, n) for n in design.get("outside_range") or []],
    }
    if design.get("facts") and _mixture(spec):  # a mixture's notes name no component, and it has no aliases
        reply["facts"] = design["facts"]
    elif design.get("facts"):
        runs = _design_runs(_runs_frame({**design, "columns": reply["columns"]}, spec))
        coded = _coded(runs, after).to_numpy(dtype=float)
        reply["facts"] = _relabelled_facts(design["facts"], coded, names)
    if simulation := request.get("simulation"):
        words = {r["name"]: r.get("label") or r["name"] for r in _responses(spec)}
        reply["simulation"] = _relabelled_process(simulation, words[spec["response"]], names)
        if isinstance(more := simulation.get("more"), dict):
            reply["simulation"]["more"] = {
                name: _relabelled_process(p, words.get(name, name), names) for name, p in more.items()
            }
    return reply


def _relabelled_spec(spec: dict, after: list[dict], names: dict[str, str]) -> dict[str, Any]:
    """Write a specification again with its factors renamed: the factors, the limits, and the models a workbook keeps.

    The limits on the settings are written with the new names all at once, so a swap works; so are the
    terms a workbook was last analysed with, for the first response and each later one.
    """

    def terms_renamed(chosen: object) -> object:
        if not isinstance(chosen, dict):
            return chosen
        return {"terms": [[names.get(n, n) for n in term] for term in chosen["terms"]]}

    spec = {**spec, "factors": after}
    if limits := spec.get("constraints"):
        spec["constraints"] = [{**limit, "expression": rename_factors(limit["expression"], names)} for limit in limits]
    if "analysis_model" in spec:
        spec["analysis_model"] = terms_renamed(spec["analysis_model"])
    if "analysis_models" in spec:
        spec["analysis_models"] = {name: terms_renamed(m) for name, m in spec["analysis_models"].items()}
    return spec


def _relabelled_process(process: dict, label: str, names: dict[str, str]) -> dict[str, Any]:
    """Write one response's simulated process again with its factors renamed: its terms, equation and quiet factors."""
    truth = [
        {"term": term, "label": _pretty(term), "coefficient": t["coefficient"]}
        for t in process.get("truth") or []
        for term in [_renamed_term(str(t["term"]), names)]
    ]
    return {
        **process,
        "truth": truth,
        "equation": _equation(label, truth),
        "inactive": [names.get(n, n) for n in process.get("inactive") or []],
    }


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
    that keep their runs inside them are listed, each made with the limits. ``left_out`` lists
    each design the page offers that the comparison leaves out, with why (see :func:`_left_out`),
    so Design can say why a design is not listed.
    """
    model = request.get("model") or "main_quadratic"
    if model not in COMPARISON_MODELS:
        raise ValueError(f"Model must be one of {', '.join(COMPARISON_MODELS)}.")
    factors = _factors(request)
    limits = _checked_constraints(request.get("constraints"), factors)
    if limits:
        _region_refusal(factors, _region(factors, [limit["expression"] for limit in limits]))
    k = len(factors)
    if k > MAX_INTERACTION_FACTORS and model != MODELS[0]:
        raise ValueError(
            f"Past {MAX_INTERACTION_FACTORS} factors, a study aims to find which factors matter, and the designs are "
            f"compared for their main effects: {k} factors have {k * (k - 1) // 2} two-factor interactions."
        )
    names = {f.name for f in factors}
    candidates = [str(request.get("response") or "").strip(), "y", *(f"y{i}" for i in range(1, k + 1))]
    response = next(name for name in candidates if name.isidentifier() and name not in names)
    plan, left_out = [], []
    labelled = any(f.type == "categorical" for f in factors)
    levels = _level_plan(request, factors, model)
    # A design taken from the comparison keeps the reader's description of the response, and aims to learn the
    # model it was compared on.
    described = _described({"response_info": request.get("response_info"), "aim": model})
    # The study's seed, so a design taken from the comparison keeps it.
    seeded = {"random_state": request["random_state"]} if "random_state" in request else {}
    for design_type in DESIGNS:
        if design_type in SPACE_FILLING:
            continue  # made for no model, so not one to compare on a model
        why = _left_out(design_type, k, labelled=labelled, limited=bool(limits), several=bool(_several_names(factors)))
        if why:
            left_out.append({"design_type": design_type, "label": COMPARE_LABELS[design_type], **why})
            continue
        options = _compare_options(design_type, model, levels)
        spec = {"design_type": design_type, "factors": request["factors"], "response": response, **options, **described}
        spec.update(seeded)
        if limits:
            spec["constraints"] = limits
        plan.append({"design_type": design_type, "label": COMPARE_LABELS[design_type], "spec": spec})
    notes = [LIMITED_DESIGNS] if limits else []
    if model != OPTIMAL_MODEL[model] and any("model_type" in DESIGNS[d["design_type"]]["options"] for d in plan):
        notes.append(
            f"The D- and I-optimal designs are built for the {OPTIMAL_MODEL[model]} model, the smallest model "
            f"they can be built for that contains this one."
        )
    # Two levels estimate the factor's main effect alone, whatever else the aim names: they are scored on it.
    scored = "main_effects" if levels else model
    return {"model": scored, "designs": plan, "notes": notes, "left_out": left_out, "levels": levels}


def _compare_options(design_type: str, model: str, levels: dict[str, Any] | None) -> dict[str, Any]:
    """Return the options a design is compared with: the page's defaults, made for the comparison's model.

    An optimal design is built for the model (see :data:`OPTIMAL_MODEL`), a Taguchi array runs its numeric factors
    at three levels for a model with squares, and a design in one factor runs each of its two levels as often as
    its plan (``levels``, from :func:`_level_plan`) asks, with no center runs.
    """
    options = dict(COMPARE_OPTIONS[design_type])
    allowed = DESIGNS[design_type]["options"]
    if levels:
        options.update(n_center_points=0, n_replicates=levels["n"])
    if "model_type" in allowed:
        options["model_type"] = OPTIMAL_MODEL[model]
    if "n_levels" in allowed and model in TAGUCHI_THREE_LEVELS:
        options["n_levels"] = TAGUCHI_LEVELS[-1]
    return options


#: The power a design in one factor is sized for: the fewest runs at each of its two levels whose t-test has this
#: chance of finding the smallest change worth finding.
LEVEL_POWER = 0.8

#: The runs at each level the plan of a design in one factor lists, beside the number that reaches its power.
LEVEL_RUNS = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, MAX_LEVEL_RUNS)


def _level_plan(request: dict, factors: list[Factor], model: str) -> dict[str, Any] | None:
    """Size a design in one factor, its levels compared: the power of its test for each number of runs at each level.

    The change is the reader's smallest change worth finding, over the run-to-run noise, or two noise standard
    deviations without them, as designs are compared (:func:`_effect_size`), between two of the factor's levels,
    the others midway, where it is hardest to find (``least_favourable_means``). The power is the one-way analysis
    of variance's (``one_way_power``), which for two levels is the two-sided t-test's, and the model's test of the
    factor. ``n`` is the fewest runs at each level that reach :data:`LEVEL_POWER`, or :data:`MAX_LEVEL_RUNS` when
    none does (``reached`` says which). None for more factors, or for a numeric factor the study aims to see curve,
    which a third level shows.
    """
    if len(factors) != 1 or (factors[0].type != "categorical" and model in (MAIN_QUADRATIC, "quadratic")):
        return None
    info = _response_info(request.get("response_info"))
    k = len(factors[0].levels or []) if factors[0].type == "categorical" else 2
    difference = 2 * _effect_size({key: info.get(key) for key in ("noise", "smallest_change")})
    means = least_favourable_means(difference, k)
    try:
        n = int(one_way_sample_size(LEVEL_POWER, 1.0, means, max_n=MAX_LEVEL_RUNS)["n"])
    except ValueError:  # no number of runs the page makes reaches it
        n = MAX_LEVEL_RUNS
    rows = [{"n": m, "runs": k * m, "power": one_way_power(m, 1.0, means)["power"]} for m in sorted({*LEVEL_RUNS, n})]
    reached = next(row["power"] for row in rows if row["n"] == n) >= LEVEL_POWER
    return {"n": n, "power": LEVEL_POWER, "reached": reached, "rows": rows}


def _left_out(
    design_type: str, k: int, *, labelled: bool, limited: bool, several: bool = False
) -> dict[str, Any] | None:
    """Say why the comparison leaves a design out for this brief, or None when it is in.

    ``reason`` is ``factors`` for a factor count outside the design's (with its ``low`` and ``high``),
    ``categorical`` for a design of numeric factors with a categorical one in the brief, ``levels``
    for a two-level design with a categorical factor of three levels or more (``several``), and
    ``limits`` for a design that cannot keep its runs inside limits on the settings, in that order.
    """
    info = DESIGNS[design_type]
    if not info["min_factors"] <= k <= info["max_factors"]:
        return {"reason": "factors", "low": info["min_factors"], "high": info["max_factors"]}
    if labelled and design_type not in CATEGORICAL_DESIGNS:
        return {"reason": "categorical"}
    if several and design_type not in SEVERAL_LEVEL_DESIGNS:
        return {"reason": "levels"}
    if limited and design_type not in REGION_DESIGNS:
        return {"reason": "limits"}
    return None


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


def _powers(request: dict) -> dict[str, Any]:
    """Score designs with a categorical factor of three levels or more: their runs, spare runs, and power.

    The comparison's measures are for factors coded from -1 to +1, which such a factor is not, so each
    design is scored on the model the reader aims to learn, over the columns it is fitted on: its spare
    runs, and, for each kind of term, the power of the F test of the term the hardest to find. A term of a
    factor of several levels is tested over its contrasts together; the change it is to find is the
    reader's, between two of its levels, the others on the average (see :func:`_least_favourable`). The reply
    has the comparison's shape, so Design lists the designs as it lists any.
    """
    effect = _effect_size(request)
    model = request.get("model") or "main_quadratic"
    designs, values, terms = [], {}, []
    for item in request.get("designs") or []:
        spec = _checked_spec(item.get("spec") or {})
        names = [f["name"] for f in spec["factors"]]
        aimed = _model_terms(model, names, _categorical_names(spec)) if model != "main_quadratic" else None
        aimed = aimed if aimed is not None else _model_terms(MAIN_QUADRATIC, names, _categorical_names(spec))
        columns = _model_columns(spec)
        frame = _design_frame(_runs_frame(item, spec), spec)
        expanded = _expanded_terms(aimed, spec)
        x = _model_matrix(frame, ["Intercept", *expanded], columns)
        n, p = x.shape
        rank = int(np.linalg.matrix_rank(x))
        fits = rank == p
        df = n - p if fits else 0
        power: dict[str, float | None] = {"main": None, "interaction": None, "quadratic": None}
        if fits and df > 0:
            inverse = np.linalg.inv(x.T @ x)
            for term in aimed:
                kind = "quadratic" if term.startswith("I(") else "interaction" if ":" in term else "main"
                members = _expanded_terms([term], spec, aimed)
                index = [expanded.index(m) + 1 for m in members]  # after the intercept
                b = _least_favourable(members, spec, effect)
                found = _group_power(b, inverse[np.ix_(index, index)], df)
                power[kind] = found if power[kind] is None else min(power[kind], found)  # type: ignore[type-var]
        sized = f"{item['label']} [n={n}]"
        designs.append(sized)
        values[sized] = {
            "runs": n,
            "fits": fits,
            "residual_df": df,
            "power_main": power["main"],
            "power_interaction": power["interaction"],
            "power_quadratic": power["quadratic"],
        }
        terms = ["Intercept", *expanded]
    return {
        "model": model,
        "terms": terms,
        "effect_size": effect,
        "significance_level": 0.05,
        "designs": designs,
        "values": values,
        "measures": [],
        "display": {},
        "fds": {},
    }


def _least_favourable(members: list[str], spec: dict, effect: float) -> np.ndarray:
    """Return the coefficients of a term's columns at the change worth finding, where it is hardest to find.

    A numeric or two-level factor's coefficient of ``effect`` noise standard deviations is the change, as
    the comparison scores power. A factor of several levels has that change between its first two
    levels, the others on the average: its first contrast's coefficient is -``effect``, its second's
    +``effect``, the rest 0; an interaction's columns take the product.
    """
    several = {f["name"] for f in spec["factors"] if _several(f)}

    def size(member: str) -> float:
        value = effect
        for part in member.split(":"):
            stem, _, j = part.rpartition(LEVEL_JOIN)
            if stem in several:
                value *= {"1": -1.0, "2": 1.0}.get(j, 0.0)
        return value

    return np.array([size(m) for m in members])


def _group_power(b: np.ndarray, block: np.ndarray, df: int) -> float:
    """Return the power of the F test of a term's columns together at the 5% level: noncentrality b' V^-1 b."""
    from scipy import stats  # noqa: PLC0415 - scipy is loaded with the library

    noncentrality = float(b @ np.linalg.solve(block, b))
    critical = stats.f.ppf(0.95, len(b), df)
    return float(stats.ncf.sf(critical, len(b), df, noncentrality))


#: From this many factors, the optimal designs' search for a model beyond main effects takes seconds
#: to tens of seconds (an I-optimal design in six factors, about 20 s in the browser), so the designs
#: listed as the reader types leave them out until the reader asks for the search.
SEARCH_FROM = 5


def _compared_sentence(k: int) -> str:
    """Say how designs in more than :data:`MAX_COMPARED_FACTORS` factors are compared, and why."""
    return (
        f"Past {MAX_COMPARED_FACTORS} factors, the designs are compared on their runs, spare runs and power alone: "
        f"the other measures search every corner of the ranges, and {k} factors have {2**k:,} of them."
    )


def candidates(request: dict) -> dict:
    """Make and score, in one call, every design the page offers for these factors and this aim.

    The designs are :func:`compare_plan`'s, each made as :func:`make_design` makes it, with
    the library's warnings for it kept beside it, and all of them scored by :func:`compare` on
    the model the reader aims to learn, at the power for the response's smallest change worth
    finding. A design that cannot be made is left out, with a note saying why. With
    :data:`SEARCH_FROM` factors or more and a model beyond main effects, the optimal designs
    are left out unless ``search`` is true, and listed in ``waiting``; with limits on the
    settings they are the only designs, so then ``made`` can be empty and ``result`` null. Past
    :data:`MAX_COMPARED_FACTORS` factors the designs are scored as for a factor of several levels
    (see :func:`_powers`), and a note says why.
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
    several = made and any(_several(f) for f in made[0]["design"]["spec"]["factors"])
    if k > MAX_COMPARED_FACTORS:
        notes.append(_compared_sentence(k))
    result = (_powers(asked) if several or k > MAX_COMPARED_FACTORS else compare(asked)) if made else None
    return {
        "model": plan["model"],
        "made": made,
        "waiting": waiting,
        "notes": notes,
        "result": result,
        "left_out": plan["left_out"],
        "levels": plan["levels"],
    }


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


# --------------------------------------------------------------------------- the best settings for the response's goal
#: The goals the page finds the best settings for: what the reader wants of the response, less ``learn``.
AIMED = ("maximize", "minimize", "target")

#: The confidence level of the intervals at the best settings.
LEVEL = 0.95

#: How far each numeric factor moves from the best settings, either way, to show how tightly it must be held: this
#: share of its range, 0.1 in coded units.
NUDGE = 0.05

#: How many confirmation runs the page adds at the best settings: enough for their average to vary less than one run.
CONFIRMATION_RUNS = 3

#: Within this, in coded units, of an end of its range a factor is at that end, and of a limit's boundary the best
#: settings are on that limit. The optimiser stops on a bound or a limit to far better than this.
AT_EDGE = 1e-5

#: How many combinations of the categorical factors' levels the optimiser searches in full: the best of them, judged
#: first at :data:`LEVEL_SAMPLE` settings of the numeric factors and every corner of their ranges. One search takes
#: about a tenth of a second in the browser, so a study with many categorical factors still answers at once.
SEARCHED_LEVELS = 3
LEVEL_SAMPLE = 64

#: The most combinations of the categorical factors' levels judged one by one: every combination of 12 factors of two
#: levels. Past it the levels are chosen a factor at a time (see :func:`_searched_levels`): 40 factors have 2^40.
LEVEL_COMBINATIONS = 4096

#: The most corners of the numeric factors' ranges each combination of levels is judged at: all of them up to 8 factors.
LEVEL_CORNERS = 256

#: The significance level below which a term is clearly larger than the noise.
CLEAR = 0.05

#: The best settings are a stationary point when no slope there is above this share of the model's spread: the
#: optimiser stops far closer to a flat point than this.
STATIONARY = 1e-4

#: A factor whose slope is below this share of the largest is left out of the way the prediction improves fastest.
SHARE = 0.01

#: How far past a limit, in its own units, the optimiser accepts its settings as inside it (the library's
#: ``_within_region``): the best settings on a limit lie on it to this, and so do the confirmation runs made there.
ON_LIMIT = 1e-6


def best_settings(request: dict) -> dict:
    """Find the settings that best meet the response's goal on the fitted model, and say how well they are known.

    The goal is the brief's (``response_info.goal``): ``maximize``, ``minimize``, or ``target`` with its
    ``target``. For ``learn``, or a target not yet typed, the reply says so and holds nothing else. A
    mixture's best blend is found by :func:`_best_blend`, whose reply follows this one's.

    The search is :func:`~process_improve.experiments.optimize_responses` with the desirability method
    and one response, over the numeric factors' ranges and inside the limits on the settings. Its
    limits are set beyond anything the model predicts in the ranges (see :func:`_goal`), so the
    desirability is a straight line in the prediction, or in its distance from the target, and its
    best is the goal's best. Each categorical factor is held at each of its levels in turn (see
    :func:`_held_at`): every combination of levels is judged at a sample of settings, and the
    :data:`SEARCHED_LEVELS` best are searched in full.

    At the best settings the reply gives the prediction with its 95% confidence interval for the
    mean and prediction interval for one new run, from the fit's covariance and residual variance;
    the leverage there against the largest leverage of the runs fitted; each factor at an end of its
    range and each limit the settings are on, with whether the goal improves beyond it, or the kind
    of stationary point when the settings are inside; and the change in the prediction when each
    numeric factor alone moves by :data:`NUDGE` of its range either way. ``confirm_at`` is where
    confirmation runs go, each setting rounded for the plant (see :func:`_practical`), and
    ``confirmation`` judges the confirmation runs on the sheet: the average of each group's results
    against the prediction interval for the average of that many runs. ``search`` is what the code
    the page shows needs to make the same search.
    """
    whole = _checked_spec(request.get("spec") or {})
    spec = _for_response(whole, request.get("response"))  # the response whose goal is met: the first, or the one named
    info = spec.get("response_info") or {}
    goal = info.get("goal", "learn")
    if goal not in AIMED:
        return {"goal": "learn"}
    target = info.get("target")
    if goal == "target" and target is None:
        return {"goal": goal, "target": None}
    if _mixture(spec):  # a blend's proportions sum to 1: the search keeps to the blends, not to each factor's range
        return _best_blend(request, whole, spec, goal, target)
    every_run = _runs_frame(request, whole)
    names = [f["name"] for f in spec["factors"]]
    categorical = _categorical_names(spec)
    model = _checked_model(request.get("model") or spec["model"], names, categorical)
    design = _design_runs(every_run)
    fitted = _fitted(design, spec, model)
    zero = ROUND_OFF * float(pd.to_numeric(design[spec["response"]], errors="coerce").abs().max())
    coefficients = [  # as the analysis shows them: a coefficient within the fit's round-off of 0 is 0
        {"term": c["term"], "coefficient": _snapped(c["coefficient"], zero)} for c in fitted.out["coefficients"]
    ]
    region = _spec_region(spec)
    found = _search(coefficients, spec, region, goal, target)
    reply: dict[str, Any] = {"goal": goal, "target": target, "flat": found is None, "confirmation": []}
    if found is None:
        return reply
    x, levels, reduced = found["x"], found["levels"], found["model"]
    numeric = [f for f in spec["factors"] if not _categorical(f)]
    point = _full_point(spec, x, levels)
    predicted, confidence, prediction, leverage = fitted.at(point)
    reached = None if goal != "target" else abs(predicted - target) <= 1e-4 * max(found["spread"], 1e-12)
    sense = _sense(goal, target, predicted, reached=bool(reached))
    gradient = _gradient(reduced, [f["name"] for f in numeric], x)
    tiny = 1e-9 * max(found["spread"], 1.0)
    in_model = _factors_in(coefficients, spec)
    reply |= {
        "predicted": predicted,
        "reached": reached,
        "level": LEVEL,
        "confidence": confidence,
        "prediction": prediction,
        "leverage": leverage,
        "max_leverage": fitted.max_leverage,
        "extrapolating": leverage > fitted.max_leverage * (1 + 1e-9) + 1e-12,
        "settings": _settings_reply(spec, point, in_model, sense * gradient, tiny),
        "coded": point,
        "limits": _limits_on(spec, region, x, sense * gradient, tiny) if not reached else [],
        "stationary": _stationary(reduced, numeric, x, sense * gradient, region, in_model, found["spread"])
        if sense
        else None,
        "ascent": _ascent(numeric, sense * gradient, tiny) if sense else None,
        "unclear": _unclear(fitted.out["coefficients"], names),
        "sensitivity": _sensitivity(reduced, numeric, x, region, in_model),
        "confirm_at": _practical(spec, point, region),
        "confirmation": _confirmation(every_run, spec, fitted),
        "search": {"held": _used(found["held"], fitted.out["coefficients"]), "goal": found["goal"]},
    }
    return reply


class _FittedModel:
    """A least-squares fit as the page made it, with what a prediction at new settings needs.

    The model matrix is built from the fitted terms over the coded runs (a mixture's over its
    proportions, see :func:`_blend_fit`), so the covariance of the coefficients is the fit's own:
    ``mse_residual`` times the inverse of ``X'X``. A new point's row holds its coded settings in the
    same terms, and zero in each block contrast, which is the average block, as the equation is written.
    """

    def __init__(self, fit: dict[str, Any], coded: pd.DataFrame, names: list[str], spec: dict | None = None) -> None:
        from scipy import stats  # noqa: PLC0415 - scipy is loaded with the library

        self.out = fit["out"]
        self.formula = fit["formula"]
        self.names = names
        self.spec = spec  # with a factor of several levels, whose point's level is written over its contrasts
        self.terms = [c["term"] for c in self.out["coefficients"]]
        self.beta = np.array([c["coefficient"] for c in self.out["coefficients"]], dtype=float)
        x = _model_matrix(coded, self.terms, names)
        self.inverse = np.linalg.pinv(x.T @ x)
        summary = self.out["model_summary"]
        self.df = int(summary["df_residual"])
        self.mse = float(summary["mse_residual"]) if self.df > 0 else None
        self.t = float(stats.t.ppf(1 - (1 - LEVEL) / 2, self.df)) if self.df > 0 else None
        leverages = np.einsum("ij,jk,ik->i", x, self.inverse, x)
        self.max_leverage = float(leverages.max())

    def row(self, point: dict[str, float]) -> np.ndarray:
        """Return a point's row of the model matrix, its settings coded, in the average block."""
        if self.spec is not None:
            point = {**point, **_held_columns(point, self.spec)}
        frame = pd.DataFrame([{name: point[name] for name in self.names}])
        return _model_matrix(frame, self.terms, self.names)[0]

    def at(self, point: dict[str, float]) -> tuple[float, list[float] | None, list[float] | None, float]:
        """Return the prediction at a point, its confidence and prediction intervals, and its leverage.

        Without residual degrees of freedom there is no estimate of the error, and both intervals are None.
        """
        row = self.row(point)
        predicted = float(row @ self.beta)
        leverage = float(row @ self.inverse @ row)
        if self.mse is None or self.t is None:
            return predicted, None, None, leverage
        mean = self.t * math.sqrt(self.mse * leverage)
        single = self.t * math.sqrt(self.mse * (1 + leverage))
        return predicted, [predicted - mean, predicted + mean], [predicted - single, predicted + single], leverage

    def average_of(self, point: dict[str, float], runs: int) -> list[float] | None:
        """Return the prediction interval for the average of ``runs`` new runs at a point; None without error left."""
        if self.mse is None or self.t is None:
            return None
        row = self.row(point)
        predicted = float(row @ self.beta)
        half = self.t * math.sqrt(self.mse * (1 / runs + float(row @ self.inverse @ row)))
        return [predicted - half, predicted + half]


def _fitted(runs: pd.DataFrame, spec: dict, model: str | list[str]) -> _FittedModel:
    """Fit the model to the runs with results, as the analysis fits it, and return the fit with its covariance."""
    names = [f["name"] for f in spec["factors"]]
    y = pd.to_numeric(runs[spec["response"]], errors="coerce")
    done = y.notna().to_numpy()
    if not done.any():
        raise ValueError("Enter results first: the best settings are found on the model fitted to them.")
    fitted, columns = _fitted_model(model, spec)
    coded = _design_frame(runs, spec, columns)[done].reset_index(drop=True)
    with _listening():
        fit = _fit_estimable(
            coded, y[done].reset_index(drop=True).rename(spec["response"]), fitted, columns, _categorical_names(spec)
        )
    return _FittedModel(fit, coded, columns, spec if columns != names else None)


def _model_matrix(coded: pd.DataFrame, terms: list[str], names: list[str]) -> np.ndarray:
    """Return the model matrix of the fitted terms over coded settings: a column per term, as the fit made it.

    A block contrast (``Block1``, ...) is coded as the library codes it, against the blocks in sorted
    order, the last one -1 in every contrast; settings without a ``Block`` are the average block, 0.
    """
    blocks = sorted(coded[BLOCK].unique(), key=str) if BLOCK in coded.columns else []
    columns = []
    for term in terms:
        if term == "Intercept":
            columns.append(np.ones(len(coded)))
        elif _block_contrast(term, names):
            if not blocks:
                columns.append(np.zeros(len(coded)))
                continue
            level = blocks[int(re.fullmatch(rf"{BLOCK}(\d+)_*", term)[1]) - 1]  # type: ignore[index]
            block = coded[BLOCK].to_numpy()
            columns.append(np.where(block == level, 1.0, np.where(block == blocks[-1], -1.0, 0.0)))
        else:
            columns.append(np.prod([coded[p].to_numpy(dtype=float) for p in _parts(term)], axis=0))
    return np.column_stack(columns)


def _used(held: dict[str, float], coefficients: list[dict]) -> dict[str, float]:
    """Keep the held columns the model's terms hold: a factor's indicators only where another factor is nested in it."""
    used = {part for c in coefficients for part in _parts(c["term"])}
    return {name: value for name, value in held.items() if name in used}


def _parts(term: str) -> tuple[str, ...]:
    """Return the factors a model term multiplies: none for the intercept, ``(T, T)`` for ``I(T ** 2)``."""
    if term == "Intercept":
        return ()
    if square := re.fullmatch(r"I\((\w+) \*\* 2\)", term):
        return (square[1], square[1])
    return tuple(term.split(":"))


def _factors_in(coefficients: list[dict], spec: dict) -> set[str]:
    """Return the factors some term of the fitted model names, block contrasts aside: a contrast names its factor."""
    columns = _model_columns(spec)
    return {
        _source_of(p, spec) for c in coefficients if not _block_contrast(c["term"], columns) for p in _parts(c["term"])
    }


def _spec_region(spec: dict) -> DesignRegion | None:
    """Return the region the limits on the settings leave, or None for a study without limits."""
    limits = spec.get("constraints") or []
    return _region(_factors(spec), [limit["expression"] for limit in limits]) if limits else None


def _searched_levels(
    spec: dict, in_model: set[str], score: Callable[[dict[str, float]], float]
) -> list[dict[str, float]]:
    """Return the combinations of the levels of the categorical factors in the model to search in full, by their codes.

    Up to :data:`LEVEL_COMBINATIONS` combinations, every one is listed, and past :data:`SEARCHED_LEVELS` of them
    the best by ``score`` are kept. With more, as 13 factors of two levels have, one combination is built a factor
    at a time: each takes its best level, the others held, until none moves.
    """
    labelled = [f for f in spec["factors"] if _categorical(f) and f["name"] in in_model]
    names = [f["name"] for f in labelled]
    codes = [np.linspace(-1.0, 1.0, len(_levels_of(f))).tolist() for f in labelled]
    if math.prod(len(c) for c in codes) <= LEVEL_COMBINATIONS:
        every = [dict(zip(names, levels, strict=True)) for levels in itertools.product(*codes)]
        if len(every) <= SEARCHED_LEVELS:
            return every
        scores = [score(levels) for levels in every]
        return [every[i] for i in sorted(range(len(every)), key=lambda i: -scores[i])[:SEARCHED_LEVELS]]
    # shortcut: a factor at a time finds the best of a model whose categorical factors do not interact with each
    # other, and a good combination otherwise; search jointly if studies of so many interacting factors appear.
    chosen = {name: levels[0] for name, levels in zip(names, codes, strict=True)}
    moved = True
    while moved:
        moved = False
        for name, levels in zip(names, codes, strict=True):
            best = max(levels, key=lambda v, name=name: score({**chosen, name: v}))
            if score({**chosen, name: best}) > score(chosen) * (1 + 1e-12) + 1e-12:
                chosen[name], moved = best, True
    return [chosen]


def _held_at(coefficients: list[dict], held: dict[str, float]) -> list[dict]:
    """Return the model with some columns held at a value: a block contrast at 0, a categorical factor at its level.

    Each term loses the columns held, its coefficient multiplied by their values, and the terms that
    are then the same are added together, in the order each first appears. The code the page shows
    (``web/python.js``) holds them in the same way, so both hand the optimiser the same model.
    """
    model: dict[str, float] = {}
    for c in coefficients:
        parts = c["term"].split(":")
        term = ":".join(p for p in parts if p not in held) or "Intercept"
        model[term] = model.get(term, 0.0) + c["coefficient"] * math.prod(held.get(p, 1.0) for p in parts)
    return [{"term": term, "coefficient": value} for term, value in model.items()]


def _predict(model: list[dict], names: list[str], points: np.ndarray) -> np.ndarray:
    """Return a model's predictions at coded settings of ``names``, one row each."""
    values = np.zeros(len(points))
    for c in model:
        column = np.ones(len(points))
        for p in _parts(c["term"]):
            column = column * points[:, names.index(p)]
        values += c["coefficient"] * column
    return values


def _gradient(model: list[dict], names: list[str], x: np.ndarray) -> np.ndarray:
    """Return a model's gradient at coded settings ``x``: the change in the prediction per coded unit of each factor."""
    gradient = np.zeros(len(names))
    for c in model:
        parts = _parts(c["term"])
        for i, p in enumerate(parts):
            rest = math.prod(x[names.index(q)] for j, q in enumerate(parts) if j != i)
            gradient[names.index(p)] += c["coefficient"] * rest
    return gradient


def _hessian(model: list[dict], names: list[str], x: np.ndarray) -> np.ndarray:
    """Return a model's matrix of second derivatives at coded settings ``x``."""
    hessian = np.zeros((len(names), len(names)))
    for c in model:
        parts = _parts(c["term"])
        for i, j in itertools.permutations(range(len(parts)), 2):
            rest = math.prod(x[names.index(q)] for m, q in enumerate(parts) if m not in (i, j))
            hessian[names.index(parts[i]), names.index(parts[j])] += c["coefficient"] * rest
    return hessian


def _objective(goal: str, value: float, target: float | None) -> float:
    """Score a prediction for the goal: larger is better."""
    if goal == "maximize":
        return value
    if goal == "minimize":
        return -value
    return -abs(value - float(target))  # type: ignore[arg-type]


def _goal(goal: str, target: float | None, model: list[dict], response: str) -> dict[str, Any]:
    """Return the desirability goal whose best is the goal's best, for one response on this model.

    In the ranges, every term of a coded model is between -1 and +1, so no prediction is further
    from the intercept than the sum of the other coefficients' sizes, the model's ``spread``. Over
    the blends, every term of a Scheffé model is between 0 and 1, and its intercept is 0, so the same
    holds of it.
    Limits twice that far out leave every prediction strictly inside them, where the desirability
    is a straight line: rising with the prediction to maximise it, falling to minimise it, and for
    a target, with limits as far out on each side, falling with the distance from the target.
    """
    intercept = next((c["coefficient"] for c in model if c["term"] == "Intercept"), 0.0)
    spread = sum(abs(c["coefficient"]) for c in model if c["term"] != "Intercept")
    if goal == "target":
        reach = abs(intercept - float(target)) + 2 * spread  # type: ignore[arg-type]
        return {"response": response, "goal": goal, "target": target, "low": target - reach, "high": target + reach}  # type: ignore[operator]
    return {"response": response, "goal": goal, "low": intercept - 2 * spread, "high": intercept + 2 * spread}


def _search(
    coefficients: list[dict], spec: dict, region: DesignRegion | None, goal: str, target: float | None
) -> dict[str, Any] | None:
    """Find the best settings on the fitted model: the numeric factors coded, and each categorical factor's level.

    Returns ``x``, the numeric factors' coded settings; ``levels``, each categorical factor's coded
    level; ``model``, the model with those levels held; ``held``, what is held (the block contrasts
    at 0 too); ``goal``, the desirability goal searched with, or None when no numeric factor is in the
    model; and ``spread``, the most the prediction moves from its intercept in the ranges. None when
    no factor is in the model, so no setting is better than another.
    """
    from process_improve.experiments import optimize_responses  # noqa: PLC0415 - loaded when a goal asks for it

    numeric = [f for f in spec["factors"] if not _categorical(f)]
    numeric_names = [f["name"] for f in numeric]
    in_model = _factors_in(coefficients, spec)
    if not in_model or all(c["coefficient"] == 0 for c in coefficients if _parts(c["term"])):
        return None
    blocks = {c["term"]: 0.0 for c in coefficients if _block_contrast(c["term"], _model_columns(spec))}
    points = _sample(len(numeric), region)

    def score(levels: dict[str, float]) -> float:
        held = _held_at(coefficients, {**blocks, **_held_columns(levels, spec)})
        return max(_objective(goal, v, target) for v in _predict(held, numeric_names, points))

    combinations = _searched_levels(spec, in_model, score)
    best: dict[str, Any] | None = None
    for levels in combinations:
        held = {**blocks, **_held_columns(levels, spec)}
        model = _held_at(coefficients, held)
        sought = _goal(goal, target, model, spec["response"])
        spread = sum(abs(c["coefficient"]) for c in model if c["term"] != "Intercept")
        if spread > 0 and numeric:
            with _listening() as heard:
                result = optimize_responses(
                    [{"response_name": spec["response"], "coefficients": model, "factor_names": numeric_names}],
                    goals=[sought],
                    factor_ranges={f["name"]: {"low": f["low"], "high": f["high"]} for f in numeric},
                    region=region,
                )["desirability"]
            _relay(heard)
            x = np.array([result["optimal_coded"][n] for n in numeric_names])
            at_end = np.abs(np.abs(x) - 1) <= 1e-9  # stopped on an end of the range, to rounding: put it there
            x[at_end] = np.sign(x[at_end])
        else:
            x, sought = _centre(numeric, region), None
        x = _unmodelled_at_centre(x, numeric_names, in_model, region)
        value = float(_predict(model, numeric_names, x[None, :])[0])
        found = {
            "x": x,
            "levels": levels,
            "model": model,
            "held": held,
            "goal": sought,
            "spread": spread,
            "value": value,
        }
        if best is None or _objective(goal, value, target) > _objective(goal, best["value"], target) + 1e-12:
            best = found
    return best


def _sample(k: int, region: DesignRegion | None) -> np.ndarray:
    """Return settings of ``k`` numeric factors, coded, to judge each combination of levels at: corners, centre, random.

    Every corner of the ranges up to 8 factors (a model without squares is best at one of them), the
    centre, and :data:`LEVEL_SAMPLE` settings drawn at random, from a fixed seed, so a study is judged
    the same way each time. Settings outside the limits are left out.
    """
    if k == 0:
        return np.zeros((1, 0))
    rng = np.random.default_rng(1)
    if 2**k <= LEVEL_CORNERS:
        corners = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
    else:
        corners = rng.choice([-1.0, 1.0], size=(LEVEL_CORNERS, k))
    points = np.vstack([corners, np.zeros((1, k)), rng.uniform(-1.0, 1.0, size=(LEVEL_SAMPLE, k))])
    if region is not None:
        points = points[region.feasible(points)]
    return points if len(points) else np.zeros((1, k))


def _centre(numeric: list[dict], region: DesignRegion | None) -> np.ndarray:
    """Return the centre of the ranges, coded, or a central setting the limits allow when they rule the centre out."""
    x = np.zeros(len(numeric))
    if region is None or region.feasible(x[None, :])[0]:
        return x
    centroid = region.centroid() or {}
    return np.array(
        [(centroid[f["name"]] - (f["low"] + f["high"]) / 2) / ((f["high"] - f["low"]) / 2) for f in numeric]
    )


def _unmodelled_at_centre(
    x: np.ndarray, names: list[str], in_model: set[str], region: DesignRegion | None
) -> np.ndarray:
    """Hold each numeric factor the model leaves out at the centre of its range, where the limits allow it.

    The prediction does not depend on such a factor, so the optimiser may stop anywhere along it.
    """
    x = x.copy()
    for i, name in enumerate(names):
        if name in in_model:
            continue
        moved = x.copy()
        moved[i] = 0.0
        if region is None or region.feasible(moved[None, :])[0]:
            x = moved
    return x


def _full_point(spec: dict, x: np.ndarray, levels: dict[str, float]) -> dict[str, float]:
    """Return the best settings of every factor, coded: a categorical factor the model leaves out at its first level."""
    numeric = iter(x.tolist())
    return {
        f["name"]: levels.get(f["name"], -1.0) if _categorical(f) else float(next(numeric)) for f in spec["factors"]
    }


def _sense(goal: str, target: float | None, predicted: float, *, reached: bool) -> float:
    """Return which way the prediction should move: +1 up, -1 down, 0 for a target reached."""
    if goal == "maximize":
        return 1.0
    if goal == "minimize":
        return -1.0
    return 0.0 if reached else float(np.sign(float(target) - predicted))  # type: ignore[arg-type]


def _settings_reply(
    spec: dict, point: dict[str, float], in_model: set[str], ascent: np.ndarray, tiny: float
) -> list[dict[str, Any]]:
    """Describe each factor's best setting: its value in its own units or its level, and where it lies in its range.

    ``end`` is ``low`` or ``high`` for a numeric factor at that end of its range, and ``beyond`` says
    whether the goal improves past that end, by the gradient's sign there (``ascent``, which points
    the way the prediction should move, coded, for the numeric factors in order).
    """
    numeric = iter(ascent.tolist())
    settings = []
    for f in spec["factors"]:
        coded = point[f["name"]]
        if _categorical(f):
            level = _level_at(f, coded)
            settings.append(
                {
                    "name": f["name"],
                    "kind": "categorical",
                    "value": level,
                    "coded": coded,
                    "in_model": f["name"] in in_model,
                    "end": None,
                    "beyond": False,
                }
            )
            continue
        slope = next(numeric)
        end = "low" if coded <= -1 + AT_EDGE else "high" if coded >= 1 - AT_EDGE else None
        beyond = (end == "high" and slope > tiny) or (end == "low" and slope < -tiny)
        value = (f["low"] + f["high"]) / 2 + coded * (f["high"] - f["low"]) / 2
        settings.append(
            {
                "name": f["name"],
                "kind": "numeric",
                "value": value,
                "coded": coded,
                "units": f["units"],
                "in_model": f["name"] in in_model,
                "end": end,
                "beyond": beyond,
            }
        )
    return settings


def _limits_on(
    spec: dict, region: DesignRegion | None, x: np.ndarray, ascent: np.ndarray, tiny: float
) -> list[dict[str, Any]]:
    """List the limits the best settings are on, numbered by their place, and whether the goal improves beyond each.

    A limit is on the boundary of the settings it allows when the best settings are within
    :data:`AT_EDGE` of it, in coded units: its value over the length of its gradient there. The goal
    improves beyond it when the way the prediction should move (``ascent``) leaves the limit's side.
    """
    if region is None:
        return []
    factors = _factors(spec)
    on = []
    for i, limit in enumerate(spec["constraints"]):
        alone = _region(factors, [limit["expression"]])
        for g in alone.inequalities:
            value, normal = _value_and_normal(g, x)
            length = float(np.linalg.norm(normal))
            if length > 0 and value / length >= -AT_EDGE:
                on.append({"index": i, "expression": limit["expression"], "beyond": bool(normal @ ascent > tiny)})
                break
    return on


def _value_and_normal(
    g: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-6
) -> tuple[float, np.ndarray]:
    """Return a limit's value at coded settings ``x`` and its gradient there, by central differences."""
    value = float(g(x[None, :])[0])
    eye = np.eye(len(x)) * step
    normal = (g(x + eye) - g(x - eye)) / (2 * step)
    return value, np.asarray(normal, dtype=float)


def _stationary(  # noqa: PLR0913 - the model, the factors, the point, the way to move, the region and the terms
    model: list[dict],
    numeric: list[dict],
    x: np.ndarray,
    ascent: np.ndarray,
    region: DesignRegion | None,
    in_model: set[str],
    spread: float,
) -> dict[str, Any] | None:
    """Say where the surface is flat at the best settings: along the factors inside their ranges, or everywhere.

    ``factors`` are the numeric factors the model holds that are inside their ranges at the best
    settings, and ``inside`` says they are all of them, so the best settings are a stationary point
    inside the region. Along those factors the surface is flat there, and its curvature over them
    names the ``kind``: ``maximum``, ``minimum``, ``saddle``, or ``ridge`` when it is flat along some
    direction too, so settings along it predict alike. None when no such factor is left, when the
    best settings are on a limit (the surface is then flat along the limit, not along the factors),
    or when a slope along them is not flat.
    """
    names = [f["name"] for f in numeric]
    held = [i for i, name in enumerate(names) if name in in_model]
    free = [i for i in held if abs(x[i]) < 1 - AT_EDGE]
    if not free:
        return None
    if region is not None:
        for g in region.inequalities:
            value, normal = _value_and_normal(g, x)
            length = float(np.linalg.norm(normal))
            if length > 0 and value / length >= -AT_EDGE:
                return None
    if np.abs(ascent[free]).max() > STATIONARY * spread:
        return None
    eigenvalues = np.linalg.eigvalsh(_hessian(model, names, x)[np.ix_(free, free)])
    scale = float(np.abs(eigenvalues).max())
    if scale == 0:
        kind = "ridge"
    else:
        curved = eigenvalues[np.abs(eigenvalues) > 1e-8 * scale]
        if len(curved) < len(eigenvalues):
            kind = "ridge"
        else:
            kind = "maximum" if np.all(curved < 0) else "minimum" if np.all(curved > 0) else "saddle"
    return {"factors": [names[i] for i in free], "inside": len(free) == len(held), "kind": kind}


def _ascent(numeric: list[dict], ascent: np.ndarray, tiny: float) -> dict[str, float] | None:
    """Return the way the prediction improves fastest from the best settings, a share per factor; None when flat.

    Each factor's share is its slope over the largest slope, in coded units: a factor with share 1
    moves half its range while another with share -0.5 moves a quarter of its range down.
    """
    if not len(ascent) or np.abs(ascent).max() <= tiny:
        return None
    largest = float(np.abs(ascent).max())
    return {f["name"]: float(s / largest) for f, s in zip(numeric, ascent, strict=True) if abs(s) >= SHARE * largest}


def _unclear(coefficients: list[dict], names: list[str]) -> bool:
    """Say whether no term of the model is clearly larger than the noise: every p-value at or above :data:`CLEAR`."""
    p_values = [
        c.get("p_value") for c in coefficients if c["term"] != "Intercept" and not _block_contrast(c["term"], names)
    ]
    known = [p for p in p_values if p is not None and math.isfinite(p)]
    return bool(known) and all(p >= CLEAR for p in known)


def _sensitivity(
    model: list[dict], numeric: list[dict], x: np.ndarray, region: DesignRegion | None, in_model: set[str]
) -> list[dict[str, Any]]:
    """Return how much the prediction changes when each numeric factor alone moves :data:`NUDGE` of its range.

    ``down`` and ``up`` are the changes from the prediction at the best settings; ``step`` is the
    move in the factor's own units; ``down_runnable`` and ``up_runnable`` say whether the setting
    moved to is still inside the ranges and the limits.
    """
    names = [f["name"] for f in numeric]
    at = float(_predict(model, names, x[None, :])[0])
    out = []
    for i, f in enumerate(numeric):
        moved = np.vstack([x, x])
        moved[0, i] -= 2 * NUDGE
        moved[1, i] += 2 * NUDGE
        down, up = (_predict(model, names, moved) - at).tolist()
        inside = np.all(np.abs(moved) <= 1 + 1e-9, axis=1)
        if region is not None:
            inside &= region.feasible(moved)
        out.append(
            {
                "name": f["name"],
                "step": NUDGE * (f["high"] - f["low"]),
                "down": down,
                "up": up,
                "down_runnable": bool(inside[0]),
                "up_runnable": bool(inside[1]),
                "in_model": f["name"] in in_model,
            }
        )
    return out


def _practical(spec: dict, point: dict[str, float], region: DesignRegion | None) -> dict[str, Any]:
    """Return the best settings as a confirmation run is made at them: each in its own units, rounded for the plant.

    A numeric setting is rounded to a thousandth of its range's order of magnitude (0.01 degC for a
    range of 50 degC), and kept inside its range; when the rounding takes the settings outside the
    limits, it is done again a digit finer, up to 12 digits. Inside means as the optimiser means it,
    to :data:`ON_LIMIT`: the best settings on a limit are on it to that. A categorical factor's
    setting is its level.
    """
    factors = spec["factors"]
    numeric = [f for f in factors if not _categorical(f)]
    for extra in range(10):
        values: dict[str, Any] = {}
        for f in factors:
            coded = point[f["name"]]
            if _categorical(f):
                values[f["name"]] = _level_at(f, coded)
                continue
            decimals = 3 - math.floor(math.log10(f["high"] - f["low"])) + extra
            exact = (f["low"] + f["high"]) / 2 + coded * (f["high"] - f["low"]) / 2
            values[f["name"]] = min(max(round(exact, decimals), f["low"]), f["high"]) + 0.0
        coded_values = np.array([[_coded_value(_factor_of(f), values[f["name"]]) for f in numeric]])
        if region is None or region.feasible(coded_values, tol=ON_LIMIT)[0]:
            return values
    return {
        f["name"]: (f["low"] + f["high"]) / 2 + point[f["name"]] * (f["high"] - f["low"]) / 2
        if not _categorical(f)
        else values[f["name"]]
        for f in factors
    }


def _factor_of(factor: dict) -> Factor:
    """Return a numeric factor the page holds as the library's ``Factor``."""
    return Factor(name=factor["name"], low=factor["low"], high=factor["high"], units=factor.get("units", ""))


def _confirmation(runs: pd.DataFrame, spec: dict, fitted: _FittedModel) -> list[dict[str, Any]]:
    """Judge the confirmation runs on the run sheet against the model fitted to the design's runs.

    The confirmation runs are grouped by their settings, in the order each group first appears.
    Each group's ``rows`` are its places on the run sheet; ``runs`` how many it has and ``done``
    how many have a result; ``mean`` their average; ``predicted`` the model's prediction there; and
    ``interval`` the prediction interval for the average of ``done`` new runs there (of ``runs``
    while none has a result), which the average falls ``inside`` when the model holds there. The
    settings are coded, or a mixture's proportions as they are, as each model is fitted.
    """
    confirming = _confirming(runs).to_numpy()
    if not confirming.any():
        return []
    names = [f["name"] for f in spec["factors"]]
    values = pd.to_numeric(runs[spec["response"]], errors="coerce")
    coded = runs[names].astype(float) if _mixture(spec) else _coded(runs, spec["factors"])  # as the model is fitted
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(confirming).tolist():
        groups.setdefault(tuple(runs.loc[i, names].tolist()), []).append(i)
    out = []
    for rows in groups.values():
        point = {name: float(coded.loc[rows[0], name]) for name in names}
        results = values.iloc[rows].dropna()
        done = len(results)
        predicted = float(fitted.row(point) @ fitted.beta)
        interval = fitted.average_of(point, done or len(rows))
        mean = float(results.mean()) if done else None
        out.append(
            {
                "rows": rows,
                "at": {name: runs.loc[rows[0], name] for name in names},
                "runs": len(rows),
                "done": done,
                "mean": mean,
                "predicted": predicted,
                "interval": interval,
                "inside": None if mean is None or interval is None else bool(interval[0] <= mean <= interval[1]),
            }
        )
    return out


# --------------------------------------------------------------------------- a mixture's best blend
#: How many decimals each proportion of the best blend is weighed out to for a confirmation run: a thousandth of
#: the blend, 1 g in 1 kg.
BLEND_DECIMALS = 3

#: A Scheffé model whose predictions over the design's blends span less than this share of the results' size predicts
#: the same at every blend. Its fit leaves more round-off than a coded fit, as its proportions sum to 1 in every run.
FLAT_BLEND = 1e-9


def _best_blend(request: dict, whole: dict, spec: dict, goal: str, target: float | None) -> dict:
    """Find the blend that best meets the response's goal on its Scheffé model, and say how well it is known.

    The search is :func:`~process_improve.experiments.optimize_responses` with the desirability method
    over the region of blends, a ``DesignRegion`` of the components (see :func:`_components`): each
    component between its bounds, and the proportions summing to 1. The model is the analysis's, on
    the proportions as they are, and the goal is a straight line in the prediction, as for factors
    with ranges (see :func:`_goal`).

    The reply holds what :func:`best_settings` holds wherever it applies to a blend. ``settings`` gives
    each component's proportion, of ``kind`` ``component`` (see :func:`_blend_settings`), and ``coded``
    the blend, the scale the model is fitted on. The prediction, its intervals and the leverage are as
    for factors with ranges. ``sensitivity`` moves each component along its Cox direction (see
    :func:`_blend_sensitivity`). ``confirm_at`` is the blend as it is weighed out, and ``rounding`` says
    how it was rounded (see :func:`_weighed`). A way to move on, a stationary point and the limits on
    the settings do not apply to a blend: ``ascent`` and ``stationary`` are None, and ``limits`` empty.
    """
    from process_improve.experiments import optimize_responses  # noqa: PLC0415 - loaded when a goal asks for it

    every_run = _runs_frame(request, whole)
    design = _design_runs(every_run)
    names = [f["name"] for f in spec["factors"]]
    fitted = _blend_fit(design, spec, _mixture_model(request.get("model") or spec["model"]))
    size = float(pd.to_numeric(design[spec["response"]], errors="coerce").abs().max())
    coefficients = [  # as the analysis shows them: a coefficient within the fit's round-off of 0 is 0
        {"term": c["term"], "coefficient": _snapped(c["coefficient"], ROUND_OFF * size)}
        for c in fitted.out["coefficients"]
    ]
    if np.ptp(_predict(coefficients, names, design[names].to_numpy(dtype=float))) <= FLAT_BLEND * size:
        return {"goal": goal, "target": target, "flat": True, "confirmation": []}  # every blend predicts the same
    sought = _goal(goal, target, coefficients, spec["response"])
    with _listening() as heard:
        found = optimize_responses(
            [{"response_name": spec["response"], "coefficients": coefficients, "factor_names": names}],
            goals=[sought],
            region=DesignRegion(_components(whole)),
        )["desirability"]
    _relay(heard)
    low, high = _bounds(spec)
    x = _on_bounds(found["optimal_coded"], names, low, high)
    point = dict(zip(names, x.tolist(), strict=True))
    predicted, confidence, prediction, leverage = fitted.at(point)
    spread = sum(abs(c["coefficient"]) for c in coefficients)
    reached = None if goal != "target" else abs(predicted - target) <= 1e-4 * max(spread, 1e-12)  # type: ignore[operator]
    ascent = _sense(goal, target, predicted, reached=bool(reached)) * _gradient(coefficients, names, x)
    confirm_at, rounding = _weighed(spec, x)
    return {
        "goal": goal,
        "target": target,
        "flat": False,
        "predicted": predicted,
        "reached": reached,
        "level": LEVEL,
        "confidence": confidence,
        "prediction": prediction,
        "leverage": leverage,
        "max_leverage": fitted.max_leverage,
        "extrapolating": leverage > fitted.max_leverage * (1 + 1e-9) + 1e-12,
        "settings": _blend_settings(spec, x, ascent, 1e-9 * max(spread, 1.0)),
        "coded": point,
        "limits": [],
        "stationary": None,
        "ascent": None,
        "unclear": _unclear_blend(fitted.out),
        "sensitivity": _blend_sensitivity(coefficients, spec, x),
        "confirm_at": confirm_at,
        "rounding": rounding,
        "confirmation": _confirmation(every_run, spec, fitted),
        "search": {"held": {}, "goal": sought},
    }


def _blend_fit(runs: pd.DataFrame, spec: dict, model: str) -> _FittedModel:
    """Fit a mixture's Scheffé model to the runs with results, as its analysis does, and return it with its covariance.

    A Scheffé model is fitted on the proportions as they are, with no intercept, so a blend's row of
    the model matrix is its proportions and their products: :class:`_FittedModel` builds it from the
    terms, as it does a coded row.
    """
    names = [f["name"] for f in spec["factors"]]
    y = pd.to_numeric(runs[spec["response"]], errors="coerce")
    done = y.notna().to_numpy()
    if not done.any():
        raise ValueError("Enter results first: the best blend is found on the model fitted to them.")
    blends = runs.loc[done, names].astype(float).reset_index(drop=True)
    y = y[done].reset_index(drop=True).rename(spec["response"])
    return _FittedModel(
        {"out": _mixture_fit(blends, y, model, ["coefficients", "anova"]), "formula": model}, blends, names
    )


def _on_bounds(found: dict[str, float], names: list[str], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Return the optimiser's blend, each proportion within :data:`PROPORTION_TOLERANCE` of a bound put on it.

    SLSQP keeps to the bounds, and stops on one to rounding (3e-17 for a component a blend leaves
    out), so a proportion at its bound is the bound itself, as the page shows it.
    """
    x = np.clip([found[n] for n in names], low, high)
    x = np.where(x <= low + PROPORTION_TOLERANCE, low, x)
    return np.where(x >= high - PROPORTION_TOLERANCE, high, x)


def _bounds(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return a mixture's lower and upper proportions, a component each, as its checked specification holds them."""
    return (
        np.array([f["low"] for f in spec["factors"]], dtype=float),
        np.array([f["high"] for f in spec["factors"]], dtype=float),
    )


def _unclear_blend(out: dict[str, Any]) -> bool:
    """Say whether no test of a mixture's ANOVA finds anything clearly larger than the noise: no p-value below 0.05.

    The mixture ANOVA tests whether the components blend alike, and each blending term on its own (a
    component's own coefficient is not tested against 0). With no runs left to estimate the noise it
    tests nothing, and nothing is said.
    """
    p_values = [row.get("p_value") for row in out.get("anova_table") or []]
    known = [p for p in p_values if p is not None and math.isfinite(p)]
    return bool(known) and all(p >= CLEAR for p in known)


def _blend_settings(spec: dict, x: np.ndarray, ascent: np.ndarray, tiny: float) -> list[dict[str, Any]]:
    """Describe each component's proportion in the best blend, and where it lies between its bounds.

    ``end`` is ``low`` or ``high`` for a component at that bound, to :data:`AT_EDGE` of half its range,
    as for a factor at an end of its range. ``beyond`` says whether the goal improves past a bound the
    reader set, by the prediction's gradient there (``ascent``, signed the way the prediction should
    move). Moving a component past its bound takes the difference from another component, or gives it
    to one, that its own bounds leave free to move that way; the goal improves past the bound when it
    does for one such component. No proportion passes 0 or 1, so a bound at either has no ``beyond``.
    """
    low, high = _bounds(spec)
    edge = AT_EDGE * (high - low) / 2
    at_low, at_high = x <= low + edge, x >= high - edge
    others = [[k for k in range(len(x)) if k != i] for i in range(len(x))]
    settings = []
    for i, f in enumerate(spec["factors"]):
        end = "low" if at_low[i] else "high" if at_high[i] else None
        gain = 0.0
        if end == "low" and low[i] > PROPORTION_TOLERANCE:  # less of it: another, not at its upper bound, takes more
            gain = max((ascent[k] - ascent[i] for k in others[i] if not at_high[k]), default=0.0)
        elif end == "high" and high[i] < 1 - PROPORTION_TOLERANCE:  # more of it: another, not at its lower, takes less
            gain = max((ascent[i] - ascent[k] for k in others[i] if not at_low[k]), default=0.0)
        settings.append(
            {
                "name": f["name"],
                "kind": "component",
                "value": float(x[i]),
                "coded": float(x[i]),
                "units": f["units"],
                "low": f["low"],
                "high": f["high"],
                "in_model": True,
                "end": end,
                "beyond": bool(gain > tiny),
            }
        )
    return settings


def _cox_moved(blend: np.ndarray, i: int, change: float) -> np.ndarray:
    """Return a blend with component ``i`` moved by ``change`` along its Cox direction, so it still sums to 1.

    On the Cox direction the other components keep their proportions to each other, as along the
    direction the analysis takes each component's effect: each is scaled by ``(1 - x_i - change) /
    (1 - x_i)``. A blend of component ``i`` alone leaves the others no proportions to keep, and they
    share the change equally.
    """
    moved = blend.copy()
    rest = 1.0 - blend[i]
    others = np.arange(len(blend)) != i
    if rest > PROPORTION_TOLERANCE:
        moved[others] *= (rest - change) / rest
    else:
        moved[others] = (rest - change) / others.sum()
    moved[i] = blend[i] + change
    return moved


def _blend_sensitivity(model: list[dict], spec: dict, x: np.ndarray) -> list[dict[str, Any]]:
    """Return how much the prediction changes when each component moves :data:`NUDGE` of its range, along its Cox line.

    ``step`` is the change in the component's proportion; ``down`` and ``up`` are the changes from the
    prediction at the best blend; ``down_runnable`` and ``up_runnable`` say whether the blend moved to
    keeps every component within its bounds. Every component is in a Scheffé model.
    """
    names = [f["name"] for f in spec["factors"]]
    low, high = _bounds(spec)
    at = float(_predict(model, names, x[None, :])[0])
    out = []
    for i, name in enumerate(names):
        step = NUDGE * (high[i] - low[i])
        moved = np.vstack([_cox_moved(x, i, -step), _cox_moved(x, i, step)])
        down, up = (_predict(model, names, moved) - at).tolist()
        inside = np.all((moved >= low - PROPORTION_TOLERANCE) & (moved <= high + PROPORTION_TOLERANCE), axis=1)
        out.append(
            {
                "name": name,
                "step": float(step),
                "down": down,
                "up": up,
                "down_runnable": bool(inside[0]),
                "up_runnable": bool(inside[1]),
                "in_model": True,
            }
        )
    return out


def _weighed(spec: dict, x: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    """Return the best blend as it is weighed out for a confirmation run, and how it was rounded.

    Each proportion is rounded to :data:`BLEND_DECIMALS` decimals, and what the rounding leaves over,
    or short, goes on the largest component whose bounds leave it room (``adjusted``; None when the
    rounding leaves nothing), so the proportions still sum to 1, as decimals. When no component has the
    room, or the rounding takes one outside its bounds, it is done again a decimal finer, up to 12
    decimals; past that the blend is given as found, with ``decimals`` None.
    """
    names = [f["name"] for f in spec["factors"]]
    low, high = _bounds(spec)
    largest_first = np.argsort(-x, kind="stable")
    for decimals in range(BLEND_DECIMALS, 13):
        scale = 10**decimals
        units = np.rint(x * scale).astype(np.int64)
        short = scale - int(units.sum())
        for k in largest_first:
            weighed = units.copy()
            weighed[k] += short
            blend = weighed / scale
            if np.all(blend >= low - PROPORTION_TOLERANCE) and np.all(blend <= high + PROPORTION_TOLERANCE):
                adjusted = names[k] if short else None
                return dict(zip(names, blend.tolist(), strict=True)), {"decimals": decimals, "adjusted": adjusted}
    return dict(zip(names, x.tolist(), strict=True)), {"decimals": None, "adjusted": None}


# --------------------------------------------------------------------------- the settings that balance the responses
#: The importance a response has in the overall desirability unless the reader sets it: the middle of 1 to 5, so a
#: response can count more, or less, than the others.
IMPORTANCE = 3

#: The importances the reader can give a response, from least to most.
IMPORTANCES = (1, 2, 3, 4, 5)

#: How strongly the good end of a response's acceptable range counts: the exponent of its desirability's ramp. At 1,
#: the desirability rises in a straight line across the range; above 1, a result counts well only near the good
#: end; below 1, most of the range counts nearly as well as the good end. The page offers these, by its words.
SHAPES = {"less": 0.5, "even": 1.0, "more": 2.0, "much_more": 4.0}

#: The shape a response's desirability has unless the reader chooses another.
SHAPE = "even"

#: How many points each axis of the overlay of the acceptable ranges has.
OVERLAY_POINTS = 51

#: The share of a response's results' size a range takes when every result is the same, so it is still a range.
FLAT_RANGE = 0.1

#: What the library's desirability takes of a goal.
DESIRABILITY_KEYS = ("response", "goal", "target", "low", "high", "weight", "importance")


def balance(request: dict) -> dict:
    """Find the settings that balance the responses with a goal, by their overall desirability, on the fitted models.

    Each response with a goal (maximise, minimise, or a target typed) is fitted with its own model
    (``models``, by its column; the design's without one) to the runs that have its result. Its goal
    becomes a Derringer-Suich desirability d, from 0 (not acceptable) to 1 (as good as it gets), over
    an acceptable range (``low`` to ``high``), with an ``importance`` from 1 to 5 and a ``shape``, how
    strongly the good end counts (see :data:`SHAPES`). ``goals`` holds the reader's, by its column; a
    part left out takes its default (see :func:`_balance_goal`). The overall desirability D is the
    geometric mean of the d's, each weighted by its importance.

    The search is :func:`~process_improve.experiments.optimize_responses` with the desirability method
    and every model, over the numeric factors' ranges and inside the limits on the settings, each
    categorical factor held at each of its levels as for the best settings (see :func:`_balance_search`).

    The reply lists each response's goal, as used and as the defaults would have it; the settings,
    coded and in their own units, with ``confirm_at`` rounded for the plant; at them, each response's
    prediction with its 95% confidence and prediction intervals, its d, and D; the response that holds D
    down most (``limiting``); the leverage there against the runs' for the model that leans furthest;
    the limits the settings are on; the confirmation runs judged response by response; the
    ``overlay`` of the acceptable ranges over two factors (see :func:`_overlay`); and ``search``, what
    the code the page shows needs. With fewer than two responses with a goal it says so and holds no
    more. A mixture's responses are balanced over its blends, each fitted with its Scheffé model (see
    :func:`_balanced_blend`).
    """
    whole = _checked_spec(request.get("spec") or {})
    every_run = _runs_frame(request, whole)
    design = _design_runs(every_run)
    aimed = [r for r in _responses(whole) if r.get("goal") in AIMED and (r["goal"] != "target" or "target" in r)]
    if len(aimed) < 2:  # two responses or more are balanced
        return {"balanced": False, "aimed": [r["name"] for r in aimed]}
    mixture = _mixture(whole)
    names = [f["name"] for f in whole["factors"]]
    categorical = _categorical_names(whole)
    raw_goals = request.get("goals") or {}
    raw_models = request.get("models") or {}
    if not isinstance(raw_goals, dict) or not isinstance(raw_models, dict):
        raise ValueError("The page sent goals for the responses this page did not write.")  # noqa: TRY004
    fits: dict[str, _FittedModel] = {}
    goals: list[dict[str, Any]] = []
    for r in aimed:
        spec = _for_response(whole, r["name"])
        model = raw_models.get(r["name"]) or spec["model"]
        model = _mixture_model(model) if mixture else _checked_model(model, names, categorical)
        results = pd.to_numeric(design[r["name"]], errors="coerce").dropna()
        if results.empty:
            raise ValueError(f"Enter results for {_words(r)} first: the balance is found on each response's model.")
        fits[r["name"]] = _blend_fit(design, spec, model) if mixture else _fitted(design, spec, model)
        goals.append(_balance_goal(r, results, raw_goals.get(r["name"])))
    if mixture:
        return _balanced_blend(every_run, whole, aimed, fits, goals)
    region = _spec_region(whole)
    found = _balance_search(fits, goals, whole, region)
    point = _full_point(whole, found["x"], found["levels"])
    in_model = set().union(*(_factors_in(fit.out["coefficients"], spec) for fit in fits.values()))
    numeric = [f for f in whole["factors"] if not _categorical(f)]
    still = np.zeros(len(numeric))  # no way to move from the balanced settings is said, so no end has a "beyond"
    return {
        **_balanced_at(aimed, goals, fits, point),
        "settings": _settings_reply(whole, point, in_model, still, 1.0),
        "limits": _limits_on(whole, region, found["x"], still, 1.0),
        "confirm_at": _practical(whole, point, region),
        "confirmation": _balanced_confirmation(every_run, whole, aimed, fits),
        "overlay": _overlay(whole, found, goals, region),
        "search": {
            "held": _used(found["held"], [c for fit in fits.values() for c in fit.out["coefficients"]]),
            "goals": goals,
            "formulas": {n: fit.formula for n, fit in fits.items()},
        },
    }


def _balanced_at(
    aimed: list[dict], goals: list[dict], fits: dict[str, _FittedModel], point: dict[str, float]
) -> dict[str, Any]:
    """Return what the balance says at the settings it found, or the blend: each response there, D and what holds it.

    Each response's prediction, intervals, d and leverage (see :func:`_balanced_responses`); the overall
    desirability D; the response that holds D down most (``limiting``), its d counted by its
    importance; and the leverage there of the model that leans furthest on its runs.
    """
    responses = _balanced_responses(aimed, goals, fits, point)
    weights = [g["importance"] for g in goals]
    share = [r["d"] ** (w / sum(weights)) for r, w in zip(responses, weights, strict=True)]
    leaning = max(responses, key=lambda r: r["leverage"] / r["max_leverage"] if r["max_leverage"] else 0.0)
    return {
        "balanced": True,
        "aimed": [r["name"] for r in aimed],
        "responses": responses,
        "composite": float(composite_desirability([r["d"] for r in responses], weights)),
        "limiting": responses[int(np.argmin(share))]["name"],
        "coded": point,
        "leverage": leaning["leverage"],
        "max_leverage": leaning["max_leverage"],
        "leverage_of": leaning["name"],
        "extrapolating": any(r["extrapolating"] for r in responses),
    }


def _words(response: dict) -> str:
    """Name a response as the reader does: its label, else its column."""
    return response.get("label") or response["name"]


def _balance_goal(response: dict, results: pd.Series, raw: object) -> dict[str, Any]:
    """Return a response's desirability goal: the reader's acceptable range, importance and shape, else the defaults.

    The default range is the results' own, the least to the greatest: for a response to maximise, the
    worst result scores 0 and the best 1, and for one to minimise the other way round. For a target it
    is centred on the target and reaches the result farthest from it, so every result is inside it. A
    range the results cannot span, all of them alike, is widened by :data:`FLAT_RANGE` of their size.
    ``default_low`` and ``default_high`` keep the defaults beside the range used.
    """
    least, most = float(results.min()), float(results.max())
    if most - least <= ROUND_OFF * max(abs(least), abs(most), 1.0):
        spread = FLAT_RANGE * max(abs(least), 1.0)
        least, most = least - spread, most + spread
    goal = response["goal"]
    if goal == "target":
        target = float(response["target"])
        half = max(target - least, most - target, (most - least) / 2)
        least, most = target - half, target + half
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("The page sent goals for the responses this page did not write.")
    raw = raw or {}
    words = _words(response)
    low = least if raw.get("low") is None else _number(raw["low"])
    high = most if raw.get("high") is None else _number(raw["high"])
    if low is None or high is None or not low < high:
        raise ValueError(f"The acceptable range of {words} must run from a lower value to a higher one.")
    if goal == "target" and not low < response["target"] < high:
        raise ValueError(f"The target of {words}, {response['target']:g}, must lie inside its acceptable range.")
    importance = raw.get("importance", IMPORTANCE)
    if isinstance(importance, bool) or importance not in IMPORTANCES:
        raise ValueError(f"The importance of {words} must be a whole number from 1 to 5.")
    shape = raw.get("shape", SHAPE)
    if shape not in SHAPES:
        raise ValueError(f"The shape of {words}'s desirability must be one the page offers.")
    return {
        "response": response["name"],
        "goal": goal,
        **({"target": float(response["target"])} if goal == "target" else {}),
        "low": float(low),
        "high": float(high),
        "weight": SHAPES[shape],
        "importance": int(importance),
        "shape": shape,
        "default_low": least,
        "default_high": most,
    }


def _desirability_of(goal: dict) -> dict[str, Any]:
    """Return a goal as the library's desirability takes it, without the page's own keys."""
    return {key: goal[key] for key in DESIRABILITY_KEYS if key in goal}


def _balance_search(
    fits: dict[str, _FittedModel], goals: list[dict], spec: dict, region: DesignRegion | None
) -> dict[str, Any]:
    """Find the settings with the best overall desirability: the numeric factors coded, each categorical one's level.

    Each categorical factor some model holds is held at each of its levels, and the block contrasts at 0,
    the average block, as for the best settings (see :func:`_search`): every combination of levels is
    judged at a sample of settings, and the :data:`SEARCHED_LEVELS` best are searched in full with the
    library's optimiser. Returns ``x``, ``levels``, ``held``, each response's ``models`` with the levels
    held, and ``D``.
    """
    from process_improve.experiments import optimize_responses  # noqa: PLC0415 - loaded when the balance is asked for

    numeric = [f for f in spec["factors"] if not _categorical(f)]
    numeric_names = [f["name"] for f in numeric]
    coefficients = {
        name: [{"term": c["term"], "coefficient": c["coefficient"]} for c in fit.out["coefficients"]]
        for name, fit in fits.items()
    }
    in_model = set().union(*(_factors_in(c, spec) for c in coefficients.values()))
    columns = _model_columns(spec)
    blocks = {c["term"]: 0.0 for cs in coefficients.values() for c in cs if _block_contrast(c["term"], columns)}
    desirable = [_desirability_of(g) for g in goals]
    weights = [g["importance"] for g in goals]

    def held_models(levels: dict[str, float]) -> dict[str, list[dict]]:
        return {name: _held_at(c, {**blocks, **_held_columns(levels, spec)}) for name, c in coefficients.items()}

    def overall(models: dict[str, list[dict]], points: np.ndarray) -> np.ndarray:
        predicted = [_predict(models[g["response"]], numeric_names, points) for g in goals]
        each = [[individual_desirability(float(v), g) for v in p] for p, g in zip(predicted, desirable, strict=True)]
        return np.array([composite_desirability(list(d), weights) for d in zip(*each, strict=True)])

    points = _sample(len(numeric), region)
    combinations = _searched_levels(spec, in_model, lambda levels: float(overall(held_models(levels), points).max()))
    best: dict[str, Any] | None = None
    for levels in combinations:
        models = held_models(levels)
        heard: list[str] = []
        x = np.zeros(0)
        if numeric:
            with _listening() as heard:
                result = optimize_responses(
                    [{"response_name": n, "coefficients": m, "factor_names": numeric_names} for n, m in models.items()],
                    goals=desirable,
                    factor_ranges={f["name"]: {"low": f["low"], "high": f["high"]} for f in numeric},
                    region=region,
                )["desirability"]
            x = np.array([result["optimal_coded"][n] for n in numeric_names])
            at_end = np.abs(np.abs(x) - 1) <= 1e-9  # stopped on an end of the range, to rounding: put it there
            x[at_end] = np.sign(x[at_end])
        x = _unmodelled_at_centre(x, numeric_names, in_model, region)
        value = float(overall(models, x[None, :])[0])
        if best is None or value > best["D"] + 1e-12:
            best = {
                "x": x,
                "levels": levels,
                "held": {**blocks, **_held_columns(levels, spec)},
                "models": models,
                "D": value,
                "heard": heard,
            }
    assert best is not None  # there is always one combination of levels, the empty one
    _relay(best.pop("heard"))
    return best


def _balanced_responses(
    aimed: list[dict], goals: list[dict], fits: dict[str, _FittedModel], point: dict[str, float]
) -> list[dict[str, Any]]:
    """Describe each response at the balanced settings: its goal, its prediction and intervals, its d and leverage."""
    out = []
    for r, goal in zip(aimed, goals, strict=True):
        fit = fits[r["name"]]
        predicted, confidence, prediction, leverage = fit.at(point)
        out.append(
            {
                "name": r["name"],
                "label": _words(r),
                "units": r.get("units", ""),
                **{key: value for key, value in goal.items() if key != "response"},
                "predicted": predicted,
                "confidence": confidence,
                "prediction": prediction,
                "d": float(individual_desirability(predicted, _desirability_of(goal))),
                "leverage": leverage,
                "max_leverage": fit.max_leverage,
                "extrapolating": leverage > fit.max_leverage * (1 + 1e-9) + 1e-12,
            }
        )
    return out


def _balanced_confirmation(
    runs: pd.DataFrame, spec: dict, aimed: list[dict], fits: dict[str, _FittedModel]
) -> list[dict[str, Any]]:
    """Judge the confirmation runs on the run sheet response by response, each against its own model.

    The groups are those of :func:`_confirmation`, by their settings; each lists every balanced
    response's verdict in ``responses``: its results' average against the prediction interval for the
    average of that many runs.
    """
    judged = {r["name"]: _confirmation(runs, _for_response(spec, r["name"]), fits[r["name"]]) for r in aimed}
    first = judged[aimed[0]["name"]]
    keep = ("done", "mean", "predicted", "interval", "inside")
    return [
        {
            "rows": group["rows"],
            "at": group["at"],
            "runs": group["runs"],
            "responses": [{"name": r["name"], **{k: judged[r["name"]][i][k] for k in keep}} for r in aimed],
        }
        for i, group in enumerate(first)
    ]


def _overlay(spec: dict, found: dict, goals: list[dict], region: DesignRegion | None) -> dict[str, Any] | None:
    """Return the overlay of the acceptable ranges over two numeric factors, the others at the balanced settings.

    The two factors are those that move the responses' desirabilities most: each factor's main effects,
    summed over the responses, each over its acceptable range's width. Over a grid of their coded
    settings, each response's prediction (``z``) and the levels its acceptable boundary is at (``levels``:
    the low end to maximise, the high end to minimise, both for a target); ``acceptable``, 1 where every
    response has a desirability above 0; and ``runnable``, 1 where the limits on the settings allow the
    setting, or None without limits. ``at`` is the balanced settings, coded; ``ticks`` mark each axis
    with the factor's own settings; ``held`` lists the other factors at the balanced settings, in their
    own units or levels. None when the study has fewer than two numeric factors.
    """
    numeric = [f for f in spec["factors"] if not _categorical(f)]
    if len(numeric) < 2:  # an overlay is over two factors
        return None
    names = [f["name"] for f in numeric]
    size = dict.fromkeys(names, 0.0)
    for goal in goals:
        for c in found["models"][goal["response"]]:
            if c["term"] in size:
                size[c["term"]] += abs(c["coefficient"]) / (goal["high"] - goal["low"])
    pair = sorted(names, key=lambda n: -size[n])[:2]
    axis = np.linspace(-1.0, 1.0, OVERLAY_POINTS)
    u, v = np.meshgrid(axis, axis)  # v down the rows, u across: z[row][column] as Plotly draws it
    points = np.tile(found["x"], (u.size, 1))
    points[:, names.index(pair[0])] = u.ravel()
    points[:, names.index(pair[1])] = v.ravel()
    acceptable = np.ones(u.size, dtype=bool)
    surfaces = []
    for goal in goals:
        z = _predict(found["models"][goal["response"]], names, points)
        inside, levels = _acceptable(goal, z)
        acceptable &= inside
        surfaces.append({"name": goal["response"], "z": _grid(z, u.shape), "levels": levels})
    runnable = None
    if region is not None:
        runnable = region.feasible(points[:, [names.index(n) for n in region.names]]).astype(int)
        runnable = runnable.reshape(u.shape).tolist()
    factors = {f["name"]: f for f in spec["factors"]}
    point = _full_point(spec, found["x"], found["levels"])
    return {
        "x": pair[0],
        "y": pair[1],
        "grid": axis.tolist(),
        "ticks": {name: _ticks(factors[name]) for name in pair},
        "responses": surfaces,
        "acceptable": acceptable.astype(int).reshape(u.shape).tolist(),
        "runnable": runnable,
        "at": [point[pair[0]], point[pair[1]]],
        "held": [
            {"name": f["name"], "value": _in_units(f, point[f["name"]])}
            for f in spec["factors"]
            if f["name"] not in pair
        ],
    }


def _acceptable(goal: dict, z: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Return where a response is acceptable, a desirability above 0, and the levels its acceptable boundary is at."""
    if goal["goal"] == "maximize":
        return z > goal["low"], [goal["low"]]
    if goal["goal"] == "minimize":
        return z < goal["high"], [goal["high"]]
    return (z > goal["low"]) & (z < goal["high"]), [goal["low"], goal["high"]]


def _in_units(factor: dict, coded: float) -> float | str:
    """Return a factor's coded setting in its own units, or a categorical factor's level."""
    if _categorical(factor):
        return _level_at(factor, coded)
    return (factor["low"] + factor["high"]) / 2 + coded * (factor["high"] - factor["low"]) / 2


def _grid(values: np.ndarray, shape: tuple[int, ...]) -> list[list[float]]:
    """Return a surface's values on its grid, rows of columns, each to six significant figures: enough to draw it."""
    return [[float(f"{v:.6g}") for v in row] for row in values.reshape(shape)]


def _balanced_blend(
    every_run: pd.DataFrame, whole: dict, aimed: list[dict], fits: dict[str, _FittedModel], goals: list[dict]
) -> dict:
    """Find the blend that balances the responses with a goal, by their overall desirability, on their Scheffé models.

    The search is :func:`balance`'s, over the region of blends (see :func:`_best_blend`): every
    response's model, fitted on the proportions, with its goal as a desirability. The reply is
    :func:`balance`'s, with each component's proportion in ``settings`` (see :func:`_blend_settings`)
    and the blend weighed out for the confirmation runs (see :func:`_weighed`). ``overlay`` is drawn
    over the triangle of three components (see :func:`_blend_overlay`), and is None for more.
    """
    from process_improve.experiments import optimize_responses  # noqa: PLC0415 - loaded when the balance is asked for

    names = [f["name"] for f in whole["factors"]]
    models = {
        name: [{"term": c["term"], "coefficient": c["coefficient"]} for c in fit.out["coefficients"]]
        for name, fit in fits.items()
    }
    with _listening() as heard:
        found = optimize_responses(
            [{"response_name": name, "coefficients": model, "factor_names": names} for name, model in models.items()],
            goals=[_desirability_of(g) for g in goals],
            region=DesignRegion(_components(whole)),
        )["desirability"]
    _relay(heard)
    low, high = _bounds(whole)
    x = _on_bounds(found["optimal_coded"], names, low, high)
    confirm_at, rounding = _weighed(whole, x)
    return {
        **_balanced_at(aimed, goals, fits, dict(zip(names, x.tolist(), strict=True))),
        "settings": _blend_settings(whole, x, np.zeros(len(x)), 1.0),  # no way to move is said: no bound has a "beyond"
        "limits": [],
        "confirm_at": confirm_at,
        "rounding": rounding,
        "confirmation": _balanced_confirmation(every_run, whole, aimed, fits),
        "overlay": _blend_overlay(whole, models, goals, x),
        "search": {"held": {}, "goals": goals, "formulas": {n: fit.formula for n, fit in fits.items()}},
    }


def _blend_overlay(
    spec: dict, models: dict[str, list[dict]], goals: list[dict], x: np.ndarray
) -> dict[str, Any] | None:
    """Return the overlay of the acceptable ranges over the triangle of three components; None for more.

    Over the triangle's grid of blends (see :func:`_ternary_grid`), as ``a``, ``b`` and ``c``: each
    response's prediction (``z``), the levels its acceptable boundary is at (``levels``, as
    :func:`_acceptable` gives them) and where it crosses each (``lines``, see :func:`_crossings`);
    ``acceptable``, 1 at a blend where every response has a desirability above 0; and ``at``, the
    balanced blend. Four components or more make a solid, which no picture of the triangle holds.
    """
    names = [f["name"] for f in spec["factors"]]
    if len(names) != 3:  # the triangle of three components
        return None
    grid, places = _ternary_grid(spec)
    acceptable = np.ones(len(grid), dtype=bool)
    surfaces = []
    for goal in goals:
        z = _predict(models[goal["response"]], names, grid)
        inside, levels = _acceptable(goal, z)
        acceptable &= inside
        surfaces.append(
            {
                "name": goal["response"],
                "z": [float(f"{v:.6g}") for v in z],
                "levels": levels,
                "lines": [_crossings(grid, places, z, level) for level in levels],
            }
        )
    return {
        "kind": "ternary",
        "components": names,
        "a": grid[:, 0].tolist(),
        "b": grid[:, 1].tolist(),
        "c": grid[:, 2].tolist(),
        "responses": surfaces,
        "acceptable": acceptable.astype(int).tolist(),
        "at": x.tolist(),
    }


def _crossings(
    grid: np.ndarray, places: list[tuple[int, int]], z: np.ndarray, level: float
) -> dict[str, list[float | None]]:
    """Return where a response crosses a level over the triangle's grid, as segments of line for a ternary plot.

    The grid's lattice splits the triangle into small triangles. One whose corners lie on both sides
    of the level holds a segment of the line, from where it crosses one side to where it crosses
    another, each found by straight-line interpolation along the side. The segments are listed end
    to end in ``a``, ``b`` and ``c``, each followed by None, so a plot draws them apart.
    """
    at = {place: k for k, place in enumerate(places)}
    ends: dict[str, list[float | None]] = {"a": [], "b": [], "c": []}
    for i, j in places:
        for corners in (((i, j), (i + 1, j), (i, j + 1)), ((i + 1, j), (i, j + 1), (i + 1, j + 1))):
            if not all(c in at for c in corners):  # a corner past the triangle, or outside the bounds
                continue
            k = [at[c] for c in corners]
            v = z[k] - level
            crossed = [
                grid[k[p]] + v[p] / (v[p] - v[q]) * (grid[k[q]] - grid[k[p]])
                for p, q in ((0, 1), (1, 2), (2, 0))
                if (v[p] >= 0) != (v[q] >= 0)
            ]
            if len(crossed) == 2:  # a straight line crosses two sides of a triangle, or none
                for point in crossed:
                    for axis, value in zip("abc", point, strict=True):
                        ends[axis].append(round(float(value), 6))
                for axis in "abc":
                    ends[axis].append(None)
    return ends


# --------------------------------------------------------------------------- what if: the models at any settings
def predictors(request: dict) -> dict:
    """Return each response's fitted model as numbers, so the page predicts at any settings without asking again.

    Each response with results is fitted with its own model (``models``, by its column; the design's
    without one) as Learn fits it (:func:`_fitted`). Its reply holds the formula fitted, for the code
    the page shows, and what :meth:`_FittedModel.at` needs: each term as the columns it multiplies
    (``I(T ** 2)`` as ``T`` twice), the coefficients, ``(X'X)^-1``, the residual mean square and the
    t value of 95% intervals (both None without residual degrees of freedom), and the largest
    leverage of the runs fitted. The page (``predictAt`` in ``web/best.js``) then gives the
    prediction, its intervals and its leverage as the reader moves a setting, at once. A block
    contrast is 0 in the average block, where the page predicts, so its term, its coefficient and
    its row and column of ``(X'X)^-1`` are left out: they add nothing there.

    ``factors`` says how a setting becomes the columns, the same for every response: a numeric
    factor is coded from its range, and a categorical factor's level is the columns
    :func:`_held_columns` writes for it, its contrasts and its indicators. A response without
    results is ``waiting``; one whose model cannot be fitted has its ``error``. A mixture's blends
    are not predicted here yet: the reply says ``mixture`` and holds no responses.
    """
    whole = _checked_spec(request.get("spec") or {})
    if _mixture(whole):
        return {"mixture": True, "factors": [], "responses": []}
    every_run = _runs_frame(request, whole)
    design = _design_runs(every_run)
    names = [f["name"] for f in whole["factors"]]
    categorical = _categorical_names(whole)
    raw_models = request.get("models") or {}
    if not isinstance(raw_models, dict):
        raise ValueError("The page sent models for the responses this page did not write.")  # noqa: TRY004
    responses = []
    for r in _responses(whole):
        words = {"name": r["name"], "label": _words(r), "units": r.get("units") or ""}
        if pd.to_numeric(design[r["name"]], errors="coerce").dropna().empty:
            responses.append(words | {"waiting": True})
            continue
        spec = _for_response(whole, r["name"])
        try:
            model = _checked_model(raw_models.get(r["name"]) or spec["model"], names, categorical)
            responses.append(words | _predictor(_fitted(design, spec, model)))
        except ValueError as exc:
            responses.append(words | {"error": str(exc)})
    return {"mixture": False, "factors": [_coding(f, whole) for f in whole["factors"]], "responses": responses}


def _predictor(fitted: _FittedModel) -> dict[str, Any]:
    """Return a fit as the page predicts from it: its terms but the block contrasts, as numbers."""
    kept = [i for i, term in enumerate(fitted.terms) if not _block_contrast(term, fitted.names)]
    return {
        "formula": fitted.formula,
        "terms": [list(_parts(fitted.terms[i])) for i in kept],
        "coefficients": [float(fitted.beta[i]) for i in kept],
        "inverse": fitted.inverse[np.ix_(kept, kept)].tolist(),
        "mse": fitted.mse,
        "t": fitted.t,
        "df": fitted.df,
        "level": LEVEL,
        "max_leverage": fitted.max_leverage,
    }


def _coding(factor: dict, spec: dict) -> dict[str, Any]:
    """Say how a factor's setting becomes the columns the models are fitted on.

    A numeric factor's column is its setting coded from its range, as :func:`_coded_column` codes it.
    A categorical factor's level, coded as the library codes it (-1 for its first, +1 for its last),
    is the columns :func:`_held_columns` writes: its own coded column, or its contrasts, and its
    indicators.
    """
    if not _categorical(factor):
        return {"name": factor["name"], "kind": "numeric", "low": factor["low"], "high": factor["high"]}
    levels = _levels_of(factor)
    codes = np.linspace(-1.0, 1.0, len(levels))
    columns = [_held_columns({factor["name"]: float(code)}, spec) for code in codes]
    return {"name": factor["name"], "kind": "categorical", "levels": levels, "columns": columns}


def outside_limits(request: dict) -> dict:
    """Name the limits on the settings that some settings break, each by its place in the study's list.

    ``settings`` holds each numeric factor's setting, a number in its own units inside its range; a
    limit names numeric factors only. Each limit is judged alone, as the library judges a point
    (:meth:`~process_improve.experiments.DesignRegion.feasible`).
    """
    spec = _checked_spec(request.get("spec") or {})
    settings = request.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("The page sent settings this page did not write.")  # noqa: TRY004
    factors = _factors(spec)
    numeric = [f for f in spec["factors"] if not _categorical(f)]
    x = np.array([[_coded_value(_factor_of(f), _setting_of(f, settings)) for f in numeric]])
    broken = []
    for i, limit in enumerate(spec.get("constraints") or []):
        if not bool(_region(factors, [limit["expression"]]).feasible(x)[0]):
            broken.append({"index": i, "expression": limit["expression"]})
    return {"broken": broken}


def _setting_of(factor: dict, settings: dict) -> float:
    """Return a numeric factor's setting as sent, refusing one that is no number or lies outside its range."""
    value = _number(settings.get(factor["name"]))
    if value is None or not min(factor["low"], factor["high"]) <= value <= max(factor["low"], factor["high"]):
        raise ValueError(f"Set {factor['name']} to a number from {factor['low']:g} to {factor['high']:g}.")
    return value


# --------------------------------------------------------------------------- runs added to a study
#: The ways Learn offers to add runs to a study, in the order it lists them, each the library's
#: ``augment_design`` with the augmentation it names:
#:
#: - ``foldover``: every run of a two-level design again with each setting switched to its other end, which
#:   separates every main effect from the two-factor interactions it is aliased with;
#: - ``axial``: runs on the faces of the ranges and at the center, which turn a two-level design that fits the
#:   interactions into a face-centred central composite design that fits the full quadratic model;
#: - ``center``: runs at the center of the ranges, to test for curvature and estimate the noise;
#: - ``optimal``: runs chosen for the model the study aims to learn (D-optimal), the runs made so far kept;
#: - ``replicate``: every run again, to estimate the noise from the repeats.
#:
#: ``blocked`` says whether the runs added are a block of their own. Runs made later can sit at a shifted level,
#: and a block keeps that shift out of the effects. Center runs are never a block of their own: the shift between
#: the blocks would then be the curvature they are added to test. ``name`` names the runs in a sentence.
EXTENSIONS: dict[str, dict[str, Any]] = {
    "foldover": {"augmentation": "foldover", "blocked": True, "name": "The runs of a fold-over"},
    "axial": {"augmentation": "upgrade_to_rsm", "blocked": True, "name": "Axial runs"},
    "center": {"augmentation": "add_center_points", "blocked": False, "name": "Center runs"},
    "optimal": {"augmentation": "add_runs_optimal", "blocked": True, "name": "Runs chosen for the model"},
    "replicate": {"augmentation": "replicate", "blocked": True, "name": "The runs made again"},
}

#: How many center runs Learn adds to a design that has none: the library's own number.
EXTRA_CENTER_RUNS = 3

#: The option that marks a study Learn added runs to: runs as given, made by the page in random order within each set,
#: rather than handed over in an order of their own.
GROWN = "grown"

#: The fewest runs a model should leave to estimate the noise, as Design marks a design's spare runs (logic.js's
#: SPARE_FOR_NOISE). With fewer, Learn offers to make every run again.
SPARE_FOR_NOISE = 3

#: Each model in the words a sentence names it with, as "the full quadratic model".
MODEL_WORDS = {
    "main_effects": "main effects",
    "interactions": "main effects and two-factor interactions",
    MAIN_QUADRATIC: "main effects and squares",
    "quadratic": "full quadratic",
}


def next_runs(request: dict) -> dict:
    """Say what the results call for next, and list the ways Learn offers to add runs to the study.

    The study is the page's run sheet (``spec``, ``columns`` and ``rows``), with its results as the
    analysis takes them (``responses``, ``more_responses`` and ``response``); its confirmation runs
    are no runs of the design, and are left out of the offers. ``recommendation`` is what the
    results call for (see :func:`_recommended`), or None, with ``waiting`` saying why when there is
    a reason the reader can act on. Each offer is one of :data:`EXTENSIONS`, in that order, and
    only where it applies: ``kind``, its ``label`` for the button, the ``runs`` it adds, ``why`` in a
    sentence or two, ``model``, the largest model the design then estimates in full, and ``blocked``.
    An extension whose runs would break a limit on the settings is not offered, and ``notes`` says so.
    A mixture is offered nothing: its runs are blends, which these extensions do not make.
    """
    spec, _, design = _study_runs(request)
    if _mixture(spec):
        return {"offers": [], "notes": [MIXTURE_EXTENSIONS], "recommendation": None, "waiting": None}
    recommendation, waiting = _recommended(request, spec)
    offers, notes = [], []
    for kind in EXTENSIONS:
        made = _extension(spec, design, kind)
        if made is None:
            continue
        if not _runnable(spec, made["new"]):
            name = EXTENSIONS[kind]["name"]
            notes.append(f"{name} would fall outside your limits on the settings, so Learn does not offer them.")
            continue
        shown = {key: made[key] for key in ("kind", "label", "why", "model", "blocked")}
        offers.append(shown | {"runs": len(made["new"])})
    return {"offers": offers, "notes": notes, "recommendation": recommendation, "waiting": waiting}


#: The option marking a study started with its ranges widened because nothing stood out in the study before it:
#: if nothing stands out again, the next step stops rather than widen them a second time.
WIDENED = "widened"

#: What the next step says while a run of the design, or a confirmation run, has no result yet.
RESULTS_FIRST = "Once every run has its result, the results say what to run next."

#: The steps whose runs Learn adds to the run sheet, as ``add_runs`` adds the runs of an extension.
ADDED_STEPS = ("add_center_points", "replicate", "de_alias", "augment_to_rsm", "add_runs_for_model")

#: The kind ``add_runs`` takes for the runs the results call for: :func:`_recommended`'s step, made again from the
#: same request, which gives the same runs.
RECOMMENDED = "recommended"


def _recommended_runs(request: dict, whole: dict, design: pd.DataFrame) -> dict[str, Any] | None:
    """Make the runs the results call for, as :func:`_extension` makes an extension's; None when they call for none.

    They are a block of their own, a later round of the study, except centre runs, whose block
    would be the curvature they are added to test.
    """
    found = _next_step(request, whole)
    if not isinstance(found, dict) or found["action"] not in ADDED_STEPS:
        return None
    kinds = ["categorical" if _categorical(f) else "numeric" for f in whole["factors"]]
    new = _coded(pd.DataFrame(found["runs"]), whole["factors"]).to_numpy(dtype=float)
    before = _coded(design, whole["factors"]).to_numpy(dtype=float)
    headline = found["headline"].replace("{", "{{").replace("}", "}}")
    return {
        "kind": RECOMMENDED,
        "new": new,
        "label": f"Add these {len(new)} runs",
        "why": found["headline"],
        "said": f"Runs {{first}} to {{last}} are the runs the results called for. {headline}",
        "model": _largest_model(np.vstack([before, new]), kinds),
        "blocked": found["action"] != "add_center_points",
    }


def _recommended(request: dict, whole: dict) -> tuple[dict[str, Any] | None, str | None]:
    """Return what the results call for next, as the page shows it, and why there is nothing to show instead.

    The decision is :func:`~process_improve.experiments.next_step.recommend_next_step`'s, for the
    response the page analyses, toward its goal, from every run of the design that has a result.
    The latest :data:`CONFIRMATION_RUNS` confirmation runs are judged as confirmation runs; any
    made before them missed (else the study would have stopped) and are runs of the study, in its
    latest block. A study whose runs are in blocks has a shift fitted for each.

    The reply holds the step (``action``, ``headline``, ``reasons``, ``after``, ``n_runs``), the
    model it read (``judged_by``, ``active``), and how the page carries it out: ``does`` is
    ``confirm`` (confirmation runs at ``settings``, rounded for the plant), ``runs`` (``add_runs``
    with the kind ``recommended``), ``study`` (a new study with the ``ranges`` given) or None.
    ``outside_limits`` says the runs would break a limit on the settings, so the page offers none.
    Nothing is shown for a goal of learning, a mixture, or before the results are sent.
    """
    found = _next_step(request, whole)
    if found is None or isinstance(found, str):
        return None, found
    return _shown_step(_for_response(whole, request.get("response")), found), None


def _next_step(request: dict, whole: dict) -> dict[str, Any] | str | None:
    """Return recommend_next_step's reply for the run sheet, or why it has none: a sentence for the page, or None.

    None is for a study the step does not apply to: a goal of learning, a target not yet typed, a
    mixture, or a request that sends no results. A study of more than :data:`MAX_INTERACTION_FACTORS` factors
    is judged on its main effects, as Learn fits them: judging every term its runs can test, the step would
    write a formula of hundreds of interactions, deeper than Python's recursion limit lets patsy read. A
    study of more than :data:`MAX_NEXT_STEP_FACTORS` factors has no step worked out, and the sentence says why.
    """
    spec = _for_response(whole, request.get("response"))
    info = spec.get("response_info") or {}
    goal, target = info.get("goal", "learn"), info.get("target")
    sent = request.get("responses") is not None or bool(request.get("more_responses"))
    if goal not in AIMED or (goal == "target" and target is None) or _mixture(spec) or not sent:
        return None
    if (k := len(spec["factors"])) > MAX_NEXT_STEP_FACTORS:
        return (
            f"Past {MAX_NEXT_STEP_FACTORS} factors, the page does not work out what to run next: it weighs runs for "
            f"every factor whose effect stands out, and with {k} factors that can take minutes here. The factors whose "
            "effects stand out are the ones to study next."
        )
    try:
        from process_improve.experiments.next_step import recommend_next_step  # noqa: PLC0415 - not in older releases
    except ImportError:
        return "This version of process-improve cannot say what the results call for yet."
    response = spec["response"]
    runs = _runs_frame(request, whole)
    confirming = _confirming(runs)
    design = runs.loc[~confirming].drop(columns=[CONFIRMATION], errors="ignore")
    confirmations = runs.loc[confirming]
    if "Run order" in runs.columns:
        confirmations = confirmations.sort_values("Run order", kind="stable")
    if design[response].isna().any() or confirmations[response].isna().any():
        return RESULTS_FIRST
    latest = confirmations.iloc[-CONFIRMATION_RUNS:]
    earlier = confirmations.iloc[: len(confirmations) - len(latest)]
    block = BLOCK if BLOCK in design.columns and len(set(design[BLOCK].dropna())) > 1 else None
    if block is not None and len(earlier):
        earlier = earlier.assign(**{BLOCK: design[BLOCK].dropna().iloc[-1]})  # made after every run of the design
    history = ["widen_ranges"] if spec.get("options", {}).get(WIDENED) else []
    history += ["confirm"] * math.ceil(len(confirmations) / CONFIRMATION_RUNS)
    factors = _factors(whole)
    mains = [f.name for f in factors] if len(factors) > MAX_INTERACTION_FACTORS else None
    try:
        with _listening() as heard:
            found = recommend_next_step(
                factors,
                pd.concat([design, earlier.drop(columns=[CONFIRMATION], errors="ignore")], ignore_index=True),
                response,
                goal=goal,  # type: ignore[arg-type]
                target=target,  # type: ignore[arg-type]
                confirmation_runs=latest if len(latest) else None,
                n_confirmation_runs=CONFIRMATION_RUNS,
                history=history,
                block=block,
                model_terms=mains,
            )
        _relay(heard)
    except ValueError as exc:
        return f"The results cannot say what to run next: {exc}"
    return found


def _shown_step(spec: dict, found: dict[str, Any]) -> dict[str, Any]:
    """Return a step of recommend_next_step as the page shows it, with how the page carries it out."""
    action = found["action"]
    shown: dict[str, Any] = {key: found[key] for key in ("action", "headline", "reasons", "after", "n_runs")} | {
        "response": spec["response"],
        "judged_by": found["model"]["judged_by"],
        "active": found["evidence"]["active_terms"],
        "does": None,
        "label": None,
        "settings": None,
        "ranges": None,
        "outside_limits": False,
    }
    if action == "confirm":
        settings = _practical(spec, found["evidence"]["optimum"]["coded"], _spec_region(spec))
        coded = np.array(
            [
                [
                    _coded_value(_factor_of(f), settings[f["name"]]) if not _categorical(f) else 0.0
                    for f in spec["factors"]
                ]
            ]
        )
        shown |= {"settings": settings, "does": "confirm", "label": f"Add {found['n_runs']} confirmation runs here"}
        shown["outside_limits"] = not _runnable(spec, coded)
    elif action in ADDED_STEPS:
        coded = _coded(pd.DataFrame(found["runs"]), spec["factors"]).to_numpy(dtype=float)
        shown |= {"does": "runs", "label": f"Add these {found['n_runs']} runs"}
        shown["outside_limits"] = not _runnable(spec, coded)
    elif action in ("widen_ranges", "move_region"):
        shown |= {"does": "study", "label": "Start a new study with these ranges", "ranges": _moved(spec, found)}
    if shown["outside_limits"]:
        shown |= {"does": None, "label": None}
    return shown


def _moved(spec: dict, found: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the factors of the new study a step points to: the ranges widened, or moved along the path.

    Moving, each factor the path moves has its range shifted by its share of half the range, to two
    significant figures, as the page's own next study moves it (best.js's ``nextRanges``); the
    others, and every categorical factor, keep theirs.
    """
    widened = found.get("new_ranges") or {}
    shares = found.get("direction") or {}
    moved = []
    for f in spec["factors"]:
        if _categorical(f):
            moved.append(dict(f))
        elif f["name"] in widened:
            low, high = widened[f["name"]]
            moved.append({**f, "low": float(f"{low:.12g}"), "high": float(f"{high:.12g}")})
        elif f["name"] in shares:
            step = float(f"{shares[f['name']] * (f['high'] - f['low']) / 2:.2g}")
            moved.append({**f, "low": float(f"{f['low'] + step:.12g}"), "high": float(f"{f['high'] + step:.12g}")})
        else:
            moved.append(dict(f))
    return moved


#: What Learn says of a mixture, to which it offers to add no runs.
MIXTURE_EXTENSIONS = "A mixture's runs are blends, and the runs Learn adds to a design are not, so it offers none here."


def add_runs(request: dict) -> dict:
    """Add the runs of one extension (``kind``, one of :data:`EXTENSIONS`) to the end of the study's run sheet.

    The runs made so far stay as they are, in their places, so the results the page holds for them still
    belong to them; the runs added follow, in random order, with blank results. The order is drawn with the
    study's seed (its ``random_state`` option) plus the number of runs before them. A blocked extension puts
    them in a block of their own: the run sheet gains its Block column the first time, the runs before
    in block 1, and a confirmation run in no block. The study is then a design of runs as given (its
    ``design_type`` is ``given``), estimating the largest model its runs estimate in full, and its notes
    say what was added and what the design now separates. The reply is a design as :func:`make_design`
    returns it, less its workbook, with ``added``, the number of runs added.
    """
    kind = request.get("kind")
    if kind not in EXTENSIONS and kind != RECOMMENDED:
        raise ValueError(f"Unknown way to add runs {kind!r}.")
    spec, runs, design = _study_runs(request)
    made = _made_for(request, spec, design, kind)
    names = [f["name"] for f in spec["factors"]]
    # The runs before are sent back as they came, cell for cell; only a Block column the sheet gains is new to them.
    columns = [str(c) for c in request["columns"]]
    rows = [list(row) for row in request["rows"]]
    if made["blocked"] and BLOCK not in columns:  # the runs before are block 1; a confirmation run is in no block
        at = columns.index(names[0])
        columns.insert(at, BLOCK)
        for row, confirming in zip(rows, _confirming(runs), strict=True):
            row.insert(at, None if confirming else 1)
    before = pd.DataFrame(rows, columns=columns)
    first_std, first_run = _last(before, "Std order") + 1, _last(before, "Run order") + 1
    block = _last(before, BLOCK) + 1 if made["blocked"] else None
    # The study's seed plus the runs before, so each set of runs added has an order of its own. A study saved before
    # studies had a seed keeps the page's one seed, as its report says.
    seed = (spec.get("options") or {}).get("random_state")
    rng = np.random.default_rng(RUN_ORDER_SEED if seed is None else int(seed) + len(before))
    order = rng.permutation(len(made["new"]))
    for place, i in enumerate(order):  # in run order, each with its place in the standard order of the runs added
        settings = {f["name"]: _setting(f, made["new"][i][j]) for j, f in enumerate(spec["factors"])}
        numbers = {"Std order": first_std + int(i), "Run order": first_run + place, BLOCK: block}
        rows.append([settings.get(c, numbers.get(c)) for c in columns])
    sheet = _runs_frame({"columns": columns, "rows": rows}, spec)
    n_blocks = _last(sheet, BLOCK) if BLOCK in columns else 1
    stored = {
        **{key: value for key, value in spec.items() if key not in ("design_type", "options", "model")},
        "design_type": GIVEN,
        "model": made["model"],
        "options": {
            GROWN: True,
            **({"random_state": int(seed)} if seed is not None else {}),
            **({"n_blocks": n_blocks} if n_blocks > 1 else {}),
            **({WIDENED: True} if spec.get("options", {}).get(WIDENED) else {}),
        },
    }
    extended = _design_runs(sheet)
    coded = _coded(extended, spec["factors"])
    groups = _aliased_groups(coded.to_numpy(dtype=float), names)
    notes = [made["said"].format(first=first_run, last=first_run + len(order) - 1), _aliasing_sentence(groups)]
    if n_blocks > 1:
        notes.append(_blocks_sentence(coded.to_numpy(dtype=float), extended[BLOCK].tolist(), stored))
    if spec.get("constraints") and _runnable(spec, coded.to_numpy(dtype=float)):
        notes.append(INSIDE_LIMITS)
    factors = _factors(spec)
    return {
        "n_runs": len(rows),
        "columns": columns,
        "rows": rows,
        "outside_range": _outside_range(extended, factors),
        "model": made["model"],
        "facts": {
            "resolution": None,
            "generators": [],
            "defining_relation": [],
            "aliases": groups,
            "center_runs": int(np.all(np.abs(coded.to_numpy(dtype=float)) < ALIAS_TOLERANCE, axis=1).sum()),
            "notes": notes,
        },
        "spec": stored,
        "added": len(order),
    }


def _made_for(request: dict, spec: dict, design: pd.DataFrame, kind: str) -> dict[str, Any]:
    """Make the runs ``add_runs`` adds of one kind, or refuse, saying why the study cannot take them."""
    if _mixture(spec):
        raise ValueError("A mixture's runs are blends, and the runs Learn adds to a design are not.")
    if kind == RECOMMENDED:
        made = _recommended_runs(request, spec, design)
        if made is None:
            raise ValueError("The results call for no runs to add to this study: see what the next step says.")
        name = "The runs the results call for"
    else:
        made = _extension(spec, design, kind)
        if made is None:
            raise ValueError(f"This study's runs cannot be extended that way: {EXTENSION_REFUSALS[kind]}")
        name = EXTENSIONS[kind]["name"]
    if not _runnable(spec, made["new"]):
        raise ValueError(f"{name} would fall outside your limits on the settings.")
    return made


#: Why a study's runs cannot be extended each way, as the refusal of :func:`add_runs` says it.
EXTENSION_REFUSALS = {
    "foldover": "a fold-over takes a two-level design whose main effects are aliased with two-factor interactions.",
    "axial": (
        "axial runs take a two-level design of numeric factors that fits the two-factor interactions and not yet "
        "the full quadratic model."
    ),
    "center": "center runs take a two-level design of numeric factors, in no blocks, with no run at the center yet.",
    "optimal": (
        "runs for the model the study aims to learn take numeric factors, no limits on the settings, and a model the "
        "design cannot fit yet."
    ),
    "replicate": "the model leaves enough runs to estimate the noise already.",
}


def _study_runs(request: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return the study's checked specification, its whole run sheet, and the runs of its design."""
    spec = _checked_spec(request.get("spec") or {})
    runs = _runs_frame({"columns": request.get("columns"), "rows": request.get("rows")}, spec)
    return spec, runs, _design_runs(runs)


def _last(sheet: pd.DataFrame, column: str) -> int:
    """Return the largest whole number in a column of the run sheet, or 0 when it has none (or no such column)."""
    if column not in sheet.columns:
        return 0
    largest = pd.to_numeric(sheet[column], errors="coerce").max()
    return int(largest) if pd.notna(largest) else 0


def _setting(factor: dict, coded: float) -> float | str:
    """Return a coded setting in the factor's own units, rounded off as a run sheet lists it, or a category's level."""
    if _categorical(factor):
        return _level_at(factor, coded)
    return float(f"{_in_units(factor, coded):.12g}")


def _runnable(spec: dict, new: np.ndarray) -> bool:
    """Say whether runs, coded, meet every limit on the settings; a study without limits runs any."""
    region = _spec_region(spec)
    if region is None or not len(new):
        return True
    names = [f["name"] for f in spec["factors"]]
    return bool(region.feasible(new[:, [names.index(name) for name in region.names]]).all())


def _model_rank(coded: np.ndarray, kinds: list[str], model: str) -> tuple[int, int]:
    """Return a model's coefficient count and the rank the runs give it: equal when they estimate it in full."""
    base = "quadratic" if model == "quadratic" else "interactions" if model == "interactions" else "main_effects"
    matrix = model_matrix(coded, kinds, base)
    if model == MAIN_QUADRATIC:
        matrix = np.column_stack([matrix, *[coded[:, [j]] ** 2 for j, kind in enumerate(kinds) if kind == "numeric"]])
    return matrix.shape[1], int(np.linalg.matrix_rank(matrix))


def _largest_model(coded: np.ndarray, kinds: list[str], blocks: list[int] | None = None) -> str:
    """Return the largest of :data:`MODELS` the runs estimate in full, as a design of runs as given is analysed.

    Runs in blocks estimate a model beside the blocks' effects, which take a column for each block after the first.
    Past :data:`MAX_INTERACTION_FACTORS` factors it is the main effects model, as the page models no interactions there.
    """
    if coded.shape[1] > MAX_INTERACTION_FACTORS:
        return MODELS[0]
    fits = estimable_models(coded, kinds)
    labels = sorted(set(blocks or []))
    if len(labels) > 1:
        indicators = np.column_stack([np.asarray(blocks) == b for b in labels[1:]]).astype(float)
        for model in (m for m in MODELS if fits.get(m)):
            matrix = np.column_stack([model_matrix(coded, kinds, model), indicators])
            fits[model] = int(np.linalg.matrix_rank(matrix)) == matrix.shape[1]
    return next((m for m in reversed(MODELS) if fits.get(m)), MODELS[0])


def _extension(spec: dict, design: pd.DataFrame, kind: str) -> dict[str, Any] | None:  # noqa: C901, PLR0911
    """Make one extension's runs for the study, coded, with what Learn says of them; None where it does not apply.

    ``new`` holds the runs added, coded, a row each; ``label`` names them for the button, ``why`` says what they
    are for, ``said`` what the design's notes say of them once added (its ``{first}`` and ``{last}`` are the run
    order of the first and last), ``model`` is the largest model the design then estimates, and ``blocked``
    whether they are a block of their own.
    """
    names = [f["name"] for f in spec["factors"]]
    kinds = ["categorical" if _categorical(f) else "numeric" for f in spec["factors"]]
    numeric = "categorical" not in kinds
    coded = _coded(design, spec["factors"]).to_numpy(dtype=float)
    measured = [j for j, k in enumerate(kinds) if k == "numeric"]  # a center run is at the center of these
    at_center = np.all(np.abs(coded[:, measured]) < ALIAS_TOLERANCE, axis=1) if measured else np.zeros(len(coded), bool)
    away = coded[~at_center]
    two_level = len(away) > 0 and bool(np.all(np.abs(np.abs(away) - 1) < ALIAS_TOLERANCE))
    blocks = len(set(design[BLOCK])) if BLOCK in design.columns and _blocked(spec) else 1
    frame = pd.DataFrame(coded, columns=names)
    info = EXTENSIONS[kind]
    out: dict[str, Any] = {"kind": kind, "blocked": info["blocked"]}
    if kind == "foldover":
        if not two_level or not _mains_aliased(away):
            return None
        result = augment_design(frame[~at_center], "foldover", factor_names=names)
        out |= {
            "label": f"Fold the design over: {len(away)} runs",
            "why": (
                "Main effects are aliased, in full or in part, with two-factor interactions, so a large effect may "
                "belong to an interaction. The design folded over, each run again with every setting switched to "
                "its other end, separates every main effect from the two-factor interactions."
            ),
            "said": "Runs {first} to {last} fold the design over: each is a run made before, every setting switched.",
        }
    elif kind == "axial":
        # Past MAX_INTERACTION_FACTORS factors the page fits no full quadratic model, so the runs are not looked at.
        fits = (
            estimable_models(coded, kinds)
            if numeric and two_level and 2 <= len(names) <= MAX_INTERACTION_FACTORS
            else {}
        )
        if not fits.get("interactions") or fits.get("quadratic"):
            return None
        result = augment_design(frame, "upgrade_to_rsm", alpha="face_centered", factor_names=names)
        n_axial, n_center = 2 * len(names), len(result["new_runs"]) - 2 * len(names)
        centres = f" and {n_center} center run{'s' if n_center != 1 else ''}" if n_center else ""
        out |= {
            "label": f"Add {n_axial} axial runs{centres}",
            "why": (
                "The runs are at two levels, so the model cannot hold the squares that curvature needs. Axial runs "
                f"on the faces of the ranges{', with runs at the center,' if n_center else ''} make the design a "
                "face-centred central composite design, which fits the full quadratic model."
            ),
            "said": (
                "Runs {first} to {last} are axial runs on the faces of the ranges"
                + (" and runs at the center" if n_center else "")
                + ": with the runs before, a face-centred central composite design."
            ),
        }
    elif kind == "center":
        if not numeric or not two_level or at_center.any() or blocks > 1:
            return None
        result = augment_design(frame, "add_center_points", n_additional_runs=EXTRA_CENTER_RUNS, factor_names=names)
        out |= {
            "label": f"Add {EXTRA_CENTER_RUNS} center runs",
            "why": (
                "No run is at the center of the ranges. Runs there show whether the response curves between the low "
                "and high settings, and their repeats estimate the noise."
            ),
            "said": "Runs {first} to {last} are at the center of the ranges, to test for curvature.",
        }
    elif kind == "optimal":
        if not numeric or spec.get("constraints"):  # the exchange takes every factor as numeric, and knows no limits
            return None
        # A study of more than MAX_INTERACTION_FACTORS factors aims at the main effects, whatever its spec says.
        aim = (spec.get("aim") or spec["model"]) if len(names) <= MAX_INTERACTION_FACTORS else MODELS[0]
        target = aim if aim in MODEL_WORDS else spec["model"] if spec["model"] in MODEL_WORDS else MODELS[0]
        n_terms, rank = _model_rank(coded, kinds, target)
        if rank == n_terms:
            return None
        if target == "quadratic" and _extension(spec, design, "axial") is not None:
            return None  # the axial runs fit it, and keep the runs on a few levels
        formula = _comparison_formula(target, names) if target == MAIN_QUADRATIC else target
        n_new = n_terms - rank + SPARE_RUNS
        with _listening():  # the exchange's own notes are about its budget, which the page sets
            result = augment_design(
                frame, "add_runs_optimal", target_model=formula, n_additional_runs=n_new, factor_names=names
            )
        n_new = len(result["new_runs"])
        out |= {
            "label": f"Add {n_new} runs for the {MODEL_WORDS[target]} model",
            "why": (
                f"The design cannot fit the {MODEL_WORDS[target]} model the study aims to learn. {n_new} runs chosen "
                "for that model (D-optimal), the runs made so far kept as they are, let it be fitted, with runs to "
                "spare to estimate the noise."
            ),
            "said": f"Runs {{first}} to {{last}} are chosen to fit the {MODEL_WORDS[target]} model (D-optimal).",
        }
    else:  # replicate
        model = spec["model"] if spec["model"] in MODEL_WORDS else MODELS[0]
        _, rank = _model_rank(coded, kinds, model)
        spare = len(coded) - rank
        if spare >= SPARE_FOR_NOISE:
            return None
        result = augment_design(frame, "replicate", factor_names=names)
        out |= {
            "label": f"Make the {len(coded)} runs again",
            "why": (
                f"The {MODEL_WORDS[model]} model leaves {spare} run{'s' if spare != 1 else ''} to estimate the noise. "
                "Every run made again estimates it from the repeats, and halves the variance of every coefficient."
            ),
            "said": "Runs {first} to {last} make every run of the design again.",
        }
    out["new"] = pd.DataFrame(result["new_runs"], columns=names).to_numpy(dtype=float)
    out["model"] = _largest_model(np.vstack([coded, out["new"]]), kinds)
    return out


def _mains_aliased(coded: np.ndarray) -> bool:
    """Say whether a main effect is correlated, in full or in part, with a two-factor interaction in these runs."""
    k = coded.shape[1]
    pairs = [coded[:, i] * coded[:, j] for i, j in itertools.combinations(range(k), 2)]
    if not pairs:
        return False
    mains, products = coded - coded.mean(axis=0), np.column_stack(pairs)
    products -= products.mean(axis=0)
    norms_m, norms_p = np.linalg.norm(mains, axis=0), np.linalg.norm(products, axis=0)
    keep_m, keep_p = norms_m > ALIAS_TOLERANCE, norms_p > ALIAS_TOLERANCE
    r = (mains[:, keep_m] / norms_m[keep_m]).T @ (products[:, keep_p] / norms_p[keep_p])
    return bool(r.size) and float(np.abs(r).max()) > ALIAS_TOLERANCE


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


def api_best_settings(payload: str) -> str:
    """JSON wrapper around :func:`best_settings`."""
    return _reply(best_settings, payload)


def api_balance(payload: str) -> str:
    """JSON wrapper around :func:`balance`."""
    return _reply(balance, payload)


def api_next_runs(payload: str) -> str:
    """JSON wrapper around :func:`next_runs`."""
    return _reply(next_runs, payload)


def api_add_runs(payload: str) -> str:
    """JSON wrapper around :func:`add_runs`."""
    return _reply(add_runs, payload)


def api_predictors(payload: str) -> str:
    """JSON wrapper around :func:`predictors`."""
    return _reply(predictors, payload)


def api_outside_limits(payload: str) -> str:
    """JSON wrapper around :func:`outside_limits`."""
    return _reply(outside_limits, payload)
