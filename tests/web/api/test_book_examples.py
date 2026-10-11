"""The book's designed-experiment examples, run through the page's entry points.

Each case is a worked example or an exercise of the design of experiments chapter of *Process
Improvement using Data* (https://learnche.org/pid, source in kgdunn/pid-book), named by its page.
The page generates the design from the factors the book gives; the book's results are typed
against the run sheet, each matched to its run by the run's coded settings, as a reader would;
the page analyses them; and every number the book prints must come back within the book's own
rounding. A change to the app that would contradict the book fails here, and so does a book
number the app does not reproduce.

The book gives some results as effects, the change from a factor's low level to its high level,
which is twice the coefficient the page reports; those cases say so. A categorical factor of the
book (a rack position, a profile required or not) is entered as a numeric stand-in from 0 to 1,
since the analysis is in coded units and the coding is the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.web._support import coded, held_of, run_sheet

if TYPE_CHECKING:
    from collections.abc import Callable

M, P = -1, 1  # the book's low and high levels
ROOT2 = 2**0.5  # the rotatable axial distance of a two-factor central composite design


def factor(name: str, low: float, high: float, units: str = "") -> dict:
    """Return a factor as the page takes it."""
    return {"name": name, "low": low, "high": high, "units": units}


def standard_order(k: int) -> list[tuple[int, ...]]:
    """Return the 2^k runs in standard (Yates) order, the first factor changing fastest, as the book lists them."""
    return [tuple(reversed(levels)) for levels in product((M, P), repeat=k)]


@dataclass(frozen=True)
class Example:
    """One example of the book: the design the page makes for it, the book's results, and what the book prints.

    ``runs`` are the book's runs in coded units, each with its result last. ``printed`` maps a
    coefficient, by the label the page shows, to the value the book prints, as text: the number
    of decimals printed sets how close the page must come. ``lenth`` holds the book's Lenth
    margins and ``active`` the effects the book finds beyond the margin of error.
    """

    page: str
    design_type: str
    factors: list[dict]
    runs: list[tuple]
    model: str
    printed: dict[str, str] = field(default_factory=dict)
    effects: bool = False  # the book prints effects, twice the page's coefficients
    options: dict = field(default_factory=dict)
    lenth: dict[str, str] = field(default_factory=dict)
    lenth_scale: float = 1.0  # what the page's (effect-scale) margins are divided by for the book's
    active: tuple[str, ...] | None = None
    r_squared: str | None = None


def tolerance(printed: str) -> float:
    """Half a unit in the last digit the book printed: the most its rounding accounts for."""
    exponent = Decimal(printed).normalize().as_tuple().exponent
    return 0.5 * 10.0 ** min(0, int(exponent)) + 1e-9


CONVERSION = [factor("T", 338, 354, "K"), factor("S", 1.25, 1.75, "g/L")]
INTERACTING = [factor("T", 390, 400, "K"), factor("S", 0.5, 1.25, "g/L")]
POLLUTANT_FACTORS = [factor("C", 0, 1, "P=0, Q=1"), factor("T", 72, 100, "degF"), factor("S", 200, 400, "rpm")]
POLLUTANT = [(*s, y) for s, y in zip(standard_order(3), [5, 30, 6, 33, 4, 3, 5, 4], strict=True)]
PARETO = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]
WEB_SALES = [factor("Shipping", 30, 50, "$"), factor("Profile", 0, 1, "no=0, yes=1")]
BIOREACTOR = [
    factor("A", 1, 2, "slow=1, medium=2"),
    factor("B", 300, 700, "g"),
    factor("C", 40, 60, "g/L"),
    factor("D", 4, 6, "mg/L"),
]
BIOREACTOR_Y = dict(
    zip(standard_order(4), [60, 59, 63, 61, 69, 61, 94, 93, 56, 63, 70, 65, 44, 45, 78, 77], strict=True)
)
REACTION = [factor("A", 140, 160, "degC"), factor("B", 7.0, 8.0, "pH"), factor("C", 45, 55, "rpm")]
REACTION_Y = [72, 73, 66, 87, 70, 73, 67, 87]
CHICKEN = [factor("D", 15, 22, "min"), factor("R", 0, 1, "middle=0, low=1"), factor("P", 0, 1, "short=0, full=1")]
CHICKEN_RUNS = [(*s, y) for s, y in zip(standard_order(3), [3, 9, 3, 7, 3, 10, 4, 7], strict=True)]
SCREENING = [
    (M, M, M, P, P, P, M, 77.1),
    (P, M, M, M, M, P, P, 68.9),
    (M, P, M, M, P, M, P, 75.5),
    (P, P, M, P, M, M, M, 72.5),
    (M, M, P, P, M, M, P, 67.9),
    (P, M, P, M, P, M, M, 68.5),
    (M, P, P, M, M, P, M, 71.5),
    (P, P, P, P, P, P, P, 63.7),
]
MOULDING = [
    (M, M, M, P, P, P, M, 14.0),
    (P, M, M, M, M, P, P, 16.8),
    (M, P, M, M, P, M, P, 15.0),
    (P, P, M, P, M, M, M, 15.4),
    (M, M, P, P, M, M, P, 27.6),
    (P, M, P, M, P, M, M, 24.0),
    (M, P, P, M, M, P, M, 27.4),
    (P, P, P, P, P, P, P, 22.6),
]
AXIAL = [(M, M), (P, M), (M, P), (P, P), (0, 0), (0, -ROOT2), (ROOT2, 0), (0, ROOT2), (-ROOT2, 0)]
MULTI_RESPONSE = [factor("T", 339, 347, "K"), factor("S", 1.40, 1.80, "g/L")]


def steepest_ascent_yield(g: int, a: int, t: int, c: int) -> float:
    """Return the exercise's yield from its saturated model, which the book gives instead of the eight runs."""
    return 24 + 3 * g - 1.0 * a + 4.0 * t - 0.2 * g * a - 0.79 * g * t - 0.25 * a * t + 3.5 * c


EXAMPLES = {
    "two-level factorial, by least squares": Example(
        "full-factorial-designs/analysis-by-least-squares-modelling.rst",
        "full_factorial",
        CONVERSION,
        [(*s, y) for s, y in zip(standard_order(2), [69, 60, 64, 53], strict=True)],
        "interactions",
        {"Intercept": "61.5", "T": "-5", "S": "-3", "T·S": "-0.5"},
    ),
    "two-level factorial, effects by hand": Example(
        "full-factorial-designs/using-two-levels-for-two-or-more-factors.rst",
        "full_factorial",
        CONVERSION,
        [(*s, y) for s, y in zip(standard_order(2), [69, 60, 64, 53], strict=True)],
        "interactions",
        {"T": "-10", "S": "-6"},
        effects=True,
    ),
    "two-level factorial with an interaction": Example(
        "full-factorial-designs/analysis-by-least-squares-modelling.rst",
        "full_factorial",
        INTERACTING,
        [(*s, y) for s, y in zip(standard_order(2), [77, 79, 81, 89], strict=True)],
        "interactions",
        {"Intercept": "81.5", "T": "2.5", "S": "3.5", "T·S": "1.5"},
    ),
    "two-level factorial with an interaction, effects by hand": Example(
        "full-factorial-designs/using-two-levels-for-two-or-more-factors.rst",
        "full_factorial",
        INTERACTING,
        [(*s, y) for s, y in zip(standard_order(2), [77, 79, 81, 89], strict=True)],
        "interactions",
        {"T": "5", "S": "7", "T·S": "3"},
        effects=True,
    ),
    "three-factor pollutant experiment": Example(
        "full-factorial-designs/example-design-and-analysis-of-a-three-factor-experiment.rst",
        "full_factorial",
        POLLUTANT_FACTORS,
        POLLUTANT,
        "interactions",
        {"Intercept": "11.25", "C": "6.25", "T": "0.75", "S": "-7.25", "C·T": "0.25", "C·S": "-6.75", "T·S": "-0.25"},
    ),
    "three-factor pollutant experiment, effects by hand": Example(
        "full-factorial-designs/example-design-and-analysis-of-a-three-factor-experiment.rst",
        "full_factorial",
        POLLUTANT_FACTORS,
        POLLUTANT,
        "interactions",
        {"C": "12.5", "T": "1.5", "S": "-14.5", "C·T": "0.5", "C·S": "-13.5", "T·S": "-0.5"},
        effects=True,
    ),
    # All fifteen effects of the 2^4, saturated, through a fifth factor E = ABCD that carries the
    # four-factor interaction: the page's Lenth margins are the library's, which the book prints.
    "Lenth's method on a 2^4 factorial": Example(
        "full-factorial-designs/assessing-significance-of-main-effects-and-interactions.rst",
        "fractional_factorial",
        [factor(n, -1, 1) for n in "ABCDE"],
        [(*s, s[0] * s[1] * s[2] * s[3], y) for s, y in zip(standard_order(4), PARETO, strict=True)],
        "interactions",
        lenth={"PSE": "2.625", "ME": "6.748", "SME": "13.699"},
        active=("A", "C", "D", "A·C", "A·D"),
        options={"resolution": 5},
    ),
    "saturated screening design": Example(
        "fractional-factorial-designs/saturated-designs-for-screening.rst",
        "fractional_factorial",
        [factor(n, -1, 1) for n in "ABCDEFG"],
        SCREENING,
        "main_effects",
        {"Intercept": "70.7", "A": "-2.3", "B": "0.1", "C": "-2.8", "D": "-0.4", "E": "0.5", "F": "-0.4", "G": "-1.7"},
        lenth={"PSE": "0.60", "ME": "2.26"},
        lenth_scale=2.0,  # the book applies Lenth's method to the coefficients
        active=("A", "C"),
        options={"resolution": 3},
    ),
    "response surface: first factorial": Example(
        "response-surface-methods.rst",
        "full_factorial",
        [factor("T", 320, 330, "K"), factor("S", 0.5, 1.0, "g/L")],
        [(*s, y) for s, y in zip(standard_order(2), [193, 310, 468, 571], strict=True)],
        "interactions",
        {"Intercept": "385.5", "T": "55", "S": "134", "T·S": "-3.50"},
    ),
    "response surface: second factorial": Example(
        "response-surface-methods.rst",
        "full_factorial",
        [factor("T", 331, 339, "K"), factor("S", 1.77, 2.17, "g/L")],
        [(*s, y) for s, y in zip(standard_order(2), [694, 725, 620, 642], strict=True)],
        "interactions",
        {"Intercept": "670.25", "T": "13.25", "S": "-39.25", "T·S": "-2.25"},
    ),
    "response surface: central composite design": Example(
        "response-surface-methods.rst",
        "ccd",
        [factor("T", 331, 339, "K"), factor("S", 1.77, 2.17, "g/L")],
        [(*s, y) for s, y in zip(AXIAL, [694, 725, 620, 642, 688, 720, 699, 610, 663], strict=True)],
        "quadratic",
        {"Intercept": "688", "T": "13", "S": "-39", "T·S": "-2.25", "T²": "-4.2", "S²": "-12.2"},
        options={"n_center_points": 1, "alpha": "rotatable"},
    ),
    "multi-response: profit": Example(
        "multi-response-optimization.rst",
        "ccd",
        MULTI_RESPONSE,
        [(*s, y) for s, y in zip(AXIAL, [720, 734, 721, 726, 738, 718, 738, 713, 726], strict=True)],
        "quadratic",
        {"Intercept": "738.0", "T": "4.496", "S": "-1.759", "T·S": "-2.25", "T²": "-2.625", "S²": "-10.875"},
        options={"n_center_points": 1, "alpha": "rotatable"},
        r_squared="0.9922",
    ),
    "multi-response: purity": Example(
        "multi-response-optimization.rst",
        "ccd",
        MULTI_RESPONSE,
        [(*s, y) for s, y in zip(AXIAL, [90.3, 82.3, 95.3, 87.2, 89.8, 83.9, 83.7, 90.9, 95.4], strict=True)],
        "quadratic",
        {"Intercept": "89.8", "T": "-4.081", "S": "2.475", "T·S": "-0.025", "T²": "-0.05", "S²": "-1.125"},
        options={"n_center_points": 1, "alpha": "rotatable"},
        r_squared="0.9989",
    ),
    # The book averages each pair of replicates, which the page cannot hold, and analyses the corners.
    "exercise: web sales": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        WEB_SALES,
        [(*s, y) for s, y in zip(standard_order(2), [3500, 3610, 3120, 2505], strict=True)],
        "interactions",
        {"Intercept": "3184", "Shipping": "-126", "Profile": "-371", "Shipping·Profile": "-181"},
    ),
    "exercise: web sales, effects": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        WEB_SALES,
        [(*s, y) for s, y in zip(standard_order(2), [3500, 3610, 3120, 2505], strict=True)],
        "interactions",
        {"Shipping": "-252.50", "Profile": "-742.50", "Shipping·Profile": "-362.50"},
        effects=True,
    ),
    "exercise: bioreactor 2^4": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        BIOREACTOR,
        [(*s, y) for s, y in BIOREACTOR_Y.items()],
        "interactions",
        {
            "Intercept": "66.125",
            "A": "-0.625",
            "B": "9",
            "C": "4",
            "D": "-3.875",
            "A·B": "-0.5",
            "A·C": "-0.5",
            "A·D": "0.875",
            "B·C": "6.375",
            "B·D": "1.25",
            "C·D": "-5.25",
        },
    ),
    "exercise: bioreactor half fraction": Example(
        "design-analysis-experiments-exercises.rst",
        "fractional_factorial",
        BIOREACTOR,
        [(*s, y) for s, y in BIOREACTOR_Y.items() if s[3] == s[0] * s[1] * s[2]],
        "interactions",
        {
            "Intercept": "66.25",
            "A": "-0.75",
            "B": "9.25",
            "C": "2.75",
            "D": "-2.75",
            "A·B": "-5.75",
            "A·C": "0.75",
            "A·D": "7.25",
        },
        options={"resolution": 4},
    ),
    "exercise: side product": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        [factor("A", 420, 440, "K"), factor("B", 10, 12, "kg")],
        [(M, M, 89), (P, M, 268), (M, P, 179), (P, P, 448), (0, 0, 196), (0, 0, 215)],
        "interactions",
        {"Intercept": "232.5", "A": "112", "B": "67.5", "A·B": "22.5"},
        options={"n_center_points": 2},
    ),
    "exercise: chemical reaction": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        REACTION,
        [(*s, y) for s, y in zip(standard_order(3), REACTION_Y, strict=True)],
        "interactions",
        {
            "Intercept": "74.375",
            "A": "5.625",
            "B": "2.375",
            "C": "-0.125",
            "A·B": "4.625",
            "A·C": "0.125",
            "B·C": "0.375",
        },
    ),
    "exercise: chemical reaction, effects by hand": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        REACTION,
        [(*s, y) for s, y in zip(standard_order(3), REACTION_Y, strict=True)],
        "interactions",
        {"A": "11.25", "B": "4.75", "C": "-0.25", "A·B": "9.25", "A·C": "0.25", "B·C": "0.75"},
        effects=True,
    ),
    "exercise: injection moulding": Example(
        "design-analysis-experiments-exercises.rst",
        "fractional_factorial",
        [factor(n, -1, 1) for n in "ABCDEFG"],
        MOULDING,
        "main_effects",
        {
            "Intercept": "20.35",
            "A": "-0.65",
            "B": "-0.25",
            "C": "5.05",
            "D": "-0.45",
            "E": "-1.45",
            "F": "-0.15",
            "G": "0.15",
        },
        lenth={"ME": "1.98"},
        lenth_scale=2.0,  # the solution quotes the margin on the coefficient scale
        active=("C",),
        options={"resolution": 3},
    ),
    "exercise: breaded chicken": Example(
        "design-analysis-experiments-exercises.rst",
        "full_factorial",
        CHICKEN,
        CHICKEN_RUNS,
        "interactions",
        {"Intercept": "5.75", "D": "2.5", "R": "-0.5", "P": "0.25", "D·R": "-0.75", "D·P": "0.0", "R·P": "0.0"},
    ),
    "exercise: breaded chicken, half fraction": Example(
        "design-analysis-experiments-exercises.rst",
        "fractional_factorial",
        CHICKEN,
        [run for run in CHICKEN_RUNS if run[2] == run[0] * run[1]],
        "main_effects",
        {"Intercept": "5.5", "D": "2.5", "R": "-0.5", "P": "-0.5"},
        options={"resolution": 3},
    ),
    "exercise: steepest ascent": Example(
        "design-analysis-experiments-exercises.rst",
        "fractional_factorial",
        [
            factor("G", 0, 1, "Gm=0, Gp=1"),
            factor("A", 10, 20, "rpm"),
            factor("T", 30, 36, "degC"),
            factor("C", 1100, 1400),
        ],
        [(*s, steepest_ascent_yield(*s)) for s in standard_order(4) if s[3] == s[0] * s[1] * s[2]],
        "interactions",
        {
            "Intercept": "24",
            "G": "3",
            "A": "-1.0",
            "T": "4.0",
            "C": "3.5",
            "G·A": "-0.2",
            "G·T": "-0.79",
            "A·T": "-0.25",
        },
        options={"resolution": 4},
    ),
}


def typed_in(example: Example, design: dict) -> list[float]:
    """Type the book's results against the page's run sheet, each matched to its run by the coded settings."""
    sheet = coded(run_sheet(design), example.factors).to_numpy(dtype=float)
    unused = [(np.asarray(run[:-1], dtype=float), run[-1]) for run in example.runs]
    results = []
    for row in sheet:
        found = next((i for i, (settings, _) in enumerate(unused) if np.allclose(settings, row, atol=1e-9)), None)
        if found is None:
            pytest.fail(f"The page's run {row.round(4).tolist()} is not one of the book's runs.")
        results.append(unused.pop(found)[1])
    assert not unused, f"The book's runs {[s.tolist() for s, _ in unused]} are not in the page's design."
    return results


def coefficient(analysis: dict, label: str) -> float:
    """Read a coefficient by the label the page shows, or by any term of its alias chain."""
    found = [
        c["coefficient"]
        for c in analysis["coefficients"]
        if label in {c["label"], *(term.replace(":", "·") for term in c["aliases"])}
    ]
    assert found, f"The page shows no coefficient {label!r}."
    return found[0]


@pytest.fixture(scope="module")
def analysis_of(api) -> Callable[[str], dict]:
    """Analyse an example, once per module, as the page does: make its design, type in its results, analyse."""

    @cache
    def analysis(name: str) -> dict:
        example = EXAMPLES[name]
        options = {"n_center_points": 0, **example.options}
        spec = {"design_type": example.design_type, "factors": example.factors, "response": "y", **options}
        design = api.ok("api_make_design", spec)
        request = {**held_of(design), "responses": typed_in(example, design), "model": example.model}
        return api.ok("api_analyze_runs", request)

    return analysis


@pytest.mark.parametrize("name", [name for name, example in EXAMPLES.items() if example.printed])
def test_every_coefficient_the_book_prints(analysis_of, name):
    example, analysis = EXAMPLES[name], analysis_of(name)
    for label, printed in example.printed.items():
        value = coefficient(analysis, label) * (2 if example.effects and label != "Intercept" else 1)
        assert abs(value - float(printed)) <= tolerance(printed), (
            f"{label}: the book prints {printed}, the page {value}"
        )


@pytest.mark.parametrize("name", [name for name, example in EXAMPLES.items() if example.active is not None])
def test_lenth_margins_and_the_effects_beyond_them(analysis_of, name):
    """The page's margins are for the effects, twice the coefficients; each example says which scale the book uses."""
    example, lenth = EXAMPLES[name], analysis_of(name)["lenth"]
    for margin, printed in example.lenth.items():
        value = lenth[margin] / example.lenth_scale
        assert abs(value - float(printed)) <= tolerance(printed), (
            f"{margin}: the book prints {printed}, the page {value}"
        )
    assert {e["label"] for e in lenth["effects"] if e["active_ME"]} == set(example.active)


@pytest.mark.parametrize("name", [name for name, example in EXAMPLES.items() if example.r_squared])
def test_r_squared_the_book_prints(analysis_of, name):
    printed = EXAMPLES[name].r_squared
    assert abs(analysis_of(name)["summary"]["r_squared"] - float(printed)) <= tolerance(printed)


def test_the_fraction_the_page_makes_is_the_book_s(api):
    """Each fraction above matches the book's runs only because the page picks the book's generators."""
    fractions = {3: ["C = A·B"], 4: ["D = A·B·C"], 7: ["D = A·B", "E = A·C", "F = B·C", "G = A·B·C"]}
    for k, generators in fractions.items():
        spec = {"design_type": "fractional_factorial", "factors": [factor(n, -1, 1) for n in "ABCDEFG"[:k]]}
        made = api.ok("api_make_design", {**spec, "n_center_points": 0, "resolution": 4 if k == 4 else 3})
        assert made["facts"]["generators"] == generators
