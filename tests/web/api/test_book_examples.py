"""The book's designed-experiment examples, run through the page's entry points.

Each case is a worked example or an exercise of the design of experiments chapter of *Process
Improvement using Data* (https://learnche.org/pid, source in kgdunn/pid-book). Its runs and
results are those of ``web/book-examples.json``, the examples the book links to by name, and
are made into the campaign that link opens (``book_example`` in ``scripts/build_web_app.py``):
the page generates the design from the factors the book gives, and each of the book's results
is placed against its run by the run's coded settings. The page then analyses them, and every
number the book prints must come back within the book's own rounding. A change to the app that
would contradict the book fails here; so does a book number the app does not reproduce, and so
does a link that would open anything other than the book's data.

The book gives some results as effects, the change from a factor's low level to its high level,
which is twice the coefficient the page reports; those cases say so. A categorical factor of the
book (a rack position, a profile required or not) is entered with its two labels, as a reader
would: the page codes the first -1 and the second +1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

import pytest

from process_improve.experiments import reference_catalogue
from tests.web._support import WEB, no_bytecode

if TYPE_CHECKING:
    from collections.abc import Callable

M, P = -1, 1  # the book's low and high levels


@cache
def catalogue() -> dict[str, dict]:
    """Return the book's examples, by the name a link carries, as ``web/book-examples.json`` lists them."""
    examples = json.loads((WEB / "book-examples.json").read_text(encoding="utf-8"))["examples"]
    return {entry["name"]: entry for entry in examples}


def factor(name: str, low: float, high: float, units: str = "") -> dict:
    """Return a factor as the page takes it."""
    return {"name": name, "low": low, "high": high, "units": units}


def standard_order(k: int) -> list[tuple[int, ...]]:
    """Return the 2^k runs in standard (Yates) order, the first factor changing fastest, as the book lists them."""
    return [tuple(reversed(run)) for run in product((M, P), repeat=k)]


def unlinked(design_type: str, factors: list[dict], runs: list[tuple], model: str, **options: object) -> dict:
    """Return runs the book gives that no link opens, as an entry of ``web/book-examples.json`` holds them."""
    spec = {"design_type": design_type, "factors": factors, "response": "y", "n_center_points": 0, **options}
    runs = [list(run) for run in runs]
    return {"name": "unlinked", "title": "", "page": "", "design": spec, "model": model, "runs": runs}


@dataclass
class Example:
    """A worked example or exercise, with what the book prints for it.

    ``data`` names the book's example in ``web/book-examples.json``, whose runs a link opens; or
    holds runs the book gives that no link opens, in the same shape. ``model`` and ``options``
    change the example's model or design, where the book analyses its runs another way.
    ``printed`` maps a coefficient, by the label the page shows, to the value the book prints,
    as text: the number of decimals printed sets how close the page must come. ``lenth`` holds
    the book's Lenth margins and ``active`` the effects the book finds beyond the margin of error.
    """

    data: str | dict
    printed: dict[str, str] = field(default_factory=dict)
    model: str | dict | None = None  # the model the book fits, where it differs from the example's
    options: dict = field(default_factory=dict)  # settings of the design that differ from the example's
    effects: bool = False  # the book prints effects, twice the page's coefficients
    lenth: dict[str, str] = field(default_factory=dict)
    lenth_scale: float = 1.0  # what the page's (effect-scale) margins are divided by for the book's
    active: tuple[str, ...] | None = None
    r_squared: str | None = None
    residual: dict[str, str] = field(default_factory=dict)  # df, pure-error df, mean square, coefficients' SE
    note: str | None = None  # the start of a note the analysis must make

    def entry(self) -> dict:
        """Return the runs as ``web/book-examples.json`` holds them, with this case's settings of the design."""
        entry = catalogue()[self.data] if isinstance(self.data, str) else self.data
        entry = {**entry, "design": {**entry["design"], **self.options}}
        if entry["design"].get("n_blocks", 1) == 1:
            entry.pop("blocks", None)  # the book's days are blocks only in a design made in blocks
        return entry


def tolerance(printed: str) -> float:
    """Half a unit in the last digit the book printed: the most its rounding accounts for."""
    exponent = Decimal(printed).normalize().as_tuple().exponent
    return 0.5 * 10.0 ** min(0, int(exponent)) + 1e-9


PARETO = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]


def steepest_ascent_yield(g: int, a: int, t: int, c: int) -> float:
    """Return the exercise's yield from its saturated model, which the book gives instead of the eight runs."""
    return 24 + 3 * g - 1.0 * a + 4.0 * t - 0.2 * g * a - 0.79 * g * t - 0.25 * a * t + 3.5 * c


EXAMPLES = {
    "two-level factorial, by least squares": Example(
        "conversion",
        {"Intercept": "61.5", "T": "-5", "S": "-3", "T·S": "-0.5"},
    ),
    "two-level factorial, effects by hand": Example("conversion", {"T": "-10", "S": "-6"}, effects=True),
    "two-level factorial with an interaction": Example(
        "conversion-interaction",
        {"Intercept": "81.5", "T": "2.5", "S": "3.5", "T·S": "1.5"},
    ),
    "two-level factorial with an interaction, effects by hand": Example(
        "conversion-interaction",
        {"T": "5", "S": "7", "T·S": "3"},
        effects=True,
    ),
    "three-factor pollutant experiment": Example(
        "pollutant",
        {"Intercept": "11.25", "C": "6.25", "T": "0.75", "S": "-7.25", "C·T": "0.25", "C·S": "-6.75", "T·S": "-0.25"},
    ),
    "three-factor pollutant experiment, effects by hand": Example(
        "pollutant",
        {"C": "12.5", "T": "1.5", "S": "-14.5", "C·T": "0.5", "C·S": "-13.5", "T·S": "-0.5"},
        effects=True,
    ),
    # All fifteen effects of the 2^4, saturated, through a fifth factor E = ABCD that carries the
    # four-factor interaction: the page's Lenth margins are the library's, which the book prints.
    # No link opens these runs: a reader would meet a factor E the book does not have.
    "Lenth's method on a 2^4 factorial": Example(
        unlinked(
            "fractional_factorial",
            [factor(n, -1, 1) for n in "ABCDE"],
            [(*s, s[0] * s[1] * s[2] * s[3], y) for s, y in zip(standard_order(4), PARETO, strict=True)],
            "interactions",
            resolution=5,
        ),
        lenth={"PSE": "2.625", "ME": "6.748", "SME": "13.699"},
        active=("A", "C", "D", "A·C", "A·D"),
    ),
    "saturated screening design": Example(
        "saturated-screening",
        {"Intercept": "70.7", "A": "-2.3", "B": "0.1", "C": "-2.8", "D": "-0.4", "E": "0.5", "F": "-0.4", "G": "-1.7"},
        lenth={"PSE": "0.60", "ME": "2.26"},
        lenth_scale=2.0,  # the book applies Lenth's method to the coefficients
        active=("A", "C"),
    ),
    "response surface: first factorial": Example(
        "rsm-first-factorial",
        {"Intercept": "385.5", "T": "55", "S": "134", "T·S": "-3.50"},
    ),
    "response surface: second factorial": Example(
        "rsm-second-factorial",
        {"Intercept": "670.25", "T": "13.25", "S": "-39.25", "T·S": "-2.25"},
    ),
    "response surface: central composite design": Example(
        "rsm-central-composite",
        {"Intercept": "688", "T": "13", "S": "-39", "T·S": "-2.25", "T²": "-4.2", "S²": "-12.2"},
    ),
    "multi-response: profit": Example(
        "multi-response-profit",
        {"Intercept": "738.0", "T": "4.496", "S": "-1.759", "T·S": "-2.25", "T²": "-2.625", "S²": "-10.875"},
        r_squared="0.9922",
    ),
    "multi-response: purity": Example(
        "multi-response-purity",
        {"Intercept": "89.8", "T": "-4.081", "S": "2.475", "T·S": "-0.025", "T²": "-0.05", "S²": "-1.125"},
        r_squared="0.9989",
    ),
    # The book averages each pair of replicates and analyses the corners; the page holds both replicates.
    "exercise: web sales": Example(
        "web-sales",
        {"Intercept": "3184", "Shipping": "-126", "Profile": "-371", "Shipping·Profile": "-181"},
        residual={"df": "4"},  # the replicates' four, all pure error: the corners' model leaves none for lack of fit
    ),
    "exercise: web sales, effects": Example(
        "web-sales",
        {"Shipping": "-252.50", "Profile": "-742.50", "Shipping·Profile": "-362.50"},
        effects=True,
    ),
    "exercise: bioreactor 2^4": Example(
        "bioreactor",
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
        "bioreactor-half-fraction",
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
    ),
    # The exercise's first parts take the six runs as one group, before the days are brought in.
    "exercise: side product": Example(
        "side-product",
        {"Intercept": "232.5", "A": "112", "B": "67.5", "A·B": "22.5"},
        options={"n_blocks": 1},
    ),
    "exercise: chemical reaction": Example(
        "chemical-reaction",
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
        "chemical-reaction",
        {"A": "11.25", "B": "4.75", "C": "-0.25", "A·B": "9.25", "A·C": "0.25", "B·C": "0.75"},
        effects=True,
    ),
    "exercise: injection moulding": Example(
        "injection-molding",
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
    ),
    "exercise: breaded chicken": Example(
        "breaded-chicken",
        {"Intercept": "5.75", "D": "2.5", "R": "-0.5", "P": "0.25", "D·R": "-0.75", "D·P": "0.0", "R·P": "0.0"},
    ),
    "exercise: breaded chicken, half fraction": Example(
        "breaded-chicken-half-fraction",
        {"Intercept": "5.5", "D": "2.5", "R": "-0.5", "P": "-0.5"},
    ),
    # The book gives the saturated model, not the eight runs: no link opens yields it did not print.
    "exercise: steepest ascent": Example(
        unlinked(
            "fractional_factorial",
            [factor("G", "Gm", "Gp"), factor("A", 10, 20, "rpm"), factor("T", 30, 36, "degC"), factor("C", 1100, 1400)],
            [(*s, steepest_ascent_yield(*s)) for s in standard_order(4) if s[3] == s[0] * s[1] * s[2]],
            "interactions",
            resolution=4,
        ),
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
    ),
    # The 2^4 of the significance section refitted with only its significant effects, the terms a reader ticks.
    "significance: the 2^4 refitted with its significant effects": Example(
        "significance",
        residual={"df": "11", "mean square": "3.55", "standard error": "0.471"},
    ),
    "significance: an effect left out of the 2^4": Example("significance", {"C": "-1.125"}, model="interactions"),
    # The full factorial with every term, the three-factor interaction ticked: the half fraction's intercept
    # is the full model's 5.75 plus its D·R·P, -0.25.
    "exercise: breaded chicken, every term": Example(
        "breaded-chicken",
        {"Intercept": "5.75", "D": "2.5", "R": "-0.5", "P": "0.25", "D·R": "-0.75", "D·R·P": "-0.25"},
        model={"terms": [["D"], ["R"], ["P"], ["D", "R"], ["D", "P"], ["R", "P"], ["D", "R", "P"]]},
    ),
    "exercise: microgel, axial runs at 1.4": Example(
        "microgel",
        residual={"df": "4", "pure error df": "1"},
    ),
    # The side product's two days are the two blocks: the page confounds A·B with them, as the book did.
    "exercise: side product, run over two days": Example(
        "side-product",
        {"Intercept": "232.5", "A": "112", "B": "67.5"},
        note="A·B is confounded with the blocks",
    ),
}


def test_every_example_a_link_opens_is_checked_against_the_book():
    """A name in the catalogue is a link the book may print: each must open runs a case here checks."""
    checked = {example.data for example in EXAMPLES.values() if isinstance(example.data, str)}
    assert checked == set(catalogue())


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
def analysis_of(api, build_script) -> Callable[[str], dict]:
    """Analyse an example, once per module, as the page does once its link has opened it.

    ``book_example`` makes the design through the page's entry point and puts the book's results
    against its runs, as the build does for the link; the page then analyses what the link opens.
    """

    @cache
    def analysis(name: str) -> dict:
        example = EXAMPLES[name]
        with no_bytecode():
            opened = build_script.book_example(example.entry())["campaign"]
        held = {key: opened["design"][key] for key in ("spec", "columns", "rows")}
        model = opened["model"] if example.model is None else example.model
        return api.ok("api_analyze_runs", {**held, "responses": opened["responses"], "model": model})

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


@pytest.mark.parametrize("name", [name for name, example in EXAMPLES.items() if example.residual])
def test_the_residual_degrees_of_freedom_and_error_the_book_prints(analysis_of, name):
    """What replicates and center runs leave for the error, and the error itself where the book prints it."""
    example, analysis = EXAMPLES[name], analysis_of(name)
    found = {
        "df": analysis["summary"]["df_residual"],
        "pure error df": (analysis["lack_of_fit"] or {}).get("df_pure_error"),
        "mean square": analysis["summary"]["mse_residual"],
        "standard error": max(c["std_error"] for c in analysis["coefficients"] if c["term"] != "Intercept"),
    }
    for key, printed in example.residual.items():
        assert found[key] is not None, f"the page gives no {key}"
        said = f"{key}: the book {printed}, the page {found[key]}"
        assert abs(found[key] - float(printed)) <= tolerance(printed), said
    if "standard error" in example.residual:
        errors = {round(c["std_error"], 9) for c in analysis["coefficients"] if c["term"] != "Intercept"}
        assert len(errors) == 1, "an orthogonal design gives every coefficient the same standard error"


@pytest.mark.parametrize("name", [name for name, example in EXAMPLES.items() if example.note])
def test_the_note_the_analysis_makes(analysis_of, name):
    assert any(note.startswith(EXAMPLES[name].note) for note in analysis_of(name)["notes"])


# --------------------------------------------------------------------------- the book's own record of it
def _reference_runs(name: str) -> list[tuple]:
    """Return the runs of the book's experiment in the reference catalogue, as printed, with the first response.

    The settings are those the book prints (1.41, say, where the reference catalogue recomputes with the square
    root of 2 it stands for): the page matches each result to a run of the design it makes by them.
    """
    experiment = next(e for e in reference_catalogue.reference_experiments("pid-book") if e["name"] == name)
    printed = {**experiment, "runs": {**experiment["runs"], "exact": None}}
    settings, responses = reference_catalogue._coded_runs(printed, None)
    return [(*row, y) for row, y in zip(settings.itertuples(index=False), responses.iloc[:, 0], strict=True)]


@pytest.mark.parametrize("name", list(catalogue()))
def test_every_example_a_link_opens_is_in_the_reference_catalogue(name):
    """The runs a link opens are runs the book prints, as the reference catalogue records them.

    The reference catalogue's experiment of the same name holds every run the book prints, and every number it
    prints for them, each with its file and line; ``tests/test_reference_catalogue.py`` reproduces those
    numbers. A link may open fewer runs than the book prints (the four corners without the baseline),
    but none the book does not.
    """
    book = [tuple(round(float(v), 9) for v in run) for run in _reference_runs(name)]
    for run in catalogue()[name]["runs"]:
        coded = tuple(round(float(v), 9) for v in run)
        assert coded in book, f"{name}: the link opens {run}, which is not a run of the book's"
        book.remove(coded)
