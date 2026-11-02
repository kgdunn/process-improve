"""The multivariate page in Chromium, its Python answered natively by ``web/mva/bootstrap.py``.

The building blocks together, as a reader uses them: open the LDPE data, lasso the upset batches
on the score plot, exclude them and see them projected, hollow, through the refitted model; ask
why they differ; refine the table with a recipe step; and save and reopen the project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._support import WEB, load_module

if TYPE_CHECKING:
    from collections.abc import Iterator

UPSET = ["52", "53", "54"]


@pytest.fixture
def page(chromium: Any, site: str, plotly_js: tuple[str, bytes]) -> Iterator[Any]:  # noqa: ANN401 - a Playwright Page
    """Open the page with its Python answered here, Plotly.js served locally, and nothing else reachable."""
    glue = load_module("mva_bootstrap_e2e", WEB / "mva" / "bootstrap.py")
    context = chromium.new_context(accept_downloads=True)
    context.route(
        re.compile(r".*"), lambda route: route.continue_() if route.request.url.startswith(site) else route.abort()
    )
    context.route(
        re.compile(r"/plotly\.min\.js$"), lambda route: route.fulfill(body=plotly_js[1], content_type="text/javascript")
    )
    opened = context.new_page()
    errors: list[str] = []
    opened.on("pageerror", lambda error: errors.append(str(error)))
    opened.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)

    def answer(fn: str, payload: str) -> str:
        entry = getattr(glue, fn, None) if re.fullmatch(r"api_[a-z_]+", fn) else None
        if not callable(entry):
            return json.dumps({"ok": False, "error": f"{fn} is not an entry point"})
        return entry(payload)

    opened.expose_function("__mvaNative", answer)
    opened.goto(f"{site}mva/index.html")
    yield opened
    context.close()
    # A request the page made for a missing file is the manifest's 404, which the page expects.
    assert [e for e in errors if "404" not in e] == [], errors


def ready(page: Any) -> None:  # noqa: ANN401
    page.wait_for_function("document.getElementById('status').dataset.state === 'ok'")


def open_ldpe(page: Any) -> None:  # noqa: ANN401
    page.click("#open-example")
    page.wait_for_function("document.getElementById('scores').data?.length > 0")
    ready(page)


def traces(page: Any, div: str) -> list[dict]:  # noqa: ANN401
    return page.evaluate(
        f"""Array.from(document.getElementById({div!r}).data, (t) => ({{
            name: t.name, ids: t.customdata ?? [], symbols: t.marker?.symbol ?? null, opacity: t.marker?.opacity ?? null
        }}))"""
    )


@pytest.mark.dataset
def test_opening_the_example_draws_the_table_and_every_linked_view(page) -> None:
    open_ldpe(page)
    assert "54 rows fitted, 0 excluded" in page.text_content("#model-summary")
    assert traces(page, "scores")[0]["ids"][:2] == ["1", "2"]
    assert page.evaluate("document.getElementById('scores').layout.xaxis.title.text").startswith("t1 [")
    painted = page.evaluate(
        """(() => { const c = document.querySelector('#table canvas');
                    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                    return d.some((v, i) => i % 4 === 3 && v > 0); })()"""
    )
    assert painted, "the table's canvas is drawn"


@pytest.mark.dataset
def test_a_lasso_on_the_score_plot_selects_the_rows_everywhere(page) -> None:
    open_ldpe(page)
    page.evaluate(
        "ids => document.getElementById('scores')"
        ".emit('plotly_selected', {points: ids.map((id) => ({customdata: id}))})",
        UPSET,
    )
    page.wait_for_function("window.__mva.store.selected('rows').size === 3")
    assert page.text_content("#selection-count") == "3 rows and 0 columns selected"
    page.wait_for_function("document.getElementById('spe').data[1].customdata.length === 3")
    faded = traces(page, "scores")[0]["opacity"]
    assert faded[0] < 1, "the rows not selected fade on the score plot"


@pytest.mark.dataset
def test_excluded_rows_are_refitted_out_and_drawn_hollow_where_the_model_projects_them(page) -> None:
    open_ldpe(page)
    page.evaluate("ids => window.__mva.store.select('rows', ids)", UPSET)
    page.click("#exclude")
    page.wait_for_function(
        "document.getElementById('model-summary').textContent.includes('51 rows fitted, 3 excluded')"
    )
    page.wait_for_function("document.getElementById('scores').data.some((t) => t.name === 'Excluded, projected')")
    projected = next(t for t in traces(page, "scores") if t["name"] == "Excluded, projected")
    assert projected["ids"] == UPSET
    assert all(symbol.endswith("-open") for symbol in projected["symbols"])
    # The charts redraw on the next animation frame: wait for the SPE chart to draw batch 54 hollow.
    page.wait_for_function("document.getElementById('spe').data[0].marker.symbol.at(-1) === 'circle-open'")
    spe = page.evaluate("document.getElementById('spe').data[0].y.at(-1)")
    assert spe == page.evaluate("window.__mva.state.fit.SPE.at(-1)")
    limit = page.evaluate("window.__mva.state.fit.limits['0.95'].SPE")
    assert spe > limit, "batch 54, projected, is beyond the SPE limit of the model that never saw it"
    page.click("#include")
    page.wait_for_function("document.getElementById('model-summary').textContent.includes('54 rows fitted')")


@pytest.mark.dataset
def test_cross_validation_runs_when_asked_and_a_bar_sets_the_number_of_components(page) -> None:
    open_ldpe(page)
    assert page.is_hidden("#cv-panel"), "cross-validation fits the model many times, so it waits to be asked"
    page.click("#cross-validate")
    page.wait_for_function("document.getElementById('cv').data?.length === 2")
    ready(page)
    assert page.text_content("#cv-sentence").startswith("Recommended: ")
    assert "PCA.select_n_components(X.loc[included], random_state=0)" in page.text_content("#code")

    # The R² bar of five components: a click on it refits with five, and the chart outlines it.
    # Plotly takes the click on its drag layer, above the bars, as a reader's click; hence force.
    page.locator("#cv .trace.bars").first.locator(".point").nth(4).click(force=True)
    page.wait_for_function("window.__mva.state.fit.scores.length === 5")
    assert page.input_value("#components") == "5"
    page.wait_for_function("document.getElementById('cv').data[0].marker.line.width[4] === 2")
    assert not page.text_content("#cv-sentence").startswith("The model has changed")

    # Excluding rows changes the model: the cross-validation is marked stale, not run again.
    page.evaluate("ids => window.__mva.store.select('rows', ids)", UPSET)
    page.click("#exclude")
    page.wait_for_function("document.getElementById('cv-sentence').textContent.startsWith('The model has changed')")
    assert page.evaluate("document.getElementById('cv').data[1].marker.opacity") < 1
    assert "select_n_components" not in page.text_content("#code")


@pytest.mark.dataset
def test_why_are_these_rows_different_ranks_the_variables_with_their_evidence(page) -> None:
    open_ldpe(page)
    page.evaluate("cols => window.__mva.store.select('cols', cols)", ["Conv", "Mn", "Mw", "LCB", "SCB"])
    page.click("#use-y")  # the quality variables are Y: PCA models the 14 process variables
    page.wait_for_function("window.__mva.state.fit.variables.length === 14")
    ready(page)
    page.evaluate("ids => window.__mva.store.select('rows', ids)", ["48", "49", "50", "51", *UPSET])
    page.click("#explain")
    page.wait_for_selector("#explanation:not([hidden])")
    page.wait_for_function("document.querySelectorAll('#evidence > div .main-svg').length > 0")
    assert page.text_content("#summary tr:nth-child(2) td:first-child") == "z2"
    assert "z2 contributes most" in page.text_content("#explanation-sentence")
    assert len(page.query_selector_all("#evidence > div")) == 3
    bars = traces(page, "contributions")[0]["ids"]
    assert bars[0] == "z2"


@pytest.mark.dataset
def test_the_table_answers_the_keyboard_and_reads_the_cell_out(page) -> None:
    open_ldpe(page)
    page.focus("#table")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Space")
    assert page.evaluate("[...window.__mva.store.selected('rows')]") == ["2"]
    assert page.text_content("#table .visually-hidden").startswith("Row 2, Tin: ")


@pytest.mark.dataset
def test_a_recipe_step_refines_the_table_and_writes_its_python(page) -> None:
    open_ldpe(page)
    page.fill("#new-name", "dT1")
    page.fill("#new-expression", "Tout1 - Tin")
    page.click("#add-column")
    page.wait_for_function("document.querySelectorAll('#steps li').length === 1")
    ready(page)
    assert page.text_content("#steps li") == "Add the column dT1 = Tout1 - Tin"
    assert 'df["dT1"] = (df["Tout1"] - df["Tin"])' in page.text_content("#code")
    page.wait_for_function("document.getElementById('loadings').data[0].customdata.includes('dT1')")


@pytest.mark.dataset
def test_a_saved_project_opens_again_with_its_exclusions_and_sets(page, tmp_path) -> None:
    open_ldpe(page)
    page.evaluate("ids => { window.__mva.store.saveSet('Upset', ids); window.__mva.store.exclude(ids); }", UPSET)
    page.wait_for_function("document.getElementById('model-summary').textContent.includes('3 excluded')")
    ready(page)
    with page.expect_download() as download:
        page.click("#save-project")
    path = Path(tmp_path / "ldpe.mvproj")
    download.value.save_as(path)
    page.reload()
    page.set_input_files("#open-project", str(path))
    page.wait_for_function("document.getElementById('model-summary').textContent.includes('3 excluded')")
    ready(page)
    assert page.text_content("#sets button").endswith("Upset (3)")
