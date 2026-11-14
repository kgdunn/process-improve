"""Each chart fits the room it has, and opens in a window of its own (``figure.js``) to be looked at closely.

A card draws its chart with no toolbar over it; its Expand button opens the figure over the page, in a shape the
reader chooses or drags to, with Plotly's buttons to zoom and turn it. From there it is saved as an image in the
shape shown, on white, under its title, or copied to the clipboard.
"""

from __future__ import annotations

import struct

import pytest

from tests.web.e2e._page import reveal
from tests.web.e2e.test_charts import MORE_TITLES, open_more

#: Whether every chart on the page is drawn at the width its box has now.
FITTED = """(box) => {
  const plots = [...document.querySelectorAll(`${box} .js-plotly-plot`)].filter((p) => p.clientWidth);
  return plots.length > 0 && plots.every((p) => p._fullLayout.width === p.clientWidth);
}"""

#: Whether the window's chart is drawn at the size of its frame.
SHOWN = """() => {
  const plot = document.querySelector('dialog.figure[open] .plot');
  const drawn = plot?._fullLayout;
  return Boolean(drawn) && drawn.width === plot.clientWidth && drawn.height === plot.clientHeight;
}"""

TOUCH = {"viewport": {"width": 820, "height": 1100}, "has_touch": True, "is_mobile": True}


def expand(page, chart: str) -> None:
    """Open one of Learn's charts in its window, and wait until it is drawn there."""
    page.click(f"figure[data-chart={chart}] .expand")
    page.wait_for_function(SHOWN)


def frame(page) -> dict[str, float]:
    """Return the size of the figure in its window."""
    return page.locator("dialog.figure .plot").evaluate("(p) => ({ width: p.clientWidth, height: p.clientHeight })")


def pressed(page) -> list[str]:
    """Return the shapes shown as chosen: one, or none once the figure is resized by hand."""
    return page.locator("dialog.figure [data-shape][aria-pressed=true]").all_text_contents()


def widths(page) -> list[int]:
    return page.locator("#charts .js-plotly-plot").evaluate_all("(plots) => plots.map((p) => p._fullLayout.width)")


def test_every_chart_is_as_wide_as_its_card(page):
    """The first chart of a row was drawn before its neighbour was added, a whole row wide, and ran under it."""
    open_more(page)
    page.wait_for_function(FITTED, arg="#charts")
    spill = page.locator("#charts figure.chart").evaluate_all(
        """(cards) => cards.filter((card) => {
             const drawn = card.querySelector('.main-svg').getBoundingClientRect();
             return drawn.right > card.getBoundingClientRect().right + 0.5;
           }).map((card) => card.dataset.chart)"""
    )
    assert spill == [], "no chart is drawn past the right of its card"
    in_a_row = page.locator("#more-charts figure.chart").evaluate_all(
        "(cards) => cards.filter((c) => c.getBoundingClientRect().top === cards[0].getBoundingClientRect().top).length"
    )
    assert in_a_row > 1, "the charts share their rows here, which is where the first one spilled"


def test_a_chart_is_drawn_again_to_fit_when_its_room_changes(page):
    open_more(page)
    page.wait_for_function(FITTED, arg="#charts")
    wide = widths(page)
    page.set_viewport_size({"width": 700, "height": 900})
    page.wait_for_function(
        f"(was) => ({FITTED})('#charts') && document.querySelector('#charts .js-plotly-plot')._fullLayout.width < was",
        arg=wide[0],
    )
    assert all(now < was for now, was in zip(widths(page), wide, strict=True)), "each in the narrower room it has"


def test_a_card_has_no_toolbar_and_no_chart_offers_to_be_sent_to_plotly(page):
    """A study's data stays in the browser: Plotly's button to share a chart uploads it, and is never shown."""
    open_more(page)
    assert page.locator("#charts .modebar-btn").count() == 0, "the toolbar sat over the legend and the colour bar"
    expand(page, "perturbation")
    buttons = page.locator("dialog.figure .modebar-btn").evaluate_all("(all) => all.map((b) => b.dataset.title)")
    assert buttons == ["Zoom", "Pan", "Zoom in", "Zoom out", "Autoscale", "Reset axes"]


def test_a_long_axis_title_is_wrapped_to_its_card_and_whole_in_the_window(page):
    title = "document.querySelector('{} .plot')._fullLayout.xaxis.title.text"
    open_more(page)
    page.wait_for_function(FITTED, arg="#charts")
    card = page.evaluate(title.format("figure[data-chart=normal_probability]"))
    assert card.replace("<br>", " ") == "Expected, were the residuals normally distributed"
    assert "<br>" in card, "on two lines, where one line was cut short"
    expand(page, "normal_probability")
    assert page.evaluate(title.format("dialog.figure")) == "Expected, were the residuals normally distributed"


def test_every_chart_has_a_button_that_opens_it_in_a_window_of_its_own(page):
    open_more(page)
    cards = page.locator("#charts figure.chart")
    assert cards.locator(".expand").count() == cards.count() == 4 + len(MORE_TITLES)
    assert cards.locator(".chart__head .expand").count() == cards.count(), "beside each chart's heading"
    card = page.locator("figure[data-chart=main_effects]")
    expand(page, "main_effects")
    assert page.text_content("#figure-title") == card.locator("h3").text_content() == "Average result at each setting"
    assert page.text_content("dialog.figure .figure__caption") == card.locator("figcaption").text_content()
    names = "(plot) => plot.data.map((trace) => trace.name)"
    assert (
        page.locator("dialog.figure .plot").evaluate(names) == card.locator(".plot").evaluate(names) == ["T", "P", "F"]
    )
    assert page.locator("dialog.figure .plot").get_attribute("aria-label") == "Average result at each setting"

    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.figure", state="hidden")
    assert page.locator("dialog.figure .plot").evaluate("(plot) => plot.data") is None, (
        "its chart is taken down with it"
    )
    assert page.locator("dialog.figure .plot .main-svg").count() == 0
    assert card.locator(".expand").evaluate("(button) => button === document.activeElement"), (
        "back where it was pressed"
    )


def test_each_chart_is_one_picture_named_by_its_heading(page):
    """To a reader who cannot see it a chart is one picture, with its numbers in the table beside it."""
    open_more(page)
    named = page.locator("#charts figure.chart").evaluate_all(
        """(cards) => cards.map((card) => {
             const plot = card.querySelector('.plot');
             return [plot.getAttribute('role'), plot.getAttribute('aria-label'), card.querySelector('h3').textContent];
           })"""
    )
    assert len(named) == 4 + len(MORE_TITLES)
    for role, label, heading in named:
        assert (role, label) == ("img", heading)


def test_what_the_pointer_rests_on_is_named_in_the_page_s_colours(page):
    """The library's label is white, for a white page: on a dark page its pale text could not be read."""
    open_more(page)
    page.emulate_media(color_scheme="dark")
    page.wait_for_function(
        "document.querySelector('figure[data-chart=main_effects] .plot')._fullLayout.hoverlabel.bgcolor !== '#ffffff'"
    )
    label = page.locator("figure[data-chart=main_effects] .plot").evaluate("(plot) => plot._fullLayout.hoverlabel")
    colours = page.evaluate(
        "['--panel', '--fg'].map((name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim())"
    )
    assert [label["bgcolor"], label["font"]["color"]] == colours


def test_the_window_is_closed_by_its_button_and_by_a_click_outside_it(page):
    open_more(page)
    expand(page, "interaction")
    page.click("dialog.figure [data-do=close]")
    page.wait_for_selector("dialog.figure", state="hidden")
    expand(page, "interaction")
    page.mouse.click(4, 4)
    page.wait_for_selector("dialog.figure", state="hidden")


def test_a_figure_opens_in_four_by_three_and_takes_the_shape_pressed(page):
    open_more(page)
    expand(page, "perturbation")
    assert pressed(page) == ["4:3"]
    size = frame(page)
    assert size["width"] / size["height"] == pytest.approx(4 / 3, abs=0.01)
    for shape, ratio in (("1:1", 1), ("16:9", 16 / 9)):
        page.click(f"dialog.figure [data-shape='{shape}']")
        page.wait_for_function(SHOWN)
        assert pressed(page) == [shape]
        size = frame(page)
        assert size["width"] / size["height"] == pytest.approx(ratio, abs=0.01)
    page.click("dialog.figure [data-shape=Fill]")
    page.wait_for_function(SHOWN)
    filled = frame(page)
    assert filled["width"] > size["width"], "all the width there is, which 16:9 left some of"
    assert filled["height"] >= size["height"]
    box = page.locator("dialog.figure").bounding_box()
    assert box["y"] >= 0, "as large as the browser's window allows, and no larger"
    assert box["y"] + box["height"] <= page.viewport_size["height"]
    assert box["x"] + box["width"] <= page.viewport_size["width"]


def test_a_figure_is_resized_by_its_corner_and_the_next_one_opens_at_that_size(page):
    open_more(page)
    expand(page, "perturbation")
    before = frame(page)
    grip = page.locator("dialog.figure .figure__grip").bounding_box()
    page.mouse.move(grip["x"] + grip["width"] / 2, grip["y"] + grip["height"] / 2)
    page.mouse.down()
    page.mouse.move(grip["x"] - 60, grip["y"] - 90, steps=4)
    page.mouse.up()
    page.wait_for_function(SHOWN)
    after = frame(page)
    assert after["width"] < before["width"] - 100, "the window stays centered: the figure narrows by twice the drag"
    assert after["height"] < before["height"] - 150
    assert pressed(page) == [], "a size of the reader's own is none of the shapes"

    page.keyboard.press("Escape")
    expand(page, "residuals_order")
    assert frame(page) == after
    page.keyboard.press("Escape")
    expand(page, "perturbation")
    page.click("dialog.figure [data-shape='4:3']")
    page.wait_for_function(SHOWN)
    assert frame(page) == before, "a shape pressed is as large as the room allows again"


def test_the_corner_is_moved_with_the_arrow_keys(page):
    open_more(page)
    expand(page, "perturbation")
    before = frame(page)
    page.focus("dialog.figure .figure__grip")
    assert "arrow keys" in page.get_attribute("dialog.figure .figure__grip", "aria-label")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("ArrowUp")
    page.wait_for_function(SHOWN)
    assert frame(page) == {"width": before["width"] - 24, "height": before["height"] - 48}
    assert pressed(page) == []


def test_in_its_window_a_surface_takes_the_shape_the_reader_gives_it(page):
    """On its card a surface is square, one coded step as long on each axis; in its window the reader shapes it."""
    sides = "(plot) => [plot._fullLayout.xaxis._length, plot._fullLayout.yaxis._length]"
    open_more(page)
    page.wait_for_function(FITTED, arg="#charts")
    across, up = page.locator("figure[data-chart=prediction_variance] .plot").evaluate(sides)
    assert across == pytest.approx(up, abs=1), "square on its card, as the surface it is drawn over"
    expand(page, "prediction_variance")
    page.click("dialog.figure [data-shape='16:9']")
    page.wait_for_function(SHOWN)
    across, up = page.locator("dialog.figure .plot").evaluate(sides)
    assert across > 1.4 * up, "its axes fill the window's shape"


def png_size(image: bytes) -> tuple[int, int]:
    """Return a PNG's width and height, in pixels, from its first chunk."""
    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", image[16:24])


def test_a_figure_is_saved_as_an_image_in_the_shape_shown_under_its_title(page, tmp_path):
    open_more(page)
    expand(page, "main_effects")
    page.click("dialog.figure [data-shape='16:9']")
    page.wait_for_function(SHOWN)
    with page.expect_download() as png:
        page.click("dialog.figure [data-do=png]")
    assert png.value.suggested_filename == "average-result-at-each-setting.png"
    png.value.save_as(tmp_path / "figure.png")
    width, height = png_size((tmp_path / "figure.png").read_bytes())
    assert width == 3 * 720, "720 pixels wide, as a report's charts, drawn three times that for print"
    assert width / height == pytest.approx(16 / 9, abs=0.01), "the shape the window shows"
    assert page.text_content("dialog.figure [role=status]") == "Saved as average-result-at-each-setting.png."

    with page.expect_download() as svg:
        page.click("dialog.figure [data-do=svg]")
    assert svg.value.suggested_filename == "average-result-at-each-setting.svg"
    svg.value.save_as(tmp_path / "figure.svg")
    drawing = (tmp_path / "figure.svg").read_text(encoding="utf-8")
    assert drawing.startswith("<svg")
    assert "Average result at each setting" in drawing, "an image that leaves the page carries its title"
    assert "rgb(255, 255, 255)" in drawing, "on white, whatever theme the page shows"


def test_a_figure_is_copied_to_the_clipboard_as_an_image(page):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_more(page)
    expand(page, "perturbation")
    page.click("dialog.figure [data-do=copy]")
    page.wait_for_function("document.querySelector('dialog.figure [role=status]').textContent.startsWith('Copied')")
    kinds = page.evaluate("async () => (await navigator.clipboard.read()).flatMap((item) => item.types)")
    assert kinds == ["image/png"]


def test_when_the_browser_will_not_copy_the_window_says_to_save_the_image(page):
    open_more(page)
    expand(page, "perturbation")
    page.evaluate("() => { navigator.clipboard.write = () => Promise.reject(new Error('refused')); }")
    page.click("dialog.figure [data-do=copy]")
    page.wait_for_function("document.querySelector('dialog.figure [role=status]').textContent !== ''")
    assert page.text_content("dialog.figure [role=status]") == "The browser would not copy the image: save it instead."


@pytest.mark.parametrize("context_options", [TOUCH])
def test_under_a_finger_a_drag_on_a_card_scrolls_the_page_and_the_window_zooms(page):
    assert page.evaluate("matchMedia('(pointer: coarse)').matches")
    open_more(page)
    mode = "(plot) => plot._fullLayout.dragmode"
    assert page.locator("figure[data-chart=perturbation] .plot").evaluate(mode) is False
    expand(page, "perturbation")
    assert page.locator("dialog.figure .plot").evaluate(mode) == "zoom"


def test_the_charts_of_the_detailed_comparison_open_over_it(page, first_paint):
    """A window opened from a dialog sits over it, and closing it leaves the dialog as it was."""
    reveal(page, "#compare")
    page.click("#compare")
    charts = 1 + len(first_paint["comparison"]["result"]["designs"])
    page.wait_for_function(f"document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}")
    assert page.locator("#compare-charts .expand").count() == charts
    page.click("#compare-charts figure.fds .expand")
    page.wait_for_function(SHOWN)
    assert page.text_content("#figure-title") == "Fraction of design space (FDS)"
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.figure", state="hidden")
    assert page.is_visible("#compare-dialog"), "Escape closed the figure, and not the comparison under it"


def test_a_chart_with_no_card_has_the_button_above_it(page):
    """The correlation map of the study's runs sits in a panel of Design, with no card and no heading of its own."""
    reveal(page, "#correlation")
    page.click("#correlation > summary")
    page.wait_for_selector("#correlation-plot.js-plotly-plot")
    button = page.locator("#correlation .chart__head .expand")
    assert button.count() == 1
    button.click()
    page.wait_for_function(SHOWN)
    assert page.text_content("#figure-title") == "Which terms these runs tell apart"


def test_a_figure_s_window_shows_the_python_that_draws_it(page):
    """The code of the design, its results and their analysis, then the figure's own: it runs as it is copied."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_more(page)
    expand(page, "interaction")
    panel = page.locator("dialog.figure details.python")
    assert panel.is_visible()
    assert not panel.evaluate("(details) => details.open"), "shut until the reader asks for it"
    panel.locator("summary").click()
    code = panel.locator("code").text_content()
    assert "result = generate_design(" in code
    assert "analysis = analyze_experiment(" in code
    assert '# The figure "Interaction of the two factors", as the library draws it' in code
    assert 'plot_type="interaction",\n    factors_to_plot=["T", "P"],' in code
    assert code.count("import plotly.graph_objects as go") == 1
    assert code.rstrip().endswith("figure.show()")

    panel.locator("[data-do=code]").click()
    page.wait_for_function(
        "document.querySelector('dialog.figure [role=status]').textContent.startsWith('Code copied')"
    )
    assert page.evaluate("navigator.clipboard.readText()") == code


def test_a_surface_s_code_holds_the_others_at_the_best_settings_as_its_caption_says(page):
    expand(page, "contour")
    assert "with F at 18.04 L/min, its best setting" in page.text_content("dialog.figure .figure__caption")
    held = page.locator("dialog.figure code").text_content().split('hold_values={"F": ')[1].split("}")[0]
    assert float(held) == pytest.approx((18.04 - 15) / 5, abs=1e-3), "18.04 L/min, in coded units"


def test_a_figure_the_page_draws_itself_shows_no_python(page):
    """The coefficients' chart is the page's own: the analysis's panel holds the code of its numbers."""
    page.click("figure[data-chart=coefficients] .expand")
    page.wait_for_function(SHOWN)
    assert page.locator("dialog.figure details.python").is_hidden()
    page.keyboard.press("Escape")
    expand(page, "true_contour")
    assert page.locator("dialog.figure details.python").is_hidden(), "drawn from the simulation, not from the results"
