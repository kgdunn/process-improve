"""The analysis in Learn as the page shows it: the fitted equation, the tables, and the sentences around them.

When the results were simulated, each coefficient stands beside the true value of what it
estimates. A saturated two-level fit is judged with Lenth's method; replicated center runs
give a lack-of-fit test; a model the design cannot fit is refused in a sentence.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from tests.web._support import FACTORS, correlated_study, factors_named, fill_workbook, workbook_of
from tests.web.e2e._page import (
    calls,
    candidates_done,
    choose,
    definition,
    enter_results,
    generate,
    headers,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    status,
    type_results,
)

#: The columns of the coefficients table every fit has: each estimate between the ends of its 95% interval.
COEFFICIENT_HEADS = ["Term", "95% low", "Coefficient", "95% high", "Std. error", "p-value"]


def coefficient_rows(page) -> list[list[str]]:
    table = page.locator("#analysis table").first
    return [row.locator("td").all_text_contents() for row in table.locator("tbody tr").all()]


def coefficient_heads(page) -> list[str]:
    return page.locator("#analysis table").first.locator("thead th button.sort").all_text_contents()


def terms_fitted(page) -> list[str]:
    """Return the terms of the coefficients table by name: it lists them by significance, which the results decide."""
    return sorted(row[0] for row in coefficient_rows(page))


def test_an_analysis_dropped_for_a_new_study_does_not_leave_the_status_line_busy(page):
    """A new study takes away the analysis on its way, and drops its reply: "Analysing..." goes with it."""
    page.evaluate("window.__pi.hold('api_analyze_runs')")
    enter_results(page, {0: 70})
    assert status(page) == ("Analysing...", "busy")
    reveal(page, "#new-study")
    page.click("#new-study")
    page.evaluate("window.__pi.release('api_analyze_runs')")
    settled(page)
    assert page.is_hidden("#status")


def test_the_coefficients_are_listed_most_significant_first_each_between_the_ends_of_its_interval(page, first_paint):
    """What the runs are surest of heads the table, the intercept comes last, and each row reads low, estimate, high."""
    fitted = first_paint["analysis"]["coefficients"]
    table = page.locator("#analysis table").first
    assert coefficient_heads(page)[:6] == COEFFICIENT_HEADS
    rows = coefficient_rows(page)
    terms = [c for c in fitted if c["term"] != "Intercept"]
    assert [row[0] for row in rows] == [*(c["label"] for c in sorted(terms, key=lambda c: c["p_value"])), "Intercept"]
    assert min(c["p_value"] for c in fitted) == next(c["p_value"] for c in fitted if c["term"] == "Intercept"), (
        "last, though nothing is more significant"
    )
    hint = table.locator("xpath=../preceding-sibling::p[1]")
    assert hint.text_content() == "The most significant first, the intercept last."
    said = table.locator("thead th").evaluate_all("(ths) => ths.map((th) => th.ariaSort)")
    assert set(said) == {"none"}, "no heading claims an order the intercept would break"
    for _, low, estimate, high, *_ in rows:
        assert float(low) <= float(estimate) <= float(high)
    looks = (
        table.locator("tbody tr")
        .first.locator("td")
        .evaluate_all(
            "(cells) => cells.map((cell) => [getComputedStyle(cell).color, Number(getComputedStyle(cell).fontWeight)])"
        )
    )
    (_, plain), (dim, _), (bright, bold), (also_dim, _), *_ = looks
    assert dim == also_dim != bright, "the ends of the interval step back"
    assert bold >= 700 > plain, "and the estimate between them stands out"

    table.locator("thead button.sort", has_text="p-value").click()
    assert [row[0] for row in coefficient_rows(page)][-1] == max(fitted, key=lambda c: c["p_value"])["label"]
    table.locator("thead button.sort", has_text="Term").click()
    by_name = [row[0] for row in coefficient_rows(page)]
    assert by_name[:2] == ["F", "F²"], "any heading sorts by its own column, as in every table"
    assert sorted(by_name) == sorted(row[0] for row in rows)


def test_a_table_moved_sideways_on_a_narrow_screen_keeps_its_terms_in_view_and_its_numbers_whole(page):
    """On a phone the table is wider than the screen: moved to its p-values, it showed numbers and no term."""
    page.set_viewport_size({"width": 390, "height": 800})
    reveal(page, "#analysis table")
    held = page.locator("#analysis .scroll").first
    assert held.evaluate("(box) => box.scrollWidth > box.clientWidth"), "the table is wider than its place"
    edge = held.bounding_box()["x"]
    first = held.locator("tbody td").first
    assert first.bounding_box()["x"] == pytest.approx(edge, abs=1)
    held.evaluate("(box) => { box.scrollLeft = box.scrollWidth; }")
    assert held.evaluate("(box) => box.scrollLeft") > 50
    assert first.bounding_box()["x"] == pytest.approx(edge, abs=1), "the terms stay where they were"
    assert first.is_visible()
    heights = held.locator("tbody tr").evaluate_all("(rows) => rows.map((row) => row.getBoundingClientRect().height)")
    assert max(heights) - min(heights) < 1, '"< 0.0001" is not broken after its sign into a second line'


def test_the_coefficients_are_saved_as_a_file_in_full_and_in_the_order_shown(page, first_paint):
    """The page shows four figures. A reader who works on with the numbers gets them as they were computed."""
    fitted = {c["label"]: c for c in first_paint["analysis"]["coefficients"]}
    button = page.get_by_role("button", name="Download this table (.csv)")
    reveal(page, "#analysis table")
    asked = page.evaluate("window.__pi.calls.length")

    def saved() -> tuple[str, list[list[str]]]:
        with page.expect_download() as caught:
            button.click()
        lines = Path(caught.value.path()).read_bytes().decode().split("\r\n")
        assert lines[-1] == "", "each line ends, the last too"
        return caught.value.suggested_filename, [line.split(",") for line in lines[:-1]]

    name, (heads, *lines) = saved()
    assert name == f"coefficients-{first_paint['design']['spec']['response']}.csv"
    assert heads == coefficient_heads(page)
    assert [line[0] for line in lines] == [row[0] for row in coefficient_rows(page)], "in the order shown"
    for term, low, estimate, high, error, p, true in lines:
        row = fitted[term]
        given = [float(low), float(estimate), float(high), float(error), float(p), float(true)]
        full = [row["ci_low"], row["coefficient"], row["ci_high"], row["std_error"], row["p_value"], row["true_value"]]
        assert given == pytest.approx(full, rel=1e-12, abs=0), term
    assert any(len(line[2].replace("-", "").replace(".", "").lstrip("0")) > 4 for line in lines), (
        "more than four figures"
    )
    assert all("<" not in line[5] for line in lines), (
        "a p-value is its number, not what the page prints for a small one"
    )

    page.locator("#analysis table").first.locator("thead button.sort", has_text="Term").click()
    _, (_, *by_term) = saved()
    assert [line[0] for line in by_term] == [row[0] for row in coefficient_rows(page)], "sorted by the reader, saved so"
    assert [line[0] for line in by_term] != [line[0] for line in lines]
    assert page.evaluate("window.__pi.calls.length") == asked, "Python is not asked"


def test_each_estimate_stands_beside_its_true_value(page, first_paint):
    analysis = first_paint["analysis"]
    assert page.text_content("#analysis .equation") == analysis["equation"]
    assert coefficient_heads(page) == [*COEFFICIENT_HEADS, "True value"]
    rows = coefficient_rows(page)
    assert sorted(row[0] for row in rows) == sorted(c["label"] for c in analysis["coefficients"])
    assert {row[0]: float(row[-1]) for row in rows} == {c["label"]: c["true_value"] for c in analysis["coefficients"]}
    shown = definition(page, '#analysis table th .term[data-term="true-value"]')
    assert shown.startswith("What the coefficient estimates for the process the results were simulated from")


def test_results_typed_in_have_no_true_values(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    type_results(page, {run: 50 + run for run in range(11)})
    assert "True value" not in headers(page, "#analysis table")


def test_lack_of_fit_is_tested_against_the_repeated_center_runs(page):
    text = page.text_content("#analysis")
    assert "Lack of fit: F = " in text
    assert "(pure error from the replicated runs: 2 degrees of freedom)" in text


def test_a_saturated_two_level_design_is_judged_with_lenth_s_method(blank_page):
    page = blank_page
    generate(page, design_type="plackett_burman", n_center_points=0)
    type_results(page, {0: 12.1, 1: 15.4, 2: 9.8, 3: 14.0})
    text = page.text_content("#analysis")
    assert "Effects, judged with Lenth's method" in text
    assert re.search(r"PSE = [-\d.]+; ME = [-\d.]+; SME = [-\d.]+\.", text)
    assert definition(page, '#analysis h3 .term[data-term="effect"]').endswith("twice its coefficient.")
    assert "Predicted R² cannot be computed: at least one run cannot be predicted from the others" in text
    lenth = page.locator("#analysis table").nth(1)
    assert lenth.locator("thead th").all_text_contents() == ["Term", "Effect", "Beyond ME", "Beyond SME"]
    assert page.get_attribute("#analysis h3 a", "href").endswith("#doe-lenth-method")


def test_an_orthogonal_design_shows_no_vif_column_and_no_note(page):
    """The worked example's terms are close to uncorrelated: a column of VIFs near 1 would say nothing."""
    assert "VIF" not in headers(page, "#analysis table")
    assert "variance inflation factor" not in page.text_content("#analysis")
    assert "evaluate_design" not in page.text_content("#py-analyse code")


def test_correlated_terms_show_their_vifs_in_a_column_and_a_note(page, api, tmp_path):
    """Runs as given in which A and B move together: the note names them, and the table shows each term's VIF."""
    design = api.ok("api_make_design", correlated_study())
    path = tmp_path / "correlated.xlsx"
    filled = fill_workbook(workbook_of(api, design), lambda runs: 10 + 2 * runs["A"] - runs["C"] + 0.1 * runs.index)
    path.write_bytes(base64.b64decode(filled))
    reveal(page, "#tab-real")
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    # The page reads the file before it calls Python, and is idle meanwhile: wait for the call, then for its answer.
    page.wait_for_function("window.__pi.count('api_read_workbook') === 1")
    settled(page)
    assert coefficient_heads(page) == [*COEFFICIENT_HEADS, "VIF"]
    assert {row[0]: row[-1] for row in coefficient_rows(page)} == {
        "Intercept": "",
        "A": "10.80",
        "B": "11.00",
        "C": "1.200",
    }
    notes = page.locator("#analysis ul.facts li").all_text_contents()
    (note,) = [n for n in notes if "VIF" in n]
    assert note.startswith("A (VIF 11) and B (VIF 11) are correlated with other terms in these runs.")
    # Once the results are read, the mouse that pressed "Enter results" can rest on a word of Learn's and open
    # its definition: move it away, and let that close, before resting it on the heading.
    page.mouse.move(0, 0)
    page.locator(".term-tip").wait_for(state="hidden")
    assert definition(page, '#analysis table th .term[data-term="vif"]').startswith("The variance inflation factor:")
    assert page.locator('#analysis ul.facts .term[data-term="vif"]').text_content() == "variance inflation factor"
    assert 'vif = evaluate_design(X, model="A + B + C", metric="vif")["vif"]' in page.text_content("#py-analyse code")


def test_choosing_a_model_analyses_the_results_again(page):
    choose(page, model="main_effects")
    settled(page)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == "main_effects"
    assert terms_fitted(page) == sorted(["Intercept", "T", "P", "F"])


def test_choosing_a_transform_analyses_the_response_on_its_scale_and_as_measured_again(page):
    choose(page, transform="log")
    settled(page)
    sent = calls(page, "api_analyze_runs")[-1]["payload"]["spec"]
    assert sent["analysis_transforms"] == {sent["response"]: "log"}
    assert page.text_content("#analysis p.equation").startswith("ln(")
    assert "Low to high" in page.locator("#analysis table").first.locator("th").all_text_contents()
    assert "np.log(y)" in page.text_content("#py-analyse code"), "the code fits what the page fits"
    choose(page, transform="")
    settled(page)
    assert "analysis_transforms" not in calls(page, "api_analyze_runs")[-1]["payload"]["spec"]
    assert not page.text_content("#analysis p.equation").startswith("ln(")


def test_box_cox_s_hint_fits_the_transform_it_points_to(page):
    """When Box-Cox's interval holds the log and not 1, the page says so, and its button fits the log."""
    choose(page, model="interactions")
    settled(page)
    reply = page.evaluate("window.__pi.replies.filter((r) => r.fn === 'api_analyze_runs').at(-1).reply")
    hint = {"lambda": 0.1, "ci": [-0.2, 0.4], "suggests": "log", "current": None}
    page.evaluate(
        "(reply) => window.__pi.fake('api_analyze_runs', reply)",
        {**reply, "result": {**reply["result"], "box_cox": hint}},
    )
    choose(page, model="main_effects")
    settled(page)
    assert page.is_visible("#box-cox")
    assert page.text_content("#box-cox-words").startswith(
        "Box-Cox's λ is 0.1 (95% interval -0.2 to 0.4): it points to the log of "
    )
    page.evaluate("() => { delete window.__pi.fakes.api_analyze_runs; }")
    page.click("#box-cox-apply")
    settled(page)
    sent = calls(page, "api_analyze_runs")[-1]["payload"]["spec"]
    assert sent["analysis_transforms"] == {sent["response"]: "log"}
    assert page.input_value("#transform") == "log"


def test_a_model_the_design_cannot_fit_is_refused_in_a_sentence(page):
    generate(page, design_type="dsd", n_center_points=1)  # nine runs, simulated again
    choose(page, model="quadratic")
    settled(page)
    assert status(page) == ("The results could not be analysed.", "error")
    assert "cannot fit the quadratic model" in page.text_content("#analysis p.error")
    assert page.is_hidden("#charts")


def test_a_box_behnken_design_with_a_factor_named_i_is_made_analysed_and_drawn(blank_page):
    """Every square term calls I(), and a factor named I is fitted beside it under its own name in each table."""
    page = blank_page
    set_factors(page, [FACTORS[0], {**FACTORS[1], "name": "I"}, FACTORS[2]])
    generate(page, design_type="box_behnken")
    assert headers(page)[2:5] == ["T", "I", "F"]
    practise(page)
    assert status(page)[1] == "ok"
    terms = ["T", "I", "F", "T·I", "T·F", "I·F", "T²", "I²", "F²"]
    assert terms_fitted(page) == sorted(["Intercept", *terms])
    anova = page.locator("#analysis > .scroll table").last.locator("tbody tr td:first-child").all_text_contents()
    assert anova == [*terms, "Residual"]
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 4")
    assert "over T and I" in page.locator("#charts figcaption").first.text_content()


def test_a_blocked_design_shows_each_run_s_block_and_fits_the_blocks(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2, n_blocks=2)
    assert headers(page)[:3] == ["Std order", "Run order", "Block"]
    assert calls(page, "api_make_design")[-1]["payload"]["n_blocks"] == 2
    practise(page)
    assert "Block 1" in [row[0] for row in coefficient_rows(page)]
    anova = page.locator("#analysis > .scroll table").last.locator("tbody tr td:first-child").all_text_contents()
    assert "Block" in anova
    assert "The blocks are fitted as a fixed effect" in page.text_content("#analysis")
    code = page.text_content("#py-analyse code")
    assert '"Block": runs["Block"],' in code, "the blocks are fitted as the page fits them"


def ticked(page) -> list[str]:
    boxes = page.locator("#term-list input:checked")
    return [boxes.nth(i).locator("xpath=..").text_content().strip() for i in range(boxes.count())]


def test_the_terms_to_fit_can_be_ticked_three_factor_interactions_included(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    assert page.is_hidden("#terms")
    choose(page, model="terms")
    settled(page)
    assert page.is_visible("#terms")
    assert ticked(page) == ["T", "P", "F", "T·P", "T·F", "P·F"], "the list starts from the design's own model"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.get_by_label("T·P·F", exact=True).check()
    page.get_by_label("P·F", exact=True).uncheck()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == {
        "terms": [["T"], ["P"], ["F"], ["T", "P"], ["T", "F"], ["T", "P", "F"]]
    }
    assert terms_fitted(page) == sorted(["Intercept", "T", "P", "F", "T·P", "T·F", "T·P·F"])
    assert 'model="T + P + F + T:P + T:F + T:P:F"' in page.text_content("#py-analyse code")


def test_the_terms_start_from_the_named_model_chosen_before(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    choose(page, model="quadratic")
    choose(page, model="terms")
    assert ticked(page)[-3:] == ["T²", "P²", "F²"]
    choose(page, model="")
    assert page.is_hidden("#terms")


def test_a_design_that_can_fit_the_aim_is_analysed_for_it(blank_page):
    """A DSD made to learn each factor's curvature is analysed with the squares, not with its own main effects."""
    page = blank_page
    generate(page, design_type="dsd")
    assert page.input_value("#model") == "terms"
    assert ticked(page) == ["T", "P", "F", "T²", "P²", "F²"]
    practise(page)
    assert terms_fitted(page) == sorted(["Intercept", "T", "P", "F", "T²", "P²", "F²"])


def test_a_design_that_cannot_fit_the_aim_is_analysed_as_designed(blank_page):
    page = blank_page
    generate(page, design_type="fractional_factorial")
    assert page.input_value("#model") == "", "a two-level design cannot tell the squares apart"


# --------------------------------------------------------------------------- what the results say, and each run
def test_learn_opens_on_what_the_results_say_in_sentences(page):
    findings = page.locator("#findings li").all_text_contents()
    assert findings[0] == "T matters most: raising T from 150 to 200 degC raises Yield by 7.9 %, on average."
    assert page.locator("#analysis h3").first.text_content() == "What the results say"
    assert page.locator("#findings").evaluate(
        "(list) => list.compareDocumentPosition(document.querySelector('#analysis .equation')) & 4"
    )


def test_each_run_is_listed_as_the_model_sees_it_under_a_heading_to_open(page):
    box = page.locator("#diagnostics")
    assert box.locator("summary").text_content() == "Each run, as the model sees it"
    assert not box.evaluate("(d) => d.open"), "no run of the worked example is one to look at again"
    box.locator("summary").click()
    assert box.locator("tbody tr").count() == 15
    heads = box.locator("thead th").all_text_contents()
    assert heads == [
        "Run",
        "Fitted",
        "Residual",
        "In standard deviations",
        "Leverage",
        "Cook's distance",
        "Look again",
        "Set aside",
    ]


# --------------------------------------------------------------------------- a simpler model, proposed and accepted
def test_a_simpler_model_is_proposed_when_asked_and_fitted_when_accepted(page):
    """The worked example's process has five terms and its model nine: the search proposes the five."""
    assert page.text_content("#simplify-words") == ""
    assert page.is_hidden("#simplify-apply"), "nothing is proposed until the reader asks"
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    said = page.text_content("#simplify-words")
    assert said.startswith(
        "The results support T, P, F, T·P and F², without T·F, P·F, T² and P². Adjusted R² would be "
    )
    assert len(coefficient_rows(page)) == 10, "the model is fitted as it was until the reader accepts"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#simplify-apply")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert terms_fitted(page) == sorted(["Intercept", "T", "P", "F", "T·P", "F²"])
    assert page.text_content("#simplify-words") == "", "the proposal was of the analysis before"
    assert page.locator("#term-list input:checked").count() == 5, "ticked in the list, for the reader to change"


def test_a_blocked_design_s_model_is_simplified_and_its_blocks_stay_in(blank_page):
    """The page once refused a blocked study: the search could not keep the blocks in, and dropped effects for it."""
    page = blank_page
    generate(page, design_type="box_behnken", n_blocks=3)
    columns = headers(page)
    rows = [dict(zip(columns, row, strict=False)) for row in run_sheet(page)]
    between = {"1": 5.0, "2": -5.0, "3": 0.0}

    def result(place: int, run: dict) -> float:
        t = (float(run["T"]) - 175) / 25
        return round(60 + 4 * t - 3 * t**2 + between[run["Block"]] + 0.05 * ((place * 7) % 5 - 2), 3)

    enter_results(page, {place: result(place, run) for place, run in enumerate(rows)})
    settled(page)
    assert {"Intercept", "Block 1", "Block 2"} <= set(terms_fitted(page))
    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    kept, dropped = page.text_content("#simplify-words").removeprefix("The results support ").split(", without ")
    assert kept.startswith("T"), "the process's own terms are among those kept"
    assert "T²" in kept
    assert "Block" not in kept + dropped, "the blocks are kept in every model: the search neither picks nor drops them"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#simplify-apply")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    fitted = terms_fitted(page)
    assert {"Intercept", "Block 1", "Block 2"} <= set(fitted), "the blocks stay in the model that is then fitted"
    assert {"T", "T²"} <= set(fitted)
    assert len(fitted) < 12, "and it is a simpler model than the nine terms and two blocks before"
    assert page.locator("#term-list input:checked").count() == len(fitted) - 3, "the blocks are no term to tick"


def test_a_factor_of_several_levels_is_kept_whole_in_the_simpler_model(fresh_page):
    """Its levels are two columns of the model and one term of the proposal, ticked once and fitted with both."""
    page = fresh_page
    set_factors(page, [*FACTORS[:2], {"name": "Catalyst", "levels": ["X", "Y", "Z"]}])
    candidates_done(page)
    generate(page, design_type="full_factorial", n_center_points=0)
    columns = headers(page)
    rows = [dict(zip(columns, row, strict=False)) for row in run_sheet(page)]
    by_level = {"X": 0.0, "Y": 6.0, "Z": -6.0}

    def result(place: int, run: dict) -> float:
        t = (float(run["T"]) - 175) / 25
        return round(50 + 3 * t + by_level[run["Catalyst"]] + 0.05 * ((place * 7) % 5 - 2), 3)

    enter_results(page, {place: result(place, run) for place, run in enumerate(rows)})
    settled(page)
    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    kept = page.text_content("#simplify-words").removeprefix("The results support ").split(", without ")[0]
    assert "Catalyst" in kept
    assert "[" not in kept, f"the factor is named, not its levels: {kept}"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#simplify-apply")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    fitted = [row[0] for row in coefficient_rows(page)]
    assert {"T", "Catalyst[X]", "Catalyst[Y]"} <= set(fitted), "both its columns are fitted"
    assert len(fitted) < 10, "a simpler model than the interactions model's ten columns"


PICKED = "The terms of this model were picked from these same results"


def test_a_nested_model_is_not_proposed_back_as_the_crossed_one(fresh_page):
    """Batches within suppliers: the reader takes Batch's own effect out, and the page does not propose it back.

    Batch 1 of one supplier has nothing to do with batch 1 of another. Searched within the interactions model, as
    it was, the nested term needed Batch's main effect, and the page said the results supported adding it. The
    page now says it proposes no simpler model for such a one, and leaves the reader's model as it is.
    """
    page = fresh_page
    supplier = {"name": "Supplier", "levels": ["X", "Y", "Z"]}
    set_factors(page, [supplier, {"name": "Batch", "levels": ["first", "second"]}])
    candidates_done(page)
    generate(page, design_type="full_factorial", n_center_points=0, n_replicates=3)
    columns = headers(page)
    rows = [dict(zip(columns, row, strict=False)) for row in run_sheet(page)]
    assert len(rows) == 18
    of_supplier = {"X": 0.0, "Y": 5.0, "Z": -5.0}
    of_batch_within = {"X": 3.0, "Y": -2.0, "Z": 0.5}

    def result(place: int, run: dict) -> float:
        sign = 1 if run["Batch"] == "first" else -1
        level = run["Supplier"]
        return round(40 + of_supplier[level] + sign * of_batch_within[level] + 0.05 * ((place * 7) % 5 - 2), 3)

    enter_results(page, {place: result(place, run) for place, run in enumerate(rows)})
    settled(page)
    reveal(page, "#model")
    page.select_option("#model", "terms")
    settled(page)
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.locator("#term-list label").filter(has_text=re.compile(r"^\s*Batch\s*$")).locator("input").uncheck()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    nested = ["Intercept", "Supplier[X]", "Supplier[Y]", "Supplier[X]·Batch", "Supplier[Y]·Batch", "Supplier[Z]·Batch"]
    assert terms_fitted(page) == sorted(nested), "Batch's effect within each supplier"

    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function("!['', 'Searching...'].includes(document.querySelector('#simplify-words').textContent)")
    assert page.text_content("#simplify-words") == (
        "The page proposes no simpler model where one factor is nested in another: "
        "take terms out, or add them, in the list of terms."
    )
    assert page.is_hidden("#simplify-apply"), "nothing to accept, and no Batch to add"
    assert calls(page, "api_simpler_model")[-1]["payload"]["model"] == {"terms": [["Supplier"], ["Supplier", "Batch"]]}
    assert terms_fitted(page) == sorted(nested), "the model the reader chose is the model still fitted"


def simplified(page) -> None:
    """Ask for a simpler model and fit the terms proposed; wait for the model fitted with them."""
    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#simplify-apply")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)


def test_a_model_of_terms_picked_from_its_results_says_so_until_the_reader_chooses_the_terms(page):
    """Its p-values were computed as if the terms had been chosen before the runs: Learn says so after its findings."""
    assert page.locator("#findings li.picked").count() == 0
    simplified(page)
    assert page.locator("#findings li").last.text_content().startswith(PICKED)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == {
        "terms": [["T"], ["P"], ["F"], ["T", "P"], ["F", "F"]]
    }, "Python is asked for the terms: that they were picked is the page's to say"
    at = page.evaluate("[...document.querySelectorAll('#term-list input')].findLastIndex((box) => box.checked)")
    box = page.locator("#term-list input").nth(at)

    def fitted_after(change) -> None:
        before = page.evaluate("window.__pi.count('api_analyze_runs')")
        change()
        page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
        settled(page)

    fitted_after(box.uncheck)
    assert page.locator("#findings li.picked").count() == 0, "the reader chose these terms"
    fitted_after(box.check)
    assert page.locator("#findings li.picked").count() == 1, "and these are the search's again"


def test_terms_picked_from_the_results_are_kept_so_with_the_study(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    practise(page)
    settled(page)
    simplified(page)
    assert page.locator("#findings li.picked").count() == 1
    saved = saved_studies(page, lambda studies: bool(studies) and studies[0]["model"].get("picked") is True)
    assert saved[0]["model"]["picked"] is True
    page.reload()
    page.wait_for_selector("#findings li.picked", state="attached")
    assert page.locator("#findings li.picked").text_content().startswith(PICKED)


def test_a_model_that_uses_every_run_is_not_simplified_and_the_page_says_why(blank_page):
    page = blank_page
    set_factors(page, factors_named(*"AB"))  # four runs, and four coefficients with the interaction
    generate(page, design_type="full_factorial", n_center_points=0)
    enter_results(page, {i: 50 + (i * 7) % 11 for i in range(4)})
    settled(page)
    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function(
        "document.querySelector('#simplify-words').textContent.startsWith('The model uses every run')"
    )
    assert page.is_hidden("#simplify-apply")


# --------------------------------------------------------------------------- a run set aside from the fit
def set_aside(page, run: int) -> None:
    """Open the runs' table and set a run aside, by its run order; wait for the model to be fitted without it."""
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    if not page.locator("#diagnostics").evaluate("(box) => box.open"):
        page.click("#diagnostics > summary")
    page.click(f'#diagnostics button[aria-label="Set run {run} aside from this fit"]')
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)


def test_a_run_set_aside_keeps_its_result_and_the_model_is_fitted_without_it(page):
    set_aside(page, 5)
    assert page.text_content("#aside-runs").startswith("Run 5 is set aside: the model is fitted without it.")
    sent = calls(page, "api_analyze_runs")[-1]["payload"]
    place = next(i for i, row in enumerate(sent["rows"]) if row[sent["columns"].index("Run order")] == 5)
    assert sent["responses"][place] is None, "Python fits the model without it"
    assert sent["set_aside"] == [5], "and is told which run that is, to name it apart from a run still to do"
    assert page.locator(f'#design-table input[data-index="{place}"]').input_value() != "", "the run sheet keeps it"
    assert "14 completed runs" in page.text_content("#analysis")
    assert "Still to run" not in page.text_content("#analysis"), "a run set aside is not one still to do"
    assert page.locator("#diagnostics tbody tr").count() == 14


def test_a_run_set_aside_is_put_back(page):
    set_aside(page, 5)
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#aside-runs button")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert page.locator("#aside-runs").count() == 0
    assert "15 completed runs" in page.text_content("#analysis")
    back = calls(page, "api_analyze_runs")[-1]["payload"]
    assert None not in back["responses"]
    assert "set_aside" not in back


def test_a_run_set_aside_is_kept_with_the_study(blank_page):
    """The study is saved with it, in the format that holds it, and a later visit fits the model without it still."""
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=4)
    practise(page)
    settled(page)
    reveal(page, "#analysis")
    set_aside(page, 3)
    page.wait_for_function("window.__pi.idle()")
    page.wait_for_timeout(600)  # the study is saved a moment after its last change
    page.reload()
    page.wait_for_selector("#aside-runs")
    assert page.text_content("#aside-runs").startswith("Run 3 is set aside: ")
    assert "11 completed runs" in page.text_content("#analysis")


def test_results_simulated_afresh_set_no_run_aside(blank_page):
    """Every result is new, so none of the runs set aside from the old ones stays out of the fit."""
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=4)
    practise(page)
    settled(page)
    reveal(page, "#analysis")
    set_aside(page, 3)
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    reveal(page, "#simulate")
    page.click("#simulate")  # again: new noise
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    reveal(page, "#analysis")
    assert page.locator("#aside-runs").count() == 0
    assert "12 completed runs" in page.text_content("#analysis")
