"""A study of one factor in the browser: two suppliers compared, from the brief to the comparison of their results.

Design sizes the design by the power of the t-test of the two levels, a row for each number of runs at each
level; Learn leads with the difference between the levels, tested four ways, and draws the runs at each level
in place of the response surface.
"""

from __future__ import annotations

from tests.web.e2e._page import headers, name_response, practise, run_sheet, set_factors, settled
from tests.web.e2e.test_report import download_report, opened

SUPPLIER = {"name": "Supplier", "low": "A", "high": "B"}


def plan_of(page) -> dict:
    """Read the plan of runs at each level: each row's runs at each level, and the row marked."""
    return page.evaluate(
        """() => {
          const rows = [...document.querySelectorAll('#candidates .levels-plan tbody tr')];
          const runs = (tr) => Number(tr.dataset.runs);
          return { runs: rows.map(runs), marked: rows.filter((tr) => tr.ariaSelected === 'true').map(runs) };
        }"""
    )


def one_factor(page) -> None:
    """Make a new study's brief one categorical factor, and wait for its plan."""
    set_factors(page, [SUPPLIER])
    page.wait_for_selector("#candidates .levels-plan tbody tr")
    settled(page)


def test_one_factor_s_design_is_sized_by_the_power_of_its_t_test(fresh_page):
    page = fresh_page  # a new study, its brief empty
    one_factor(page)
    plan = plan_of(page)
    assert plan["runs"][:3] == [2, 3, 4]
    assert plan["marked"] == [6], "the fewest runs at each level for 80% power to find two noise standard deviations"
    assert (page.input_value("#n_replicates"), page.input_value("#n_center_points")) == ("6", "0")
    assert page.is_hidden("#compare"), "one design, its two levels: nothing to compare in detail"
    assert "6 runs at each level, 12 in all, are the fewest" in page.text_content("#candidates")


def test_a_row_picked_sets_the_runs_at_each_level_and_the_run_sheet_makes_them(fresh_page):
    page = fresh_page  # a new study, its brief empty
    one_factor(page)
    page.click("#candidates .levels-plan tr[data-runs='10']")
    assert page.input_value("#n_replicates") == "10"
    assert plan_of(page)["marked"] == [10]
    page.fill("#n_replicates", "8")
    page.dispatch_event("#n_replicates", "input")
    assert plan_of(page)["marked"] == [8], "the plan follows the replicates typed in"
    page.click("#generate")
    settled(page)
    column = headers(page).index("Supplier")
    assert sorted(run[column] for run in run_sheet(page)) == ["A"] * 8 + ["B"] * 8


def test_the_noise_and_the_change_worth_finding_size_the_design(fresh_page):
    page = fresh_page  # a new study, its brief empty
    name_response(page, "Yield", "%", noise=2, change=3)
    one_factor(page)
    page.wait_for_function("document.querySelector('#n_replicates').value === '9'")
    assert plan_of(page)["marked"] == [9]
    assert "Power to find a change of 3 %" in page.text_content("#candidates .levels-plan thead")


def test_learn_compares_the_two_levels_and_draws_the_runs_at_each(fresh_page):
    page = fresh_page  # a new study, its brief empty
    one_factor(page)
    page.click("#generate")
    settled(page)
    practise(page)
    analysis = page.text_content("#analysis")
    assert "Whether B differs from A" in analysis
    assert analysis.index("On average,") < analysis.index("In coded units."), "the comparison leads, the model follows"
    tests = page.locator("#analysis h4 + .scroll tbody tr")
    assert tests.count() == 4
    names = tests.locator("td:first-child .term").all_text_contents()
    assert names == ["t-test", "Welch's t-test", "Randomization test", "Rank-sum test"]
    page.wait_for_function(
        """() => [...document.querySelectorAll('#charts figure.chart')].some(
             (f) => f.querySelector('h3').textContent === 'The runs at each level' && f.querySelector('.main-svg'))"""
    )
    settled(page)
    titles = page.locator("#charts figure.chart h3").all_text_contents()
    assert titles[:2] == ["Coefficients", "The runs at each level"]
    assert "Fitted response surface" not in titles
    assert "randomization_test(low, high" in page.text_content("#py-analyse code"), "the Python panel tests them too"


# --------------------------------------------------------------------------- one factor of three levels or more
CATALYST = {"name": "Catalyst", "levels": ["Copper", "Nickel", "Platinum", "Zinc"]}


def test_a_factor_of_four_levels_is_sized_for_the_one_way_analysis_of_variance(fresh_page):
    page = fresh_page  # a new study, its brief empty
    set_factors(page, [CATALYST])
    page.wait_for_selector("#candidates .levels-plan tbody tr")
    settled(page)
    plan = plan_of(page)
    marked = plan["marked"][0]
    hint = page.text_content("#candidates")
    assert f"{marked} runs at each level, {4 * marked} in all, are the fewest that give the one-way analysis" in hint
    assert "the design runs its 4 levels in random order" in hint


def test_learn_compares_four_levels_every_pair_and_each_against_the_control_picked(fresh_page):
    page = fresh_page  # a new study, its brief empty
    set_factors(page, [CATALYST])
    page.wait_for_selector("#candidates .levels-plan tbody tr")
    settled(page)
    page.click("#generate")
    settled(page)
    practise(page)
    page.click("#go-learn")
    analysis = page.locator("#analysis")
    assert "Whether the levels of Catalyst differ" in analysis.text_content()
    words = analysis.locator(".term").all_text_contents()
    assert {"one-way analysis of variance", "Tukey's intervals", "Dunnett's intervals"} <= set(words)
    tables = analysis.locator("table")  # each level's runs, the checks, Tukey's pairs, and Dunnett's
    pairs = tables.nth(2).locator("tbody td:first-child").all_text_contents()
    assert pairs == [
        "Nickel less Copper",
        "Platinum less Copper",
        "Zinc less Copper",
        "Platinum less Nickel",
        "Zinc less Nickel",
        "Zinc less Platinum",
    ], "every pair of the four levels, in their order"
    assert page.input_value("#control") == "Copper", "the first level is the control until another is picked"
    assert tables.nth(3).locator("tbody td:first-child").all_text_contents() == ["Nickel", "Platinum", "Zinc"]
    page.select_option("#control", "Platinum")
    settled(page)
    assert page.input_value("#control") == "Platinum"
    assert tables.nth(3).locator("tbody td:first-child").all_text_contents() == ["Copper", "Nickel", "Zinc"]
    sent = [c["payload"].get("control") for c in page.evaluate("window.__pi.calls") if c["fn"] == "api_analyze_runs"]
    assert sent[-1] == "Platinum"


def test_three_levels_are_checked_for_what_the_analysis_of_variance_assumes_in_learn_its_code_and_its_report(
    fresh_page,
):
    """Bartlett's test and the Kruskal-Wallis test, each with a sentence, beside the analysis of variance."""
    page = fresh_page  # a new study, its brief empty
    set_factors(page, [{"name": "Catalyst", "levels": ["Copper", "Nickel", "Zinc"]}])
    page.wait_for_selector("#candidates .levels-plan tbody tr")
    settled(page)
    page.click("#generate")
    settled(page)
    practise(page)
    page.click("#go-learn")
    analysis = page.locator("#analysis")
    checks = analysis.locator("h4:text-is('Checks of what the analysis of variance assumes') + .scroll tbody tr")
    assert checks.locator("td:first-child .term").all_text_contents() == ["Bartlett's test", "Kruskal-Wallis test"]
    assert checks.locator("td:nth-child(3)").all_text_contents() == ["2", "2"], "a degree of freedom per level but one"
    said = analysis.locator("h4:text-is('Checks of what the analysis of variance assumes') ~ ul.facts").first
    sentences = said.locator("li").all_text_contents()
    assert len(sentences) == 2, "a sentence for each check"
    assert sentences[0].startswith("Bartlett's test ")
    assert sentences[1].startswith("The Kruskal-Wallis test, which ranks the results and so does not assume normal")
    code = page.text_content("#py-analyse code")
    assert "spread = bartlett_test(" in code
    assert "ranks = kruskal_wallis_test(" in code, "the Python panel checks them too"
    _, html = download_report(page, "#learn-report-button")
    report = opened(page, html)
    assert "Checks of what the analysis of variance assumes" in report.locator("h4").all_text_contents()
    assert sentences[1] in report.text_content("body"), "the report holds the checks, as Learn shows them"
