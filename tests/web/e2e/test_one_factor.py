"""A study of one factor in the browser: two suppliers compared, from the brief to the comparison of their results.

Design sizes the design by the power of the t-test of the two levels, a row for each number of runs at each
level; Learn leads with the difference between the levels, tested four ways, and draws the runs at each level
in place of the response surface.
"""

from __future__ import annotations

from tests.web.e2e._page import headers, name_response, practise, run_sheet, set_factors, settled

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
