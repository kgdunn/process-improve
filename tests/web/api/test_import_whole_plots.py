"""A table of runs made in whole plots, a split-plot experiment run elsewhere, made a study with its blocks and plots.

Montgomery (2013), Section 14.4: the tensile strength of paper from three pulp preparation methods and four cooking
temperatures, in three replicates. Each replicate makes one batch of pulp by each method (a whole plot) and splits it
into four samples, one for each temperature. Learn tests each term against the spread it meets, as JMP does in the
book's Table 14.19.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.reference_catalogue import reference_experiments


def paper_table(*, whole_plot: str = "Whole plot") -> str:
    """Return the book's runs as a table: the replicate as the block, its batch of pulp as the whole plot."""
    case = next(e for e in reference_experiments() if e["name"] == "montgomery-tensile-strength-split-plot")
    lines = [f"Block\t{whole_plot}\tMethod\tTemperature\tStrength"]
    lines += [f"R{r}\tR{r}M{m}\tM{m}\tT{t}\t{y}" for r, m, t, y in case["runs"]["rows"]]
    return "\n".join(lines)


def opened(api, text: str, **roles: str) -> dict:
    """Make a study of a table, with the roles guessed and any given here by heading."""
    table = api.ok("api_read_table", {"text": text})
    for column in table["columns"]:
        column["role"] = roles.get(column["heading"], column["role"])
    return api.ok("api_import_study", {"columns": table["columns"], "rows": table["rows"]})["campaign"]


def analysed(api, campaign: dict) -> dict:
    """Analyze a study's response with the model it opens on, as Learn does."""
    design = campaign["design"]
    request = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    return api.ok("api_analyze_runs", {**request, "model": campaign["model"], "responses": campaign["responses"]})


def test_the_block_and_whole_plot_columns_are_guessed_from_their_headings(api):
    table = api.ok("api_read_table", {"text": paper_table()})
    assert [c["role"] for c in table["columns"]] == ["block", "whole_plot", "factor", "factor", "response"]


def test_the_factor_set_once_for_each_whole_plot_is_found_hard_to_change(api):
    design = opened(api, paper_table())["design"]
    assert design["spec"]["options"] == {"n_blocks": 3, "hard_to_change": ["Method"]}
    assert design["columns"][:4] == ["Std order", "Run order", "Block", "Whole plot"]
    assert sorted({row[3] for row in design["rows"]}) == list(range(1, 10))
    assert any("Method is hard to change, so the runs are made in 9 groups" in n for n in design["facts"]["notes"])


def test_each_term_is_tested_against_the_spread_it_meets_as_the_book_prints_it(api):
    split = analysed(api, opened(api, paper_table()))["split_plot"]
    tests = {t["source"]: t for t in split["tests"]}
    # Table 14.19: the fixed-effect tests, and the variance components of the batches and of the samples.
    printed = {
        "Method": (2, 4, 7.0781, 0.0485),
        "Temperature": (3, 18, 36.4266, None),
        "Method·Temperature": (6, 18, 3.1538, 0.0271),
    }
    for source, (df, df_error, f_value, p_value) in printed.items():
        assert (tests[source]["df"], round(tests[source]["df_error"])) == (df, df_error)
        assert tests[source]["F"] == pytest.approx(f_value, abs=5e-5)
        if p_value is not None:
            assert tests[source]["p_value"] == pytest.approx(p_value, abs=5e-5)
    assert tests["Temperature"]["p_value"] < 1e-4  # printed <.0001
    assert [tests[s]["against"] for s in printed] == ["whole plots", "runs", "runs"]
    assert tests["Blocks"]["against"] == "whole plots"
    assert split["sd_whole_plot"] ** 2 == pytest.approx(1.2743056, abs=5e-7)
    assert split["sd_run"] ** 2 == pytest.approx(3.9722222, abs=5e-7)
    assert split["whole_plot_terms"] == ["Method"]


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (
            lambda rows: [
                [*row[:1], f"{row[0]}{(int(row[2][1:]) + int(row[3][1:]) // 25) % 2}", *row[2:]] for row in rows
            ],
            "No factor keeps one setting",
        ),
        (lambda rows: [[*row[:1], f"P{i}", *row[2:]] for i, row in enumerate(rows)], "must group the runs"),
        (lambda rows: [[*row[:1], row[2], *row[2:]] for row in rows], "A whole plot lies in two blocks"),
    ],
    ids=["no-factor-set-once", "a-plot-for-each-run", "plots-across-blocks"],
)
def test_whole_plots_that_are_not_whole_plots_are_refused_in_a_sentence(api, change, refusal):
    table = api.ok("api_read_table", {"text": paper_table()})
    rows = change(table["rows"])
    assert refusal in api.error("api_import_study", {"columns": table["columns"], "rows": rows})


def test_two_columns_as_the_whole_plot_are_refused(api):
    error = api.error("api_import_study", {**_two_plot_columns(api)})
    assert "Choose one column at most as the block, and one at most as the whole plot." in error


def _two_plot_columns(api) -> dict:
    """Return a read table in which two columns are chosen as the whole plot."""
    table = api.ok("api_read_table", {"text": paper_table()})
    table["columns"][0]["role"] = "whole_plot"
    return {"columns": table["columns"], "rows": table["rows"]}


def test_a_study_without_whole_plots_still_has_its_analysis_of_variance(api):
    reply = analysed(api, opened(api, paper_table(whole_plot="Batch"), Batch="ignore"))
    assert "split_plot" not in reply
    assert [a["source"] for a in reply["anova"]][:4] == ["Block", "Method", "Temperature", "Method·Temperature"]


def test_a_model_with_a_nested_factor_has_no_table_of_tests(bootstrap):
    """A nested factor's indicators have no factor of their own to stand for them, so no term is tested as a whole."""
    fit = {"formula": "T + Supplier_I1:Batch + Supplier_I2:Batch"}
    assert bootstrap._split_tests(None, fit, None, {"factors": []}) is None
