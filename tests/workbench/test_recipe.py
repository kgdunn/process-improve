"""The workbench recipe: every step, and every script it writes, rebuilds the same table bit for bit."""

from __future__ import annotations

import ast
import json
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from process_improve.workbench import (
    STEPS,
    TRANSFORMS,
    AddColumn,
    DropColumns,
    DropRows,
    FilterRows,
    KeepColumns,
    Recipe,
    RenameColumns,
    SetIndex,
    SourceFile,
    ToNumeric,
    Transform,
    step_from_dict,
)
from process_improve.workbench._expressions import Expression, python_literal

if TYPE_CHECKING:
    from pathlib import Path

    from process_improve.workbench import Step


def table() -> pd.DataFrame:
    """Return a small table with what real data carries: missing values, integers, text, a name with a space."""
    rng = np.random.default_rng(5)
    n = 12
    return pd.DataFrame(
        {
            "Batch": [f"B{i:02d}" for i in range(n)],
            "Tin": rng.normal(200, 5, n).round(3),
            "Tout": rng.normal(240, 8, n).round(3),
            "flow rate": rng.uniform(0.1, 0.9, n),
            "count": rng.integers(1, 50, n),
            "grade": rng.choice(["A", "B", "C"], n),
            "noisy": [str(v) if i % 4 else "n/a" for i, v in enumerate(rng.normal(1, 0.2, n).round(4))],
        },
        index=pd.RangeIndex(100, 100 + n, name="row"),
    ).assign(Tout=lambda d: d["Tout"].where(d.index % 5 != 0))


def write_source(directory: Path, df: pd.DataFrame, name: str = "plant.csv") -> tuple[SourceFile, bytes]:
    content = df.to_csv().encode()
    (directory / name).write_bytes(content)
    return SourceFile.of(name, content, index_col=0), content


def run_script(recipe: Recipe, directory: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Run the recipe's script where its source file is, and return the table it built."""
    monkeypatch.chdir(directory)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # log of a negative, as the recipe met it too
        exec(compile(recipe.to_python(), "recipe.py", "exec"), namespace)  # noqa: S102 - the test runs its own script
    built = namespace["df"]
    assert isinstance(built, pd.DataFrame)
    return built


def assert_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Equal to the bit: same labels, dtypes and values, with every float compared by its bytes."""
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    for column in left.columns:
        if pd.api.types.is_float_dtype(left[column]):
            assert left[column].to_numpy().tobytes() == right[column].to_numpy().tobytes(), column


ONE_STEP: list[Step] = [
    DropColumns(("Tin", "grade")),
    KeepColumns(("grade", "Tout", "Tin")),
    RenameColumns((("Tin", "T in"), ("grade", "Grade"))),
    DropRows((101, 105), reason="sensor failure"),
    FilterRows("Tin > 199 and not (grade == 'C')"),
    FilterRows("`flow rate` < 0.5 or count >= 30"),
    AddColumn("dT", "Tout - Tin"),
    AddColumn("ratio", "log(Tout / Tin) ** 2 + sqrt(`flow rate`) - abs(-count)"),
    AddColumn("constant", "2.5"),
    ToNumeric(("noisy",)),
    SetIndex("Batch"),
    *[Transform("flow rate", name) for name in TRANSFORMS],
    Transform("Tin", "log", constant=-150.0, new_name="log Tin"),
    Transform("count", "power", constant=1.5),
]


@pytest.mark.parametrize("step", ONE_STEP, ids=lambda s: s.describe())
def test_each_step_s_script_rebuilds_the_table_it_applies(step: Step, tmp_path, monkeypatch) -> None:
    source, content = write_source(tmp_path, table())
    recipe = Recipe(source).then(step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        applied = recipe.run(content)
    assert_identical(run_script(recipe, tmp_path, monkeypatch), applied)


def random_step(rng: random.Random, df: pd.DataFrame) -> Step | None:
    """Return a random step that can apply to ``df``, or None when the kind drawn cannot."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    columns, wide, long = list(df.columns), len(df.columns) > 3, len(df) > 4
    a, b = rng.sample(numeric, 2) if len(numeric) >= 2 else (None, None)
    candidates: dict[str, Step | None] = {
        "drop": DropColumns((rng.choice(columns),)) if wide else None,
        "keep": KeepColumns(tuple(rng.sample(columns, len(columns) - 1))) if wide else None,
        "rename": RenameColumns(((columns[0], f"{columns[0]} {rng.randint(0, 99)}"),)),
        "rows": DropRows((rng.choice(list(df.index)),), reason="random") if long else None,
        "filter": FilterRows(f"`{a}` >= {float(df[a].min())} or `{a}` != `{a}`")
        if a and long and pd.notna(df[a].min())
        else None,
        "add": AddColumn(f"new {rng.randint(0, 999)}", f"`{a}` * {rng.uniform(-2, 2)!r} - exp(`{b}` / 1000)")
        if a
        else None,
        "transform": Transform(rng.choice(numeric), rng.choice(list(TRANSFORMS))) if numeric else None,
    }
    return candidates[rng.choice(list(candidates))]


@pytest.mark.parametrize("seed", range(25))
def test_random_recipes_rebuild_the_same_table(seed: int, tmp_path, monkeypatch) -> None:
    """Twenty-five recipes of up to eight random steps: each script rebuilds its table bit for bit."""
    rng = random.Random(seed)  # noqa: S311 - drawing test cases, not secrets
    source, content = write_source(tmp_path, table().drop(columns=["noisy"]))
    recipe = Recipe(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        current = recipe.run(content)
        for _ in range(8):
            step = random_step(rng, current)
            if step is not None:
                recipe = recipe.then(step)
                current = step.apply(current)
    assert len(recipe.steps) >= 4
    assert_identical(run_script(recipe, tmp_path, monkeypatch), current)


def test_a_recipe_round_trips_through_json_and_applies_the_same(tmp_path) -> None:
    source, content = write_source(tmp_path, table())
    recipe = Recipe(source).then(*ONE_STEP[3:8], Transform("count", "log10", constant=1.0, new_name="lc"))
    again = Recipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    assert again == recipe
    assert_identical(again.run(content), recipe.run(content))


def test_every_step_kind_round_trips() -> None:
    seen = {type(step) for step in ONE_STEP}
    assert seen == set(STEPS.values()), "ONE_STEP covers every kind"
    for step in ONE_STEP:
        assert step_from_dict(json.loads(json.dumps(step.to_dict()))) == step


def test_the_script_refuses_a_different_file_of_the_same_name(tmp_path, monkeypatch) -> None:
    source, _ = write_source(tmp_path, table())
    (tmp_path / "plant.csv").write_bytes(b"Batch,Tin\nB00,1\n")
    with pytest.raises(ValueError, match="not the file this recipe was written for"):
        run_script(Recipe(source), tmp_path, monkeypatch)


def test_an_excel_source_reads_the_same_in_the_script(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "plant.xlsx"
    table().to_excel(path)
    content = path.read_bytes()
    recipe = Recipe(SourceFile.of("plant.xlsx", content, format="excel", index_col=0)).then(
        AddColumn("dT", "Tout - Tin")
    )
    assert_identical(run_script(recipe, tmp_path, monkeypatch), recipe.run(content))


def test_a_failing_step_names_its_number_and_the_reason(tmp_path) -> None:
    source, content = write_source(tmp_path, table())
    recipe = Recipe(source).then(DropColumns(("Tin",)), AddColumn("dT", "Tout - Tin"))
    with pytest.raises(ValueError, match=r"^Step 2 \(Add the column dT = Tout - Tin\) failed: .*'Tin'"):
        recipe.run(content)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (DropColumns(("absent",)), "no column absent"),
        (DropRows((999,)), "no row 999"),
        (RenameColumns((("Tin", "Tout"),)), "two columns the same name"),
        (SetIndex("grade"), "repeated values"),
        (FilterRows("Tin + 1"), "not a condition"),
    ],
)
def test_a_step_that_cannot_apply_says_why(step: Step, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        step.apply(table())


# ---------------------------------------------------------------------------
# A recipe is data: nothing in it becomes code
# ---------------------------------------------------------------------------

HOSTILE = 'x"]); __import__("os").system("echo owned"); (["'


def test_hostile_names_and_reasons_stay_literals_in_the_script(tmp_path) -> None:
    """A name or a reason that tries to close its quote, or start a line, adds no statement to the script."""
    df = table().rename(columns={"Tin": HOSTILE})
    source, content = write_source(tmp_path, df)
    recipe = Recipe(source).then(
        AddColumn(f"{HOSTILE}2", f"`{HOSTILE}` * 2"),
        DropRows((100,), reason='fine\nimport os\nos.system("echo owned")'),
        RenameColumns(((HOSTILE, "Tin"),)),
    )
    recipe.run(content)
    tree = ast.parse(recipe.to_python())
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and hasattr(node.func, "attr")}
    assert "system" not in calls
    assert not any(isinstance(node, ast.Import) and node.names[0].name == "os" for node in ast.walk(tree))
    assert sum(isinstance(node, ast.Assign) for node in tree.body) == 5  # SOURCE, df, and one per step


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os")',
        "Tin.real",
        "Tin[0]",
        "(lambda: 1)()",
        "[Tin]",
        "Tin if Tout else 1",
        "print(Tin)",
        "log(Tin, 2)",
        "'text'",
        "Tin @ Tout",
        "x := 1",
    ],
)
def test_an_expression_outside_the_language_is_refused(expression: str) -> None:
    with pytest.raises(ValueError, match=r"not allowed|does not parse|not a column"):
        Expression(expression, ["Tin", "Tout"])


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "eval", "code": "1"},
        {"kind": "drop_columns", "columns": "Tin"},
        {"kind": "drop_rows", "rows": [True]},
        {"kind": "drop_rows", "rows": [float("nan")]},
        {"kind": "rename_columns", "pairs": [["a"]]},
        {"kind": "add_column", "name": "x", "expression": "__import__('os')"},
        {"kind": "transform", "column": "Tin", "function": "eval"},
        {"kind": "transform", "column": "Tin", "function": "log", "constant": "1"},
    ],
)
def test_a_step_read_from_a_file_is_checked(data: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        step_from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"name": "a.csv", "format": "pickle"},
        {"name": "a.csv", "options": {"engine": "python"}},
        {"name": "a.csv", "options": {"converters": {}}},
        {"name": "a.csv", "sha256": "not a digest"},
    ],
)
def test_a_source_read_from_a_file_is_checked(data: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceFile.from_dict(data)


# ---------------------------------------------------------------------------
# The expression language
# ---------------------------------------------------------------------------


def test_a_name_with_spaces_is_written_in_back_ticks() -> None:
    expression = Expression("`flow rate` * 2", ["flow rate"])
    assert expression.columns == ["flow rate"]
    assert expression.to_code() == '(df["flow rate"] * 2)'


def test_a_plain_name_matches_its_column_however_its_accent_is_encoded() -> None:
    decomposed = "Témp"  # e and a combining acute; the parser composes it
    expression = Expression("Témp + 1", [decomposed])
    assert expression.columns == [decomposed]
    frame = pd.DataFrame({decomposed: [1.0, 2.0]})
    pd.testing.assert_series_equal(expression.evaluate(frame), frame[decomposed] + 1)


def test_and_or_not_bind_after_comparisons_and_are_written_as_pandas_operators() -> None:
    expression = Expression("a > 1 and not b < 2 or a == 0", ["a", "b"])
    assert expression.to_code() == '(((df["a"] > 1) & (~(df["b"] < 2))) | (df["a"] == 0))'


def test_a_chained_comparison_is_refused_with_what_to_write_instead() -> None:
    with pytest.raises(ValueError, match="Join them with 'and'"):
        Expression("Tin > 1 & Tout < 2", ["Tin", "Tout"])


def test_an_expression_has_a_length_limit() -> None:
    with pytest.raises(ValueError, match="at most 500 characters"):
        Expression("+".join(["Tin"] * 200), ["Tin"])


@pytest.mark.parametrize("value", ["plain", 'say "hi"', "it's", "both ' and \"", "tab\there", "", "µ"])
def test_python_literal_reads_back_as_the_same_text(value: str) -> None:
    assert ast.literal_eval(python_literal(value)) == value


def test_python_literal_writes_numbers_and_containers_exactly() -> None:
    values = [0.1, 1e-300, -0.0, 7, [1, "a"], {"k": 2.5}, (3,)]
    for value in values:
        assert ast.literal_eval(python_literal(value)) == value
    assert python_literal(float("inf")) == "float('inf')"


def test_a_column_whose_name_holds_a_back_tick_cannot_be_transformed() -> None:
    with pytest.raises(ValueError, match="back-tick"):
        Transform("odd`name", "log")


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "transform", "column": "Tin", "function": "log", "new_name": 3},
    ],
)
def test_a_transform_s_new_name_must_be_text(data: dict) -> None:
    with pytest.raises(TypeError, match="new_name"):
        step_from_dict(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "a.csv", "options": ["index_col"]}, "must be a mapping"),
        ({"name": "a.csv", "sha256": 12}, "must be text"),
    ],
)
def test_a_source_s_fields_must_have_their_types(data: dict, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        SourceFile.from_dict(data)


def test_a_recipe_needs_a_source_and_a_list_of_steps() -> None:
    with pytest.raises(TypeError, match="needs a 'source' mapping"):
        Recipe.from_dict({"source": "a.csv", "steps": []})


def test_an_expression_that_cannot_be_tokenized_is_reported_as_not_parsing() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        Expression("'unterminated", None)


def test_a_source_without_a_digest_writes_a_script_that_reads_it_unchecked(tmp_path, monkeypatch) -> None:
    content = table().to_csv().encode()
    (tmp_path / "plant.csv").write_bytes(content)
    recipe = Recipe(SourceFile("plant.csv", options={"index_col": 0})).then(AddColumn("dT", "Tout - Tin"))
    script = recipe.to_python()
    assert "hashlib" not in script
    assert_identical(run_script(recipe, tmp_path, monkeypatch), recipe.run(content))
