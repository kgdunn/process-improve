"""The project file: it round-trips, it reproduces its fits on opening, and nothing in it is executed."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.workbench import AddColumn, DropRows, Recipe, SourceFile, Variant
from process_improve.workbench import project as project_module
from process_improve.workbench.project import Project

LDPE = Path(__file__).parents[2] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE" / "LDPE.csv"


def make_project() -> Project:
    """Return the LDPE data with a recipe step, a PCA and a PLS variant, and one saved selection."""
    content = LDPE.read_bytes()
    recipe = Recipe(SourceFile.of("LDPE.csv", content, index_col=0)).then(
        AddColumn("dT1", "Tout1 - Tin"), DropRows((1,), reason="start-up batch")
    )
    x = ("Tin", "Tmax1", "Tout1", "Tmax2", "Tout2", "z1", "z2", "Fi1", "Fi2", "Press", "dT1")
    return Project(
        recipe=recipe,
        source=content,
        variants=(
            Variant("PCA", 3, x_columns=x, excluded=(52, 53, 54), name="Without the upset"),
            Variant("PLS", 2, x_columns=x, y_columns=("Conv", "Mn"), name="Quality"),
        ),
        selections={"Upset": [52, 53, 54]},
    )


def rewrite(content: bytes, replace: dict[str, bytes]) -> bytes:
    """Return the zip ``content`` with the members in ``replace`` swapped for new bytes."""
    source = zipfile.ZipFile(io.BytesIO(content))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for info in source.infolist():
            target.writestr(info.filename, replace.get(info.filename, source.read(info.filename)))
        for name in set(replace) - set(source.namelist()):
            target.writestr(name, replace[name])
    return out.getvalue()


@pytest.mark.dataset
def test_a_project_round_trips_through_its_file(tmp_path) -> None:
    project = make_project()
    path = tmp_path / "ldpe.mvproj"
    project.save(path)
    again = Project.load(path)
    assert again == project
    pd.testing.assert_frame_equal(again.table(), project.table(), check_exact=True)


@pytest.mark.dataset
def test_opening_a_project_refits_every_variant_and_finds_the_same_numbers() -> None:
    opened = Project.open(make_project().to_bytes())
    assert opened.reproduces
    assert {(c.variant, c.array) for c in opened.checks} == {
        (v, a) for v in ("Without the upset", "Quality") for a in ("scores", "loadings", "T2", "SPE")
    }
    assert all(c.max_difference == 0.0 for c in opened.checks), "the same library refits to the bit"
    assert opened.fits[0].rows().loc[[52, 53, 54], "excluded"].all()


@pytest.mark.dataset
def test_a_stored_array_that_no_longer_reproduces_is_named() -> None:
    content = make_project().to_bytes()
    stored = np.load(io.BytesIO(zipfile.ZipFile(io.BytesIO(content)).read("fits/1/SPE.npy")))
    changed = io.BytesIO()
    np.save(changed, stored * 1.01)
    opened = Project.open(rewrite(content, {"fits/1/SPE.npy": changed.getvalue()}))
    assert not opened.reproduces
    failed = [(c.variant, c.array) for c in opened.checks if not c.ok]
    assert failed == [("Quality", "SPE")]


@pytest.mark.dataset
def test_the_project_carries_a_script_that_rebuilds_its_table(tmp_path) -> None:
    project = make_project()
    with zipfile.ZipFile(io.BytesIO(project.to_bytes())) as archive:
        (tmp_path / "LDPE.csv").write_bytes(archive.read("source/LDPE.csv"))
        script = archive.read("recipe.py").decode()
    (tmp_path / "recipe.py").write_text(script + "\ndf.to_pickle('built.pkl')\n", encoding="utf-8")
    subprocess.run([sys.executable, "recipe.py"], cwd=tmp_path, check=True)
    pd.testing.assert_frame_equal(pd.read_pickle(tmp_path / "built.pkl"), project.table(), check_exact=True)  # noqa: S301


def test_a_source_that_is_not_the_recipe_s_file_is_refused() -> None:
    project = make_project()
    with pytest.raises(ValueError, match="not the file the recipe was written for"):
        Project(recipe=project.recipe, source=project.source + b"\n")


@pytest.mark.dataset
def test_a_project_whose_source_was_swapped_is_refused() -> None:
    content = make_project().to_bytes()
    with pytest.raises(ValueError, match="not the file the recipe was written for"):
        Project.from_bytes(rewrite(content, {"source/LDPE.csv": LDPE.read_bytes() + b"\n"}))


@pytest.mark.dataset
def test_an_array_that_needs_pickle_to_load_is_refused() -> None:
    content = make_project().to_bytes()
    hostile = io.BytesIO()
    np.save(hostile, np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle"):
        Project.open(rewrite(content, {"fits/0/scores.npy": hostile.getvalue()}))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not a zip", "not a zip archive"),
        (zipfile.ZipFile, "no manifest.json"),
    ],
)
def test_a_file_that_is_not_a_project_says_so(content, message) -> None:
    if content is zipfile.ZipFile:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "hello")
        content = buffer.getvalue()
    with pytest.raises(ValueError, match=message):
        Project.from_bytes(content)


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "something-else"}, "names another format"),
        ({"version": 99}, "newer process-improve"),
        ({"variants": {"not": "a list"}}, "must be a list"),
        ({"selections": {"Upset": "52"}}, "list of row ids"),
        ({"recipe": {"source": {"name": "a.csv"}, "steps": [{"kind": "exec"}]}}, "Unknown recipe step"),
    ],
)
def test_a_manifest_is_checked_field_by_field(change, message) -> None:
    content = make_project().to_bytes()
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(content)).read("manifest.json"))
    manifest.update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        Project.from_bytes(rewrite(content, {"manifest.json": json.dumps(manifest).encode()}))


def test_a_project_too_large_to_unpack_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(project_module, "MAX_UNPACKED_BYTES", 10)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", "{}" * 20)
    with pytest.raises(ValueError, match="too large"):
        Project.from_bytes(buffer.getvalue())


def test_a_source_name_with_a_path_is_stored_under_its_base_name() -> None:
    content = b"a,b\n1,2\n3,4\n"
    project = Project(recipe=Recipe(SourceFile.of("../../elsewhere/data.csv", content)), source=content)
    names = zipfile.ZipFile(io.BytesIO(project.to_bytes())).namelist()
    assert "source/data.csv" in names
    assert not any(".." in name for name in names)


@pytest.mark.dataset
def test_a_stored_array_that_is_missing_counts_as_not_reproduced() -> None:
    content = make_project().to_bytes()
    source = zipfile.ZipFile(io.BytesIO(content))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for info in source.infolist():
            if info.filename != "fits/0/T2.npy":
                target.writestr(info.filename, source.read(info.filename))
    opened = Project.open(out.getvalue())
    assert [(c.variant, c.array) for c in opened.checks if not c.ok] == [("Without the upset", "T2")]


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("replace", "drop", "message"),
    [
        ({"manifest.json": b"{not json"}, None, "not valid JSON"),
        ({}, "source/LDPE.csv", "has no source/LDPE.csv"),
    ],
)
def test_a_damaged_project_file_says_what_is_wrong(replace, drop, message) -> None:
    content = make_project().to_bytes()
    source = zipfile.ZipFile(io.BytesIO(content))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for info in source.infolist():
            if info.filename != drop:
                target.writestr(info.filename, replace.get(info.filename, source.read(info.filename)))
    with pytest.raises(ValueError, match=message):
        Project.from_bytes(out.getvalue())
