"""Changelog fragments: every pull request's entry is a file of its own, filed into CHANGELOG.md at release."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "changelog.py"

_CHANGELOG = """# Changelog

## [Unreleased]

### Added

- **An earlier entry.** Filed before fragments existed.

### Fixed

- A fix already filed.

## [1.0.0] - 2026-01-01

### Added

- The first release.
"""


@pytest.fixture(scope="module")
def changelog():
    """Import `tools/changelog.py`, which is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("changelog_tool", _TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["changelog_tool"] = module
    spec.loader.exec_module(module)
    return module


def test_every_fragment_in_the_repository_is_well_formed(changelog):
    assert changelog.problems(changelog.FRAGMENTS) == []


@pytest.mark.parametrize(
    ("name", "text", "complaint"),
    [
        ("2026-10-18-glm.new.md", "- An entry.\n", "name it <yyyy-mm-dd>-<slug>.<type>.md"),
        ("2026-10-18-GLM.added.md", "- An entry.\n", "name it <yyyy-mm-dd>-<slug>.<type>.md"),
        ("glm.added.md", "- An entry.\n", "name it <yyyy-mm-dd>-<slug>.<type>.md"),
        ("18-10-2026-glm.added.md", "- An entry.\n", "name it <yyyy-mm-dd>-<slug>.<type>.md"),
        ("2026-02-30-glm.added.md", "- An entry.\n", "2026-02-30 is no date"),
        ("2026-10-18-glm.added.md", "An entry.\n", "starts with '- '"),
        ("2026-10-18-glm.added.md", "- An entry.\n\n### Added\n", "holds none"),
        ("2026-10-18-glm.added.md", "- An entry \u2014 with a dash.\n", "no em-dashes"),
    ],
)
def test_a_fragment_names_its_date_and_heading_and_holds_bullets_only(changelog, tmp_path, name, text, complaint):
    (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "README.md").write_text("# Not a fragment\n", encoding="utf-8")
    found = changelog.problems(tmp_path)
    assert len(found) == 1
    assert complaint in found[0]


def test_build_files_each_fragment_under_its_heading_and_deletes_it(changelog, tmp_path):
    """New bullets follow the ones already filed, oldest first; a heading not yet present opens in its order."""
    path = tmp_path / "CHANGELOG.md"
    path.write_text(_CHANGELOG, encoding="utf-8")
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    (folder / "README.md").write_text("# Kept\n", encoding="utf-8")
    # Written on different days: each heading files them in the order they were written, whatever their slugs.
    (folder / "2026-10-18-a-glm.added.md").write_text(
        "- **GLM.** Counts and proportions.\n  A second line.\n", encoding="utf-8"
    )
    (folder / "2026-10-17-z-lsmeans.added.md").write_text("- **Least-squares means.**\n", encoding="utf-8")
    (folder / "2026-10-18-pb.changed.md").write_text("- Plackett-Burman designs are cyclic.\n", encoding="utf-8")
    (folder / "2026-10-18-old.removed.md").write_text("- An old name.\n", encoding="utf-8")

    assert changelog.build(path, folder) == 4
    unreleased = path.read_text(encoding="utf-8").split("## [1.0.0]")[0]
    assert unreleased == (
        "# Changelog\n\n## [Unreleased]\n\n"
        "### Added\n\n- **An earlier entry.** Filed before fragments existed.\n"
        "- **Least-squares means.**\n- **GLM.** Counts and proportions.\n  A second line.\n\n"
        "### Changed\n\n- Plackett-Burman designs are cyclic.\n\n"
        "### Removed\n\n- An old name.\n\n"
        "### Fixed\n\n- A fix already filed.\n\n"
    )
    assert path.read_text(encoding="utf-8").endswith("## [1.0.0] - 2026-01-01\n\n### Added\n\n- The first release.\n")
    assert [p.name for p in folder.iterdir()] == ["README.md"]


def test_build_opens_the_first_heading_of_an_empty_unreleased_section(changelog, tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text("# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n", encoding="utf-8")
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    (folder / "2026-10-18-doc.documentation.md").write_text("- A page.\n", encoding="utf-8")
    assert changelog.build(path, folder) == 1
    assert path.read_text(encoding="utf-8") == (
        "# Changelog\n\n## [Unreleased]\n\n### Documentation\n\n- A page.\n\n## [1.0.0] - 2026-01-01\n"
    )


def test_build_without_fragments_leaves_the_changelog_as_it_is(changelog, tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(_CHANGELOG, encoding="utf-8")
    (tmp_path / "changelog.d").mkdir()
    assert changelog.build(path, tmp_path / "changelog.d") == 0
    assert path.read_text(encoding="utf-8") == _CHANGELOG


def test_the_command_line_refuses_to_build_from_a_malformed_fragment(changelog, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    (folder / "glm.added.md").write_text("- An entry.\n", encoding="utf-8")
    monkeypatch.setattr(changelog, "FRAGMENTS", folder)
    assert changelog.main(["changelog.py", "build"]) == 1
    assert "glm.added.md: name it <yyyy-mm-dd>-<slug>.<type>.md" in capsys.readouterr().out
    assert changelog.main(["changelog.py", "release"]) == 2
