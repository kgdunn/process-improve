# (c) Kevin Dunn, 2010-2026. MIT License.
"""The project file: a source file, its recipe, its model variants and saved selections, in one zip.

=========================  ====================================================================
Member                     Holds
=========================  ====================================================================
``manifest.json``          the format and library versions, the recipe, the variants, the
                           saved selections
``source/<file name>``     the source file's own bytes, exactly as read
``recipe.py``              the recipe as a script, to run next to the source file
``fits/<k>/<array>.npy``   variant ``k``'s fitted scores, loadings, T² and SPE
=========================  ====================================================================

Opening a project executes nothing from it: the manifest is JSON, checked field by
field; the source is read as the recipe says (CSV or Excel); the arrays are read with
``numpy.load(allow_pickle=False)``. ``recipe.py`` is written for the reader and never
read back. :meth:`Project.open` refits every variant from the source and the recipe and
compares the result with the stored arrays, so a project that no longer reproduces,
because the library or the file changed, says so instead of being trusted.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from process_improve.multivariate import PCA

from .recipe import Recipe, RowId, _is_row_id, _release
from .variant import Variant

if TYPE_CHECKING:
    import pandas as pd

    from .variant import Fit

#: What ``manifest.json`` calls this format, and the version of it this module writes.
FORMAT, FORMAT_VERSION = "process-improve-project", 1

#: The largest project this module opens, unpacked, and the most members it reads.
MAX_UNPACKED_BYTES, MAX_MEMBERS = 512 * 2**20, 2_000

#: The arrays stored for each variant, by the name of the member that holds them.
ARRAYS = ("scores", "loadings", "T2", "SPE")


@dataclass(frozen=True)
class Check:
    """How one stored array compares with the same array refitted on opening."""

    variant: str
    array: str
    max_difference: float
    ok: bool


@dataclass(frozen=True)
class Opened:
    """A project opened: its table, every variant fitted, and how each fit compares with the stored one."""

    project: Project
    table: pd.DataFrame
    fits: list[Fit]
    checks: list[Check]
    library: str

    @property
    def reproduces(self) -> bool:
        """Whether every refitted array matches the one stored."""
        return all(check.ok for check in self.checks)


@dataclass(frozen=True)
class Project:
    """A source file's content, its recipe, the model variants over it, and the reader's saved selections.

    Parameters
    ----------
    recipe : Recipe
    source : bytes
        The source file's content. Its SHA-256 digest must match the recipe's, when the
        recipe records one.
    variants : tuple of Variant
    selections : dict
        Named sets of row ids, as the page saves them (a lasso kept under a name).
    """

    recipe: Recipe
    source: bytes
    variants: tuple[Variant, ...] = ()
    selections: dict[str, list[RowId]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a source whose digest is not the recipe's, and selections that are not named lists of row ids."""
        expected = self.recipe.source.sha256
        if expected is not None and _digest(self.source) != expected:
            raise ValueError(f"The source is not the file the recipe was written for ({self.recipe.source.name}).")
        for name, rows in self.selections.items():
            if not isinstance(name, str) or not isinstance(rows, list) or not all(_is_row_id(r) for r in rows):
                raise TypeError(f"A selection is a name and a list of row ids, got {name!r}: {rows!r}.")

    def table(self) -> pd.DataFrame:
        """Return the recipe's table, read from the source and refined by every step."""
        return self.recipe.run(self.source)

    def to_bytes(self) -> bytes:
        """Return the project as a zip file, each variant fitted so its arrays are stored with it."""
        table = self.table()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(self._manifest(), indent=2))
            archive.writestr(_source_member(self.recipe), self.source)
            archive.writestr("recipe.py", self.recipe.to_python())
            for k, variant in enumerate(self.variants):
                for name, values in _arrays(variant.fit(table)).items():
                    member = io.BytesIO()
                    np.save(member, values, allow_pickle=False)
                    archive.writestr(f"fits/{k}/{name}.npy", member.getvalue())
        return buffer.getvalue()

    def save(self, path: str | Path) -> None:
        """Write the project to ``path``."""
        Path(path).write_bytes(self.to_bytes())

    def _manifest(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "library": _release(),
            "recipe": self.recipe.to_dict(),
            "source": _source_member(self.recipe),
            "variants": [v.to_dict() for v in self.variants],
            "selections": self.selections,
        }

    @classmethod
    def from_bytes(cls, content: bytes) -> Project:
        """Read a project from its zip file's content, checking every field; nothing in it is executed."""
        return _read(content)[0]

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """Read a project from ``path``."""
        return cls.from_bytes(Path(path).read_bytes())

    @staticmethod
    def open(content: bytes, *, rtol: float = 1e-9) -> Opened:
        """Read a project, refit each variant, and compare every refitted array with the stored one.

        Parameters
        ----------
        content : bytes
            The project file's content.
        rtol : float, default 1e-9
            The largest difference, relative to the array's largest magnitude, that still
            counts as the same.
        """
        project, stored, library = _read(content)
        table = project.table()
        fits = [variant.fit(table) for variant in project.variants]
        checks = []
        for variant, fit, arrays in zip(project.variants, fits, stored, strict=True):
            for name, refitted in _arrays(fit).items():
                saved = arrays.get(name)
                if saved is None or saved.shape != refitted.shape:
                    checks.append(Check(variant.name, name, float("inf"), ok=False))
                    continue
                scale = max(float(np.nanmax(np.abs(saved), initial=0.0)), np.finfo(float).tiny)
                difference = float(np.nanmax(np.abs(saved - refitted), initial=0.0)) / scale
                same_gaps = bool(np.array_equal(np.isnan(saved), np.isnan(refitted)))
                checks.append(Check(variant.name, name, difference, ok=same_gaps and difference <= rtol))
        return Opened(project=project, table=table, fits=fits, checks=checks, library=library)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_member(recipe: Recipe) -> str:
    # Only the base name: a recipe from a file may carry a path, and the member name is ours.
    name = recipe.source.name.replace("\\", "/").rsplit("/", 1)[-1] or "source"
    return f"source/{name}"


def _arrays(fit: Fit) -> dict[str, np.ndarray]:
    rows = fit.rows()
    a = fit.variant.n_components
    return {
        "scores": rows.iloc[:, :a].to_numpy(dtype=float),
        "loadings": (fit.model.loadings_ if isinstance(fit.model, PCA) else fit.model.x_loadings_).to_numpy(),
        "T2": rows["T2"].to_numpy(dtype=float),
        "SPE": rows["SPE"].to_numpy(dtype=float),
    }


def _read(content: bytes) -> tuple[Project, list[dict[str, np.ndarray]], str]:
    """Return the project, each variant's stored arrays, and the library version that wrote it."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise ValueError("This is not a project file: it is not a zip archive.") from None
    with archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
            raise ValueError("This project file is too large to open.")
        names = {m.filename for m in members}
        if "manifest.json" not in names:
            raise ValueError("This is not a project file: it has no manifest.json.")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"The project's manifest.json is not valid JSON: {error}.") from None
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ValueError("This is not a project file: its manifest names another format.")
        if not isinstance(manifest.get("version"), int) or manifest["version"] > FORMAT_VERSION:
            raise ValueError("This project file was written by a newer process-improve; update to open it.")
        recipe = Recipe.from_dict(manifest.get("recipe", {}))
        member = _source_member(recipe)
        if member not in names:
            raise ValueError(f"The project has no {member}.")
        variants_raw = manifest.get("variants", [])
        selections = manifest.get("selections", {})
        if not isinstance(variants_raw, list) or not isinstance(selections, dict):
            raise TypeError("The manifest's 'variants' must be a list and its 'selections' a mapping.")
        project = Project(
            recipe=recipe,
            source=archive.read(member),
            variants=tuple(Variant.from_dict(v) for v in variants_raw),
            selections=selections,
        )
        stored = [
            {
                name: np.load(io.BytesIO(archive.read(f"fits/{k}/{name}.npy")), allow_pickle=False)
                for name in ARRAYS
                if f"fits/{k}/{name}.npy" in names
            }
            for k in range(len(project.variants))
        ]
    library = manifest.get("library")
    return project, stored, library if isinstance(library, str) else "unknown"
