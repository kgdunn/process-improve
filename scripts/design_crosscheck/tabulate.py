"""Turn the rows that compare.py wrote into the table and the downloads of docs/validation.rst.

Run it from the repository root after compare.py, with the same ``DOE_OUT``::

    DOE_OUT=results uv run python scripts/design_crosscheck/tabulate.py

It writes ``docs/validation/design_crosscheck.csv``, one row per package,
``docs/validation/differences_<package>.csv``, every check on which that package and process-improve differ, and
``web/checked.json``, which the browser app reads: for each kind of design and each number of factors, the
sources that print it and the packages that build it as process-improve does.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("DOE_OUT", "."))
DOCS = Path(__file__).resolve().parents[2] / "docs" / "validation"

#: The packages in the order of the page: the name compare.py records in ``built_by``, then what the page prints.
PACKAGES = {
    "FrF2": "FrF2 2.3-5 (R)",
    "DoE.base": "DoE.base 1.2-5 (R)",
    "rsm": "rsm 2.10.6 (R)",
    "daewr": "daewr 1.2-11 (R)",
    "dexpy": "dexpy 0.12 (Python)",
    "oapackage": "OApackage 2.7.21 (Python)",
    "dsdpkg": "definitive-screening-design 0.5.1 (Python)",
    "pydoe": "pydoe 1.7.0 (Python)",
}
WEB = Path(__file__).resolve().parents[2] / "web"

#: A package as the app names it, and its language. Each is reported at its latest release; pyDOE3, which pydoe
#: continues, has no row.
SHOWN = {
    "FrF2": ("FrF2", "R"),
    "DoE.base": ("DoE.base", "R"),
    "rsm": ("rsm", "R"),
    "daewr": ("daewr", "R"),
    "dexpy": ("dexpy", "Python"),
    "oapackage": ("OApackage", "Python"),
    "dsdpkg": ("definitive-screening-design", "Python"),
    "pydoe": ("pydoe", "Python"),
}
DIFFERS = ["other differs", "process-improve differs"]
DESIGNS = {
    "full_factorial": "full factorial",
    "fractional_factorial": "fractional factorial",
    "plackett_burman": "Plackett-Burman",
    "box_behnken": "Box-Behnken",
    "ccd": "central composite",
    "dsd": "definitive screening",
    "taguchi": "orthogonal array",
}


def main() -> None:
    rows = pd.concat([pd.read_csv(path) for path in sorted(OUT.glob("rows_*_all.csv"))], ignore_index=True)
    rows["design_type"] = rows["design"].str.split(" ").str[0]
    DOCS.mkdir(parents=True, exist_ok=True)
    table = []
    for name, label in PACKAGES.items():
        mine = rows[rows["built_by"] == name]
        mine = mine[
            ~mine["other"].str.contains("harness does not map", na=False)
        ]  # a limit of compare.py, not the package
        # A package is compared on the kinds of design it builds at least once. A number that neither library
        # gives as printed is the catalog's subject, not this comparison's.
        built = mine["outcome"].isin(["agree", "other differs", "process-improve differs"])
        builds = mine.loc[built, "design_type"].unique()
        mine = mine[mine["design_type"].isin(builds) & (mine["outcome"] != "neither gives the printed number")]
        count = mine["outcome"].value_counts()
        table.append(
            {
                "Package": label,
                "Designs compared": ", ".join(text for kind, text in DESIGNS.items() if kind in builds),
                "Checks": f"{len(mine):,}",
                "Agree": f"{count.get('agree', 0):,}",
                "Differ": f"{count.get('other differs', 0) + count.get('process-improve differs', 0):,}",
                "Not built": f"{count.get('other cannot build', 0):,}",
            }
        )
        differs = mine[mine["outcome"].isin(["other differs", "process-improve differs"])]
        if len(differs):
            differs.drop(columns=["design_type", "built_by"]).to_csv(DOCS / f"differences_{name}.csv", index=False)
    pd.DataFrame(table).to_csv(DOCS / "design_crosscheck.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))
    write_checked(rows)


def _agreeing(rows: pd.DataFrame) -> dict[str, list[str]]:
    """Return the sources and the packages of the checks that agree, each in the order of its first check."""
    agree = rows[rows["outcome"] == "agree"]
    packages = [SHOWN[name][0] for name in SHOWN if name in set(agree["built_by"])]
    return {"sources": sorted(agree["source"].unique()), "packages": packages}


def write_checked(rows: pd.DataFrame) -> None:
    """Write ``web/checked.json``: what each kind of design, at each number of factors, was checked against.

    A kind of design lists every package that builds it as process-improve does at one size or more. A size
    lists a package only when no check of that size differs, so a design's own note in the app names no package
    that builds that design another way.
    """
    rows = rows[rows["built_by"].isin(set(SHOWN)) & rows["design_type"].isin(DESIGNS)]
    rows = rows[~rows["other"].str.contains("harness does not map", na=False)]
    rows = rows.assign(factors=rows["design"].str.extract(r"k=(\d+)$", expand=False))
    designs = {}
    for kind in DESIGNS:
        mine = rows[rows["design_type"] == kind]
        sizes = {}
        for factors, size in mine.groupby("factors"):
            differs = size.groupby("built_by")["outcome"].agg(lambda outcomes: outcomes.isin(DIFFERS).any())
            found = _agreeing(size[~size["built_by"].map(differs)])
            if found["packages"]:
                sizes[factors] = found
        designs[kind] = {**_agreeing(mine), "factors": dict(sorted(sizes.items(), key=lambda item: int(item[0])))}
    newest = max(path.stat().st_mtime for path in OUT.glob("rows_*_all.csv"))
    checked = {
        "checked": datetime.fromtimestamp(newest, tz=UTC).date().isoformat(),
        "packages": dict(SHOWN.values()),
        "designs": designs,
    }
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "checked.json").write_text(json.dumps(checked, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
