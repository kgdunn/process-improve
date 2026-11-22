"""Turn the rows that compare.py wrote into the table and the downloads of docs/validation.rst.

Run it from the repository root after compare.py, with the same ``DOE_OUT``::

    DOE_OUT=results uv run python scripts/design_crosscheck/tabulate.py

It writes ``docs/validation/design_crosscheck.csv``, one row per package, and
``docs/validation/differences_<package>.csv``, every check on which that package and process-improve differ.
"""

import os
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("DOE_OUT", "."))
DOCS = Path(__file__).resolve().parents[2] / "docs" / "validation"

#: The packages in the order of the page: the name compare.py records in ``built_by``, then what the page prints.
PACKAGES = {
    "FrF2": ("FrF2 2.3-5", "R"),
    "DoE.base": ("DoE.base 1.2-5", "R"),
    "rsm": ("rsm 2.10.6", "R"),
    "daewr": ("daewr 1.2-11", "R"),
    "dexpy": ("dexpy 0.12", "Python"),
    "oapackage": ("OApackage 2.7.21", "Python, C++"),
    "dsdpkg": ("definitive-screening-design 0.5.1", "Python"),
    "pydoe": ("pydoe 1.6.0", "Python"),
    "pyDOE3": ("pyDOE3 1.6.2", "Python"),
}
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
    for name, (label, language) in PACKAGES.items():
        mine = rows[rows["built_by"] == name]
        mine = mine[
            ~mine["other"].str.contains("harness does not map", na=False)
        ]  # a limit of compare.py, not the package
        # A package is compared on the kinds of design it builds at least once; a misprint both libraries
        # reproduce the same way is the catalogue's subject, not this comparison's.
        builds = mine.loc[mine["outcome"] != "other cannot build", "design_type"].unique()
        mine = mine[mine["design_type"].isin(builds) & (mine["outcome"] != "neither gives the printed number")]
        count = mine["outcome"].value_counts()
        table.append(
            {
                "Package": label,
                "Language": language,
                "Designs compared": ", ".join(text for kind, text in DESIGNS.items() if kind in builds),
                "Checks": len(mine),
                "Agree": int(count.get("agree", 0)),
                "Differ": int(count.get("other differs", 0) + count.get("process-improve differs", 0)),
                "Not built": int(count.get("other cannot build", 0)),
            }
        )
        differs = mine[mine["outcome"].isin(["other differs", "process-improve differs"])]
        if len(differs):
            differs.drop(columns=["design_type", "built_by"]).to_csv(DOCS / f"differences_{name}.csv", index=False)
    pd.DataFrame(table).to_csv(DOCS / "design_crosscheck.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
