# Cross-check of designs against other libraries

`compare.py` builds the designs of the reference catalog with another library and compares
every printed number that depends on how a design is built. The catalog's own checker
measures both designs, so only the construction differs. The results are reported in
`docs/validation.rst`.

## Running it

From the repository root, with the library named in `DOE_LIB`:

```bash
DOE_LIB=R DOE_OUT=results uv run python scripts/design_crosscheck/compare.py all
```

The argument is `all`, or a seed for a random sample of 20 entries (a quick trial).
`DOE_OUT` is the folder for the two files it writes: `rows_<library>_all.csv`, one row per
check, and `differences_<library>_all.csv`, the rows that do not agree. A run takes 5 to 40
minutes.

A check that takes more than 60 seconds in either library is recorded as a timeout, and counts
as not built. `DOE_TIMEOUT` sets another limit, in seconds, and `DOE_ENTRIES` names the
catalogue entries to run, separated by commas, to give the slow ones a second look. The rows of
`docs/validation.rst` gave each check that timed out a second run of 180 seconds:

```bash
DOE_TIMEOUT=180 DOE_ENTRIES=montgomery-table-x-2-10-5 DOE_LIB=pydoe DOE_OUT=again \
    uv run --with pydoe python scripts/design_crosscheck/compare.py all
```

pydoe's `fracfact_opt` scores every generator set, which takes up to 4.9 hours for 11 factors in
128 runs. A run searches each size once, and the checks of that size share it. The 14 checks of
10 and 11 factors that pass 180 seconds come from five entries, run with a limit of 6 hours
(about nine hours in all):

```bash
DOE_TIMEOUT=21600 DOE_LIB=pydoe DOE_OUT=again \
DOE_ENTRIES=box-draper-5-table-5-15-fractional-designs,fries-hunter-table-2-maximum-resolution,\
montgomery-2-k-p-selection,montgomery-table-x-2-10-4,montgomery-table-x-2-11-6 \
    uv run --with pydoe python scripts/design_crosscheck/compare.py all
```

## What each library needs

| `DOE_LIB` | Builds with | Environment |
| --- | --- | --- |
| `pydoe` | pydoe | `uv run --with pydoe ...` |
| `pyDOE3` | pyDOE3 1.6.2, which pydoe continues | In the development environment already; `tabulate.py` leaves it out |
| `R` | FrF2, DoE.base, rsm, daewr | R with these packages and jsonlite; `R_LIBS_USER` if they are in a library of their own |
| `dexpy` | dexpy | An interpreter with dexpy and pandas 1.x, named in `DEXPY_PYTHON` |
| `oapackage` | OApackage | `uv run --with oapackage ...` |
| `dsdpkg` | definitive-screening-design | `uv run --with definitive-screening-design ...` |

`r_worker.R` and `dexpy_worker.py` are started by `compare.py`. `oa_builders.py` searches for
a conference design or an orthogonal array with OApackage.

## Reading a row

| `outcome` | Meaning |
| --- | --- |
| `agree` | Both libraries give the printed number |
| `other differs` | process-improve gives the printed number; the other library gives another |
| `other cannot build` | The other library has no such design, or declines the request |
| `process-improve differs` | The other library gives the printed number; process-improve does not |
| `neither gives the printed number` | Neither does; the catalog records the cause, most often a misprint |

A row that differs is a starting point, not a verdict. Some come from how a request is mapped
to the other library (for example which center runs go to which block), so each group of
differences is examined before it is reported.
