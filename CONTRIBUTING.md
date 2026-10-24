# Contributing to process-improve

Bug reports, feature requests, and pull requests are welcome. This guide
covers how to set up a development environment and the conventions the
project follows.

By participating in this project you agree to abide by its
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security vulnerability,
follow the [security policy](SECURITY.md) instead of opening a public issue.

## Development setup

Clone the repository and install it in editable mode with the development
extras:

```bash
git clone https://github.com/kgdunn/process-improve.git
cd process-improve
pip install -e ".[dev,all]"
```

`dev` brings in the test runner, linter, formatter, type checker, and the
documentation toolchain. `all` brings in the optional runtime dependencies
(`plotly`, `numba`, `ruptures`, ...) that the gated public-API surfaces
need. Install both: with `dev` alone a large part of the suite fails on
`ImportError` rather than skipping, because those tests exercise the extras.
CI installs the equivalent with `uv sync --dev --all-extras`.

## Running tests

The full suite:

```bash
pytest -o "addopts="
```

A focused subset, for example the multivariate tests:

```bash
pytest tests/test_multivariate.py -v -o "addopts="
```

Only the tests your branch can affect, judged from a static import graph of
`src/` and `tests/` (see `tests/_selection.py`):

```bash
pytest --no-cov --affected-since=origin/main
```

It counts committed, staged, unstaged and untracked changes since the merge
base. A test reaches a changed file through `import` statements, module names
in string literals, and Python snippets held in strings; a changed data file or
document affects the code that names it. A change to `pyproject.toml`,
`pytest.ini` or `.coveragerc`, or a deleted file, runs everything. It is a tool
for fast local feedback: the analysis is static and cannot see every import-time
side effect, so CI does not use it. CI runs the whole suite on every platform
and Python it tests, split into duration-balanced shards: two for each `test`
and `test-free-threaded` row, four for the `coverage` job. The 3.12 `test` row
runs under `python -O` (`PYTHONOPTIMIZE=1`), to catch validation that an
`assert` would lose:

```bash
pytest --shard=2/4                     # the second of four shards
pytest --no-cov --store-durations      # refresh tests/.test_durations.json
```

The shards of a job together are exactly the full suite. A stale durations file
only unbalances them. The `coverage-report` job warns when it is stale, and
uploads a fresh one, measured by the coverage shards, as the `test-durations`
artifact: refreshing the file is a download of that artifact from a recent run
on `main`.

Tests use real datasets (LDPE, SIMCA) alongside synthetic data; please keep
the real-dataset tests in place when adding new ones. New methods should have
tests for both basic functionality and edge cases.

### Test tiers

Tests are categorised via pytest markers so contributors can skip the ones
that are slow or network-dependent during local iteration. The full suite
runs in CI; the markers exist so a contributor doesn't pay for
multi-second or network-bound tests on every save.

| Marker | Meaning | Example | When to add |
|--------|---------|---------|-------------|
| (none) / `unit` | Fast in-process unit test. The implicit default; no decorator needed. | `test_check_random_state_resolves_int` | Default. |
| `integration` | Crosses module boundaries or exercises an external library's surface (statsmodels, plotly). | `test_pca_round_trip_via_sklearn_pipeline` | Add when the test wires several modules together. |
| `slow` | Multi-second wall-clock time, even on a fast laptop. | `test_tpls_full_fit_on_simca_dataset` | Add when the test takes >= 2 s. |
| `dataset` | Loads a bundled or remote real-world dataset (network or large file). | `test_oildoe_loads` | Add when the test depends on a real-data fixture, especially remote ones. |

Common opt-in / opt-out invocations:

```bash
# Skip network-dependent dataset tests (useful when offline):
pytest -m "not dataset" -o "addopts="

# Skip slow tests during local iteration:
pytest -m "not slow" -o "addopts="

# Run only the network-dependent dataset tests:
pytest -m dataset -o "addopts="

# CI runs the full suite (no marker filter).
```

The markers are registered in `pytest.ini`, and `addopts` includes
`--strict-markers`, so a typo'd marker name is a collection error rather
than a silent warning. Tag a test by adding the decorator above the
function (or above the `class` to tag the whole class):

```python
import pytest


@pytest.mark.dataset
def test_loads_real_data(): ...
```

### The README is tested

The README is the first code a new user runs, so CI checks it like code:

- `tests/test_readme.py` runs every Python block of `README.md` and
  `SKLEARN_COMPATIBILITY.md` in order, and compares each `print(...)  # output`
  comment with what the line prints. When a change moves a number, update the
  comment in the same pull request.
- `tests/test_readme_guards.py` checks the rest of the page, offline: every call
  quoted in the text or a table names a real function, method and keyword; every
  link into the repository or the documentation resolves; and the installation
  lines match `pyproject.toml`, so a new extra has to be documented.
- `tools/check_readme_links.py` requests every link to another site, and checks
  that the openmv.net data files still match the bundled copies behind the
  README's numbers. `.github/workflows/readme-links.yml` runs it weekly and on
  pull requests that touch the README, and opens an issue when a weekly run fails.

### The browser app is tested

The no-install app in `web/` runs the package in the reader's browser, through
`web/bootstrap.py`. Its tests are in `tests/web/` and run with the rest of the suite:

- `tests/web/api/` drives `bootstrap.py` through its JSON entry points, as the page
  does: a module per entry point, plus `test_errors.py` (every refusal the page can
  show), `test_envelope.py` (the shape of every reply) and `test_contract.py` (every
  key, and its type, that `app.js`, `charts.js` and `python.js` read from a reply).
- `tests/web/page/` checks the page without a browser: each control and rule the HTML
  and JavaScript repeat from `bootstrap.py`, the build in `scripts/build_web_app.py`,
  and the code the "Show the Python" panels write. The tests that run the page's
  JavaScript need Node.js, and skip without it.
- `tests/web/js/` unit-tests the page's JavaScript with Node.js's own test runner
  (`node --test`, Node.js 22.7 or later): `web/logic.js`, which holds what the page
  counts, decides, formats and sorts apart from the document, and the figures
  `web/charts.js` builds.

`pytest tests/web --no-cov` runs them, and `--affected-since` selects all of them when any
file of the app changes. `web/bootstrap.py` counts towards coverage, so the patch gate
applies to it.

Three slower tiers run only when asked for, since they need a browser, Pyodide, or both.
CI's `web-app` job runs all three:

- `tests/web/e2e` (`--e2e`): the page in Chromium, through Playwright, with its Python
  answered natively by `bootstrap.py` in the test process. It needs neither Pyodide nor
  the network, and each test gets a fresh page; a console error fails the test.
- `tests/web/live` (`--live`): the built app in Chromium, with Pyodide in its worker. Its
  `smoke` tests also run in `docs.yml` against the built `/app`, so an app that does not
  work is not deployed; again once the deploy is served (`tools/wait_for_web_app.py` waits
  for the published manifest to name the commit); and weekly against the published app, in
  `web-app-live.yml`, which also reports a pin in `web/runtime.json` that is no longer the
  latest release (`tools/check_web_runtime.py`). When npm has a newer Pyodide, that workflow
  runs the wasm and live tiers on it and says in its summary whether moving the pin is safe.
- `tests/web/wasm` (`--wasm`): the app's Python under Pyodide in Node.js, asked what
  native Python is asked; the answers must agree.

To run them locally:

```bash
uv run --with playwright==1.64.1 playwright install chromium     # once, for --e2e and --live
npm install --prefix ~/.cache/process-improve/node --no-save \
  pyodide@$(jq -r .pyodide web/runtime.json)                      # once, for --wasm
uv run --with playwright==1.64.1 pytest tests/web --e2e --live --wasm --no-cov
```

`--live` and `--wasm` build the app into `~/.cache/process-improve/web-app`, unless
`--web-app` names a build directory or a URL (the published app, for instance). Behind a
proxy, set `HTTPS_PROXY`: Chromium is pointed at it. `--e2e-headed` shows the browser, and
`--e2e-artifacts DIR` keeps a trace and screenshots of each failing browser test.

When you change the app, the tests move with it:

- A new entry point, or a new key the page reads from a reply: a test in
  `tests/web/api/`, and the key in `READS` in `test_contract.py`.
- A new refusal (a `raise ValueError(...)` in `bootstrap.py`): a case in `REFUSALS` in
  `test_errors.py`. The suite fails, naming the message, until there is one.
- A choice the page repeats from `bootstrap.py`, so the form is right before Python
  loads: a test in `tests/web/page/test_parity.py`.
- A helper that needs no document (a rule, a format, a sentence): in `web/logic.js`, with a
  test in `tests/web/js/logic.test.mjs`.
- A new control, step or message on the page: a test in `tests/web/e2e`, which drives the
  page as a reader would. `window.__pi` (`tests/web/harness.js`) records each call to
  Python, and can hold a reply back or give one, for races and error paths. The page shows
  one phase at a time (Design, Run or Learn): a test shows the phase that holds what it types
  into or clicks first, with `reveal` in `tests/web/e2e/_page.py`, as a reader presses its tab.
- A new package the page loads into Pyodide: in `web/runtime.json` only, which
  `web/worker.js`, the wasm tier and CI read. One from PyPI is pinned (`name==version`)
  to the version the wasm tier installs; `tests/web/page/test_runtime.py` checks both.
- A new file in `web/`: in `APP_FILES` in `scripts/build_web_app.py` if the page loads
  it, and in `APP_SOURCES` in `tests/web/_support.py` either way.
- A new example the book links to (`https://factori.al/?book=<name>`): an entry in
  `web/book-examples.json`, and a case in `tests/web/api/test_book_examples.py` holding
  the numbers the book prints for it; the suite fails until there is one. Its name is
  printed in the book, so keep it once published: the book checks every name it uses
  against `book/index.json` on factori.al.

One build is served in two places: https://factori.al/, the app's address, and GitHub
Pages at `/app`. On each push to `main`, `docs.yml` publishes the build the smoke tests
passed to both (`deploy-factorial` sends it to factori.al's server over SSH) and runs the
smoke tests against each once it is served. [`deploy/factorial/README.md`](deploy/factorial/README.md)
covers the server: its one-time setup, what it logs, and how to roll back.
- A new design type or simulated process: in `ALL_DESIGNS` or `ALL_SCENARIOS` in
  `tests/web/_support.py` (a mixture design in `ALL_MIXTURES`), so the tests that run over every one
  include it.

## Linting and formatting

```bash
ruff check .                # lint
ruff format .               # format
ruff format --check .       # what CI runs: fails instead of rewriting
mypy src/process_improve    # type check
```

The line length is 120 characters.

Ruff covers linting, formatting, and import sorting; do not add black, flake8,
or isort configuration.

The lint job runs `ruff check .` **and** `ruff format --check .` as two
independent gates, so a clean `ruff check .` says nothing about formatting. Run
both before pushing, or install the pre-commit hooks, which run the same pair.
Keep the `ruff-pre-commit` `rev` in `.pre-commit-config.yaml` in step with the
`ruff` pin in `pyproject.toml`; if they diverge, code formatted locally can
still be rejected by CI. `tests/test_tooling_versions.py` enforces this for ruff
and mypy, so an update of either hook fails until its pin moves with it.

[pre-commit.ci](https://pre-commit.ci) runs the same hooks on every pull request
and pushes the fixes it can make, so a contributor without the hooks installed
still lands formatted code. Its quarterly autoupdate PRs bump the hook
revisions; when one moves ruff or mypy past its pin, finish that PR by moving
the pin in `pyproject.toml` too.

## Code style

- **Docstrings:** NumPy style throughout (Parameters, Returns, Examples,
  See Also, References), with type annotations in both signatures and
  docstrings.
- **Fitted attributes:** follow the sklearn convention of a trailing
  underscore (`scores_`, `loadings_`, `spe_`, ...).
- **Scaling:** prefer `MCUVScaler` (mean-center, unit-variance) for preparing
  data before fitting models.
- **Prose:** do not use em-dashes in code, comments, docstrings, or commit
  messages; use a hyphen, a semicolon, or split the sentence.

### Complexity budget

The four complexity rules (`C901` too-complex, `PLR0912` too-many-branches,
`PLR0913` too-many-arguments, `PLR0915` too-many-statements) are silenced
in-line at many sites. They are a real signal: the functions carrying them are
the ones no single reader can hold in their head, and they are where the bugs
turn up (#307).

The count is a **ratchet**, not a style preference:

```bash
python tools/complexity_budget.py        # counts, budget, and what to split next
```

The script asks ruff with `--ignore-noqa`, so it counts functions that genuinely
breach a threshold rather than `# noqa` comments, and it ranks them by how far
past the threshold they are. `tests/test_complexity_ratchet.py` turns that into
a CI gate, in both directions:

- A count **above** its budget fails: the change made the code worse in exactly
  the way #307 is about. Split the function, or make the case in #307 for
  raising the budget.
- A count **below** its budget also fails, with the one-line fix in the message:
  lower `BUDGET` in `tools/complexity_budget.py`. That is what stops a refactor
  being silently spent by the next change.

**Target: halve the 2026-06 baseline of 185 breaches, to 91, by v2.0.** Roughly
one function split per release gets there; picking the top entry of the report
is a good default.

A new `# noqa` for one of these four rules needs a trailing comment saying why
the refactor is deferred, the same as any other deliberate suppression.

### API consistency

Functions and estimators that do the same job must present the same API.
Concretely, across a family of siblings (`PCA` / `PLS` / `TPLS` / `MBPLS` /
`MBPCA`, or `robust_regression` / `multiple_linear_regression`):

- **The same key or attribute name means the same quantity everywhere.** If
  one function returns `t_value`, every sibling that can compute it returns
  `t_value`, holding the same statistic. Reusing a name for a different
  quantity is worse than not providing it at all.
- **A key that is documented must be populated.** Do not document a return
  value the code never assigns; either compute it or remove it from the
  docstring. Where a value is genuinely undefined (a degenerate fit, for
  example), return `NaN` and say so in the docstring.
- **Sibling methods return the same shape of thing.** A method that returns a
  `DataFrame` on one estimator should not return a `Series` on its sibling.
- **A convention chosen once applies everywhere.** Degrees of freedom,
  permutation-test corrections, centred versus uncentred moments, and similar
  choices should not vary between neighbouring functions. If two conventions
  are genuinely both needed, name the difference in both docstrings.

When a fix makes one member of a family better, check the others in the same
change; a divergence introduced quietly is expensive to find later.

### Naming conventions

- **`pi_` prefix on attributes** stands for **"process-improve"** and marks
  library-managed metadata on `Expt` (and related) objects: `pi_title`,
  `pi_source`, `pi_units`, `pi_range`, `pi_lo`, `pi_hi`, `pi_center`,
  `pi_name`. The prefix exists to keep these reserved names from colliding
  with column names from a user-supplied DataFrame. Treat `pi_*` attributes
  as part of the public API surface; new metadata fields should use the
  same prefix.
- **`Expt`** is the canonical (and only) name for the experiment container
  class, an abbreviation of "Experiment". It is a `pd.DataFrame` subclass
  carrying the `pi_*` metadata above. The abbreviation predates the rest of
  the package's spell-out-everything convention; it is retained for backwards
  compatibility.

## Adding new methods to PCA / PLS

1. Add the method to the class. The implementation is split across focused
   submodules: PCA is in `src/process_improve/multivariate/_pca.py` and PLS in
   `_pls.py`, with shared plotting in `plots.py`, limits in `_limits.py` and
   diagnostics in `_diagnostics.py`. `methods.py` is only a re-export shim that
   keeps every public name importable from one stable path, so adding a method
   there has no effect.
2. Use a NumPy-style docstring with Parameters, Returns, and Examples
   sections.
3. Add tests in `tests/test_multivariate.py`, or in the more specific
   `tests/test_multivariate_*.py` file, using both real datasets and
   synthetic data.
4. If the method belongs on both PCA and PLS, implement it on both with the
   same API signature.
5. Verify with `pytest tests/test_multivariate.py -v -o "addopts="`.

## Submitting changes

- Open a pull request against `main` with a clear description of the change
  and why it is needed.
- Do not bump the version. Add your changelog entry as a fragment,
  `changelog.d/<yyyy-mm-dd>-<slug>.<type>.md` (see [`changelog.d/README.md`](changelog.d/README.md)),
  and leave `CHANGELOG.md`, `pyproject.toml` and `CITATION.cff` alone; the
  version is set at release time (see "Versioning policy" below).
- Bugs and feature requests can also be filed on the
  [issue tracker](https://github.com/kgdunn/process-improve/issues).

## Versioning policy

The package uses three-part semantic versioning: `MAJOR.MINOR.PATCH`.

- **PATCH** (`1.22.8 -> 1.22.9`): bug fixes, CI / workflow changes,
  documentation updates, dependency bumps, small refactors. Public
  signatures are unchanged. Behaviour changes are only allowed when
  fixing a documented bug.
- **MINOR** (`1.22.9 -> 1.23.0`): new features, new modules,
  significant API additions, or meaningful behavioural changes.
  Backwards-compatible by default; deprecations follow the schedule
  in [`docs/development/deprecation_policy.rst`](docs/development/deprecation_policy.rst).
- **MAJOR** (`1.x.x -> 2.0.0`): incompatible removals. Anything
  previously deprecated under the policy can be removed here.

Reset rules:

- Bumping MINOR resets PATCH to 0 (`1.22.9 -> 1.23.0`).
- Bumping MAJOR resets MINOR and PATCH to 0 (`1.23.4 -> 2.0.0`).

### When the version is bumped

**Not in your pull request.** A pull request adds its entry as a fragment in
`changelog.d/` and leaves `CHANGELOG.md`, `pyproject.toml` and
`CITATION.cff` alone. The version is set once, at release time, from
everything that accumulated in the fragments and under `[Unreleased]`.

This is deliberate. Every pull request bumping the same line of
`pyproject.toml`, the same line of `CITATION.cff`, and inserting at the
same anchor in `CHANGELOG.md` makes every open pull request conflict with
every other one, on three lines that have nothing to do with the work.
The conflicts are not telling you anything, and resolving them costs a
rebase per merge. A fragment is a file of its own, so two pull requests'
entries never meet until the release files them.

### Choosing the level at release

The headings under `[Unreleased]` decide it, so the level is read off the
changelog rather than argued about. A fragment's type is the heading it is
filed under, so `changelog.d/2026-10-18-glm.added.md` counts as `### Added`:

| Highest heading present under `[Unreleased]` | Level |
|---|---|
| `### Removed` | MAJOR |
| `### Added` or `### Deprecated` | MINOR |
| anything else (`Changed`, `Fixed`, `Security`, `Documentation`) | PATCH |

`### Deprecated` sits at MINOR to match
[`docs/development/deprecation_policy.rst`](docs/development/deprecation_policy.rst),
which announces a deprecation in an `X.Y.0` release. Announcing one breaks
nothing, but it does add a warning the caller did not get before, and the
message has to name the version that announced it, so it needs a version of
its own to name. The removal it schedules is the MAJOR, and lands later
under `### Removed`.

This makes the level a consequence of how each entry was filed, which is
the decision the author already made when writing it. If an entry is
filed under the wrong heading the level comes out wrong, so file it where
it belongs; that is easier to review in a pull request than a version
number is.

### Cutting a release

1. Run `python tools/changelog.py build`. It files every fragment in
   `changelog.d/` under its heading in `## [Unreleased]` and deletes it.
2. Read the headings under `[Unreleased]` and pick the level from the
   table above.
3. In one commit, with the result of step 1: set `pyproject.toml` `version`,
   set `CITATION.cff` `version` to the identical value and `date-released`
   to today, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, open a
   fresh empty `## [Unreleased]` above it, and add the compare link to the
   footer.
4. Tag `vX.Y.Z` and push the tag.

`publish.yml` refuses to ship if the tag does not equal
`v<pyproject version>`, or if `CHANGELOG.md` has no `## [X.Y.Z]` heading,
so a release that skips step 2 fails loudly rather than shipping a wrong
version.

### What counts as a breaking change

A change is breaking if any of these are true:

- A documented public name (function, class, attribute, kwarg) is
  removed, renamed, or moved.
- A public function's signature changes in a way that fails
  existing valid call sites (a positional becomes keyword-only,
  a required kwarg is added without a default, a kwarg default
  changes meaning).
- A `Bunch` return type loses or renames a field.
- A `@tool_spec` tool is renamed, has a schema field removed, or
  changes the meaning of an existing field.
- A documented behaviour (e.g. "PCA returns sign-flipped loadings
  by the Wold convention") is silently changed.

A change is **not** breaking if:

- A new optional kwarg is added with a default that preserves the
  previous behaviour.
- A new public function or attribute is added.
- A `Bunch` return type gains a field.
- A documented bug is fixed.
- Internal (`_`-prefixed) code is refactored without affecting any
  public surface.

Breaking changes always require a deprecation cycle; see the
linked policy below.

## Performance-regression policy

Public-API algorithms (PCA fit, PLS fit/predict, ``analyze_experiment``,
``evaluate_design``, batch alignment) are expected not to regress in
wall-clock time. The rule of thumb:

- A PR that knowingly slows a hot path by more than ~10% must say
  so in the PR description and justify the trade-off (correctness
  fix, security guard, deprecation prep).
- The tests in `tests/perf/` are **not** wall-clock benchmarks. They
  assert deterministic cost-shape properties of the ENG-18 lazy-frame
  design (the public DataFrame views are built once and cached,
  pickling excludes the cache, `transform` / `predict` never touch the
  frame views) and of `check_random_state` (generator pass-through by
  identity). They fail on a real structural regression, such as losing
  a cache or re-routing a hot path through a DataFrame conversion, but
  a plain slowdown does not trip them.
- Wall-clock regression gating is the perf-baseline CI job, which is
  still **planned**
  ([ENG-15](https://github.com/kgdunn/process-improve/issues/297)).
  Until that job lands, the maintainer eyeballs perf on the hot
  paths during review.
- Internal refactors (private modules, internal classes) carry no
  perf SLA.

If a perf regression is unavoidable, the CHANGELOG entry says so
under "Changed".

## Policies referenced from this guide

These four documents pin the project's contributor-facing policies.
They are short, opinionated, and review feedback will reference
them directly:

- [Error-handling style](docs/development/error_handling.rst)
  ([ENG-11](https://github.com/kgdunn/process-improve/issues/293)) -- when to
  raise, when to assert, when to warn, when to use a dedicated
  exception class.
- [Reproducibility contract](docs/development/reproducibility.rst)
  ([ENG-08](https://github.com/kgdunn/process-improve/issues/290)) --
  ``random_state`` handling for every public function that
  touches an RNG.
- [Deprecation policy](docs/development/deprecation_policy.rst)
  ([ENG-22](https://github.com/kgdunn/process-improve/issues/304)) --
  the schedule for retiring public API, including the
  ``DeprecationWarning`` message format.
- This guide, the "Versioning policy" and "Performance-regression
  policy" sections above
  ([ENG-28](https://github.com/kgdunn/process-improve/issues/310)).
