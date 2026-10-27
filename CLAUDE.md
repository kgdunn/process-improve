# CLAUDE.md - Project Context for AI Agents

## Overview

`process-improve` is a Python package for process improvement using data. It accompanies the online textbook [Process Improvement using Data](https://learnche.org/pid). The package provides multivariate analysis (PCA, PLS, and TPLS - *PLS for T-shaped data structures*, not "Total PLS" or "Three-way PLS"), designed experiments, process monitoring, batch data analysis, and visualization tools.

**Repository:** https://github.com/kgdunn/process-improve
**License:** MIT
**Python:** >= 3.11 (CI tests 3.11-3.13, with 3.11 at the lowest dependency versions allowed; 3.13 is the primary version for lint, typecheck, and coverage)

## Authoritative documents

This file is the agent-facing map: the repo layout, the decisions that are not
obvious from reading the code, and the session workflow. It is **not** the
canonical home for contributor policy. Where a policy document exists, read it
rather than assuming; do not restate its content here, because two copies of a
rule is how they drift apart.

| Topic | Canonical source |
|---|---|
| Dev setup, test tiers, code style, breaking changes, perf policy | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| When to raise / assert / warn, exception classes | `docs/development/error_handling.rst` |
| `random_state` contract for every public function touching an RNG | `docs/development/reproducibility.rst` |
| Retiring public API, `DeprecationWarning` message format | `docs/development/deprecation_policy.rst` |
| Logging conventions | `docs/development/logging.rst` |
| Writing a new `@tool_spec` tool | `docs/development/tool_authoring.rst` |
| sklearn interoperability details | `SKLEARN_COMPATIBILITY.md` |
| Adding a case to the reference catalogue, and the `cause` each number records | [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-case-to-the-reference-catalogue) |
| The browser app in `web/`: its tests, and what a change to it adds to them | [`CONTRIBUTING.md`](CONTRIBUTING.md#the-browser-app-is-tested) |
| factori.al: how the app is deployed and rolled back, and the server's one-time setup | [`deploy/factorial/README.md`](deploy/factorial/README.md) |

## Package Structure

The package uses a src layout: all code lives under `src/process_improve/`.

```
src/process_improve/
    multivariate/    # PCA, PLS, TPLS, MBPLS, MBPCA, MCUVScaler, center, scale
    univariate/      # t_value, outlier_detection_multiple, confidence_interval
    experiments/     # Factorial designs (full, fractional, response surface, optimal, OMARS)
    monitoring/      # Control charts (Shewhart, CUSUM, EWMA)
    batch/           # Batch process data alignment, features, preprocessing
    regression/      # Robust regression (repeated median, Theil-Sen)
    bivariate/       # Elbow detection, peak finding, area under curve
    sensory/         # Descriptive panel-data analysis (validate, panel check, relate)
    simulation/      # Fake-data / process-simulator subpackage
    visualization/   # Plotting utilities (raincloud plots, chart spec/IR, adapters)
    chemistry/       # Product-by-compound block preprocessing (sensory-to-chemistry analysis)
    datasets/        # Sample datasets for examples and tests
    interactions.py  # Two-factor interaction terms, and whether the data can support them
    recipes.py       # Reusable analysis-recipe framework
    tool_spec.py     # @tool_spec decorator, tool registry, Anthropic tool-use specs
    tool_safety.py   # Subprocess isolation for tool calls over untrusted transports
    mcp_server.py    # FastMCP server exposing the registered tools (mcp extra)
    config.py        # Settings singleton reading PROCESS_IMPROVE_* env vars
    _extras.py       # Clean ImportError messages for missing optional extras
    _linalg.py       # Shared numerical-linear-algebra guards
    _random.py       # Shared random_state resolver (see reproducibility.rst)
    _remote_data.py  # Bounded download of the sample datasets that are not bundled
```

Outside the package:

- `web/` is the no-install browser app (https://factori.al/). `scripts/build_web_app.py`
  builds it, `tests/web/` tests it, and `deploy/factorial/` is its server.
- `skills/doe-designer/` is an agent skill that calls the `@tool_spec` registry.
- `tools/` holds the release and gate scripts: `changelog.py`, `complexity_budget.py`,
  and the link and runtime checkers.
- `changelog.d/` holds the changelog fragments.

## Key Architectural Decisions

These are the things that are costly to infer from the source. For the list of
fitted attributes on any estimator, read the class: they are documented in its
NumPy-style `Attributes` section, and any list duplicated here would rot.

### sklearn API Compatibility
- **PCA** inherits from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`
- **PLS** inherits from `sklearn.base.BaseEstimator`, `TransformerMixin`, and `RegressorMixin`.
  It deliberately does **not** inherit `sklearn.cross_decomposition.PLSRegression` (ENG-07 / #289):
  the estimators keep the lightweight sklearn *mixins* for API compatibility (`get_params`/`set_params`,
  `clone`, Pipeline support) but never couple to a concrete sklearn estimator's private attribute
  layout. The same applies to TPLS / MBPLS / MBPCA. `do not inherit from sklearn` here means the
  concrete estimators, not the mixins.
- Fitted attributes use the trailing `_` convention and are set only in `fit()`, never in `__init__`
  (sklearn requires `__init__` to set only the constructor parameters).
- `diagnose()` returns `sklearn.utils.Bunch` with named fields (not custom classes); `PLS.predict` / `OPLS.predict` return a DataFrame (sklearn convention)
- `score()` follows sklearn convention (higher is better)
- `fit()` returns `self`

### Frame-valued fitted attributes are descriptors
`scores_`, `loadings_`, `spe_` and their PLS counterparts are **not** assigned in
`fit()`. As of ENG-18 they are `_LazyFrame` descriptors (`multivariate/_base.py`)
over a private ndarray (`_scores`, `_loadings`, ...), which stays the source of
truth. The `pd.DataFrame` is built on first access from the array plus the
index/column metadata, cached, and excluded from pickling via `__getstate__`;
internal math reads the ndarray directly and skips the per-call `.values`
conversion. On an unfitted model the backing array is absent, so access raises
`AttributeError`, keeping `hasattr` / `check_is_fitted` behaviour intact.

When adding a frame-valued fitted attribute, follow that pattern: store the
array privately in `fit()` and declare the descriptor at class-body level.

### Convenience Method Binding
PCA / PLS / TPLS / MBPLS / MBPCA expose plot, limit, and diagnostic convenience methods
(`score_plot`, `spe_plot`, `loading_plot`, `spe_limit`, `score_limit`, `vip`, `eigenvalue_summary`,
`hotellings_t2_limit`, `ellipse_coordinates`, ...) that forward to the standalone functions in
`plots.py` / `_limits.py` / `_diagnostics.py`. As of ENG-05 (#287) these are **real methods defined
on the class** via `_model_method` from `multivariate/_common.py`, not `functools.partial` instances
bound in `fit()`. Methods needing fitted state are written out explicitly instead. This keeps
`help` / `inspect.signature` accurate, the fitted model picklable, and the methods overridable by
subclasses. The standalone functions remain importable for advanced callers. (TPLS's `spe_limit` is
a separate nested dict-of-callables API and is intentionally not a method.)

### Migration Helpers
Both PCA and PLS have `__getattr__` methods that raise `AttributeError` with helpful rename messages when old attribute names are used (e.g., `model.x_scores` tells you to use `model.scores_`).

### `Expt` and the `pi_` prefix
`Expt` (`experiments/structures.py`) is the only experiment container. Its
library-managed metadata is namespaced `pi_*`, is public API, and new fields take
the same prefix: see [`CONTRIBUTING.md`](CONTRIBUTING.md#naming-conventions).

## Coding Conventions

Full style guide: [`CONTRIBUTING.md`](CONTRIBUTING.md#code-style). The rules
below are the ones most often got wrong in an agent session.

- Line length 120. Linter: ruff, `select = ["ALL"]` with ignores in `pyproject.toml`.
  Formatter: ruff-format. Do not add black, flake8, or isort config; ruff covers all three.
- Type checking: mypy, a blocking CI gate over `src/process_improve`.
- **The lint gate is two commands, not one:** `ruff check .` **and**
  `ruff format --check .`. They fail independently, so a clean `ruff check .`
  says nothing about formatting. Run both before pushing.
- Keep the `ruff-pre-commit` `rev` in `.pre-commit-config.yaml` in step with the
  `ruff` pin in `pyproject.toml`; if they diverge, locally formatted code can
  still be rejected by CI. `tests/test_tooling_versions.py` fails the PR until
  they agree.
- The four complexity rules (`C901`, `PLR0912`, `PLR0913`, `PLR0915`) are a
  ratchet, gated in both directions by `tests/test_complexity_ratchet.py`: a new
  breach fails, and so does a refactor that removes one without lowering `BUDGET`
  in `tools/complexity_budget.py`. See
  [`CONTRIBUTING.md`](CONTRIBUTING.md#complexity-budget).
- Reformatting can strand a trailing `# noqa` on a line the reflow splits, so the
  directive no longer sits on the code it suppresses. Ruff then reports both the
  unsuppressed rule and the now-unused directive (`RUF100`): move the directive to
  the line the rule is reported against rather than deleting it.
- Docstrings are NumPy style throughout, with type annotations in both the
  signature and the docstring.
- Prefer `MCUVScaler` (mean-center, unit-variance) over the standalone `center()`
  / `scale()` for preparing data before fitting.
- `N` = samples, `K` = features, `M` = targets, `A` = components as local
  variables; stored as `n_samples_`, `n_features_in_`, `n_targets_`.
- Name arguments and result keys for what they are, never by a letter or a source's
  label, and never reuse a retired name for something else: see
  [`CONTRIBUTING.md`](CONTRIBUTING.md#naming-conventions).
- **No em-dashes** (U+2014) anywhere: docs, docstrings, comments, commit messages,
  PR descriptions, Markdown, reStructuredText or YAML prose. Use a hyphen, a
  semicolon, or split the sentence.

## Making a change

Solve the whole problem with the least new code.

**Before you write**, list every place the change must reach: callers, tests,
fixtures, `__init__` exports, the `@tool_spec` registry, the README (its code
blocks run in `tests/test_readme.py`), the docs, the browser app in `web/`, and
`skills/doe-designer/`. The companion book
([kgdunn/pid-book](https://github.com/kgdunn/pid-book)) runs its code against the
PyPI release, so renaming a public name breaks the book's CI once the release
reaches PyPI. Then check
what the change could break for users: a removed name, a changed default, a
different number. That is the scope. Extra features are not.

**Take the first option that fully works:**

1. Is it needed? Skip features, options and flexibility that nobody asked for,
   and name them in one line.
2. Is it already in the package? Reuse the shared helper the way the surrounding
   code does: `multivariate/_common.py` and `_base.py`, `_linalg.py`,
   `_random.py`, `_extras.py`.
3. Does the standard library or an installed dependency do it? Use it. Do not add
   a runtime dependency for a few lines: an optional surface goes in an extra,
   imported through `_extras.py`.
4. Otherwise, write the minimum code that works.

- Keep the solution small, never the change: finish every part the task needs,
  including the callers, tests and fixtures the change breaks.
- No wrapper, option, config or "for later" code that nobody asked for. Deletion
  beats addition. Keep the structure the codebase already has.
- Comment the why that the code cannot show, not the what.
- Bug fix: grep every caller of the function first, then fix the root cause once,
  in the shared code. A fix to one estimator of a family (PCA / PLS / TPLS / MBPLS
  / MBPCA) is checked on the others in the same change
  ([API consistency](CONTRIBUTING.md#api-consistency)).
- Code you move or merge keeps its validation and error handling.
- Between options of equal size, take the one that is correct on the edge cases:
  missing values, a constant column, a single sample, one component.
- A shortcut with a known limit carries a comment of this form:
  `shortcut: <the limit>, <when to upgrade>`.
- A new case in the reference catalogue records a `cause` for every number it does not
  recompute and every number that does not agree with its source; the tests fail without it. The
  causes and how to choose one: [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-case-to-the-reference-catalogue).

Never cut: validation at trust boundaries (`@tool_spec` inputs, `mcp_server.py`,
`tool_safety.py`), error handling that prevents data loss, security, the browser
app's accessibility, or anything the user asked for.

## Testing

Defaults in `pytest.ini` include xdist parallelism and the coverage gate, so a
plain `pytest` works. Debug helpers (`--pdb`, `-x`, `-v`) are deliberately not
defaults; pass them manually.

```bash
uv run pytest                                        # full suite, parallel, coverage gate
uv run pytest tests/test_multivariate.py --no-cov    # one file
uv run pytest -k "pls" --no-cov                      # by keyword
uv run pytest --no-cov --affected-since=origin/main  # only tests your branch can affect
```

`--affected-since` is for fast local iteration before pushing. On a pull request
CI uses it for the `test` and `test-free-threaded` rows, while the `coverage` job
runs the whole suite. How it selects tests is in `CONTRIBUTING.md`.

Install with `uv sync --dev --all-extras`. Without the extras a large part of
the suite fails on `ImportError` rather than skipping.

**Tag new tests with the right tier marker.** `pytest.ini` registers `unit`
(implicit default), `integration`, `slow` (>= 2 s) and `dataset` (loads a
bundled or remote real dataset); the browser app's tiers (`e2e`, `live`, `wasm`,
`smoke`) run only with their flags. Untagged slow or network-bound tests silently
break `-m 'not slow'` for everyone. The table of when to use each is in
[`CONTRIBUTING.md`](CONTRIBUTING.md#test-tiers).

Other conventions:
- Use **real datasets** (LDPE, SIMCA) alongside synthetic data; do not remove real dataset tests.
- Scale with `MCUVScaler().fit_transform(X)` (not just `center()`).
- For synthetic PLS data use `X.values @ beta` (not `X @ beta`), which otherwise produces NaN via pandas column mismatch.
- Fixtures load CSV data from `src/process_improve/datasets/multivariate/`.
- New methods need tests for both basic functionality and edge cases.
- Guard optional dependencies with `pytest.importorskip`, and probe binaries that
  can be present but non-executable (e.g. pulp's bundled CBC solver).
- The four coverage shards are balanced by `tests/.test_durations.json`. When
  their times drift apart in CI, refresh it the way the shards run:
  `uv run pytest --store-durations --cov-fail-under=0` on Python 3.13.

## Versioning and changelog

Version lives in `pyproject.toml` under `[project] version`, 3-part semver.
The full policy, including what counts as a breaking change, is in
[`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-policy).

**Do not bump the version in a PR.** `pyproject.toml` `version` and
`CITATION.cff` are set once, at release time, from whatever accumulated under
`## [Unreleased]`. A PR that touches either file for the sake of a bump is
wrong: it makes that PR conflict with every other open PR on three lines that
have nothing to do with the work. The level (MAJOR / MINOR / PATCH) is read off
the `[Unreleased]` headings at release; the rule and the release steps are in
[`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-policy).

The one thing a PR owes the release is a correctly *filed* changelog entry,
because the heading it sits under is what picks the level.

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com). **Prompt the user to confirm whether an entry is required.** User-facing changes (features, API changes, bug fixes, behavioural changes) generally need one; internal-only changes (refactors, CI tweaks, edits to this file) generally do not. New entries go in a fragment, `changelog.d/<yyyy-mm-dd>-<slug>.<type>.md` (the date is the day you write it), never in `CHANGELOG.md` itself; `<type>` is the heading it belongs under (`added`, `changed`, `fixed`, `removed`, `deprecated`, `security`, `documentation`). The release step folds the fragments into `CHANGELOG.md`.

Publishing is **manually gated** (ENG-21 / #303): `publish.yml` runs only on a `v*` tag or a maintainer's `workflow_dispatch`. Nothing in a PR publishes; the release commit and its tag do. Releases carry a sigstore attestation (PEP 740) and a CycloneDX SBOM, with notes pulled from the matching `CHANGELOG.md` section.

## CI/CD

Workflows in `.github/workflows/`:

- **run-tests.yml**: `lint` (`ruff check .` and `ruff format --check .`, two
  independent gates), `typecheck` (blocking `mypy src/process_improve`), `web-app`
  (builds the browser app and runs its Chromium, Pyodide and Node.js tests), `test`
  (the full suite without coverage, over Python 3.11-3.13; the 3.11 rows install
  the lowest release of each dependency that `pyproject.toml` allows; a pull request runs
  ubuntu only, and pushes to main, the nightly schedule and manual runs add
  windows and macos at 3.11 and 3.13; ubuntu/3.13 is the `coverage` job
  instead), `coverage`
  (the full suite on ubuntu/3.13 with coverage, in duration-balanced shards via
  `--shard`), `coverage-report` (combines the shards for the 96% gate and the
  diff-cover patch gate), `test-under-dash-O` (the suite with
  `PYTHONOPTIMIZE=1`, to catch load-bearing asserts; plain `python -O` would not
  reach the xdist workers), and `test-free-threaded`. On a pull request the
  `test` and `test-free-threaded` rows run only the affected tests
  (`--affected-since`); every other run, and the `coverage` job always, runs the
  whole suite. `CI passed` is the one
  check to require in branch protection: it fails unless every other job except
  `coverage-report` succeeded. All jobs install with `uv sync --dev --all-extras`.
- **main-status.yml**: the README badge; passes or fails as the latest run of
  the tests on main that was not cancelled. Merges to main share one
  concurrency group, so a merge that lands while one run goes and another waits
  cancels the waiting run; the next run tests every merge before it.
- **docs.yml**: strict Sphinx build (`-W`, notebooks executed) and GitHub Pages deploy on main.
  On main it also sends the browser app (the same build its smoke tests just
  passed) to https://factori.al/ (`deploy-factorial`) and smoke-tests it there
  (`check-factorial`). Both jobs are skipped until the repository variable
  `FACTORIAL_KNOWN_HOSTS` is set.
- **factorial-rollback.yml**: manual; switches factori.al back to the build
  before, or to a commit you name.
- **web-app-live.yml**: weekly. Smoke-tests the published app, which loads from
  Pyodide's CDN, PyPI and Plotly's CDN and so can break with no change here, and
  reports a pin in `web/runtime.json` that is no longer the latest release; a
  scheduled failure of either opens an issue. Its `newest-pyodide` job is advisory: it
  tries the newest Pyodide release and says whether moving the pin is safe.
- **publish.yml**: tag-gated PyPI publish (see above).
- **codeql.yml**: weekly and per-PR security scanning.
- **readme-links.yml**: requests every external link in `README.md` and
  `SKLEARN_COMPATIBILITY.md`, weekly (a failure opens an issue) and on PRs that
  touch them. Links inside the repository, and the badges of its own workflows,
  are checked offline on every PR by `tests/test_readme_guards.py`.

Docs are Sphinx with the PyData theme, NumPy docstrings via `sphinx.ext.napoleon`; build with `cd docs && make html`.

## Git & PR workflow (for Claude Code sessions)

**Default workflow for any non-trivial task:**

1. **Open a blank PR up front**, before doing the work. Push an empty commit
   (or the first micro-commit) to the session's working branch (for example
   `claude/<task-slug>`) and open the PR as
   ready-for-review with a description of what's about to be done. Do not
   open it as a draft.
2. **Micro-commit as you go.** Each commit should be a small, self-contained
   step (one module's tests, one config tweak, one bug fix) - not a single
   end-of-session megacommit.
3. **Push regularly.** After every micro-commit (or at worst every 2-3),
   `git push` to the same branch so the PR reflects current progress and the
   user can watch it land.

This is the default - don't ask the user whether to do it, just do it.

**Never push lock files.** Claude Code sessions must not stage, commit, or push
any dependency lock file: `uv.lock`, `poetry.lock`, `Pipfile.lock`,
`requirements.lock` / pip-tools compiled lockfiles, or any equivalent
regenerated artifact. If a command (`uv sync`, `pip install`) regenerates one
during a session, leave it uncommitted; if it is already staged, unstage it with
`git restore --staged <lockfile>`. The user refreshes lock files manually
outside of Claude Code sessions.

**No scheduled PR check-ins.** Do not schedule recurring self check-ins
(`send_later`, routines, `/loop` or anything similar) to poll open PRs for
merges, CI results or reviews, and do not send "nothing changed" status
messages. Once the work is pushed and the PR is open, report and end the turn.
Handle PR events that arrive on their own (a CI failure, a review comment, a
merge-conflict notice) when they arrive; otherwise the user asks for a status
when they want one.

## Communication style

- Lead with the result. Your first sentence answers "what happened" or "what's
  the answer."
- Cut narration. Don't restate the request, the plan, or each step you took.
  Report outcomes, decisions, and anything the user must act on.
- Short by default. Answer simple questions in 1-3 sentences of plain prose.
- Never trade correctness for brevity. Error reports and other technical
  content keep their full content.
- Write in the spirit of ASD-STE100 (Simplified Technical English). Keep
  sentence structures direct. Limit each procedural step to a single action.
  Avoid unnecessary synonyms: use one term for one thing.
- End a report with one or two lines: what you skipped or did not check, and any
  risk the user must know.
- Every PR named in chat carries its full link, in this repository or another,
  never a bare number:
  [process-improve #958](https://github.com/kgdunn/process-improve/pull/958).

## Updating this file (CLAUDE.md)

If during a session you notice a recurring pattern, convention, or piece of
project context that you think belongs in `CLAUDE.md`, **do not add it
yourself**. Surface the proposed addition in chat - the wording you would
add, where you would put it, and why you think it is reusable - and ask the
user whether it should be recorded here. The user decides what gets
canonised in this file.

Before adding anything, check whether one of the documents in
[Authoritative documents](#authoritative-documents) already owns that fact. If
it does, the right change is usually a pointer, or a fix to that document, not a
second copy here.
