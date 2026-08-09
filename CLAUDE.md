# CLAUDE.md - Project Context for AI Agents

## Overview

`process-improve` is a Python package for process improvement using data. It accompanies the online textbook [Process Improvement using Data](https://learnche.org/pid). The package provides multivariate analysis (PCA, PLS, and TPLS - *PLS for T-shaped data structures*, not "Total PLS" or "Three-way PLS"), designed experiments, process monitoring, batch data analysis, and visualization tools.

**Repository:** https://github.com/kgdunn/process-improve
**License:** MIT
**Python:** >= 3.10 (CI tests 3.10-3.13; 3.13 is the primary version for lint, typecheck, and coverage)

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
    datasets/        # Sample datasets for examples and tests
    recipes.py       # Reusable analysis-recipe framework
    tool_spec.py     # @tool_spec decorator, tool registry, Anthropic tool-use specs
    tool_safety.py   # Subprocess isolation for tool calls over untrusted transports
    mcp_server.py    # FastMCP server exposing the registered tools (mcp extra)
    config.py        # Settings singleton reading PROCESS_IMPROVE_* env vars
    _extras.py       # Clean ImportError messages for missing optional extras
    _linalg.py       # Shared numerical-linear-algebra guards
    _random.py       # Shared random_state resolver (see reproducibility.rst)
```

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
- `predict()` returns `sklearn.utils.Bunch` with named fields (not custom classes)
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
`Expt` (`experiments/structures.py`) is the canonical and only name for the
experiment container, a `pd.DataFrame` subclass. Library-managed metadata on it
is namespaced with a `pi_` prefix (`pi_title`, `pi_units`, `pi_range`, `pi_lo`,
`pi_hi`, `pi_center`, `pi_name`, `pi_levels`, `pi_is_coded`, ...), which exists
so these reserved names cannot collide with columns from a user-supplied
DataFrame. Treat `pi_*` as public API, and give new metadata fields the same
prefix.

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
  still be rejected by CI.
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
- **No em-dashes** (U+2014) anywhere: docs, docstrings, comments, commit messages,
  PR descriptions, Markdown, reStructuredText or YAML prose. Use a hyphen, a
  semicolon, or split the sentence.

## Testing

Defaults in `pytest.ini` include xdist parallelism and the coverage gate, so a
plain `pytest` works. Debug helpers (`--pdb`, `-x`, `-v`) are deliberately not
defaults; pass them manually.

```bash
uv run pytest                                        # full suite, parallel, coverage gate
uv run pytest tests/test_multivariate.py --no-cov    # one file
uv run pytest -k "pls" --no-cov                      # by keyword
```

Install with `uv sync --dev --all-extras`. Without the extras a large part of
the suite fails on `ImportError` rather than skipping.

**Tag new tests with the right tier marker.** `pytest.ini` registers `unit`
(implicit default), `integration`, `slow` (>= 2 s) and `dataset` (loads a
bundled or remote real dataset). Untagged slow or network-bound tests silently
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

## Versioning and changelog

Version lives in `pyproject.toml` under `[project] version`, 3-part semver.
The full policy, including what counts as a breaking change, is in
[`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-policy).

**Auto-bump the version with every PR that changes code or configuration:**
- **PATCH**: bug fixes, CI/workflow changes, docs updates, dependency bumps, small refactors.
- **MINOR**: new features, new modules, significant API additions, meaningful behavioural changes. Resets PATCH to 0.
- **MAJOR**: incompatible removals, which are only permitted after the
  `docs/development/deprecation_policy.rst` schedule has run. Resets MINOR and PATCH to 0.
- **If unsure which level applies, ask the user** before bumping.

**Keep `CITATION.cff` in sync.** In the *same commit* as a version bump, set its `version:` to the identical value and `date-released:` to the current date. The two files must never disagree.

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com). **Prompt the user to confirm whether an entry is required.** User-facing changes (features, API changes, bug fixes, behavioural changes) generally need one; internal-only changes (refactors, CI tweaks, edits to this file) generally do not. New lines go under `## [Unreleased]`. When bumping the version, in the same commit rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and update the link-reference footer (the `[Unreleased]` compare link plus a new `[X.Y.Z]` link).

Publishing is **manually gated** (ENG-21 / #303): `publish.yml` runs only on a `v*` tag or a maintainer's `workflow_dispatch`. Bumping the version in a PR does not publish. Releases carry a sigstore attestation (PEP 740) and a CycloneDX SBOM, with notes pulled from the matching `CHANGELOG.md` section.

## CI/CD

Workflows in `.github/workflows/`:

- **run-tests.yml**: `lint` (`ruff check .` and `ruff format --check .`, two
  independent gates), `typecheck` (blocking `mypy src/process_improve`), `test`
  (pytest matrix over Python 3.10-3.13 and ubuntu/windows/macos), and
  `test-under-dash-O` (the suite under `python -O`, to catch load-bearing
  asserts). All jobs install with `uv sync --dev --all-extras`.
- **docs.yml**: strict Sphinx build (`-W`, notebooks executed) and GitHub Pages deploy on main.
- **publish.yml**: tag-gated PyPI publish (see above).
- **codeql.yml**: weekly and per-PR security scanning.

Docs are Sphinx with the PyData theme, NumPy docstrings via `sphinx.ext.napoleon`; build with `cd docs && make html`.

## Git & PR workflow (for Claude Code sessions)

**Default workflow for any non-trivial task:**

1. **Open a blank PR up front**, before doing the work. Push an empty commit
   (or the first micro-commit) to `claude/<task-slug>` and open the PR as
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
