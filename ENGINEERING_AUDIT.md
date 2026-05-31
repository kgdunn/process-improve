# Engineering Audit

A second-pass review of `process-improve` from the perspective of a
senior engineer with a long career in production APIs and scientific
ETL pipelines. The scope **excludes** security and numerical
correctness (those live in `SECURITY_AUDIT.md`); this document
critiques the things that survive even after every `SEC-NN` is fixed:
architecture, API design, type safety, testing strategy, ETL plumbing,
configuration, packaging, observability, and release engineering.

The tone is intentionally critical -- "what would you flag in a
code review for a 1.x library that other people build on" -- not
"what's good about it". A separate "what's good" note appears at the
end.

Findings are grouped by theme and rated:
- **Major** -- structural issue that compounds over time; should be on
  a roadmap.
- **Moderate** -- annoying or risk-prone but not currently painful;
  worth fixing opportunistically.
- **Minor** -- style or polish.

Tracking ID format mirrors `SECURITY_AUDIT.md`: `ENG-NN`.

---

## Status overview

| ID | Title | Severity |
|----|-------|----------|
| ENG-01 | `multivariate/methods.py` is a 5,863-line monolith | Major |
| ENG-02 | `experiments/tools.py` and `analysis.py` are 1,400-line god modules | Major |
| ENG-03 | Type system is opt-out (`disallow_untyped_defs = false`, ~370 `Any`) | Major |
| ENG-04 | `pydantic` is a hard dependency but only the MCP layer uses it | Moderate |
| ENG-05 | Convenience methods bound via `functools.partial` after `fit()` are not introspectable | Moderate |
| ENG-06 | Three plotting backends (matplotlib + plotly + echarts) for a small team | Major |
| ENG-07 | Inheriting from `sklearn.cross_decomposition.PLSRegression` couples to a private layout | Moderate |
| ENG-08 | RNG handling is inconsistent; "reproducibility" is a vibe, not a contract | Major |
| ENG-09 | Configuration sprawl: env vars at import time, magic numbers, no central config | Moderate |
| ENG-10 | Tool-spec decorator duplicates the function signature in JSON Schema | Moderate |
| ENG-11 | Error-handling style varies wildly across the codebase | Major |
| ENG-12 | Logging is sparse; no structured logs; no metrics | Major |
| ENG-13 | 18 hard runtime dependencies; no extras for plotting / experiments / mcp | Major |
| ENG-14 | Flat package layout (no `src/`); notebooks shipped inside the package | Moderate |
| ENG-15 | Test suite under-uses property-based testing, `python -O`, perf, and fuzzing | Major |
| ENG-16 | `Expt` and `Model` mutate state via dynamic attributes (`pi_title`, `data`) | Moderate |
| ENG-17 | PCA / PLS / TPLS / MBPLS / MBPCA duplicate fit / predict / plot scaffolding | Major |
| ENG-18 | Pandas-shaped fitted attributes (`scores_` as `DataFrame`) -- ergonomic but slow / dtype-fragile | Moderate |
| ENG-19 | Every algorithm assumes the whole matrix fits in RAM; no chunking / streaming | Moderate |
| ENG-20 | CI does not run mypy / pyright; type-check is local-only | Moderate |
| ENG-21 | Auto-publish-on-version-bump release flow is novel; no signing / SBOM / provenance | Moderate |
| ENG-22 | No deprecation policy; rename helpers via `__getattr__` will accumulate | Moderate |
| ENG-23 | `methods.py` is the same generic filename in three packages | Minor |
| ENG-24 | `pi_` prefix and `Expt` abbreviation are undocumented | Minor |
| ENG-25 | Many `noqa: C901 / PLR0912 / PLR0915 / PLR0913` suppressions (69 instances) | Moderate |
| ENG-26 | Sphinx docs and architecture overview are thin | Moderate |
| ENG-27 | `tool_safety.py` reads env vars at import time -- can't be re-configured at runtime | Minor |
| ENG-28 | No CONTRIBUTING-level guidance on breaking-change policy or perf regressions | Minor |
| ENG-29 | `tests/` mixes tier-1 unit, tier-2 integration, and tier-3 dataset tests with no marking | Moderate |
| ENG-30 | The MCP dispatch path is sync inside an async server | Moderate |

---

## Architecture

### ENG-01 -- `multivariate/methods.py` is a 5,863-line monolith *(Major)*
Five public classes (`MCUVScaler`, `PCA`, `PLS`, `TPLS`, `MBPLS`,
`MBPCA`), shared private helpers, sign-flip conventions, NIPALS
inner loops, score / loading / limit helpers, and the `Resampler`
all live in one file. The file scores 5,863 LoC and at least 14
distinct logical sections. Editing one class touches the others'
git history; reviewing a PR that says "fix PCA NIPALS" requires
reading half the file. **Action:** split into
`multivariate/{base,mcuv,pca,pls,tpls,multiblock,resampler,limits,plotting_bindings}.py`,
with shared scaffolding (NIPALS step, sign convention, missing-data
imputation, R^2 bookkeeping) in `_internal/`. The package's
`__init__.py` keeps the existing imports so this is a no-op for
users.

### ENG-02 -- `experiments/{tools,analysis}.py` are 1.4k-line god modules *(Major)*
`experiments/tools.py` (1,473 LoC) bundles 10 MCP tool wrappers,
each ~150 lines (schema + dispatch + clean-up). `experiments/
analysis.py` (778 LoC) bundles ANOVA, effects, coefficients,
significance, residual diagnostics, lack-of-fit, curvature, model
selection, Box-Cox, Lenth's, confidence intervals, prediction, and
confirmation analyses behind a single `analyze_experiment()`. The
single-entry-point pattern is convenient for the MCP tool spec but
makes the implementation hard to test, hard to extend, and hard to
deprecate single sub-analyses. **Action:** keep
`analyze_experiment()` as a dispatcher, but move each
`_run_anova`, `_run_effects`, etc. to its own module
`experiments/_analyses/{anova,effects,...}.py`; same for the tool
wrappers (`experiments/_tools/{create_factorial,analyze,evaluate,...}.py`)
with the `_register(...)` calls re-exported from `tools.py`.

### ENG-17 -- PCA / PLS / TPLS / MBPLS / MBPCA duplicate scaffolding *(Major)*
Each class re-implements: a fit dispatcher, a `fit_transform`, a
`predict` -> `Bunch` builder, attribute-rename `__getattr__`, the
`functools.partial`-binding of plot helpers, and the post-fit
`scaling_factor_for_scores_` computation. None of this is in a
shared base class; the result is that every `SEC-NN` numerical
fix has to be applied 3-5 times (this audit alone duplicates work
across PCA, PLS, TPLS, MBPLS in SEC-21). **Action:** introduce a
`_LatentVariableModel` base class that owns the common state
(`n_features_in_`, `scores_`, `loadings_`, `spe_`, the
`_bind_plot_helpers()` step, `__getattr__` migration) and let each
algorithm provide just the inner-loop step (`_fit_one_component`).

### ENG-23 -- `methods.py` is the same filename in three packages *(Minor)*
`multivariate/methods.py`, `regression/methods.py`,
`bivariate/methods.py`, `experiments/models.py` -- generic file
names hide the domain. Tooling that ranks by filename (Jump To
File, codecov reports, fuzzy search) gets ambiguous. **Action:**
rename to `_pca_pls.py`, `_robust_regression.py`,
`_elbow_peak.py`, etc., and keep `methods.py` as a thin
re-exporter.

---

## API design

### ENG-05 -- `functools.partial` plot bindings are not introspectable *(Moderate)*
After `fit()` the code does:
```python
self.score_plot = partial(score_plot, model=self)
self.spe_plot   = partial(spe_plot,   model=self)
...
```
This works in a notebook, but:
- `help(model.score_plot)` shows `functools.partial`, not the
  function's docstring.
- IDEs cannot autocomplete the function's parameters.
- `inspect.signature` shows the *partial* signature, not the
  underlying one.
- Pickling / serialising a fitted model breaks because `partial`
  references the bound callable.
- Subclassers cannot override these "methods" cleanly.

**Action:** define them as actual methods on the class
(`def score_plot(self, ...): return _score_plot(self, ...)`) -- a
mechanical change. Keep the standalone function for advanced
callers.

### ENG-07 -- Inheriting from `sklearn.cross_decomposition.PLSRegression` couples to private layout *(Moderate)*
`PLS` inherits from sklearn's `PLSRegression`, which means the
class picks up sklearn's `__init__`, `_validate_params`,
`get_params`, `set_params`, the `BaseEstimator` machinery -- and
sklearn's *current* attribute layout. The package then *overrides*
most of those attributes (`x_loadings_`, `y_loadings_`,
`scores_`, etc.) with its own pandas-shaped versions, leaving a
zombie sklearn fit underneath. This will break on a sklearn major
bump (sklearn renames attributes on a major every few years), and
sklearn-style introspection (`get_params`) returns a confusing
mix. **Action:** compose, don't inherit -- hold a private
`self._sklearn_pls = PLSRegression(...)` and expose only the
attributes the package documents. CLAUDE.md already names the
"sklearn API compatibility" as a goal; that doesn't require
inheritance -- `BaseEstimator + TransformerMixin + RegressorMixin`
plus the right fitted-attribute conventions is enough.

### ENG-16 -- Dynamic-attribute objects (`Expt`, `Model`) *(Moderate)*
`Expt(df).pi_title = None; expt_data.pi_source = None;
expt_data.pi_units = None` is the construction idiom for
`fit_linear_model`. The class accepts attributes that aren't in
`__init__`, aren't typed, aren't validated, and aren't documented.
Similarly `Model` sets `self.data = None` in `__init__` and then
expects callers to overwrite it. **Action:** make these
`@dataclass(slots=True)` (or pydantic) so the supported attribute
set is explicit, attribute typos fail at construction, and IDE
auto-complete works.

### ENG-22 -- No deprecation policy; rename helpers accumulate *(Moderate)*
The `__getattr__` migration helper on PCA / PLS that suggests new
attribute names is a great pattern. It also means every renamed
attribute lives forever -- there is no "removed in v2.0" timeline.
At semver 1.x with 18 dependencies and a hand-curated MCP surface,
this will calcify. **Action:** add a deprecation policy to
`CONTRIBUTING.md` (e.g. "marked deprecated for one MINOR cycle,
warn for two, remove on the next MAJOR") and use
`warnings.warn(..., DeprecationWarning, stacklevel=2)` in the
`__getattr__` migration responses.

### ENG-24 -- `pi_` prefix and `Expt` are undocumented *(Minor)*
`pi_title`, `pi_source`, `pi_units`, `pi_range` -- the `pi_`
prefix is presumably `process_improve`, but new readers will have
to grep the codebase to find out. `Expt` abbreviates "Experiment"
in a codebase that otherwise spells out names. **Action:**
document `pi_` in `CONTRIBUTING.md` (or rename to `pio_` /
`piinfo_`); spell out `Expt` as `Experiment` with `Expt` left as
a deprecated alias.

---

## Type system

### ENG-03 -- mypy is opt-out; ~370 `Any` annotations *(Major)*
`pyproject.toml [tool.mypy]` sets:
```toml
disallow_untyped_defs = false
disable_error_code = ["import-untyped"]
```
which makes the type-check optional and silences library-stubs
problems. Combined with 373 `Any` occurrences in production code,
the type system gives the false comfort of decorations without the
actual safety net. Many of those `Any`s are easily replaceable:
`dict[str, float]`, `pd.DataFrame`, `np.ndarray[np.float64]`, etc.
**Action:** turn on `disallow_untyped_defs` in
`process_improve/`, leave it off in `tests/`, fix the cascade.
Drop `disable_error_code = ["import-untyped"]`; add explicit
`# type: ignore[import-untyped]` per import that genuinely lacks
stubs (with the library name and a "TODO: contribute stubs"
comment).

### ENG-20 -- CI does not run mypy *(Moderate)*
The CI workflow runs `ruff check .` and `pytest`. mypy is in
`[dev]` extras but never invoked in CI. Even with a permissive
mypy config the run would catch broken annotations on PR. **Action:**
add a `lint` matrix step that runs `mypy process_improve`,
initially in non-blocking mode, then ratchet to blocking once
`disallow_untyped_defs` lands per ENG-03.

### ENG-25 -- 69 `noqa: C901 / PLR0912 / PLR0915 / PLR0913` suppressions *(Moderate)*
ruff's complexity checks are off-by-default for these functions
via in-line `# noqa`. The presence of so many is a code-smell
canary: these are exactly the functions most likely to grow
SEC-NN-style bugs (they're complex enough that no single reader
can hold them in their head). **Action:** treat each `noqa: C901`
as a refactor candidate; pick the worst offender per release
cycle. SEC-21 sub-items 1-9 all live inside functions that carry
these suppressions.

---

## Validation, configuration, and tooling

### ENG-04 -- pydantic is a hard dep but only MCP uses it *(Moderate)*
`pydantic>=2.12.5` is in `[project] dependencies`. Searching the
codebase finds it imported nowhere outside the MCP-server path
(FastMCP pulls it in transitively). Either commit to pydantic at
every external boundary (replace the ad-hoc `validate_input`,
`validate_against_schema`, `tool_spec` JSON Schemas) or drop it
from the core and re-add it under `[mcp]` extras. The current
state is the worst of both: a heavy dep without the payoff.
**Action:** decide; the ergonomic + safety upside of pydantic
models at the tool boundary is substantial (see ENG-10).

### ENG-10 -- `@tool_spec` duplicates the function signature in JSON Schema *(Moderate)*
Every `@tool_spec` decorator writes the parameter list once as a
Python signature and again as a hand-written JSON Schema. The
schemas drift from the signatures over time (SEC-15, SEC-20 are
direct consequences). **Action:** generate the JSON Schema from a
pydantic model per tool (which doubles as the signature) -- a
single source of truth. This also fixes SEC-15 (the
`reveal_simulator` kwarg-injection) because pydantic refuses
unknown fields by default.

### ENG-09 -- Configuration sprawl *(Moderate)*
`tool_safety.py` reads `PROCESS_IMPROVE_TOOL_TIMEOUT`,
`PROCESS_IMPROVE_MAX_CELLS`, `PROCESS_IMPROVE_MAX_STRING`,
`PROCESS_IMPROVE_MAX_DEPTH`, `PROCESS_IMPROVE_MAX_MEMORY_MB` at
import time. `mcp_server.py` reads `PROCESS_IMPROVE_MCP_SAFE_MODE`.
Plus 30+ magic numbers (`epsqrt`, `0.995`, `0.5`, `_NOISE_FRACTIONS`,
`MODEL_VERSION`, etc.). **Action:** introduce a single
`process_improve.config` module with a pydantic-settings model;
read env once; expose a `Config` object that the rest of the code
can override in tests (per-call) rather than at import time.

### ENG-27 -- env vars read at import time *(Minor)*
Follows from ENG-09: `DEFAULT_TIMEOUT_S = float(os.environ.get(...,
"10"))` at module-level means a test that mutates the env after
import has no effect. The pool-recreation logic in `get_pool`
papers over this for `memory_mb`, but not for the other knobs.
**Action:** read on first use, not at import; or use the
`Config` object from ENG-09.

---

## Error handling and observability

### ENG-11 -- Error-handling style varies wildly *(Major)*
A non-exhaustive list of patterns in current production code:
- `if not ...: raise ValueError(...)`
- `assert condition, "message"` (subject to SEC-08 / SEC-17)
- `try: ... except Exception: return None`
- `try: ... except Exception as e: return {"error": str(e)}`
  (SEC-18)
- `try: ... except Exception: pass`
- `if cond: raise RuntimeError(...)`
- `warnings.warn(...)`
- Silent NaN propagation (SEC-21)

A reader landing in a new module cannot predict how a failure will
surface. **Action:** publish an error-handling style guide in
`CONTRIBUTING.md`:
- Internal invariants -> `assert` (rare) or `RuntimeError`
- User input violations -> `ValueError` / `TypeError`
- Library degeneracies -> dedicated exception classes
  (`SingularDesignError`, `NotConvergedError`, ...)
- MCP boundary -> structured `ToolSafetyError` subclasses (good
  pattern; extend it)
- Soft warnings -> `warnings.warn(..., category=...)`, never
  `print`

Then sweep the codebase.

### ENG-12 -- Logging is sparse; no structured logs; no metrics *(Major)*
Only ~10 modules import `logging`. There are no module-level
loggers in `multivariate/methods.py`, `experiments/analysis.py`,
`batch/preprocessing.py`, or anywhere a long-running algorithm
loops. There is no convergence-iteration log, no "rejected because
input was X" trace, no performance counter. Diagnosing a wrong
PCA result in production requires re-running with `import pdb;
pdb.set_trace()`. **Action:** add a `logger =
logging.getLogger(__name__)` to every module that does real work;
emit `logger.debug("NIPALS converged in %d iters, tol=%g", k,
tol)` at component boundaries; emit `logger.info("...")` when a
guard rail fires.

---

## Dependencies and packaging

### ENG-13 -- 18 hard runtime deps; no extras *(Major)*
`matplotlib`, `numba`, `numpy`, `openpyxl`, `pandas`, `patsy`,
`plotly`, `pydantic`, `pyyaml`, `pyDOE3`, `ridgeplot`,
`scikit-image`, `scikit-learn`, `seaborn`, `statsmodels`,
`tqdm`, `mcp` (in extras already). For a library that other
people import inside *their* science pipelines, this is a
substantial transitive footprint. A user who only wants
`detect_multivariate_outliers` is forced to install `plotly`,
`seaborn`, `ridgeplot`, `openpyxl`, `numba`, ... 

**Action:** split into:
- core: numpy, pandas, scipy, scikit-learn, statsmodels, patsy
- `[plotting]`: matplotlib, plotly, seaborn, ridgeplot
- `[experiments]`: pyDOE3, patsy (already core)
- `[batch]`: scikit-image, openpyxl
- `[mcp]`: mcp, pydantic, fastmcp
- `[notebooks]`: ipykernel, jupyter
- meta extra `[all]` to keep current install behaviour.

`numba` is doubly suspicious: a runtime dep that's used in only a
handful of inner loops (and that ships large prebuilt wheels);
should be an extra (`[fast]`).

### ENG-14 -- Flat package layout; notebooks shipped inside the package *(Moderate)*
`pyproject.toml` notes "we do not (yet) use a `src/` layout".
That's a recipe for accidental local-import shadowing (running
`python` in the repo root imports the *source* `process_improve/`
package even if a different version is `pip install`-ed). It also
means tests under `tests/` import the local source by accident
when run elsewhere. The `process_improve/notebooks_examples/`
folder is *inside* the installable package -- it ships in every
wheel and is `from process_improve.notebooks_examples.batch
import batch_llm` reachable. **Actions:**
- Move source to `src/process_improve/`.
- Move `notebooks_examples/` to a top-level `examples/` (not
  installed) -- or to a separate `process-improve-examples` repo.

### ENG-21 -- Auto-publish-on-version-bump release flow; no signing / SBOM *(Moderate)*
Per `CLAUDE.md`: "The PyPI publish workflow automatically detects
version changes on push to main and publishes to PyPI when the
version differs from the previous commit." This is convenient but:
- A typo in `pyproject.toml` ships a release; there's no human
  press-the-button step.
- No PEP 740 / sigstore attestation that the wheel came from CI.
- No SBOM (SPDX or CycloneDX) shipped with the release.
- No release notes in the `gh release` -- only the `CHANGELOG.md`
  entry, which is text on the repo.

**Actions:** require a git tag (not just a version bump), produce
attestations via `pypa/gh-action-pypi-publish` (built-in sigstore
support), generate an SBOM from the wheel, push release notes to
the GitHub release page from `CHANGELOG.md`.

### ENG-28 -- CONTRIBUTING has no breaking-change or perf-regression policy *(Minor)*
The current `CONTRIBUTING.md` covers setup and conventions. It
does not say what counts as a breaking change, how to deprecate,
how to add a performance benchmark, or how to negotiate behaviour
changes with downstream users. **Action:** add sections on
versioning policy, deprecation timeline (ENG-22), and perf
regression handling.

---

## Tests

### ENG-15 -- Test suite under-uses property-based testing, `-O`, perf, and fuzzing *(Major)*
- `hypothesis` is in `[dev]` but no `tests/test_*.py` imports it.
  PCA / PLS are perfect property-based targets (round-trips,
  invariants under permutation, sign-flip equivalence).
- No tests run under `python -O`. SEC-17 exists precisely
  because nobody noticed that `assert`-stripping silently disabled
  shape-checks.
- No performance baselines / regression tests. `repeated_median`,
  PCA NIPALS, and TPLS each have O(N^2) or O(N x K^2) paths that
  could silently get 10x slower without anyone noticing.
- The MCP boundary has no fuzz suite. SEC-14, SEC-15, SEC-20
  would all have been caught by a "throw random JSON-Schema-
  matching inputs at every `@tool_spec` and assert no uncaught
  exception" test.

**Action:**
1. Add `tests/properties/test_pca_invariants.py` etc. using
   hypothesis.
2. Add a `tox.ini` env / pytest mark `@pytest.mark.under_dash_O`
   and run it in CI with `python -O`.
3. Add `tests/perf/` benchmarks using `pytest-benchmark` and a
   CI job that fails on >25% regression.
4. Add `tests/fuzz/test_mcp_boundary.py` using hypothesis
   strategies derived from each tool's `input_schema`.

### ENG-29 -- `tests/` mixes tier-1 unit, tier-2 integration, tier-3 dataset tests with no marking *(Moderate)*
43 test files, ~19,300 LoC, no `pytest.mark.*` markers separate
unit vs integration vs slow-dataset tests. A contributor running
`pytest` waits for the full suite. **Action:** mark long-running
or network-dependent tests with `@pytest.mark.slow` /
`@pytest.mark.dataset`, default-skip them, and run the full suite
in CI.

---

## ETL / data plumbing

### ENG-18 -- Pandas-shaped fitted attributes are ergonomic but slow / dtype-fragile *(Moderate)*
PCA / PLS / TPLS store `loadings_`, `scores_`, `spe_`,
`r2_per_component_`, etc. as `pd.DataFrame` / `pd.Series`. This
is great for notebook display and for preserving the original
sample / feature index. It is also:
- Slower than `np.ndarray` for any subsequent `@`-multiply.
- Promotes ints to floats silently.
- Tied to pandas's evolving API (`.iloc` / `.loc` semantics shift
  between major versions).
- Heavier to (de)serialise.

**Action:** store *both*: a private `_loadings: np.ndarray` for
internal use and a public `loadings_: pd.DataFrame` view built
lazily. Methods that operate on `model.loadings_` then no longer
pay the `.values` conversion cost on every call. Where memory is
tight, the DataFrame view can be a property that builds on
demand.

### ENG-19 -- Every algorithm assumes the whole matrix fits in RAM *(Moderate)*
None of PCA / PLS / TPLS / batch features supports an out-of-core
or streaming code path. A 10M-row x 200-col matrix needs ~16 GB
in float64 -- well within plant-data scale, well outside RAM on a
laptop. **Action:** prototype an incremental PCA (sklearn has one;
the package could wrap it), or use `dask.array` / `numpy.memmap`
as an optional path. Not a near-term concern but a credibility
issue for "production-grade" claims in the README.

### ENG-30 -- The MCP dispatch is sync inside an async server *(Moderate)*
`mcp_server.py` defines `async def handler(...)` that calls
`safe_execute_tool_call` (sync, blocks the event loop on
`future.result(timeout=...)`). FastMCP is async-native; serving
multiple agents (e.g. via SSE) over a stdio bridge will queue
unnecessarily. **Action:** offload the
`ProcessPoolExecutor.submit().result()` to a thread (`await
asyncio.wrap_future(...)`) so the event loop can serve other
requests.

---

## Reproducibility

### ENG-08 -- RNG handling is inconsistent; "reproducibility" is a vibe, not a contract *(Major)*
Production code that creates an *unseeded* RNG:
- `process_improve/experiments/simulations.py:101`
  (`np.random.default_rng().normal(0, 1)`)
- `process_improve/simulation/model.py:425` (`noise_rng =
  np.random.default_rng()` -- documented as "fresh noise on every
  call"; arguably intentional)
- `process_improve/multivariate/methods.py:5770, :5794`
  (`Resampler.bootstrap` / `.fractional` -- SEC-21 sub-item 9)

Hard-coded seeds in production code:
- `process_improve/experiments/optimization.py:564` (`rng =
  np.random.default_rng(42)`)
- `process_improve/experiments/visualization/plots/surfaces.py:242`,
  `design_quality.py:85` (`seed=0`, `seed=42`).

The `@tool_spec(rng={...})` metadata is well-intentioned but
enforces nothing -- the decorator only validates the *shape* of
the metadata, not that the function obeys it. **Action:** publish
a one-page reproducibility contract:
1. Every public function that touches an RNG MUST accept
   `random_state: int | np.random.Generator | None`.
2. Hard-coded seeds in production code are forbidden (move
   defaults into function signatures).
3. The `rng={}` decorator should *unit-test itself*: a test that
   imports every registered tool, builds a random input, runs it
   twice with the declared seed, and asserts the outputs match.

---

## Documentation

### ENG-06 -- Three plotting backends for a small team *(Major)*
The codebase carries adapters for matplotlib, plotly, and Apache
ECharts (`process_improve/visualization/adapters/`). The `ChartSpec`
abstraction is well-designed and worth keeping, but maintaining
three live backends is a heavy commitment: every new chart type
must work in all three, and divergence will accumulate. **Action:**
pick one *blessed* backend (plotly is the natural fit for
interactive scientific use), maintain the others as best-effort
adapters, and document the support matrix.

### ENG-26 -- Sphinx docs and architecture overview are thin *(Moderate)*
The Sphinx setup at `docs/` is wired up, but skimming the README
shows mostly an algorithm catalogue. There's no architecture
overview, no "how to add a new MCP tool" tutorial, no
"how PCA differs from PLS for our purposes" page, no
hosted "tools cookbook" for LLM consumers, no reproducibility
contract, no error-handling guide. **Action:** flesh out
`docs/architecture.rst`, `docs/contributing.rst`,
`docs/tool_authoring.rst`, `docs/error_handling.rst`. These
double as onboarding material for the next contributor.

---

## What this audit deliberately *does not* fault

For balance:

- The `SECURITY_AUDIT.md` workflow itself is exemplary. Numbered
  findings, separate code+tests PRs per fix, a consolidated
  release commit, table that gets struck off as fixes land --
  this is the right way to do it. Most projects don't do this at
  all.
- The `MCUVScaler` and `_linalg.py` modules are small, focused,
  well-tested examples of the rest-of-codebase ideal.
- The `visualization/spec.py` + adapter approach is the right
  abstraction (the criticism in ENG-06 is about *maintaining*
  three backends, not about the design).
- The MCP tool registry has good guard rails already (timeout,
  memory cap, kill-runaway-workers, schema validator), even if
  ENG-04, ENG-10, and the SEC items show where to harden further.
- The `@tool_spec(rng={...})` metadata is forward-looking; it
  just needs the enforcement layer (ENG-08).
- The choice to ship a `CLAUDE.md` and to keep `CHANGELOG.md`
  in Keep-a-Changelog format is mature.
- The CI matrix is unusually broad (3.10..3.13 x three OSes,
  plus the experimental free-threaded builds) for a small
  project.

---

## Suggested triage order

If this list landed on my desk:

1. **ENG-15** (testing rigour) -- single biggest leverage for
   shipping quality.
2. **ENG-03** (turn the type system on) -- unblocks many of the
   rename / refactor items.
3. **ENG-13** (split runtime deps into extras) -- one-week task,
   immediate user benefit, reduces transitive blast radius.
4. **ENG-11** (error-handling style guide) -- prerequisite for
   any further SEC-NN sweep.
5. **ENG-17** + **ENG-01** (factor out a `_LatentVariableModel`
   base; split `methods.py`) -- compounds with every future
   numerical fix.
6. **ENG-10** (pydantic-derive the tool schemas) -- closes a class
   of MCP-boundary bugs (overlaps with SEC-15 / SEC-20).
7. **ENG-09** + **ENG-27** (config module) -- enabler for
   ENG-15-style perf and fuzz tests that need to override caps.
8. Everything else, opportunistically.

Each of the above is structural; none requires the SEC-NN fixes to
land first. They can be done in parallel.
