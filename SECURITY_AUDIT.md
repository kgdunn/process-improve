# Security & Robustness Audit

This document catalogues exploitable holes and robustness defects found in
`process-improve`, grouped so that each entry (or logically connected group)
can become a single GitHub issue, be fixed with a regression test, and then be
struck off here.

## Threat models

`process-improve` exposes its analysis registry as agent-callable tools
(`@tool_spec`) over an **MCP server** (`process_improve/mcp_server.py`), and the
code anticipates being "fronted by HTTP or otherwise reachable from untrusted
clients" (`PROCESS_IMPROVE_MCP_SAFE_MODE`, `tool_safety.py`). Each finding is
therefore ranked under two models:

- **U (Untrusted):** the MCP server / tool registry is reachable by hostile or
  LLM-driven callers who fully control tool inputs.
- **L (Local-trusted):** the server only ever drives the owner's own LLM on the
  owner's machine.

## Status overview

| ID | Title | U | L | Status |
|----|-------|---|---|--------|
| SEC-01 | RCE via patsy formula in `fit_linear_model` | Critical | Low | done (#238, v1.22.5) |
| SEC-02 | Timeout does not terminate runaway worker | High | Low | done (#239, v1.22.7) |
| SEC-03 | No per-call worker-pool isolation | Medium | Low | done (#239, v1.22.7) |
| SEC-04 | Declared `input_schema` never enforced | High | Low | done (#240, v1.22.6) |
| SEC-05 | Div-by-zero in NIPALS / multiblock methods | High | High | done (#241, v1.22.8) |
| SEC-06 | Non-convergence not flagged; `fractional()` 1/0 | Medium | Medium | done (#241, v1.22.8) |
| SEC-07 | Matrix inversion without conditioning checks | Medium | Medium | done (#242, v1.22.9) |
| SEC-08 | `assert` used for validation (stripped by `-O`) | Medium | Low | done (#243, v1.22.11) |
| SEC-09 | Exception suppression; tool errors leak internals | Medium | Low | done (#244, v1.22.10) |
| SEC-10 | Latent path traversal; unverified remote fetch | Low | Low | open |
| SEC-11 | `discover_tools` swallows all `ImportError`s | Low | Low | open |
| SEC-12 | `DataFrame.query` built with f-strings | Low | Low | open |

---

## SEC-01 - Arbitrary code execution via patsy formula in `fit_linear_model` [RESOLVED]
- **Status:** Fixed in v1.22.5 (issue #238). `validate_formula_is_safe` now
  rejects any non-Wilkinson formula before it reaches patsy.
- **Severity:** U = Critical, L = Low
- **Where:** `process_improve/experiments/tools.py:172` (`fit_linear_model`) ->
  `process_improve/experiments/models.py:178` (`lm`) ->
  `smf.ols(model_spec, data=data)` (`models.py:277,281`).
- **Issue:** The MCP tool `fit_linear_model` takes a free-text `formula` string
  and passes it unmodified to statsmodels/patsy. Patsy evaluates each formula
  term as a Python expression in an environment with builtins and numpy, so a
  formula such as `y ~ I(__import__('os').system('id'))` (or any callable term)
  executes arbitrary code. The `safe_execute_tool_call` sandbox limits time and
  memory but does not restrict what code runs (file read, outbound network), so
  it is not a security boundary against code execution.
- **Fix direction:** Do not feed raw user formulas to patsy. Validate `formula`
  against a strict allowlist grammar (known column names; operators
  `+ - * : ^`; integer powers; `I()`/`Q()` of a bare column only) and reject any
  token with a function call `(`, attribute access `.`, or dunder. Prefer
  building the RHS from structured terms (the shorthand mapping already exists in
  `experiments/evaluate.py:95`). Add tests asserting malicious formulas are
  rejected before reaching patsy.

## SEC-02 - Timeout guard does not terminate runaway workers (CPU-exhaustion DoS) [RESOLVED]
- **Status:** Fixed in v1.22.7 (issue #239). `shutdown_pool` now force-terminates
  workers; `safe_execute_tool_call` recycles the pool after every call.
- **Severity:** U = High, L = Low
- **Where:** `process_improve/tool_safety.py:344-353` (`safe_execute_tool_call`),
  `shutdown_pool` (`tool_safety.py:279-285`).
- **Issue:** On timeout the code calls `shutdown_pool()`, which does
  `ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)`. That does not
  kill a worker already running a task; `cancel_futures` only drops
  not-yet-started futures. A CPU-bound or infinite-loop tool keeps a core pegged
  after `ToolTimeoutError` is raised, contradicting the docstring ("the current
  subprocess is terminated"). The pool defaults to `max_workers=1`, so each
  timeout both leaks a runaway process and forces a fresh pool; repeated calls
  accumulate live busy processes.
- **Fix direction:** On timeout, forcibly terminate the worker process(es) (hold
  the underlying `multiprocessing.Process` and call `.terminate()`/`.kill()`,
  then join) before raising. Correct the docstring. Add a regression test that
  submits a busy-loop tool and asserts the child process is gone after timeout.

## SEC-03 - Worker pool has no per-call isolation / memory reset [RESOLVED]
- **Status:** Fixed in v1.22.7 (issue #239). The module pool is recycled after
  every call. (`max_tasks_per_child=1` is incompatible with the `fork` start
  method this module uses, so per-call recycling is used instead.)
- **Severity:** U = Medium, L = Low
- **Where:** `process_improve/tool_safety.py:259-276` (`get_pool`),
  `_pool_initializer`, `_apply_memory_limit`.
- **Issue:** The Linux `fork` pool reuses one long-lived worker across calls and
  clients. Process-global state (numpy global RNG seeds, module registries,
  matplotlib state) persists between unrelated tool calls. `RLIMIT_AS` is
  cumulative address space and Python does not reliably return memory to the OS,
  so a later legitimate call can spuriously hit `ToolMemoryExceededError`.
- **Fix direction:** Set `max_tasks_per_child=1` (or recycle the pool every N
  calls) so each tool call gets a clean process; document the isolation
  guarantee. Can be implemented together with SEC-02 as "pool hardening."

## SEC-04 - Declared `input_schema` is never enforced at dispatch [RESOLVED]
- **Status:** Fixed in v1.22.6 (issue #240). `safe_execute_tool_call` now calls
  `validate_against_schema` (types, bounds, item counts, enum, required, unknown
  keys) before dispatch; the in-process `execute_tool_call` fast path is unchanged.
- **Severity:** U = High, L = Low
- **Where:** `process_improve/tool_spec.py:310-335` (`execute_tool_call`) and
  `process_improve/tool_safety.py:293-339` (`safe_execute_tool_call`), both ending
  in `_TOOL_REGISTRY[tool_name](**tool_input)`.
- **Issue:** Inputs pass straight through as `**kwargs`; the per-tool JSON Schema
  (types, `minimum`/`maximum`, `minItems`, `enum`, `required`,
  `additionalProperties`) is advisory and never validated. `_SCALAR_CAPS`
  (`tool_safety.py:65-73`) caps only a hard-coded handful of keys; other
  cost-driving parameters (design size, `conf_level`, factor counts) are
  unbounded and types are unchecked, allowing type confusion, silent wrong
  results, or expensive work below the cell cap.
- **Fix direction:** Validate `tool_input` against the tool's own `input_schema`
  before dispatch (e.g. `jsonschema`, or a pydantic model derived from the
  schema): enforce types, bounds, `enum`, `required`, and reject
  `additionalProperties`. This also closes the `_SCALAR_CAPS` bypass. Add tests
  for type/bounds rejection and unknown-key rejection.

## SEC-05 - Unguarded division by zero in multivariate NIPALS / multiblock methods [RESOLVED]
- **Status:** Fixed in v1.22.8 (issue #241). NIPALS denominators are floored via
  `_nz`; zero-variance R-squared is reported as NaN.
- **Severity:** U = High, L = High (correctness)
- **Where:** `process_improve/multivariate/methods.py` around lines 4444-4458,
  5277-5283, 5435-5439 (e.g. `/(p_b @ p_b)`, `/(t_super @ t_super)`,
  `/(c_a @ c_a)`), and the `np.where(... > 0, ..., 1.0)` R-squared fallbacks at
  5309-5311, 5378, 5458.
- **Issue:** Loading/score dot-product denominators are divided without a
  near-zero guard; degenerate or collinear blocks yield silent `inf`/`nan` that
  poison the fitted model with no warning. The R-squared `1.0` fallback for
  all-zero columns is semantically wrong (should be undefined/flagged).
- **Fix direction:** Guard denominators with an epsilon / `np.isclose` check;
  raise or flag a structured non-convergence condition; record degenerate columns
  in `fitting_info_` instead of silently substituting `1.0`. Add tests with
  collinear / all-constant columns.

## SEC-06 - Iterative algorithms do not flag non-convergence; `fractional()` divide-by-zero [RESOLVED]
- **Status:** Fixed in v1.22.8 (issue #241). `fitting_info_["converged"]` plus a
  `SpecificationWarning`; `fractional()` re-validates `fraction_excluded`.
- **Severity:** U = Medium, L = Medium (correctness)
- **Where:** `process_improve/multivariate/methods.py:4434`, `5258` (NIPALS
  `while ... and itern < self.max_iter`), `5740`
  (`n_groups = int(1 / self.fraction_excluded)`).
- **Issue:** When NIPALS hits `max_iter` without converging it returns a
  non-converged solution silently (no `fitting_info_` flag, no warning).
  `fractional()` divides by `self.fraction_excluded`; the `__init__` guard
  (`5673`) is bypassable by mutating the attribute to `0.0` before calling,
  giving `ZeroDivisionError`.
- **Fix direction:** Record convergence status (iterations, converged bool) in
  `fitting_info_` and warn on non-convergence; re-validate `fraction_excluded`
  inside `fractional()`. Tests: pathological data hitting `max_iter`;
  `fraction_excluded = 0` after init.

## SEC-07 - Matrix inversion without conditioning / singularity checks [RESOLVED]
- **Status:** Fixed in v1.22.9 (issue #242). New `process_improve._linalg`
  (`safe_inverse` / `is_singular`) guards the two previously-unguarded
  multivariate sites and upgrades the surface/design-quality plot fallbacks to
  catch ill-conditioning. `optimal.py` and `evaluate.py` were already guarded
  (try/except and rank check).
- **Severity:** U = Medium, L = Medium (correctness)
- **Where:** `process_improve/experiments/optimal.py:21,91`,
  `process_improve/experiments/evaluate.py:137`,
  `process_improve/multivariate/methods.py:1618,3841`,
  `process_improve/experiments/visualization/plots/surfaces.py:474`,
  `process_improve/experiments/visualization/plots/design_quality.py:77`.
- **Issue:** `np.linalg.inv(XtX)` is applied to matrices that can be
  singular/ill-conditioned (rank-deficient designs, collinear factors),
  producing overflow-driven garbage rather than a clear error.
- **Fix direction:** Use `np.linalg.solve`/`lstsq`/`pinv` when solving a system,
  or check `np.linalg.cond` / rank first and raise a clear
  "singular/rank-deficient design" error. Tests with a deliberately singular
  design matrix.

## SEC-08 - `assert` used for input/state validation (stripped under `python -O`) [RESOLVED]
- **Status:** Fixed in v1.22.11 (issue #243). The flagged validation asserts were
  converted to explicit `raise` statements; remaining asserts are internal
  invariants.
- **Severity:** U = Medium, L = Low
- **Where:** ~105 instances; validation-style examples:
  `process_improve/regression/methods.py:486-489,615`,
  `process_improve/experiments/optimal.py:79-80`,
  `process_improve/batch/data_input.py:67,74,81,86,104,146,152`,
  `process_improve/monitoring/control_charts.py:104-105,181-186`,
  `process_improve/univariate/metrics.py:397,583,733-734`.
- **Issue:** Public-facing argument/shape validation done via `assert` is removed
  when Python runs with `-O`, silently disabling the checks and allowing invalid
  input to corrupt results or crash later in an obscure place.
- **Fix direction:** Convert validation asserts (those checking user-supplied
  arguments or external data) to explicit `if not ...: raise
  ValueError/TypeError(...)`. Genuine internal invariants may remain asserts but
  prefer explicit raises at API boundaries. Tests: invalid input raises even
  under `-O`.

## SEC-09 - Broad exception suppression hides errors; tool errors leak internals [RESOLVED]
- **Status:** Fixed in v1.22.10 (issue #244). `analysis.py` and `augment.py`
  narrow their `except` and log; the MCP server logs unexpected exceptions
  server-side and returns a generic message instead of `str(exc)`. (Individual
  tool wrappers already log server-side and mostly carry domain-validation
  messages; SEC-04 now rejects malformed inputs before dispatch.)
- **Severity:** U = Medium (information disclosure), L = Low
- **Where:** `process_improve/experiments/analysis.py:191` (`except Exception` ->
  `None, None`), `process_improve/experiments/augment.py:69` (`except Exception`
  -> `{}`); plus every tool wrapper's `except Exception: return {"error":
  str(exc)}` and the MCP handler `process_improve/mcp_server.py:93-94`.
- **Issue:** Broad `except Exception` blocks swallow real bugs (type errors,
  library faults) indistinguishably from expected failures. The tool/MCP layer
  returns `str(exc)` verbatim to the caller, which under the untrusted model
  leaks file paths, library internals, and stack details.
- **Fix direction:** Narrow the caught types and log unexpected ones in
  `analysis.py`/`augment.py`; at the tool/MCP boundary, return a generic error
  code/message to the caller while logging detail server-side (the structured
  `ToolSafetyError.to_dict()` pattern is a good model). Tests: forced failure
  returns a sanitized error with no path leakage.

## SEC-10 - Latent path traversal in file helpers; unverified remote dataset fetch
- **Severity:** U = Low, L = Low
- **Where:** `process_improve/experiments/knowledge/engine.py:35-42`
  (`_load_yaml(filename)` joins `_DATA_DIR / filename`, no traversal check),
  `process_improve/batch/plotting.py:37` (`plot_to_HTML(filename, ...)` writes to
  an arbitrary path), `process_improve/experiments/datasets.py:30,94`
  (`pd.read_csv("https://openmv.net/...")`).
- **Issue:** The two helpers accept unsanitized filenames (all current callers
  pass hardcoded literals, so not presently reachable from untrusted input, but
  they are footguns). The dataset loaders fetch CSVs from a remote host with no
  integrity check or graceful failure; a content MITM or host change can feed
  arbitrary data and an outage raises an opaque error.
- **Fix direction:** In the helpers, resolve the path and assert it stays within
  the intended base dir (`Path.resolve().is_relative_to(base)`); raise on
  traversal. For datasets, wrap remote fetches with clear error handling and
  document the trust assumption (optionally checksum-pin). Tests: traversal
  filename rejected.

## SEC-11 - `discover_tools` silently swallows all `ImportError`s
- **Severity:** U = Low, L = Low (robustness)
- **Where:** `process_improve/tool_spec.py:255-269`
  (`contextlib.suppress(ImportError)` around each `import_module`).
- **Issue:** A genuine bug inside a `tools.py` module (a bad import nested deep in
  the module) raises `ImportError` and is silently suppressed, so that entire
  tool category vanishes from the registry with no signal, producing
  hard-to-diagnose "missing tool" failures.
- **Fix direction:** Only tolerate a missing top-level optional dependency
  (`ModuleNotFoundError` whose name is the module itself); log a warning and
  surface unexpected import failures. Test: a module raising a nested ImportError
  surfaces a warning rather than disappearing silently.

## SEC-12 - `pandas.DataFrame.query` built with f-strings
- **Severity:** U = Low, L = Low (code smell)
- **Where:** `process_improve/batch/preprocessing.py:637,639`
  (`metrics.query(f"SPE < {pca_second.spe_limit(...)}")`).
- **Issue:** Injecting computed values into a `query` expression string. The
  values here are internal numerics (not user input), so not currently
  exploitable, but `DataFrame.query` evaluates expressions and the pattern is
  fragile.
- **Fix direction:** Replace with boolean-mask indexing or `@`-variable binding
  so no expression string is assembled. Test: equivalent filtering result.

---

## Out of scope / checked clean

No `eval`/`exec`/`compile`/`pickle`/`yaml.unsafe_load`/`shell=True`/XXE or
hardcoded secrets were found. CI workflows use trusted publishing and do not
expose secrets to fork pull requests. CSV/Excel parsing uses pandas defaults (no
formula evaluation). Tools accept structured data, not file paths or URLs, and
YAML loading uses `yaml.safe_load`.

## Workflow

1. One GitHub issue per `SEC-NN` (or per logically grouped pair, e.g.
   SEC-02 + SEC-03 "pool hardening"; SEC-05 + SEC-06 "multivariate numerical
   robustness").
2. For each: write a failing repro/test, apply the minimal fix, convert the
   repro to a regression test, run the relevant `pytest` plus `ruff check .`,
   bump the version per `CLAUDE.md`, update `CHANGELOG.md`, then close the issue
   and set its row above to `done` (or remove it).
