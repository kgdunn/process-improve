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
| SEC-08 | `assert` used for validation (stripped by `-O`) | Medium | Low | done (#243, v1.22.9) |
| SEC-09 | Exception suppression; tool errors leak internals | Medium | Low | done (#244, v1.22.9) |
| SEC-10 | Latent path traversal; unverified remote fetch | Low | Low | done (#245, v1.22.9) |
| SEC-11 | `discover_tools` swallows all `ImportError`s | Low | Low | done (#246, v1.22.9) |
| SEC-12 | `DataFrame.query` built with f-strings | Low | Low | done (#247, v1.22.9) |
| SEC-13 | RCE via patsy formula in `analyze_experiment`/`evaluate_design`/`augment_design`/`lm` | Critical | Low | open |
| SEC-14 | `reveal_simulator` confirmation gate bypassable via kwarg injection | Critical | Low | open |
| SEC-15 | `TPLS.score` raises `NameError` when the `Y` dict is empty | High | High | open |
| SEC-16 | `assert` for validation in multivariate/bivariate/batch/structures (SEC-08 follow-up) | Medium | Low | open |
| SEC-17 | Tool wrappers still leak raw exception text via `{"error": str(e)}` | High | Low | open |
| SEC-18 | MCP DoS - unbounded combinatorial generators, O(N^2) regression, unbounded matrices | High | Medium | open |
| SEC-19 | `validate_against_schema` gaps (`oneOf`, nested items, str-encoded numerics, non-object root) | High | Low | open |
| SEC-20 | NaN-poisoning in single-block PCA/PLS/TPLS (SEC-05 follow-up) | High | High | open |
| SEC-21 | Holt-Winters control chart divides by zero on constant warm-up window | High | High | open |
| SEC-22 | `regression.OLS.predict` accepts wrong-shape `X` silently | High | Medium | open |
| SEC-23 | `confidence_interval`, paired `t_value`, and `calculate_cpk` crash on n <= 1 / zero spread | Medium | Medium | open |
| SEC-24 | `pca_predict` / `pls_predict` accept untrusted `model_params` (no caps, no integrity) | High | Low | open |
| SEC-25 | `analyze_experiment` `transform="inverse"` divides by user data | Medium | Low | open |
| SEC-26 | Quadratic-term regex misses the `np.power(A, 2)` form | Low | Medium | open |
| SEC-27 | Simulator seed entropy truncated to 31 bits | Medium | Low | open |
| SEC-28 | `_SIGNIFICANT_FACTOR_PATTERN` is O(n^2) on multi-KB input | Medium | Low | open |
| SEC-29 | Knowledge YAML loader has no file-size cap (anchor-bomb DoS) | Low | Low | open |
| SEC-30 | `_terminate_workers` relies on CPython private `_processes` attribute | Low | Low | open |
| SEC-31 | `json.loads(key)` in batch plotting raises unhandled `JSONDecodeError` | Low | Low | open |
| SEC-32 | Miscellaneous numerical / correctness cleanup | Low | Medium | open |

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
- **Status:** Fixed in v1.22.9 (issue #243). The flagged validation asserts were
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
- **Status:** Fixed in v1.22.9 (issue #244). `analysis.py` and `augment.py`
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

## SEC-10 - Latent path traversal in file helpers; unverified remote dataset fetch [RESOLVED]
- **Status:** Fixed in v1.22.9 (issue #245). `_load_yaml` confines the resolved
  path to its data directory; the remote dataset loaders wrap the fetch and raise
  a clear error. `plot_to_HTML` is an explicit save-to-path API (the path is the
  intended output), so it is left as-is by design.
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

## SEC-11 - `discover_tools` silently swallows all `ImportError`s [RESOLVED]
- **Status:** Fixed in v1.22.9 (issue #246). Only `ModuleNotFoundError` is
  tolerated (and logged); other `ImportError`s propagate.
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

## SEC-12 - `pandas.DataFrame.query` built with f-strings [RESOLVED]
- **Status:** Fixed in v1.22.9 (issue #247). Replaced with boolean-mask indexing.
- **Severity:** U = Low, L = Low (code smell)
- **Where:** `process_improve/batch/preprocessing.py:637,639`
  (`metrics.query(f"SPE < {pca_second.spe_limit(...)}")`).
- **Issue:** Injecting computed values into a `query` expression string. The
  values here are internal numerics (not user input), so not currently
  exploitable, but `DataFrame.query` evaluates expressions and the pattern is
  fragile.
- **Fix direction:** Replace with boolean-mask indexing or `@`-variable binding
  so no expression string is assembled. Test: equivalent filtering result.

## SEC-13 - Arbitrary code execution via patsy formula in additional tool sinks
- **Status:** Open. SEC-01 plugged `fit_linear_model`; the same class of bug
  remains in three other tools and in `lm()` itself.
- **Severity:** U = Critical, L = Low
- **Where:**
  - `process_improve/experiments/tools.py:630-661` ->
    `process_improve/experiments/analysis.py:713-714`
    (`analyze_experiment_tool`: `model`, `response_column`, *and*
    attacker-controlled dict keys in `design_matrix` reach
    `smf.ols(build_formula(...), data=df).fit()`).
  - `process_improve/experiments/tools.py:496-521` ->
    `process_improve/experiments/evaluate.py:104-114`
    (`evaluate_design_tool`: `model` flows into
    `patsy.dmatrix(rhs, design_df, return_type="dataframe")`).
  - `process_improve/experiments/tools.py:980-1007` ->
    `process_improve/experiments/augment.py:415-425, 445`
    (`augment_design_tool`: `target_model` flows into
    `_greedy_d_optimal_select` -> `dmatrix(rhs, trial, ...)` via
    `_build_model_rhs`, whose final branch is `return model`).
  - `process_improve/experiments/models.py:244-345` (`lm()`: validation lives in
    `fit_linear_model`, *not* in `lm()`, so any other caller that hands an
    untrusted formula to `lm` is exposed).
- **Issue:** Patsy / statsmodels evaluate every formula term as a Python
  expression with builtins + numpy in scope. The schema's `enum` on `model`
  is only enforced in `PROCESS_IMPROVE_MCP_SAFE_MODE`; the default stdio MCP
  path skips it entirely. Even in safe mode, `response_column`
  (`{"type":"string"}`, no `pattern`) and *dict keys* in `design_matrix`
  (`{"type":"object","additionalProperties":{"type":"number"}}` constrains
  values, not keys) are interpolated into the formula and run as Python.
- **Fix direction:** Call `validate_formula_is_safe(formula, df.columns)`
  inside `lm()` itself and inside `_build_model_matrix` / `_build_model_rhs`
  (against `design_df.columns` / `factor_names`) immediately before
  `smf.ols(...)` / `dmatrix(...)`. Also validate each column name and
  `response_column` with the same identifier regex. Tests: malicious
  formulas, malicious column names, malicious `response_column` are all
  rejected before reaching patsy.

## SEC-14 - `reveal_simulator` confirmation gate bypassable via kwarg injection
- **Status:** Open.
- **Severity:** U = Critical, L = Low
- **Where:** `process_improve/simulation/tools.py:370-414` (and
  `simulate_process` at `:305-336`); dispatch via
  `process_improve/tool_spec.py:327-352`
  (`_TOOL_REGISTRY[tool_name](**tool_input)`).
- **Issue:** `reveal_simulator(*, sim_id, simulator_state=None,
  confirmed=False)` deliberately omits `simulator_state` and `confirmed`
  from the JSON schema so the host can inject them. `execute_tool_call`
  forwards `**tool_input` verbatim. Under the default (non-safe) MCP path
  there is no schema validation, so a prompt-injected agent can call
  `reveal_simulator(sim_id="...", confirmed=True, simulator_state={...})`,
  bypassing the double-confirmation gate and feeding the function a
  fabricated state (whose contents are echoed back). Same trick lets
  `simulate_process` accept an attacker-supplied `simulator_state`.
- **Fix direction:** In `execute_tool_call` (both the safe and unsafe path)
  filter `tool_input` down to the keys declared in
  `input_schema["properties"]` *before* invoking the function. Independently,
  move `simulator_state` / `confirmed` out of the function signature and
  pass them through a side channel (e.g. a contextvar populated by the host
  before dispatch) so they cannot be re-introduced as kwargs even if the
  registry filter regresses. Tests: forwarding `confirmed=True` /
  `simulator_state=...` from the dispatch path raises
  `ToolInputInvalidError` (or is silently dropped) and the gate still
  fires.

## SEC-15 - `TPLS.score` raises `NameError` when the `Y` dict is empty
- **Status:** Open.
- **Severity:** U = High (crash), L = High (crash on legitimate edge case)
- **Where:** `process_improve/multivariate/methods.py:~3496-3499`

  ```python
  for _idx, key in enumerate(y_actual):
      r2_key += r2_score(...)
  return r2_key / (_idx + 1)
  ```
- **Issue:** When `y_actual` (i.e. `X["Y"]`) is an empty dict the loop body
  never runs, `_idx` is never bound, and the `return` line raises
  `NameError: name '_idx' is not defined`. Any malformed test bundle
  triggers it.
- **Fix direction:** Initialise `count = 0` before the loop, increment
  inside, and `return r2_key / count` (or raise `ValueError("y_actual is
  empty")` if `count == 0`). Test: `TPLS.score` on an empty Y dict raises
  a clear error.

## SEC-16 - Validation `assert`s missed by SEC-08 (stripped under `python -O`)
- **Status:** Open. SEC-08 swept ~105 sites but the multivariate / batch /
  bivariate / experiments modules still carry validation asserts.
- **Severity:** U = Medium, L = Low (silent wrong results under `-O`)
- **Where:**
  - `process_improve/multivariate/methods.py` -- `:968`, `:992`, `:1599`,
    `:1823`, `:2625-2626`, `:2678-2679`, `:2752-2755`, `:3064-3073`,
    `:3100`, `:3280`, `:3316-3324`, `:3541-3552`, `:4274-4275`,
    `:5173-5174` (and `:690-691`).
  - `process_improve/bivariate/methods.py:49, 57`.
  - `process_improve/univariate/tools.py:574, 660`.
  - `process_improve/experiments/structures.py:82, 279, 301`.
  - `process_improve/batch/features.py:67, 325, 599-600`.
  - `process_improve/batch/plotting.py:29, 141, 143, 408`.
  - `process_improve/batch/preprocessing.py:331-334, 593-595`.
  - `process_improve/monitoring/metrics.py:52-53`.
  - `process_improve/experiments/evaluate.py:301` (internal invariant -
    keep as `assert` with a comment, or convert to `RuntimeError`).
- **Issue:** Every `assert` listed validates user-supplied arguments or
  external data and is stripped by `python -O`. Most prominently
  `PCA.transform` / `PCA.predict` / `PLS.predict` lose their shape check
  and silently accept the wrong-shape input.
- **Fix direction:** Same template as SEC-08:
  ```python
  if X.shape[1] != self.n_features_in_:
      raise ValueError(
          f"Prediction data must have {self.n_features_in_} columns, "
          f"got {X.shape[1]}."
      )
  ```
  Where the `assert` truly is an internal invariant, leave it but add a
  `# noqa: S101` comment that names the invariant. Tests: each public
  entry point still raises under `python -O` (run the relevant test file
  with `python -O -m pytest`).

## SEC-17 - Tool wrappers still leak raw exception text via `{"error": str(e)}`
- **Status:** Open. SEC-09 fixed the MCP server's outer handler and the two
  worst offenders in `analysis.py` / `augment.py`; ~25 individual tool
  wrappers still catch `Exception` and serialise `str(exc)` directly,
  bypassing `_serialise_tool_error`.
- **Severity:** U = High (information disclosure), L = Low
- **Where:**
  - `process_improve/experiments/tools.py:110-112, 220-222, 390-392,
    519-521, 659-661, 861-863, 1005-1007, 1156-1158, 1268-1270,
    1459-1460`.
  - `process_improve/multivariate/tools.py:132-133, 249-250, 312-313,
    390-391, 483-484, 580-581`.
  - `process_improve/monitoring/tools.py:114-115, 202-203`.
  - `process_improve/regression/tools.py:121-122, 175-176`.
  - `process_improve/bivariate/tools.py:81-82`.
  - `process_improve/batch/tools.py:169-170`.
  - `process_improve/visualization/tools.py:278-280`.
- **Issue:** Library exceptions (`pandas`, `numpy`, `statsmodels`,
  `scipy`) carry filesystem paths and library internals in the message
  text. `mcp_server._serialise_tool_error` only runs for *unhandled*
  exceptions; the per-tool `except` catches first and leaks `str(exc)`.
- **Fix direction:** Either (a) drop the broad `except Exception` block
  and let the exception propagate to `_serialise_tool_error`, or (b)
  narrow to `(ValueError, KeyError, LinAlgError, ...)` and route any
  remaining `Exception` through `_serialise_tool_error` with a
  server-side log. Pattern to follow: the PCA/PLS tools that already
  narrow to `(ValueError, TypeError, KeyError, LinAlgError)`. Tests:
  forced library failure (e.g. malformed numeric coercion) returns a
  generic message; the raw message appears only in the server log.

## SEC-18 - MCP DoS surface: unbounded inputs and missing scalar caps
- **Status:** Open. SEC-04 plugged the per-tool schema gap for declared
  parameters, but many parameters that drive algorithm cost remain
  unbounded.
- **Severity:** U = High, L = Medium
- **Where:**
  1. **Combinatorial design generators:**
     `process_improve/experiments/designs.py:~45` (`ff2n(k)` allocates
     `2**k` rows -- 40 factors -> 2^40 rows);
     `process_improve/experiments/designs_optimal.py:~217`
     (`fullfact([3] * k)` -- 3^k candidate set);
     `process_improve/experiments/designs_mixture.py:~48, ~80`
     (`2**k - 1` simplex-centroid; `itertools.product(grid, repeat=k)`
     blows up for `(degree+1)**k`).
  2. **O(N^2) regression kernels:**
     `process_improve/regression/tools.py:76-122, 162-176` ->
     `process_improve/regression/methods.py:~73-80`
     (`repeated_median_slope` is quadratic, no `maxItems` on `x`/`y`;
     `DEFAULT_MAX_CELLS=1_000_000` is far too generous).
  3. **Unbounded matrix dimensions:**
     `process_improve/multivariate/tools.py:90-133, 197-250, 293-313,
     363-391` (`data` / `x_data` schemas have only `minItems`; a 1 x 1M
     matrix passes `max_cells=1_000_000` and blows up SVD).
  4. **Missing `_SCALAR_CAPS` keys:**
     `process_improve/tool_safety.py:66-74` (no caps for `n_steps`,
     `n_additional_runs`, `center_points`, `replicates`).
  5. **Unbounded `fit_linear_model` data + formula expansion:**
     `process_improve/experiments/tools.py:172-225`
     (`formula="y ~ (A+B+C+D+E)**5"` expands to 2^5 terms; combined
     with large `data` is a CPU sink).
- **Fix direction:** Cap `k` at the entry of each combinatorial
  generator (e.g. 15) and mirror as `maxItems` on the tool schema. Add
  `maxItems` (rows and inner cols) on every matrix-shaped input. Add
  cost-driving keys to `_SCALAR_CAPS`. For `fit_linear_model`, cap
  `formula` length and the number of expanded terms after patsy parses
  the RHS. Tests: each cap fires with a clear structured error before
  any expensive work starts.

## SEC-19 - `validate_against_schema` / `validate_input` gaps
- **Status:** Open. SEC-04 introduced the validator; the following
  schema features are still silently bypassed.
- **Severity:** U = High, L = Low
- **Where:** `process_improve/tool_safety.py:194-285, 310-311`.
  1. `validate_input`'s `_SCALAR_CAPS` check at `:194-201` only
     compares numeric values, so `{"n_components": "50000"}` slips
     through (any downstream `int(...)` coercion accepts it).
  2. `validate_against_schema._validate_value` at `:237-285` only looks
     at top-level `type` / `enum` / `minimum` / `maximum` /
     `minItems`/`maxItems`. `oneOf` (used by
     `evaluate_design.metric` and `analyze_experiment.analysis_type`)
     and nested `items` / `properties` / `additionalProperties` are
     silently unchecked.
  3. `validate_against_schema` at `:310-311` returns silently when
     `schema.get("type") != "object"` -- a non-object schema disables
     validation entirely.
  4. `evaluate_design.metric` array branch at
     `process_improve/experiments/tools.py:436-464` is
     `{"type":"array","items":{"type":"string"}}` with no inner enum.
- **Issue:** Any of the above gaps re-opens the validation hole that
  SEC-04 was meant to close. In particular, the string-vs-number gap
  lets a caller bypass `_SCALAR_CAPS["n_components"]` and other CPU
  caps.
- **Fix direction:**
  1. Coerce-and-cap or raise `ToolInputInvalidError` when a capped key
     carries a non-numeric value.
  2. Support `oneOf` (validate against the first matching branch,
     reject if none match); recurse into `items`, `properties`, and
     `additionalProperties`.
  3. Fail-closed (raise) on unknown root types.
  4. Add the enum to the array branch of every `oneOf` metric/list
     parameter.
  Tests: each gap rejected with a clear structured error.

## SEC-20 - NaN-poisoning in single-block PCA / PLS / TPLS (SEC-05 follow-up)
- **Status:** Open. SEC-05 fixed multiblock NIPALS divisions; the
  single-block PLS and TPLS paths and several adjacent helpers still
  divide by quantities that can be zero on legitimate-but-degenerate
  inputs, producing silent `inf` / `NaN` loadings, scores, and limits.
- **Severity:** U = High, L = High (silent wrong models)
- **Where:**
  1. `process_improve/multivariate/methods.py:~2783`
     (`w_i = w_i / np.linalg.norm(w_i)`),
     `:~2794` (convergence ratio `||u_i - u_new|| / ||u_i||`),
     same pattern at `:~4022`, `:~4048`, `:~3608-3613`, `:~4081`.
     Fix: floor every norm with `_nz(...)`.
  2. PCA NIPALS mutates caller's column-0 NaNs in-place at
     `:~851` (`t_a_guess = Xd[:, [0]]` is a view; `[np.isnan(...)] = 0`
     poisons `Xd[:, 0]` before the algorithm runs). Fix:
     `t_a_guess = Xd[:, [0]].copy()` (matches PLS at `:1516-1517`).
  3. `spe_calculation` divides by zero at `:~2685-2689`
     (`g = variance_spe / (2 * center_spe)`; `h = ... / variance_spe`).
     Fix: short-circuit when either is 0.
  4. `r2_per_variable_` divides by `prior_ssx_col` (which can be 0)
     at `:~818, ~875, ~1719`. Fix:
     `np.where(prior_ssx_col > 0, 1 - col_ssx / prior_ssx_col, np.nan)`.
  5. Score-contribution weighting divides by
     `sqrt(explained_variance_[idx])` without guard at `:~1207, ~2060,
     ~4757, ~5602`. Fix: clamp divisor.
  6. `explained_variance_` divides by `N - 1` with no `N >= 2` guard
     at `:~822, ~3457` (MBPLS / MBPCA already use `max(1, N-1)`).
  7. `quick_regress` silently returns the un-normalised numerator at
     `:~2486-2511` (`if abs(denom) > epsqrt: b[k] /= denom`; else
     branch leaves `b[k]` as `sum(x*y)`). Fix: `b[k] = 0.0` in the
     else branch.
  8. `np.argmin` on all-NaN RMSECV at `:~1989` silently returns
     `n_components = 1`. Fix: check `np.all(np.isnan(...))` first.
  9. `Resampler.bootstrap` / `.fractional` at `:~5770, ~5794` use
     `np.random.default_rng()` (no seed) despite documenting
     `random_state`. Fix: thread `random_state` through.
- **Issue:** Each path turns a legitimate degenerate input
  (rank-deficient, constant column, single sample) into a silently
  poisoned fitted model rather than a clear error.
- **Fix direction:** Apply the floor / guard / clamp at each site as
  noted. Tests: collinear, all-constant, single-row, all-NaN inputs
  raise (or are clearly flagged in `fitting_info_`) instead of
  returning NaN.

## SEC-21 - Holt-Winters control chart divides by zero on constant warm-up window
- **Status:** Open.
- **Severity:** U = High, L = High (silent wrong control-chart limits)
- **Where:** `process_improve/monitoring/control_charts.py:~299-302`
  (`rho_i = error_i / df["sigma_hat"][i - 1]`;
  `psi_i = error_i / sigma_i`).
- **Issue:** A constant (zero-variance) warm-up window makes
  `sigma_0 = MAD = 0`, so `rho_i = +/-inf`, propagating into the
  computed control-chart limits which become `0` / `NaN`. No guard.
- **Fix direction:** Clamp `sigma_0`/`sigma_hat` to a small positive
  floor (e.g. `max(sigma_0, eps * abs(target))`) before dividing, or
  raise `ValueError("variance is zero in warm-up window; supply more
  representative data")`. Test: constant warm-up series produces a
  clear error rather than NaN limits.

## SEC-22 - `regression.OLS.predict` accepts wrong-shape `X` silently
- **Status:** Open.
- **Severity:** U = High, L = Medium
- **Where:** `process_improve/regression/methods.py:~642-653`
  (`return intercept + X_arr @ self.coefficients_`).
- **Issue:** `n_features_in_` is stored at fit time but never checked
  at predict; numpy raises a confusing `ValueError` (or, worse, chains
  shapes and produces wrong output).
- **Fix direction:** Validate
  `X_arr.shape[1] == self.n_features_in_` and raise a clear
  `ValueError`, mirroring the sklearn-style `check_is_fitted` pattern
  used by PCA / PLS. Test: predict with wrong column count raises.

## SEC-23 - `confidence_interval`, paired `t_value`, and `calculate_cpk` crash on n <= 1 / zero spread
- **Status:** Open.
- **Severity:** U = Medium, L = Medium
- **Where:**
  - `process_improve/univariate/metrics.py:~452-453`
    (`confidence_interval`: `t_value(1 - (1-conflevel)/2, n - 1)`,
    divides by `np.sqrt(n)`; `n == 0` -> `1/0`; `n == 1` -> NaN).
  - `process_improve/univariate/metrics.py:~344-345`
    (paired `t_value`: `sd_z_variate = diff_svar * np.sqrt(1.0 /
    (dof + 1))`; `dof == -1` divides by zero).
  - `process_improve/monitoring/metrics.py:~73-81` (`calculate_cpk`
    returns `inf` / `NaN` silently on a constant column).
- **Fix direction:** Add explicit `n >= 2` (or `>= 1` paired) early
  return / raise; add explicit zero-spread branch in `calculate_cpk`
  that returns `inf` (or `NaN`) with a warning. Tests: each function
  raises (or returns a documented value) for degenerate input.

## SEC-24 - `pca_predict` / `pls_predict` accept untrusted `model_params`
- **Status:** Open.
- **Severity:** U = High, L = Low
- **Where:** `process_improve/multivariate/tools.py:430-484, 524-581`.
- **Issue:** `model_params` schema is `{"type":"object"}` (no
  `properties`, no caps). `loadings`, `train_spe_values`, `n_components`,
  `n_samples` are read straight from the attacker-controlled dict and
  used to allocate arrays. The per-key cap on `n_components` in
  `_SCALAR_CAPS` only applies to the top-level kwarg, not to a
  nested `model_params.n_components`.
- **Fix direction:** Either
  1. require a fresh `fit_pca` / `fit_pls` per call (best -- removes
     the trust boundary entirely), or
  2. enumerate sub-keys in the schema with `maxItems`, bound array
     dimensions before `np.array(...)`, and HMAC-sign the returned
     `model_params` blob server-side so a tampered blob is rejected.
  Tests: tampered / oversized `model_params` returns a structured
  error before any allocation.

## SEC-25 - `analyze_experiment` `transform="inverse"` divides by user data
- **Status:** Open.
- **Severity:** U = Medium, L = Low
- **Where:** `process_improve/experiments/analysis.py:~703-704`
  (`df[response_col] = 1.0 / df[response_col]`).
- **Issue:** A zero in the response column produces `inf`; the
  subsequent `LinAlgError` text is currently leaked via the broad
  `except` in the tool wrapper (compare SEC-17).
- **Fix direction:** Pre-check `(df[response_col] != 0).all()` for the
  `inverse` transform and raise a structured `ValueError`. Test: zero
  response raises a clear error rather than `inf` / leaking
  `LinAlgError` text.

## SEC-26 - Quadratic-term regex misses the `np.power(A, 2)` form
- **Status:** Open. Logic bug rather than security issue.
- **Severity:** U = Low, L = Medium (silent wrong surfaces)
- **Where:** `process_improve/experiments/optimization.py:~66` and
  `process_improve/experiments/visualization/plots/surfaces.py:~57`
  (`re.match(r"I\((\w+)\s*\*\*\s*2\)", term)`).
- **Issue:** Newer statsmodels emits `np.power(A, 2)`; the term falls
  through to the linear branch and produces wrong predictions /
  surfaces with no warning.
- **Fix direction:** Add an alternative pattern for `np.power(\w+,\s*2)`
  and `power(\w+,\s*2)`. Test: surface / optimisation reproduces the
  correct quadratic shape under both term spellings.

## SEC-27 - Simulator seed entropy truncated to 31 bits
- **Status:** Open.
- **Severity:** U = Medium, L = Low
- **Where:** `process_improve/simulation/model.py:~443`
  (`return int(np.random.SeedSequence().entropy % (2**31))`).
- **Issue:** Reduces simulator-seed space from 128 bits to 31 bits;
  combined with observable simulator outputs an attacker can
  brute-force the seed and recover the hidden coefficients that
  `reveal_simulator` is supposed to gate.
- **Fix direction:** Return the full 128-bit `entropy` (numpy's
  `default_rng` accepts arbitrary-size seeds); or use
  `secrets.randbits(63)` if a JSON-int round-trip is required.
  Test: `draw_initial_seed` returns at least 63 bits of entropy.

## SEC-28 - `_SIGNIFICANT_FACTOR_PATTERN` is O(n^2) on multi-KB input
- **Status:** Open.
- **Severity:** U = Medium, L = Low
- **Where:** `process_improve/experiments/strategy/engine.py:~54-57`
  (`(\w[\w\s]*?)\s+(?:is|are)\s+(?:known\s+to\s+be\s+)?(?:significant|...)`,
  `re.IGNORECASE`).
- **Issue:** `[\w\s]*?` over long whitespace runs is O(n^2). Combined
  with `DEFAULT_MAX_STRING=100_000` from `tool_safety.py`, an attacker
  can send a near-100KB string and burn meaningful CPU. Also captures
  phrases like `"data is significant"` and returns the wrong factor.
- **Fix direction:** Anchor the capture with a length bound,
  e.g. `\b(\w+(?:\s\w+){0,4}?)`; reject inputs longer than e.g. 4096
  chars at the strategy-tool boundary. Test: a 50KB whitespace-heavy
  payload returns in under N ms.

## SEC-29 - Knowledge YAML loader has no file-size cap
- **Status:** Open. SEC-10 added path-traversal protection; size is not
  bounded.
- **Severity:** U = Low, L = Low (defence in depth)
- **Where:** `process_improve/experiments/knowledge/engine.py:~48`
  (`yaml.safe_load(fh)` on `_DATA_DIR/<file>.yaml`).
- **Issue:** A tampered YAML (e.g. a malicious package post-install,
  or an attacker with write access to the data directory) can ship an
  anchor-bomb that explodes on first load; `safe_load` resolves
  anchors and merges. No size cap, no anchor-disable.
- **Fix direction:** Check `path.stat().st_size` against a cap
  (e.g. 1 MB) and reject; use `yaml.CSafeLoader` (or a custom loader
  that disables anchors) for the actual parse. Test: a billion-laughs
  YAML is rejected.

## SEC-30 - `_terminate_workers` relies on CPython private `_processes` attribute
- **Status:** Open.
- **Severity:** U = Low, L = Low (silent timeout-guarantee degradation)
- **Where:** `process_improve/tool_safety.py:~412-436`
  (`getattr(pool, "_processes", None)` under a blanket `suppress`).
- **Issue:** Documented as best-effort, but if CPython renames the
  attribute the runaway worker keeps a CPU and `ToolTimeoutError`
  silently degrades to its pre-SEC-02 behaviour.
- **Fix direction:** Add a `tests/test_tool_safety.py` assertion that
  the attribute exists on the supported Python versions, so CI fails
  loudly (not silently degrades) on a future upgrade. Test:
  `assert hasattr(ProcessPoolExecutor(max_workers=1), "_processes")`.

## SEC-31 - `json.loads(key)` on untrusted dict keys in batch plotting
- **Status:** Open.
- **Severity:** U = Low, L = Low (robustness)
- **Where:** `process_improve/batch/plotting.py:~130, 255`
  (`json.loads(key)` inside a comprehension, no try/except).
- **Issue:** A caller that passes a plain-string key gets a
  `json.JSONDecodeError` deep in plotting code, masking the real input
  mistake.
- **Fix direction:** Validate keys with a clear error message at
  function entry, or pre-decode outside the comprehension with a
  try/except that re-raises a documented `ValueError`. Test: a
  non-JSON key raises a clear `ValueError` at the API surface.

## SEC-32 - Miscellaneous numerical / correctness cleanup
- **Status:** Open. Bundle of small fixes; each is mechanical.
- **Severity:** U = Low, L = Medium
- **Items:**
  - `process_improve/multivariate/methods.py:~2473-2482` -- off-by-one
    in `terminate_check` (`>` vs `>=` against `md_max_iter`); the loop
    runs `md_max_iter + 1` iterations.
  - `process_improve/multivariate/methods.py:~2803-2807` -- `np.var`
    on an empty negative-only slice returns NaN; the comparison
    silently skips a sign-flip that may have been required.
  - `process_improve/multivariate/methods.py:~5031` -- float `==` zero
    guard on `norm(t) * norm(u)`; a near-zero (but non-zero) denom
    still yields a meaningless ratio that is treated as an observed
    statistic.
  - `process_improve/bivariate/methods.py:~98` (currently dead branch)
    -- `np.arccos` argument not clamped to [-1, 1]; if the branch is
    ever re-enabled, a floating-point excursion produces silent NaN.
  - `process_improve/experiments/optimization.py:~564` -- hard-coded
    `rng = np.random.default_rng(42)`; accept an optional
    `random_seed` parameter (default `None` for true randomness).
  - `process_improve/experiments/visualization/plots/registry.py:~266-269`
    -- reserved-word filter (`{"I", "np", "power"}`) is incomplete;
    cross-reference returned names against the actual design columns
    instead of a hard-coded blocklist.
- **Fix direction:** Address each in a single PR with one
  micro-commit per item. Test: a small regression test per item.

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
