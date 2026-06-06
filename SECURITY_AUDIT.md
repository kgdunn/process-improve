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
| SEC-14 | RCE via patsy formula in analyze/evaluate/augment/`lm` | Critical | Low | done (#314, v1.22.11) |
| SEC-15 | `reveal_simulator` gate bypassable via kwarg injection | Critical | Low | done (#264, v1.22.12) |
| SEC-22 | Holt-Winters chart divides by zero on constant warm-up | High | High | done (#271, v1.22.13) |
| SEC-23 | `regression.OLS.predict` accepts wrong-shape `X` silently | High | Medium | done (#272, v1.22.13) |
| SEC-34 | Unguarded `json.loads` of highlight keys in multivariate plots | Low | Low | done (v1.26.1) |
| SEC-35 | Blanket `setattr(**kwargs)` in `ControlChart.calculate_limits` | Low | Low | done (v1.26.1) |
| SEC-36 | No PEP 561 `py.typed` marker; published types invisible | Low | Low | done (v1.26.1) |
| SEC-37 | No security disclosure policy (`SECURITY.md`) | Low | Low | done (v1.26.1) |

> Note: SEC-16 through SEC-33 were tracked in their own PRs / CHANGELOG
> entries and are not all transcribed into this table; the rows above resume
> the running ledger from the latest audit pass.

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

## SEC-14 - Arbitrary code execution via patsy formula in analyze/evaluate/augment/`lm` [RESOLVED]
- **Status:** Fixed in v1.22.11 (issue #263). The formula guard now runs inside
  the library functions (the default path) rather than only the
  `fit_linear_model` wrapper.
- **Severity:** U = Critical, L = Low
- **Where:**
  - `process_improve/experiments/analysis.py` (`analyze_experiment` ->
    `smf.ols(build_formula(...), data=df).fit()`): `model`, `response_column`,
    and `design_matrix` dict keys reached patsy.
  - `process_improve/experiments/evaluate.py` (`_build_model_matrix` ->
    `dmatrix(rhs, design_df, ...)`): `model` flowed in as `rhs`.
  - `process_improve/experiments/augment.py` (`_build_model_rhs` ->
    `_greedy_d_optimal_select` -> `dmatrix(rhs, trial, ...)`): `target_model`
    flowed in.
  - `process_improve/experiments/models.py` (`lm`): `smf.ols(model_spec, ...)`
    with no guard; the SEC-01 check lived only in the tool wrapper.
- **Issue:** Patsy evaluates each formula term as a Python expression with
  builtins and numpy in scope, so a string such as
  `y ~ I(__import__('os').system('id'))` (or a malicious column name /
  `response_column`) executes arbitrary code. The schema `enum` on `model` is
  only enforced under `PROCESS_IMPROVE_MCP_SAFE_MODE`; the default stdio MCP path
  skips it, and even in safe mode `response_column` and dict keys were
  unconstrained.
- **Fix:** `validate_formula_is_safe` gained `allow_transforms` / `allow_numpy`
  flags backed by an AST walker that admits only `I()`/`Q()` and a curated
  allowlist of element-wise `np.<func>` calls (rejecting attribute access, string
  literals, dunders, and any other call); the default strict path is unchanged.
  `validate_identifier_is_safe` rejects non-identifier column / response names.
  `lm()` validates with transforms+numpy enabled (the textbook API uses
  `I(np.power(...))` etc.); the three design paths validate the built formula /
  RHS with transforms enabled (so the `quadratic` shorthand's `I(f ** 2)` still
  fits). Tests assert malicious formulas, models, and names are rejected before
  reaching patsy, with sentinel-file checks confirming no side effect runs.

## SEC-15 - `reveal_simulator` confirmation gate bypassable via kwarg injection [RESOLVED]
- **Status:** Fixed in v1.22.12 (issue #264). `execute_tool_call` filters
  `tool_input` down to the keys declared in the tool's `input_schema` before
  dispatch; `simulator_state` / `confirmed` moved off the function signatures
  into a `contextvars` side channel (`simulation/context.py`) only the host
  populates.
- **Severity:** U = Critical, L = Low
- **Where:** `process_improve/simulation/tools.py` (`reveal_simulator`,
  `simulate_process`) and `process_improve/tool_spec.py` (`execute_tool_call`),
  which dispatched via `_TOOL_REGISTRY[tool_name](**tool_input)`.
- **Issue:** `reveal_simulator` and `simulate_process` accepted `simulator_state`
  / `confirmed` as keyword arguments deliberately omitted from the JSON schema so
  the host could inject them server-side. But the dispatch path forwarded
  `**tool_input` verbatim, and the default (non-safe) MCP path ran no schema
  validation, so a prompt-injected agent could pass `confirmed=True` plus a
  fabricated `simulator_state` as ordinary kwargs - bypassing the
  double-confirmation reveal gate and feeding the function attacker-controlled
  state.
- **Fix direction:** (1) Filter `tool_input` to the schema's declared
  `properties` before invoking the function (the safe path already rejected
  unknown keys via `validate_against_schema`). (2) Remove `simulator_state` /
  `confirmed` from the kwarg surface entirely and inject them through a
  `contextvars.ContextVar` populated by the host before dispatch, so they cannot
  be re-introduced as kwargs even if the registry filter regresses. Tests:
  forwarding `confirmed` / `simulator_state` through dispatch is dropped (or
  raises `ToolInputInvalidError` on the safe path), and the gate still fires for
  legitimate host calls.

## SEC-22 - Holt-Winters control chart divides by zero on constant warm-up window [RESOLVED]
- **Status:** Fixed in v1.22.13 (issue #271). `_holt_winters_warmup_fit` now
  raises a clear `ValueError` when the warm-up `sigma_0` is zero or non-finite,
  instead of propagating `inf`/`NaN` into the control limits.
- **Severity:** U = High, L = High (silent wrong control-chart limits)
- **Where:** `process_improve/monitoring/control_charts.py` (`_holt_winters_warmup_fit`,
  around the `rho_input` / `rho_i = error_i / sigma_hat` divisions).
- **Issue:** A constant (zero-variance) warm-up window makes
  `sigma_0 = MAD = 0`, so `rho_i = +/-inf` propagates into every downstream
  control-chart limit, which silently become `0` / `NaN`. No guard existed.
- **Fix direction:** Validate `sigma_0` immediately after it is computed; raise
  `ValueError("...variance is zero in warm-up window; supply more representative
  data...")` rather than dividing by it. Test: a constant warm-up series raises a
  clear error instead of producing NaN limits.

## SEC-23 - `regression.OLS.predict` accepts wrong-shape `X` silently [RESOLVED]
- **Status:** Fixed in v1.22.13 (issue #272). `OLS.predict` now checks the
  feature count against `n_features_in_` and raises a clear `ValueError`.
- **Severity:** U = High, L = Medium
- **Where:** `process_improve/regression/methods.py` (`OLS.predict`,
  `return intercept + X_arr @ self.coefficients_`).
- **Issue:** `n_features_in_` is stored at fit time but never checked at predict.
  A wrong column count either raised a confusing numpy `ValueError` or, worse,
  broadcast shapes (e.g. a 1-column `X` against a scalar coefficient) and
  produced silently wrong output.
- **Fix direction:** Validate `X_arr.shape[1] == self.n_features_in_` and raise a
  clear `ValueError`, mirroring the sklearn-style shape check used by PCA / PLS.
  Tests: predict with the wrong column count raises a clear `ValueError`; predict
  with the correct shape still returns finite values.

## SEC-34 - Unguarded `json.loads` of highlight keys in multivariate plots [RESOLVED]
- **Status:** Fixed in v1.26.1. The three plot helpers decode highlight keys via
  a shared `_decode_highlight_style` helper that raises a clear `ValueError`,
  matching the SEC-32 guard already applied in `batch/plotting.py`.
- **Severity:** U = Low, L = Low (robustness)
- **Where:** `process_improve/multivariate/plots.py` `score_plot` (2D + 3D
  branches), `spe_plot`, `t2_plot`.
- **Issue:** Each `items_to_highlight` key is a JSON-encoded Plotly style spec
  decoded with a bare `json.loads(key)`. A malformed key raised an uncaught
  `json.JSONDecodeError` from deep inside the trace-building loop rather than a
  clear error at the API boundary. SEC-32 (#281) fixed the identical pattern in
  `batch/plotting.py` but the multivariate plot sites were missed.
- **Fix:** Introduced `_decode_highlight_style(key)` which wraps `json.loads`
  and re-raises as a `ValueError` with the offending key and an example of the
  expected format. All four call sites now use it. Regression test:
  `test_highlight_key_must_be_json` (parametrised over score/spe/t2).

## SEC-35 - Blanket `setattr(**kwargs)` in `ControlChart.calculate_limits` [RESOLVED]
- **Status:** Fixed in v1.26.1. `**kwargs` is now validated against an allowlist
  (`ld_1`, `ld_2`) before any attribute is set.
- **Severity:** U = Low, L = Low (defensive programming / footgun)
- **Where:** `process_improve/monitoring/control_charts.py`
  (`ControlChart.calculate_limits` -> `for key, val in kwargs.items():
  setattr(self, key, val)`).
- **Issue:** The method copied every keyword argument onto the instance with an
  unrestricted `setattr`. The only legitimate kwargs are the Holt-Winters
  smoothing lambdas `ld_1` / `ld_2`; any other key (a typo, or a deliberately
  crafted name such as `target`, `s`, `train_samples`, or a method name) would
  silently overwrite internal state and corrupt the fitted limits with no error.
  The current tool wrapper passes no kwargs, so it was not reachable from the
  MCP surface, but it is a footgun for direct API users.
- **Fix:** Extracted `_apply_tuning_kwargs`, which rejects any key outside the
  `_TUNING_KWARGS` allowlist with a clear `ValueError` before applying the
  remaining (allowlisted) values. Regression test:
  `test_calculate_limits_rejects_unknown_kwargs`.

## SEC-36 - No PEP 561 `py.typed` marker; published types invisible [RESOLVED]
- **Status:** Fixed in v1.26.1. `src/process_improve/py.typed` is added and ships
  in the wheel (verified: the `uv_build` backend bundles all non-`.py` files in
  the package tree).
- **Severity:** U = Low, L = Low (maintenance / downstream usability)
- **Where:** packaging (`src/process_improve/`).
- **Issue:** The project is fully type-annotated and `mypy src/process_improve`
  runs as a blocking CI gate (ENG-03 / ENG-20), yet the distribution carried no
  `py.typed` marker. Under PEP 561 a type-checker therefore treated
  `process-improve` as an untyped third-party package and ignored every
  annotation, so downstream users got none of the benefit of the typing work.
- **Fix:** Added the marker file. Downstream mypy / pyright now consume the
  published annotations.

## SEC-37 - No security disclosure policy (`SECURITY.md`) [RESOLVED]
- **Status:** Fixed in v1.26.1. `SECURITY.md` added at the repo root.
- **Severity:** U = Low, L = Low (process / community)
- **Where:** repository root.
- **Issue:** The project ships an agent-callable MCP tool surface and maintains
  this detailed `SECURITY_AUDIT.md`, but provided no private channel for a
  security researcher to report a vulnerability and no statement of supported
  versions or response expectations. GitHub surfaces a repo's `SECURITY.md` as
  its "Security policy"; its absence pushes reporters toward public issues.
- **Fix:** Added `SECURITY.md` documenting private reporting (GitHub private
  advisories + email), supported versions, response timeline, threat-model
  scope, and out-of-scope items, cross-linked with this file.

---

## Recommendations (not yet actioned)

These are lower-priority maintenance / contribution-health items surfaced
during the v1.26.1 audit pass. They are documented here rather than fixed in the
same change because each is a maintainer judgement call.

- **`CODE_OF_CONDUCT.md` is absent.** A code of conduct is a standard
  community-health file that GitHub surfaces and that many contributors look for
  before engaging. Adopting the Contributor Covenant (with the maintainer's
  contact as the enforcement channel) would round out the community files
  alongside `CONTRIBUTING.md` and the new `SECURITY.md`.
- **`.pre-commit-config.yaml` runs `flake8` and `isort` alongside `ruff`.**
  `ruff` already replaces both (lint + import sorting via the `I` rules), and CI
  only runs `ruff`. Keeping `flake8` (plus a separate `.flake8` config) and
  `mirrors-isort` in the pre-commit hooks is redundant and can produce
  conflicting fixups. The stale comment `# Remove MyPy: conflicts in python 3.9
  with pytz` also no longer applies (the project requires Python >= 3.10 and
  mypy is an active hook). Consider trimming pre-commit to `ruff` + `ruff-format`
  + the hygiene hooks, and either dropping `.flake8` or aligning it with the
  120-character line length.

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
