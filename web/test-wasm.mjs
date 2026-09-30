// Smoke test: the browser app's Python side, run under real Pyodide in Node.
//
// It boots Pyodide the way worker.js does, installs the wheel built from this
// checkout with deps=False, loads bootstrap.py, and drives a full design ->
// fill in -> analyse round trip. A dependency that has no WebAssembly build, or
// an import that reaches for one, fails here rather than on a reader's screen.
//
//   npm install --no-save pyodide@0.28.3
//   node web/test-wasm.mjs dist/web/process_improve-*.whl

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";

const wheelPath = process.argv[2];
assert.ok(wheelPath && fs.existsSync(wheelPath), "pass the path of the built wheel");

const pyodide = await loadPyodide();
// Same list as worker.js.
await pyodide.loadPackage(
  ["micropip", "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "patsy", "pydantic", "pyyaml"],
  { messageCallback: () => {} },
);
const wheel = path.basename(wheelPath);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheel}`, fs.readFileSync(wheelPath));
await pyodide.runPythonAsync(`
import micropip
await micropip.install(["tqdm", "pyDOE3", "openpyxl"])
await micropip.install("emfs:/wheels/${wheel}", deps=False)
`);
await pyodide.runPythonAsync(fs.readFileSync(new URL("bootstrap.py", import.meta.url), "utf8"));

const call = (fn, payload) => JSON.parse(pyodide.globals.get(fn)(JSON.stringify(payload)));

const catalogue = call("api_catalogue", {});
assert.ok(catalogue.ok);

const factors = [
  { name: "T", low: 150, high: 200, units: "degC" },
  { name: "P", low: 1, high: 3, units: "bar" },
  { name: "F", low: 10, high: 20, units: "" },
];
for (const design_type of Object.keys(catalogue.result.designs)) {
  if (design_type === "fractional_factorial") continue; // kgdunn/process-improve#620
  const reply = call("api_make_design", { design_type, factors, budget: 14 });
  assert.ok(reply.ok, `${design_type}: ${reply.error}`);
  console.log(`ok  ${design_type}: ${reply.result.n_runs} runs`);
}

// Round trip: fill the response column with a known quadratic, analyse, and
// check the fit recovers it.
const design = call("api_make_design", { design_type: "box_behnken", factors }).result;
pyodide.globals.set("xlsx_in", design.xlsx_base64);
const filled = await pyodide.runPythonAsync(`
import base64, io
import numpy as np, pandas as pd
sheets = pd.read_excel(io.BytesIO(base64.b64decode(xlsx_in)), sheet_name=None)
runs = sheets["Runs"]
t, p = (runs["T"] - 175) / 25, (runs["P"] - 2) / 1
runs["y"] = 50 + 4 * t - 3 * p + 2 * t * p - 5 * t**2 + np.random.default_rng(0).normal(0, 0.1, len(runs))
buf = io.BytesIO()
with pd.ExcelWriter(buf) as xl:
    runs.to_excel(xl, sheet_name="Runs", index=False)
    sheets["_pi_design"].to_excel(xl, sheet_name="_pi_design", index=False)
base64.b64encode(buf.getvalue()).decode()
`);
const analysis = call("api_analyze", { xlsx_base64: filled });
assert.ok(analysis.ok, analysis.error);
const coef = Object.fromEntries(analysis.result.coefficients.map((c) => [c.term, c.coefficient]));
assert.ok(Math.abs(coef.T - 4) < 0.2 && Math.abs(coef["I(T ** 2)"] + 5) < 0.3, JSON.stringify(coef));
assert.ok(analysis.result.summary.r_squared > 0.99);
console.log(`ok  analyse: ${analysis.result.equation}`);

// A workbook's model name is data, never code.
const hostile = call("api_analyze", { xlsx_base64: filled, model: "__import__('os')" });
assert.equal(hostile.ok, false);
console.log("ok  hostile model name rejected");
