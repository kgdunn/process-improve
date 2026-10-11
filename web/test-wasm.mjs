// The browser app's Python side, run under real Pyodide in Node.
//
// It boots Pyodide the way worker.js does, installs the wheel built from this
// checkout with deps=False and loads bootstrap.py. Then, for every design the page
// offers, at three and at six factors with the page's defaults, it generates the
// design, fills the downloaded workbook with a known model in the design's own
// terms plus small noise, and analyses it as uploaded: the coefficients must equal
// an independent numpy fit made in the same interpreter. A dependency with no
// WebAssembly build, an import that reaches for one, or a numerical difference
// under WebAssembly fails here rather than on a reader's screen. The timings it
// prints are what a reader waits for.
//
//   npm install --no-save pyodide@0.28.3
//   node web/test-wasm.mjs dist/web/process_improve-*.whl

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";

const wheelPath = process.argv[2];
assert.ok(wheelPath && fs.existsSync(wheelPath), "pass the path of the built wheel");

const started = performance.now();
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
console.log(`ok  Python ready in ${((performance.now() - started) / 1000).toFixed(1)} s`);

const call = (fn, payload) => {
  const reply = JSON.parse(pyodide.globals.get(fn)(JSON.stringify(payload)));
  assert.ok(Array.isArray(reply.warnings), `${fn} returns its warnings`);
  return reply;
};

// The round trip, in the same interpreter: fill the workbook with a known model in
// the design's own terms, analyse it as uploaded, and fit the same model with numpy.
await pyodide.runPythonAsync(`
import base64, io, itertools, json, time
import numpy as np, pandas as pd

def _terms(k, model):
    terms = [(), *[(i,) for i in range(k)]]
    if model in {"interactions", "quadratic"}:
        terms += list(itertools.combinations(range(k), 2))
    if model == "quadratic":
        terms += [(i, i) for i in range(k)]
    return terms

def _name(term, names):
    if not term:
        return "Intercept"
    if len(term) == 2 and term[0] == term[1]:
        return f"I({names[term[0]]} ** 2)"
    return ":".join(names[i] for i in term)

def round_trip(spec_json, model=""):
    spec = json.loads(spec_json)
    names = [f["name"] for f in spec["factors"]]
    start = time.perf_counter()
    made = json.loads(api_make_design(spec_json))
    design_s = time.perf_counter() - start
    if not made["ok"]:
        return json.dumps({"ok": False, "error": made["error"]})
    design = made["result"]
    sheets = pd.read_excel(io.BytesIO(base64.b64decode(design["xlsx_base64"])), sheet_name=None)
    runs = sheets["Runs"]
    x = np.column_stack([(runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2)
                         for f in spec["factors"]])
    terms = _terms(len(names), design["model"])
    matrix = np.column_stack([np.prod(x[:, list(t)], axis=1) if t else np.ones(len(x)) for t in terms])
    beta = np.linspace(1, 2, len(terms)) * np.resize([1.0, -1.0], len(terms))
    y = matrix @ beta + np.random.default_rng(0).normal(0, 0.01, len(x))
    runs["y"] = y
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        for sheet, frame in sheets.items():
            frame.to_excel(xl, sheet_name=sheet, index=False)
    start = time.perf_counter()
    workbook = base64.b64encode(buf.getvalue()).decode()
    analysed = json.loads(api_analyze(json.dumps({"xlsx_base64": workbook, **({"model": model} if model else {})})))
    analysis_s = time.perf_counter() - start
    if not analysed["ok"]:
        return json.dumps({"ok": False, "error": analysed["error"]})
    got = {c["term"]: c["coefficient"] for c in analysed["result"]["coefficients"]}
    expected = dict(zip([_name(t, names) for t in terms], np.linalg.lstsq(matrix, y, rcond=None)[0]))
    if set(got) != set(expected):
        return json.dumps({"ok": False, "error": f"terms {sorted(got)} differ from {sorted(expected)}"})
    return json.dumps({
        "ok": True,
        "runs": len(x),
        "model": design["model"],
        "worst": max(abs(got[t] - b) for t, b in expected.items()),
        "randomised": runs["Std order"].tolist() != sorted(runs["Std order"]),
        "lenth": analysed["result"]["lenth"] is not None,
        "warnings": made["warnings"] + analysed["warnings"],
        "design_s": design_s,
        "analysis_s": analysis_s,
    })
`);
const roundTrip = (spec, model = "") => JSON.parse(pyodide.globals.get("round_trip")(JSON.stringify(spec), model));

const catalogue = call("api_catalogue", {});
assert.ok(catalogue.ok);
const NAMES = ["T", "P", "F", "C", "S", "R"];
const factors = (k) => NAMES.slice(0, k).map((name, i) => ({ name, low: 100 + 10 * i, high: 102 + 12 * i, units: "" }));

for (const k of [3, 6]) {
  for (const design_type of Object.keys(catalogue.result.designs)) {
    const r = roundTrip({ design_type, factors: factors(k) });
    assert.ok(r.ok, `${design_type}, ${k} factors: ${r.error}`);
    assert.ok(r.worst < 1e-6, `${design_type}, ${k} factors: the fit differs from numpy by ${r.worst}`);
    assert.ok(r.randomised, `${design_type}, ${k} factors: the runs are in standard order`);
    const timing = `design ${r.design_s.toFixed(2)} s, analysis ${r.analysis_s.toFixed(2)} s`;
    console.log(`ok  ${design_type}, ${k} factors: ${r.runs} runs, ${r.model} model; ${timing}`);
    for (const w of r.warnings) console.log(`      warning: ${w}`);
  }
}

// A saturated two-level design is judged with Lenth's method.
const saturated = roundTrip({ design_type: "plackett_burman", factors: factors(3), n_center_points: 0 });
assert.ok(saturated.ok && saturated.lenth, saturated.error);
console.log("ok  saturated Plackett-Burman analysed with Lenth's method");

// A model the design cannot fit is refused with the reason, not fitted.
const refused = roundTrip({ design_type: "dsd", factors: factors(4) }, "quadratic");
assert.ok(!refused.ok && refused.error.includes("cannot fit the quadratic model"), refused.error);
console.log(`ok  refused: ${refused.error}`);

// The guided page: the worked example through the entry points the page calls, then the
// charts, with plotly installed late as worker.js installs it on the first chart.
const example = call("api_catalogue", {}).result.demo;
const made = call("api_make_design", example.design).result;
const held = { spec: made.spec, columns: made.columns, rows: made.rows };
const simulated = call("api_simulate", { ...held, scenario: example.scenario, sigma: example.sigma, seed: example.seed });
assert.ok(simulated.ok, simulated.error);
const analysed = call("api_analyze_runs", {
  ...held,
  responses: simulated.result.responses,
  model: example.model,
  truth: simulated.result.truth,
});
assert.ok(analysed.ok, analysed.error);
for (const c of analysed.result.coefficients) {
  assert.ok(c.ci_low <= c.true_value && c.true_value <= c.ci_high, `${c.term}: the interval misses the truth`);
}
console.log(`ok  worked example: ${analysed.result.equation}`);
let clock = performance.now();
await pyodide.runPythonAsync('import micropip\nawait micropip.install("plotly")');
console.log(`ok  plotly installed after process-improve in ${((performance.now() - clock) / 1000).toFixed(1)} s`);
clock = performance.now();
const charts = call("api_plots", { ...held, responses: simulated.result.responses, truth: simulated.result.truth });
assert.ok(charts.ok, charts.error);
assert.deepEqual(Object.keys(charts.result.figures).sort(), ["contour", "residuals", "true_contour"]);
const timing = `${((performance.now() - clock) / 1000).toFixed(2)} s`;
console.log(`ok  charts: ${Object.keys(charts.result.figures).join(", ")} for Plotly.js ${charts.result.plotly_js}; ${timing}`);

// The compare step: every design for three factors, scored on the nine measures.
clock = performance.now();
const plan = call("api_compare_plan", { factors: factors(3), model: "main_quadratic" });
assert.ok(plan.ok, plan.error);
const contenders = plan.result.designs.map((item) => {
  const generated = call("api_make_design", item.spec).result;
  return { label: item.label, spec: generated.spec, columns: generated.columns, rows: generated.rows };
});
const compared = call("api_compare", { model: "main_quadratic", designs: contenders });
assert.ok(compared.ok, compared.error);
const aOverP = compared.result.measures.findIndex((m) => m.key === "a");
assert.equal(compared.result.display["Box-Behnken [n=15]"][aOverP], "0.2173", "the book's A/p for this design");
const comparedIn = `${((performance.now() - clock) / 1000).toFixed(1)} s`;
console.log(`ok  compared ${compared.result.designs.length} designs on the nine measures in ${comparedIn}`);

// A workbook's model name is data, never code.
const hostile = roundTrip({ design_type: "dsd", factors: factors(4) }, "__import__('os')");
assert.equal(hostile.ok, false);
console.log("ok  hostile model name rejected");
