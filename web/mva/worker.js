// The multivariate page's Python: Pyodide in a web worker, so a fit never freezes the page.
//
// It starts Python as web/worker.js does for the designed-experiments page: the Pyodide release and the
// packages ../runtime.json names, its pinned PyPI packages, then the process-improve wheel with deps=False,
// then bootstrap.py. The page calls only the api_* entry points bootstrap.py defines.
// (This duplicates web/worker.js's start-up on purpose, until the pieces both pages share move to one place.)

const runtime = fetch("../runtime.json", { cache: "no-store" }).then((res) => res.json());

// The build writes the app's wheels/manifest.json, one level up, naming the wheel built from this checkout;
// both pages install that one wheel. Without a manifest (web/ served as it is), PyPI's release.
const manifest = fetch("../wheels/manifest.json", { cache: "no-store" })
  .then((res) => (res.ok ? res.json() : {}))
  .catch(() => ({}));

const oneLine = (err) =>
  String(err?.message ?? err)
    .split("\n")
    .filter(Boolean)
    .at(-1) ?? "unknown error";

async function boot() {
  const say = (status, state = "busy") => self.postMessage({ status, state });
  say("Loading Python (WebAssembly)...");
  const { pyodide: release, packages, micropip } = await runtime;
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${release}/full/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL });
  say("Loading numpy, pandas, scipy, scikit-learn...");
  await pyodide.loadPackage(packages);
  say("Installing process-improve...");
  const { wheel } = await manifest;
  pyodide.globals.set("_pypi", pyodide.toPy(micropip));
  pyodide.globals.set("_wheel", wheel ? new URL(`../wheels/${wheel}`, self.location.href).href : "process-improve");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(list(_pypi))
await micropip.install(_wheel, deps=False)
`);
  pyodide.globals.delete("_pypi");
  pyodide.globals.delete("_wheel");
  await pyodide.runPythonAsync(await (await fetch("bootstrap.py", { cache: "no-store" })).text());
  say("Ready", "ok");
  return pyodide;
}

const ready = boot().catch((err) => {
  const reason = `Could not start Python: ${oneLine(err)}`;
  self.postMessage({ status: reason, state: "error" });
  throw new Error(reason);
});
ready.catch(() => {});

const ENTRY_POINT = /^api_[a-z_]+$/;

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const refused = new Error(`${JSON.stringify(fn)} is not one of the page's entry points.`);
    if (!ENTRY_POINT.test(fn)) throw refused;
    const pyodide = await ready;
    const entry = pyodide.globals.get(fn);
    if (typeof entry !== "function") throw refused;
    self.postMessage({ id, reply: entry(payload ?? "{}") });
  } catch (err) {
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: oneLine(err) }) });
  }
};
