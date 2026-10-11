// Runs Python in a Web Worker, so the page never freezes while Pyodide loads
// (about 10 s on first visit) or while a design is optimised. It is a module
// worker (app.js makes it with type "module"): Pyodide 314 refuses a classic one.
//
// Protocol: the page posts {id, fn, payload}; the worker answers {id, reply}
// where reply is the JSON string the Python function returned, or
// {status, state} while it is still starting up, where state is "busy",
// "ok" or "error" so the page can show a spinner, a tick or an error. If Python
// cannot start, every call is answered with the reason, so nothing waits for it.

// runtime.json names the Pyodide release, the packages loaded from it and the PyPI packages
// micropip installs, each pinned; the wasm tests and CI read the same file.
const runtime = fetch("runtime.json", { cache: "no-store" }).then((res) => {
  if (!res.ok) throw new Error(`runtime.json could not be read (HTTP ${res.status})`);
  return res.json();
});

/** An error in one line: a Python traceback's last line names the exception and says what went wrong. */
const oneLine = (err) => String(err?.message || err).trim().split("\n").at(-1);

// The deploy step writes wheels/manifest.json naming the wheel built from this
// commit, and the plotly release its worked example was drawn with. Without it (a
// local preview) the page falls back to the PyPI release of each.
async function readManifest() {
  try {
    const res = await fetch("wheels/manifest.json", { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch (_) {
    /* fall through to PyPI */
  }
  return {};
}
const manifest = readManifest();

async function wheelSource() {
  const { wheel } = await manifest;
  return wheel ? new URL(`wheels/${wheel}`, self.location.href).href : "process-improve";
}

// plotly is installed on the first chart rather than at start-up: designing and
// analysing do not need it, and it would add its download to every visit.
let plotly = null;
function installPlotly(pyodide) {
  plotly ??= (async () => {
    self.postMessage({ status: "Installing plotly for the charts...", state: "busy" });
    const { plotly: version } = await manifest;
    pyodide.globals.set("_plotly", version ? `plotly==${version}` : "plotly");
    await pyodide.runPythonAsync("import micropip\nawait micropip.install(_plotly)");
    pyodide.globals.delete("_plotly");
  })();
  return plotly;
}

async function boot() {
  const say = (status, state = "busy") => self.postMessage({ status, state });
  say("Loading Python (WebAssembly)...");
  const { pyodide: release, packages, micropip } = await runtime;
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${release}/full/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`); // here, so a refused download stops boot()
  const pyodide = await loadPyodide({ indexURL });

  say("Loading numpy, pandas, scipy, statsmodels, scikit-learn...");
  // Compiled packages come prebuilt from the Pyodide distribution.
  await pyodide.loadPackage(packages);

  say("Installing process-improve...");
  pyodide.globals.set("_pypi", pyodide.toPy(micropip));
  pyodide.globals.set("_wheel", await wheelSource());
  await pyodide.runPythonAsync(`
import micropip
# The pure-Python packages the Pyodide distribution lacks come from PyPI, at the pinned versions.
await micropip.install(list(_pypi))
# deps=False: the wheel's metadata bounds numpy and pandas, and the Pyodide release
# need not fall inside the bounds (Pyodide 314 ships pandas 3.0, where the package
# says <3.0). The packages loaded above are what the experiments module needs
# (tests/web/api/test_browser_safe.py), and the wasm tier checks its answers under
# them against native Python's.
await micropip.install(_wheel, deps=False)
`);
  pyodide.globals.delete("_pypi");
  pyodide.globals.delete("_wheel");

  const glue = await (await fetch("bootstrap.py", { cache: "no-store" })).text();
  await pyodide.runPythonAsync(glue);
  const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
  say(`Ready: process-improve ${version}`, "ok");
  return pyodide;
}

const ready = boot().catch((err) => {
  const reason = `Could not start Python: ${oneLine(err)}`;
  self.postMessage({ status: reason, state: "error" });
  throw new Error(reason);
});
ready.catch(() => {}); // each call is answered with the reason

// The page calls only the api_* entry points bootstrap.py defines; nothing else in its namespace answers.
const ENTRY_POINT = /^api_[a-z_]+$/;

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const refused = new Error(`${JSON.stringify(fn)} is not one of the page's entry points.`);
    if (!ENTRY_POINT.test(fn)) throw refused;
    const pyodide = await ready;
    const entry = pyodide.globals.get(fn);
    if (typeof entry !== "function") throw refused;
    if (fn === "api_plots") await installPlotly(pyodide);
    const reply = entry(payload ?? "{}");
    self.postMessage({ id, reply });
  } catch (err) {
    if (fn === "api_plots") plotly = null; // the next chart tries the install again
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: oneLine(err) }) });
  }
};
