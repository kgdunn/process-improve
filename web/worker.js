// Runs Python in a Web Worker, so the page never freezes while Pyodide loads
// (about 10 s on first visit) or while a design is optimised.
//
// Protocol: the page posts {id, fn, payload}; the worker answers {id, reply}
// where reply is the JSON string the Python function returned, or
// {status, state} while it is still starting up, where state is "busy",
// "ok" or "error" so the page can show a spinner, a tick or an error.

const PYODIDE_VERSION = "0.28.3";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

importScripts(`${PYODIDE_URL}pyodide.js`);

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
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  say("Loading numpy, pandas, scipy, statsmodels, scikit-learn...");
  // Compiled packages come prebuilt from the Pyodide distribution.
  await pyodide.loadPackage([
    "micropip", "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "patsy", "pydantic", "pyyaml",
  ]);

  say("Installing process-improve...");
  pyodide.globals.set("_wheel", await wheelSource());
  await pyodide.runPythonAsync(`
import micropip
# Pure-Python dependencies come from PyPI.
await micropip.install(["tqdm", "openpyxl"])
# deps=False: the wheel's metadata pins numpy/pandas versions that can run
# ahead of the Pyodide distribution, and the ones loaded above are what the
# experiments module actually needs (tests/test_web_app.py checks this).
await micropip.install(_wheel, deps=False)
`);
  pyodide.globals.delete("_wheel");

  const glue = await (await fetch("bootstrap.py", { cache: "no-store" })).text();
  await pyodide.runPythonAsync(glue);
  const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
  say(`Ready: process-improve ${version}`, "ok");
  return pyodide;
}

const ready = boot();
ready.catch((err) => self.postMessage({ status: `Could not start Python: ${err.message}`, state: "error" }));

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const pyodide = await ready;
    if (fn === "api_plots") await installPlotly(pyodide);
    const reply = pyodide.globals.get(fn)(payload ?? "{}");
    self.postMessage({ id, reply });
  } catch (err) {
    if (fn === "api_plots") plotly = null; // the next chart tries the install again
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: String(err.message || err) }) });
  }
};
