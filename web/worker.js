// Runs Python in a Web Worker, so the page never freezes while Pyodide loads or while a design is optimised.
// It is a module worker (app.js makes it with type "module"): Pyodide 314 refuses a classic one.
//
// Protocol: the page posts {id, fn, payload}; the worker answers {id, reply}, where reply is the JSON string the
// Python function returned, or {status, state} while it starts up or installs a package, where state is "busy",
// "ok" or "error" so the page can show a spinner, a tick or an error. If Python cannot start, every call is
// answered with the reason, so nothing waits for it.

import { answer, install, loadGlue, oneLine, startPython } from "./python-boot.js";

const say = (status, state = "busy") => self.postMessage({ status, state });
const here = new URL(".", self.location.href);

/**
 * What start-up leaves out, since designing and analysing need neither: the module Python says a call lacks, what
 * micropip installs for it, and what the status line says it is for. openpyxl is installed with the first
 * workbook, written or read, and plotly with the first chart, at the release the build drew the worked example
 * with; neither adds its download to a visit that makes a run sheet and no more.
 */
const onDemand = ({ runtime, manifest }) => ({
  openpyxl: { install: runtime.micropip, purpose: "Excel workbooks" },
  plotly: { install: [manifest.plotly ? `plotly==${manifest.plotly}` : "plotly"], purpose: "the charts" },
});

async function boot() {
  const python = await startPython(here, say);
  const { pyodide, manifest } = python;
  // The page's own Python: the build's archive of it, kept by the browser as the wheel is, or the file as it is.
  const archive = manifest.glue && new URL(`wheels/${manifest.glue}`, here).href;
  const glue = await loadGlue(pyodide, "bootstrap", { archive, source: new URL("bootstrap.py", here).href });
  const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
  say(`Ready: process-improve ${version}`, "ok");
  return { ...python, glue };
}

const ready = boot().catch((err) => {
  const reason = `Could not start Python: ${oneLine(err)}`;
  say(reason, "error");
  throw new Error(reason);
});
ready.catch(() => {}); // each call is answered with the reason

// One install of each package, shared by the calls that find it missing at the same time. An install ends its own
// note, as start-up does: the first chart may be the curves' fit, after which the page says nothing more.
const installs = new Map();
function provide(python, name) {
  const wanted = onDemand(python)[name];
  if (!wanted) throw new Error(`Python needs ${name}, which this page does not install.`);
  if (!installs.has(name)) {
    const installing = (async () => {
      say(`Installing ${name} for ${wanted.purpose}...`);
      try {
        await install(python.pyodide, wanted.install);
      } catch (err) {
        installs.delete(name); // the next call that needs it tries the install again
        const reason = `Could not install ${name}: ${oneLine(err)}`;
        say(reason, "error");
        throw new Error(reason); // what the call is answered with, too
      }
      say(`Installed ${name} for ${wanted.purpose}.`, "ok");
    })();
    installs.set(name, installing);
  }
  return installs.get(name);
}

// The page calls only the api_* entry points bootstrap.py defines; nothing else in the module answers.
const ENTRY_POINT = /^api_[a-z_]+$/;

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const refused = new Error(`${JSON.stringify(fn)} is not one of the page's entry points.`);
    if (!ENTRY_POINT.test(fn)) throw refused;
    const python = await ready;
    const entry = python.glue[fn];
    if (typeof entry !== "function") throw refused;
    const reply = await answer(entry, payload ?? "{}", (name) => provide(python, name));
    self.postMessage({ id, reply });
  } catch (err) {
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: oneLine(err) }) });
  }
};
