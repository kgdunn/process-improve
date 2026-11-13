// Runs Python in a Web Worker, so the page never freezes while Pyodide loads or while a design is optimised.
// It is a module worker (app.js makes it with type "module"): Pyodide 314 refuses a classic one.
//
// Protocol: the page posts {id, fn, payload}; the worker answers {id, reply}, where reply is the JSON string the
// Python function returned, or {status, state} while it starts up or installs a package, where state is "busy",
// "ok" or "error" so the page can show a spinner, a tick or an error. If Python cannot start, every call is
// answered with the reason, so nothing waits for it. Once what a model is fitted with is in, it posts {warmed},
// which holds what the page now runs on, or false if the worker could not get it unasked.

import { answer, loadGlue, onDemand, oneLine, provider, startPython } from "./python-boot.js";

const say = (status, state = "busy") => self.postMessage({ status, state });
const here = new URL(".", self.location.href);

async function boot() {
  const python = await startPython(here, say);
  const { pyodide, runtime, manifest } = python;
  // The page's own Python: the build's archive of it, kept by the browser as the wheel is, or the file as it is.
  const archive = manifest.glue && new URL(`wheels/${manifest.glue}`, here).href;
  const glue = await loadGlue(pyodide, "bootstrap", { archive, source: new URL("bootstrap.py", here).href });
  const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
  say(`Ready: process-improve ${version}`, "ok");
  const table = onDemand(runtime, manifest);
  // Once what a model is fitted with is in, whichever call or step asked for it: the library's analysis code is
  // imported, so that a first fit waits for nothing, and the page is told what it now runs on.
  const provide = provider(pyodide, glue, table, say, (installed) => {
    if (installed === table.statsmodels) self.postMessage({ warmed: JSON.parse(glue.warm()).result ?? false });
  });
  return { glue, provide };
}

const ready = boot().catch((err) => {
  const reason = `Could not start Python: ${oneLine(err)}`;
  say(reason, "error");
  throw new Error(reason);
});

// The page is ready once a design can be made: what a model is fitted with is left out of the start, so a visit
// that only makes a run sheet waits for none of it. It is fetched here instead, in the background and without a
// word, as soon as the page is ready; a call that needs it sooner waits for this same install, and says so. If it
// cannot be had, or Python did not start, the page is told that much: its copy for use without a network waits to
// hear (offline.js), and a call that needs it tries again.
ready.then(({ provide }) => provide("statsmodels", { quiet: true })).catch(() => self.postMessage({ warmed: false }));

// The page calls only the api_* entry points bootstrap.py defines; nothing else in the module answers.
const ENTRY_POINT = /^api_[a-z_]+$/;

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const refused = new Error(`${JSON.stringify(fn)} is not one of the page's entry points.`);
    if (!ENTRY_POINT.test(fn)) throw refused;
    const { glue, provide } = await ready;
    const entry = glue[fn];
    if (typeof entry !== "function") throw refused;
    self.postMessage({ id, reply: await answer(entry, payload ?? "{}", provide) });
  } catch (err) {
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: oneLine(err) }) });
  }
};
