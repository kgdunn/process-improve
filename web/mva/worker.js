// The multivariate page's Python: Pyodide in a web worker, so a fit never freezes the page.
//
// It starts Python as web/worker.js does for the designed-experiments page, with the code they share
// (../python-boot.js): the Pyodide release and the packages ../runtime.json names, then the process-improve wheel.
// This page reads workbooks from its first step, so the pinned PyPI packages are installed at the start. Then its
// own Python, bootstrap.py, whose api_* entry points are all the page calls.

import { install, loadGlue, oneLine, startPython } from "../python-boot.js";

const say = (status, state = "busy") => self.postMessage({ status, state });

async function boot() {
  const { pyodide, runtime } = await startPython(new URL("..", self.location.href), say);
  await install(pyodide, runtime.micropip);
  const glue = await loadGlue(pyodide, "bootstrap", { source: new URL("bootstrap.py", self.location.href).href });
  say("Ready", "ok");
  return glue;
}

const ready = boot().catch((err) => {
  const reason = `Could not start Python: ${oneLine(err)}`;
  say(reason, "error");
  throw new Error(reason);
});
ready.catch(() => {});

const ENTRY_POINT = /^api_[a-z_]+$/;

self.onmessage = async ({ data: { id, fn, payload } }) => {
  try {
    const refused = new Error(`${JSON.stringify(fn)} is not one of the page's entry points.`);
    if (!ENTRY_POINT.test(fn)) throw refused;
    const entry = (await ready)[fn];
    if (typeof entry !== "function") throw refused;
    self.postMessage({ id, reply: entry(payload ?? "{}") });
  } catch (err) {
    self.postMessage({ id, reply: JSON.stringify({ ok: false, error: oneLine(err) }) });
  }
};
