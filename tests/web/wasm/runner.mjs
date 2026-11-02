// web/bootstrap.py under Pyodide in Node.js, answering the tests' calls one JSON line at a time.
//
//   node runner.mjs <pyodide package directory> <runtime.json> <wheel> <bootstrap.py>
//
// It starts Python as web/worker.js does: the Pyodide packages web/runtime.json names, its
// micropip packages, the wheel with deps=False, then bootstrap.py. It prints
// {"ready": <process-improve version>}, then answers each line on stdin, {"fn", "payload"},
// with the JSON text the entry point returns, on one line; an exception comes back as
// worker.js sends one, {"ok": false, "error"}. What Pyodide and Python print goes to stderr,
// so stdout carries only the answers.

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const [pyodideDir, runtimePath, wheelPath, bootstrapPath] = process.argv.slice(2);
const runtime = JSON.parse(fs.readFileSync(runtimePath, "utf8"));
const aside = (text) => process.stderr.write(`${text}\n`);

const { loadPyodide } = await import(pathToFileURL(path.join(pyodideDir, "pyodide.mjs")).href);
const pyodide = await loadPyodide({ stdout: aside, stderr: aside });
await pyodide.loadPackage(runtime.packages, { messageCallback: aside, errorCallback: aside });
const wheel = path.basename(wheelPath);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheel}`, fs.readFileSync(wheelPath));
pyodide.globals.set("_micropip", pyodide.toPy(runtime.micropip));
await pyodide.runPythonAsync(`
import micropip
await micropip.install(list(_micropip))
await micropip.install("emfs:/wheels/${wheel}", deps=False)
`);
pyodide.globals.delete("_micropip");
await pyodide.runPythonAsync(fs.readFileSync(bootstrapPath, "utf8"));
const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
process.stdout.write(`${JSON.stringify({ ready: version })}\n`);

for await (const line of readline.createInterface({ input: process.stdin })) {
  const { fn, payload } = JSON.parse(line);
  let reply;
  try {
    reply = pyodide.globals.get(fn)(payload);
  } catch (err) {
    reply = JSON.stringify({ ok: false, error: String(err.message || err) });
  }
  process.stdout.write(`${reply}\n`);
}
