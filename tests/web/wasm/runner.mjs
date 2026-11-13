// web/bootstrap.py under Pyodide in Node.js, answering the tests' calls one JSON line at a time.
//
//   node runner.mjs <pyodide package directory> <runtime.json> <wheel> <the page's Python: its archive, or its file>
//                   [<the build's manifest>]
//
// It starts Python as the page's worker does (web/python-boot.js and web/worker.js): the Pyodide release with the
// packages web/runtime.json names, from the folder of the release it names; the wheel with deps=False; then the
// page's Python, imported as the module bootstrap. A call that says it lacks a package start-up leaves out has the
// package installed and is made again. It prints {"ready": <process-improve version>, "python": <its version>,
// "bytecode": <whether the wheel and the page's Python were loaded from the build's bytecode>}, then answers each
// line on stdin, {"fn", "payload"}, with the JSON text the entry point returns, on one line; an exception comes
// back as worker.js sends one, {"ok": false, "error"}. What Pyodide and Python print goes to stderr, so stdout
// carries only the answers.

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";

// The worker's own way of installing a package and of answering a call that lacks one.
const boot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../web/python-boot.js");
const { answer, install, installExact } = await import(pathToFileURL(boot).href);

const [pyodideDir, runtimePath, wheelPath, gluePath, manifestPath] = process.argv.slice(2);
const runtime = JSON.parse(fs.readFileSync(runtimePath, "utf8"));
const manifest = manifestPath ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : {};
const aside = (text) => process.stderr.write(`${text}\n`);

// The interpreter is the npm package's, the same file in every folder of the release; the folder's own standard
// library, lock file and wheels are kept beside it, under the folder's name (tests/web/_pyodide.py fetches them).
const folder = path.join(pyodideDir, runtime.distribution);
const { loadPyodide } = await import(pathToFileURL(path.join(pyodideDir, "pyodide.mjs")).href);
const pyodide = await loadPyodide({
  stdLibURL: path.join(folder, "python_stdlib.zip"),
  lockFileURL: path.join(folder, "pyodide-lock.json"),
  packageCacheDir: folder,
  packages: runtime.packages,
  stdout: aside,
  stderr: aside,
});

const wheel = path.basename(wheelPath);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheel}`, fs.readFileSync(wheelPath));
await install(pyodide, [`emfs:/wheels/${wheel}`], { deps: false });

const site = pyodide.runPython("import site; site.getsitepackages()[0]");
const packed = new Uint8Array(fs.readFileSync(gluePath)); // a Node.js Buffer is no array Pyodide takes
if (gluePath.endsWith(".zip")) pyodide.unpackArchive(packed, "zip", { extractDir: site });
else pyodide.FS.writeFile(`${site}/bootstrap.py`, packed);
pyodide.runPython("import importlib; importlib.invalidate_caches()");
const glue = pyodide.pyimport("bootstrap");

// What the page's worker installs when a call first needs it, as it installs it (onDemand and provide in
// web/worker.js): the exact files the build resolved, or else the pins, through the package index. `installed`
// lists the packages as they are asked for, so a test can see what a call needed.
const onDemand = { openpyxl: runtime.micropip, plotly: [manifest.plotly ? `plotly==${manifest.plotly}` : "plotly"] };
const installed = [];
async function provide(name) {
  if (!onDemand[name]) throw new Error(`Python needs ${name}, which the page does not install.`);
  const exact = manifest.on_demand?.[name];
  await (exact ? installExact(pyodide, exact) : install(pyodide, onDemand[name]));
  installed.push(name);
}

// Whether a module was loaded from bytecode the build made: its header is this Python's, and says the bytecode
// is never checked against its source (PEP 552), as the build writes it.
const fromBytecode = pyodide.runPython(`
import importlib.util

def from_bytecode(name):
    module = importlib.import_module(name)
    try:
        with open(importlib.util.cache_from_source(module.__file__), "rb") as compiled:
            header = compiled.read(8)
    except OSError:
        return False
    return header[:4] == importlib.util.MAGIC_NUMBER and int.from_bytes(header[4:], "little") == 1

from_bytecode
`);
const version = pyodide.runPython("from importlib.metadata import version; version('process-improve')");
const python = pyodide.runPython("import platform; platform.python_version()");
const bytecode = { wheel: fromBytecode("process_improve.experiments.designs"), glue: fromBytecode("bootstrap") };
process.stdout.write(`${JSON.stringify({ ready: version, python, bytecode })}\n`);

for await (const line of readline.createInterface({ input: process.stdin })) {
  const { fn, payload } = JSON.parse(line);
  let reply;
  try {
    // "installed" is the runner's own question: the packages it has installed for the calls so far.
    reply = fn === "installed" ? JSON.stringify(installed) : await answer(glue[fn], payload, provide);
  } catch (err) {
    reply = JSON.stringify({ ok: false, error: String(err.message || err) });
  }
  process.stdout.write(`${reply}\n`);
}
