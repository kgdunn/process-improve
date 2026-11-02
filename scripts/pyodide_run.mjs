// Run a Python file, or pytest, from this checkout under Pyodide in Node.js, to time it the way the
// browser app would run it.
//
//   uv build --wheel -o dist
//   node scripts/pyodide_run.mjs dist/<wheel> scripts/multivariate_timings.py [args]
//   node scripts/pyodide_run.mjs dist/<wheel> -m pytest -o addopts= tests/test_multivariate.py --durations=20
//
// It starts Python as web/worker.js does: the Pyodide release and packages web/runtime.json names,
// its micropip packages, then the wheel with deps=False. pytest and plotly are added, since the
// tests need them. The checkout is mounted at /repo and is the working directory, so the tests and
// scripts find their data; process_improve itself comes from the wheel. Pyodide is the npm package
// the wasm tests use: PI_PYODIDE, or else ~/.cache/process-improve/node/node_modules/pyodide.
// Behind a proxy, Node.js needs NODE_USE_ENV_PROXY=1 to fetch Pyodide's packages.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [wheelPath, ...argv] = process.argv.slice(2);
if (!wheelPath || argv.length === 0) {
  process.stderr.write("usage: node scripts/pyodide_run.mjs <wheel> <script.py | -m module> [args]\n");
  process.exit(2);
}
const pyodideDir =
  process.env.PI_PYODIDE || path.join(os.homedir(), ".cache", "process-improve", "node", "node_modules", "pyodide");
const runtime = JSON.parse(fs.readFileSync(path.join(repo, "web", "runtime.json"), "utf8"));

const start = performance.now();
const { loadPyodide } = await import(pathToFileURL(path.join(pyodideDir, "pyodide.mjs")).href);
const pyodide = await loadPyodide();
await pyodide.loadPackage([...runtime.packages, "pytest"], { messageCallback: () => {} });
pyodide.FS.mkdirTree("/repo");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: repo }, "/repo");
const wheel = path.basename(wheelPath);
pyodide.FS.mkdirTree("/wheels");
pyodide.FS.writeFile(`/wheels/${wheel}`, fs.readFileSync(wheelPath));
pyodide.globals.set("_pypi", pyodide.toPy([...runtime.micropip, "plotly"]));
await pyodide.runPythonAsync(`
import micropip
await micropip.install(list(_pypi))
await micropip.install("emfs:/wheels/${wheel}", deps=False)
`);
process.stderr.write(`Pyodide ${runtime.pyodide} ready in ${((performance.now() - start) / 1000).toFixed(1)} s\n`);

pyodide.globals.set("_argv", pyodide.toPy(argv));
const code = await pyodide.runPythonAsync(`
import os, runpy, sys
os.chdir("/repo")
args = list(_argv)
sys.argv = args[1:] if args[0] == "-m" else args
try:
    if args[0] == "-m":
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(args[0], run_name="__main__")
    exit_code = 0
except SystemExit as stop:
    exit_code = stop.code if isinstance(stop.code, int) else int(stop.code is not None)
exit_code
`);
process.exit(code);
