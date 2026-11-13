// Starting Python in a page's worker: the Pyodide release and packages runtime.json names, the process-improve
// wheel the build made, and then the page's own Python. The designed-experiments page and the multivariate page
// both start it here, so both run one Python and start it one way.

/** An error in one line: a Python traceback's last line names the exception and says what went wrong. */
export const oneLine = (err) =>
  String(err?.message || err)
    .trim()
    .split("\n")
    .filter(Boolean)
    .at(-1) ?? "unknown error";

/** One of the app's JSON files, fetched past the browser's cache: a deploy changes them, and they are small. */
async function read(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${new URL(url).pathname.split("/").at(-1)} could not be read (HTTP ${res.status})`);
  return res.json();
}

/** Install packages with micropip, which takes them from PyPI, or from the address given. */
export async function install(pyodide, requirements, { deps = true } = {}) {
  const micropip = pyodide.pyimport("micropip");
  const wanted = pyodide.toPy(requirements);
  try {
    await micropip.install.callKwargs(wanted, { deps });
  } finally {
    wanted.destroy();
    micropip.destroy();
  }
}

/**
 * Start Python in this worker, and resolve to {pyodide, runtime, manifest}.
 *
 * `root` is the address of the folder that holds runtime.json and wheels/, and `say(status)` is told how far the
 * start has got. runtime.json names the Pyodide release, the folder of it on the CDN, and the packages loaded
 * from it; the wasm tests and CI read the same file. The build writes wheels/manifest.json, which names the wheel
 * built from its commit; without one (web/ served as it is) the wheel is PyPI's release.
 */
export async function startPython(root, say) {
  const runtime = await read(new URL("runtime.json", root));
  const manifest = await read(new URL("wheels/manifest.json", root)).catch(() => ({}));
  say("Loading Python and its packages...");
  // The release's "pyc" folder holds the same packages as "full", their Python compiled to bytecode already:
  // nothing is compiled as it is imported, which halves the start.
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${runtime.pyodide}/${runtime.distribution}/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`); // here, so a refused download stops the start
  // Named here, the packages download while the interpreter starts, not after it.
  const pyodide = await loadPyodide({ indexURL, packages: runtime.packages });
  say("Installing process-improve...");
  // deps: false, since every package the page runs is loaded above, from the Pyodide release, or installed when a
  // call first needs it, pinned: micropip must not fetch an unpinned dependency of its own, and the release need
  // not meet the wheel's version floors. What the experiments module needs is checked by
  // tests/web/api/test_browser_safe.py, and the wasm tier checks its answers against native Python's.
  const wheel = manifest.wheel ? new URL(`wheels/${manifest.wheel}`, root).href : "process-improve";
  await install(pyodide, [wheel], { deps: false });
  return { pyodide, runtime, manifest };
}

/** The package a reply says its call lacks (bootstrap.py's _Needs). Such a reply is short; no other is read here. */
const lacking = (reply) => (reply.length < 200 ? JSON.parse(reply).needs : undefined);

/**
 * Call an entry point, and resolve to its reply, the JSON text Python returned.
 *
 * Start-up leaves out the packages few visits need. A call that needs one says so, in place of its answer;
 * `provide(name)` then installs it, and the call is made again: an entry point changes nothing outside its reply,
 * so asking twice is asking once. A package still lacking once it was provided is an error, not a loop.
 */
export async function answer(entry, payload, provide) {
  let reply = entry(payload);
  for (let needs = lacking(reply), had = new Set(); needs; needs = lacking(reply)) {
    if (had.has(needs)) throw new Error(`Python still lacks ${needs} after installing it.`);
    had.add(needs);
    await provide(needs);
    reply = entry(payload);
  }
  return reply;
}

/**
 * Load a page's own Python as the module `name`, and return it.
 *
 * `archive` is the address of the build's zip of it: its source and, when the build compiled it for the Python
 * Pyodide runs, its bytecode, which is then not compiled here. Without one, `source` is the address of the file
 * itself, as a plain preview of web/ serves it.
 */
export async function loadGlue(pyodide, name, { archive, source }) {
  const site = pyodide.runPython("import site; site.getsitepackages()[0]");
  const fetched = await fetch(archive ?? source, archive ? {} : { cache: "no-store" });
  if (!fetched.ok) throw new Error(`${name}.py could not be read (HTTP ${fetched.status})`);
  if (archive) pyodide.unpackArchive(await fetched.arrayBuffer(), "zip", { extractDir: site });
  else pyodide.FS.writeFile(`${site}/${name}.py`, await fetched.text());
  pyodide.runPython("import importlib; importlib.invalidate_caches()"); // the folder was listed before the file came
  return pyodide.pyimport(name);
}
