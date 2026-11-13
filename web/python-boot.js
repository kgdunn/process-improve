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

/** A package's name as a lock file's keys compare: lower case, with one kind of separator. */
const normal = (name) => name.toLowerCase().replace(/[-_.]+/g, "-");

/** A Pyodide release's lock file entry for each of these packages and for all each depends on, those named first. */
export function lockEntries(lock, packages) {
  const entries = new Map(Object.entries(lock.packages).map(([name, entry]) => [normal(name), entry]));
  const found = new Map();
  const todo = packages.map(normal);
  for (let at = 0; at < todo.length; at += 1) {
    const name = todo[at];
    if (found.has(name)) continue;
    const entry = entries.get(name);
    if (!entry) throw new Error(`Pyodide's lock file has no package ${name}`);
    found.set(name, entry);
    todo.push(...entry.depends.map(normal));
  }
  return [...found.values()];
}

/**
 * Install what the build resolved to exact files (exact_files in scripts/build_web_app.py): packages of the Pyodide
 * release, loaded from it by name, and wheels from the addresses the build found for them. Nothing is resolved
 * here, so no package index is asked and every visit installs the same files.
 *
 * Each package is asked for by name with all it depends on, as the release's lock file lists them. Pyodide loads
 * them unasked, but installs a package although one it depends on could not be downloaded, which it only logs,
 * and never looks again at what a loaded package depends on. Named, a package that is missing fails the install
 * here, and the next try asks for it again.
 */
export async function installExact(pyodide, { packages, wheels }) {
  const names = lockEntries(pyodide.lockfile, packages).map((entry) => entry.name);
  if (names.length) await pyodide.loadPackage(names);
  const lacking = names.filter((name) => !pyodide.loadedPackages[name]);
  if (lacking.length) throw new Error(`${lacking.join(", ")} could not be loaded from the Pyodide release`);
  if (wheels.length) await install(pyodide, wheels, { deps: false });
}

/**
 * What start-up leaves out, since designing an experiment needs none of it: for each module Python may say a call
 * lacks (bootstrap.py's ON_DEMAND), what is installed for it, and what the status line calls it and says it is for.
 *
 * - openpyxl comes with the first workbook, written or read, and plotly with the first chart, at the release the
 *   build drew the worked example with. A build resolves these pins to exact files (its manifest's on_demand),
 *   which are installed as they are; the pins themselves are resolved through the package index by a plain preview
 *   of web/, which has no manifest, and by a build that could not ask PyPI.
 * - statsmodels and scikit-learn are what a model is fitted with (runtime.json's `fitting`): packages of the
 *   Pyodide release, a third of what the page downloads. They are loaded together, with joblib and threadpoolctl,
 *   which the release loads for scikit-learn, whichever of the four modules a call lacks first.
 */
export function onDemand(runtime, manifest) {
  const plotly = manifest.plotly ? `plotly==${manifest.plotly}` : "plotly";
  const fits = { what: runtime.fitting.join(" and "), purpose: "fitting models", exact: { packages: runtime.fitting, wheels: [] } };
  return {
    openpyxl: { what: "openpyxl", purpose: "Excel workbooks", exact: manifest.on_demand?.openpyxl, pins: runtime.micropip },
    plotly: { what: "plotly", purpose: "the charts", exact: manifest.on_demand?.plotly, pins: [plotly] },
    statsmodels: fits,
    sklearn: fits,
    joblib: fits,
    threadpoolctl: fits,
  };
}

/**
 * Return `provide(name, { quiet })`, which installs what `table` holds for the module `name` and resolves when it
 * is there. An entry is installed once, however many calls find it missing at the same time and whichever of its
 * modules they name; one that could not be installed is tried again by the next call that needs it.
 *
 * `say(status, state)` is told as start-up tells it, once a call waits for the install: it ends its own note, since
 * the page may say nothing more after it (the first chart may be the curves' fit). An install nothing waits for is
 * quiet: the worker's own, made in the background once the page is ready. A call that then needs it joins the same
 * install, and from there on it is said.
 *
 * Python refuses to import any of these packages until it is told that one is in (bootstrap.py's _NotYet): a
 * package's files are unpacked before its compiled modules are loaded, and a call answered in that gap would
 * import it half loaded. So once an entry is in, `glue.installed` is told the modules it holds, and then
 * `installed(entry)` is called, both before any call that waited for the entry is made again.
 */
export function provider(pyodide, glue, table, say, installed = () => {}) {
  const installs = new Map(); // by entry: its install, whether it is in, and whether a call waits for it
  const modulesOf = (entry) => Object.keys(table).filter((name) => table[name] === entry);
  return function provide(name, { quiet = false } = {}) {
    const wanted = table[name];
    if (!wanted) return Promise.reject(new Error(`Python needs ${name}, which this page does not install.`));
    let going = installs.get(wanted);
    if (!going) {
      going = { there: false, aloud: false };
      going.done = (async () => {
        try {
          await (wanted.exact ? installExact(pyodide, wanted.exact) : install(pyodide, wanted.pins));
        } catch (err) {
          installs.delete(wanted); // the next call that needs it tries the install again
          const reason = `Could not install ${wanted.what}: ${oneLine(err)}`;
          if (going.aloud) say(reason, "error");
          throw new Error(reason); // what the call is answered with, too
        }
        going.there = true;
        glue.installed(JSON.stringify(modulesOf(wanted)));
        if (going.aloud) say(`Installed ${wanted.what} for ${wanted.purpose}.`, "ok");
        installed(wanted);
      })();
      installs.set(wanted, going);
    }
    if (!quiet && !going.aloud && !going.there) {
      going.aloud = true;
      say(`Installing ${wanted.what} for ${wanted.purpose}...`, "busy");
    }
    return going.done;
  };
}

/**
 * Start Python in this worker, and resolve to {pyodide, runtime, manifest}.
 *
 * `root` is the address of the folder that holds runtime.json and wheels/, and `say(status)` is told how far the
 * start has got. runtime.json names the Pyodide release, the folder of it on the CDN, and the packages loaded
 * from it: `packages` at the start, and `fitting` when a model is first fitted, or at the start too for a page
 * that fits from its first step (`fitting: true`). The wasm tests and CI read the same file. The build writes
 * wheels/manifest.json, which names the wheel built from its commit; without one (web/ served as it is) the wheel
 * is PyPI's release.
 */
export async function startPython(root, say, { fitting = false } = {}) {
  const runtime = await read(new URL("runtime.json", root));
  const manifest = await read(new URL("wheels/manifest.json", root)).catch(() => ({}));
  say("Loading Python and its packages...");
  // The release's "pyc" folder holds the same packages as "full", their Python compiled to bytecode already:
  // nothing is compiled as it is imported, which halves the start.
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${runtime.pyodide}/${runtime.distribution}/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`); // here, so a refused download stops the start
  // Named here, the packages download while the interpreter starts, not after it.
  const packages = fitting ? [...runtime.packages, ...runtime.fitting] : runtime.packages;
  const pyodide = await loadPyodide({ indexURL, packages });
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
