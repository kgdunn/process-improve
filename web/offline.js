// Keeping a copy of Factorial on the reader's device, so the page opens and works without a network.
//
// sw.js, the page's service worker, serves the copy and adds to it what passes through it. This module owns what
// the copy holds: it makes the copy, drops what earlier builds left in it, says in the page's footer what is kept,
// and removes it.
//
// 1. Once Python has started, the worker is registered and what this visit loaded is copied: the page's own files,
//    the Pyodide release with its packages, the wheel. The browser has just fetched them all, so the copy costs no
//    download. The worker serves a page from its next visit on, never one that is open: a first visit starts
//    Python and makes its copy with no worker in its way, and from the next visit on the worker keeps the copy up
//    with each build as its files pass. A browser that registers no worker is asked to keep nothing.
// 2. Charts and workbooks need plotly, openpyxl and Plotly.js, which start-up leaves out. A reader who wants the
//    whole page without a network asks for them with the button; a reader who never asks downloads none of it
//    until a chart or a workbook is first made.
// 3. What the footer says is read from the copy itself, never from a note of what was once copied: the page "opens
//    without a network" while the copy holds every file Python starts from, and "works" there while it holds every
//    file of the charts and workbooks too. A browser may drop a copy, and a note would go on claiming it.
//
// A build's manifest names every one of those files (scripts/build_web_app.py), so the copy can be checked file by
// file. The one note kept is the reader's own wish: a reader who removes the copy has none kept until they ask.
//
// Nothing here touches the document or the browser until installOffline is called, so tests/web/js reads the lists
// and the words under Node.js, and runs installOffline against a stand-in for the browser.

/** The copies' names, as sw.js has them (tests/web/page/test_offline.py keeps the two in step). */
export const KEPT = "factorial-kept-1";
export const APP = "factorial-app-1";

/** Where the reader's wish for no copy is noted. */
const WISH = "factorial-copy";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/";
const PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@";
const PYPI_FILES = "files.pythonhosted.org";

/** The folder of the Pyodide release a page loads, and of that release as an address on the CDN. */
const releaseOf = (runtime) => `v${runtime.pyodide}/${runtime.distribution}`;
const pyodideAt = (runtime) => `${PYODIDE}${releaseOf(runtime)}/`;

/** A package's name as a lock file's keys compare: lower case, with one kind of separator. */
const normal = (name) => name.toLowerCase().replace(/[-_.]+/g, "-");

/**
 * The files of a Pyodide release that a page with these packages loads: the interpreter, its standard library,
 * the lock file, and the wheel of each package and of all it depends on, as the lock file lists them.
 */
export function pyodideFiles(lock, packages) {
  const entries = new Map(Object.entries(lock.packages).map(([name, entry]) => [normal(name), entry]));
  const wheels = new Map();
  const todo = packages.map(normal);
  while (todo.length) {
    const name = todo.pop();
    if (wheels.has(name)) continue;
    const entry = entries.get(name);
    if (!entry) throw new Error(`Pyodide's lock file has no package ${name}`);
    wheels.set(name, entry.file_name);
    todo.push(...entry.depends.map(normal));
  }
  return ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json", ...wheels.values()];
}

/** The files Python starts from: the Pyodide release with the packages start-up loads, the wheel, the page's Python. */
export function startFiles({ scope, lock, runtime, manifest }) {
  const python = [manifest.wheel, manifest.glue].filter(Boolean).map((file) => `${scope}wheels/${file}`);
  return [...pyodideFiles(lock, runtime.packages).map((file) => pyodideAt(runtime) + file), ...python];
}

/**
 * The files of the charts and workbooks, which start-up leaves out: what the build resolved for the packages
 * installed on demand (its manifest's on_demand), and Plotly.js. null when the build could not resolve them: the
 * page then installs them through a package index, and what that will fetch cannot be listed, so cannot be kept
 * for certain.
 */
export function laterFiles({ lock, runtime, manifest }) {
  if (!manifest.on_demand || !manifest.plotly_js) return null;
  const groups = Object.values(manifest.on_demand);
  const started = new Set(pyodideFiles(lock, runtime.packages));
  const packages = pyodideFiles(lock, groups.flatMap((group) => group.packages)).filter((file) => !started.has(file));
  return [
    ...packages.map((file) => pyodideAt(runtime) + file),
    ...groups.flatMap((group) => group.wheels),
    `${PLOTLY_JS}${manifest.plotly_js}/plotly.min.js`,
  ];
}

/**
 * Whether a kept file is of an earlier build or release than the page now loads. `now` holds what this build
 * loads, each part or none: `folder`, the wheels' folder named by their digest; `pyodide`, the release and the
 * folder of it, as "v314.0.7/pyc"; `plotlyJs`, the Plotly.js version; `wheels`, the addresses of PyPI's files it
 * installs on demand. A file `now` says nothing of is kept.
 */
export function stale(href, scope, { folder, pyodide, plotlyJs, wheels }) {
  if (folder && href.startsWith(`${scope}wheels/`)) return !href.startsWith(`${scope}wheels/${folder}/`);
  if (pyodide && href.startsWith(PYODIDE)) return !href.startsWith(`${PYODIDE}${pyodide}/`);
  if (plotlyJs && href.startsWith(PLOTLY_JS)) return !href.startsWith(`${PLOTLY_JS}${plotlyJs}/`);
  if (wheels && new URL(href).hostname === PYPI_FILES) return !wheels.includes(href);
  return false;
}

/** What this build loads, as `stale` reads it. */
export const loadedNow = (runtime, manifest) => ({
  folder: manifest.wheel.split("/")[0],
  pyodide: releaseOf(runtime),
  plotlyJs: manifest.plotly_js,
  wheels: manifest.on_demand ? Object.values(manifest.on_demand).flatMap((group) => group.wheels) : undefined,
});

/** The words on the button that keeps more than is kept, or tries again what could not be finished. */
const KEEP_PAGE = "Keep a copy, to open it without a network";
const KEEP_WHOLE = "Keep its charts and workbooks too (about 12 MB)";
const TRY_AGAIN = "Try again";

/**
 * What the footer says of the copy in each state, and the buttons it offers: `keep` holds the words of the button
 * that keeps more, or none, and `remove` says whether the copy can be removed. `exact` says whether the build names
 * the files of the charts and workbooks: only then can they be kept for certain, and the button offered. `size` is
 * what the copy takes of the device's storage, in MB, where the browser says: the copy is made unasked, so a
 * reader is told what it costs beside the button that removes it.
 */
export function copyWords(state, { reason = "", exact = true, size = null } = {}) {
  const kept = size ? `a copy of it, about ${size} MB, is kept here` : "a copy of it is kept here";
  return {
    none: { said: "Factorial keeps no copy of itself on this device.", keep: KEEP_PAGE, remove: false },
    copying: { said: "Keeping a copy of Factorial...", keep: null, remove: false },
    opens: { said: `Factorial opens on this device without a network: ${kept}.`, keep: exact ? KEEP_WHOLE : null, remove: true },
    keeping: { said: "Keeping the charts and the workbooks, about 12 MB...", keep: null, remove: false },
    whole: {
      said: `Factorial opens and works on this device without a network: ${kept}, with its charts and workbooks.`,
      keep: null,
      remove: true,
    },
    failed: { said: `The copy of Factorial on this device is not whole: ${reason}`, keep: TRY_AGAIN, remove: true },
  }[state];
}

/**
 * Copy the addresses a cache lacks into it. A file the browser has just loaded comes from its own cache. One that
 * cannot be had fails the copy, unless the copy can do without it (`needed` false): it is then left out.
 */
async function copyInto(name, addresses, { needed = true } = {}) {
  const cache = await caches.open(name);
  for (const address of addresses) {
    if (await cache.match(address, { ignoreVary: true })) continue;
    const added = cache.add(address);
    await (needed ? added : added.catch(() => {}));
  }
}

/** Whether a cache is there and holds every one of these addresses. */
async function holds(name, addresses) {
  if (!(await caches.has(name))) return false;
  const cache = await caches.open(name);
  for (const address of addresses) {
    if (!(await cache.match(address, { ignoreVary: true }))) return false;
  }
  return true;
}

/** What this site keeps on the device, in MB to the nearest ten, as the browser estimates it; null where it says nothing. */
async function megabytesKept() {
  try {
    const { usage } = await navigator.storage.estimate();
    return usage ? Math.round(usage / 1e7) * 10 : null;
  } catch {
    return null; // a browser without the estimate, or one that refuses it
  }
}

/** Drop the kept files of earlier builds and releases, which `now` names the successors of. */
async function prune(scope, now) {
  const cache = await caches.open(KEPT);
  for (const request of await cache.keys()) if (stale(request.url, scope, now)) await cache.delete(request);
}

/** The app's own files no page opens without: the page itself, and what starts Python in its worker. */
const opening = (scope) => ["", "worker.js", "python-boot.js", "runtime.json", "wheels/manifest.json"].map((file) => scope + file);

/**
 * What else this visit loaded of the app, as the browser's record of the visit has it: its modules, its styles,
 * its pictures. The record also holds what was asked for and refused. A browser that notes how each request was
 * answered has those left out here; in any other, each file is copied if it can be had.
 */
const alsoLoaded = (scope) =>
  performance
    .getEntriesByType("resource")
    .filter((entry) => (entry.responseStatus ?? 200) === 200)
    .map((entry) => entry.name)
    .filter((address) => address.startsWith(scope) && !/\/(wheels\/[0-9a-f]{16}|api)\//.test(address));

/** The reader's wish for no copy; a browser that refuses storage keeps none, and the page asks nothing of it. */
function wishesNone() {
  try {
    return localStorage.getItem(WISH) === "none";
  } catch {
    return false;
  }
}
function wish(none) {
  try {
    if (none) localStorage.setItem(WISH, "none");
    else localStorage.removeItem(WISH);
  } catch {
    /* storage refused: the copy is made again at the next visit, and can be removed again */
  }
}

/** Unregister the worker and delete its copies: the build retires it, or the reader removes the copy. */
async function removeCopy() {
  const registration = await navigator.serviceWorker.getRegistration();
  await registration?.unregister();
  for (const name of await caches.keys()) if (name.startsWith("factorial-")) await caches.delete(name);
}

const read = async (address, options) => (await fetch(address, options)).json();

/**
 * Keep the copy and have the service worker serve it, saying in `box` what is kept.
 *
 * `manifest` is the build's wheels/manifest.json, or {} for a plain preview of web/, which keeps no copy: its
 * files change under the reader's hands. `started` resolves once Python has. `box` holds the words and the two
 * buttons, and takes the keyboard when a button pressed goes.
 */
export async function installOffline({ manifest, started, box }) {
  if (!globalThis.navigator?.serviceWorker || !globalThis.caches || !manifest.wheel) return;
  if (manifest.service_worker === false) {
    await removeCopy().catch(() => {}); // the build retires the worker; what the reader wished of the copy stays noted
    return;
  }
  const scope = new URL(".", location.href).href;
  const [said, keep, remove] = ["span", "#offline-keep", "#offline-remove"].map((part) => box.querySelector(part));
  let exact = true; // whether the build names the files of the charts and workbooks
  let wanted = "page"; // what was last asked for, which "Try again" tries: the page's copy, or the whole of it
  let busy = false; // one thing at a time is done to the copy

  const show = (state, { reason, size } = {}) => {
    const words = copyWords(state, { reason, exact, size });
    const within = box.contains(document.activeElement); // the button pressed may be about to go
    box.dataset.state = state;
    said.textContent = words.said;
    keep.textContent = words.keep ?? "";
    keep.hidden = !words.keep;
    remove.hidden = !words.remove;
    box.hidden = false;
    if (within) [keep, remove, box].find((part) => !part.hidden).focus();
  };

  /** The files of this build's copy, listed once Python has started: the browser then has each of them at hand. */
  async function listed() {
    await started;
    const runtime = await read("runtime.json", { cache: "no-store" });
    const lock = await read(`${pyodideAt(runtime)}pyodide-lock.json`);
    const later = laterFiles({ lock, runtime, manifest });
    exact = later !== null;
    return { start: startFiles({ scope, lock, runtime, manifest }), later, now: loadedNow(runtime, manifest) };
  }

  /** Say what the copy holds, as the copy itself has it, and what it takes of the device's storage. */
  async function say({ start, later }) {
    const opens = (await holds(KEPT, start)) && (await holds(APP, [scope]));
    const state = !opens ? "none" : later && (await holds(KEPT, later)) ? "whole" : "opens";
    show(state, { size: await megabytesKept() });
  }

  /**
   * Keep the page's copy ("page"), or its charts and workbooks too ("whole"), and say what is then kept. The
   * worker is registered first, so a browser that refuses one is asked to keep nothing: a copy with no worker to
   * serve it is of no use. Then the page's copy: what it lacks is added, and what earlier builds left is dropped.
   * `meanwhile` is the state shown while it is done; without one the reader asked for nothing, and a browser that
   * registers no worker is told nothing.
   */
  async function run(what, meanwhile) {
    if (busy) return;
    [busy, wanted] = [true, what];
    if (meanwhile) show(meanwhile);
    try {
      const files = await listed();
      const served = await navigator.serviceWorker.register("sw.js").then(
        () => true,
        (refused) => (meanwhile ? Promise.reject(refused) : false),
      );
      if (!served) return;
      await copyInto(KEPT, files.start);
      await copyInto(APP, opening(scope));
      await copyInto(APP, alsoLoaded(scope), { needed: false });
      await prune(scope, files.now);
      if (what === "whole" && files.later) await copyInto(KEPT, files.later);
      await say(files);
    } catch (err) {
      // No worker though the reader asked for a copy, a full disk, no network for a file, or Python did not start.
      // Unasked, it is said only of a copy that is there: with none, the reader was promised nothing.
      const copies = await caches.keys().catch(() => []);
      const reason = String(err?.message || err);
      if (meanwhile || copies.some((name) => name.startsWith("factorial-"))) show("failed", { reason });
    } finally {
      busy = false;
    }
  }

  keep.onclick = () => {
    const state = box.dataset.state;
    if (state === "none") wish(false); // the reader wants a copy again
    const what = state === "none" ? "page" : state === "failed" ? wanted : "whole";
    return run(what, what === "whole" ? "keeping" : "copying");
  };
  remove.onclick = async () => {
    if (busy) return;
    await removeCopy().catch(() => {});
    wish(true);
    show("none");
  };
  if (wishesNone()) show("none");
  else await run("page"); // quietly: on a visit after the first, the copy lacks nothing
}
