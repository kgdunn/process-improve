// Keeping a copy of Factorial on the reader's device, so the page opens and works without a network.
//
// sw.js, the page's service worker, serves the copy; this module registers it, fills the copy, and says in the
// page's footer what is kept. Three steps, each asked of the reader no more than it must be:
//
// 1. The worker is registered for a built app (its wheels/manifest.json names a wheel), unless the build says
//    "service_worker": false, which retires the worker and its copies instead.
// 2. Once Python has started, what this visit loaded is copied: the page's own files, the Pyodide release with its
//    packages, the wheel. The browser has just fetched them all, so the copy costs no download, and the page then
//    opens without a network. The worker alone would keep them only from the reader's second visit.
// 3. Charts and workbooks need plotly, openpyxl and Plotly.js, which start-up leaves out. A reader who wants the
//    whole page without a network asks for them with the button; a reader who never asks downloads none of it.
//
// Nothing here touches the document or the browser until installOffline is called, so tests/web/js reads the
// parts that need neither under Node.js.

/** The copies' names, as sw.js has them (tests/web/page/test_offline.py keeps the two in step). */
export const KEPT = "factorial-kept-1";
export const APP = "factorial-app-1";

/** Where the page notes what it has copied, and for which build: a later build copies again. */
const COPIED = "factorial-copy";

const PYODIDE = (runtime) => `https://cdn.jsdelivr.net/pyodide/v${runtime.pyodide}/${runtime.distribution}/`;
const PLOTLY_JS = (version) => `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/plotly.min.js`;

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

/** What names a copy: the build's Python, and the Pyodide it runs on. Another build's copy is made again. */
export const copyKey = (runtime, manifest) => `${manifest.wheel}|${runtime.pyodide}/${runtime.distribution}`;

/** What names the copy of the charts and workbooks: the copy it adds to, and the releases of what it adds. */
export const wholeKey = (runtime, manifest) =>
  [copyKey(runtime, manifest), manifest.plotly, manifest.plotly_js, ...runtime.micropip].join("|");

/** The words on the button that keeps more than is kept: the page itself, or its charts and workbooks too. */
const KEEP_PAGE = "Keep a copy, to open it without a network";
const KEEP_WHOLE = "Keep its charts and workbooks too (about 12 MB)";

/**
 * What the footer says of the copy in each state, and the buttons it offers: `keep` holds the words of the button
 * that keeps more, or none, and `remove` says whether the copy can be removed.
 */
export function copyWords(state, reason = "") {
  const kept = "a copy of it is kept here";
  return {
    none: { said: "Factorial keeps no copy of itself on this device.", keep: KEEP_PAGE, remove: false },
    opens: { said: `Factorial opens on this device without a network: ${kept}.`, keep: KEEP_WHOLE, remove: true },
    keeping: { said: "Keeping the charts and the workbooks, about 12 MB...", keep: null, remove: false },
    whole: {
      said: `Factorial opens and works on this device without a network: ${kept}, with its charts and workbooks.`,
      keep: null,
      remove: true,
    },
    failed: { said: `The copy of the charts and workbooks could not be finished: ${reason}`, keep: KEEP_WHOLE, remove: true },
  }[state];
}

/** Copy the addresses a cache lacks into it. A file the browser has just loaded comes from its own cache. */
async function copyInto(name, addresses) {
  const cache = await caches.open(name);
  for (const address of addresses) {
    if (!(await cache.match(address, { ignoreVary: true }))) await cache.add(address);
  }
}

/** The app's own files a page needs to open: itself, what this visit loaded of the app, and what its worker loads. */
function shell(scope) {
  const loaded = performance
    .getEntriesByType("resource")
    .map((entry) => entry.name)
    .filter((address) => address.startsWith(scope) && !/\/(wheels\/[0-9a-f]{16}|api)\//.test(address));
  const named = ["", "worker.js", "python-boot.js", "runtime.json", "wheels/manifest.json", "manifest.webmanifest"];
  return [...new Set([...named.map((file) => scope + file), ...loaded])];
}

/** Copy what the page needs to open and to start Python: its own files, the Pyodide release, the wheel. */
async function copyPage(scope, runtime, manifest) {
  const release = PYODIDE(runtime);
  const lock = await (await fetch(`${release}pyodide-lock.json`)).json();
  const python = [manifest.wheel, manifest.glue].filter(Boolean).map((file) => `${scope}wheels/${file}`);
  await copyInto(KEPT, [...pyodideFiles(lock, runtime.packages).map((file) => release + file), ...python]);
  await copyInto(APP, shell(scope));
}

/** Read and write the note of what is copied; a browser that refuses storage keeps no note, and copies again. */
function noted() {
  try {
    return JSON.parse(localStorage.getItem(COPIED)) ?? {};
  } catch {
    return {};
  }
}
function note(value) {
  try {
    if (value) localStorage.setItem(COPIED, JSON.stringify(value));
    else localStorage.removeItem(COPIED);
  } catch {
    /* storage refused */
  }
}

/** Unregister the worker and delete its copies: the build retires it, or the reader removes the copy. */
async function removeCopy() {
  note(null);
  const registration = await navigator.serviceWorker.getRegistration();
  await registration?.unregister();
  for (const name of await caches.keys()) if (name.startsWith("factorial-")) await caches.delete(name);
}

/**
 * Register the service worker and keep the copy, saying in `box` what is kept.
 *
 * `manifest` is the build's wheels/manifest.json, or {} for a plain preview of web/, which keeps no copy: its
 * files change under the reader's hands. `started` resolves once Python has, and `keepPackages()` asks the Python
 * worker for every package a call might need, resolving to its reply. `box` holds the words and the two buttons.
 * A reader who removes the copy has none kept again until they ask for one: the note of that is kept instead.
 */
export async function installOffline({ manifest, started, keepPackages, box }) {
  if (!("serviceWorker" in navigator) || !("caches" in window) || !manifest.wheel) return;
  if (manifest.service_worker === false) {
    await removeCopy().catch(() => {});
    return;
  }
  const scope = new URL(".", location.href).href;
  const [said, keep, remove] = ["span", "#offline-keep", "#offline-remove"].map((part) => box.querySelector(part));
  const show = (state, reason) => {
    const words = copyWords(state, reason);
    said.textContent = words.said;
    keep.textContent = words.keep ?? "";
    keep.hidden = !words.keep;
    remove.hidden = !words.remove;
    box.hidden = false;
  };

  /** Copy the page, unless this build's is kept already, and say what is kept. */
  async function keepPage() {
    await navigator.serviceWorker.register("sw.js");
    await started;
    const runtime = await (await fetch("runtime.json", { cache: "no-store" })).json();
    const [page, whole] = [copyKey(runtime, manifest), wholeKey(runtime, manifest)];
    // Copied for this build already, unless the browser has since dropped the copy, as it may when short of room.
    if (noted().page !== page || !(await caches.has(KEPT))) {
      await copyPage(scope, runtime, manifest);
      note({ page });
    }
    show(noted().whole === whole ? "whole" : "opens");
    return { page, whole };
  }

  /** Add the charts and workbooks: the packages a call may need, which the worker installs, and Plotly.js. */
  async function keepWhole() {
    show("keeping");
    try {
      const keys = await keepPage();
      const reply = await keepPackages();
      if (!reply.ok) throw new Error(reply.error);
      if (manifest.plotly_js) await copyInto(KEPT, [PLOTLY_JS(manifest.plotly_js)]);
      note(keys);
      show("whole");
    } catch (err) {
      show("failed", err.message);
    }
  }

  keep.onclick = () => {
    if (!noted().off) return keepWhole();
    note(null); // the reader wants a copy again
    return keepPage().catch(() => show("none"));
  };
  remove.onclick = async () => {
    await removeCopy().catch(() => {});
    note({ off: true });
    show("none");
  };
  if (noted().off) show("none");
  else await keepPage().catch(() => {}); // no worker, no storage, or no network for the copy: the page works without
}
