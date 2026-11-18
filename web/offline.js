// Keeping a copy of Factorial on the reader's device, so the page opens and works without a network.
//
// sw.js, the page's service worker, serves the copy and adds to it what passes through it. This module owns what
// the copy holds: it makes the copy, drops what earlier builds left in it, says in the page's footer what is kept,
// and removes it.
//
// 1. Once Python has started, and has fetched what a model is fitted with, the worker is registered and what this
//    visit loaded is copied: the page's own files, the Pyodide release with its packages, the wheel. The browser
//    has just fetched them all, so the copy costs no download. The worker serves a page from its next visit on,
//    never one that is open: a first visit starts Python and makes its copy with no worker in its way, and from
//    the next visit on the worker keeps the copy up with each build as its files pass. A browser that registers
//    no worker is asked to keep nothing.
// 2. Charts and workbooks need plotly, openpyxl and Plotly.js, which start-up leaves out: about 12 MB, and a
//    download of its own unless the visit has made a chart and a workbook. They are copied too, unasked, so that
//    the whole page works without a network and the footer has one thing to say. A browser that asks sites to
//    save data (navigator.connection.saveData) is not sent them unasked: its reader adds them with the button.
//    So does a reader whose copy of them could not be finished, which the next visit tries again in any case.
// 3. What the footer says is read from the copy itself, never from a note of what was once copied: the page "opens
//    without a network" while the copy holds every file Python starts from, and "works" there while it holds every
//    file of the charts and workbooks too. A browser may drop a copy, and a note would go on claiming it.
//
// A build's manifest names every one of those files (scripts/build_web_app.py), so the copy can be checked file by
// file. The one note kept is the reader's own wish: a reader who removes the copy has none kept until they ask.
//
// Nothing here touches the document or the browser until installOffline is called, so tests/web/js reads the lists
// and the words under Node.js, and runs installOffline against a stand-in for the browser.

import { ASK_AGAIN, lockEntries } from "./python-boot.js";

/**
 * The copies' names, as sw.js has them (tests/web/page/test_offline.py keeps the two in step), and the mark that
 * the page has made a copy: an empty cache, made before the copies and removed before them. The worker adds
 * nothing to a copy without its mark, so a copy removed here is not brought back by a worker still at work.
 */
export const KEPT = "factorial-kept-1";
export const APP = "factorial-app-1";
export const MADE = "factorial-made-1";

/** Where the reader's wish for no copy is noted. */
const WISH = "factorial-copy";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/";
const PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@";
const PYPI_FILES = "files.pythonhosted.org";

/** The folder of the Pyodide release a page loads, and of that release as an address on the CDN. */
const releaseOf = (runtime) => `v${runtime.pyodide}/${runtime.distribution}`;
const pyodideAt = (runtime) => `${PYODIDE}${releaseOf(runtime)}/`;

/**
 * The files of a Pyodide release that a page with these packages loads: the interpreter, its standard library,
 * the lock file, and the wheel of each package and of all it depends on, as the lock file lists them.
 */
export function pyodideFiles(lock, packages) {
  const wheels = lockEntries(lock, packages).map((entry) => entry.file_name);
  return ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json", ...wheels];
}

/** The packages of the Pyodide release every visit loads: those Python starts with, and those a model is fitted with. */
const loaded = (runtime) => [...runtime.packages, ...runtime.fitting];

/**
 * The files Python starts from and fits a model with: the Pyodide release with the packages every visit loads, the
 * wheel, the page's Python. A page that opens without a network designs and analyses there.
 */
export function startFiles({ scope, lock, runtime, manifest }) {
  const python = [manifest.wheel, manifest.glue].filter(Boolean).map((file) => `${scope}wheels/${file}`);
  return [...pyodideFiles(lock, loaded(runtime)).map((file) => pyodideAt(runtime) + file), ...python];
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
  const started = new Set(pyodideFiles(lock, loaded(runtime)));
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

/** The words on the button that saves more than is saved, or tries again what could not be finished. */
const KEEP_PAGE = "Save Factorial for offline use";
const KEEP_WHOLE = "Add charts and Excel files (12 MB more)";
const TRY_AGAIN = "Try again";

/**
 * What the footer says of the copy in each state, and the buttons it offers. Each state opens with the answer to
 * the one question a reader has, whether Factorial works without internet here, and each button says what pressing
 * it does. What is saved is called Factorial's tools, so that no reader takes it for their own studies; the whole
 * copy says where working offline helps. `keep` holds the words of the button that saves more, or none, and `remove` says whether the copy can
 * be removed. `exact` says whether the build names the files of the charts and Excel files: only then can they be
 * saved for certain, and the button offered. `size` is what the copy takes of the device's storage, in MB, where the
 * browser says: the copy is made unasked, so a reader is told what it costs beside the button that removes it.
 * The whole copy is made unasked, so `opens` is said only where the charts and Excel files are not there: a
 * build that names none, a browser that asks to save data, or a download of them that did not finish.
 */
export function copyWords(state, { reason = "", exact = true, size = null } = {}) {
  const saved = size ? `Factorial's tools are saved on this device (about ${size} MB).` : "Factorial's tools are saved on this device.";
  const where = "for example when travelling, or in a factory or laboratory without internet";
  return {
    none: { said: "Needs internet. Factorial's tools are not saved on this device.", keep: KEEP_PAGE, remove: false },
    copying: { said: "Saving Factorial for offline use...", keep: null, remove: false },
    opens: { said: `Works offline, except charts and Excel files. ${saved}`, keep: exact ? KEEP_WHOLE : null, remove: true },
    keeping: { said: "Adding charts and Excel files, about 12 MB...", keep: null, remove: false },
    whole: { said: `Works fully offline, ${where}. ${saved}`, keep: null, remove: true },
    failed: { said: `Not fully saved: ${reason}`, keep: TRY_AGAIN, remove: true },
  }[state];
}

/**
 * Copy the addresses a cache lacks into it. A file the browser has just loaded comes from its own cache. One that
 * cannot be had fails the copy, unless the copy can do without it (`needed` false): it is then left out.
 *
 * A file the copy needs that does not arrive is asked for again, as the start asks for its packages again
 * (python-boot.js ASK_AGAIN): a CDN that refuses a file serves it a moment later, and a copy of some hundred files
 * would else fail whole for one of them. `pause(ms)` waits before each further try.
 */
async function copyInto(name, addresses, { needed = true, pause = paused } = {}) {
  const cache = await caches.open(name);
  for (const address of addresses) {
    if (await cache.match(address, { ignoreVary: true })) continue;
    if (!needed) {
      await cache.add(address).catch(() => {});
      continue;
    }
    for (const ms of [0, ...ASK_AGAIN]) {
      if (ms) await pause(ms);
      const refused = await cache.add(address).then(() => null, (error) => error); // prettier-ignore
      if (!refused) break;
      if (ms === ASK_AGAIN.at(-1)) throw refused;
    }
  }
}

/** Wait some milliseconds. */
const paused = (ms) => new Promise((go) => setTimeout(go, ms));

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
 * The app's modules a visit loads only when it has drawn a chart: the window a figure opens in (figure.js). A
 * copy made by a visit that drew none holds them all the same, so a chart drawn without a network opens.
 */
const onDemand = (scope) => ["figure.js"].map((file) => scope + file);

/**
 * The pages of the app this page never loads, which its footer leads to and which are small: the list of what
 * Factorial does (features.html), with the module that reads its document and the document; how Factorial works
 * (about.html), which reads only files the copy holds; and the mark beside their way back. A copy holds them all
 * the same, so both open without a network. The design catalog and the concept map are megabytes each: they are
 * kept once a visit with a network opens them, and until then the worker says they need one (sw.js).
 */
const linked = (scope) => ["features.html", "features.js", "FEATURES.md", "about.html", "favicon.svg", "site.js"].map((file) => scope + file);

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

/**
 * Unregister the worker and delete the copies: the build retires it, or the reader removes the copy. The mark
 * goes first, so the worker adds nothing more; then each copy, whatever becomes of the one before it.
 */
async function removeCopy() {
  await caches.delete(MADE);
  const registration = await navigator.serviceWorker.getRegistration();
  await registration?.unregister();
  const copies = (await caches.keys()).filter((name) => name.startsWith("factorial-"));
  await Promise.allSettled(copies.map((name) => caches.delete(name)));
}

const read = async (address, options) => (await fetch(address, options)).json();

/**
 * Keep the copy and have the service worker serve it, saying in `box` what is kept.
 *
 * `manifest` is the build's wheels/manifest.json, or {} for a plain preview of web/, which keeps no copy: its
 * files change under the reader's hands. `started` resolves once Python has started and has fetched what a model
 * is fitted with, and rejects if it cannot start. `box` holds the words and the two buttons, and takes the keyboard
 * when a button pressed goes.
 */
export async function installOffline({ manifest, started, box, pause = paused }) {
  if (!globalThis.navigator?.serviceWorker || !globalThis.caches || !manifest.wheel) return;
  if (manifest.service_worker === false) {
    await removeCopy().catch(() => {}); // the build retires the worker; what the reader wished of the copy stays noted
    // The worker serves this page until it is closed. A look it had begun in the copies as they went can bring one
    // of them back, empty, some tens of milliseconds later (Chromium, issue 1248): no call of the page or of the
    // worker makes it, and so none can be guarded. By the time Python has started, or has failed to, every file
    // the page asked for as it opened has arrived and no such look is left: what came back meanwhile goes then.
    await started.catch(() => {});
    await removeCopy().catch(() => {});
    return;
  }
  const scope = new URL(".", location.href).href;
  const [said, keep, remove] = ["span", "#offline-keep", "#offline-remove"].map((part) => box.querySelector(part));
  let exact = true; // whether the build names the files of the charts and workbooks
  // What a copy is made of unasked: the whole of it, or the page's copy alone in a browser that asks to save data.
  const most = globalThis.navigator.connection?.saveData === true ? "page" : "whole";
  let wanted = most; // what was last asked for, which "Try again" tries: the page's copy, or the whole of it
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

  /** The files of this build's copy, listed once Python has them all: the browser then has each of them at hand. */
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
   * registers no worker is told nothing. Nor is a reader told that the charts and workbooks could not be added
   * unasked: the copy is said as it stands, with the button that adds them, and the next visit tries again.
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
      await caches.open(MADE); // the mark first: from here on the worker may add to the copy
      await copyInto(KEPT, files.start, { pause });
      await copyInto(APP, opening(scope), { pause });
      await copyInto(APP, [...alsoLoaded(scope), ...onDemand(scope), ...linked(scope)], { needed: false });
      await prune(scope, files.now);
      if (what === "whole" && files.later) {
        const added = copyInto(KEPT, files.later, { pause });
        await (meanwhile ? added : added.catch(() => {}));
      }
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
    const what = state === "none" ? most : state === "failed" ? wanted : "whole";
    return run(what, state === "opens" ? "keeping" : "copying");
  };
  remove.onclick = async () => {
    if (busy) return;
    await removeCopy().catch(() => {});
    wish(true);
    show("none");
  };
  if (wishesNone()) {
    // A write the worker was in the middle of as the copy went can bring a cache back: it goes at the next visit.
    await removeCopy().catch(() => {});
    show("none");
  } else await run(most); // quietly: on a visit after the first, the copy lacks nothing
}
