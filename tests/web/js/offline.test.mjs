// web/offline.js: what the copy for use without a network is made of, what the footer says of it, and
// installOffline run against a stand-in for the browser (standins.mjs), visit after visit. sw.test.mjs has the
// service worker's side; that a real browser opens the page from the copy is the live tier's test.

import assert from "node:assert/strict";
import { beforeEach, describe, test } from "node:test";

import { APP, KEPT, copyWords, installOffline, laterFiles, loadedNow, pyodideFiles, stale, startFiles } from "../../../web/offline.js";
import worker from "../../../web/sw.js";
import { CacheStorage, Network } from "./standins.mjs";

const SCOPE = "https://factori.al/";
const RELEASE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/pyc/";
const PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js";
const PYPI = "https://files.pythonhosted.org/packages/ab/cd/";
const [OPENPYXL, ET_XMLFILE, PLOTLY] = ["openpyxl-3.1.5-py2.py3", "et_xmlfile-2.0.0-py3", "plotly-7.1.0-py3"].map((name) => `${PYPI}${name}-none-any.whl`);

const lock = {
  packages: {
    numpy: { file_name: "numpy.whl", depends: [] },
    pandas: { file_name: "pandas.whl", depends: ["numpy", "python-dateutil"] },
    "python-dateutil": { file_name: "dateutil.whl", depends: ["six"] },
    six: { file_name: "six.whl", depends: [] },
    "scikit-learn": { file_name: "sklearn.whl", depends: ["numpy"] },
    narwhals: { file_name: "narwhals.whl", depends: [] },
    packaging: { file_name: "packaging.whl", depends: [] },
    unused: { file_name: "unused.whl", depends: [] },
  },
};
const runtime = { pyodide: "314.0.7", distribution: "pyc", packages: ["pandas"], micropip: ["openpyxl==3.1.5", "et-xmlfile==2.0.0"] };
const FOLDER = "0123456789abcdef";
const manifest = {
  wheel: `${FOLDER}/process_improve-2.5.0-py3-none-any.whl`,
  glue: `${FOLDER}/bootstrap.zip`,
  service_worker: true,
  plotly: "7.1.0",
  plotly_js: "4.1.1",
  on_demand: {
    openpyxl: { packages: [], wheels: [OPENPYXL, ET_XMLFILE] },
    plotly: { packages: ["narwhals", "packaging"], wheels: [PLOTLY] },
  },
};
const listing = { scope: SCOPE, lock, runtime, manifest };
const START = startFiles(listing);
const LATER = laterFiles(listing);

describe("the files the copy is made of", () => {
  test("a release's are the interpreter, its library, the lock file, and each package with all it depends on, once", () => {
    const files = pyodideFiles(lock, ["pandas", "scikit_learn"]);
    assert.deepEqual(files.slice(0, 5), ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]);
    assert.deepEqual(files.slice(5).sort(), ["dateutil.whl", "numpy.whl", "pandas.whl", "six.whl", "sklearn.whl"]);
  });

  test("a package the release does not have is an error, not a gap in the copy", () => {
    assert.throws(() => pyodideFiles(lock, ["pandas", "polars"]), /no package polars/);
  });

  test("Python starts from the release with its start-up packages, the wheel and the page's own Python", () => {
    assert.deepEqual(START.slice(0, 5), ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"].map((file) => RELEASE + file));
    assert.deepEqual(START.slice(5).sort(), [
      ...["dateutil.whl", "numpy.whl", "pandas.whl", "six.whl"].map((file) => RELEASE + file),
      `${SCOPE}wheels/${FOLDER}/bootstrap.zip`,
      `${SCOPE}wheels/${FOLDER}/process_improve-2.5.0-py3-none-any.whl`,
    ]);
  });

  test("the charts and workbooks add what the build resolved, with the release's packages they need, and Plotly.js", () => {
    assert.deepEqual(LATER, [`${RELEASE}packaging.whl`, `${RELEASE}narwhals.whl`, OPENPYXL, ET_XMLFILE, PLOTLY, PLOTLY_JS]);
    assert.ok(!LATER.some((file) => START.includes(file)), "nothing Python started from is listed again");
  });

  test("a build that resolved nothing lists none: what an index would give cannot be named", () => {
    assert.equal(laterFiles({ ...listing, manifest: { ...manifest, on_demand: null } }), null);
  });

  test("are under the names the service worker serves from", () => {
    assert.deepEqual([KEPT, APP], [worker.KEPT, worker.APP]);
  });
});

describe("which kept files a later build leaves behind", () => {
  const now = loadedNow(runtime, { ...manifest, wheel: "bbbbbbbbbbbbbbbb/process_improve.whl" });

  test("what the build loads is read from its manifest and the release it runs on", () => {
    assert.deepEqual(now, { folder: "bbbbbbbbbbbbbbbb", pyodide: "v314.0.7/pyc", plotlyJs: "4.1.1", wheels: [OPENPYXL, ET_XMLFILE, PLOTLY] });
  });

  test("the wheel and the page's Python of an earlier build go, and this build's stay", () => {
    assert.ok(stale(`${SCOPE}wheels/aaaaaaaaaaaaaaaa/bootstrap.zip`, SCOPE, now));
    assert.ok(!stale(`${SCOPE}wheels/bbbbbbbbbbbbbbbb/bootstrap.zip`, SCOPE, now));
  });

  test("another Pyodide release, or another folder of this one, goes", () => {
    assert.ok(stale("https://cdn.jsdelivr.net/pyodide/v314.0.6/pyc/pyodide.asm.wasm", SCOPE, now));
    assert.ok(stale("https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.asm.wasm", SCOPE, now));
    assert.ok(!stale(`${RELEASE}pyodide.asm.wasm`, SCOPE, now));
  });

  test("another Plotly.js goes, and so does a file of PyPI's the build no longer installs", () => {
    assert.ok(stale("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.0.0/plotly.min.js", SCOPE, now));
    assert.ok(!stale(PLOTLY_JS, SCOPE, now));
    assert.ok(stale(`${PYPI}plotly-7.0.0-py3-none-any.whl`, SCOPE, now));
    assert.ok(!stale(PLOTLY, SCOPE, now));
  });

  test("a file is kept when nothing is known of what replaced its kind", () => {
    assert.ok(!stale(`${SCOPE}wheels/aaaaaaaaaaaaaaaa/bootstrap.zip`, SCOPE, { pyodide: "v314.0.7/pyc" }));
    assert.ok(!stale("https://cdn.jsdelivr.net/pyodide/v314.0.6/pyc/x.whl", SCOPE, { folder: "bbbbbbbbbbbbbbbb" }));
    assert.ok(!stale(`${PYPI}plotly-7.0.0-py3-none-any.whl`, SCOPE, loadedNow(runtime, { ...manifest, on_demand: null })));
  });
});

describe("what the footer says of the copy", () => {
  test("a page that opens without a network offers its charts and workbooks, and to remove the copy", () => {
    const words = copyWords("opens");
    assert.match(words.said, /^Factorial opens on this device without a network/);
    assert.equal(words.keep, "Keep its charts and workbooks too (about 12 MB)");
    assert.ok(words.remove);
  });

  test("but offers no more than it can keep for certain: a build must name the files", () => {
    assert.deepEqual([copyWords("opens", { exact: false }).keep, copyWords("opens", { exact: false }).remove], [null, true]);
  });

  test("the whole copy offers nothing more to keep", () => {
    const words = copyWords("whole");
    assert.match(words.said, /opens and works on this device without a network/);
    assert.deepEqual([words.keep, words.remove], [null, true]);
  });

  test("while a copy is being made, neither button is offered", () => {
    for (const state of ["copying", "keeping"]) assert.deepEqual([copyWords(state).keep, copyWords(state).remove], [null, false]);
  });

  test("a copy that could not be finished says why, and can be tried again or removed", () => {
    const words = copyWords("failed", { reason: "Request failed: numpy.whl" });
    assert.equal(words.said, "The copy of Factorial on this device is not whole: Request failed: numpy.whl");
    assert.deepEqual([words.keep, words.remove], ["Try again", true]);
  });

  test("a reader who removed the copy is offered one again", () => {
    assert.deepEqual(copyWords("none"), {
      said: "Factorial keeps no copy of itself on this device.",
      keep: "Keep a copy, to open it without a network",
      remove: false,
    });
  });
});

// ---------------------------------------------------------------- installOffline, visit after visit
/**
 * The page's own files a visit loads, as the browser's record of the visit lists them: one of them asked for and
 * refused, as an optional file is. SHELL is what the copy then holds of the app: what opens the page, and the rest.
 */
const LOADED = ["app.js", "style.css", "manifest.webmanifest", "book/absent.json", `wheels/${manifest.wheel}`, "api/feedback"].map((file) => SCOPE + file);
const SHELL = ["", "worker.js", "python-boot.js", "runtime.json", "wheels/manifest.json", "app.js", "style.css", "manifest.webmanifest"].map((file) => SCOPE + file);

/** Every address a visit may ask for, each with some text; runtime.json and the lock file with their own. */
const everything = () => ({
  ...Object.fromEntries([...START, ...LATER, ...SHELL].map((address) => [address, `the file at ${address}`])),
  [`${SCOPE}runtime.json`]: JSON.stringify(runtime),
  [`${RELEASE}pyodide-lock.json`]: JSON.stringify(lock),
});

/** The footer's box: its words, of which `seen` keeps each one said, and its two buttons. */
function footer() {
  const seen = [];
  const button = () => ({ textContent: "", hidden: true, onclick: null, focus() { globalThis.document.activeElement = this; } });
  const said = {
    get textContent() { return seen.at(-1) ?? ""; },
    set textContent(text) { seen.push(text); },
  };
  const [keep, remove] = [button(), button()];
  const box = {
    hidden: true,
    dataset: {},
    querySelector: (part) => ({ span: said, "#offline-keep": keep, "#offline-remove": remove })[part],
    contains: (element) => [keep, remove, box].includes(element),
    focus() { globalThis.document.activeElement = box; },
  };
  return { box, said, keep, remove, seen };
}

/** A reader's browser: its network, what it keeps between visits (copies, a note, a registered worker), and visits. */
class Browser {
  constructor() {
    this.network = new Network(everything(), SCOPE);
    this.caches = new CacheStorage(this.network);
    this.notes = new Map();
    this.worker = null; // the script registered, and how many requests the page had made by then
  }

  /** Open the page and run installOffline to its end; return the footer as the visit leaves it. */
  async visit({ build = manifest, started = Promise.resolve(), serviceWorkers = true } = {}) {
    const browser = this;
    const page = footer();
    const set = (name, value) => Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
    set("location", { href: `${SCOPE}?example=worked` });
    set("document", { activeElement: null });
    set("fetch", this.network.fetch);
    set("caches", this.caches);
    set("localStorage", { getItem: (key) => this.notes.get(key) ?? null, setItem: (key, value) => this.notes.set(key, value), removeItem: (key) => this.notes.delete(key) });
    set("navigator", {
      serviceWorker: serviceWorkers && {
        register: async (script) => void (browser.worker = { script, after: browser.network.asked.length }),
        getRegistration: async () => browser.worker && { unregister: async () => void (browser.worker = null) },
      },
    });
    // The browser notes how each request was answered: the file the server refused is known as such.
    const record = LOADED.map((name) => ({ name, responseStatus: name.endsWith("absent.json") ? 404 : 200 }));
    globalThis.performance.getEntriesByType = (type) => (type === "resource" ? record : []);
    await installOffline({ manifest: build, started, box: page.box });
    return page;
  }

  /** The addresses the page asked the network for since `from`, less the two files every visit reads. */
  downloaded(from = 0) {
    return this.network.asked.slice(from).filter((address) => ![`${SCOPE}runtime.json`, `${RELEASE}pyodide-lock.json`].includes(address));
  }
}

describe("a first visit", () => {
  let browser;
  let page;
  beforeEach(async () => {
    browser = new Browser();
    page = await browser.visit();
  });

  test("copies what Python started from and the page's own files, and says the page opens without a network", () => {
    assert.deepEqual(browser.caches.held(KEPT).sort(), [...START].sort());
    assert.deepEqual(browser.caches.held(APP).sort(), [...SHELL].sort(), "not the wheel's folder, a reader's report, or a file the server refused");
    assert.deepEqual(page.seen, ["Factorial opens on this device without a network: a copy of it is kept here."]);
    assert.deepEqual([page.box.hidden, page.keep.hidden, page.remove.hidden], [false, false, false]);
    assert.equal(page.keep.textContent, "Keep its charts and workbooks too (about 12 MB)");
  });

  test("registers the worker once the copy is made, so Python starts with no worker in its way", () => {
    assert.equal(browser.worker.script, "sw.js");
    assert.equal(browser.worker.after, browser.network.asked.length, "nothing was copied after it");
    assert.ok(START.every((address) => browser.network.asked.includes(address)));
  });

  test("downloads nothing of the charts and workbooks", () => {
    assert.ok(!LATER.some((address) => browser.network.asked.includes(address)));
  });

  test("is followed by visits that copy nothing again", async () => {
    const before = browser.network.asked.length;
    const again = await browser.visit();
    assert.deepEqual(browser.downloaded(before), []);
    assert.equal(again.said.textContent, page.said.textContent);
  });
});

describe("the whole copy", () => {
  let browser;
  let page;
  beforeEach(async () => {
    browser = new Browser();
    page = await browser.visit();
  });

  test("is made when the reader asks: each file of the charts and workbooks, and nothing twice", async () => {
    const before = browser.network.asked.length;
    await page.keep.onclick();
    assert.deepEqual(browser.downloaded(before).sort(), [...LATER].sort());
    assert.deepEqual(page.seen.slice(1), [
      "Keeping the charts and the workbooks, about 12 MB...",
      "Factorial opens and works on this device without a network: a copy of it is kept here, with its charts and workbooks.",
    ]);
    assert.deepEqual([page.keep.hidden, page.remove.hidden], [true, false]);
  });

  test("is what the footer says only while the copy holds every file of it: nothing is taken from a note", async () => {
    await page.keep.onclick();
    assert.match((await browser.visit()).said.textContent, /^Factorial opens and works/);
    await (await browser.caches.open(KEPT)).delete(PLOTLY);
    assert.match((await browser.visit()).said.textContent, /^Factorial opens on this device/, "a browser may drop what it kept");
  });

  test("is what the footer says once the files have all passed through the worker, unasked", async () => {
    const kept = await browser.caches.open(KEPT);
    for (const address of LATER) await kept.put(address, { body: "kept as it passed" });
    assert.match((await browser.visit()).said.textContent, /^Factorial opens and works/);
  });

  test("is not offered by a build that could not name its files", async () => {
    const unnamed = await new Browser().visit({ build: { ...manifest, on_demand: null } });
    assert.match(unnamed.said.textContent, /^Factorial opens on this device/);
    assert.deepEqual([unnamed.keep.hidden, unnamed.remove.hidden], [true, false]);
  });
});

describe("a copy the browser has lost part of", () => {
  test("is made whole again at the next visit, file by file", async () => {
    const browser = new Browser();
    await browser.visit();
    await (await browser.caches.open(KEPT)).delete(`${RELEASE}pyodide.asm.wasm`);
    const before = browser.network.asked.length;
    const page = await browser.visit();
    assert.deepEqual(browser.downloaded(before), [`${RELEASE}pyodide.asm.wasm`]);
    assert.match(page.said.textContent, /^Factorial opens on this device/);
  });

  test("is made again when the browser dropped it all, though the worker's empty copies are there", async () => {
    const browser = new Browser();
    await browser.visit();
    for (const name of [KEPT, APP]) (await browser.caches.delete(name), await browser.caches.open(name));
    await browser.visit();
    assert.deepEqual(browser.caches.held(KEPT).sort(), [...START].sort());
  });
});

describe("a copy that cannot be finished", () => {
  test("says why, registers no worker to serve it, and is whole once tried again with the file there", async () => {
    const browser = new Browser();
    browser.network.files.delete(`${RELEASE}numpy.whl`);
    const page = await browser.visit();
    assert.equal(page.said.textContent, `The copy of Factorial on this device is not whole: Request failed: ${RELEASE}numpy.whl`);
    assert.deepEqual([page.keep.textContent, page.keep.hidden, page.remove.hidden], ["Try again", false, false]);
    assert.equal(browser.worker, null);
    browser.network.files.set(`${RELEASE}numpy.whl`, "numpy");
    await page.keep.onclick();
    assert.deepEqual(page.seen.slice(1), ["Keeping a copy of Factorial...", "Factorial opens on this device without a network: a copy of it is kept here."]);
    assert.equal(browser.worker.script, "sw.js");
  });

  test("for want of a file the page opens with says so, where one it merely loaded is left out", async () => {
    const browser = new Browser();
    browser.network.files.delete(`${SCOPE}style.css`); // gone since the visit loaded it
    assert.match((await browser.visit()).said.textContent, /^Factorial opens on this device/);
    assert.ok(!browser.caches.held(APP).includes(`${SCOPE}style.css`));
    const without = new Browser();
    without.network.files.delete(`${SCOPE}worker.js`);
    assert.equal((await without.visit()).said.textContent, `The copy of Factorial on this device is not whole: Request failed: ${SCOPE}worker.js`);
  });

  test("of the charts and workbooks keeps the page's copy, and Try again tries the whole of it", async () => {
    const browser = new Browser();
    const page = await browser.visit();
    browser.network.files.delete(PLOTLY);
    await page.keep.onclick();
    assert.equal(page.said.textContent, `The copy of Factorial on this device is not whole: Request failed: ${PLOTLY}`);
    browser.network.files.set(PLOTLY, "plotly");
    await page.keep.onclick();
    assert.match(page.said.textContent, /^Factorial opens and works/);
  });

  test("because Python did not start says so, and offers to remove a copy that may be the cause", async () => {
    const browser = new Browser();
    await browser.visit();
    const page = await browser.visit({ started: Promise.reject(new Error("Could not start Python: the wasm is cut short")) });
    assert.equal(page.said.textContent, "The copy of Factorial on this device is not whole: Could not start Python: the wasm is cut short");
    assert.equal(page.remove.hidden, false);
  });
});

describe("a reader who removes the copy", () => {
  let browser;
  let page;
  beforeEach(async () => {
    browser = new Browser();
    page = await browser.visit();
    await page.remove.onclick();
  });

  test("has the copies deleted and the worker unregistered, and is told so", async () => {
    assert.deepEqual(await browser.caches.keys(), []);
    assert.equal(browser.worker, null);
    assert.equal(page.said.textContent, "Factorial keeps no copy of itself on this device.");
    assert.deepEqual([page.keep.textContent, page.remove.hidden], ["Keep a copy, to open it without a network", true]);
  });

  test("has none made at later visits, until they ask for one", async () => {
    const before = browser.network.asked.length;
    const later = await browser.visit();
    assert.deepEqual(browser.network.asked.slice(before), [], "nothing is read, and nothing copied");
    assert.equal(later.said.textContent, "Factorial keeps no copy of itself on this device.");
    await later.keep.onclick();
    assert.match(later.said.textContent, /^Factorial opens on this device/);
    assert.match((await browser.visit()).said.textContent, /^Factorial opens on this device/, "and it is kept from then on");
  });
});

describe("the keyboard", () => {
  test("stays in the footer when the button pressed goes: on the next button, or on the words", async () => {
    const browser = new Browser();
    const page = await browser.visit();
    globalThis.document.activeElement = page.keep;
    await page.keep.onclick();
    assert.equal(globalThis.document.activeElement, page.remove, "the whole copy offers only its removal");
    await page.remove.onclick();
    assert.equal(globalThis.document.activeElement, page.keep, "and no copy offers only to keep one");
  });

  test("is left where it is when the reader is elsewhere on the page", async () => {
    const page = await new Browser().visit();
    assert.equal(globalThis.document.activeElement, null);
    assert.equal(page.box.dataset.state, "opens");
  });
});

describe("a later build", () => {
  const next = { ...manifest, wheel: "fedcba9876543210/process_improve-2.6.0-py3-none-any.whl", glue: "fedcba9876543210/bootstrap.zip" };

  test("has its files added to the copy, and the build before's dropped once they are all there", async () => {
    const browser = new Browser();
    await browser.visit();
    for (const file of [next.wheel, next.glue]) browser.network.files.set(`${SCOPE}wheels/${file}`, "the next build's");
    const stray = "https://cdn.jsdelivr.net/pyodide/v314.0.6/pyc/pyodide.asm.wasm";
    await (await browser.caches.open(KEPT)).put(stray, { body: "of the release before" });
    await browser.visit({ build: next });
    const held = browser.caches.held(KEPT);
    assert.ok([next.wheel, next.glue].every((file) => held.includes(`${SCOPE}wheels/${file}`)));
    assert.ok(![manifest.wheel, manifest.glue].some((file) => held.includes(`${SCOPE}wheels/${file}`)));
    assert.ok(!held.includes(stray));
  });

  test("that cannot be copied whole leaves the build before's copy as it was", async () => {
    const browser = new Browser();
    await browser.visit();
    const page = await browser.visit({ build: next }); // its wheel is not on the network
    assert.match(page.said.textContent, /is not whole: Request failed/);
    assert.ok(START.every((address) => browser.caches.held(KEPT).includes(address)), "what opened the page before still does");
  });

  test("made to retire the worker deletes the copies and unregisters, and leaves the reader's wish as it was", async () => {
    const browser = new Browser();
    await (await browser.visit()).remove.onclick();
    await (await browser.visit()).keep.onclick();
    assert.equal(browser.worker.script, "sw.js");
    const page = await browser.visit({ build: { ...manifest, service_worker: false } });
    assert.deepEqual([await browser.caches.keys(), browser.worker, page.box.hidden], [[], null, true]);
    const off = new Browser();
    await (await off.visit()).remove.onclick();
    await off.visit({ build: { ...manifest, service_worker: false } });
    assert.equal((await off.visit()).said.textContent, "Factorial keeps no copy of itself on this device.", "a reader who wanted none still has none");
  });
});

describe("a page that keeps no copy", () => {
  test("is a plain preview of web/, which has no build to copy", async () => {
    const browser = new Browser();
    const page = await browser.visit({ build: {} });
    assert.deepEqual([page.box.hidden, browser.worker, browser.network.asked], [true, null, []]);
  });

  test("is one in a browser without service workers", async () => {
    const browser = new Browser();
    const page = await browser.visit({ serviceWorkers: false });
    assert.deepEqual([page.box.hidden, browser.network.asked], [true, []]);
  });
});
