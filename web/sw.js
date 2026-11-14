// The service worker of Factorial: it serves the copy of the page kept on the reader's device, so the page opens
// without a network, and adds to the copy what passes through it.
//
// It holds no list of files and no version of the app, and it reads none: it has one rule for each kind of
// address, so nothing in it can fall behind a deploy.
//
// - A file whose address names its version is kept as it is, and served from the copy: Pyodide's release, the
//   app's wheel and its own Python in their folder named by digest, Plotly.js, and PyPI's files.
// - The app's own files, and PyPI's index pages, come from the network whenever it answers, and the copy is the
//   last one it gave: a deploy reaches a reader at their next visit with a network, as it does without this worker.
//   A page asked for with no network that the copy does not hold is answered with a few words and the way back.
// - Any other request, and any that is not a GET, is left alone: the feedback form's among them.
//
// What the copy holds is the page's business (offline.js): it makes the copy, drops what earlier builds left in
// it, and removes it. This worker makes no copy: it reads and adds to one the page has made and marked as made
// (MADE), so a copy removed stays removed, whatever this worker was in the middle of. And it serves a page from
// the page's next visit on: it never takes over a page that is open, so the visit that registers it loads, and
// makes its copy, with no worker in its way.
//
// The way out, should this worker ever serve a page wrongly, is a build made with --no-service-worker
// (scripts/build_web_app.py): it puts sw-retire.js at this address, which the browser fetches at the reader's next
// visit whatever the page's own code does, and which deletes the copies and unregisters.
//
// This is a classic script, as every browser's service worker takes one; tests/web/js reads its rules, and runs
// its handlers against a stand-in for the browser, under Node.js.

const KEPT = "factorial-kept-1"; // files whose address names their version
const APP = "factorial-app-1"; // the app's own files and the package indexes: the network's copy when there is one
const MADE = "factorial-made-1"; // empty: there while the page has a copy, made first and removed first

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/";
const PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@";

/**
 * What a request is to this worker: "kept" for a file whose address names its version, "fresh" for one the
 * network's copy is wanted of, and null for one it leaves alone. `scope` is the app's folder, as an address.
 */
function kind(href, scope) {
  if (href.startsWith(PYODIDE) || href.startsWith(PLOTLY_JS)) return "kept";
  const { hostname } = new URL(href);
  if (hostname === "files.pythonhosted.org") return "kept";
  if (hostname === "pypi.org") return "fresh";
  if (!href.startsWith(scope)) return null;
  const path = href.slice(scope.length);
  if (/^wheels\/[0-9a-f]{16}\//.test(path)) return "kept";
  if (path.startsWith("api/")) return null; // the feedback relay: a reader's report is never kept
  return "fresh";
}

/**
 * The address a response is kept under. A page's is without its query, which the page itself reads: one copy
 * of the page then serves every link to it, and a link's query adds no copy.
 */
function address(request) {
  if (request.mode !== "navigate") return request.url;
  const page = new URL(request.url);
  page.search = page.hash = "";
  return page.href;
}

if (typeof module === "object") module.exports = { kind, address, KEPT, APP, MADE }; // Node.js, for the tests

if (typeof ServiceWorkerGlobalScope !== "undefined") {
  const scope = self.registration.scope;
  const whole = (response) => response.status === 200 && response.type !== "opaque";

  /**
   * Read or write one of the copies, only while the page's mark is there; resolve to what `work` gave, or to
   * nothing without the mark. A reader may remove the copy while this worker still serves their page, and a
   * browser makes a cache anew when it is opened and, in Chromium, when it is only looked up by its name. So the
   * mark is looked for before the work and after it, however the work ended: without it nothing is done, and a
   * cache the work brought back, because the copy went in between, is deleted again.
   */
  async function inCopy(name, work) {
    if (!(await caches.has(MADE))) return undefined;
    try {
      return await work();
    } finally {
      if (!(await caches.has(MADE))) await caches.delete(name);
    }
  }

  /** Keep a response under its address. A full disk, or a browser that refuses, loses the copy and not the page. */
  const keep = (name, href, response) => inCopy(name, async () => (await caches.open(name)).put(href, response)).catch(() => {});

  /** The copy of a request, whatever headers it was asked with; none if the copy cannot be read. */
  const held = (name, request) =>
    inCopy(name, () => caches.match(address(request), { cacheName: name, ignoreVary: true })).catch(() => undefined);

  /** A file whose address names its version: the copy, or the network's once, kept from then on. */
  async function kept(request, event) {
    const copy = await held(KEPT, request);
    if (copy) return copy;
    const response = await fetch(request);
    if (whole(response)) event.waitUntil(keep(KEPT, request.url, response.clone()));
    return response;
  }

  /**
   * What a reader is told of a page asked for with no network that the copy does not hold: the design catalog or
   * the concept map, say, which are kept once a visit with a network has opened them. It names the way back to
   * the app, where the browser's own page for a site it cannot reach names none.
   */
  const needsNetwork = () =>
    new Response(
      '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>This page needs a network</title>' +
        "<style>:root{color-scheme:light dark}body{font:1rem/1.5 system-ui,sans-serif;max-width:34rem;margin:15vh auto;" +
        "padding:0 1rem}</style></head><body><h1>This page needs a network</h1>" +
        "<p>Factorial opens on this device without a network, from a copy of it kept here. This page is not in the " +
        "copy yet: it is kept once it has been opened with a network.</p>" +
        `<p><a href="${scope}">Back to Factorial</a></p></body></html>`,
      { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } },
    );

  /** One of the app's own files: the network's whenever it answers, kept for when it does not. */
  async function fresh(request, event) {
    let response;
    try {
      response = await fetch(request);
    } catch (offline) {
      const copy = await held(APP, request);
      if (copy) return copy;
      // A page the copy lacks says so, while the copy holds the app's own page for the reader to go back to.
      const page = request.mode === "navigate" && address(request) !== scope;
      if (page && (await held(APP, { mode: "navigate", url: scope }))) return needsNetwork();
      throw offline;
    }
    if (whole(response)) event.waitUntil(keep(APP, address(request), response.clone()));
    return response;
  }

  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => {
    event.waitUntil(
      (async () => {
        // A later version of this worker names its copies anew: the ones before it go. It makes none itself.
        for (const name of await caches.keys()) {
          if (name.startsWith("factorial-") && ![KEPT, APP, MADE].includes(name)) await caches.delete(name);
        }
      })(),
    );
  });
  self.addEventListener("fetch", (event) => {
    const { request } = event;
    if (request.method !== "GET") return;
    const what = kind(request.url, scope);
    if (what === "kept") event.respondWith(kept(request, event));
    else if (what === "fresh") event.respondWith(fresh(request, event));
  });
}
