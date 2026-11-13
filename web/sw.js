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
// - Any other request, and any that is not a GET, is left alone: the feedback form's among them.
//
// What the copy holds is the page's business (offline.js): it makes the copy, drops what earlier builds left in
// it, and removes it. This worker only adds to a copy that is there, so one removed stays removed. And it serves a
// page from the page's next visit on: it never takes over a page that is open, so the visit that registers it
// loads, and makes its copy, with no worker in its way.
//
// The way out, should this worker ever serve a page wrongly, is a build made with --no-service-worker
// (scripts/build_web_app.py): it puts sw-retire.js at this address, which the browser fetches at the reader's next
// visit whatever the page's own code does, and which deletes the copies and unregisters.
//
// This is a classic script, as every browser's service worker takes one; tests/web/js reads its rules, and runs
// its handlers against a stand-in for the browser, under Node.js.

const KEPT = "factorial-kept-1"; // files whose address names their version
const APP = "factorial-app-1"; // the app's own files and the package indexes: the network's copy when there is one

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

if (typeof module === "object") module.exports = { kind, address, KEPT, APP }; // Node.js, for the tests

if (typeof ServiceWorkerGlobalScope !== "undefined") {
  const scope = self.registration.scope;
  const whole = (response) => response.status === 200 && response.type !== "opaque";

  /**
   * Keep a response under its address, in a copy that is there. The copies are made by the page (offline.js) and
   * when this worker takes over (activate); once a reader removes them, a page this worker still serves must not
   * bring them back. A full disk, or a browser that refuses, loses the copy and not the page.
   */
  const keep = (name, href, response) =>
    caches
      .has(name)
      .then((there) => there && caches.open(name).then((cache) => cache.put(href, response)))
      .catch(() => {});

  /** The copy of a request, whatever headers it was asked with. Looked up by the copy's name: looking makes none. */
  const held = (name, request) => caches.match(address(request), { cacheName: name, ignoreVary: true });

  /** A file whose address names its version: the copy, or the network's once, kept from then on. */
  async function kept(request, event) {
    const copy = await held(KEPT, request);
    if (copy) return copy;
    const response = await fetch(request);
    if (whole(response)) event.waitUntil(keep(KEPT, request.url, response.clone()));
    return response;
  }

  /** One of the app's own files: the network's whenever it answers, kept for when it does not. */
  async function fresh(request, event) {
    let response;
    try {
      response = await fetch(request);
    } catch (offline) {
      const copy = await held(APP, request);
      if (copy) return copy;
      throw offline;
    }
    if (whole(response)) event.waitUntil(keep(APP, address(request), response.clone()));
    return response;
  }

  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => {
    event.waitUntil(
      (async () => {
        // A later version of this worker names its copies anew: the ones before it go.
        for (const name of await caches.keys()) {
          if (name.startsWith("factorial-") && name !== KEPT && name !== APP) await caches.delete(name);
        }
        await Promise.all([caches.open(KEPT), caches.open(APP)]); // the copies, which keep() adds to while they last
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
