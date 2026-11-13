// The service worker of Factorial: it keeps a copy of what the page loads, so the page opens without a network,
// and Python starts from the copy when Stop starts it again.
//
// It holds no list of files and no version of the app. It has one rule for each kind of address, so nothing in it
// can fall behind a deploy:
//
// - A file whose address names its version is kept as it is, and served from the copy: Pyodide's release, the
//   app's wheel and its own Python in their folder named by digest, Plotly.js, and PyPI's files.
// - The app's own files, and PyPI's index pages, come from the network whenever it answers, and the copy is the
//   last one it gave: a deploy reaches a reader at their next visit with a network, as it does without this worker.
// - Any other request, and any that is not a GET, is left alone: the feedback form's among them.
//
// wheels/manifest.json is the switch. A build made with "service_worker": false there retires this worker and
// its copies at the reader's next visit (scripts/build_web_app.py --no-service-worker), with no change here. The
// same file, and runtime.json, say which wheel, Pyodide release and Plotly.js the page now loads: copies of
// earlier ones are dropped as these two pass by.
//
// This is a classic script, as every browser's service worker takes one; tests/web/js reads its two rules under
// Node.js through the export at the foot.

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
 * Whether a kept file is of an earlier build or release than the page now loads. `now` holds what is known of
 * that, each part or none: `folder`, the wheels' folder named by their digest; `pyodide`, the release and the
 * folder of it, as "v314.0.7/pyc"; `plotlyJs`, the Plotly.js version. A file `now` says nothing of is kept.
 */
function stale(href, scope, { folder, pyodide, plotlyJs }) {
  if (folder && href.startsWith(`${scope}wheels/`)) return !href.startsWith(`${scope}wheels/${folder}/`);
  if (pyodide && href.startsWith(PYODIDE)) return !href.startsWith(`${PYODIDE}${pyodide}/`);
  if (plotlyJs && href.startsWith(PLOTLY_JS)) return !href.startsWith(`${PLOTLY_JS}${plotlyJs}/`);
  return false;
}

if (typeof module === "object") module.exports = { kind, stale, KEPT, APP }; // Node.js, for the tests

if (typeof ServiceWorkerGlobalScope !== "undefined") {
  const scope = self.registration.scope;
  const whole = (response) => response.status === 200 && response.type !== "opaque";
  /** The two files that say what the page now loads, and whether this worker is wanted. */
  const [MANIFEST, RUNTIME] = [`${scope}wheels/manifest.json`, `${scope}runtime.json`];
  const SAYS = [MANIFEST, RUNTIME];

  /**
   * Keep a response under its address, in a copy that is there. The copies are made when this worker takes over
   * (activate) and by the page (offline.js); once they are deleted, by a build that retires the worker or by a
   * reader who removes the copy, a page this worker still serves must not bring them back. A full disk, or a
   * browser that refuses, loses the copy and not the page.
   */
  const keep = (name, href, response) =>
    caches
      .has(name)
      .then((there) => there && caches.open(name).then((cache) => cache.put(href, response)))
      .catch(() => {});

  /** The copy of an address, whatever headers it was asked with; a page's, with or without its query. */
  async function held(name, request) {
    const within = { cacheName: name, ignoreVary: true }; // looked up by name, so that looking makes no copy
    const copy = await caches.match(request.url, within);
    if (copy || request.mode !== "navigate") return copy;
    return caches.match(request.url.split("?")[0], within); // the page reads its own query
  }

  /** A file whose address names its version: the copy, or the network's once, kept from then on. */
  async function kept(request) {
    const copy = await held(KEPT, request);
    if (copy) return copy;
    const response = await fetch(request);
    if (whole(response)) keep(KEPT, request.url, response.clone());
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
    if (whole(response)) {
      event.waitUntil(keep(APP, request.url, response.clone()));
      if (SAYS.includes(request.url)) event.waitUntil(follow(request.url, response.clone()).catch(() => {}));
    }
    return response;
  }

  /** Retire this worker and every copy it kept: the build says it is not wanted. */
  async function retire() {
    for (const name of await caches.keys()) if (name.startsWith("factorial-")) await caches.delete(name);
    await self.registration.unregister();
  }

  /** Drop the kept files of earlier builds and releases, which `now` names the successors of. */
  async function prune(now) {
    if (!(await caches.has(KEPT))) return;
    const cache = await caches.open(KEPT);
    for (const request of await cache.keys()) if (stale(request.url, scope, now)) await cache.delete(request);
  }

  /** Read the two files that say what the page now loads, as the network gives them, and act on what they say. */
  async function follow(href, response) {
    const said = await response.json();
    if (href === RUNTIME) return prune({ pyodide: `v${said.pyodide}/${said.distribution}` });
    if (said.service_worker === false) return retire();
    return prune({ folder: said.wheel?.split("/")[0], plotlyJs: said.plotly_js });
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
        await self.clients.claim(); // the page that registered this worker is served by it from here on
      })(),
    );
  });
  self.addEventListener("fetch", (event) => {
    const { request } = event;
    if (request.method !== "GET") return;
    const what = kind(request.url, scope);
    if (what === "kept") event.respondWith(kept(request));
    else if (what === "fresh") event.respondWith(fresh(request, event));
  });
}
