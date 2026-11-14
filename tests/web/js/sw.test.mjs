// web/sw.js and web/sw-retire.js: the rule for each kind of address, and the worker's handlers run against a
// stand-in for the browser (standins.mjs). That a real browser opens the page from the copy is the live tier's test.

import assert from "node:assert/strict";
import path from "node:path";
import { describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import worker from "../../../web/sw.js";
import { CacheStorage, Network, request, response, serviceWorker } from "./standins.mjs";

const web = (file) => path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../web", file);

const SCOPE = "https://factori.al/";
const RELEASE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/pyc/";
const WASM = `${RELEASE}pyodide.asm.wasm`;
const APP_JS = `${SCOPE}app.js`;
const { KEPT, APP, MADE } = worker;

describe("how the service worker serves an address", () => {
  test("a file whose address names its version is kept, and served from the copy", () => {
    for (const address of [
      WASM,
      `${RELEASE}numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl`,
      "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js",
      "https://files.pythonhosted.org/packages/ab/cd/plotly-7.1.0-py3-none-any.whl",
      `${SCOPE}wheels/0123456789abcdef/process_improve-2.5.0-py3-none-any.whl`,
      `${SCOPE}wheels/0123456789abcdef/bootstrap.zip`,
    ]) {
      assert.equal(worker.kind(address, SCOPE), "kept", address);
    }
  });

  test("the app's own files and the package indexes come from the network when it answers", () => {
    for (const address of [
      SCOPE,
      `${SCOPE}?example=worked`,
      APP_JS,
      `${SCOPE}runtime.json`,
      `${SCOPE}wheels/manifest.json`,
      `${SCOPE}catalog/index.json`,
      "https://pypi.org/simple/plotly/",
    ]) {
      assert.equal(worker.kind(address, SCOPE), "fresh", address);
    }
  });

  test("a reader's report, and any other site, are left alone", () => {
    assert.equal(worker.kind(`${SCOPE}api/feedback`, SCOPE), null);
    assert.equal(worker.kind("https://example.org/app.js", SCOPE), null);
    assert.equal(worker.kind("https://factori.al.example.org/app.js", SCOPE), null);
  });

  test("an app served under a folder keeps to its folder", () => {
    const pages = "https://kgdunn.github.io/process-improve/app/";
    assert.equal(worker.kind(`${pages}app.js`, pages), "fresh");
    assert.equal(worker.kind(`${pages}wheels/0123456789abcdef/bootstrap.zip`, pages), "kept");
    assert.equal(worker.kind("https://kgdunn.github.io/process-improve/index.html", pages), null);
  });
});

describe("the address a response is kept under", () => {
  test("a page is kept once, whatever query or place in it a link names", () => {
    for (const url of [SCOPE, `${SCOPE}?example=worked`, `${SCOPE}?example=worked#learn`, `${SCOPE}#plan`]) {
      assert.equal(worker.address({ mode: "navigate", url }), SCOPE, url);
    }
    assert.equal(worker.address({ mode: "navigate", url: `${SCOPE}about.html?from=app` }), `${SCOPE}about.html`);
  });

  test("any other file is kept under the address it was asked by", () => {
    const url = `${SCOPE}catalog/index.json?v=2`;
    assert.equal(worker.address({ mode: "cors", url }), url);
  });
});

/** The worker, taken over a browser whose network gives these files and whose page has made a copy, mark first. */
async function running(files = { [WASM]: "wasm", [APP_JS]: "app, as deployed", [SCOPE]: "the page" }) {
  const network = new Network(files, SCOPE);
  const caches = new CacheStorage(network);
  for (const name of [MADE, KEPT, APP]) await caches.open(name);
  const sw = serviceWorker(web("sw.js"), { caches, network, scope: SCOPE });
  await sw.fire("activate");
  const fetched = (url, how) => sw.fire("fetch", { request: request(url, how) });
  return { ...sw, network, caches, fetched };
}

describe("the worker, taking over", () => {
  test("waits for no page of an earlier version to close, and takes over no page that is open", async () => {
    const { self, fire } = await running();
    await fire("install");
    assert.ok(self.skipped, "a later version of the worker replaces the one before at once");
    assert.equal(self.claimed, undefined, "the visit that registers it loads, and copies, with no worker in its way");
  });

  test("makes no copy itself, and deletes the copies an earlier version of it named", async () => {
    const network = new Network({}, SCOPE);
    const caches = new CacheStorage(network);
    for (const name of ["factorial-kept-0", "factorial-app-0", MADE, KEPT, "another-app"]) await (await caches.open(name)).put(`${SCOPE}x`, response("x"));
    await serviceWorker(web("sw.js"), { caches, network, scope: SCOPE }).fire("activate");
    assert.deepEqual((await caches.keys()).sort(), [MADE, KEPT, "another-app"].sort(), "and no copy the page has not made");
    assert.deepEqual(caches.held(KEPT), [`${SCOPE}x`], "a copy of its own is kept as it is");
  });
});

describe("a file whose address names its version", () => {
  test("comes from the network once, and from the copy from then on, with or without a network", async () => {
    const { fetched, network, caches } = await running();
    assert.equal((await fetched(WASM)).got.body, "wasm");
    assert.deepEqual(caches.held(KEPT), [WASM]);
    network.down = true;
    assert.equal((await fetched(WASM)).got.body, "wasm");
    assert.equal(network.asked.length, 1, "the copy is not asked of the network again");
  });

  test("that the network refuses is passed on, and not kept", async () => {
    const { fetched, caches } = await running();
    assert.equal((await fetched(`${RELEASE}gone.whl`)).got.status, 404);
    assert.deepEqual(caches.held(KEPT), []);
  });

  test("with no network and no copy fails as it would without the worker", async () => {
    const { fetched, network } = await running();
    network.down = true;
    assert.match((await fetched(WASM)).error.message, /Failed to fetch/);
  });

  test("is not kept once the copy is removed: a page still served must not bring it back", async () => {
    const { fetched, caches } = await running();
    for (const name of [MADE, KEPT, APP]) await caches.delete(name); // as offline.js removes a copy: its mark first
    assert.equal((await fetched(WASM)).got.body, "wasm");
    assert.deepEqual(await caches.keys(), []);
  });

  test("is not kept in a cache the page has not marked as its copy", async () => {
    const { fetched, caches } = await running();
    await caches.delete(MADE);
    await fetched(WASM);
    assert.deepEqual(caches.held(KEPT), []);
  });

  test("does not bring the copy back when it is removed just as the worker goes to write to it", async () => {
    const { fetched, caches } = await running();
    const open = caches.open;
    caches.open = async (name) => {
      for (const gone of [MADE, KEPT, APP]) await caches.delete(gone); // the reader's removal, between the look and the write
      return open(name); // which makes the cache anew
    };
    assert.equal((await fetched(WASM)).got.body, "wasm");
    assert.deepEqual(await caches.keys(), [], "the cache the write brought back is deleted again");
  });

  test("does not bring the copy back when the write that made it anew then fails", async () => {
    const { fetched, caches } = await running();
    const open = caches.open;
    caches.open = async (name) => {
      for (const gone of [MADE, KEPT, APP]) await caches.delete(gone);
      const cache = await open(name);
      cache.put = async () => Promise.reject(new Error("QuotaExceededError")); // a full disk, say
      return cache;
    };
    assert.equal((await fetched(WASM)).got.body, "wasm", "the page has its file all the same");
    assert.deepEqual(await caches.keys(), []);
  });

  test("is looked up in no copy once the copy is removed: in Chromium a look by its name would make it anew", async () => {
    const { fetched, caches } = await running();
    for (const name of [MADE, KEPT, APP]) await caches.delete(name);
    let looked = 0;
    const match = caches.match;
    caches.match = (...asked) => ((looked += 1), match(...asked));
    assert.equal((await fetched(WASM)).got.body, "wasm");
    assert.equal(looked, 0);
    assert.deepEqual(await caches.keys(), []);
  });

  test("is looked for in every cache at once, and in none by its name, so a look makes no cache", async () => {
    const { fetched, caches } = await running();
    await fetched(WASM); // kept
    const asked = [];
    const match = caches.match;
    caches.match = (address, how) => (asked.push(how), match(address, how));
    assert.equal((await fetched(WASM)).got.body, "wasm", "from the copy");
    await fetched(APP_JS);
    assert.ok(asked.length >= 1);
    assert.ok(asked.every((how) => how.cacheName === undefined && how.ignoreVary === true));
  });

  test("brings no cache back by a look, so none is left if the worker is stopped before it can delete it", async () => {
    const { fetched, caches } = await running();
    await fetched(WASM); // kept
    const match = caches.match;
    let deletes = 0;
    caches.match = async (...asked) => {
      for (const gone of [MADE, KEPT, APP]) await caches.delete(gone); // the reader's removal, after the mark was seen
      const remove = caches.delete;
      caches.delete = (name) => ((deletes += 1), remove(name)); // any the worker makes from here on
      return match(...asked);
    };
    assert.equal((await fetched(WASM)).got.body, "wasm", "from the network, there being no copy");
    assert.deepEqual([await caches.keys(), deletes], [[], 0], "there was nothing for the worker to delete");
  });

  test("does not bring the copy back when it is removed just as the worker goes to look in it", async () => {
    const { fetched, caches } = await running();
    await fetched(WASM); // kept
    const match = caches.match;
    caches.match = async (...asked) => {
      for (const gone of [MADE, KEPT, APP]) await caches.delete(gone); // the reader's removal, after the mark was seen
      return match(...asked); // which makes the cache anew, and finds nothing in it
    };
    assert.equal((await fetched(WASM)).got.body, "wasm", "from the network, there being no copy");
    assert.deepEqual(await caches.keys(), [], "the cache the look brought back is deleted again");
  });
});

describe("one of the app's own files", () => {
  test("comes from the network whenever it answers, so a deploy is seen at once", async () => {
    const { fetched, network, caches } = await running();
    await fetched(APP_JS);
    network.files.set(APP_JS, "app, deployed again");
    assert.equal((await fetched(APP_JS)).got.body, "app, deployed again");
    assert.equal((await caches.match(APP_JS, { cacheName: APP })).body, "app, deployed again", "and the copy follows");
  });

  test("comes from the copy when the network does not answer", async () => {
    const { fetched, network } = await running();
    await fetched(APP_JS);
    network.down = true;
    assert.equal((await fetched(APP_JS)).got.body, "app, as deployed");
  });

  test("with no network and no copy fails as it would without the worker", async () => {
    const { fetched, network } = await running();
    network.down = true;
    assert.match((await fetched(APP_JS)).error.message, /Failed to fetch/);
  });

  test("with no network, once the copy is removed, fails the same way and brings no copy back", async () => {
    const { fetched, network, caches } = await running();
    await fetched(APP_JS); // kept
    for (const name of [MADE, KEPT, APP]) await caches.delete(name);
    network.down = true;
    assert.match((await fetched(APP_JS)).error.message, /Failed to fetch/);
    assert.deepEqual(await caches.keys(), []);
  });

  test("that is the page is kept once, and opens from the copy by any link to it", async () => {
    const { fetched, network, caches } = await running();
    await fetched(`${SCOPE}?example=worked`, { mode: "navigate" });
    assert.deepEqual(caches.held(APP), [SCOPE]);
    network.down = true;
    assert.equal((await fetched(`${SCOPE}?book=reactor#learn`, { mode: "navigate" })).got.body, "the page");
  });
});

describe("a page asked for with no network that the copy does not hold", () => {
  const CATALOG = `${SCOPE}catalog/?q=dsd`;

  test("says that it needs a network, with the way back to the app", async () => {
    const { fetched, network } = await running();
    await fetched(SCOPE, { mode: "navigate" }); // the app's own page, kept
    network.down = true;
    const { got } = await fetched(CATALOG, { mode: "navigate" });
    assert.equal(got.status, 503);
    assert.match(got.headers.get("Content-Type"), /^text\/html/);
    const said = await got.text();
    assert.match(said, /<h1>This page needs a network<\/h1>/);
    assert.ok(said.includes(`<a href="${SCOPE}">Back to Factorial</a>`));
  });

  test("is served as any page is, once a visit with a network has opened it", async () => {
    const { fetched, network } = await running({ [SCOPE]: "the page", [`${SCOPE}catalog/`]: "the catalog" });
    await fetched(SCOPE, { mode: "navigate" });
    await fetched(CATALOG, { mode: "navigate" });
    network.down = true;
    assert.equal((await fetched(`${SCOPE}catalog/`, { mode: "navigate" })).got.body, "the catalog");
  });

  test("is a page only: a file the copy lacks fails as it would without the worker", async () => {
    const { fetched, network } = await running();
    await fetched(SCOPE, { mode: "navigate" });
    network.down = true;
    assert.match((await fetched(`${SCOPE}catalog/index.json`)).error.message, /Failed to fetch/);
  });

  test("fails as it would without the worker when the copy does not hold the app's page to go back to", async () => {
    const { fetched, network } = await running();
    network.down = true;
    assert.match((await fetched(CATALOG, { mode: "navigate" })).error.message, /Failed to fetch/);
    assert.match((await fetched(SCOPE, { mode: "navigate" })).error.message, /Failed to fetch/, "nor the app's own page");
  });

  test("fails the same way once the copy is removed, and brings no copy back", async () => {
    const { fetched, network, caches } = await running();
    await fetched(SCOPE, { mode: "navigate" });
    for (const name of [MADE, KEPT, APP]) await caches.delete(name);
    network.down = true;
    assert.match((await fetched(CATALOG, { mode: "navigate" })).error.message, /Failed to fetch/);
    assert.deepEqual(await caches.keys(), []);
  });
});

describe("a request the worker leaves alone", () => {
  test("is not answered by it: a report sent, another site's file, and the feedback relay", async () => {
    const { fetched, network } = await running();
    for (const [url, how] of [
      [APP_JS, { method: "POST" }],
      ["https://example.org/app.js", {}],
      [`${SCOPE}api/feedback`, {}],
    ]) {
      assert.equal(await fetched(url, how), undefined, url);
    }
    assert.deepEqual(network.asked, []);
  });
});

describe("the worker a build puts in its place to retire it", () => {
  test("takes over at once, deletes every copy the worker kept, and unregisters", async () => {
    const network = new Network({}, SCOPE);
    const caches = new CacheStorage(network);
    for (const name of [KEPT, APP, MADE, "another-app"]) await caches.open(name);
    const deleted = [];
    const remove = caches.delete;
    caches.delete = (name) => (deleted.push(name), remove(name));
    const retiring = serviceWorker(web("sw-retire.js"), { caches, network, scope: SCOPE });
    await retiring.fire("install");
    await retiring.fire("activate");
    assert.ok(retiring.self.skipped && retiring.self.unregistered);
    assert.deepEqual(await caches.keys(), ["another-app"]);
    assert.equal(deleted[0], MADE, "the mark first: the worker being retired then adds nothing more");
    assert.ok(!retiring.listens("fetch"), "it serves nothing: the page loads from the network");
  });
});
