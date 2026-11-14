// Stand-ins for the parts of a browser that the page's copy for use without a network touches: a network that
// answers from a table of addresses, Cache Storage kept in memory, and the service worker's own scope. They do
// what the code under test relies on and no more; that a real browser does the same is the live tier's test.

import fs from "node:fs";
import vm from "node:vm";

/** A response, as far as the page and the worker read one. */
export function response(body, { status = 200, type = "basic" } = {}) {
  return {
    body,
    status,
    type,
    ok: status === 200,
    clone: () => response(body, { status, type }),
    json: async () => JSON.parse(body),
  };
}

/** An address without the place in the page it names: Cache Storage matches without it. */
const plain = (address) => String(address.url ?? address).split("#")[0];

/**
 * A network that answers each address in `files` with its text, whatever query it is asked with, as a server of
 * files does; refuses any other with a 404; and fails every request while `down`, as a browser with no network
 * does. `asked` lists what was asked for, in order.
 */
export class Network {
  constructor(files, base) {
    Object.assign(this, { files: new Map(Object.entries(files)), base, asked: [], down: false });
  }

  fetch = async (request) => {
    const address = new URL(plain(request), this.base).href;
    this.asked.push(address);
    if (this.down) throw new TypeError("Failed to fetch");
    const body = this.files.get(address) ?? this.files.get(address.split("?")[0]);
    return body === undefined ? response("", { status: 404 }) : response(body);
  };
}

/** One cache: responses by address. Adding an address fetches it, and refuses one the network does not give whole. */
class Cache {
  constructor(entries, network) {
    Object.assign(this, { entries, network });
  }

  match = async (address) => this.entries.get(plain(address));
  put = async (address, kept) => void this.entries.set(plain(address), kept);
  keys = async () => [...this.entries.keys()].map((url) => ({ url }));
  delete = async (address) => this.entries.delete(plain(address));
  add = async (address) => {
    const got = await this.network.fetch(address);
    if (!got.ok) throw new TypeError(`Request failed: ${address}`);
    this.entries.set(plain(address), got);
  };
}

/** Cache Storage: caches by name, made when first opened. `held(name)` lists the addresses one holds. */
export class CacheStorage {
  constructor(network) {
    Object.assign(this, { network, caches: new Map() });
  }

  has = async (name) => this.caches.has(name);
  keys = async () => [...this.caches.keys()];
  delete = async (name) => this.caches.delete(name);
  open = async (name) => {
    if (!this.caches.has(name)) this.caches.set(name, new Map());
    return new Cache(this.caches.get(name), this.network);
  };
  // As Chromium has it, and the specification does not: looking a cache up by its name makes the cache, as
  // opening it does. A removed copy's name came back this way, empty, so sw.js looks in every cache at once, which
  // makes none: the first cache that holds the address answers.
  match = async (address, { cacheName } = {}) => {
    if (cacheName !== undefined) return (await this.open(cacheName)).match(address);
    for (const held of this.caches.values()) {
      const found = await new Cache(held, this.network).match(address);
      if (found) return found;
    }
    return undefined;
  };
  held = (name) => [...(this.caches.get(name)?.keys() ?? [])];
}

/**
 * A service worker's script, run as a browser runs one: with `self`, `caches` and `fetch` for globals. Returns
 * `self`, which notes what the script asked of it (`skipped`, `claimed`, `unregistered`), and `fire(type, event)`,
 * which hands an event to the listener the script added and resolves once all it waits for is done.
 */
export function serviceWorker(file, { caches, network, scope }) {
  const listeners = {};
  const self = {
    registration: { scope, unregister: async () => (self.unregistered = true) },
    clients: { claim: async () => (self.claimed = true) },
    skipWaiting: () => (self.skipped = true),
    addEventListener: (type, listener) => (listeners[type] = listener),
  };
  const globals = { self, caches, fetch: network.fetch, URL, Response, ServiceWorkerGlobalScope: class {} };
  vm.runInNewContext(fs.readFileSync(file, "utf8"), globals);
  const fire = async (type, event = {}) => {
    const waits = [];
    Object.assign(event, { waitUntil: (promise) => waits.push(promise), respondWith: (promise) => (event.response = promise) });
    listeners[type](event);
    const answer = await event.response?.then(
      (got) => ({ got }),
      (error) => ({ error }),
    );
    while (waits.length) await waits.shift(); // a wait may be added while another is awaited
    return answer;
  };
  return { self, fire, listens: (type) => type in listeners };
}

/** The request of a fetch event: an address, and how it was asked for. */
export const request = (url, { method = "GET", mode = "cors" } = {}) => ({ url, method, mode });
