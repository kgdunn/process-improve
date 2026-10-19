// The browser tests' harness, loaded into the page before app.js.
//
// It records every call the page makes to its Python worker and every reply, so a test can
// wait until the page has nothing in flight, hold a reply back to stage a race, or give a
// reply of its own to reach an error path. With window.__piTransport = "native" (set by the
// test before this script runs) the worker is never started: each call is answered in the
// test process, by web/bootstrap.py run natively, through the binding window.__piNative, with a
// fixed stand-in for each study seed (see pinSeeds).
// Otherwise the page's own worker runs, Pyodide and all, and is only watched.

(() => {
  const post = Worker.prototype.postMessage; // the browser's own, before Worker is replaced below
  const pi = {
    calls: [], // {fn, payload} for each call, in the order the page made it
    replies: [], // {fn, reply} for each reply delivered to the page
    pending: 0, // calls made and not yet answered
    held: new Set(), // entry points whose replies wait for release()
    queue: [], // replies held back: {fn, deliver}
    fakes: {}, // entry point -> the reply to give instead of asking Python
    asked: new Map(), // id -> resolve, for each call a test makes through ask()
    worker: null, // the page's own worker, once it has made one
    seeds: new Map(), // each study seed the page sent -> the stand-in Python was given for it
  };
  pi.idle = () => pi.pending === 0 && pi.queue.length === 0;
  pi.hold = (fn) => pi.held.add(fn);
  pi.release = (fn) => {
    pi.held.delete(fn);
    const due = pi.queue.filter((r) => r.fn === fn);
    pi.queue = pi.queue.filter((r) => r.fn !== fn);
    due.forEach((r) => r.deliver());
  };
  pi.fake = (fn, reply) => {
    pi.fakes[fn] = reply;
  };
  pi.count = (fn) => pi.calls.filter((c) => c.fn === fn).length;
  /** Ask Python directly, as the page would, without the page seeing the call or its reply. */
  pi.ask = (fn, payload = {}) => {
    if (window.__piTransport === "native") return window.__piNative(fn, JSON.stringify(payload)).then(JSON.parse);
    return new Promise((resolve) => {
      const id = `asked-${pi.asked.size}-${performance.now()}`;
      pi.asked.set(id, resolve);
      post.call(pi.worker, { id, fn, payload: JSON.stringify(payload) });
    });
  };
  window.__pi = pi;

  /**
   * A study's seed is the second it began (app.js's studySeed), so a test's runs would come in another order on
   * each run of it. Python is given a stand-in for each such seed instead: 42 for the first the page sends, 43 for
   * the next, and so on. A test's runs are then the same on every run, a seed the page keeps is still kept, and a
   * new one is still new. The reply gives the page its own seed back, wherever Python wrote the stand-in, so the
   * page sees the seeds it drew. A seed below SECONDS, such as the worked example's 42, goes as it is.
   */
  const SECONDS = 1e9;

  /** A copy of ``value`` with each random_state that ``swap`` holds replaced by its image there. */
  function swapSeeds(value, swap) {
    if (Array.isArray(value)) return value.map((v) => swapSeeds(v, swap));
    if (value === null || typeof value !== "object") return value;
    const entries = Object.entries(value).map(([k, v]) => [k, k === "random_state" && swap.has(v) ? swap.get(v) : swapSeeds(v, swap)]);
    return Object.fromEntries(entries);
  }

  /** The payload Python is given for the page's, and a way to give the page its seeds back in the reply. */
  function pinSeeds(sent) {
    const drawn = [...sent.matchAll(/"random_state":\s*(\d+)/g)].map((m) => Number(m[1])).filter((v) => v > SECONDS);
    if (!drawn.length) return { payload: sent, unpin: (text) => text };
    for (const seed of drawn) if (!pi.seeds.has(seed)) pi.seeds.set(seed, 42 + pi.seeds.size);
    const back = new Map(drawn.map((seed) => [pi.seeds.get(seed), seed]));
    return {
      payload: JSON.stringify(swapSeeds(JSON.parse(sent), pi.seeds)),
      unpin: (text) => JSON.stringify(swapSeeds(JSON.parse(text), back)),
    };
  }

  /** Route one reply to the page, now or when its entry point is released. */
  function answer(fn, deliverNow) {
    const deliver = () => {
      pi.pending -= 1;
      deliverNow();
    };
    if (pi.held.has(fn)) pi.queue.push({ fn, deliver });
    else deliver();
  }

  /**
   * A worker answered by the test process: the messages worker.js sends, without Pyodide.
   * Like worker.js, it says it is ready just before its first reply, since every call waits
   * for Python to start.
   */
  class NativeWorker extends EventTarget {
    constructor(url) {
      super();
      this.url = url;
      this.onmessage = null;
      this.onerror = null;
      this.started = false;
    }

    postMessage({ id, fn, payload: sent }) {
      const { payload, unpin } = pinSeeds(sent);
      pi.calls.push({ fn, payload: JSON.parse(sent) });
      pi.pending += 1;
      const reply = fn in pi.fakes ? Promise.resolve(JSON.stringify(pi.fakes[fn])) : window.__piNative(fn, payload).then(unpin);
      reply.then((text) =>
        answer(fn, () => {
          if (!this.started) {
            this.started = true;
            this.emit({ status: "Ready: process-improve (native)", state: "ok" });
          }
          pi.replies.push({ fn, reply: JSON.parse(text) });
          this.emit({ id, reply: text });
        }),
      );
    }

    emit(data) {
      const event = new MessageEvent("message", { data });
      this.onmessage?.(event);
      this.dispatchEvent(event);
    }

    terminate() {}
  }

  /** The page's own worker, watched: calls and replies are recorded on their way past. */
  class WatchedWorker extends Worker {
    constructor(url, options) {
      super(url, options);
      pi.worker = this;
      this.fns = new Map();
      this.handler = null;
      super.onmessage = (event) => {
        const { id, reply } = event.data;
        if (pi.asked.has(id)) {
          pi.asked.get(id)(JSON.parse(reply));
          pi.asked.delete(id);
          return;
        }
        if (id === undefined || !this.fns.has(id)) {
          this.handler?.(event);
          return;
        }
        const fn = this.fns.get(id);
        this.fns.delete(id);
        answer(fn, () => {
          pi.replies.push({ fn, reply: JSON.parse(reply) });
          this.handler?.(event);
        });
      };
    }

    set onmessage(handler) {
      this.handler = handler;
    }

    get onmessage() {
      return this.handler;
    }

    postMessage(message) {
      pi.calls.push({ fn: message.fn, payload: JSON.parse(message.payload) });
      pi.pending += 1;
      if (message.fn in pi.fakes) {
        answer(message.fn, () => {
          const reply = JSON.stringify(pi.fakes[message.fn]);
          pi.replies.push({ fn: message.fn, reply: pi.fakes[message.fn] });
          this.handler?.(new MessageEvent("message", { data: { id: message.id, reply } }));
        });
        return;
      }
      this.fns.set(message.id, message.fn);
      super.postMessage(message);
    }
  }

  window.Worker = window.__piTransport === "native" ? NativeWorker : WatchedWorker;
})();
