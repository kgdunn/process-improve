// The browser tests' harness, loaded into the page before app.js.
//
// It records every call the page makes to its Python worker and every reply, so a test can
// wait until the page has nothing in flight, hold a reply back to stage a race, or give a
// reply of its own to reach an error path. With window.__piTransport = "native" (set by the
// test before this script runs) the worker is never started: each call is answered in the
// test process, by web/bootstrap.py run natively, through the binding window.__piNative.
// Otherwise the page's own worker runs, Pyodide and all, and is only watched.

(() => {
  const pi = {
    calls: [], // {fn, payload} for each call, in the order the page made it
    replies: [], // {fn, reply} for each reply delivered to the page
    pending: 0, // calls made and not yet answered
    held: new Set(), // entry points whose replies wait for release()
    queue: [], // replies held back: {fn, deliver}
    fakes: {}, // entry point -> the reply to give instead of asking Python
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
  window.__pi = pi;

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

    postMessage({ id, fn, payload }) {
      pi.calls.push({ fn, payload: JSON.parse(payload) });
      pi.pending += 1;
      const reply = fn in pi.fakes ? Promise.resolve(JSON.stringify(pi.fakes[fn])) : window.__piNative(fn, payload);
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
      this.fns = new Map();
      this.handler = null;
      super.onmessage = (event) => {
        const { id, reply } = event.data;
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
