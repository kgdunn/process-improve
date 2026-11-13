// Calls to the Python worker: start it, call an api_* entry point and wait for its reply, and stop it.
//
// The page has one worker at a time. A worker cannot be interrupted, so stopping Python ends the worker and starts
// another; every call that was waiting is answered as stopped, and later calls wait for the new worker as the first
// calls waited for the first. A message from a worker that was ended is not the page's any more.
//
// This module touches no document: what the page shows is up to the hooks it is given, so it is tested under
// Node.js with a stand-in worker.

import { STOPPED } from "./logic.js";

/** How long Python may take over a call before the page offers to stop it, in milliseconds. */
export const STOP_AFTER = 2500;

/**
 * The page's line to Python. ``make`` starts a worker. The hooks say what happened, for the page to show:
 * ``onStatus(text, state)`` for a note from the worker on how far it has got; ``onStoppable(offered)`` when a call
 * has waited STOP_AFTER, or none waits any more; ``onRefusal(fn, error)`` for a call Python refused;
 * ``onCrash(message)`` when the worker itself did not load or run; ``onStopped()`` once a worker is ended, before
 * its calls are answered. ``now``, ``setTimer`` and ``clearTimer`` are the clock, for a test to hold.
 */
export function pythonLine({ make, onStatus, onStoppable, onRefusal, onCrash, onStopped, now = Date.now, setTimer = setTimeout, clearTimer = clearTimeout }) {
  let nextId = 0;
  const pending = new Map(); // each call waiting for its reply, by its id: the entry point, who waits, and since when
  let failure = null; // why Python cannot answer, once it cannot: every call is then answered with it
  let timer = null;
  let worker = null;

  /**
   * Offer to stop Python once a call has waited STOP_AFTER, and take the offer away when nothing waits. Python
   * starting is not a call to stop, the first time or after a stop: the offer waits until this worker has started.
   */
  function offerStop() {
    clearTimer(timer);
    const oldest = Math.min(...[...pending.values()].map((waiting) => waiting.since));
    const wait = pending.size && worker.started ? STOP_AFTER - (now() - oldest) : Infinity;
    onStoppable(wait <= 0);
    if (wait > 0 && wait !== Infinity) timer = setTimer(offerStop, wait);
  }

  function start() {
    const made = make();
    made.onmessage = ({ data }) => {
      if (made !== worker) return;
      if (data.status) {
        if (data.state === "ok") made.started = true; // it says it is ready: a call it has now can be stopped
        onStatus(data.status, data.state);
        return;
      }
      const waiting = pending.get(data.id);
      if (!waiting) return; // stopped before its reply came
      pending.delete(data.id);
      offerStop();
      const reply = JSON.parse(data.reply);
      if (!reply.ok) onRefusal(waiting.fn, reply.error);
      waiting.resolve(reply);
    };
    // worker.js itself did not load or run: nothing will answer, so say so rather than wait.
    made.onerror = (event) => {
      event.preventDefault();
      if (made === worker) onCrash(event.message);
    };
    return made;
  }
  worker = start();

  /** Answer every call that waits, and forget them. */
  function answerAll(reply) {
    const waiting = [...pending.values()];
    pending.clear();
    offerStop();
    return () => waiting.forEach(({ resolve }) => resolve(reply));
  }

  return {
    /** Call a Python ``api_*`` function; resolves to its decoded {ok, result|error, warnings}. */
    call(fn, payload = {}) {
      if (failure) return Promise.resolve({ ok: false, error: failure, warnings: [] });
      const id = nextId++;
      return new Promise((resolve) => {
        pending.set(id, { fn, resolve, since: now() });
        worker.postMessage({ id, fn, payload: JSON.stringify(payload) });
        offerStop();
      });
    },

    /** Stop what Python is doing, when a call waits for it; says whether it did. */
    stop() {
      if (!pending.size || failure) return false;
      const answer = answerAll({ ok: false, error: STOPPED, warnings: [], stopped: true });
      worker.terminate();
      worker = start();
      onStopped();
      answer();
      return true;
    },

    /** Python cannot answer: every call that waits, and every later one, is answered with the reason. */
    fail(reason) {
      if (failure) return;
      failure = reason;
      answerAll({ ok: false, error: reason, warnings: [] })();
    },
  };
}
