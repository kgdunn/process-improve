// web/worker-rpc.js: the page's line to Python, with a stand-in worker and a clock the test holds.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { STOPPED } from "../../../web/logic.js";
import { STOP_AFTER, pythonLine } from "../../../web/worker-rpc.js";

/** A line to stand-in workers: what each was sent, what the hooks were told, and a clock moved by hand. */
function line() {
  const workers = [];
  const told = { status: [], stoppable: [], refusals: [], crashes: [], stopped: 0 };
  const clock = { now: 0, timers: [] };
  const python = pythonLine({
    make() {
      const worker = { sent: [], ended: false, postMessage: (message) => worker.sent.push(message), terminate: () => (worker.ended = true) };
      workers.push(worker);
      return worker;
    },
    onStatus: (text, state) => told.status.push([text, state]),
    onStoppable: (offered) => told.stoppable.push(offered),
    onRefusal: (fn, error) => told.refusals.push([fn, error]),
    onCrash: (message) => told.crashes.push(message),
    onStopped: () => (told.stopped += 1),
    now: () => clock.now,
    setTimer: (run, wait) => clock.timers.push({ run, at: clock.now + wait }) - 1,
    clearTimer: (id) => id !== null && (clock.timers[id] = null),
  });
  /** Move the clock on, and run the timers that are due. */
  const wait = (ms) => {
    clock.now += ms;
    clock.timers.forEach((timer, id) => {
      if (timer && timer.at <= clock.now) {
        clock.timers[id] = null;
        timer.run();
      }
    });
  };
  const ready = (worker = workers.at(-1)) => worker.onmessage({ data: { status: "Ready", state: "ok" } });
  const reply = (worker, id, body) => worker.onmessage({ data: { id, reply: JSON.stringify(body) } });
  return { python, workers, told, wait, ready, reply };
}

describe("the line to Python", () => {
  test("a call is posted to the worker and resolves to its reply", async () => {
    const { python, workers, reply } = line();
    const asked = python.call("api_make_design", { k: 3 });
    assert.deepEqual(workers[0].sent, [{ id: 0, fn: "api_make_design", payload: '{"k":3}' }]);
    reply(workers[0], 0, { ok: true, result: { n_runs: 8 }, warnings: [] });
    assert.deepEqual(await asked, { ok: true, result: { n_runs: 8 }, warnings: [] });
  });

  test("a call Python refuses is told to the page, and still resolves", async () => {
    const { python, workers, told, reply } = line();
    const asked = python.call("api_plots");
    reply(workers[0], 0, { ok: false, error: "No runs.", warnings: [] });
    assert.equal((await asked).ok, false);
    assert.deepEqual(told.refusals, [["api_plots", "No runs."]]);
  });

  test("Stop is offered once a call has waited a while, and not while Python is starting", () => {
    const { python, told, wait, ready } = line();
    python.call("api_candidates");
    wait(STOP_AFTER + 1);
    assert.ok(!told.stoppable.includes(true), "the first worker has not said it is ready");
    ready();
    python.call("api_candidates");
    assert.equal(told.stoppable.at(-1), true, "the first call has waited long enough already");
  });

  test("Stop is not offered for a call answered quickly, and is taken away with the last call", () => {
    const { python, workers, told, wait, ready, reply } = line();
    ready();
    python.call("api_make_design");
    wait(STOP_AFTER - 1);
    assert.ok(!told.stoppable.includes(true));
    wait(2);
    assert.equal(told.stoppable.at(-1), true);
    reply(workers[0], 0, { ok: true, result: {}, warnings: [] });
    assert.equal(told.stoppable.at(-1), false);
  });

  test("stopping ends the worker, starts another, and answers every waiting call as stopped", async () => {
    const { python, workers, told, ready } = line();
    ready();
    const [first, second] = [python.call("api_candidates"), python.call("api_plots")];
    assert.equal(python.stop(), true);
    assert.equal(workers[0].ended, true);
    assert.equal(workers.length, 2);
    assert.equal(told.stopped, 1);
    for (const answer of [await first, await second]) assert.deepEqual(answer, { ok: false, error: STOPPED, warnings: [], stopped: true });
    assert.deepEqual(told.refusals, [], "a call the reader stopped is not an error of Python's");
  });

  test("a reply or a note from the worker that was ended is not heard", async () => {
    const { python, workers, told, ready, reply } = line();
    ready();
    const stopped = python.call("api_candidates");
    python.stop();
    await stopped;
    const noted = told.status.length;
    reply(workers[0], 0, { ok: true, result: { late: true }, warnings: [] });
    ready(workers[0]);
    assert.equal(told.status.length, noted);
    const next = python.call("api_make_design");
    assert.equal(workers[1].sent[0].id, 1, "ids are not used again");
    reply(workers[1], 1, { ok: true, result: { n_runs: 8 }, warnings: [] });
    assert.equal((await next).result.n_runs, 8);
  });

  test("there is nothing to stop when no call waits", () => {
    const { python, workers } = line();
    assert.equal(python.stop(), false);
    assert.equal(workers.length, 1);
  });

  test("Stop is not offered again while the new worker is starting", () => {
    const { python, told, wait, ready } = line();
    ready();
    python.call("api_candidates");
    python.stop();
    python.call("api_candidates");
    wait(STOP_AFTER * 2);
    assert.equal(told.stoppable.at(-1), false, "the worker started after the stop has not said it is ready");
    ready();
    python.call("api_plots");
    assert.equal(told.stoppable.at(-1), true);
  });

  test("when Python cannot answer, every waiting call and every later one is answered with the reason", async () => {
    const { python, workers } = line();
    const waiting = python.call("api_catalogue");
    python.fail("Could not start Python: offline");
    assert.deepEqual(await waiting, { ok: false, error: "Could not start Python: offline", warnings: [] });
    assert.deepEqual(await python.call("api_make_design"), { ok: false, error: "Could not start Python: offline", warnings: [] });
    assert.equal(workers[0].sent.length, 1, "nothing more is posted");
    assert.equal(python.stop(), false);
  });

  test("a worker that did not load is told to the page", () => {
    const { workers, told } = line();
    let prevented = false;
    workers[0].onerror({ message: "404", preventDefault: () => (prevented = true) });
    assert.deepEqual(told.crashes, ["404"]);
    assert.ok(prevented);
  });
});
