// web/python-boot.js: answering a call whose reply says it lacks a package, and an error in one line.
// Starting Pyodide itself is the live and wasm tiers' business; here the entry points are stand-ins.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { answer, installExact, oneLine } from "../../../web/python-boot.js";

const lacks = (name) => JSON.stringify({ ok: false, needs: name });
const done = JSON.stringify({ ok: true, result: { runs: 8 }, warnings: [] });

/** An entry point that lacks each of `missing` in turn until it is provided, then answers. */
function entryThatLacks(...missing) {
  const calls = [];
  const provided = [];
  const entry = (payload) => {
    calls.push(payload);
    const still = missing.find((name) => !provided.includes(name));
    return still ? lacks(still) : done;
  };
  return { entry, calls, provided, provide: async (name) => provided.push(name) };
}

describe("a call to Python", () => {
  test("is answered as it is when it lacks nothing, and nothing is installed", async () => {
    const { entry, calls, provided, provide } = entryThatLacks();
    assert.equal(await answer(entry, "{}", provide), done);
    assert.deepEqual([calls, provided], [["{}"], []]);
  });

  test("that lacks a package has it provided, and is made again with the same request", async () => {
    const { entry, calls, provided, provide } = entryThatLacks("openpyxl");
    assert.equal(await answer(entry, '{"rows": []}', provide), done);
    assert.deepEqual(provided, ["openpyxl"]);
    assert.deepEqual(calls, ['{"rows": []}', '{"rows": []}']);
  });

  test("that lacks two packages has each provided once", async () => {
    const { entry, calls, provided, provide } = entryThatLacks("openpyxl", "plotly");
    assert.equal(await answer(entry, "{}", provide), done);
    assert.deepEqual(provided, ["openpyxl", "plotly"]);
    assert.equal(calls.length, 3);
  });

  test("that still lacks a package once it was provided is an error, not a loop", async () => {
    let asked = 0;
    const entry = () => lacks("plotly");
    const provide = async () => (asked += 1);
    await assert.rejects(answer(entry, "{}", provide), /Python still lacks plotly after installing it/);
    assert.equal(asked, 1);
  });

  test("whose package cannot be provided fails with the reason, and is not made again", async () => {
    const { entry, calls } = entryThatLacks("openpyxl");
    const provide = async () => {
      throw new Error("Could not install openpyxl: no network");
    };
    await assert.rejects(answer(entry, "{}", provide), /Could not install openpyxl: no network/);
    assert.equal(calls.length, 1);
  });

  test("is not read for what it lacks when its reply is long: only Python's short reply says so", async () => {
    // A megabyte of charts is not parsed a second time to look for the word: were it read, this reply would ask.
    const long = JSON.stringify({ ok: true, needs: "plotly", result: { rows: Array(200).fill(0) }, warnings: [] });
    const provide = async () => assert.fail("nothing is lacking");
    assert.equal(await answer(() => long, "{}", provide), long);
  });

  test("that is refused in a short sentence lacks nothing", async () => {
    const refused = JSON.stringify({ ok: false, error: "Unknown design type 'x'.", warnings: [] });
    assert.equal(await answer(() => refused, "{}", async () => assert.fail("nothing is lacking")), refused);
  });
});

describe("what the build resolved to exact files", () => {
  /** A stand-in for Pyodide that notes what it is asked to load and to install; `fails` names a package that does not load. */
  function standIn(fails = null) {
    const asked = [];
    const pyodide = {
      loadedPackages: {},
      async loadPackage(names) {
        asked.push(["from the release", names]);
        for (const name of names) if (name !== fails) this.loadedPackages[name] = "default channel";
      },
      toPy: (value) => ({ value, destroy() {} }),
      pyimport: () => ({ destroy() {}, install: { callKwargs: async (wanted, options) => asked.push(["wheels", wanted.value, options]) } }),
    };
    return { pyodide, asked };
  }
  const plotly = { packages: ["narwhals", "packaging"], wheels: ["https://files.pythonhosted.org/packages/ab/plotly-7.1.0-py3-none-any.whl"] };

  test("is installed as it is: the release's packages by name, then the wheels, with nothing resolved", async () => {
    const { pyodide, asked } = standIn();
    await installExact(pyodide, plotly);
    assert.deepEqual(asked, [
      ["from the release", ["narwhals", "packaging"]],
      ["wheels", plotly.wheels, { deps: false }],
    ]);
  });

  test("with nothing needed of the release, asks it for nothing", async () => {
    const { pyodide, asked } = standIn();
    await installExact(pyodide, { packages: [], wheels: ["a.whl", "b.whl"] });
    assert.deepEqual(asked, [["wheels", ["a.whl", "b.whl"], { deps: false }]]);
  });

  test("a package of the release that did not load is an error, and no wheel is installed over the gap", async () => {
    const { pyodide, asked } = standIn("narwhals");
    await assert.rejects(installExact(pyodide, plotly), /narwhals could not be loaded from the Pyodide release/);
    assert.equal(asked.length, 1);
  });
});

describe("an error in one line", () => {
  test("is the last line of a Python traceback, which names the exception", () => {
    const traceback = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: no such design\n';
    assert.equal(oneLine(new Error(traceback)), "ValueError: no such design");
  });

  test("is the message of an error that has one line, or the thing thrown", () => {
    assert.equal(oneLine(new Error("worker.js did not load")), "worker.js did not load");
    assert.equal(oneLine("offline"), "offline");
  });
});
