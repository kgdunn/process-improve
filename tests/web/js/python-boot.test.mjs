// web/python-boot.js: answering a call whose reply says it lacks a package, what is installed for each package
// start-up leaves out, and an error in one line. Starting Pyodide itself is the live and wasm tiers' business;
// here Pyodide and the entry points are stand-ins.

import assert from "node:assert/strict";
import fs from "node:fs";
import { describe, test } from "node:test";

import { answer, installExact, onDemand, oneLine, provider } from "../../../web/python-boot.js";

const RUNTIME = JSON.parse(fs.readFileSync(new URL("../../../web/runtime.json", import.meta.url), "utf8"));

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

/** A release's lock file, as far as these tests go: each package's name as Pyodide loads it, and what it depends on. */
const LOCK = {
  packages: {
    narwhals: { name: "narwhals", depends: [] },
    packaging: { name: "packaging", depends: [] },
    numpy: { name: "numpy", depends: [] },
    patsy: { name: "patsy", depends: ["numpy"] },
    statsmodels: { name: "statsmodels", depends: ["numpy", "patsy", "packaging"] },
    "scikit-learn": { name: "scikit-learn", depends: ["numpy", "joblib", "threadpoolctl"] },
    joblib: { name: "joblib", depends: [] },
    threadpoolctl: { name: "threadpoolctl", depends: [] },
  },
};

/** What a model is fitted with, and all of it that the release's lock file says those two need. */
const FITTING = ["statsmodels", "scikit-learn", "numpy", "patsy", "packaging", "joblib", "threadpoolctl"];

/**
 * A stand-in for Pyodide that notes what it is asked to load and to install. `fails` names a package of the release
 * that does not load, which Pyodide only logs: every other package asked for is loaded all the same. `gate` is
 * waited for before anything is in; and while `refused` holds a reason, micropip installs nothing and gives it.
 */
function standIn(fails = null, gate = Promise.resolve()) {
  const asked = [];
  const pyodide = {
    lockfile: LOCK,
    loadedPackages: {},
    fails,
    refused: null,
    async loadPackage(names) {
      asked.push(["from the release", names]);
      await gate;
      for (const name of names) if (name !== this.fails) this.loadedPackages[name] = "default channel";
    },
    toPy: (value) => ({ value, destroy() {} }),
    pyimport: () => ({
      destroy() {},
      install: {
        async callKwargs(wanted, options) {
          asked.push(["wheels", wanted.value, options]);
          await gate;
          if (pyodide.refused) throw new Error(`Traceback (most recent call last):\nValueError: ${pyodide.refused}`);
        },
      },
    }),
  };
  return { pyodide, asked };
}

describe("what the build resolved to exact files", () => {
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

  test("with no wheel to install, asks micropip for nothing", async () => {
    const { pyodide, asked } = standIn();
    await installExact(pyodide, { packages: ["patsy"], wheels: [] });
    assert.deepEqual(asked, [["from the release", ["patsy", "numpy"]]]);
  });

  test("asks by name for all a package depends on, those named first and each once", async () => {
    const { pyodide, asked } = standIn();
    await installExact(pyodide, { packages: ["statsmodels", "scikit_learn"], wheels: [] });
    assert.deepEqual(asked, [["from the release", FITTING]]);
  });

  test("a package another depends on that did not load is an error, though Pyodide loaded the other", async () => {
    const { pyodide } = standIn("joblib");
    const fitting = { packages: ["statsmodels", "scikit-learn"], wheels: [] };
    await assert.rejects(installExact(pyodide, fitting), /^Error: joblib could not be loaded from the Pyodide release$/);
    assert.ok(pyodide.loadedPackages["scikit-learn"], "which is why the two that were asked for cannot be all that is checked");
  });

  test("a package the release's lock file does not have is an error", async () => {
    const { pyodide, asked } = standIn();
    await assert.rejects(installExact(pyodide, { packages: ["polars"], wheels: [] }), /lock file has no package polars/);
    assert.deepEqual(asked, []);
  });
});

describe("what start-up leaves out", () => {
  const exact = {
    openpyxl: { packages: [], wheels: ["https://files.pythonhosted.org/packages/a/openpyxl-3.1.5-py2.py3-none-any.whl"] },
    plotly: { packages: ["narwhals"], wheels: ["https://files.pythonhosted.org/packages/b/plotly-7.1.0-py3-none-any.whl"] },
  };

  test("is, for a build, the exact files it resolved for a workbook and for a chart", () => {
    const table = onDemand(RUNTIME, { plotly: "7.1.0", on_demand: exact });
    assert.equal(table.openpyxl.exact, exact.openpyxl);
    assert.equal(table.plotly.exact, exact.plotly);
  });

  test("is, with no build, the pins: openpyxl's from runtime.json, and plotly at the release the build drew with", () => {
    assert.deepEqual(onDemand(RUNTIME, {}).openpyxl.pins, RUNTIME.micropip);
    assert.equal(onDemand(RUNTIME, {}).openpyxl.exact, undefined);
    assert.deepEqual(onDemand(RUNTIME, {}).plotly.pins, ["plotly"]);
    assert.deepEqual(onDemand(RUNTIME, { plotly: "7.1.0" }).plotly.pins, ["plotly==7.1.0"]);
  });

  test("holds what a model is fitted with once, by each module a call may lack: packages of the release", () => {
    const table = onDemand(RUNTIME, {});
    for (const module of ["sklearn", "joblib", "threadpoolctl"]) assert.equal(table[module], table.statsmodels, "one install, whichever is missed first");
    assert.deepEqual(table.statsmodels.exact, { packages: RUNTIME.fitting, wheels: [] });
    assert.equal(table.statsmodels.what, "statsmodels and scikit-learn");
  });

  test("is not loaded at the start: the page is ready without what a model is fitted with", () => {
    assert.deepEqual(RUNTIME.fitting, ["statsmodels", "scikit-learn"]);
    for (const name of RUNTIME.fitting) assert.ok(!RUNTIME.packages.includes(name), name);
  });
});

describe("providing a package a call lacks", () => {
  /** A provider over a stand-in Pyodide, with what it said, what it told Python is in, and the entries it reported. */
  function providing({ fails = null, gate, manifest = {} } = {}) {
    const { pyodide, asked } = standIn(fails, gate);
    const said = [];
    const reported = [];
    const told = [];
    const glue = { installed: (modules) => told.push(JSON.parse(modules)) };
    const table = onDemand(RUNTIME, manifest);
    const provide = provider(pyodide, glue, table, (status, state) => said.push([status, state]), (entry) => reported.push(entry));
    return { provide, pyodide, asked, said, reported, told, table };
  }

  test("installs its pins through the package index, and says so from start to end", async () => {
    const { provide, asked, said } = providing();
    await provide("openpyxl");
    assert.deepEqual(asked, [["wheels", RUNTIME.micropip, { deps: true }]]);
    assert.deepEqual(said, [
      ["Installing openpyxl for Excel workbooks...", "busy"],
      ["Installed openpyxl for Excel workbooks.", "ok"],
    ]);
  });

  test("installs the exact files a build resolved, with nothing resolved here", async () => {
    const wheels = ["https://files.pythonhosted.org/packages/b/plotly-7.1.0-py3-none-any.whl"];
    const { provide, asked, said } = providing({ manifest: { plotly: "7.1.0", on_demand: { plotly: { packages: ["narwhals"], wheels } } } });
    await provide("plotly");
    assert.deepEqual(asked, [
      ["from the release", ["narwhals"]],
      ["wheels", wheels, { deps: false }],
    ]);
    assert.deepEqual(said.at(-1), ["Installed plotly for the charts.", "ok"]);
  });

  test("installs a package once, however many calls lack it at once and whichever of its modules they name", async () => {
    const { provide, asked, said } = providing();
    await Promise.all([provide("statsmodels"), provide("sklearn"), provide("statsmodels")]);
    await provide("sklearn");
    assert.deepEqual(asked, [["from the release", FITTING]]);
    assert.deepEqual(said, [
      ["Installing statsmodels and scikit-learn for fitting models...", "busy"],
      ["Installed statsmodels and scikit-learn for fitting models.", "ok"],
    ]);
  });

  test("says nothing of an install nothing waits for, the worker's own once the page is ready", async () => {
    const { provide, asked, said, reported, table } = providing();
    await provide("statsmodels", { quiet: true });
    assert.deepEqual(asked, [["from the release", FITTING]]);
    assert.deepEqual(said, []);
    assert.deepEqual(reported, [table.statsmodels]);
    await provide("sklearn");
    assert.deepEqual(said, [], "nor of a package that is already in, when a call then needs it");
    assert.equal(asked.length, 1);
  });

  test("says so from the moment a call waits for an install that began unasked", async () => {
    let open;
    const gate = new Promise((resolve) => (open = resolve));
    const { provide, asked, said } = providing({ gate });
    const unasked = provide("statsmodels", { quiet: true });
    assert.deepEqual(said, []);
    const waiting = provide("sklearn"); // an analysis, made before the worker had fetched what it needs
    assert.deepEqual(said, [["Installing statsmodels and scikit-learn for fitting models...", "busy"]]);
    open();
    await Promise.all([unasked, waiting]);
    assert.deepEqual(said.at(-1), ["Installed statsmodels and scikit-learn for fitting models.", "ok"]);
    assert.equal(asked.length, 1, "the call joined the install; it began no other");
  });

  test("tells Python the modules of an entry are in, then reports it, once, before any call that waited goes on", async () => {
    const order = [];
    const { pyodide } = standIn();
    const table = onDemand(RUNTIME, {});
    const glue = { installed: (modules) => order.push(`Python told of ${JSON.parse(modules).join(", ")}`) };
    const provide = provider(pyodide, glue, table, () => {}, (entry) => order.push(entry === table.statsmodels ? "reported" : "another"));
    await Promise.all([provide("statsmodels").then(() => order.push("first call")), provide("sklearn").then(() => order.push("second call"))]);
    await provide("statsmodels");
    assert.deepEqual(order, ["Python told of statsmodels, sklearn, joblib, threadpoolctl", "reported", "first call", "second call"]);
  });

  test("tells Python of a package only once it is in: not while it installs, and not if it could not be", async () => {
    let open;
    const gate = new Promise((resolve) => (open = resolve));
    const { provide, pyodide, told } = providing({ gate });
    const installing = provide("openpyxl");
    assert.deepEqual(told, [], "Python still refuses it: its files may be in place, and it is not");
    open();
    await installing;
    assert.deepEqual(told, [["openpyxl"]]);
    pyodide.refused = "no network";
    await assert.rejects(provide("plotly"));
    assert.deepEqual(told, [["openpyxl"]]);
  });

  test("that cannot be installed fails with the reason in one line, says so, and is tried again by the next call", async () => {
    const { provide, pyodide, asked, said, reported } = providing();
    pyodide.refused = "no network";
    await assert.rejects(provide("openpyxl"), { message: "Could not install openpyxl: ValueError: no network" });
    assert.deepEqual(said.at(-1), ["Could not install openpyxl: ValueError: no network", "error"]);
    assert.deepEqual(reported, []);
    pyodide.refused = null;
    await provide("openpyxl");
    assert.equal(asked.length, 2);
    assert.deepEqual(said.at(-1), ["Installed openpyxl for Excel workbooks.", "ok"]);
  });

  test("that could not be fetched unasked fails without a word, and the call that needs it tries again aloud", async () => {
    const { provide, pyodide, asked, said } = providing({ fails: "scikit-learn" });
    await assert.rejects(provide("statsmodels", { quiet: true }), /Could not install statsmodels and scikit-learn: scikit-learn could not be loaded/);
    assert.deepEqual(said, []);
    pyodide.fails = null; // the network is back
    await provide("statsmodels");
    assert.equal(asked.length, 2);
    assert.deepEqual(said.map(([, state]) => state), ["busy", "ok"]);
  });

  test("one of whose packages did not load is not said to Python to be in, and the next call asks for that package again", async () => {
    const { provide, pyodide, asked, told } = providing({ fails: "joblib" });
    await assert.rejects(provide("sklearn"), /Could not install statsmodels and scikit-learn: joblib could not be loaded/);
    assert.deepEqual(told, [], "Python still refuses scikit-learn, which would fail for want of joblib at every fit");
    pyodide.fails = null; // the network is back
    await provide("sklearn");
    assert.ok(asked.at(-1)[1].includes("joblib"), "asked for by name: Pyodide does not look again at what a loaded package needs");
    assert.ok(pyodide.loadedPackages.joblib);
    assert.deepEqual(told, [["statsmodels", "sklearn", "joblib", "threadpoolctl"]]);
  });

  test("refuses a module the page installs nothing for", async () => {
    const { provide, asked } = providing();
    await assert.rejects(provide("torch"), { message: "Python needs torch, which this page does not install." });
    assert.deepEqual(asked, []);
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
