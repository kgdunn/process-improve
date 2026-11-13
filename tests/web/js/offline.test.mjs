// web/offline.js and web/sw.js: what is kept of the page for use without a network, and how each address is served.
// The service worker's two rules are read here under Node.js; that the copy opens the page is the live tier's test.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { APP, KEPT, copyKey, copyWords, pyodideFiles, wholeKey } from "../../../web/offline.js";
import worker from "../../../web/sw.js";

const SCOPE = "https://factori.al/";
const RELEASE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/pyc/";

describe("how the service worker serves an address", () => {
  test("a file whose address names its version is kept, and served from the copy", () => {
    for (const address of [
      `${RELEASE}pyodide.asm.wasm`,
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
      `${SCOPE}app.js`,
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

describe("which kept files a later build leaves behind", () => {
  const now = { folder: "bbbbbbbbbbbbbbbb", pyodide: "v314.0.7/pyc", plotlyJs: "4.1.1" };

  test("the wheel and the page's Python of an earlier build go, and this build's stay", () => {
    assert.ok(worker.stale(`${SCOPE}wheels/aaaaaaaaaaaaaaaa/bootstrap.zip`, SCOPE, now));
    assert.ok(!worker.stale(`${SCOPE}wheels/bbbbbbbbbbbbbbbb/bootstrap.zip`, SCOPE, now));
  });

  test("another Pyodide release, or another folder of this one, goes", () => {
    assert.ok(worker.stale("https://cdn.jsdelivr.net/pyodide/v314.0.6/pyc/pyodide.asm.wasm", SCOPE, now));
    assert.ok(worker.stale("https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.asm.wasm", SCOPE, now));
    assert.ok(!worker.stale(`${RELEASE}pyodide.asm.wasm`, SCOPE, now));
  });

  test("another Plotly.js goes, and PyPI's files stay: nothing names their successors", () => {
    assert.ok(worker.stale("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.0.0/plotly.min.js", SCOPE, now));
    assert.ok(!worker.stale("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js", SCOPE, now));
    assert.ok(!worker.stale("https://files.pythonhosted.org/packages/ab/cd/plotly-7.0.0-py3-none-any.whl", SCOPE, now));
  });

  test("a file is kept when nothing is known of what replaced its kind", () => {
    assert.ok(!worker.stale(`${SCOPE}wheels/aaaaaaaaaaaaaaaa/bootstrap.zip`, SCOPE, { pyodide: "v314.0.7/pyc" }));
    assert.ok(!worker.stale("https://cdn.jsdelivr.net/pyodide/v314.0.6/pyc/x.whl", SCOPE, { folder: "bbbbbbbbbbbbbbbb" }));
  });
});

describe("the copy the page makes", () => {
  const lock = {
    packages: {
      numpy: { file_name: "numpy.whl", depends: [] },
      pandas: { file_name: "pandas.whl", depends: ["numpy", "python-dateutil"] },
      "python-dateutil": { file_name: "dateutil.whl", depends: ["six"] },
      six: { file_name: "six.whl", depends: [] },
      "scikit-learn": { file_name: "sklearn.whl", depends: ["numpy"] },
      unused: { file_name: "unused.whl", depends: [] },
    },
  };

  test("holds the interpreter, its library, the lock file, and each package with all it depends on, once", () => {
    const files = pyodideFiles(lock, ["pandas", "scikit_learn"]);
    assert.deepEqual(files.slice(0, 5), ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]);
    assert.deepEqual(files.slice(5).sort(), ["dateutil.whl", "numpy.whl", "pandas.whl", "six.whl", "sklearn.whl"]);
  });

  test("is not made of a package the release does not have", () => {
    assert.throws(() => pyodideFiles(lock, ["pandas", "polars"]), /no package polars/);
  });

  test("is named by the build and the Pyodide it runs on, and the whole copy by what it adds", () => {
    const runtime = { pyodide: "314.0.7", distribution: "pyc", micropip: ["openpyxl==3.1.5"] };
    const manifest = { wheel: "abc/process_improve.whl", plotly: "7.1.0", plotly_js: "4.1.1" };
    assert.equal(copyKey(runtime, manifest), "abc/process_improve.whl|314.0.7/pyc");
    assert.notEqual(copyKey(runtime, manifest), copyKey(runtime, { ...manifest, wheel: "abd/process_improve.whl" }));
    assert.notEqual(wholeKey(runtime, manifest), wholeKey(runtime, { ...manifest, plotly: "7.2.0" }));
    assert.ok(wholeKey(runtime, manifest).startsWith(copyKey(runtime, manifest)));
  });

  test("is under the names the service worker serves from", () => {
    assert.deepEqual([KEPT, APP], [worker.KEPT, worker.APP]);
  });
});

describe("what the footer says of the copy", () => {
  test("a page that opens without a network offers its charts and workbooks, and to remove the copy", () => {
    const words = copyWords("opens");
    assert.match(words.said, /^Factorial opens on this device without a network/);
    assert.equal(words.keep, "Keep its charts and workbooks too (about 12 MB)");
    assert.ok(words.remove);
  });

  test("the whole copy offers nothing more to keep", () => {
    const words = copyWords("whole");
    assert.match(words.said, /opens and works on this device without a network/);
    assert.deepEqual([words.keep, words.remove], [null, true]);
  });

  test("while the charts and workbooks download, neither button is offered", () => {
    assert.deepEqual([copyWords("keeping").keep, copyWords("keeping").remove], [null, false]);
  });

  test("a copy that could not be finished says why, and can be asked for again", () => {
    const words = copyWords("failed", "Could not install plotly: no network");
    assert.ok(words.said.endsWith("Could not install plotly: no network"));
    assert.equal(words.keep, copyWords("opens").keep);
  });

  test("a reader who removed the copy is offered one again", () => {
    assert.deepEqual(copyWords("none"), {
      said: "Factorial keeps no copy of itself on this device.",
      keep: "Keep a copy, to open it without a network",
      remove: false,
    });
  });
});
