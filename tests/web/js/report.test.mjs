// web/report.js: the study as a report, and the stamp that says how to get its numbers again.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { describe, test } from "node:test";

import {
  RUN_ORDER_SEED,
  briefHtml,
  canonicalJson,
  chartsHtml,
  orderWords,
  report,
  resultWords,
  runSheetHtml,
  runtimeWords,
  sha256,
  stampRows,
  tableHtml,
  whenWords,
} from "../../../web/report.js";

const spec = {
  design_type: "full_factorial",
  factors: [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "Catalyst", low: "Pd", high: "Pt", units: "" },
  ],
  response: "y",
  response_info: { label: "Yield", units: "%", goal: "target", target: 80 },
  model: "interactions",
  options: {},
  constraints: [{ expression: "T <= 190", note: "the seal's rating" }],
};
const design = {
  spec,
  columns: ["Std order", "Run order", "T", "Catalyst", "y"],
  rows: [
    [1, 2, 150, "Pd", null],
    [2, 1, 200, "Pd", null],
    [3, 4, 150, "Pt", null],
    [4, 3, 200, "Pt", null],
  ],
};
const simulation = { label: "A maximum inside the ranges", sigma: 1, seed: 7 };

describe("the digest of the study's inputs", () => {
  test("does not depend on the order the keys were written in", () => {
    assert.equal(canonicalJson({ b: 1, a: { d: [2, { f: 1, e: 0 }], c: null } }), '{"a":{"c":null,"d":[2,{"e":0,"f":1}]},"b":1}');
    assert.equal(canonicalJson({ a: 1, b: 2 }), canonicalJson({ b: 2, a: 1 }));
  });

  test("leaves out a key with no value, as a link's JSON does, and writes a list's gap as null", () => {
    assert.equal(canonicalJson({ a: undefined, b: [undefined, 1] }), '{"b":[null,1]}');
    assert.equal(canonicalJson({ x: "é<" }), '{"x":"é<"}');
  });

  test("is the SHA-256 of the text, in hex, the same as any other tool gives", async () => {
    const text = canonicalJson({ design: design.rows, title: "Yield" });
    assert.equal(await sha256(text), createHash("sha256").update(text, "utf8").digest("hex"));
    assert.match(await sha256(""), /^e3b0c442/, "the digest of nothing is SHA-256's own");
  });
});

describe("the stamp", () => {
  test("gives the moment in UTC, to the minute", () => {
    assert.equal(whenWords(new Date(Date.UTC(2026, 9, 17, 21, 4, 59))), "2026-10-17 21:04 UTC");
  });

  test("names Python, Pyodide in a browser, and the numeric packages' releases", () => {
    const runtime = { python: "3.13.2", pyodide: "314.0.7", packages: { numpy: "2.2.5", scipy: "1.15.3" } };
    assert.equal(runtimeWords(runtime), "Python 3.13.2 in Pyodide 314.0.7, with numpy 2.2.5 and scipy 1.15.3");
    assert.equal(runtimeWords({ python: "3.13.2", pyodide: null, packages: {} }), "Python 3.13.2");
    assert.equal(runtimeWords(null), "");
  });

  test("says how the run order came about: by the seed, or as given", () => {
    assert.equal(orderWords(spec), `Drawn at random with seed ${RUN_ORDER_SEED}.`);
    assert.equal(orderWords({ design_type: "given", options: {} }), "As given.");
    assert.match(orderWords({ design_type: "given", options: { grown: true } }), /then the runs added, drawn at random with seed 42/);
  });

  test("names the study's own seed, and how the runs added to it were drawn", () => {
    const seeded = { ...spec, options: { ...spec.options, random_state: 1792419802 } };
    assert.equal(orderWords(seeded), "Drawn at random with seed 1792419802.");
    assert.equal(orderWords({ design_type: "given", options: { random_state: 1792419802 } }), "As given.");
    assert.equal(
      orderWords({ design_type: "given", options: { grown: true, random_state: 1792419802 } }),
      "As the study had them, then each set of runs added, drawn at random with seed 1792419802 plus the number of runs before it.",
    );
  });

  test("counts the runs with results, entered or simulated, with the process, noise and seed", () => {
    const none = { y: [null, null, null, null] };
    assert.equal(resultWords(none, { y: [null, null, null, null] }, null, 4), "None yet.");
    const typed = { y: [1, 2, null, 4] };
    assert.equal(resultWords(typed, { y: ["entered", "entered", null, "entered"] }, null, 4), "3 of 4 runs, entered.");
    const all = { y: [1, 2, 3, 4] };
    const marks = { y: ["simulated", "simulated", "simulated", "simulated"] };
    assert.equal(
      resultWords(all, marks, simulation, 4),
      "4 of 4 runs, simulated: A maximum inside the ranges, with noise 1 and seed 7.",
    );
    const mixed = { y: ["entered", "simulated", "entered", "simulated"] };
    assert.match(resultWords(all, mixed, simulation, 4), /^4 of 4 runs, 2 entered and 2 simulated: A maximum/);
  });

  test("a run with any response's result counts, and is simulated if any of them is", () => {
    const results = { y: [1, null], z: [null, 2] };
    const marks = { y: ["entered", null], z: [null, "simulated"] };
    assert.equal(resultWords(results, marks, null, 2), "2 of 2 runs, 1 entered and 1 simulated.");
  });

  test("holds a row for each fact, and leaves out what the page does not know", () => {
    const facts = {
      made: new Date(Date.UTC(2026, 9, 17, 21, 4)),
      version: "1.4.0",
      runtime: { python: "3.13.2", pyodide: null, packages: {} },
      url: "https://factori.al/",
      spec,
      runs: 4,
      results: { y: [1, 2, 3, 4] },
      marks: { y: ["entered", "entered", "entered", "entered"] },
      simulation: null,
      sha: "ab".repeat(32),
      link: "https://factori.al/#c=xyz",
    };
    const rows = stampRows(facts);
    assert.deepEqual(
      rows.map(([label]) => label),
      ["Made", "With", "On", "Run order", "Results", "Inputs, SHA-256", "Open it again"],
    );
    assert.equal(rows[1][1], "process-improve 1.4.0, in Factorial at https://factori.al/");
    assert.equal(rows.at(-1)[2], "https://factori.al/#c=xyz");
    const unknown = stampRows({ ...facts, version: null, runtime: null, sha: null, link: null });
    assert.deepEqual(
      unknown.map(([label]) => label),
      ["Made", "With", "Run order", "Results"],
    );
    assert.equal(unknown[1][1], "process-improve, from its source, in Factorial at https://factori.al/");
    const undesigned = stampRows({ ...facts, spec: null, results: {}, marks: {}, sha: null, link: null });
    assert.deepEqual(
      undesigned.map(([label]) => label),
      ["Made", "With", "On"],
      "a page with no design yet, as the Feedback dialog stamps one",
    );
  });
});

describe("the study's tables", () => {
  test("escape every cell", () => {
    assert.equal(
      tableHtml(["<a>"], [["x & y"]]),
      "<table><thead><tr><th>&lt;a&gt;</th></tr></thead><tbody><tr><td>x &amp; y</td></tr></tbody></table>",
    );
  });

  test("the brief: the factors' ranges, the responses' goals, the aim and the limits", () => {
    const html = briefHtml(spec, { aim: "Which factors matter, and how they act together" });
    assert.match(html, /<td>T<\/td><td>150<\/td><td>200<\/td><td>degC<\/td>/);
    assert.match(html, /<td>Catalyst<\/td><td>Pd<\/td><td>Pt<\/td><td><\/td>/, "a categorical factor has no units");
    assert.match(html, /<td>Yield<\/td><td>%<\/td><td>Hit a target: 80<\/td>/);
    assert.match(html, /<p>Aim: Which factors matter, and how they act together<\/p>/);
    assert.match(html, /<li><code>T &lt;= 190<\/code>: the seal's rating<\/li>/);
    assert.match(briefHtml(spec, { mixture: true }), /<th>Component<\/th><th>Least<\/th><th>Most<\/th>/);
    assert.doesNotMatch(briefHtml({ ...spec, constraints: [] }), /Limits/);
  });

  test("the run sheet holds each response's results, a simulated one in italics", () => {
    const html = runSheetHtml(design, { y: [71.2, null, 80, 79.5] }, { y: ["entered", null, "simulated", "simulated"] });
    assert.match(html, /<tr><td>1<\/td><td>2<\/td><td>150<\/td><td>Pd<\/td><td>71.2<\/td><\/tr>/);
    assert.match(html, /<tr><td>2<\/td><td>1<\/td><td>200<\/td><td>Pd<\/td><td><\/td><\/tr>/, "a run without its result");
    assert.match(html, /<td class="simulated">80<\/td>/);
    assert.match(html, /A result in italics was simulated/);
    assert.doesNotMatch(runSheetHtml(design, { y: [1, 2, 3, 4] }, { y: ["entered", "entered", "entered", "entered"] }), /italics/);
  });

  test("the charts are figures with their heading, image and caption", () => {
    const html = chartsHtml([{ title: "Residuals", caption: "No pattern.", src: "data:image/png;base64,AA==" }]);
    assert.equal(
      html,
      '<figure><h4>Residuals</h4><img src="data:image/png;base64,AA==" alt="Residuals"><figcaption>No pattern.</figcaption></figure>',
    );
  });
});

describe("the report", () => {
  const stamp = [
    ["Made", "2026-10-17 21:04 UTC", null],
    ["Inputs, SHA-256", "ab".repeat(32), null],
    ["Open it again", "This study in Factorial, with its results", "https://factori.al/#c=a&b"],
  ];
  const page = report({ title: "Yield <study>", stamp, sections: [{ heading: "The brief", html: "<p>held</p>" }] });

  test("is a whole page, its title escaped, that needs nothing from outside", () => {
    assert.match(page, /^<!doctype html>/);
    assert.match(page, /<title>Yield &lt;study&gt;: report<\/title>/);
    assert.match(page, /<h1>Yield &lt;study&gt;<\/h1>/);
    assert.doesNotMatch(page, /<script|<link |src="http/, "no script, no stylesheet, no image from elsewhere");
  });

  test("carries the stamp, the digest set apart, the link to open the study again, and says what the digest is of", () => {
    assert.match(page, /<th>Made<\/th><td>2026-10-17 21:04 UTC<\/td>/);
    assert.match(page, new RegExp(`<td class="digest">${"ab".repeat(32)}</td>`));
    assert.match(page, /<a href="https:\/\/factori.al\/#c=a&amp;b">This study in Factorial, with its results<\/a>/);
    assert.match(page, /The SHA-256 is of the study as Factorial reads it from its link/);
    assert.doesNotMatch(report({ title: "t", stamp: stamp.slice(0, 1), sections: [] }), /The SHA-256 is of/);
  });

  test("puts each section under its heading, its HTML as given", () => {
    assert.match(page, /<section>\n<h2>The brief<\/h2>\n<p>held<\/p>\n<\/section>/);
  });
});
