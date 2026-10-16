// web/catalog-logic.js: what the catalog page reads from a search, which designs pass its filters,
// what its address keeps, and how it draws and copies a design, apart from the page.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  DEFAULT_STATE,
  checkIndex,
  checkLine,
  csvText,
  decodeGlyph,
  factLine,
  familyCounts,
  filtersOf,
  fromHash,
  INDEX_FIELDS,
  minimumRuns,
  parseQuery,
  pythonCode,
  results,
  settingColour,
  sorted,
  sourceLine,
  switchCount,
  toHash,
  yearBins,
} from "../../../web/catalog-logic.js";

const state = (changes = {}) => ({ ...structuredClone({ ...DEFAULT_STATE }), ...changes });

const design = (changes) => ({
  id: "x",
  title: "A design",
  collection: "published",
  kind: "design",
  family: "full_factorial",
  design_type: "full_factorial",
  source: "bhh",
  year: 1978,
  added: "2026-10-14",
  where: "pp. 307-324; Table 10.1",
  n_runs: 8,
  n_factors: 3,
  n_numeric: 3,
  n_categorical: 0,
  n_mixture: 0,
  levels: [2, 2, 2],
  n_centre: 0,
  runs_per_parameter: 1.14,
  models: { main_effects: true, interactions: true, quadratic: false },
  blocked: false,
  n_blocks: null,
  split_plot: false,
  has_responses: false,
  n_responses: 0,
  checks: { printed: 4, checked: 3, agree: 3, explained: 0 },
  glyph: "",
  regenerable: true,
  open: true,
  detail: "bhh",
  ...changes,
});

const CATALOG = {
  sources: {
    bhh: { short: "Box, Hunter and Hunter", year: 1978, cite: "Box, G. E. P. (1978)", title: "Statistics for Experimenters" },
    jn: { short: "Jones and Nachtsheim", year: 2013, cite: "Jones and Nachtsheim (2013)", title: "Definitive Screening" },
  },
  families: {
    full_factorial: { label: "Full factorial" },
    dsd: { label: "Definitive screening" },
    ccd: { label: "Central composite" },
  },
  designs: [
    design({ id: "pilot-plant", title: "Pilot plant", has_responses: true, n_responses: 1, n_categorical: 1 }),
    design({
      id: "dsd-categorical",
      title: "DSD with a categorical factor",
      family: "dsd",
      source: "jn",
      year: 2013,
      added: "2026-10-13",
      n_runs: 14,
      n_factors: 5,
      n_categorical: 1,
      checks: { printed: 9, checked: 9, agree: 8, explained: 1 },
      open: false,
      regenerable: false,
    }),
    design({
      id: "lib-ccd-k3",
      title: "Central composite, 3 factors",
      collection: "generated",
      family: "ccd",
      source: null,
      year: null,
      added: null,
      where: "",
      n_runs: 17,
      models: { main_effects: true, interactions: true, quadratic: true },
      checks: null,
      regenerable: null,
    }),
  ],
};

const ids = (designs) => designs.map((d) => d.id);

describe("the search box", () => {
  test("reads factors, runs, a model and categorical factors", () => {
    const asked = parseQuery("5 factors, <=20 runs, quadratic, 1 categorical");
    assert.deepEqual(asked.factors, [5, 5]);
    assert.deepEqual(asked.runs, [null, 20]);
    assert.equal(asked.model, "quadratic");
    assert.deepEqual(asked.categorical, [1, null]);
    assert.deepEqual(asked.words, []);
  });

  test("reads ranges, dates, blocks and data", () => {
    const asked = parseQuery("3-7 factors 8 to 16 runs before 1980 blocked with data");
    assert.deepEqual(asked.factors, [3, 7]);
    assert.deepEqual(asked.runs, [8, 16]);
    assert.deepEqual(asked.years, [null, 1979]);
    assert.equal(asked.blocked, true);
    assert.equal(asked.hasData, true);
  });

  test("reads under, at least, since, a single year and a page", () => {
    assert.deepEqual(parseQuery("under 13 runs").runs, [null, 12]);
    assert.deepEqual(parseQuery("at least 20 runs").runs, [20, null]);
    assert.deepEqual(parseQuery("since 2010").years, [2010, null]);
    const asked = parseQuery("box hunter 1978 p. 272");
    assert.deepEqual(asked.years, [1978, 1978]);
    assert.deepEqual(asked.words, ["272", "box", "hunter"]);
  });

  test("a family's name is a family, not a model", () => {
    const asked = parseQuery("definitive screening designs");
    assert.deepEqual(asked.families, ["dsd"]);
    assert.equal(asked.model, undefined, "the word screening is the family's here");
    assert.deepEqual(parseQuery("CCD or Box-Behnken").families, ["ccd", "box_behnken"]);
  });

  test("no categorical factors is a filter of its own", () => {
    assert.deepEqual(parseQuery("no categorical factors").categorical, [null, 0]);
  });
});

describe("the filters", () => {
  test("the toggle shows the published designs, the generated ones, or both", () => {
    assert.deepEqual(ids(results(CATALOG, state())), ["pilot-plant", "dsd-categorical", "lib-ccd-k3"]);
    assert.deepEqual(ids(results(CATALOG, state({ collection: "published" }))), ["pilot-plant", "dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ collection: "generated" }))), ["lib-ccd-k3"]);
  });

  test("a model keeps the designs that estimate it in full", () => {
    assert.deepEqual(ids(results(CATALOG, state({ model: "quadratic" }))), ["lib-ccd-k3"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "quadratic" }))), ["lib-ccd-k3"]);
  });

  test("a year keeps published designs only: a generated one has none", () => {
    assert.deepEqual(ids(results(CATALOG, state({ years: [2000, null] }))), ["dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ addedSince: "2026-10-14" }))), ["pilot-plant"]);
  });

  test("the yes-or-no filters", () => {
    assert.deepEqual(ids(results(CATALOG, state({ hasData: true }))), ["pilot-plant"]);
    assert.deepEqual(ids(results(CATALOG, state({ differs: true }))), ["dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ recomputed: true }))), ["pilot-plant", "dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ opens: true }))), ["pilot-plant", "lib-ccd-k3"]);
  });

  test("the sidebar and the search box must both hold", () => {
    assert.deepEqual(ids(results(CATALOG, state({ families: ["dsd"], query: "ccd" }))), []);
    assert.deepEqual(ids(results(CATALOG, state({ families: ["dsd", "ccd"], query: "ccd" }))), ["lib-ccd-k3"]);
    assert.deepEqual(filtersOf(state({ runs: [10, null], query: "<=15 runs" })).runs, [10, 15]);
  });

  test("words are looked for in the title, the family and the source", () => {
    assert.deepEqual(ids(results(CATALOG, state({ query: "nachtsheim" }))), ["dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "pilot plant" }))), ["pilot-plant"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "table 10.1" }))).sort(), ["dsd-categorical", "pilot-plant"]);
  });

  test("a family's count leaves out the family filter and keeps the others", () => {
    assert.deepEqual(familyCounts(CATALOG, state({ families: ["dsd"] })), { full_factorial: 1, dsd: 1, ccd: 1 });
    assert.deepEqual(familyCounts(CATALOG, state({ collection: "published" })), { full_factorial: 1, dsd: 1 });
  });

  test("a switch's count is what ticking it would leave", () => {
    assert.equal(switchCount(CATALOG, state(), "hasData"), 1);
    assert.equal(switchCount(CATALOG, state({ collection: "generated" }), "opens"), 1);
  });

  test("the orders", () => {
    assert.deepEqual(ids(sorted(CATALOG.designs, "runs")), ["pilot-plant", "dsd-categorical", "lib-ccd-k3"]);
    assert.deepEqual(ids(sorted(CATALOG.designs, "newest")), ["dsd-categorical", "pilot-plant", "lib-ccd-k3"]);
    assert.deepEqual(ids(sorted(CATALOG.designs, "title")), ["lib-ccd-k3", "dsd-categorical", "pilot-plant"]);
  });

  test("the catalogue's order lists designs with their runs ahead of comparisons and entries without runs", () => {
    const list = [
      design({ id: "comparison", family: "comparison", glyph: "fp" }),
      design({ id: "no-runs", glyph: "" }),
      design({ id: "factorial", glyph: "fp" }),
    ];
    assert.deepEqual(ids(sorted(list, "default")), ["factorial", "comparison", "no-runs"]);
  });

  test("the year bars count published designs only, in bins of five years", () => {
    const bins = yearBins(CATALOG, state());
    assert.deepEqual(bins[0], { from: 1975, to: 1979, count: 1 });
    assert.deepEqual(bins.at(-1), { from: 2010, to: 2014, count: 1 });
    assert.equal(bins.reduce((n, b) => n + b.count, 0), 2);
  });
});

describe("the address", () => {
  test("the default filters write nothing", () => {
    assert.equal(toHash(state()), "");
  });

  test("the filters come back as they were written", () => {
    const chosen = state({
      collection: "published",
      query: "5 factors",
      families: ["dsd", "ccd"],
      runs: [8, null],
      years: [1990, 2020],
      hasData: true,
      sort: "newest",
      design: "pilot-plant",
    });
    assert.deepEqual(fromHash(toHash(chosen), CATALOG.families), chosen);
  });

  test("anything the page did not write is ignored", () => {
    const read = fromHash("#col=all&fam=dsd,nothing&runs=x-9&model=cubic&added=yesterday&d=../etc&blk=yes", CATALOG.families);
    assert.equal(read.collection, "both");
    assert.deepEqual(read.families, ["dsd"]);
    assert.deepEqual(read.runs, [null, null]);
    assert.equal(read.model, "");
    assert.equal(read.addedSince, "");
    assert.equal(read.design, "");
    assert.equal(read.blocked, false);
  });
});

describe("the picture and the words", () => {
  test("a glyph reads back into coded settings, as the library wrote them", () => {
    assert.deepEqual(decodeGlyph("afkpu.", 3), [
      [-2, -1, 0],
      [1, 2, null],
    ]);
    assert.deepEqual(decodeGlyph("", 3), []);
  });

  test("settings are coloured as the book colours a run matrix", () => {
    assert.equal(settingColour(-1), "#0072b2");
    assert.equal(settingColour(0), "#ededed");
    assert.equal(settingColour(1), "#d55e00");
    assert.equal(settingColour(null), "transparent");
  });

  test("the fewest runs each model needs", () => {
    assert.deepEqual([3, 5].map((k) => minimumRuns(k, "main_effects")), [4, 6]);
    assert.deepEqual([3, 5].map((k) => minimumRuns(k, "interactions")), [7, 16]);
    assert.deepEqual([3, 5].map((k) => minimumRuns(k, "quadratic")), [10, 21]);
  });

  test("a design's facts, its checks and its source in a line each", () => {
    const pilot = CATALOG.designs[0];
    assert.equal(factLine(pilot, CATALOG), "Full factorial · 3 factors (1 categorical) · 8 runs · estimates two-factor interactions");
    assert.equal(checkLine(CATALOG.designs[1].checks), "8 of 9 recomputed numbers agree; 1 known to differ, with the reason");
    assert.equal(checkLine({ printed: 2, checked: 0, agree: 0, explained: 0 }), "2 printed numbers, none recomputed");
    assert.equal(sourceLine(pilot, CATALOG), "Box, Hunter and Hunter (1978), pp. 307-324; Table 10.1");
    assert.equal(sourceLine(CATALOG.designs[2], CATALOG), "Made by process-improve");
  });
});

describe("what a design is copied as", () => {
  test("a generated design is the library call that makes it", () => {
    const detail = {
      spec: { design_type: "ccd", k: 2, options: { n_center_points: 3, alpha: "face_centered" } },
      factors: [{ name: "A", low: -1, high: 1 }, { name: "B", low: -1, high: 1 }],
    };
    const code = pythonCode(CATALOG.designs[2], detail);
    assert.match(code, /from process_improve\.experiments import Factor, generate_design/);
    assert.match(code, /Factor\(name="B", low=-1, high=1\),/);
    assert.match(code, /generate_design\(factors, design_type="ccd", n_center_points=3, alpha="face_centered"\)/);
  });

  test("a published design is its entry in the reference catalogue", () => {
    const code = pythonCode(CATALOG.designs[0], { listed: "experiments" });
    assert.match(code, /from process_improve\.experiments\.reference_catalogue import reference_experiments/);
    assert.match(code, /reference_experiments\("bhh"\) if e\["name"\] == "pilot-plant"/);
  });

  test("the CSV holds the coded runs and the printed results", () => {
    const detail = { factors: [{ name: "T" }, { name: "K" }], coded_runs: [[-1, -1], [1, null]], response_values: { "y, %": [60, 72] } };
    assert.equal(csvText(detail), 'T (coded),K (coded),"y, %"\n-1,-1,60\n1,,72\n');
  });

  test("an index of another format is refused", () => {
    assert.throws(() => checkIndex({ version: 2, fields: INDEX_FIELDS, designs: [] }), /format 2/);
    assert.throws(() => checkIndex({ version: 1, fields: ["id"], designs: [] }), /fields/);
    assert.deepEqual(checkIndex({ version: 1, fields: INDEX_FIELDS, designs: [] }).designs, []);
  });
});
