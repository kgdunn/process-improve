// web/catalog-logic.js: what the catalog page reads from a search, which designs pass its filters,
// what its address keeps, and how it draws and copies a design, apart from the page.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  COMPARE_MOST,
  DEFAULT_STATE,
  applicationLine,
  applications,
  caseFacts,
  cubeRuns,
  topicCounts,
  topicOrder,
  placesAfter,
  whereParts,
  LETTERS,
  limitInUnits,
  limitLine,
  limitOutcome,
  meets,
  parseLimit,
  checkIndex,
  compareRefusal,
  compareRows,
  correlationMap,
  errataCounts,
  errataShown,
  generatorsLine,
  familyMarks,
  largestSecondOrderR,
  markInSource,
  nextSteps,
  projected,
  projectionSummary,
  projectionText,
  rank,
  sharedModel,
  shownValue,
  toggled,
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
  labelOf,
  leverageWords,
  levelsText,
  manyLabels,
  projectionLines,
  pythonCode,
  results,
  runLossLine,
  sameDesigns,
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
  survives_run_loss: true,
  critical_runs: 0,
  worst_d_ratio: 0.7430,
  quadratic_projection: 0,
  projectivity: 3,
  design_class: "cube",
  topics: [],
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
  topics: {
    "food-beverages": { label: "Food and beverages", description: "Cooking, baking, fermentation of food." },
    "chemical-processes": { label: "Chemical reactions and processes", description: "Reactors, yield, conversion." },
    general: { label: "General (no stated application)", description: "Abstract factors (A, B, x1, ...)." },
  },
  designs: [
    design({
      id: "pilot-plant",
      title: "Pilot plant",
      has_responses: true,
      n_responses: 1,
      n_categorical: 1,
      topics: ["chemical-processes"],
    }),
    design({
      id: "dsd-categorical",
      title: "DSD with a categorical factor",
      topics: ["general"],
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
      survives_run_loss: false,
      critical_runs: 14,
      worst_d_ratio: 0,
      quadratic_projection: null,
      projectivity: null,
      design_class: "dsd-augmented",
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
      quadratic_projection: 3,
      design_class: "ccd",
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

  test("reads what a design withstands: a lost run, its projections, its projectivity", () => {
    const asked = parseQuery("3-factor projection, survives a lost run, projectivity 4");
    assert.equal(asked.projection, 3);
    assert.equal(asked.survives, true);
    assert.equal(asked.projectivity, 4);
    assert.equal(asked.factors, undefined, "a 3-factor projection names no number of factors");
    assert.equal(parseQuery("projects onto any 2").projection, 2);
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

  test("what a design withstands", () => {
    assert.deepEqual(ids(results(CATALOG, state({ survives: true }))), ["pilot-plant", "lib-ccd-k3"]);
    assert.deepEqual(ids(results(CATALOG, state({ projection: 3 }))), ["lib-ccd-k3"]);
    assert.deepEqual(ids(results(CATALOG, state({ projectivity: 3 }))), ["pilot-plant", "lib-ccd-k3"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "survives a lost run, 3-factor projection" }))), ["lib-ccd-k3"]);
    assert.deepEqual(filtersOf(state({ projection: 2, query: "4-factor projection" })).projection, 4);
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
    assert.deepEqual(bins[0], { from: 1975, to: 1979, count: 1, families: { full_factorial: 1 } });
    assert.deepEqual(bins.at(-1), { from: 2010, to: 2014, count: 1, families: { dsd: 1 } });
    assert.equal(bins.reduce((n, b) => n + b.count, 0), 2);
  });
});

describe("the applications", () => {
  test("an application keeps the designs that list it, and none that the library generates", () => {
    assert.deepEqual(ids(results(CATALOG, state({ topics: ["chemical-processes"] }))), ["pilot-plant"]);
    assert.deepEqual(ids(results(CATALOG, state({ topics: ["chemical-processes", "general"] }))), ["pilot-plant", "dsd-categorical"]);
    assert.deepEqual(ids(results(CATALOG, state({ topics: ["general"], collection: "generated" }))), []);
    assert.deepEqual(ids(results(CATALOG, state({ topics: [] }))), ["pilot-plant", "dsd-categorical", "lib-ccd-k3"]);
  });

  test("an application's count leaves out the application filter and keeps the others", () => {
    assert.deepEqual(topicCounts(CATALOG, state({ topics: ["general"] })), { "chemical-processes": 1, general: 1 });
    assert.deepEqual(topicCounts(CATALOG, state({ hasData: true })), { "chemical-processes": 1 });
    assert.deepEqual(topicCounts(CATALOG, state({ collection: "generated" })), {});
  });

  test("the filter lists the applications some design has, the most designs first, and general last", () => {
    const catalog = {
      ...CATALOG,
      designs: [
        design({ id: "a", topics: ["general"] }),
        design({ id: "b", topics: ["general"] }),
        design({ id: "c", topics: ["food-beverages"] }),
        design({ id: "d", topics: ["chemical-processes", "food-beverages"] }),
      ],
    };
    assert.deepEqual(topicOrder(catalog), ["food-beverages", "chemical-processes", "general"]);
    assert.deepEqual(topicOrder({ ...catalog, designs: [design({ topics: [] })] }), []);
  });

  test("a word of an application's name finds its designs; general is no word to find", () => {
    assert.deepEqual(ids(results(CATALOG, state({ query: "chemical" }))), ["pilot-plant"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "chemical reactions" }))), ["pilot-plant"]);
    assert.deepEqual(ids(results(CATALOG, state({ query: "food" }))), []);
    assert.deepEqual(ids(results(CATALOG, state({ query: "stated application" }))), []);
  });

  test("a design's applications in words, without general", () => {
    assert.deepEqual(applications(CATALOG.designs[0], CATALOG), ["Chemical reactions and processes"]);
    assert.equal(applicationLine(CATALOG.designs[0], CATALOG), "Application: Chemical reactions and processes.");
    const two = design({ topics: ["food-beverages", "chemical-processes"] });
    assert.equal(applicationLine(two, CATALOG), "Applications: Food and beverages; Chemical reactions and processes.");
    assert.equal(applicationLine(CATALOG.designs[1], CATALOG), "", "a source that states no application");
    assert.equal(applicationLine(CATALOG.designs[2], CATALOG), "", "a generated design");
  });
});

describe("where a design is printed", () => {
  test("its section, pages, tables and figures, each a part of its own", () => {
    const where = { section: "10.2", pages: "307-324", tables: ["Table 10.1"], figures: ["Cube plot", "Pareto plot, sorted"] };
    assert.deepEqual(whereParts(where), ["section 10.2", "pages 307-324", "Table 10.1", "Cube plot", "Pareto plot, sorted"]);
    assert.deepEqual(whereParts({ pages: 73 }), ["page 73"]);
    assert.deepEqual(whereParts("Table 4"), ["Table 4"]);
    assert.deepEqual(whereParts(null), []);
  });

  test("after its first part, the line names four places and counts the rest", () => {
    const rest = ["T1", "T2", "F1", "F2", "F3", "F4"];
    assert.equal(placesAfter(rest), "; T1; T2; F1; F2; and 2 more tables and figures.");
    assert.equal(placesAfter(rest.slice(0, 5)), "; T1; T2; F1; F2; and 1 more table or figure.");
    assert.equal(placesAfter(rest.slice(0, 2)), "; T1; T2.");
    assert.equal(placesAfter([]), ".");
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
      survives: true,
      projection: 3,
      projectivity: 2,
      sort: "newest",
      design: "pilot-plant",
      source: "jn",
      unprinted: true,
      compare: ["lib-ccd-k3", "pilot-plant"],
      view: "compare",
      limit: "A + B <= 1",
      limitAll: true,
      topics: ["food-beverages", "general"],
    });
    assert.deepEqual(fromHash(toHash(chosen), CATALOG.families, CATALOG.sources, CATALOG.topics), chosen);
  });

  test("the applications chosen are kept under app, and one the catalog does not have is ignored", () => {
    assert.equal(toHash(state({ topics: ["food-beverages", "general"] })), "#app=food-beverages%2Cgeneral");
    const read = fromHash("#app=food-beverages,cooking,general,food-beverages", CATALOG.families, CATALOG.sources, CATALOG.topics);
    assert.deepEqual(read.topics, ["food-beverages", "general"]);
    assert.deepEqual(fromHash("#app=food-beverages").topics, [], "no application without the catalog's list");
  });

  test("anything the page did not write is ignored", () => {
    const read = fromHash("#col=all&fam=dsd,nothing&runs=x-9&model=cubic&added=yesterday&d=../etc&blk=yes&qp=12&proj=x", CATALOG.families);
    assert.equal(read.collection, "both");
    assert.deepEqual(read.families, ["dsd"]);
    assert.deepEqual(read.runs, [null, null]);
    assert.equal(read.model, "");
    assert.equal(read.addedSince, "");
    assert.equal(read.design, "");
    assert.equal(read.blocked, false);
    assert.equal(read.projection, 0);
    assert.equal(read.projectivity, 0);
    const more = fromHash("#src=nobody&cmp=a,../b,a,c,d,e,f&view=everything", CATALOG.families, CATALOG.sources);
    assert.equal(more.source, "");
    assert.deepEqual(more.compare, ["a", "c", "d", "e"], "valid ids, once each, four at most");
    assert.equal(more.view, "");
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
    assert.deepEqual(caseFacts(pilot), ["3 factors (1 categorical)", "8 runs", "Estimates two-factor interactions"], "its own, not its family's");
    assert.equal(checkLine(CATALOG.designs[1].checks), "8 of 9 recomputed numbers agree; 1 known to differ, with the reason");
    assert.equal(checkLine({ printed: 2, checked: 0, agree: 0, explained: 0 }), "2 printed numbers, none recomputed");
    assert.equal(sourceLine(pilot, CATALOG), "Box, Hunter and Hunter (1978), pp. 307-324; Table 10.1");
    assert.equal(sourceLine(CATALOG.designs[2], CATALOG), "Made by process-improve");
  });
});

describe("what a design withstands, in words", () => {
  test("losing a run", () => {
    const pilot = CATALOG.designs[0];
    assert.equal(
      runLossLine(pilot, "interactions"),
      "If any one run is lost, the design still estimates two-factor interactions; the worst loss leaves 74% of its D-efficiency.",
    );
    assert.equal(
      runLossLine(CATALOG.designs[1], "main_effects"),
      "None of its 14 runs can be lost: without any one of them the design no longer estimates main effects.",
    );
    assert.equal(
      runLossLine(design({ survives_run_loss: false, critical_runs: 1, n_runs: 9 }), "main_effects"),
      "1 of its 9 runs cannot be lost: without it the design no longer estimates main effects.",
    );
    assert.equal(runLossLine(design({ survives_run_loss: null }), "main_effects"), "");
  });

  test("projections", () => {
    assert.deepEqual(projectionLines(design({ quadratic_projection: 3, n_numeric: 6, projectivity: null })), [
      "Any 3 of its factors: the runs estimate the full quadratic in them, so if only 3 matter, their response surface needs no more runs.",
    ]);
    assert.deepEqual(projectionLines(design({ quadratic_projection: 0, projectivity: 3, n_factors: 3 })), [
      "Its two-level runs hold every combination of its 3 factors' levels.",
    ]);
    assert.deepEqual(projectionLines(design({ quadratic_projection: 0, projectivity: 3, n_factors: 11 })), [
      "Projectivity 3: any 3 of its factors hold all 8 combinations of their two levels.",
    ]);
  });

  test("projectivity says how many runs it is about when only some are two-level runs", () => {
    const ccd = design({ quadratic_projection: 0, projectivity: 3, n_factors: 3 });
    assert.equal(cubeRuns(CCD), 8);
    assert.deepEqual(projectionLines(ccd, CCD), ["Its 8 two-level runs, of 17, hold every combination of its 3 factors' levels."]);
    assert.deepEqual(projectionLines(ccd, CUBE), ["Its two-level runs hold every combination of its 3 factors' levels."]);
    const wide = design({ quadratic_projection: 0, projectivity: 2, n_factors: 3 });
    assert.deepEqual(projectionLines(wide, CCD), [
      "Projectivity 2: in its 8 two-level runs, of 17, any 2 of its factors hold all 4 combinations of their two levels.",
    ]);
  });

  test("a categorical factor of more than two labels reads back its label from its coded setting", () => {
    const part = { name: "part", type: "categorical", levels: Array.from({ length: 20 }, (_, i) => String(i + 1)) };
    assert.equal(manyLabels(part), true);
    assert.equal(manyLabels({ type: "categorical", levels: ["A", "B"] }), false, "two labels are a factor's two levels");
    assert.equal(manyLabels({ type: "numeric", levels: [1, 2, 3] }), false);
    assert.equal(labelOf(part, -1), "1");
    assert.equal(labelOf(part, -1 + (2 * 2) / 19), "3", "the third label of twenty");
    assert.equal(labelOf(part, Number((-0.789).toFixed(3))), "3", "as the catalog rounds it");
    assert.equal(labelOf(part, 1), "20");
    assert.equal(labelOf(part, 0), null, "between two labels");
    assert.equal(labelOf(part, null), null);
  });

  test("a factor's levels in a few words", () => {
    const labels = Array.from({ length: 20 }, (_, i) => String(i + 1));
    assert.equal(levelsText({ type: "categorical", low: "1", high: "2", levels: labels }), "1 to 20 (20 labels)");
    assert.equal(levelsText({ type: "categorical", levels: ["A", "B"] }), "A, B");
    assert.equal(levelsText({ type: "categorical", low: "old", high: "new" }), "old, new");
    assert.equal(levelsText({ type: "numeric", low: 150, high: 200 }), "150 to 200");
    assert.equal(levelsText({ type: "numeric" }), "");
  });

  test("leverage in words: one sentence when every run has the same, else the runs that carry the most", () => {
    const even = leverageWords(Array(120).fill(0.5));
    assert.equal(even.even, true);
    assert.equal(
      even.text,
      "Every run has the same leverage, 0.50: the model's 60 coefficients are spread evenly over its 120 runs, so it leans on no run more than another.",
    );
    assert.deepEqual(leverageWords([0.95, 0.95, 0.95, 0.95, 0.2]), {
      even: false,
      text: "Runs 1, 2, 3 and 4 carry the most weight, with leverage 0.95; the least is 0.20.",
    });
    assert.equal(leverageWords([0.3, 1, 0.3]).text, "Run 2 carries the most weight, with leverage 1.00; the least is 0.30.");
    assert.match(leverageWords([...Array(6).fill(0.9), 0.1]).text, /^6 runs carry the most weight/);
    assert.deepEqual(leverageWords(null), { even: false, text: "" });
  });

  test("the same design elsewhere", () => {
    const twin = design({ id: "twin", design_class: "cube" });
    const catalog = { ...CATALOG, designs: [...CATALOG.designs, twin] };
    assert.deepEqual(ids(sameDesigns(catalog, catalog.designs[0])), ["twin"]);
    assert.deepEqual(sameDesigns(catalog, design({ design_class: null })), []);
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
    assert.match(code, /design = generate_design\(\n {4}factors,\n {4}design_type="ccd",\n {4}n_center_points=3,\n {4}alpha="face_centered",\n\)/);
  });

  test("a published design is its entry in the reference catalogue", () => {
    const code = pythonCode(CATALOG.designs[0], { listed: "experiments", factors: [{ name: "T" }], coded_runs: [[-1]] });
    assert.match(code, /from process_improve\.experiments\.reference_catalogue import \(\n {4}reference_experiments,\n\)/);
    assert.match(code, /entries = reference_experiments\("bhh"\)\nby_name = \{item\["name"\]: item for item in entries\}\nentry = by_name\["pilot-plant"\]/);
    assert.match(code, /factors = pd\.DataFrame\(entry\["factors"\], columns=columns\)/);
    assert.match(code, /table = pd\.DataFrame\(runs\["rows"\], columns=runs\["columns"\]\)/);
  });

  test("a published design with no printed runs prints what the source says of it, and needs no pandas", () => {
    const code = pythonCode(CATALOG.designs[0], { listed: "designs", factors: [], coded_runs: null, response_values: {} });
    assert.match(code, /\n {4}reference_designs,\n/);
    assert.doesNotMatch(code, /pandas|runs\["rows"\]/);
    assert.match(code, /print\(entry\["design"\]\)$/);
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

// ---------------------------------------------------------------- phase 3: the views over the list
const CUBE = [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]];
const HALF = [[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]];
const FACES = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
const CCD = [...CUBE, ...FACES, [0, 0, 0], [0, 0, 0], [0, 0, 0]];

describe("which designs no source prints", () => {
  test("a generated design of a published design's class is printed; one of no such class is a gap", () => {
    const designs = markInSource([
      design({ id: "printed", design_class: "cube" }),
      design({ id: "made-too", collection: "generated", design_class: "cube" }),
      design({ id: "only-made", collection: "generated", design_class: "omars" }),
      design({ id: "no-class", collection: "generated", design_class: null }),
    ]);
    assert.deepEqual(designs.map((d) => d.in_source), [true, true, false, false]);
    const catalog = { ...CATALOG, designs };
    assert.deepEqual(ids(results(catalog, state({ unprinted: true }))), ["only-made", "no-class"]);
    assert.deepEqual(ids(results(catalog, state({ query: "gaps" }))), ["only-made", "no-class"]);
  });

  test("a source keeps its own designs", () => {
    assert.deepEqual(ids(results(CATALOG, state({ source: "jn" }))), ["dsd-categorical"]);
  });
});

describe("designs side by side", () => {
  const catalog = {
    ...CATALOG,
    designs: [
      design({ id: "a", compared: true }),
      design({ id: "b", compared: true }),
      design({ id: "c", compared: true, n_factors: 4 }),
      design({ id: "d", compared: false }),
    ],
  };

  test("a design joins when it is scored, has the others' factors, and there is room", () => {
    assert.equal(compareRefusal(catalog, [], "a"), "");
    assert.equal(compareRefusal(catalog, ["a"], "b"), "");
    assert.match(compareRefusal(catalog, ["a"], "c"), /have 3 factors, and this one has 4/);
    assert.match(compareRefusal(catalog, [], "d"), /all numeric/);
    assert.match(compareRefusal(catalog, ["a", "b", "a", "b"], "b"), new RegExp(`${COMPARE_MOST} designs at most`));
    assert.deepEqual(toggled(["a", "b"], "a"), ["b"]);
    assert.deepEqual(toggled(["b"], "a"), ["b", "a"]);
  });

  test("the comparison opens on the largest model every design fits", () => {
    const fits = (m, i, q) => ({ main_effects: { fits: m }, interactions: { fits: i }, quadratic: { fits: q } });
    assert.equal(sharedModel([fits(true, true, true), fits(true, true, false)]), "interactions");
    assert.equal(sharedModel([fits(true, true, true)]), "quadratic");
    assert.equal(sharedModel([fits(false, false, false)]), "main_effects");
  });

  test("values are printed as compare_designs prints its table", () => {
    const values = { runs: 17, fits: true, residual_df: 7, d: 7.02, e: 2, i: 0.3234, power_main: null, rank: 10 };
    assert.equal(shownValue("runs", values), "17");
    assert.equal(shownValue("fits", values), "yes");
    assert.equal(shownValue("fits", { ...values, fits: false, rank: 7 }), "no (rank 7)");
    assert.equal(shownValue("d", values), "7.020");
    assert.equal(shownValue("e", values), "2.000");
    assert.equal(shownValue("i", values), "0.3234");
    assert.equal(shownValue("power_main", values), "");
    const measures = [{ key: "runs", label: "Runs", direction: "" }, { key: "i", label: "I", direction: "lower" }];
    const rows = compareRows(measures, [{ quadratic: values }, { quadratic: { runs: 9, i: 0.5 } }], "quadratic");
    assert.deepEqual(rows.map((r) => [r.label, r.arrow, r.cells]), [["Runs", "", ["17", "9"]], ["I", "↓", ["0.3234", "0.5000"]]]);
  });
});

describe("a design's projections", () => {
  test("the rank of a matrix", () => {
    assert.equal(rank([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3);
    assert.equal(rank([[1, 2], [2, 4], [3, 6]]), 1);
    assert.equal(rank([]), 0);
  });

  test("runs that land on one setting are counted once, with their number", () => {
    const points = projected(CCD, [0, 1]);
    assert.equal(points.length, 9, "a 3-by-3 grid");
    assert.equal(points.find((p) => p.point[0] === 0 && p.point[1] === 0).count, 5, "two face runs and three centre runs");
  });

  test("a central composite design fits the quadratic in any two factors; a two-level design does not", () => {
    const ccd = projectionSummary(CCD, [0, 2]);
    assert.deepEqual([ccd.distinct, ccd.terms, ccd.quadratic, ccd.allCorners], [9, 6, true, true]);
    const cube = projectionSummary(CUBE, [1, 2]);
    assert.deepEqual([cube.distinct, cube.rank, cube.quadratic], [4, 4, false]);
    assert.equal(
      projectionText(cube, ["B", "C"]),
      "In B and C, the runs sit at 4 distinct settings and do not estimate the full quadratic in them: its 6 terms have rank 4. They hold all 4 corners: a full 2^2 factorial in these factors. Every point holds 2 runs.",
    );
    assert.match(projectionText(projectionSummary(CCD, [0, 1, 2]), ["A", "B", "C"]), /15 distinct settings and estimate the full quadratic in them \(10 terms\)/);
  });

  test("the runs on each point: one sentence when every point holds the same, else what a label counts", () => {
    const flat = projectionSummary(CCD, [0, 1]);
    assert.deepEqual([flat.even, flat.most, flat.single], [null, 5, true]);
    assert.match(projectionText(flat, ["A", "B"]), / A label such as 5× counts the runs at a point; a point with no label holds one run\.$/);
    const twice = projectionSummary([...CUBE, ...CUBE], [0, 1, 2]);
    assert.deepEqual([twice.even, twice.single], [2, false]);
    assert.match(projectionText(twice, ["A", "B", "C"]), / Every point holds 2 runs\.$/);
    const once = projectionText(projectionSummary(CUBE, [0, 1, 2]), ["A", "B", "C"]);
    assert.doesNotMatch(once, /holds|label/, "one run on every point needs no word on it");
  });
});

describe("a design's correlations", () => {
  test("a half fraction aliases each main effect with an interaction", () => {
    const map = correlationMap(HALF, ["A", "B", "C"]);
    const at = (a, b) => map.matrix[map.terms.indexOf(a)][map.terms.indexOf(b)];
    assert.deepEqual(map.terms, ["A", "B", "C", "I(A ** 2)", "I(B ** 2)", "I(C ** 2)", "A:B", "A:C", "B:C"]);
    assert.equal(at("A", "B:C"), 1);
    assert.equal(at("A", "B"), 0);
    assert.equal(at("I(A ** 2)", "A"), null, "the squares of a two-level design do not vary");
    assert.equal(largestSecondOrderR(map), 0, "AB is C and AC is B: aliased with main effects, not with each other");
  });

  test("a full factorial's terms are uncorrelated; the face-centred CCD's are as compare_designs finds", () => {
    assert.equal(largestSecondOrderR(correlationMap(CUBE, ["A", "B", "C"])), 0);
    const ccd = correlationMap(CCD, ["A", "B", "C"]);
    assert.ok(ccd.matrix.flat().every((r) => r !== null), "every term varies in a CCD");
    assert.ok(Math.abs(largestSecondOrderR(ccd) - 0.5143) < 1e-4, "max |r| 0.5143, between two squares");
  });
});

describe("the errata", () => {
  const errata = [
    { id: "a", source: "bhh", what: "variance", printed: "8.29", where: "p. 590", reason: "the square of a rounded value", rounding: true },
    { id: "b", source: "bhh", what: "effect", printed: "-3.1", where: "p. 310", reason: "a misprint: the sign", rounding: false },
    { id: "c", source: "jn", what: "D-efficiency", printed: "0.9", where: "p. 4", reason: "another method", rounding: false },
  ];

  test("by source, by reason and by words", () => {
    assert.deepEqual(errataShown(errata).map((e) => e.id), ["a", "b", "c"]);
    assert.deepEqual(errataShown(errata, { source: "bhh" }).map((e) => e.id), ["a", "b"]);
    assert.deepEqual(errataShown(errata, { rounding: false }).map((e) => e.id), ["b", "c"]);
    assert.deepEqual(errataShown(errata, { words: "Misprint 310" }).map((e) => e.id), ["b"]);
    assert.deepEqual(errataCounts(errata), { bhh: 2, jn: 1 });
  });
});

describe("the timeline and where a design leads", () => {
  test("the families first published within the years the bars span are marked", () => {
    const catalog = { ...CATALOG, families: { ...CATALOG.families, full_factorial: { label: "Full factorial", year: 1926 }, dsd: { label: "Definitive screening", year: 2011 } } };
    const marks = familyMarks(catalog, yearBins(catalog, state()));
    assert.deepEqual(marks, [{ family: "dsd", year: 2011, label: "Definitive screening" }]);
    assert.deepEqual(familyMarks(catalog, []), []);
  });

  test("a step leads to the family's designs of at most as many factors, or at least as many for more factors", () => {
    const catalog = {
      ...CATALOG,
      families: {
        ...CATALOG.families,
        full_factorial: {
          label: "Full factorial",
          next: [{ family: "ccd", how: "Add center + axial points" }, { family: "fractional_factorial", how: "Use a fraction for more factors" }],
        },
      },
    };
    const steps = nextSteps(CATALOG.designs[0], catalog);
    assert.deepEqual(steps.map((s) => [s.label, s.hash]), [
      ["Central composite", "#fam=ccd&k=-3"],
      ["fractional_factorial", "#fam=fractional_factorial&k=3-"],
    ]);
    assert.deepEqual(nextSteps(CATALOG.designs[1], CATALOG), []);
  });
});

// ---------------------------------------------------------------- phase 4: limits on the coded settings
describe("a limit on the coded settings", () => {
  test("factors by their letters, numbers, + - * / ^, brackets and one or more signs", () => {
    for (const text of ["A + B <= 1", "-0.5 <= A - C <= 0.5", "2A + B^2 >= -1", "A*B < 0.5", "A ≤ 0.5", "(A + B) / 2 > -1", "A**2 <= 1"]) {
      assert.equal(parseLimit(text).ok, true, text);
    }
    assert.deepEqual(parseLimit("-0.5 <= A - C <= 0.5").factors, [0, 2]);
    assert.equal(LETTERS.slice(7, 10), "HJK", "no I, as the library names factors");
  });

  test("a limit that cannot be read says why", () => {
    assert.match(parseLimit("A + B").error, /write <= for at most/);
    assert.match(parseLimit("(A + B <= 1").error, /a bracket is not closed/);
    assert.match(parseLimit("A + I <= 1").error, /I names no factor/);
    assert.match(parseLimit("A <= 1 1").error, /is not where it can be read/);
    assert.match(parseLimit("2 <= 3").error, /names no factor/);
    assert.match(parseLimit("A <= 1; alert(1)").error, /cannot be read from/);
    assert.equal(parseLimit("").error, "");
  });

  test("a run meets a limit, within a hair of its boundary", () => {
    const limit = parseLimit("-0.5 <= A - C <= 0.5");
    assert.equal(meets(limit, [0.5, 9, 0]), true);
    assert.equal(meets(limit, [0.5 + 1e-12, 9, 0]), true);
    assert.equal(meets(limit, [1, 0, 0]), false);
    assert.equal(meets(parseLimit("A / B <= 1"), [1, 0]), false, "a division by zero meets no limit");
  });

  test("a design is judged on its numeric factors with every setting given", () => {
    const limit = parseLimit("A + B <= 1");
    const runs = [[1, 1, 0], [1, 0, 0], [-1, 1, 0], [0.5, 0.5, 0]];
    assert.deepEqual(limitOutcome(limit, runs, ["numeric", "numeric", "numeric"]), { judged: true, outside: [0] });
    assert.match(limitOutcome(parseLimit("D <= 0"), runs, ["numeric", "numeric", "numeric"]).why, /names D, and the design has 3 factors/);
    assert.match(limitOutcome(limit, runs, ["numeric", "categorical", "numeric"]).why, /B is a categorical factor/);
    assert.match(limitOutcome(limit, [[null, 1, 0]], ["numeric", "numeric", "numeric"]).why, /not printed/);
    assert.match(limitOutcome(limit, [], ["numeric"]).why, /no table of its runs/);
  });

  test("its outcome, in a sentence", () => {
    const limit = parseLimit("A + B <= 1");
    assert.equal(limitLine(limit, { judged: true, outside: [] }, 8), "Every one of its 8 runs meets the limit A + B <= 1.");
    assert.equal(limitLine(limit, { judged: true, outside: [3, 7] }, 8), "Runs 4 and 8 are outside the limit A + B <= 1, of its 8 runs.");
    assert.equal(limitLine(limit, { judged: false, why: "no runs" }, 8), "The limit A + B <= 1 cannot be judged on this design: no runs.");
  });

  test("it is written in a design's factors and their own units, for Factorial", () => {
    const factors = [{ name: "T", low: 150, high: 200 }, { name: "P", low: 1, high: 3 }, { name: "F", low: -2, high: 2 }];
    assert.equal(limitInUnits(parseLimit("A + B <= 1"), factors), "((T - 175) / 25) + (P - 2) <= 1");
    assert.equal(limitInUnits(parseLimit("-0.5 <= B - 2*C <= 0.5"), factors), "-0.5 <= (P - 2) - (2 * (F / 2)) <= 0.5");
    assert.equal(limitInUnits(parseLimit("A^2 <= 1"), factors), "((T - 175) / 25) ** 2 <= 1");
  });

  test("the list keeps the designs whose every run meets it, or every design it can judge", () => {
    const catalog = {
      ...CATALOG,
      designs: [design({ id: "in", limit_outside: 0 }), design({ id: "out", limit_outside: 2 }), design({ id: "unjudged", limit_outside: null })],
    };
    assert.deepEqual(ids(results(catalog, state({ limit: "A + B <= 1" }))), ["in"]);
    assert.deepEqual(ids(results(catalog, state({ limit: "A + B <= 1", limitAll: true }))), ["in", "out"]);
    assert.deepEqual(ids(results(catalog, state({ limit: "A + B" }))), ["in", "out", "unjudged"], "a limit that cannot be read keeps every design");
  });
});

describe("a fraction named by its generators", () => {
  test("its generators are spaced and listed as the source writes them", () => {
    assert.equal(generatorsLine(["C=AB"]), "C = AB");
    assert.equal(generatorsLine(["D=AB", "E=AC"]), "D = AB and E = AC");
    assert.equal(generatorsLine(["5=234", "6 = 1234", "F=-ABCDE"]), "5 = 234, 6 = 1234 and F = -ABCDE");
  });
});
