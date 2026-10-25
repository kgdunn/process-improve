// web/reproduced-logic.js: the tiers a case and a number fall in, what the reproduction page counts,
// which cases a search keeps, the bands it counts differences in, and what its address keeps.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BANDS,
  DEFAULT_VIEW,
  TIER_KEYS,
  SOURCE_ORDERS,
  byForm,
  bySource,
  errorBands,
  filterCases,
  formatError,
  fromHash,
  groupByTier,
  kinds,
  percent,
  quantity,
  summarise,
  tierOf,
  toHash,
} from "../../../web/reproduced-logic.js";

const FORMS = { book: "Books", paper: "Journal papers", other: "Handbooks, encyclopedias and courses" };
const LIMITS = { reproduced: 0, within_1: 0.01, within_5: 0.05, not_reproduced: null };
const SOURCES = {
  lenth: { short: "Lenth", year: 1989, title: "Quick and easy analysis of unreplicated factorials", form: "paper" },
  davies: { short: "Davies", year: 1956, title: "The Design and Analysis of Industrial Experiments", form: "book" },
};

const number = (error, reason = "a misprint") => ({ what: "coefficient", term: "x1", printed: "1", found: "1.1", error, where: "p. 1", reason });
const CASES = [
  { id: "a", title: "Isatin yield", source: "lenth", kind: "example", checked: 20, tier: "reproduced", worst: 0, differs: [] },
  { id: "b", title: "Tensile strength", source: "lenth", kind: "example", checked: 50, tier: "not_reproduced", worst: null, differs: [number(null, "an alias chain misprinted"), number(0.03)] },
  { id: "c", title: "Penicillin yield", source: "davies", kind: "exercise", checked: 10, tier: "within_1", worst: 0.004, differs: [number(0.004, "the book's rounding")] },
  { id: "d", title: "Dyestuff", source: "davies", kind: "case-study", checked: 8, tier: "within_5", worst: 0.02, differs: [number(0.02)] },
  { id: "e", title: "A design no number of which is recomputed", source: "davies", kind: null, checked: 0, tier: null, worst: 0, differs: [] },
];

describe("the tiers", () => {
  test("a number that differs falls in the best tier whose limit holds it, and never in the first", () => {
    assert.equal(tierOf(0, LIMITS), "within_1", "a difference below the six figures reported still differs");
    assert.equal(tierOf(0.01, LIMITS), "within_1");
    assert.equal(tierOf(0.0101, LIMITS), "within_5");
    assert.equal(tierOf(0.05, LIMITS), "within_5");
    assert.equal(tierOf(0.2, LIMITS), "not_reproduced");
    assert.equal(tierOf(null, LIMITS), "not_reproduced", "a difference that is not a number");
  });

  test("the cases count once each, the numbers each in its own tier, and a case with no number apart", () => {
    const s = summarise(CASES, LIMITS);
    assert.deepEqual(s.cases, { reproduced: 1, within_1: 1, within_5: 1, not_reproduced: 1 });
    assert.deepEqual(s.numbers, { reproduced: 20 + 48 + 9 + 7, within_1: 1, within_5: 2, not_reproduced: 1 });
    assert.equal(s.numbersChecked, 88);
    assert.equal(s.recomputed, 4);
    assert.equal(s.unchecked, 1);
  });
});

describe("the words", () => {
  test("a share and a difference", () => {
    assert.equal(percent(1, 3), "33.3%");
    assert.equal(percent(3, 3), "100%");
    assert.equal(percent(0, 0), "0%");
    assert.equal(formatError(0.00046), "0.046%");
    assert.equal(formatError(0.0328), "3.3%");
    assert.equal(formatError(9.99), "999%");
    assert.equal(formatError(0), "< 0.0001%");
    assert.equal(formatError(null), "not a number");
    assert.equal(quantity({ what: "lack_of_fit_ss", term: null }), "lack of fit ss");
    assert.equal(quantity({ what: "coefficient", term: "x1" }), "coefficient: x1");
  });
});

describe("the search", () => {
  const ids = (list) => list.map((c) => c.id);

  test("keeps every case with a recomputed number until a filter is chosen", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES)), ["a", "b", "c", "d"]);
  });

  test("reads the title, the source's name and the reasons, every word of it", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "yield" })), ["a", "c"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "davies rounding" })), ["c"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "alias misprinted" })), ["b"]);
  });

  test("keeps a source and a kind, the catalogue's two spellings of a case study as one", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { source: "lenth" })), ["a", "b"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { kind: "case study" })), ["d"]);
    assert.deepEqual(kinds(CASES), [["example", 2], ["case study", 1], ["exercise", 1]]);
  });
});

describe("the sources and the lists", () => {
  test("each source counts its cases by tier and its numbers that agree, the oldest first", () => {
    const rows = bySource(CASES, SOURCES);
    assert.deepEqual(rows.map((r) => [r.key, r.form, r.total, r.tiers.reproduced]), [["davies", "book", 2, 0], ["lenth", "paper", 2, 1]]);
    assert.deepEqual([rows[1].numbers, rows[1].agree], [70, 68]);
    assert.deepEqual(bySource(CASES, SOURCES, "name").map((r) => r.key), ["davies", "lenth"]);
  });

  test("no order ranks the sources by how much of them reproduces", () => {
    assert.deepEqual(Object.keys(SOURCE_ORDERS), ["year", "name", "cases"]);
  });

  test("the sources group by the form of their source, each group with its totals, and all of them", () => {
    const { groups, totals } = byForm(bySource(CASES, SOURCES), FORMS);
    assert.deepEqual(groups.map((g) => [g.form, g.label, g.rows.map((r) => r.key)]), [["book", "Books", ["davies"]], ["paper", "Journal papers", ["lenth"]]]);
    assert.deepEqual(groups[1].totals, { total: 2, tiers: { reproduced: 1, within_1: 0, within_5: 0, not_reproduced: 1 }, numbers: 70, agree: 68 });
    assert.equal(totals.total, 4);
    assert.equal(totals.numbers, 88);
    assert.equal(totals.agree, 88 - 4);
  });

  test("a form chosen keeps the cases of its sources", () => {
    assert.deepEqual(filterCases(CASES, SOURCES, { form: "paper" }).map((c) => c.id), ["a", "b"]);
  });

  test("a tier's list groups its cases by source, the worst first", () => {
    const more = [...CASES, { ...CASES[2], id: "c2", worst: 0.009 }];
    assert.deepEqual(
      groupByTier(more, "within_1", ["lenth", "davies"]).map((g) => [g.source, g.cases.map((c) => c.id)]),
      [["davies", ["c2", "c"]]],
    );
  });

  test("the bands count each difference once, their edges at the tiers' limits", () => {
    const bands = errorBands(CASES, LIMITS);
    assert.equal(bands.length, BANDS.length);
    assert.equal(bands.reduce((sum, band) => sum + band.count, 0), 4);
    const at = (label) => bands.find((band) => band.label === label);
    assert.equal(at("0.3-1%").count, 1);
    assert.equal(at("0.3-1%").tier, "within_1");
    assert.equal(at("1-2%").tier, "within_5");
    assert.equal(at("1-2%").count, 1, "a band holds its upper edge");
    assert.equal(at("2-5%").count, 1);
    assert.equal(at("5-20%").tier, "not_reproduced");
    assert.equal(at("Not a number").count, 1);
  });
});

describe("the address", () => {
  test("keeps the view, leaving out what is at its default", () => {
    assert.equal(toHash(DEFAULT_VIEW), "");
    const view = { query: "misprint", source: "lenth", form: "paper", kind: "example", open: ["within_1", "not_reproduced"], focus: "b" };
    assert.equal(toHash(view), "#q=misprint&src=lenth&form=paper&kind=example&open=within_1%2Cnot_reproduced&case=b");
    assert.deepEqual(fromHash(toHash(view), SOURCES, FORMS), view);
  });

  test("ignores what it does not know", () => {
    const read = fromHash("#src=nobody&form=pamphlet&open=within_1,everything&case=<script>", SOURCES, FORMS);
    assert.deepEqual(read, { ...DEFAULT_VIEW, open: ["within_1"] });
    assert.deepEqual(TIER_KEYS, ["reproduced", "within_1", "within_5", "not_reproduced"]);
  });
});
