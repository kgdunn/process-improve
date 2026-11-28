// web/reproduced-logic.js: the states a printed number and a case are in and the reasons a number is no match,
// what the reproduction page counts, the tiers a number that differs falls in, which cases a search keeps, the
// plain names of the quantities, and what its address keeps.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  CASE_STATES,
  CAUSES,
  DEFAULT_VIEW,
  REASONS,
  STATES,
  TIER_KEYS,
  SOURCE_ORDERS,
  byForm,
  bySource,
  byState,
  caseState,
  cascade,
  examples,
  missingByReason,
  missingKey,
  filterCases,
  formatError,
  fromHash,
  groupByState,
  kinds,
  percent,
  quantity,
  quantityName,
  sizes,
  tally,
  tallyWords,
  tierOf,
  toHash,
} from "../../../web/reproduced-logic.js";

const FORMS = { book: "Books", paper: "Journal papers", other: "Handbooks, encyclopedias and courses" };
const LIMITS = { reproduced: 0, within_1: 0.01, within_5: 0.05, not_reproduced: null };
const SOURCES = {
  lenth: { short: "Lenth", year: 1989, title: "Quick and easy analysis of unreplicated factorials", form: "paper" },
  davies: { short: "Davies", year: 1956, title: "The Design and Analysis of Industrial Experiments", form: "book" },
};

const number = (error, reason = "a misprint", cause = "method") => ({ what: "coefficient", term: "x1", printed: "1", found: "1.1", error, where: "p. 1", reason, cause });
const RIGHT = ["incorrect_option", "source_error"];
/** Plain names of quantities, as the JSON's ``quantities`` holds them (from web/quantities.json). */
const NAMES = { coefficient: "a coefficient", anova_sum_sq: "a sum of squares", f_regression: "an F ratio" };
const CASES = [
  { id: "a", title: "Isatin yield", source: "lenth", kind: "example", checked: 20, tier: "reproduced", worst: 0, differs: [] },
  { id: "b", title: "Tensile strength", source: "lenth", kind: "example", checked: 50, tier: "not_reproduced", worst: null, differs: [number(null, "an alias chain misprinted"), number(0.03)] },
  { id: "c", title: "Penicillin yield", source: "davies", kind: "exercise", checked: 10, tier: "within_1", worst: 0.004, differs: [number(0.004, "the book's rounding")] },
  { id: "d", title: "Dyestuff", source: "davies", kind: "case-study", checked: 8, tier: "within_5", worst: 0.02, differs: [number(0.02)] },
  { id: "e", title: "A design no number of which is recomputed", source: "davies", kind: null, checked: 0, tier: null, worst: 0, differs: [], unchecked: { library_gap: 2 }, missing: [{ what: "estimate", term: null, printed: "4", where: "p. 9", reason: "the library has no IRLS fit" }, { what: "estimate", term: null, printed: "5", where: "p. 9", reason: "the library has no IRLS fit" }] },
  { id: "f", title: "Misprinted pilot plant", source: "davies", kind: "exercise", checked: 12, tier: "reproduced", worst: 0, differs: [number(0.4, "a misprint in the book", "source_error")], unchecked: { library_gap: 1, not_a_result: 3 }, missing: [{ what: "radius", term: null, printed: "2", where: "p. 3", reason: "the library has no IRLS fit" }] },
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

  test("the sizes of the differences leave out those the library is right about", () => {
    assert.deepEqual(sizes(CASES, LIMITS, RIGHT), { within_1: 1, within_5: 2, not_reproduced: 1 });
    assert.equal(sizes(CASES, LIMITS, []).not_reproduced, 2, "the misprint, held to its size");
  });
});

describe("the states and the reasons", () => {
  test("the numbers count once each: those that match, and each of the others under its cause", () => {
    const counts = tally(CASES);
    assert.equal(counts.checked, 100);
    assert.equal(counts.match, 95);
    assert.deepEqual(counts.causes, { source_rounding: 0, source_error: 1, method: 4, library: 0, incorrect_option: 0, unexplained: 0, library_gap: 3 });
    assert.equal(tally([CASES[4]]).checked, 0, "a number not in the library yet is no part of those recomputed");
  });

  test("a reason's numbers are under its state, the quiz and what the library lacks apart", () => {
    assert.deepEqual(byState(tally(CASES)), { match: 95, known: 5, quiz: 0, open: 0, gap: 3 });
    assert.deepEqual(STATES.map((s) => s.key), ["match", "known", "quiz", "open", "gap"]);
    assert.deepEqual(REASONS.map((r) => r.state), ["known", "known", "known", "known", "quiz", "open", "gap"]);
  });

  test("the headline says the page explains each number that differs, but those it cannot yet", () => {
    const said = "printed numbers match. And where they don't, we can explain why";
    assert.equal(tallyWords(byState(tally(CASES))), `${said}. See below.`, "every reason is known");
    assert.equal(tallyWords({ match: 34765, known: 2210, quiz: 92, open: 275, gap: 1549 }), `${said}, for all but 275. See below.`);
    assert.equal(tallyWords({ match: 9, known: 0, quiz: 0, open: 1275, gap: 0 }), `${said}, for all but 1,275. See below.`);
  });

  test("a case is in the state of its number that comes last in the bar's order", () => {
    const of = (...causes) => caseState({ differs: causes.map((cause) => number(0.1, "a reason", cause)) });
    assert.equal(of(), "match", "no number differs");
    assert.equal(of("source_error", "method"), "known");
    assert.equal(of("source_error", "incorrect_option"), "quiz");
    assert.equal(of("incorrect_option", "unexplained", "source_rounding"), "open", "a reason not known yet is never hidden");
    assert.equal(of("a cause the page does not know"), "open");
    assert.equal(caseState(CASES[5]), "known", "a misprint differs, though the library is right about it");
    assert.equal(caseState(CASES[4]), "match", "numbers not in the library yet leave a case's state as it is");
    assert.deepEqual(CASE_STATES.map((s) => [s.key, s.label]), STATES.slice(0, 4).map((s) => [s.key, s.label]), "every state but gap");
  });

  test("every cause of a difference is a reason, by the name the page gives the cause", () => {
    assert.deepEqual(new Set(REASONS.map((r) => r.cause)), new Set([...Object.keys(CAUSES.differs), "library_gap"]));
    assert.equal(REASONS.find((r) => r.cause === "incorrect_option").name, "Multiple choice: wrong on purpose");
    assert.equal(REASONS.at(-1).name, CAUSES.unchecked.library_gap);
  });

  test("in the cascade a bar starts where the one before it ended, and a reason with no number has no row", () => {
    const { steps, whole, differ } = cascade(tally(CASES));
    assert.deepEqual(steps.map((s) => [s.cause, s.n]), [["source_error", 1], ["method", 4], ["library_gap", 3]]);
    assert.equal(whole, 8);
    assert.equal(differ, 5, "those not in the library yet do not differ");
    assert.deepEqual(steps.map((s) => s.from), [0, 1 / 8, 5 / 8]);
    assert.equal(steps[2].from + steps[2].share, 1);
  });

  test("the examples of a reason are plain numbers, one source after another in turn", () => {
    const cases = [
      { ...CASES[0], differs: [number(0.1), { ...number(0.1), printed: ["a", "b"] }, { ...number(0.1), found: "x1, x2" }, { ...number(0.1), printed: "7", found: "8" }] },
      { ...CASES[2], differs: [{ ...number(0.1), what: "other", term: "mode", printed: "3", found: "4" }] },
    ];
    const found = examples(cases, "method", NAMES);
    assert.deepEqual(found.map((e) => [e.source, e.printed, e.found]), [["lenth", "1", "1.1"], ["davies", "3", "4"], ["lenth", "7", "8"]]);
    assert.deepEqual(found[0], { source: "lenth", where: "p. 1", what: "a coefficient", printed: "1", found: "1.1" });
    assert.equal(found[1].what, "mode", "a quantity the catalog has no name for is named by its term");
    assert.equal(examples(cases, "method")[0].what, "coefficient", "the catalog's own word, where no name is given");
    assert.deepEqual(examples(CASES, "unexplained"), []);
  });

  test("an example of what the library lacks has the printed number, and no value of the library's", () => {
    assert.deepEqual(
      examples(CASES, "library_gap").map((e) => [e.source, e.what, e.printed, e.found]),
      [["davies", "estimate", "4", null], ["davies", "estimate", "5", null], ["davies", "radius", "2", null]],
    );
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

  test("a quantity is named by its plain name, and one the catalog calls other by its term alone", () => {
    assert.equal(quantity({ what: "anova_sum_sq", term: "Error" }, NAMES), "a sum of squares: Error");
    assert.equal(quantity({ what: "f_regression", term: null }, NAMES), "an F ratio");
    assert.equal(quantity({ what: "other", term: "mode" }, NAMES), "mode");
    assert.equal(quantity({ what: "other", term: null }, NAMES), "");
    assert.equal(quantityName("lack_of_fit_ss", NAMES), "lack of fit ss", "no name: the catalog's own word");
  });
});

describe("the search", () => {
  const ids = (list) => list.map((c) => c.id);

  test("keeps every case with a recomputed number until a filter is chosen", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES)), ["a", "b", "c", "d", "f"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, {}, { all: true })), ["a", "b", "c", "d", "e", "f"]);
  });

  test("reads the name of a cause, and what the library cannot compute yet", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "slip in the book" })), ["f"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "irls" }, { all: true })), ["e", "f"]);
  });

  test("reads a quantity by its plain name", () => {
    const cases = [{ ...CASES[2], differs: [{ ...number(0.004), what: "anova_sum_sq", term: "Error" }] }, CASES[3]];
    assert.deepEqual(ids(filterCases(cases, SOURCES, { query: "squares" }, { names: NAMES })), ["c"]);
    assert.deepEqual(ids(filterCases(cases, SOURCES, { query: "squares" })), [], "no names: the catalog's own words");
  });

  test("reads the title, the source's name and the reasons, every word of it", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "yield" })), ["a", "c"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "davies rounding" })), ["c"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { query: "alias misprinted" })), ["b"]);
  });

  test("keeps a source and a kind, the catalog's two spellings of a case study as one", () => {
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { source: "lenth" })), ["a", "b"]);
    assert.deepEqual(ids(filterCases(CASES, SOURCES, { kind: "case study" })), ["d"]);
    assert.deepEqual(kinds(CASES), [["example", 2], ["exercise", 2], ["case study", 1]]);
  });
});

describe("the sources and the lists", () => {
  test("each source counts its cases by state and its numbers that agree, the oldest first", () => {
    const rows = bySource(CASES, SOURCES);
    assert.deepEqual(rows.map((r) => [r.key, r.form, r.total, r.states.match]), [["davies", "book", 3, 0], ["lenth", "paper", 2, 1]]);
    assert.deepEqual([rows[1].numbers, rows[1].agree], [70, 68]);
    assert.deepEqual(bySource(CASES, SOURCES, "name").map((r) => r.key), ["davies", "lenth"]);
  });

  test("no order ranks the sources by how much of them reproduces", () => {
    assert.deepEqual(Object.keys(SOURCE_ORDERS), ["year", "name", "numbers"]);
    assert.deepEqual(bySource(CASES, SOURCES, "numbers").map((r) => r.key), ["lenth", "davies"]);
  });

  test("the sources group by the form of their source, each group with its totals, and all of them", () => {
    const { groups, totals } = byForm(bySource(CASES, SOURCES), FORMS);
    assert.deepEqual(groups.map((g) => [g.form, g.label, g.rows.map((r) => r.key)]), [["book", "Books", ["davies"]], ["paper", "Journal papers", ["lenth"]]]);
    assert.deepEqual(groups[1].totals, { total: 2, states: { match: 1, known: 1, quiz: 0, open: 0 }, numbers: 70, agree: 68 });
    assert.equal(totals.total, 5);
    assert.equal(totals.numbers, 100);
    assert.equal(totals.agree, 100 - 5);
  });

  test("a form chosen keeps the cases of its sources", () => {
    assert.deepEqual(filterCases(CASES, SOURCES, { form: "paper" }).map((c) => c.id), ["a", "b"]);
  });

  test("a state's list groups its cases by source, the worst first", () => {
    const more = [...CASES, { ...CASES[2], id: "c2", worst: 0.009 }];
    assert.deepEqual(
      groupByState(more, "known", ["lenth", "davies"]).map((g) => [g.source, g.cases.map((c) => c.id)]),
      [["lenth", ["b"]], ["davies", ["d", "c2", "c", "f"]]],
    );
    assert.deepEqual(groupByState(more, "match", ["lenth", "davies"]).map((g) => g.cases.map((c) => c.id)), [["a"]], "not e: nothing recomputed");
  });
});

describe("the causes", () => {
  test("the numbers of one missing method fall together, whatever each number is", () => {
    const ridge = "the library has no ridge analysis under linear restrictions: its ridge_analysis traces spheres";
    assert.equal(missingKey(`a point of path A at 400; ${ridge}`), "The library has no ridge analysis under linear restrictions");
    assert.equal(missingKey(`x1 of path B; ${ridge}`), missingKey(`a statement; ${ridge}`));
    const vague = "Larntz and Whitcomb's multiplier, which the library does not compute";
    assert.equal(missingKey(vague), vague, "too short a clause to tell one missing method from another");
  });

  test("what the library lacks is grouped by its reason, with the numbers and cases that wait on it", () => {
    const groups = missingByReason(CASES);
    assert.deepEqual(groups.map((g) => [g.reason, g.numbers, g.cases.map((c) => c.id)]), [["The library has no IRLS fit", 3, ["e", "f"]]]);
  });
});

describe("the address", () => {
  test("keeps the view, leaving out what is at its default", () => {
    assert.equal(toHash(DEFAULT_VIEW), "");
    const view = { query: "misprint", source: "lenth", form: "paper", kind: "example", open: ["known", "open"], focus: "b" };
    assert.equal(toHash(view), "#q=misprint&src=lenth&form=paper&kind=example&open=known%2Copen&case=b");
    assert.deepEqual(fromHash(toHash(view), SOURCES, FORMS), view);
  });

  test("ignores what it does not know", () => {
    const read = fromHash("#src=nobody&form=pamphlet&open=known,within_1,everything&case=<script>", SOURCES, FORMS);
    assert.deepEqual(read, { ...DEFAULT_VIEW, open: ["known"] }, "a tier an older link names is no list now");
    assert.deepEqual(TIER_KEYS, ["reproduced", "within_1", "within_5", "not_reproduced"]);
  });
});
