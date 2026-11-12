// web/import.js: a table of the reader's own, and what stops it from being a study.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { MAX_RESPONSES, ROLES, holdsWords, importProblems, isWorkbook, openedWords, takesRange } from "../../../web/import.js";

const temp = { heading: "Temp (degC)", name: "Temp", units: "degC", kind: "number", settings: 2, low: 150, high: 200, role: "factor" };
const catalyst = { heading: "Catalyst", name: "Catalyst", units: "", kind: "text", settings: 2, levels: ["A", "B"], role: "factor" };
const yieldOf = { heading: "Yield (%)", name: "Yield", units: "%", kind: "number", settings: 16, low: 51, high: 68, role: "response" };

describe("a table of the reader's own", () => {
  test("a column is a factor, a response, or left out", () => {
    assert.deepEqual(ROLES.map(([value]) => value), ["factor", "response", "ignore"]);
  });

  test("what a column holds is said in a few words", () => {
    assert.equal(holdsWords(temp), "numbers, 150 to 200, at 2 settings");
    assert.equal(holdsWords({ ...temp, low: 5, high: 5, settings: 1 }), "numbers, 5, at 1 setting");
    assert.equal(holdsWords(catalyst), "labels: A, B");
    assert.equal(holdsWords({ ...catalyst, settings: 12 }), "labels: A, B, ...");
    assert.equal(holdsWords({ kind: "empty" }), "empty");
  });

  test("only a numeric factor takes a range", () => {
    assert.ok(takesRange(temp));
    assert.ok(!takesRange(catalyst));
    assert.ok(!takesRange(yieldOf));
  });

  test("a workbook is told from text by its name", () => {
    assert.ok(isWorkbook("Trials.XLSX"));
    assert.ok(!isWorkbook("trials.csv"));
  });

  test("a table with a factor and a response, each named, has nothing to mend", () => {
    assert.deepEqual(importProblems([temp, catalyst, yieldOf]), []);
  });

  test("a study needs a factor and a response", () => {
    assert.deepEqual(importProblems([{ ...temp, role: "ignore" }, yieldOf]), ["Choose at least one column as a factor."]);
    assert.deepEqual(importProblems([temp, { ...yieldOf, role: "ignore" }]), ["Choose at least one column as a response."]);
  });

  test("each column of the study has a name a model can hold, and its own", () => {
    assert.match(importProblems([{ ...temp, name: "Temp C" }, yieldOf])[0], /^Temp C: a name is letters, digits and _/);
    assert.deepEqual(importProblems([{ ...temp, name: "" }, yieldOf]), ["Temp (degC) needs a name."]);
    assert.deepEqual(importProblems([{ ...temp, name: "Yield" }, yieldOf]), ["Yield: two columns of the study cannot share a name."]);
    assert.deepEqual(importProblems([{ ...temp, role: "ignore", name: "" }, catalyst, yieldOf]), [], "a column left out needs none");
  });

  test("a response is a number, and a factor's low is below its high", () => {
    assert.deepEqual(importProblems([temp, { ...catalyst, role: "response" }]), ["Catalyst holds labels, and a response is a number."]);
    assert.deepEqual(importProblems([{ ...temp, low: 200, high: 150 }, yieldOf]), ["Temp: the low value must be below the high value."]);
    assert.deepEqual(importProblems([{ ...temp, low: null }, yieldOf]), ["Temp: the low value must be below the high value."]);
  });

  test("a study measures a few responses at most", () => {
    const many = Array.from({ length: MAX_RESPONSES + 1 }, (_, i) => ({ ...yieldOf, heading: `y${i}`, name: `y${i}` }));
    assert.deepEqual(importProblems([temp, ...many]), [`A study measures at most ${MAX_RESPONSES} responses; ${MAX_RESPONSES + 1} are chosen.`]);
  });

  test("the page says what it opened, and what it left out", () => {
    assert.equal(openedWords(16, 0), "Opened your table as a study of 16 runs, taken as they are. It is saved in this browser, under Your studies.");
    assert.match(openedWords(15, 1), /1 run with no setting for some factor was left out\./);
    assert.match(openedWords(14, 2), /2 runs with no setting for some factor were left out\./);
  });
});
