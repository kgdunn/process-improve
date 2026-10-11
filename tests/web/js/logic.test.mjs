// web/logic.js: what the page counts, decides, formats and sorts, apart from the page.
// The rules it shares with web/bootstrap.py are held to Python by tests/web/page/test_parity.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  CAPTIONS,
  compareCells,
  defaultResolution,
  escape,
  fitStatistics,
  fmt,
  formMatches,
  isBlank,
  listNames,
  nTerms,
  needsCentre,
  numberOrNull,
  pval,
  sortNumber,
} from "../../../web/logic.js";

describe("the design's settings", () => {
  test("a model's coefficients, intercept included", () => {
    assert.equal(nTerms(3, "main_effects"), 4);
    assert.equal(nTerms(3, "interactions"), 7);
    assert.equal(nTerms(3, "quadratic"), 10);
  });

  test("a fraction opens at resolution IV, or III when three factors allow no more", () => {
    assert.deepEqual([3, 4, 5, 11].map(defaultResolution), [3, 4, 4, 4]);
  });

  test("the designs that keep a center run", () => {
    assert.ok(needsCentre("box_behnken", "face_centered"));
    assert.ok(needsCentre("dsd", "face_centered"));
    assert.ok(needsCentre("ccd", "rotatable"));
    assert.ok(!needsCentre("ccd", "face_centered"));
    assert.ok(!needsCentre("full_factorial", "rotatable"));
  });
});

describe("the form against the design", () => {
  const spec = { response: "y", factors: [{ name: "T", low: 150, high: 200, units: "degC" }, { name: "P", low: 1, high: 3 }] };
  const form = [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "P", low: 1, high: 3, units: "" },
  ];

  test("the same factors and response match, a missing unit reading as none", () => {
    assert.ok(formMatches(form, "y", spec));
  });

  test("a renamed factor, a changed range, an added factor or another response do not", () => {
    assert.ok(!formMatches([{ ...form[0], name: "Temp" }, form[1]], "y", spec));
    assert.ok(!formMatches([form[0], { ...form[1], high: 4 }], "y", spec));
    assert.ok(!formMatches([...form, { name: "F", low: 0, high: 1, units: "" }], "y", spec));
    assert.ok(!formMatches(form, "Yield", spec));
  });

  test("a blank response is the y the design was made with", () => {
    assert.ok(formMatches(form, "", spec));
    assert.ok(!formMatches(form, "", { ...spec, response: "Yield" }));
  });

  test("a design read from a workbook keeps its ranges as numbers or as text", () => {
    assert.ok(formMatches(form, "y", { ...spec, factors: spec.factors.map((f) => ({ ...f, low: String(f.low) })) }));
  });

  test("anything matches before there is a design", () => {
    assert.ok(formMatches(form, "y", undefined));
  });

  test("a blank box is sent as null, never as zero", () => {
    assert.equal(numberOrNull(""), null);
    assert.equal(numberOrNull("  "), null);
    assert.equal(numberOrNull("1.5"), 1.5);
    assert.ok(Number.isNaN(numberOrNull("abc")), "Python refuses it, with a message");
  });
});

describe("words and numbers", () => {
  test("names are joined as a sentence joins them", () => {
    assert.deepEqual([[], ["A"], ["A", "B"], ["A", "B", "C"]].map(listNames), ["", "A", "A and B", "A, B and C"]);
  });

  test("numbers to four significant figures, and nothing for a value the fit could not give", () => {
    assert.deepEqual([null, undefined, 1.23456, 0, 1234567].map(fmt), ["", "", "1.235", "0.000", "1.235e+6"]);
  });

  test("p-values to four decimals, the smallest as a bound", () => {
    assert.deepEqual([null, 0.00001, 0.5, 0.012345].map(pval), ["", "< 0.0001", "0.5000", "0.0123"]);
  });

  test("the fit's statistics, and why a predicted R² is missing", () => {
    const all = { r_squared: 0.9, r_squared_adj: 0.85, r_squared_pred: 0.7, df_residual: 5 };
    assert.equal(
      fitStatistics(all),
      "R² = 0.9000, adjusted R² = 0.8500, predicted R² = 0.7000, residual degrees of freedom = 5.",
    );
    assert.match(fitStatistics({ ...all, r_squared_pred: null }), /Predicted R² cannot be computed/);
    assert.equal(fitStatistics({ r_squared: 1, df_residual: 0 }).startsWith("R² = 1.000, residual degrees"), true);
  });

  test("text placed in HTML is escaped", () => {
    assert.equal(escape('<b a="1">&</b>'), "&lt;b a=&quot;1&quot;&gt;&amp;&lt;/b&gt;");
    assert.equal(escape(12), "12");
  });

  test("the captions name the factors drawn", () => {
    assert.match(CAPTIONS.contour(["T", "P"]), /over T and P, the two factors/);
    assert.match(CAPTIONS.coefficients(true), /A diamond marks the true value/);
    assert.doesNotMatch(CAPTIONS.coefficients(false), /diamond/);
  });
});

describe("sorting a table", () => {
  test("blank cells are blank, zero is not", () => {
    assert.deepEqual([null, undefined, "", 0, "0"].map(isBlank), [true, true, true, false, false]);
  });

  test("numbers sort as numbers, including text that leads with one", () => {
    assert.equal(sortNumber("< 0.0001"), 0.0001);
    assert.equal(sortNumber("4.623 to 5.300"), 4.623);
    assert.deepEqual([10, 9, "< 0.0001", 2.5].sort(compareCells), ["< 0.0001", 2.5, 9, 10]);
  });

  test("text sorts alphabetically, numbers within it in order", () => {
    assert.deepEqual(["x10", "x9", "a"].sort(compareCells), ["a", "x9", "x10"]);
  });
});
