// web/logic.js: what the page counts, decides, formats and sorts, apart from the page.
// The rules it shares with web/bootstrap.py are held to Python by tests/web/page/test_parity.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  AXIAL_DISTANCE,
  CAPTIONS,
  CATEGORICAL,
  MODES,
  NUMERIC,
  axialDistance,
  candidateTerms,
  combinations,
  compareCells,
  defaultResolution,
  escape,
  factorType,
  fitStatistics,
  fmt,
  formMatches,
  isBlank,
  isCategorical,
  isLabel,
  levelKind,
  levelProblem,
  levelValue,
  listNames,
  nTerms,
  namedModelTerms,
  needsCentre,
  nextMode,
  numberOrNull,
  pval,
  shownTheme,
  sortNumber,
  termLabel,
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
    assert.ok(needsCentre("box_behnken", "face_centered", 3));
    assert.ok(needsCentre("dsd", "face_centered", 3));
    assert.ok(needsCentre("ccd", "rotatable", 3));
    assert.ok(!needsCentre("ccd", "face_centered", 3));
    assert.ok(!needsCentre("full_factorial", "rotatable", 3));
  });

  test("a CCD whose typed axial distance is the corners' own keeps a center run", () => {
    assert.ok(needsCentre("ccd", Math.SQRT2, 2));
    assert.ok(needsCentre("ccd", 2, 4));
    assert.ok(!needsCentre("ccd", 1.4, 2), "near the corners' distance is not on their sphere");
    assert.ok(!needsCentre("ccd", 2, 3));
  });

  test("the axial distance the form sends", () => {
    assert.equal(axialDistance("rotatable", "1.4"), "rotatable", "a named choice ignores the box");
    assert.equal(axialDistance(AXIAL_DISTANCE, " 1.4 "), 1.4);
    assert.equal(axialDistance(AXIAL_DISTANCE, ""), null, "a blank box is null, which Python refuses in words");
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

  test("the fit's statistics, a line each, and why a predicted R² is missing", () => {
    const all = { r_squared: 0.9, r_squared_adj: 0.85, r_squared_pred: 0.7, df_residual: 5 };
    assert.deepEqual(fitStatistics(all, 15), [
      "15 completed runs",
      "R² = 0.9000",
      "Adjusted R² = 0.8500",
      "Predicted R² = 0.7000",
      "Residual degrees of freedom = 5",
    ]);
    assert.equal(fitStatistics(all, 13, [14, 15])[1], "Still to run: 14, 15");
    assert.match(fitStatistics({ ...all, r_squared_pred: null }, 15)[3], /^Predicted R² cannot be computed/);
    const saturated = fitStatistics({ r_squared: 1, df_residual: 0 }, 4);
    assert.deepEqual([saturated[1], saturated.at(-1)], ["R² = 1.000", "Residual degrees of freedom = 0"]);
  });

  test("text placed in HTML is escaped", () => {
    assert.equal(escape('<b a="1">&</b>'), "&lt;b a=&quot;1&quot;&gt;&amp;&lt;/b&gt;");
    assert.equal(escape(12), "12");
  });

  test("the captions name the factors drawn", () => {
    assert.match(CAPTIONS.contour(["T", "P"]), /over T and P, the two factors/);
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

describe("the terms a reader ticks", () => {
  test("pairs and triples of factors, in order", () => {
    assert.deepEqual(combinations(["A", "B", "C"], 2), [["A", "B"], ["A", "C"], ["B", "C"]]);
    assert.deepEqual(combinations(["A", "B", "C"], 3), [["A", "B", "C"]]);
    assert.deepEqual(combinations(["A", "B"], 3), []);
  });

  const three = ["T", "P", "F"].map((name) => ({ name, low: -1, high: 1 }));

  test("every term three factors allow, grouped, three-factor interactions included", () => {
    const terms = candidateTerms(three);
    assert.equal(terms.length, 3 + 3 + 1 + 3);
    assert.deepEqual(
      terms.filter((t) => t.group === "Three-factor interactions").map((t) => t.label),
      ["T·P·F"],
    );
    assert.deepEqual(terms.find((t) => t.group === "Squares"), { term: ["T", "T"], label: "T²", group: "Squares" });
  });

  test("a term written as the page shows it", () => {
    assert.deepEqual([["A"], ["A", "B"], ["A", "B", "C"], ["A", "A"]].map(termLabel), ["A", "A·B", "A·B·C", "A²"]);
  });

  test("a named model's terms, where the list starts", () => {
    assert.equal(namedModelTerms("main_effects", three).length, nTerms(3, "main_effects") - 1);
    assert.equal(namedModelTerms("interactions", three).length, nTerms(3, "interactions") - 1);
    assert.equal(namedModelTerms("quadratic", three).length, nTerms(3, "quadratic") - 1);
  });

  test("a categorical factor has no square to tick, nor one in the quadratic model", () => {
    const factors = [three[0], { name: "S", low: "A", high: "B" }];
    assert.deepEqual(candidateTerms(factors).filter((t) => t.group === "Squares").map((t) => t.label), ["T²"]);
    assert.deepEqual(namedModelTerms("quadratic", factors).at(-1), ["T", "T"]);
  });
});

describe("a factor's levels", () => {
  test("a label is text that is no number", () => {
    assert.deepEqual(["A", " old ", "Supplier 2"].map(isLabel), [true, true, true]);
    assert.deepEqual(["150", "-1.5", "1e3", "", " ", "inf", "-Infinity", "NaN", null].map(isLabel), Array(9).fill(false));
  });

  test("a level is sent as a label, a number, or null for a blank box", () => {
    assert.deepEqual([" old ", "150", "", "1e3"].map(levelValue), ["old", 150, null, 1000]);
  });

  test("a factor is categorical when both its levels are labels", () => {
    assert.ok(isCategorical({ low: "A", high: "B" }));
    assert.ok(!isCategorical({ low: "A", high: 3 }));
    assert.ok(!isCategorical({ low: 1, high: 3 }));
  });

  test("a design whose labels are as the form shows them still matches the form", () => {
    const spec = { response: "y", factors: [{ name: "S", low: "A", high: "B", units: "" }] };
    assert.ok(formMatches([{ name: "S", low: "A", high: "B", units: "" }], "y", spec));
    assert.ok(!formMatches([{ name: "S", low: "A", high: "C", units: "" }], "y", spec));
  });

  test("each box holds a number, a label, or nothing", () => {
    assert.deepEqual(["150", " -1.5 ", "old", "inf", "", "  ", null].map(levelKind), [
      "number",
      "number",
      "label",
      "number",
      "blank",
      "blank",
      "blank",
    ]);
  });

  test("two labels make a factor categorical and two numbers numeric, whatever type it had", () => {
    assert.equal(factorType("old", "new", NUMERIC), CATEGORICAL);
    assert.equal(factorType("150", "200", CATEGORICAL), NUMERIC);
  });

  test("a factor half typed in keeps the type the reader chose", () => {
    for (const [low, high] of [["old", ""], ["", "200"], ["old", "200"], ["", ""]]) {
      assert.equal(factorType(low, high, CATEGORICAL), CATEGORICAL, `${low}, ${high}`);
      assert.equal(factorType(low, high, NUMERIC), NUMERIC, `${low}, ${high}`);
    }
  });

  test("a numeric factor needs two numbers, the low below the high", () => {
    assert.equal(levelProblem({ name: "T", low: "150", high: "200" }, NUMERIC), "");
    assert.equal(
      levelProblem({ name: "T", low: "150", high: "" }, NUMERIC),
      "T: give a number for both the low and the high value, or make it categorical for two labels.",
    );
    assert.match(levelProblem({ name: "T", low: "old", high: "200" }, NUMERIC), /or make it categorical/);
    assert.equal(levelProblem({ name: "T", low: "200", high: "150" }, NUMERIC), "T: the low value must be below the high value.");
    assert.equal(levelProblem({ name: "T", low: "5", high: "5" }, NUMERIC), "T: the low value must be below the high value.");
  });

  test("a categorical factor needs two labels that differ, neither of them a number", () => {
    assert.equal(levelProblem({ name: "Supplier", low: "A", high: "B" }, CATEGORICAL), "");
    assert.equal(levelProblem({ name: "Supplier", low: "A", high: " " }, CATEGORICAL), "Supplier: give it two labels, such as A and B.");
    assert.equal(
      levelProblem({ name: "Supplier", low: "1", high: "B" }, CATEGORICAL),
      "Supplier: a label cannot be a number; write C1 rather than 1.",
    );
    assert.equal(levelProblem({ name: "Supplier", low: "A", high: " A " }, CATEGORICAL), "Supplier: its two labels must differ.");
  });

  test("a factor without a name is still named in the sentence", () => {
    assert.match(levelProblem({ name: " ", low: "A", high: "" }, CATEGORICAL), /^A factor without a name: /);
  });
});

describe("the colour mode", () => {
  test("the switch offers the system's mode, light and dark, as the book's does", () => {
    assert.deepEqual(MODES, ["auto", "light", "dark"]);
  });

  test("from the system's mode the first press shows the other theme, so it always changes the page", () => {
    assert.equal(nextMode("auto", false), "dark");
    assert.equal(nextMode("auto", true), "light");
    assert.equal(shownTheme(nextMode("auto", false), false), "dark");
    assert.equal(shownTheme(nextMode("auto", true), true), "light");
  });

  test("three presses come back to the system's mode", () => {
    for (const dark of [false, true]) {
      const seen = [MODES[0]];
      for (let i = 0; i < 3; i += 1) seen.push(nextMode(seen.at(-1), dark));
      assert.equal(seen.at(-1), "auto");
      assert.deepEqual(new Set(seen.slice(0, 3)), new Set(MODES));
    }
  });

  test("a mode the page does not know starts the cycle again", () => {
    assert.equal(nextMode("sepia", false), "auto");
  });
});
