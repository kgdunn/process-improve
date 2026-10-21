// web/logic.js: what the page counts, decides, formats and sorts, apart from the page.
// The rules it shares with web/bootstrap.py are held to Python by tests/web/page/test_parity.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  AXIAL_DISTANCE,
  alsoRuledOut,
  cornersWords,
  cutWords,
  filterCorners,
  settingLabel,
  insertAt,
  limitEffect,
  limitsSummary,
  positional,
  renameLimit,
  shareText,
  withBounds,
  CAPTIONS,
  CATEGORICAL,
  MODES,
  NUMERIC,
  axialDistance,
  candidateTerms,
  combinations,
  compareCells,
  comparedRow,
  rowDecimals,
  defaultResolution,
  escape,
  factorType,
  fitStatistics,
  fittedFormula,
  fmt,
  GOALS,
  OPTIMAL_MODEL,
  SCREEN_FROM,
  SPACE_FILLING_RUNS,
  defaultBudget,
  aimTerms,
  analysisModel,
  candidateRows,
  comparisonOrder,
  changeWords,
  defaultAim,
  describeResponse,
  holds,
  nameProblem,
  percent,
  pickCandidate,
  pickSentence,
  pastedCells,
  pastedNumber,
  placePasted,
  leftOutSentences,
  standouts,
  responseProblems,
  runBadge,
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
  powerScale,
  pval,
  renameTerms,
  renaming,
  responseName,
  responseTitle,
  shownTheme,
  showsVif,
  sortNumber,
  termLabel,
  VIF_HIGH,
  withUnits,
  FULL,
  RESOLUTIONS,
  TRADE_OFF,
  TRADE_OFF_FACTORS,
  TRADE_OFF_RUNS,
  cellBand,
  cellDesign,
  cellResolution,
  cellText,
  fractionRuns,
  tradeOffCell,
  MAX_RESPONSES,
  anyResult,
  completeRuns,
  joinResults,
  partialRuns,
  partialSentence,
  responseColumns,
  responseLabelProblems,
  responsesOf,
  responsesTitle,
  splitResults,
  storedResponses,
  withResponses,
  MIXTURES,
  boundedComponents,
  componentProblem,
  isMixture,
  mixtureProblem,
  proportionValue,
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

describe("the trade-off grid of a fraction", () => {
  const fraction = (resolution) => ({ design_type: "fractional_factorial", resolution });
  const full = { design_type: "full_factorial" };

  test("holds a cell for every run count and number of factors", () => {
    assert.equal(TRADE_OFF.length, TRADE_OFF_RUNS.length);
    for (const row of TRADE_OFF) assert.equal(row.length, TRADE_OFF_FACTORS.length);
    assert.equal(tradeOffCell(8, 5), 3);
    assert.equal(tradeOffCell(32, 5), FULL);
    assert.equal(tradeOffCell(4, 4), null, "four factors do not fit in four runs");
    assert.equal(tradeOffCell(16, 3), null, "below the full factorial the runs only repeat it");
    assert.equal(tradeOffCell(16, 12), null, "no column past the page's eleven factors");
    assert.equal(tradeOffCell(12, 5), null, "no row for a run count that is not a power of two");
  });

  test("a cell shows Full or its resolution in roman numerals, coloured in three bands", () => {
    assert.deepEqual([FULL, 8, 5, 4, 3, null].map(cellText), ["Full", "VIII", "V", "IV", "III", ""]);
    assert.deepEqual([FULL, 8, 5, 4, 3, null].map(cellBand), ["high", "high", "high", "iv", "iii", null]);
  });

  test("the page's fraction at a resolution has the fewest runs that reach it", () => {
    assert.deepEqual([3, 4, 5].map((r) => fractionRuns(5, r)), [8, 16, 16], "IV in five factors is the 16-run V");
    assert.deepEqual([3, 4, 5].map((r) => fractionRuns(7, r)), [8, 16, 64]);
    assert.deepEqual([3, 4].map((r) => fractionRuns(4, r)), [8, 8], "the half fraction, of resolution IV");
    assert.equal(fractionRuns(2, 3), null, "no column for two factors");
  });

  test("a cell of resolution III, IV or V sets that resolution", () => {
    assert.deepEqual(cellDesign(4, 3), fraction(3));
    assert.deepEqual(cellDesign(8, 5), fraction(3));
    assert.deepEqual(cellDesign(16, 5), fraction(5));
    assert.deepEqual(cellDesign(16, 6), fraction(4));
    assert.deepEqual(cellDesign(128, 10), fraction(5));
  });

  test("a fraction cell asks for its own resolution, or V above it", () => {
    assert.deepEqual([3, 4, 5, 6, 8].map(cellResolution), [3, 4, 5, 5, 5]);
  });

  test("a cell of a higher resolution sets V when V gives exactly its runs", () => {
    assert.deepEqual(cellDesign(32, 6), fraction(5), "the 32-run VI in six factors");
    assert.deepEqual(cellDesign(64, 7), fraction(5), "the 64-run VII in seven factors");
    assert.deepEqual(cellDesign(128, 9), fraction(5), "the 128-run VI in nine factors");
    assert.equal(cellDesign(128, 8), null, "V in eight factors takes 64 runs, so the 128-run VIII is not offered");
  });

  test("a cell whose runs the page makes at a lower run count is not offered", () => {
    assert.equal(cellDesign(32, 7), null, "IV in seven factors takes 16 runs");
    assert.equal(cellDesign(32, 8), null);
    assert.equal(cellDesign(64, 9), null);
  });

  test("a Full cell switches to the full factorial", () => {
    for (const k of [3, 4, 5, 6, 7]) assert.deepEqual(cellDesign(2 ** k, k), full, `${k} factors`);
  });

  test("a blank cell sets nothing", () => {
    assert.equal(cellDesign(4, 5), null);
    assert.equal(cellDesign(64, 3), null);
  });

  test("every resolution the page offers marks a cell a reader can pick", () => {
    for (const k of TRADE_OFF_FACTORS) {
      for (const r of RESOLUTIONS.filter((r) => r <= k)) {
        const runs = fractionRuns(k, r);
        const picked = cellDesign(runs, k);
        assert.equal(picked.design_type, "fractional_factorial", `${k} factors, resolution ${r}`);
        assert.equal(fractionRuns(k, picked.resolution), runs, `${k} factors, resolution ${r}`);
      }
    }
  });
});

describe("a blank box", () => {
  test("is sent as null, never as zero", () => {
    assert.equal(numberOrNull(""), null);
    assert.equal(numberOrNull("  "), null);
    assert.equal(numberOrNull("1.5"), 1.5);
    assert.ok(Number.isNaN(numberOrNull("abc")), "Python refuses it, with a message");
  });
});

describe("the response, as the reader names it", () => {
  test("a name in words becomes a column every spreadsheet and Python read alike", () => {
    assert.equal(responseName("Yield"), "Yield");
    assert.equal(responseName("Tensile strength (MPa)"), "Tensile_strength_MPa");
    assert.equal(responseName("  Purity, % "), "Purity");
    assert.equal(responseName("Viscosité"), "Viscosite", "an accent is dropped, the letter kept");
  });

  test("nothing left is y, and a leading digit is prefixed", () => {
    assert.deepEqual(["", "  ", "%", null, undefined].map((label) => responseName(label)), Array(5).fill("y"));
    assert.equal(responseName("2nd pass yield"), "y_2nd_pass_yield");
    assert.equal(responseName("yield"), "yield", "the run sheet's column, never a variable of the code");
  });

  test("a name a factor or the blocks already take is numbered", () => {
    assert.equal(responseName("T", ["T", "P"]), "T_2");
    assert.equal(responseName("T", ["T", "T_2"]), "T_3");
    assert.equal(responseName("Block"), "Block_2");
  });

  test("a long name is cut to forty characters, never ending in _", () => {
    const name = responseName(`${"a".repeat(39)} b`);
    assert.equal(name, "a".repeat(39));
    assert.ok(responseName("x".repeat(80)).length <= 40);
  });

  test("a quantity with its units, or the name alone", () => {
    assert.equal(withUnits("T", "degC"), "T (degC)");
    assert.equal(withUnits("T", ""), "T");
  });

  test("the study's title names the response, its units and its goal", () => {
    assert.equal(responseTitle({ label: "Yield", units: "%", goal: "maximize" }), "Yield (%), maximise");
    assert.equal(responseTitle({ label: "Cost", goal: "minimize" }), "Cost, minimise");
    assert.equal(responseTitle({ label: "Purity", units: "%", goal: "target", target: 99.5 }), "Purity (%), target 99.5");
    assert.equal(responseTitle({ label: "Purity", goal: "target" }), "Purity, on target");
    assert.equal(responseTitle({ label: "Hardness", goal: "learn" }), "Hardness");
    assert.equal(responseTitle(undefined, "Yield_pct"), "Yield_pct", "a design made before the page asked");
    assert.equal(responseTitle({}, ""), "y");
  });

  test("each goal has its words", () => {
    assert.deepEqual(Object.keys(GOALS), ["maximize", "minimize", "target", "learn"]);
  });

  test("power is for the smallest change worth finding: half of it, over the noise", () => {
    assert.deepEqual(powerScale({ noise: 1.5, smallest_change: 6 }), { change: 6, coefficient: 2 });
    assert.equal(powerScale({ noise: 1.5 }), null);
    assert.equal(powerScale({ noise: 0, smallest_change: 2 }), null);
    assert.equal(powerScale(undefined), null);
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

  test("a mixture's captions say what the star marks once the best blend is found", () => {
    assert.match(CAPTIONS.ternary([]), /the white dots are the runs\.$/);
    assert.match(CAPTIONS.ternary([], { starred: true }), /the white dots are the runs, and the star the best blend\.$/);
    assert.match(CAPTIONS.cox([]), /the lines cross at the reference blend\.$/);
    assert.match(CAPTIONS.cox([], { starred: true }), /^Each line changes one component's share of the best blend .* which the star marks\.$/);
  });

  test("with limits, the caption says what is grey, and where the others are held when the center cannot be run", () => {
    const factors = [{ name: "F", units: "L/min" }];
    assert.equal(
      CAPTIONS.contour(["T", "P"], { shaded: true, held: [], factors }),
      "Drawn over T and P, the two factors with the largest main effects, with the others at their center; grey: " +
        "settings outside your limits; the dots are the runs.",
    );
    assert.equal(
      CAPTIONS.contour(["T", "P"], { shaded: true, held: [{ name: "F", value: 16.666666667 }], factors }),
      "Drawn over T and P, the two factors with the largest main effects, with F at 16.67 L/min, since the center " +
        "of the ranges cannot be run; grey: settings outside your limits; the dots are the runs.",
    );
  });
});

describe("a row of the detailed comparison", () => {
  const texts = (values, direction = "lower", ranked) => comparedRow(values, direction, ranked).map((c) => c.text);
  const goods = (values, direction = "lower", ranked) =>
    comparedRow(values, direction, ranked).map((c) => (c.good === null ? null : Math.round(c.good * 1000) / 1000));

  test("prints whole numbers whole, as a count or an eigenvalue that happens to be whole", () => {
    assert.deepEqual(texts([9, 11, 15]), ["9", "11", "15"]);
    assert.deepEqual(texts([4.000000000001, 6, 5]), ["4", "6", "5"], "a float's last bit does not make it fractional");
  });

  test("gives the row's largest value two significant figures, with one number of decimals for the row", () => {
    assert.deepEqual(texts([0.2143, 0.25, 0.1414]), ["0.21", "0.25", "0.14"]);
    assert.deepEqual(texts([4.82, 4, 7.343]), ["4.8", "4.0", "7.3"]);
    assert.deepEqual(texts([48.2, 52.9]), ["48", "53"]);
    assert.equal(rowDecimals([0.8312, 2.04]), 1, "the smaller values take the largest's decimals");
  });

  test("takes a decimal more while two values more than 1% apart would print alike, two at most", () => {
    assert.equal(rowDecimals([0.2998, 0.2953, 0.071]), 3, "both print 0.30, and differ by 1.5%");
    assert.deepEqual(texts([0.2998, 0.2953, 0.071]), ["0.300", "0.295", "0.071"]);
    assert.equal(rowDecimals([0.2143, 0.2141]), 2, "0.1% apart: alike is right");
    assert.equal(rowDecimals([1, 1.011, 1.5]), 2);
    assert.equal(rowDecimals([0.0102, 0.01, 5]), 3, "two decimals more at most, though 0.010 and 0.010 are 2% apart");
  });

  test("scores each value from 0, the row's worst, to 1, its best, in the direction the measure prefers", () => {
    assert.deepEqual(goods([0.2, 0.3, 0.4], "lower"), [1, 0.5, 0]);
    assert.deepEqual(goods([0.2, 0.3, 0.4], "higher"), [0, 0.5, 1]);
    assert.deepEqual(goods([0.2143, 0.2141, 0.3], "lower"), [1, 1, 0], "values printed alike score alike");
  });

  test("leaves unscored a missing value, a row with no direction, and a row where every design prints the same", () => {
    assert.deepEqual(comparedRow([0.2, null], "lower"), [{ text: "0.20", good: null }, { text: "", good: null }]);
    assert.deepEqual(goods([9, 11, 15], ""), [null, null, null]);
    assert.deepEqual(goods([0.5, 0.5], "higher"), [null, null]);
    assert.deepEqual(goods([null, null], "higher"), [null, null]);
  });

  test("scores only the designs ranked, and only they set the best and the worst", () => {
    const values = [0.071, 0.606, 0.354, 1];
    const ranked = [true, true, true, false];
    assert.deepEqual(texts(values, "lower", ranked), ["0.07", "0.61", "0.35", "1.00"], "the decimals of the ranked");
    assert.deepEqual(goods(values, "lower", ranked), [1, 0, 0.481, null], "(0.61 - 0.35) / (0.61 - 0.07)");
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

describe("the coefficients table's VIF column", () => {
  const row = (term, vif, block = false) => ({ term, vif, block });

  test("shown only when a term's VIF is above VIF_HIGH, as the analysis's note says", () => {
    const orthogonal = [row("Intercept", null), row("A", 1), row("B", 1)];
    assert.equal(showsVif(orthogonal), false, "no column of ones");
    assert.equal(showsVif([row("Intercept", null), row("A", VIF_HIGH), row("B", 1)]), false, "above it, not at it");
    assert.equal(showsVif([row("Intercept", null), row("A", 10.8), row("B", 11)]), true);
    assert.equal(showsVif([row("Intercept", null), row("Block1", 9, true), row("A", 1)]), false, "a block's own does not count");
    assert.equal(showsVif([row("Intercept", undefined), row("A", null)]), false, "a fit with no VIFs");
  });

  test("the terms fitted, as the formula that gives their VIFs: the blocks first, as the fit has them", () => {
    const fitted = [row("Intercept", null), row("Block1", 1.5, true), row("Block2", 1.5, true), row("A", 1), row("A:B", 1)];
    assert.equal(fittedFormula(fitted), "Block + A + A:B");
    assert.equal(fittedFormula([row("Intercept", null), row("A", 1), row("I(A ** 2)", 1)]), "A + I(A ** 2)");
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

describe("a factor's name", () => {
  test("is a variable name, as Python reads one", () => {
    for (const name of ["T", "Temp_2", "_x", "Température", "pH"]) assert.equal(nameProblem(name), "", name);
    assert.match(nameProblem("a b"), /^a b: a name is letters, digits and _/);
    assert.match(nameProblem("2x"), /^2x: /);
    assert.match(nameProblem("x-y"), /^x-y: /);
  });

  test("is given, and is no other factor's", () => {
    assert.equal(nameProblem("  ", [], 3), "Factor 3 needs a name.");
    assert.equal(nameProblem("T", ["T", "P"]), "T: two factors cannot share a name.");
  });

  test("is not a column the run sheet has besides its factors, when it is renamed on one", () => {
    const taken = ["Yield", "Block"];
    assert.equal(nameProblem("Yield", [], 1, taken), "Yield: the run sheet has a Yield column already.");
    assert.equal(nameProblem("Block", [], 1, taken), "Block: the run sheet has a Block column already.");
    assert.equal(nameProblem("Block"), "", "without a run sheet to rename, Python takes it");
  });
});

describe("renaming the factors of a run sheet", () => {
  const made = [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "Supplier", low: "old", high: "new", units: "" },
  ];

  test("maps each old name to its new one when only names and units change", () => {
    const after = [{ ...made[0], name: "Temperature", units: "K" }, made[1]];
    assert.deepEqual(renaming(made, after), { T: "Temperature", Supplier: "Supplier" });
    assert.deepEqual(renaming(made, [{ ...made[0], units: "K" }, made[1]]), { T: "T", Supplier: "Supplier" });
    const swapped = [{ ...made[0], name: "Supplier" }, { ...made[1], name: "T" }];
    assert.deepEqual(renaming(made, swapped), { T: "Supplier", Supplier: "T" });
  });

  test("is no rename when the factors' number or a level changes, or nothing does", () => {
    assert.equal(renaming(made, made.map((f) => ({ ...f }))), null);
    assert.equal(renaming(made, [{ ...made[0], name: "Temp", high: 210 }, made[1]]), null);
    assert.equal(renaming(made, [made[0], { ...made[1], name: "Vendor", high: "newer" }]), null);
    assert.equal(renaming(made, [{ ...made[0], low: "150", high: "200" }, made[1]]), null, "a label is not a number");
    assert.equal(renaming(made, [made[0]]), null);
  });

  test("renames the factors of chosen terms: main effects, interactions and squares", () => {
    const terms = [["T"], ["T", "P"], ["T", "T"], ["F"]];
    assert.deepEqual(renameTerms(terms, { T: "Temp", P: "Pres" }), [["Temp"], ["Temp", "Pres"], ["Temp", "Temp"], ["F"]]);
  });
});

describe("the response's description", () => {
  const fields = (more) => ({ label: "", units: "", goal: "learn", target: "", typical: "", noise: "", smallest_change: "", ...more });

  test("keeps the words trimmed and the numbers that are numbers", () => {
    assert.deepEqual(describeResponse(fields({ label: " Yield ", units: "%", typical: "62", noise: "1", smallest_change: "2" })), {
      label: "Yield",
      units: "%",
      goal: "learn",
      typical: 62,
      noise: 1,
      smallest_change: 2,
    });
  });

  test("leaves out an empty box, a target without its goal, and a noise or change that is not above 0", () => {
    assert.deepEqual(describeResponse(fields({ target: "5", noise: "0", smallest_change: "-1" })), { goal: "learn" });
    assert.deepEqual(describeResponse(fields({ goal: "target", target: "99.5" })), { goal: "target", target: 99.5 });
    assert.deepEqual(describeResponse(fields({ goal: "soar" })), { goal: "learn" });
  });

  test("says what a number still needs, as bootstrap.py refuses it", () => {
    assert.deepEqual(responseProblems(fields({ noise: "0", smallest_change: "-2" })), [
      "The run-to-run noise must be a number above 0.",
      "The smallest change worth finding must be a number above 0.",
    ]);
    assert.deepEqual(responseProblems(fields({ noise: "", smallest_change: "3" })), []);
  });
});

describe("what a study aims to learn", () => {
  const numeric = ["T", "P", "F"].map((name) => ({ name, low: -1, high: 1 }));

  test("starts from each factor's curvature, and from screening for many factors", () => {
    assert.equal(defaultAim(3), "main_quadratic");
    assert.equal(defaultAim(SCREEN_FROM), "main_effects");
  });

  test("an aim's model has its coefficients, and no square of a categorical factor", () => {
    assert.deepEqual(["main_effects", "interactions", "main_quadratic", "quadratic"].map((aim) => aimTerms(numeric, aim)), [4, 7, 7, 10]);
    assert.equal(aimTerms([...numeric.slice(0, 2), { name: "S", low: "A", high: "B" }], "main_quadratic"), 6);
  });

  test("an optimal design is built for the smallest model it takes that holds the aim", () => {
    assert.equal(OPTIMAL_MODEL.main_quadratic, "quadratic");
    for (const [aim, model] of Object.entries(OPTIMAL_MODEL)) assert.ok(holds(model, aim), aim);
  });

  test("a model holds another when it has every kind of term the other has", () => {
    assert.ok(holds("quadratic", "main_quadratic"));
    assert.ok(holds("interactions", "main_effects"));
    assert.ok(!holds("interactions", "main_quadratic"));
    assert.ok(!holds("main_effects", "interactions"));
  });

  test("a run sheet is analysed for the aim when its design can fit it, and holds less", () => {
    assert.equal(analysisModel("quadratic", "main_quadratic", true, numeric), "", "the design's own holds it");
    assert.deepEqual(analysisModel("main_effects", "main_quadratic", true, numeric), {
      terms: [["T"], ["P"], ["F"], ["T", "T"], ["P", "P"], ["F", "F"]],
    });
    assert.equal(analysisModel("main_effects", "interactions", true, numeric), "interactions");
    assert.equal(analysisModel("main_effects", "main_quadratic", false, numeric), "", "a design that cannot fit it");
    assert.equal(analysisModel("main_effects", undefined, true, numeric), "", "a study made before aims");
  });
});

describe("the designs that can learn the brief", () => {
  const value = (runs, fits, df, main, quadratic = null) => ({
    runs,
    fits,
    residual_df: df,
    power_main: main,
    power_interaction: null,
    power_quadratic: quadratic,
  });
  const result = {
    designs: ["Full factorial [n=11]", "DSD [n=11]", "Box-Behnken [n=15]", "D-optimal [n=8]"],
    values: {
      "Full factorial [n=11]": value(11, false, 6, null),
      "DSD [n=11]": value(11, true, 4, 0.46, 0.23),
      "Box-Behnken [n=15]": value(15, true, 8, 0.7, 0.39),
      "D-optimal [n=8]": value(8, true, 1, 0.3, 0.1),
    },
  };
  const made = [
    { design_type: "full_factorial", label: "Full factorial" },
    { design_type: "dsd", label: "DSD" },
    { design_type: "box_behnken", label: "Box-Behnken" },
    { design_type: "d_optimal", label: "D-optimal" },
  ];

  test("the detailed comparison puts the designs that cannot fit the model at its right, each group as scored", () => {
    assert.deepEqual(comparisonOrder(result), [1, 2, 3, 0], "the full factorial cannot fit, and goes last");
    const none = { designs: ["A", "B"], values: { A: value(4, false, 0, null), B: value(4, false, 0, null) } };
    assert.deepEqual(comparisonOrder(none), [0, 1], "with none that fits, the order is kept");
  });

  test("the designs the reader hides go to the far right, after those that cannot fit, in the same order", () => {
    const [full, dsd, boxBehnken] = result.designs;
    assert.deepEqual(comparisonOrder(result, new Set([boxBehnken])), [1, 3, 0, 2]);
    assert.deepEqual(comparisonOrder(result, new Set([full, dsd])), [2, 3, 1, 0], "fitting first, as shown");
    assert.deepEqual(comparisonOrder(result, new Set(result.designs)), comparisonOrder(result), "all hidden: as before");
  });

  test("are listed by their runs, those that fit first, with their spare runs and power", () => {
    const rows = candidateRows(result, made);
    assert.deepEqual(rows.map((r) => r.design_type), ["d_optimal", "dsd", "box_behnken", "full_factorial"]);
    assert.deepEqual(rows[1], {
      design_type: "dsd",
      label: "DSD",
      runs: 11,
      fits: true,
      spare: 4,
      power: { main: 0.46, interaction: null, quadratic: 0.23 },
      order: 1,
    });
    assert.equal(rows[3].spare, null, "no spare runs for a design that cannot fit the model");
  });

  test("the one picked has the fewest runs that leave three to estimate the noise", () => {
    assert.equal(pickCandidate(candidateRows(result, made)), "dsd");
    const tight = { ...result, values: { ...result.values, "DSD [n=11]": value(11, true, 2, 0.4), "Box-Behnken [n=15]": value(15, true, 2, 0.5) } };
    assert.equal(pickCandidate(candidateRows(tight, made)), "d_optimal", "else the fewest runs that fit");
    assert.equal(pickCandidate([]), null);
  });

  test("power reads as a percentage, for the change in the reader's units", () => {
    assert.deepEqual([0.4626, 1, null].map(percent), ["46%", "100%", ""]);
    assert.equal(changeWords({ units: "%", noise: 1, smallest_change: 2 }), "a change of 2 %");
    assert.equal(changeWords({ noise: 1 }), "a change of two noise standard deviations");
  });

  test("the pick says why: the fewest runs that fit and leave three for the noise", () => {
    const rows = candidateRows(result, made);
    assert.equal(
      pickSentence(rows, "dsd"),
      "Picked for you: DSD, the fewest runs with 3 or more spare.",
    );
    const tight = { ...result, values: { ...result.values, "DSD [n=11]": value(11, true, 2, 0.4), "Box-Behnken [n=15]": value(15, true, 2, 0.5) } };
    assert.equal(
      pickSentence(candidateRows(tight, made), "d_optimal"),
      "Picked for you: D-optimal, the fewest runs, as none here has 3 spare.",
    );
    assert.equal(pickSentence(rows, "d_optimal"), "", "a design the rule would not pick has no reason to give");
    assert.equal(pickSentence(rows, "full_factorial"), "", "nor one that cannot fit");
    assert.equal(pickSentence([], null), "");
  });

  describe("what each design does best of those that fit", () => {
    const measures = [
      { key: "a", direction: "lower" },
      { key: "d", direction: "higher" },
      { key: "max_r", direction: "lower" },
      { key: "power_main", direction: "higher" },
      { key: "power_quadratic", direction: "higher" },
    ];
    const scored = (runs, fits, extra) => ({ ...value(runs, fits, 4, null), ...extra });
    const compared = {
      measures,
      designs: ["Full factorial [n=11]", "DSD [n=11]", "Box-Behnken [n=15]", "D-optimal [n=8]"],
      values: {
        "Full factorial [n=11]": scored(11, false, { a: 0.01, d: 9, power_main: 1 }),
        "DSD [n=11]": scored(11, true, { a: 0.2, d: 0.5, max_r: 0.5, power_main: 0.46, power_quadratic: 0.23 }),
        "Box-Behnken [n=15]": scored(15, true, { a: 0.1, d: 0.6, max_r: 0.0, power_main: 0.7, power_quadratic: 0.39 }),
        "D-optimal [n=8]": scored(8, true, { a: 0.3, d: 0.604, max_r: 0.0, power_main: 0.3, power_quadratic: 0.1 }),
      },
    };
    const words = (out) => Object.fromEntries(Object.entries(out).map(([type, best]) => [type, best.map((b) => b.words)]));

    test("names the measures it leads, lower or higher as each measure reads, ties within 1% shared", () => {
      assert.deepEqual(words(standouts(compared, made)), {
        dsd: [],
        box_behnken: [
          "precise coefficients",
          "joint precision",
          "least correlated terms",
          "power for main effects",
          "power for curvature",
        ],
        d_optimal: ["fewest runs", "joint precision", "least correlated terms"],
      });
    });

    test("a design that cannot fit stands out on nothing, nor does it set the best", () => {
      assert.equal(standouts(compared, made).full_factorial, undefined);
    });

    test("a measure every design shares, or one only a design knows, is not a lead", () => {
      const even = { ...compared, values: { ...compared.values } };
      for (const key of ["DSD [n=11]", "Box-Behnken [n=15]", "D-optimal [n=8]"]) {
        even.values[key] = { ...even.values[key], max_r: 0.25 };
      }
      even.values["DSD [n=11]"] = { ...even.values["DSD [n=11]"], power_interaction: 0.9 };
      const leads = Object.values(words(standouts(even, made))).flat();
      assert.ok(!leads.includes("least correlated terms"));
      assert.ok(!leads.includes("power for interactions"));
    });

    test("with one design that fits, none stands out", () => {
      const alone = { ...compared, values: { ...compared.values, "DSD [n=11]": value(11, false, 4, null) } };
      alone.values["Box-Behnken [n=15]"] = value(15, false, 8, null);
      assert.deepEqual(standouts(alone, made), { d_optimal: [] });
    });
  });

  test("the designs left out say why: the factor count they take, or numeric factors", () => {
    const leftOut = [
      { design_type: "box_behnken", label: "Box-Behnken", reason: "factors", low: 3, high: 7 },
      { design_type: "dsd", label: "DSD", reason: "factors", low: 3, high: 12 },
      { design_type: "taguchi", label: "Taguchi", reason: "factors", low: 4, high: 12 },
      { design_type: "ccd", label: "Central composite", reason: "categorical" },
      { design_type: "d_optimal", label: "D-optimal", reason: "limits" },
    ];
    assert.deepEqual(leftOutSentences(leftOut, 2, ["Catalyst"]), [
      "Box-Behnken and DSD are not listed: they take 3 factors or more, and the brief has 2.",
      "Taguchi is not listed: it takes 4 factors or more, and the brief has 2.",
      "Central composite is not listed: it needs numeric factors, and Catalyst is categorical.",
    ]);
    assert.deepEqual(leftOutSentences(leftOut.slice(0, 1), 9), [
      "Box-Behnken is not listed: it takes at most 7 factors, and the brief has 9.",
    ]);
    assert.deepEqual(leftOutSentences(leftOut.slice(3), 2, ["Catalyst", "Mill"]), [
      "Central composite is not listed: it needs numeric factors, and Catalyst and Mill are categorical.",
    ]);
    assert.deepEqual(leftOutSentences(leftOut.slice(3), 2), [], "no sentence on numeric factors for a brief without");
    assert.deepEqual(leftOutSentences([], 4), []);
  });

  test("a design that takes a number of runs opens on its own: ten a factor, or the model's coefficients and spare", () => {
    assert.equal(defaultBudget("maximin_lhs", 3, "interactions"), 3 * SPACE_FILLING_RUNS, "a space-filling design has no model");
    assert.equal(defaultBudget("d_optimal", 3, "interactions"), 7 + 4);
    assert.equal(defaultBudget("i_optimal", 4, "quadratic"), 15 + 4);
  });

  test("the Run tab counts the runs, then the results in", () => {
    assert.equal(runBadge([]), "");
    assert.equal(runBadge([null, null]), "2 runs");
    assert.equal(runBadge([1, null, 3]), "2 of 3");
  });
});

describe("limits on the settings", () => {
  const names = ["T", "P", "F"];

  test("a limit's key names each factor by its place, so a rename leaves it and an edit changes it", () => {
    const typed = "T + 25*P <= 250";
    assert.equal(positional(typed, names), "#0+25*#1<=250");
    const renamed = renameLimit(typed, { T: "Temp" });
    assert.equal(positional(renamed, ["Temp", "P", "F"]), positional(typed, names));
    assert.notEqual(positional("T + 25*P <= 240", names), positional(typed, names));
    assert.notEqual(positional(typed, ["P", "T", "F"]), positional(typed, names), "the factors' order is part of it");
  });

  test("the key reads spaces and the signs a phone types as the canonical text does", () => {
    assert.equal(positional("T+25·P ≤ 250", names), positional("T + 25 * P <= 250", names));
    assert.equal(positional("T^2 − P ≥ 1", names), positional("T ** 2 - P >= 1", names));
  });

  test("a function's name, a number's exponent and a longer name are not taken for a factor", () => {
    assert.equal(positional("sqrt(T) + 1e5*F + T1 <= 3", ["T", "F", "T1", "e5"]), "sqrt(#0)+1e5*#1+#2<=3");
    assert.equal(positional("log(F) <= 2", ["log", "F"]), "log(#1)<=2");
  });

  test("renaming swaps two names at once, and keeps the spacing and the functions as typed", () => {
    assert.equal(renameLimit("T + 25*P <= sqrt(T)", { T: "P", P: "T" }), "P + 25*T <= sqrt(P)");
    assert.equal(renameLimit("Tank_T  <=  T", { T: "Temp" }), "Tank_T  <=  Temp");
    assert.equal(renameLimit("exp(exp) <= 3", { exp: "E" }), "exp(E) <= 3", "a factor may share a function's name");
  });

  test("a bound box rebuilds the limit from its left side", () => {
    assert.equal(withBounds("T + 25 * P", null, 240), "T + 25 * P <= 240");
    assert.equal(withBounds("-T + 20 * P", -140, null), "-T + 20 * P >= -140");
    assert.equal(withBounds("3 * T + 5 * P", 400, 600), "400 <= 3 * T + 5 * P <= 600");
  });

  test("a name tapped in is set apart from a name or number beside it, and the caret goes after it", () => {
    assert.deepEqual(insertAt("T+", 2, 2, "P"), { text: "T+P", caret: 3 });
    assert.deepEqual(insertAt("T", 1, 1, "<="), { text: "T <=", caret: 4 });
    assert.deepEqual(insertAt("T + P <= 3", 4, 5, "F"), { text: "T + F <= 3", caret: 5 });
    assert.deepEqual(insertAt("", 0, 0, "T"), { text: "T", caret: 1 });
  });

  test("a share is never shown as 0% or 100% when it is not", () => {
    assert.deepEqual([0, 0.0004, 0.004, 0.126953125, 0.997, 0.9995, 1].map(shareText), [
      "none",
      "less than 0.1%",
      "0.4%",
      "13%",
      "99.7%",
      "more than 99.9%",
      "100%",
    ]);
  });

  test("a share and what it leaves sum to 100%, an exact half included", () => {
    assert.deepEqual([0.125, 0.875, 0.375, 0.625].map(shareText), ["12%", "88%", "38%", "62%"]);
    for (let k = 1; k < 2000; k += 1) {
      const [share, left] = [k / 2000, 1 - k / 2000].map(shareText);
      const sum = [share, left].map((text) => Number(text.replace("%", ""))).reduce((a, b) => a + b);
      if (!share.includes("than")) assert.equal(Math.round(sum * 10) / 10, 100, `${share} and ${left}`);
    }
  });

  const probes = { center: 8 };

  test("what a limit rules out: its share of the ranges, and the corners", () => {
    assert.equal(
      limitEffect({ removed: 0.127, removes: [6, 7], redundant: null }, probes),
      "Rules out 13% of your ranges: 2 of the 8 corners.",
    );
    assert.equal(
      limitEffect({ removed: 0.68, removes: [2, 3, 8], redundant: null }, probes),
      "Rules out 68% of your ranges: 2 of the 8 corners, and their center.",
    );
    assert.equal(limitEffect({ removed: 0.04, removes: [], redundant: null }, probes), "Rules out 4% of your ranges: none of the 8 corners.");
    assert.equal(limitEffect({ removed: 0.1, removes: null, redundant: null }, null), "Rules out 10% of your ranges.");
  });

  test("a limit that rules out nothing says so, and one that cannot be read says nothing here", () => {
    assert.equal(limitEffect({ removed: 0, removes: [], redundant: "alone" }, probes), "Rules out nothing within your ranges.");
    assert.equal(limitEffect({ removed: 0.05, removes: [7], redundant: "others" }, probes), "Rules out nothing the other limits do not.");
    assert.equal(limitEffect({ expression: "T +", problem: "The limit 'T +' cannot be read." }, probes), "");
  });

  test("a limit that rules out corners never rules out none, however thin a part the sample sees", () => {
    // T * F <= 3990 cuts about 0.003% off the corners where T and F are high: too little for the fixed sample.
    assert.equal(
      limitEffect({ removed: 0, removes: [5, 7], redundant: null }, probes),
      "Rules out less than 0.1% of your ranges: 2 of the 8 corners.",
    );
  });

  const factors = [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "P", low: 1, high: 3, units: "bar" },
  ];
  const reply = (change) => ({
    constraints: [{ expression: "T + 25 * P <= 215" }],
    kept: 0.31689453125,
    empty: false,
    center_allowed: true,
    bounds: [],
    refusal: null,
    ...change,
  });

  test("what all the limits leave: the share, the center, and each range narrowed, in its units", () => {
    const said = limitsSummary(
      reply({
        center_allowed: false,
        bounds: [
          { name: "T", low: null, high: 190 },
          { name: "P", low: 1.25, high: 2.6 },
        ],
      }),
      factors,
    ).map((s) => s.text);
    assert.deepEqual(said, [
      "The limits leave 32% of your ranges to run.",
      "The center of the ranges cannot be run.",
      "So T never goes above 190 degC.",
      "So P stays between 1.25 and 2.6 bar.",
    ]);
    const below = limitsSummary(reply({ bounds: [{ name: "T", low: 155.55555, high: null }] }), factors);
    assert.equal(below.at(-1).text, "So T never goes below 155.6 degC.");
  });

  test("a refusal comes first, as an error, and nothing left says only that", () => {
    const refusal = "Limits on the settings work with numeric factors only on this page, and S is categorical.";
    assert.deepEqual(limitsSummary(reply({ refusal }), factors)[0], { text: refusal, error: true });
    const problem = "The limit 'Z <= 3' names Z, which is not one of your factors: T and P.";
    const own = reply({ constraints: [{ expression: "Z <= 3", problem }, { expression: "T <= 190" }], refusal: problem });
    assert.deepEqual(limitsSummary(own, factors)[0], { text: "The limits leave 32% of your ranges to run.", error: false }, "said under the limit, not twice");
    const empty = limitsSummary(reply({ empty: true, kept: 0, refusal: "No setting in your ranges meets every limit." }), factors);
    assert.deepEqual(empty, [{ text: "No setting in your ranges meets every limit.", error: true }]);
    assert.deepEqual(limitsSummary(reply({ constraints: [] }), factors), []);
  });

  test("what the limits leave is never all while one rules something out, nor none while a setting can be run", () => {
    const leave = (change) => limitsSummary(reply(change), factors)[0].text;
    const sliver = [{ expression: "T * F <= 3990", redundant: null }];
    assert.equal(leave({ constraints: sliver, kept: 1 }), "The limits leave more than 99.9% of your ranges to run.");
    const nothing = [{ expression: "T <= 200", redundant: "alone" }];
    assert.equal(leave({ constraints: nothing, kept: 1 }), "The limits leave 100% of your ranges to run.");
    assert.equal(leave({ kept: 0 }), "The limits leave less than 0.1% of your ranges to run.");
  });
});

describe("the table of corners", () => {
  const factors = [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "P", low: 1, high: 3, units: "bar" },
    { name: "F", low: 10, high: 20, units: "" },
  ];
  const columns = ["T", "P", "F"];
  const rows = [
    [150, 1, 10],
    [150, 1, 20],
    [150, 3, 10],
    [150, 3, 20],
    [200, 1, 10],
    [200, 1, 20],
    [200, 3, 10],
    [200, 3, 20],
  ];

  test("a setting is named factor by factor, in its units, as a screen reader says it", () => {
    assert.equal(settingLabel([200, 3, 20], columns, factors), "T 200 degC, P 3 bar, F 20");
    assert.equal(settingLabel([175, 2.0000001, 15], columns, factors), "T 175 degC, P 2 bar, F 15");
  });

  test("the filters leave the corners at the ends asked for", () => {
    assert.deepEqual(filterCorners(rows, columns, factors, { T: "high" }), [4, 5, 6, 7]);
    assert.deepEqual(filterCorners(rows, columns, factors, { T: "high", F: "low" }), [4, 6]);
    assert.deepEqual(filterCorners(rows, columns, factors, {}), [0, 1, 2, 3, 4, 5, 6, 7]);
  });

  test("a cut is asked from the corner's end of the factor", () => {
    assert.equal(cutWords({ name: "T", at: 200, cut: 175 }, factors[0]), "can go up to");
    assert.equal(cutWords({ name: "T", at: 150, cut: 175 }, factors[0]), "must be at least");
  });

  test("a proposal flags the corners it rules out that were not marked, unless a limit rules them out already", () => {
    const ruledOutBy = [[], [], [], [], [], [], [], [0], []];
    assert.deepEqual(alsoRuledOut({ removes: [5, 6, 7] }, [6], ruledOutBy), [5]);
    assert.deepEqual(alsoRuledOut({ problem: "no cut" }, [6], ruledOutBy), []);
  });

  test("corners are named one by one, or by the face of the ranges they make up", () => {
    const probes = { columns, rows: [...rows, [175, 2, 15]], center: 8 };
    const words = (marked) => cornersWords(marked, probes, factors);
    assert.equal(words([7]), "the corner T 200 degC, P 3 bar, F 20");
    assert.equal(words([6, 7]), "the 2 corners where T is 200 degC and P is 3 bar");
    assert.equal(words([4, 5, 6, 7, 8]), "the 4 corners where T is 200 degC, and the center of the ranges");
    assert.equal(words([0, 1, 2, 3, 4, 5, 6, 7]), "every corner");
    assert.equal(words([8]), "the center of the ranges");
    assert.equal(
      words([3, 6, 7]),
      "the 3 corners (T 150 degC, P 3 bar, F 20; T 200 degC, P 3 bar, F 10; T 200 degC, P 3 bar, F 20)",
      "three corners that are not a face are named one by one",
    );
    assert.equal(words([0, 3, 5, 6]), "the 4 corners (T 150 degC, P 1 bar, F 10; T 150 degC, P 3 bar, F 20; and 2 more)");
    assert.equal(words([0, 7]), "the 2 corners (T 150 degC, P 1 bar, F 10; T 200 degC, P 3 bar, F 20)");
  });
});

describe("several responses", () => {
  const YIELD = { name: "Yield", label: "Yield", units: "%", goal: "maximize", typical: 62 };
  const COST = { name: "Cost", label: "Cost", units: "$/kg", goal: "minimize" };
  const PURITY = { name: "Purity", label: "Purity", units: "%", goal: "target", target: 99 };

  test("a study made before measures the one response it names, described as it was", () => {
    const old = { response: "Yield", response_info: { label: "Yield", units: "%" } };
    assert.deepEqual(responsesOf(old), [{ name: "Yield", label: "Yield", units: "%" }]);
    assert.deepEqual(responsesOf({ response: "y" }), [{ name: "y" }]);
    assert.deepEqual(responsesOf({ response: "Yield", responses: [YIELD, COST] }), [YIELD, COST]);
  });

  test("one response is stored as before, several as a list beside the first", () => {
    const { name: _, ...info } = YIELD;
    assert.deepEqual(storedResponses([YIELD]), { response: "Yield", response_info: info });
    assert.deepEqual(storedResponses([{ name: "y" }]), { response: "y" }, "no description, none kept");
    assert.deepEqual(storedResponses([YIELD, COST]), { response: "Yield", response_info: info, responses: [YIELD, COST] });
  });

  test("a specification takes new responses, and loses the list when one is left", () => {
    const spec = { design_type: "ccd", response: "Yield", response_info: { label: "Yield" }, responses: [YIELD, COST], model: "quadratic" };
    const one = withResponses(spec, [{ name: "Yield", label: "Yield", units: "%" }]);
    assert.deepEqual(one, { design_type: "ccd", model: "quadratic", response: "Yield", response_info: { label: "Yield", units: "%" } });
    assert.deepEqual(withResponses(one, [YIELD, COST, PURITY]).responses, [YIELD, COST, PURITY]);
  });

  test("the most responses a study measures", () => {
    assert.equal(MAX_RESPONSES, 5);
  });

  test("a run is complete when every response has its result", () => {
    const results = { Yield: [60, null, 61, 62], Cost: [9, 8, null, 7] };
    assert.deepEqual(completeRuns(results, ["Yield", "Cost"]), [true, false, false, true]);
    assert.deepEqual(completeRuns(results, ["Yield"]), [true, false, true, true]);
    assert.deepEqual(partialRuns(results, ["Yield", "Cost"]), [1, 2], "runs with some of their results");
    assert.deepEqual(partialRuns({ Yield: [null, 1], Cost: [null, 2] }, ["Yield", "Cost"]), [], "a run with none is to do");
    assert.equal(anyResult({ Yield: [null], Cost: [null] }), false);
    assert.equal(anyResult({ Yield: [null], Cost: [0] }), true, "a result of 0 is a result");
  });

  test("the results go to Python as a study of one response sends them, the others beside", () => {
    const results = { Yield: [60, 61], Cost: [9, null] };
    assert.deepEqual(splitResults(["Yield"], results), { responses: [60, 61] });
    assert.deepEqual(splitResults(["Yield", "Cost"], results), { responses: [60, 61], more_responses: { Cost: [9, null] } });
    const marks = { Yield: ["entered", "simulated"], Cost: ["entered", null] };
    assert.deepEqual(splitResults(["Yield", "Cost"], marks, "sources"), {
      sources: ["entered", "simulated"],
      more_sources: { Cost: ["entered", null] },
    });
    assert.deepEqual(joinResults(["Yield", "Cost"], [60, 61], { Cost: [9, null] }), results);
    assert.deepEqual(joinResults(["Yield", "Cost"], [60, 61]), { Yield: [60, 61], Cost: [null, null] }, "no results yet");
  });

  test("a study's title names its responses", () => {
    assert.equal(responsesTitle([YIELD]), "Yield (%), maximise");
    assert.equal(responsesTitle([YIELD, COST, PURITY]), "Yield, Cost and Purity");
    assert.equal(responsesTitle([YIELD, { name: "Cost_2" }]), "Yield and Cost_2", "a response not described, by its column");
  });

  test("two responses cannot share a name, whatever its case", () => {
    assert.deepEqual(responseLabelProblems(["Yield", "Cost", "Purity"]), []);
    assert.deepEqual(responseLabelProblems(["Yield", "", "  "]), [], "blank rows are no responses yet");
    assert.deepEqual(responseLabelProblems(["Yield", "Cost", "cost"]), ["cost: two responses cannot share a name."]);
    assert.deepEqual(responseLabelProblems(["Cost", "Cost", "Cost"]), ["Cost: two responses cannot share a name."], "said once");
  });

  test("each response's column is named from its words, unlike the factors and the responses before it", () => {
    assert.deepEqual(responseColumns(["Yield", "Cost ($/kg)", "Purity"], ["T", "P"]), ["Yield", "Cost_kg", "Purity"]);
    assert.deepEqual(responseColumns(["Yield", "yield", "Yield"], ["T"]), ["Yield", "yield", "Yield_2"]);
    assert.deepEqual(responseColumns(["", ""], ["T"]), ["y", "y_2"]);
    assert.deepEqual(responseColumns(["T", "Block"], ["T"]), ["T_2", "Block_2"]);
  });

  test("a response on the run sheet keeps its column, which no new one takes", () => {
    const kept = ["Yield", undefined, "Cost"];
    assert.deepEqual(responseColumns(["Yield of A", "Cost", "Costs"], ["T"], kept), ["Yield", "Cost_2", "Cost"]);
  });

  test("Learn says how a run with some of its results is fitted", () => {
    assert.equal(
      partialSentence([3], "Cost", 14),
      "Run 3 has some of its results, not all. Each response is fitted to the runs that have its result: Cost to 14.",
    );
    assert.equal(
      partialSentence([3, 7, 9], "Yield", 12),
      "Runs 3, 7 and 9 have some of their results, not all. Each response is fitted to the runs that have its result: Yield to 12.",
    );
  });
});

describe("a mixture's components", () => {
  test("the mixture designs, and no other, read the factors as a blend's components", () => {
    assert.deepEqual(MIXTURES.filter(isMixture), MIXTURES);
    assert.equal(isMixture("box_behnken"), false);
    assert.equal(isMixture(undefined), false);
  });

  test("a blank bound is 0 for the lower proportion and 1 for the upper; anything else is read as a level", () => {
    assert.equal(proportionValue("", 0), 0);
    assert.equal(proportionValue("  ", 1), 1);
    assert.equal(proportionValue("0.25", 0), 0.25);
    assert.equal(proportionValue("most", 1), "most");
  });

  test("each component's bounds are proportions, the lower below the upper", () => {
    assert.equal(componentProblem({ name: "A", low: "", high: "" }), "");
    assert.equal(componentProblem({ name: "A", low: "0.1", high: "0.6" }), "");
    assert.match(componentProblem({ name: "A", low: "", high: "most" }), /^A: enter its lower and upper proportions as numbers/);
    assert.match(componentProblem({ name: "T", low: "150", high: "200" }), /^T runs from 150 to 200, which is no proportion/);
    assert.equal(componentProblem({ name: "A", low: "0.6", high: "0.4" }), "A: its lower proportion must be below its upper proportion.");
    assert.match(componentProblem({ name: "", low: "inf", high: "" }), /^A component without a name:/);
  });

  test("together, the bounds leave more than one blend, and suit the design type", () => {
    const blank = ["A", "B", "C"].map((name) => ({ name, low: 0, high: 1 }));
    const bounded = [{ ...blank[0], low: 0.2 }, blank[1], { ...blank[2], high: 0.5 }];
    assert.deepEqual(boundedComponents(bounded), ["A", "C"]);
    assert.equal(mixtureProblem(blank, "simplex_lattice", "Simplex lattice (mixture)"), "");
    assert.equal(mixtureProblem(bounded, "extreme_vertices", "Extreme vertices"), "");
    assert.match(mixtureProblem(bounded, "simplex_lattice", "Simplex lattice (mixture)"), /A and C have bounds: choose extreme vertices/);
    assert.match(mixtureProblem(blank, "extreme_vertices", "Extreme vertices"), /^Extreme vertices are for a mixture with bounds/);
    const heavy = blank.map((f) => ({ ...f, low: 0.4 }));
    assert.match(mixtureProblem(heavy, "extreme_vertices", ""), /^The lower proportions sum to 1.2, and must sum to less than 1/);
    const light = blank.map((f) => ({ ...f, high: 1 / 3 }));
    assert.match(mixtureProblem(light, "extreme_vertices", ""), /^The upper proportions sum to 1, and must sum to more than 1/);
  });
});

describe("results pasted from a spreadsheet", () => {
  // Five runs, listed as made but sorted on the sheet by their Std order: row 0 is run 3, row 1 run 1, and so on.
  const sheet = {
    runs: [3, 1, 5, 2, 4],
    order: [1, 3, 0, 4, 2],
    names: ["y", "cost"],
    labels: { y: "Yield", cost: "Cost" },
  };
  const paste = (text, at = { index: 1, name: "y" }) => placePasted(pastedCells(text), sheet, at);
  const of = (placed) => placed.values.map(({ index, name, value }) => `${name}[${index}]=${value}`);

  test("a block is a line a row and a tab between cells, as a spreadsheet copies it", () => {
    assert.deepEqual(pastedCells("1\t2\r\n3\t 4 \n\n"), [["1", "2"], ["3", "4"]]);
  });

  test("a cell reads as a number however a spreadsheet writes it, a blank as none", () => {
    assert.deepEqual(["2.5", "2,5", " 1 234,5 ", "\u22121.5", "1e3", "", "n/a", "1,234.5", "1,2,3"].map(pastedNumber), [
      2.5, 2.5, 1234.5, -1.5, 1000, null, undefined, undefined, undefined,
    ]);
  });

  test("a column goes in place, down the sheet as it is shown, from the box pasted into", () => {
    const placed = paste("61.2\n62,5\n63");
    assert.deepEqual(of(placed), ["y[1]=61.2", "y[3]=62.5", "y[0]=63"]);
    assert.equal(placed.said, "Pasted 3 results of Yield, down the run sheet from run 1.");
  });

  test("two columns go across the responses from the one pasted into, and a blank cell leaves a result as it is", () => {
    assert.deepEqual(of(paste("61\t9\n\t8")), ["y[1]=61", "cost[1]=9", "cost[3]=8"]);
  });

  test("a first column of run numbers matches each line to its run, whatever the order", () => {
    const placed = paste("5\t70\n2\t71");
    assert.deepEqual(of(placed), ["y[2]=70", "y[3]=71"]);
    assert.equal(placed.said, "Pasted 2 results of Yield, matched to the runs by the run numbers in the first column.");
  });

  test("headings name the columns: Run order matches the runs, a response's name or column its results", () => {
    const placed = paste("Std order\tRun order\tT\tCost ($/kg)\tYield [%]\n1\t4\t150\t9.5\t60\n2\t1\t200\t8.5\t65");
    assert.deepEqual(of(placed), ["cost[4]=9.5", "y[4]=60", "cost[1]=8.5", "y[1]=65"]);
    assert.equal(placed.said, "Pasted 4 results of Yield and Cost, matched to the runs by the Run order column.");
    assert.deepEqual(of(paste("y\n50\n51", { index: 0, name: "cost" })), ["y[0]=50", "y[4]=51"], "no Run order: in place");
  });

  test("a block that does not fit is refused whole, saying where", () => {
    assert.equal(
      paste("1\n2\n3\n4\n5\n6").problem,
      "You pasted 6 lines, and the run sheet shows 5 runs from the one pasted into down. Nothing was pasted.",
    );
    assert.equal(
      paste("61\t2\t3").problem,
      "You pasted 3 columns of results, and the run sheet has 2 from Yield on. Nothing was pasted.",
    );
    assert.equal(
      paste("1\t60\t9\t2").problem,
      "You pasted 3 columns of results, and the run sheet has 2 from Yield on. Nothing was pasted.",
      "after a first column of run numbers",
    );
    assert.equal(paste("Run\tYield\n9\t60").problem, "Line 2 of what you pasted is for run 9, which the run sheet does not have. Nothing was pasted.");
    assert.equal(paste("Run\tYield\n2\t60\n2\t61").problem, "Line 3 of what you pasted is for run 2 again. Nothing was pasted.");
    assert.equal(paste("60\nabout 61").problem, 'Line 2 of what you pasted holds "about 61", which is not a number. Nothing was pasted.');
    assert.equal(
      paste("Temperature\tPressure\n1\t2").problem,
      "No heading of what you pasted names a response of the run sheet: Yield and Cost. Nothing was pasted.",
    );
  });

  test("run numbers that are not all on the sheet read as results instead", () => {
    assert.deepEqual(of(paste("7\t1\n8\t2")), ["y[1]=7", "cost[1]=1", "y[3]=8", "cost[3]=2"]);
  });

  test("a block of blanks pastes nothing, and says so", () => {
    assert.deepEqual(paste("\t\n\t"), { values: [], said: "Nothing to paste: every cell was blank." });
    assert.deepEqual(paste(""), { values: [], said: "Nothing to paste: every cell was blank." });
  });
});
