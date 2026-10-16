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
  defaultResolution,
  escape,
  factorType,
  fitStatistics,
  fmt,
  GOALS,
  OPTIMAL_MODEL,
  SCREEN_FROM,
  aimTerms,
  analysisModel,
  candidateRows,
  changeWords,
  defaultAim,
  describeResponse,
  holds,
  nameProblem,
  percent,
  pickCandidate,
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
  sortNumber,
  termLabel,
  withUnits,
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
    assert.deepEqual([0, 0.0004, 0.004, 0.126953125, 0.9995, 1].map(shareText), [
      "none",
      "less than 0.1%",
      "0.4%",
      "13%",
      "more than 99.9%",
      "100%",
    ]);
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
