// web/best.js: what Learn says of the best settings for the response's goal, and the run sheet's confirmation runs.
// The facts come from bootstrap.best_settings; tests/web/api/test_best_settings.py holds them to the model.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BLEND_SENSITIVITY,
  BLEND_UNCLEAR,
  CONFIRMATION,
  NO_BLEND_OVERLAY,
  CONFIRMATION_RUNS,
  CONFIRMS,
  GOOD_END,
  IMPORTANCE,
  IMPORTANCES,
  SHAPE,
  UNCLEAR,
  balanceConfirmation,
  balanceLead,
  balanceLeverage,
  balanceRows,
  balanceWhere,
  blendOverlayCaption,
  dWords,
  goalProblem,
  goalWords,
  limitingSentence,
  missedBoundaries,
  overlayCaption,
  ascentSentence,
  bestContourCaption,
  blendExtrapolation,
  blendLead,
  blendSentences,
  blendWhere,
  confirmationRows,
  confirmationSentences,
  confirmedAt,
  apartRows,
  failedRows,
  withFailed,
  designRows,
  pathRows,
  withPathRuns,
  withoutPathRuns,
  extrapolationSentence,
  flatSentence,
  intervalRows,
  leadSentence,
  answerWords,
  learnChecks as checksOf,
  movesOn,
  nextRanges,
  noGoalSentence,
  noIntervalSentence,
  num,
  sensitivityCaption,
  settingWords,
  weighingSentence,
  whereSentences,
  whereWords,
  withConfirmationRuns,
  withUnit,
  withoutConfirmationRuns,
  WHAT_IF_BACK,
  columnsAt,
  keptAs,
  predictAt,
  putBack,
  rangeWords,
  scenarioProblem,
  settingsFit,
  settingsWords,
  sliderStep,
  whatIfRows,
  PROFILE_POINTS,
  MAP_POINTS,
  mapPair,
  surfaceOf,
  profileOf,
  runsAt,
  insideWindow,
  dotPiles,
  runsWords,
  profilePieces,
  profileRange,
  profileWords,
  whatIfSentences,
  whatIfStart,
} from "../../../web/best.js";
import { findingWords, fmt, pval, shareWords as wholeShare } from "../../../web/logic.js";
import { CONFIRMATION as PYTHON_CONFIRMATION, fittedRuns } from "../../../web/python.js";

const WORDS = { label: "Yield", units: "%" };
const FACTORS = [
  { name: "T", low: 150, high: 200, units: "degC" },
  { name: "P", low: 1, high: 3, units: "bar" },
  { name: "F", low: 10, high: 20, units: "L/min" },
];

/** The worked example's best settings, as bootstrap.best_settings gives them. */
const WORKED = {
  goal: "maximize",
  target: null,
  flat: false,
  predicted: 68.2522,
  reached: null,
  level: 0.95,
  confidence: [66.1728, 70.3316],
  prediction: [65.2526, 71.2518],
  leverage: 0.9252,
  max_leverage: 0.75,
  extrapolating: true,
  settings: [
    { name: "T", kind: "numeric", value: 200, coded: 1, units: "degC", in_model: true, end: "high", beyond: true },
    { name: "P", kind: "numeric", value: 1, coded: -1, units: "bar", in_model: true, end: "low", beyond: true },
    { name: "F", kind: "numeric", value: 18.0372, coded: 0.6074, units: "L/min", in_model: true, end: null, beyond: false },
  ],
  coded: { T: 1, P: -1, F: 0.6074 },
  limits: [],
  stationary: { factors: ["F"], inside: false, kind: "maximum" },
  ascent: { T: 1, P: -0.6712 },
  unclear: false,
  sensitivity: [
    { name: "T", step: 2.5, down: -0.2466, up: 0.2467, down_runnable: true, up_runnable: false, in_model: true },
    { name: "P", step: 0.1, down: 0.1639, up: -0.1672, down_runnable: false, up_runnable: true, in_model: true },
    { name: "F", step: 0.5, down: -0.0182, up: -0.0181, down_runnable: true, up_runnable: true, in_model: true },
  ],
  confirm_at: { T: 200, P: 1, F: 18.04 },
  confirmation: [],
};

describe("numbers and settings in a sentence", () => {
  test("four significant figures, without trailing zeros", () => {
    assert.equal(num(200), "200");
    assert.equal(num(1.0000000000000004), "1");
    assert.equal(num(18.037189919), "18.04");
    assert.equal(num(0.92524096), "0.9252");
  });

  test("units after the number, a percent sign against it", () => {
    assert.equal(withUnit(200, "degC"), "200 degC");
    assert.equal(withUnit(68.25220041, "%"), "68.25%");
    assert.equal(withUnit(3), "3");
  });

  test("a setting, numeric or a level", () => {
    assert.equal(settingWords(WORKED.settings[0], "degC"), "T = 200 degC");
    assert.equal(settingWords({ name: "Supplier", kind: "categorical", value: "B" }), "Supplier at B");
  });

  test("where each setting lies, in the table's words: at an end, and better past it; or at its peak inside", () => {
    assert.deepEqual(
      WORKED.settings.map((s) => whereWords(s, WORKED, WORDS)),
      ["high end; a higher Yield may lie above", "low end; a higher Yield may lie below", "inside its range, at its peak"],
    );
    const minimum = { ...WORKED, goal: "minimize", stationary: null };
    assert.equal(whereWords(WORKED.settings[0], minimum, WORDS), "high end; a lower Yield may lie above");
    assert.equal(whereWords(WORKED.settings[2], minimum, WORDS), "inside its range");
    assert.equal(whereWords({ ...WORKED.settings[0], beyond: false }, WORKED, WORDS), "high end of its range");
    assert.equal(whereWords({ ...WORKED.settings[2], in_model: false }), "not in the model");
    assert.equal(whereWords({ name: "S", kind: "categorical", value: "B", in_model: true }), "the better level");
  });
});

describe("the answer Learn opens with", () => {
  test("where to run, and what the model expects there, in the panel's own numbers", () => {
    assert.deepEqual(answerWords(WORKED, WORDS), {
      run: "Run at T = 200 degC, P = 1 bar and F = 18.04 L/min.",
      expect: "The model predicts the highest Yield there: 68.25% (95% confidence interval for the mean: 66.17 to 70.33%).",
    });
    const [interval] = intervalRows(WORKED, WORDS);
    assert.ok(answerWords(WORKED, WORDS).expect.includes(`${interval.label}: ${interval.value}`), "the panel's row, to the digit");
    assert.match(answerWords({ ...WORKED, goal: "minimize" }, WORDS).expect, /^The model predicts the lowest Yield there: 68\.25%/);
  });

  test("a target reached, or approached as closely as the ranges allow", () => {
    const target = { ...WORKED, goal: "target", target: 60, predicted: 60.00000003, reached: true, confidence: [58.1, 61.9] };
    assert.equal(
      answerWords(target, WORDS).expect,
      "The model predicts Yield on its target there: 60% (95% confidence interval for the mean: 58.1 to 61.9%).",
    );
    const far = { ...WORKED, goal: "target", target: 90, reached: false };
    assert.equal(
      answerWords(far, WORDS).expect,
      "No setting in your ranges reaches the target of 90%: these come closest, at 68.25% " +
        "(95% confidence interval for the mean: 66.17 to 70.33%).",
    );
  });

  test("a categorical factor is at its level, and a fit with no runs to spare gives no interval", () => {
    const supplier = { name: "Supplier", kind: "categorical", value: "B", in_model: true };
    const said = answerWords({ ...WORKED, settings: [WORKED.settings[0], supplier], confidence: null }, WORDS);
    assert.deepEqual(said, { run: "Run at T = 200 degC and Supplier at B.", expect: "The model predicts the highest Yield there: 68.25%." });
  });

  test("a mixture's answer is its blend", () => {
    const share = (name, value) => ({ name, kind: "numeric", value, in_model: true });
    const blend = { ...WORKED, settings: [share("A", 0.5), share("B", 0.25), share("C", 0.25)], goal: "target", target: 90, reached: false };
    const said = answerWords(blend, { label: "Hardness", units: "" }, { blend: true });
    assert.equal(said.run, "Blend A = 0.5, B = 0.25 and C = 0.25.");
    assert.match(said.expect, /^No blend within your bounds reaches the target of 90: this one comes closest, at 68\.25 /);
  });

  test("nothing to state without settings: a goal of learning, a target not typed, a flat model", () => {
    assert.equal(answerWords({ goal: "learn", settings: null }, WORDS), null);
    assert.equal(answerWords({ goal: "maximize", flat: true, settings: null }, WORDS), null);
    assert.equal(answerWords(null, WORDS), null);
  });
});

describe("the checks under the answer", () => {
  const ANALYSIS = {
    summary: { r_squared: 0.9927, r_squared_adj: 0.9862, r_squared_pred: 0.9705, df_residual: 8 },
    findings: [{ kind: "fit", explained: 0.9927, unseen: 0.9705 }],
    diagnostics: Array.from({ length: 15 }, (_, i) => ({ run: i + 1, unusual: false })),
    lack_of_fit: { f_statistic: 0.8124, p_value: 0.6012, df_pure_error: 2, significant: false },
  };
  const by = (checks) => Object.fromEntries(checks.map((check) => [check.key, check]));
  /** The page's words for an analysis, as app.js hands them over: best.js stands alone and imports none of them. */
  const told = (analysis) => ({
    said: (kind) => findingWords((analysis.findings ?? []).filter((finding) => finding.kind === kind), WORDS)[0] ?? "",
    number: fmt,
    p: pval,
    share: wholeShare,
  });
  const learnChecks = (analysis, best, _words, options) => checksOf(analysis, best, told(analysis), options);

  test("four checks, each a verdict, a sentence and its statistic", () => {
    const checks = learnChecks(ANALYSIS, { ...WORKED, extrapolating: false }, WORDS);
    assert.deepEqual(checks.map((check) => [check.key, check.state]), [["fit", "ok"], ["runs", "ok"], ["repeats", "ok"], ["reach", "ok"]]);
    const { fit, runs, repeats, reach } = by(checks);
    assert.equal(fit.title, "The model explains 99%");
    assert.equal(fit.says, "The model explains 99% of the variation in Yield, and about 97% in runs it has not seen.");
    assert.equal(fit.numbers, "R² = 0.9927, Adjusted R² = 0.9862, Predicted R² = 0.9705");
    assert.deepEqual([runs.title, runs.numbers], ["No run stands apart", "15 runs compared"]);
    assert.equal(repeats.title, "The repeats agree with the model");
    assert.equal(repeats.numbers, "Lack of fit: F = 0.8124, p = 0.6012; pure error from the repeated runs, 2 degrees of freedom");
    assert.deepEqual([reach.title, reach.numbers], ["Within the runs' reach", "Leverage 0.93; the runs' is 0.75 at most"]);
  });

  test("each caution Python raises is a check to look at, in the words the page already uses", () => {
    const analysis = {
      ...ANALYSIS,
      findings: [...ANALYSIS.findings, { kind: "lack_of_fit" }, { kind: "unusual", runs: [7] }],
      lack_of_fit: { ...ANALYSIS.lack_of_fit, p_value: 0.012, significant: true },
    };
    const { fit, runs, repeats, reach } = by(learnChecks(analysis, WORKED, WORDS));
    assert.equal(fit.state, "ok");
    assert.deepEqual([runs.state, runs.title], ["look", "Run 7 stands apart"]);
    assert.equal(runs.says, "Run 7's result is far from what the model expects: check before trusting the fit.");
    assert.deepEqual([repeats.state, repeats.title], ["look", "A term may be missing"]);
    assert.deepEqual([reach.state, reach.title, reach.says], ["look", "Beyond the runs", extrapolationSentence(WORKED)]);
    const several = { ...analysis, findings: [{ kind: "unusual", runs: [2, 7] }] };
    assert.equal(by(learnChecks(several, WORKED, WORDS)).runs.title, "2 runs stand apart");
  });

  test("no clear effect is said of the fit, from the findings or from the best settings", () => {
    const none = { ...ANALYSIS, findings: [{ kind: "none" }] };
    const { fit } = by(learnChecks(none, null, WORDS));
    assert.deepEqual([fit.state, fit.title, fit.says], ["look", "No clear effect", "No factor's effect on Yield stands out from the noise."]);
    assert.equal(by(learnChecks(ANALYSIS, { ...WORKED, unclear: true }, WORDS)).fit.says, UNCLEAR);
    assert.equal(by(learnChecks(ANALYSIS, { ...WORKED, unclear: true }, WORDS, { blend: true })).fit.says, BLEND_UNCLEAR);
  });

  test("what the runs cannot tell is said so: a model that uses every run, and no repeats", () => {
    const saturated = {
      summary: { r_squared: 1, r_squared_adj: null, r_squared_pred: null, df_residual: 0 },
      findings: [{ kind: "saturated", lenth: true }],
      diagnostics: null,
      lack_of_fit: null,
    };
    const checks = learnChecks(saturated, null, WORDS);
    assert.deepEqual(checks.map((check) => [check.key, check.state]), [["fit", "unknown"], ["runs", "unknown"], ["repeats", "unknown"]]);
    assert.match(by(checks).fit.says, /^The model uses every run, so none is left to measure the noise with/);
    assert.equal(by(checks).repeats.title, "No repeated runs");
  });

  test("with no settings to check there is no check of their reach", () => {
    assert.deepEqual(learnChecks(ANALYSIS, { goal: "learn", settings: null }, WORDS).map((check) => check.key), ["fit", "runs", "repeats"]);
  });
});

describe("what the panel says first", () => {
  test("the prediction, as the goal's best", () => {
    assert.equal(leadSentence(WORKED, WORDS), "These settings give the highest Yield the model predicts: 68.25%.");
    assert.equal(leadSentence({ ...WORKED, goal: "minimize" }, WORDS), "These settings give the lowest Yield the model predicts: 68.25%.");
  });

  test("a target reached, or approached as closely as the ranges allow", () => {
    const target = { ...WORKED, goal: "target", target: 60, predicted: 60.00000003, reached: true };
    assert.equal(leadSentence(target, WORDS), "These settings give Yield on its target, 60%, as the model predicts.");
    const far = { ...WORKED, goal: "target", target: 90, reached: false };
    assert.equal(leadSentence(far, WORDS), "No setting in your ranges reaches the target of 90%. These come closest: 68.25%.");
  });

  test("the two intervals and the leverage, a row each, each label naming its definition", () => {
    assert.deepEqual(intervalRows(WORKED, WORDS), [
      {
        label: "95% confidence interval for the mean",
        term: "confidence interval",
        key: "confidence-interval",
        value: "66.17 to 70.33%",
      },
      {
        label: "95% prediction interval for one new run",
        term: "prediction interval",
        key: "prediction-interval",
        value: "65.25 to 71.25%",
      },
      { label: "Leverage", term: "Leverage", key: "leverage", value: "0.93, above the runs': 0.75 at most" },
    ]);
    for (const row of intervalRows(WORKED, WORDS)) assert.ok(row.label.includes(row.term), row.label);
    assert.equal(noIntervalSentence(WORKED), "");
  });

  test("with no runs left for the noise, no interval, and the page says why", () => {
    const saturated = { ...WORKED, confidence: null, prediction: null, extrapolating: false, leverage: 1, max_leverage: 1 };
    assert.deepEqual(intervalRows(saturated, WORDS).map((row) => row.value), ["1, within the runs': 1 at most"]);
    assert.match(noIntervalSentence(saturated), /^No runs are left to estimate the noise/);
  });

  test("without a goal, one line that asks for one", () => {
    assert.equal(
      noGoalSentence({ goal: "learn" }, WORDS),
      "Set a goal for Yield in the brief, to maximise, minimise or hit a target, and the page finds the best settings here.",
    );
    assert.match(noGoalSentence({ goal: "target", target: null }, WORDS), /^Type the target for Yield in the brief/);
    assert.equal(flatSentence(WORDS), "No term of the model changes Yield, so no setting is better than another.");
  });

  test("of a mixture, the same lines name the blend", () => {
    const blend = { blend: true };
    assert.match(noGoalSentence({ goal: "learn" }, WORDS, blend), /and the page finds the best blend here\.$/);
    assert.match(noGoalSentence({ goal: "target", target: null }, WORDS, blend), /finds the blend that reaches it\.$/);
    assert.equal(flatSentence(WORDS, blend), "The model predicts the same Yield for every blend, so no blend is better than another.");
  });
});

describe("where the best settings lie", () => {
  test("two factors at the ends of their ranges: the way a next study could move them", () => {
    assert.deepEqual(whereSentences(WORKED, FACTORS, WORDS), [
      "A next study could move the ranges along the steepest ascent: T up 25 degC, P down 0.67 bar.",
    ]);
  });

  test("one factor beyond: the next study moves its range", () => {
    const one = {
      ...WORKED,
      settings: [WORKED.settings[0], { ...WORKED.settings[1], end: null, beyond: false }, WORKED.settings[2]],
      ascent: { T: 1 },
    };
    assert.deepEqual(whereSentences(one, FACTORS, WORDS), ["A next study could move the range of T up."]);
  });

  test("nothing past an end, no way to move", () => {
    const flat = { ...WORKED, settings: WORKED.settings.map((s) => ({ ...s, beyond: false })) };
    assert.deepEqual(whereSentences(flat, FACTORS, WORDS), []);
  });

  test("a minimum's way is the steepest descent", () => {
    const low = { ...WORKED, goal: "minimize", ascent: { T: -1, P: 0.5 } };
    assert.equal(ascentSentence(low, FACTORS), "A next study could move the ranges along the steepest descent: T down 25 degC, P up 0.5 bar.");
  });

  test("a stationary point inside the ranges", () => {
    const inside = {
      ...WORKED,
      settings: WORKED.settings.map((s) => ({ ...s, end: null, beyond: false })),
      stationary: { factors: ["T", "P", "F"], inside: true, kind: "maximum" },
    };
    assert.deepEqual(whereSentences(inside, FACTORS, WORDS), ["These settings are a stationary point: the top of the fitted surface."]);
  });

  test("on a limit, the better settings past it cannot be run, and no range is moved", () => {
    const limited = {
      ...WORKED,
      settings: WORKED.settings.map((s) => ({ ...s, end: null, beyond: false })),
      limits: [{ index: 0, expression: "T + 25 * P <= 215", beyond: true }],
      stationary: null,
    };
    assert.deepEqual(whereSentences(limited, FACTORS, WORDS), [
      "These settings are on your limit 1, T + 25 * P <= 215: the model predicts a higher Yield past it, but those settings cannot be run.",
    ]);
    const atEnd = { ...limited, settings: WORKED.settings };
    assert.equal(whereSentences(atEnd, FACTORS, WORDS).length, 1, "a range moved would cross the limit");
  });

  test("a next study is pointed to past an end, not past a limit nor at a target reached", () => {
    assert.equal(movesOn(WORKED), true);
    assert.equal(movesOn({ ...WORKED, settings: WORKED.settings.map((s) => ({ ...s, beyond: false })) }), false);
    assert.equal(movesOn({ ...WORKED, limits: [{ index: 0, expression: "T <= 190", beyond: true }] }), false);
    assert.equal(movesOn({ ...WORKED, goal: "target", target: 60, reached: true }), false);
    assert.equal(movesOn({ ...WORKED, ascent: null }), false);
    const outOfModel = { ...WORKED, settings: WORKED.settings.map((s) => ({ ...s, in_model: false })) };
    assert.equal(movesOn(outOfModel), false, "a factor the model leaves out points nowhere");
  });

  test("the next study's ranges move as the steepest ascent says, the others and a category kept", () => {
    const supplier = { name: "S", low: "A", high: "B", units: "" };
    const moved = nextRanges({ ...WORKED, ascent: { T: 1, P: -0.6712, S: 1 } }, [...FACTORS, supplier]);
    assert.deepEqual(moved, [
      { name: "T", low: 175, high: 225, units: "degC" }, // half the range up, as ascentSentence's "T up 25 degC"
      { name: "P", low: 0.33, high: 2.33, units: "bar" }, // "P down 0.67 bar"
      { name: "F", low: 10, high: 20, units: "L/min" },
      supplier,
    ]);
    assert.notEqual(moved[2], FACTORS[2], "a copy, so the study's own factors stay as they are");
  });

  test("a target reached: other settings reach it too", () => {
    const target = { ...WORKED, goal: "target", target: 60, reached: true, stationary: null, ascent: null };
    assert.deepEqual(whereSentences(target, FACTORS, WORDS), [
      "Other settings reach the target too: its contour on the response surface shows them.",
    ]);
  });

  test("factors the model leaves out", () => {
    const left = {
      ...WORKED,
      settings: [
        ...WORKED.settings.slice(0, 2),
        { ...WORKED.settings[2], in_model: false, value: 15 },
        { name: "S", kind: "categorical", value: "A", coded: -1, in_model: false, end: null, beyond: false },
      ],
      stationary: null,
    };
    const said = whereSentences(left, FACTORS, WORDS);
    assert.ok(said.includes("F is not in the model, so it does not change the prediction; these settings hold it at the center."));
    assert.ok(said.includes("S is not in the model, so either level serves; these settings use A."));
  });
});

describe("how far the model reaches", () => {
  test("beyond the runs, the prediction leans on the model", () => {
    assert.equal(
      extrapolationSentence(WORKED),
      "These settings lie beyond the runs, so the prediction leans on the model there: confirm it with a run.",
    );
  });

  test("among the runs, the leverage's row says so, and no sentence more", () => {
    const among = { ...WORKED, leverage: 0.4028, extrapolating: false };
    assert.equal(extrapolationSentence(among), "");
    assert.equal(intervalRows(among, WORDS).at(-1).value, "0.4, within the runs': 0.75 at most");
  });

  test("no term clearly larger than the noise is said plainly", () => {
    assert.match(UNCLEAR, /^No term of the model is clearly larger than the noise/);
  });
});

describe("how tightly to hold each factor", () => {
  test("the legend, the axis and the heading's definition say how to read it: no caption is needed", () => {
    assert.equal(sensitivityCaption(WORKED), "");
  });

  test("but the factors the model leaves out, which the chart does not draw, are named", () => {
    const left = { ...WORKED, sensitivity: [...WORKED.sensitivity.slice(0, 2), { ...WORKED.sensitivity[2], in_model: false }] };
    assert.equal(sensitivityCaption(left), "F is not in the model, so moving it changes nothing.");
  });
});

describe("the response surface drawn through the best settings", () => {
  test("the factors held at their best settings, and the star", () => {
    const caption = bestContourCaption(["T", "P"], { held: [{ name: "F", value: 18.0372 }], factors: FACTORS, starred: true });
    assert.equal(
      caption,
      "Drawn over T and P, the two factors with the largest main effects, with F at 18.04 L/min, its best setting; " +
        "the star marks the best settings; the dots are the runs.",
    );
  });

  test("a level held, the limits shaded, and no star", () => {
    const caption = bestContourCaption(["T", "P"], { held: [{ name: "S", value: "B" }], shaded: true, factors: FACTORS });
    assert.match(caption, /with S at B, its best setting; grey: settings outside your limits; the dots are the runs\.$/);
  });
});

// ---------------------------------------------------------------- confirmation runs
const SHEET = {
  columns: ["Std order", "Run order", "T", "P", "F", "Yield"],
  rows: [
    [3, 1, 150, 1, 15, null],
    [1, 2, 200, 3, 15, null],
  ],
};

describe("confirmation runs on the run sheet", () => {
  test("Python, the run sheet and the code shown mark them alike", () => {
    assert.equal(CONFIRMATION, "Confirmation run");
    assert.equal(PYTHON_CONFIRMATION, CONFIRMATION, "python.js leaves them out of the fit by the same column");
    assert.equal(CONFIRMS, "yes");
    assert.equal(CONFIRMATION_RUNS, 3);
  });

  test("added at the end, next in the run order, with no standard order", () => {
    const added = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 });
    assert.deepEqual(added.columns, [...SHEET.columns, CONFIRMATION]);
    assert.deepEqual(added.rows, [
      [3, 1, 150, 1, 15, null, null],
      [1, 2, 200, 3, 15, null, null],
      [null, 3, 200, 1, 18.04, null, "yes"],
      [null, 4, 200, 1, 18.04, null, "yes"],
      [null, 5, 200, 1, 18.04, null, "yes"],
    ]);
    assert.deepEqual(SHEET.columns.length, 6, "the run sheet given is left as it was");
    assert.deepEqual(confirmationRows(added), [2, 3, 4]);
    assert.deepEqual(fittedRuns(added), [true, true, false, false, false]);
  });

  test("added again, the column is not repeated", () => {
    const twice = withConfirmationRuns(withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18 }), { T: 175, P: 2, F: 15 }, 1);
    assert.equal(twice.columns.filter((c) => c === CONFIRMATION).length, 1);
    assert.deepEqual(twice.rows.at(-1), [null, 6, 175, 2, 15, null, "yes"]);
  });

  test("whether runs at these settings are there already", () => {
    const added = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 });
    assert.equal(confirmedAt(added, { T: 200, P: 1, F: 18.04 }), true);
    assert.equal(confirmedAt(added, { T: 200, P: 1, F: 18 }), false);
    assert.equal(confirmedAt(SHEET, { T: 150, P: 1, F: 15 }), false, "a run of the design is no confirmation run");
  });

  test("the design's runs, as the fit and the picture of the limits take them", () => {
    const added = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 });
    const design = designRows(added);
    assert.deepEqual(design.columns, SHEET.columns);
    assert.deepEqual(design.rows, SHEET.rows);
    assert.deepEqual(design.kept, [true, true, false, false, false]);
  });

  test("the runs along a path join the sheet in the path's order, apart from the design's and the confirmation runs", () => {
    const confirmed = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 }, 1);
    const added = withPathRuns(confirmed, [
      { T: 210, P: 1, F: 15 },
      { T: 220, P: 0.5, F: 15 },
    ]);
    const column = (name) => added.rows.map((row) => row[added.columns.indexOf(name)]);
    assert.deepEqual(column("Confirmation run"), [null, null, "yes", "path", "path"]);
    assert.deepEqual(column("T").slice(3), [210, 220]);
    assert.deepEqual(column("Run order").slice(2), [3, 4, 5]);
    assert.deepEqual([confirmationRows(added), pathRows(added), apartRows(added)], [[2], [3, 4], [2, 3, 4]]);
    assert.deepEqual(designRows(added).rows, SHEET.rows, "the fit takes neither");
    const results = { Yield: [50, 51, 67, 70, 75] };
    const off = withoutPathRuns(added, results, { Yield: [null, null, null, null, null] });
    assert.deepEqual(off.rows, confirmed.rows, "the confirmation run stays, with its column");
    assert.deepEqual(off.results, { Yield: [50, 51, 67] });
    const bare = withoutPathRuns(withPathRuns(SHEET, [{ T: 210, P: 1, F: 15 }]), { Yield: [50, 51, 70] }, { Yield: [null, null, null] });
    assert.deepEqual([bare.columns, bare.rows], [SHEET.columns, SHEET.rows], "the column goes once it marks no run");
  });

  test("a run that failed is marked on the sheet, stays a run of the design, and can be made again", () => {
    const marked = withFailed(SHEET, 1, true);
    assert.deepEqual(marked.columns, [...SHEET.columns, "Confirmation run"]);
    assert.deepEqual(failedRows(marked), [1]);
    assert.deepEqual(apartRows(marked), [], "it is a run of the design still");
    assert.equal(designRows(marked).rows.length, SHEET.rows.length);
    const again = withFailed(marked, 1, false);
    assert.deepEqual([again.columns, again.rows], [SHEET.columns, SHEET.rows], "the column goes with its last mark");
    const confirmed = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 }, 1);
    const both = withFailed(confirmed, 0, true);
    assert.deepEqual([failedRows(both), confirmationRows(both)], [[0], [2]]);
    assert.deepEqual(withFailed(both, 0, false).rows, confirmed.rows, "the confirmation run keeps its mark");
  });

  test("taken off again, with every response's results and sources", () => {
    const added = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 });
    const results = { Yield: [50, 51, 67, 68, null], Cost: [9, null, 8, 7, 6] };
    const marks = {
      Yield: ["entered", "entered", "simulated", "simulated", null],
      Cost: ["entered", null, "simulated", "simulated", "entered"],
    };
    const off = withoutConfirmationRuns(added, results, marks);
    assert.deepEqual(off, {
      columns: SHEET.columns,
      rows: SHEET.rows,
      results: { Yield: [50, 51], Cost: [9, null] },
      marks: { Yield: ["entered", "entered"], Cost: ["entered", null] },
      curves: null,
    });
    const curves = [[0, 1, 2], null, [0, 2, 3], null, null];
    assert.deepEqual(withoutConfirmationRuns(added, results, marks, curves).curves, [[0, 1, 2], null], "each run's curve too");
  });

  test("a mark typed in a workbook is read whatever its case", () => {
    const typed = { columns: [...SHEET.columns, CONFIRMATION], rows: [[...SHEET.rows[0], " Yes "], [...SHEET.rows[1], ""]] };
    assert.deepEqual(confirmationRows(typed), [0]);
  });
});

describe("what the confirmation runs show", () => {
  const group = {
    rows: [15, 16, 17],
    at: { T: 200, P: 1, F: 18.04 },
    runs: 3,
    done: 0,
    mean: null,
    predicted: 68.25,
    interval: [65.98, 70.52],
    inside: null,
  };

  test("before their results: where to make them, and what to expect", () => {
    assert.deepEqual(confirmationSentences(group, FACTORS, WORDS), [
      "3 confirmation runs at T = 200 degC, P = 1 bar and F = 18.04 L/min. Enter their results in Run. " +
        "When their average is between 65.98 and 70.52%, the model holds there.",
    ]);
  });

  test("their average inside the interval: the model holds", () => {
    const done = { ...group, done: 3, mean: 67.9, inside: true };
    assert.equal(
      confirmationSentences(done, FACTORS, WORDS)[1],
      "The average of their 3 runs, 67.9%, is inside the 95% prediction interval for the average of 3 runs, 65.98 to 70.52%: " +
        "the model holds at these settings.",
    );
  });

  test("outside it: the process differs from the model there", () => {
    const off = { ...group, done: 2, mean: 61.5, inside: false };
    const said = confirmationSentences(off, FACTORS, WORDS);
    assert.match(said[0], /2 of 3 results are in\.$/);
    assert.match(said[1], /^The average of their 2 runs, 61.5%, is outside the 95% prediction interval for the average of 2 runs/);
    assert.match(said[1], /the process differs from the model at these settings/);
  });

  test("a categorical factor's level, and no interval without runs left for the noise", () => {
    const factors = [...FACTORS.slice(0, 2), { name: "S", low: "A", high: "B", units: "" }];
    const level = { ...group, at: { T: 200, P: 1, S: "B" }, done: 1, mean: 60, interval: null, inside: null };
    const said = confirmationSentences(level, factors, WORDS);
    assert.match(said[0], /at T = 200 degC, P = 1 bar and S at B\./);
    assert.match(said[0], /the page cannot judge their average\.$/);
  });
});

describe("a mixture's best blend", () => {
  const HARDNESS = { label: "Hardness", units: "HV" };
  const component = (name, value, end = null, more = {}) => ({ name, kind: "component", value, low: 0, high: 1, in_model: true, end, beyond: false, ...more });
  const BLEND = {
    goal: "maximize",
    target: null,
    predicted: 54.686,
    reached: null,
    settings: [component("A", 0.9379), component("B", 0.0621), component("C", 0, "low")],
    extrapolating: false,
    rounding: { decimals: 3, adjusted: "A" },
  };

  test("the lead names the blend and what the model predicts there", () => {
    assert.equal(blendLead(BLEND, HARDNESS), "This blend gives the highest Hardness the model predicts: 54.69 HV.");
    assert.equal(blendLead({ ...BLEND, goal: "minimize" }, HARDNESS), "This blend gives the lowest Hardness the model predicts: 54.69 HV.");
    const reached = { ...BLEND, goal: "target", target: 50, predicted: 50, reached: true };
    assert.equal(blendLead(reached, HARDNESS), "This blend gives Hardness on its target, 50 HV, as the model predicts.");
    const missed = { ...BLEND, goal: "target", target: 60, reached: false };
    assert.equal(blendLead(missed, HARDNESS), "No blend within your bounds reaches the target of 60 HV. This one comes closest: 54.69 HV.");
  });

  test("each component at a bound, or between its bounds, and whether the goal may be better past a bound set", () => {
    assert.equal(blendWhere(BLEND.settings[0], BLEND, HARDNESS), "between its bounds");
    assert.equal(blendWhere(BLEND.settings[2], BLEND, HARDNESS), "at its lower bound, 0");
    const capped = component("A", 0.7, "high", { low: 0.2, high: 0.7, beyond: true });
    assert.equal(blendWhere(capped, BLEND, HARDNESS), "at its upper bound, 0.7; a higher Hardness may lie above");
    const floored = component("B", 0.1, "low", { low: 0.1, high: 0.6, beyond: true });
    assert.equal(blendWhere(floored, { ...BLEND, goal: "minimize" }, HARDNESS), "at its lower bound, 0.1; a lower Hardness may lie below");
  });

  test("a blend inside every bound is the top or the bottom of the surface; a target reached, one of many", () => {
    assert.deepEqual(blendSentences(BLEND), [], "C is at its bound: the table says so");
    const inside = { ...BLEND, settings: BLEND.settings.map((s) => ({ ...s, end: null })) };
    assert.deepEqual(blendSentences(inside), ["Every component is between its bounds: this blend is the top of the fitted surface."]);
    assert.match(blendSentences({ ...inside, goal: "minimize" })[0], /the bottom of the fitted surface\.$/);
    assert.deepEqual(blendSentences({ ...BLEND, goal: "target", reached: true }), ["Other blends reach the target too."]);
    assert.deepEqual(blendSentences({ ...inside, goal: "target", reached: false }), []);
  });

  test("the cautions and the chart's caption name the blend", () => {
    assert.equal(blendExtrapolation(BLEND), "");
    assert.match(blendExtrapolation({ ...BLEND, extrapolating: true }), /^This blend lies beyond the runs/);
    assert.match(BLEND_UNCLEAR, /so this blend may follow the noise\.$/);
    assert.match(BLEND_SENSITIVITY, /the others keeping their proportions to each other, so the blend still sums to 1\.$/);
  });

  test("the confirmation runs weigh out the blend, and say which component takes the rounding", () => {
    assert.equal(
      weighingSentence(BLEND),
      "The confirmation runs weigh out each proportion to 0.001, and A takes what the rounding leaves, so the blend still sums to 1.",
    );
    assert.equal(weighingSentence({ ...BLEND, rounding: { decimals: 4, adjusted: null } }), "The confirmation runs weigh out each proportion to 0.0001; the blend still sums to 1.");
    assert.equal(weighingSentence({ ...BLEND, rounding: { decimals: null, adjusted: null } }), "The confirmation runs use the blend as the search found it.");
  });

  test("confirmation runs at a blend say so", () => {
    const factors = ["A", "B", "C"].map((name) => ({ name, low: 0, high: 1, units: "" }));
    const group = { rows: [14], at: { A: 0.938, B: 0.062, C: 0 }, runs: 3, done: 3, mean: 54.2, predicted: 54.69, interval: [53, 56.4], inside: true };
    const said = confirmationSentences(group, factors, HARDNESS, "this blend");
    assert.equal(said[0], "3 confirmation runs at A = 0.938, B = 0.062 and C = 0.");
    assert.match(said[1], /: the model holds at this blend\.$/);
  });
});

describe("the settings that balance the responses", () => {
  /** A balance as bootstrap.balance gives it: a yield to maximise and a purity with a target, at T and P. */
  const BALANCE = {
    balanced: true,
    composite: 0.7305,
    limiting: "Purity",
    leverage: 0.2564,
    max_leverage: 0.7939,
    leverage_of: "Yield",
    extrapolating: false,
    limits: [],
    settings: [
      { name: "T", kind: "numeric", value: 169.3, coded: -0.228, units: "degC", in_model: true, end: null, beyond: false },
      { name: "F", kind: "numeric", value: 15, coded: 0, units: "L/min", in_model: false, end: null, beyond: false },
    ],
    responses: [
      { name: "Yield", label: "Yield", units: "%", goal: "maximize", low: 49, high: 61, importance: 3, d: 0.83, predicted: 58.96, prediction: [57.1, 60.8], default_low: 49, default_high: 61 },
      { name: "Purity", label: "Purity", units: "%", goal: "target", target: 98.5, low: 97, high: 100, importance: 3, d: 0.64, predicted: 98.04, prediction: null, default_low: 97, default_high: 100 },
    ],
  };

  test("the goal as the table says it", () => {
    assert.equal(goalWords({ goal: "maximize" }), "Maximise");
    assert.equal(goalWords({ goal: "minimize" }), "Minimise");
    assert.equal(goalWords({ goal: "target", target: 98.5, units: "%" }), "Target 98.5%");
  });

  test("a range typed upside down, or one that leaves the target out, is said as Python says it", () => {
    const [yieldR, purity] = BALANCE.responses;
    assert.equal(goalProblem(yieldR, 55, 60), "");
    assert.equal(goalProblem(yieldR, null, null), "", "blank boxes take the results' range");
    assert.equal(goalProblem(yieldR, 61, null), "The acceptable range of Yield must run from a lower value to a higher one.");
    assert.equal(goalProblem(yieldR, 50, 50), "The acceptable range of Yield must run from a lower value to a higher one.");
    assert.equal(goalProblem(purity, 99, 100), "The target of Purity, 98.5, must lie inside its acceptable range.");
    assert.equal(goalProblem(purity, 97, null), "");
  });

  test("the panel says first how good the balance is", () => {
    assert.equal(balanceLead(BALANCE), "These settings balance the responses best: an overall desirability of 0.73.");
    assert.equal(balanceLead({ ...BALANCE, composite: 1 }), "These settings make every response as good as it gets: an overall desirability of 1.");
    assert.match(balanceLead({ ...BALANCE, composite: 0 }), /^No setting in your ranges makes every response acceptable\./);
    const blend = { blend: true };
    assert.equal(balanceLead(BALANCE, blend), "This blend balances the responses best: an overall desirability of 0.73.");
    assert.equal(balanceLead({ ...BALANCE, composite: 1 }, blend), "This blend makes every response as good as it gets: an overall desirability of 1.");
    assert.match(balanceLead({ ...BALANCE, composite: 0 }, blend), /^No blend within your bounds makes every response acceptable\. This one comes closest/);
    const none = { ...BALANCE, composite: 0, responses: [{ ...BALANCE.responses[0], d: 0 }, BALANCE.responses[1]] };
    assert.match(limitingSentence(none, blend), /^At this blend Yield is not acceptable/);
  });

  test("a mixture's overlay is over its triangle, and four components or more have none", () => {
    const overlay = {
      kind: "ternary",
      components: ["A", "B", "C"],
      responses: [
        { name: "Yield", z: [50, 56, 60], levels: [55] },
        { name: "Purity", z: [98, 98.2, 98.4], levels: [97.5, 99.5] },
      ],
    };
    assert.equal(
      blendOverlayCaption(overlay, BALANCE.responses),
      "Each corner is a pure component and each point a blend the bounds allow. Shaded where every response is " +
        "acceptable; each line is a response's acceptable boundary; the star marks the balanced blend. No line for " +
        "Purity: it is acceptable at every blend drawn. Narrow a range in the table to draw its boundary.",
    );
    assert.match(NO_BLEND_OVERLAY, /^With four components or more the blends fill a solid/);
  });

  test("the response that holds D down most is named, with its importance when they differ", () => {
    assert.equal(limitingSentence(BALANCE), "Purity holds the overall desirability down most: its desirability is 0.64.");
    const weighted = { ...BALANCE, responses: [BALANCE.responses[0], { ...BALANCE.responses[1], importance: 5 }] };
    assert.equal(limitingSentence(weighted), "Purity holds the overall desirability down most: its desirability is 0.64, at importance 5.");
    assert.equal(limitingSentence({ ...BALANCE, composite: 1 }), "");
    const none = { ...BALANCE, composite: 0, responses: [{ ...BALANCE.responses[0], d: 0 }, BALANCE.responses[1]] };
    assert.equal(limitingSentence(none), "At these settings Yield is not acceptable: a desirability of 0.");
  });

  test("each response at the balanced settings, a row each", () => {
    assert.deepEqual(balanceRows(BALANCE), [
      ["Yield", "58.96%", "57.1 to 60.8%", "0.83"],
      ["Purity", "98.04%", "", "0.64"],
    ]);
    assert.equal(dWords(0.6), "0.60");
  });

  test("the leverage, for the model that leans furthest on the runs", () => {
    assert.equal(balanceLeverage(BALANCE), "The leverage here is 0.26 for Yield's model, within its runs': 0.79 at most.");
    assert.match(balanceLeverage({ ...BALANCE, extrapolating: true }), /above its runs'/);
  });

  test("the limits the settings are on, and the factors no model holds", () => {
    assert.deepEqual(balanceWhere(BALANCE), ["F is in no response's model, so it changes nothing; these settings hold it at the center."]);
    const limited = { ...BALANCE, settings: [], limits: [{ index: 1, expression: "T + 25*P <= 215", beyond: false }] };
    assert.deepEqual(balanceWhere(limited), ["These settings are on your limit 2, T + 25*P <= 215."]);
  });

  test("the overlay's caption says what it is drawn over, where the others are held, and what each mark means", () => {
    const overlay = { x: "T", y: "P", held: [{ name: "F", value: 16.25 }], runnable: null };
    assert.equal(
      overlayCaption(overlay, FACTORS),
      "Drawn over T and P, the two factors that move the responses' desirabilities most, with F at 16.25 L/min, its " +
        "balanced setting. Shaded where every response is acceptable; each line is a response's acceptable boundary; the " +
        "star marks the balanced settings.",
    );
    assert.match(overlayCaption({ ...overlay, held: [], runnable: [[1]] }, FACTORS), /, the two factors that move the responses' desirabilities most\. .*; grey: settings outside your limits;/);
  });

  test("a boundary that misses the settings drawn has no line, and the caption says why", () => {
    const responses = [
      { name: "Yield", label: "Yield", goal: "maximize", low: 55, high: 64 },
      { name: "Cost", label: "Cost", goal: "minimize", low: 10, high: 12 },
      { name: "Purity", label: "Purity", goal: "target", low: 97.5, high: 99.5 },
    ];
    const surfaces = [
      { name: "Yield", z: [[50, 60]], levels: [55] }, // crosses: a line
      { name: "Cost", z: [[10.5, 11]], levels: [12] }, // below its high end throughout: acceptable everywhere
      { name: "Purity", z: [[96, 97]], levels: [97.5, 99.5] }, // below the range throughout: acceptable nowhere
    ];
    const overlay = { x: "T", y: "P", held: [], runnable: null, responses: surfaces };
    assert.deepEqual(missedBoundaries(overlay, responses), { everywhere: ["Cost"], nowhere: ["Purity"] });
    assert.match(
      overlayCaption(overlay, FACTORS, responses),
      /the star marks the balanced settings\. No line for Cost: it is acceptable at every setting drawn\. Narrow a range in the table to draw its boundary\. No line for Purity: it is not acceptable at any setting drawn\.$/,
    );
    const all = { ...overlay, responses: [{ name: "Yield", z: [[60, 62]], levels: [55] }, surfaces[1]] };
    assert.match(overlayCaption(all, FACTORS, responses), /No line for Yield and Cost: they are acceptable at every setting drawn\./);
    assert.ok(overlayCaption({ ...overlay, responses: [surfaces[0]] }, FACTORS, responses).endsWith("the star marks the balanced settings."));
  });

  test("confirmation runs are judged response by response", () => {
    const group = {
      runs: 3,
      at: { T: 169.3, P: 2.016, F: 15 },
      responses: [
        { name: "Yield", done: 3, mean: 58.5, predicted: 58.96, interval: [56.9, 61.0], inside: true },
        { name: "Purity", done: 3, mean: 95, predicted: 98.04, interval: [97.1, 99.0], inside: false },
      ],
    };
    const { said, state } = balanceConfirmation(group, FACTORS, BALANCE.responses);
    assert.equal(state, "outside");
    assert.deepEqual(said, [
      "3 confirmation runs at T = 169.3 degC, P = 2.016 bar and F = 15 L/min.",
      "Yield: the average of 3 results, 58.5%, is inside the 95% prediction interval, 56.9 to 61%: the model holds here.",
      "Purity: the average of 3 results, 95%, is outside the 95% prediction interval, 97.1 to 99%: the process differs from the model here.",
    ]);
    const waiting = { ...group, responses: [group.responses[0], { name: "Purity", done: 0, mean: null, predicted: 98, interval: [97, 99], inside: null }] };
    assert.equal(balanceConfirmation(waiting, FACTORS, BALANCE.responses).state, "waiting");
    assert.equal(balanceConfirmation(waiting, FACTORS, BALANCE.responses).said[2], "Purity: no result yet; enter it in Run.");
    const both = { ...group, responses: [group.responses[0], { ...group.responses[1], mean: 98, inside: true }] };
    assert.equal(balanceConfirmation(both, FACTORS, BALANCE.responses).state, "inside");
  });

  test("the choices the table offers", () => {
    assert.deepEqual(Object.keys(GOOD_END), ["less", "even", "more", "much_more"]);
    assert.deepEqual(IMPORTANCES, [1, 2, 3, 4, 5]);
    assert.equal(IMPORTANCE, 3);
    assert.equal(SHAPE, "even");
  });
});

// What if: tests/web/api/test_what_if.py holds predictAt to statsmodels and to the best settings; these hold its arithmetic
// on a fit small enough to work out by hand, and the panel's words.
describe("what if", () => {
  /** T numeric over 150 to 200, Catalyst of three levels as bootstrap.predictors codes them. */
  const PREDICTED = [
    { name: "T", kind: "numeric", low: 150, high: 200 },
    {
      name: "Catalyst",
      kind: "categorical",
      levels: ["A", "B", "C"],
      columns: [
        { Catalyst_I1: 1, Catalyst_I2: 0, Catalyst_I3: 0, Catalyst_L1: 1, Catalyst_L2: 0 },
        { Catalyst_I1: 0, Catalyst_I2: 1, Catalyst_I3: 0, Catalyst_L1: 0, Catalyst_L2: 1 },
        { Catalyst_I1: 0, Catalyst_I2: 0, Catalyst_I3: 1, Catalyst_L1: -1, Catalyst_L2: -1 },
      ],
    },
  ];
  /** y = 10 + 2 T + 3 T², fitted to runs whose (X'X)^-1 is a quarter of the identity, with s² = 4 on 6 df. */
  const FIT = {
    name: "y",
    label: "Yield",
    units: "%",
    terms: [[], ["T"], ["T", "T"]],
    coefficients: [10, 2, 3],
    inverse: [
      [0.25, 0, 0],
      [0, 0.25, 0],
      [0, 0, 0.25],
    ],
    mse: 4,
    t: 2.447,
    df: 6,
    level: 0.95,
    max_leverage: 0.5,
  };

  test("a numeric setting is coded from its range, a level as the columns Python wrote for it", () => {
    assert.deepEqual(columnsAt(PREDICTED, { T: 187.5, Catalyst: "C" }), {
      T: 0.5,
      Catalyst_I1: 0,
      Catalyst_I2: 0,
      Catalyst_I3: 1,
      Catalyst_L1: -1,
      Catalyst_L2: -1,
    });
  });

  test("the prediction, its intervals and its leverage, as _FittedModel.at works them out", () => {
    const at = predictAt(FIT, { T: 1 });
    assert.equal(at.predicted, 15); // 10 + 2 + 3
    assert.equal(at.leverage, 0.75); // a row of ones, each term a quarter
    const mean = 2.447 * Math.sqrt(4 * 0.75);
    const single = 2.447 * Math.sqrt(4 * 1.75);
    assert.deepEqual(at.confidence, [15 - mean, 15 + mean]);
    assert.deepEqual(at.prediction, [15 - single, 15 + single]);
    assert.equal(at.extrapolating, true, "0.75 is above the runs' 0.5");
    assert.equal(predictAt(FIT, { T: 0 }).extrapolating, false, "the center's 0.25 is within");
  });

  test("a prediction as runs: one for each t value Python sent, the outer two at the ends of the prediction interval", () => {
    // Five equally likely runs here: the t values at the middle of five equal shares, the outer two the interval's own.
    const fit = { ...FIT, run_quantiles: [-2.447, -0.8, 0, 0.8, 2.447] };
    const runs = runsAt(fit, { T: 1 });
    const single = Math.sqrt(4 * 1.75);
    assert.deepEqual(runs, [-2.447, -0.8, 0, 0.8, 2.447].map((q) => 15 + q * single));
    assert.deepEqual([runs[0], runs.at(-1)], predictAt(fit, { T: 1 }).prediction, "the same two numbers as the interval");
    assert.equal(runsAt({ ...fit, mse: null, t: null, run_quantiles: null }, { T: 1 }), null, "a fit of every run has no noise to draw");
    assert.equal(runsAt(FIT, { T: 1 }), null, "nor has a fit sent without its runs' t values");

    // A response fitted on a log scale has its runs put back, as its interval is; one that cannot be put back draws none.
    const logged = runsAt({ ...fit, coefficients: [1, 0.1, 0], transform: "log" }, { T: 1 });
    assert.deepEqual([logged[0], logged.at(-1)], predictAt({ ...fit, coefficients: [1, 0.1, 0], transform: "log" }, { T: 1 }).prediction);
    assert.ok(logged.every((run, i) => i === 0 || run > logged[i - 1]), "least first, as on the scale it is fitted on");
    assert.equal(runsAt({ ...fit, transform: "inverse" }, { T: 1 }), null, "-1/y above 0 is a result without bound");
  });

  test("the runs inside a window, an end of which may be open", () => {
    const runs = [1, 2, 3, 4, 5];
    assert.deepEqual(insideWindow(runs, [2, 4]), [false, true, true, true, false], "an end is inside");
    assert.deepEqual(insideWindow(runs, [null, 2.5]), [true, true, false, false, false]);
    assert.deepEqual(insideWindow(runs, [4.5, null]), [false, false, false, false, true]);
    assert.deepEqual(insideWindow(runs), [true, true, true, true, true], "with no window, every one");
  });

  test("runs pile up in the column they fall in", () => {
    assert.deepEqual(dotPiles([0, 1, 1.1, 1.2, 4], [0, 4], 4), [
      { column: 0, level: 0 },
      { column: 1, level: 0 },
      { column: 1, level: 1 },
      { column: 1, level: 2 },
      { column: 3, level: 0 }, // the high end is in the last column, not past it
    ]);
    assert.deepEqual(dotPiles([2, 2], [2, 2], 4), [{ column: 0, level: 0 }, { column: 0, level: 1 }], "runs that are all one value"); // prettier-ignore
  });

  test("the runs in words: how far they reach, and how many a window passes", () => {
    const runs = [58.21, 60, 62, 64, 66.14];
    const words = { label: "Yield", units: "%" };
    assert.equal(runsWords(runs, insideWindow(runs), words), "Yield as 5 runs you might get: 58.21 to 66.14%.");
    assert.equal(runsWords(runs, insideWindow(runs, [60, null]), words, true), "Yield as 5 runs you might get: 58.21 to 66.14%. 4 of them are inside your window.");
    assert.equal(runsWords([1, 3], [false, false], { label: "Cost", units: "$/kg" }, true), "Cost as 2 runs you might get: 1 to 3 $/kg. 0 of them are inside your window.");
  });

  test("a factor's profile is the prediction as it alone moves, the others where they are set", () => {
    const settings = { T: 160, Catalyst: "B" };
    const profile = profileOf(FIT, PREDICTED, settings, "T");
    assert.equal(profile.length, PROFILE_POINTS);
    assert.deepEqual([profile[0].value, profile.at(-1).value], [150, 200], "from its low end to its high end");
    assert.deepEqual([profile[0].predicted, profile[12].predicted, profile.at(-1).predicted], [11, 10, 15]); // 10 + 2 x + 3 x²
    for (const point of profile) {
      const at = predictAt(FIT, columnsAt(PREDICTED, { ...settings, T: point.value }));
      assert.deepEqual([point.predicted, point.confidence, point.extrapolating], [at.predicted, at.confidence, at.extrapolating]);
    }
    assert.deepEqual(profile.map((point) => point.extrapolating).join("").replaceAll("true", "1").replaceAll("false", "0"), "1110000000000000000000111");
    assert.equal(profileOf(FIT, [{ ...PREDICTED[0], low: 200, high: 150 }, PREDICTED[1]], settings, "T")[0].value, 150, "a range typed high to low");

    const levels = profileOf(FIT, PREDICTED, settings, "Catalyst");
    assert.deepEqual(levels.map((point) => point.value), ["A", "B", "C"]); // prettier-ignore
    const there = predictAt(FIT, columnsAt(PREDICTED, settings)).predicted;
    assert.deepEqual(levels.map((point) => point.predicted), [there, there, there], "a factor the model does not hold moves nothing"); // prettier-ignore
    assert.equal(profileOf({ ...FIT, mse: null, t: null }, PREDICTED, settings, "T")[0].confidence, null, "a fit of every run has no band");
  });

  test("the profiles of a row share one scale, which holds still while the reader drags", () => {
    const profiles = ["T", "Catalyst"].map((name) => profileOf(FIT, PREDICTED, { T: 160, Catalyst: "B" }, name));
    const ends = profiles.flat().flatMap((point) => [point.predicted, ...point.confidence]);
    assert.deepEqual(profileRange(profiles), [Math.min(...ends), Math.max(...ends)], "the bands are inside it");
    assert.deepEqual(profileRange(profiles, [-100, 100]), [-100, 100]);
    assert.deepEqual(profileRange([[{ predicted: 5, confidence: null }]]), [4.5, 5.5], "a level response has room about it");
    assert.deepEqual(profileRange([[{ predicted: 400, confidence: null }]]), [380, 420]);
    assert.equal(profileRange([]), null);
  });

  test("a profile's line is in pieces, dashed where the settings lie beyond the runs, and has no gap", () => {
    const profile = profileOf(FIT, PREDICTED, { T: 160, Catalyst: "B" }, "T");
    const pieces = profilePieces(profile);
    assert.deepEqual(pieces.map((piece) => [piece.beyond, piece.points.length]), [[true, 4], [false, 19], [true, 4]]); // prettier-ignore
    assert.equal(pieces[0].points.at(-1), pieces[1].points[0], "a dashed piece ends where the solid one starts");
    assert.equal(pieces[1].points.at(-1), pieces[2].points[0]);
    const within = profilePieces(profile.map((point) => ({ ...point, extrapolating: false })));
    assert.deepEqual(within.map((piece) => [piece.beyond, piece.points.length]), [[false, PROFILE_POINTS]]); // prettier-ignore
    assert.deepEqual(profilePieces([]), []);
  });

  test("a profile is said in words: the two ends, a turn inside the range, and whether part lies beyond the runs", () => {
    const words = { label: "Yield", units: "%" };
    const settings = { T: 160, Catalyst: "B" };
    assert.equal(
      profileWords(PREDICTED[0], profileOf(FIT, PREDICTED, settings, "T"), words, "degC"),
      "Yield against T, the other factors where they are set: 11% at 150 degC, 15% at 200 degC, lowest at 9.667% at 166.7 degC. Part of it lies beyond the runs.",
    );
    const hill = { ...FIT, coefficients: [10, 2, -3], max_leverage: 2 };
    assert.equal(
      profileWords(PREDICTED[0], profileOf(hill, PREDICTED, settings, "T"), words),
      "Yield against T, the other factors where they are set: 5% at 150, 9% at 200, highest at 10.33% at 183.3.",
    );
    const slope = { ...FIT, coefficients: [10, 2, 0], max_leverage: 2 };
    assert.equal(profileWords(PREDICTED[0], profileOf(slope, PREDICTED, settings, "T"), words), "Yield against T, the other factors where they are set: 8% at 150, 12% at 200.");
    assert.equal(
      profileWords(PREDICTED[1], profileOf(slope, PREDICTED, settings, "Catalyst"), words),
      "Yield at each level of Catalyst: 8.8% at A, 8.8% at B and 8.8% at C.",
    );
  });

  test("the map is over the two factors chosen when both are numeric, else the first two numeric ones", () => {
    const three = [PREDICTED[0], PREDICTED[1], { name: "P", kind: "numeric", low: 1, high: 3 }, { name: "F", kind: "numeric", low: 10, high: 20 }];
    assert.deepEqual(mapPair(three), ["T", "P"], "a factor with levels is not an axis");
    assert.deepEqual(mapPair(three, ["F", "T"]), ["F", "T"]);
    assert.deepEqual(mapPair(three, ["Catalyst", "T"]), ["T", "P"]);
    assert.deepEqual(mapPair(three, ["T", "T"]), ["T", "P"]);
    assert.deepEqual(mapPair(three, ["T", "gone"]), ["T", "P"]);
    assert.equal(mapPair(PREDICTED), null, "one numeric factor has no map");
  });

  test("a response's map is its prediction over two factors, the others where they are set", () => {
    const factors = [PREDICTED[0], PREDICTED[1], { name: "P", kind: "numeric", low: 3, high: 1 }];
    const fit = { ...FIT, terms: [[], ["T"], ["T", "T"], ["P"]], coefficients: [10, 2, 3, -1], inverse: [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => (i === j ? 0.2 : 0))), max_leverage: 0.5 };
    const settings = { T: 160, Catalyst: "B", P: 2 };
    const surface = surfaceOf(fit, factors, settings, ["T", "P"]);
    assert.deepEqual([surface.x.length, surface.y.length, surface.z.length, surface.z[0].length], [MAP_POINTS, MAP_POINTS, MAP_POINTS, MAP_POINTS]);
    assert.deepEqual([surface.x[0], surface.x.at(-1), surface.y[0], surface.y.at(-1)], [150, 200, 1, 3], "each from its low end to its high end");
    for (const [j, i] of [[0, 0], [5, 17], [20, 20]]) { // prettier-ignore
      const at = predictAt(fit, columnsAt(factors, { ...settings, T: surface.x[i], P: surface.y[j] }));
      assert.equal(surface.z[j][i], at.predicted);
      assert.equal(surface.beyond[j][i], Number(at.extrapolating));
    }
    assert.equal(surface.beyond[10][10], 0, "the center is within the runs");
    assert.equal(surface.beyond[0][0], 1, "a corner is beyond them here");
  });

  test("a fit on the log scale predicts the median, its intervals put back end by end", () => {
    const at = predictAt({ ...FIT, transform: "log" }, { T: 1 });
    const onScale = predictAt(FIT, { T: 1 });
    assert.equal(at.predicted, Math.exp(15));
    assert.deepEqual(at.confidence, onScale.confidence.map(Math.exp));
    assert.equal(at.leverage, onScale.leverage, "the leverage is the design's, on any scale");
  });

  test("a number on a transform's scale is put back as bootstrap._put_back puts it", () => {
    assert.ok(Math.abs(putBack(Math.log(20), "log") - 20) < 1e-12);
    assert.equal(putBack(3, "sqrt"), 9);
    assert.equal(putBack(-1, "sqrt"), 0, "a square root below 0 is a result of 0");
    assert.equal(putBack(-0.25, "inverse"), 4, "-1/y = -0.25 is y = 4");
    assert.equal(putBack(0, "inverse"), null, "-1/y at 0 is a result without bound");
    assert.equal(putBack(800, "log"), null, "a log too large to put back");
    assert.deepEqual(putBack([-0.5, 0.1], "inverse"), [2, null]);
    assert.equal(putBack(5, null), 5);
    assert.equal(putBack(null, "log"), null);
  });

  test("an interval with no bound on one side says so", () => {
    assert.equal(rangeWords([61.2, 68.4], "%"), "61.2 to 68.4%");
    assert.equal(rangeWords([61.2, 68.4], "%", "and"), "61.2 and 68.4%");
    assert.equal(rangeWords([2, null], "s"), "2 s or more");
    assert.equal(rangeWords([null, 7], "s"), "up to 7 s");
  });

  test("a fit with no runs left over predicts with no interval", () => {
    const at = predictAt({ ...FIT, mse: null, t: null, df: 0 }, { T: 0 });
    assert.deepEqual([at.predicted, at.confidence, at.prediction], [10, null, null]);
  });

  test("the panel opens at the balanced settings, else the best, else the center of the ranges", () => {
    const balance = { settings: [{ name: "T", value: 160 }, { name: "Catalyst", value: "B" }] };
    const best = { settings: [{ name: "T", value: 190 }, { name: "Catalyst", value: "A" }] };
    assert.deepEqual(whatIfStart(PREDICTED, balance, best), { from: "balanced", settings: { T: 160, Catalyst: "B" } });
    assert.deepEqual(whatIfStart(PREDICTED, null, best), { from: "best", settings: { T: 190, Catalyst: "A" } });
    assert.deepEqual(whatIfStart(PREDICTED, null, { goal: "learn" }), { from: "center", settings: { T: 175, Catalyst: "A" } });
    assert.equal(WHAT_IF_BACK.center, "Back to the center of the ranges");
  });

  test("settings are kept while they fit the factors", () => {
    assert.equal(settingsFit(PREDICTED, { T: 160, Catalyst: "B" }), true);
    assert.equal(settingsFit(PREDICTED, { T: 210, Catalyst: "B" }), false, "outside the range");
    assert.equal(settingsFit(PREDICTED, { T: 160, Catalyst: "D" }), false, "a level the factor lost");
    assert.equal(settingsFit(PREDICTED, { T: 160 }), false, "a factor added");
    assert.equal(settingsFit(PREDICTED, null), false);
  });

  test("settings kept from the panel say why the models do not predict there, once the study has changed", () => {
    const units = { T: "degC" };
    assert.equal(scenarioProblem(PREDICTED, { T: 160, Catalyst: "B" }, units), null);
    assert.equal(
      scenarioProblem(PREDICTED, { T: 210, Catalyst: "B" }, units),
      "T = 210 degC is outside its range, 150 to 200 degC, so the models do not predict there.",
    );
    assert.equal(
      scenarioProblem(PREDICTED, { T: 160, Catalyst: "D" }, units),
      "Catalyst at D is none of its levels, A, B and C, so the models do not predict there.",
    );
    assert.equal(scenarioProblem(PREDICTED, { Catalyst: "B" }), "These settings do not set T, so the models do not predict there.");
    assert.equal(scenarioProblem(PREDICTED, { T: "hot", Catalyst: "B" }), "T = hot is outside its range, 150 to 200, so the models do not predict there.");
    const broken = [
      { index: 0, expression: "T <= 190" },
      { index: 2, expression: "T >= 155" },
    ];
    assert.equal(
      scenarioProblem(PREDICTED, { T: 160, Catalyst: "B" }, units, broken),
      "These settings break your limit 1, T <= 190 and your limit 3, T >= 155, so the models do not predict there.",
    );
    assert.equal(settingsFit(PREDICTED, { T: 160, Catalyst: "B", Gone: 1 }), true, "a setting of no factor is not read");
  });

  test("settings kept from the panel are written as the best settings are", () => {
    assert.equal(settingsWords(PREDICTED, { T: 162.5, Catalyst: "B" }, { T: "degC" }), "T = 162.5 degC, Catalyst at B");
    assert.equal(settingsWords(PREDICTED, { Catalyst: "C", Gone: 3 }), "Catalyst at C", "each factor they set");
  });

  test("settings are kept under the name typed, on one line, or the next one free", () => {
    const held = { count: 3, name: 12 };
    const hot = { name: "Hot", settings: { T: 200 } };
    assert.deepEqual(keptAs("  Cooler,\tmore\npressure ", [], held), { name: "Cooler, more", problem: null });
    assert.deepEqual(keptAs("", [], held), { name: "Settings 1", problem: null });
    assert.deepEqual(keptAs(" ", [hot, { ...hot, name: "Settings 2" }], held), { name: "Settings 3", problem: null });
    assert.deepEqual(keptAs("Hot", [hot], held), { name: "Hot", problem: "Settings are kept as Hot already: give these another name." });
    const full = ["A", "B", "C"].map((name) => ({ ...hot, name }));
    assert.deepEqual(keptAs("D", full, held), { name: "D", problem: "A study keeps 3 settings at most: delete one to keep these." });
  });

  test("a slider steps by about a hundredth of the range", () => {
    assert.deepEqual(
      [
        [150, 200],
        [1, 3],
        [0, 0.37],
        [0, 950],
      ].map(([low, high]) => sliderStep({ low, high })),
      [0.5, 0.02, 0.005, 10],
    );
  });

  test("a row per response, and what the table cannot say", () => {
    const at = predictAt(FIT, { T: 1 });
    const responses = [
      { ...FIT, at },
      { name: "Cost", label: "Cost", units: "$/kg", waiting: true },
      { name: "Purity", label: "Purity", units: "%", error: "The term Supplier·Batch needs Supplier in the model too." },
    ];
    assert.deepEqual(whatIfRows(responses), [
      ["Yield", "15%", "10.76 to 19.24%", "8.526 to 21.47%", "0.75, above 0.5"],
      ["Cost", "No results yet: enter them in Run.", "", "", ""],
      ["Purity", "The term Supplier·Batch needs Supplier in the model too.", "", "", ""],
    ]);
    const broken = [{ index: 1, expression: "T + 25 * P <= 215" }];
    assert.deepEqual(whatIfSentences(responses, broken), [
      "These settings break your limit 2, T + 25 * P <= 215.",
      "These settings lie beyond the runs for Yield's model: confirm its prediction with a run.",
    ]);
    const bare = { ...FIT, label: "Cost", at: predictAt({ ...FIT, mse: null, t: null }, { T: 0 }) };
    assert.deepEqual(whatIfRows([bare])[0].slice(2, 4), ["none", "none"]);
    assert.deepEqual(whatIfSentences([{ ...FIT, at }, { ...bare, at: { ...bare.at, extrapolating: true } }]), [
      "These settings lie beyond the runs for the models of Yield and Cost: confirm their predictions with a run.",
      "No runs are left to estimate the noise of Cost's model, so it gives no interval.",
    ]);
  });
});

describe("the trade-offs between the responses", async () => {
  const { frontier, meetsLimits, shortfall } = await import("../../../web/best.js");

  test("a value's shortfall is smaller where its goal is better met", () => {
    assert.ok(shortfall({ goal: "maximize" }, 70) < shortfall({ goal: "maximize" }, 60));
    assert.ok(shortfall({ goal: "minimize" }, 60) < shortfall({ goal: "minimize" }, 70));
    assert.ok(shortfall({ goal: "target", target: 63 }, 62) < shortfall({ goal: "target", target: 63 }, 65));
    assert.equal(shortfall({ goal: "target", target: 63 }, 63), 0);
  });

  test("a candidate meets the limits when each limited response is between its two ends", () => {
    const responses = { Yield: [60, 65, 70, null], Cost: [50, 55, 60, 52] };
    assert.deepEqual(meetsLimits(responses, {}), [true, true, true, true]);
    assert.deepEqual(meetsLimits(responses, { Yield: [62, null] }), [false, true, true, false]);
    assert.deepEqual(meetsLimits(responses, { Yield: [62, null], Cost: [null, 58] }), [false, true, false, false]);
    assert.deepEqual(meetsLimits(responses, { Yield: [null, null], Other: [1, 2] }), [true, true, true, true]);
  });

  test("the best trade-off holds the candidates no other beats on both responses, in order", () => {
    // Shortfalls: smaller is better on both. Candidate 3 is beaten by 1 on both; 4 ties 0 across and is worse up.
    const across = [1, 2, 3, 2.5, 1];
    const up = [9, 5, 2, 6, 10];
    assert.deepEqual(frontier(across, up, [true, true, true, true, true]), [0, 1, 2]);
    assert.deepEqual(frontier(across, up, [false, true, true, true, true]), [4, 1, 2], "without the first");
    assert.deepEqual(frontier(across, up, [false, false, false, false, false]), []);
    assert.deepEqual(frontier([1, null, 3], [3, 1, null], [true, true, true]), [0], "one with no prediction is left out");
  });
});

describe("the settings that give a target", async () => {
  const { solveSentences, targetCaption, targetLead, targetRows, targetShare } = await import("../../../web/best.js");
  const yieldAt = { name: "Yield", label: "Yield", units: "%", target: 62, predicted: 62, prediction: [59.5, 64.5] };
  const found = { reached: true, share: 0.0698, tried: 20000, near: 0.02, responses: [yieldAt] };

  test("the lead says whether the models give the targets inside the ranges", () => {
    assert.equal(targetLead(found), "These settings give the target, by the models.");
    const two = { ...found, reached: false, responses: [yieldAt, { ...yieldAt, name: "Cost" }] };
    assert.equal(targetLead(two), "No settings inside your ranges give every target, by the models. These come closest.");
    assert.match(targetLead(two, { limits: true }), /^No settings inside your ranges and limits give every target/);
  });

  test("a mixture's answer is said as a blend", () => {
    assert.equal(targetLead(found, { blend: true }), "This blend gives the target, by the models.");
    assert.equal(targetLead({ ...found, reached: false }, { blend: true }), "No blend within your bounds gives the target, by the models. This one comes closest.");
    assert.equal(
      targetShare(found, { blend: true }),
      "About 7% of the 20,000 blends tried give the target to within 2% of the range of the response's results: " +
        "more than one blend does. This is the one nearest the middle of the blends your bounds allow.",
    );
    const triangle = { kind: "ternary", components: ["A", "B", "C"], responses: [{ name: "Yield", z: [60, 64], levels: [62] }] };
    assert.equal(
      targetCaption(triangle, [], [yieldAt]),
      "Each corner is a pure component and each point a blend the bounds allow. The line is where the response is predicted at its " +
        "target: every blend on it gives the target. The star marks the blend found, and the dots your runs.",
    );
    const missed = { ...triangle, responses: [...triangle.responses, { name: "Cost", z: [10, 12], levels: [20] }] };
    const said = targetCaption(missed, [], [yieldAt, { name: "Cost", label: "Cost" }]);
    assert.match(said, /where the lines cross, the blend gives every target\./);
    assert.match(said, / No line for Cost: no blend drawn brings it to its target\.$/);
  });

  test("the share of the settings tried that give the targets is said when some do", () => {
    assert.equal(
      targetShare(found),
      "About 7% of the 20,000 settings tried give the target to within 2% of the range of the response's results: " +
        "more than one setting does. These are the ones nearest the center of your ranges, where your runs are gathered.",
    );
    assert.match(targetShare({ ...found, share: 0.0004 }), /^Fewer than 1 in 1,000 of the 20,000 settings tried/);
    assert.equal(targetShare({ ...found, reached: false, share: 0 }), "");
  });

  test("each response is listed with its target, its prediction and its interval", () => {
    assert.deepEqual(targetRows(found), [["Yield", "62%", "62%", "59.5 to 64.5%"]]);
    assert.deepEqual(targetRows({ responses: [{ ...yieldAt, prediction: null }] }), [["Yield", "62%", "62%", ""]]);
  });

  test("the chart's caption says what a line means, where the others are held, and which response has no line", () => {
    const surface = (name, z, target) => ({ name, z, levels: [target] });
    const factors = [{ name: "T", units: "degC" }, { name: "P", units: "bar" }, { name: "F", units: "L/min" }];
    const one = { x: "T", y: "P", held: [{ name: "F", value: 15 }], responses: [surface("Yield", [[55, 60], [62, 70]], 62)], runnable: null };
    assert.equal(
      targetCaption(one, factors, [yieldAt]),
      "Drawn over T and P, with F at 15 L/min, as at the settings found. The line is where the response is predicted " +
        "at its target: every setting on it gives the target. The star marks the settings found, and the dots your runs.",
    );
    const two = { ...one, held: [], runnable: [[1]], responses: [...one.responses, surface("Cost", [[10, 12], [14, 16]], 40)] };
    const said = targetCaption(two, factors, [yieldAt, { name: "Cost", label: "Cost per kg" }]);
    assert.match(said, /^Drawn over T and P\. Each line is where a response is predicted at its target: /);
    assert.match(said, /where the lines cross, the settings give every target\. The star marks/);
    assert.match(said, / Grey: settings outside your limits\. No line for Cost per kg: no setting drawn brings it to its target\.$/);
  });

  test("one factor solved from the others is said with the settings that still hold the target", () => {
    const solved = { response: "Yield", factor: "T", target: 62, solutions: [{ value: 175.3, interval: [167.6, 182.9] }], nearest: {} };
    const words = { held: "P = 2 bar, F = 15 L/min", units: "degC", label: "Yield", responseUnits: "%" };
    assert.deepEqual(solveSentences(solved, words), [
      "With P = 2 bar, F = 15 L/min held, T = 175.3 degC gives Yield 62%, by the model. From 167.6 degC to 182.9 degC, " +
        "the 95% confidence interval of the prediction still holds the target.",
    ]);
    const open = { ...solved, solutions: [{ value: 175.3, interval: [null, 182.9] }, { value: 190, interval: null }] };
    const [first, second] = solveSentences(open, words);
    assert.match(first, /From the low end of its range to 182.9 degC/);
    assert.match(second, /T = 190 degC gives Yield 62%, by the model. The fit uses every run, so it has no interval.$/);
    const none = { ...solved, target: 90, solutions: [], nearest: { value: 200, predicted: 65.9 } };
    assert.deepEqual(solveSentences(none, words), [
      "With P = 2 bar, F = 15 L/min held, no setting of T in its range gives Yield 90%, by the model. " +
        "The prediction comes closest at 65.9%, at T = 200 degC.",
    ]);
  });
});

describe("the settings that meet a specification", async () => {
  const { shareWords, surestSentence, windowCaption, windowChance, windowLead, windowLevels } = await import("../../../web/best.js");
  const one = { responses: [{ name: "Yield" }], held: [], share: 0.17325, likely: { 0.5: 0.1702, 0.9: 0.08655 } };
  const two = { ...one, responses: [{ name: "Yield" }, { name: "Cost" }], held: [{ name: "S", value: "B" }] };

  test("a share is written to a whole percent, to a tenth below ten, and as less than a tenth", () => {
    assert.deepEqual([0.17325, 0.0344, 0.0004, 0, 1].map(shareWords), ["17%", "3.4%", "less than 0.1%", "0%", "100%"]);
  });

  test("the lead says over what share of the settings the predictions are inside the windows", () => {
    assert.equal(windowLead(one), "By the models, 17% of the settings inside your ranges give the response inside its window.");
    assert.equal(
      windowLead(two, { limits: true }),
      "By the models, 17% of the settings inside your ranges and limits give every response inside every window, with S at B.",
    );
    assert.equal(windowLead({ ...one, share: 0 }), "By the models, no setting inside your ranges gives the response inside its window.");
  });

  test("the chance is said for each line drawn, or why none can be given", () => {
    assert.match(windowChance(one), /^One new run meets its window with a chance of 90% or more at 8\.7% of the settings, and of 50% or more at 17%\./);
    assert.match(windowChance({ ...two, likely: { 0.5: 0.03, 0.9: 0 } }), /^At no setting does one new run meet every window with a chance of 90% or more; it does with 50% or more at 3% of them\./);
    assert.match(windowChance({ ...one, likely: { 0.5: 0, 0.9: 0 } }), /with a chance of 90% or more, nor of 50%\. The noise is wide/);
    assert.equal(windowChance({ ...one, likely: null }), "No runs are left to estimate the noise, so the chance that a new run meets its window cannot be given.");
    assert.equal(windowChance({ ...one, share: 0 }), "");
  });

  test("a mixture's windows are met by blends, and its chart is its triangle", () => {
    assert.equal(windowLead(one, { blend: true }), "By the models, 17% of the blends within your bounds give the response inside its window.");
    assert.equal(windowLead({ ...one, share: 0 }, { blend: true }), "By the models, no blend within your bounds gives the response inside its window.");
    assert.match(windowChance(one, { blend: true }), /^One new run meets its window with a chance of 90% or more at 8\.7% of the blends, and of 50% or more at 17%\./);
    assert.match(windowChance({ ...one, likely: { 0.5: 0.03, 0.9: 0 } }, { blend: true }), /^At no blend does one new run meet its window with a chance of 90% or more/);
    const components = [{ name: "A", kind: "numeric" }, { name: "B", kind: "numeric" }];
    const surest = { settings: { A: 0.62, B: 0.38 }, probability: 0.69 };
    assert.equal(surestSentence({ ...one, surest }, components, {}, { blend: true }), "A new run is surest to meet its window at the blend A = 0.62, B = 0.38: a chance of 69%.");
    const line = (chance, a) => ({ chance, a, b: a, c: a });
    const triangle = { kind: "ternary", at: [0.6, 0.3, 0.1], chances: [line(0.5, [0.5, 0.4, null]), line(0.9, [])] };
    const said = windowCaption(triangle);
    assert.match(said, /^Each corner is a pure component and each point a blend the bounds allow\. Shaded where every response is predicted inside its window; /);
    assert.match(said, /Inside the line of 50%, a new run meets every window at least that often/, "the surest line drawn, not one no blend reaches");
    assert.match(said, /The star marks the blend at which a new run is surest to meet them, and the dots your runs\.$/);
    const unknown = windowCaption({ ...triangle, at: null, chances: null });
    assert.doesNotMatch(unknown, /Inside the line|The star/);
    assert.match(unknown, / The dots are your runs\.$/);
  });

  test("the chance says when the responses' errors were taken as correlated, and only then", () => {
    const together = "The responses' errors are taken as correlated as the residuals of their fits are.";
    const correlated = { ...two, search: { correlation: [[1, 0.6], [0.6, 1]] } };
    assert.ok(windowChance(correlated).endsWith(` count too. ${together}`));
    assert.ok(windowChance({ ...correlated, likely: { 0.5: 0.03, 0.9: 0 } }).endsWith(` the runs are few. ${together}`));
    for (const apart of [two, { ...two, search: { correlation: null } }]) assert.ok(!windowChance(apart).includes("correlated"));
    assert.ok(!windowChance({ ...correlated, likely: null }).includes("correlated"), "no chance is given, so nothing is joined");
  });

  test("each combination of levels has a row: its share, its highest chance, and whether What if has it", () => {
    const at = (supplier, lot) => [{ name: "S", value: supplier }, { name: "Lot", value: lot }];
    const levels = [
      { levels: at("A", 1), share: 0, surest: 0.0312, here: false },
      { levels: at("B", 2), share: 0.17325, surest: 0.9925, here: true },
    ];
    assert.deepEqual(windowLevels({ ...two, levels }), {
      columns: ["S", "Lot", "Settings that meet it", "Highest chance", "Under What if"],
      rows: [["A", "1", "0%", "3.1%", ""], ["B", "2", "17%", "99%", "yes"]],
    });
    const unknown = windowLevels({ ...two, levels: [{ ...levels[0], surest: null }] });
    assert.equal(unknown.rows[0][3], "none given", "no runs are left to estimate the noise");
    assert.equal(windowLevels({ ...one, levels: [] }), null, "a study with no categorical factor has no table");
    assert.equal(windowLevels(one), null, "nor has a reply from before the levels were judged");
  });

  test("the surest settings are said with their chance", () => {
    const factors = [{ name: "T", kind: "numeric" }, { name: "S", kind: "categorical" }];
    const surest = { settings: { T: 180.7, S: "B" }, probability: 0.9925 };
    assert.equal(surestSentence({ ...two, surest }, factors, { T: "degC" }), "A new run is surest to meet every window at T = 180.7 degC, S at B: a chance of 99%.");
    assert.match(surestSentence({ ...one, surest: { ...surest, probability: 0.9996 } }, factors), /: a chance of more than 99%\.$/);
    assert.equal(surestSentence({ ...one, surest: null }, factors), "");
  });

  test("the chart's caption says what is drawn, and the line of the surest chance when there is one", () => {
    const overlay = { x: "T", y: "P", held: [{ name: "F", value: 15 }], probability: [[0.2]], chances: [0.5, 0.9], runnable: [[1]] };
    const said = windowCaption(overlay, [{ name: "F", units: "L/min" }]);
    assert.match(said, /^Drawn over T and P, with F at 15 L\/min, as What if has it\. Shaded where every response is predicted inside its window; /);
    assert.match(said, /Inside the line of 90%, a new run meets every window at least that often/);
    assert.match(said, /The star marks the settings What if holds, and the dots your runs\. Grey: settings outside your limits\.$/);
    assert.doesNotMatch(windowCaption({ ...overlay, probability: null, runnable: null, held: [] }), /Inside the line|Grey|as What if has/);
  });
});
