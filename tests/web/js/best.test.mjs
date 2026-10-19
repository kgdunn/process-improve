// web/best.js: what Learn says of the best settings for the response's goal, and the run sheet's confirmation runs.
// The facts come from bootstrap.best_settings; tests/web/api/test_best_settings.py holds them to the model.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BLEND_SENSITIVITY,
  BLEND_UNCLEAR,
  CONFIRMATION,
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
  designRows,
  extrapolationSentence,
  flatSentence,
  intervalRows,
  leadSentence,
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
} from "../../../web/best.js";
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
    });
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
