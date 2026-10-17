// web/best.js: what Learn says of the best settings for the response's goal, and the run sheet's confirmation runs.
// The facts come from bootstrap.best_settings; tests/web/api/test_best_settings.py holds them to the model.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  CONFIRMATION,
  CONFIRMATION_RUNS,
  CONFIRMS,
  UNCLEAR,
  ascentSentence,
  bestContourCaption,
  confirmationRows,
  confirmationSentences,
  confirmedAt,
  designRows,
  extrapolationSentence,
  flatSentence,
  intervalRows,
  leadSentence,
  noGoalSentence,
  noIntervalSentence,
  num,
  sensitivityCaption,
  settingWords,
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

  test("taken off again, with their results and sources", () => {
    const added = withConfirmationRuns(SHEET, { T: 200, P: 1, F: 18.04 });
    const off = withoutConfirmationRuns(added, [50, 51, 67, 68, null], ["entered", "entered", "simulated", "simulated", null]);
    assert.deepEqual(off, {
      columns: SHEET.columns,
      rows: SHEET.rows,
      responses: [50, 51],
      sources: ["entered", "entered"],
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
