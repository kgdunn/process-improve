// A study of one factor, its levels compared: the sentence Learn leads with (web/logic.js) and the chart of the runs
// at each level (web/charts.js), from the comparison bootstrap.py's analysis gives.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { LEVEL_AVERAGE, levelsFigure, spreadOf } from "../../../web/charts.js";
import { CAPTIONS, COMPARISON_TESTS, TITLES, comparisonSentence } from "../../../web/logic.js";

const COLOURS = { fg: "#111", muted: "#666", line: "#ddd", accent: "#06c" };

/** A comparison as the analysis gives one: two levels, and the difference between them tested four ways. */
function compared({ estimate = 1.3, low = -1.798, high = 4.398 } = {}) {
  return {
    factor: "method",
    levels: [
      { level: "A", n: 2, mean: 84, std_dev: 1, ci_low: 75, ci_high: 93, runs: [{ run: 1, value: 83 }, { run: 3, value: 85 }] },
      { level: "B", n: 2, mean: 85.3, std_dev: 1, ci_low: 76, ci_high: 94, runs: [{ run: 2, value: 85 }, { run: 4, value: 85.6 }] },
    ],
    difference: {
      estimate,
      from: "A",
      to: "B",
      true: null,
      tests: [
        { test: "pooled", ci_low: low, ci_high: high, p_value: 0.39 },
        { test: "welch", ci_low: low, ci_high: high, p_value: 0.39 },
        { test: "randomization", ci_low: null, ci_high: null, p_value: 0.39, exact: true, n_arrangements: 6 },
        { test: "rank_sum", p_value: 0.5, exact: true },
      ],
    },
  };
}

describe("the comparison of one factor's two levels, in a sentence", () => {
  test("says how far apart the averages are, which way, and that an interval holding zero shows no difference", () => {
    assert.equal(
      comparisonSentence(compared(), { response: "Yield", units: "%" }),
      "On average, Yield is 1.3% higher at B than at A. The 95% interval for the difference, -1.798 to 4.398, holds " +
        "zero, so these runs do not show that B and A differ by more than their noise.",
    );
  });

  test("says an interval clear of zero shows a difference, either way, with the factor's units", () => {
    const lower = compared({ estimate: -12, low: -20, high: -4 });
    for (const level of lower.levels) level.level = level.level === "A" ? 150 : 200;
    Object.assign(lower.difference, { from: 150, to: 200 });
    assert.equal(
      comparisonSentence(lower, { response: "Time", units: "min", factorUnits: "degC" }),
      "On average, Time is 12 min lower at 200 degC than at 150 degC. The 95% interval for the difference, -20 to -4, " +
        "lies below zero, so the difference is larger than the noise in these runs.",
    );
  });

  test("says nothing without a difference: levels with too few runs, or more than two", () => {
    assert.equal(comparisonSentence({ ...compared(), difference: null }, { response: "y" }), null);
    assert.equal(comparisonSentence(null, { response: "y" }), null);
  });

  test("names each test the analysis gives, with the glossary entry that says what it is", () => {
    assert.deepEqual(Object.keys(COMPARISON_TESTS), compared().difference.tests.map((t) => t.test));
    assert.deepEqual(
      Object.values(COMPARISON_TESTS).map((t) => t.key),
      ["pooled-t", "welch-t", "randomization-test", "rank-sum"],
    );
  });
});

describe("the chart of the runs at each level", () => {
  const figure = levelsFigure(compared(), { response: "Yield", units: "%" }, COLOURS);
  const [runs, averages] = figure.data;

  test("draws every run as a dot at its level, spread by its run number so equal results stand apart", () => {
    assert.deepEqual(runs.y, [83, 85, 85, 85.6]);
    assert.deepEqual(runs.x, [0 + spreadOf(1), 0 + spreadOf(3), 1 + spreadOf(2), 1 + spreadOf(4)]);
    assert.ok([1, 2, 3, 4, 20].every((run) => Math.abs(spreadOf(run)) <= 0.15), "each dot stays by its level");
    assert.notEqual(spreadOf(2), spreadOf(4));
    assert.deepEqual(runs.customdata[2], [2, "B"]);
  });

  test("draws each level's average with its 95% interval beside the dots", () => {
    assert.equal(averages.name, LEVEL_AVERAGE);
    assert.deepEqual(averages.x, [0.3, 1.3]);
    assert.deepEqual(averages.y, [84, 85.3]);
    assert.deepEqual(averages.error_y.array, [9, 94 - 85.3]);
    assert.deepEqual(averages.error_y.arrayminus, [9, 85.3 - 76]);
  });

  test("names the levels on the axis, and a level with one run has no interval", () => {
    assert.deepEqual(figure.layout.xaxis.ticktext, ["A", "B"]);
    assert.equal(figure.layout.xaxis.title.text, "method");
    assert.equal(figure.layout.yaxis.title.text, "Yield (%)");
    const one = compared();
    Object.assign(one.levels[1], { n: 1, std_dev: null, ci_low: null, ci_high: null });
    assert.deepEqual(levelsFigure(one, { response: "y" }, COLOURS).data[1].error_y.array, [9, 0]);
  });

  test("has a title and a caption, as every chart of the analysis has", () => {
    assert.equal(TITLES.levels, "The runs at each level");
    assert.match(CAPTIONS.levels, /^Each dot is a run/);
  });
});
