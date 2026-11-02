// The chart of a field's experiments (web/field-chart.js): where each mark is drawn, without a document.
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { FRAME, KINDS, MOST_FACTORS, RUN_LINES, chartMarks, chartX, chartY } from "../../../web/field-chart.js";

describe("placing a design on a field's chart", () => {
  test("puts one factor at the left edge and the most factors at the right", () => {
    assert.equal(chartX(1), FRAME.left);
    assert.equal(chartX(MOST_FACTORS), FRAME.width - FRAME.right);
    assert.equal(chartX(24), chartX(MOST_FACTORS), "more factors than the chart spans sit at its edge");
  });

  test("doubles the runs at each line, the fewest at the bottom and the most at the top", () => {
    assert.equal(chartY(RUN_LINES[0]), FRAME.height - FRAME.bottom);
    assert.equal(chartY(RUN_LINES.at(-1)), FRAME.top);
    const steps = RUN_LINES.slice(1).map((runs, i) => chartY(RUN_LINES[i]) - chartY(runs));
    assert.ok(steps.every((step) => Math.abs(step - steps[0]) < 1e-9), "each doubling is one equal step up");
    assert.equal(chartY(2), chartY(RUN_LINES[0]));
    assert.equal(chartY(500), chartY(RUN_LINES.at(-1)));
  });

  test("draws a larger mark for more experiments of one size and kind, up to a limit", () => {
    const [one, four, many] = chartMarks([[3, 8, "full", 1], [3, 8, "full", 4], [3, 8, "full", 400]]);
    assert.ok(one.size < four.size);
    assert.ok(four.size < many.size);
    assert.equal(many.size, 16);
  });

  test("sets kinds of one size side by side, and reads an unknown kind as other", () => {
    const marks = chartMarks([[3, 8, "full", 1], [3, 8, "screening", 1], [3, 8, "something", 1]]);
    assert.ok(marks[0].x < marks[1].x, "a full factorial left of a screening design of the same size");
    assert.equal(marks[2].kind, "other");
    assert.deepEqual(KINDS, ["full", "screening", "surface", "mixture", "other"]);
  });
});
