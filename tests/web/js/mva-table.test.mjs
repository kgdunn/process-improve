// web/mva/table.js: what the data table draws, worked out apart from the canvas it draws on.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BINS,
  HEADER_HEIGHT,
  ID_WIDTH,
  ROW_HEIGHT,
  cellSentence,
  columnProfile,
  envelope,
  formatCell,
  histogram,
  hitTest,
  isMissing,
  missingStrip,
  rowSpan,
  visibleColumns,
  visibleRows,
} from "../../../web/mva/table.js";

describe("cells", () => {
  test("numbers show six significant figures without trailing zeros; missing cells are blank", () => {
    assert.equal(formatCell(208.17), "208.17");
    assert.equal(formatCell(1 / 3), "0.333333");
    assert.equal(formatCell(160326), "160326");
    assert.equal(formatCell(1234567.891), "1234570");
    assert.equal(formatCell(2.5e-7), "2.5000e-7");
    assert.equal(formatCell(Number.NaN), "");
    assert.equal(formatCell(null), "");
    assert.equal(formatCell("B12"), "B12");
  });

  test("missing is null, undefined, NaN or empty text, and nothing else", () => {
    assert.deepEqual([null, undefined, Number.NaN, "", 0, "0", false].map(isMissing), [
      true,
      true,
      true,
      true,
      false,
      false,
      false,
    ]);
  });

  test("the live region reads the cell, and says whether its row is selected or excluded", () => {
    assert.equal(cellSentence("52", "Tin", 208.17), "Row 52, Tin: 208.17.");
    assert.equal(
      cellSentence("52", "Tin", null, { selected: true, excluded: true }),
      "Row 52, Tin: missing, selected, excluded from the model.",
    );
  });
});

describe("a column's header", () => {
  test("a number column gets a histogram, its run-order line and its missing strip", () => {
    const values = [1, 2, 3, null, 5, 6, 7, 8, 9, 10];
    const profile = columnProfile(values, { bins: 3, points: 5 });
    assert.equal(profile.kind, "number");
    assert.equal(profile.missing, 1);
    assert.deepEqual(profile.flags, ["10% missing"]);
    assert.deepEqual(profile.histogram, [3, 2, 4]);
    assert.equal(profile.sequence.length, 5);
    assert.deepEqual(profile.strip, [0, 0.5, 0, 0, 0]);
  });

  test("a constant column is flagged; its values fall in the middle bin", () => {
    const profile = columnProfile([4, 4, 4]);
    assert.deepEqual(profile.flags, ["constant"]);
    assert.equal(profile.histogram[Math.floor(BINS / 2)], 3);
  });

  test("a text column shows its commonest categories, and text among numbers is flagged", () => {
    const grades = columnProfile(["A", "B", "A", "C", "A", "B", "D", "E", "F", "G"]);
    assert.equal(grades.kind, "text");
    assert.deepEqual(grades.categories.slice(0, 2), [
      { name: "A", count: 3 },
      { name: "B", count: 2 },
    ]);
    assert.deepEqual(grades.categories.at(-1), { name: "others", count: 2 });
    assert.equal(grades.distinct, 7);
    assert.deepEqual(columnProfile([1, 2, "n/a"]).flags, ["1 not numbers"]);
  });

  test("the histogram puts the maximum in the last bin, not past it", () => {
    assert.deepEqual(histogram([0, 5, 10], 0, 10, 2), [1, 2]);
  });

  test("a long column's line keeps its spikes: each point is a bucket's minimum and maximum", () => {
    const values = Array.from({ length: 1000 }, (_, i) => (i === 517 ? 100 : 0));
    const line = envelope(values, 100);
    assert.equal(line.length, 100);
    assert.deepEqual(line[51], { at: 510, lo: 0, hi: 100 });
    assert.deepEqual(envelope([null, null, 1], 3)[0], { at: 0, lo: null, hi: null });
    assert.deepEqual(missingStrip([null, 1, null, null], 2), [0.5, 1]);
  });
});

describe("what is in view", () => {
  test("only the rows in view are drawn, with a few more each side", () => {
    assert.deepEqual(visibleRows(0, 220, 100_000), { first: 0, last: 14 });
    assert.deepEqual(visibleRows(ROW_HEIGHT * 50_000, 220, 100_000), { first: 49_996, last: 50_014 });
    assert.deepEqual(visibleRows(0, 220, 3), { first: 0, last: 3 });
  });

  test("only the columns in view are drawn, after the frozen id column", () => {
    const widths = new Array(1000).fill(100);
    const { first, last, offsets, total } = visibleColumns(5000, 500, widths);
    assert.equal(offsets[0], ID_WIDTH);
    assert.equal(first, 50);
    assert.equal(last, 55);
    assert.equal(total, ID_WIDTH + 100_000);
  });

  test("a point of the grid names its header, row id or cell", () => {
    const widths = [100, 100];
    const { offsets } = visibleColumns(0, 400, widths);
    const at = (x, y, scrollLeft = 0) => hitTest(x, y, { offsets, widths, count: 10, scrollLeft });
    assert.deepEqual(at(ID_WIDTH + 150, 10), { kind: "header", row: null, column: 1 });
    assert.deepEqual(at(10, HEADER_HEIGHT + ROW_HEIGHT * 2 + 1), { kind: "id", row: 2, column: null });
    assert.deepEqual(at(ID_WIDTH + 5, HEADER_HEIGHT + 1), { kind: "cell", row: 0, column: 0 });
    assert.equal(at(ID_WIDTH + 5, HEADER_HEIGHT + ROW_HEIGHT * 10 + 1), null, "past the last row");
    assert.deepEqual(
      at(150 + 10, HEADER_HEIGHT + 1, 150),
      { kind: "id", row: 0, column: null },
      "the id column stays put",
    );
  });

  test("Shift-click selects the run of rows from the anchor, either way", () => {
    const ids = ["a", "b", "c", "d"];
    assert.deepEqual(rowSpan(ids, 1, 3), ["b", "c", "d"]);
    assert.deepEqual(rowSpan(ids, 3, 1), ["b", "c", "d"]);
  });
});
