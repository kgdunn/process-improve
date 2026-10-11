// web/charts.js: the figures the page draws, built without a browser.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  coefficientFigure,
  correlationFigure,
  fdsFigure,
  niceStep,
  pageLayout,
  prettyTerm,
  shareContourScale,
} from "../../../web/charts.js";

const COLOURS = { fg: "#111", muted: "#666", line: "#ddd", accent: "#06c", truth: "#0a0" };

describe("terms as the page writes them", () => {
  test("squares and products", () => {
    assert.equal(prettyTerm("I(T ** 2)"), "T²");
    assert.equal(prettyTerm("T:P"), "T·P");
    assert.equal(prettyTerm("I(Temp ** 2) + T:P:F"), "Temp² + T·P·F");
  });

  test("a sentence's own colons are left alone", () => {
    assert.equal(prettyTerm("Note: T is aliased"), "Note: T is aliased");
  });
});

describe("the coefficient chart", () => {
  const coefficients = [
    { term: "Intercept", label: "Intercept", coefficient: 62, ci_low: 61, ci_high: 63, true_value: 62 },
    { term: "T", label: "T", coefficient: 4.1, ci_low: 3.5, ci_high: 4.6, true_value: 4 },
    { term: "T:P", label: "T·P", coefficient: 1.4, ci_low: null, ci_high: null, true_value: 1.5 },
  ];

  test("each term but the intercept, with its interval", () => {
    const [estimates] = coefficientFigure(coefficients, false, COLOURS).data;
    assert.deepEqual(estimates.y, ["T", "T·P"]);
    assert.deepEqual(estimates.x, [4.1, 1.4]);
    assert.deepEqual(estimates.error_x.array.map((v) => +v.toFixed(9)), [0.5, 0]);
    assert.deepEqual(estimates.error_x.arrayminus.map((v) => +v.toFixed(9)), [0.6, 0]);
  });

  test("the true values beside the estimates, only when the results were simulated", () => {
    assert.equal(coefficientFigure(coefficients, false, COLOURS).data.length, 1);
    const truth = coefficientFigure(coefficients, true, COLOURS).data[1];
    assert.equal(truth.name, "True value");
    assert.deepEqual(truth.x, [4, 1.5]);
  });

  test("a row for every term, whatever their number", () => {
    assert.equal(coefficientFigure(coefficients, false, COLOURS).height, 220);
    const many = Array.from({ length: 20 }, (_, i) => ({ term: `X${i}`, label: `X${i}`, coefficient: i }));
    assert.equal(coefficientFigure(many, false, COLOURS).height, 30 * 20 + 100);
  });
});

describe("the response surfaces", () => {
  test("one colour scale and one set of levels for both", () => {
    const fitted = { data: [{ z: [[0, 1], [2, 3]] }] };
    const truth = { data: [{ z: [[-1, 1], [2, 4.5]] }] };
    shareContourScale(fitted, truth);
    for (const figure of [fitted, truth]) {
      assert.equal(figure.data[0].zmin, -1);
      assert.equal(figure.data[0].zmax, 4.5);
      assert.equal(figure.data[0].contours.size, 0.5);
      assert.equal(figure.data[0].contours.start, -1);
    }
  });

  test("round steps between contour levels", () => {
    assert.deepEqual([0.37, 1, 1.2, 3, 7, 0].map(niceStep), [0.5, 1, 2, 5, 10, 1]);
  });
});

describe("the comparison's charts", () => {
  test("an FDS curve for each design", () => {
    const fds = { "BBD [n=15]": { fraction: [0, 1], variance: [0.2, 0.9] }, "CCD [n=17]": { fraction: [0, 1], variance: [0.3, 0.6] } };
    const figure = fdsFigure(fds);
    assert.deepEqual(figure.data.map((t) => t.name), ["BBD [n=15]", "CCD [n=17]"]);
    assert.deepEqual(figure.data[1].y, [0.3, 0.6]);
  });

  test("a correlation map with lines between the main effects, the squares and the interactions", () => {
    const terms = ["A", "B", "I(A ** 2)", "I(B ** 2)", "A:B"];
    const matrix = terms.map(() => terms.map(() => 0.5));
    const figure = correlationFigure(terms, matrix, 2, COLOURS);
    assert.deepEqual(figure.data[0].x, ["A", "B", "A²", "B²", "A·B"]);
    assert.equal(figure.layout.shapes.length, 4, "two lines at each of the two block edges");
  });
});

describe("a card's layout", () => {
  test("the page's colours, its own height, no title, and no fixed width", () => {
    const layout = pageLayout({ title: "x", width: 900, xaxis: { scaleanchor: "y" } }, 300, COLOURS);
    assert.equal(layout.height, 300);
    assert.equal(layout.font.color, COLOURS.fg);
    assert.equal(layout.title, undefined);
    assert.equal(layout.width, undefined);
    assert.equal(layout.xaxis.constrain, "domain", "equal axes keep their range and shrink the plot");
  });
});
