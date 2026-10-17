// web/charts.js: the figures the page draws, built without a browser.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  BEST_MARKER,
  MOVES,
  PALE,
  SIGN_COLOURS,
  TRUE_MARKER,
  coefficientFigure,
  correlationFigure,
  fdsFigure,
  markBest,
  niceStep,
  pageLayout,
  paretoFigure,
  prettyTerm,
  sensitivityFigure,
  sensitivityHeight,
  shareContourScale,
  termRowsHeight,
} from "../../../web/charts.js";

const COLOURS = { fg: "#111", muted: "#666", line: "#ddd", accent: "#06c" };

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

  test("the legend says what the line across each dot is: its 95% interval", () => {
    const names = (figure) => figure.data.map((trace) => trace.name);
    assert.deepEqual(names(coefficientFigure(coefficients, false, COLOURS)), ["Estimate", "95% interval"]);
    const key = coefficientFigure(coefficients, false, COLOURS).data[1];
    assert.deepEqual([key.mode, key.x, key.line.color], ["lines", [null], COLOURS.accent], "a key, not data");
    const saturated = coefficients.map((c) => ({ ...c, ci_low: null, ci_high: null }));
    assert.deepEqual(names(coefficientFigure(saturated, false, COLOURS)), ["Estimate"], "no interval, no key");
  });

  test("the true values beside the estimates, only when the results were simulated", () => {
    const { data } = coefficientFigure(coefficients, true, COLOURS);
    const [truth, estimates] = data;
    assert.equal(truth.name, "True value", "drawn first, so an estimate on its true value shows inside the diamond");
    assert.equal(estimates.name, "Estimate");
    assert.deepEqual(truth.x, [4, 1.5]);
    assert.deepEqual(data.map((trace) => trace.legendrank), [3, 1, 2], "the legend still lists the estimate first");
    assert.equal(truth.marker, TRUE_MARKER);
  });

  test("a row for every term, whatever their number, each with its label", () => {
    assert.equal(coefficientFigure(coefficients, false, COLOURS).height, 240);
    const many = Array.from({ length: 20 }, (_, i) => ({ term: `X${i}`, label: `X${i}`, coefficient: i }));
    assert.equal(coefficientFigure(many, false, COLOURS).height, termRowsHeight(20));
    assert.equal(termRowsHeight(20), 28 * 20 + 130);
    const { yaxis } = coefficientFigure(many, false, COLOURS).layout;
    assert.equal(yaxis.dtick, 1, "every term labelled, none skipped for want of room");
  });

  test("the legend sits above the plot, clear of the axis title below it", () => {
    const layout = pageLayout(coefficientFigure(coefficients, true, COLOURS).layout, 300, COLOURS);
    assert.equal(layout.legend.yanchor, "bottom");
    assert.ok(layout.legend.y > 1);
  });
});

describe("the book's Pareto plot of the coefficients", () => {
  const coefficients = [
    { term: "Intercept", label: "Intercept", coefficient: 62, ci_low: 61, ci_high: 63, true_value: 62 },
    { term: "T", label: "T", coefficient: 1.5, ci_low: 1.0, ci_high: 2.0, true_value: 2 },
    { term: "P", label: "P", coefficient: -4, ci_low: -4.5, ci_high: -3.5, true_value: -3 },
    { term: "T:P", label: "T·P", coefficient: 0.25, ci_low: -0.25, ci_high: 0.75, true_value: 0 },
  ];

  test("sizes sorted, the largest at the top, orange for positive and blue for negative", () => {
    const { data, layout } = paretoFigure(coefficients, false, null, COLOURS);
    assert.deepEqual(layout.yaxis.categoryarray, ["P", "T", "T·P"]);
    assert.equal(layout.yaxis.autorange, "reversed");
    const [positive, negative] = data;
    assert.deepEqual([positive.name, positive.y, positive.x], ["Positive", ["T", "T·P"], [1.5, 0.25]]);
    assert.deepEqual([negative.name, negative.y, negative.x], ["Negative", ["P"], [4]]);
    assert.equal(positive.marker.color, SIGN_COLOURS.positive);
    assert.equal(negative.marker.color, SIGN_COLOURS.negative);
    assert.deepEqual(SIGN_COLOURS, { positive: "#E69F00", negative: "#0072B2" }, "the book's Okabe-Ito pair");
  });

  test("each bar carries its 95% interval when there are degrees of freedom for one", () => {
    const [positive, negative, key] = paretoFigure(coefficients, false, null, COLOURS).data;
    assert.deepEqual(positive.error_x.array, [0.5, 0.5]);
    assert.deepEqual(negative.error_x.array, [0.5]);
    assert.deepEqual([key.name, key.line.color], ["95% interval", COLOURS.fg], "the legend names the line");
    assert.equal(paretoFigure(coefficients, false, null, COLOURS).layout.shapes.length, 0);
  });

  test("with none, Lenth's margins are drawn on the coefficients' scale: half the effects'", () => {
    const saturated = coefficients.map((c) => ({ ...c, ci_low: null, ci_high: null }));
    const { data, layout } = paretoFigure(saturated, false, { PSE: 1, ME: 3, SME: 6 }, COLOURS);
    assert.equal(data[0].error_x.visible, false);
    assert.deepEqual(layout.shapes.map((s) => [s.x0, s.line.dash]), [[1.5, "dash"], [3, "dot"]]);
    assert.deepEqual(layout.annotations.map((a) => a.text), ["ME", "SME"]);
    const keys = data.filter((trace) => trace.mode === "lines").map((trace) => [trace.name, trace.line.dash]);
    assert.deepEqual(keys, [
      ["Margin of error (ME)", "dash"],
      ["Simultaneous margin of error (SME)", "dot"],
    ]);
  });

  test("the true sizes beside the bars, only when the results were simulated", () => {
    assert.ok(paretoFigure(coefficients, false, null, COLOURS).data.every((trace) => trace.name !== "True size"));
    const truth = paretoFigure(coefficients, true, null, COLOURS).data.at(-1);
    assert.equal(truth.name, "True size");
    assert.deepEqual([truth.marker.color, truth.marker.line.color], ["#ffffff", "#1a1a1a"], "white face, dark edge");
    assert.deepEqual(truth.x, [3, 2, 0]);
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

  test("with limits on the settings, the curves run over the settings you can run, not the cube", () => {
    const fds = { "D-optimal [n=14]": { fraction: [0, 1], variance: [0.2, 0.5] } };
    assert.equal(fdsFigure(fds).layout.xaxis.title.text, "Fraction of the cube");
    const limited = fdsFigure(fds, { region: true });
    assert.equal(limited.layout.xaxis.title.text, "Fraction of the settings you can run");
    assert.match(limited.data[0].hovertemplate, /of the settings you can run</);
  });

  test("a correlation map with lines between the main effects, the squares and the interactions", () => {
    const terms = ["A", "B", "I(A ** 2)", "I(B ** 2)", "A:B"];
    const matrix = terms.map(() => terms.map(() => 0.5));
    const figure = correlationFigure(terms, matrix, 2, COLOURS);
    assert.deepEqual(figure.data[0].x, ["A", "B", "A²", "B²", "A·B"]);
    assert.equal(figure.layout.shapes.length, 4, "two lines at each of the two block edges");
  });
});

describe("the best settings", () => {
  const SENSITIVITY = [
    { name: "F", step: 0.5, down: -0.018, up: -0.018, down_runnable: true, up_runnable: true, in_model: true, units: "L/min" },
    { name: "T", step: 2.5, down: -0.247, up: 0.247, down_runnable: true, up_runnable: false, in_model: true, units: "degC" },
    { name: "P", step: 0.1, down: 0.164, up: -0.167, down_runnable: false, up_runnable: true, in_model: true, units: "bar" },
    { name: "Q", step: 1, down: 0, up: 0, down_runnable: true, up_runnable: true, in_model: false, units: "" },
  ];
  const WORDS = { label: "Yield", units: "%" };
  const colours = { ...COLOURS, warn: "#a35f00" };

  test("a bar each way for each factor in the model, the one that moves the prediction most at the top", () => {
    const figure = sensitivityFigure(SENSITIVITY, WORDS, colours);
    const [down, up] = figure.data;
    assert.deepEqual([down.name, up.name], [MOVES.down, MOVES.up]);
    assert.deepEqual(down.y, ["T", "P", "F"], "Q is not in the model");
    assert.deepEqual(down.x, [-0.247, 0.164, -0.018]);
    assert.deepEqual(up.x, [0.247, -0.167, -0.018]);
    assert.deepEqual([down.marker.color, up.marker.color], [colours.warn, colours.accent]);
    assert.equal(figure.layout.barmode, "group");
    assert.deepEqual(figure.layout.yaxis.categoryarray, ["T", "P", "F"]);
    assert.equal(figure.layout.yaxis.autorange, "reversed", "the first factor at the top");
    assert.equal(figure.layout.xaxis.title.text, "Change in Yield (%)");
    assert.equal(figure.height, sensitivityHeight(3));
  });

  test("a move outside the ranges or limits is pale, and the legend says what pale is", () => {
    const figure = sensitivityFigure(SENSITIVITY, WORDS, colours);
    const [down, up, key] = figure.data;
    assert.deepEqual(down.marker.opacity, [1, 0.3, 1]);
    assert.deepEqual(up.marker.opacity, [0.3, 1, 1]);
    assert.match(up.customdata[0], /raised 2.5 degC, outside your ranges or limits/);
    assert.equal(key.name, PALE);
    assert.deepEqual([key.type, key.x], ["scatter", [null]], "a key, which takes no place among the bars");
    const inside = sensitivityFigure([SENSITIVITY[0]], WORDS, colours);
    assert.equal(inside.data.length, 2, "nothing pale, no key");
  });

  test("the star at the best settings, on the surface's coded axes, drawn whole at its edge", () => {
    const figure = { data: [{ type: "contour", z: [[1]] }], layout: {} };
    const marked = markBest(figure, { T: 1, P: -1, F: 0.6 }, ["T", "P"], "Best settings: T = 200 degC, P = 1 bar");
    const star = marked.data.at(-1);
    assert.deepEqual([star.x, star.y], [[1], [-1]]);
    assert.deepEqual(star.marker, BEST_MARKER);
    assert.equal(star.marker.symbol, "star");
    assert.equal(star.cliponaxis, false);
    assert.equal(star.showlegend, false, "the caption says what the star is");
    assert.equal(star.hovertemplate, "Best settings: T = 200 degC, P = 1 bar<extra></extra>");
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
