// web/mva/views.js and web/mva/link.js: the linked figures, and how the reader's lasso reaches the store.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { bindView, idsOf, modeOf } from "../../../web/mva/link.js";
import { Selection } from "../../../web/mva/selection.js";
import {
  CATEGORY_COLOURS,
  FADED,
  MARKER,
  SIGN_COLOURS,
  WEBGL_ABOVE,
  barFigure,
  chooseLabels,
  componentTitle,
  componentsFigure,
  componentsSentence,
  encodingKind,
  labelPositions,
  markerSizes,
  plain,
  pointStyles,
  scatterFigure,
  sequenceFigure,
  viridis,
} from "../../../web/mva/views.js";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A score plot's points: a cloud, and three far away. */
const cloud = () => {
  const points = [];
  for (let i = 0; i < 30; i += 1)
    points.push({ id: String(i), x: Math.cos(i) * 0.5, y: Math.sin(2 * i) * 0.5, label: `B${i}` });
  points.push(
    { id: "52", x: 4, y: 3, label: "B52" },
    { id: "53", x: 4.5, y: 3.2, label: "B53" },
    { id: "54", x: 5, y: 3.5, label: "B54" },
  );
  return points;
};

const state = (overrides = {}) => ({ selected: new Set(), hovered: null, excluded: new Set(), sets: [], ...overrides });

describe("encodings", () => {
  test("an axis title carries the percent of variance in square brackets", () => {
    assert.equal(componentTitle("t", 1, 0.2771), "t1 [27.7%]");
    assert.equal(componentTitle("p", 2, Number.NaN), "p2");
  });

  test("numbers encode on a scale, anything else as categories", () => {
    assert.equal(encodingKind([1, 2.5, null]), "number");
    assert.equal(encodingKind([1, "A"]), "category");
    assert.equal(encodingKind([null, ""]), "category");
  });

  test("marker area, not diameter, is in proportion to the value", () => {
    const [small, middle, large] = markerSizes([0, 5, 10]);
    assert.equal(small, MARKER.min);
    assert.equal(large, MARKER.max);
    const area = (d) => d * d;
    assert.ok(Math.abs(area(middle) - (area(small) + area(large)) / 2) < 1e-9);
    assert.deepEqual(markerSizes([3, 3]), [
      Math.sqrt((MARKER.min ** 2 + MARKER.max ** 2) / 2),
      Math.sqrt((MARKER.min ** 2 + MARKER.max ** 2) / 2),
    ]);
  });

  test("the colour scale runs from dark to light", () => {
    assert.equal(viridis(0), "rgb(68, 1, 84)");
    assert.equal(viridis(1), "rgb(253, 231, 37)");
    assert.equal(viridis(2), viridis(1), "clamped");
  });

  test("text bound for Plotly is escaped, because Plotly reads a label as HTML", () => {
    assert.equal(plain('<img src=x onerror="alert(1)">'), "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;");
  });
});

describe("the look of each point follows the store", () => {
  const points = cloud();
  test("an excluded point is hollow, a selected one outlined, and the rest faded while one is selected", () => {
    const styles = pointStyles(points, state({ selected: new Set(["1"]), excluded: new Set(["52"]) }), {
      colour: null,
    });
    const at = (id) => styles[points.findIndex((p) => p.id === id)];
    assert.equal(at("52").symbol, "circle-open");
    assert.equal(at("1").outlineWidth, 2);
    assert.equal(at("1").opacity, 1);
    assert.equal(at("2").opacity, FADED);
  });

  test("the hovered point is drawn larger in every view", () => {
    const styles = pointStyles(points, state({ hovered: "3" }), { colour: null });
    assert.equal(styles[3].size, MARKER.size * 1.6);
  });

  test("a row in a saved set takes the set's colour, unless the points encode colour themselves", () => {
    const sets = [{ name: "Upset", rows: new Set(["52"]), colour: "#E69F00" }];
    assert.equal(pointStyles(points, state({ sets }), { colour: null })[30].colour, "#E69F00");
    const coloured = points.map((p, i) => ({ ...p, colour: i % 2 ? "A" : "B" }));
    assert.equal(
      pointStyles(coloured, state({ sets }), { colour: "category" })[30].colour,
      CATEGORY_COLOURS[0],
      "B comes first",
    );
  });
});

describe("labels", () => {
  test("the selection and the extremes carry labels, up to the limit", () => {
    const chosen = chooseLabels(cloud(), state({ selected: new Set(["4"]) }), { limit: 4 });
    assert.deepEqual([...chosen].sort(), ["4", "52", "53", "54"]);
    assert.equal(chooseLabels(cloud(), state(), { labels: "none" }).size, 0);
    assert.equal(chooseLabels(cloud(), state(), { labels: "all" }).size, 33);
  });

  test("a label goes on the side away from its neighbours", () => {
    const points = [
      { id: "a", x: 0, y: 0 },
      { id: "left", x: -1, y: 0 },
      { id: "left2", x: -1.2, y: 0.1 },
    ];
    assert.equal(labelPositions(points, new Set(["a"]))[0], "middle right");
    const above = [
      { id: "a", x: 0, y: 0 },
      { id: "b", x: 0, y: 1 },
    ];
    assert.equal(labelPositions(above, new Set(["a"]))[0], "bottom center");
  });
});

describe("the scatter", () => {
  test("fitted and excluded points are two traces, each point carrying its id for the lasso", () => {
    const figure = scatterFigure(cloud(), state({ excluded: new Set(["52", "53", "54"]) }), {
      xTitle: "t1 [27.7%]",
    });
    const [fitted, excluded] = figure.data;
    assert.equal(fitted.name, "Fitted");
    assert.equal(excluded.name, "Excluded, projected");
    assert.deepEqual(excluded.customdata, ["52", "53", "54"]);
    assert.ok(excluded.marker.symbol.every((s) => s.endsWith("-open")));
    assert.equal(figure.layout.xaxis.title.text, "t1 [27.7%]");
    assert.equal(figure.layout.dragmode, "lasso");
    assert.equal(fitted.selectedpoints, null);
  });

  test("with nothing excluded there is one trace and no legend for it", () => {
    const figure = scatterFigure(cloud(), state());
    assert.equal(figure.data.length, 1);
    assert.equal(figure.data[0].showlegend, false);
  });

  test("colour categories get legend keys, the ellipse a dashed line, and an annotation an arrow", () => {
    const points = cloud().map((p, i) => ({ ...p, colour: i < 30 ? "Normal" : "Upset" }));
    const ellipse = { x: [1, 0, -1, 0, 1], y: [0, 1, 0, -1, 0] };
    const figure = scatterFigure(points, state(), {
      colourTitle: "Regime",
      ellipse,
      annotations: [{ id: "54", text: "Batch 54" }],
    });
    const keys = figure.data.filter((t) => t.legendgroup === "colour");
    assert.deepEqual(
      keys.map((k) => k.name),
      ["Normal", "Upset"],
    );
    assert.equal(keys[0].legendgrouptitle.text, "Regime");
    assert.equal(figure.data.at(-1).line.dash, "dash");
    assert.equal(figure.layout.annotations[0].x, 5);
    assert.equal(figure.layout.annotations[0].showarrow, true);
  });

  test("many points are drawn with WebGL", () => {
    const many = Array.from({ length: WEBGL_ABOVE + 1 }, (_, i) => ({ id: String(i), x: i, y: i }));
    assert.equal(scatterFigure(many, state()).data[0].type, "scattergl");
  });

  test("a hostile label reaches Plotly escaped", () => {
    const points = [{ id: "1", x: 0, y: 0, label: "<b onmouseover=x>" }];
    const figure = scatterFigure(points, state(), { labels: "all" });
    assert.equal(figure.data[0].text[0], "&lt;b onmouseover=x&gt;");
  });
});

describe("bars and sequences", () => {
  test("bars take their sign's colour and can be sorted by size", () => {
    const items = [
      { id: "Tin", value: 0.2 },
      { id: "z2", value: -1.3 },
      { id: "Tmax2", value: 0.7 },
    ];
    const figure = barFigure(items, state({ selected: new Set(["z2"]) }), {
      sort: true,
      limits: [{ value: 1, label: "VIP = 1" }],
    });
    const [bars] = figure.data;
    assert.deepEqual(bars.customdata, ["z2", "Tmax2", "Tin"]);
    assert.deepEqual(bars.marker.color, [SIGN_COLOURS.negative, SIGN_COLOURS.positive, SIGN_COLOURS.positive]);
    assert.deepEqual(bars.marker.opacity, [1, FADED, FADED]);
    assert.equal(figure.layout.shapes[0].y0, 1);
  });

  test("a sequence marks the group's rows and draws excluded rows hollow", () => {
    const series = [{ name: "z2", ids: ["1", "2", "3", "4"], y: [0.58, 0.57, 0.61, 0.62] }];
    const figure = sequenceFigure(series, state({ excluded: new Set(["4"]) }), { group: new Set(["3", "4"]) });
    const [line, marks] = figure.data;
    assert.deepEqual(line.x, [1, 2, 3, 4]);
    assert.equal(line.marker.symbol[3], "circle-open");
    assert.deepEqual(marks.customdata, ["3", "4"]);
  });
});

describe("how many components", () => {
  const cv = {
    components: [1, 2, 3],
    r2: [0.4, 0.7, 0.8],
    q2: [0.3, 0.5, 0.45],
    q2se: [0.05, 0.04, 0.06],
    recommended: 2,
    rule: "the lowest prediction error",
    heldOut: "cells",
    folds: 5,
    block: "X",
  };

  test("R² and Q² stand side by side, Q² with its standard error, the count in use outlined", () => {
    const figure = componentsFigure(cv, { current: 3 });
    const [r2, q2] = figure.data;
    assert.deepEqual([r2.name, q2.name], ["R²X, fitted rows", "Q²X, held out"]);
    assert.deepEqual(q2.error_y.array, cv.q2se);
    assert.deepEqual(q2.marker.line.width, [0, 0, 2]);
    assert.equal(figure.layout.xaxis.dtick, 1);
    assert.equal(figure.layout.annotations[0].x, 2);
    assert.equal(figure.layout.annotations[0].y, 0.7);
  });

  test("a cross-validation of a model since changed is drawn faded", () => {
    const figure = componentsFigure({ ...cv, block: "Y" }, { stale: true });
    assert.equal(figure.data[1].name, "Q²Y, held out");
    assert.ok(figure.data.every((trace) => trace.marker.opacity === FADED));
  });

  test("a recommendation outside the counts draws no arrow", () => {
    assert.deepEqual(componentsFigure({ ...cv, recommended: 9 }).layout.annotations, []);
  });

  test("the sentence names the recommendation, its rule, the count in use and what was held out", () => {
    assert.equal(
      componentsSentence(cv, 3),
      "Recommended: 2 components, by the lowest prediction error. With 3: Q²X 0.45 ± 0.06, R²X 0.80. Q²X is R²X " +
        "measured on single cells held out of the fitted rows, each predicted from the rest of its row by a model " +
        "that never saw them (5 fits per count).",
    );
    assert.match(componentsSentence({ ...cv, recommended: 1, heldOut: "rows" }), /^Recommended: 1 component, .*rows/);
    assert.match(componentsSentence({ ...cv, q2: [0.3, 0.5, null] }, 3), /With 3: Q²X not available ± 0.06/);
    assert.doesNotMatch(componentsSentence(cv, 7), /With/);
  });
});

describe("binding a view to the store", () => {
  /** A stand-in for Plotly and a <div>: it records what is drawn and lets a test fire Plotly's events. */
  const standIn = () => {
    const handlers = {};
    const div = { on: (name, fn) => (handlers[name] = fn) };
    const drawn = [];
    const Plotly = {
      newPlot: (_div, data) => drawn.push(["newPlot", data]),
      react: (_div, data) => drawn.push(["react", data]),
      purge: () => drawn.push(["purge"]),
    };
    const keys = {
      listeners: {},
      addEventListener(name, fn) {
        this.listeners[name] = fn;
      },
      removeEventListener(name) {
        delete this.listeners[name];
      },
    };
    return { div, handlers, drawn, Plotly, keys, schedule: (fn) => fn() };
  };

  test("a lasso selects in the store, and every view bound to the store redraws once", async () => {
    const store = new Selection();
    const a = standIn();
    const b = standIn();
    const build = (s) => scatterFigure(cloud(), s);
    const first = bindView(a.div, store, "rows", build, a);
    const second = bindView(b.div, store, "rows", build, b);
    await Promise.all([first.ready, second.ready]);
    a.handlers.plotly_selected({ points: [{ customdata: "52" }, { customdata: "53" }] });
    await tick();
    assert.deepEqual([...store.selected("rows")], ["52", "53"]);
    assert.deepEqual(
      b.drawn.map(([kind]) => kind),
      ["newPlot", "react"],
    );
    const redrawn = b.drawn[1][1][0];
    assert.equal(redrawn.marker.opacity[0], FADED, "the other view fades what is not selected");
  });

  test("Shift adds and Alt removes, from the keys held while the lasso is drawn", async () => {
    const store = new Selection();
    const v = standIn();
    await bindView(v.div, store, "rows", (s) => scatterFigure(cloud(), s), v).ready;
    v.handlers.plotly_selected({ points: [{ customdata: "1" }] });
    v.keys.listeners.keydown({ shiftKey: true });
    v.handlers.plotly_selected({ points: [{ customdata: "2" }] });
    v.keys.listeners.keyup({ shiftKey: false, altKey: true });
    v.handlers.plotly_selected({ points: [{ customdata: "1" }] });
    assert.deepEqual([...store.selected("rows")], ["2"]);
  });

  test("a click picks one point, Ctrl-click toggles it, a double-click clears, and hover is echoed", async () => {
    const store = new Selection();
    const v = standIn();
    await bindView(v.div, store, "cols", (s) => barFigure([{ id: "Tin", value: 1 }], s), v).ready;
    v.handlers.plotly_click({ points: [{ customdata: "Tin" }], event: {} });
    assert.ok(store.has("cols", "Tin"));
    v.handlers.plotly_click({ points: [{ customdata: "Tin" }], event: { ctrlKey: true } });
    assert.equal(store.selected("cols").size, 0);
    v.handlers.plotly_click({ points: [{ customdata: "Tin" }], event: {} });
    v.handlers.plotly_doubleclick();
    assert.equal(store.selected("cols").size, 0);
    v.handlers.plotly_hover({ points: [{ customdata: "Tin" }] });
    assert.equal(store.hovered("cols"), "Tin");
    v.handlers.plotly_unhover();
    assert.equal(store.hovered("cols"), null);
  });

  test("a view on columns does not redraw for a change of rows, and a destroyed view never again", async () => {
    const store = new Selection();
    const v = standIn();
    const view = bindView(v.div, store, "cols", (s) => barFigure([{ id: "Tin", value: 1 }], s), v);
    await view.ready;
    store.select("rows", ["1"]);
    await tick();
    assert.equal(v.drawn.length, 1);
    view.destroy();
    store.select("cols", ["Tin"]);
    await tick();
    assert.deepEqual(
      v.drawn.map(([kind]) => kind),
      ["newPlot", "purge"],
    );
    assert.deepEqual(Object.keys(v.keys.listeners), []);
  });

  test("the helpers read Plotly's events", () => {
    assert.deepEqual(idsOf({ points: [{ customdata: 1 }, { customdata: null }] }), ["1"]);
    assert.deepEqual(idsOf(undefined), []);
    assert.equal(modeOf({ metaKey: true }), "toggle");
    assert.equal(modeOf(null), "replace");
  });
});
