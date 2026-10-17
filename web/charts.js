// Charts: Plotly.js, loaded once from jsDelivr at the version the Python plotly package
// names, drawing the library's figures and the page's coefficient chart in the page's
// colours. Every chart sits beside a table that holds its numbers, so a CDN that cannot
// be reached costs the picture, not the result. Nothing runs against the document until a
// chart is built or drawn, so tests/web/js imports this module under Node.

const CDN = (version) => `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/plotly.min.js`;
/** Plotly's settings for a chart on this page; on a phone the toolbar would sit over the chart for good. */
const plotConfig = () => ({
  responsive: true,
  displaylogo: false,
  displayModeBar: window.matchMedia("(max-width: 600px)").matches ? false : "hover",
  modeBarButtonsToRemove: ["select2d", "lasso2d"],
});

let loading = null;

/** Load Plotly.js once; later calls get the copy already loading, whatever version they name. */
export function loadPlotly(version) {
  loading ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = CDN(version);
    script.onload = () => resolve(window.Plotly);
    script.onerror = () => {
      loading = null;
      reject(new Error(`Plotly.js ${version} could not be loaded from ${CDN(version)}`));
    };
    document.head.append(script);
  });
  return loading;
}

/** The page's colours, read from its CSS, so the charts follow the light and dark themes. */
export function palette() {
  const css = getComputedStyle(document.documentElement);
  const read = (name) => css.getPropertyValue(name).trim();
  return {
    fg: read("--fg"),
    muted: read("--muted"),
    line: read("--line"),
    accent: read("--accent"),
    warn: read("--warn"),
    ok: read("--ok"),
    error: read("--error"),
  };
}

/**
 * The true values' marker: a diamond with a white face and a dark edge, as the book draws a mark
 * that sits on bars. It reads on the orange and the blue bars, and on a light or a dark page,
 * where any one colour would be too like one of them.
 */
export const TRUE_MARKER = { symbol: "diamond", size: 12, color: "#ffffff", line: { width: 1.5, color: "#1a1a1a" } };

/**
 * A legend entry with nothing drawn on the plot: a line in the legend that says what a line on the
 * plot is, such as a 95% interval, so the chart needs no caption to explain it.
 */
export const legendKey = (name, color, dash = "solid") => ({
  type: "scatter",
  mode: "lines",
  name,
  x: [null],
  y: [null],
  line: { color, width: 2, dash },
  hoverinfo: "skip",
});

/** Restyle a figure's layout for a card on the page: its colours, its size, no title, the legend below. */
export function pageLayout(layout, height, colours = palette()) {
  const axis = (given) => ({
    ...(given || {}),
    gridcolor: colours.line,
    zerolinecolor: colours.muted,
    linecolor: colours.line,
    color: colours.muted,
    automargin: true,
  });
  const restyled = {
    ...layout,
    height,
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { ...(layout.font || {}), color: colours.fg, size: 12 },
    margin: { l: 56, r: 16, t: 12, b: 52, ...(layout.margin || {}) },
    legend: { orientation: "h", x: 0, y: -0.22, bgcolor: "rgba(0,0,0,0)", ...(layout.legend || {}) },
    xaxis: axis(layout.xaxis),
    yaxis: axis(layout.yaxis),
  };
  // A figure drawn with equal axes (a response surface in coded units) keeps its data
  // range and shrinks its plot area instead, rather than stretching an axis to fill a card.
  if (restyled.yaxis.scaleanchor || restyled.xaxis.scaleanchor) {
    restyled.xaxis.constrain = "domain";
    restyled.yaxis.constrain = "domain";
  }
  delete restyled.width;
  delete restyled.title;
  return restyled;
}

/** Draw a figure ({data, layout}) into a card's plot area. */
/** What each chart on the page was drawn from, so a change of theme can draw it again in the new colours. */
const drawn = new Map();

/**
 * Draw a figure into a div. ``figure`` is the figure, or a function that makes it, for one whose
 * colours are the page's own (palette()): it is made again whenever the chart is redrawn.
 */
export async function draw(div, figure, version, height = 340) {
  const make = typeof figure === "function" ? figure : () => figure;
  for (const old of drawn.keys()) if (!old.isConnected) drawn.delete(old);
  drawn.set(div, { make, version, height });
  await react(div, make(), version, height);
}

/** Draw every chart on the page again, in the colours of the theme it shows now. */
export async function redrawAll() {
  for (const [div, { make, version, height }] of drawn) {
    if (div.isConnected) await react(div, make(), version, height);
    else drawn.delete(div);
  }
}

/**
 * Fit the charts inside ``root`` to the room they have now: a chart drawn while its phase was hidden
 * took Plotly's default width, and a phase shown again gives it its own.
 */
export function resizeWithin(root) {
  const Plotly = window.Plotly;
  if (!Plotly) return;
  for (const div of root.querySelectorAll(".js-plotly-plot")) Plotly.Plots.resize(div);
}

async function react(div, figure, version, height) {
  const Plotly = await loadPlotly(version);
  await Plotly.react(div, figure.data, pageLayout(figure.layout || {}, height), plotConfig());
}

/**
 * Give two contour figures one colour scale and one set of contour levels, so the fitted
 * surface and the true one can be read against each other.
 */
export function shareContourScale(...figures) {
  const values = figures.flatMap((f) => f.data[0].z.flat().filter(Number.isFinite));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const size = niceStep((high - low) / 12);
  for (const figure of figures) {
    Object.assign(figure.data[0], {
      zauto: false,
      zmin: low,
      zmax: high,
      autocontour: false,
      contours: { ...(figure.data[0].contours || {}), start: Math.floor(low / size) * size, end: high, size },
    });
  }
}

/** A round step near the one asked for: 1, 2 or 5 times a power of ten. */
export function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * power).find((step) => step >= raw);
}

/**
 * The estimated coefficients with their 95% intervals, one row per term, and the true
 * value beside each when the results were simulated. The intercept is left out: it sits
 * on another scale.
 */
/**
 * The height of a chart with a row per term: room for every term's label, plus the legend above
 * the plot and the axis below it, which a fixed height took from the rows on a narrow screen.
 */
export const termRowsHeight = (n) => Math.max(240, 28 * n + 130);

/** A legend above the plot, where it never meets the axis title below it, however narrow the card. */
const LEGEND_ABOVE = { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom" };

/** A category axis down the side with every term's label, the first term at the top. */
const termAxis = (labels) => ({
  type: "category",
  categoryorder: "array",
  categoryarray: labels,
  autorange: "reversed",
  tickmode: "linear",
  dtick: 1,
});

export function coefficientFigure(coefficients, simulated, colours = palette()) {
  const terms = coefficients.filter((c) => c.term !== "Intercept");
  const labels = terms.map((c) => c.label);
  const above = terms.map((c) => (c.ci_high === null || c.ci_high === undefined ? 0 : c.ci_high - c.coefficient));
  const below = terms.map((c) => (c.ci_low === null || c.ci_low === undefined ? 0 : c.coefficient - c.ci_low));
  const data = [
    {
      type: "scatter",
      mode: "markers",
      name: "Estimate",
      legendrank: 1,
      x: terms.map((c) => c.coefficient),
      y: labels,
      marker: { color: colours.accent, size: 9 },
      error_x: {
        type: "data",
        symmetric: false,
        array: above,
        arrayminus: below,
        color: colours.accent,
        thickness: 2,
        width: 0,
      },
      hovertemplate: "%{y}: %{x:.4g}<extra></extra>",
    },
  ];
  if (terms.some((c) => c.ci_low !== null && c.ci_low !== undefined)) {
    data.push({ ...legendKey("95% interval", colours.accent), legendrank: 2 });
  }
  if (simulated) {
    // Drawn first, under the estimates, so a dot on its true value shows inside the diamond.
    data.unshift({
      type: "scatter",
      mode: "markers",
      name: "True value",
      legendrank: 3,
      x: terms.map((c) => c.true_value ?? 0),
      y: labels,
      marker: TRUE_MARKER,
      hovertemplate: "%{y}, true: %{x:.4g}<extra></extra>",
    });
  }
  return {
    data,
    layout: {
      xaxis: { title: { text: "Coefficient, coded units" }, zeroline: true },
      yaxis: termAxis(labels),
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
    height: termRowsHeight(terms.length),
  };
}

/** The book's colours for a coefficient's sign: the colourblind-safe Okabe-Ito orange and blue. */
export const SIGN_COLOURS = { positive: "#E69F00", negative: "#0072B2" };

/**
 * The coefficients as the book's Pareto plot draws them: each one's size as a bar, the largest at
 * the top, orange for a positive coefficient and blue for a negative one. With residual degrees of
 * freedom each bar carries its 95% interval; with none, Lenth's margins of error (``lenth``, which
 * the analysis gives on the effects' scale, twice the coefficients') are drawn on the coefficients'
 * scale, dashed for ME and dotted for SME. The true sizes are marked when the results were simulated.
 */
export function paretoFigure(coefficients, simulated, lenth = null, colours = palette()) {
  const terms = coefficients
    .filter((c) => c.term !== "Intercept")
    .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient));
  const labels = terms.map((c) => c.label);
  const intervals = terms.some((c) => c.ci_high !== null && c.ci_high !== undefined);
  const bars = (positive) => {
    const picked = terms.filter((c) => c.coefficient > 0 === positive);
    return {
      type: "bar",
      orientation: "h",
      name: positive ? "Positive" : "Negative",
      x: picked.map((c) => Math.abs(c.coefficient)),
      y: picked.map((c) => c.label),
      customdata: picked.map((c) => c.coefficient),
      marker: { color: positive ? SIGN_COLOURS.positive : SIGN_COLOURS.negative },
      error_x: intervals
        ? { type: "data", array: picked.map((c) => c.ci_high - c.coefficient), color: colours.fg, thickness: 1.5, width: 5 }
        : { visible: false },
      hovertemplate: "%{y}: %{customdata:.4g}<extra></extra>",
    };
  };
  const data = [bars(true), bars(false)].filter((trace) => trace.x.length > 0);
  if (intervals) data.push(legendKey("95% interval", colours.fg));
  if (simulated) {
    data.push({
      type: "scatter",
      mode: "markers",
      name: "True size",
      x: terms.map((c) => Math.abs(c.true_value ?? 0)),
      y: labels,
      marker: TRUE_MARKER,
      hovertemplate: "%{y}, true size: %{x:.4g}<extra></extra>",
    });
  }
  const margins = !intervals && lenth ? [["ME", "dash"], ["SME", "dot"]].filter(([key]) => Number.isFinite(lenth[key])) : [];
  const named = { ME: "Margin of error (ME)", SME: "Simultaneous margin of error (SME)" };
  data.push(...margins.map(([key, dash]) => legendKey(named[key], colours.muted, dash)));
  return {
    data,
    layout: {
      barmode: "overlay",
      xaxis: { title: { text: "|coefficient|, coded units" }, rangemode: "tozero" },
      yaxis: termAxis(labels),
      legend: LEGEND_ABOVE,
      showlegend: true,
      shapes: margins.map(([key, dash]) => ({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: lenth[key] / 2,
        x1: lenth[key] / 2,
        y0: 0,
        y1: 1,
        line: { color: colours.muted, dash, width: 1.5 },
      })),
      annotations: margins.map(([key]) => ({
        xref: "x",
        yref: "paper",
        x: lenth[key] / 2,
        y: 1,
        xanchor: "left",
        yanchor: "top",
        xshift: 3,
        text: key,
        showarrow: false,
        font: { size: 11, color: colours.muted },
      })),
    },
    height: termRowsHeight(terms.length),
  };
}

// ColorBrewer's Blues, light at zero and dark at one, as the book draws its correlation maps.
// Plotly.js's own "Blues" runs the other way, so the scale is written out.
const BLUES = [
  [0, "#f7fbff"],
  [0.125, "#deebf7"],
  [0.25, "#c6dbef"],
  [0.375, "#9ecae1"],
  [0.5, "#6baed6"],
  [0.625, "#4292c6"],
  [0.75, "#2171b5"],
  [0.875, "#08519c"],
  [1, "#08306b"],
];

/** Write a model term as the page shows it: ``I(T ** 2)`` as ``T²`` and ``T:P`` as ``T·P``. */
// Only a colon between two names is a product, so a sentence's own colons are left alone.
export const prettyTerm = (term) => term.replace(/I\((\w+) \*\* 2\)/g, "$1²").replace(/(\w):(?=\w)/g, "$1·");

/**
 * FDS curves of several designs on one chart: each design's prediction variance, in units
 * of the noise variance, sorted over points spread evenly through the cube. The right end
 * of a curve is its worst case, G.
 */
export function fdsFigure(fds, { region = false } = {}) {
  const over = region ? "the settings you can run" : "the cube"; // with limits, the curves are over what they leave
  const data = Object.entries(fds).map(([label, curve]) => ({
    type: "scatter",
    mode: "lines",
    name: label,
    x: curve.fraction,
    y: curve.variance,
    line: { width: 2 },
    hovertemplate: `${label}: %{y:.3g} or less over %{x:.0%} of ${over}<extra></extra>`,
  }));
  return {
    data,
    layout: {
      xaxis: { title: { text: `Fraction of ${over}` }, tickformat: ".0%", range: [0, 1] },
      yaxis: { title: { text: "Prediction variance / σ²" }, rangemode: "tozero" },
      showlegend: true,
    },
  };
}

/**
 * One design's correlation map: |r| between its main effects, squares and two-factor
 * interactions, in that order, with lines between the three blocks. A term that does not
 * vary in the design has no correlation, and its row and column are blank.
 */
export function correlationFigure(terms, matrix, k, colours = palette()) {
  const labels = terms.map(prettyTerm);
  const n = labels.length;
  const line = { width: 1, color: colours.muted };
  const shapes = [k, 2 * k]
    .filter((b) => b < n)
    .flatMap((b) => [
      { type: "line", xref: "x", yref: "y", x0: b - 0.5, x1: b - 0.5, y0: -0.5, y1: n - 0.5, line },
      { type: "line", xref: "x", yref: "y", x0: -0.5, x1: n - 0.5, y0: b - 0.5, y1: b - 0.5, line },
    ]);
  return {
    data: [
      {
        type: "heatmap",
        x: labels,
        y: labels,
        z: matrix,
        zmin: 0,
        zmax: 1,
        colorscale: BLUES,
        showscale: false,
        hovertemplate: "%{y} with %{x}: |r| = %{z:.2f}<extra></extra>",
      },
    ],
    layout: {
      xaxis: { type: "category", tickangle: -45, showgrid: false },
      yaxis: { type: "category", autorange: "reversed", showgrid: false },
      shapes,
      showlegend: false,
    },
  };
}

// ---------------------------------------------------------------- the best settings for the response's goal
/**
 * The best settings' marker on a response surface: a star with a white face and a dark edge, as the true values'
 * diamond, so it reads on every colour of the surface, on a light page or a dark one.
 */
export const BEST_MARKER = { symbol: "star", size: 18, color: "#ffffff", line: { width: 1.5, color: "#1a1a1a" } };

/**
 * Mark the best settings on a response surface drawn over the factors ``pair``: a star at their coded settings,
 * which the surface's axes are in, drawn whole even at the edge of the plot. ``text`` is what hovering over it says.
 * Returns the figure, the star its last trace.
 */
export function markBest(figure, coded, pair, text) {
  figure.data.push({
    type: "scatter",
    mode: "markers",
    name: "Best settings",
    x: [coded[pair[0]]],
    y: [coded[pair[1]]],
    marker: BEST_MARKER,
    cliponaxis: false,
    showlegend: false,
    hovertemplate: `${text}<extra></extra>`,
  });
  return figure;
}

/** How the chart of how tightly to hold each factor names its two moves, and what a pale bar is. */
export const MOVES = { down: "Lowered 5% of its range", up: "Raised 5% of its range" };
export const PALE = "Outside your ranges or limits";

/** The height of the chart of how tightly to hold each factor: two bars a factor, and the legend and the axis. */
export const sensitivityHeight = (n) => Math.max(200, 54 * n + 120);

/**
 * How tightly to hold each factor at the best settings: for each factor in the model, the change in the predicted
 * response when it alone is lowered, and when it is raised, by 5% of its range, the others held at their best
 * settings. The factor that moves the prediction most is at the top. A move that leaves the ranges or the limits is
 * drawn pale: those settings cannot be run, and the model is extrapolating there. ``words`` are the response's
 * {label, units}; each item of ``sensitivity`` is as bootstrap.best_settings gives it, with the factor's ``units``.
 */
export function sensitivityFigure(sensitivity, { label, units = "" }, colours = palette()) {
  const size = (s) => Math.max(Math.abs(s.down), Math.abs(s.up));
  const rows = sensitivity.filter((s) => s.in_model).sort((a, b) => size(b) - size(a));
  const names = rows.map((s) => s.name);
  const colour = { down: colours.warn, up: colours.accent };
  const step = (s) => `${Number(s.step.toPrecision(4))}${s.units ? ` ${s.units}` : ""}`;
  const bars = (side) => ({
    type: "bar",
    orientation: "h",
    name: MOVES[side],
    x: rows.map((s) => s[side]),
    y: names,
    marker: { color: colour[side], opacity: rows.map((s) => (s[`${side}_runnable`] ? 1 : 0.3)), line: { width: 0 } },
    customdata: rows.map(
      (s) => `${side === "down" ? "lowered" : "raised"} ${step(s)}${s[`${side}_runnable`] ? "" : `, ${PALE.toLowerCase()}`}`,
    ),
    hovertemplate: `%{y} %{customdata}: %{x:+.3g}${units ? ` ${units}` : ""}<extra></extra>`,
  });
  const data = [bars("down"), bars("up")];
  if (rows.some((s) => !s.down_runnable || !s.up_runnable)) {
    // A key, not data: a scatter takes no place among the grouped bars, as a third bar trace would.
    data.push({
      type: "scatter",
      mode: "markers",
      name: PALE,
      x: [null],
      y: [null],
      marker: { symbol: "square", size: 12, color: colours.muted, opacity: 0.3 },
      hoverinfo: "skip",
    });
  }
  return {
    data,
    layout: {
      barmode: "group",
      bargap: 0.25,
      xaxis: { title: { text: `Change in ${label}${units ? ` (${units})` : ""}` }, zeroline: true },
      yaxis: termAxis(names),
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
    height: sensitivityHeight(rows.length),
  };
}

// ---------------------------------------------------------------- the settings that balance the responses
/** A colour of the page's, ``#2c5f7c``, as an rgba with this opacity, for a shading that lets the lines through. */
export function withAlpha(colour, alpha) {
  const hex = String(colour).trim().replace(/^#/, "");
  const full = hex.length === 3 ? [...hex].map((c) => c + c).join("") : hex;
  if (!/^[0-9a-f]{6}$/i.test(full)) return colour;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Each response's line on the overlay, in turn: a colour of the page's and a dash, so a line reads without its colour. */
export const RESPONSE_DASHES = ["solid", "dash", "dot", "dashdot", "longdash"];
export const responseColours = (colours) => [colours.accent, colours.warn, colours.ok, colours.error, colours.fg];

/** The star on the overlay at the balanced settings: the best settings' marker, which reads on every shading. */
export const BALANCED = "Balanced settings";

/** What the overlay's shading means, as its legend says it. */
export const SHADED = "Every response acceptable";
export const GREYED = "Outside your limits";

/**
 * What a response's line on the overlay is, as the legend names it: the boundary of the results it accepts, in its
 * units. ``r`` is the response as bootstrap.balance gives it, its acceptable range from ``low`` to ``high``.
 */
export function boundaryName(r) {
  const at = (x) => `${Number(Number(x).toPrecision(4))}${r.units === "%" ? "%" : r.units ? ` ${r.units}` : ""}`;
  if (r.goal === "maximize") return `${r.label}: ${at(r.low)} and above`;
  if (r.goal === "minimize") return `${r.label}: ${at(r.high)} and below`;
  return `${r.label}: ${at(r.low).replace(/%$/, "")} to ${at(r.high)}`;
}

/** Whether a response's acceptable boundary crosses the settings the overlay draws, so it has a line there. */
export function boundaryDrawn(surface) {
  const values = surface.z.flat();
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  return surface.levels.some((level) => level > lo && level < hi);
}

/**
 * The overlay of the responses' acceptable ranges over two factors, the others at the balanced settings: shaded where
 * every response is acceptable (a desirability above 0), each response's acceptable boundary a labelled line, greyed
 * where a limit on the settings rules the setting out, and a star at the balanced settings. ``overlay`` is
 * bootstrap.balance's, in coded units, ticked with the factors' own settings; ``responses`` are the reply's, for each
 * response's words and range; ``axes`` names each axis with its units.
 */
export function overlayFigure(overlay, responses, axes, colours = palette()) {
  const { grid } = overlay;
  const byName = Object.fromEntries(responses.map((r) => [r.name, r]));
  const lineColours = responseColours(colours);
  // A shading is a constraint on a mask of 1s and 0s. Plotly fills the side a constraint rules out, so each keeps what
  // is below 0.5 and fills the 1s. It draws only where the mask crosses 0.5, so each mask gets a border of 0s beyond
  // the axes, outside what is shown: a mask of 1s everywhere still crosses there, and is shaded whole.
  const step = grid.length > 1 ? grid[1] - grid[0] : 1;
  const padded = [grid[0] - step, ...grid, grid.at(-1) + step];
  const pad = (z) => [Array(padded.length).fill(0), ...z.map((row) => [0, ...row, 0]), Array(padded.length).fill(0)];
  const region = (z, fill, name) => ({
    type: "contour",
    x: padded,
    y: padded,
    z: pad(z),
    name,
    showscale: false,
    showlegend: false,
    hoverinfo: "skip",
    contours: { type: "constraint", operation: "<", value: 0.5 },
    fillcolor: fill,
    line: { width: 0 },
  });
  const data = [region(overlay.acceptable, withAlpha(colours.accent, 0.22), SHADED)];
  if (overlay.runnable) data.push(region(overlay.runnable.map((row) => row.map((v) => 1 - v)), withAlpha(colours.muted, 0.45), GREYED));
  overlay.responses.forEach((surface, i) => {
    const r = byName[surface.name];
    const colour = lineColours[i % lineColours.length];
    const dash = RESPONSE_DASHES[i % RESPONSE_DASHES.length];
    // One level is drawn as the first of two, the second beyond every value, as a contour of start and end alike is not.
    const values = surface.z.flat();
    const beyond = Math.max(...values) - Math.min(...values) + 1;
    const [low, high = low] = surface.levels;
    data.push({
      type: "contour",
      x: grid,
      y: grid,
      z: surface.z,
      name: boundaryName(r),
      showscale: false,
      showlegend: false,
      autocontour: false,
      contours: {
        coloring: "none",
        start: low,
        end: high > low ? high : low + beyond,
        size: high > low ? high - low : beyond,
        showlabels: true,
        labelformat: ".4~g",
        labelfont: { size: 11, color: colour },
      },
      line: { color: colour, width: 2.5, dash },
      hovertemplate: `${r.label}: %{z:.4g}${r.units ? ` ${r.units}` : ""}<extra></extra>`,
    });
  });
  data.push(
    {
      type: "scatter",
      mode: "markers",
      name: SHADED,
      x: [null],
      y: [null],
      marker: { symbol: "square", size: 12, color: withAlpha(colours.accent, 0.35) },
      hoverinfo: "skip",
    },
    ...(overlay.runnable
      ? [{ type: "scatter", mode: "markers", name: GREYED, x: [null], y: [null], marker: { symbol: "square", size: 12, color: withAlpha(colours.muted, 0.55) }, hoverinfo: "skip" }]
      : []),
    // A boundary that misses the settings drawn has no line, so no key: the caption says why.
    ...overlay.responses.flatMap((surface, i) =>
      boundaryDrawn(surface) ? [legendKey(boundaryName(byName[surface.name]), lineColours[i % lineColours.length], RESPONSE_DASHES[i % RESPONSE_DASHES.length])] : [],
    ),
    {
      type: "scatter",
      mode: "markers",
      name: BALANCED,
      x: [overlay.at[0]],
      y: [overlay.at[1]],
      marker: BEST_MARKER,
      cliponaxis: false,
      hovertemplate: `${BALANCED}<extra></extra>`,
    },
  );
  const axis = (name) => ({
    title: { text: axes[name] ?? name },
    range: [-1, 1],
    showgrid: false,
    zeroline: false,
    ...(overlay.ticks?.[name] ?? {}),
  });
  return {
    data,
    layout: {
      xaxis: axis(overlay.x),
      yaxis: { ...axis(overlay.y), scaleanchor: "x" },
      legend: LEGEND_ABOVE,
      showlegend: true,
      margin: { t: 8 },
    },
    height: 420 + 22 * Math.ceil((overlay.responses.length + 2) / 2),
  };
}
