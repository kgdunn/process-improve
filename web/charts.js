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
  return { fg: read("--fg"), muted: read("--muted"), line: read("--line"), accent: read("--accent") };
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

export function coefficientFigure(coefficients, simulated, colours = palette(), { mixture = false } = {}) {
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
      // A mixture is fitted in its proportions: a component's coefficient is its response alone, in the response's units.
      xaxis: { title: { text: mixture ? "Coefficient" : "Coefficient, coded units" }, zeroline: true },
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

// ---------------------------------------------------------------- a mixture's pictures
/** The library's colours for a response surface, light for low and dark for high, as its contour plots draw them. */
export const SURFACE = [
  [0, "#EFF6FF"],
  [0.25, "#93C5FD"],
  [0.5, "#3B82F6"],
  [0.75, "#1D4ED8"],
  [1, "#1E3A8A"],
];

/** A run's marker on a surface: a white face and a dark edge, which read on its light and its dark blues alike. */
const RUN_MARKER = { symbol: "circle", size: 10, color: "#ffffff", line: { width: 1.5, color: "#1a1a1a" } };

/** A quantity's name with its units, as an axis or a colour bar shows it. */
const withUnits = (name, units) => (units ? `${name} (${units})` : name);

/**
 * The lowest and highest response a ternary plot colours: over the fitted values, and the true ones when the
 * results were simulated, so the fitted and the true surface share one scale.
 */
export function ternaryRange(ternary) {
  const values = [...ternary.grid.fitted, ...(ternary.grid.true ?? [])].filter(Number.isFinite);
  return [Math.min(...values), Math.max(...values)];
}

/**
 * A mixture of three components over its triangle (``ternary``, as api_plots gives it): each blend of the grid
 * coloured by the response, the ``fitted`` one or the ``true`` one, and the runs marked on top. Each corner is
 * a pure component, labelled with its name; ``range`` is the colour scale's ends, shared by the fitted and the
 * true surface.
 */
export function ternaryFigure(ternary, { response, units = "", values = "fitted", range = ternaryRange(ternary) }, colours = palette()) {
  const [a, b, c] = ternary.components;
  const { grid, runs } = ternary;
  const axis = (title) => ({
    title: { text: title, font: { size: 13 } },
    min: 0,
    linecolor: colours.muted,
    gridcolor: colours.line,
    tickfont: { color: colours.muted, size: 10 },
  });
  const at = `${a} %{a:.3f}<br>${b} %{b:.3f}<br>${c} %{c:.3f}`;
  return {
    data: [
      {
        type: "scatterternary",
        mode: "markers",
        name: values === "true" ? "True response" : "Fitted response",
        a: grid.a,
        b: grid.b,
        c: grid.c,
        customdata: grid[values],
        marker: {
          symbol: "circle",
          size: 7,
          color: grid[values],
          colorscale: SURFACE,
          cmin: range[0],
          cmax: range[1],
          line: { width: 0 },
          colorbar: { title: { text: withUnits(response, units) }, thickness: 12, len: 0.8 },
        },
        hovertemplate: `${at}<br>${response}: %{customdata:.4g}<extra></extra>`,
        showlegend: false,
      },
      {
        type: "scatterternary",
        mode: "markers",
        name: "Runs",
        a: runs.a,
        b: runs.b,
        c: runs.c,
        customdata: runs.y,
        marker: RUN_MARKER,
        hovertemplate: `${at}<br>${response}, run: %{customdata:.4g}<extra></extra>`,
        showlegend: false,
      },
    ],
    layout: {
      ternary: { sum: 1, bgcolor: "rgba(0,0,0,0)", aaxis: axis(a), baxis: axis(b), caxis: axis(c) },
      margin: { l: 36, r: 36, t: 36, b: 36 },
      showlegend: false,
    },
  };
}

/** Okabe and Ito's colourblind-safe colours, one per component of a Cox trace plot. */
const COMPONENT_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"];

/**
 * A mixture of four components or more, along each component's Cox direction (``cox``, as api_plots gives it):
 * the fitted response as the component's share moves from the reference blend, the others keeping their
 * proportions to each other, a line per component; with the true process, its response dotted in the same colour.
 */
export function coxFigure(cox, { response, units = "" }, colours = palette()) {
  const data = cox.traces.flatMap((trace, i) => {
    const color = COMPONENT_COLOURS[i % COMPONENT_COLOURS.length];
    const fitted = {
      type: "scatter",
      mode: "lines",
      name: trace.component,
      legendgroup: trace.component,
      x: trace.change,
      y: trace.fitted,
      line: { color, width: 2.5 },
      hovertemplate: `${trace.component} %{x:+.3f}: %{y:.4g}<extra></extra>`,
    };
    if (!trace.true) return [fitted];
    const truth = {
      type: "scatter",
      mode: "lines",
      name: `${trace.component}, true`,
      legendgroup: trace.component,
      showlegend: false,
      x: trace.change,
      y: trace.true,
      line: { color, width: 1.5, dash: "dot" },
      hovertemplate: `${trace.component} %{x:+.3f}, true: %{y:.4g}<extra></extra>`,
    };
    return [fitted, truth];
  });
  if (cox.traces.some((trace) => trace.true)) data.push(legendKey("True process", colours.muted, "dot"));
  return {
    data,
    layout: {
      xaxis: { title: { text: "Change in the component's proportion from the reference blend" }, zeroline: true },
      yaxis: { title: { text: withUnits(`Fitted ${response}`, units) } },
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
  };
}

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
