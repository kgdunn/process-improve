// Charts: Plotly.js, loaded once from jsDelivr at the version the Python plotly package
// names, drawing the library's figures and the page's coefficient chart in the page's
// colours. Every chart sits beside a table that holds its numbers, so a CDN that cannot
// be reached costs the picture, not the result. Nothing runs against the document until a
// chart is built or drawn, so tests/web/js imports this module under Node.

const CDN = (version) => `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/plotly.min.js`;
/**
 * Plotly's settings for a chart. A card has no toolbar, which sat over its legend and its colour bar's title: a
 * figure's own window (figure.js) has one, with room for it. Two of Plotly's buttons are never shown. The one to
 * save an image saves the shape on the screen: the window saves the figure in a shape the reader chooses. The
 * one to share a chart sends it to Plotly's servers, and a study's data stays in the reader's browser.
 */
const plotConfig = (toolbar) => ({
  displaylogo: false,
  displayModeBar: toolbar,
  showSendToCloud: false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage", "sendChartToCloud"],
});

let loading = null;

/** Load Plotly.js once; later calls get the copy already loading, whatever version they name. */
export function loadPlotly(version) {
  loading ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.crossOrigin = "anonymous"; // a response the service worker can read, and so keep (sw.js)
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

/** The size of an axis title's letters, which Plotly draws a fifth larger than the figure's other text. */
const TITLE_SIZE = 14;

/**
 * ``text`` on as many lines as it takes to fit in ``room`` pixels, broken at its spaces. Plotly draws an axis
 * title on one line and cuts what does not fit, and a title holds the reader's own names and units, of any
 * length. A letter is taken as 0.55 of its size wide, which is a little more than this text averages.
 */
export function wrapped(text, room, size = TITLE_SIZE) {
  const most = Math.max(8, Math.floor(room / (0.55 * size)));
  const lines = [];
  for (const word of String(text).split(" ")) {
    const last = lines.length - 1;
    if (last >= 0 && lines[last].length + 1 + word.length <= most) lines[last] += ` ${word}`;
    else lines.push(word);
  }
  return lines.join("<br>");
}

/**
 * Where a figure in three dimensions is seen from: far enough back that its corners and their labels are in
 * view. Plotly fits the scene to the room's height, and cuts its sides off in a narrow room: a cube seen from
 * here is 0.9 times as wide as it is tall, so in a room narrower than that the camera stands back by as much.
 */
const EYE = { x: 1.65, y: 1.65, z: 1.1 };
const SCENE_SHAPE = 0.9;

/**
 * Restyle a figure's layout for the room it is drawn in: the page's colours, its size, no title, the legend
 * below. ``width`` is the room's, in pixels, where it is known, and ``bar`` the part of it a colour bar takes:
 * each axis title is then wrapped to fit its axis, and a scene in three dimensions is seen whole. ``free`` is
 * for a figure in a window of its own, whose shape is the reader's to give: its axes fill the room.
 */
export function pageLayout(layout, height, colours = palette(), { width = 0, bar = 0, free = false } = {}) {
  const margin = { l: 56, r: 16, t: 12, b: 52, ...(layout.margin || {}) };
  const axis = (given, room = 0) => {
    const title = given?.title?.text;
    return {
      ...(given || {}),
      // A title with markup of its own is left as it is written.
      ...(title && room > 0 && !title.includes("<") ? { title: { ...given.title, text: wrapped(title, room) } } : {}),
      gridcolor: colours.line,
      zerolinecolor: colours.muted,
      linecolor: colours.line,
      color: colours.muted,
      automargin: true,
    };
  };
  const restyled = {
    ...layout,
    height,
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { ...(layout.font || {}), color: colours.fg, size: 12 },
    margin,
    // No box around the legend: the library's own figures draw one, in black on a white page.
    legend: { orientation: "h", x: 0, y: -0.22, bgcolor: "rgba(0,0,0,0)", borderwidth: 0, ...(layout.legend || {}) },
    xaxis: axis(layout.xaxis, width && width - margin.l - margin.r - bar),
    yaxis: axis(layout.yaxis, height - margin.t - margin.b),
  };
  // A figure drawn with equal axes (a response surface in coded units) keeps its data
  // range and shrinks its plot area instead, rather than stretching an axis to fill a card.
  if (free) {
    for (const key of ["scaleanchor", "scaleratio"]) for (const axis of [restyled.xaxis, restyled.yaxis]) delete axis[key];
  } else if (restyled.yaxis.scaleanchor || restyled.xaxis.scaleanchor) {
    restyled.xaxis.constrain = "domain";
    restyled.yaxis.constrain = "domain";
  }
  if (layout.scene) {
    // A chart in three dimensions has its axes in a scene of its own, and needs no room for the flat ones.
    const { xaxis, yaxis, zaxis } = layout.scene;
    const back = width > bar ? Math.max(1, (SCENE_SHAPE * height) / (width - bar)) : 1;
    const camera = { eye: { x: EYE.x * back, y: EYE.y * back, z: EYE.z * back } };
    restyled.scene = { camera, ...layout.scene, xaxis: axis(xaxis), yaxis: axis(yaxis), zaxis: axis(zaxis) };
    restyled.margin = { l: 0, r: 0, t: 0, b: 0 };
  }
  delete restyled.width;
  delete restyled.title;
  return restyled;
}

/** The kinds of trace that show their values on a colour bar beside the plot. */
const SCALED = new Set(["contour", "heatmap", "surface"]);
const hasBar = (trace) => SCALED.has(trace.type) && trace.showscale !== false;

/** The width a colour bar takes from the plot, as pageData() draws it: the bar, its numbers and its title. */
const BAR = 64;

/**
 * A figure's traces as the page draws them: a colour bar is narrow, with its title along its side. Plotly's own
 * is wide, with its title on top, and takes a third of a card's width from the plot.
 */
export function pageData(data) {
  const slim = (bar = {}) => ({ thickness: 12, outlinewidth: 0, ...bar, title: { side: "right", ...(bar.title || {}) } });
  return data.map((trace) => (hasBar(trace) ? { ...trace, colorbar: slim(trace.colorbar) } : trace));
}

/**
 * A chart as the page draws it in a room ``width`` pixels by ``height``, in ``colours``: its traces and its
 * layout. A chart with Plotly's toolbar is the one in a figure's own window, where the reader gives it its shape.
 */
function dressed(shown, height, colours, width) {
  const figure = shown.make();
  const data = pageData(figure.data);
  const room = { width, bar: data.some(hasBar) ? BAR : 0, free: shown.toolbar };
  return { data, layout: pageLayout(figure.layout || {}, height, colours, room) };
}

/**
 * What each chart on the page was drawn from, so it can be drawn again: in the new colours when the theme
 * changes, and to fit when its room does.
 */
const drawn = new Map();
let draws = 0;

/** The room a chart has: its width, and its height too where the chart fills it (``height`` null). */
const roomOf = (div, shown) => `${div.clientWidth}x${shown.height ?? div.clientHeight}`;

/**
 * Watches every chart's room. Plotly draws a chart at the width its box has at that moment and leaves it there,
 * so the first chart of a row, drawn before its neighbour was added, stayed a whole row wide; and one drawn while
 * its part of the page was hidden had no width to take. A chart whose room changes is drawn again to fit it.
 */
let rooms = null;
const watch = (div) => {
  rooms ??= new ResizeObserver((entries) => {
    for (const { target } of entries) {
      const shown = drawn.get(target);
      if (!shown) rooms.unobserve(target);
      else if (target.clientWidth && roomOf(target, shown) !== shown.room) refit(target, shown);
    }
  });
  rooms.observe(div);
};

/** Draw a chart again in the room it has now; a change of room while it is drawing is drawn once more, after. */
async function refit(div, shown) {
  shown.again = true;
  if (shown.fitting) return;
  shown.fitting = true;
  try {
    while (shown.again && drawn.get(div) === shown) {
      shown.again = false;
      await react(div, shown);
    }
  } finally {
    shown.fitting = false;
  }
}

/**
 * Draw a figure into a div. ``figure`` is the figure, or a function that makes it, for one whose
 * colours are the page's own (palette()): it is made again whenever the chart is redrawn. A ``height`` of null
 * fills the div's own height. With ``toolbar``, Plotly's buttons to zoom and turn the figure are shown.
 * ``python`` gives the code that draws the figure with the library, for the figure's own window to show.
 */
export async function draw(div, figure, version, height = 340, { toolbar = false, python = null } = {}) {
  const make = typeof figure === "function" ? figure : () => figure;
  for (const old of drawn.keys()) if (!old.isConnected) drawn.delete(old);
  // The reader's zoom, or the way a figure is turned, is kept while this figure is drawn again, and no longer.
  const shown = { make, version, height, toolbar, python, revision: ++draws };
  drawn.set(div, shown);
  offerWindow(div);
  await react(div, shown);
  watch(div);
}

/** What a chart was drawn from, as draw() was given it; undefined for a div with no chart. */
export const figureOf = (div) => drawn.get(div);

/** Take a chart down, and forget what it was drawn from. */
export function erase(div) {
  drawn.delete(div);
  window.Plotly?.purge(div);
}

/** A chart's heading and what is written under it, as its card has them, for its own window. */
function wordsOf(div) {
  const card = div.closest("figure.chart");
  const said = (node) => node?.textContent.replace(/\s+/g, " ").trim();
  const title =
    said(card?.querySelector("h3, h4")) ||
    div.getAttribute("aria-label") ||
    said(div.closest("details")?.querySelector("summary")) ||
    "Figure";
  const captions = [...(card?.querySelectorAll("figcaption") ?? [])].filter((caption) => !caption.hidden);
  return { title, caption: captions.map(said).join(" ") };
}

/**
 * Put the button that opens a chart in a window of its own (figure.js) at the right of its heading, once. Every
 * chart is drawn by draw(), so every chart has one. A chart in a card has the card's heading; one with no card
 * has the button alone, above it. The window's module is loaded when a figure is first opened.
 */
function offerWindow(div) {
  if (div.closest("dialog.figure")) return; // the window's own chart
  const card = div.closest("figure.chart");
  const above = div.previousElementSibling;
  let head = card ? card.querySelector(".chart__head") : above?.matches(".chart__head") ? above : null;
  if (head?.querySelector(".expand")) return;
  if (!head) {
    head = document.createElement("div");
    head.className = "chart__head";
    const heading = card?.querySelector("h3, h4");
    if (heading) heading.replaceWith(head);
    else div.before(head);
    if (heading) head.append(heading);
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = "link expand";
  button.textContent = "Expand";
  button.setAttribute("aria-label", "Expand this figure: open it in a window of its own, to look at it closely");
  button.onclick = async () => (await import("./figure.js")).openFigure(div, wordsOf(div));
  head.append(button);
}

/** The light theme's colours, as style.css gives them: a report is read on white, whatever theme the page shows. */
export const LIGHT = {
  fg: "#1d1d1f",
  muted: "#5f6368",
  line: "#dcd8ce",
  accent: "#2c5f7c",
  warn: "#a35f00",
  ok: "#1e7b3a",
  error: "#b3261e",
};

/**
 * A chart the page drew, as an image on white in the light theme's axes and text, for a report or a file: a data
 * URL. It is ``width`` CSS pixels wide and ``height`` tall (its height on the page, when none is given), and drawn
 * ``scale`` times that for print. With a ``title``, the image carries it, as one that leaves the page must. Null
 * for a div no chart was drawn in.
 */
export async function chartImage(div, width = 720, { height = null, title = "", format = "png", scale = 2 } = {}) {
  const shown = drawn.get(div);
  if (!shown || !window.Plotly) return null;
  const tall = height ?? shown.height ?? div.clientHeight;
  const { data, layout: fitted } = dressed(shown, tall, LIGHT, width);
  const layout = { ...fitted, width, autosize: false, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff" };
  if (title) {
    const head = 34; // room for the title, above a legend that sits over the plot
    layout.margin = { ...layout.margin, t: layout.margin.t + head };
    layout.title = { text: title, x: 0.01, xanchor: "left", y: 1, yref: "container", yanchor: "top", pad: { t: 10 } };
    layout.title.font = { size: 16, color: LIGHT.fg };
  }
  return window.Plotly.toImage({ data, layout }, { format, width, height: tall, scale });
}

/** Draw every chart on the page again, in the colours of the theme it shows now. */
export async function redrawAll() {
  for (const [div, shown] of drawn) {
    if (div.isConnected) await react(div, shown);
    else drawn.delete(div);
  }
}

async function react(div, shown) {
  const Plotly = await loadPlotly(shown.version);
  shown.room = roomOf(div, shown);
  const { data, layout } = dressed(shown, shown.height ?? div.clientHeight, palette(), div.clientWidth);
  layout.uirevision = shown.revision;
  // Under a finger, a drag on a card scrolls the page, as a drag anywhere else on it does, and does not zoom the
  // chart: the figure's own window is where it is zoomed. A chart in three dimensions is still turned by a drag.
  if (!shown.toolbar && window.matchMedia("(pointer: coarse)").matches) layout.dragmode = false;
  if (shown.toolbar) {
    // The toolbar has a row of its own above the plot, in the page's colours.
    const { muted, accent } = palette();
    layout.margin = { ...layout.margin, t: Math.max(layout.margin.t, 36) };
    layout.modebar = { bgcolor: "rgba(0,0,0,0)", color: muted, activecolor: accent };
  }
  await Plotly.react(div, data, layout, plotConfig(shown.toolbar));
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
 * true surface. ``best``, the best blend's proportions by component, puts a star there.
 */
export function ternaryFigure(
  ternary,
  { response, units = "", values = "fitted", range = ternaryRange(ternary), best = null },
  colours = palette(),
) {
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
      ...(best
        ? [
            {
              type: "scatterternary",
              mode: "markers",
              name: BEST_BLEND,
              a: [best[a]],
              b: [best[b]],
              c: [best[c]],
              marker: BEST_MARKER,
              cliponaxis: false,
              hovertemplate: `${BEST_BLEND}<br>${at}<extra></extra>`,
              showlegend: false,
            },
          ]
        : []),
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

/** What the star on a mixture's pictures marks. */
export const BEST_BLEND = "Best blend";

/**
 * A mixture of four components or more, along each component's Cox direction (``cox``, as api_plots gives it):
 * the fitted response as the component's share moves from the reference blend, the others keeping their
 * proportions to each other, a line per component; with the true process, its response dotted in the same colour.
 * Drawn through the best blend (``cox.through`` is "best"), the lines cross there, at its prediction ``best``,
 * where a star marks it.
 */
export function coxFigure(cox, { response, units = "", best = null }, colours = palette()) {
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
  if (best !== null) {
    data.push({
      type: "scatter",
      mode: "markers",
      name: BEST_BLEND,
      x: [0],
      y: [best],
      marker: BEST_MARKER,
      cliponaxis: false,
      showlegend: false,
      hovertemplate: `${BEST_BLEND}: %{y:.4g}<extra></extra>`,
    });
  }
  const from = cox.through === "best" ? "the best blend" : "the reference blend";
  return {
    data,
    layout: {
      xaxis: { title: { text: `Change in the component's proportion from ${from}` }, zeroline: true },
      yaxis: { title: { text: withUnits(`Fitted ${response}`, units) } },
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
  };
}

/** What the chart of one factor's levels marks beside each level's runs. */
export const LEVEL_AVERAGE = "Average, with its 95% interval";

/**
 * How far a run's dot sits from its level's line, a fraction of the space between two levels: spread by the run's
 * number, so dots at the same result stand apart, and the same run sits in the same place each time it is drawn.
 */
export const spreadOf = (run) => ((((run * 0.6180339887) % 1) + 1) % 1 - 0.5) * 0.3;

/**
 * A study of one factor (``comparison``, as the analysis gives it): every run's result as a dot at its level, and
 * beside the dots the level's average with its 95% interval, from that level's runs alone. The spread of the dots
 * within a level is the noise a difference between the levels is judged against.
 */
export function levelsFigure(comparison, { response, units = "", factorUnits = "" }, colours = palette()) {
  const levels = comparison.levels;
  const at = levels.map((_, i) => i);
  const named = levels.map((level) => (typeof level.level === "number" ? `${level.level}` : level.level));
  const runs = {
    type: "scatter",
    mode: "markers",
    name: "Runs",
    x: levels.flatMap((level, i) => level.runs.map((r) => i + spreadOf(r.run))),
    y: levels.flatMap((level) => level.runs.map((r) => r.value)),
    customdata: levels.flatMap((level, i) => level.runs.map((r) => [r.run, named[i]])),
    marker: { color: withAlpha(colours.accent, 0.7), size: 8, line: { width: 1, color: colours.accent } },
    hovertemplate: "Run %{customdata[0]}, at %{customdata[1]}: %{y:.4g}<extra></extra>",
  };
  const spread = levels.map((level) => level.ci_low !== null && level.ci_low !== undefined);
  const averages = {
    type: "scatter",
    mode: "markers",
    name: LEVEL_AVERAGE,
    x: at.map((i) => i + 0.3),
    y: levels.map((level) => level.mean),
    customdata: named,
    marker: { color: colours.fg, size: 11, symbol: "line-ew", line: { width: 3, color: colours.fg } },
    error_y: {
      type: "data",
      symmetric: false,
      array: levels.map((level, i) => (spread[i] ? level.ci_high - level.mean : 0)),
      arrayminus: levels.map((level, i) => (spread[i] ? level.mean - level.ci_low : 0)),
      color: colours.fg,
      thickness: 2,
      width: 6,
    },
    hovertemplate: "Average at %{customdata}: %{y:.4g}<extra></extra>",
  };
  return {
    data: [runs, averages],
    layout: {
      xaxis: {
        title: { text: withUnits(comparison.factor, factorUnits) },
        tickvals: at,
        ticktext: named,
        range: [-0.6, levels.length - 0.4],
        zeroline: false,
      },
      yaxis: { title: { text: withUnits(response, units) } },
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
  };
}

/** Write a model term as the page shows it: ``I(T ** 2)`` as ``T²`` and ``T:P`` as ``T·P``. */
// Only a colon between two names is a product, so a sentence's own colons are left alone.
export const prettyTerm = (term) => term.replace(/I\((\w+) \*\* 2\)/g, "$1²").replace(/(\w):(?=\w)/g, "$1·");

/** Plotly's default sequence of trace colours, so a curve's colour can be fixed whatever else is drawn. */
const PLOTLY_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

/**
 * FDS curves of several designs on one chart: each design's prediction variance, in units
 * of the noise variance, sorted over points spread evenly through the cube. The right end
 * of a curve is its worst case, G.
 */
export function fdsFigure(fds, { region = false, hidden = new Set() } = {}) {
  const over = region ? "the settings you can run" : "the cube"; // with limits, the curves are over what they leave
  // A design the reader hides is not drawn. Plotly would then hand its colour to the next curve, so each curve's
  // colour is fixed by its place in the list, in Plotly's own sequence.
  const data = Object.entries(fds).map(([label, curve], i) => ({
    type: "scatter",
    mode: "lines",
    name: label,
    visible: !hidden.has(label),
    x: curve.fraction,
    y: curve.variance,
    line: { width: 2, color: PLOTLY_COLOURS[i % PLOTLY_COLOURS.length] },
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

/** The star on a mixture's overlay, at the balanced blend. */
export const BALANCED_BLEND = "Balanced blend";

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

/**
 * The overlay of the responses' acceptable ranges over the triangle of a mixture's three components: the blends where
 * every response is acceptable shaded, each response's acceptable boundary a line in its colour and dash, and a star
 * at the balanced blend. ``overlay`` is bootstrap.balance's for a mixture: its grid of blends, each response's
 * prediction there and its boundary as segments of line; ``responses`` are the reply's, for each response's words.
 */
export function blendOverlayFigure(overlay, responses, colours = palette()) {
  const [a, b, c] = overlay.components;
  const byName = Object.fromEntries(responses.map((r) => [r.name, r]));
  const lineColours = responseColours(colours);
  const kept = (values) => values.filter((_, k) => overlay.acceptable[k]);
  const at = `${a} %{a:.3f}<br>${b} %{b:.3f}<br>${c} %{c:.3f}`;
  const data = [
    {
      type: "scatterternary",
      mode: "markers",
      name: SHADED,
      a: kept(overlay.a),
      b: kept(overlay.b),
      c: kept(overlay.c),
      marker: { symbol: "circle", size: 7, color: withAlpha(colours.accent, 0.35), line: { width: 0 } },
      hoverinfo: "skip",
    },
  ];
  overlay.responses.forEach((surface, i) => {
    const r = byName[surface.name];
    const line = { color: lineColours[i % lineColours.length], width: 2.5, dash: RESPONSE_DASHES[i % RESPONSE_DASHES.length] };
    // A boundary that misses the blends drawn has no line, so no key: the caption says why.
    surface.lines.forEach((segments, j) => {
      data.push({
        type: "scatterternary",
        mode: "lines",
        name: boundaryName(r),
        legendgroup: surface.name,
        showlegend: j === 0 && boundaryDrawn(surface),
        a: segments.a,
        b: segments.b,
        c: segments.c,
        line,
        hoverinfo: "skip",
      });
    });
  });
  data.push({
    type: "scatterternary",
    mode: "markers",
    name: BALANCED_BLEND,
    a: [overlay.at[0]],
    b: [overlay.at[1]],
    c: [overlay.at[2]],
    marker: BEST_MARKER,
    cliponaxis: false,
    hovertemplate: `${BALANCED_BLEND}<br>${at}<extra></extra>`,
  });
  const axis = (title) => ({
    title: { text: title, font: { size: 13 } },
    min: 0,
    linecolor: colours.muted,
    gridcolor: colours.line,
    tickfont: { color: colours.muted, size: 10 },
  });
  return {
    data,
    layout: {
      ternary: { sum: 1, bgcolor: "rgba(0,0,0,0)", aaxis: axis(a), baxis: axis(b), caxis: axis(c) },
      legend: LEGEND_ABOVE,
      showlegend: true,
      margin: { l: 36, r: 36, t: 48, b: 36 },
    },
    height: 440 + 22 * Math.ceil((overlay.responses.length + 2) / 2),
  };
}

// ---------------------------------------------------------------- a curve each run gives
/** What the curve chart's legend calls each line, and the band it shades. */
export const CURVE_LINES = {
  curve: "Predicted curve",
  pointwise: "95% band at each point",
  simultaneous: "95% band over the whole curve",
  target: "Target",
};

/**
 * A curve predicted at the what-if panel's settings (best.js curveAt), over the ``points`` along it: the curve, the
 * 95% band at each point shaded, and the band over the whole curve dashed. ``words`` holds the curve's label and
 * units and the points' (``along``, ``alongUnits``). A ``target`` the reader gave, ``{points, values}``, is dotted,
 * with a gap where it has no value.
 */
export function curveFigure(points, predicted, words, target = null, colours = palette()) {
  const { label = "Curve", units = "", along = "Along the curve", alongUnits = "" } = words;
  const shade = withAlpha(colours.accent, 0.22);
  const line = (name, x, y, style, more = {}) => ({ type: "scatter", mode: "lines", name, x, y, line: style, hoverinfo: "skip", ...more });
  const edge = { color: colours.accent, width: 1, dash: "dash" };
  const data = [
    line(CURVE_LINES.pointwise, points, predicted.pointwise[0], { width: 0, color: shade }, { showlegend: false }),
    line(CURVE_LINES.pointwise, points, predicted.pointwise[1], { width: 0, color: shade }, { fill: "tonexty", fillcolor: shade }),
    line(CURVE_LINES.simultaneous, points, predicted.simultaneous[0], edge, { legendgroup: "whole" }),
    line(CURVE_LINES.simultaneous, points, predicted.simultaneous[1], edge, { legendgroup: "whole", showlegend: false }),
    {
      ...line(CURVE_LINES.curve, points, predicted.curve, { color: colours.accent, width: 2.5 }),
      hoverinfo: undefined,
      hovertemplate: "%{x:.4g}: %{y:.4g}<extra></extra>",
    },
  ];
  if (target) data.push(line(CURVE_LINES.target, target.points, target.values, { color: colours.fg, width: 2, dash: "dot" }, { connectgaps: false }));
  return {
    data,
    layout: {
      xaxis: { title: { text: withUnits(along, alongUnits) } },
      yaxis: { title: { text: withUnits(label, units) } },
      legend: LEGEND_ABOVE,
      showlegend: true,
    },
  };
}
