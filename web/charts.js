// Charts: Plotly.js, loaded once from jsDelivr at the version the Python plotly package
// names, drawing the library's figures and the page's coefficient chart in the page's
// colours. Every chart sits beside a table that holds its numbers, so a CDN that cannot
// be reached costs the picture, not the result.

const CDN = (version) => `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/plotly.min.js`;
// On a phone the toolbar would sit over the chart for good, since there is no hover to hide it.
const CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: window.matchMedia("(max-width: 600px)").matches ? false : "hover",
  modeBarButtonsToRemove: ["select2d", "lasso2d"],
};

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
function palette() {
  const css = getComputedStyle(document.documentElement);
  const read = (name) => css.getPropertyValue(name).trim();
  return { fg: read("--fg"), muted: read("--muted"), line: read("--line"), accent: read("--accent"), truth: read("--ok") };
}

/** Restyle a figure's layout for a card on the page: its colours, its size, no title, the legend below. */
function pageLayout(layout, height) {
  const colours = palette();
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
    margin: { l: 56, r: 16, t: 12, b: 52 },
    legend: { orientation: "h", x: 0, y: -0.22, bgcolor: "rgba(0,0,0,0)" },
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
export async function draw(div, figure, version, height = 340) {
  const Plotly = await loadPlotly(version);
  await Plotly.react(div, figure.data, pageLayout(figure.layout || {}, height), CONFIG);
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
function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * power).find((step) => step >= raw);
}

/**
 * The estimated coefficients with their 95% intervals, one row per term, and the true
 * value beside each when the results were simulated. The intercept is left out: it sits
 * on another scale.
 */
export function coefficientFigure(coefficients, simulated) {
  const colours = palette();
  const terms = coefficients.filter((c) => c.term !== "Intercept");
  const labels = terms.map((c) => c.label);
  const above = terms.map((c) => (c.ci_high === null || c.ci_high === undefined ? 0 : c.ci_high - c.coefficient));
  const below = terms.map((c) => (c.ci_low === null || c.ci_low === undefined ? 0 : c.coefficient - c.ci_low));
  const data = [
    {
      type: "scatter",
      mode: "markers",
      name: terms.some((c) => c.ci_low !== null && c.ci_low !== undefined) ? "Estimate, with its 95% interval" : "Estimate",
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
  if (simulated) {
    data.push({
      type: "scatter",
      mode: "markers",
      name: "True value",
      x: terms.map((c) => c.true_value ?? 0),
      y: labels,
      marker: { symbol: "diamond-open", size: 13, color: colours.truth, line: { width: 2, color: colours.truth } },
      hovertemplate: "%{y}, true: %{x:.4g}<extra></extra>",
    });
  }
  return {
    data,
    layout: {
      xaxis: { title: { text: "Coefficient, coded units" }, zeroline: true },
      yaxis: { type: "category", autorange: "reversed" },
      showlegend: true,
    },
    height: Math.max(220, 30 * terms.length + 100),
  };
}
