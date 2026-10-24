// Linked views: the Plotly figures of the multivariate app, built from plain points and the selection's state.
// Every function here returns a figure ({ data, layout }) and touches nothing else, so tests/web/mva runs it
// under Node; web/mva/link.js draws the figures and turns the reader's clicks and lassos into selections.
//
// A point is { id, x, y, label?, colour?, size?, symbol?, hover? }. `id` names the row (or the column) in the
// selection store; `colour` is a number (a colour scale) or a category (a palette); `size` a number, drawn with
// the marker's area in proportion; `symbol` a category. A state is what Selection.snapshot() returns.

/** Colours for categories: Okabe-Ito, which reads apart with every colour vision; black last. */
export const CATEGORY_COLOURS = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#CC79A7",
  "#56B4E9",
  "#D55E00",
  "#F0E442",
  "#000000",
];

/** Plotly's marker symbols for categories, each with a hollow form ("-open") for an excluded point. */
export const CATEGORY_SYMBOLS = [
  "circle",
  "square",
  "diamond",
  "triangle-up",
  "triangle-down",
  "pentagon",
  "hexagon",
  "star",
];

/** The colours of a contribution's sign: one hue each way, as in the designed-experiments app. */
export const SIGN_COLOURS = { positive: "#E69F00", negative: "#0072B2" };

/** The page's colours when the caller passes none: the light theme. */
export const LIGHT = { fg: "#1a1a1a", muted: "#666666", line: "#d0d0d0", bg: "#ffffff", accent: "#0072B2" };

/** Marker diameters, in pixels: the usual one, the smallest and largest a size encoding draws. */
export const MARKER = { size: 9, min: 6, max: 26 };

/** At more points than this a scatter is drawn with WebGL, which draws fast but does not animate. */
export const WEBGL_ABOVE = 1500;

/** The opacity of a point that is not selected, while something is. */
export const FADED = 0.25;

/** The eight places Plotly can put a point's label, clockwise from the right. */
const POSITIONS = [
  "middle right",
  "top right",
  "top center",
  "top left",
  "middle left",
  "bottom left",
  "bottom center",
  "bottom right",
];

/** The title of a component's axis, with the percent of variance it explains in square brackets: "t1 [27.7%]". */
export function componentTitle(prefix, component, r2) {
  const share = Number.isFinite(r2) ? ` [${(100 * r2).toFixed(1)}%]` : "";
  return `${prefix}${component}${share}`;
}

/** How a list of values encodes: "number" when every value present is a finite number, else "category". */
export function encodingKind(values) {
  const present = values.filter((v) => v !== null && v !== undefined && v !== "");
  return present.length && present.every((v) => typeof v === "number" && Number.isFinite(v)) ? "number" : "category";
}

/** Marker diameters with the area in proportion to each value: MARKER.min at the smallest, MARKER.max at the largest. */
export function markerSizes(values, { min = MARKER.min, max = MARKER.max } = {}) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return values.map(() => MARKER.size);
  const lo = Math.min(...finite);
  const span = Math.max(...finite) - lo;
  return values.map((v) => {
    if (!Number.isFinite(v)) return min;
    const share = span > 0 ? (v - lo) / span : 0.5;
    return Math.sqrt(min * min + share * (max * max - min * min));
  });
}

/** The categories of a list of values, in the order they first appear, each with its colour and symbol. */
export function categories(values) {
  const seen = [];
  for (const v of values) if (v !== null && v !== undefined && !seen.includes(String(v))) seen.push(String(v));
  return seen.map((name, k) => ({
    name,
    colour: CATEGORY_COLOURS[k % CATEGORY_COLOURS.length],
    symbol: CATEGORY_SYMBOLS[k % CATEGORY_SYMBOLS.length],
  }));
}

/**
 * Which points carry a label: the hovered one, the selected ones, and the most extreme of the rest, up to
 * `limit` in all. Extreme is the distance from the centre in units of each axis's spread, so a label goes
 * where a reader looks first. `labels` is "auto", "all" or "none".
 */
export function chooseLabels(points, state, { labels = "auto", limit = 12 } = {}) {
  if (labels === "none") return new Set();
  if (labels === "all") return new Set(points.map((p) => p.id));
  const chosen = new Set();
  if (state.hovered !== null && points.some((p) => p.id === state.hovered)) chosen.add(state.hovered);
  for (const p of points) if (chosen.size < limit && state.selected.has(p.id)) chosen.add(p.id);
  const sx = spread(points.map((p) => p.x));
  const sy = spread(points.map((p) => p.y));
  const ranked = points
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && !chosen.has(p.id))
    .map((p) => ({ id: p.id, d: (p.x / sx) ** 2 + (p.y / sy) ** 2 }))
    .sort((a, b) => b.d - a.d);
  for (const { id } of ranked) {
    if (chosen.size >= limit) break;
    chosen.add(id);
  }
  return chosen;
}

/**
 * Where each labelled point's label goes: on the side away from its nearest neighbours, so a label never sits
 * on another point. Returns one Plotly textposition per point (unlabelled points get "middle right").
 */
export function labelPositions(points, labelled, { neighbours = 5 } = {}) {
  const sx = spread(points.map((p) => p.x));
  const sy = spread(points.map((p) => p.y));
  const scaled = points.map((p) => [p.x / sx, p.y / sy]);
  return points.map((p, i) => {
    if (!labelled.has(p.id) || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return POSITIONS[0];
    const near = scaled
      .map((q, j) => ({ j, d: (q[0] - scaled[i][0]) ** 2 + (q[1] - scaled[i][1]) ** 2 }))
      .filter(({ j, d }) => j !== i && Number.isFinite(d))
      .sort((a, b) => a.d - b.d)
      .slice(0, neighbours);
    if (!near.length) return POSITIONS[0];
    // Push away from the neighbours, the nearest weighing most; with none in the way, away from the centre.
    let dx = 0;
    let dy = 0;
    for (const { j, d } of near) {
      const w = 1 / (d + 1e-6);
      dx += w * (scaled[i][0] - scaled[j][0]);
      dy += w * (scaled[i][1] - scaled[j][1]);
    }
    if (Math.hypot(dx, dy) < 1e-9) [dx, dy] = scaled[i];
    const octant = Math.round(Math.atan2(dy, dx) / (Math.PI / 4));
    return POSITIONS[(octant + 8) % 8];
  });
}

const spread = (values) => {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return 1;
  const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
  const sd = Math.sqrt(finite.reduce((a, b) => a + (b - mean) ** 2, 0) / (finite.length - 1));
  return sd > 0 ? sd : 1;
};

/**
 * Each point's look: colour, diameter, symbol (hollow when excluded), opacity (faded while something else is
 * selected), and outline (dark around a selected point). `encode` names how points carry colour, size and
 * symbol: { colour: "number" | "category" | null, size: bool, symbol: bool }.
 */
export function pointStyles(points, state, encode, colours = LIGHT) {
  const anySelected = state.selected.size > 0;
  const colourOf = colourScale(
    points.map((p) => p.colour),
    encode.colour,
    colours,
  );
  const symbolCats = encode.symbol ? categories(points.map((p) => p.symbol)) : [];
  const sizes = encode.size ? markerSizes(points.map((p) => p.size)) : points.map(() => MARKER.size);
  const inSet = setMembership(state.sets);
  return points.map((p, i) => {
    const selected = state.selected.has(p.id);
    const hovered = state.hovered === p.id;
    const excluded = state.excluded.has(p.id);
    const base = encode.symbol ? (symbolCats.find((c) => c.name === String(p.symbol))?.symbol ?? "circle") : "circle";
    const setColour = inSet.get(p.id);
    return {
      colour: encode.colour ? colourOf(p.colour) : (setColour ?? colours.accent),
      size: sizes[i] * (hovered ? 1.6 : 1),
      symbol: excluded ? `${base}-open` : base,
      opacity: anySelected && !selected && !hovered ? FADED : 1,
      outline: selected ? colours.fg : excluded ? "rgba(0,0,0,0)" : colours.bg,
      outlineWidth: selected ? 2 : 1,
    };
  });
}

/** Each row in a saved set, with the colour of the last set it is in. */
const setMembership = (sets) => {
  const out = new Map();
  for (const { rows, colour } of sets ?? []) for (const id of rows) out.set(id, colour);
  return out;
};

/** A function from a value to its colour: Viridis-like for numbers, the category palette for categories. */
function colourScale(values, kind, colours) {
  if (kind === "category") {
    const cats = categories(values);
    return (v) => cats.find((c) => c.name === String(v))?.colour ?? colours.muted;
  }
  if (kind !== "number") return () => colours.accent;
  const finite = values.filter(Number.isFinite);
  const lo = Math.min(...finite);
  const span = Math.max(...finite) - lo || 1;
  return (v) => (Number.isFinite(v) ? viridis((v - lo) / span) : colours.muted);
}

/** A colour of the Viridis scale, which keeps its order in grey and with every colour vision; t in [0, 1]. */
export function viridis(t) {
  const stops = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const k = Math.min(stops.length - 2, Math.floor(x));
  const mix = (a, b) => Math.round(a + (b - a) * (x - k));
  const [r1, g1, b1] = rgb(stops[k]);
  const [r2, g2, b2] = rgb(stops[k + 1]);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
}

const rgb = (hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));

/** Text safe to put in a Plotly label or hover, which Plotly reads as a little HTML. */
export const plain = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/**
 * A scatter of points, linked by their ids: a score plot (rows) or a loading plot (columns).
 *
 * options: {
 *   xTitle, yTitle,            the axis titles (componentTitle writes a score plot's)
 *   colourTitle, sizeTitle,    what the colour and size encode, for the legend and the colour bar
 *   labels: "auto" | "all" | "none", maxLabels,
 *   ellipse: { x: [], y: [], name },   the T² ellipse, or any closed curve
 *   origin: true,              dashed lines through zero
 *   annotations: [{ id, text }],       an arrow at a point, with its text
 *   height, colours
 * }
 */
export function scatterFigure(points, state, options = {}) {
  const colours = options.colours ?? LIGHT;
  const encode = {
    colour: points.some((p) => p.colour !== undefined && p.colour !== null)
      ? encodingKind(points.map((p) => p.colour))
      : null,
    size: points.some((p) => Number.isFinite(p.size)),
    symbol: points.some((p) => p.symbol !== undefined && p.symbol !== null),
  };
  const styles = pointStyles(points, state, encode, colours);
  const labelled = chooseLabels(points, state, { labels: options.labels, limit: options.maxLabels });
  const positions = labelPositions(points, labelled);
  const type = points.length > WEBGL_ABOVE ? "scattergl" : "scatter";

  // Fitted and excluded points are two traces, so the legend can tell them apart; draw order puts the
  // excluded, hollow ones on top, where they cannot hide behind the cloud.
  const groups = [
    { name: "Fitted", keep: (p) => !state.excluded.has(p.id) },
    { name: "Excluded, projected", keep: (p) => state.excluded.has(p.id) },
  ];
  const data = [];
  for (const group of groups) {
    const index = points.map((p, i) => i).filter((i) => group.keep(points[i]));
    if (!index.length) continue;
    data.push({
      type,
      mode: "markers+text",
      name: group.name,
      x: index.map((i) => points[i].x),
      y: index.map((i) => points[i].y),
      customdata: index.map((i) => points[i].id),
      text: index.map((i) => (labelled.has(points[i].id) ? plain(points[i].label ?? points[i].id) : "")),
      textposition: index.map((i) => positions[i]),
      textfont: { size: 11, color: colours.fg },
      hovertext: index.map((i) => plain(points[i].hover ?? points[i].label ?? points[i].id)),
      hoverinfo: "text",
      marker: {
        color: index.map((i) => styles[i].colour),
        size: index.map((i) => styles[i].size),
        symbol: index.map((i) => styles[i].symbol),
        opacity: index.map((i) => styles[i].opacity),
        line: { color: index.map((i) => styles[i].outline), width: index.map((i) => styles[i].outlineWidth) },
      },
      selectedpoints: null, // the store draws the selection; Plotly's own dimming would fight it
      showlegend: groups.every((g) => points.some(g.keep)),
    });
  }
  if (encode.colour === "category")
    data.push(
      ...categoryKeys(
        points.map((p) => p.colour),
        options.colourTitle,
        type,
      ),
    );
  if (options.ellipse) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: options.ellipse.name ?? "T² limit",
      x: options.ellipse.x,
      y: options.ellipse.y,
      line: { color: colours.muted, dash: "dash", width: 1 },
      hoverinfo: "skip",
    });
  }
  const shapes = options.origin === false ? [] : originLines(colours);
  return {
    data,
    layout: {
      height: options.height ?? 420,
      margin: { l: 60, r: 20, t: 20, b: 50 },
      paper_bgcolor: colours.bg,
      plot_bgcolor: colours.bg,
      font: { color: colours.fg },
      xaxis: { title: { text: options.xTitle ?? "" }, zeroline: false, gridcolor: colours.line },
      yaxis: { title: { text: options.yTitle ?? "" }, zeroline: false, gridcolor: colours.line },
      dragmode: "lasso",
      hovermode: "closest",
      shapes,
      annotations: pointAnnotations(points, options.annotations ?? [], colours),
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" },
      selections: [],
    },
  };
}

/** Legend entries for the colour categories: one empty trace each, so the legend shows what each colour is. */
function categoryKeys(values, title, type) {
  return categories(values).map((c, k) => ({
    type,
    mode: "markers",
    x: [null],
    y: [null],
    name: plain(c.name),
    legendgroup: "colour",
    legendgrouptitle: k === 0 && title ? { text: plain(title) } : undefined,
    marker: { color: c.colour, size: MARKER.size },
    hoverinfo: "skip",
  }));
}

/** Dashed lines through the origin, in paper coordinates along their axis. */
const originLines = (colours) => [
  {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    line: { color: colours.muted, dash: "dot", width: 1 },
  },
  {
    type: "line",
    xref: "paper",
    yref: "y",
    x0: 0,
    x1: 1,
    y0: 0,
    y1: 0,
    line: { color: colours.muted, dash: "dot", width: 1 },
  },
];

/** An arrow at each named point, with its text, small, as the text that discusses it points there. */
function pointAnnotations(points, wanted, colours) {
  return wanted
    .map(({ id, text }) => {
      const p = points.find((q) => q.id === id);
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
      return {
        x: p.x,
        y: p.y,
        text: plain(text),
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 1,
        arrowcolor: colours.fg,
        ax: 40,
        ay: -40,
        font: { size: 11, color: colours.fg },
        bgcolor: colours.bg,
      };
    })
    .filter(Boolean);
}

/**
 * A bar per item, linked by id: contributions, loadings of one component, VIP. Bars take the colour of their
 * sign; a selected item is outlined, the rest faded while one is selected.
 *
 * items: [{ id, value, label? }]; options: { yTitle, limits: [{ value, label }], sort: bool, height, colours }
 */
export function barFigure(items, state, options = {}) {
  const colours = options.colours ?? LIGHT;
  const shown = options.sort ? [...items].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) : items;
  const anySelected = state.selected.size > 0;
  return {
    data: [
      {
        type: "bar",
        x: shown.map((it) => plain(it.label ?? it.id)),
        y: shown.map((it) => it.value),
        customdata: shown.map((it) => it.id),
        marker: {
          color: shown.map((it) => (it.value >= 0 ? SIGN_COLOURS.positive : SIGN_COLOURS.negative)),
          opacity: shown.map((it) => (anySelected && !state.selected.has(it.id) ? FADED : 1)),
          line: {
            color: shown.map((it) => (state.selected.has(it.id) ? colours.fg : colours.bg)),
            width: shown.map((it) => (state.selected.has(it.id) || state.hovered === it.id ? 2 : 0)),
          },
        },
        hovertemplate: "%{x}: %{y:.3g}<extra></extra>",
        selectedpoints: null,
      },
    ],
    layout: {
      height: options.height ?? 320,
      margin: { l: 60, r: 20, t: 20, b: 90 },
      paper_bgcolor: colours.bg,
      plot_bgcolor: colours.bg,
      font: { color: colours.fg },
      xaxis: { tickangle: -45, automargin: true },
      yaxis: { title: { text: options.yTitle ?? "" }, gridcolor: colours.line, zerolinecolor: colours.muted },
      shapes: (options.limits ?? []).map((limit) => limitLine(limit.value, colours)),
      annotations: (options.limits ?? []).map((limit) => limitLabel(limit, colours)),
      dragmode: "select",
      selections: [],
      bargap: 0.25,
    },
  };
}

/**
 * A line per series in run order, with the rows of a group marked: the raw data behind a contribution bar, or a
 * T² or SPE chart against its limits. Rows are linked by id; an excluded row is hollow.
 *
 * series: [{ name, ids: [], y: [] }]; options: { group: Set of ids, limits: [{ value, label }], yTitle, colours }
 */
export function sequenceFigure(series, state, options = {}) {
  const colours = options.colours ?? LIGHT;
  const group = options.group ?? new Set();
  const data = series.flatMap((s, k) => {
    const colour = CATEGORY_COLOURS[k % CATEGORY_COLOURS.length];
    const order = s.ids.map((_, i) => i + 1);
    const marked = s.ids.map((id, i) => i).filter((i) => group.has(s.ids[i]) || state.selected.has(s.ids[i]));
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name: plain(s.name),
        x: order,
        y: s.y,
        customdata: s.ids,
        text: s.ids.map(plain),
        hovertemplate: "%{text}: %{y:.4g}<extra></extra>",
        line: { color: colour, width: 1.5 },
        marker: {
          color: colour,
          size: 5,
          symbol: s.ids.map((id) => (state.excluded.has(id) ? "circle-open" : "circle")),
        },
        selectedpoints: null,
      },
      {
        type: "scatter",
        mode: "markers",
        name: `${plain(s.name)}: the group`,
        x: marked.map((i) => order[i]),
        y: marked.map((i) => s.y[i]),
        customdata: marked.map((i) => s.ids[i]),
        hoverinfo: "skip",
        // Solid and outlined, as a selected point is on a scatter: hollow means excluded, and only that.
        marker: { color: colour, size: 9, line: { color: colours.fg, width: 1.5 } },
        showlegend: false,
      },
    ];
  });
  return {
    data,
    layout: {
      height: options.height ?? 260,
      margin: { l: 60, r: 20, t: 10, b: 40 },
      paper_bgcolor: colours.bg,
      plot_bgcolor: colours.bg,
      font: { color: colours.fg },
      xaxis: {
        title: { text: options.xTitle ?? "Run order" },
        gridcolor: colours.line,
        dtick: series[0]?.ids.length > 40 ? undefined : 5,
      },
      yaxis: { title: { text: options.yTitle ?? "" }, gridcolor: colours.line },
      shapes: (options.limits ?? []).map((limit) => limitLine(limit.value, colours)),
      annotations: (options.limits ?? []).map((limit) => limitLabel(limit, colours)),
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" },
      dragmode: "select",
      selections: [],
    },
  };
}

const limitLine = (value, colours) => ({
  type: "line",
  xref: "paper",
  yref: "y",
  x0: 0,
  x1: 1,
  y0: value,
  y1: value,
  line: { color: colours.muted, dash: "dash", width: 1 },
});

const limitLabel = (limit, colours) => ({
  xref: "paper",
  yref: "y",
  x: 1,
  y: limit.value,
  xanchor: "right",
  yanchor: "bottom",
  text: plain(limit.label ?? ""),
  showarrow: false,
  font: { size: 10, color: colours.muted },
});
