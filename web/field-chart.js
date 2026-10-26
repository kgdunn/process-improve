// The chart of a field's published experiments, on the field's page and in the planning panel: a mark for each size
// and kind of design, placed by its factors (across) and its runs (up, doubling at each line). Each kind has a shape
// of its own, so the kinds are told apart without their colours. A field's page writes its points into the svg's
// data-field-points, which this module draws on load; the panel calls drawFieldChart itself, with the reader's plan.

const NS = "http://www.w3.org/2000/svg";

/** The chart's frame, in the units of its viewBox. */
export const FRAME = { width: 360, height: 248, left: 38, right: 14, top: 26, bottom: 42 };

/** The most factors drawn apart: a design with more is drawn at the edge, labelled "9+". */
export const MOST_FACTORS = 9;

/** The runs at each line of the vertical axis, doubling; fewer or more are drawn at the bottom or the top. */
export const RUN_LINES = [4, 8, 16, 32, 64, 128];

/** The kinds of design, as the build names them (FIELD_GROUPS in scripts/build_web_app.py). */
export const KINDS = ["full", "screening", "surface", "mixture", "other"];

/** Kinds of one size sit side by side, so that each mark shows. */
const NUDGE = { full: -5, screening: 5, surface: 0, mixture: -2.5, other: 2.5 };

/** Where a number of factors is drawn across the chart. */
export function chartX(factors) {
  const inner = FRAME.width - FRAME.left - FRAME.right;
  return FRAME.left + ((Math.min(Math.max(factors, 1), MOST_FACTORS) - 1) / (MOST_FACTORS - 1)) * inner;
}

/** Where a number of runs is drawn up the chart, on a scale that doubles at each line. */
export function chartY(runs) {
  const inner = FRAME.height - FRAME.top - FRAME.bottom;
  const low = Math.log2(RUN_LINES[0]);
  const high = Math.log2(RUN_LINES.at(-1));
  const held = Math.min(Math.max(runs, RUN_LINES[0]), RUN_LINES.at(-1));
  return FRAME.top + (1 - (Math.log2(held) - low) / (high - low)) * inner;
}

/** The marks for a field's points, each ``[factors, runs, kind, how many]``: a larger mark for more experiments. */
export function chartMarks(points) {
  return points.map(([factors, runs, kind, count]) => ({
    kind: KINDS.includes(kind) ? kind : "other",
    x: chartX(factors) + (NUDGE[kind] ?? 0),
    y: chartY(runs),
    size: Math.min(16, 6 + 2.5 * Math.sqrt(count)),
  }));
}

function shape(tag, attributes = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== null) node.textContent = text;
  return node;
}

/** One mark, in its kind's shape: a disc, a square, a diamond, a triangle, or a ring for the other designs. */
function mark({ kind, x, y, size }) {
  const half = size / 2;
  const attributes = { class: `field-chart__mark field-chart__mark--${kind}` };
  if (kind === "screening") return shape("rect", { ...attributes, x: x - half * 0.9, y: y - half * 0.9, width: size * 0.9, height: size * 0.9 });
  if (kind === "surface") return shape("polygon", { ...attributes, points: `${x},${y - half} ${x + half},${y} ${x},${y + half} ${x - half},${y}` });
  if (kind === "mixture") return shape("polygon", { ...attributes, points: `${x},${y - half} ${x + half},${y + half * 0.8} ${x - half},${y + half * 0.8}` });
  return shape("circle", { ...attributes, cx: x, cy: y, r: kind === "other" ? half * 0.8 : half });
}

/**
 * Draw the chart into ``svg``: its lines and labels, a mark for each point, and the reader's plan when given, as
 * ``{factors, runs}``. A plan without runs yet is drawn as a band at its number of factors.
 */
export function drawFieldChart(svg, points, plan = null) {
  const { width, height, left, right, top, bottom } = FRAME;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const parts = [];
  for (const runs of RUN_LINES) {
    const y = chartY(runs);
    parts.push(shape("line", { class: "field-chart__line", x1: left, x2: width - right, y1: y, y2: y }));
    parts.push(shape("text", { class: "field-chart__tick", x: left - 6, y: y + 4, "text-anchor": "end" }, String(runs)));
  }
  for (let factors = 1; factors <= MOST_FACTORS; factors += 1) {
    const label = factors === MOST_FACTORS ? `${factors}+` : String(factors);
    parts.push(shape("text", { class: "field-chart__tick", x: chartX(factors), y: height - bottom + 16, "text-anchor": "middle" }, label));
  }
  parts.push(shape("text", { class: "field-chart__axis", x: (left + width - right) / 2, y: height - 6, "text-anchor": "middle" }, "Factors"));
  parts.push(shape("text", { class: "field-chart__axis", x: 4, y: 12 }, "Runs"));
  parts.push(...chartMarks(points).map(mark));
  if (plan?.factors) {
    const x = chartX(plan.factors);
    if (plan.runs) {
      const y = chartY(plan.runs);
      parts.push(shape("circle", { class: "field-chart__plan", cx: x, cy: y, r: 9 }));
      parts.push(shape("text", { class: "field-chart__plan-label", x: x + 13, y: y - 9 }, "Your plan"));
    } else {
      parts.push(shape("rect", { class: "field-chart__band", x: x - 9, y: top, width: 18, height: height - top - bottom }));
      parts.push(shape("text", { class: "field-chart__plan-label", x: x + 12, y: top + 10 }, "Your factors"));
    }
  }
  svg.replaceChildren(...parts);
}

// A field's page: draw the chart its build wrote the points of.
if (typeof document !== "undefined") {
  for (const svg of document.querySelectorAll("svg[data-field-points]")) drawFieldChart(svg, JSON.parse(svg.dataset.fieldPoints));
}
