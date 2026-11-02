// The design catalog's views, built as pieces of the document: the runs of a design projected onto two
// or three of its factors, the timeline of the published designs, the comparison of designs side by side
// and the errata. catalog.js decides when each is shown, and catalog-logic.js holds the rules they draw.

import {
  FAMILY_COLOURS,
  OTHER_COLOUR,
  compareRows,
  projected,
  projectionSummary,
  projectionText,
} from "./catalog-logic.js";

// ---------------------------------------------------------------- small pieces of the document
export function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.includes("-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

const NS = "http://www.w3.org/2000/svg";

function shape(tag, attributes = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}

const titled = (node, text) => {
  node.append(shape("title", {}, text));
  return node;
};

/** A table in a box that scrolls; ``rowClass`` gives each row's class from its index, such as a run outside a limit. */
/** A table of heads and rows: a cell is text, a node, or a <td> made with its own class and style, used as it is. */
export function table(head, rows, className = "data", rowClass = () => "") {
  return element(
    "div",
    { class: "scroll" },
    element(
      "table",
      { class: className },
      element("thead", {}, element("tr", {}, ...head.map((h) => element("th", { scope: "col", text: h })))),
      element(
        "tbody",
        {},
        ...rows.map((row, i) =>
          element("tr", { class: rowClass(i) || null }, ...row.map((cell) => (cell?.tagName === "TD" ? cell : element("td", {}, cell ?? "")))),
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------- a design's projections
/** A point's radius on a projection: its area grows with the runs that land on it. */
const radius = (count) => Math.min(16, 4.5 * Math.sqrt(count));

/**
 * Draw a design's runs projected onto two factors, as a square, or three, as a cube seen at an angle:
 * one point per distinct setting, its area in proportion to the runs that land on it. The box is the
 * factors' low and high levels; a run beyond it, such as an axial run of a central composite
 * design, is drawn outside. When the points hold different numbers of runs, a point with more than
 * one is labelled with its count, as "3×", so the label is not read as a response at that corner.
 */
export function projectionPlot(runs, columns, names) {
  const points = projected(runs, columns);
  const reach = Math.max(1, ...points.flatMap((p) => p.point.map(Math.abs)));
  const three = columns.length === 3;
  // The cube's depth axis goes up and to the right, at half length, as an oblique drawing does.
  const place = ([x, y, z = 0]) => [x + (three ? 0.42 * z : 0), y + (three ? 0.3 * z : 0)];
  const corners = three
    ? [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => [x, y, z])))
    : [[-1, -1], [1, -1], [-1, 1], [1, 1]];
  const placed = [...corners, ...points.map((p) => p.point)].map(place);
  const xs = placed.map((p) => p[0]);
  const ys = placed.map((p) => p[1]);
  const pad = 0.25 * reach;
  const [x0, x1, y0, y1] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  const size = 260;
  const scale = (size - 40) / Math.max(x1 - x0, y1 - y0);
  const px = ([x, y]) => [20 + (x - x0) * scale, size - 20 - (y - y0) * scale];
  const svg = shape("svg", { viewBox: `0 0 ${size} ${size}`, class: "projection", role: "img" });
  svg.setAttribute("aria-label", `The runs projected onto ${names.join(", ")}`);
  const edge = (a, b, cls = "projection__box") => {
    const [[ax, ay], [bx, by]] = [px(place(a)), px(place(b))];
    return shape("line", { x1: ax, y1: ay, x2: bx, y2: by, class: cls });
  };
  const edges = corners.flatMap((a, i) =>
    corners.slice(i + 1).filter((b) => a.filter((v, j) => v !== b[j]).length === 1).map((b) => edge(a, b)),
  );
  svg.append(...edges);
  // Each factor's name beside the middle of an edge along it, outside the box: below, to the left, and,
  // for the depth, below and to the right of the cube's bottom right edge, clear of the labels of its points.
  const out = 1 + 0.2 * reach;
  const labels = three ? [[0, -out, -1], [-out, 0, -1], [out, -out, 0]] : [[0, -out], [-out, 0]];
  labels.forEach((at, j) => {
    const [x, y] = px(place(at));
    svg.append(shape("text", { x, y, class: "projection__axis", "text-anchor": "middle", "dominant-baseline": "middle" }, names[j]));
  });
  // Far points first, so a nearer point is drawn over them.
  const order = [...points].sort((a, b) => (b.point[2] ?? 0) - (a.point[2] ?? 0));
  const labelled = new Set(points.map((p) => p.count)).size > 1;
  for (const { point, count } of order) {
    const [x, y] = px(place(point));
    const dot = titled(shape("circle", { cx: x, cy: y, r: radius(count), class: "projection__point" }), `${point.map((v) => +v.toFixed(3)).join(", ")}: ${count} ${count === 1 ? "run" : "runs"}`);
    svg.append(dot);
    if (labelled && count > 1) svg.append(shape("text", { x: x + radius(count) + 2, y: y - radius(count) + 2, class: "projection__count" }, `${count}×`));
  }
  return svg;
}

/**
 * The projection explorer: buttons that choose two or three of a design's factors, the runs projected
 * onto them, and what that projection estimates. Choosing a fourth factor lets go of the first chosen.
 */
export function projectionExplorer(runs, names) {
  let order = names.length >= 3 ? [0, 1, 2] : [0, 1];
  const picks = element("div", { class: "explorer__picks", role: "group", "aria-label": "The factors to project onto" });
  const plot = element("div", { class: "explorer__plot" });
  const text = element("p", { class: "explorer__text", "aria-live": "polite" });
  const show = () => {
    const chosen = [...order].sort((a, b) => a - b);
    for (const button of picks.children) button.setAttribute("aria-pressed", String(chosen.includes(Number(button.dataset.column))));
    const named = chosen.map((c) => names[c]);
    plot.replaceChildren(projectionPlot(runs, chosen, named));
    text.textContent = projectionText(projectionSummary(runs, chosen), named);
  };
  names.forEach((name, column) => picks.append(element("button", { type: "button", class: "chip", "data-column": column, text: name })));
  picks.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const column = Number(button.dataset.column);
    if (order.includes(column)) {
      if (order.length > 2) order = order.filter((c) => c !== column);
    } else {
      order = [...order, column].slice(-3);
    }
    show();
  });
  show();
  return element(
    "section",
    { class: "explorer" },
    element("p", { class: "hint", text: "Choose two or three factors. The plot shows the runs as those factors alone see them, each point sized by the runs that land on it; the box is the low and high levels." }),
    picks,
    plot,
    text,
  );
}

// ---------------------------------------------------------------- the timeline
/** The families in the order their colours are listed, then the rest: the order of a bar's segments. */
const STACK = [...Object.keys(FAMILY_COLOURS)];

/**
 * Draw the published designs by the year their source appeared: a bar per span of years, in the colours
 * of their families, the families first published within the span marked above, and the years
 * chosen in full colour. A bar is a button; the page also lets a reader drag across the years.
 */
export function drawTimeline(svg, bins, marks, [from, to]) {
  const width = 720;
  const top = 22;
  const height = 84;
  svg.setAttribute("viewBox", `0 0 ${width} ${top + height + 18}`);
  if (!bins.length) return svg.replaceChildren();
  const first = bins[0].from;
  const last = bins.at(-1).to + 1;
  svg.dataset.first = first;
  svg.dataset.last = last;
  const x = (year) => ((year - first) / (last - first)) * width;
  const most = Math.max(...bins.map((b) => b.count), 1);
  const groups = bins.map((bin) => {
    const chosen = (from === null || bin.to >= from) && (to === null || bin.from <= to);
    const g = shape("g", { class: chosen ? "bin" : "bin bin--out", tabindex: "0", role: "button", "data-from": bin.from, "data-to": bin.to });
    g.setAttribute("aria-label", `${bin.from} to ${bin.to}: ${bin.count} designs`);
    let base = top + height;
    const families = Object.keys(bin.families).sort((a, b) => (STACK.indexOf(a) + 1 || 99) - (STACK.indexOf(b) + 1 || 99));
    for (const family of families) {
      const h = (height * bin.families[family]) / most;
      base -= h;
      g.append(shape("rect", { x: x(bin.from) + 1, y: base, width: Math.max(x(bin.to + 1) - x(bin.from) - 2, 1), height: h, fill: FAMILY_COLOURS[family] ?? OTHER_COLOUR }));
    }
    // A bar with few designs is still a target as tall as the plot.
    g.append(shape("rect", { x: x(bin.from), y: top, width: x(bin.to + 1) - x(bin.from), height, class: "bin__hit" }));
    return titled(g, `${bin.from} to ${bin.to}: ${bin.count} designs`);
  });
  const ticks = [];
  for (let year = Math.ceil(first / 10) * 10; year < last; year += 10) {
    ticks.push(shape("text", { x: x(year), y: top + height + 14, class: "timeline__year", "text-anchor": "middle" }, String(year)));
  }
  const flags = marks.map((mark, i) => {
    const cx = x(mark.year + 0.5);
    const cy = 6 + (i % 2) * 9;
    const flag = shape("path", { d: `M${cx} ${cy - 4} L${cx + 4} ${cy} L${cx} ${cy + 4} L${cx - 4} ${cy} Z`, fill: FAMILY_COLOURS[mark.family] ?? OTHER_COLOUR, class: "timeline__mark" });
    return titled(flag, `${mark.label} designs, first published in ${mark.year}`);
  });
  const brush = shape("rect", { x: 0, y: top, width: 0, height, class: "timeline__brush", visibility: "hidden" });
  svg.replaceChildren(...groups, ...ticks, ...flags, brush);
  return svg;
}

// ---------------------------------------------------------------- designs side by side
/**
 * The comparison of designs: a row per measure of compare_designs, grouped as the book groups them,
 * with the direction each measure prefers and every design's value. No design is marked best.
 */
export function compareTable(measures, designs, scored, model) {
  const rows = compareRows(measures, scored, model);
  const head = element(
    "thead",
    {},
    element("tr", {}, element("th", { scope: "col", text: "Measure" }), element("th", { scope: "col", text: "Better" }), ...designs.map((d) => element("th", { scope: "col", text: d.title }))),
  );
  const bodies = [];
  for (const row of rows) {
    if (!bodies.length || bodies.at(-1).dataset.group !== row.group) {
      const body = element("tbody", { "data-group": row.group }, element("tr", { class: "compare__group" }, element("th", { scope: "rowgroup", colSpan: 2 + designs.length, text: row.group })));
      bodies.push(body);
    }
    bodies.at(-1).append(
      element(
        "tr",
        {},
        element("th", { scope: "row", title: row.description, text: row.label }),
        element("td", { class: "compare__arrow", text: row.arrow }),
        ...row.cells.map((cell) => element("td", { text: cell })),
      ),
    );
  }
  return element("div", { class: "scroll" }, element("table", { class: "data compare" }, head, ...bodies));
}

// ---------------------------------------------------------------- the errata
/** The errata as a table: the entry that prints each number, the quantity, the value printed, where, and why. */
export function errataTable(errata, catalog) {
  const known = new Set(catalog.designs.map((d) => d.id));
  const rows = errata.map((e) => {
    const source = catalog.sources[e.source];
    const entry = known.has(e.id) ? element("button", { type: "button", class: "link", "data-id": e.id, text: e.id }) : e.id;
    const where = e.url ? element("a", { href: e.url, target: "_blank", rel: "noopener", text: e.where || "web page" }) : e.where;
    const quantity = [String(e.what ?? "").replace(/_/g, " "), e.term].filter(Boolean).join(", ");
    return [source ? `${source.short} (${source.year})` : e.source, entry, quantity, String(e.printed ?? ""), where, e.reason];
  });
  return table(["Source", "Entry", "Quantity", "Printed", "Where", "Why the library's value differs"], rows, "data errata__table");
}
