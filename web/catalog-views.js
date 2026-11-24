// The design catalog's views, built as pieces of the document: the runs of a design projected onto two
// or three of its factors, the timeline of the published designs, the comparison of designs side by side
// and the errata. catalog.js decides when each is shown, and catalog-logic.js holds the rules they draw.

import { tabOf } from "./campaign.js";
import {
  MAP_ROWS,
  RUN_BANDS,
  STEP_WORDS,
  bandPlace,
  bandWords,
  compareRows,
  projected,
  projectionSummary,
  projectionText,
  rowWords,
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
 * one is labeled with its count, as "3×", so the label is not read as a response at that corner.
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
/** The height of the strip of years, in the units it is drawn in: style.css gives it the same height on the page. */
const STRIP = 28;

/**
 * Draw the published designs by the year their source appeared, as a strip for the filter panel: a bar per
 * span of years, as tall as its share of the fullest span, with the years chosen in full color. A bar is
 * a button; the page also lets a reader drag across the years. The strip is stretched to the panel's width,
 * so it holds no words: the page writes the first and last year under it.
 */
export function drawTimeline(svg, bins, [from, to]) {
  const width = 720;
  svg.setAttribute("viewBox", `0 0 ${width} ${STRIP}`);
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
    // A span with designs shows at least a line, so a reader sees it is not empty.
    const h = bin.count ? Math.max((STRIP * bin.count) / most, 1.5) : 0;
    g.append(shape("rect", { x: x(bin.from) + 1, y: STRIP - h, width: Math.max(x(bin.to + 1) - x(bin.from) - 2, 1), height: h, class: "bin__bar" }));
    // A bar with few designs is still a target as tall as the strip.
    g.append(shape("rect", { x: x(bin.from), y: 0, width: x(bin.to + 1) - x(bin.from), height: STRIP, class: "bin__hit" }));
    return titled(g, `${bin.from} to ${bin.to}: ${bin.count} designs`);
  });
  const brush = shape("rect", { x: 0, y: 0, width: 0, height: STRIP, class: "timeline__brush", visibility: "hidden" });
  svg.replaceChildren(...groups, brush);
  return svg;
}

// ---------------------------------------------------------------- the map of factors and runs
/** The fewest designs of each step of the map's colors, when a cell's color says how many it holds. */
/** The fill of a cell or a mark, as style.css names it: by the most that one of its designs estimates. */
const fillOf = (item) => `fill-${item.step ? item.step + 1 : 0}`;

/** What a cell's or a mark's designs estimate at most, as a sentence. */
const atMost = (item) => `${item.count > 1 ? "At most, they estimate" : "It estimates"} ${STEP_WORDS[item.step]}.`;

const designs = (n) => `${n} design${n === 1 ? "" : "s"}`;

/** Where each row of each map was last drawn, so that a row moves from there when another opens. */
const drawnAt = new WeakMap();

/**
 * Draw the map of the designs by their factors and runs (sizeMap, in catalog-logic.js) at the width its holder
 * has. A row is a button with its number of factors, then a cell for each band of runs with the designs it
 * holds; a cell with none in the list is pale. Over the cells run two lines: the fewest runs the main effects
 * need, and the fewest the full quadratic needs. A row in ``opened`` has a lane for each family under it, with a
 * mark for each run size, and the lines run straight down through them. A cell's color is the most that one of
 * its designs estimates, which says what a design of that size can tell a reader. The buttons, the cells and the
 * marks name themselves (data-unfold, data-band, data-runs and data-family) for the page, which decides what a
 * press does.
 * ``moved`` is the row just opened or shut: the rows then slide from where they were.
 */
export function drawSizeMap(holder, rows, { opened, moved = null }) {
  const width = holder.clientWidth;
  const head = 58, top = 20, rowH = 32, nameH = 15, laneH = 24;
  const colW = (width - head) / RUN_BANDS.length;
  const x = (runs) => head + bandPlace(runs) * colW;
  const svg = shape("svg");
  const names = shape("g", { class: "sizemap__bands", "aria-hidden": "true" });
  RUN_BANDS.forEach((most, band) => {
    const name = Number.isFinite(most) ? `≤${most}` : `${RUN_BANDS[band - 1] + 1}+`;
    names.append(shape("text", { x: head + (band + 0.5) * colW, y: 12, "text-anchor": "middle" }, name));
  });
  const cells = shape("g");
  const counts = shape("g", { "aria-hidden": "true" });
  const lanes = shape("g");
  const lines = [[], []]; // the points each of the two lines passes through
  const at = {};
  const parts = [];
  let y = top;
  let fresh = null;
  for (const row of rows) {
    const isOpen = opened.has(row.row);
    const group = shape("g");
    const numbers = shape("g");
    at[row.row] = y;
    parts.push([row.row, group], [row.row, numbers]);

    const name = row.row > MAP_ROWS ? `${row.row}+` : String(row.row);
    const chip = shape("g", { class: "sizemap__row", role: "button", tabindex: "0", "aria-expanded": String(isOpen), "data-unfold": row.row });
    chip.setAttribute("aria-label", `${rowWords(row.row)}, ${designs(row.count)}: ${isOpen ? "close" : "open"} this row's families`);
    // An arrow beside the number: to the right on a shut row, down on an open one, as a fold's own mark is.
    const arrow = isOpen ? `M${head - 24} ${rowH / 2 - 2}l4.5 4.5 4.5 -4.5` : `M${head - 20} ${rowH / 2 - 4.5}l4.5 4.5 -4.5 4.5`;
    chip.append(
      shape("rect", { x: 1, y: 2, width: head - 8, height: rowH - 4, rx: (rowH - 4) / 2 }),
      shape("text", { x: 11, y: rowH / 2 + 4, "text-anchor": "start" }, name),
      shape("path", { class: "sizemap__arrow", d: arrow, "aria-hidden": "true" }),
    );
    group.append(chip);

    row.cells.forEach((cell, band) => {
      if (!cell.count) return;
      const state = `${fillOf(cell)}${cell.listed ? "" : " out"}`;
      const words = `${rowWords(row.row)}, ${bandWords(band)}: ${designs(cell.count)}. ${atMost(cell)}`;
      const button = shape("g", { class: `sizemap__cell ${state}`, role: "button", tabindex: "-1", "data-row": row.row, "data-band": band });
      button.setAttribute("aria-label", `${words} Press to list ${cell.count === 1 ? "it" : "them"}.`);
      button.append(shape("rect", { x: head + band * colW + 1, y: 1, width: colW - 2, height: rowH - 2, rx: 3 }));
      group.append(titled(button, words));
      numbers.append(shape("text", { class: `sizemap__count ${state}`, x: head + (band + 0.5) * colW, y: rowH / 2 + 4, "text-anchor": "middle" }, String(cell.count)));
    });
    cells.append(group);
    counts.append(numbers);

    let tall = rowH;
    if (isOpen) {
      const block = shape("g");
      parts.push([row.row, block]);
      if (row.row === moved) fresh = block;
      row.fewest.forEach((runs, i) => {
        block.append(shape("text", { class: "sizemap__fewest", x: x(runs) + (i ? 5 : -5), y: rowH + 12, "text-anchor": i ? "start" : "end", "aria-hidden": "true" }, String(runs)));
      });
      let ly = rowH + 18;
      row.lanes.forEach((lane, index) => {
        const cy = ly + nameH + laneH / 2;
        block.append(
          shape("line", { class: "sizemap__rule", x1: head, x2: width, y1: ly + nameH + laneH, y2: ly + nameH + laneH }),
          // A press beside a mark finds the nearest one: the marks of neighboring sizes are a few pixels apart.
          shape("rect", { class: "sizemap__lane", x: head, y: ly + nameH, width: width - head, height: laneH, "data-row": row.row, "data-lane": index }),
        );
        // The largest first, so that a small mark sits on a large one.
        for (const mark of [...lane.marks].sort((a, b) => b.count - a.count)) {
          const words = `${mark.runs} runs, ${lane.label}: ${designs(mark.count)} of ${rowWords(row.row)}. ${atMost(mark)}`;
          const dot = shape("circle", {
            class: `sizemap__mark ${fillOf(mark)}${mark.listed ? "" : " out"}`,
            cx: x(mark.runs), cy, r: Math.min(10, 3 + 1.4 * Math.sqrt(mark.count)),
            role: "button", tabindex: "-1", "data-row": row.row, "data-lane": index, "data-runs": mark.runs, "data-family": lane.family,
          });
          dot.setAttribute("aria-label", `${words} Press to list ${mark.count === 1 ? "it" : "them"}.`);
          block.append(titled(dot, words));
        }
        block.append(shape("text", { class: "sizemap__family", x: head, y: ly + 11, "aria-hidden": "true" }, `${lane.label} · ${lane.count}`));
        ly += nameH + laneH + 4;
      });
      tall = ly + 8;
      lanes.append(block);
    }
    row.fewest.forEach((runs, i) => {
      lines[i].push([x(runs), y + rowH / 2]);
      if (isOpen) lines[i].push([x(runs), y + tall - 4]); // straight down through the row's families
    });
    y += tall;
  }
  const drawn = shape("g", { "aria-hidden": "true" });
  lines.forEach((points, i) => {
    const d = points.map(([px, py], n) => `${n ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
    const endX = points.at(-1)[0];
    drawn.append(
      shape("path", { class: "sizemap__line sizemap__line--under", d }),
      shape("path", { class: `sizemap__line${i ? "" : " sizemap__line--main"}`, d }),
      shape("text", { class: "sizemap__name", x: Math.min(width - 40, Math.max(head + 34, endX)), y: y + 12, "text-anchor": "middle" }, i ? "full quadratic" : "main effects"),
    );
  });
  svg.setAttribute("viewBox", `0 0 ${width} ${y + 20}`);
  for (const [row, part] of parts) part.style.transform = `translate(0px, ${at[row]}px)`;
  svg.append(names, cells, drawn, counts, lanes);
  holder.replaceChildren(svg);

  const before = drawnAt.get(holder);
  drawnAt.set(holder, at);
  if (moved === null || !before || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  for (const [row, part] of parts) {
    if (before[row] === undefined || before[row] === at[row]) continue;
    part.animate([{ transform: `translate(0px, ${before[row]}px)` }, { transform: `translate(0px, ${at[row]}px)` }], { duration: 260, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" });
  }
  for (const part of [fresh, drawn]) part?.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 320, easing: "ease-out" });
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
    const where = e.url ? element("a", { href: e.url, ...tabOf(e.url), text: e.where || "web page" }) : e.where;
    const quantity = [String(e.what ?? "").replace(/_/g, " "), e.term].filter(Boolean).join(", ");
    return [source ? `${source.short} (${source.year})` : e.source, entry, quantity, String(e.printed ?? ""), where, e.reason];
  });
  return table(["Source", "Entry", "Quantity", "Printed", "Where", "Why the library's value differs"], rows, "data errata__table");
}
