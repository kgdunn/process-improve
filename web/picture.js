// The picture of the settings that can be run, drawn from api_limits' reply (limits.js asks for it
// with the limits). Each pair of numeric factors has a panel in their own units, hatched where no
// setting can be run and shaded where some setting of the other factors meets every limit, with the
// boundary of each straight limit that names just those two, numbered as the list numbers it, the
// center of the ranges and the run sheet's runs. With three numeric factors or more, a panel opens
// to look inside the region: the other factors are held where the reader sets them, and the
// settings that can be run there are drawn exactly, with every straight limit's boundary.

import { isCategorical, listNames } from "./logic.js";

const $ = (sel) => document.querySelector(sel);
const NS = "http://www.w3.org/2000/svg";

/**
 * A panel's plotting square and the room around it for the axes, in the units of its viewBox: a
 * small panel of the matrix, or the large one that looks inside a pair. The text is the same size in
 * both, so the large one draws more square around it.
 */
const SMALL = { left: 44, top: 10, size: 150, right: 12, bottom: 38 };
const LARGE = { left: 48, top: 12, size: 300, right: 14, bottom: 42 };

/** From this many numeric factors the picture can show every pair, or only the pairs the limits name (bootstrap.PICTURE_EVERY_PAIR + 1). */
export const ASK_EVERY_PAIR_FROM = 5;

/** What the page lends the picture: what to do when the reader asks for another view, and the run sheet's runs. */
let hooks = null;

/** The reply the picture was drawn from, and the factors Python read for it. */
let drawn = null;
let factorsDrawn = [];

/** Whether every pair is asked for, and the pair being looked inside, ``{x, y, at}``, or null. */
let everyPair = false;
let inside = null;

/** The runs drawn, as JSON, so a run sheet that has not changed is not drawn again. */
let runsDrawn = null;

/** Each panel names its description by a number of its own. */
let panels = 0;

function el(tag, cls, text = "") {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

function svg(tag, attrs = {}, text = "") {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  if (text) node.textContent = text;
  return node;
}

/** A value as the picture writes it: four significant figures. */
const num = (value) => String(Number(Number(value).toPrecision(4)));

/** A numeric factor's ends, low first, and its units, from the factors Python read. */
function endsOf(name) {
  const f = factorsDrawn.find((factor) => factor.name === name);
  return { low: Math.min(f.low, f.high), high: Math.max(f.low, f.high), units: f.units ?? "" };
}

/** A value with its factor's units: "15 L/min". */
const inUnits = (name, value) => `${num(value)}${endsOf(name).units ? ` ${endsOf(name).units}` : ""}`;

/** Show a part with this text, or hide it when there is none; text that has not changed is left, so it is not read out again. */
function say(node, text) {
  node.hidden = !text;
  if (node.textContent !== text) node.textContent = text;
}

// --------------------------------------------------------------------------- what the page asks and is told
/** What the picture adds to the request: every pair, when asked for, and the pair looked inside, while its factors are numeric. */
export function pictureAsked(factors) {
  const numeric = new Map(factors.filter((f) => !isCategorical(f)).map((f) => [f.name, f]));
  const asked = everyPair ? { every_pair: true } : {};
  const fits = inside && numeric.has(inside.x) && numeric.has(inside.y) && Object.keys(inside.at).every((n) => numeric.has(n));
  if (!fits) return asked;
  const at = Object.fromEntries(
    Object.entries(inside.at).map(([name, value]) => {
      const f = numeric.get(name);
      return [name, Math.min(Math.max(value, Math.min(f.low, f.high)), Math.max(f.low, f.high))]; // a range narrowed since: held at its end
    }),
  );
  return { ...asked, slice: { x: inside.x, y: inside.y, at } };
}

/** Rename factors in the pair looked inside, as ``names`` maps them (old to new), as the limits are renamed. */
export function renameInPicture(names) {
  if (!inside) return;
  const to = (name) => names[name] ?? name;
  inside = { x: to(inside.x), y: to(inside.y), at: Object.fromEntries(Object.entries(inside.at).map(([n, v]) => [to(n), v])) };
  $("#slice-held").dataset.held = ""; // its rows are named afresh
}

/** Stop looking inside: the factors changed in number or kind, so the pair may be gone. */
export function closePicture() {
  inside = null;
  $("#region-slice").hidden = true;
  $("#region-panels").hidden = false;
  $("#region-legend").hidden = false;
}

/** Draw the picture from api_limits' result, for the factors Python read; ``shown`` when there is a limit to picture. */
export function renderPicture(result, factors, shown) {
  drawn = result;
  factorsDrawn = factors;
  const picture = result?.picture ?? null;
  const numeric = factors.filter((f) => !isCategorical(f)).length;
  $("#region").hidden = !shown || !picture || numeric < 2;
  if (!shown && inside) closePicture(); // no limit to picture: the next one opens on every pair
  if ($("#region").hidden) return;
  const every = $("#every-pair");
  every.hidden = numeric < ASK_EVERY_PAIR_FROM;
  every.setAttribute("aria-pressed", String(everyPair));
  every.textContent = "Show every pair of factors";
  renderPanels(picture, numeric);
  renderInside(result);
  say($("#region-key"), keyText(numeric));
  $("#region-legend").hidden = Boolean(inside);
}

function keyText(numeric) {
  if (inside) return "Shaded: what you can run with the other factors held as set here; hatched: what you cannot.";
  return numeric === 2
    ? "Shaded: what you can run; hatched: what you cannot. Each dashed line is a limit, numbered as in the list."
    : "Shaded: where some setting of the other factors can be run. Choose a pair to look inside.";
}

/** Say what the markers are: the center of the ranges, and the runs when there are some. */
function renderLegend(runs) {
  const legend = $("#region-legend");
  const parts = [[el("span", "region__glyph region__glyph--centre", "◇"), " the center of the ranges"]];
  if (runs.length) parts.push([el("span", "region__glyph region__glyph--run", "○"), ` a run of the run sheet (${runs.length})`]);
  legend.replaceChildren(...parts.flatMap((part, i) => (i ? ["; ", ...part] : part)), ".");
}

// --------------------------------------------------------------------------- the panels
function renderPanels(picture, numeric) {
  const holder = $("#region-panels");
  const layout = JSON.stringify([picture.pairs, picture.shown, factorsDrawn]);
  if (holder.dataset.layout === layout) {
    redrawRuns(); // the same picture: only the run sheet may have changed
    return;
  }
  holder.dataset.layout = layout;
  const runs = hooks.runs();
  runsDrawn = JSON.stringify(runs);
  renderLegend(runs);
  holder.classList.toggle("one", picture.pairs.length === 1);
  holder.replaceChildren(
    ...picture.pairs.map((pair) => {
      const region = pair.trace ? { kind: "trace", ...pair.trace } : pair.shadow;
      const bounds = boundaryLines(pair.lines, pair.curves ?? [], pair.x, pair.y);
      const drawing = panel(pair.x, pair.y, region, bounds, runs, SMALL);
      const description = describe(pair.x, pair.y, pair.shadow, bounds, numeric);
      if (numeric === 2) {
        drawing.setAttribute("role", "img");
        drawing.setAttribute("aria-label", description);
        const cell = el("figure", "region__cell");
        cell.append(drawing);
        return cell;
      }
      const id = `region-panel-${++panels}`;
      const pick = el("button", "region__pick");
      pick.type = "button";
      pick.dataset.x = pair.x;
      pick.dataset.y = pair.y;
      drawing.setAttribute("aria-hidden", "true");
      pick.append(drawing, el("span", "sr", `Look inside ${pair.x} and ${pair.y}`));
      pick.setAttribute("aria-describedby", id);
      const cell = el("div", "region__cell");
      cell.append(pick, Object.assign(el("span", "sr", description), { id }));
      return cell;
    }),
  );
  if (!picture.pairs.length) {
    const named = picture.shown.length ? `only ${listNames(picture.shown)}` : "no factor";
    holder.append(el("p", "hint", `No pair of factors to picture: the limits name ${named}.`));
  }
}

/** The run sheet changed, or its plan did: draw its runs again in each panel. */
export function redrawRuns() {
  if (!hooks || !drawn?.picture || $("#region").hidden) return;
  const runs = hooks.runs();
  if (JSON.stringify(runs) === runsDrawn) return;
  runsDrawn = JSON.stringify(runs);
  renderLegend(runs);
  for (const drawing of document.querySelectorAll("#region-panels .region__panel")) {
    const box = drawing.dataset.size === "large" ? LARGE : SMALL;
    drawing.querySelector(".region__runs").replaceWith(runsLayer(drawing.dataset.x, drawing.dataset.y, runs, box));
  }
}

/**
 * Each limit's boundary over two factors, as lines in the factors' units: a straight limit's one
 * segment, and a curved limit's boundary traced on its own, its pieces joined into lines.
 */
function boundaryLines(lines, curves, x, y) {
  const ex = endsOf(x);
  const ey = endsOf(y);
  return [
    ...lines.map((line) => ({ limit: line.limit, curved: false, lines: [line.points] })),
    ...curves.map((curve) => ({
      limit: curve.limit,
      curved: true,
      lines: chains(traced(curve.trace).edge, [ex.low, ex.high], [ey.low, ey.high]),
    })),
  ].filter((bound) => bound.lines.length);
}

/**
 * One panel: the square of two factors' ranges, hatched where nothing can be run, the settings that
 * can be run shaded over it, each limit's boundary with its number, the center of the ranges, and
 * the runs given.
 */
function panel(x, y, region, bounds, runs, box, { centred = true } = {}) {
  const sx = scale(x, box.left, box.left + box.size);
  const sy = scale(y, box.top + box.size, box.top);
  const width = box.left + box.size + box.right;
  const height = box.top + box.size + box.bottom;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "region__panel" });
  Object.assign(root.dataset, { x, y, size: box === LARGE ? "large" : "small" });
  const square = { x: box.left, y: box.top, width: box.size, height: box.size };
  root.append(svg("rect", { ...square, class: "region__out" }));
  root.append(allowed(region, sx, sy, x, y));
  const onPanel = bounds.map((bound) => ({ ...bound, lines: bound.lines.map((points) => points.map(([px, py]) => [sx(px), sy(py)])) }));
  for (const bound of onPanel) {
    root.append(svg("path", { d: bound.lines.map((points) => `M${points.map((pt) => pt.join(",")).join("L")}`).join(""), class: "region__limit" }));
  }
  root.append(runsLayer(x, y, runs, box));
  const centre = [sx((endsOf(x).low + endsOf(x).high) / 2), sy((endsOf(y).low + endsOf(y).high) / 2)];
  if (centred) root.append(svg("path", { d: diamond(centre[0], centre[1], 6.5), class: "region__centre" })); // over a run there
  root.append(svg("rect", { ...square, class: "region__frame" }));
  root.append(...axes(x, y, box));
  root.append(...tags(onPanel, centred ? centre : null));
  return root;
}

/** A factor's values onto the panel, from its low end at ``from`` to its high end at ``to``. */
function scale(name, from, to) {
  const { low, high } = endsOf(name);
  return (value) => from + ((value - low) / (high - low || 1)) * (to - from);
}

const pathOf = (polygons) => polygons.map((points) => `M${points.map(([px, py]) => `${px},${py}`).join("L")}Z`).join("");

/**
 * The settings that can be run: the exact polygon of a straight region; a curved one traced through
 * the points where its boundary crosses the lines of a grid; or the cells of a sampled one. Each is
 * one path, so cells that meet show no seam between them.
 */
function allowed(region, sx, sy, x, y) {
  const group = svg("g", { class: "region__in" });
  if (region.kind === "trace") {
    const { fill, edge } = traced(region);
    const at = ([px, py]) => [sx(px), sy(py)];
    group.append(svg("path", { d: pathOf(fill.map((points) => points.map(at))), class: "region__allowed region__fill" }));
    const d = edge.map(([a, b]) => `M${at(a).join(",")}L${at(b).join(",")}`).join("");
    group.append(svg("path", { d, class: "region__edge" }));
    return group;
  }
  if (region.kind === "mask") {
    const xs = edges(region.x, endsOf(x));
    const ys = edges(region.y, endsOf(y));
    const cells = [];
    region.z.forEach((row, i) => {
      let start = null;
      row.forEach((cell, j) => {
        if (cell && start === null) start = j;
        if (start !== null && (!cell || j === row.length - 1)) {
          const end = cell ? j : j - 1;
          const [x0, x1, y0, y1] = [sx(xs[start]), sx(xs[end + 1]), sy(ys[i]), sy(ys[i + 1])];
          cells.push([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
          ]);
          start = null;
        }
      });
    });
    group.append(svg("path", { d: pathOf(cells), class: "region__allowed region__fill" }));
    return group;
  }
  const points = region.points.map(([vx, vy]) => [sx(vx), sy(vy)]);
  if (points.length >= 3) group.append(svg("path", { d: pathOf([points]), class: "region__allowed" }));
  if (points.length === 2) group.append(svg("path", { d: `M${points[0].join(",")}L${points[1].join(",")}`, class: "region__edge region__edge--only" }));
  if (points.length === 1) group.append(svg("circle", { cx: points[0][0], cy: points[0][1], r: 3, class: "region__dot" }));
  return group;
}

/** The edges of the cells around sampled values: halfway between neighbours, and the range's ends outside. */
function edges(values, { low, high }) {
  return [low, ...values.slice(1).map((v, i) => (values[i] + v) / 2), high];
}

/**
 * The settings that can be run on a grid, with the boundary through the points where it crosses the
 * grid's lines (api_limits' ``trace``): each cell's part that can be run, walking its corners
 * counter-clockwise and cutting each side where the boundary crosses it, and the boundary as the
 * cuts each part joins.
 */
export function traced({ x, y, z, crossings }) {
  const cut = new Map(crossings.map(([i, j, along, t]) => [`${i} ${j} ${along}`, t]));
  const fill = [];
  const edge = [];
  for (let i = 0; i + 1 < y.length; i++) {
    let run = null; // cells wholly inside, side by side along the row, are one rectangle
    const close = () => {
      if (run) fill.push([[x[run[0]], y[i]], [x[run[1]], y[i]], [x[run[1]], y[i + 1]], [x[run[0]], y[i + 1]]]);
      run = null;
    };
    for (let j = 0; j + 1 < x.length; j++) {
      const corners = [
        [i, j],
        [i, j + 1],
        [i + 1, j + 1],
        [i + 1, j],
      ];
      const ins = corners.map(([ci, cj]) => z[ci][cj] === 1);
      if (ins.every(Boolean)) {
        run = run ? [run[0], j + 1] : [j, j + 1];
        continue;
      }
      close();
      if (!ins.some(Boolean)) continue;
      // The four sides from the first corner to the next: the grid line each lies on, and which way it runs.
      const sides = [
        [`${i} ${j} 0`, false],
        [`${i} ${j + 1} 1`, false],
        [`${i + 1} ${j} 0`, true],
        [`${i} ${j} 1`, true],
      ];
      const points = [];
      corners.forEach(([ci, cj], k) => {
        const [ni, nj] = corners[(k + 1) % 4];
        if (ins[k]) points.push({ at: [x[cj], y[ci]], cut: false });
        if (ins[k] !== ins[(k + 1) % 4]) {
          const [key, backwards] = sides[k];
          const along = cut.get(key) ?? 0.5;
          const t = backwards ? 1 - along : along;
          points.push({ at: [x[cj] + t * (x[nj] - x[cj]), y[ci] + t * (y[ni] - y[ci])], cut: true });
        }
      });
      fill.push(points.map((p) => p.at));
      points.forEach((p, k) => {
        const next = points[(k + 1) % points.length];
        if (p.cut && next.cut) edge.push([p.at, next.at]);
      });
    }
    close();
  }
  return { fill, edge };
}

/**
 * Join the pieces of a traced boundary into lines, each from one end to the other, or once round a
 * loop. Two pieces join where they share an end, to a millionth of each factor's range (``xs`` and
 * ``ys`` are the ranges' ends), so a dashed boundary is dashed evenly along its whole length.
 */
export function chains(edge, [x0, x1], [y0, y1]) {
  const key = ([px, py]) => `${Math.round(((px - x0) / (x1 - x0 || 1)) * 1e6)} ${Math.round(((py - y0) / (y1 - y0 || 1)) * 1e6)}`;
  const pieces = edge.filter(([a, b]) => key(a) !== key(b));
  const ends = new Map();
  pieces.forEach(([a, b], n) => {
    for (const [point, end] of [
      [a, 0],
      [b, 1],
    ]) {
      ends.set(key(point), [...(ends.get(key(point)) ?? []), [n, end]]);
    }
  });
  const used = new Set();
  const walk = (n, end) => {
    used.add(n);
    const line = [pieces[n][end], pieces[n][1 - end]];
    for (;;) {
      const next = (ends.get(key(line.at(-1))) ?? []).find(([m]) => !used.has(m));
      if (!next) return line;
      used.add(next[0]);
      line.push(pieces[next[0]][1 - next[1]]);
    }
  };
  const alone = (point) => ends.get(key(point)).length === 1;
  const lines = [];
  pieces.forEach(([a, b], n) => {
    if (!used.has(n) && (alone(a) || alone(b))) lines.push(walk(n, alone(a) ? 0 : 1)); // from an open end
  });
  pieces.forEach((_, n) => {
    if (!used.has(n)) lines.push(walk(n, 0)); // a loop
  });
  return lines;
}

/** The distance from a point to a segment, in the panel's units. */
function distance([px, py], [[ax, ay], [bx, by]]) {
  const [dx, dy] = [bx - ax, by - ay];
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** The pieces of a line, each from one point to the next. */
const piecesOf = (points) => points.slice(1).map((point, k) => [points[k], point]);

/** The length of a line. */
const lengthOf = (points) => piecesOf(points).reduce((sum, [a, b]) => sum + Math.hypot(b[0] - a[0], b[1] - a[1]), 0);

/** The point a share ``t`` of the way along a line. */
function along(points, t) {
  let left = t * lengthOf(points);
  for (const [a, b] of piecesOf(points)) {
    const step = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (left <= step) return [a[0] + (left / (step || 1)) * (b[0] - a[0]), a[1] + (left / (step || 1)) * (b[1] - a[1])];
    left -= step;
  }
  return points.at(-1);
}

/**
 * Each limit's number, on its boundary, where it sits clear of the other boundaries, the numbers
 * already placed and the center's marker: halfway along its longest line when it can, else nearer
 * an end.
 */
function tags(bounds, centre) {
  const placed = [];
  return bounds.map((bound, n) => {
    const longest = bound.lines.reduce((best, points) => (lengthOf(points) > lengthOf(best) ? points : best));
    const spots = [0.5, 0.32, 0.68, 0.18, 0.82].map((t) => along(longest, t));
    const clear = (spot) =>
      bounds.every((other, m) => m === n || other.lines.every((points) => piecesOf(points).every((piece) => distance(spot, piece) >= 10))) &&
      placed.every((other) => Math.hypot(spot[0] - other[0], spot[1] - other[1]) >= 18) &&
      (!centre || Math.hypot(spot[0] - centre[0], spot[1] - centre[1]) >= 12);
    const [cx, cy] = spots.find(clear) ?? spots[0];
    placed.push([cx, cy]);
    const group = svg("g", { class: "region__tag" });
    group.append(svg("circle", { cx, cy, r: 8 }), svg("text", { x: cx, y: cy }, String(bound.limit + 1)));
    return group;
  });
}

const diamond = (cx, cy, r) => `M${cx},${cy - r}L${cx + r},${cy}L${cx},${cy + r}L${cx - r},${cy}Z`;

function runsLayer(x, y, runs, box) {
  const group = svg("g", { class: "region__runs" });
  const sx = scale(x, box.left, box.left + box.size);
  const sy = scale(y, box.top + box.size, box.top);
  for (const run of runs) {
    if (Number.isFinite(run[x]) && Number.isFinite(run[y])) group.append(svg("circle", { cx: sx(run[x]), cy: sy(run[y]), r: 3.2, class: "region__run" }));
  }
  return group;
}

/** The ends of each axis, and its name with its units. */
function axes(x, y, box) {
  const ex = endsOf(x);
  const ey = endsOf(y);
  const bottom = box.top + box.size;
  const right = box.left + box.size;
  const middle = box.top + box.size / 2;
  const named = (name, units) => (units ? `${name} (${units})` : name);
  return [
    svg("text", { x: box.left, y: bottom + 14, class: "region__tick", "text-anchor": "start" }, num(ex.low)),
    svg("text", { x: right, y: bottom + 14, class: "region__tick", "text-anchor": "end" }, num(ex.high)),
    svg("text", { x: box.left - 5, y: bottom, class: "region__tick", "text-anchor": "end" }, num(ey.low)),
    svg("text", { x: box.left - 5, y: box.top + 10, class: "region__tick", "text-anchor": "end" }, num(ey.high)),
    svg("text", { x: box.left + box.size / 2, y: bottom + 31, class: "region__axis", "text-anchor": "middle" }, named(x, ex.units)),
    svg("text", { x: 14, y: middle, class: "region__axis", "text-anchor": "middle", transform: `rotate(-90 14 ${middle})` }, named(y, ey.units)),
  ];
}

/** What a panel shows, for a reader who cannot see it: the settings that can be run, and where each boundary crosses. */
function describe(x, y, region, bounds, numeric, held = null) {
  const at = (point) => `${x} ${inUnits(x, point[0])}, ${y} ${inUnits(y, point[1])}`;
  const whose = held
    ? `with ${listNames(Object.entries(held).map(([name, value]) => `${name} at ${inUnits(name, value)}`))}`
    : numeric > 2
      ? "for some setting of the other factors"
      : "";
  const none = region.kind === "polygon" ? region.points.length === 0 : !region.z.some((row) => row.some(Boolean));
  const head = `${x} against ${y}${whose ? `, ${whose}` : ""}: `;
  if (none) return `${head}no setting can be run.`;
  const shape =
    region.kind === "polygon" && region.points.length >= 3
      ? `the settings that can be run have the corners ${region.points.map(at).join("; ")}`
      : region.kind === "polygon"
        ? `only ${region.points.map(at).join(" to ")} can be run`
        : "the settings that can be run are shaded, inside a curved limit";
  const crossings = bounds.map((bound) => {
    const longest = bound.lines.reduce((best, points) => (points.length > best.length ? points : best));
    const how = bound.curved ? "curves" : "crosses";
    return `limit ${bound.limit + 1} ${how} from ${at(longest[0])} to ${at(longest.at(-1))}`;
  });
  return `${head}${[shape, ...crossings].join("; ")}.`;
}

// --------------------------------------------------------------------------- looking inside a pair
/** Start looking inside a pair of factors, the others held where the picture puts them: the center, or a central setting that can be run. */
function lookInside(x, y) {
  const centre = drawn.picture.center ?? {};
  const others = factorsDrawn.filter((f) => !isCategorical(f) && f.name !== x && f.name !== y);
  inside = { x, y, at: Object.fromEntries(others.map((f) => [f.name, Number(num(centre[f.name] ?? (f.low + f.high) / 2))])) };
  $("#region-panels").hidden = true;
  $("#region-legend").hidden = true;
  $("#region-slice").hidden = false;
  $("#slice-panel").replaceChildren();
  say($("#region-key"), keyText(0));
  renderHeld();
  $("#slice-held input")?.focus();
  hooks.changed();
}

function closeInside() {
  const { x, y } = inside ?? {};
  closePicture();
  $(`#region-panels .region__pick[data-x="${x}"][data-y="${y}"]`)?.focus();
  hooks.changed();
}

/** A slider and a number box for each factor held, as the reader set them. */
function renderHeld() {
  const holder = $("#slice-held");
  const wanted = JSON.stringify([inside.x, inside.y, Object.keys(inside.at)]);
  $("#h-slice").textContent = `${inside.x} and ${inside.y}, with the other factors held at:`;
  if (holder.dataset.held === wanted) return;
  holder.dataset.held = wanted;
  holder.replaceChildren(
    ...Object.entries(inside.at).map(([name, value]) => {
      const { low, high, units } = endsOf(name);
      const row = el("div", "held");
      row.dataset.name = name;
      const slider = el("input");
      Object.assign(slider, { type: "range", name: `held-${name}`, min: String(low), max: String(high), step: String((high - low) / 100) });
      slider.value = String(value);
      slider.setAttribute("aria-label", `${name}, held at`);
      const box = el("input");
      Object.assign(box, { type: "number", name: `held-box-${name}`, min: String(low), max: String(high), step: "any" });
      box.value = String(value);
      box.setAttribute("aria-label", `${name}, held at${units ? `, in ${units}` : ""}`);
      row.append(el("span", "held__name", name), slider, box);
      if (units) row.append(el("span", "held__units", units));
      return row;
    }),
  );
}

/** Draw the settings that can be run inside the pair, at the values the reply holds the others at. */
function renderInside(result) {
  if (!inside) {
    closePicture();
    return;
  }
  const slice = result.slice;
  if (!slice || slice.x !== inside.x || slice.y !== inside.y) return; // the reply is for another view: the next one draws it
  renderHeld();
  const region = slice.trace ? { kind: "trace", ...slice.trace } : slice.region;
  const bounds = boundaryLines(slice.lines, slice.curves ?? [], slice.x, slice.y);
  const drawing = panel(slice.x, slice.y, region, bounds, [], LARGE, { centred: false });
  drawing.setAttribute("role", "img");
  drawing.setAttribute("aria-label", describe(slice.x, slice.y, slice.region, bounds, 3, slice.at));
  $("#slice-panel").replaceChildren(drawing);
}

/** The reader moved a slider, or typed in its box: hold the factor there, and ask for the slice again. */
function heldChanged(target) {
  const row = target.closest(".held");
  if (!row || !inside) return;
  const value = Number(target.value);
  if (target.type === "number" && (target.validity.badInput || target.value.trim() === "" || !Number.isFinite(value))) return;
  const other = row.querySelector(target.type === "range" ? 'input[type="number"]' : 'input[type="range"]');
  other.value = String(value);
  inside = { ...inside, at: { ...inside.at, [row.dataset.name]: value } };
  hooks.changed();
}

/** Wire the picture to the panel: ``changed`` asks Python again, ``runs`` gives the run sheet's runs, by factor. */
export function installPicture({ changed, runs }) {
  hooks = { changed, runs };
  $("#region-panels").addEventListener("click", (event) => {
    const pick = event.target.closest(".region__pick");
    if (pick && drawn?.picture) lookInside(pick.dataset.x, pick.dataset.y);
  });
  $("#every-pair").onclick = () => {
    everyPair = !everyPair;
    $("#every-pair").setAttribute("aria-pressed", String(everyPair));
    changed();
  };
  $("#slice-close").onclick = closeInside;
  $("#slice-held").addEventListener("input", (event) => heldChanged(event.target));
}
