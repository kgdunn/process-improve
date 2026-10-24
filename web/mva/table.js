// The data table: a canvas-drawn grid that renders only the cells in view, so 100,000 rows or 1,000 columns
// scroll smoothly. Each column's header draws its histogram, its values in run order and where it is missing,
// and the rows selected anywhere on the page are overlaid on all three: lasso a cluster on a score plot and
// every header shows where those rows sit in that variable.
//
// The functions before DataTable decide what to draw and touch nothing, so tests/web/mva runs them under Node.
// A canvas is invisible to a screen reader, so the grid is also a focusable ARIA grid: arrow keys move the
// active cell, Space selects its row, and a live region reads the cell out.

/** Heights, in CSS pixels, of a body row and of the header's four bands. */
export const ROW_HEIGHT = 22;
export const HEADER = { title: 18, histogram: 34, sequence: 24, missing: 6, gap: 4 };
export const HEADER_HEIGHT = HEADER.title + HEADER.histogram + HEADER.sequence + HEADER.missing + 4 * HEADER.gap;

/** Widths, in CSS pixels, of the row-id column and of a value column. */
export const ID_WIDTH = 84;
export const COLUMN_WIDTH = 112;

/** Bars in a header's histogram, and the most points its run-order line keeps (a min-max envelope beyond). */
export const BINS = 12;
export const SEQUENCE_POINTS = 120;

/** The categories a text column's header shows before "others". */
export const TOP_CATEGORIES = 5;

/** A missing cell: null, undefined, NaN or the empty text. */
export const isMissing = (v) => v === null || v === undefined || v === "" || (typeof v === "number" && Number.isNaN(v));

/** A cell as text: numbers to six significant figures, without a trailing zero; missing cells blank. */
export function formatCell(value) {
  if (isMissing(value)) return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value) && Math.abs(value) < 1e15) return String(value);
    const text =
      Math.abs(value) >= 1e-4 && Math.abs(value) < 1e9 ? String(Number(value.toPrecision(6))) : value.toExponential(4);
    return text;
  }
  return String(value);
}

/**
 * What a column's header draws, from its values (in table order): its kind, counts of missing and of text in a
 * number column, the histogram (or the top categories), the run-order line, the missing strip, and its flags.
 */
export function columnProfile(values, { bins = BINS, points = SEQUENCE_POINTS } = {}) {
  const n = values.length;
  const missing = values.filter(isMissing).length;
  const numbers = values.filter((v) => typeof v === "number" && Number.isFinite(v));
  const present = n - missing;
  const kind = present > 0 && numbers.length === present ? "number" : "text";
  const flags = [];
  if (missing) flags.push(`${((100 * missing) / n).toPrecision(2)}% missing`);
  const profile = { kind, n, missing, flags, strip: missingStrip(values, points) };
  if (kind === "number") {
    const min = Math.min(...numbers);
    const max = Math.max(...numbers);
    if (min === max) flags.push("constant");
    return { ...profile, min, max, histogram: histogram(numbers, min, max, bins), sequence: envelope(values, points) };
  }
  const counts = new Map();
  for (const v of values) if (!isMissing(v)) counts.set(String(v), (counts.get(String(v)) ?? 0) + 1);
  const ranked = [...counts].sort((a, b) => b[1] - a[1]);
  const top = ranked.slice(0, TOP_CATEGORIES).map(([name, count]) => ({ name, count }));
  const others = ranked.slice(TOP_CATEGORIES).reduce((sum, [, c]) => sum + c, 0);
  if (others) top.push({ name: "others", count: others });
  if (numbers.length) flags.push(`${present - numbers.length} not numbers`);
  return { ...profile, categories: top, distinct: counts.size };
}

/** Counts of `values` in `bins` equal bins from min to max; the last bin holds the maximum. */
export function histogram(values, min, max, bins = BINS) {
  const counts = new Array(bins).fill(0);
  const width = (max - min) / bins;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const k = width > 0 ? Math.min(bins - 1, Math.floor((v - min) / width)) : Math.floor(bins / 2);
    if (k >= 0) counts[k] += 1;
  }
  return counts;
}

/**
 * The run-order line of a column, at most `points` wide: each point is { at, lo, hi } over a bucket of rows, so
 * a long column keeps its spikes (the bucket's minimum and maximum) rather than losing them to a thinning.
 * `at` is the bucket's first row; a bucket of missing values only is { at, lo: null, hi: null }.
 */
export function envelope(values, points = SEQUENCE_POINTS) {
  const n = values.length;
  const per = Math.max(1, Math.ceil(n / points));
  const out = [];
  for (let at = 0; at < n; at += per) {
    let lo = null;
    let hi = null;
    for (let i = at; i < Math.min(n, at + per); i += 1) {
      const v = values[i];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      lo = lo === null ? v : Math.min(lo, v);
      hi = hi === null ? v : Math.max(hi, v);
    }
    out.push({ at, lo, hi });
  }
  return out;
}

/** Where a column is missing, as the fraction missing in each of at most `points` buckets of rows. */
export function missingStrip(values, points = SEQUENCE_POINTS) {
  const n = values.length;
  const per = Math.max(1, Math.ceil(n / points));
  const out = [];
  for (let at = 0; at < n; at += per) {
    const bucket = values.slice(at, at + per);
    out.push(bucket.filter(isMissing).length / bucket.length);
  }
  return out;
}

/** The rows in view, with `overscan` more each side: { first, last }, last exclusive. */
export function visibleRows(scrollTop, height, count, { rowHeight = ROW_HEIGHT, overscan = 4 } = {}) {
  const first = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const last = Math.min(count, Math.ceil((scrollTop + height) / rowHeight) + overscan);
  return { first, last: Math.max(first, last) };
}

/** The value columns in view after the frozen id column: { first, last, offsets }, offsets[k] the left edge of column k. */
export function visibleColumns(scrollLeft, width, widths, { idWidth = ID_WIDTH } = {}) {
  const offsets = [];
  let x = idWidth;
  for (const w of widths) {
    offsets.push(x);
    x += w;
  }
  const left = scrollLeft + idWidth;
  const right = scrollLeft + width;
  let first = offsets.findIndex((o, k) => o + widths[k] > left);
  if (first < 0) first = widths.length;
  let last = first;
  while (last < widths.length && offsets[last] < right) last += 1;
  return { first, last, offsets, total: x };
}

/**
 * What lies under a point of the grid, in the grid's own pixels (scrolling already added): { kind: "header" |
 * "cell" | "id", row, column } or null. `column` is an index into the value columns.
 */
export function hitTest(x, y, { offsets, widths, count, idWidth = ID_WIDTH, rowHeight = ROW_HEIGHT, scrollLeft = 0 }) {
  const row = y < HEADER_HEIGHT ? null : Math.floor((y - HEADER_HEIGHT) / rowHeight);
  if (row !== null && row >= count) return null;
  if (x - scrollLeft < idWidth) return { kind: row === null ? "header" : "id", row, column: null };
  const column = offsets.findIndex((o, k) => x >= o && x < o + widths[k]);
  if (column < 0) return null;
  return { kind: row === null ? "header" : "cell", row, column };
}

/** The ids of the rows from `anchor` to `row`, inclusive, in table order: what a Shift-click selects. */
export function rowSpan(ids, anchor, row) {
  const [a, b] = anchor <= row ? [anchor, row] : [row, anchor];
  return ids.slice(a, b + 1);
}

/** The sentence the live region reads for the active cell. */
export function cellSentence(rowId, column, value, { selected = false, excluded = false } = {}) {
  const shown = formatCell(value) || "missing";
  const notes = [selected ? "selected" : "", excluded ? "excluded from the model" : ""].filter(Boolean);
  return `Row ${rowId}, ${column}: ${shown}${notes.length ? `, ${notes.join(", ")}` : ""}.`;
}

/**
 * The grid on the page. `columns` is [{ id, title?, values }], every values list in the order of `rowIds`.
 * The grid reads and writes `selection`: rows (click, Shift-click, Ctrl or Cmd-click, drag, Space) and columns
 * (a header click). Returns nothing until `destroy()`.
 */
export class DataTable {
  #root;
  #scroller;
  #canvas;
  #live;
  #selection;
  #rowIds;
  #columns;
  #profiles;
  #widths;
  #colours;
  #unsubscribe;
  #active = { row: 0, column: 0 };
  #anchor = 0;
  #dragging = false;
  #frame = null;

  constructor(root, { rowIds, columns, selection, colours }) {
    this.#root = root;
    this.#selection = selection;
    this.#rowIds = rowIds.map(String);
    this.#columns = columns;
    this.#profiles = columns.map((c) => columnProfile(c.values));
    this.#widths = columns.map(() => COLUMN_WIDTH);
    this.#colours = colours ?? readColours(root);

    const doc = root.ownerDocument;
    root.classList.add("mva-table");
    root.setAttribute("role", "grid");
    root.setAttribute("aria-rowcount", String(this.#rowIds.length + 1));
    root.setAttribute("aria-colcount", String(columns.length + 1));
    root.setAttribute("aria-label", "Data table");
    root.tabIndex = 0;
    this.#scroller = doc.createElement("div");
    this.#scroller.className = "mva-table-scroller";
    const spacer = doc.createElement("div");
    spacer.className = "mva-table-spacer";
    this.#canvas = doc.createElement("canvas");
    this.#canvas.setAttribute("aria-hidden", "true");
    this.#live = doc.createElement("div");
    this.#live.className = "visually-hidden";
    this.#live.setAttribute("aria-live", "polite");
    this.#scroller.append(spacer);
    root.append(this.#canvas, this.#scroller, this.#live);
    const { total } = visibleColumns(0, 0, this.#widths);
    spacer.style.width = `${total}px`;
    spacer.style.height = `${HEADER_HEIGHT + this.#rowIds.length * ROW_HEIGHT}px`;

    this.#scroller.addEventListener("scroll", () => this.redraw());
    this.#scroller.addEventListener("pointerdown", (e) => this.#pointerDown(e));
    this.#scroller.addEventListener("pointermove", (e) => this.#pointerMove(e));
    this.#scroller.addEventListener("pointerleave", () => this.#hover(null));
    doc.addEventListener("pointerup", (this.#release = () => (this.#dragging = false)));
    root.addEventListener("keydown", (e) => this.#key(e));
    this.#resizer = new ResizeObserver(() => this.redraw());
    this.#resizer.observe(root);
    this.#unsubscribe = selection.subscribe(() => this.redraw());
    this.redraw();
  }

  #release;
  #resizer;

  /** Draw on the next animation frame; several calls in one frame draw once. */
  redraw() {
    if (this.#frame !== null) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = null;
      this.#draw();
    });
  }

  destroy() {
    this.#unsubscribe();
    this.#resizer.disconnect();
    this.#root.ownerDocument.removeEventListener("pointerup", this.#release);
    if (this.#frame !== null) cancelAnimationFrame(this.#frame);
    this.#root.replaceChildren();
  }

  #point(event) {
    const box = this.#scroller.getBoundingClientRect();
    return {
      x: event.clientX - box.left + this.#scroller.scrollLeft,
      y: event.clientY - box.top + (event.clientY - box.top < HEADER_HEIGHT ? 0 : this.#scroller.scrollTop),
    };
  }

  #hit(event) {
    const { offsets } = visibleColumns(0, 0, this.#widths);
    const { x, y } = this.#point(event);
    return hitTest(x, y, {
      offsets,
      widths: this.#widths,
      count: this.#rowIds.length,
      scrollLeft: this.#scroller.scrollLeft,
    });
  }

  #pointerDown(event) {
    const hit = this.#hit(event);
    if (!hit) return;
    this.#root.focus({ preventScroll: true });
    if (hit.kind === "header") {
      if (hit.column === null) return this.#selection.clear("cols");
      const id = this.#columns[hit.column].id;
      return this.#selection.select(
        "cols",
        [id],
        event.shiftKey || event.ctrlKey || event.metaKey ? "toggle" : "replace",
      );
    }
    this.#active = { row: hit.row, column: hit.column ?? this.#active.column };
    if (event.shiftKey) {
      this.#selection.select("rows", rowSpan(this.#rowIds, this.#anchor, hit.row), "replace");
    } else {
      this.#anchor = hit.row;
      this.#selection.select("rows", [this.#rowIds[hit.row]], event.ctrlKey || event.metaKey ? "toggle" : "replace");
      this.#dragging = !(event.ctrlKey || event.metaKey);
    }
    this.#announce();
  }

  #pointerMove(event) {
    const hit = this.#hit(event);
    if (this.#dragging && hit?.row !== null && hit?.row !== undefined) {
      this.#selection.select("rows", rowSpan(this.#rowIds, this.#anchor, hit.row), "replace");
    }
    this.#hover(hit);
  }

  #hover(hit) {
    this.#selection.hover("rows", hit?.row === null || hit?.row === undefined ? null : this.#rowIds[hit.row]);
    this.#selection.hover(
      "cols",
      hit?.column === null || hit?.column === undefined ? null : this.#columns[hit.column].id,
    );
  }

  #key(event) {
    const moves = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
      PageUp: [-20, 0],
      PageDown: [20, 0],
    };
    if (event.key in moves) {
      const [dr, dc] = moves[event.key];
      this.#active = {
        row: Math.min(this.#rowIds.length - 1, Math.max(0, this.#active.row + dr)),
        column: Math.min(this.#columns.length - 1, Math.max(0, this.#active.column + dc)),
      };
      if (event.shiftKey && dr) this.#selection.select("rows", rowSpan(this.#rowIds, this.#anchor, this.#active.row));
      this.#scrollIntoView();
    } else if (event.key === " ") {
      const id = this.#rowIds[this.#active.row];
      this.#anchor = this.#active.row;
      this.#selection.select("rows", [id], event.shiftKey ? "add" : "toggle");
    } else if (event.key === "Escape") {
      this.#selection.clear();
    } else {
      return;
    }
    event.preventDefault();
    this.#announce();
    this.redraw();
  }

  #scrollIntoView() {
    const top = this.#active.row * ROW_HEIGHT;
    const view = this.#scroller.clientHeight - HEADER_HEIGHT;
    if (top < this.#scroller.scrollTop) this.#scroller.scrollTop = top;
    else if (top + ROW_HEIGHT > this.#scroller.scrollTop + view) this.#scroller.scrollTop = top + ROW_HEIGHT - view;
  }

  #announce() {
    const { row, column } = this.#active;
    const id = this.#rowIds[row];
    const col = this.#columns[column];
    this.#live.textContent = cellSentence(id, col.title ?? col.id, col.values[row], {
      selected: this.#selection.has("rows", id),
      excluded: this.#selection.isExcluded(id),
    });
  }

  #draw() {
    const width = this.#scroller.clientWidth;
    const height = this.#scroller.clientHeight;
    const ratio = globalThis.devicePixelRatio || 1;
    this.#canvas.width = Math.round(width * ratio);
    this.#canvas.height = Math.round(height * ratio);
    this.#canvas.style.width = `${width}px`;
    this.#canvas.style.height = `${height}px`;
    const ctx = this.#canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const { scrollTop, scrollLeft } = this.#scroller;
    const rows = visibleRows(scrollTop, height - HEADER_HEIGHT, this.#rowIds.length);
    const cols = visibleColumns(scrollLeft, width, this.#widths);
    const selectedRows = this.#selection.selected("rows");
    const selectedCols = this.#selection.selected("cols");
    const state = {
      selectedRows,
      selectedCols,
      hoveredRow: this.#selection.hovered("rows"),
      hoveredCol: this.#selection.hovered("cols"),
    };
    const c = this.#colours;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    // Body rows, under the header; the frozen id column last, over the scrolled cells.
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, HEADER_HEIGHT, width, height - HEADER_HEIGHT);
    ctx.clip();
    for (let r = rows.first; r < rows.last; r += 1) {
      const y = HEADER_HEIGHT + r * ROW_HEIGHT - scrollTop;
      const id = this.#rowIds[r];
      const excluded = this.#selection.isExcluded(id);
      if (selectedRows.has(id)) fill(ctx, c.selection, 0, y, width, ROW_HEIGHT);
      else if (r % 2) fill(ctx, c.stripe, 0, y, width, ROW_HEIGHT);
      for (let k = cols.first; k < cols.last; k += 1) {
        const x = cols.offsets[k] - scrollLeft;
        if (selectedCols.has(this.#columns[k].id)) fill(ctx, c.selection, x, y, this.#widths[k], ROW_HEIGHT);
        const text = formatCell(this.#columns[k].values[r]);
        ctx.fillStyle = excluded ? c.muted : c.fg;
        ctx.textAlign = "right";
        ctx.fillText(clip(ctx, text, this.#widths[k] - 10), x + this.#widths[k] - 6, y + ROW_HEIGHT / 2);
      }
      if (excluded) stroke(ctx, c.muted, 0, y + ROW_HEIGHT / 2, cols.total - scrollLeft, y + ROW_HEIGHT / 2);
      fill(ctx, selectedRows.has(id) ? c.selection : c.bg, 0, y, ID_WIDTH, ROW_HEIGHT);
      ctx.fillStyle = excluded ? c.muted : c.fg;
      ctx.textAlign = "left";
      ctx.fillText(clip(ctx, id, ID_WIDTH - 10), 6, y + ROW_HEIGHT / 2);
      if (state.hoveredRow === id) outline(ctx, c.fg, 0.5, y + 0.5, width - 1, ROW_HEIGHT - 1);
      if (r === this.#active.row && this.#root.ownerDocument.activeElement === this.#root) {
        const x = cols.offsets[this.#active.column] - scrollLeft;
        outline(ctx, c.accent, x + 1, y + 1, this.#widths[this.#active.column] - 2, ROW_HEIGHT - 2, 2);
      }
    }
    ctx.restore();

    // Headers: title, histogram (with the selected rows' over it), run-order line (selected rows marked), missing strip.
    fill(ctx, c.bg, 0, 0, width, HEADER_HEIGHT);
    for (let k = cols.first; k < cols.last; k += 1) {
      const x = cols.offsets[k] - scrollLeft;
      this.#drawHeader(ctx, k, x, state);
    }
    fill(ctx, c.bg, 0, 0, ID_WIDTH, HEADER_HEIGHT);
    ctx.fillStyle = c.muted;
    ctx.textAlign = "left";
    ctx.fillText(`${this.#rowIds.length} rows`, 6, HEADER.title / 2 + 2);
    if (selectedRows.size) ctx.fillText(`${selectedRows.size} selected`, 6, HEADER.title + 16);
    stroke(ctx, c.line, 0, HEADER_HEIGHT - 0.5, width, HEADER_HEIGHT - 0.5);
    stroke(ctx, c.line, ID_WIDTH - 0.5, 0, ID_WIDTH - 0.5, height);
  }

  #drawHeader(ctx, k, x, state) {
    const c = this.#colours;
    const column = this.#columns[k];
    const profile = this.#profiles[k];
    const w = this.#widths[k];
    const selected = state.selectedCols.has(column.id);
    if (selected) fill(ctx, c.selection, x, 0, w, HEADER_HEIGHT);
    ctx.fillStyle = c.fg;
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(
      clip(ctx, column.title ?? column.id, w - (profile.flags.length ? 20 : 8)),
      x + 4,
      HEADER.title / 2 + 2,
    );
    ctx.font = "12px system-ui, sans-serif";
    if (profile.flags.length) {
      ctx.fillStyle = c.warning;
      ctx.beginPath();
      ctx.arc(x + w - 9, HEADER.title / 2 + 2, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    let y = HEADER.title + HEADER.gap;
    const inner = { x: x + 4, w: w - 8 };
    const rows = [...state.selectedRows].map((id) => this.#rowIndex(id)).filter((r) => r >= 0);
    if (profile.kind === "number") {
      const mine = histogram(
        rows.map((r) => column.values[r]),
        profile.min,
        profile.max,
        profile.histogram.length,
      );
      drawBars(ctx, profile.histogram, inner.x, y, inner.w, HEADER.histogram, c.bar);
      if (rows.length)
        drawBars(ctx, mine, inner.x, y, inner.w, HEADER.histogram, c.accent, Math.max(...profile.histogram));
    } else {
      drawBars(
        ctx,
        profile.categories.map((cat) => cat.count),
        inner.x,
        y,
        inner.w,
        HEADER.histogram,
        c.bar,
      );
    }
    y += HEADER.histogram + HEADER.gap;
    if (profile.kind === "number") {
      drawEnvelope(
        ctx,
        profile.sequence,
        profile.min,
        profile.max,
        profile.n,
        inner.x,
        y,
        inner.w,
        HEADER.sequence,
        c.muted,
      );
      ctx.fillStyle = c.accent;
      for (const r of rows) {
        const v = column.values[r];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        const span = profile.max - profile.min || 1;
        ctx.fillRect(
          inner.x + (r / Math.max(1, profile.n - 1)) * inner.w - 1.5,
          y + HEADER.sequence - ((v - profile.min) / span) * HEADER.sequence - 1.5,
          3,
          3,
        );
      }
    }
    y += HEADER.sequence + HEADER.gap;
    const step = inner.w / Math.max(1, profile.strip.length);
    ctx.fillStyle = c.warning;
    profile.strip.forEach((share, b) => {
      if (share > 0) ctx.fillRect(inner.x + b * step, y, Math.max(1, step), HEADER.missing * Math.max(0.35, share));
    });
    if (state.hoveredCol === column.id) outline(ctx, c.fg, x + 0.5, 0.5, w - 1, HEADER_HEIGHT - 1);
    stroke(ctx, c.line, x + w - 0.5, 0, x + w - 0.5, HEADER_HEIGHT);
  }

  #rowIndex(id) {
    this.#index ??= new Map(this.#rowIds.map((r, i) => [r, i]));
    return this.#index.get(id) ?? -1;
  }

  #index = null;
}

/** The page's colours, read from its CSS custom properties, with the light theme where one is not set. */
function readColours(root) {
  const css = getComputedStyle(root);
  const read = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  return {
    fg: read("--fg", "#1a1a1a"),
    muted: read("--muted", "#666666"),
    line: read("--line", "#d0d0d0"),
    bg: read("--bg", "#ffffff"),
    stripe: read("--stripe", "rgba(0, 0, 0, 0.03)"),
    selection: read("--selection", "rgba(0, 114, 178, 0.16)"),
    accent: read("--accent", "#0072B2"),
    bar: read("--bar", "#b8c4cc"),
    warning: read("--warning", "#D55E00"),
  };
}

const fill = (ctx, colour, x, y, w, h) => {
  ctx.fillStyle = colour;
  ctx.fillRect(x, y, w, h);
};

const stroke = (ctx, colour, x1, y1, x2, y2) => {
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
};

const outline = (ctx, colour, x, y, w, h, width = 1) => {
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.strokeRect(x, y, w, h);
};

/** Text cut with an ellipsis to fit `width` pixels. */
function clip(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  let cut = text.length;
  while (cut > 0 && ctx.measureText(`${text.slice(0, cut)}…`).width > width) cut -= 1;
  return `${text.slice(0, cut)}…`;
}

function drawBars(ctx, counts, x, y, w, h, colour, scaleTo = Math.max(...counts)) {
  const top = scaleTo || 1;
  const bw = w / Math.max(1, counts.length);
  ctx.fillStyle = colour;
  counts.forEach((count, k) => {
    const bh = (count / top) * h;
    if (bh > 0) ctx.fillRect(x + k * bw + 0.5, y + h - bh, Math.max(1, bw - 1), bh);
  });
}

function drawEnvelope(ctx, sequence, min, max, n, x, y, w, h, colour) {
  const span = max - min || 1;
  const at = (i) => x + (i / Math.max(1, n - 1)) * w;
  const level = (v) => y + h - ((v - min) / span) * h;
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1;
  ctx.beginPath();
  let drawing = false;
  for (const { at: i, lo, hi } of sequence) {
    if (lo === null) {
      drawing = false;
      continue;
    }
    if (drawing) ctx.lineTo(at(i), level(lo));
    else ctx.moveTo(at(i), level(lo));
    ctx.lineTo(at(i), level(hi));
    drawing = true;
  }
  ctx.stroke();
}
