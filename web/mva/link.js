// Linked views, live: draw a figure into a <div>, turn the reader's lasso, box, clicks and hover into
// selections in the store, and redraw when the store changes. Every view bound to one store shows the same
// selection: lasso points on a score plot and their rows light up in the table and in every other plot.
//
// The caller hands in Plotly and the scheduler, so tests/web/mva runs this under Node with stand-ins.

/** The parts of the store a view redraws for; "sets" only matters to the rows, where the sets colour the points. */
const REDRAWS_FOR = { rows: ["rows", "hover", "excluded", "sets"], cols: ["cols", "hover"] };

/**
 * How a lasso or a click combines with the selection: Shift adds, Alt (Option) removes, Ctrl or Cmd toggles a
 * clicked point; otherwise it replaces.
 */
export function modeOf(event) {
  if (event?.altKey) return "remove";
  if (event?.shiftKey) return "add";
  if (event?.ctrlKey || event?.metaKey) return "toggle";
  return "replace";
}

/** The ids of the points a Plotly event names, from each point's customdata. */
export const idsOf = (event) =>
  (event?.points ?? [])
    .map((p) => p.customdata)
    .filter((id) => id !== undefined && id !== null)
    .map(String);

/**
 * Bind a figure to `div` and to `selection`. `build(state)` returns the figure for the store's state on `dim`
 * ("rows" for a score plot, "cols" for a loading plot). Returns { redraw, destroy }.
 *
 * options: { Plotly, schedule = requestAnimationFrame, config, keys = window }
 */
export function bindView(div, selection, dim, build, options = {}) {
  const { Plotly } = options;
  const schedule = options.schedule ?? ((fn) => requestAnimationFrame(fn));
  const config = { responsive: true, displaylogo: false, ...(options.config ?? {}) };
  // Plotly's selection events carry no keys, so the keys held while the lasso is drawn are tracked here.
  const keys = { altKey: false, shiftKey: false, ctrlKey: false, metaKey: false };
  const track = (event) =>
    Object.assign(keys, {
      altKey: event.altKey,
      shiftKey: event.shiftKey,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
    });
  const target = options.keys ?? (typeof window === "undefined" ? null : window);
  target?.addEventListener("keydown", track);
  target?.addEventListener("keyup", track);

  let pending = false;
  let alive = true;
  const draw = (first = false) => {
    const figure = build(selection.snapshot(dim));
    return first
      ? Plotly.newPlot(div, figure.data, figure.layout, config)
      : Plotly.react(div, figure.data, figure.layout, config);
  };
  const redraw = () => {
    if (pending || !alive) return;
    pending = true;
    schedule(() => {
      pending = false;
      if (alive) draw();
    });
  };

  const ready = Promise.resolve(draw(true)).then(() => {
    div.on("plotly_selected", (event) => {
      if (!event) return; // a click on empty space, which Plotly reports as a selection of nothing
      selection.select(dim, idsOf(event), modeOf(keys));
    });
    div.on("plotly_click", (event) => {
      const ids = idsOf(event);
      if (ids.length)
        selection.select(dim, ids.slice(0, 1), modeOf(event?.event ?? keys) === "replace" ? "replace" : "toggle");
    });
    div.on("plotly_doubleclick", () => selection.clear(dim));
    div.on("plotly_hover", (event) => selection.hover(dim, idsOf(event)[0] ?? null));
    div.on("plotly_unhover", () => selection.hover(dim, null));
  });
  const unsubscribe = selection.subscribe((changes) => {
    if (REDRAWS_FOR[dim].some((part) => changes.has(part))) redraw();
  });

  return {
    ready,
    redraw,
    destroy() {
      alive = false;
      unsubscribe();
      target?.removeEventListener("keydown", track);
      target?.removeEventListener("keyup", track);
      Plotly.purge?.(div);
    },
  };
}
