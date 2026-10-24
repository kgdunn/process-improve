// The selection store: which rows and columns the reader has picked, hovered, excluded or saved under a name.
// One store serves a page, and every view subscribes to it. Rows and columns are named by their ids, as text,
// never by position, so a selection keeps its meaning through a sort, a filter, a dropped row or a refit.
// Nothing here touches the document, so tests/web/mva runs it under Node.

/** The two things a reader can select: observations (rows) and variables (columns). */
export const DIMENSIONS = ["rows", "cols"];

/** How a new pick combines with what is already selected. */
export const MODES = ["replace", "add", "remove", "toggle"];

/** The set operations a reader can combine two saved sets with. */
export const OPERATIONS = ["union", "intersection", "difference", "symmetric"];

/** Colours given to saved sets in turn: the Okabe-Ito palette, which reads apart with every colour vision. */
export const SET_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"];

/** A row or column id as the store holds it: text, whatever the table's index holds. */
export const idOf = (value) => String(value);

/** Combine two sets of ids by one of OPERATIONS, into a new set. */
export function combine(a, operation, b) {
  switch (operation) {
    case "union":
      return new Set([...a, ...b]);
    case "intersection":
      return new Set([...a].filter((id) => b.has(id)));
    case "difference":
      return new Set([...a].filter((id) => !b.has(id)));
    case "symmetric":
      return new Set([...[...a].filter((id) => !b.has(id)), ...[...b].filter((id) => !a.has(id))]);
    default:
      throw new Error(`Unknown set operation ${JSON.stringify(operation)}; choose from ${OPERATIONS.join(", ")}.`);
  }
}

const checkDimension = (dim) => {
  if (!DIMENSIONS.includes(dim)) throw new Error(`Unknown dimension ${JSON.stringify(dim)}: use "rows" or "cols".`);
};

export class Selection {
  #picked = { rows: new Set(), cols: new Set() };
  #hovered = { rows: null, cols: null };
  #excluded = new Set();
  #sets = new Map(); // name -> { rows: Set, colour }
  #listeners = new Set();
  #changed = new Set();
  #scheduled = false;

  /**
   * Call `listener(changes)` after every change, with the set of what changed: "rows", "cols", "hover",
   * "excluded", "sets". Changes made in one task arrive together, so a lasso of 5,000 points is one call.
   * Returns the function that stops the calls.
   */
  subscribe(listener) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #notify(part) {
    this.#changed.add(part);
    if (this.#scheduled) return;
    this.#scheduled = true;
    queueMicrotask(() => {
      const changes = this.#changed;
      this.#changed = new Set();
      this.#scheduled = false;
      for (const listener of this.#listeners) listener(changes);
    });
  }

  /** The ids selected on a dimension, as a new set the caller may keep. */
  selected(dim) {
    checkDimension(dim);
    return new Set(this.#picked[dim]);
  }

  /** Whether an id is selected on a dimension. */
  has(dim, id) {
    checkDimension(dim);
    return this.#picked[dim].has(idOf(id));
  }

  /** Select ids on a dimension: replace the selection, add to it, remove from it, or toggle each id. */
  select(dim, ids, mode = "replace") {
    checkDimension(dim);
    if (!MODES.includes(mode))
      throw new Error(`Unknown mode ${JSON.stringify(mode)}; choose from ${MODES.join(", ")}.`);
    const picked = this.#picked[dim];
    const before = new Set(picked);
    if (mode === "replace") picked.clear();
    for (const id of Array.from(ids, idOf)) {
      if (mode === "remove" || (mode === "toggle" && picked.has(id))) picked.delete(id);
      else picked.add(id);
    }
    if (picked.size !== before.size || [...picked].some((id) => !before.has(id))) this.#notify(dim);
  }

  /** Clear the selection on a dimension, or on both. */
  clear(dim = null) {
    for (const d of dim ? [dim] : DIMENSIONS) this.select(d, [], "replace");
  }

  /** Echo the id under the pointer in one view to every other view; null when the pointer leaves. */
  hover(dim, id) {
    checkDimension(dim);
    const value = id === null || id === undefined ? null : idOf(id);
    if (this.#hovered[dim] === value) return;
    this.#hovered[dim] = value;
    this.#notify("hover");
  }

  /** The id hovered on a dimension, or null. */
  hovered(dim) {
    checkDimension(dim);
    return this.#hovered[dim];
  }

  /** Exclude rows from the model: they stay on every plot, hollow, projected through the model. */
  exclude(ids) {
    const before = this.#excluded.size;
    for (const id of Array.from(ids, idOf)) this.#excluded.add(id);
    if (this.#excluded.size !== before) this.#notify("excluded");
  }

  /** Put excluded rows back into the model. */
  include(ids) {
    const before = this.#excluded.size;
    for (const id of Array.from(ids, idOf)) this.#excluded.delete(id);
    if (this.#excluded.size !== before) this.#notify("excluded");
  }

  /** The excluded rows, as a new set. */
  excluded() {
    return new Set(this.#excluded);
  }

  /** Whether a row is excluded. */
  isExcluded(id) {
    return this.#excluded.has(idOf(id));
  }

  /** Save rows under a name (by default the rows selected now), with the next colour unless one is given. */
  saveSet(name, ids = this.#picked.rows, colour = null) {
    const label = String(name).trim();
    if (!label) throw new Error("A saved set needs a name.");
    const previous = this.#sets.get(label);
    const next = colour ?? previous?.colour ?? SET_COLOURS[this.#sets.size % SET_COLOURS.length];
    this.#sets.set(label, { rows: new Set(Array.from(ids, idOf)), colour: next });
    this.#notify("sets");
  }

  /** Remove a saved set. */
  deleteSet(name) {
    if (this.#sets.delete(String(name))) this.#notify("sets");
  }

  /** Rename a saved set, keeping its rows, its colour and its place. */
  renameSet(name, newName) {
    const old = String(name);
    const label = String(newName).trim();
    if (!this.#sets.has(old)) throw new Error(`There is no saved set ${JSON.stringify(old)}.`);
    if (!label) throw new Error("A saved set needs a name.");
    if (label !== old && this.#sets.has(label)) throw new Error(`A set named ${JSON.stringify(label)} exists already.`);
    this.#sets = new Map([...this.#sets].map(([key, value]) => [key === old ? label : key, value]));
    this.#notify("sets");
  }

  /** The saved sets, in the order they were saved: [{ name, rows, colour }], each rows a new set. */
  sets() {
    return [...this.#sets].map(([name, { rows, colour }]) => ({ name, rows: new Set(rows), colour }));
  }

  /** The rows of a saved set, as a new set. */
  setRows(name) {
    const found = this.#sets.get(String(name));
    if (!found) throw new Error(`There is no saved set ${JSON.stringify(String(name))}.`);
    return new Set(found.rows);
  }

  /** Select the rows of a saved set, or of two sets combined by one of OPERATIONS. */
  selectSet(name, { operation = null, other = null, mode = "replace" } = {}) {
    const rows = operation ? combine(this.setRows(name), operation, this.setRows(other)) : this.setRows(name);
    this.select("rows", rows, mode);
  }

  /** What a project file keeps of the store: the saved sets, by name, and the excluded rows. */
  toJSON() {
    return {
      selections: Object.fromEntries([...this.#sets].map(([name, { rows }]) => [name, [...rows]])),
      colours: Object.fromEntries([...this.#sets].map(([name, { colour }]) => [name, colour])),
      excluded: [...this.#excluded],
    };
  }

  /** A store holding what toJSON wrote; anything else in `data` is ignored. */
  static fromJSON(data) {
    const store = new Selection();
    const selections = data?.selections && typeof data.selections === "object" ? data.selections : {};
    for (const [name, rows] of Object.entries(selections)) {
      if (Array.isArray(rows)) store.#sets.set(name, { rows: new Set(rows.map(idOf)), colour: null });
    }
    let k = 0;
    for (const [name, entry] of store.#sets) {
      const given = data?.colours?.[name];
      entry.colour =
        typeof given === "string" && /^#[0-9a-fA-F]{6}$/.test(given) ? given : SET_COLOURS[k % SET_COLOURS.length];
      k += 1;
    }
    if (Array.isArray(data?.excluded)) for (const id of data.excluded) store.#excluded.add(idOf(id));
    return store;
  }

  /** The state a figure is built from: frozen copies, so a view cannot change the store by accident. */
  snapshot(dim) {
    checkDimension(dim);
    return Object.freeze({
      selected: new Set(this.#picked[dim]),
      hovered: this.#hovered[dim],
      excluded: dim === "rows" ? new Set(this.#excluded) : new Set(),
      sets: dim === "rows" ? this.sets() : [],
    });
  }
}
