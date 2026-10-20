// The design catalog's rules, apart from the page: what a search asks for, which designs pass the
// filters, how many each filter would leave, what the address keeps, and the picture of a design's
// runs. catalog.js draws the page from these; nothing here touches the document, so tests/web/js
// runs it under Node.

/** The fields of each design in catalog/index.json, in order: the library's facets, then the build's. */
export const INDEX_FIELDS = [
  "id", "title", "collection", "kind", "family", "design_type", "source", "year", "added", "where", "topics",
  "n_runs", "n_factors", "n_numeric", "n_categorical", "n_mixture", "levels", "n_centre", "runs_per_parameter",
  "models", "blocked", "n_blocks", "split_plot", "has_responses", "n_responses", "checks", "glyph",
  "survives_run_loss", "critical_runs", "worst_d_ratio", "quadratic_projection", "projectivity", "design_class",
  "regenerable", "open", "detail", "compared",
];

/** The format of catalog/index.json this page reads. */
export const CATALOG_VERSION = 1;

/** The models a design is filtered on, each including the ones before it, as the library names them. */
export const MODELS = { main_effects: "main effects", interactions: "two-factor interactions", quadratic: "the full quadratic" };

/** Which designs to show: the published ones, the ones the library makes, or both. */
export const COLLECTIONS = ["published", "generated", "both"];

/** How the list can be ordered. */
export const SORTS = { default: "Catalogue order", runs: "Fewest runs", factors: "Fewest factors", newest: "Most recent", title: "Title" };

/** The colour of each family on the map: the Okabe-Ito colours first, for the families most designs are in. */
export const FAMILY_COLOURS = {
  full_factorial: "#0072B2",
  fractional_factorial: "#56B4E9",
  plackett_burman: "#009E73",
  ccd: "#D55E00",
  box_behnken: "#E69F00",
  dsd: "#CC79A7",
  omars: "#7B3294",
  optimal: "#999933",
  mixture: "#44AA99",
  orthogonal_array: "#882255",
  supersaturated: "#661100",
};

/** The colour of a family the map does not single out. */
export const OTHER_COLOUR = "#A6A49E";

/** The filters with nothing chosen: every design passes. */
export const DEFAULT_STATE = Object.freeze({
  collection: "both",
  query: "",
  families: [],
  topics: [],
  runs: [null, null],
  factors: [null, null],
  model: "",
  categorical: [null, null],
  blocked: false,
  splitPlot: false,
  mixture: false,
  hasData: false,
  recomputed: false,
  differs: false,
  opens: false,
  source: "",
  unprinted: false,
  survives: false,
  projection: 0,
  projectivity: 0,
  years: [null, null],
  addedSince: "",
  sort: "default",
  design: "",
  compare: [],
  view: "",
  limit: "",
  limitAll: false,
});

/** The views the page opens over the list: the comparison of the designs chosen, and the errata. */
export const VIEWS = ["compare", "errata"];

/** The fewest runs a model in k factors needs: one per coefficient, intercept included. */
export function minimumRuns(k, model) {
  if (model === "main_effects") return 1 + k;
  if (model === "interactions") return 1 + k + (k * (k - 1)) / 2;
  return ((k + 1) * (k + 2)) / 2;
}

// ---------------------------------------------------------------- the search box
/** The words that name a family in a search, each with the family it names. */
const FAMILY_WORDS = [
  [/\bdsds?\b|definitive screening/, "dsd"],
  [/\bomars\b/, "omars"],
  [/\bccds?\b|central composite/, "ccd"],
  [/box[- ]behnken|\bbbds?\b/, "box_behnken"],
  [/plackett|\bpb\b/, "plackett_burman"],
  [/fractional|fractions?\b/, "fractional_factorial"],
  [/full factorials?/, "full_factorial"],
  [/taguchi|orthogonal arrays?/, "orthogonal_array"],
  [/latin squares?/, "latin_square"],
  [/\b(?:d|i|g)-optimal|optimal designs?/, "optimal"],
  [/supersaturated/, "supersaturated"],
  [/mixtures?/, "mixture"],
  [/split[- ]plots?/, "split_plot"],
];

const range = (low, high) => [low, high];

/**
 * Read what a search asks for. "5 factors, <=20 runs, quadratic, 1 categorical, since 2010" becomes
 * filters; every word that is no filter is searched for in the title, the family, the source and
 * where it is printed. A filter written twice keeps the last.
 */
export function parseQuery(text) {
  let rest = ` ${String(text ?? "").toLowerCase()} `;
  const found = { families: [], words: [] };
  const take = (pattern, use) => {
    rest = rest.replace(pattern, (...m) => {
      use(m);
      return " ";
    });
  };
  // The projections first: "3-factor projection" names no number of factors.
  take(/projectivity\s*(?:of\s*)?(\d)/g, (m) => (found.projectivity = +m[1]));
  take(/(?:every|any)?\s*(\d)[- ]factor projections?|projects? (?:on)?to (?:any )?(\d)/g, (m) => (found.projection = +(m[1] ?? m[2])));
  take(/(\d+)\s*(?:-|to)\s*(\d+)\s*factors?/g, (m) => (found.factors = range(+m[1], +m[2])));
  take(/(?:k\s*=\s*(\d+))|(\d+)\s*factors?/g, (m) => (found.factors = range(+(m[1] ?? m[2]), +(m[1] ?? m[2]))));
  take(/(\d+)\s*(?:-|to)\s*(\d+)\s*runs?/g, (m) => (found.runs = range(+m[1], +m[2])));
  take(/(?:<=|≤|at most|up to)\s*(\d+)\s*runs?/g, (m) => (found.runs = range(null, +m[1])));
  take(/(?:<|under|fewer than|less than)\s*(\d+)\s*runs?/g, (m) => (found.runs = range(null, +m[1] - 1)));
  take(/(?:>=|≥|at least)\s*(\d+)\s*runs?/g, (m) => (found.runs = range(+m[1], null)));
  take(/(\d+)\s*runs?/g, (m) => (found.runs = range(+m[1], +m[1])));
  for (const [pattern, family] of FAMILY_WORDS) {
    take(new RegExp(pattern.source, "g"), () => {
      if (!found.families.includes(family)) found.families.push(family);
    });
  }
  take(/no categorical(?: factors?)?/g, () => (found.categorical = range(null, 0)));
  take(/(\d+)\s*categorical(?: factors?)?/g, (m) => (found.categorical = range(+m[1], null)));
  take(/categorical(?: factors?)?/g, () => (found.categorical = range(1, null)));
  take(/quadratic|response surface|curvature/g, () => (found.model = "quadratic"));
  take(/two-factor interactions?|interactions?|\b2fi\b/g, () => (found.model = "interactions"));
  take(/main effects?|screening/g, () => (found.model = "main_effects"));
  take(/\bblock(?:s|ed)?\b/g, () => (found.blocked = true));
  take(/(?:has|with) data|results|responses/g, () => (found.hasData = true));
  take(/reproduced|recomputed|verified|checked/g, () => (found.recomputed = true));
  take(/errata|does not reproduce|disagree(?:s|ment)?/g, () => (found.differs = true));
  take(/\bgaps?\b|not (?:yet )?(?:printed|published)|no source prints it/g, () => (found.unprinted = true));
  take(/surviv(?:es|ing)?(?: (?:the )?loss of (?:a|any) run)?|robust to a lost run|(?:any|a) (?:lost|missing) run/g, () => (found.survives = true));
  take(/(\d{4})\s*(?:-|to)\s*(\d{4})/g, (m) => (found.years = range(+m[1], +m[2])));
  take(/since\s*(\d{4})|after\s*(\d{4})/g, (m) => (found.years = range(m[1] ? +m[1] : +m[2] + 1, null)));
  take(/before\s*(\d{4})/g, (m) => (found.years = range(null, +m[1] - 1)));
  take(/\b(1[89]\d\d|20\d\d)\b/g, (m) => (found.years = range(+m[1], +m[1])));
  take(/\bp(?:p|age)?\.?\s*(\d+)\b/g, (m) => found.words.push(String(m[1])));
  found.words.push(...rest.split(/[\s,;]+/).filter((w) => w.length > 1 && !/^(and|the|with|of|in|a|an|or|for|design|designs)$/.test(w)));
  return found;
}

// ---------------------------------------------------------------- the filters
const between = (value, [low, high]) =>
  (low === null || (value !== null && value !== undefined && value >= low)) &&
  (high === null || (value !== null && value !== undefined && value <= high));

const narrower = (a = [null, null], b = [null, null]) => [
  a[0] === null ? b[0] : b[0] === null ? a[0] : Math.max(a[0], b[0]),
  a[1] === null ? b[1] : b[1] === null ? a[1] : Math.min(a[1], b[1]),
];

/** Join the filters chosen beside the list with those the search box asks for: a design must pass both. */
export function filtersOf(state) {
  const asked = parseQuery(state.query);
  const families = state.families.length && asked.families.length
    ? state.families.filter((f) => asked.families.includes(f))
    : state.families.length ? state.families : asked.families;
  return {
    collection: state.collection,
    families: state.families.length && asked.families.length && !families.length ? ["(none)"] : families,
    topics: state.topics,
    runs: narrower(state.runs, asked.runs),
    factors: narrower(state.factors, asked.factors),
    model: state.model || asked.model || "",
    categorical: narrower(state.categorical, asked.categorical),
    blocked: state.blocked || Boolean(asked.blocked),
    splitPlot: state.splitPlot,
    mixture: state.mixture,
    hasData: state.hasData || Boolean(asked.hasData),
    recomputed: state.recomputed || Boolean(asked.recomputed),
    differs: state.differs || Boolean(asked.differs),
    opens: state.opens,
    source: state.source,
    unprinted: state.unprinted || Boolean(asked.unprinted),
    survives: state.survives || Boolean(asked.survives),
    projection: Math.max(state.projection, asked.projection ?? 0),
    projectivity: Math.max(state.projectivity, asked.projectivity ?? 0),
    years: narrower(state.years, asked.years),
    addedSince: state.addedSince,
    limit: Boolean(state.limit) && parseLimit(state.limit).ok,
    limitAll: state.limitAll,
    words: asked.words,
  };
}

/**
 * The text a word of a search is looked for in: the design's title, family, source, where it is printed
 * and its applications, so "food" finds the designs of food and beverages.
 */
export function haystack(design, catalog) {
  const source = catalog.sources?.[design.source];
  const family = catalog.families?.[design.family];
  return [design.id, design.title, design.where, family?.label, source?.cite, source?.title, design.year, ...applications(design, catalog)]
    .filter((v) => v !== null && v !== undefined)
    .join(" ")
    .toLowerCase();
}

/** Each filter, by name, as a test of one design. */
const TESTS = {
  collection: (d, f) => f.collection === "both" || d.collection === f.collection,
  families: (d, f) => !f.families.length || f.families.includes(d.family),
  // A generated design has no application, so any application keeps published designs only.
  topics: (d, f) => !f.topics.length || (d.topics ?? []).some((t) => f.topics.includes(t)),
  runs: (d, f) => between(d.n_runs, f.runs),
  factors: (d, f) => between(d.n_factors, f.factors),
  model: (d, f) => !f.model || d.models?.[f.model] === true,
  categorical: (d, f) => between(d.n_categorical, f.categorical),
  blocked: (d, f) => !f.blocked || d.blocked,
  splitPlot: (d, f) => !f.splitPlot || d.split_plot,
  mixture: (d, f) => !f.mixture || d.n_mixture > 0,
  hasData: (d, f) => !f.hasData || d.has_responses,
  recomputed: (d, f) => !f.recomputed || (d.checks?.checked ?? 0) > 0,
  differs: (d, f) => !f.differs || (d.checks?.explained ?? 0) > 0,
  opens: (d, f) => !f.opens || d.open,
  source: (d, f) => !f.source || d.source === f.source,
  unprinted: (d, f) => !f.unprinted || (d.collection === "generated" && !d.in_source),
  survives: (d, f) => !f.survives || d.survives_run_loss === true,
  projection: (d, f) => !f.projection || (d.quadratic_projection ?? 0) >= f.projection,
  projectivity: (d, f) => !f.projectivity || (d.projectivity ?? 0) >= f.projectivity,
  years: (d, f) => between(d.year, f.years),
  addedSince: (d, f) => !f.addedSince || (d.added !== null && d.added >= f.addedSince),
  // A design is judged against a limit once the page has read its runs (limitOutcome): it passes when every
  // run meets the limit, or, with limitAll, whenever it can be judged at all.
  limit: (d, f) => !f.limit || (Number.isInteger(d.limit_outside) && (f.limitAll || d.limit_outside === 0)),
};

/** Say whether a design passes every filter, leaving out the one named ``except``. */
export function passes(design, filters, catalog, except = null) {
  for (const [name, test] of Object.entries(TESTS)) if (name !== except && !test(design, filters)) return false;
  if (!filters.words.length) return true;
  const text = haystack(design, catalog);
  return filters.words.every((word) => text.includes(word));
}

/** The designs that pass every filter, in the order chosen. */
export function results(catalog, state) {
  const filters = filtersOf(state);
  return sorted(catalog.designs.filter((d) => passes(d, filters, catalog)), state.sort);
}

/** How many designs each family would show, with every other filter as it is. */
export function familyCounts(catalog, state) {
  const filters = filtersOf(state);
  const counts = {};
  for (const d of catalog.designs) {
    if (passes(d, filters, catalog, "families")) counts[d.family] = (counts[d.family] ?? 0) + 1;
  }
  return counts;
}

// ---------------------------------------------------------------- the applications
/** The topic of a design whose source states no application: listed last, and shown on no card. */
export const GENERAL = "general";

/** How many designs each application would show, with every other filter as it is. */
export function topicCounts(catalog, state) {
  const filters = filtersOf(state);
  const counts = {};
  for (const d of catalog.designs) {
    if (!passes(d, filters, catalog, "topics")) continue;
    for (const t of d.topics ?? []) counts[t] = (counts[t] ?? 0) + 1;
  }
  return counts;
}

/** The applications the filter lists: those some design has, the most designs first, and "general" last. */
export function topicOrder(catalog) {
  const totals = {};
  for (const d of catalog.designs) for (const t of d.topics ?? []) totals[t] = (totals[t] ?? 0) + 1;
  return Object.keys(catalog.topics ?? {})
    .filter((t) => totals[t])
    .sort((a, b) => (a === GENERAL) - (b === GENERAL) || totals[b] - totals[a]);
}

/** A design's applications, by name, without "general": what its card's badges and its detail say. */
export function applications(design, catalog) {
  return (design.topics ?? []).filter((t) => t !== GENERAL).map((t) => catalog.topics?.[t]?.label ?? t);
}

/** A design's applications in a sentence, as its detail says them; empty when its source states none. */
export function applicationLine(design, catalog) {
  const names = applications(design, catalog);
  if (!names.length) return "";
  return `${names.length === 1 ? "Application" : "Applications"}: ${names.join("; ")}.`;
}

/** How many designs a yes-or-no filter would leave if it were ticked, with every other filter as it is. */
export function switchCount(catalog, state, name) {
  const filters = { ...filtersOf(state), [name]: true };
  return catalog.designs.filter((d) => passes(d, filters, catalog)).length;
}

/** Families whose entries are no design of their own: comparisons, regressions and the like. */
const NOT_A_DESIGN = new Set(["comparison", "other"]);

/** Whether the catalog lists a design ahead of the rest: it prints its runs and is a designed experiment. */
const ahead = (d) => (d.glyph && !NOT_A_DESIGN.has(d.family) ? 0 : 1);

/**
 * Order designs: as the catalogue lists them (designs with their runs first, then comparisons and
 * entries without runs), by size, by year (newest first) or by title.
 */
export function sorted(designs, how) {
  const by = {
    default: (a, b) => ahead(a) - ahead(b),
    runs: (a, b) => (a.n_runs ?? Infinity) - (b.n_runs ?? Infinity) || (a.n_factors ?? 0) - (b.n_factors ?? 0),
    factors: (a, b) => (a.n_factors ?? Infinity) - (b.n_factors ?? Infinity) || (a.n_runs ?? 0) - (b.n_runs ?? 0),
    newest: (a, b) => (b.year ?? -Infinity) - (a.year ?? -Infinity),
    title: (a, b) => a.title.localeCompare(b.title),
  }[how];
  return [...designs].sort(by ?? (() => 0));
}

/**
 * Count published designs by the year their source appeared, in bins of ``width`` years, with every
 * filter but the years as it is: the bars of the date filter.
 */
export function yearBins(catalog, state, width = 5) {
  const filters = filtersOf(state);
  const years = catalog.designs.filter((d) => d.year !== null && passes(d, filters, catalog, "years")).map((d) => d.year);
  if (!years.length) return [];
  const first = Math.floor(Math.min(...years) / width) * width;
  const last = Math.floor(Math.max(...years) / width) * width;
  const bins = [];
  const kept = catalog.designs.filter((d) => d.year !== null && passes(d, filters, catalog, "years"));
  for (let from = first; from <= last; from += width) {
    const inside = kept.filter((d) => d.year >= from && d.year < from + width);
    const families = {};
    for (const d of inside) families[d.family] = (families[d.family] ?? 0) + 1;
    bins.push({ from, to: from + width - 1, count: inside.length, families });
  }
  return bins;
}

/** The families first published within the years the bars span, each with its year: the marks on the timeline. */
export function familyMarks(catalog, bins) {
  if (!bins.length) return [];
  const [first, last] = [bins[0].from, bins.at(-1).to];
  return Object.entries(catalog.families ?? {})
    .filter(([, f]) => f.year && f.year >= first && f.year <= last)
    .map(([family, f]) => ({ family, year: f.year, label: f.label }))
    .sort((a, b) => a.year - b.year);
}

// ---------------------------------------------------------------- the address
const KEYS = {
  collection: "col", query: "q", families: "fam", topics: "app", runs: "runs", factors: "k", model: "model",
  categorical: "cat", blocked: "blk", splitPlot: "sp", mixture: "mix", hasData: "data", recomputed: "rec",
  differs: "err", opens: "open", source: "src", unprinted: "gap", survives: "surv", projection: "qp",
  projectivity: "proj", years: "year", addedSince: "added", sort: "sort", design: "d", compare: "cmp", view: "view",
  limit: "lim", limitAll: "lima",
};

/** The filters held as a list of names, written into the address joined by commas. */
const LISTS = new Set(["families", "topics", "compare"]);

/** A design's id, as the address may name one. */
const AN_ID = /^[a-z0-9][a-z0-9-]{0,120}$/;

const writeRange = ([low, high]) => `${low ?? ""}-${high ?? ""}`;

function readRange(text) {
  const m = /^(\d{0,5})-(\d{0,5})$/.exec(text ?? "");
  if (!m) return [null, null];
  return [m[1] === "" ? null : +m[1], m[2] === "" ? null : +m[2]];
}

/** Write the filters into the part of the address after "#", leaving out every filter at its default. */
export function toHash(state) {
  const params = new URLSearchParams();
  for (const [name, key] of Object.entries(KEYS)) {
    const value = state[name];
    const base = DEFAULT_STATE[name];
    if (Array.isArray(value)) {
      if (LISTS.has(name) ? value.length : value[0] !== null || value[1] !== null) {
        params.set(key, LISTS.has(name) ? value.join(",") : writeRange(value));
      }
    } else if (value !== base) {
      params.set(key, value === true ? "1" : String(value));
    }
  }
  const text = params.toString();
  return text ? `#${text}` : "";
}

/**
 * Read the filters back from an address. The address is text anyone can write: a value of the
 * wrong kind, a family or an application the catalog does not have, or an unknown key is ignored.
 */
export function fromHash(text, families = {}, sources = {}, topics = {}) {
  const params = new URLSearchParams(String(text ?? "").replace(/^#/, ""));
  const state = structuredClone({ ...DEFAULT_STATE });
  const read = (name) => params.get(KEYS[name]);
  if (COLLECTIONS.includes(read("collection"))) state.collection = read("collection");
  state.query = (read("query") ?? "").slice(0, 200);
  state.limit = (read("limit") ?? "").slice(0, LIMIT_LENGTH);
  state.families = (read("families") ?? "").split(",").filter((f) => Object.hasOwn(families, f));
  state.topics = [...new Set((read("topics") ?? "").split(",").filter((t) => Object.hasOwn(topics, t)))];
  for (const name of ["runs", "factors", "categorical", "years"]) state[name] = readRange(read(name));
  if (Object.hasOwn(MODELS, read("model"))) state.model = read("model");
  if (Object.hasOwn(sources, read("source"))) state.source = read("source");
  for (const name of ["blocked", "splitPlot", "mixture", "hasData", "recomputed", "differs", "opens", "unprinted", "survives", "limitAll"]) {
    state[name] = read(name) === "1";
  }
  for (const name of ["projection", "projectivity"]) {
    const n = Number(read(name));
    if (Number.isInteger(n) && n >= 1 && n <= 9) state[name] = n;
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(read("addedSince") ?? "")) state.addedSince = read("addedSince");
  if (Object.hasOwn(SORTS, read("sort"))) state.sort = read("sort");
  if (AN_ID.test(read("design") ?? "")) state.design = read("design");
  state.compare = [...new Set((read("compare") ?? "").split(",").filter((id) => AN_ID.test(id)))].slice(0, COMPARE_MOST);
  if (VIEWS.includes(read("view"))) state.view = read("view");
  return state;
}

// ---------------------------------------------------------------- the picture of a design's runs
/** Read a design's glyph back into its coded settings: one row per run, one number (or null) per factor. */
export function decodeGlyph(glyph, k) {
  if (!glyph || !k) return [];
  const rows = [];
  for (let at = 0; at + k <= glyph.length; at += k) {
    rows.push([...glyph.slice(at, at + k)].map((c) => (c === "." ? null : (c.charCodeAt(0) - 97) / 5 - 2)));
  }
  return rows;
}

const LOW = [0x00, 0x72, 0xb2]; // #0072B2 at -1
const MID = [0xed, 0xed, 0xed]; // #EDEDED at 0
const HIGH = [0xd5, 0x5e, 0x00]; // #D55E00 at +1

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const hex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

/**
 * The colour of a coded setting, as the book draws a run matrix: blue at -1, light grey at 0 and
 * vermillion at +1, darker beyond, as an axial run of a central composite design is.
 */
export function settingColour(value) {
  if (value === null || value === undefined) return "transparent";
  const v = Math.max(-2, Math.min(2, value));
  if (v <= -1) return hex(mix(LOW, [0x00, 0x30, 0x4c], -1 - v));
  if (v >= 1) return hex(mix(HIGH, [0x6b, 0x2f, 0x00], v - 1));
  return v < 0 ? hex(mix(MID, LOW, -v)) : hex(mix(MID, HIGH, v));
}

// ---------------------------------------------------------------- words for a design
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The largest model a design estimates in full, in words; null when the catalog does not state it. */
export function largestModel(models) {
  if (!models) return null;
  return ["quadratic", "interactions", "main_effects"].find((m) => models[m] === true) ?? (models.main_effects === false ? "" : null);
}

/** One line of facts about a design: its family, its size, its factors and the largest model it estimates. */
export function factLine(design, catalog) {
  return [catalog.families?.[design.family]?.label ?? design.family, ...sizeFacts(design)].join(" · ");
}

/** The facts of factLine that are this design's own, not its family's, each as a sentence's start: "48 runs". */
export function caseFacts(design) {
  return sizeFacts(design).map((fact) => fact[0].toUpperCase() + fact.slice(1));
}

/** A design's factors, runs, the largest model it estimates, and its blocks, in words. */
function sizeFacts(design) {
  const parts = [];
  if (design.n_factors) {
    const kinds = [];
    if (design.n_categorical) kinds.push(`${design.n_categorical} categorical`);
    if (design.n_mixture) kinds.push(`${design.n_mixture} mixture`);
    parts.push(plural(design.n_factors, "factor") + (kinds.length ? ` (${kinds.join(", ")})` : ""));
  }
  if (design.n_runs) parts.push(plural(design.n_runs, "run"));
  const model = largestModel(design.models);
  if (model) parts.push(`estimates ${MODELS[model]}`);
  else if (model === "") parts.push("estimates fewer coefficients than it has factors");
  if (design.blocked) parts.push(design.n_blocks ? `${design.n_blocks} blocks` : "in blocks");
  if (design.split_plot) parts.push("split plot");
  return parts;
}

/** What the catalog can say of a published design's printed numbers. */
export function checkLine(checks) {
  if (!checks || !checks.printed) return "";
  if (!checks.checked) return `${plural(checks.printed, "printed number")}, none recomputed`;
  const agree = `${checks.agree} of ${plural(checks.checked, "recomputed number")} agree`;
  return checks.explained ? `${agree}; ${checks.explained} known to differ, with the reason` : agree;
}

/** The other designs of the catalog that are this one, with its runs, factors or signs in another order. */
export function sameDesigns(catalog, design) {
  if (!design.design_class) return [];
  return catalog.designs.filter((d) => d.design_class === design.design_class && d.id !== design.id);
}

/** What losing a run costs a design, in a sentence; empty when the catalog does not know. */
export function runLossLine(design, model) {
  if (design.survives_run_loss === null || design.survives_run_loss === undefined) return "";
  const what = MODELS[model] ?? "its model";
  if (design.survives_run_loss) {
    const kept = Math.round(100 * design.worst_d_ratio);
    return `If any one run is lost, the design still estimates ${what}; the worst loss leaves ${kept}% of its D-efficiency.`;
  }
  const n = design.critical_runs;
  const without = `without ${n === 1 ? "it" : "any one of them"} the design no longer estimates ${what}`;
  if (n === design.n_runs) return `None of its ${n} runs can be lost: ${without}.`;
  return `${n} of its ${design.n_runs} runs cannot be lost: ${without}.`;
}

/** What a design's projections hold, in sentences: onto how many factors it fits the quadratic, its projectivity. */
export function projectionLines(design) {
  const lines = [];
  const p = design.quadratic_projection;
  if (p && p >= 2) {
    lines.push(
      p === design.n_numeric
        ? `It estimates the full quadratic in all ${p} of its numeric factors.`
        : `Any ${p} of its factors: the runs estimate the full quadratic in them, so if only ${p} matter, their response surface needs no more runs.`,
    );
  }
  const q = design.projectivity;
  if (q && q >= 2) {
    lines.push(
      q === design.n_factors
        ? `Its two-level runs hold every combination of its ${q} factors' levels.`
        : `Projectivity ${q}: any ${q} of its factors hold all ${2 ** q} combinations of their two levels.`,
    );
  }
  return lines;
}

/**
 * Where a source prints a design, as its detail says it: the section, the pages, then each table and
 * each figure, by the names the build gives them. The detail links the first part to the place itself.
 */
export function whereParts(where) {
  if (!where) return [];
  if (typeof where === "string") return [where];
  const parts = [];
  if (where.section) parts.push(`section ${where.section}`);
  if (where.pages) parts.push(`${/[-,]/.test(String(where.pages)) ? "pages" : "page"} ${where.pages}`);
  return [...parts, ...(where.tables ?? []), ...(where.figures ?? [])].map(String);
}

/** The most tables and figures a design's "Printed in" line names after its first part; the rest are counted. */
export const PLACES_SHOWN = 4;

/**
 * The rest of a "Printed in" line, after its first part: the next places, at most PLACES_SHOWN of them, then a
 * count of the others, which the link to the section leads to. A section of the book can print twenty.
 */
export function placesAfter(rest, most = PLACES_SHOWN) {
  const shown = rest.slice(0, most);
  const more = rest.length - shown.length;
  const count = more ? `; and ${more} more ${more === 1 ? "table or figure" : "tables and figures"}` : "";
  return shown.length ? `; ${shown.join("; ")}${count}.` : ".";
}

/** Where a published design is printed, in a few words: its source's authors and year, and its pages. */
export function sourceLine(design, catalog) {
  const source = catalog.sources?.[design.source];
  if (!source) return "Made by process-improve";
  return [`${source.short} (${source.year})`, design.where].filter(Boolean).join(", ");
}

// ---------------------------------------------------------------- what a design is copied as
const repr = (value) => (typeof value === "string" ? JSON.stringify(value) : String(value));

/**
 * The Python that gives a reader the design: the library call that makes it, or the catalogue entry that holds it.
 * It is written to be read: one step a line, each step with a comment, and no line wider than the box shows.
 */
export function pythonCode(design, detail) {
  const spec = detail?.spec;
  if (spec && !spec.array) {
    const factors = (detail.factors ?? []).map((f) => {
      const kind = f.type === "mixture" ? ', type="mixture"' : "";
      return `    Factor(name=${repr(f.name)}, low=${repr(f.low)}, high=${repr(f.high)}${kind}),`;
    });
    const options = Object.entries(spec.options ?? {}).map(([key, value]) => `    ${key}=${repr(value)},`);
    return [
      "from process_improve.experiments import Factor, generate_design",
      "",
      "factors = [",
      ...factors,
      "]",
      "design = generate_design(",
      "    factors,",
      `    design_type=${repr(spec.design_type)},`,
      ...options,
      ")",
      "print(design.design)",
    ].join("\n");
  }
  if (spec?.array) {
    return [
      "from process_improve.experiments._orthogonal_arrays import orthogonal_array",
      "",
      `array = orthogonal_array(${repr(spec.array)})  # levels numbered from 0`,
      "print(array)",
    ].join("\n");
  }
  return catalogueCode(design, detail);
}

/** Whether the source prints the entry's runs: the page holds them coded, or holds the results printed beside them. */
const printsRuns = (detail) => Boolean(detail?.coded_runs?.length || Object.keys(detail?.response_values ?? {}).length);

/** The Python that reads a published design's entry from the reference catalogue, and prints its factors and runs. */
function catalogueCode(design, detail) {
  const listed = detail?.listed === "designs" ? "reference_designs" : "reference_experiments";
  const factors = Boolean(detail?.factors?.length);
  const runs = printsRuns(detail);
  const lines = [
    ...(factors || runs ? ["import pandas as pd", ""] : []),
    "from process_improve.experiments.reference_catalogue import (",
    `    ${listed},`,
    ")",
    "",
    "# Every entry of this source, looked up by its name.",
    `entries = ${listed}(${repr(design.source)})`,
    'by_name = {item["name"]: item for item in entries}',
    `entry = by_name[${repr(design.id)}]`,
  ];
  if (factors) {
    lines.push(
      "",
      "# The factors: their names, ranges and units.",
      'columns = ["name", "low", "high", "units"]',
      'factors = pd.DataFrame(entry["factors"], columns=columns)',
      "print(factors)",
    );
  }
  if (runs) {
    lines.push(
      "",
      "# The runs, in the units and the order the source prints them.",
      'runs = entry["runs"]',
      'table = pd.DataFrame(runs["rows"], columns=runs["columns"])',
      "print(table)",
    );
  } else {
    lines.push("", "# The source prints no table of runs: this is what it says of the design.", 'print(entry["design"])');
  }
  return lines.join("\n");
}

const csvCell = (value) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

/** The design's coded runs, and each run's printed results, as a CSV file's text. */
export function csvText(detail) {
  const names = (detail.factors ?? []).map((f) => f.name);
  const responses = Object.keys(detail.response_values ?? {});
  const lines = [[...names.map((n) => `${n} (coded)`), ...responses].map(csvCell).join(",")];
  (detail.coded_runs ?? []).forEach((row, i) => {
    lines.push([...row, ...responses.map((r) => detail.response_values[r][i])].map(csvCell).join(","));
  });
  return `${lines.join("\n")}\n`;
}

/** Check the catalog's index as read from the server: a file of another format is refused, not guessed at. */
export function checkIndex(data) {
  if (!data || typeof data !== "object" || data.version !== CATALOG_VERSION) {
    throw new Error(`The catalog is in format ${JSON.stringify(data?.version)}; this page reads format ${CATALOG_VERSION}.`);
  }
  if (JSON.stringify(data.fields) !== JSON.stringify(INDEX_FIELDS)) throw new Error("The catalog's fields are not the ones this page reads.");
  if (!Array.isArray(data.designs)) throw new Error("The catalog lists no designs.");
  markInSource(data.designs);
  return data;
}

/**
 * Mark each design that some source prints: a published design, and a generated one of the same
 * class as a published one (see sameDesigns). A generated design left unmarked is one no source in
 * the catalogue prints.
 */
export function markInSource(designs) {
  const printed = new Set(designs.filter((d) => d.collection === "published" && d.design_class).map((d) => d.design_class));
  for (const d of designs) d.in_source = d.collection === "published" || printed.has(d.design_class);
  return designs;
}

// ---------------------------------------------------------------- designs side by side
/** The most designs the comparison sets side by side. */
export const COMPARE_MOST = 4;

/** Why a design cannot join the designs being compared, in words; empty when it can. */
export function compareRefusal(catalog, ids, id) {
  const design = catalog.designs.find((d) => d.id === id);
  if (!design?.compared) {
    return "the catalog compares designs whose factors are all numeric, with at most 12 factors and 1000 runs";
  }
  if (ids.length >= COMPARE_MOST) return `the comparison holds ${COMPARE_MOST} designs at most`;
  const first = catalog.designs.find((d) => d.id === ids[0]);
  if (first && first.n_factors !== design.n_factors) {
    return `the designs compared have ${plural(first.n_factors, "factor")}, and this one has ${design.n_factors}`;
  }
  return "";
}

/** The list of ids with this one added, or taken out when it is there. */
export function toggled(ids, id) {
  return ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
}

/** The largest model every design compared fits: the model the comparison opens on. */
export function sharedModel(scored) {
  return ["quadratic", "interactions", "main_effects"].find((m) => scored.every((s) => s?.[m]?.fits)) ?? "main_effects";
}

const COUNTS = new Set(["runs", "residual_df"]);

/** A measure's value as compare_designs' table prints it: counts whole, the rest to four significant figures. */
export function shownValue(key, values) {
  const value = values?.[key];
  if (key === "fits") return value ? "yes" : `no (rank ${values?.rank})`;
  if (value === null || value === undefined) return "";
  return COUNTS.has(key) ? String(value) : Number(value).toPrecision(4);
}

/** The comparison's rows, one per measure: its label, the direction preferred and each design's value. */
export function compareRows(measures, scored, model) {
  const arrows = { higher: "↑", lower: "↓" };
  return measures.map((m) => ({ ...m, arrow: arrows[m.direction] ?? "", cells: scored.map((s) => shownValue(m.key, s?.[model])) }));
}

// ---------------------------------------------------------------- a design's projections
/** The distinct settings of the chosen factors, each with how many runs land on it. */
export function projected(runs, columns) {
  const held = new Map();
  for (const row of runs) {
    const point = columns.map((c) => row[c]);
    if (point.some((v) => v === null || v === undefined)) continue;
    const key = point.map((v) => Math.round(v * 1e4)).join(",");
    if (held.has(key)) held.get(key).count += 1;
    else held.set(key, { point, count: 1 });
  }
  return [...held.values()];
}

/** The rank of a matrix, by Gaussian elimination with partial pivoting. */
export function rank(matrix, tolerance = 1e-8) {
  const a = matrix.map((row) => [...row]);
  const rows = a.length;
  const cols = rows ? a[0].length : 0;
  const scale = Math.max(1, ...a.map((row) => Math.max(...row.map(Math.abs))));
  let found = 0;
  for (let c = 0; c < cols && found < rows; c++) {
    let pivot = found;
    for (let i = found + 1; i < rows; i++) if (Math.abs(a[i][c]) > Math.abs(a[pivot][c])) pivot = i;
    if (Math.abs(a[pivot][c]) <= tolerance * scale) continue;
    [a[found], a[pivot]] = [a[pivot], a[found]];
    for (let i = found + 1; i < rows; i++) {
      const f = a[i][c] / a[found][c];
      if (f !== 0) for (let j = c; j < cols; j++) a[i][j] -= f * a[found][j];
    }
    found += 1;
  }
  return found;
}

/** The full quadratic's model matrix in the chosen factors: intercept, main effects, interactions, squares. */
export function quadraticRows(runs, columns) {
  return runs
    .filter((row) => columns.every((c) => row[c] !== null && row[c] !== undefined))
    .map((row) => {
      const x = columns.map((c) => row[c]);
      const out = [1, ...x];
      for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) out.push(x[i] * x[j]);
      return [...out, ...x.map((v) => v * v)];
    });
}

/**
 * What a projection holds: its distinct settings, whether it estimates the full quadratic, its corners, and the runs
 * on each point: ``even``, the number every point holds when they all hold the same, else null, and ``most``.
 */
export function projectionSummary(runs, columns) {
  const points = projected(runs, columns);
  const matrix = quadraticRows(runs, columns);
  const terms = ((columns.length + 1) * (columns.length + 2)) / 2;
  const found = rank(matrix);
  const corners = points.filter((p) => p.point.every((v) => Math.abs(Math.abs(v) - 1) < 1e-6)).length;
  const counts = points.map((p) => p.count);
  const even = counts.length && counts.every((c) => c === counts[0]) ? counts[0] : null;
  return {
    distinct: points.length,
    terms,
    rank: found,
    quadratic: found === terms,
    corners,
    allCorners: corners === 2 ** columns.length,
    even,
    most: Math.max(0, ...counts),
    single: counts.includes(1),
  };
}

const joined = (names) => (names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`);

/** A projection's summary, in a sentence or two. */
export function projectionText(summary, names) {
  const fits = summary.quadratic
    ? `estimate the full quadratic in them (${summary.terms} terms)`
    : `do not estimate the full quadratic in them: its ${summary.terms} terms have rank ${summary.rank}`;
  const lines = [`In ${joined(names)}, the runs sit at ${plural(summary.distinct, "distinct setting")} and ${fits}.`];
  if (summary.allCorners) {
    lines.push(`They hold all ${summary.corners} corners: a full 2^${names.length} factorial in these factors.`);
  }
  // The plot labels a point with its runs only when the points differ; when they do not, one sentence says it.
  if (summary.even > 1) lines.push(`Every point holds ${summary.even} runs.`);
  else if (summary.even === null) {
    lines.push(`A label such as ${summary.most}× counts the runs at a point${summary.single ? "; a point with no label holds one run" : ""}.`);
  }
  return lines.join(" ");
}

// ---------------------------------------------------------------- a design's correlations
/**
 * The absolute correlation between a design's main effects, squares and two-factor interactions, as
 * compare_designs reads them (its correlations) and names them: each column centred and scaled to
 * length 1, in that order. A column that does not vary, such as the square of a two-level factor, has
 * no correlation: its row and column are null, as the app's correlation map leaves them blank.
 */
export function correlationMap(runs, names) {
  const rows = runs.filter((row) => row.every((v) => v !== null && v !== undefined));
  const k = names.length;
  const columns = [];
  for (let j = 0; j < k; j++) columns.push({ term: names[j], kind: "main", values: rows.map((r) => r[j]) });
  for (let j = 0; j < k; j++) columns.push({ term: `I(${names[j]} ** 2)`, kind: "square", values: rows.map((r) => r[j] ** 2) });
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) columns.push({ term: `${names[i]}:${names[j]}`, kind: "interaction", values: rows.map((r) => r[i] * r[j]) });
  }
  const n = rows.length;
  const units = columns.map(({ values }) => {
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const centred = values.map((v) => v - mean);
    const norm = Math.sqrt(centred.reduce((a, v) => a + v * v, 0));
    return norm > 1e-9 * Math.sqrt(n) ? centred.map((v) => v / norm) : null;
  });
  const matrix = units.map((a) =>
    units.map((b) => {
      if (!a || !b) return null;
      const v = Math.min(1, Math.abs(a.reduce((sum, x, i) => sum + x * b[i], 0)));
      return v < 1e-9 ? 0 : v;
    }),
  );
  return { terms: columns.map((c) => c.term), kinds: columns.map((c) => c.kind), matrix };
}

/** The largest |r| between two different second-order terms (squares and interactions); null without a pair. */
export function largestSecondOrderR(map) {
  let most = null;
  map.kinds.forEach((a, i) =>
    map.kinds.forEach((b, j) => {
      const r = map.matrix[i][j];
      if (i !== j && a !== "main" && b !== "main" && r !== null) most = Math.max(most ?? 0, r);
    }),
  );
  return most;
}

// ---------------------------------------------------------------- the errata
/** The errata a reader asks for: of one source or all, the ones whose reason mentions rounding or not, with words. */
export function errataShown(errata, { source = "", words = "", rounding = true } = {}) {
  const wanted = String(words).toLowerCase().split(/\s+/).filter(Boolean);
  return errata.filter((e) => {
    if ((source && e.source !== source) || (!rounding && e.rounding)) return false;
    const text = [e.id, e.what, e.term, e.printed, e.where, e.reason, e.quote].filter(Boolean).join(" ").toLowerCase();
    return wanted.every((w) => text.includes(w));
  });
}

/** How many errata each source has. */
export function errataCounts(errata) {
  const counts = {};
  for (const e of errata) counts[e.source] = (counts[e.source] ?? 0) + 1;
  return counts;
}

// ---------------------------------------------------------------- where a design leads
/**
 * The families a design can lead to, each with how and the address of the catalog filtered to its
 * designs: of at most as many factors (a follow-up keeps the factors that matter), or at least as
 * many when the step is to more factors.
 */
export function nextSteps(design, catalog) {
  const k = design.n_factors;
  return (catalog.families?.[design.family]?.next ?? []).map((step) => {
    const factors = !k ? [null, null] : /more factors/i.test(step.how) ? [k, null] : [null, k];
    const filters = { ...structuredClone({ ...DEFAULT_STATE }), families: [step.family], factors };
    return { ...step, label: catalog.families?.[step.family]?.label ?? step.family, hash: toHash(filters) };
  });
}

// ---------------------------------------------------------------- limits on the settings
/**
 * The letters a limit names a design's factors by, in coded units: A for its first factor, B for its
 * second, and so on, without I, as the library names factors (A to H, then J).
 */
export const LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

/** The most characters of a limit typed in the catalog. */
export const LIMIT_LENGTH = 200;

/** How far a run may sit past a limit, in the limit's own units, and still meet it (bootstrap.LIMIT_TOLERANCE). */
const LIMIT_TOLERANCE = 1e-9;

const SIGNS = {
  "<=": (a, b) => a <= b + LIMIT_TOLERANCE,
  ">=": (a, b) => a >= b - LIMIT_TOLERANCE,
  "<": (a, b) => a < b + LIMIT_TOLERANCE,
  ">": (a, b) => a > b - LIMIT_TOLERANCE,
};

/** The symbols a reader may type or paste for an operator, written as the page reads them. */
const TYPED = [[/≤/g, "<="], [/≥/g, ">="], [/[−–]/g, "-"], [/[×·]/g, "*"], [/÷/g, "/"], [/\*\*/g, "^"]];

function limitTokens(text) {
  let source = text;
  for (const [pattern, written] of TYPED) source = source.replace(pattern, written);
  const pattern = /\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?)|([A-Z])|(<=|>=|<|>|[-+*/^()]))\s*/y;
  const out = [];
  for (let at = 0; at < source.length; ) {
    pattern.lastIndex = at;
    const m = pattern.exec(source);
    if (!m || pattern.lastIndex === at) throw new Error(`it cannot be read from "${source.slice(at).trim().slice(0, 12)}"`);
    if (m[1] !== undefined) out.push({ kind: "number", value: Number(m[1]) });
    else if (m[2] !== undefined) {
      const index = LETTERS.indexOf(m[2]);
      if (index < 0) throw new Error(`${m[2]} names no factor: the factors are A, B, C and so on, without I`);
      out.push({ kind: "factor", index });
    } else if (m[3] !== undefined) out.push({ kind: "op", value: m[3] });
    at = pattern.lastIndex;
  }
  return out;
}

/** Read the tokens of a limit into its sides and the signs between them, by recursive descent. */
function limitTree(tokens) {
  let i = 0;
  const op = (value) => tokens[i]?.kind === "op" && tokens[i].value === value && ++i;
  const startsTerm = (t) => t && (t.kind === "factor" || (t.kind === "op" && t.value === "("));
  function primary() {
    const t = tokens[i++];
    if (!t) throw new Error("it ends too soon");
    if (t.kind === "number") return startsTerm(tokens[i]) ? { kind: "*", a: { kind: "number", value: t.value }, b: power() } : { kind: "number", value: t.value };
    if (t.kind === "factor") return { kind: "factor", index: t.index };
    if (t.value === "(") {
      const inside = sum();
      if (!op(")")) throw new Error("a bracket is not closed");
      return inside;
    }
    throw new Error(`"${t.value}" cannot start a term`);
  }
  function power() {
    const base = primary();
    return op("^") ? { kind: "^", a: base, b: unary() } : base;
  }
  function unary() {
    if (op("-")) return { kind: "neg", a: unary() };
    if (op("+")) return unary();
    return power();
  }
  function product() {
    let a = unary();
    for (;;) {
      if (op("*")) a = { kind: "*", a, b: unary() };
      else if (op("/")) a = { kind: "/", a, b: unary() };
      else return a;
    }
  }
  function sum() {
    let a = product();
    for (;;) {
      if (op("+")) a = { kind: "+", a, b: product() };
      else if (op("-")) a = { kind: "-", a, b: product() };
      else return a;
    }
  }
  const sides = [sum()];
  const signs = [];
  while (tokens[i]?.kind === "op" && Object.hasOwn(SIGNS, tokens[i].value)) {
    signs.push(tokens[i++].value);
    sides.push(sum());
  }
  if (i < tokens.length) throw new Error(`"${tokens[i].value ?? ""}" is not where it can be read`);
  if (!signs.length) throw new Error("write <= for at most, or >= for at least");
  return { sides, signs };
}

function factorsOf(node, found = new Set()) {
  if (node.kind === "factor") found.add(node.index);
  for (const child of [node.a, node.b]) if (child) factorsOf(child, found);
  return found;
}

/**
 * Read a limit on the coded settings, such as "A + B <= 1" or "-0.5 <= A - C <= 0.5": factors by
 * their letters, numbers, + - * / ^ and brackets, and one or more of <=, >=, < and >. Nothing is
 * evaluated as code. Returns {ok: true, text, sides, signs, factors} or {ok: false, error}.
 */
export function parseLimit(text) {
  const written = String(text ?? "").trim();
  if (!written) return { ok: false, error: "" };
  if (written.length > LIMIT_LENGTH) return { ok: false, error: `the limit is longer than ${LIMIT_LENGTH} characters` };
  try {
    const { sides, signs } = limitTree(limitTokens(written));
    const factors = [...sides.reduce((found, side) => factorsOf(side, found), new Set())].sort((a, b) => a - b);
    if (!factors.length) return { ok: false, error: "the limit names no factor" };
    return { ok: true, text: written, sides, signs, factors };
  } catch (err) {
    return { ok: false, error: `the limit cannot be read: ${err.message}` };
  }
}

function value(node, x) {
  switch (node.kind) {
    case "number":
      return node.value;
    case "factor":
      return x[node.index];
    case "neg":
      return -value(node.a, x);
    case "+":
      return value(node.a, x) + value(node.b, x);
    case "-":
      return value(node.a, x) - value(node.b, x);
    case "*":
      return value(node.a, x) * value(node.b, x);
    case "/":
      return value(node.a, x) / value(node.b, x);
    default:
      return value(node.a, x) ** value(node.b, x);
  }
}

/** Whether a run, its coded settings in factor order, meets a limit read by parseLimit. */
export function meets(limit, run) {
  const values = limit.sides.map((side) => value(side, run));
  return limit.signs.every((sign, i) => SIGNS[sign](values[i], values[i + 1]));
}

/**
 * Judge a design's runs against a limit: {judged: true, outside: [run indices]} when every factor the
 * limit names is one of the design's numeric factors with every run's setting given, else
 * {judged: false, why}.
 */
export function limitOutcome(limit, runs, kinds) {
  if (!runs?.length) return { judged: false, why: "the catalog holds no table of its runs" };
  const beyond = limit.factors.find((j) => j >= kinds.length);
  if (beyond !== undefined) return { judged: false, why: `the limit names ${LETTERS[beyond]}, and the design has ${plural(kinds.length, "factor")}` };
  const labelled = limit.factors.find((j) => kinds[j] === "categorical");
  if (labelled !== undefined) return { judged: false, why: `${LETTERS[labelled]} is a categorical factor, which has no coded value between its levels` };
  if (runs.some((run) => limit.factors.some((j) => run[j] === null || run[j] === undefined))) {
    return { judged: false, why: "a run's setting of a factor the limit names is not printed" };
  }
  return { judged: true, outside: runs.flatMap((run, i) => (meets(limit, run) ? [] : [i])) };
}

/** A limit's outcome on a design, in a sentence. */
export function limitLine(limit, outcome, n) {
  if (!outcome.judged) return `The limit ${limit.text} cannot be judged on this design: ${outcome.why}.`;
  if (!outcome.outside.length) return `Every one of its ${plural(n, "run")} meets the limit ${limit.text}.`;
  const shown = outcome.outside.map((i) => String(i + 1));
  const which = shown.length === 1 ? `Run ${shown[0]} is` : `Runs ${joined(shown)} are`;
  return `${which} outside the limit ${limit.text}, of its ${plural(n, "run")}.`;
}

const number = (x) => String(Number(x.toPrecision(12)));

/**
 * Write a limit on the coded settings in a design's factors and their own units, as Factorial reads
 * a limit: each letter becomes ((name - centre) / half-range). Every operation is bracketed, so the
 * text means what the tree means.
 */
export function limitInUnits(limit, factors) {
  const coded = (j) => {
    const f = factors[j];
    const centre = (f.low + f.high) / 2;
    const half = (f.high - f.low) / 2;
    if (centre === 0) return half === 1 ? f.name : `(${f.name} / ${number(half)})`;
    const shifted = `${f.name} ${centre < 0 ? "+" : "-"} ${number(Math.abs(centre))}`;
    return half === 1 ? `(${shifted})` : `((${shifted}) / ${number(half)})`;
  };
  const write = (node) => {
    switch (node.kind) {
      case "number":
        return number(node.value);
      case "factor":
        return coded(node.index);
      case "neg":
        return `(-${write(node.a)})`;
      case "^":
        return `(${write(node.a)} ** ${write(node.b)})`;
      default:
        return `(${write(node.a)} ${node.kind} ${write(node.b)})`;
    }
  };
  const sides = limit.sides.map((side) => write(side).replace(/^\((.*)\)$/, (whole, inner) => (balanced(inner) ? inner : whole)));
  return sides.map((side, i) => (i ? ` ${limit.signs[i - 1]} ${side}` : side)).join("");
}

/** Whether a text's brackets pair up, so the brackets around it can be dropped. */
function balanced(text) {
  let depth = 0;
  for (const c of text) {
    depth += c === "(" ? 1 : c === ")" ? -1 : 0;
    if (depth < 0) return false;
  }
  return depth === 0;
}
