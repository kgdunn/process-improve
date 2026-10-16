// The design catalog's rules, apart from the page: what a search asks for, which designs pass the
// filters, how many each filter would leave, what the address keeps, and the picture of a design's
// runs. catalog.js draws the page from these; nothing here touches the document, so tests/web/js
// runs it under Node.

/** The fields of each design in catalog/index.json, in order: the library's facets, then the build's. */
export const INDEX_FIELDS = [
  "id", "title", "collection", "kind", "family", "design_type", "source", "year", "added", "where",
  "n_runs", "n_factors", "n_numeric", "n_categorical", "n_mixture", "levels", "n_centre", "runs_per_parameter",
  "models", "blocked", "n_blocks", "split_plot", "has_responses", "n_responses", "checks", "glyph",
  "survives_run_loss", "critical_runs", "worst_d_ratio", "quadratic_projection", "projectivity", "design_class",
  "regenerable", "open", "detail",
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
  years: [null, null],
  addedSince: "",
  sort: "default",
  design: "",
});

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
    years: narrower(state.years, asked.years),
    addedSince: state.addedSince,
    words: asked.words,
  };
}

/** The text a word of a search is looked for in: the design's title, family, source and where it is printed. */
export function haystack(design, catalog) {
  const source = catalog.sources?.[design.source];
  const family = catalog.families?.[design.family];
  return [design.id, design.title, design.where, family?.label, source?.cite, source?.title, design.year]
    .filter((v) => v !== null && v !== undefined)
    .join(" ")
    .toLowerCase();
}

/** Each filter, by name, as a test of one design. */
const TESTS = {
  collection: (d, f) => f.collection === "both" || d.collection === f.collection,
  families: (d, f) => !f.families.length || f.families.includes(d.family),
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
  years: (d, f) => between(d.year, f.years),
  addedSince: (d, f) => !f.addedSince || (d.added !== null && d.added >= f.addedSince),
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
  for (let from = first; from <= last; from += width) {
    bins.push({ from, to: from + width - 1, count: years.filter((y) => y >= from && y < from + width).length });
  }
  return bins;
}

// ---------------------------------------------------------------- the address
const KEYS = {
  collection: "col", query: "q", families: "fam", runs: "runs", factors: "k", model: "model", categorical: "cat",
  blocked: "blk", splitPlot: "sp", mixture: "mix", hasData: "data", recomputed: "rec", differs: "err", opens: "open",
  years: "year", addedSince: "added", sort: "sort", design: "d",
};

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
      if (name === "families" ? value.length : value[0] !== null || value[1] !== null) {
        params.set(key, name === "families" ? value.join(",") : writeRange(value));
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
 * wrong kind, a family the catalog does not have, or an unknown key is ignored.
 */
export function fromHash(text, families = {}) {
  const params = new URLSearchParams(String(text ?? "").replace(/^#/, ""));
  const state = structuredClone({ ...DEFAULT_STATE });
  const read = (name) => params.get(KEYS[name]);
  if (COLLECTIONS.includes(read("collection"))) state.collection = read("collection");
  state.query = (read("query") ?? "").slice(0, 200);
  state.families = (read("families") ?? "").split(",").filter((f) => Object.hasOwn(families, f));
  for (const name of ["runs", "factors", "categorical", "years"]) state[name] = readRange(read(name));
  if (Object.hasOwn(MODELS, read("model"))) state.model = read("model");
  for (const name of ["blocked", "splitPlot", "mixture", "hasData", "recomputed", "differs", "opens"]) {
    state[name] = read(name) === "1";
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(read("addedSince") ?? "")) state.addedSince = read("addedSince");
  if (Object.hasOwn(SORTS, read("sort"))) state.sort = read("sort");
  if (/^[a-z0-9][a-z0-9-]{0,120}$/.test(read("design") ?? "")) state.design = read("design");
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
  const parts = [catalog.families?.[design.family]?.label ?? design.family];
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
  return parts.join(" · ");
}

/** What the catalog can say of a published design's printed numbers. */
export function checkLine(checks) {
  if (!checks || !checks.printed) return "";
  if (!checks.checked) return `${plural(checks.printed, "printed number")}, none recomputed`;
  const agree = `${checks.agree} of ${plural(checks.checked, "recomputed number")} agree`;
  return checks.explained ? `${agree}; ${checks.explained} known to differ, with the reason` : agree;
}

/** Where a published design is printed, in a few words: its source's authors and year, and its pages. */
export function sourceLine(design, catalog) {
  const source = catalog.sources?.[design.source];
  if (!source) return "Made by process-improve";
  return [`${source.short} (${source.year})`, design.where].filter(Boolean).join(", ");
}

// ---------------------------------------------------------------- what a design is copied as
const repr = (value) => (typeof value === "string" ? JSON.stringify(value) : String(value));

/** The Python that gives a reader the design: the library call that makes it, or the catalogue entry that holds it. */
export function pythonCode(design, detail) {
  const spec = detail?.spec;
  if (spec && !spec.array) {
    const factors = (detail.factors ?? []).map((f) => {
      const kind = f.type === "mixture" ? ', type="mixture"' : "";
      return `    Factor(name=${repr(f.name)}, low=${repr(f.low)}, high=${repr(f.high)}${kind}),`;
    });
    const options = Object.entries(spec.options ?? {}).map(([key, value]) => `, ${key}=${repr(value)}`).join("");
    return [
      "from process_improve.experiments import Factor, generate_design",
      "",
      "factors = [",
      ...factors,
      "]",
      `design = generate_design(factors, design_type=${repr(spec.design_type)}${options})`,
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
  const listed = detail?.listed === "designs" ? "reference_designs" : "reference_experiments";
  return [
    `from process_improve.experiments.reference_catalogue import ${listed}`,
    "",
    `entry = next(e for e in ${listed}(${repr(design.source)}) if e["name"] == ${repr(design.id)})`,
    'print(entry["factors"], entry["runs"], sep="\\n")',
  ].join("\n");
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
  return data;
}
