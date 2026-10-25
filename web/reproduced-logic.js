// The rules of the reproduction page (catalog/reproduced/index.html), apart from the page: the tiers a
// case falls in, what the page counts, which cases a search keeps, how far off each difference is, and
// what the address keeps. catalog/reproduction.json, which the build writes, holds every case.

/**
 * The tiers, best first, as the library's REPRODUCTION_TIERS has them. ``mark`` names a tier beside
 * its colour, so a reader who cannot tell the colours apart still can the tiers.
 */
export const TIERS = [
  {
    key: "reproduced",
    label: "Fully reproduced",
    mark: "✓",
    means:
      "Every number the library recomputes agrees with the source to its last printed digit, apart from those the " +
      "source prints in error and quiz options its answer key marks wrong.",
  },
  {
    key: "within_1",
    label: "Within 1%",
    mark: "<1%",
    means: "A number differs by more than the source's rounding, and none by more than 1% of the printed value.",
  },
  {
    key: "within_5",
    label: "Within 5%",
    mark: "<5%",
    means: "The largest difference is more than 1% and at most 5% of the printed value.",
  },
  {
    key: "not_reproduced",
    label: "Unable to reproduce",
    mark: "✕",
    means:
      "A number differs by more than 5%, or a result that is not a number (an alias chain, a list of runs) differs.",
  },
];

export const TIER_KEYS = TIERS.map((tier) => tier.key);

/**
 * The causes the catalogue records, as the page names them: why a number differs from its source, and why
 * a printed number is not recomputed. The build's JSON holds what each means; these are its short names.
 */
export const CAUSES = {
  differs: {
    source_error: "Error in the source",
    incorrect_option: "Wrong option of a quiz",
    source_rounding: "The source's rounding",
    method: "Another method or convention",
    library: "The library",
    unexplained: "Not known yet",
  },
  unchecked: {
    library_gap: "Not in the library yet",
    check_gap: "Not read by a check yet",
    data_not_printed: "Data not printed",
    not_a_result: "Nothing to recompute",
  },
};

/**
 * Whether the library's value is the right one for a number that differs (the JSON's ``library_right``
 * causes): such a number is reproduced, and leaves a case's tier as it is.
 */
export const libraryRight = (number, right) => right.includes(number.cause);

/**
 * The tier a number that differs falls in, by the largest relative difference each tier allows (``limits``,
 * from the JSON). It differs by more than the source's rounding, so it is never in the first tier, even when
 * the six figures the library reports make its difference read as 0.
 */
export function tierOf(error, limits) {
  if (error === null || error === undefined) return "not_reproduced";
  return TIER_KEYS.slice(1).find((key) => limits[key] === null || error <= limits[key]);
}

/**
 * Count the cases and the numbers in each tier. A case counts once, in its own tier; a number that
 * agrees counts as reproduced, so does one whose difference the library is right about (see
 * libraryRight), and any other number that differs in the tier its own difference falls in. A case
 * with no recomputed number is no reproduction, so it is counted apart (``unchecked``).
 */
export function summarise(cases, limits, right = []) {
  const zero = () => Object.fromEntries(TIER_KEYS.map((key) => [key, 0]));
  const out = { cases: zero(), numbers: zero(), recomputed: 0, numbersChecked: 0, unchecked: 0 };
  for (const c of cases) {
    if (!c.tier) {
      out.unchecked += 1;
      continue;
    }
    out.recomputed += 1;
    out.cases[c.tier] += 1;
    out.numbersChecked += c.checked;
    out.numbers.reproduced += c.checked - c.differs.length;
    for (const number of c.differs) {
      out.numbers[libraryRight(number, right) ? "reproduced" : tierOf(number.error, limits)] += 1;
    }
  }
  return out;
}

/** The share ``part`` is of ``whole``, as a percentage with one decimal; "0%" of nothing. */
export function percent(part, whole) {
  if (!whole) return "0%";
  const value = (100 * part) / whole;
  return `${value >= 99.95 || value === 0 ? value.toFixed(0) : value.toFixed(1)}%`;
}

/**
 * A relative difference in words: a percentage with two significant figures, or why it has none.
 * Differences far below the source's rounding are written in full, so "0.00046%" is not read as zero.
 */
export function formatError(error) {
  if (error === null || error === undefined) return "not a number";
  if (error === 0) return "< 0.0001%"; // below the six figures the library reports
  const value = 100 * error;
  if (value >= 100) return `${Math.round(value).toLocaleString("en")}%`;
  return `${Number(value.toPrecision(2))}%`;
}

/** The quantity a number is, in a few words: what it is and of which term. */
export const quantity = (number) =>
  [String(number.what ?? "").replaceAll("_", " "), number.term].filter((part) => part).join(": ");

/**
 * The text a search reads of a case: its title, id, kind, source, each differing number with its reason and
 * the name of its cause, and the reason of each number the library cannot compute yet.
 */
function haystack(c, sources) {
  const source = sources[c.source] ?? {};
  const numbers = c.differs.map((n) => `${quantity(n)} ${n.reason} ${n.where} ${CAUSES.differs[n.cause] ?? ""}`);
  const missing = (c.missing ?? []).map((n) => `${quantity(n)} ${n.reason}`);
  return [c.title, c.id, c.kind, c.source, source.short, source.title, ...numbers, ...missing].join(" ").toLowerCase();
}

/**
 * The cases a reader's filters keep: every word of ``query`` in the case's text, its ``source``, the
 * ``form`` of its source and its ``kind`` where chosen. A case with no recomputed number is kept only
 * when ``all`` is asked for: it is no reproduction, though it may hold numbers the library cannot compute yet.
 */
export function filterCases(cases, sources, { query = "", source = "", form = "", kind = "" } = {}, { all = false } = {}) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return cases.filter(
    (c) =>
      (all || c.tier) &&
      (!source || c.source === source) &&
      (!form || sources[c.source]?.form === form) &&
      (!kind || kindOf(c) === kind) &&
      (!words.length || words.every((word) => haystack(c, sources).includes(word))),
  );
}

/** A case's kind as the page names it: the catalogue writes "case study" two ways, and a design has none. */
export const kindOf = (c) => String(c.kind ?? "design").replace("-", " ");

/** Each kind of case among ``cases``, with how many there are, most first. */
export function kinds(cases) {
  const counts = new Map();
  for (const c of cases) if (c.tier) counts.set(kindOf(c), (counts.get(kindOf(c)) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** Counts of cases by tier and of printed numbers, recomputed (``numbers``) and agreeing (``agree``), from none. */
const tally = () => ({ total: 0, tiers: Object.fromEntries(TIER_KEYS.map((key) => [key, 0])), numbers: 0, agree: 0 });

function count(into, c) {
  into.total += 1;
  into.tiers[c.tier] += 1;
  into.numbers += c.checked;
  into.agree += c.checked - c.differs.length;
}

/** The orders the page lists the sources in: none ranks them by how much of them reproduces. */
export const SOURCE_ORDERS = {
  year: (a, b) => (a.year ?? 9999) - (b.year ?? 9999) || a.short.localeCompare(b.short),
  name: (a, b) => a.short.localeCompare(b.short) || (a.year ?? 9999) - (b.year ?? 9999),
  cases: (a, b) => b.total - a.total || a.short.localeCompare(b.short),
};

/**
 * Each source's counts: its cases by tier, and its printed numbers recomputed and agreeing (see
 * ``tally``), with its name, year and form, in the ``order`` of SOURCE_ORDERS (oldest first unless told).
 */
export function bySource(cases, sources, order = "year") {
  const rows = new Map();
  for (const c of cases) {
    if (!c.tier) continue;
    if (!rows.has(c.source)) {
      const info = sources[c.source] ?? {};
      rows.set(c.source, { key: c.source, short: info.short ?? c.source, year: info.year ?? null, title: info.title ?? "", form: info.form ?? "other", ...tally() });
    }
    count(rows.get(c.source), c);
  }
  return [...rows.values()].sort(SOURCE_ORDERS[order] ?? SOURCE_ORDERS.year);
}

/**
 * The sources' rows grouped by the form their source takes, in the order of ``forms`` (the JSON's, form to
 * its name), each group with its totals, and the totals of all of them. A form with no row is left out.
 */
export function byForm(rows, forms) {
  const sum = (list) => {
    const out = tally();
    for (const row of list) {
      out.total += row.total;
      out.numbers += row.numbers;
      out.agree += row.agree;
      for (const key of TIER_KEYS) out.tiers[key] += row.tiers[key];
    }
    return out;
  };
  const groups = Object.entries(forms)
    .map(([form, label]) => ({ form, label, rows: rows.filter((row) => row.form === form) }))
    .filter((group) => group.rows.length)
    .map((group) => ({ ...group, totals: sum(group.rows) }));
  return { groups, totals: sum(rows) };
}

/**
 * The bands the page counts differences in, each by its largest relative difference, aligned so the
 * tiers' limits (1% and 5%) are edges. A difference with no relative size has a band of its own.
 */
export const BANDS = [
  { upTo: 1e-5, label: "< 0.001%" },
  { upTo: 1e-4, label: "0.001-0.01%" },
  { upTo: 1e-3, label: "0.01-0.1%" },
  { upTo: 3e-3, label: "0.1-0.3%" },
  { upTo: 1e-2, label: "0.3-1%" },
  { upTo: 2e-2, label: "1-2%" },
  { upTo: 5e-2, label: "2-5%" },
  { upTo: 0.2, label: "5-20%" },
  { upTo: 1, label: "20-100%" },
  { upTo: Infinity, label: "> 100%" },
  { upTo: null, label: "Not a number" },
];

/**
 * How many differing numbers of ``cases`` fall in each band, with the tier each band belongs to. A number
 * the library is right about (see libraryRight) is left out: its difference measures the source.
 */
export function errorBands(cases, limits, right = []) {
  const counts = BANDS.map((band) => ({ ...band, count: 0, tier: band.upTo === null ? "not_reproduced" : tierOf(band.upTo, limits) }));
  for (const c of cases) {
    for (const number of c.differs) {
      if (libraryRight(number, right)) continue;
      const at = number.error === null ? BANDS.length - 1 : BANDS.findIndex((band) => band.upTo !== null && number.error <= band.upTo);
      counts[at].count += 1;
    }
  }
  return counts;
}

/** The cases of one tier, grouped by source in the order of ``order`` (source keys), each group worst case first. */
export function groupByTier(cases, tier, order) {
  const groups = new Map(order.map((key) => [key, []]));
  for (const c of cases) if (c.tier === tier) groups.get(c.source)?.push(c);
  const worst = (c) => (c.worst === null ? Infinity : c.worst);
  return [...groups]
    .filter(([, list]) => list.length)
    .map(([source, list]) => ({ source, cases: list.sort((a, b) => worst(b) - worst(a) || a.title.localeCompare(b.title)) }));
}

/**
 * How many numbers of ``cases`` differ for each cause, and how many are not recomputed for each cause, from
 * every case (``all`` filtered or not): the page shows these totals for all the sources together only.
 */
export function causeTotals(cases) {
  const out = { differs: {}, unchecked: {} };
  for (const c of cases) {
    for (const number of c.differs) out.differs[number.cause] = (out.differs[number.cause] ?? 0) + 1;
    for (const [cause, n] of Object.entries(c.unchecked ?? {})) out.unchecked[cause] = (out.unchecked[cause] ?? 0) + n;
  }
  return out;
}

/**
 * The numbers the library cannot compute yet, grouped by the reason the catalogue gives, the reasons that
 * would recompute most numbers first: a list of what to add to the library. Each holds its cases.
 */
export function missingByReason(cases) {
  const groups = new Map();
  for (const c of cases) {
    for (const number of c.missing ?? []) {
      if (!groups.has(number.reason)) groups.set(number.reason, { reason: number.reason, numbers: 0, cases: [] });
      const group = groups.get(number.reason);
      group.numbers += 1;
      if (!group.cases.includes(c)) group.cases.push(c);
    }
  }
  return [...groups.values()].sort((a, b) => b.numbers - a.numbers || a.reason.localeCompare(b.reason));
}

// ---------------------------------------------------------------- the address
const KEYS = { query: "q", source: "src", form: "form", kind: "kind", open: "open", focus: "case" };
const AN_ID = /^[a-z0-9][a-z0-9-]{0,120}$/;

export const DEFAULT_VIEW = { query: "", source: "", form: "", kind: "", open: [], focus: "" };

/** Write the reader's view into the part of the address after "#", leaving out whatever is at its default. */
export function toHash(view) {
  const params = new URLSearchParams();
  if (view.query) params.set(KEYS.query, view.query);
  if (view.source) params.set(KEYS.source, view.source);
  if (view.form) params.set(KEYS.form, view.form);
  if (view.kind) params.set(KEYS.kind, view.kind);
  if (view.open.length) params.set(KEYS.open, view.open.join(","));
  if (view.focus) params.set(KEYS.focus, view.focus);
  const text = params.toString();
  return text ? `#${text}` : "";
}

/** Read the view back from an address, which anyone can write: a value the page does not know is ignored. */
export function fromHash(text, sources = {}, forms = {}) {
  const params = new URLSearchParams(String(text ?? "").replace(/^#/, ""));
  const read = (name) => params.get(KEYS[name]) ?? "";
  return {
    query: read("query").slice(0, 200),
    source: Object.hasOwn(sources, read("source")) ? read("source") : "",
    form: Object.hasOwn(forms, read("form")) ? read("form") : "",
    kind: read("kind").slice(0, 40),
    open: [...new Set(read("open").split(",").filter((key) => TIER_KEYS.includes(key)))],
    focus: AN_ID.test(read("focus")) ? read("focus") : "",
  };
}
