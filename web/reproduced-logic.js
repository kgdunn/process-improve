// The rules of the reproduction page (catalog/reproduced/index.html), apart from the page: the states a
// printed number and a case are in and the reasons a number is no match, what the page counts, how far off the
// numbers that differ are, which cases a search keeps, and what the address keeps. catalog/reproduction.json,
// which the build writes, holds every case.

/**
 * The tiers, best first, as the library's REPRODUCTION_TIERS has them: the page counts the numbers that differ
 * within each one's limit (sizes), and lists cases by their state, not by tier.
 */
export const TIER_KEYS = ["reproduced", "within_1", "within_5", "not_reproduced"];

/**
 * The causes the catalogue records, as the page names them: why a number differs from its source, and why
 * a printed number is not recomputed. The build's JSON holds what each means; these are its short names.
 */
export const CAUSES = {
  differs: {
    source_error: "A misprint or slip in the book",
    incorrect_option: "Multiple choice: wrong on purpose",
    source_rounding: "The book rounds sooner",
    method: "Another method or convention",
    library: "A limit of Factorial",
    unexplained: "Not known yet",
  },
  unchecked: {
    library_gap: "Not in Factorial yet",
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
 * The states a printed number is in, in the order of the page's bar. ``gap`` is apart from the rest: its
 * numbers are not recomputed, so they are no part of those that match or differ.
 */
export const STATES = [
  { key: "match", label: "Matches the book" },
  { key: "known", label: "Differs, and we know why" },
  { key: "quiz", label: "Multiple choice, wrong on purpose" },
  { key: "open", label: "Differs, reason not known yet" },
  { key: "gap", label: "Not in Factorial yet" },
];

/** Why a printed number is no match: each cause the page counts, with its state, in the order of the bar. */
export const REASONS = [
  { cause: "source_rounding", state: "known" },
  { cause: "source_error", state: "known" },
  { cause: "method", state: "known" },
  { cause: "library", state: "known" },
  { cause: "incorrect_option", state: "quiz" },
  { cause: "unexplained", state: "open" },
  { cause: "library_gap", state: "gap" },
].map((reason) => ({ ...reason, name: CAUSES.differs[reason.cause] ?? CAUSES.unchecked[reason.cause] }));

/** The state of a number that differs, from its cause: a cause the page does not know is a reason not known yet. */
const stateOf = (number) => REASONS.find((reason) => reason.cause === number.cause)?.state ?? "open";

/**
 * The state a case is in, which its list is named by, and a book's page names an example by (``example_state`` in
 * scripts/build_web_app.py): that of its number that comes last in the order of STATES, the bar's order. So a case
 * matches the book only when every number recomputed does, and a reason not known yet is never hidden behind a
 * known one. Its numbers not in the library yet are apart from those recomputed: they leave it as it is.
 */
export function caseState(c) {
  const order = STATES.map((state) => state.key);
  return c.differs.map(stateOf).reduce((last, state) => (order.indexOf(state) > order.indexOf(last) ? state : last), "match");
}

/** A list of cases for each state a case can be in (caseState), with what puts a case in it. */
export const CASE_STATES = [
  { key: "match", means: "Every number Factorial worked again matches the book." },
  { key: "known", means: "A number differs, and the reason is known for each one that does." },
  {
    key: "quiz",
    means: "A number is a multiple-choice option that is wrong on purpose. Any other that differs has a known reason.",
  },
  { key: "open", means: "A number differs, and the reason is not known yet." },
].map((state) => ({ ...state, label: STATES.find((s) => s.key === state.key).label }));

const CASE_KEYS = CASE_STATES.map((state) => state.key);

/**
 * Count the printed numbers of ``cases``: those recomputed (``checked``), those of them that match, and
 * for each cause of REASONS those that are no match. A number not in the library yet is counted from
 * every case, recomputed or not; it is no part of ``checked``.
 */
export function tally(cases) {
  const out = { checked: 0, match: 0, causes: Object.fromEntries(REASONS.map((reason) => [reason.cause, 0])) };
  for (const c of cases) {
    out.causes.library_gap += c.unchecked?.library_gap ?? 0;
    out.checked += c.checked;
    out.match += c.checked - c.differs.length;
    for (const number of c.differs) out.causes[number.cause] = (out.causes[number.cause] ?? 0) + 1;
  }
  return out;
}

/** A tally's numbers by state (see STATES): those that match, and each reason's under its state. */
export function byState(counts) {
  const out = Object.fromEntries(STATES.map((state) => [state.key, 0]));
  out.match = counts.match;
  for (const reason of REASONS) out[reason.state] += counts.causes[reason.cause];
  return out;
}

/**
 * The reasons a tally has numbers for, as a cascade: each with its count (``n``), where its bar starts
 * (``from``) and how long it is (``share``), as shares of all of them (``whole``), so a bar starts where
 * the one before it ended. ``differ`` counts those that differ, which leaves out those not in the library yet.
 */
export function cascade(counts) {
  const whole = REASONS.reduce((sum, reason) => sum + counts.causes[reason.cause], 0);
  let before = 0;
  const steps = REASONS.filter((reason) => counts.causes[reason.cause]).map((reason) => {
    const n = counts.causes[reason.cause];
    const step = { ...reason, n, from: before / whole, share: n / whole };
    before += n;
    return step;
  });
  return { steps, whole, differ: whole - counts.causes.library_gap };
}

/** What an example's box shows: a number as a source prints one, with no more than 14 characters. */
const A_NUMBER = /^[-−+]?[\d,]*\.?\d+%?$/;
const FITS = 14;

/**
 * The numbers of ``cases`` that are no match for one ``cause`` and fit an example's box (a list of runs or
 * an alias chain does not): what the source prints (``printed``) and the library's value (``found``, null
 * for a number not in the library yet), with where it is printed and what it is, by its plain name in ``names``
 * (the JSON's ``quantities``). They come one source after another in turn, so the next example is from another
 * source.
 */
export function examples(cases, cause, names = {}) {
  const fits = (value) => typeof value === "string" && value.length <= FITS && A_NUMBER.test(value);
  const lists = new Map();
  for (const c of cases) {
    const numbers = cause === "library_gap" ? (c.missing ?? []) : c.differs.filter((number) => number.cause === cause);
    for (const number of numbers) {
      const found = cause === "library_gap" ? null : number.found;
      if (!number.where || !fits(number.printed) || !(found === null || fits(found))) continue;
      if (!lists.has(c.source)) lists.set(c.source, []);
      const what = number.what === "other" ? String(number.term ?? "") : quantityName(number.what, names);
      lists.get(c.source).push({ source: c.source, where: number.where, what, printed: number.printed, found });
    }
  }
  const out = [];
  const longest = Math.max(0, ...[...lists.values()].map((list) => list.length));
  for (let i = 0; i < longest; i += 1) for (const list of lists.values()) if (i < list.length) out.push(list[i]);
  return out;
}

/**
 * How far off the numbers that differ are: how many fall within each tier's limit, leaving out those the
 * library is right about (see libraryRight), whose difference measures the source.
 */
export function sizes(cases, limits, right = []) {
  const out = Object.fromEntries(TIER_KEYS.slice(1).map((key) => [key, 0]));
  for (const c of cases) for (const number of c.differs) if (!libraryRight(number, right)) out[tierOf(number.error, limits)] += 1;
  return out;
}

/**
 * The tier a number that differs falls in, by the largest relative difference each tier allows (``limits``,
 * from the JSON). It differs by more than the source's rounding, so it is never in the first tier, even when
 * the six figures the library reports make its difference read as 0.
 */
export function tierOf(error, limits) {
  if (error === null || error === undefined) return "not_reproduced";
  return TIER_KEYS.slice(1).find((key) => limits[key] === null || error <= limits[key]);
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

/**
 * What a number is, by its plain name in ``names`` (the JSON's ``quantities``, which the build takes from
 * web/quantities.json), or by the catalogue's own word where it has none.
 */
export const quantityName = (what, names = {}) => names[what] ?? String(what ?? "").replaceAll("_", " ");

/**
 * The quantity a number is, in a few words, as ``_quantity`` in scripts/build_web_app.py names it: what it is and
 * of which term. The catalogue calls a quantity it has no name for "other", and says what it is in its term.
 */
export const quantity = (number, names = {}) =>
  [number.what === "other" ? "" : quantityName(number.what, names), number.term].filter((part) => part).join(": ");

/**
 * The text a search reads of a case: its title, id, kind, source, each differing number with its reason and
 * the name of its cause, and the reason of each number the library cannot compute yet.
 */
function haystack(c, sources, names) {
  const source = sources[c.source] ?? {};
  const numbers = c.differs.map((n) => `${quantity(n, names)} ${n.reason} ${n.where} ${CAUSES.differs[n.cause] ?? ""}`);
  const missing = (c.missing ?? []).map((n) => `${quantity(n, names)} ${n.reason}`);
  return [c.title, c.id, c.kind, c.source, source.short, source.title, ...numbers, ...missing].join(" ").toLowerCase();
}

/**
 * The cases a reader's filters keep: every word of ``query`` in the case's text, its ``source``, the
 * ``form`` of its source and its ``kind`` where chosen. A case with no recomputed number is kept only
 * when ``all`` is asked for: it is no reproduction, though it may hold numbers the library cannot compute yet.
 * ``names`` are the plain names of the quantities (see quantity), which the search reads.
 */
export function filterCases(cases, sources, { query = "", source = "", form = "", kind = "" } = {}, { all = false, names = {} } = {}) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return cases.filter(
    (c) =>
      (all || c.tier) &&
      (!source || c.source === source) &&
      (!form || sources[c.source]?.form === form) &&
      (!kind || kindOf(c) === kind) &&
      (!words.length || words.every((word) => haystack(c, sources, names).includes(word))),
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

/** Counts of cases by state and of printed numbers, recomputed (``numbers``) and agreeing (``agree``), from none. */
const blank = () => ({ total: 0, states: Object.fromEntries(CASE_KEYS.map((key) => [key, 0])), numbers: 0, agree: 0 });

function count(into, c) {
  into.total += 1;
  into.states[caseState(c)] += 1;
  into.numbers += c.checked;
  into.agree += c.checked - c.differs.length;
}

/** The orders the page lists the sources in: none ranks them by how much of them reproduces. */
export const SOURCE_ORDERS = {
  year: (a, b) => (a.year ?? 9999) - (b.year ?? 9999) || a.short.localeCompare(b.short),
  name: (a, b) => a.short.localeCompare(b.short) || (a.year ?? 9999) - (b.year ?? 9999),
  numbers: (a, b) => b.numbers - a.numbers || a.short.localeCompare(b.short),
};

/**
 * Each source's counts: its cases by state, and its printed numbers recomputed and agreeing (see
 * ``blank``), with its name, year and form, in the ``order`` of SOURCE_ORDERS (oldest first unless told).
 */
export function bySource(cases, sources, order = "year") {
  const rows = new Map();
  for (const c of cases) {
    if (!c.tier) continue;
    if (!rows.has(c.source)) {
      const info = sources[c.source] ?? {};
      rows.set(c.source, { key: c.source, short: info.short ?? c.source, year: info.year ?? null, title: info.title ?? "", form: info.form ?? "other", ...blank() });
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
    const out = blank();
    for (const row of list) {
      out.total += row.total;
      out.numbers += row.numbers;
      out.agree += row.agree;
      for (const key of CASE_KEYS) out.states[key] += row.states[key];
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
 * The cases with a recomputed number in one state (caseState), grouped by source in the order of ``order``
 * (source keys), each group worst case first.
 */
export function groupByState(cases, state, order) {
  const groups = new Map(order.map((key) => [key, []]));
  for (const c of cases) if (c.tier && caseState(c) === state) groups.get(c.source)?.push(c);
  const worst = (c) => (c.worst === null ? Infinity : c.worst);
  return [...groups]
    .filter(([, list]) => list.length)
    .map(([source, list]) => ({ source, cases: list.sort((a, b) => worst(b) - worst(a) || a.title.localeCompare(b.title)) }));
}

/** The clause of a reason that names what the library lacks: "the library has no ..." up to a ";" or ":". */
const LACKS = /\bthe library (?:has no|does not|cannot|lacks|makes no|offers no|works)\b[^;:]*/i;

/**
 * What a number not in the library yet waits on: the clause of its reason that names what the library lacks,
 * so the numbers of one missing method fall together, or the whole reason where that clause is too short to
 * tell one missing method from another ("the library does not compute").
 */
export function missingKey(reason) {
  const clause = LACKS.exec(reason)?.[0].trim().replace(/[,.]$/, "");
  if (!clause || clause.split(/\s+/).length < 6) return reason;
  return clause[0].toUpperCase() + clause.slice(1);
}

/**
 * The numbers the library cannot compute yet, grouped by what they wait on (missingKey), those that would
 * recompute most numbers first: a list of what to add to the library. Each holds its cases.
 */
export function missingByReason(cases) {
  const groups = new Map();
  for (const c of cases) {
    for (const number of c.missing ?? []) {
      const key = missingKey(number.reason);
      if (!groups.has(key)) groups.set(key, { reason: key, numbers: 0, cases: [] });
      const group = groups.get(key);
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
    open: [...new Set(read("open").split(",").filter((key) => CASE_KEYS.includes(key)))],
    focus: AN_ID.test(read("focus")) ? read("focus") : "",
  };
}
