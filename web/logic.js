// The page's own logic, apart from the page: what it counts, decides, formats and sorts.
// Nothing here touches the document, so tests/web/js runs it under Node (node --test), and
// tests/web/page/test_parity.py holds the rules that mirror web/bootstrap.py to it.

/** Spare runs an optimal design gets beyond its model's coefficients; mirrors bootstrap.SPARE_RUNS. */
export const SPARE_RUNS = 4;

/** Count the coefficients, intercept included, of each named model in k factors. */
export function nTerms(k, model) {
  const pairs = (k * (k - 1)) / 2;
  return 1 + k + (model === "main_effects" ? 0 : pairs) + (model === "quadratic" ? k : 0);
}

/** The space-filling designs, made for no model; mirrors bootstrap.SPACE_FILLING. */
export const SPACE_FILLING = ["maximin_lhs"];

/** Runs a factor in a space-filling design, unless the reader gives a number; mirrors bootstrap.SPACE_FILLING_RUNS. */
export const SPACE_FILLING_RUNS = 10;

/**
 * The runs a design that takes a number of runs opens with, as bootstrap.py gives them: SPACE_FILLING_RUNS a factor
 * for a space-filling design, else the coefficients of the model it is made for and SPARE_RUNS more.
 */
export const defaultBudget = (designType, k, model) =>
  SPACE_FILLING.includes(designType) ? SPACE_FILLING_RUNS * k : nTerms(k, model) + SPARE_RUNS;

/** The resolution a fraction in k factors opens on: IV, or III when three factors allow no more. */
export const defaultResolution = (k) => Math.max(3, Math.min(4, k));

/** The resolutions the page offers a fraction, lowest first; mirrors bootstrap.RESOLUTIONS. */
export const RESOLUTIONS = [3, 4, 5];

/** A resolution in roman numerals, as a fraction's is written. */
export const ROMAN = { 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII" };

/** A cell of the trade-off grid whose runs are the full factorial's 2^k. */
export const FULL = "full";

/** The trade-off grid's rows, run counts, and its columns, numbers of factors: a fraction takes 3 to 11 on the page. */
export const TRADE_OFF_RUNS = [4, 8, 16, 32, 64, 128];
export const TRADE_OFF_FACTORS = [3, 4, 5, 6, 7, 8, 9, 10, 11];

const F = FULL;
const _ = null;

/**
 * The trade-off of a two-level fraction, a row per run count and a column per number of factors, as
 * process_improve.experiments.trade_off.trade_off_table finds it: each cell is the resolution of the
 * minimum-aberration fraction, FULL where the runs are the full factorial's, and null where the factors
 * do not fit in the runs. Below the full factorial the runs would only repeat it, which Replicates does,
 * so those cells are null too. tests/web/page/test_parity.py compares every cell with the library's.
 */
export const TRADE_OFF = [
  // 3  4  5  6  7  8  9 10 11 factors
  [3, _, _, _, _, _, _, _, _], // 4 runs
  [F, 4, 3, 3, 3, _, _, _, _], // 8 runs
  [_, F, 5, 4, 4, 4, 3, 3, 3], // 16 runs
  [_, _, F, 6, 4, 4, 4, 4, 4], // 32 runs
  [_, _, _, F, 7, 5, 4, 4, 4], // 64 runs
  [_, _, _, _, F, 8, 6, 5, 5], // 128 runs
];

/** The grid's cell at a run count and a number of factors: a resolution, FULL, or null. */
export function tradeOffCell(runs, k) {
  const row = TRADE_OFF_RUNS.indexOf(runs);
  const column = TRADE_OFF_FACTORS.indexOf(k);
  return row < 0 || column < 0 ? null : TRADE_OFF[row][column];
}

/** What a cell shows: Full, its resolution in roman numerals, or nothing. */
export const cellText = (cell) => (cell === FULL ? "Full" : cell === null ? "" : ROMAN[cell]);

/** The band a cell is coloured by: the full factorial or resolution V and above, IV, III; null for no design. */
export const cellBand = (cell) => (cell === null ? null : cell === FULL || cell >= 5 ? "high" : cell === 4 ? "iv" : "iii");

/**
 * The runs of the fraction the page makes in k factors at a resolution: the fewest of the grid's whose
 * fraction reaches it, as the library takes the minimum-aberration fraction with the fewest runs that does.
 * Null when the grid has no column for k.
 */
export function fractionRuns(k, resolution) {
  const reaches = (runs) => typeof tradeOffCell(runs, k) === "number" && tradeOffCell(runs, k) >= resolution;
  return TRADE_OFF_RUNS.find(reaches) ?? null;
}

/** The resolution the form asks for to make a fraction cell: the cell's own, or the highest the page offers (V). */
export const cellResolution = (cell) => Math.min(cell, Math.max(...RESOLUTIONS));

/**
 * The design a pick of a cell sets, or null for a cell the page cannot make: the full factorial for a FULL
 * cell; else a fraction at the cell's resolution, or at V for a higher one, when the page makes that
 * resolution in exactly the cell's runs. In seven factors resolution IV takes 16 runs, so the 32-run cell
 * of resolution IV is not offered.
 */
export function cellDesign(runs, k) {
  const cell = tradeOffCell(runs, k);
  if (cell === null) return null;
  if (cell === FULL) return { design_type: "full_factorial" };
  const resolution = cellResolution(cell);
  return fractionRuns(k, resolution) === runs ? { design_type: "fractional_factorial", resolution } : null;
}

/** The select value that asks for an axial distance typed as a number. */
export const AXIAL_DISTANCE = "distance";

/** The axial distance the form asks for: a named choice, or the number typed beside it (null while blank). */
export const axialDistance = (choice, typed) => (choice === AXIAL_DISTANCE ? numberOrNull(typed) : choice);

/** Whether two numbers agree as Python's math.isclose finds them, to a relative 1e-9. */
const isClose = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b));

/**
 * Whether a design needs a center run to fit the squares, because its other runs all lie on one
 * sphere: a Box-Behnken design's do, and a CCD's whose axial distance is the corners' own, the
 * square root of the k factors. The page asks for one at any rotatable CCD, as bootstrap.py
 * does, and a DSD counts its own built-in center run among them.
 */
export const needsCentre = (designType, alpha, k) =>
  ["box_behnken", "dsd"].includes(designType) ||
  (designType === "ccd" && (alpha === "rotatable" || (typeof alpha === "number" && isClose(alpha * alpha, k))));

/** The analysis model's select value for a model of the terms the reader ticks. */
export const CHOSEN_TERMS = "terms";

/** Every way of picking `size` of the items, in order: the pairs or triples of factors for interactions. */
export function combinations(items, size) {
  if (size === 0) return [[]];
  return items.flatMap((item, i) => combinations(items.slice(i + 1), size - 1).map((rest) => [item, ...rest]));
}

/**
 * Every term a model of these factors may hold, as the page lists them to tick: {term, label, group},
 * where a term is a list of factor names: one for a main effect, two or three for an interaction,
 * and one name twice for a square, as bootstrap.py reads them.
 */
export function candidateTerms(factors) {
  const names = factors.map((f) => f.name);
  const group = (label, terms) => terms.map((term) => ({ term, label: termLabel(term), group: label }));
  return [
    ...group("Main effects", names.map((n) => [n])),
    ...group("Two-factor interactions", combinations(names, 2)),
    ...group("Three-factor interactions", combinations(names, 3)),
    ...group("Squares", squared(factors)),
  ];
}

/** The squares a model may hold: one per numeric factor, since a categorical one is +1 at both its levels. */
const squared = (factors) => factors.filter((f) => !isCategorical(f)).map((f) => [f.name, f.name]);

/** A term as the page writes it: T, T·P, T·P·F, or T² for a square. */
export const termLabel = (term) => (term.length === 2 && term[0] === term[1] ? `${term[0]}²` : term.join("·"));

/** The terms of a named model, as lists of factor names: where a list of terms to tick starts. */
export function namedModelTerms(model, factors) {
  const names = factors.map((f) => f.name);
  const terms = names.map((n) => [n]);
  if (model === "interactions" || model === "quadratic") terms.push(...combinations(names, 2));
  if (model === "quadratic" || model === "main_quadratic") terms.push(...squared(factors));
  return terms;
}

/** The kinds of term each model holds: main effects, two-factor interactions, squares. */
const KINDS = {
  main_effects: ["main"],
  interactions: ["main", "interaction"],
  main_quadratic: ["main", "square"],
  quadratic: ["main", "interaction", "square"],
};

/** Whether a model holds every kind of term another does: the full quadratic holds them all. */
export const holds = (model, aim) => (KINDS[aim] ?? []).every((kind) => (KINDS[model] ?? []).includes(kind));

/**
 * The model a new run sheet's results are analysed with: "" for the design's own when it holds what
 * the study aims to learn; else the aim's, when the design can fit it (``fits``, as the comparison
 * found), as a named model or, for main effects and squares, as their terms; else the design's own.
 */
export function analysisModel(designModel, aim, fits, factors) {
  if (!aim || !fits || holds(designModel, aim)) return "";
  return aim === "main_quadratic" ? { terms: namedModelTerms(aim, factors) } : aim;
}

/** The designs that take a categorical factor; mirrors bootstrap.CATEGORICAL_DESIGNS. */
export const CATEGORICAL_DESIGNS = ["full_factorial", "fractional_factorial", "plackett_burman", "taguchi"];

/** Of those, the designs that take a categorical factor of three levels or more; mirrors bootstrap.SEVERAL_LEVEL_DESIGNS. */
export const SEVERAL_LEVEL_DESIGNS = ["full_factorial", "taguchi"];

/** The most levels a categorical factor takes; mirrors bootstrap.MAX_LEVELS. */
export const MAX_LEVELS = 8;

/** The longest label a level may have; mirrors bootstrap.LABEL_MOST. */
export const LABEL_MOST = 64;

/**
 * The design the page holds but does not make, as bootstrap.GIVEN names it: runs taken as they are
 * given, such as a design a book prints, opened from the design catalog. It takes categorical factors
 * and limits on the settings, and keeps its runs as they are.
 */
export const GIVEN = "given";

// ---------------------------------------------------------------- mixtures
/**
 * The mixture designs, as bootstrap.MIXTURES names them: the factors are the components of a blend, each a
 * proportion between its lower and upper bounds, and every run's proportions sum to 1.
 */
export const MIXTURES = ["simplex_lattice", "simplex_centroid", "extreme_vertices"];

/** Whether a design type is a mixture's. */
export const isMixture = (designType) => MIXTURES.includes(designType);

/** The models a mixture is analysed with; mirrors bootstrap.MIXTURE_MODELS. */
export const MIXTURE_MODELS = ["scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic"];

/** A component's bound as the form sends it: a blank box is the bound's default, 0 for the lower and 1 for the upper. */
export const proportionValue = (text, blank) => (String(text ?? "").trim() === "" ? blank : levelValue(text));

/**
 * What a component's bounds need, in a sentence the page shows under the table; "" when they are ready. A
 * blank box is 0 or 1. The rules are those bootstrap.py applies to a mixture's components.
 */
export function componentProblem({ name, low, high }) {
  const who = String(name ?? "").trim() || "A component without a name";
  const [lower, upper] = [proportionValue(low, 0), proportionValue(high, 1)];
  if (typeof lower !== "number" || typeof upper !== "number" || !Number.isFinite(lower) || !Number.isFinite(upper)) {
    return `${who}: enter its lower and upper proportions as numbers, or leave them blank for 0 and 1.`;
  }
  if (lower < 0 || upper > 1) {
    return (
      `${who} runs from ${lower} to ${upper}, which is no proportion: each component of a mixture lies between 0 ` +
      "and 1, and a mixture on this page takes no process variables, such as a temperature."
    );
  }
  if (!(lower < upper)) return `${who}: its lower proportion must be below its upper proportion.`;
  return "";
}

/** The components whose bounds cut the simplex: a lower proportion above 0, or an upper one below 1. */
export const boundedComponents = (factors) => factors.filter((f) => f.low > 0 || f.high < 1).map((f) => f.name);

/**
 * What a mixture's components, each ready on its own, still need together for the design type, in a sentence;
 * "" when nothing. The lower proportions must sum to less than 1 and the upper ones to more than 1; extreme
 * vertices need a bound, and the simplex designs none. ``factors`` are as the form sends them for a mixture.
 */
export function mixtureProblem(factors, designType, label) {
  const sum = (key) => factors.reduce((total, f) => total + f[key], 0);
  const figures = (x) => Number(x.toPrecision(4));
  if (sum("low") > 1 - 1e-9) {
    return (
      `The lower proportions sum to ${figures(sum("low"))}, and must sum to less than 1: above 1 no blend gives every ` +
      "component its lower proportion, and at 1 only one blend does. Lower some of them."
    );
  }
  if (sum("high") < 1 + 1e-9) {
    return (
      `The upper proportions sum to ${figures(sum("high"))}, and must sum to more than 1: below 1 no blend of the ` +
      "components reaches 1, and at 1 only one does. Raise some of them."
    );
  }
  const bounded = boundedComponents(factors);
  if (designType === "extreme_vertices" && !bounded.length) {
    return (
      "Extreme vertices are for a mixture with bounds: give a component a lower or upper proportion, or choose the " +
      "simplex lattice or simplex centroid."
    );
  }
  if (designType !== "extreme_vertices" && bounded.length) {
    return (
      `The ${label} design covers every blend, each component from 0 to 1, and ${listNames(bounded)} ` +
      `${bounded.length === 1 ? "has" : "have"} bounds: choose extreme vertices for a mixture with bounds.`
    );
  }
  return "";
}

/** Whether text typed as a level is a label, such as old: neither blank nor a number (nor inf or nan). */
export function isLabel(text) {
  const t = String(text ?? "").trim();
  return t !== "" && Number.isNaN(Number(t)) && !/^[+-]?(inf(inity)?|nan)$/i.test(t);
}

/** A level as the form sends it: a label as its text, a blank box as null, and anything else as a number. */
export const levelValue = (text) => (isLabel(text) ? String(text).trim() : numberOrNull(String(text ?? "")));

/**
 * Whether a factor is categorical: its type says so, as the brief and a study of several levels send it, or both its
 * levels are labels, as every study before them kept a factor of two.
 */
export const isCategorical = (f) => f?.type === "categorical" || (isLabel(f?.low) && isLabel(f?.high));

/** A categorical factor's levels, in their order: its list of levels, or its two labels. */
export const levelsOf = (f) => (Array.isArray(f.levels) ? [...f.levels] : [f.low, f.high]);

/** Whether a factor is categorical with three levels or more: a column of the model per level but one. */
export const isSeveral = (f) => isCategorical(f) && levelsOf(f).length > 2;

/**
 * A categorical factor as the brief sends it: its type and its levels, any text, with a factor of two levels' first
 * and second as its low and high too, as the two-level designs read them.
 */
export function categoricalFactor(name, levels, units = "") {
  const two = levels.length === 2 ? { low: levels[0], high: levels[1] } : {};
  return { name, type: "categorical", levels: [...levels], ...two, units };
}

/** What a level reads as, to tell two apart: the number it is (1 and 1.0 alike), or its text. */
const levelKey = (text) => {
  const t = String(text).trim();
  return t !== "" && !Number.isNaN(Number(t)) ? `#${Number(t)}` : t;
};

/**
 * What a categorical factor's levels need, in a sentence the page shows under the table as the reader adds them;
 * "" when they are ready. The rules are those bootstrap.py applies: 2 to MAX_LEVELS labels, none blank, none longer
 * than LABEL_MOST characters, and no two alike, nor two that read as the same number.
 */
export function labelsProblem(name, levels) {
  const who = String(name ?? "").trim() || "A factor without a name";
  const texts = levels.map((v) => String(v ?? "").trim());
  if (texts.length < 2) return `${who}: give it two levels or more, such as A and B.`;
  if (texts.length > MAX_LEVELS) return `${who}: give it at most ${MAX_LEVELS} levels; it has ${texts.length}.`;
  if (texts.includes("")) return `${who}: a level is blank; give each level a label.`;
  const long = texts.find((t) => t.length > LABEL_MOST);
  if (long) return `${who}: the level ${long.slice(0, 20)}... is longer than ${LABEL_MOST} characters.`;
  const seen = new Map();
  for (const t of texts) {
    const key = levelKey(t);
    if (seen.has(key)) return seen.get(key) === t ? `${who}: ${t} is listed twice.` : `${who}: ${seen.get(key)} and ${t} read as the same number.`;
    seen.set(key, t);
  }
  return "";
}

/** A factor's two types, as the Type column offers them. */
export const NUMERIC = "numeric";
export const CATEGORICAL = "categorical";

/** What a level box holds: "blank", a "number" or a "label". */
export const levelKind = (text) => (String(text ?? "").trim() === "" ? "blank" : isLabel(text) ? "label" : "number");

/**
 * The type a factor's two levels give it: categorical for two labels, numeric for two numbers.
 * Anything in between (a blank box, or a label beside a number) keeps the type the reader chose.
 */
export function factorType(low, high, chosen) {
  const kinds = [levelKind(low), levelKind(high)];
  if (kinds.every((k) => k === "label")) return CATEGORICAL;
  if (kinds.every((k) => k === "number")) return NUMERIC;
  return chosen;
}

/**
 * What a factor's levels need for its type, in a sentence the page shows under the table as the
 * reader types; "" when they are ready. The rules are those bootstrap.py applies to the levels.
 */
export function levelProblem({ name, low, high, levels }, type) {
  const who = String(name ?? "").trim() || "A factor without a name";
  const kinds = [levelKind(low), levelKind(high)];
  if (type === CATEGORICAL) return labelsProblem(name, levels ?? [low, high]);
  if (kinds.includes("blank") || kinds.includes("label")) {
    return `${who}: give a number for both the low and the high value, or make it categorical to give it labels.`;
  }
  if (!(Number(low) < Number(high))) return `${who}: the low value must be below the high value.`;
  return "";
}

/** Read a number box as the page sends it: a blank box as null, rather than as Number("") = 0. */
export const numberOrNull = (text) => (text.trim() === "" ? null : Number(text));

/** The response a blank box names: y, as Python names it when it makes the design. */
export const DEFAULT_RESPONSE = "y";

/** The run sheet's column of each run's block, which no factor or response may be named; mirrors bootstrap.BLOCK. */
export const BLOCK = "Block";

/**
 * The run sheet's column for a response the reader names in words: "Yield" stays Yield, and "Tensile
 * strength (MPa)" becomes Tensile_strength_MPa. Only ASCII letters, digits and _, so Python and every
 * spreadsheet read it alike, and never a factor's name or Block: y when nothing is left.
 */
export function responseName(label, factorNames = []) {
  const plain = String(label ?? "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  let name = plain.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40).replace(/_+$/, "");
  if (!name) name = DEFAULT_RESPONSE;
  if (/^[0-9]/.test(name)) name = `y_${name}`;
  const taken = new Set([...factorNames, BLOCK]);
  if (!taken.has(name)) return name;
  let i = 2;
  while (taken.has(`${name}_${i}`)) i += 1;
  return `${name}_${i}`;
}

/** What the reader can want of the response, as the response card words it; the keys are bootstrap.GOALS. */
export const GOALS = { maximize: "Maximise", minimize: "Minimise", target: "Hit a target", learn: "Just learn" };

// ---------------------------------------------------------------- several responses
/** The most responses a study measures; mirrors bootstrap.MAX_RESPONSES. */
export const MAX_RESPONSES = 5;

/**
 * A study's responses, the first first, as bootstrap._responses reads a specification: each is its column's
 * ``name`` and its description. A study made before the page measured several holds one, ``response``,
 * described by ``response_info``; one with several lists them all in ``responses``.
 */
export function responsesOf(spec) {
  if (Array.isArray(spec?.responses) && spec.responses.length) return spec.responses;
  return [{ name: spec?.response ?? DEFAULT_RESPONSE, ...(spec?.response_info ?? {}) }];
}

/** A response's description without its column's name: what a specification's ``response_info`` holds. */
export const infoOf = ({ name: _, ...info }) => info;

/**
 * What a specification keeps of its responses, as bootstrap._stored_responses writes it: the first one's name
 * and description, as a study of one response has always kept them, and with several the whole list.
 */
export function storedResponses(responses) {
  const [first] = responses;
  const info = infoOf(first);
  return {
    response: first.name,
    ...(Object.keys(info).length ? { response_info: info } : {}),
    ...(responses.length > 1 ? { responses } : {}),
  };
}

/** A specification with its responses replaced: the first one's name and description, and the list with several. */
export function withResponses(spec, responses) {
  const { response_info: _, responses: __, ...rest } = spec;
  return { ...rest, ...storedResponses(responses) };
}

/** Each response's results, by its column: no result yet for any of ``n`` runs. */
export const noResults = (names, n) => Object.fromEntries(names.map((name) => [name, Array(n).fill(null)]));

/** Whether a value is a result: a number typed, simulated or read, not a blank. */
const isResult = (v) => v !== null && v !== undefined;

/**
 * Whether each run is complete: it has a result for every response. ``results`` holds each response's results
 * by its column, and ``names`` are the columns. A run with some of its results counts for those responses only.
 */
export function completeRuns(results, names) {
  const n = Math.max(0, ...names.map((name) => results[name]?.length ?? 0));
  return Array.from({ length: n }, (_, i) => names.every((name) => isResult(results[name]?.[i])));
}

/** Whether any response has a result yet. */
export const anyResult = (results) => Object.values(results).some((values) => values.some(isResult));

/**
 * The runs that have some of their results and not all: their places on the run sheet. Each response is fitted
 * to the runs that have its result, so these count for some responses and not for others.
 */
export function partialRuns(results, names) {
  const complete = completeRuns(results, names);
  return complete.flatMap((done, i) => (!done && names.some((name) => isResult(results[name]?.[i])) ? [i] : []));
}

/**
 * The results as the page sends them to Python, and as a campaign keeps them: the first response's in
 * ``responses``, as a study of one response always has, and each later one's in ``more_responses``, by its
 * column, only when there are several. ``key`` names the pair: ``responses``, or ``sources`` for their sources.
 */
export function splitResults(names, byName, key = "responses") {
  const [first, ...more] = names;
  const out = { [key]: byName[first] ?? [] };
  if (more.length) out[`more_${key}`] = Object.fromEntries(more.map((name) => [name, byName[name] ?? []]));
  return out;
}

/** The reverse of splitResults: each response's results by its column, from the first's list and the others'. */
export function joinResults(names, first, more = {}) {
  const [head, ...rest] = names;
  const n = first?.length ?? 0;
  return { [head]: first ?? [], ...Object.fromEntries(rest.map((name) => [name, more?.[name] ?? Array(n).fill(null)])) };
}

/**
 * What Learn says of the runs that have some of their results and not all, with several responses: ``runs`` are
 * their numbers, as the run sheet's Run order gives them, and each response is fitted to the runs that have its
 * result: the one analysed, named ``label``, to ``fitted`` runs.
 */
export function partialSentence(runs, label, fitted) {
  const one = runs.length === 1;
  const which = one ? `Run ${runs[0]} has some of its results` : `Runs ${listNames(runs.map(String))} have some of their results`;
  return `${which}, not all. Each response is fitted to the runs that have its result: ${label} to ${fitted}.`;
}

/**
 * The responses as a study's title names them: one as responseTitle does ("Yield (%), maximise"), several by
 * their names ("Yield, Cost and Purity").
 */
export function responsesTitle(responses) {
  if (responses.length === 1) return responseTitle(infoOf(responses[0]), responses[0].name);
  return listNames(responses.map((r) => r.label || r.name));
}

/**
 * What the brief's responses still need, a sentence each, as the reader types: the later responses must differ
 * in name from the first and from each other. ``labels`` are the names typed, in order, blanks left out.
 */
export function responseLabelProblems(labels) {
  const seen = new Set();
  const said = [];
  for (const label of labels.map((l) => String(l ?? "").trim()).filter(Boolean)) {
    const key = label.toLowerCase();
    if (seen.has(key)) said.push(`${label}: two responses cannot share a name.`);
    seen.add(key);
  }
  return [...new Set(said)];
}

/**
 * The run sheet's columns for the brief's responses: each named from its label as responseName names it, unlike
 * every factor and every response before it. ``kept`` gives a column a response holds already, by its place.
 */
export function responseColumns(labels, factorNames = [], kept = []) {
  const taken = [...factorNames, ...kept.filter(Boolean)];
  return labels.map((label, i) => {
    if (kept[i]) return kept[i];
    const name = responseName(label, taken);
    taken.push(name);
    return name;
  });
}

/** A quantity's name with its units, as a heading or an axis shows it: "T (degC)", or the name alone. */
export const withUnits = (name, units) => (units ? `${name} (${units})` : name);

/**
 * The response as the study's title names it: "Yield (%), maximise", "Purity (%), target 99.5",
 * "Hardness", or the column's name for a design made before the page asked for more.
 */
export function responseTitle(info, name = DEFAULT_RESPONSE) {
  const head = withUnits(info?.label || name || DEFAULT_RESPONSE, info?.units ?? "");
  if (info?.goal === "maximize") return `${head}, maximise`;
  if (info?.goal === "minimize") return `${head}, minimise`;
  if (info?.goal === "target") return isFiniteNumber(info.target) ? `${head}, target ${info.target}` : `${head}, on target`;
  return head;
}

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * The power the designs are compared at, in the reader's words: the smallest change worth finding,
 * from a factor's low setting to its high one, and the coefficient in noise standard deviations it
 * is (half the change, over the noise); null without both numbers, when the default of one noise
 * standard deviation holds.
 */
export function powerScale(info) {
  if (!isFiniteNumber(info?.noise) || !isFiniteNumber(info?.smallest_change)) return null;
  if (info.noise <= 0 || info.smallest_change <= 0) return null;
  return { change: info.smallest_change, coefficient: info.smallest_change / (2 * info.noise) };
}

/** A variable name, as Python's str.isidentifier reads one: a letter or _, then letters, digits and _. */
const IDENTIFIER = /^[\p{ID_Start}_][\p{ID_Continue}]*$/u;

/**
 * What a factor's name needs, in a sentence the page shows as the reader types; "" when it is ready.
 * The rules are those bootstrap.py applies: a variable name, unlike every other factor's, and unlike
 * ``taken``, the run sheet's other columns a factor renamed on it cannot take (its results, its blocks).
 */
export function nameProblem(name, others = [], place = 1, taken = []) {
  const n = String(name ?? "").trim();
  if (!n) return `Factor ${place} needs a name.`;
  if (!IDENTIFIER.test(n)) return `${n}: a name is letters, digits and _, and starts with a letter, such as Temp_2.`;
  if (others.includes(n)) return `${n}: two factors cannot share a name.`;
  if (taken.includes(n)) return `${n}: the run sheet has a ${n} column already.`;
  return "";
}

/**
 * What the factor table renames, when that is all it changes: ``after`` holds the design's factors in
 * the same places with the same levels, and only names or units differ. Returns each old name's new
 * one, or null when the number of factors or a level differs, or when nothing does.
 */
export function renaming(before, after) {
  const sameLevels = (f, i) => f.low === after[i].low && f.high === after[i].high;
  if (before.length !== after.length || !before.every(sameLevels)) return null;
  const relabelled = (f, i) => f.name !== after[i].name || (f.units ?? "") !== (after[i].units ?? "");
  return before.some(relabelled) ? Object.fromEntries(before.map((f, i) => [f.name, after[i].name])) : null;
}

/** Chosen model terms, each a list of factor names, with the factors renamed as ``names`` maps them. */
export const renameTerms = (terms, names) => terms.map((term) => term.map((name) => names[name] ?? name));

/**
 * The response as the reader describes it, cleaned as bootstrap.py cleans it: the words trimmed and an
 * empty one left out, the goal (with its target only when the goal is one), and only the numbers that
 * are numbers; the noise and the smallest change only above 0. ``fields`` holds the boxes' text.
 */
export function describeResponse(fields) {
  const info = {};
  for (const key of ["label", "units"]) {
    const text = String(fields[key] ?? "").trim();
    if (text) info[key] = text;
  }
  info.goal = GOALS[fields.goal] ? fields.goal : "learn";
  const number = (key) => {
    const text = String(fields[key] ?? "").trim();
    const value = text === "" ? null : Number(text);
    return value !== null && Number.isFinite(value) ? value : null;
  };
  if (info.goal === "target" && number("target") !== null) info.target = number("target");
  if (number("typical") !== null) info.typical = number("typical");
  for (const key of ["noise", "smallest_change"]) if (number(key) > 0) info[key] = number(key);
  return info;
}

/** What the response's numbers still need, a sentence each; the rules are bootstrap.py's. */
export function responseProblems(fields) {
  const said = [];
  const given = (key) => String(fields[key] ?? "").trim() !== "";
  if (given("noise") && !(Number(fields.noise) > 0)) said.push("The run-to-run noise must be a number above 0.");
  if (given("smallest_change") && !(Number(fields.smallest_change) > 0)) {
    said.push("The smallest change worth finding must be a number above 0.");
  }
  return said;
}

/** The model an optimal design is built for, for each aim: the smallest it takes that contains it; mirrors bootstrap.OPTIMAL_MODEL. */
export const OPTIMAL_MODEL = {
  main_effects: "main_effects",
  interactions: "interactions",
  main_quadratic: "quadratic",
  quadratic: "quadratic",
};

/** From this many factors a study starts from the screening aim, which the reader can change. */
export const SCREEN_FROM = 5;

/**
 * The aim a study starts from: which factors matter, for many factors, or for one, whose two levels are compared;
 * else where each one's effect curves.
 */
export const defaultAim = (k) => (k === 1 || k >= SCREEN_FROM ? "main_effects" : "main_quadratic");

/**
 * Count the coefficients, intercept included, of the model an aim names, over these factors: a categorical factor of
 * k levels has k - 1 columns, and so has each of its interactions per column of the other factor; no square of a
 * categorical one.
 */
export function aimTerms(factors, aim) {
  const columns = factors.map((f) => (isCategorical(f) ? Math.max(1, levelsOf(f).length - 1) : 1));
  const mains = columns.reduce((a, b) => a + b, 0);
  let pairs = 0;
  if (aim === "interactions" || aim === "quadratic") {
    for (let i = 0; i < columns.length; i++) for (let j = i + 1; j < columns.length; j++) pairs += columns[i] * columns[j];
  }
  const squares = aim === "main_quadratic" || aim === "quadratic" ? factors.filter((f) => !isCategorical(f)).length : 0;
  return 1 + mains + pairs + squares;
}

/** Spare runs a design must leave for the noise to be the one picked when the reader has not chosen one. */
export const SPARE_FOR_NOISE = 3;

/**
 * The designs of a comparison as Design lists them, sorted by their runs, those that can fit the model
 * first: each design's type, its short name, its runs, the runs left over to estimate the noise, and
 * its power for each kind of term the model has (null for a kind it lacks, or with no runs left over).
 * ``made`` is the comparison's designs in the order scored, each {design_type, label}.
 */
export function candidateRows(result, made) {
  const rows = result.designs.map((sized, i) => {
    const v = result.values[sized];
    return {
      design_type: made[i].design_type,
      label: made[i].label,
      runs: v.runs,
      fits: Boolean(v.fits),
      spare: v.fits ? v.residual_df : null,
      power: { main: v.power_main ?? null, interaction: v.power_interaction ?? null, quadratic: v.power_quadratic ?? null },
      order: i,
    };
  });
  return rows.sort((a, b) => Number(b.fits) - Number(a.fits) || a.runs - b.runs || a.order - b.order);
}

/**
 * The order of the detailed comparison's columns: the designs that can fit the model first, then those that cannot,
 * whose measures are mostly blank, each group in the order scored; and the designs the reader has hidden (their
 * labels in ``hidden``) after all of those, in the same order. Each is the design's place in ``result.designs``.
 */
export function comparisonOrder(result, hidden = new Set()) {
  const places = result.designs.map((_, i) => i);
  const fits = (i) => Boolean(result.values[result.designs[i]]?.fits);
  const ordered = [...places.filter(fits), ...places.filter((i) => !fits(i))];
  const shown = (i) => !hidden.has(result.designs[i]);
  return [...ordered.filter(shown), ...ordered.filter((i) => !shown(i))];
}

/**
 * The design picked when the reader has not chosen one: of those that can fit the model, the fewest
 * runs that leave SPARE_FOR_NOISE to estimate the noise, else the fewest runs; null when none fits.
 */
export function pickCandidate(rows) {
  const fitting = rows.filter((r) => r.fits);
  const spare = fitting.filter((r) => r.spare >= SPARE_FOR_NOISE);
  return (spare[0] ?? fitting[0] ?? null)?.design_type ?? null;
}

/**
 * What Design says of the design it picked for a reader who has not chosen one: why that one, by pickCandidate's rule.
 * ``rows`` are candidateRows', ``picked`` the design type the plan holds; "" when the rule would not pick it.
 */
export function pickSentence(rows, picked) {
  if (!picked || pickCandidate(rows) !== picked) return "";
  const row = rows.find((r) => r.design_type === picked);
  const spare = row.spare >= SPARE_FOR_NOISE;
  const why = spare
    ? `the fewest runs with ${SPARE_FOR_NOISE} or more spare`
    : `the fewest runs, as none here has ${SPARE_FOR_NOISE} spare`;
  return `Picked for you: ${row.label}, ${why}.`;
}

/**
 * The comparison's measures a design can stand out on among the designs listed, each with the words the list names it
 * with: the fewest runs, the precision of the coefficients and of the predictions, the correlation between the
 * second-order terms, and the power for each kind of term. Each is a key of the comparison's values.
 */
export const STANDOUTS = [
  ["runs", "fewest runs"],
  ["a", "precise coefficients"],
  ["d", "joint precision"],
  ["i", "precise predictions"],
  ["g", "precise worst-case prediction"],
  ["max_r", "least correlated terms"],
  ["power_main", "power for main effects"],
  ["power_interaction", "power for interactions"],
  ["power_quadratic", "power for curvature"],
];

/** How close to the best value a design's must be, as a share of the best, to share the lead on a measure. */
export const STANDOUT_TIE = 0.01;

/**
 * What each design that fits the model does best of those that fit, by its design type: each measure of STANDOUTS on
 * which it is best, or within STANDOUT_TIE of the best, while some other design that fits is not. A measure no design
 * leads alone, as when every design has the same, is left out; with fewer than two designs that fit, none stands out.
 * Each is {key, words}. ``result`` is the comparison's, ``made`` its designs in the order scored.
 */
export function standouts(result, made) {
  const lower = new Set(result.measures.filter((m) => m.direction === "lower").map((m) => m.key).concat("runs"));
  const fitting = result.designs
    .map((sized, i) => ({ type: made[i].design_type, values: result.values[sized] }))
    .filter((d) => d.values.fits);
  const out = Object.fromEntries(fitting.map((d) => [d.type, []]));
  if (fitting.length < 2) return out;
  for (const [key, words] of STANDOUTS) {
    const known = fitting.filter((d) => typeof d.values[key] === "number" && Number.isFinite(d.values[key]));
    if (known.length < 2) continue;
    const values = known.map((d) => d.values[key]);
    const best = lower.has(key) ? Math.min(...values) : Math.max(...values);
    const near = (x) => Math.abs(x - best) <= STANDOUT_TIE * Math.abs(best);
    const leaders = known.filter((d) => near(d.values[key]));
    if (leaders.length === known.length) continue;
    for (const d of leaders) out[d.type].push({ key, words });
  }
  return out;
}

/**
 * Why Design lists none of some designs the page offers, a sentence for each reason: the factor count a design takes,
 * by the bound the brief's count breaks, the designs of numeric factors when the brief has a categorical one, and the
 * two-level designs when it has a categorical factor of three levels or more. ``leftOut`` is api_candidates', ``k``
 * the brief's factor count, ``categorical`` the names of its categorical factors and ``several`` its factors of three
 * levels or more. Limits on the settings leave designs out too, and the comparison's own note says so, so they get
 * no sentence here.
 */
export function leftOutSentences(leftOut, k, categorical = [], several = []) {
  const said = [];
  const bounds = new Map();
  for (const d of leftOut.filter((one) => one.reason === "factors")) {
    const bound = k < d.low ? `${d.low} factors or more` : `at most ${d.high} factors`;
    bounds.set(bound, [...(bounds.get(bound) ?? []), d.label]);
  }
  const verb = (labels, singular, plural) => (labels.length === 1 ? `is not listed: it ${singular}` : `are not listed: they ${plural}`);
  for (const [bound, labels] of bounds) {
    said.push(`${listNames(labels)} ${verb(labels, "takes", "take")} ${bound}, and the brief has ${k}.`);
  }
  const numeric = leftOut.filter((one) => one.reason === "categorical").map((d) => d.label);
  if (numeric.length && categorical.length) {
    const are = categorical.length === 1 ? "is" : "are";
    said.push(`${listNames(numeric)} ${verb(numeric, "needs", "need")} numeric factors, and ${listNames(categorical)} ${are} categorical.`);
  }
  const twoLevel = leftOut.filter((one) => one.reason === "levels").map((d) => d.label);
  if (twoLevel.length && several.length) {
    const levels = several.map((f) => `${f.name} has ${levelsOf(f).length}`);
    said.push(`${listNames(twoLevel)} ${verb(twoLevel, "takes", "take")} factors of two levels, and ${listNames(levels)}.`);
  }
  return said;
}

/** A power as the page prints it: a whole percentage. */
export const percent = (p) => (p === null || p === undefined ? "" : `${Math.round(100 * p)}%`);

/**
 * The change the designs' power is for, in the reader's words: the smallest change worth finding in
 * the response's units, else two noise standard deviations, which the comparison's default is.
 */
export function changeWords(info) {
  const scale = powerScale(info);
  if (!scale) return "a change of two noise standard deviations";
  return `a change of ${scale.change}${info.units ? ` ${info.units}` : ""}`;
}

/** The run sheet's progress, as the Run tab's badge shows it: "15 runs", or "9 of 15" once results come in. */
export function runBadge(responses) {
  const n = responses.length;
  const done = responses.filter((v) => v !== null && v !== undefined).length;
  if (!n) return "";
  return done ? `${done} of ${n}` : `${n} runs`;
}

/** Join names as a sentence does: "A", "A and B", "A, B and C". */
export const listNames = (names) =>
  names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names.join("");

/** A number to four significant figures, or nothing for a value the fit could not give. */
export const fmt = (x) => (x === null || x === undefined ? "" : Number(x).toPrecision(4));

const isNumber = (v) => typeof v === "number" && Number.isFinite(v);
const isWhole = (v) => Math.abs(v - Math.round(v)) < 1e-9 * Math.max(1, Math.abs(v));
/** Two values a reader should see as different: more than 1% apart. */
const apart = (a, b) => Math.abs(a - b) > 0.01 * Math.max(Math.abs(a), Math.abs(b));

/**
 * The decimals one row of the detailed comparison prints with, the same for every design so the
 * row's numbers line up. A row of whole numbers prints whole. Otherwise the row starts at two
 * significant figures of its largest value, and takes one more decimal, twice at most, while two
 * values more than 1% apart would print alike.
 */
export function rowDecimals(numbers) {
  if (!numbers.length || numbers.every(isWhole)) return 0;
  const largest = Math.max(...numbers.map(Math.abs));
  const first = Math.max(0, 1 - Math.floor(Math.log10(largest)));
  const alike = (d) => numbers.some((a, i) => numbers.slice(i + 1).some((b) => apart(a, b) && a.toFixed(d) === b.toFixed(d)));
  let decimals = first;
  while (decimals < first + 2 && alike(decimals)) decimals += 1;
  return decimals;
}

/**
 * One row of the detailed comparison as the page shows it: each design's value as text, and how
 * near it is to the row's best. `good` runs from 0, the row's worst, to 1, its best, read off the
 * printed numbers so that values printed alike score alike. Only the designs `ranked` marks are
 * scored, and only they set the row's decimals, best and worst: a design that cannot fit the model
 * prints its value, unscored. `good` is null for those, for a missing value, for a measure that
 * prefers no direction (the feasibility rows), and when every design scored prints the same.
 *
 * @param {(number|null)[]} values each design's value, in the table's order
 * @param {string} direction "higher" or "lower", whichever the measure prefers; "" for neither
 * @param {boolean[]} [ranked] which designs are scored; all of them when left out
 * @returns {{text: string, good: number|null}[]}
 */
export function comparedRow(values, direction, ranked = values.map(() => true)) {
  const rankedNumbers = values.filter((v, j) => isNumber(v) && ranked[j]);
  const decimals = rowDecimals(rankedNumbers.length ? rankedNumbers : values.filter(isNumber));
  const printed = values.map((v) => (isNumber(v) ? Number(v.toFixed(decimals)) : null));
  const scored = printed.filter((v, j) => v !== null && ranked[j]);
  const low = Math.min(...scored);
  const spread = Math.max(...scored) - low;
  const directed = spread > 0 && (direction === "higher" || direction === "lower");
  return printed.map((v, j) => ({
    text: v === null ? "" : v.toFixed(decimals),
    good: v === null || !ranked[j] || !directed ? null : direction === "higher" ? (v - low) / spread : 1 - (v - low) / spread,
  }));
}

/** A p-value to four decimals, or "< 0.0001". */
export const pval = (p) => (p === null || p === undefined ? "" : p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4));

/** The variance inflation factor above which a term is correlated enough for a note; bootstrap.py's VIF_HIGH. */
export const VIF_HIGH = 5;

/**
 * Whether the coefficients table shows each term's VIF: only when some term's is above VIF_HIGH, as the
 * analysis's note says, so an orthogonal design (every VIF 1) gets no column of ones. A block's own VIF
 * does not count, as the note leaves it out.
 */
export const showsVif = (coefficients) =>
  coefficients.some((c) => !c.block && typeof c.vif === "number" && c.vif > VIF_HIGH);

/**
 * The terms an analysis fitted, as a formula ``evaluate_design`` reads to give their VIFs: the blocks
 * first, as the fit has them, then each term of the model, less the intercept.
 */
export function fittedFormula(coefficients) {
  const terms = coefficients.filter((c) => !c.block && c.term !== "Intercept").map((c) => c.term);
  return [...(coefficients.some((c) => c.block) ? [BLOCK] : []), ...terms].join(" + ");
}

/**
 * The fit at a glance, a line each, as the page lists them: the runs fitted and any still to do,
 * then the fit statistics. One the fit cannot give is left out, and a missing predicted R² is explained.
 */
export function fitStatistics(s, completed, pending = []) {
  const known = (x) => x !== null && x !== undefined;
  const stats = [
    ["R²", s.r_squared],
    ["Adjusted R²", s.r_squared_adj],
    ["Predicted R²", s.r_squared_pred],
  ];
  // A run with leverage 1 is fitted exactly whatever its result, so the fit without it cannot predict it.
  const predicted = known(s.r_squared_pred)
    ? []
    : ["Predicted R² cannot be computed: at least one run cannot be predicted from the others"];
  return [
    `${completed} completed runs`,
    ...(pending.length ? [`Still to run: ${pending.join(", ")}`] : []),
    ...stats.filter(([, x]) => known(x)).map(([name, x]) => `${name} = ${fmt(x)}`),
    ...predicted,
    `Residual degrees of freedom = ${s.df_residual}`,
  ];
}

/** Blank cells (an unfilled response) sort last in either direction. */
export const isBlank = (v) => v === null || v === undefined || v === "";

/** A cell's number for sorting, including text that leads with one ("< 0.0001", "4.623 to 5.300"). */
export const sortNumber = (v) => (typeof v === "number" ? v : parseFloat(String(v).replace(/^[<>≤≥]\s*/, "")));

/** Order two non-blank cells: numbers numerically, then everything else alphabetically, numbers within text in order. */
export function compareCells(a, b) {
  const na = sortNumber(a);
  const nb = sortNumber(b);
  if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

/** Text made safe to place in HTML. */
export const escape = (v) =>
  String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/** Each chart's heading. */
export const TITLES = {
  coefficients: "Coefficients",
  contour: "Fitted response surface",
  true_contour: "True response surface",
  residuals: "Residuals against fitted values",
  half_normal: "Half-normal plot of the effects",
  ternary: "Fitted response over the blends",
  true_ternary: "True response over the blends",
  cox: "Fitted response along each Cox direction",
  levels: "The runs at each level",
};

/** What each chart shows and how to read it; some name the factors drawn. */
export const CAPTIONS = {
  /**
   * With limits on the settings the surface is greyed where a setting cannot be run, and when the
   * center of the ranges cannot be run, the others are held at a central setting that can (``held``,
   * in their own units).
   */
  contour: (pair, { shaded = false, held = [], factors = [] } = {}) => {
    const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
    const others = held.length
      ? `, with ${listNames(held.map((h) => `${h.name} at ${inUnits(h.value, units[h.name])}`))}, since the center of ` +
        "the ranges cannot be run"
      : ", with the others at their center";
    const grey = shaded ? "; grey: settings outside your limits" : "";
    return `Drawn over ${pair.join(" and ")}, the two factors with the largest main effects${others}${grey}; the dots are the runs.`;
  },
  true_contour: "The process the results were simulated from, on the same scale.",
  /** With the best blend found, a star marks it (``starred``). */
  ternary: (_, { starred = false } = {}) =>
    "Each corner is a pure component and each point a blend, coloured by the fitted response, over the blends the " +
    `bounds allow; the white dots are the runs${starred ? ", and the star the best blend" : ""}.`,
  true_ternary: "The process the results were simulated from, on the same scale.",
  /** With the best blend found, the lines go through it and a star marks it (``starred``); else through the reference blend. */
  cox: (_, { starred = false } = {}) =>
    starred
      ? "Each line changes one component's share of the best blend while the others keep their proportions to each " +
        "other; the lines cross at the best blend, which the star marks."
      : "Each line changes one component's share of the reference blend, the centroid of the design's blends, while " +
        "the others keep their proportions to each other; the lines cross at the reference blend.",
  residuals: "A curve or a funnel points to a missing term or a changing noise level.",
  half_normal: "Effects far above the line, past Lenth's margin of error, are likely real.",
  levels:
    "Each dot is a run, and beside the dots is the level's average with its 95% interval. The spread of the dots " +
    "within a level is the noise a difference between the levels is judged against.",
};

/** The most times each run of a design is made: bootstrap.MAX_REPLICATES. */
export const MAX_REPLICATES = 10;

/** The most runs at each level of a design in one factor, which are its replicates: bootstrap.MAX_LEVEL_RUNS. */
export const MAX_LEVEL_RUNS = 50;

/**
 * The tests of the difference between one factor's two levels, as the comparison lists them: each test's name, the
 * glossary entry that says what it is, and what it assumes, a few words for the table.
 */
export const COMPARISON_TESTS = {
  pooled: { name: "t-test", key: "pooled-t", how: "one variance for both levels" },
  welch: { name: "Welch's t-test", key: "welch-t", how: "each level's own variance" },
  randomization: { name: "Randomization test", key: "randomization-test", how: "every split of the runs" },
  rank_sum: { name: "Rank-sum test", key: "rank-sum", how: "the order of the results" },
};

/**
 * The tests of the difference between one factor's two levels run in pairs, a block of two runs, one at each level:
 * each test's name, its glossary entry and what it rests on, as for ``COMPARISON_TESTS``.
 */
export const PAIRED_TESTS = {
  paired: { name: "Paired t-test", key: "paired-t", how: "the difference within each pair" },
  sign_flip: { name: "Randomization test", key: "randomization-test", how: "every swap within the pairs" },
  signed_rank: { name: "Signed-rank test", key: "signed-rank", how: "the order of the differences" },
};

/**
 * A study of one factor's two levels in a sentence: how far apart their averages are and which way, and whether the
 * 95% interval for the difference, from the t-test with one variance for both levels (the model's own), holds zero.
 * Runs in pairs (``difference.paired``, the number of pairs) are compared within each pair, by the paired t-test.
 * ``words`` are the response's name and units and the factor's units; null without a difference to say.
 */
export function comparisonSentence(comparison, { response, units = "", factorUnits = "" }) {
  const d = comparison?.difference;
  const pooled = d?.tests.find((t) => t.test === (d.paired ? "paired" : "pooled"));
  if (!pooled || pooled.ci_low === null || pooled.ci_high === null) return null;
  const num = (x) => String(Number(Number(x).toPrecision(4)));
  const after = (u) => (!u ? "" : u === "%" ? "%" : ` ${u}`); // no space before a percent sign
  const at = (level) => `${level}${after(factorUnits)}`;
  const size = `${num(Math.abs(d.estimate))}${after(units)}`;
  const way = d.estimate < 0 ? "lower" : "higher";
  const interval = `${num(pooled.ci_low)} to ${num(pooled.ci_high)}`;
  const verdict =
    pooled.ci_low <= 0 && pooled.ci_high >= 0
      ? `holds zero, so these runs do not show that ${at(d.to)} and ${at(d.from)} differ by more than their noise`
      : `lies ${pooled.ci_low > 0 ? "above" : "below"} zero, so the difference is larger than the noise in these runs`;
  const lead = d.paired
    ? `Within the ${d.paired} pairs, ${response} is on average ${size} ${way} at ${at(d.to)} than at ${at(d.from)}.`
    : `On average, ${response} is ${size} ${way} at ${at(d.to)} than at ${at(d.from)}.`;
  return `${lead} The 95% interval for the difference, ${interval}, ${verdict}.`;
}

/**
 * What a comparison of three levels or more says (``comparison.several``), in a sentence: whether the one-way analysis
 * of variance finds that the levels' averages differ by more than the noise in the runs, at the 5% level. Null while
 * the runs leave no degree of freedom for the noise.
 */
export function severalSentence(comparison, { response }) {
  const test = comparison?.several?.test;
  if (!test) return null;
  const k = comparison.levels.length;
  const value = test.p_value < 0.0001 ? "below 0.0001" : `of ${pval(test.p_value)}`;
  const p = `the one-way analysis of variance gives a p-value ${value}`;
  return test.p_value < 0.05
    ? `The averages of ${response} at the ${k} levels differ by more than the noise in these runs: ${p}. Tukey's ` +
        "intervals below say which pairs differ."
    : `These runs do not show that the averages of ${response} at the ${k} levels differ by more than their noise: ${p}.`;
}

// ---------------------------------------------------------------- results pasted from a spreadsheet
/** The headings, in lower case, a pasted column of run numbers may have: the run sheet's own, and the usual others. */
export const RUN_HEADINGS = ["run order", "run", "run number", "run no.", "run #"];

/** A block of cells as a spreadsheet copies it: a line a row, its cells split by tabs; a trailing blank line dropped. */
export function pastedCells(text) {
  const lines = String(text).replace(/\r\n?/g, "\n").split("\n");
  while (lines.length && lines.at(-1) === "") lines.pop(); // a line of tabs is a row of blank cells, and stays
  return lines.map((line) => line.split("\t").map((cell) => cell.trim()));
}

/**
 * A pasted cell as a result: null for a blank, a number however a spreadsheet writes it (2.5 or 2,5, with a hyphen
 * or a minus sign, with spaces between its thousands), and undefined for text that is no number.
 */
export function pastedNumber(cell) {
  const text = String(cell).replace(/[\s  ]/g, "").replace(/−/g, "-");
  if (text === "") return null;
  const decimal = !text.includes(".") && text.split(",").length === 2 ? text.replace(",", ".") : text;
  const value = Number(decimal);
  return Number.isFinite(value) ? value : undefined;
}

/** A heading as it names a column, in lower case and without the units after it: "Yield (%)" names yield. */
const headingKey = (text) => text.replace(/\s*[([].*$/, "").trim().toLowerCase();

/**
 * Where each line of a pasted block goes when its rows are matched by run number: the row index of each line's
 * run (``at`` the column holding them), or a problem: a run the sheet does not have, or one given twice.
 */
function rowsByRun(lines, at, sheet, first) {
  const rowOf = new Map(sheet.runs.map((run, i) => [run, i]));
  const seen = new Set();
  const rows = [];
  for (const [j, line] of lines.entries()) {
    const run = pastedNumber(line[at] ?? "");
    const where = `Line ${first + j} of what you pasted`;
    if (!rowOf.has(run)) return { problem: `${where} is for run ${line[at] || "(none)"}, which the run sheet does not have.` };
    if (seen.has(run)) return { problem: `${where} is for run ${run} again.` };
    seen.add(run);
    rows.push(rowOf.get(run));
  }
  return { rows };
}

/** The rows a pasted block of ``count`` lines fills in place: down the sheet as it is shown, from the row pasted into. */
function rowsInPlace(count, sheet, at) {
  const start = sheet.order.indexOf(at.index);
  if (start + count > sheet.order.length) {
    const room = sheet.order.length - start;
    return { problem: `You pasted ${count} lines, and the run sheet shows ${room} runs from the one pasted into down.` };
  }
  return { rows: sheet.order.slice(start, start + count) };
}

/** The response columns a pasted block of ``width`` columns of results fills: from the one pasted into, across. */
function columnsInPlace(width, sheet, at) {
  const names = sheet.names.slice(sheet.names.indexOf(at.name));
  if (width > names.length) {
    const from = sheet.labels[at.name] ?? at.name;
    return { problem: `You pasted ${width} columns of results, and the run sheet has ${names.length} from ${from} on.` };
  }
  return { names: names.slice(0, width) };
}

/**
 * How a pasted block maps onto the run sheet: the row and the response column of each of its cells, or a problem.
 * A first line of headings names the columns: the one headed Run order (or Run) matches each line to its run, and
 * each one headed by a response's name or column holds its results, the others (the factors', say) passed over;
 * without a Run order column the lines go in place. With no headings, a first column of run numbers, each on the
 * sheet once, matches the lines to their runs; else the block goes in place.
 */
function pastedPlan(cells, sheet, at) {
  const heading = cells[0].some((cell) => cell !== "" && pastedNumber(cell) === undefined);
  if (heading) {
    const [head, ...lines] = cells;
    const label = (name) => (sheet.labels[name] ?? name).toLowerCase();
    const names = head.map((h) => sheet.names.find((name) => [name.toLowerCase(), label(name)].includes(headingKey(h))) ?? null);
    if (!names.some(Boolean)) {
      const wanted = listNames(sheet.names.map((name) => sheet.labels[name] ?? name));
      return { problem: `No heading of what you pasted names a response of the run sheet: ${wanted}.` };
    }
    const runAt = head.findIndex((h) => RUN_HEADINGS.includes(headingKey(h)));
    const rows = runAt >= 0 ? rowsByRun(lines, runAt, sheet, 2) : rowsInPlace(lines.length, sheet, at);
    const how = runAt >= 0 ? "matched to the runs by the Run order column" : null;
    return rows.problem ? rows : { lines, first: 2, rows: rows.rows, names, how };
  }
  const width = Math.max(...cells.map((line) => line.length));
  const byRun = width >= 2 ? rowsByRun(cells, 0, sheet, 1) : { problem: "one column" };
  if (!byRun.problem) {
    const columns = columnsInPlace(width - 1, sheet, at);
    if (columns.problem) return columns;
    const how = "matched to the runs by the run numbers in the first column";
    return { lines: cells, first: 1, rows: byRun.rows, names: [null, ...columns.names], how };
  }
  const rows = rowsInPlace(cells.length, sheet, at);
  const columns = columnsInPlace(width, sheet, at);
  return rows.problem ? rows : columns.problem ? columns : { lines: cells, first: 1, rows: rows.rows, names: columns.names, how: null };
}

/**
 * Where the results of a block pasted into the run sheet go, as a spreadsheet copies it (see pastedPlan). ``sheet``
 * is {runs, order, names, labels}: each row's run number, the row indices in the order the sheet shows them, the
 * response columns, and each response's name as the reader gave it. ``at`` is the box pasted into, {index, name}.
 * Returns {values: [{index, name, value}], said}, or {problem} for a block that does not fit, of which nothing is
 * pasted. A blank cell leaves the result there as it is.
 */
export function placePasted(cells, sheet, at) {
  if (!cells.length) return { values: [], said: "Nothing to paste: every cell was blank." };
  const plan = pastedPlan(cells, sheet, at);
  if (plan.problem) return { problem: `${plan.problem} Nothing was pasted.` };
  const values = [];
  for (const [j, line] of plan.lines.entries()) {
    for (const [c, name] of plan.names.entries()) {
      if (!name) continue;
      const value = pastedNumber(line[c] ?? "");
      if (value === undefined) {
        return { problem: `Line ${plan.first + j} of what you pasted holds "${line[c]}", which is not a number. Nothing was pasted.` };
      }
      if (value !== null) values.push({ index: plan.rows[j], name, value });
    }
  }
  const touched = sheet.names.filter((name) => values.some((v) => v.name === name)).map((name) => sheet.labels[name] ?? name);
  const how = plan.how ?? `down the run sheet from run ${sheet.runs[at.index]}`;
  const said = values.length
    ? `Pasted ${values.length} result${values.length === 1 ? "" : "s"} of ${listNames(touched)}, ${how}.`
    : "Nothing to paste: every cell was blank.";
  return { values, said };
}

// ---------------------------------------------------------------- limits on the settings
/** The most limits a study can have on its settings; mirrors bootstrap.MAX_CONSTRAINTS. */
export const MAX_CONSTRAINTS = 8;

/** The most characters of a limit's note, which the page shows beside it and never reads; mirrors bootstrap.NOTE_LENGTH. */
export const NOTE_LENGTH = 80;

/** The longest limit Python reads; mirrors process_improve.experiments.designs_constrained.MAX_EXPRESSION_LENGTH. */
export const EXPRESSION_LENGTH = 500;

/** The designs that keep every run inside limits on the settings; mirrors bootstrap.REGION_DESIGNS. */
export const REGION_DESIGNS = ["d_optimal", "i_optimal"];

/**
 * What a reader may type or paste in a limit for an operator, and the operator a limit is written with;
 * mirrors bootstrap._SYMBOLS: the signs for at most, at least and not equal, the minus sign and the en
 * dash, the multiplication sign and the middle dot, the division sign, and ^ for a power, as a
 * spreadsheet writes it.
 */
export const LIMIT_SYMBOLS = {
  "≤": "<=",
  "≥": ">=",
  "≠": "!=",
  "−": "-",
  "–": "-",
  "×": "*",
  "·": "*",
  "÷": "/",
  "^": "**",
};

/**
 * A limit's text in tokens, as Python reads it: a number (so the e of 1e5 is no name), a name, a
 * run of spaces, or any other character. A name followed by "(" is a function's, such as sqrt.
 */
const TOKEN =
  /(0[xXoObB][\dA-Fa-f_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?[jJ]?)|([\p{ID_Start}_][\p{ID_Continue}]*)|(\s+)|([^])/gu;

function limitTokens(text) {
  const tokens = [...String(text).matchAll(TOKEN)].map(([whole, number, name, space]) => ({
    text: whole,
    kind: number ? "number" : name ? "name" : space ? "space" : "other",
  }));
  for (const [i, token] of tokens.entries()) {
    const next = tokens.slice(i + 1).find((t) => t.kind !== "space");
    token.called = token.kind === "name" && next?.text === "(";
  }
  return tokens;
}

/**
 * A limit with each factor's name replaced by its place in the brief (#0, #1, ...), spaces dropped and
 * typed signs read as the operators they stand for: what keys a run sheet to the limits it was made
 * with. Renaming a factor leaves it as it was; changing a limit, or the order of the factors, does not.
 */
export function positional(expression, names) {
  const read = Object.entries(LIMIT_SYMBOLS).reduce((text, [sign, op]) => text.replaceAll(sign, op), String(expression));
  return limitTokens(read)
    .filter((t) => t.kind !== "space")
    .map((t) => (t.kind === "name" && !t.called && names.includes(t.text) ? `#${names.indexOf(t.text)}` : t.text))
    .join("");
}

/**
 * A limit with its factors renamed as ``names`` maps them (old name to new), all at once, so a swap
 * works; a function's name, such as sqrt, and the spacing typed are left as they were.
 */
export const renameLimit = (expression, names) =>
  limitTokens(expression)
    .map((t) => (t.kind === "name" && !t.called && Object.hasOwn(names, t.text) ? names[t.text] : t.text))
    .join("");

/** A straight limit written from its left side and its bounds, as a bound box rebuilds it; null for an open side. */
export function withBounds(lhs, low, high) {
  if (low !== null && high !== null) return `${low} <= ${lhs} <= ${high}`;
  if (high !== null) return `${lhs} <= ${high}`;
  return `${lhs} >= ${low}`;
}

/**
 * Text with ``piece`` put in place of the selection from ``start`` to ``end``, set apart by a space from
 * a name, a number or a bracket beside it, so a name tapped in after another does not join it; and where
 * the caret goes after it.
 */
export function insertAt(text, start, end, piece) {
  const before = text.slice(0, start);
  const after = text.slice(end);
  const lead = /[\p{ID_Continue})]$/u.test(before) ? " " : "";
  const trail = /^[\p{ID_Continue}(]/u.test(after) ? " " : "";
  const placed = `${lead}${piece}${trail}`;
  return { text: `${before}${placed}${after}`, caret: start + lead.length + piece.length };
}

/**
 * A percent to ``digits`` decimals, a half to the even neighbour. Rounding a half up writes an eighth and
 * the seven eighths it leaves as 13% and 88%; to the even neighbour they are 12% and 88%, so a share and
 * what it leaves always sum to 100%.
 */
function evenPercent(percent, digits) {
  const scaled = percent * 10 ** digits;
  const below = Math.floor(scaled);
  const half = Math.abs(scaled - below - 0.5) < 1e-6;
  return (half ? below + (below % 2) : Math.round(scaled)) / 10 ** digits;
}

/**
 * A share of the factors' ranges as the page writes it: a whole percent, a tenth within 1% of none or of
 * all, and never a false 0% or 100%. A share and what it leaves sum to 100%, so a lone limit's "rules out"
 * agrees with the "leave" written beside it.
 */
export function shareText(share) {
  if (share <= 0) return "none";
  if (share < 0.001) return "less than 0.1%";
  if (share > 0.999 && share < 1) return "more than 99.9%";
  const digits = share < 0.01 || (share > 0.99 && share < 1) ? 1 : 0;
  return `${evenPercent(100 * share, digits).toFixed(digits)}%`;
}

/**
 * What one limit does over the factors' ranges, as the panel says it under the limit: the share it rules
 * out and how many corners of the ranges, from api_limits' description of it. "" for a limit that cannot be
 * read, which says its problem instead.
 */
export function limitEffect(entry, probes) {
  if (entry.problem) return "";
  if (entry.redundant === "alone") return "Rules out nothing within your ranges.";
  if (entry.redundant === "others") return "Rules out nothing the other limits do not.";
  // From here the limit rules out a part of the ranges, even one too thin for the sample or the rounding to see.
  const share = `Rules out ${entry.removed > 0 ? shareText(entry.removed) : "less than 0.1%"} of your ranges`;
  if (!probes || !entry.removes) return `${share}.`;
  const corners = entry.removes.filter((row) => row !== probes.center).length;
  const centre = entry.removes.includes(probes.center) ? ", and their center" : "";
  return `${share}: ${corners || "none"} of the ${probes.center} corners${centre}.`;
}

/** A value in a factor's units, as the panel writes it: four significant figures, and the units after. */
const inUnits = (value, units) => `${Number(Number(value).toPrecision(4))}${units ? ` ${units}` : ""}`;

/**
 * What all the limits leave of the factors' ranges, a sentence each, from api_limits' reply: the share
 * kept, whether the center of the ranges can still be run, and each range they narrow, in the factors'
 * units. Each is ``{text, error}``. A refusal that holds the brief back comes first, unless it is a
 * limit's own problem, which is said under that limit.
 */
export function limitsSummary(reply, factors) {
  const read = reply.constraints.filter((c) => !c.problem);
  const own = reply.constraints.some((c) => c.problem === reply.refusal);
  const said = reply.refusal && !own ? [{ text: reply.refusal, error: true }] : [];
  if (!read.length || reply.empty) return said;
  // Some setting can be run, and a limit that rules out anything leaves less than all, however thin either part.
  const cuts = read.some((c) => c.redundant !== "alone");
  const kept =
    reply.kept <= 0 ? "less than 0.1%" : cuts && reply.kept >= 1 ? "more than 99.9%" : shareText(reply.kept);
  said.push({ text: `The limits leave ${kept} of your ranges to run.`, error: false });
  if (!reply.center_allowed) said.push({ text: "The center of the ranges cannot be run.", error: false });
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  for (const b of reply.bounds ?? []) {
    const u = units[b.name];
    const text =
      b.low !== null && b.high !== null
        ? `So ${b.name} stays between ${inUnits(b.low, "")} and ${inUnits(b.high, u)}.`
        : b.high !== null
          ? `So ${b.name} never goes above ${inUnits(b.high, u)}.`
          : `So ${b.name} never goes below ${inUnits(b.low, u)}.`;
    said.push({ text, error: false });
  }
  return said;
}

/** A setting of the factors as the page names it: "T 200 degC, P 3 bar, F 10 L/min", each value to four figures. */
export function settingLabel(values, columns, factors) {
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  return columns.map((name, j) => `${name} ${inUnits(values[j], units[name])}`).join(", ");
}

/** From this many numeric factors the table of corners can be narrowed, factor by factor, to the corners wanted. */
export const FILTER_CORNERS_FROM = 5;

/** The most rows of corners shown at once; more come on request. */
export const CORNERS_SHOWN = 32;

/**
 * The rows of the table of corners a reader's filters leave: each factor ``"low"``, ``"high"`` or
 * ``"any"`` (the default), by column. The centre is no corner, and is left to the caller.
 */
export function filterCorners(rows, columns, factors, filters = {}) {
  const ends = Object.fromEntries(factors.map((f) => [f.name, f]));
  return rows
    .map((row, i) => [row, i])
    .filter(([row]) =>
      columns.every((name, j) => {
        const want = filters[name] ?? "any";
        return want === "any" || row[j] === (want === "low" ? ends[name].low : ends[name].high);
      }),
    )
    .map(([, i]) => i);
}

/**
 * The question asked of one factor when a proposed limit is adjusted from its corner: with the others
 * held at the corner, how far the factor can go before the limit is met. From a factor's high end it
 * "can go up to" the cut; from its low end it "must be at least" the cut.
 */
export const cutWords = (cut, factor) => (cut.at === factor.high ? "can go up to" : "must be at least");

/** The settings a proposed limit rules out that the reader did not mark, and no limit rules out already. */
export const alsoRuledOut = (proposal, marked, ruledOutBy) =>
  (proposal.removes ?? []).filter((row) => !marked.includes(row) && !ruledOutBy[row]?.length);

/** Up to this many corners a sentence names one by one; from more, the first two and how many others. */
const CORNERS_NAMED = 3;

/**
 * Settings of the table of corners named in a sentence, by their rows of ``probes``. One corner is named
 * by its settings. Corners that make up a face of the ranges are named by what they share ("the 2
 * corners where T is 200 degC and P is 3 bar"), since that is how a reader marks them; other corners
 * one by one, up to three. The center of the ranges comes last.
 */
export function cornersWords(rows, probes, factors) {
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const { columns } = probes;
  const corners = rows.filter((row) => row !== probes.center);
  const label = (row) => settingLabel(probes.rows[row], columns, factors);
  const parts = [];
  if (corners.length === 1) parts.push(`the corner ${label(corners[0])}`);
  if (corners.length > 1) {
    const fixed = columns.filter((_, j) => new Set(corners.map((row) => probes.rows[row][j])).size === 1);
    const at = (name) => `${name} is ${inUnits(probes.rows[corners[0]][columns.indexOf(name)], units[name])}`;
    if (corners.length === 2 ** (columns.length - fixed.length)) {
      parts.push(fixed.length ? `the ${corners.length} corners where ${listNames(fixed.map(at))}` : "every corner");
    } else {
      const named =
        corners.length <= CORNERS_NAMED ? corners.map(label) : [...corners.slice(0, 2).map(label), `and ${corners.length - 2} more`];
      parts.push(`the ${corners.length} corners (${named.join("; ")})`);
    }
  }
  if (rows.includes(probes.center)) parts.push("the center of the ranges");
  return parts.join(", and ");
}

// ---------------------------------------------------------------- colour mode
/** The colour modes the switch in the header offers: the system's, light and dark, as the book's does. */
export const MODES = ["auto", "light", "dark"];

/**
 * The mode the switch goes to next, in the book's order: from the system's own to the opposite of
 * what the system shows, so the first press always changes the page, then to the other.
 */
export function nextMode(current, systemDark) {
  const order = systemDark ? ["auto", "light", "dark"] : ["auto", "dark", "light"];
  return order[(order.indexOf(current) + 1) % order.length];
}

/** The theme a mode shows, given whether the system is dark. */
export const shownTheme = (mode, systemDark) => (mode === "auto" ? (systemDark ? "dark" : "light") : mode);
