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

/** The resolution a fraction in k factors opens on: IV, or III when three factors allow no more. */
export const defaultResolution = (k) => Math.max(3, Math.min(4, k));

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
export const CATEGORICAL_DESIGNS = ["full_factorial", "fractional_factorial", "plackett_burman"];

/** Whether text typed as a level is a label, such as old: neither blank nor a number (nor inf or nan). */
export function isLabel(text) {
  const t = String(text ?? "").trim();
  return t !== "" && Number.isNaN(Number(t)) && !/^[+-]?(inf(inity)?|nan)$/i.test(t);
}

/** A level as the form sends it: a label as its text, a blank box as null, and anything else as a number. */
export const levelValue = (text) => (isLabel(text) ? String(text).trim() : numberOrNull(String(text ?? "")));

/** Whether a factor is categorical: both its levels are labels, as bootstrap.py reads them. */
export const isCategorical = (f) => isLabel(f.low) && isLabel(f.high);

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
export function levelProblem({ name, low, high }, type) {
  const who = String(name ?? "").trim() || "A factor without a name";
  const kinds = [levelKind(low), levelKind(high)];
  if (type === CATEGORICAL) {
    if (kinds.includes("blank")) return `${who}: give it two labels, such as A and B.`;
    if (kinds.includes("number")) return `${who}: a label cannot be a number; write C1 rather than 1.`;
    if (String(low).trim() === String(high).trim()) return `${who}: its two labels must differ.`;
    return "";
  }
  if (kinds.includes("blank") || kinds.includes("label")) {
    return `${who}: give a number for both the low and the high value, or make it categorical for two labels.`;
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

/** The aim a study starts from: which factors matter, for many factors; else where each one's effect curves. */
export const defaultAim = (k) => (k >= SCREEN_FROM ? "main_effects" : "main_quadratic");

/** Count the coefficients, intercept included, of the model an aim names, over these factors: no square of a categorical one. */
export function aimTerms(factors, aim) {
  const k = factors.length;
  const pairs = aim === "interactions" || aim === "quadratic" ? (k * (k - 1)) / 2 : 0;
  const squares = aim === "main_quadratic" || aim === "quadratic" ? factors.filter((f) => !isCategorical(f)).length : 0;
  return 1 + k + pairs + squares;
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
 * The design picked when the reader has not chosen one: of those that can fit the model, the fewest
 * runs that leave SPARE_FOR_NOISE to estimate the noise, else the fewest runs; null when none fits.
 */
export function pickCandidate(rows) {
  const fitting = rows.filter((r) => r.fits);
  const spare = fitting.filter((r) => r.spare >= SPARE_FOR_NOISE);
  return (spare[0] ?? fitting[0] ?? null)?.design_type ?? null;
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

/** A p-value to four decimals, or "< 0.0001". */
export const pval = (p) => (p === null || p === undefined ? "" : p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4));

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
};

/** What each chart shows and how to read it; some name the factors drawn. */
export const CAPTIONS = {
  contour: (pair) =>
    `Drawn over ${pair.join(" and ")}, the two factors with the largest main effects, with the others at their ` +
    "center; the dots are the runs.",
  true_contour: "The process the results were simulated from, on the same scale.",
  residuals: "A curve or a funnel points to a missing term or a changing noise level.",
  half_normal: "Effects far above the line, past Lenth's margin of error, are likely real.",
};

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

/** A share of the factors' ranges as the page writes it: a whole percent, a tenth below 1%, and never a false 0% or 100%. */
export function shareText(share) {
  if (share <= 0) return "none";
  if (share < 0.001) return "less than 0.1%";
  if (share > 0.999 && share < 1) return "more than 99.9%";
  if (share < 0.01) return `${(100 * share).toFixed(1)}%`;
  return `${Math.round(100 * share)}%`;
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
  const share = `Rules out ${shareText(entry.removed)} of your ranges`;
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
  said.push({ text: `The limits leave ${shareText(reply.kept)} of your ranges to run.`, error: false });
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
