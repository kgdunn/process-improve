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
