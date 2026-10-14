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
  if (model === "quadratic") terms.push(...squared(factors));
  return terms;
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

/** Whether the factors in the form still describe the design ``spec`` holds. */
export function formMatches(factors, spec) {
  if (!spec?.factors) return true;
  const level = (v) => (isLabel(v) ? String(v).trim() : Number(v));
  const same = (f, g) =>
    f.name === g.name && f.low === level(g.low) && f.high === level(g.high) && f.units === (g.units ?? "");
  return factors.length === spec.factors.length && factors.every((f, i) => same(f, spec.factors[i]));
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
    `The fitted model over ${pair.join(" and ")}, the two factors with the largest main effects, with any other ` +
    "factor at the center of its range; the dots are the runs, in coded units.",
  true_contour: (pair) => `The process the results were simulated from, over ${pair.join(" and ")}, on the same scale.`,
  residuals:
    "What the model leaves unexplained in each run, against its fitted value. A curve or a funnel points to a " +
    "missing term or a changing noise level.",
  half_normal:
    "With no degrees of freedom left to estimate the error, the effects are judged against each other: effects " +
    "far above the line, past Lenth's margin of error, are likely real.",
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
