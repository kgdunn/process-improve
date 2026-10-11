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

/** Read a number box as the page sends it: a blank box as null, rather than as Number("") = 0. */
export const numberOrNull = (text) => (text.trim() === "" ? null : Number(text));

/** The response a blank box names: y, as Python names it when it makes the design. */
export const DEFAULT_RESPONSE = "y";

/** Whether the factors and response in the form still describe the design ``spec`` holds; a blank response is y. */
export function formMatches(factors, response, spec) {
  if (!spec?.factors) return true;
  const same = (f, g) =>
    f.name === g.name && f.low === Number(g.low) && f.high === Number(g.high) && f.units === (g.units ?? "");
  return (
    (response || DEFAULT_RESPONSE) === spec.response &&
    factors.length === spec.factors.length &&
    factors.every((f, i) => same(f, spec.factors[i]))
  );
}

/** Join names as a sentence does: "A", "A and B", "A, B and C". */
export const listNames = (names) =>
  names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names.join("");

/** A number to four significant figures, or nothing for a value the fit could not give. */
export const fmt = (x) => (x === null || x === undefined ? "" : Number(x).toPrecision(4));

/** A p-value to four decimals, or "< 0.0001". */
export const pval = (p) => (p === null || p === undefined ? "" : p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4));

/** The fit statistics as a sentence: one the fit cannot give is left out, and a missing predicted R² is explained. */
export function fitStatistics(s) {
  const known = (x) => x !== null && x !== undefined;
  const stats = [
    ["R²", s.r_squared],
    ["adjusted R²", s.r_squared_adj],
    ["predicted R²", s.r_squared_pred],
  ];
  const given = stats.filter(([, x]) => known(x)).map(([name, x]) => `${name} = ${fmt(x)}`);
  const text = `${[...given, `residual degrees of freedom = ${s.df_residual}`].join(", ")}.`;
  // A run with leverage 1 is fitted exactly whatever its result, so the fit without it cannot predict it.
  return known(s.r_squared_pred)
    ? text
    : `${text} Predicted R² cannot be computed: at least one run cannot be predicted from the others.`;
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
  coefficients: (simulated) =>
    "Each dot is an estimate with its 95% interval; an interval that crosses zero is a term the data cannot tell " +
    "apart from no effect." +
    (simulated ? " A diamond marks the true value, so you can see what the analysis recovered and what it missed." : ""),
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
