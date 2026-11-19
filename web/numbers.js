// The decimal mark the page writes its numbers with: a period unless the study's settings say a comma. A number
// is formatted as it always was, with a point, and passed through marked() where it is shown. What a machine
// reads stays with a point: a box of type number, the Python code, a link and the study's file. This module
// needs no document, so it is tested under Node.js. The study's significance level is kept here too: the page's
// words and its tests read it where they name an interval or judge a p-value.

/** The decimal marks a study's settings choose among, by the names the settings keep; the first is the default. */
export const DECIMAL_MARKS = { period: ".", comma: "," };

let mark = ".";

/** Write numbers with this mark from here on: "comma", or the period for any other name. */
export function setDecimalMark(name) {
  mark = name === "comma" ? "," : ".";
}

/** The name of the mark numbers are written with now: "period" or "comma". */
export const decimalMark = () => (mark === "," ? "comma" : "period");

/** A number written with a decimal point, as the page shows it: with the reader's mark. Text with no point is as it was. */
export const marked = (text) => (mark === "." ? text : String(text).replace(".", ","));

/**
 * A number typed with the reader's mark, written with a point: "1,5" is "1.5" while the mark is a comma. Text that
 * is no such number is as it was, so a label such as "A,B" stays a label.
 */
export const pointed = (text) => (mark === "," && /^\s*[-+]?\d*,\d+\s*$/.test(String(text)) ? String(text).replace(",", ".") : text);

/**
 * An equation Python wrote, "Yield = 61.95 + 3.95·T", with the reader's mark in each of its numbers. Only what
 * follows the "=" is changed: the response's name before it is the reader's own text.
 */
export function markedEquation(equation) {
  if (mark === ".") return equation;
  const at = String(equation).indexOf(" = ");
  if (at < 0) return equation;
  return equation.slice(0, at) + equation.slice(at).replace(/(\d)\.(\d)/g, "$1,$2");
}

/** The significance levels a study's settings choose among, as bootstrap.py takes them. */
export const SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.1];
/** The level of a study that sets none, as every study before the setting. */
export const DEFAULT_SIGNIFICANCE = 0.05;

let level = DEFAULT_SIGNIFICANCE;

/** Judge every test at this significance level from here on; a level the page does not offer is the default. */
export function setSignificance(value) {
  level = SIGNIFICANCE_LEVELS.includes(value) ? value : DEFAULT_SIGNIFICANCE;
}

/** The study's significance level: a p-value below it is a finding. */
export const significance = () => level;

/** The confidence of every interval, one minus the significance level, as the page writes it: "95%". */
export const confidencePercent = () => `${Math.round((1 - level) * 100)}%`;
