// The reading level of the page: Beginner, Intermediate or Adventurous, as the concept map has them. A level shows
// every part of the page of the levels below it, so a reader meets the designs and charts in the book's order and
// the page is no fuller than they need. This module needs no document, so it is tested under Node.js.

/** The levels, in order: level 1 is the first. */
export const LEVELS = ["Beginner", "Intermediate", "Adventurous"];

/** Where the level is kept: the concept map's own key, so a choice made on either page holds on both. */
export const LEVEL_KEY = "factorial-map-level";

/** The level of a first visit. The map opens on Beginner; this page opens on the designs most studies need. */
export const DEFAULT_LEVEL = 2;

/**
 * Each part of the page that a level brings, but for the charts, whose levels are in logic.CHARTS: with the term
 * of the concept map (map/graph.json) whose level it takes. tests/web/page/test_levels.py keeps the two in step. A
 * part with no term of its own names why in `because`. A part that is in index.html from the first paint carries
 * its key there as data-feature, beside its level, and the same test checks the two agree.
 */
export const FEATURES = {
  "design:full_factorial": { level: 1, term: "full_factorial" },
  "design:fractional_factorial": { level: 2, term: "fractional_factorial" },
  "design:plackett_burman": { level: 2, term: "plackett_burman" },
  "design:taguchi": { level: 2, because: "a screening array, read as a fractional factorial is" },
  "design:box_behnken": { level: 2, term: "box_behnken" },
  "design:ccd": { level: 2, term: "ccd" },
  "design:dsd": { level: 3, term: "dsd" },
  "design:omars": { level: 3, term: "omars_design" },
  "design:d_optimal": { level: 3, term: "d_optimal" },
  "design:i_optimal": { level: 3, term: "i_optimal" },
  "design:a_optimal": { level: 3, term: "a_optimal" },
  "design:maximin_lhs": { level: 3, because: "made for no model, which the optimal designs lead up to" },
  "design:sobol": { level: 3, because: "made for no model, as the Latin hypercube is" },
  "design:simplex_lattice": { level: 3, term: "mixture_design" },
  "design:simplex_centroid": { level: 3, term: "mixture_design" },
  "design:extreme_vertices": { level: 3, term: "mixture_design" },
  "action:simplify": { level: 2, because: "choosing a model's terms from the results, past fitting the model as designed" },
  "table:diagnostics": { level: 2, because: "leverage and Cook's distance, past the residuals a beginner reads" },
  "option:blocks": { level: 2, term: "blocking" },
  "panel:limits": { level: 2, term: "constraint" },
  "action:add-response": { level: 2, term: "multiple_responses" },
  "action:compare": { level: 2, because: "nine measures of each design, past its runs and the model it fits" },
  "panel:curves": { level: 2, because: "a response that is a whole curve at each run, past one number a run" },
};

/** The level that first shows a part of the page: 1 for a part every level has. */
export const levelOf = (feature) => FEATURES[feature]?.level ?? 1;

/**
 * Whether the list of designs offers a design at a level: a design of that level or of one below, the design the
 * plan holds, whatever its level, and any design a study has called for besides (runs as given, which no level
 * offers, once a study of given runs has been open).
 */
export function designListed(designType, level, { chosen = null, also = [] } = {}) {
  if (designType === chosen || also.includes(designType)) return true;
  const feature = FEATURES[`design:${designType}`];
  return Boolean(feature) && feature.level <= level;
}

/** The level a stored choice names, or the first visit's when there is none or it is not a level. */
export function readLevel(stored) {
  const level = Number(stored);
  return stored !== null && stored !== "" && Number.isInteger(level) && level >= 1 && level <= LEVELS.length ? level : DEFAULT_LEVEL;
}

/**
 * What a level leaves out of a list, said where the list is: "3 more designs at Adventurous", or over two levels
 * "2 more charts at Intermediate, 1 at Adventurous". ``hidden`` holds the level of each part left out; "" for none.
 */
export function moreWords(hidden, noun) {
  const at = LEVELS.map((name, i) => [name, hidden.filter((level) => level === i + 1).length]).filter(([, n]) => n);
  if (!at.length) return "";
  const [[first, n], ...rest] = at;
  const plural = n === 1 && !rest.length ? noun : `${noun}s`;
  return [`${n} more ${plural} at ${first}`, ...rest.map(([name, count]) => `${count} at ${name}`)].join(", ");
}
