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
 * Each part of the page that a level brings, with the term of the concept map (map/graph.json) whose level it
 * takes: tests/web/page/test_levels.py keeps the two in step. A part with no term of its own names why in `because`.
 */
export const FEATURES = {
  "design:full_factorial": { level: 1, term: "full_factorial" },
  "design:fractional_factorial": { level: 2, term: "fractional_factorial" },
  "design:plackett_burman": { level: 2, term: "plackett_burman" },
  "design:taguchi": { level: 2, because: "a screening array, read as a fractional factorial is" },
  "design:box_behnken": { level: 2, term: "box_behnken" },
  "design:ccd": { level: 2, term: "ccd" },
  "design:dsd": { level: 3, term: "dsd" },
  "design:d_optimal": { level: 3, term: "d_optimal" },
  "design:i_optimal": { level: 3, term: "i_optimal" },
  "design:maximin_lhs": { level: 3, because: "made for no model, which the optimal designs lead up to" },
  "design:simplex_lattice": { level: 3, term: "mixture_design" },
  "design:simplex_centroid": { level: 3, term: "mixture_design" },
  "design:extreme_vertices": { level: 3, term: "mixture_design" },
  "chart:main_effects": { level: 1, term: "main_effect" },
  "chart:interaction": { level: 1, term: "interaction_plot" },
  "chart:cube": { level: 1, term: "cube_plot" },
  "chart:normal_probability": { level: 1, term: "residual_analysis" },
  "chart:residuals_order": { level: 1, term: "residual_analysis" },
  "chart:perturbation": { level: 2, term: "response_surface" },
  "chart:surface_3d": { level: 2, term: "response_surface" },
  "chart:prediction_variance": { level: 3, term: "prediction_variance" },
  "table:diagnostics": { level: 2, because: "leverage and Cook's distance, past the residuals a beginner reads" },
};

/** The level that first shows a part of the page: 1 for a part every level has. */
export const levelOf = (feature) => FEATURES[feature]?.level ?? 1;

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
