// The best settings for the response's goal: what Learn says of them, and the confirmation runs made
// at them. Python finds them (bootstrap.best_settings) and says where they lie as facts; this module
// turns the facts into the page's sentences and keeps the run sheet's confirmation runs. Nothing here
// touches the document, so tests/web/js runs it under Node.

/** The run sheet's column marking a confirmation run, and what it holds for one; mirrors bootstrap.CONFIRMATION and CONFIRMS. */
export const CONFIRMATION = "Confirmation run";
export const CONFIRMS = "yes";

/** How many confirmation runs the page adds at the best settings; mirrors bootstrap.CONFIRMATION_RUNS. */
export const CONFIRMATION_RUNS = 3;

/** How far each factor moves for the chart of how tightly to hold it: this share of its range; mirrors bootstrap.NUDGE. */
export const NUDGE = 0.05;

/** A number as a sentence gives it: four significant figures, with no trailing zeros. */
export const num = (x) => String(Number(Number(x).toPrecision(4)));

/** A quantity with its units: "200 degC", "68.25%" (no space before a percent sign), or the number alone. */
export const withUnit = (x, units = "") => (!units ? num(x) : units === "%" ? `${num(x)}%` : `${num(x)} ${units}`);

/** Join words as a sentence does: "A", "A and B", "A, B and C". */
const listed = (items) => (items.length > 1 ? `${items.slice(0, -1).join(", ")} and ${items.at(-1)}` : items.join(""));

/** A factor's best setting as the page writes it: "T = 200 degC", or "Supplier at B" for a categorical factor. */
export function settingWords(setting, units = "") {
  return setting.kind === "categorical" ? `${setting.name} at ${setting.value}` : `${setting.name} = ${withUnit(setting.value, units)}`;
}

/** What the response does past an end or a limit, for the goal: "a higher Yield", "a lower Yield", "Yield nearer its target". */
function better(best, label) {
  if (best.goal === "maximize") return `a higher ${label}`;
  if (best.goal === "minimize") return `a lower ${label}`;
  return `${label} nearer its target`;
}

/** What the surface does along a factor inside its range, when others are at an end of theirs. */
const ALONG = { maximum: "at its peak", minimum: "at its lowest", saddle: "at a saddle", ridge: "on a ridge" };

/**
 * Where a factor's best setting lies, as the table of settings says it: at an end of its range, and whether the goal
 * may be better past it; inside it, and at the surface's peak along it; or not in the model.
 */
export function whereWords(setting, best = null, { label = "" } = {}) {
  if (!setting.in_model) return "not in the model";
  if (setting.kind === "categorical") return "the better level";
  if (setting.end) {
    const past = setting.end === "high" ? "above" : "below";
    return setting.beyond ? `${setting.end} end; ${better(best, label)} may lie ${past}` : `${setting.end} end of its range`;
  }
  const along = best?.stationary && !best.stationary.inside && best.stationary.factors.includes(setting.name);
  return along ? `inside its range, ${ALONG[best.stationary.kind]}` : "inside its range";
}

/** The goal in words, for the response named ``label``: "the highest Yield", "the lowest Yield". */
const SUPERLATIVE = { maximize: "highest", minimize: "lowest" };

/**
 * What the panel says first: the prediction the best settings give, and that it is the goal's best. The settings
 * themselves follow in a table. ``words`` is the response's {label, units}.
 */
export function leadSentence(best, { label, units }) {
  const predicted = withUnit(best.predicted, units);
  if (best.goal !== "target") return `These settings give the ${SUPERLATIVE[best.goal]} ${label} the model predicts: ${predicted}.`;
  if (best.reached) return `These settings give ${label} on its target, ${withUnit(best.target, units)}, as the model predicts.`;
  return `No setting in your ranges reaches the target of ${withUnit(best.target, units)}. These come closest: ${predicted}.`;
}

/** A leverage as the page writes it: two significant figures, enough to compare with the runs'. */
const lev = (h) => String(Number(Number(h).toPrecision(2)));

/**
 * How well the model knows the response at the best settings, a row each for a small table: its two intervals, and
 * its leverage beside the runs'. Each row is its ``label``, the ``term`` in the label the glossary entry ``key``
 * defines, and its ``value``.
 */
export function intervalRows(best, { units }) {
  const level = `${Math.round(100 * (best.level ?? 0.95))}%`;
  const range = ([low, high]) => `${num(low)} to ${withUnit(high, units)}`;
  const rows = [];
  if (best.confidence) {
    const label = `${level} confidence interval for the mean`;
    rows.push({ label, term: "confidence interval", key: "confidence-interval", value: range(best.confidence) });
  }
  if (best.prediction) {
    const label = `${level} prediction interval for one new run`;
    rows.push({ label, term: "prediction interval", key: "prediction-interval", value: range(best.prediction) });
  }
  const value = `${lev(best.leverage)}, ${best.extrapolating ? "above" : "within"} the runs': ${lev(best.max_leverage)} at most`;
  rows.push({ label: "Leverage", term: "Leverage", key: "leverage", value });
  return rows;
}

/** What the panel says when the fit left no runs to estimate the noise, so there is no interval; else nothing. */
export const noIntervalSentence = (best) =>
  best.confidence ? "" : "No runs are left to estimate the noise, so the model gives no interval here.";

/** What the panel says when the best settings lean on the model beyond the runs; else nothing. */
export const extrapolationSentence = (best) =>
  best.extrapolating ? "These settings lie beyond the runs, so the prediction leans on the model there: confirm it with a run." : "";

/**
 * The way a next study could move its ranges, along the steepest ascent (or descent): each factor that moves, up or
 * down, by its share of half its range, to two significant figures, as a direction is, not a setting. ``factors``
 * are the spec's, for their ranges and units.
 */
export function ascentSentence(best, factors) {
  const shares = Object.entries(best.ascent ?? {});
  if (!shares.length) return "";
  const range = Object.fromEntries(factors.map((f) => [f.name, f]));
  const step = ([name, share]) => {
    const f = range[name];
    const size = Number(((Math.abs(share) * (f.high - f.low)) / 2).toPrecision(2));
    return { name, share, size: withUnit(size, f.units ?? "") };
  };
  const steps = shares.map(step).sort((a, b) => Math.abs(b.share) - Math.abs(a.share));
  const way = (s) => (s.share > 0 ? "up" : "down");
  if (steps.length === 1) return `A next study could move the range of ${steps[0].name} ${way(steps[0])}.`;
  const climb = best.goal === "minimize" || (best.goal === "target" && best.predicted > best.target) ? "descent" : "ascent";
  return `A next study could move the ranges along the steepest ${climb}: ${steps.map((s) => `${s.name} ${way(s)} ${s.size}`).join(", ")}.`;
}

/** The shape of the fitted surface at a stationary point inside the ranges, as the page says it. */
const SHAPES = {
  maximum: "the top of the fitted surface",
  minimum: "the bottom of the fitted surface",
  saddle: "a saddle of the fitted surface",
  ridge: "on a ridge of the fitted surface, along which settings predict about the same",
};

/**
 * What the table of settings cannot say, a sentence each: the way a next study could move its ranges, the limits the
 * settings are on, a stationary point inside the ranges, a target that other settings reach too, and the factors
 * the model leaves out. The rest (each factor at an end, and past it) the table says.
 */
export function whereSentences(best, factors, { label }) {
  const said = [];
  const settings = best.settings ?? [];
  if (best.goal === "target" && best.reached) {
    said.push("Other settings reach the target too: its contour on the response surface shows them.");
  } else {
    const limits = best.limits ?? [];
    for (const limit of limits) {
      const which = `limit ${limit.index + 1}, ${limit.expression}`;
      said.push(
        limit.beyond
          ? `These settings are on your ${which}: the model predicts ${better(best, label)} past it, but those settings cannot be run.`
          : `These settings are on your ${which}.`,
      );
    }
    if (settings.some((s) => s.in_model && s.beyond) && !limits.some((l) => l.beyond)) {
      const sentence = ascentSentence(best, factors);
      if (sentence) said.push(sentence);
    }
    if (best.stationary?.inside) said.push(`These settings are a stationary point: ${SHAPES[best.stationary.kind]}.`);
  }
  for (const s of settings.filter((s) => !s.in_model)) {
    said.push(
      s.kind === "categorical"
        ? `${s.name} is not in the model, so either level serves; these settings use ${s.value}.`
        : `${s.name} is not in the model, so it does not change the prediction; these settings hold it at the center.`,
    );
  }
  return said;
}

/** What the chart of how tightly to hold each factor leaves out: the factors the model does not hold; else nothing. */
export function sensitivityCaption(best) {
  const left = (best.sensitivity ?? []).filter((s) => !s.in_model).map((s) => s.name);
  if (!left.length) return "";
  return `${listed(left)} ${left.length === 1 ? "is" : "are"} not in the model, so moving ${left.length === 1 ? "it" : "them"} changes nothing.`;
}

/** The caution when no term of the model is clearly larger than the noise. */
export const UNCLEAR =
  "No term of the model is clearly larger than the noise (every p-value is 0.05 or above), so these settings may follow the noise.";

/** What the panel says for a goal of learning, or a target not yet typed. */
export function noGoalSentence(best, { label }) {
  if (best.goal === "target") return `Type the target for ${label} in the brief, and the page finds the settings that reach it.`;
  return `Set a goal for ${label} in the brief, to maximise, minimise or hit a target, and the page finds the best settings here.`;
}

/** What the panel says when no term of the model changes the response. */
export const flatSentence = ({ label }) => `No term of the model changes ${label}, so no setting is better than another.`;

/**
 * What the page says of one group of confirmation runs: where they are, and once their results are in, whether
 * their average falls inside the prediction interval for an average of that many runs, and what that means.
 */
export function confirmationSentences(group, factors, { label, units }) {
  const unitsOf = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = listed(
    factors.map((f) => (typeof group.at[f.name] === "string" ? `${f.name} at ${group.at[f.name]}` : `${f.name} = ${withUnit(group.at[f.name], unitsOf[f.name])}`)),
  );
  const runs = (n) => `${n} run${n === 1 ? "" : "s"}`;
  const head = `${group.runs} confirmation ${group.runs === 1 ? "run" : "runs"} at ${at}.`;
  const range = group.interval ? `${num(group.interval[0])} to ${withUnit(group.interval[1], units)}` : null;
  if (!group.done) {
    const between = group.interval ? `${num(group.interval[0])} and ${withUnit(group.interval[1], units)}` : null;
    const expect = between ? ` When their average is between ${between}, the model holds there.` : "";
    return [`${head} Enter their results in Run.${expect}`];
  }
  const some = group.done < group.runs ? ` ${group.done} of ${group.runs} results are in.` : "";
  if (!range) {
    return [`${head}${some} With no runs left to estimate the noise, the page cannot judge their average.`];
  }
  const average = `The average of ${group.done === 1 ? "the one result" : `their ${runs(group.done)}`}, ${withUnit(group.mean, units)},`;
  const interval = `the 95% prediction interval for the average of ${runs(group.done)}, ${range}`;
  return group.inside
    ? [`${head}${some}`, `${average} is inside ${interval}: the model holds at these settings.`]
    : [
        `${head}${some}`,
        `${average} is outside ${interval}: the process differs from the model at these settings, so the model misses something there. Look for a term it lacks, or a change in the process.`,
      ];
}

/**
 * The caption of the response surface drawn through the best settings: the factors not drawn are held at their
 * best settings, and a star marks the settings when both factors drawn are in the model.
 */
export function bestContourCaption(pair, { shaded = false, held = [], factors = [], starred = false } = {}) {
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = (h) => (typeof h.value === "string" ? `${h.name} at ${h.value}` : `${h.name} at ${withUnit(h.value, units[h.name])}`);
  const others = held.length ? `, with ${listed(held.map(at))}, ${held.length === 1 ? "its best setting" : "their best settings"}` : "";
  const grey = shaded ? "; grey: settings outside your limits" : "";
  const star = starred ? "; the star marks the best settings" : "";
  return `Drawn over ${pair.join(" and ")}, the two factors with the largest main effects${others}${grey}${star}; the dots are the runs.`;
}

// ---------------------------------------------------------------- the run sheet's confirmation runs
/** The places on a run sheet of its confirmation runs, in order. */
export function confirmationRows(design) {
  const at = design.columns.indexOf(CONFIRMATION);
  if (at < 0) return [];
  return design.rows.flatMap((row, i) => (String(row[at] ?? "").trim().toLowerCase() === CONFIRMS ? [i] : []));
}

/** The run sheet without its confirmation runs: what the model is fitted to, and what the picture of the limits draws. */
export function designRows(design) {
  const confirming = new Set(confirmationRows(design));
  const at = design.columns.indexOf(CONFIRMATION);
  const columns = at < 0 ? design.columns : design.columns.filter((_, j) => j !== at);
  const keep = (row) => (at < 0 ? row : row.filter((_, j) => j !== at));
  return { columns, rows: design.rows.filter((_, i) => !confirming.has(i)).map(keep), kept: design.rows.map((_, i) => !confirming.has(i)) };
}

/**
 * Add ``count`` confirmation runs at the settings ``at`` (each factor's value, or a categorical factor's level) to
 * the end of a run sheet. They take the next places in the run order, no place in the standard order and no block,
 * and the run sheet gains its confirmation column the first time. Returns the new columns and rows.
 */
export function withConfirmationRuns(design, at, count = CONFIRMATION_RUNS) {
  const has = design.columns.includes(CONFIRMATION);
  const columns = has ? [...design.columns] : [...design.columns, CONFIRMATION];
  const rows = design.rows.map((row) => (has ? [...row] : [...row, null]));
  const order = columns.indexOf("Run order");
  const last = Math.max(0, ...rows.map((row) => Number(row[order]) || 0));
  for (let i = 1; i <= count; i++) {
    rows.push(columns.map((c) => (c === "Run order" ? last + i : c === CONFIRMATION ? CONFIRMS : c in at ? at[c] : null)));
  }
  return { columns, rows };
}

/** Take a run sheet's confirmation runs away, with their results and sources, and its confirmation column with them. */
export function withoutConfirmationRuns(design, responses, sources) {
  const { columns, rows, kept } = designRows(design);
  return { columns, rows, responses: responses.filter((_, i) => kept[i]), sources: sources.filter((_, i) => kept[i]) };
}

/** Whether a run sheet holds confirmation runs at these settings already, the settings ``at`` as the page adds them. */
export function confirmedAt(design, at) {
  return confirmationRows(design).some((i) => Object.entries(at).every(([name, value]) => design.rows[i][design.columns.indexOf(name)] === value));
}
