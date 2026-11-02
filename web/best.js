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

/**
 * An interval as the page writes it: "61.2 to 68.4%", or with ``join`` "and", for "between ... and ...". A transformed
 * response's interval, put back in its units, can have no bound on one side (bootstrap._put_back gives null there):
 * "61.2% or more", "up to 68.4%".
 */
export function rangeWords([low, high], units = "", join = "to") {
  if (high === null || high === undefined) return `${withUnit(low, units)} or more`;
  if (low === null || low === undefined) return `up to ${withUnit(high, units)}`;
  return `${num(low)} ${join} ${withUnit(high, units)}`;
}

/**
 * A number on a transform's scale put back in the response's units, as bootstrap._put_back puts it: a prediction
 * becomes the response's median. ``transform`` is a key of bootstrap.TRANSFORMS (log, sqrt or inverse, which fits
 * -1/y), or none. A square root below 0 is a result of 0; -1/y at 0 or above, like a log too large to put back, is a
 * result without bound: null. Null stays null, and an interval is put back end by end.
 */
export function putBack(value, transform) {
  if (!transform || value === null || value === undefined) return value;
  if (Array.isArray(value)) return value.map((v) => putBack(v, transform));
  if (transform === "sqrt") return Math.max(value, 0) ** 2;
  if (transform === "inverse") return value < 0 ? -1 / value : null;
  const out = Math.exp(value);
  return Number.isFinite(out) ? out : null;
}

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
  const range = (pair) => rangeWords(pair, units);
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
  return `A next study could move the ranges along the ${climb(best)}: ${steps.map((s) => `${s.name} ${way(s)} ${s.size}`).join(", ")}.`;
}

/** The way the goal improves fastest, as a sentence names it: the steepest descent to minimise or to come down to a target, else the steepest ascent. */
export const climb = (best) =>
  best.goal === "minimize" || (best.goal === "target" && best.predicted > best.target) ? "steepest descent" : "steepest ascent";

/**
 * Whether the best settings point to a next study with moved ranges: the goal may be better past an end of a factor's
 * range that the model holds, no limit on the settings stands in the way, and there is a way to move. A target the
 * settings reach is met inside the ranges, so it points nowhere.
 */
export function movesOn(best) {
  if (best.goal === "target" && best.reached) return false;
  if ((best.limits ?? []).some((l) => l.beyond)) return false;
  return (best.settings ?? []).some((s) => s.in_model && s.beyond) && Object.keys(best.ascent ?? {}).length > 0;
}

/**
 * The factors of the next study the best settings point to: each factor the steepest ascent (or descent) moves has its
 * range shifted by its share of half the range, to two significant figures, as ascentSentence says it; the others, and
 * every categorical factor, keep theirs. ``factors`` are the spec's.
 */
export function nextRanges(best, factors) {
  const shares = best.ascent ?? {};
  const clean = (x) => Number(x.toPrecision(12)); // 1 - 0.6 is 0.4, not 0.40000000000000002
  return factors.map((f) => {
    if (!(f.name in shares) || typeof f.low !== "number") return { ...f };
    const step = Number(((shares[f.name] * (f.high - f.low)) / 2).toPrecision(2));
    return { ...f, low: clean(f.low + step), high: clean(f.high + step) };
  });
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
    if (movesOn(best)) said.push(ascentSentence(best, factors));
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

/** What the panel says for a goal of learning, or a target not yet typed; of a mixture's best blend with ``blend``. */
export function noGoalSentence(best, { label }, { blend = false } = {}) {
  if (best.goal === "target") {
    return `Type the target for ${label} in the brief, and the page finds the ${blend ? "blend that reaches" : "settings that reach"} it.`;
  }
  return `Set a goal for ${label} in the brief, to maximise, minimise or hit a target, and the page finds the best ${blend ? "blend" : "settings"} here.`;
}

/** What the panel says when no term of the model changes the response; of a mixture's blends with ``blend``. */
export const flatSentence = ({ label }, { blend = false } = {}) =>
  blend
    ? `The model predicts the same ${label} for every blend, so no blend is better than another.`
    : `No term of the model changes ${label}, so no setting is better than another.`;

/**
 * What the page says of one group of confirmation runs: where they are, and once their results are in, whether
 * their average falls inside the prediction interval for an average of that many runs, and what that means.
 * ``where`` names the place the runs test: "these settings", or "this blend" for a mixture's.
 */
export function confirmationSentences(group, factors, { label, units }, where = "these settings") {
  const unitsOf = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = listed(
    factors.map((f) => (typeof group.at[f.name] === "string" ? `${f.name} at ${group.at[f.name]}` : `${f.name} = ${withUnit(group.at[f.name], unitsOf[f.name])}`)),
  );
  const runs = (n) => `${n} run${n === 1 ? "" : "s"}`;
  const head = `${group.runs} confirmation ${group.runs === 1 ? "run" : "runs"} at ${at}.`;
  const range = group.interval ? rangeWords(group.interval, units) : null;
  if (!group.done) {
    const bounded = group.interval && group.interval.every((end) => end !== null);
    const between = group.interval ? (bounded ? `between ${rangeWords(group.interval, units, "and")}` : rangeWords(group.interval, units)) : null;
    const expect = between ? ` When their average is ${between}, the model holds there.` : "";
    return [`${head} Enter their results in Run.${expect}`];
  }
  const some = group.done < group.runs ? ` ${group.done} of ${group.runs} results are in.` : "";
  if (!range) {
    return [`${head}${some} With no runs left to estimate the noise, the page cannot judge their average.`];
  }
  const average = `The average of ${group.done === 1 ? "the one result" : `their ${runs(group.done)}`}, ${withUnit(group.mean, units)},`;
  const interval = `the 95% prediction interval for the average of ${runs(group.done)}, ${range}`;
  return group.inside
    ? [`${head}${some}`, `${average} is inside ${interval}: the model holds at ${where}.`]
    : [
        `${head}${some}`,
        `${average} is outside ${interval}: the process differs from the model at ${where}, so the model misses something there. Look for a term it lacks, or a change in the process.`,
      ];
}

// ---------------------------------------------------------------- a mixture's best blend
/**
 * What the panel says first of a mixture's best blend: the prediction it gives, and that it is the goal's best. The
 * proportions follow in a table. ``words`` is the response's {label, units}.
 */
export function blendLead(best, { label, units }) {
  const predicted = withUnit(best.predicted, units);
  if (best.goal !== "target") return `This blend gives the ${SUPERLATIVE[best.goal]} ${label} the model predicts: ${predicted}.`;
  if (best.reached) return `This blend gives ${label} on its target, ${withUnit(best.target, units)}, as the model predicts.`;
  return `No blend within your bounds reaches the target of ${withUnit(best.target, units)}. This one comes closest: ${predicted}.`;
}

/**
 * Where a component's proportion lies, as the table of the best blend says it: at its lower or upper bound, and
 * whether the goal may be better past a bound you set, or between its bounds.
 */
export function blendWhere(setting, best = null, { label = "" } = {}) {
  if (!setting.end) return "between its bounds";
  const [which, bound, past] = setting.end === "high" ? ["upper", setting.high, "above"] : ["lower", setting.low, "below"];
  const at = `at its ${which} bound, ${num(bound)}`;
  return setting.beyond ? `${at}; ${better(best, label)} may lie ${past}` : at;
}

/**
 * What the table of the best blend cannot say, a sentence each: a target that other blends reach too, or a blend with
 * every component between its bounds, which is then the top (or the bottom) of the fitted surface.
 */
export function blendSentences(best) {
  if (best.goal === "target") return best.reached ? ["Other blends reach the target too."] : [];
  if ((best.settings ?? []).some((s) => s.end)) return [];
  return [`Every component is between its bounds: this blend is the ${best.goal === "maximize" ? "top" : "bottom"} of the fitted surface.`];
}

/** What the panel says when the best blend leans on the model beyond the runs; else nothing. */
export const blendExtrapolation = (best) =>
  best.extrapolating ? "This blend lies beyond the runs, so the prediction leans on the model there: confirm it with a run." : "";

/** The caution when no test of a mixture's model finds anything clearly larger than the noise. */
export const BLEND_UNCLEAR =
  "No test of the model finds the blends differ by more than the noise (every p-value is 0.05 or above), so this blend may follow the noise.";

/** What the chart of how tightly to weigh out each component moves: each along its Cox direction. */
export const BLEND_SENSITIVITY =
  "Each component is moved with the others keeping their proportions to each other, so the blend still sums to 1.";

/**
 * How the confirmation runs weigh out the best blend: each proportion rounded, and the component that takes what the
 * rounding leaves, so the blend still sums to 1 (bootstrap._weighed).
 */
export function weighingSentence(best) {
  const { decimals = null, adjusted = null } = best.rounding ?? {};
  if (decimals === null) return "The confirmation runs use the blend as the search found it.";
  const each = `The confirmation runs weigh out each proportion to ${(10 ** -decimals).toFixed(decimals)}`;
  return adjusted ? `${each}, and ${adjusted} takes what the rounding leaves, so the blend still sums to 1.` : `${each}; the blend still sums to 1.`;
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

/**
 * Take a run sheet's confirmation runs away, with their results and sources, and its confirmation column with them.
 * ``results`` and ``marks`` hold each response's results and their sources, by its column.
 */
export function withoutConfirmationRuns(design, results, marks) {
  const { columns, rows, kept } = designRows(design);
  const keep = (byName) => Object.fromEntries(Object.entries(byName).map(([name, values]) => [name, values.filter((_, i) => kept[i])]));
  return { columns, rows, results: keep(results), marks: keep(marks) };
}

/** Whether a run sheet holds confirmation runs at these settings already, the settings ``at`` as the page adds them. */
export function confirmedAt(design, at) {
  return confirmationRows(design).some((i) => Object.entries(at).every(([name, value]) => design.rows[i][design.columns.indexOf(name)] === value));
}

// ---------------------------------------------------------------- the settings that balance the responses
/** How strongly the good end of a response's acceptable range counts, as the balance's table words it; mirrors bootstrap.SHAPES. */
export const GOOD_END = { less: "Less", even: "Evenly", more: "More", much_more: "Much more" };

/** The shape a response's desirability has unless the reader chooses another; mirrors bootstrap.SHAPE. */
export const SHAPE = "even";

/** The importances a response can have, and the one it has unless the reader sets it; mirror bootstrap.IMPORTANCES and IMPORTANCE. */
export const IMPORTANCES = [1, 2, 3, 4, 5];
export const IMPORTANCE = 3;

/** A desirability as the page writes it: two decimals, enough to compare one response's with another's. */
export const dWords = (d) => Number(d).toFixed(2);

/** A response's goal as the balance's table says it: "Maximise", "Minimise", or "Target 63%". */
export function goalWords(r) {
  if (r.goal === "maximize") return "Maximise";
  if (r.goal === "minimize") return "Minimise";
  return `Target ${withUnit(r.target, r.units)}`;
}

/**
 * What a response's acceptable range still needs, as the reader types it, in bootstrap._balance_goal's words; "" when
 * it is ready. ``low`` and ``high`` are the numbers typed, null for a box left blank (its default holds).
 */
export function goalProblem(r, low, high) {
  const from = low ?? r.default_low;
  const to = high ?? r.default_high;
  if (!Number.isFinite(from) || !Number.isFinite(to) || !(from < to)) {
    return `The acceptable range of ${r.label} must run from a lower value to a higher one.`;
  }
  if (r.goal === "target" && !(from < r.target && r.target < to)) {
    return `The target of ${r.label}, ${num(r.target)}, must lie inside its acceptable range.`;
  }
  return "";
}

/**
 * What the balance panel says first: how good the balance is, as the overall desirability D. With D of 0, no setting
 * makes every response acceptable, and the settings shown are those that come closest. Of a mixture's blend with
 * ``blend``.
 */
export function balanceLead(b, { blend = false } = {}) {
  if (b.composite <= 0) {
    return blend
      ? "No blend within your bounds makes every response acceptable. This one comes closest: widen an acceptable range to find a balance."
      : "No setting in your ranges makes every response acceptable. These settings come closest: widen an acceptable range to find a balance.";
  }
  const these = blend ? "This blend" : "These settings";
  if (b.composite >= 1) return `${these} ${blend ? "makes" : "make"} every response as good as it gets: an overall desirability of 1.`;
  return `${these} ${blend ? "balances" : "balance"} the responses best: an overall desirability of ${dWords(b.composite)}.`;
}

/**
 * Which response holds the overall desirability down most, and by how much: the one whose desirability, counted by its
 * importance, is lowest. Its importance is said when the importances differ. Nothing when every response is as good
 * as it gets; with D of 0, the responses not acceptable at these settings.
 */
export function limitingSentence(b, { blend = false } = {}) {
  if (b.composite >= 1) return "";
  if (b.composite <= 0) {
    const out = b.responses.filter((r) => r.d <= 0).map((r) => r.label);
    return `At ${blend ? "this blend" : "these settings"} ${listed(out)} ${out.length === 1 ? "is" : "are"} not acceptable: a desirability of 0.`;
  }
  const r = b.responses.find((one) => one.name === b.limiting);
  const differ = new Set(b.responses.map((one) => one.importance)).size > 1;
  return `${r.label} holds the overall desirability down most: its desirability is ${dWords(r.d)}${differ ? `, at importance ${r.importance}` : ""}.`;
}

/**
 * Each response at the balanced settings, a row each for a table: its name, its prediction, its 95% prediction
 * interval for one new run, and its desirability.
 */
export function balanceRows(b) {
  return b.responses.map((r) => [
    r.label,
    withUnit(r.predicted, r.units),
    r.prediction ? rangeWords(r.prediction, r.units) : "",
    dWords(r.d),
  ]);
}

/** The leverage at the balanced settings, for the model that leans furthest on the runs, as a sentence. */
export function balanceLeverage(b) {
  const r = b.responses.find((one) => one.name === b.leverage_of);
  const where = b.extrapolating ? "above" : "within";
  return `The leverage here is ${lev(b.leverage)} for ${r.label}'s model, ${where} its runs': ${lev(b.max_leverage)} at most.`;
}

/**
 * What the table of settings cannot say of the balanced settings: the limits they are on, and the factors no response's
 * model holds, each held at the center or at its first level.
 */
export function balanceWhere(b) {
  const said = (b.limits ?? []).map((limit) => `These settings are on your limit ${limit.index + 1}, ${limit.expression}.`);
  for (const s of (b.settings ?? []).filter((one) => !one.in_model)) {
    said.push(
      s.kind === "categorical"
        ? `${s.name} is in no response's model, so either level serves; these settings use ${s.value}.`
        : `${s.name} is in no response's model, so it changes nothing; these settings hold it at the center.`,
    );
  }
  return said;
}

/**
 * The caption of the overlay of the acceptable ranges: the two factors drawn and why, where the others are held,
 * and what the shading, the lines, the grey and the star mean. ``factors`` are the spec's, for their units.
 */
export function overlayCaption(overlay, factors = [], responses = []) {
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = (h) => (typeof h.value === "string" ? `${h.name} at ${h.value}` : `${h.name} at ${withUnit(h.value, units[h.name])}`);
  const held = overlay.held ?? [];
  const others = held.length ? `, with ${listed(held.map(at))}, ${held.length === 1 ? "its balanced setting" : "their balanced settings"}` : "";
  const grey = overlay.runnable ? "; grey: settings outside your limits" : "";
  const { everywhere, nowhere } = missedBoundaries(overlay, responses);
  const missed = (labels, said) => (labels.length ? ` No line for ${listed(labels)}: ${labels.length === 1 ? "it is" : "they are"} ${said}.` : "");
  const narrow = everywhere.length ? " Narrow a range in the table to draw its boundary." : "";
  return (
    `Drawn over ${overlay.x} and ${overlay.y}, the two factors that move the responses' desirabilities most${others}. ` +
    `Shaded where every response is acceptable; each line is a response's acceptable boundary${grey}; the star marks the balanced settings.` +
    missed(everywhere, "acceptable at every setting drawn") +
    narrow +
    missed(nowhere, "not acceptable at any setting drawn")
  );
}

/**
 * The caption of a mixture's overlay of the acceptable ranges over its triangle: what the shading, the lines and the
 * star mean, and why a response has no line. ``responses`` are the balance's, for each one's label, goal and range.
 */
export function blendOverlayCaption(overlay, responses = []) {
  const { everywhere, nowhere } = missedBoundaries(overlay, responses);
  const missed = (labels, said) => (labels.length ? ` No line for ${listed(labels)}: ${labels.length === 1 ? "it is" : "they are"} ${said}.` : "");
  const narrow = everywhere.length ? " Narrow a range in the table to draw its boundary." : "";
  return (
    "Each corner is a pure component and each point a blend the bounds allow. Shaded where every response is " +
    "acceptable; each line is a response's acceptable boundary; the star marks the balanced blend." +
    missed(everywhere, "acceptable at every blend drawn") +
    narrow +
    missed(nowhere, "not acceptable at any blend drawn")
  );
}

/** Why a mixture of four components or more has no overlay of the acceptable ranges. */
export const NO_BLEND_OVERLAY =
  "With four components or more the blends fill a solid, which no picture of a triangle holds, so no overlay is drawn.";

/**
 * The responses whose acceptable boundary misses the settings the overlay draws, by what that means: acceptable at
 * every setting drawn, or at none. ``responses`` are the balance's, for each one's label, goal and range.
 */
export function missedBoundaries(overlay, responses = []) {
  const byName = Object.fromEntries(responses.map((r) => [r.name, r]));
  const everywhere = [];
  const nowhere = [];
  for (const surface of overlay.responses ?? []) {
    const r = byName[surface.name];
    if (!r) continue;
    const values = surface.z.flat();
    const [lo, hi] = [Math.min(...values), Math.max(...values)];
    if (surface.levels.some((level) => level > lo && level < hi)) continue;
    const ok = (r.goal === "minimize" || lo >= r.low) && (r.goal === "maximize" || hi <= r.high);
    (ok ? everywhere : nowhere).push(r.label);
  }
  return { everywhere, nowhere };
}

/**
 * What the page says of one group of confirmation runs, response by response: where they are, and for each response
 * with a goal, once its results are in, whether their average falls inside the prediction interval for an average of
 * that many runs. ``responses`` are the balance's, for each one's words. Returns the sentences and the group's state:
 * ``inside`` when every response's average is, ``outside`` when one is not, else ``waiting``.
 */
export function balanceConfirmation(group, factors, responses) {
  const unitsOf = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const words = Object.fromEntries(responses.map((r) => [r.name, r]));
  const at = listed(
    factors.map((f) => (typeof group.at[f.name] === "string" ? `${f.name} at ${group.at[f.name]}` : `${f.name} = ${withUnit(group.at[f.name], unitsOf[f.name])}`)),
  );
  const said = [`${group.runs} confirmation ${group.runs === 1 ? "run" : "runs"} at ${at}.`];
  for (const v of group.responses) {
    const { label, units } = words[v.name];
    if (!v.done) {
      said.push(`${label}: no result yet; enter it in Run.`);
      continue;
    }
    if (!v.interval) {
      said.push(`${label}: with no runs left to estimate the noise, the page cannot judge its average.`);
      continue;
    }
    const average = `the average of ${v.done === 1 ? "the one result" : `${v.done} results`}, ${withUnit(v.mean, units)}`;
    const range = `${num(v.interval[0])} to ${withUnit(v.interval[1], units)}`;
    said.push(
      v.inside
        ? `${label}: ${average}, is inside the 95% prediction interval, ${range}: the model holds here.`
        : `${label}: ${average}, is outside the 95% prediction interval, ${range}: the process differs from the model here.`,
    );
  }
  const judged = group.responses.filter((v) => v.inside !== null);
  const state = judged.some((v) => !v.inside) ? "outside" : judged.length === group.responses.length ? "inside" : "waiting";
  return { said, state };
}

// ---------------------------------------------------------------- what if: each model at the reader's settings
/**
 * The columns the models are fitted on at the reader's settings: a numeric factor's setting coded from its range, -1 at
 * its low end and +1 at its high end, as bootstrap._coded_column codes it, and a categorical factor's level as the
 * columns Python wrote for it (bootstrap._coding). ``factors`` is bootstrap.predictors' ``factors``; ``settings``
 * holds each factor's number in its own units, or its level.
 */
export function columnsAt(factors, settings) {
  const columns = {};
  for (const f of factors) {
    if (f.kind === "numeric") columns[f.name] = (settings[f.name] - (f.high + f.low) / 2) / ((f.high - f.low) / 2);
    else Object.assign(columns, f.columns[f.levels.indexOf(settings[f.name])]);
  }
  return columns;
}

/**
 * A response's prediction at the columns of some settings (see columnsAt), from its fit as bootstrap.predictors sends
 * it, as bootstrap._FittedModel.at gives it: the prediction, its confidence interval for the mean and prediction
 * interval for one new run (null without residual degrees of freedom), and the leverage there, with whether it is
 * above the largest of the runs', as best_settings judges it. Each term is the product of the columns it names. A
 * response fitted on a transform's scale (its ``transform``) has its prediction and intervals put back in its units.
 */
export function predictAt(fit, columns) {
  const row = fit.terms.map((parts) => parts.reduce((product, name) => product * columns[name], 1));
  const predicted = row.reduce((sum, x, i) => sum + x * fit.coefficients[i], 0);
  const spread = row.map((_, i) => fit.inverse[i].reduce((sum, v, j) => sum + v * row[j], 0));
  const leverage = Math.max(0, row.reduce((sum, x, i) => sum + x * spread[i], 0)); // never below 0 but by round-off
  const extrapolating = leverage > fit.max_leverage * (1 + 1e-9) + 1e-12;
  const out = { predicted, confidence: null, prediction: null, leverage, max_leverage: fit.max_leverage, extrapolating };
  if (fit.mse === null || fit.t === null) return putBackAt(out, fit.transform);
  const mean = fit.t * Math.sqrt(fit.mse * leverage);
  const single = fit.t * Math.sqrt(fit.mse * (1 + leverage));
  const intervals = { confidence: [predicted - mean, predicted + mean], prediction: [predicted - single, predicted + single] };
  return putBackAt({ ...out, ...intervals }, fit.transform);
}

/** A prediction and its intervals on a transform's scale, put back in the response's units; the leverage as it is. */
function putBackAt(at, transform) {
  if (!transform) return at;
  return { ...at, ...Object.fromEntries(["predicted", "confidence", "prediction"].map((key) => [key, putBack(at[key], transform)])) };
}

/**
 * Where the what-if panel starts, and comes back to: the balanced settings while the responses are balanced, else the
 * best settings for the goal of the response Learn shows, else each numeric factor at the center of its range and each
 * categorical one at its first level. ``factors`` is bootstrap.predictors' ``factors``; ``balance`` and ``best`` are
 * what bootstrap.balance and best_settings answered, or null. Returns ``from``, which of the three, and the settings.
 */
export function whatIfStart(factors, balance, best) {
  const [from, found] = balance?.settings ? ["balanced", balance.settings] : best?.settings ? ["best", best.settings] : ["center", []];
  const given = Object.fromEntries(found.map((s) => [s.name, s.value]));
  const middle = (f) => (f.kind === "numeric" ? (f.low + f.high) / 2 : f.levels[0]);
  return { from, settings: Object.fromEntries(factors.map((f) => [f.name, given[f.name] ?? middle(f)])) };
}

/** The button that brings the what-if panel back to where it started, by whatIfStart's ``from``. */
export const WHAT_IF_BACK = {
  balanced: "Back to the balanced settings",
  best: "Back to the best settings",
  center: "Back to the center of the ranges",
};

/**
 * Whether settings the panel holds still fit the factors: every factor has one, a number inside its range or one of
 * its levels. Settings kept from before an analysis are kept after it when they do.
 */
export function settingsFit(factors, settings) {
  return Boolean(settings) && factors.every((f) => {
    const v = settings[f.name];
    return f.kind === "numeric" ? Number.isFinite(v) && v >= Math.min(f.low, f.high) && v <= Math.max(f.low, f.high) : f.levels.includes(v);
  });
}

/** The step of a numeric factor's slider: about a hundredth of its range, as 1, 2 or 5 times a power of ten. */
export function sliderStep(f) {
  const raw = Math.abs(f.high - f.low) / 100;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5].find((m) => m * power >= raw) ?? 10) * power;
  return Number(step.toPrecision(1)); // 0.3, not 0.30000000000000004
}

/**
 * Each response at the what-if settings, a row each for a table: its name, its prediction, its 95% confidence interval
 * for the mean, its 95% prediction interval for one new run, and its leverage beside the runs' largest. ``responses`` is
 * bootstrap.predictors' list, each with ``at``, what predictAt gave at the settings; one without results, or whose
 * model cannot be fitted, says so in its row.
 */
export function whatIfRows(responses) {
  const range = (pair, units) => (pair ? rangeWords(pair, units) : "none");
  return responses.map((r) => {
    if (r.waiting) return [r.label, "No results yet: enter them in Run.", "", "", ""];
    if (r.error) return [r.label, r.error, "", "", ""];
    const { at } = r;
    const leverage = `${lev(at.leverage)}, ${at.extrapolating ? "above" : "within"} ${lev(at.max_leverage)}`;
    return [r.label, withUnit(at.predicted, r.units), range(at.confidence, r.units), range(at.prediction, r.units), leverage];
  });
}

/**
 * What the panel says beside the table: the limits the settings break, numbered as the brief lists them; the models
 * the settings lean on beyond their runs; and the models with no runs left to estimate the noise. ``broken`` is what
 * bootstrap.outside_limits answered.
 */
export function whatIfSentences(responses, broken = []) {
  const said = broken.map((limit) => `These settings break your limit ${limit.index + 1}, ${limit.expression}.`);
  const beyond = responses.filter((r) => r.at?.extrapolating).map((r) => r.label);
  if (beyond.length === 1) {
    said.push(`These settings lie beyond the runs for ${beyond[0]}'s model: confirm its prediction with a run.`);
  } else if (beyond.length) {
    said.push(`These settings lie beyond the runs for the models of ${listed(beyond)}: confirm their predictions with a run.`);
  }
  const bare = responses.filter((r) => r.at && !r.at.confidence).map((r) => r.label);
  if (bare.length) {
    const whose = bare.length === 1 ? `${bare[0]}'s model, so it gives` : `the models of ${listed(bare)}, so they give`;
    said.push(`No runs are left to estimate the noise of ${whose} no interval.`);
  }
  return said;
}
