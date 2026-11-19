// The best settings for the response's goal: what Learn says of them, and the confirmation runs made
// at them. Python finds them (bootstrap.best_settings) and says where they lie as facts; this module
// turns the facts into the page's sentences and keeps the run sheet's confirmation runs. Nothing here
// touches the document, so tests/web/js runs it under Node.

/** The run sheet's column marking a confirmation run, and what it holds for one; mirrors bootstrap.CONFIRMATION and CONFIRMS. */
import { marked } from "./numbers.js";

export const CONFIRMATION = "Confirmation run";
export const CONFIRMS = "yes";
/** What that column holds for a run on the path of steepest ascent (or descent); mirrors bootstrap.ON_PATH. */
export const ON_PATH = "path";

/** How many confirmation runs the page adds at the best settings; mirrors bootstrap.CONFIRMATION_RUNS. */
export const CONFIRMATION_RUNS = 3;

/** How far each factor moves for the chart of how tightly to hold it: this share of its range; mirrors bootstrap.NUDGE. */
export const NUDGE = 0.05;

/** A number as a sentence gives it: four significant figures, with no trailing zeros. */
export const num = (x) => marked(String(Number(Number(x).toPrecision(4))));

/** The same number for a box of type number, which takes a point whatever the reader's mark. */
export const numPlain = (x) => String(Number(Number(x).toPrecision(4)));

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

/**
 * The answer Learn opens with, from the best settings: where to run (``run``), and what the model expects there
 * with its interval for the mean (``expect``). Each number is written as the panel of the best settings writes
 * it. Null with no settings to state: a goal of learning, a target not yet typed, or a model no term of which
 * changes the response. ``words`` is the response's {label, units}; ``blend`` words a mixture's best blend.
 */
export function answerWords(best, { label, units }, { blend = false } = {}) {
  if (!best?.settings) return null;
  const at = listed(best.settings.map((setting) => settingWords(setting, setting.units)));
  const level = `${Math.round(100 * (best.level ?? 0.95))}%`;
  const mean = best.confidence ? ` (${level} confidence interval for the mean: ${rangeWords(best.confidence, units)})` : "";
  const predicted = `${withUnit(best.predicted, units)}${mean}`;
  const expect =
    best.goal !== "target"
      ? `The model predicts the ${SUPERLATIVE[best.goal]} ${label} there: ${predicted}.`
      : best.reached
        ? `The model predicts ${label} on its target there: ${withUnit(best.target, units)}${mean}.`
        : `No ${blend ? "blend within your bounds" : "setting in your ranges"} reaches the target of ` +
          `${withUnit(best.target, units)}: ${blend ? "this one comes" : "these come"} closest, at ${predicted}.`;
  return { run: `${blend ? "Blend" : "Run at"} ${at}.`, expect };
}

/** A leverage as the page writes it: two significant figures, enough to compare with the runs'. */
const lev = (h) => marked(String(Number(Number(h).toPrecision(2))));

/**
 * The checks under Learn's answer: whether to rely on it, from what the analysis (``analysis``, as
 * bootstrap.analyze_runs gives it) and the best settings (``best``, or null) already say. Nothing is judged here:
 * Python marks the runs to look at again, a lack of fit, a model with no clear term and settings beyond the runs.
 *
 * Each check is {key, state, title, says, numbers}. ``state`` is "ok", "look" (read this before relying on the
 * answer) or "unknown" (these runs cannot tell); ``title`` is the verdict in a few words, ``says`` a sentence, and
 * ``numbers`` the statistic behind it, or "". The fit, the runs one by one, and the repeated runs are always
 * there; the reach of the runs only with settings to check.
 *
 * This module stands alone, so the page's other words come in as ``tell``: ``said(kind)``, the sentence Learn's
 * findings give for a kind of finding, or ""; ``number(x)`` and ``p(x)``, a statistic and a p-value as the tables
 * write them; and ``share(x)``, a share as a whole percentage (logic.js findingWords, fmt, pval and shareWords).
 */
export function learnChecks(analysis, best, tell, { blend = false } = {}) {
  const found = (kind) => (analysis.findings ?? []).filter((finding) => finding.kind === kind);
  const { said, number: fmt, p: pval, share: wholeShare } = tell;
  const known = (x) => x !== null && x !== undefined;
  const s = analysis.summary;
  const fits = [
    ["R²", s.r_squared],
    ["Adjusted R²", s.r_squared_adj],
    ["Predicted R²", s.r_squared_pred],
  ];
  const statistics = fits.filter(([, x]) => known(x)).map(([name, x]) => `${name} = ${fmt(x)}`).join(", ");
  const checks = [];

  if (found("saturated").length) {
    checks.push({ key: "fit", state: "unknown", title: "No runs left to judge the fit", says: said("saturated"), numbers: "Residual degrees of freedom = 0" });
  } else if (found("none").length || best?.unclear) {
    const says = best?.unclear ? (blend ? BLEND_UNCLEAR : UNCLEAR) : said("none");
    checks.push({ key: "fit", state: "look", title: "No clear effect", says, numbers: statistics });
  } else {
    const [fit] = found("fit");
    const title = fit ? `The model explains ${wholeShare(fit.explained)}` : "The model's fit";
    checks.push({ key: "fit", state: "ok", title, says: said("fit"), numbers: statistics });
  }

  const [unusual] = found("unusual");
  if (!analysis.diagnostics?.length) {
    const says = "The model uses every run, so no run can be compared with what the model expects of it.";
    checks.push({ key: "runs", state: "unknown", title: "No run can be judged alone", says, numbers: "" });
  } else if (unusual) {
    const title = unusual.runs.length === 1 ? `Run ${unusual.runs[0]} stands apart` : `${unusual.runs.length} runs stand apart`;
    checks.push({ key: "runs", state: "look", title, says: said("unusual"), numbers: "" });
  } else {
    const says = "No run's result is far from what the model expects of it.";
    checks.push({ key: "runs", state: "ok", title: "No run stands apart", says, numbers: `${analysis.diagnostics.length} runs compared` });
  }

  const lof = analysis.lack_of_fit;
  if (!lof || !known(lof.p_value)) {
    const says = "No run was repeated, so these runs cannot tell a missing term from noise.";
    checks.push({ key: "repeats", state: "unknown", title: "No repeated runs", says, numbers: "" });
  } else {
    const freedom = `${lof.df_pure_error} degree${lof.df_pure_error === 1 ? "" : "s"} of freedom`;
    const numbers = `Lack of fit: F = ${fmt(lof.f_statistic)}, p = ${pval(lof.p_value)}; pure error from the repeated runs, ${freedom}`;
    const says = lof.significant ? said("lack_of_fit") : "The repeated runs agree with the model about as closely as with each other.";
    const [state, title] = lof.significant ? ["look", "A term may be missing"] : ["ok", "The repeats agree with the model"];
    checks.push({ key: "repeats", state, title, says, numbers });
  }

  if (best?.settings) {
    const numbers = `Leverage ${lev(best.leverage)}; the runs' is ${lev(best.max_leverage)} at most`;
    const beyond = (blend ? blendExtrapolation : extrapolationSentence)(best);
    const within = `${blend ? "This blend lies" : "These settings lie"} where the runs can speak for the prediction.`;
    const [state, title] = beyond ? ["look", "Beyond the runs"] : ["ok", "Within the runs' reach"];
    checks.push({ key: "reach", state, title, says: beyond || within, numbers });
  }
  return checks;
}

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
  const each = `The confirmation runs weigh out each proportion to ${marked((10 ** -decimals).toFixed(decimals))}`;
  return adjusted ? `${each}, and ${adjusted} takes what the rounding leaves, so the blend still sums to 1.` : `${each}; the blend still sums to 1.`;
}

/**
 * The caption of the response surface drawn through the best settings: the factors not drawn are held at their
 * best settings, and a star marks the settings when both factors drawn are in the model.
 */
export function bestContourCaption(pair, { shaded = false, held = [], factors = [], starred = false, chosen = false } = {}) {
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = (h) => (typeof h.value === "string" ? `${h.name} at ${h.value}` : `${h.name} at ${withUnit(h.value, units[h.name])}`);
  const others = held.length ? `, with ${listed(held.map(at))}, ${held.length === 1 ? "its best setting" : "their best settings"}` : "";
  const grey = shaded ? "; grey: settings outside your limits" : "";
  const star = starred ? "; the star marks the best settings" : "";
  const which = chosen ? "the two factors you chose" : "the two factors with the largest main effects"; // logic.pairWords
  return `Drawn over ${pair.join(" and ")}, ${which}${others}${grey}${star}; the dots are the runs.`;
}

// ---------------------------------------------------------------- the run sheet's confirmation runs
/** The places on a run sheet of the runs its confirmation column marks with ``mark``, in order. */
function markedRows(design, mark) {
  const at = design.columns.indexOf(CONFIRMATION);
  if (at < 0) return [];
  return design.rows.flatMap((row, i) => (String(row[at] ?? "").trim().toLowerCase() === mark ? [i] : []));
}

/** What that column holds for a run of the design that failed, and has no result; mirrors bootstrap.FAILED. */
export const FAILED = "failed";

/** The places on a run sheet of the runs that failed, in order: runs of the design with no result to come. */
export const failedRows = (design) => markedRows(design, FAILED);

/**
 * A run sheet with the run at ``place`` marked as failed, or (``failed`` false) as a run to make again. The sheet
 * gains its confirmation column with the first mark and loses it with the last. Returns the new columns and rows.
 */
export function withFailed(design, place, failed) {
  const has = design.columns.includes(CONFIRMATION);
  let columns = has ? [...design.columns] : [...design.columns, CONFIRMATION];
  const at = columns.indexOf(CONFIRMATION);
  let rows = design.rows.map((row, i) => {
    const copy = has ? [...row] : [...row, null];
    if (i === place) copy[at] = failed ? FAILED : null;
    return copy;
  });
  if (rows.every((row) => !String(row[at] ?? "").trim())) {
    columns = columns.filter((_, j) => j !== at);
    rows = rows.map((row) => row.filter((_, j) => j !== at));
  }
  return { columns, rows };
}

/** The places on a run sheet of its confirmation runs, in order. */
export const confirmationRows = (design) => markedRows(design, CONFIRMS);

/** The places on a run sheet of its runs along the path of steepest ascent, in the path's order. */
export const pathRows = (design) => markedRows(design, ON_PATH);

/** The places on a run sheet of the runs that are no runs of its design: its confirmation runs and its runs on the path. */
export const apartRows = (design) => [...confirmationRows(design), ...pathRows(design)].sort((a, b) => a - b);

/**
 * The run sheet without its confirmation runs and its runs on the path: what the model is fitted to, and what the
 * picture of the limits draws.
 */
export function designRows(design) {
  const apart = new Set(apartRows(design));
  const at = design.columns.indexOf(CONFIRMATION);
  const columns = at < 0 ? design.columns : design.columns.filter((_, j) => j !== at);
  const keep = (row) => (at < 0 ? row : row.filter((_, j) => j !== at));
  return { columns, rows: design.rows.filter((_, i) => !apart.has(i)).map(keep), kept: design.rows.map((_, i) => !apart.has(i)) };
}

/**
 * Add ``count`` confirmation runs at the settings ``at`` (each factor's value, or a categorical factor's level) to
 * the end of a run sheet. They take the next places in the run order, no place in the standard order and no block,
 * and the run sheet gains its confirmation column the first time. Returns the new columns and rows.
 */
export const withConfirmationRuns = (design, at, count = CONFIRMATION_RUNS) => withMarkedRuns(design, Array(count).fill(at), CONFIRMS);

/**
 * Add the runs along a path of steepest ascent to the end of a run sheet, in the path's order, which is the order
 * they are made in: ``path`` holds each run's settings. As confirmation runs, they are no runs of the design.
 */
export const withPathRuns = (design, path) => withMarkedRuns(design, path, ON_PATH);

/** Add runs at the settings given, one each, to the end of a run sheet, marked ``mark`` in its confirmation column. */
function withMarkedRuns(design, settings, mark) {
  const has = design.columns.includes(CONFIRMATION);
  const columns = has ? [...design.columns] : [...design.columns, CONFIRMATION];
  const rows = design.rows.map((row) => (has ? [...row] : [...row, null]));
  const order = columns.indexOf("Run order");
  const last = Math.max(0, ...rows.map((row) => Number(row[order]) || 0));
  settings.forEach((at, i) => {
    rows.push(columns.map((c) => (c === "Run order" ? last + i + 1 : c === CONFIRMATION ? mark : c in at ? at[c] : null)));
  });
  return { columns, rows };
}

/**
 * Take a run sheet's confirmation runs away, with their results and sources. ``results`` and ``marks`` hold each
 * response's results and their sources, by its column.
 */
export const withoutConfirmationRuns = (design, results, marks, curves = null) => withoutRuns(design, confirmationRows(design), results, marks, curves);

/** Take a run sheet's runs on the path away, with their results and sources. */
export const withoutPathRuns = (design, results, marks, curves = null) => withoutRuns(design, pathRows(design), results, marks, curves);

/** A run sheet less the runs at the places ``gone``; its confirmation column goes too once it marks no run. */
function withoutRuns(design, gone, results, marks, curves) {
  const out = new Set(gone);
  const kept = design.rows.map((_, i) => !out.has(i));
  const at = design.columns.indexOf(CONFIRMATION);
  let rows = design.rows.filter((_, i) => kept[i]);
  let columns = design.columns;
  if (at >= 0 && rows.every((row) => !String(row[at] ?? "").trim())) {
    columns = columns.filter((_, j) => j !== at);
    rows = rows.map((row) => row.filter((_, j) => j !== at));
  }
  const keep = (byName) => Object.fromEntries(Object.entries(byName).map(([name, values]) => [name, values.filter((_, i) => kept[i])]));
  return { columns, rows, results: keep(results), marks: keep(marks), curves: curves && curves.filter((_, i) => kept[i]) };
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
export const dWords = (d) => marked(Number(d).toFixed(2));

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
    `${TRIANGLE} Shaded where every response is ` +
    "acceptable; each line is a response's acceptable boundary; the star marks the balanced blend." +
    missed(everywhere, "acceptable at every blend drawn") +
    narrow +
    missed(nowhere, "not acceptable at any blend drawn")
  );
}

/** What a mixture's triangle is, as the caption of each chart drawn over it starts. */
const TRIANGLE = "Each corner is a pure component and each point a blend the bounds allow.";

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
  const { predicted, leverage } = modelAt(fit, columns);
  const extrapolating = leverage > fit.max_leverage * (1 + 1e-9) + 1e-12;
  const out = { predicted, confidence: null, prediction: null, leverage, max_leverage: fit.max_leverage, extrapolating };
  if (fit.mse === null || fit.t === null) return putBackAt(out, fit.transform);
  const mean = fit.t * Math.sqrt(fit.mse * leverage);
  const single = fit.t * Math.sqrt(fit.mse * (1 + leverage));
  const intervals = { confidence: [predicted - mean, predicted + mean], prediction: [predicted - single, predicted + single] };
  return putBackAt({ ...out, ...intervals }, fit.transform);
}

/** A fit's prediction at the columns of some settings, on the scale it is fitted on, and the leverage there. */
function modelAt(fit, columns) {
  const row = fit.terms.map((parts) => parts.reduce((product, name) => product * columns[name], 1));
  const predicted = row.reduce((sum, x, i) => sum + x * fit.coefficients[i], 0);
  const spread = row.map((_, i) => fit.inverse[i].reduce((sum, v, j) => sum + v * row[j], 0));
  const leverage = Math.max(0, row.reduce((sum, x, i) => sum + x * spread[i], 0)); // never below 0 but by round-off
  return { predicted, leverage };
}

/**
 * A prediction as the runs a reader might get at the columns of some settings (see columnsAt), least first: one for
 * each of the fit's ``run_quantiles``, the t values of equally likely runs (bootstrap.PREDICTED_RUNS of them), about
 * the prediction with the spread of one new run there. The least and the greatest are the two ends of predictAt's
 * prediction interval, so the two cannot disagree. In the response's units; null without residual degrees of
 * freedom, or when a transform cannot put one of them back.
 */
export function runsAt(fit, columns) {
  if (fit.mse === null || !fit.run_quantiles) return null;
  const { predicted, leverage } = modelAt(fit, columns);
  const single = Math.sqrt(fit.mse * (1 + leverage));
  const runs = putBack(fit.run_quantiles.map((q) => predicted + q * single), fit.transform);
  return runs.every(Number.isFinite) ? runs : null; // each transform rises with the scale it is fitted on, so the order holds
}

/** Whether each of some runs is inside a window, ``[least, most]`` with null for an end left open; with no window, every one. */
export const insideWindow = (runs, [least, most] = [null, null]) => runs.map((run) => (least === null || run >= least) && (most === null || run <= most));

/**
 * Where to draw runs as dots that pile up where the runs are close: each run's column of ``columns`` across
 * ``[low, high]``, and its place in that column's pile, 0 at the foot.
 */
export function dotPiles(runs, [low, high], columns) {
  const piled = new Map();
  return runs.map((run) => {
    const column = Math.min(columns - 1, Math.max(0, Math.floor(((run - low) / (high - low || 1)) * columns)));
    piled.set(column, (piled.get(column) ?? -1) + 1);
    return { column, level: piled.get(column) };
  });
}

/**
 * The runs in words: how far they reach, and how many are inside the reader's window when one is typed. ``words`` is
 * the response's ``label`` and ``units``; ``inside`` is insideWindow's answer.
 */
export function runsWords(runs, inside, { label, units }, windowed = false) {
  const reach = `${label} as ${runs.length} runs you might get: ${rangeWords([runs[0], runs.at(-1)], units)}.`;
  return windowed ? `${reach} ${inside.filter(Boolean).length} of them are inside your window.` : reach;
}

/** A prediction and its intervals on a transform's scale, put back in the response's units; the leverage as it is. */
function putBackAt(at, transform) {
  if (!transform) return at;
  return { ...at, ...Object.fromEntries(["predicted", "confidence", "prediction"].map((key) => [key, putBack(at[key], transform)])) };
}

/**
 * A curve predicted at the columns of some settings (see columnsAt), from its fit as bootstrap.functional sends it,
 * with the 95% bands the library's FunctionalFit.predict gives there. The curve is ``mean`` plus the model row times
 * ``map``, at each of ``points``; each bootstrap replicate's is the same from its own. The band at each point is the
 * curve plus or minus ``t`` times the replicates' standard deviation there. The band over the whole curve takes the
 * ``level`` quantile of each replicate's largest departure, in those standard deviations, times ``widening``.
 */
export function curveAt(fit, columns) {
  const row = fit.terms.map((parts) => parts.reduce((product, name) => product * columns[name], 1));
  const along = (mean, map) => mean.map((m, g) => row.reduce((sum, x, j) => sum + x * map[j][g], m));
  const curve = along(fit.mean, fit.map);
  const replicates = fit.replicates.map((r) => along(r.mean, r.map));
  const error = curve.map((_, g) => spread(replicates.map((r) => r[g])));
  const largest = replicates.map((r) => Math.max(...r.map((v, g) => (error[g] > 0 ? Math.abs(v - curve[g]) / error[g] : 0))));
  const critical = fit.widening * quantile(largest, fit.level);
  const band = (k) => [curve.map((c, g) => c - k * error[g]), curve.map((c, g) => c + k * error[g])];
  return { curve, error, pointwise: band(fit.t), simultaneous: band(critical), critical };
}

/** The standard deviation of some numbers, about their mean, over one fewer than there are. */
function spread(values) {
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (values.length - 1));
}

/** The ``q`` quantile of some numbers, between the two nearest by a straight line, as numpy's quantile gives it. */
function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (sorted.length - 1) * q;
  const low = Math.floor(at);
  return low + 1 < sorted.length ? sorted[low] + (at - low) * (sorted[low + 1] - sorted[low]) : sorted[low];
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
export const settingsFit = (factors, settings) => Boolean(settings) && scenarioProblem(factors, settings) === null;

/**
 * Why the models do not predict at settings kept from the panel, in a sentence, or null when they do: a factor the
 * settings do not set, a number outside its factor's range, a level the factor does not have, or a limit on the
 * settings they break (``broken``, as bootstrap.outside_limits answered). A study or a workbook edited since the settings
 * were kept can hold the first three. ``factors`` is bootstrap.predictors' ``factors``; ``units`` each factor's units.
 */
/** The comparison that holds where a limit for one level's own does not: "not (... and T > 170)" is T <= 170. */
const OTHERWISE = { ">": "<=", ">=": "<", "<": ">=", "<=": ">" };

/**
 * A limit as a sentence shows it. A limit for one level of a categorical factor, which Python reads as
 * ``not (RM == 'A' and conc > 1.5)``, reads "conc <= 1.5 when RM is A"; two levels never run together read
 * "never RM is A and Line is 2 together". Any other limit is shown as it is written.
 */
export function limitWords(expression) {
  // shortcut: a label that holds " and " is not split right, and the limit is then shown as written.
  const inner = /^not \((.+)\)$/.exec(expression);
  if (!inner) return expression;
  const when = [];
  const holds = [];
  for (const part of inner[1].split(" and ")) {
    const level = /^(\S+) == '(.*)'$/.exec(part);
    const compared = /^(.+) (<=|>=|<|>) (.+)$/.exec(part);
    if (level) when.push(`${level[1]} is ${level[2].replaceAll("\\'", "'")}`);
    else if (compared) holds.push(`${compared[1]} ${OTHERWISE[compared[2]]} ${compared[3]}`);
    else return expression;
  }
  if (!when.length || holds.length > 1) return expression;
  return holds.length ? `${holds[0]} when ${when.join(" and ")}` : `never ${when.join(" and ")} together`;
}

export function scenarioProblem(factors, settings, units = {}, broken = []) {
  const not = "so the models do not predict there";
  for (const f of factors) {
    const v = settings[f.name];
    if (v === undefined) return `These settings do not set ${f.name}, ${not}.`;
    if (f.kind === "categorical") {
      if (!f.levels.includes(v)) return `${f.name} at ${v} is none of its levels, ${listed(f.levels)}, ${not}.`;
      continue;
    }
    const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
    if (!(Number.isFinite(v) && v >= low && v <= high)) {
      const value = typeof v === "number" ? withUnit(v, units[f.name]) : v;
      return `${f.name} = ${value} is outside its range, ${num(low)} to ${withUnit(high, units[f.name])}, ${not}.`;
    }
  }
  if (!broken.length) return null;
  return `These settings break ${listed(broken.map((limit) => `your limit ${limit.index + 1}, ${limitWords(limit.expression)}`))}, ${not}.`;
}

/** Settings as the list of kept settings writes them: "T = 175 degC, Catalyst at B", for each factor they set. */
export function settingsWords(factors, settings, units = {}) {
  return factors
    .filter((f) => f.name in settings)
    .map((f) => settingWords({ kind: f.kind, name: f.name, value: settings[f.name] }, units[f.name]))
    .join(", ");
}

/**
 * The name settings are kept under, from what the reader typed (on one line, at most ``held.name`` characters), or
 * "Settings 1" and on when nothing was typed, and what stops them being kept, in a sentence, or null: the study keeps
 * ``held.count`` already, or keeps settings under that name. ``kept`` is the study's, each {name, settings}.
 */
export function keptAs(typed, kept, held) {
  const taken = new Set(kept.map((s) => s.name));
  let name = typed.replace(/[\s\u0000-\u001f\u007f-\u009f]+/g, " ").trim().slice(0, held.name).trim();
  if (name && taken.has(name)) return { name, problem: `Settings are kept as ${name} already: give these another name.` };
  for (let n = kept.length + 1; !name || taken.has(name); n++) name = `Settings ${n}`;
  if (kept.length >= held.count) return { name, problem: `A study keeps ${held.count} settings at most: delete one to keep these.` };
  return { name, problem: null };
}

/** How many points a numeric factor's profile is drawn through, its two ends among them. */
export const PROFILE_POINTS = 25;

/**
 * One factor's profile for a response: what the response's model predicts as that factor alone moves, the others
 * where ``settings`` has them. ``fit`` is the response's, as bootstrap.predictors sends it, and ``factors`` its list.
 * A numeric factor has a point at each of PROFILE_POINTS values from its low end to its high end, a categorical one
 * at each of its levels. Each point is its ``value`` with what predictAt gives there: ``predicted``, the
 * ``confidence`` interval for the mean (null when the fit has no run left to measure the noise), and whether the
 * settings lie beyond the runs (``extrapolating``).
 */
export function profileOf(fit, factors, settings, name) {
  const f = factors.find((one) => one.name === name);
  const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
  const across = Array.from({ length: PROFILE_POINTS }, (_, i) => low + ((high - low) * i) / (PROFILE_POINTS - 1));
  return (f.kind === "numeric" ? across : f.levels).map((value) => {
    const { predicted, confidence, extrapolating } = predictAt(fit, columnsAt(factors, { ...settings, [name]: value }));
    return { value, predicted, confidence, extrapolating };
  });
}

/**
 * The lowest and the highest that any of the ``profiles`` reaches, band and all: the one scale a row of profiles is
 * drawn to, so a steeper line is a larger effect. A level response is given room about its value. ``within`` is a
 * range to hold as well, which keeps the scale still while the reader drags.
 */
export function profileRange(profiles, within = null) {
  const reached = profiles.flat().flatMap((point) => [point.predicted, ...(point.confidence ?? [])]);
  const all = [...reached, ...(within ?? [])].filter(Number.isFinite);
  if (!all.length) return null;
  const [low, high] = [Math.min(...all), Math.max(...all)];
  const room = high > low ? 0 : Math.max(Math.abs(low) * 0.05, 0.5);
  return [low - room, high + room];
}

/**
 * A profile's line as the pieces it is drawn in: each a run of neighbouring points that are all within the runs, or
 * all beyond them (``beyond``), which the page draws dashed. A piece beyond the runs starts and ends at the points
 * within them that it joins, so the line has no gap.
 */
export function profilePieces(points) {
  const pieces = [];
  points.forEach((point, i) => {
    const last = pieces.at(-1);
    if (last && last.beyond === point.extrapolating) last.points.push(point);
    else pieces.push({ beyond: point.extrapolating, points: last && point.extrapolating ? [points[i - 1], point] : [point] });
    if (last && last.beyond && !point.extrapolating) last.points.push(point);
  });
  return pieces;
}

/**
 * A profile in words, for a reader who does not see it: the response at the factor's two ends (or at each level of
 * a categorical factor), where it is highest when that is inside the range, and whether part of it lies beyond the
 * runs. ``words`` is the response's ``label`` and ``units``; ``units`` the factor's.
 */
export function profileWords(f, points, { label, units }, factorUnits = "") {
  const at = (point) => `${withUnit(point.predicted, units)} at ${f.kind === "numeric" ? withUnit(point.value, factorUnits) : point.value}`;
  const beyond = points.some((point) => point.extrapolating) ? " Part of it lies beyond the runs." : "";
  if (f.kind !== "numeric") return `${label} at each level of ${f.name}: ${listed(points.map(at))}.${beyond}`;
  const [first, last] = [points[0], points.at(-1)];
  const top = points.reduce((best, point) => (point.predicted > best.predicted ? point : best));
  const bottom = points.reduce((best, point) => (point.predicted < best.predicted ? point : best));
  const inside = [first, last].includes(top) ? ([first, last].includes(bottom) ? "" : `, lowest at ${at(bottom)}`) : `, highest at ${at(top)}`;
  return `${label} against ${f.name}, the other factors where they are set: ${at(first)}, ${at(last)}${inside}.${beyond}`;
}

/** How many points along each side a response's map is drawn through, its two ends among them. */
export const MAP_POINTS = 21;

/**
 * The two factors a response's map is drawn over: the two ``chosen`` when both are numeric factors of the study,
 * else its first two numeric factors; null for a study with fewer than two, which has no map.
 */
export function mapPair(factors, chosen = null) {
  const numeric = factors.filter((f) => f.kind === "numeric").map((f) => f.name);
  if (chosen?.length === 2 && chosen[0] !== chosen[1] && chosen.every((name) => numeric.includes(name))) return chosen;
  return numeric.length < 2 ? null : numeric.slice(0, 2);
}

/**
 * A response's map: what its model predicts over two factors (``pair``), the others where ``settings`` has them.
 * ``x`` and ``y`` are MAP_POINTS values of the first and the second factor, from its low end to its high end;
 * ``z[j][i]`` is the prediction at ``x[i]`` and ``y[j]``, and ``beyond[j][i]`` is 1 where those settings lie beyond
 * the runs, as predictAt judges it, and 0 where they do not.
 */
export function surfaceOf(fit, factors, settings, pair) {
  const side = (name) => {
    const f = factors.find((one) => one.name === name);
    const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
    return Array.from({ length: MAP_POINTS }, (_, i) => low + ((high - low) * i) / (MAP_POINTS - 1));
  };
  const [x, y] = pair.map(side);
  const at = y.map((second) => x.map((first) => predictAt(fit, columnsAt(factors, { ...settings, [pair[0]]: first, [pair[1]]: second }))));
  return { x, y, z: at.map((row) => row.map((one) => one.predicted)), beyond: at.map((row) => row.map((one) => Number(one.extrapolating))) };
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
  const said = broken.map((limit) => `These settings break your limit ${limit.index + 1}, ${limitWords(limit.expression)}.`);
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

// ---------------------------------------------------------------- the trade-offs between the responses
/**
 * How far a response's value falls short of what its goal asks for, to compare two values by: the smaller, the
 * better. To maximise, a larger value is better; to minimise, a smaller one; for a target, one nearer the target.
 */
export const shortfall = (goal, value) =>
  goal.goal === "maximize" ? -value : goal.goal === "minimize" ? value : Math.abs(value - goal.target);

/**
 * Which candidate settings meet every limit the reader typed: ``responses`` holds each response's prediction at
 * every candidate (bootstrap.trade_offs), and ``limits`` a response's two ends, the least and the most it may be,
 * null for an end left open. A candidate with no prediction for a limited response, one past its transform's range,
 * does not meet the limit.
 */
export function meetsLimits(responses, limits) {
  const n = Object.values(responses)[0]?.length ?? 0;
  const ends = Object.entries(limits).filter(([name, [low, high]]) => name in responses && (low !== null || high !== null));
  return Array.from({ length: n }, (_, i) =>
    ends.every(([name, [low, high]]) => {
      const value = responses[name][i];
      return value !== null && (low === null || value >= low) && (high === null || value <= high);
    }),
  );
}

/**
 * The best trade-off between two responses among the candidates kept: those that no other candidate kept beats
 * on both. ``across`` and ``up`` are each candidate's shortfall on the two responses (see ``shortfall``). Returns
 * their places among the candidates, from the best on the first response to the best on the second. It is found
 * among the settings tried, so it is as close to the true trade-off as they are dense.
 */
export function frontier(across, up, kept) {
  const order = across.map((_, i) => i).filter((i) => kept[i] && across[i] !== null && up[i] !== null);
  order.sort((a, b) => across[a] - across[b] || up[a] - up[b]);
  const front = [];
  let best = Infinity;
  for (const i of order) {
    if (up[i] < best) {
      front.push(i);
      best = up[i];
    }
  }
  return front;
}

// ---------------------------------------------------------------- the settings that give a target
/** "the target" for one response with a target, "every target" for more. */
const theTargets = (found) => (found.responses.length === 1 ? "the target" : "every target");

/** Whether the models can give the targets inside the ranges, as the sentence over the settings found says it. */
export function targetLead(found, { limits = false, blend = false } = {}) {
  if (blend) {
    return found.reached
      ? `This blend gives ${theTargets(found)}, by the models.`
      : `No blend within your bounds gives ${theTargets(found)}, by the models. This one comes closest.`;
  }
  if (found.reached) return `These settings give ${theTargets(found)}, by the models.`;
  const inside = limits ? "inside your ranges and limits" : "inside your ranges";
  return `No settings ${inside} give ${theTargets(found)}, by the models. These come closest.`;
}

/**
 * How many of the settings tried give the targets, and which of them the answer is; nothing when none does.
 * ``found`` is bootstrap.target_settings' reply: its ``share`` of the ``tried`` settings, each response within
 * ``near`` of the range of its results from its target.
 */
export function targetShare(found, { blend = false } = {}) {
  if (!found.reached || !found.share) return "";
  const percent = Number((100 * found.share).toPrecision(2));
  const many = percent < 0.1 ? "Fewer than 1 in 1,000" : `About ${percent}%`;
  const one = found.responses.length === 1;
  const which = blend
    ? "more than one blend does. This is the one nearest the middle of the blends your bounds allow."
    : "more than one setting does. These are the ones nearest the center of your ranges, where your runs are gathered.";
  return (
    `${many} of the ${found.tried.toLocaleString("en")} ${blend ? "blends" : "settings"} tried give ${theTargets(found)} to within ` +
    `${marked(String(Number((100 * found.near).toPrecision(2))))}% of the range of ${one ? "the response's" : "each response's"} results: ${which}`
  );
}

/** The rows of the table of each response at the settings found: its target, its prediction and its interval. */
export function targetRows(found) {
  return found.responses.map((r) => [
    r.label,
    withUnit(r.target, r.units),
    withUnit(r.predicted, r.units),
    r.prediction ? rangeWords(r.prediction, r.units) : "",
  ]);
}

/**
 * What one factor must be for a response to meet its target, the others held: a sentence for each setting that does,
 * or the nearest the response comes. ``solved`` is bootstrap.solve_for's reply; ``held`` says where the other factors
 * are, as settingsWords writes them; ``units`` are the factor's and the response's, and ``label`` the response's.
 */
export function solveSentences(solved, { held = "", units = "", label = solved.response, responseUnits = "" } = {}) {
  const others = held ? `With ${held} held, ` : "";
  const target = `${label} ${withUnit(solved.target, responseUnits)}`;
  if (!solved.solutions.length) {
    const closest = `${withUnit(solved.nearest.predicted, responseUnits)}, at ${solved.factor} = ${withUnit(solved.nearest.value, units)}`;
    return [`${others}no setting of ${solved.factor} in its range gives ${target}, by the model. The prediction comes closest at ${closest}.`];
  }
  return solved.solutions.map((s) => {
    const said = `${others}${solved.factor} = ${withUnit(s.value, units)} gives ${target}, by the model.`;
    if (!s.interval) return `${said} The fit uses every run, so it has no interval.`;
    const [low, high] = s.interval;
    const from = low === null ? "the low end of its range" : withUnit(low, units);
    const to = high === null ? "the high end of its range" : withUnit(high, units);
    return `${said} From ${from} to ${to}, the 95% confidence interval of the prediction still holds the target.`;
  });
}

/**
 * The caption of the chart of the settings that give the targets: the two factors drawn and where the others are held,
 * what a line, the star, the dots and the grey mean, and the responses with no line, which no setting drawn brings to
 * its target. ``overlay`` and ``responses`` are bootstrap.target_settings'; ``factors`` the spec's, for their units.
 */
export function targetCaption(overlay, factors = [], responses = []) {
  const blend = overlay.kind === "ternary"; // a mixture's chart is over the triangle of its three components
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = (h) => (typeof h.value === "string" ? `${h.name} at ${h.value}` : `${h.name} at ${withUnit(h.value, units[h.name])}`);
  const held = overlay.held ?? [];
  const others = held.length ? `, with ${listed(held.map(at))}, as at the settings found` : "";
  const where = blend ? TRIANGLE : `Drawn over ${overlay.x} and ${overlay.y}${others}.`;
  const one = blend ? "blend" : "setting";
  const lines =
    overlay.responses.length === 1
      ? `The line is where the response is predicted at its target: every ${one} on it gives the target.`
      : `Each line is where a response is predicted at its target: every ${one} on it gives that target, and where the lines cross, the ${blend ? "blend gives" : "settings give"} every target.`;
  const label = Object.fromEntries(responses.map((r) => [r.name, r.label]));
  const undrawn = overlay.responses
    .filter(({ z, levels }) => !(levels[0] > Math.min(...z.flat()) && levels[0] < Math.max(...z.flat())))
    .map((surface) => label[surface.name] ?? surface.name);
  const missed = undrawn.length ? ` No line for ${listed(undrawn)}: no ${one} drawn brings ${undrawn.length === 1 ? "it to its target" : "them to their targets"}.` : "";
  const grey = overlay.runnable ? " Grey: settings outside your limits." : "";
  return `${where} ${lines} The star marks the ${blend ? "blend" : "settings"} found, and the dots your runs.${grey}${missed}`;
}

// ---------------------------------------------------------------- the settings that meet a specification
/** A share as the page writes it: "17%", "3.4%", "less than 0.1%". */
export function shareWords(share) {
  const percent = 100 * share;
  if (percent > 0 && percent < 0.1) return `less than ${marked("0.1")}%`;
  return `${percent >= 10 ? Math.round(percent) : marked(String(Number(percent.toFixed(1))))}%`;
}

/** "the window" for one response with a window, "every window" for more. */
const theWindows = (found) => (found.responses.length === 1 ? "its window" : "every window");

/** The categorical factors a specification is judged with, each at its level, as a clause; nothing without any. */
const heldClause = (found) => (found.held.length ? `, with ${listed(found.held.map((h) => `${h.name} at ${h.value}`))}` : "");

/**
 * Over what share of the settings the models predict every response inside its window, as the sentence that leads the
 * section. ``found`` is bootstrap.specification's reply; ``limits`` says the study has limits on its settings.
 */
export function windowLead(found, { limits = false, blend = false } = {}) {
  const inside = blend ? "within your bounds" : limits ? "inside your ranges and limits" : "inside your ranges";
  const each = found.responses.length === 1 ? "the response" : "every response";
  const [one, many] = blend ? ["blend", "blends"] : ["setting", "settings"];
  if (found.share === 0) return `By the models, no ${one} ${inside} gives ${each} inside ${theWindows(found)}${heldClause(found)}.`;
  return `By the models, ${shareWords(found.share)} of the ${many} ${inside} give ${each} inside ${theWindows(found)}${heldClause(found)}.`;
}

/**
 * What the windows give at each combination of the categorical factors' levels, as a table's columns and rows: the
 * share of the settings predicted inside every window there, the highest chance that one new run meets them ("none
 * given" without runs to estimate the noise), and which combination What if has, since the sentences and the chart
 * are of that one. ``found`` is bootstrap.specification's reply. Null for a study with no categorical factor.
 */
export function windowLevels(found) {
  const levels = found.levels ?? [];
  if (!levels.length) return null;
  const rows = levels.map((row) => [
    ...row.levels.map((level) => String(level.value)),
    shareWords(row.share),
    row.surest === null ? "none given" : shareWords(row.surest),
    row.here ? "yes" : "",
  ]);
  return { columns: [...levels[0].levels.map((level) => level.name), "Settings that meet it", "Highest chance", "Under What if"], rows };
}

/**
 * How surely one new run meets the windows: the share of the settings at which it does with each chance drawn, and
 * that the responses' errors were taken as correlated when they were (``found.search.correlation``), or why
 * no chance can be given. Nothing when no setting is predicted inside the windows.
 */
export function windowChance(found, { blend = false } = {}) {
  if (found.share === 0) return "";
  const all = found.responses.length === 1 ? "its window" : "every window";
  const [one, many] = blend ? ["blend", "blends"] : ["setting", "settings"];
  // Said when it is so: with too few runs to estimate it, the responses' errors are taken as independent.
  const together = found.search?.correlation ? " The responses' errors are taken as correlated as the residuals of their fits are." : "";
  if (!found.likely) return `No runs are left to estimate the noise, so the chance that a new run meets ${all} cannot be given.`;
  const [low, high] = Object.keys(found.likely).map(Number).sort((a, b) => a - b);
  const at = (chance) => found.likely[String(chance)];
  const words = (chance) => `${Math.round(100 * chance)}%`;
  if (at(high) === 0) {
    const half = at(low) === 0 ? `, nor of ${words(low)}` : `; it does with ${words(low)} or more at ${shareWords(at(low))} of them`;
    return `At no ${one} does one new run meet ${all} with a chance of ${words(high)} or more${half}. The noise is wide beside the window, or the runs are few.${together}`;
  }
  return (
    `One new run meets ${all} with a chance of ${words(high)} or more at ${shareWords(at(high))} of the ${many}, and of ` +
    `${words(low)} or more at ${shareWords(at(low))}. A prediction inside a window is not a run inside it: the noise and the fit's own doubt count too.${together}`
  );
}

/**
 * Where one new run is surest to meet the windows, and how sure. ``factors`` are the study's with their kinds, and
 * ``units`` each one's units, as settingsWords takes them. Nothing when no chance can be given.
 */
export function surestSentence(found, factors, units = {}, { blend = false } = {}) {
  if (!found.surest) return "";
  const all = found.responses.length === 1 ? "its window" : "every window";
  const chance = found.surest.probability;
  const sure = chance >= 0.995 && chance < 1 ? "more than 99%" : shareWords(chance); // never "100%" of a chance that is not
  const at = `${blend ? "the blend " : ""}${settingsWords(factors, found.surest.settings, units)}`;
  return `A new run is surest to meet ${all} at ${at}: a chance of ${sure}.`;
}

/**
 * The caption of the chart of a specification: the two factors drawn and where the others are held, and what the
 * shading, the lines, the lines of equal chance, the star, the dots and the grey mean.
 */
export function windowCaption(overlay, factors = []) {
  const blend = overlay.kind === "ternary"; // a mixture's chart is over the triangle of its three components
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const at = (h) => (typeof h.value === "string" ? `${h.name} at ${h.value}` : `${h.name} at ${withUnit(h.value, units[h.name])}`);
  const held = overlay.held ?? [];
  const others = held.length ? `, with ${listed(held.map(at))}, as What if has ${held.length === 1 ? "it" : "them"}` : "";
  // The triangle has each chance as a line of its own; the chart over two factors, the chances and their surface.
  const drawn = (line) => line.a.some((v) => v !== null); // a chance no blend reaches has no line
  const chances = blend ? (overlay.chances ?? []).filter(drawn).map((line) => line.chance) : overlay.probability ? overlay.chances : [];
  const chance = chances.length
    ? ` Inside the line of ${Math.round(100 * Math.max(...chances))}%, a new run meets every window at least that often: the fit's doubt and the noise are counted.`
    : "";
  const grey = overlay.runnable ? " Grey: settings outside your limits." : "";
  const star = blend ? (overlay.at ? " The star marks the blend at which a new run is surest to meet them, and the dots your runs." : " The dots are your runs.") : " The star marks the settings What if holds, and the dots your runs.";
  return (
    `${blend ? TRIANGLE : `Drawn over ${overlay.x} and ${overlay.y}${others}.`} Shaded where every response is predicted inside its window; each ` +
    `coloured line is the end of a response's window.${chance}${star}${grey}`
  );
}
