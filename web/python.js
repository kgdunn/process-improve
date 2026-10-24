// "Show the Python code": the process-improve code that does what the page just did, step by
// step, written from the page's state. Each panel shows its step after the steps it builds on,
// so the code copied from any one of them runs as it is, in a script or a notebook.

/**
 * A value as Python reads it back: a string in double quotes, True, False and None, a number as
 * the shortest text with its value, and a list or a dict with its items written the same way. A
 * string is never rewritten, so a label such as "true or null" stays as it was typed.
 */
function repr(value) {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "None";
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(repr).join(", ")}]`;
  const items = Object.entries(value).filter(([, v]) => v !== undefined);
  return `{${items.map(([key, v]) => `${JSON.stringify(key)}: ${repr(v)}`).join(", ")}}`;
}

/** Python keyword arguments from an object, in the order given, skipping undefined values. */
const kwargs = (options) =>
  Object.entries(options)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${repr(v)}`)
    .join(", ");

/** Whether a factor is categorical: it says so, or its two levels are labels, held as its low and high. */
const labelled = (f) => f.type === "categorical" || typeof f.low === "string";

/** A categorical factor's levels, in their order: those it lists, or its two labels. */
const levelsOf = (f) => f.levels ?? [f.low, f.high];

/** Whether a factor is categorical of three levels or more, which the model fits as its contrasts. */
const isSeveral = (f) => labelled(f) && levelsOf(f).length > 2;

/** What joins a factor's name to the number of its contrast column; mirrors bootstrap.LEVEL_JOIN. */
export const LEVEL_JOIN = "_L";

/** What joins a factor's name to the number of its indicator column; mirrors bootstrap.INDICATOR_JOIN. */
export const INDICATOR_JOIN = "_I";

/** A categorical factor's indicator columns that a model's formula uses, each with its level, for a factor nested in it. */
function indicatorsIn(formula, f) {
  if (!labelled(f) || typeof formula !== "string") return [];
  return levelsOf(f)
    .map((level, j) => ({ name: `${f.name}${INDICATOR_JOIN}${j + 1}`, level }))
    .filter(({ name }) => new RegExp(`\\b${name}\\b`).test(formula));
}

/**
 * The lines that add a categorical factor's indicators to ``X`` where the model nests another factor in it, as
 * bootstrap._nested codes it: 1 at the level and 0 elsewhere, so the term holds the other factor within each level.
 */
function indicatorLines(spec, formula) {
  const lines = (spec?.factors ?? []).flatMap((f) =>
    indicatorsIn(formula, f).map(({ name, level }) => `X[${repr(name)}] = (runs[${repr(f.name)}] == ${repr(level)}).astype(float)`),
  );
  return lines.length ? ["# A factor nested in another: an indicator for each level of the other, 1 at it and 0 elsewhere.", ...lines] : [];
}

/**
 * A level of a factor of three levels or more as its contrast columns, as bootstrap._contrasts codes it: a column for
 * each level but the last, +1 at its own level, -1 at the last level, and 0 elsewhere.
 */
function contrastsAt(f, level) {
  const levels = levelsOf(f);
  const at = levels.indexOf(level);
  return Object.fromEntries(
    levels.slice(0, -1).map((_, j) => [`${f.name}${LEVEL_JOIN}${j + 1}`, at === levels.length - 1 ? -1 : at === j ? 1 : 0]),
  );
}

/**
 * The page's mixture designs, as bootstrap.MIXTURES names them; each is the library's design type "mixture",
 * built for the Scheffé model its options name, and its factors are the components of a blend.
 */
export const MIXTURES = ["simplex_lattice", "simplex_centroid", "extreme_vertices"];

const isMixture = (spec) => MIXTURES.includes(spec.design_type);

/**
 * The factors as Python makes them, as the list ``name``. With ``threeLevels``, each numeric factor is
 * given its low, center and high values as its levels, which generate_design reads as a three-level
 * factor: the page's option n_levels=3 for a Taguchi array, which is no argument of generate_design.
 * With ``mixture``, each is a component of a blend.
 */
function factorList(factors, threeLevels = false, name = "factors", mixture = false) {
  const levels = (f) => (threeLevels ? `, levels=[${f.low}, ${(f.low + f.high) / 2}, ${f.high}]` : "");
  const units = (f) => (f.units ? `, units=${repr(f.units)}` : "");
  const rows = factors.map((f) =>
    labelled(f)
      ? `    Factor(name=${repr(f.name)}, type="categorical", levels=${repr(levelsOf(f))}),`
      : `    Factor(name=${repr(f.name)}${mixture ? ', type="mixture"' : ""}, low=${f.low}, high=${f.high}${levels(f)}${units(f)}),`,
  );
  const said = threeLevels ? "  # each numeric factor at its low, center and high values" : "";
  return [`${name} = [${said}`, ...rows, "]"].join("\n");
}

/** A design's options as generate_design takes them, and whether its numeric factors are at three levels. */
function libraryOptions(options) {
  const { n_levels: levels, ...rest } = options || {};
  return { options: rest, threeLevels: levels === 3 };
}

/**
 * The limits on the settings as Python takes them, one to a line: each limit's text, in the
 * factors' own units. A limit's note is the reader's, and stays on the page.
 */
function constraintLines(limits) {
  if (!limits?.length) return [];
  return ["constraints = [  # the settings that can be run, in the factors' own units", ...limits.map((l) => `    ${repr(l.expression)},`), "]"];
}

/** The space-filling designs, whose settings the page rounds as the reader can set them; mirrors bootstrap.SPACE_FILLING. */
export const SPACE_FILLING = ["maximin_lhs"];

/**
 * The lines that round a space-filling design's settings as the page lists them (bootstrap._round_spread): each to
 * the decimal place of a tenth of the step between its settings, so it stays in its own step.
 */
function spreadLines(budget) {
  return [
    "# Each setting rounded to a tenth of the step between a factor's settings, as the page lists it.",
    "for f in factors:",
    `    step = (f.high - f.low) / ${budget}`,
    "    result.design_actual[f.name] = result.design_actual[f.name].round(max(0, math.ceil(1 - math.log10(step))))",
  ];
}

/** Step 1: the design, as generate_design makes it, inside the limits on the settings when there are some. */
export function designCode(spec) {
  const { options, threeLevels } = libraryOptions(spec.options);
  const limits = spec.constraints ?? [];
  if (spec.design_type === "given") {
    // Runs as given: the page made none of them, so step 1 holds the factors and step 2 the runs themselves.
    return [
      "from process_improve.experiments import Factor",
      "",
      factorList(spec.factors),
      ...constraintLines(limits),
      "# The runs are taken as they were given: step 2 lists them, in the order given.",
    ].join("\n");
  }
  if (isMixture(spec)) {
    // Each component a proportion between its bounds; the library's mixture design, for the model its options name.
    return [
      "from process_improve.experiments import Factor, generate_design",
      "",
      "# The components of the blend, each a proportion between its lower and upper bounds.",
      factorList(spec.factors, false, "factors", true),
      `result = generate_design(factors, design_type="mixture", ${kwargs(options)})`,
      "print(result.design_actual)  # the blends, in randomised order: each run's proportions sum to 1",
    ].join("\n");
  }
  const spread = SPACE_FILLING.includes(spec.design_type);
  return [
    ...(spread ? ["import math", ""] : []),
    "from process_improve.experiments import Factor, generate_design",
    "",
    factorList(spec.factors, threeLevels),
    ...constraintLines(limits),
    `result = generate_design(factors, design_type=${repr(spec.design_type)}, ${limits.length ? "constraints=constraints, " : ""}${kwargs(options)})`,
    ...(spread ? spreadLines(spec.options.budget) : []),
    "print(result.design_actual)  # the runs, in randomised order, in the factors' own units",
  ].join("\n");
}

/** A number as Python reads it back to the same value: JavaScript and Python both write the shortest such text. */
const num = (x) => String(x);

/**
 * One column of a DataFrame literal, ``"T": [1, 2, ...],``: on one line while it fits in 100
 * characters, else wrapped to lines of that width, so a long run sheet stays readable.
 */
function columnLines(key, values, indent = "        ") {
  const items = values.map((v) => `${repr(v)},`);
  const one = `${indent}${repr(key)}: [${items.join(" ").slice(0, -1)}],`;
  if (one.length <= 100) return [one];
  const lines = [`${indent}${repr(key)}: [`];
  let line = "";
  for (const item of items) {
    if (line && indent.length + 4 + line.length + 1 + item.length > 100) {
      lines.push(`${indent}    ${line}`);
      line = item;
    } else {
      line = line ? `${line} ${item}` : item;
    }
  }
  return [...lines, `${indent}    ${line}`, `${indent}],`];
}

/**
 * How the page codes a factor's settings, as a line of the ``X`` literal: (setting - centre) /
 * half-range for a number, and -1 and +1 for a category's two labels. A factor of three levels or
 * more is a line for each of its contrast columns. The centre and half-range are worked out as
 * bootstrap.py works them out, so the coded values agree to the last digit.
 */
function codedLine(f) {
  const column = `runs[${repr(f.name)}]`;
  if (isSeveral(f)) {
    const coded = levelsOf(f).map((level) => contrastsAt(f, level));
    return [
      `        # ${f.name}: a column for each level but the last, +1 at its level, -1 at the last level, 0 elsewhere.`,
      ...Object.keys(coded[0]).map((name) => {
        const map = levelsOf(f).map((level, i) => `${repr(level)}: ${coded[i][name].toFixed(1)}`);
        return `        ${repr(name)}: ${column}.map({${map.join(", ")}}),`;
      }),
    ].join("\n");
  }
  if (labelled(f)) return `        ${repr(f.name)}: ${column}.map({${repr(f.low)}: -1.0, ${repr(f.high)}: 1.0}),`;
  const centre = (f.high + f.low) / 2;
  const half = (f.high - f.low) / 2;
  const shifted = centre === 0 ? column : `${column} ${centre < 0 ? "+" : "-"} ${num(Math.abs(centre))}`;
  const scaled = half === 1 ? shifted : `${centre === 0 ? shifted : `(${shifted})`} / ${num(half)}`;
  return `        ${repr(f.name)}: ${scaled},`;
}

/** The run sheet's column marking a confirmation run; mirrors best.js, which this module does not import. */
export const CONFIRMATION = "Confirmation run";

/** Whether each run of a run sheet is fitted: every run but a confirmation run, which tests the model. */
export function fittedRuns(design) {
  const at = design.columns.indexOf(CONFIRMATION);
  return design.rows.map((row) => at < 0 || String(row[at] ?? "").trim().toLowerCase() !== "yes");
}

/** The decimals the page rounds simulated results to: two significant decimals below the noise's size. */
const decimals = (sigma) => Math.max(0, 2 - Math.floor(Math.log10(sigma)));

/** What ``y`` is, as a comment: the response in the reader's words and units ("# Yield, in %"), or nothing. */
function responseComment(spec) {
  const info = spec.response_info ?? {};
  const name = info.label || spec.response || "y";
  if (!info.label && !info.units) return "";
  return `  # ${name}${info.units ? `, in ${info.units}` : ""}`;
}

/** A specification's responses' columns, the first first, as logic.js responsesOf names them. */
const responseNames = (spec) => (spec.responses?.length ? spec.responses.map((r) => r.name) : [spec.response || "y"]);

/** A study's specification as one of its responses' code reads it: that response as its ``response``. */
export function forResponse(spec, name) {
  const found = (spec.responses ?? []).find((r) => r.name === name);
  if (!found || name === spec.response) return spec;
  const { name: _, ...info } = found;
  return { ...spec, response: name, response_info: info };
}

/**
 * Step 2: the run sheet as the page lists it, its settings in coded units as ``X``, and the
 * results as ``y``: simulated from ``simulation``'s process when it is given, else the results
 * on the page, None for a run still to do. The runs are written out rather than taken from
 * step 1's ``result``, because the page may list them in another order: as the book's table
 * does, or as a workbook was sorted.
 *
 * ``results`` is the first response's results, one per run, or each response's by its column. With
 * several responses the run sheet holds a column for each, and ``y`` is the first: each response's
 * analysis takes its own column (analyseCode). Simulated, each has its process and its seed.
 */
export function resultsCode(design, results, simulation = null) {
  const { spec, columns } = design;
  const names = responseNames(spec);
  const byName = Array.isArray(results) ? { [names[0]]: results } : results;
  const kept = fittedRuns(design);
  const rows = design.rows.filter((_, i) => kept[i]);
  const values = Object.fromEntries(names.map((name) => [name, (byName[name] ?? []).filter((_, i) => kept[i])]));
  const left = kept.includes(false) ? ["# The confirmation runs are left out: they test the model, so they do not enter its fit."] : [];
  const at = (name) => columns.indexOf(name);
  const settings = (f) => rows.map((row) => (labelled(f) ? row[at(f.name)] : Number(row[at(f.name)])));
  const blocked = at("Block") >= 0 && (spec.options?.n_blocks ?? 1) > 1;
  const mixture = isMixture(spec);
  const response = names[0];
  const several = names.length > 1;
  const runs = [
    ...spec.factors.flatMap((f) => columnLines(f.name, settings(f))),
    ...(blocked ? columnLines("Block", rows.map((row) => row[at("Block")])) : []),
    ...(simulation ? [] : names.flatMap((name) => columnLines(name, values[name]))),
  ];
  const waiting = names.some((name) => values[name].includes(null));
  const lines = [
    "import pandas as pd",
    ...(simulation ? ["from process_improve.experiments import TrueProcess"] : []),
    "",
    simulation
      ? "# The run sheet, as the page lists it: each run's settings, in the factors' own units."
      : `# The run sheet, as the page lists it: each run's settings and ${several ? "results" : "result"}${
          waiting ? "; None for a run still to do" : ""
        }.`,
    ...left,
    "runs = pd.DataFrame(",
    "    {",
    ...runs,
    "    }",
    ")",
    mixture
      ? "# The proportions, as the Scheffé model is fitted: a mixture's components are not coded."
      : "# The settings in coded units, as the model is fitted: -1 at a factor's low level, +1 at its high.",
    "X = pd.DataFrame(",
    "    {",
    ...spec.factors.map((f) => (mixture ? `        ${repr(f.name)}: runs[${repr(f.name)}],` : codedLine(f))),
    ...(blocked ? ['        "Block": runs["Block"],'] : []),
    "    }",
    ")",
  ];
  if (!simulation) return [...lines, `y = runs[${repr(response)}]${responseComment(spec)}`].join("\n");
  const process = (one) => ["TrueProcess(", "    {", ...one.truth.map((t) => `        ${repr(t.term)}: ${t.coefficient},`), "    },", `    sigma=${one.sigma},`, ")"];
  if (!several) {
    return [
      ...lines,
      "",
      `process = ${process(simulation).join("\n")}`,
      ...(responseComment(spec) ? [`#${responseComment(spec).slice(3)}, simulated from a known process`] : []),
      `y = process.simulate(X, random_state=${simulation.seed}).round(${decimals(simulation.sigma)})  # as the page rounds them`,
    ].join("\n");
  }
  const each = names.map((name, i) => [name, i === 0 ? simulation : simulation.more?.[name]]).filter(([, one]) => one);
  return [
    ...lines,
    "",
    "# Each response's process, simulated with a seed of its own and rounded as the page rounds it.",
    "processes = {}",
    ...each.map(([name, one]) => `processes[${repr(name)}] = ${process(one).join("\n")}`),
    `seeds = ${repr(Object.fromEntries(each.map(([name, one]) => [name, one.seed])))}`,
    `decimals = ${repr(Object.fromEntries(each.map(([name, one]) => [name, decimals(one.sigma)])))}`,
    "for name, process in processes.items():",
    "    runs[name] = process.simulate(X, random_state=seeds[name]).round(decimals[name])",
    `y = runs[${repr(response)}]${responseComment(spec)}`,
  ].join("\n");
}

/**
 * Each transform a response can be analysed with (bootstrap.TRANSFORMS), as the code writes it: how the results are
 * put on its scale, and how a number on its scale is put back in the response's units. -1/y keeps the results'
 * order, so the goals mean the same on its scale, and it tests as 1/y does.
 */
const SCALES = {
  log: { on: (y) => `np.log(${y})`, back: "np.exp", words: "its log" },
  sqrt: { on: (y) => `np.sqrt(${y})`, back: "np.square", words: "its square root" },
  inverse: { on: (y) => `-1 / ${y}`, back: "lambda v: -1 / v", words: "-1/y, which tests as 1/y does" },
};

/** The transform a specification's response is analysed with, or null: bootstrap._transform_of. */
const transformOf = (spec) => spec?.analysis_transforms?.[spec?.response] ?? null;

/** The results as the model is fitted to them: ``y``, or its transform's expression of it. */
const onScale = (y, transform) => (transform ? SCALES[transform].on(y) : y);

/** The transformed responses of a code that fits several, as a dict of how each is put on its scale, and back. */
function scaleLines(spec, names) {
  const kept = Object.entries(spec.analysis_transforms ?? {}).filter(([name]) => names.includes(name));
  if (!kept.length) return { lines: [], on: (y) => y, back: (v) => v };
  const as = (key) => `{${kept.map(([name, t]) => `${repr(name)}: ${key === "on" ? SCALES[t].on("v").replace(/^/, "lambda v: ") : SCALES[t].back}`).join(", ")}}`;
  return {
    lines: [
      "import numpy as np",
      "# The responses fitted on a transform's scale, as the page fits them, and how a number is put back in their units.",
      `on_scale = ${as("on")}`,
      `put_back = ${as("back")}`,
    ],
    on: (y) => `on_scale.get(name, lambda v: v)(${y})`,
    back: (v) => `put_back.get(name, lambda v: v)(${v})`,
  };
}

/**
 * Step 3: the analysis of step 2's ``X`` and ``y``, with the model the page fitted. ``complete``
 * says every run has its result; when one does not, the runs still to do are left out first.
 * ``spec`` names a response other than the study's first when it is that response's analysis: ``y``
 * is its column then. ``vifModel``, the terms fitted as a formula, is given when the page shows their
 * VIFs: the code then works them out from the same runs, as the page's VIF column has them.
 */
export function analyseCode(model, complete = true, spec = null, vifModel = null, { compared = false, control = null } = {}) {
  const transform = transformOf(spec);
  const fitted = complete ? `X, ${onScale("y", transform)}` : `X[done], ${onScale("y[done]", transform)}`;
  const own = Boolean(spec?.response) && spec.response !== responseNames(spec)[0];
  const vif = vifModel
    ? [
        "# Each term's variance inflation factor: how many times larger its coefficient's variance is",
        "# because the term is correlated with the others in these runs (the page's VIF column).",
        `vif = evaluate_design(${complete ? "X" : "X[done]"}, model=${repr(vifModel)}, metric="vif")["vif"]`,
        "print(vif)",
      ]
    : [];
  return [
    ...(transform ? ["import numpy as np"] : []),
    `from process_improve.experiments import analyze_experiment${vifModel ? ", evaluate_design" : ""}`,
    "",
    ...(own ? [`y = runs[${repr(spec.response)}]${responseComment(spec)}`] : []),
    ...(transform ? [`# The model is fitted to ${SCALES[transform].words}, as the page fits it; y stays as measured.`] : []),
    ...indicatorLines(spec, model),
    ...(complete ? [] : ["done = y.notna()  # leave out the runs still to do, as the page does"]),
    "analysis = analyze_experiment(",
    `    ${fitted}, model=${repr(model)}, analysis_type=["coefficients", "anova", "lack_of_fit"],`,
    '    coding="actual",  # the coefficients of X as it is, as the page shows them',
    ")",
    'print(analysis["coefficients"])',
    ...vif,
    ...(compared ? comparedCode(spec, complete, control) : []),
  ].join("\n");
}

/** The arrangements the randomization test counts at most, drawing that many at random beyond: bootstrap.ARRANGEMENTS. */
export const ARRANGEMENTS = 200_000;

/** The code that compares one factor's levels as the page does: three levels or more, two in pairs, or two. */
function comparedCode(spec, complete, control) {
  if (isSeveral(spec.factors[0])) return severalCode(spec, complete, control);
  return (spec.options?.n_blocks ?? 1) > 1 ? pairedCode(spec, complete) : comparisonCode(spec, complete);
}

/**
 * A study of one factor at two levels run in pairs, each block one run at each level (``compared``, as the page's
 * analysis compares them): the high level's result less the low's within each pair, tested as the page tests it.
 * The randomization tests flip the signs of the differences, and of their ranks, with the study's seed.
 */
function pairedCode(spec, complete) {
  const name = repr(spec.factors[0].name);
  const seed = spec.options?.random_state ?? 0;
  const most = String(ARRANGEMENTS).replace(/\B(?=(\d{3})+$)/g, "_");
  return [
    "",
    `# The two levels of ${spec.factors[0].name} compared within each pair: each block holds one run at each level.`,
    "import numpy as np",
    "import pandas as pd",
    "from process_improve.univariate import randomization_test, ttest_paired",
    "",
    `pairs = pd.DataFrame({"block": runs["Block"], "level": X[${name}], "y": y})${complete ? "" : "[done]"}`,
    'pairs = pairs.pivot(index="block", columns="level", values="y").dropna()  # a pair with a run to do waits',
    "differences = pairs[1] - pairs[-1]  # the high level's result less the low's, pair by pair",
    "paired = ttest_paired(differences, conflevel=0.95)",
    `swapped = randomization_test(differences, max_arrangements=${most}, random_state=${seed})`,
    "moved = differences[differences != 0]  # a pair that does not differ has no sign, and drops out",
    "signed_ranks = moved.abs().rank() * np.sign(moved)",
    `ranks = randomization_test(signed_ranks, max_arrangements=${most}, random_state=${seed})`,
    'print(paired, swapped, ranks, sep="\\n")',
  ];
}

/**
 * A study of one factor at two levels (``compared``, as the page's analysis compares them): the high level's results
 * less the low's, tested as the page tests them, with the seed the study's runs were put in order with, which the
 * randomization test draws its arrangements with when there are too many to count.
 */
function comparisonCode(spec, complete) {
  const name = repr(spec.factors[0].name);
  const done = complete ? "" : "done & ";
  const seed = spec.options?.random_state ?? 0;
  const most = String(ARRANGEMENTS).replace(/\B(?=(\d{3})+$)/g, "_");
  return [
    "",
    `# The two levels of ${spec.factors[0].name} compared: the high level's results less the low's.`,
    "from process_improve.univariate import randomization_test, rank_sum_test, ttest_independent",
    "",
    `low, high = y[${done}(X[${name}] == -1)], y[${done}(X[${name}] == 1)]`,
    "pooled = ttest_independent(low, high, conflevel=0.95)  # one variance for both levels",
    "welch = ttest_independent(low, high, conflevel=0.95, equal_var=False)  # each level's own variance",
    `shuffled = randomization_test(low, high, max_arrangements=${most}, random_state=${seed})`,
    `ranks = rank_sum_test(low, high, max_arrangements=${most})`,
    "print(pooled, welch, shuffled, ranks, sep=\"\\n\")",
  ];
}

/**
 * A study of one categorical factor at three levels or more (``compared``): its levels tested as the page tests them,
 * with scipy, which the library is built on. The one-way analysis of variance asks whether the averages differ at all,
 * Tukey's intervals compare every pair of levels, and Dunnett's each level with the control, the level the page
 * compares them with. Dunnett's intervals are the page's when each level has the same runs; with unequal runs the
 * page takes the comparisons' average correlation, and scipy each one's own.
 */
function severalCode(spec, complete, control) {
  const f = spec.factors[0];
  const levels = levelsOf(f);
  const base = levels.includes(control) ? control : levels[0];
  const done = complete ? "" : "done & ";
  return [
    "",
    `# The ${levels.length} levels of ${f.name} compared: whether their averages differ at all, every pair of levels by`,
    `# Tukey's intervals, and each level against the control, ${base}, by Dunnett's intervals.`,
    "from scipy import stats",
    "",
    `levels, control = ${repr(levels)}, ${repr(base)}`,
    `groups = {level: y[${done}(runs[${repr(f.name)}] == level)] for level in levels}`,
    ...(complete ? [] : ["groups = {level: g for level, g in groups.items() if len(g)}  # a level with no result yet is left out"]),
    "anova = stats.f_oneway(*groups.values())",
    "tukey = stats.tukey_hsd(*groups.values())",
    "# scipy integrates Dunnett's distribution by randomized quasi-Monte Carlo: seeded, it is the same on every run.",
    "others = (g for level, g in groups.items() if level != control)",
    "dunnett = stats.dunnett(*others, control=groups[control], random_state=1)",
    'print(anova, tukey, dunnett, sep="\\n")',
  ];
}

/** How the code shown holds some columns of the model at a value: as bootstrap._held_at does, line for line. */
const HELD_AT = [
  "def held_at(coefficients, held):",
  '    """Return the model with some columns held at a value: a block contrast at 0, a categorical factor at its level."""',
  "    model = {}",
  "    for c in coefficients:",
  '        parts = c["term"].split(":")',
  '        term = ":".join(p for p in parts if p not in held) or "Intercept"',
  '        model[term] = model.get(term, 0.0) + c["coefficient"] * math.prod(held.get(p, 1.0) for p in parts)',
  '    return [{"term": term, "coefficient": value} for term, value in model.items()]',
  "",
  "",
];

/** The columns a factor is fitted as: its own, or a factor of three levels or more's contrasts. */
const columnsOf = (f) => (isSeveral(f) ? Object.keys(contrastsAt(f, levelsOf(f)[0])) : [f.name]);

/** The level a categorical factor is held at, as its columns are held: -1 or +1 for two levels, else its contrasts. */
function heldLevel(f, held) {
  if (!isSeveral(f)) return held[f.name] < 0 ? f.low : f.high;
  return levelsOf(f).find((level) => Object.entries(contrastsAt(f, level)).every(([name, v]) => held[name] === v));
}

/**
 * The best settings in the columns the model is fitted on: each in coded units as the page found it, a categorical
 * factor of three levels or more as the contrasts of its best level.
 */
function codedPoint(best, spec) {
  const level = (f) => best.settings?.find((s) => s.name === f.name)?.value;
  return Object.assign(
    {},
    ...Object.entries(best.coded).map(([name, v]) => {
      const f = spec.factors.find((one) => one.name === name);
      return f && isSeveral(f) ? contrastsAt(f, level(f)) : { [name]: v };
    }),
  );
}

/** What the code shown says of the held columns: each categorical factor at the level found best, the blocks at their average. */
function heldComment(held, factors) {
  const levels = factors
    .filter((f) => labelled(f) && columnsOf(f).every((name) => name in held))
    .map((f) => `${f.name} at ${heldLevel(f, held)}`);
  const indicators = factors.flatMap((f) => (labelled(f) ? levelsOf(f).map((_, j) => `${f.name}${INDICATOR_JOIN}${j + 1}`) : []));
  const columns = [...factors.flatMap(columnsOf), ...indicators];
  const blocks = Object.keys(held).some((name) => !columns.includes(name));
  const said = [];
  if (levels.length) said.push(`The page tried each level and found ${levels.join(" and ")} best: the model is held there.`);
  if (blocks) said.push("The blocks are held at the average block, as the equation is.");
  return said.map((line) => `# ${line}`).join("\n");
}

/**
 * Step 4: the best settings for the response's goal, found on step 3's model with the library's optimiser, and the
 * prediction there with its intervals. ``best`` is what bootstrap.best_settings answered: its ``search`` holds what
 * the page held each categorical factor and block at, and the desirability goal it searched with. ``model`` is the
 * model step 3 fitted, and ``complete`` says every run of the design has its result, as for step 3.
 */
export function bestCode(best, spec, model, complete = true) {
  if (isMixture(spec)) return blendCode(best, spec, model, complete);
  const numeric = spec.factors.filter((f) => !labelled(f));
  const held = best.search?.held ?? {};
  const holds = Object.keys(held).length > 0;
  const limits = spec.constraints ?? [];
  const response = spec.response || "y";
  const searched = Boolean(best.search?.goal);
  const transform = transformOf(spec);
  const fitted = complete ? `X, ${onScale("y", transform)}` : `X[done], ${onScale("y[done]", transform)}`;
  const point = codedPoint(best, spec);
  const chosen = (f) => best.settings?.find((s) => s.name === f.name)?.value;
  const levels = Object.fromEntries(
    spec.factors.filter(labelled).flatMap((f) => [
      ...columnsOf(f).map((name) => [name, point[name]]),
      ...indicatorsIn(model, f).map(({ name, level }) => [name, level === chosen(f) ? 1 : 0]),
    ]),
  );
  const imports = [
    ...(holds && searched ? ["import math"] : []),
    ...(transform ? ["import numpy as np"] : []),
    "import pandas as pd",
    `from process_improve.experiments import ${limits.length && searched ? "DesignRegion, " : ""}analyze_experiment${searched ? ", optimize_responses" : ""}`,
    "",
  ];
  const at = searched
    ? `at = pd.DataFrame([{**best["optimal_coded"]${Object.keys(levels).length ? `, **${repr(levels)}` : ""}}])`
    : `at = pd.DataFrame([${repr({ ...point, ...levels })}])`;
  const prediction = [
    "# The prediction there, with its 95% confidence interval for the mean and prediction interval for one new run.",
    at,
    `prediction = analyze_experiment(${fitted}, model=${repr(model)}, analysis_type="prediction", new_points=at)`,
    'print(prediction["predictions"][0])',
    ...(transform
      ? [
          "# Put back in the response's units, as the page says them: the prediction is its median there.",
          `print({key: float((${SCALES[transform].back})(v)) for key, v in prediction["predictions"][0].items()})`,
        ]
      : []),
  ];
  if (!searched) {
    return [
      ...imports,
      "# Every factor the model holds is categorical: the page compared each combination of their levels, and these",
      "# are best, in coded units.",
      ...prediction,
    ].join("\n");
  }
  const ranges = numeric.map((f) => `${repr(f.name)}: {"low": ${f.low}, "high": ${f.high}}`).join(", ");
  return [
    ...imports,
    ...(holds ? [heldComment(held, spec.factors), ...HELD_AT, `coefficients = held_at(analysis["coefficients"], ${repr(held)})`] : []),
    "# The goal as a desirability, its limits beyond every prediction in the ranges: a straight line in the prediction.",
    `goal = ${repr(best.search.goal)}`,
    "best = optimize_responses(",
    "    [",
    "        {",
    `            "response_name": ${repr(response)},`,
    `            "coefficients": ${holds ? "coefficients" : 'analysis["coefficients"]'},`,
    `            "factor_names": ${repr(numeric.map((f) => f.name))},`,
    "        }",
    "    ],",
    "    goals=[goal],",
    `    factor_ranges={${ranges}},`,
    ...(limits.length ? ["    region=DesignRegion(factors, constraints),  # inside the limits on the settings"] : []),
    ')["desirability"]',
    'print(best["optimal_actual"])  # the best settings, in the factors\' own units',
    `print(best["predicted_responses"][${repr(response)}])  # what the model predicts there`,
    "",
    ...prediction,
  ].join("\n");
}

/**
 * Step 4 for a mixture: the best blend for the response's goal, found on step 3's Scheffé model with the library's
 * optimiser over the region of blends, a ``DesignRegion`` of step 1's components (each between its bounds, the
 * proportions summing to 1), and the prediction there with its intervals. ``best`` is what bootstrap.best_settings
 * answered: its ``search`` holds the desirability goal it searched with.
 */
function blendCode(best, spec, model, complete) {
  const response = spec.response || "y";
  const fitted = complete ? "X, y" : "X[done], y[done]";
  return [
    "import pandas as pd",
    "from process_improve.experiments import DesignRegion, analyze_experiment, optimize_responses",
    "",
    "# The goal as a desirability, its limits beyond every prediction over the blends: a straight line in the prediction.",
    `goal = ${repr(best.search.goal)}`,
    "best = optimize_responses(",
    "    [",
    "        {",
    `            "response_name": ${repr(response)},`,
    '            "coefficients": analysis["coefficients"],',
    `            "factor_names": ${repr(spec.factors.map((f) => f.name))},`,
    "        }",
    "    ],",
    "    goals=[goal],",
    "    region=DesignRegion(factors),  # the blends: each component between its bounds, the proportions summing to 1",
    ')["desirability"]',
    'print(best["optimal_coded"])  # the best blend: each component\'s proportion',
    `print(best["predicted_responses"][${repr(response)}])  # what the model predicts there`,
    "",
    "# The prediction there, with its 95% confidence interval for the mean and prediction interval for one new run.",
    'at = pd.DataFrame([best["optimal_coded"]])',
    "prediction = analyze_experiment(",
    `    ${fitted}, model=${repr(model)}, analysis_type="prediction", coding="actual", new_points=at`,
    ")",
    'print(prediction["predictions"][0])',
  ].join("\n");
}

const IMPORT = /^(import|from) /;

/**
 * Steps' code as one script that runs as it is: their imports first, merged and sorted as ruff
 * sorts them, then each step's own lines in turn. A panel shows its step this way, after the
 * steps it builds on, so the code copied from any panel runs without the others.
 */
export function cumulative(...steps) {
  const plain = new Set();
  const named = new Map();
  const bodies = steps.map((code) => {
    const lines = code.split("\n");
    let i = 0;
    for (; i < lines.length && (IMPORT.test(lines[i]) || lines[i] === ""); i++) {
      const from = lines[i].match(/^from (\S+) import (.+)$/);
      if (from) {
        const names = named.get(from[1]) ?? new Set();
        for (const name of from[2].split(", ")) names.add(name);
        named.set(from[1], names);
      } else if (lines[i]) {
        plain.add(lines[i]);
      }
    }
    return lines.slice(i).join("\n");
  });
  // As ruff orders imported names: CONSTANTS, then Classes, then functions, each group alphabetically.
  const kind = (name) => (name === name.toUpperCase() ? 0 : /^[A-Z]/.test(name) ? 1 : 2);
  const sorted = (items) => [...items].sort((a, b) => kind(a) - kind(b) || (a < b ? -1 : a > b ? 1 : 0));
  const fromLines = sorted(named.keys()).map((module) => `from ${module} import ${sorted(named.get(module)).join(", ")}`);
  return [...sorted(plain), ...fromLines, "", bodies.join("\n\n")].join("\n");
}

/**
 * Step 4: the comparison, each design generated with the options the page used for it
 * (``designs`` is a list of {label, design_type, options}). With limits on the settings, each
 * design keeps to them, and the designs are compared over the settings that can be run.
 */
export function compareCode(factors, designs, model, limits = []) {
  const kept = limits.length ? ", constraints=constraints" : "";
  const asked = designs.map((d) => libraryOptions(d.options));
  const lines = designs.map((d, i) => {
    const extra = kwargs(asked[i].options);
    const made = asked[i].threeLevels ? "three_levels" : "factors";
    return `    ${repr(d.label)}: generate_design(${made}, design_type=${repr(d.design_type)}${kept}${extra ? `, ${extra}` : ""}),`;
  });
  return [
    `from process_improve.experiments import ${limits.length ? "DesignRegion, " : ""}Factor, compare_designs, generate_design`,
    "",
    factorList(factors),
    ...(asked.some((a) => a.threeLevels) ? [factorList(factors, true, "three_levels")] : []),
    ...constraintLines(limits),
    "designs = {",
    ...lines,
    "}",
    `comparison = compare_designs(designs, model=${repr(model)}${limits.length ? ", region=DesignRegion(factors, constraints)" : ""})`,
    "print(comparison.table.to_string())",
  ].join("\n");
}

/** A notebook cell's source as nbformat keeps it: a string per line, each but the last ending in a newline. */
const lines = (text) => text.split("\n").map((line, i, all) => (i < all.length - 1 ? `${line}\n` : line));

const markdownCell = (id, text) => ({ cell_type: "markdown", id, metadata: {}, source: lines(text) });
const codeCell = (id, text) => ({ cell_type: "code", id, metadata: {}, execution_count: null, outputs: [], source: lines(text) });

/**
 * A Jupyter notebook of the campaign on the page: the install line, then each step's code, in
 * order, so running it top to bottom gives the page's design, results and analysis. Results
 * simulated and left as they were come from the same process and seed; results typed in, edited
 * or read from a workbook are written out. ``model`` is the model the analysis fitted, or null
 * when there is no analysis yet, and ``vifModel`` its terms when the page shows their VIFs (see
 * analyseCode). ``analyses`` lists each response's ``{response, model, best, vifModel}`` for a
 * study of several responses; ``version`` pins the release the page ran, when known.
 */
export function notebook({ title, design, responses, sources, simulation, model, vifModel = null, compared = false, control = null, version, url, best = null, analyses = null, balance = null }) {
  const names = responseNames(design.spec);
  const results = Array.isArray(responses) ? { [names[0]]: responses } : responses;
  const marks = Array.isArray(sources) ? { [names[0]]: sources } : sources;
  const simulated = Boolean(simulation) && Object.values(marks).every((list) => list.every((s) => s === "simulated"));
  const cells = [
    markdownCell(
      "about",
      `# ${title}\n\nThe design, its results and their analysis, as [Factorial](${url}) ran them with ` +
        `process-improve${version ? ` ${version}` : ""}. Run the cells in order.`,
    ),
    codeCell("install", version ? `%pip install "process-improve==${version}"` : "%pip install process-improve"),
    markdownCell("design-step", "## 1. The design"),
    codeCell("design", designCode(design.spec)),
    markdownCell("results-step", simulated ? "## 2. The results, simulated from a known process" : "## 2. The results"),
    codeCell("results", resultsCode(design, results, simulated ? simulation : null)),
  ];
  // Each response's analysis, and its best settings when it has a goal: the first's alone for a study of one response.
  const each = analyses ?? (model !== null && model !== undefined ? [{ response: names[0], model, best, vifModel, compared, control }] : []);
  const fitted = fittedRuns(design);
  let step = 3;
  for (const { response, model: fittedModel, best: found, vifModel: vifTerms = null, compared: levels = false, control: base = null } of each) {
    if (fittedModel === null || fittedModel === undefined) continue;
    const spec = forResponse(design.spec, response);
    const values = results[response] ?? [];
    const complete = values.every((v, i) => !fitted[i] || (v !== null && v !== undefined));
    const words = spec.response_info?.label || response;
    const id = names.length > 1 ? `-${names.indexOf(response) + 1}` : ""; // a cell's id is ASCII, as nbformat asks
    cells.push(
      markdownCell(`analysis-step${id}`, names.length > 1 ? `## ${step}. The analysis of ${words}` : `## ${step}. The analysis`),
      codeCell(`analysis${id}`, analyseCode(fittedModel, complete, spec, vifTerms, { compared: levels, control: base })),
    );
    step += 1;
    if (found?.settings) {
      cells.push(
        markdownCell(`best-step${id}`, names.length > 1 ? `## ${step}. The best settings for ${words}` : `## ${step}. The best settings for the goal`),
        codeCell(`best${id}`, bestCode(found, spec, fittedModel, complete)),
      );
      step += 1;
    }
  }
  if (balance?.balanced) {
    cells.push(markdownCell("balance-step", `## ${step}. The balance of the responses`), codeCell("balance", balanceCode(balance, design.spec)));
  }
  return {
    cells,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python" },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

/** What a goal holds as the library's desirability takes it: mirrors bootstrap.DESIRABILITY_KEYS. */
const DESIRABILITY_KEYS = ["response", "goal", "target", "low", "high", "weight", "importance"];

/**
 * The balance of the responses: each response with a goal fitted with its own model to the runs that have its result,
 * its goal as a desirability over its acceptable range, and the settings with the best overall desirability found
 * with the library's optimiser over every model. It builds on step 2's ``runs`` and ``X``. ``balance`` is what
 * bootstrap.balance answered: its ``search`` holds each response's fitted formula, its goal, and what the page held
 * each categorical factor and block at. A mixture's balance is searched over the region of blends that step 1's
 * components make.
 */
export function balanceCode(balance, spec) {
  const mixture = isMixture(spec); // a mixture's components are searched over its blends, not each over a range
  const numeric = spec.factors.filter((f) => !labelled(f));
  const held = balance.search?.held ?? {};
  const holds = Object.keys(held).length > 0;
  const limits = spec.constraints ?? [];
  const formulas = balance.search.formulas;
  const goals = balance.search.goals.map((g) => Object.fromEntries(DESIRABILITY_KEYS.filter((k) => k in g).map((k) => [k, g[k]])));
  const ranges = numeric.map((f) => `${repr(f.name)}: {"low": ${f.low}, "high": ${f.high}}`).join(", ");
  const scale = scaleLines(spec, Object.keys(formulas));
  const imports = [
    ...(holds ? ["import math"] : []),
    `from process_improve.experiments import ${limits.length || mixture ? "DesignRegion, " : ""}analyze_experiment, optimize_responses`,
    "",
    ...scale.lines,
  ];
  const searched = mixture
    ? ["    region=DesignRegion(factors),  # the blends: each component between its bounds, the proportions summing to 1"]
    : [
        `    factor_ranges={${ranges}},`,
        ...(limits.length ? ["    region=DesignRegion(factors, constraints),  # inside the limits on the settings"] : []),
      ];
  return [
    ...imports,
    ...(holds ? [heldComment(held, spec.factors), ...HELD_AT] : []),
    `# Each response with a goal, fitted with its own ${mixture ? "Scheffé model" : "model"} to the runs that have its result.`,
    "models = []",
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    fit = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="coefficients", coding="actual")`,
    `    coefficients = ${holds ? `held_at(fit["coefficients"], ${repr(held)})` : 'fit["coefficients"]'}`,
    `    models.append({"response_name": name, "coefficients": coefficients, "factor_names": ${repr(numeric.map((f) => f.name))}})`,
    "",
    "# Each goal as a desirability: 0 outside its acceptable range (low to high), 1 at its good end; its weight shapes",
    "# the ramp between, and its importance is how much it counts in the overall desirability.",
    ...(scale.lines.length ? ["# A transformed response's range and target are on its scale, as the page balances it."] : []),
    "goals = [",
    ...goals.map((g) => `    ${repr(g)},`),
    "]",
    "balanced = optimize_responses(",
    "    models,",
    "    goals=goals,",
    ...searched,
    ')["desirability"]',
    mixture
      ? 'print(balanced["optimal_coded"])  # the balanced blend: each component\'s proportion'
      : 'print(balanced["optimal_actual"])  # the balanced settings, in the factors\' own units',
    'print(balanced["predicted_responses"])  # what each model predicts there',
    'print(balanced["individual_desirability"])  # each response\'s desirability, d',
    'print(balanced["composite_desirability"])  # the overall desirability, D',
  ].join("\n");
}

/**
 * What if: each response's prediction at the reader's settings, from its own model fitted to the runs that have its
 * result, with its 95% confidence interval for the mean and prediction interval for one new run. ``formulas`` holds
 * each response's model as bootstrap.predictors fitted it, by its column; ``at`` the settings as the columns of ``X``
 * (best.js columnsAt), and ``settings`` as the reader set them, for the comment. ``kept`` holds the settings kept with
 * the study that the models predict at, each its ``name`` and its columns ``at``: each response is predicted at them
 * too. It builds on the design and its results.
 */
export function whatIfCode(spec, formulas, at, settings, kept = []) {
  const words = spec.factors.map((f) =>
    labelled(f) ? `${f.name} at ${settings[f.name]}` : `${f.name} = ${Number(settings[f.name].toPrecision(4))}${f.units ? ` ${f.units}` : ""}`,
  );
  const models = Object.values(formulas).join(" + ");
  const used = new Set([...spec.factors.flatMap(columnsOf), ...spec.factors.flatMap((f) => indicatorsIn(models, f).map(({ name }) => name))]);
  const only = (columns) => Object.fromEntries(Object.entries(columns).filter(([name]) => used.has(name)));
  const scale = scaleLines(spec, Object.keys(formulas));
  const keptLines = [
    "",
    "# The settings kept with the study, by their names, and each response's prediction at each.",
    `kept = pd.DataFrame(${repr(kept.map((k) => only(k.at)))}, index=${repr(kept.map((k) => k.name))})`,
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    prediction = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="prediction", new_points=kept)`,
    '    for scenario, predicted in zip(kept.index, prediction["predictions"], strict=True):',
    scale.lines.length
      ? `        print(f"{name}, {scenario}:", {key: float(${scale.back("v")}) for key, v in predicted.items()})  # in its units`
      : '        print(f"{name}, {scenario}:", predicted)',
  ];
  return [
    "import pandas as pd",
    "from process_improve.experiments import analyze_experiment",
    "",
    ...scale.lines,
    ...indicatorLines(spec, models),
    `# The settings, ${words.join(", ")}, in coded units as the columns of X.`,
    `at = pd.DataFrame([${repr(only(at))}])`,
    "# Each response's prediction there, from its own model fitted to the runs that have its result, with its 95%",
    "# confidence interval for the mean (ci_low to ci_high) and prediction interval for one new run (pi_low to pi_high).",
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    prediction = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="prediction", new_points=at)`,
    scale.lines.length
      ? `    print(name, {key: float(${scale.back("v")}) for key, v in prediction["predictions"][0].items()})  # in its units`
      : '    print(name, prediction["predictions"][0])',
    ...(kept.length ? keptLines : []),
  ].join("\n");
}
