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
export const SPACE_FILLING = ["maximin_lhs", "sobol"];

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
export function analyseCode(model, complete = true, spec = null, vifModel = null, { compared = false, control = null, spread = false } = {}) {
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
    ...(spread ? spreadCode(spec) : []),
  ].join("\n");
}

/**
 * Step 3, one figure: the code that draws one of Learn's figures with the library, after the analysis's own code.
 * ``options`` is what the page asked the library for beyond the runs and their analysis (``plot_options`` of
 * bootstrap.plots): the library's name for the plot, the factors it draws, and the coded values the others are
 * held at. The figure is the library's own: each factor in coded units, under the library's titles, where the page
 * puts the axes in the reader's units and its own colours. It is drawn from the analysis the page draws it from:
 * Lenth's margins for the half-normal plot of the effects, the fit and its residuals for every other figure.
 */
export function figureCode(title, options, model, complete = true, spec = null) {
  const transform = transformOf(spec);
  const X = complete ? "X" : "X[done]";
  const y = onScale(complete ? "y" : "y[done]", transform);
  const response = repr(spec?.response ?? "y");
  const lenth = options.plot_type === "half_normal";
  return [
    "import plotly.graph_objects as go",
    "from process_improve.experiments import analyze_experiment, visualize_doe",
    "",
    `# The figure "${title}", as the library draws it: under its own titles, with each factor in`,
    "# coded units, from -1 at its low setting to +1 at its high. The page puts the axes in your own units.",
    // shortcut: the region of a study with limits is not written out, so the code draws the whole cube; #1242.
    ...(spec?.constraints?.length ? ["# The page also shades the settings your limits rule out; this figure is of every setting."] : []),
    "drawn = analyze_experiment(",
    lenth
      ? `    ${X}, ${y}, model=${repr(model)}, analysis_type=["effects", "lenth_method"],`
      : `    ${X}, ${y}, model=${repr(model)}, analysis_type=["coefficients", "residual_diagnostics"], coding="actual",`,
    ")",
    "chart = visualize_doe(",
    ...Object.entries(options).map(([name, value]) => `    ${name}=${repr(value)},`),
    "    analysis_results=drawn,",
    `    design_data=${X}.assign(**{${response}: ${y}}).to_dict("records"),`,
    `    response_column=${response},`,
    '    backend="plotly",',
    ")",
    'figure = go.Figure(chart["plotly"])',
    "figure.show()",
  ].join("\n");
}

/** The runs and results a model is fitted to, as Python writes them: those with a result, on the model's scale. */
const fittedTo = (spec, complete) =>
  complete ? `X, ${onScale("y", transformOf(spec))}` : `X[done], ${onScale("y[done]", transformOf(spec))}`;

/** A response as an axis names it: the reader's words for it, with its units. */
function axisWords(spec) {
  const info = spec?.response_info ?? {};
  const label = info.label || spec?.response || "y";
  return info.units ? `${label} (${info.units})` : label;
}

/** The lines that end a figure's code: its axes named, and the figure shown. */
const shown = (x, y) => [`figure.update_layout(xaxis_title=${repr(x)}, yaxis_title=${repr(y)})`, "figure.show()"];

/** A Scheffé model's response over the blends of the frame ``at``, from the coefficients ``of``, as Python writes it. */
const blended = (at, of) => `sum(value * ${at}[term.split(":")].prod(axis=1) for term, value in ${of}.items())`;

/** A known process as Python makes it, named ``truth``: a figure of it draws its response with no noise. */
const knownProcess = (one) => [
  "truth = TrueProcess(",
  "    {",
  ...one.truth.map((t) => `        ${repr(t.term)}: ${t.coefficient},`),
  "    },",
  `    sigma=${one.sigma},`,
  ")",
];

/**
 * The page's own figures, which the library has no plot for: each a function of the figure's title, the study's
 * specification and what the page drew it from, giving the lines that draw it with Plotly. Each builds on step 2's
 * ``runs``, ``X`` and ``y`` and on step 3's ``analysis``, and holds the numbers the page's figure holds.
 */
const OWN_FIGURES = {
  coefficients: (title, spec) => [
    "import plotly.graph_objects as go",
    "",
    `# The figure "${title}": each term's estimate with its 95% interval, from the analysis above. A fit that`,
    "# uses every run has no interval. Sorted by size, the estimates are the page's Pareto plot.",
    'terms = [c for c in analysis["coefficients"] if c["term"] != "Intercept"]',
    'reach = lambda c, end: 0 if c.get(end) is None else abs(c[end] - c["coefficient"])  # none: no interval',
    "figure = go.Figure(",
    "    go.Scatter(",
    '        x=[c["coefficient"] for c in terms],',
    '        y=[c["term"] for c in terms],',
    '        mode="markers",',
    '        name="Estimate",',
    "        error_x={",
    '            "type": "data",',
    '            "symmetric": False,',
    '            "array": [reach(c, "ci_high") for c in terms],',
    '            "arrayminus": [reach(c, "ci_low") for c in terms],',
    "        },",
    "    )",
    ")",
    'figure.update_yaxes(autorange="reversed")  # the first term at the top, as the page lists them',
    ...shown(`Coefficient: ${axisWords(spec)}${transformOf(spec) ? ", on the scale the model is fitted on" : ""}`, "Term"),
  ],
  // The coefficients' third view: the effects in the response's units, as charts.js effectsFigure draws them, with
  // the band of the smallest change worth finding (``about.change``) when the reader gave one.
  effects: (title, spec, { change = null } = {}) => [
    "import re",
    "",
    "import plotly.graph_objects as go",
    "",
    `# The figure "${title}", as the effects in the response's units: how far each term moves the response over`,
    "# the factors' ranges. A factor from its low end to its high end is two coded units, so its effect is twice",
    "# its coefficient; a square moves the response by its coefficient from the center to an end.",
    'span = lambda c: 1 if re.fullmatch(r"I\\(.+ \\*\\* 2\\)", c["term"]) else 2',
    'terms = [c for c in analysis["coefficients"] if c["term"] != "Intercept"]',
    'terms = sorted(terms, key=lambda c: -abs(span(c) * c["coefficient"]))',
    'reach = lambda c: 0 if c.get("ci_high") is None else span(c) * (c["ci_high"] - c["coefficient"])  # none: no interval',
    "figure = go.Figure(",
    "    go.Bar(",
    '        x=[span(c) * c["coefficient"] for c in terms],',
    '        y=[c["term"] for c in terms],',
    '        orientation="h",',
    '        marker_color=["#E69F00" if c["coefficient"] > 0 else "#0072B2" for c in terms],',
    '        error_x={"type": "data", "array": [reach(c) for c in terms]},',
    "    )",
    ")",
    ...(change
      ? [
          "# A change smaller than the one you said is worth finding lies inside the band.",
          `figure.add_vrect(x0=${-change}, x1=${change}, fillcolor="gray", opacity=0.18, line_dash="dot", line_width=1)`,
        ]
      : []),
    'figure.update_yaxes(autorange="reversed")  # the largest at the top',
    ...shown(`Change in ${axisWords(spec)} over the range`, "Term"),
  ],
  // The coefficients' other view. With ``about.lenth`` the fit uses every run, which leaves no interval: Lenth's
  // margins judge the sizes then, from the model (``about.model``) fitted to the runs that have their result
  // (``about.complete``: all of them).
  pareto: (title, spec, { lenth = false, model = "", complete = true } = {}) => [
    "import plotly.graph_objects as go",
    ...(lenth ? ["from process_improve.experiments import analyze_experiment"] : []),
    "",
    `# The figure "${title}", as a Pareto plot: each term's size as a bar, the largest at the top, orange for a`,
    "# positive coefficient and blue for a negative one.",
    'terms = sorted((c for c in analysis["coefficients"] if c["term"] != "Intercept"), key=lambda c: -abs(c["coefficient"]))',
    "figure = go.Figure(",
    "    go.Bar(",
    '        x=[abs(c["coefficient"]) for c in terms],',
    '        y=[c["term"] for c in terms],',
    '        orientation="h",',
    '        marker_color=["#E69F00" if c["coefficient"] > 0 else "#0072B2" for c in terms],',
    ...(lenth
      ? []
      : ['        error_x={"type": "data", "array": [0 if c.get("ci_high") is None else c["ci_high"] - c["coefficient"] for c in terms]},']),
    "    )",
    ")",
    ...(lenth
      ? [
          "# The fit uses every run, so no coefficient has an interval: Lenth's margins of error judge their sizes.",
          "# They are on the effects' scale, and an effect is twice its coefficient.",
          `lenth = analyze_experiment(${fittedTo(spec, complete)}, model=${repr(model)}, analysis_type="lenth_method")["lenth_method"]`,
          'figure.add_vline(x=lenth["ME"] / 2, line_dash="dash", annotation_text="ME")',
          'figure.add_vline(x=lenth["SME"] / 2, line_dash="dot", annotation_text="SME")',
        ]
      : []),
    'figure.update_yaxes(autorange="reversed")  # the largest at the top',
    ...shown("|coefficient|, coded units", "Term"),
  ],
  levels: (title, spec) => {
    const factor = spec.factors[0];
    return [
      "import plotly.graph_objects as go",
      "",
      `# The figure "${title}": every run at its level of ${factor.name}, and each level's average.`,
      "had = y.notna()  # the runs that have their result",
      `levels = runs[${repr(factor.name)}][had].astype(str)`,
      "averages = y[had].groupby(levels, sort=False).mean()",
      "figure = go.Figure(",
      "    [",
      '        go.Scatter(x=levels, y=y[had], mode="markers", name="Runs"),',
      "        go.Scatter(",
      "            x=averages.index,",
      "            y=averages,",
      '            mode="markers",',
      '            name="Average at each level",',
      '            marker={"symbol": "line-ew", "size": 24, "line": {"width": 3}},',
      "        ),",
      "    ]",
      ")",
      ...shown(factor.units ? `${factor.name} (${factor.units})` : factor.name, axisWords(spec)),
    ];
  },
  true_contour: (title, spec, { contour, process }) => {
    if (!contour || !process) return null;
    const [across, up] = contour.factors_to_plot;
    const held = Object.entries(contour.hold_values ?? {});
    return [
      "import numpy as np",
      "import pandas as pd",
      "import plotly.graph_objects as go",
      "from process_improve.experiments import TrueProcess",
      "",
      `# The figure "${title}": the process the results were simulated from, with no noise, over`,
      `# ${across} and ${up} in coded units, from -1 at a factor's low setting to +1 at its high.`,
      held.length ? "# Every other factor is held where the page holds it, in coded units too." : "# Every other factor is at its center, 0.",
      ...knownProcess(process),
      "side = np.linspace(-1, 1, 50)",
      "points = pd.DataFrame({name: 0.0 for name in truth.factors}, index=range(side.size**2))",
      `points[${repr(across)}], points[${repr(up)}] = np.tile(side, side.size), np.repeat(side, side.size)`,
      ...held.map(([name, value]) => `points[${repr(name)}] = ${value}`),
      "surface = np.asarray(truth.mean(points)).reshape(side.size, side.size)",
      `figure = go.Figure(go.Contour(x=side, y=side, z=surface, colorbar={"title": {"text": ${repr(axisWords(spec))}}}))`,
      ...shown(`${across} (coded)`, `${up} (coded)`),
    ];
  },
  ternary: (title, spec, { process }, values = "fitted") => {
    const names = spec.factors.map((f) => f.name);
    const [a, b, c] = names.map(repr);
    const model = values === "fitted";
    if (!model && !process) return null;
    return [
      "import numpy as np",
      "import pandas as pd",
      "import plotly.graph_objects as go",
      ...(model ? [] : ["from process_improve.experiments import TrueProcess"]),
      "",
      model
        ? `# The figure "${title}": the fitted model's response over a grid of blends, and the runs.`
        : `# The figure "${title}": the process the results were simulated from, with no noise, over a grid of blends.`,
      "# The grid covers the blends the components' lower and upper proportions allow, in 40 steps a side.",
      `low, high = np.array(${repr(spec.factors.map((f) => f.low))}), np.array(${repr(spec.factors.map((f) => f.high))})`,
      "steps = 40",
      "lattice = np.array([(i, j, steps - i - j) for i in range(steps + 1) for j in range(steps + 1 - i)]) / steps",
      `grid = pd.DataFrame(low + (1 - low.sum()) * lattice, columns=${repr(names)})`,
      "grid = grid[(grid <= high + 1e-9).all(axis=1)]",
      ...(model
        ? [
            "# A Scheffé model's response at a blend: each term's coefficient times the product of its components.",
            'fitted = {c["term"]: c["coefficient"] for c in analysis["coefficients"]}',
            `response = ${blended("grid", "fitted")}`,
          ]
        : [...knownProcess(process), "response = truth.mean(grid)"]),
      "had = y.notna()  # the runs that have their result",
      "figure = go.Figure(",
      "    [",
      "        go.Scatterternary(",
      `            a=grid[${a}], b=grid[${b}], c=grid[${c}], mode="markers", name=${repr(model ? "Fitted response" : "True response")},`,
      `            marker={"color": response, "colorscale": "Blues", "size": 7, "colorbar": {"title": {"text": ${repr(axisWords(spec))}}}},`,
      "        ),",
      "        go.Scatterternary(",
      `            a=X[${a}][had], b=X[${b}][had], c=X[${c}][had], mode="markers", name="Runs",`,
      '            marker={"color": "white", "size": 10, "line": {"color": "black", "width": 1.5}},',
      "        ),",
      "    ]",
      ")",
      `figure.update_layout(ternary={axis: {"title": {"text": name}} for axis, name in zip(("aaxis", "baxis", "caxis"), ${repr(names)})})`,
      "figure.show()",
    ];
  },
  true_ternary: (title, spec, about) => OWN_FIGURES.ternary(title, spec, about, "true"),
  cox: (title, spec, { cox }) => {
    if (!cox) return null;
    const spans = Object.fromEntries(cox.traces.map((trace) => [trace.component, [trace.change[0], trace.change.at(-1)]]));
    const best = cox.through === "best";
    return [
      "import numpy as np",
      "import pandas as pd",
      "import plotly.graph_objects as go",
      "",
      `# The figure "${title}": the fitted response as each component moves away from one blend along its`,
      "# Cox direction, on which the other components keep their proportions to each other.",
      `through = pd.Series(${repr(cox.reference)})  # ${best ? "the best blend" : "the blend the effects are taken from"}`,
      `spans = ${repr(spans)}  # how far each component moves from it, as the page draws it`,
      'fitted = {c["term"]: c["coefficient"] for c in analysis["coefficients"]}',
      "figure = go.Figure()",
      "for name, (start, stop) in spans.items():",
      best
        ? "    change = np.unique(np.append(np.linspace(start, stop, 41), 0.0))  # and the best blend itself, at 0"
        : "    change = np.linspace(start, stop, 41)",
      "    share = through[name] + change",
      "    blends = pd.DataFrame(np.outer((1 - share) / (1 - through[name]), through), columns=through.index)",
      "    blends[name] = share",
      "    # A Scheffé model's response at a blend: each term's coefficient times the product of its components.",
      `    figure.add_scatter(x=change, y=${blended("blends", "fitted")}, mode="lines", name=name)`,
      ...shown("Change in a component's proportion", axisWords(spec)),
    ];
  },
};

/**
 * Step 3, one of the page's own figures: the code that draws it with Plotly, after the analysis's own code. The
 * library draws most of Learn's figures (figureCode); these the page draws itself, from the same analysis. ``about``
 * holds what the page drew the figure from beyond the runs and their analysis: ``contour``, the two factors of the
 * response surface and the coded values the others are held at; ``cox``, the blend a mixture's traces go through
 * and how far each runs; and ``process``, the known process a response's results were simulated from, its terms and
 * its noise. Null for a figure it has no code for, or one whose figure the page could not have drawn.
 */
export function ownFigureCode(key, title, spec, about = {}) {
  return OWN_FIGURES[key]?.(title, spec, about)?.join("\n") ?? null;
}

/** A map of how far each pair of terms moves together, drawn from the frame ``correlations``: the page's colours. */
const correlationMap = [
  "figure = go.Figure(",
  "    go.Heatmap(",
  "        z=correlations.to_numpy(), x=list(correlations.columns), y=list(correlations.index),",
  '        zmin=0, zmax=1, colorscale="Blues",',
  "    )",
  ")",
  'figure.update_yaxes(autorange="reversed")  # the first term at the top',
  "figure.show()",
];

/**
 * The design's own correlation map, after step 2's ``X``: the absolute correlation between each pair of terms over
 * the runs, as the comparison reckons it. ``factors`` names the columns of ``X`` that are factors.
 */
export function correlationCode(factors) {
  return [
    "import plotly.graph_objects as go",
    "from process_improve.experiments import compare_designs",
    "",
    '# The figure "Correlation map": how far each pair of terms moves together over the runs. 0: the runs tell the',
    "# two apart in full. 1: they cannot tell them apart at all.",
    `compared = compare_designs({"runs": X[${repr(factors)}]})`,
    "correlations = list(compared.correlations.values())[0]  # of the one design compared: these runs",
    ...correlationMap,
  ].join("\n");
}

/**
 * A figure of the comparison, after its ``designs`` and ``comparison`` (compareCode): the FDS plot of every design,
 * from what the page asked each curve with (``asked``: bootstrap.compare's ``fds_asked``), over the settings the
 * limits leave when there are some; or, with ``label``, that design's correlation map.
 */
export function comparisonFigureCode({ asked = null, limits = [], label = null }) {
  if (label !== null) {
    return [
      "import plotly.graph_objects as go",
      "",
      `# The figure "Correlation map, ${label}": how far each pair of terms moves together over the design's runs.`,
      `correlations = comparison.correlations[${repr(label)}]`,
      ...correlationMap,
    ].join("\n");
  }
  return [
    "import plotly.graph_objects as go",
    "from process_improve.experiments import evaluate_design",
    "",
    '# The figure "Fraction of design space (FDS)": each design\'s prediction variance, in units of the noise',
    "# variance, sorted over settings drawn evenly from the region. A design that cannot fit the model has no curve.",
    "figure = go.Figure()",
    "for label, design in zip(comparison.designs, designs.values()):",
    '    if not comparison.values[label]["fits"]:',
    "        continue",
    "    curve = evaluate_design(",
    `        design, model=${repr(asked.model)}, metric="fds", n_samples=${asked.n_samples},`,
    `        fds_resolution=${asked.fds_resolution}, random_state=${asked.random_state},${limits.length ? " region=DesignRegion(factors, constraints)," : ""}`,
    '    )["fds"]["curve"]',
    '    figure.add_scatter(x=curve["fraction"], y=curve["prediction_variance"], name=label)',
    ...shown("Fraction of the design space", "Prediction variance, in units of the noise variance"),
  ].join("\n");
}

/**
 * "How tightly to hold each factor", after the best settings' code (bestCode), which leaves the settings as ``at``
 * and the prediction there as ``prediction``: each of ``names``, the numeric factors the model holds, moved alone
 * by 5% of its range, 0.1 in coded units, and the change in the prediction. Null for a mixture, whose components
 * do not move alone.
 */
export function sensitivityCode(names, spec, model, complete = true) {
  if (isMixture(spec) || !names.length) return null;
  const transform = transformOf(spec);
  const fitted = fittedTo(spec, complete);
  const back = (value) => (transform ? `(${SCALES[transform].back})(${value})` : value);
  return [
    "import plotly.graph_objects as go",
    "",
    '# The figure "How tightly to hold each factor": how much the prediction changes when one factor alone moves',
    "# 5% of its range from the best settings, which is 0.1 in coded units.",
    `names = ${repr(names)}`,
    "moved = pd.concat([at.assign(**{name: at[name] + step}) for name in names for step in (-0.1, 0.1)], ignore_index=True)",
    `there = analyze_experiment(${fitted}, model=${repr(model)}, analysis_type="prediction", new_points=moved)["predictions"]`,
    `here = ${back('prediction["predictions"][0]["predicted"]')}`,
    `change = [${back('p["predicted"]')} - here for p in there]  # lowered, then raised, for each factor in turn`,
    "figure = go.Figure(",
    "    [",
    '        go.Bar(name="Lowered 5% of its range", y=names, x=change[0::2], orientation="h"),',
    '        go.Bar(name="Raised 5% of its range", y=names, x=change[1::2], orientation="h"),',
    "    ]",
    ")",
    ...shown(`Change in the prediction: ${axisWords(spec)}`, "Factor"),
  ].join("\n");
}

/**
 * The overlay of the responses' acceptable ranges, after the balance's code (balanceCode), which leaves each
 * response's ``models`` entry, its ``goals`` entry and the ``balanced`` settings: the library's overlay plot over
 * the two factors the page draws (``overlay.x`` and ``overlay.y``), the others held at the balanced settings. A
 * response is bounded where the page draws its boundary: below to maximise, above to minimise, both for a target.
 * Null for a mixture, whose overlay is over its triangle.
 */
export function overlayCode(overlay, spec) {
  if (isMixture(spec) || overlay?.kind === "ternary" || !overlay) return null;
  return [
    "import plotly.graph_objects as go",
    "from process_improve.experiments import visualize_doe",
    "",
    `# The figure of the acceptable ranges over ${overlay.x} and ${overlay.y}, in coded units, the other factors at the`,
    "# balanced settings: each response's boundary, and the shaded region where every response is acceptable.",
    "# A response to maximise is bounded below, one to minimise above, and one with a target on both sides.",
    'bounds = {"maximize": ["low"], "minimize": ["high"], "target": ["low", "high"]}',
    "responses = [",
    '    {**model, "name": model["response_name"], **{end: goal[end] for end in bounds[goal["goal"]]}}',
    "    for model, goal in zip(models, goals)",
    "]",
    `pair = ${repr([overlay.x, overlay.y])}`,
    "chart = visualize_doe(",
    '    plot_type="overlay",',
    '    analysis_results={"optimization": {"responses": responses}},',
    "    factors_to_plot=pair,",
    '    hold_values={name: at for name, at in balanced["optimal_coded"].items() if name not in pair},',
    '    backend="plotly",',
    ")",
    'figure = go.Figure(chart["plotly"])',
    "figure.show()",
  ].join("\n");
}

/**
 * The chart of the trade-offs between the responses, after the balance's code (balanceCode), which leaves each
 * response's ``models`` entry and the ``balanced`` settings: the settings the page tried (``view.asked``, what
 * bootstrap.trade_offs drew them with), each response's prediction at each, those that meet the limits the reader
 * typed (``view.limits``, each response's two ends, null for one left open), and the best trade-off among them between
 * the responses on the two axes (``view.across`` and ``view.up``), each compared by its goal (``balance.responses``).
 * The points are coloured by ``view.colour`` when it is a response's column. Null for a mixture.
 */
export function tradeOffCode(view, balance, spec) {
  if (isMixture(spec)) return null;
  const numeric = spec.factors.filter((f) => !labelled(f)).map((f) => f.name);
  const limits = spec.constraints ?? [];
  const scale = scaleLines(spec, Object.keys(balance.search.formulas));
  const goalOf = (name) => balance.responses.find((r) => r.name === name);
  const short = (name) => {
    const { goal, target } = goalOf(name);
    const column = `kept[${repr(name)}]`;
    return goal === "maximize" ? `-${column}` : goal === "minimize" ? column : `(${column} - ${target}).abs()`;
  };
  const ends = Object.entries(view.limits ?? {}).flatMap(([name, [low, high]]) => [
    ...(low === null ? [] : [`(cloud[${repr(name)}] >= ${low})`]),
    ...(high === null ? [] : [`(cloud[${repr(name)}] <= ${high})`]),
  ]);
  const tint = view.colour in balance.search.formulas ? `, "color": kept[${repr(view.colour)}], "showscale": True` : "";
  const [across, up] = [repr(view.across), repr(view.up)];
  return [
    "import numpy as np",
    "import pandas as pd",
    "import plotly.graph_objects as go",
    "",
    `# The figure "What one response costs another": ${view.asked.drawn} settings drawn evenly from the factors' ranges, in`,
    "# coded units, and each response's prediction at each, on the models of the balance.",
    `names = ${repr(numeric)}`,
    `tried = pd.DataFrame(np.random.default_rng(${view.asked.seed}).uniform(-1, 1, size=(${view.asked.drawn}, ${numeric.length})), columns=names)`,
    ...(limits.length
      ? [
          "region = DesignRegion(factors, constraints)",
          "tried = tried[region.feasible(tried[list(region.names)].to_numpy())].reset_index(drop=True)  # inside the limits",
        ]
      : []),
    "",
    "",
    "def predicted(coefficients, at):",
    "    # A model's prediction at each row of at: each coefficient times its term; a square is written I(name ** 2).",
    '    part = lambda name: at[name[2:-6]] ** 2 if name.startswith("I(") else at[name]',
    '    product = lambda term: 1.0 if term == "Intercept" else np.prod([part(name) for name in term.split(":")], axis=0)',
    '    return sum(c["coefficient"] * product(c["term"]) for c in coefficients)',
    "",
    "",
    "cloud, star = pd.DataFrame(index=tried.index), {}",
    "for model in models:",
    '    name = model["response_name"]',
    `    cloud[name] = ${scale.back('predicted(model["coefficients"], tried)')}`,
    `    star[name] = ${scale.back('balanced["predicted_responses"][name]')}  # at the balanced settings`,
    ends.length ? `kept = cloud[${ends.join(" & ")}]  # the settings that meet your limits` : "kept = cloud",
    `# The best trade-off between ${view.across} and ${view.up}: the settings tried that no other betters on both.`,
    "# Each response is compared by how far it falls short of its goal: the smaller, the better.",
    `shortfall = pd.DataFrame({"across": ${short(view.across)}, "up": ${short(view.up)}}).sort_values(["across", "up"])`,
    'front = shortfall[shortfall["up"] < shortfall["up"].cummin().shift(fill_value=np.inf)].index',
    `line = kept.loc[front].sort_values(${across})  # joined from left to right, as the page draws it`,
    "figure = go.Figure(",
    "    [",
    `        go.Scatter(x=kept[${across}], y=kept[${up}], mode="markers", name="Settings tried", marker={"size": 5, "opacity": 0.5${tint}}),`,
    `        go.Scatter(x=line[${across}], y=line[${up}], mode="lines", name="Best trade-off"),`,
    `        go.Scatter(x=[star[${across}]], y=[star[${up}]], mode="markers", name="Balanced settings", marker={"symbol": "star", "size": 16}),`,
    "    ]",
    ")",
    ...shown(axisOf(goalOf(view.across)), axisOf(goalOf(view.up))),
  ].join("\n");
}

/** A response of the balance as an axis names it: its label, with its units. */
const axisOf = (r) => (r.units ? `${r.label} (${r.units})` : r.label);

/**
 * The settings that give each response its target, after step 2's ``runs`` and ``X``: each response with a target
 * fitted with its own model, the settings the page tried, those at which every response is predicted within reach of
 * its target, the one of them nearest the center of the ranges, and the least-squares step from it that meets the
 * targets where the models can. ``found`` is what bootstrap.target_settings answered: its ``search`` holds each
 * response's fitted formula, its target and its reach on its model's scale, what the page held each categorical
 * factor and block at, and how the settings tried were drawn. Null for a mixture.
 */
export function targetCode(found, spec) {
  if (isMixture(spec)) return null;
  const numeric = spec.factors.filter((f) => !labelled(f));
  const names = numeric.map((f) => f.name);
  const { held, formulas, targets, reach, drawn, seed } = found.search;
  const holds = Object.keys(held).length > 0;
  const limits = spec.constraints ?? [];
  const scale = scaleLines(spec, Object.keys(formulas));
  const levels = spec.factors.filter((f) => labelled(f) && columnsOf(f).every((name) => name in held)).map((f) => `${f.name} at ${heldLevel(f, held)}`);
  return [
    ...(holds ? ["import math"] : []),
    "import numpy as np",
    "import pandas as pd",
    "from scipy.optimize import least_squares",
    `from process_improve.experiments import ${limits.length ? "DesignRegion, " : ""}analyze_experiment`,
    "",
    ...scale.lines.filter((line) => line !== "import numpy as np"),
    ...indicatorLines(spec, Object.values(formulas).join(" + ")),
    ...(holds
      ? [
          `# ${levels.length ? `The page tried each level: these settings are with ${levels.join(" and ")}. ` : ""}The blocks, if any, are held at the average block.`,
          ...HELD_AT,
        ]
      : []),
    "# What each response should be, and how near counts as giving it: 2% of the range of its results.",
    ...(scale.lines.length ? ["# A transformed response's target and reach are on its scale, as the page searches."] : []),
    `targets = ${repr(targets)}`,
    `reach = ${repr(reach)}`,
    "# Each response with a target, fitted with its own model to the runs that have its result.",
    "models = {}",
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    fit = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="coefficients", coding="actual")`,
    `    models[name] = ${holds ? `held_at(fit["coefficients"], ${repr(held)})` : 'fit["coefficients"]'}`,
    "",
    "",
    "def predicted(coefficients, at):",
    "    # A model's prediction at the settings at: each coefficient times its term; a square is written I(name ** 2).",
    '    part = lambda name: at[name[2:-6]] ** 2 if name.startswith("I(") else at[name]',
    '    product = lambda term: 1.0 if term == "Intercept" else np.prod([part(name) for name in term.split(":")], axis=0)',
    '    return sum(c["coefficient"] * product(c["term"]) for c in coefficients)',
    "",
    "",
    "def missed(at):",
    "    # How far each response is predicted from its target at the settings at, in units of its reach.",
    "    return pd.DataFrame({name: (predicted(models[name], at) - targets[name]) / reach[name] for name in targets})",
    "",
    "",
    `# ${drawn} settings drawn evenly from the factors' ranges, in coded units: -1 is a factor's low end, +1 its high end.`,
    `names = ${repr(names)}`,
    `tried = pd.DataFrame(np.random.default_rng(${seed}).uniform(-1, 1, size=(${drawn}, ${names.length})), columns=names)`,
    ...(limits.length
      ? [
          "region = DesignRegion(factors, constraints)",
          "tried = tried[region.feasible(tried[list(region.names)].to_numpy())].reset_index(drop=True)  # inside the limits",
        ]
      : []),
    "miss = missed(tried)",
    "giving = miss.abs().max(axis=1) <= 1  # the settings that give every target",
    "print(giving.mean())  # the share of the settings tried that do",
    "# Of those, the settings nearest the center of the ranges; with none, the settings that miss by least.",
    "start = tried.loc[(tried[giving] ** 2).sum(axis=1).idxmin() if giving.any() else (miss**2).sum(axis=1).idxmin()]",
    "# From there, a least-squares step meets the targets where the models can; a small pull keeps it by that setting.",
    "one = lambda x: pd.DataFrame([x], columns=names)",
    "step = least_squares(lambda x: np.concatenate([missed(one(x)).iloc[0], 0.01 * (x - start)]), start.to_numpy(), bounds=(-1, 1))",
    limits.length
      ? "found = one(step.x) if region.feasible(one(step.x)[list(region.names)].to_numpy())[0] else one(start.to_numpy())  # inside the limits"
      : "found = one(step.x)",
    `ranges = ${repr(Object.fromEntries(numeric.map((f) => [f.name, [f.low, f.high]])))}`,
    "settings = {name: low + (found[name][0] + 1) / 2 * (high - low) for name, (low, high) in ranges.items()}",
    "print(settings)  # the settings that give the targets, in the factors' own units",
    `reached = {name: ${scale.back("predicted(models[name], found)[0]")} for name in targets}`,
    "print(reached)  # what each model predicts there",
  ].join("\n");
}

/**
 * The chart of where the settings give the targets, after the code of the settings found (targetCode), which leaves
 * each response's ``models`` entry, its target, ``predicted`` and the settings ``found``: each response's prediction
 * over the two factors the page draws (``overlay.x`` and ``overlay.y``), on the page's grid, the others at the
 * settings found, with one line where it is predicted at its target. Null without an overlay, and for a mixture.
 */
export function targetOverlayCode(overlay, spec) {
  if (isMixture(spec) || !overlay) return null;
  const scale = scaleLines(spec, overlay.responses.map((r) => r.name));
  return [
    "import plotly.graph_objects as go",
    "",
    `# The figure "Where the settings give the targets" over ${overlay.x} and ${overlay.y}, in coded units, the other factors at`,
    "# the settings found: each response's line is where its model predicts its target, so every setting on it gives",
    "# that target, and where the lines cross, the settings give every target.",
    `pair = ${repr([overlay.x, overlay.y])}`,
    `axis = np.linspace(-1, 1, ${overlay.grid.length})`,
    "across, up = np.meshgrid(axis, axis)",
    "over = pd.concat([found] * across.size, ignore_index=True)",
    "over[pair[0]], over[pair[1]] = across.ravel(), up.ravel()",
    "surfaces, figure = {}, go.Figure()",
    "for name in targets:",
    `    surfaces[name] = np.reshape(${scale.back("np.zeros(across.size) + predicted(models[name], over)")}, across.shape)`,
    `    line = {"coloring": "none", "start": ${scale.back("targets[name]")}, "end": ${scale.back("targets[name]")}, "size": 1}`,
    "    figure.add_trace(go.Contour(x=axis, y=axis, z=surfaces[name], name=name, contours=line, showscale=False, showlegend=True))",
    'star = {"symbol": "star", "size": 16}',
    'figure.add_trace(go.Scatter(x=found[pair[0]], y=found[pair[1]], mode="markers", name="Settings found", marker=star))',
    ...shown(`${overlay.x} (coded)`, `${overlay.y} (coded)`),
  ].join("\n");
}

/** A Python value for an end of a window: a number, or None for an end left open. */
const endOf = (end) => (end === null ? "None" : String(end));

/**
 * The settings that meet a specification, after step 2's ``runs`` and ``X``: each response with a window fitted with
 * its own model, with what the scatter of a new run about its prediction is worked from; the settings the page tried,
 * every other column where the what-if panel has it; and the library's ``feasible_region`` over them, for the share
 * at which every prediction is inside its window, how surely a new run is, and where it is surest. ``found`` is what
 * bootstrap.specification answered: its ``search`` holds each response's fitted formula, its window on its model's
 * scale, the settings held as the models' columns, and how the settings tried were drawn. Null for a mixture.
 */
export function windowCode(found, spec) {
  if (isMixture(spec)) return null;
  const numeric = spec.factors.filter((f) => !labelled(f));
  const names = numeric.map((f) => f.name);
  const { formulas, windows, at, drawn, seed } = found.search;
  const correlated = Boolean(found.search.correlation);
  const limits = spec.constraints ?? [];
  const scale = scaleLines(spec, Object.keys(formulas));
  const blocked = Object.keys(at).some((name) => /^Block\d+_*$/.test(name));
  const windowed = Object.entries(windows).map(([name, [low, high]]) => `${repr(name)}: (${endOf(low)}, ${endOf(high)})`);
  return [
    "import numpy as np",
    "import pandas as pd",
    `from process_improve.experiments import ${limits.length ? "DesignRegion, " : ""}analyze_experiment, feasible_region`,
    "",
    ...scale.lines.filter((line) => line !== "import numpy as np"),
    ...indicatorLines(spec, Object.values(formulas).join(" + ")),
    "# The window each response must fall in, (low, high), an end of None left open.",
    ...(scale.lines.length ? ["# A transformed response's window is on its scale, as the page judges it."] : []),
    `windows = {${windowed.join(", ")}}`,
    ...(blocked
      ? [
          "# The columns the models are fitted on: X, with the library's block contrasts in place of the blocks, a column",
          "# for each block but the last, which is -1 in every one.",
          'blocks = sorted(runs["Block"].unique(), key=str)',
          "contrasts = {",
          '    f"Block{i + 1}": np.where(runs["Block"] == block, 1.0, np.where(runs["Block"] == blocks[-1], -1.0, 0.0))',
          "    for i, block in enumerate(blocks[:-1])",
          "}",
          'columns = X.drop(columns="Block").assign(**contrasts)',
        ]
      : ["columns = X  # the columns the models are fitted on"]),
    "",
    "",
    "def term_columns(terms, at):",
    "    # Each term's column over the settings at: the product of its parts; a square is written I(name ** 2).",
    '    part = lambda name: at[name[2:-6]] ** 2 if name.startswith("I(") else at[name]',
    '    product = lambda term: np.ones(len(at)) if term == "Intercept" else np.prod([part(name) for name in term.split(":")], axis=0)',
    "    return np.column_stack([product(term) for term in terms])",
    "",
    "",
    "# Each response with a window, fitted with its own model to the runs that have its result, with what the scatter",
    "# of a new run about its prediction is worked from: the fit's error, and the inverse of X'X of its model matrix.",
    "models = []",
    ...(correlated ? ["residuals = {}  # each fit's, by its response, for how the responses' errors move together"] : []),
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    fit = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="coefficients", coding="actual")`,
    '    matrix = term_columns([c["term"] for c in fit["coefficients"]], columns[done])',
    ...(correlated ? [`    residuals[name] = ${scale.on("runs[name][done]")} - matrix @ [c["coefficient"] for c in fit["coefficients"]]`] : []),
    "    models.append(",
    "        {",
    '            "response_name": name,',
    '            "coefficients": fit["coefficients"],',
    '            "factor_names": list(columns.columns),',
    '            "mse_residual": fit["model_summary"]["mse_residual"],',
    '            "df_residual": fit["model_summary"]["df_residual"],',
    '            "unscaled_covariance": np.linalg.pinv(matrix.T @ matrix),',
    "        }",
    "    )",
    "",
    `# ${drawn} settings of the factors with a range, drawn evenly from their ranges in coded units (-1 is a factor's low`,
    "# end, +1 its high end); every other column is where What if has it.",
    `names = ${repr(names)}`,
    `at = ${repr(at)}`,
    `tried = pd.DataFrame(np.random.default_rng(${seed}).uniform(-1, 1, size=(${drawn}, ${names.length})), columns=names)`,
    ...(limits.length
      ? [
          "region = DesignRegion(factors, constraints)",
          "tried = tried[region.feasible(tried[list(region.names)].to_numpy())].reset_index(drop=True)  # inside the limits",
        ]
      : []),
    ...(correlated
      ? [
          "# The responses' errors on one run are taken as correlated as the residuals of their fits are, over the runs",
          "# every fit used: two windows are then likelier to be met together, or missed together.",
          "correlation = pd.DataFrame(residuals).dropna().corr().to_numpy()",
        ]
      : []),
    `judged = feasible_region(models, windows, tried.assign(**{name: value for name, value in at.items() if name not in names})${correlated ? ", residual_correlation=correlation" : ""})`,
    'print(judged["feasible"].mean())  # the share of the settings at which every prediction is inside its window',
    ...(found.likely
      ? [
          `likely = {chance: (judged["probability"] >= chance).mean() for chance in ${repr(Object.keys(found.likely).map(Number))}}`,
          "print(likely)  # the share at which one new run meets every window with each chance or more",
          'surest = judged["probability"].idxmax()',
          `ranges = ${repr(Object.fromEntries(numeric.map((f) => [f.name, [f.low, f.high]])))}`,
          "settings = {name: low + (tried[name][surest] + 1) / 2 * (high - low) for name, (low, high) in ranges.items()}",
          'print(settings, judged["probability"][surest])  # where a new run is surest to meet them, and how sure',
        ]
      : []),
  ].join("\n");
}

/**
 * The chart of a specification, after its code (windowCode), which leaves the ``models``, the ``windows`` and the
 * settings ``at``: the library's ``feasible_region`` over the page's grid of the two factors drawn (``overlay.x`` and
 * ``overlay.y``), the other columns where the what-if panel has them, shaded where every prediction is inside its
 * window, with a line for each chance the page draws when the chance can be given. ``correlated`` says the code
 * before left the ``correlation`` of the responses' errors, which the chance is then drawn with. Null without an overlay.
 */
export function windowOverlayCode(overlay, spec, correlated = false) {
  if (isMixture(spec) || !overlay) return null;
  const [low, high] = [Math.min(...overlay.chances), Math.max(...overlay.chances)];
  const lines = `{"coloring": "none", "start": ${low}, "end": ${high}, "size": ${high - low}, "showlabels": True}`;
  return [
    "import plotly.graph_objects as go",
    "",
    `# The figure "Where every response is inside its window" over ${overlay.x} and ${overlay.y}, in coded units, every other`,
    "# column where What if has it: shaded where every prediction is inside its window, with a line of equal chance",
    "# that one new run meets every window.",
    `pair = ${repr([overlay.x, overlay.y])}`,
    `axis = np.linspace(-1, 1, ${overlay.grid.length})`,
    "across, up = np.meshgrid(axis, axis)",
    "over = pd.DataFrame({name: np.full(across.size, value) for name, value in at.items()})",
    "over[pair[0]], over[pair[1]] = across.ravel(), up.ravel()",
    `drawn = feasible_region(models, windows, over${correlated ? ", residual_correlation=correlation" : ""})`,
    "grid = lambda column: drawn[column].to_numpy(dtype=float).reshape(across.shape)",
    "figure = go.Figure(",
    "    [",
    '        go.Heatmap(x=axis, y=axis, z=grid("feasible"), name="Every response inside its window", opacity=0.3, showscale=False),',
    ...(overlay.probability
      ? [`        go.Contour(x=axis, y=axis, z=grid("probability"), name="Chance a new run meets every window", contours=${lines}, showscale=False),`]
      : []),
    '        go.Scatter(x=[at[pair[0]]], y=[at[pair[1]]], mode="markers", name="What-if settings", marker={"symbol": "star", "size": 16}),',
    "    ]",
    ")",
    ...shown(`${overlay.x} (coded)`, `${overlay.y} (coded)`),
  ].join("\n");
}

/**
 * The curve the what-if panel draws, after step 2's ``X``: each run's curve (``curves``, a list per run of the
 * design of one value per point, null where it was not read, or null for a run not measured), reduced and modelled as the
 * page models it (``fit``: bootstrap.functional's reply, for its formula and its number of replicates), and the
 * curve predicted at the panel's settings (``at``, the columns of ``X``) with its 95% bands.
 */
export function curveCode(design, curves, fit, at) {
  const { spec } = design;
  const kept = fittedRuns(design);
  const { label, units, argument } = spec.curve;
  const used = new Set([...spec.factors.flatMap(columnsOf), ...spec.factors.flatMap((f) => indicatorsIn(fit.formula, f).map(({ name }) => name))]);
  const only = Object.fromEntries(Object.entries(at).filter(([name]) => used.has(name)));
  const along = argument.units ? `${argument.label} (${argument.units})` : argument.label;
  return [
    "import pandas as pd",
    "import plotly.graph_objects as go",
    "from process_improve.experiments import analyze_functional_experiment",
    "",
    ...indicatorLines(spec, fit.formula),
    "# Each run's curve, a row per run in the order of the runs above and a column per point along the curve.",
    `points = ${repr(argument.values)}`,
    "curves = pd.DataFrame(",
    "    [",
    ...curves.filter((_, i) => kept[i]).map((curve) => `        ${repr(curve ?? argument.values.map(() => null))},`),
    "    ],",
    "    columns=points,",
    "    dtype=float,",
    ")",
    "measured = curves.notna().any(axis=1)  # a run with no point read has no curve yet",
    "# The curves are reduced to a few scores, and each score is modelled against the factors, as a response is.",
    "fit = analyze_functional_experiment(",
    `    X[measured].reset_index(drop=True), curves[measured].reset_index(drop=True), model=${repr(fit.formula)},`,
    ")",
    "# The curve at the what-if settings, in coded units as the columns of X, with its 95% band for the whole curve.",
    `at = pd.DataFrame([${repr(only)}])`,
    `predicted = fit.predict(at, n_bootstrap=${fit.replicates.length}, random_state=0)`,
    "figure = go.Figure(",
    "    [",
    '        go.Scatter(x=points, y=predicted["simultaneous_high"].iloc[0], line_width=0, showlegend=False),',
    '        go.Scatter(x=points, y=predicted["simultaneous_low"].iloc[0], line_width=0, fill="tonexty", name="95% band"),',
    '        go.Scatter(x=points, y=predicted["curve"].iloc[0], name="Predicted"),',
    "    ]",
    ")",
    ...shown(along, units ? `${label} (${units})` : label),
  ].join("\n");
}

/** The signal-to-noise ratio each goal reads a setting's readings by, of Taguchi's three: bootstrap.SN_KINDS. */
export const SN_KINDS = { maximize: "larger", minimize: "smaller", target: "nominal" };

/** Each signal-to-noise ratio of the readings ``v`` at one setting, as Python writes it, and its name. */
const SN_FORMULAS = {
  larger: ["-10 * np.log10((1 / v**2).mean())", "larger is better"],
  smaller: ["-10 * np.log10((v**2).mean())", "smaller is better"],
  nominal: ["10 * np.log10(v.mean() ** 2 / v.var())", "nominal is best"],
};

/**
 * The results read more than once at a setting (``spread``, as the page's analysis sums them up): each setting's
 * readings summed up, with the signal-to-noise ratio of the response's goal, and the response table, each summary's
 * average at each level of each factor and its delta. A setting read once, and a summary a setting cannot give, such
 * as the ln s of readings that do not vary, are left out, as the page leaves them out.
 */
function spreadCode(spec) {
  const kind = SN_KINDS[spec?.response_info?.goal];
  const summaries = ["mean", "ln_std_dev", ...(kind ? ["sn"] : [])];
  return [
    "",
    "# The results read more than once at a setting: each setting's readings summed up, and the response table.",
    "import numpy as np",
    "",
    `factors = ${repr(spec.factors.map((f) => f.name))}`,
    'readings = runs[factors].assign(y=y).dropna(subset=["y"]).groupby(factors)["y"]',
    'settings = readings.agg(n="count", mean="mean", std_dev="std")',
    'settings["ln_std_dev"] = np.log(settings["std_dev"].where(settings["std_dev"] > 0))',
    ...(kind ? [`settings["sn"] = readings.agg(lambda v: ${SN_FORMULAS[kind][0]})  # ${SN_FORMULAS[kind][1]}`] : []),
    'settings = settings[settings["n"] > 1].replace([np.inf, -np.inf], np.nan)  # a setting read once has no spread',
    "print(settings)",
    "for factor in factors:  # each summary's average at each of the factor's levels, and its delta",
    `    table = settings.groupby(level=factor)[${repr(summaries)}].mean()`,
    '    print(table, (table.max() - table.min()).rename("delta"), sep="\\n")',
  ];
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
 * page takes the comparisons' average correlation, and scipy each one's own. The library's Bartlett's test and
 * Kruskal-Wallis test then check what the analysis of variance assumes, as the page checks it.
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
    "from process_improve.univariate import bartlett_test, kruskal_wallis_test",
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
    "# The checks of what the analysis of variance assumes: whether each level's runs spread alike, which needs two",
    "# different results or more at each level, and the analysis of variance of the results' ranks.",
    "spread = bartlett_test(*groups.values()) if all(g.nunique() > 1 for g in groups.values()) else None",
    "ranks = kruskal_wallis_test(*groups.values())",
    'print(anova, tukey, dunnett, spread, ranks, sep="\\n")',
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
 * study of several responses; ``version`` pins the release the page ran, when known. ``figures`` holds the
 * figures the page drew of one response's analysis, ``{response, codes}`` with each figure's ``title`` and the
 * ``code`` that draws it (figureCode, ownFigureCode): each is a cell after that analysis.
 */
export function notebook({ title, design, responses, sources, simulation, model, vifModel = null, compared = false, control = null, spread = false, version, url, best = null, analyses = null, balance = null, figures = null }) {
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
  const each = analyses ?? (model !== null && model !== undefined ? [{ response: names[0], model, best, vifModel, compared, control, spread }] : []);
  const fitted = fittedRuns(design);
  let step = 3;
  for (const { response, model: fittedModel, best: found, vifModel: vifTerms = null, compared: levels = false, control: base = null, spread: repeated = false } of each) {
    if (fittedModel === null || fittedModel === undefined) continue;
    const spec = forResponse(design.spec, response);
    const values = results[response] ?? [];
    const complete = values.every((v, i) => !fitted[i] || (v !== null && v !== undefined));
    const words = spec.response_info?.label || response;
    const id = names.length > 1 ? `-${names.indexOf(response) + 1}` : ""; // a cell's id is ASCII, as nbformat asks
    cells.push(
      markdownCell(`analysis-step${id}`, names.length > 1 ? `## ${step}. The analysis of ${words}` : `## ${step}. The analysis`),
      codeCell(`analysis${id}`, analyseCode(fittedModel, complete, spec, vifTerms, { compared: levels, control: base, spread: repeated })),
    );
    step += 1;
    // The figures the page drew of this response's analysis, a cell each, so each is shown under its own code.
    if (figures?.response === response) {
      figures.codes.forEach(({ title: named, code }, i) => {
        cells.push(markdownCell(`figure-step${id}-${i + 1}`, `### ${named}`), codeCell(`figure${id}-${i + 1}`, code));
      });
    }
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
 * The what-if map, for its window: after the what-if code of its response (``name``, fitted as ``formula``), the
 * response predicted over two factors (``pair``), each from its low end to its high end in ``points`` steps, with
 * the other factors where ``at`` has them, drawn with its lines labelled.
 */
export function mapCode(spec, name, formula, at, settings, pair, points) {
  const scale = scaleLines(spec, [name]);
  const [first, second] = pair.map((one) => spec.factors.find((f) => f.name === one));
  const side = (f) => `${f.low} + (side + 1) / 2 * (${f.high} - ${f.low})`;
  const title = (f) => (f.units ? `${f.name} (${f.units})` : f.name);
  return [
    "import numpy as np",
    "import plotly.graph_objects as go",
    whatIfCode(spec, { [name]: formula }, at, settings),
    "",
    `# The map: ${name} over ${pair[0]} and ${pair[1]}, each from its low end to its high end (-1 to 1 in coded`,
    "# units), with the other factors where they are set above.",
    `name, model = ${repr(name)}, ${repr(formula)}`,
    "done = runs[name].notna()",
    `side = np.linspace(-1, 1, ${points})`,
    `grid = pd.concat([at] * ${points * points}, ignore_index=True)`,
    `grid[${repr(pair[0])}] = np.tile(side, ${points})`,
    `grid[${repr(pair[1])}] = np.repeat(side, ${points})`,
    `surface = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="prediction", new_points=grid)`,
    `z = np.array([float(${scale.back('p["predicted"]')}) for p in surface["predictions"]]).reshape(${points}, ${points})`,
    `fig = go.Figure(go.Contour(x=${side(first)}, y=${side(second)}, z=z, contours={"showlabels": True}))`,
    `fig.update_layout(xaxis_title=${repr(title(first))}, yaxis_title=${repr(title(second))})`,
    "fig.show()",
  ].join("\n");
}

/**
 * What if: each response's prediction at the reader's settings, from its own model fitted to the runs that have its
 * result, with its 95% confidence interval for the mean and prediction interval for one new run. ``formulas`` holds
 * each response's model as bootstrap.predictors fitted it, by its column; ``at`` the settings as the columns of ``X``
 * (best.js columnsAt), and ``settings`` as the reader set them, for the comment. ``kept`` holds the settings kept with
 * the study that the models predict at, each its ``name`` and its columns ``at``: each response is predicted at them
 * too. ``profiled`` is the column of the response whose profiles the panel draws, with how many ``points`` each is
 * drawn through (best.js profileOf): the code then predicts it as each numeric factor alone moves across its range.
 * It builds on the design and its results.
 */
export function whatIfCode(spec, formulas, at, settings, kept = [], profiled = null) {
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
  const moved = spec.factors.filter((f) => !labelled(f)).map((f) => f.name);
  const profileLines = [
    "",
    `# The profiles the panel draws: ${profiled?.name} as each factor alone moves from its low end to its high end (-1 to 1`,
    "# in coded units), with the other factors where they are set above.",
    `name, model = ${repr(profiled?.name)}, ${repr(formulas[profiled?.name])}`,
    "done = runs[name].notna()",
    `for factor in ${repr(moved)}:`,
    `    along = pd.concat([at] * ${profiled?.points}, ignore_index=True)`,
    `    along[factor] = np.linspace(-1, 1, ${profiled?.points})`,
    `    profile = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="prediction", new_points=along)`,
    `    print(f"Profile against {factor}:", [float(${scale.back('p["predicted"]')}) for p in profile["predictions"]])`,
  ];
  const profiles = profiled && formulas[profiled.name] && moved.length ? profileLines : [];
  return [
    ...(profiles.length && !scale.lines.length ? ["import numpy as np"] : []),
    "import pandas as pd",
    "from process_improve.experiments import analyze_experiment",
    "",
    ...scale.lines,
    ...indicatorLines(spec, models),
    `# The settings, ${words.join(", ")}, in coded units as the columns of X.`,
    `at = pd.DataFrame([${repr(only(at))}])`,
    "# Each response's prediction there, from its own model fitted to the runs that have its result, with its 95%",
    "# confidence interval for the mean (ci_low to ci_high) and prediction interval for one new run (pi_low to pi_high).",
    "# The 20 runs the panel draws are the 2.5%, 7.5%, ... 97.5% points of one new run there: pi_low is the least",
    "# of them and pi_high the greatest.",
    `for name, model in ${repr(formulas)}.items():`,
    "    done = runs[name].notna()",
    `    prediction = analyze_experiment(X[done], ${scale.on("runs[name][done]")}, model=model, analysis_type="prediction", new_points=at)`,
    scale.lines.length
      ? `    print(name, {key: float(${scale.back("v")}) for key, v in prediction["predictions"][0].items()})  # in its units`
      : '    print(name, prediction["predictions"][0])',
    ...(kept.length ? keptLines : []),
    ...profiles,
  ].join("\n");
}
