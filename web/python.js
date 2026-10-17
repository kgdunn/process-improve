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

/** Whether a factor is categorical: its levels are labels, which the page holds as its low and high. */
const labelled = (f) => typeof f.low === "string";

function factorList(factors) {
  const rows = factors.map((f) =>
    labelled(f)
      ? `    Factor(name=${repr(f.name)}, type="categorical", levels=[${repr(f.low)}, ${repr(f.high)}]),`
      : `    Factor(name=${repr(f.name)}, low=${f.low}, high=${f.high}${f.units ? `, units=${repr(f.units)}` : ""}),`,
  );
  return ["factors = [", ...rows, "]"].join("\n");
}

/**
 * The limits on the settings as Python takes them, one to a line: each limit's text, in the
 * factors' own units. A limit's note is the reader's, and stays on the page.
 */
function constraintLines(limits) {
  if (!limits?.length) return [];
  return ["constraints = [  # the settings that can be run, in the factors' own units", ...limits.map((l) => `    ${repr(l.expression)},`), "]"];
}

/** Step 1: the design, as generate_design makes it, inside the limits on the settings when there are some. */
export function designCode(spec) {
  const options = { ...(spec.options || {}) };
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
  return [
    "from process_improve.experiments import Factor, generate_design",
    "",
    factorList(spec.factors),
    ...constraintLines(limits),
    `result = generate_design(factors, design_type=${repr(spec.design_type)}, ${limits.length ? "constraints=constraints, " : ""}${kwargs(options)})`,
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
 * half-range for a number, and -1 and +1 for a category's two labels. The centre and half-range
 * are worked out as bootstrap.py works them out, so the coded values agree to the last digit.
 */
function codedLine(f) {
  const column = `runs[${repr(f.name)}]`;
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
    "# The settings in coded units, as the model is fitted: -1 at a factor's low level, +1 at its high.",
    "X = pd.DataFrame(",
    "    {",
    ...spec.factors.map(codedLine),
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
 * Step 3: the analysis of step 2's ``X`` and ``y``, with the model the page fitted. ``complete``
 * says every run has its result; when one does not, the runs still to do are left out first.
 * ``spec`` names a response other than the study's first when it is that response's analysis: ``y``
 * is its column then.
 */
export function analyseCode(model, complete = true, spec = null) {
  const fitted = complete ? "X, y" : "X[done], y[done]";
  const own = spec && spec.response !== responseNames(spec)[0];
  return [
    "from process_improve.experiments import analyze_experiment",
    "",
    ...(own ? [`y = runs[${repr(spec.response)}]${responseComment(spec)}`] : []),
    ...(complete ? [] : ["done = y.notna()  # leave out the runs still to do, as the page does"]),
    "analysis = analyze_experiment(",
    `    ${fitted}, model=${repr(model)}, analysis_type=["coefficients", "anova", "lack_of_fit"]`,
    ")",
    'print(analysis["coefficients"])',
  ].join("\n");
}

/** How the code shown holds some columns of the model at a value: as bootstrap._held_at does, line for line. */
const HELD_AT = [
  "def held_at(coefficients, held):",
  '    """Return the model with some columns held at a value: a block contrast at 0, a categorical factor at -1 or +1."""',
  "    model = {}",
  "    for c in coefficients:",
  '        parts = c["term"].split(":")',
  '        term = ":".join(p for p in parts if p not in held) or "Intercept"',
  '        model[term] = model.get(term, 0.0) + c["coefficient"] * math.prod(held.get(p, 1.0) for p in parts)',
  '    return [{"term": term, "coefficient": value} for term, value in model.items()]',
  "",
  "",
];

/** What the code shown says of the held columns: each categorical factor at the level found best, the blocks at their average. */
function heldComment(held, factors) {
  const levels = factors
    .filter((f) => labelled(f) && f.name in held)
    .map((f) => `${f.name} at ${held[f.name] < 0 ? f.low : f.high}`);
  const blocks = Object.keys(held).some((name) => !factors.some((f) => f.name === name));
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
  const numeric = spec.factors.filter((f) => !labelled(f));
  const held = best.search?.held ?? {};
  const holds = Object.keys(held).length > 0;
  const limits = spec.constraints ?? [];
  const response = spec.response || "y";
  const searched = Boolean(best.search?.goal);
  const fitted = complete ? "X, y" : "X[done], y[done]";
  const levels = Object.fromEntries(spec.factors.filter(labelled).map((f) => [f.name, best.coded[f.name]]));
  const imports = [
    ...(holds && searched ? ["import math"] : []),
    "import pandas as pd",
    `from process_improve.experiments import ${limits.length && searched ? "DesignRegion, " : ""}analyze_experiment${searched ? ", optimize_responses" : ""}`,
    "",
  ];
  const at = searched
    ? `at = pd.DataFrame([{**best["optimal_coded"]${Object.keys(levels).length ? `, **${repr(levels)}` : ""}}])`
    : `at = pd.DataFrame([${repr(best.coded)}])`;
  const prediction = [
    "# The prediction there, with its 95% confidence interval for the mean and prediction interval for one new run.",
    at,
    `prediction = analyze_experiment(${fitted}, model=${repr(model)}, analysis_type="prediction", new_points=at)`,
    'print(prediction["predictions"][0])',
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
  const lines = designs.map((d) => {
    const extra = kwargs(d.options || {});
    return `    ${repr(d.label)}: generate_design(factors, design_type=${repr(d.design_type)}${kept}${extra ? `, ${extra}` : ""}),`;
  });
  return [
    `from process_improve.experiments import ${limits.length ? "DesignRegion, " : ""}Factor, compare_designs, generate_design`,
    "",
    factorList(factors),
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
 * when there is no analysis yet; ``version`` pins the release the page ran, when known.
 */
export function notebook({ title, design, responses, sources, simulation, model, version, url, best = null, analyses = null }) {
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
  const each = analyses ?? (model !== null && model !== undefined ? [{ response: names[0], model, best }] : []);
  const fitted = fittedRuns(design);
  let step = 3;
  for (const { response, model: fittedModel, best: found } of each) {
    if (fittedModel === null || fittedModel === undefined) continue;
    const spec = forResponse(design.spec, response);
    const values = results[response] ?? [];
    const complete = values.every((v, i) => !fitted[i] || (v !== null && v !== undefined));
    const words = spec.response_info?.label || response;
    const id = names.length > 1 ? `-${names.indexOf(response) + 1}` : ""; // a cell's id is ASCII, as nbformat asks
    cells.push(
      markdownCell(`analysis-step${id}`, names.length > 1 ? `## ${step}. The analysis of ${words}` : `## ${step}. The analysis`),
      codeCell(`analysis${id}`, analyseCode(fittedModel, complete, spec)),
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
