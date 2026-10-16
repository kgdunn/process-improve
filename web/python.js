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

/**
 * The page's mixture designs, as bootstrap.MIXTURES names them; each is the library's design type "mixture",
 * built for the Scheffé model its options name, and its factors are the components of a blend.
 */
export const MIXTURES = ["simplex_lattice", "simplex_centroid", "extreme_vertices"];

const isMixture = (spec) => MIXTURES.includes(spec.design_type);

function factorList(factors, mixture = false) {
  const units = (f) => (f.units ? `, units=${repr(f.units)}` : "");
  const rows = factors.map((f) =>
    labelled(f)
      ? `    Factor(name=${repr(f.name)}, type="categorical", levels=[${repr(f.low)}, ${repr(f.high)}]),`
      : `    Factor(name=${repr(f.name)}${mixture ? ', type="mixture"' : ""}, low=${f.low}, high=${f.high}${units(f)}),`,
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
  if (isMixture(spec)) {
    // Each component a proportion between its bounds; the library's mixture design, for the model its options name.
    return [
      "from process_improve.experiments import Factor, generate_design",
      "",
      "# The components of the blend, each a proportion between its lower and upper bounds.",
      factorList(spec.factors, true),
      `result = generate_design(factors, design_type="mixture", ${kwargs(options)})`,
      "print(result.design_actual)  # the blends, in randomised order: each run's proportions sum to 1",
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

/** The decimals the page rounds simulated results to: two significant decimals below the noise's size. */
const decimals = (sigma) => Math.max(0, 2 - Math.floor(Math.log10(sigma)));

/** What ``y`` is, as a comment: the response in the reader's words and units ("# Yield, in %"), or nothing. */
function responseComment(spec) {
  const info = spec.response_info ?? {};
  const name = info.label || spec.response || "y";
  if (!info.label && !info.units) return "";
  return `  # ${name}${info.units ? `, in ${info.units}` : ""}`;
}

/**
 * Step 2: the run sheet as the page lists it, its settings in coded units as ``X``, and the
 * results as ``y``: simulated from ``simulation``'s process when it is given, else the results
 * on the page, None for a run still to do. The runs are written out rather than taken from
 * step 1's ``result``, because the page may list them in another order: as the book's table
 * does, or as a workbook was sorted.
 */
export function resultsCode(design, responses, simulation = null) {
  const { spec, columns, rows } = design;
  const at = (name) => columns.indexOf(name);
  const settings = (f) => rows.map((row) => (labelled(f) ? row[at(f.name)] : Number(row[at(f.name)])));
  const blocked = at("Block") >= 0 && (spec.options?.n_blocks ?? 1) > 1;
  const mixture = isMixture(spec);
  const response = spec.response || "y";
  const runs = [
    ...spec.factors.flatMap((f) => columnLines(f.name, settings(f))),
    ...(blocked ? columnLines("Block", rows.map((row) => row[at("Block")])) : []),
    ...(simulation ? [] : columnLines(response, responses)),
  ];
  const lines = [
    "import pandas as pd",
    ...(simulation ? ["from process_improve.experiments import TrueProcess"] : []),
    "",
    simulation
      ? "# The run sheet, as the page lists it: each run's settings, in the factors' own units."
      : `# The run sheet, as the page lists it: each run's settings and result${
          responses.includes(null) ? "; None for a run still to do" : ""
        }.`,
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
  const terms = simulation.truth.map((t) => `        ${repr(t.term)}: ${t.coefficient},`);
  return [
    ...lines,
    "",
    "process = TrueProcess(",
    "    {",
    ...terms,
    "    },",
    `    sigma=${simulation.sigma},`,
    ")",
    ...(responseComment(spec) ? [`#${responseComment(spec).slice(3)}, simulated from a known process`] : []),
    `y = process.simulate(X, random_state=${simulation.seed}).round(${decimals(simulation.sigma)})  # as the page rounds them`,
  ].join("\n");
}

/**
 * Step 3: the analysis of step 2's ``X`` and ``y``, with the model the page fitted. ``complete``
 * says every run has its result; when one does not, the runs still to do are left out first.
 */
export function analyseCode(model, complete = true) {
  const fitted = complete ? "X, y" : "X[done], y[done]";
  return [
    "from process_improve.experiments import analyze_experiment",
    "",
    ...(complete ? [] : ["done = y.notna()  # leave out the runs still to do, as the page does"]),
    "analysis = analyze_experiment(",
    `    ${fitted}, model=${repr(model)}, analysis_type=["coefficients", "anova", "lack_of_fit"]`,
    ")",
    'print(analysis["coefficients"])',
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
export function notebook({ title, design, responses, sources, simulation, model, version, url }) {
  const simulated = Boolean(simulation) && sources.every((s) => s === "simulated");
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
    codeCell("results", resultsCode(design, responses, simulated ? simulation : null)),
  ];
  if (model !== null && model !== undefined) {
    const complete = responses.every((v) => v !== null && v !== undefined);
    cells.push(markdownCell("analysis-step", "## 3. The analysis"), codeCell("analysis", analyseCode(model, complete)));
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
