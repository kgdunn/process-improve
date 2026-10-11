// "Show the Python": the process-improve code that does what the page just did, step by
// step, written from the page's state so it can be copied into a script or notebook.

const repr = (value) => (typeof value === "string" ? JSON.stringify(value) : JSON.stringify(value).replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None"));

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
 * The lines that make ``runs``, the design's coded runs as the page fits them: a categorical
 * factor's labels coded -1 and +1, and a blocked design's runs with their Block. ``result.design``
 * holds the labels themselves, which a model in coded units cannot multiply.
 */
function codedRuns(spec) {
  const blocked = (spec.options?.n_blocks ?? 1) > 1;
  const lines = [
    `names = [${spec.factors.map((f) => repr(f.name)).join(", ")}]`,
    `runs = result.design[${blocked ? 'names + ["Block"]' : "names"}].copy()`,
  ];
  for (const f of spec.factors.filter(labelled)) {
    lines.push(`runs[${repr(f.name)}] = runs[${repr(f.name)}].map({${repr(f.low)}: -1.0, ${repr(f.high)}: 1.0})  # as the page codes them`);
  }
  return lines;
}

/** Step 1: the design, as generate_design makes it. */
export function designCode(spec) {
  const options = { ...(spec.options || {}) };
  return [
    "from process_improve.experiments import Factor, generate_design",
    "",
    factorList(spec.factors),
    `result = generate_design(factors, design_type=${repr(spec.design_type)}, ${kwargs(options)})`,
    "print(result.design_actual)  # the runs, in randomised order, in the factors' own units",
  ].join("\n");
}

/** Step 2: the simulated results, from the same process the page simulated, at the design's coded runs. */
export function simulateCode(simulation, spec) {
  const terms = simulation.truth.map((t) => `    ${repr(t.term)}: ${t.coefficient},`);
  const coded = spec?.factors.some(labelled);
  return [
    "from process_improve.experiments import TrueProcess",
    "",
    ...(coded ? [...codedRuns(spec), ""] : []),
    "process = TrueProcess(",
    "    {",
    ...terms.map((line) => `    ${line}`),
    "    },",
    `    sigma=${simulation.sigma},`,
    ")",
    `y = process.simulate(${coded ? "runs" : "result"}, random_state=${simulation.seed})`,
  ].join("\n");
}

/** Step 3: the analysis, in coded units as the page fits it; a blocked design's runs keep their Block. */
export function analyseCode(spec, model) {
  return [
    "from process_improve.experiments import analyze_experiment",
    "",
    ...codedRuns(spec),
    "analysis = analyze_experiment(",
    `    runs, y, model=${repr(model)}, analysis_type=["coefficients", "anova", "lack_of_fit"]`,
    ")",
    'print(analysis["coefficients"])',
  ].join("\n");
}

/** Step 2, for results typed in or read from a workbook: the responses, in the runs' order. */
export function enteredCode(responses) {
  const values = responses.map((v) => (v === null || v === undefined ? "None" : String(v)));
  return [
    "import pandas as pd",
    "",
    "# One result per run, in the order of result.design; None for a run still to do.",
    `y = pd.Series([${values.join(", ")}], name="y")`,
  ].join("\n");
}

/**
 * Step 4: the comparison, each design generated with the options the page used for it
 * (``designs`` is a list of {label, design_type, options}).
 */
export function compareCode(factors, designs, model) {
  const lines = designs.map((d) => {
    const extra = kwargs(d.options || {});
    return `    ${repr(d.label)}: generate_design(factors, design_type=${repr(d.design_type)}${extra ? `, ${extra}` : ""}),`;
  });
  return [
    "from process_improve.experiments import Factor, compare_designs, generate_design",
    "",
    factorList(factors),
    "designs = {",
    ...lines,
    "}",
    `comparison = compare_designs(designs, model=${repr(model)})`,
    "print(comparison.table.to_string())",
  ].join("\n");
}
