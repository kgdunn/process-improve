// "Show the Python": the process-improve code that does what the page just did, step by
// step, written from the page's state so it can be copied into a script or notebook.

const repr = (value) => (typeof value === "string" ? JSON.stringify(value) : JSON.stringify(value).replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None"));

/** Python keyword arguments from an object, in the order given, skipping undefined values. */
const kwargs = (options) =>
  Object.entries(options)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${repr(v)}`)
    .join(", ");

function factorList(factors) {
  const rows = factors.map((f) => `    Factor(name=${repr(f.name)}, low=${f.low}, high=${f.high}${f.units ? `, units=${repr(f.units)}` : ""}),`);
  return ["factors = [", ...rows, "]"].join("\n");
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

/** Step 2: the simulated results, from the same process the page simulated. */
export function simulateCode(simulation) {
  const terms = simulation.truth.map((t) => `    ${repr(t.term)}: ${t.coefficient},`);
  return [
    "from process_improve.experiments import TrueProcess",
    "",
    "process = TrueProcess(",
    "    {",
    ...terms.map((line) => `    ${line}`),
    "    },",
    `    sigma=${simulation.sigma},`,
    ")",
    `y = process.simulate(result, random_state=${simulation.seed})`,
  ].join("\n");
}

/** Step 3: the analysis, in coded units as the page fits it. */
export function analyseCode(spec, model) {
  const names = spec.factors.map((f) => repr(f.name)).join(", ");
  return [
    "from process_improve.experiments import analyze_experiment",
    "",
    `names = [${names}]`,
    "analysis = analyze_experiment(",
    `    result.design[names], y, model=${repr(model)}, analysis_type=["coefficients", "anova", "lack_of_fit"]`,
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
