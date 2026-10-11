// The page: builds the form, talks to the Python worker, renders results.
// All computation happens in worker.js; this file moves JSON around and draws it.
//
// The page holds one campaign at a time: a design's runs, a response per run (typed,
// simulated or read from a workbook), the process they were simulated from, and the
// analysis. Each step re-renders from that state, so the worked example drawn at first
// paint, the live run and an uploaded workbook all show the same way.

import { coefficientFigure, correlationFigure, draw, fdsFigure, prettyTerm, shareContourScale } from "./charts.js";
import {
  CAPTIONS,
  SPARE_RUNS,
  TITLES,
  compareCells,
  defaultResolution,
  escape,
  fitStatistics,
  fmt,
  formMatches,
  isBlank,
  listNames,
  nTerms,
  needsCentre,
  numberOrNull,
  pval,
} from "./logic.js";
import { analyseCode, compareCode, designCode, enteredCode, simulateCode } from "./python.js";

// Where Process Improvement using Data explains each design, method and measure the page uses.
const BOOK = "https://learnche.org/pid/design-analysis-experiments";
const BOOK_PAGES = {
  full_factorial: `${BOOK}/full-factorial-designs/using-two-levels-for-two-or-more-factors#doe-two-level-factorials`,
  fractional_factorial: `${BOOK}/fractional-factorial-designs/design-resolution#doe-design-resolution`,
  plackett_burman: `${BOOK}/fractional-factorial-designs/saturated-designs-for-screening#doe-saturated-screening-designs`,
  dsd: `${BOOK}/definitive-screening-designs#doe-definitive-screening-designs`,
  box_behnken: `${BOOK}/response-surface-methods#doe-box-behnken-designs`,
  ccd: `${BOOK}/response-surface-methods#doe-central-composite-designs`,
  d_optimal: `${BOOK}/optimal-designs#doe-optimal-designs`,
  i_optimal: `${BOOK}/optimal-designs#doe-optimal-designs`,
  lenth: `${BOOK}/full-factorial-designs/assessing-significance-of-main-effects-and-interactions#doe-lenth-method`,
  measures: `${BOOK}/omars-designs#doe-omars-metric-choice`,
  fds: `${BOOK}/judging-and-comparing-designs#doe-fds-plot`,
};

const $ = (sel) => document.querySelector(sel);
const worker = new Worker("worker.js");

// ---------------------------------------------------------------- worker RPC
let nextId = 0;
const pending = new Map();

worker.onmessage = ({ data }) => {
  if (data.status) {
    if (!pythonFailure) setStatus(data.status, data.state);
    return;
  }
  const resolve = pending.get(data.id);
  pending.delete(data.id);
  resolve(JSON.parse(data.reply));
};

// worker.js itself did not load or run: nothing will answer, so say so rather than wait.
worker.onerror = (event) => {
  event.preventDefault();
  pythonFailed(`Could not start Python: ${event.message || "worker.js did not load"}`);
};

/** Call a Python ``api_*`` function; resolves to its decoded {ok, result|error, warnings}. */
function call(fn, payload = {}) {
  if (pythonFailure) return Promise.resolve({ ok: false, error: pythonFailure, warnings: [] });
  const id = nextId++;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker.postMessage({ id, fn, payload: JSON.stringify(payload) });
  });
}

/**
 * Python could not start: say why, and stop everything that waits for it. The worked
 * example drawn at first paint stays, and every call is answered with the reason.
 */
function pythonFailed(reason) {
  if (pythonFailure) return;
  pythonFailure = reason;
  setStatus(reason, "error");
  $("#run-example").disabled = true;
  for (const resolve of pending.values()) resolve({ ok: false, error: reason, warnings: [] });
  pending.clear();
  rejectReady(new Error(reason));
}

/** Show a status message; state is "busy" (spinner), "ok" (tick) or "error". */
function setStatus(text, state = "ok") {
  const el = $("#status");
  el.querySelector(".status__text").textContent = text;
  el.dataset.state = state;
}

// ---------------------------------------------------------------- state
const state = {
  design: null, // what api_make_design returned, or the runs of an uploaded workbook
  responses: [], // one per run, in the design's row order: a number, or null while still to run
  sources: [], // one per run: "simulated", "entered" or null
  simulation: null, // what api_simulate returned: the true process, its noise and seed
  analysis: null,
  plotlyJs: null, // the Plotly.js version the Python plotly package names
};
let seed = 1;
let catalogue = null;
let pythonReady = false;
let pythonFailure = null; // why Python could not start, once it is known
let resolveReady;
let rejectReady;
const whenReady = new Promise((resolve, reject) => {
  resolveReady = resolve;
  rejectReady = reject;
});
whenReady.catch(() => {}); // a failure is shown in the status line, whoever waits for it

/** The runs as the Python side expects them back. */
const held = () => ({ spec: state.design.spec, columns: state.design.columns, rows: state.design.rows });

// ---------------------------------------------------------------- factor table
const DEFAULT_FACTORS = [
  { name: "T", low: 150, high: 200, units: "degC" },
  { name: "P", low: 1, high: 3, units: "bar" },
  { name: "F", low: 10, high: 20, units: "L/min" },
];

function addFactor(f = { name: "", low: -1, high: 1, units: "" }) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input name="name" aria-label="Factor name" size="8"></td>
    <td><input name="low" type="number" step="any" aria-label="Low"></td>
    <td><input name="high" type="number" step="any" aria-label="High"></td>
    <td><input name="units" aria-label="Units" size="6"></td>
    <td><button type="button" class="link" aria-label="Remove factor">✕</button></td>`;
  for (const key of ["name", "low", "high", "units"]) tr.querySelector(`[name=${key}]`).value = f[key] ?? "";
  tr.querySelector("button").onclick = () => {
    tr.remove();
    refreshDesign();
  };
  $("#factors tbody").append(tr);
}

const factorCount = () => document.querySelectorAll("#factors tbody tr").length;

function readFactors() {
  // A blank box is sent as null, which Python reports, rather than as Number("") = 0.
  return [...document.querySelectorAll("#factors tbody tr")].map((tr) => ({
    name: tr.querySelector("[name=name]").value.trim(),
    low: numberOrNull(tr.querySelector("[name=low]").value),
    high: numberOrNull(tr.querySelector("[name=high]").value),
    units: tr.querySelector("[name=units]").value.trim(),
  }));
}

/** Put a design specification (the worked example's, or a workbook's) into the form. */
function setForm(spec) {
  $("#factors tbody").replaceChildren();
  spec.factors.forEach(addFactor);
  $("#design-type").value = spec.design_type;
  $("#response").value = spec.response || "y";
  const options = spec.options || spec;
  if (options.n_center_points !== undefined) $("#n_center_points").value = options.n_center_points;
  if (options.alpha) $("#alpha").value = options.alpha;
  if (options.model_type) $("#model_type").value = options.model_type;
  // A resolution or a number of runs other than the page's own pick is kept, as if the reader had chosen it.
  const k = spec.factors.length;
  resolutionEdited = Boolean(options.resolution) && options.resolution !== defaultResolution(k);
  if (options.resolution) $("#resolution").value = String(options.resolution);
  budgetEdited = Boolean(options.budget) && options.budget !== nTerms(k, options.model_type) + SPARE_RUNS;
  if (options.budget) $("#budget").value = options.budget;
  refreshDesign();
}

// ---------------------------------------------------------------- design step
// The design list, its hints, which settings each design takes and how many
// factors it accepts are in the HTML, so the form is right from the first paint,
// before Python has loaded.
let budgetEdited = false;
let resolutionEdited = false;

/**
 * Bring the design settings in line with the factors: designs the factor count rules
 * out are disabled (and the hint says why if one is selected), a fraction's resolution
 * is at most the number of factors, Box-Behnken and a rotatable CCD keep a center
 * point, and an optimal design's runs follow its model unless the reader set them.
 */
function refreshDesign() {
  const k = factorCount();
  const select = $("#design-type");
  for (const option of select.options) {
    option.disabled = k < Number(option.dataset.minFactors) || k > Number(option.dataset.maxFactors);
  }
  const option = select.selectedOptions[0];
  const allowed = option.dataset.options.split(" ");
  for (const el of document.querySelectorAll("[data-option]")) el.hidden = !allowed.includes(el.dataset.option);

  // Resolution IV unless the reader chose another; three factors allow only III.
  const resolution = $("#resolution");
  for (const choice of resolution.options) choice.disabled = Number(choice.value) > k;
  if (!resolutionEdited || resolution.selectedOptions[0].disabled) resolution.value = String(defaultResolution(k));

  // A Box-Behnken design or a rotatable CCD needs a center run to fit the squares; a DSD has one built in.
  const centre = $("#n_center_points");
  const centreNeeded = needsCentre(option.value, $("#alpha").value);
  centre.min = centreNeeded ? 1 : 0;
  if (centreNeeded && Number(centre.value) < 1) centre.value = 1;

  const p = nTerms(k, $("#model_type").value);
  if (!budgetEdited) $("#budget").value = p + SPARE_RUNS;
  $("#budget-hint").textContent = `The model has ${p} coefficients.`;

  const hint = $("#design-hint");
  if (option.disabled) {
    hint.textContent = `${option.textContent} takes ${option.dataset.minFactors} to ${option.dataset.maxFactors} factors on this page; you have ${k}.`;
  } else {
    hint.replaceChildren(`${option.dataset.hint} `, link(BOOK_PAGES[option.value], "More in the book"));
  }
  hint.classList.toggle("error", option.disabled);
  $("#generate").disabled = !pythonReady || option.disabled;
  markStale(); // a factor added or removed
}

/** Generate the design the form describes; resolves to true when it is on the page. */
async function generate() {
  const button = $("#generate");
  button.disabled = true;
  setStatus("Generating design...", "busy");
  const spec = {
    design_type: $("#design-type").value,
    factors: readFactors(),
    response: $("#response").value.trim(),
    n_center_points: Number($("#n_center_points").value),
    alpha: $("#alpha").value,
    resolution: Number($("#resolution").value),
    budget: Number($("#budget").value),
    model_type: $("#model_type").value,
  };
  const reply = await call("api_make_design", spec);
  button.disabled = false;
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return false;
  }
  setStatus(`${reply.result.n_runs} runs generated.`);
  $("#demo-banner").hidden = true;
  showDesign(reply.result, reply.warnings);
  return true;
}

/** Make a new set of runs the campaign: no results yet, nothing simulated, nothing analysed. */
function showDesign(design, warnings = []) {
  state.design = design;
  state.responses = design.rows.map(() => null);
  state.sources = design.rows.map(() => null);
  state.simulation = null;
  state.analysis = null;
  renderDesign(warnings);
  renderRuns();
  renderTruth(); // the process behind the old design's results goes with them
  clearAnalysis();
  if (design.spec.options) showPython("design", designCode(design.spec));
  $("#py-run").hidden = true;
}

function renderDesign(warnings = []) {
  const r = state.design;
  const parts = [];
  if (r.facts) {
    const centre = r.facts.center_runs ? `, ${r.facts.center_runs} of them at the center of the ranges` : "";
    const outside = r.outside_range?.length
      ? ` Some runs of ${r.outside_range.join(", ")} lie outside the ranges you entered; check they are safe to run.`
      : "";
    parts.push(
      el(
        "p",
        "",
        `${r.n_runs} runs${centre}, listed in step 2 in randomised run order. The results will be analysed with ` +
          `the ${r.model.replace("_", " ")} model unless you choose another.${outside}`,
      ),
      ...noteList(r.facts.notes),
    );
  } else {
    parts.push(el("p", "", `${r.rows.length} runs, read from the workbook.`));
  }
  parts.push(...warningList(warnings));
  $("#design-summary").replaceChildren(...parts);
  $("#design-out").hidden = false;
  markStale();
}

/** Whether the factor table and response name still describe the design the page holds. */
function formMatchesDesign() {
  return formMatches(readFactors(), $("#response").value.trim(), state.design?.spec);
}

/** Say when the factors have been edited since the design was generated, until it is generated again. */
function markStale() {
  $("#design-stale").hidden = formMatchesDesign();
}

// ---------------------------------------------------------------- run step
/**
 * The run sheet, sortable by any column, with the response column as inputs: a reader
 * types each result as it comes in. Simulated values are marked until they are edited.
 */
function renderRuns() {
  const d = state.design;
  $("#run-empty").hidden = true;
  $("#run-out").hidden = false;
  const t = $("#design-table");
  const response = d.spec.response;
  const at = d.columns.indexOf(response);
  const rows = d.rows.map((row, i) => Object.assign([...row], { index: i }));
  t._cell = (row, col) => {
    if (col !== at) return null;
    const i = row.index;
    const value = state.responses[i];
    const cls = state.sources[i] === "simulated" ? "sim" : "";
    return `<input class="${cls}" type="number" step="any" inputmode="decimal" data-index="${i}" aria-label="${escape(response)}, run ${i + 1}" value="${value ?? ""}">`;
  };
  renderTable(t, d.columns, rows, "Run order");
  t.oninput = (e) => {
    const input = e.target.closest("input[data-index]");
    if (!input) return;
    const i = Number(input.dataset.index);
    state.responses[i] = input.value === "" ? null : Number(input.value);
    state.sources[i] = input.value === "" ? null : "entered";
    input.classList.remove("sim");
    describeSource();
    showPython("run", enteredCode(state.responses));
    scheduleAnalysis();
  };
  describeSource();
}

/** Say where the results in the run sheet came from. */
function describeSource() {
  const n = state.responses.length;
  const done = state.responses.filter((v) => v !== null).length;
  const simulated = state.sources.filter((s) => s === "simulated").length;
  const sim = state.simulation;
  let text = done ? `${done} of ${n} results in.` : `No results yet: simulate them, or type them into the ${state.design.spec.response} column.`;
  if (simulated && sim) {
    text =
      `${simulated} of ${n} results simulated from "${sim.label}", with noise of standard deviation ${sim.sigma} ` +
      `(seed ${sim.seed}). Simulated values are in italics; type over any of them to enter your own.`;
  }
  $("#run-source").textContent = text;
}

/** The true process behind the simulated results, behind a reveal. */
function renderTruth() {
  const sim = state.simulation;
  $("#truth").hidden = !sim;
  if (!sim) return;
  $("#truth-equation").textContent = sim.equation;
  // A factor the process leaves out is still a factor of the design: say so, so a new factor visibly counts.
  const quiet = sim.inactive ?? [];
  const note = $("#truth-inactive");
  note.hidden = !quiet.length;
  note.textContent = quiet.length
    ? `${listNames(quiet)} ${quiet.length === 1 ? "does" : "do"} not affect ${state.design.spec.response} in this process.`
    : "";
}

async function simulateResults() {
  if (!state.design) return;
  setStatus("Simulating the results...", "busy");
  const reply = await call("api_simulate", {
    ...held(),
    scenario: $("#scenario").value,
    sigma: Number($("#sigma").value),
    seed,
  });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  state.simulation = reply.result;
  state.responses = [...reply.result.responses];
  state.sources = state.responses.map(() => "simulated");
  renderRuns();
  renderTruth();
  showPython("run", simulateCode(reply.result));
  await analyseRuns();
}

function selectTab(name) {
  for (const tab of ["simulate", "real"]) {
    $(`#tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`#panel-${tab}`).hidden = tab !== name;
  }
}

async function downloadWorkbook() {
  if (!state.design) return;
  const reply = await call("api_workbook", {
    ...held(),
    responses: state.responses,
    sources: state.sources,
    simulated: state.sources.includes("simulated") ? state.simulation : null,
  });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const bytes = Uint8Array.from(atob(reply.result.xlsx_base64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.design.spec.design_type}-${state.responses.length}-runs.xlsx`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
}

async function readUpload(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  setStatus("Reading the workbook...", "busy");
  const reply = await call("api_read_workbook", { xlsx_base64: btoa(binary) });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const r = reply.result;
  $("#demo-banner").hidden = true;
  setForm(r.spec);
  state.design = { spec: r.spec, columns: r.columns, rows: r.rows, model: r.spec.model, facts: null };
  state.responses = r.responses;
  state.sources = r.sources;
  state.simulation = null;
  renderDesign(reply.warnings);
  renderRuns();
  renderTruth();
  showPython("design", designCode(r.spec));
  showPython("run", enteredCode(state.responses));
  await analyseRuns();
}

// ---------------------------------------------------------------- analyse step
let analysisTimer = null;
let ticket = 0;

/** Analyse a moment after the last keystroke, rather than on every one. */
function scheduleAnalysis() {
  clearTimeout(analysisTimer);
  analysisTimer = setTimeout(analyseRuns, 700);
}

/** Take the analysis away: its runs or their results are gone. A reply still on its way is dropped. */
function clearAnalysis() {
  ticket += 1;
  $("#analysis").hidden = true;
  $("#py-analyse").hidden = true;
  $("#charts").hidden = true;
  $("#charts").replaceChildren();
  $("#analyse-empty").hidden = false;
}

async function analyseRuns() {
  if (!state.design || !state.responses.some((v) => v !== null)) {
    clearAnalysis();
    setStatus("No results to analyse yet.");
    return;
  }
  const mine = ++ticket;
  setStatus("Analysing...", "busy");
  const request = {
    ...held(),
    responses: state.responses,
    model: $("#model").value,
    truth: state.sources.includes("simulated") ? state.simulation?.truth : null,
  };
  const reply = await call("api_analyze_runs", request);
  if (mine !== ticket) return;
  state.analysis = reply.ok ? reply.result : null;
  renderAnalysis(reply);
  if (reply.ok) await drawCharts(request, mine);
}

function renderAnalysis(reply) {
  const out = $("#analysis");
  out.hidden = false;
  $("#analyse-empty").hidden = true;
  if (!reply.ok) {
    setStatus("The results could not be analysed.", "error");
    out.replaceChildren(el("p", "error", reply.error), ...warningList(reply.warnings));
    $("#charts").hidden = true;
    $("#py-analyse").hidden = true;
    return;
  }
  const r = reply.result;
  const s = r.summary;
  const simulated = r.coefficients.some((c) => c.true_value !== undefined && c.true_value !== null);
  setStatus("Analysis complete.");
  showPython("analyse", analyseCode(r.spec, r.model));
  const parts = [
    el("p", "equation", r.equation),
    el("p", "hint", "Coefficients are in coded units: each factor runs from -1 at its low value to +1 at its high value."),
    el(
      "p",
      "",
      `${r.n_completed} completed runs. ${fitStatistics(s)}` +
        (r.pending_runs.length ? ` Still to run: ${r.pending_runs.join(", ")}.` : ""),
    ),
    ...noteList(r.notes),
    ...warningList(reply.warnings),
    el("h3", "", "Coefficients"),
    table(
      ["Term", "Coefficient", "Std. error", "95% interval", "p-value", ...(simulated ? ["True value"] : [])],
      r.coefficients.map((c) => [
        c.label,
        fmt(c.coefficient),
        fmt(c.std_error),
        c.std_error === null ? "" : `${fmt(c.ci_low)} to ${fmt(c.ci_high)}`,
        pval(c.p_value),
        ...(simulated ? [fmt(c.true_value)] : []),
      ]),
    ),
  ];
  if (simulated) {
    parts.push(
      el(
        "p",
        "hint",
        "The true value is what the coefficient estimates for the process the results were simulated from: the " +
          "estimate the same fit gives without noise.",
      ),
    );
  }
  if (r.lenth) {
    const heading = el("h3", "", "Effects, judged with ");
    heading.append(link(BOOK_PAGES.lenth, "Lenth's method"));
    parts.push(
      heading,
      el("p", "hint", `Pseudo standard error PSE = ${fmt(r.lenth.PSE)}; margins ME = ${fmt(r.lenth.ME)} and SME = ${fmt(r.lenth.SME)}.`),
      table(
        ["Term", "Effect", "Beyond ME", "Beyond SME"],
        r.lenth.effects.map((e) => [e.label, fmt(e.effect), e.active_ME ? "yes" : "no", e.active_SME ? "yes" : "no"]),
      ),
    );
  }
  if (r.anova.length) {
    parts.push(
      el("h3", "", "ANOVA"),
      table(
        ["Source", "df", "Sum of squares", "F", "p-value"],
        r.anova.map((a) => [a.source, a.df, fmt(a.sum_sq), fmt(a.F), pval(a.p_value)]),
      ),
    );
  }
  const lof = r.lack_of_fit;
  if (lof && lof.p_value !== null && lof.p_value !== undefined) {
    parts.push(
      el(
        "p",
        "",
        `Lack of fit: F = ${fmt(lof.f_statistic)}, p = ${pval(lof.p_value)} ` +
          `(pure error from the replicated runs: ${lof.df_pure_error} degree${lof.df_pure_error === 1 ? "" : "s"} of freedom).`,
      ),
    );
  }
  out.replaceChildren(...parts);
}

// ---------------------------------------------------------------- charts
function chartCard(key, caption, wide = false) {
  const card = el("figure", wide ? "chart wide" : "chart", "");
  const plot = el("div", "plot", "");
  card.append(el("h3", "", TITLES[key]), plot, el("figcaption", "hint", caption));
  return { card, plot };
}

/** Draw the coefficient chart, then the library's plots for the same analysis. */
async function drawCharts(request, mine, given = null) {
  const box = $("#charts");
  box.hidden = false;
  const simulated = Boolean(request.truth);
  const coefficients = chartCard("coefficients", CAPTIONS.coefficients(simulated), true);
  box.replaceChildren(coefficients.card);
  const figure = coefficientFigure(state.analysis.coefficients, simulated);

  const plots = given ?? (await libraryPlots(request, mine));
  if (mine !== ticket) return;
  const version = plots?.plotly_js ?? state.plotlyJs;
  if (!version) {
    coefficients.card.append(el("p", "hint", "The charts could not be drawn here; the tables above hold the numbers."));
    return;
  }
  state.plotlyJs = version;
  try {
    await draw(coefficients.plot, figure, version, figure.height);
    if (!plots) return;
    const figures = Object.fromEntries(Object.entries(plots.figures).map(([k, v]) => [k, JSON.parse(v)]));
    if (figures.true_contour) shareContourScale(figures.contour, figures.true_contour);
    for (const key of ["contour", "true_contour", "residuals", "half_normal"]) {
      if (!figures[key]) continue;
      const caption = typeof CAPTIONS[key] === "function" ? CAPTIONS[key](plots.contour_factors) : CAPTIONS[key];
      const card = chartCard(key, caption);
      box.append(card.card);
      await draw(card.plot, figures[key], version);
    }
  } catch (err) {
    box.append(el("p", "hint", `The charts could not be drawn (${err.message}); the tables above hold the numbers.`));
  }
}

async function libraryPlots(request, mine) {
  const reply = await call("api_plots", request);
  if (mine === ticket) setStatus(reply.ok ? "Analysis complete." : `Charts: ${reply.error}`, reply.ok ? "ok" : "error");
  return reply.ok ? reply.result : null;
}

// ---------------------------------------------------------------- the worked example
/** Draw the worked example computed when the page was built, while Python loads. */
async function showBuiltExample() {
  let example;
  try {
    const res = await fetch("demo.json", { cache: "no-store" });
    if (!res.ok) return;
    example = await res.json();
  } catch {
    return; // a plain local preview has no demo.json
  }
  if (state.design) return; // the reader has already started
  setForm(example.inputs.design);
  $("#scenario").value = example.inputs.scenario;
  $("#sigma").value = example.inputs.sigma;
  showScenarioHint();
  showDesign(example.design);
  state.simulation = example.simulation;
  state.responses = [...example.simulation.responses];
  state.sources = state.responses.map(() => "simulated");
  state.plotlyJs = example.plots.plotly_js;
  renderRuns();
  renderTruth();
  showPython("run", simulateCode(example.simulation));
  state.analysis = example.analysis;
  renderAnalysis({ ok: true, result: example.analysis, warnings: [] });
  $("#demo-version").textContent = `process-improve ${example.version}`;
  $("#demo-banner").hidden = false;
  if (!pythonReady && !pythonFailure) setStatus("Showing the worked example while Python starts...", "busy");
  const request = { truth: example.simulation.truth };
  await drawCharts(request, ticket, example.plots);
  const compared = example.comparison;
  if (compared && !$("#comparison").textContent) {
    $("#compare-model").value = compared.model;
    renderComparison(compared.result, compared.made, compared.notes);
    showPython(
      "compare",
      compareCode(
        example.inputs.design.factors,
        compared.made.map((m) => ({ label: m.label, design_type: m.design_type, options: m.design.spec.options })),
        compared.model,
      ),
    );
    await drawComparison(compared.result, example.inputs.design.factors.length);
  }
}

async function runExample() {
  const button = $("#run-example");
  if (!pythonReady) {
    button.disabled = true;
    button.textContent = "Waiting for Python to start...";
    const started = await whenReady.then(() => true, () => false);
    button.textContent = "Run the worked example";
    button.disabled = !started;
    if (!started) return;
  }
  const example = catalogue.demo;
  setForm(example.design);
  $("#scenario").value = example.scenario;
  $("#sigma").value = example.sigma;
  $("#model").value = example.model;
  showScenarioHint();
  seed = example.seed;
  selectTab("simulate");
  if (!(await generate())) return;
  await simulateResults();
  $("#step-analyse").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showScenarioHint() {
  $("#scenario-hint").textContent = $("#scenario").selectedOptions[0].dataset.hint;
}

// ---------------------------------------------------------------- compare step
/**
 * Generate every design the factor count allows, one call each so the page can say how
 * far it has got, then score them all for one model on the nine measures.
 */
async function compareDesigns() {
  const button = $("#compare");
  button.disabled = true;
  const factors = readFactors();
  const model = $("#compare-model").value;
  setStatus("Choosing the designs to compare...", "busy");
  const plan = await call("api_compare_plan", { factors, model, response: $("#response").value.trim() });
  if (!plan.ok) {
    setStatus(plan.error, "error");
    button.disabled = false;
    return;
  }
  const made = [];
  const notes = [...plan.result.notes];
  for (const [i, item] of plan.result.designs.entries()) {
    setStatus(`Generating ${item.label}, ${i + 1} of ${plan.result.designs.length}...`, "busy");
    const reply = await call("api_make_design", item.spec);
    if (reply.ok) made.push({ ...item, design: reply.result });
    else notes.push(`${item.label} could not be generated: ${reply.error}`);
  }
  setStatus("Scoring the designs...", "busy");
  const designs = made.map((m) => ({ label: m.label, spec: m.design.spec, columns: m.design.columns, rows: m.design.rows }));
  const reply = await call("api_compare", { model, designs });
  button.disabled = false;
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  setStatus(`${made.length} designs compared.`);
  renderComparison(reply.result, made, notes);
  showPython(
    "compare",
    compareCode(
      factors,
      made.map((m) => ({ label: m.label, design_type: m.design_type, options: m.design.spec.options })),
      model,
    ),
  );
  await drawComparison(reply.result, factors.length);
}

let comparisonFit = null; // watches whether the comparison table fits its box

function renderComparison(result, made, notes) {
  const out = $("#comparison");
  out.hidden = false;
  const t = el("table", "data compare", "");
  const head = el("tr", "", "");
  head.append(el("th", "", "Measure"), el("th", "", "Better"));
  result.designs.forEach((label, i) => {
    const th = el("th", "", label);
    const use = el("button", "link", "Use this design");
    use.type = "button";
    use.onclick = () => useDesign(made[i]);
    th.append(el("br", "", ""), use);
    head.append(th);
  });
  const thead = el("thead", "", "");
  thead.append(head);
  const tbody = el("tbody", "", "");
  let group = null;
  result.measures.forEach((m, row) => {
    // A row no design can fill, such as interaction power for a model without interactions, is left out.
    if (result.designs.every((label) => result.display[label][row] === "")) return;
    if (m.group !== group) {
      group = m.group;
      const tr = el("tr", "group", "");
      const td = el("td", "", "");
      td.append(el("span", "", group)); // a span, so the label can stay in view as the table scrolls
      td.colSpan = result.designs.length + 2;
      tr.append(td);
      tbody.append(tr);
    }
    const tr = el("tr", "", "");
    const name = el("td", "", "");
    name.append(m.group === "Feasibility" ? m.label : link(BOOK_PAGES.measures, m.label));
    name.title = m.description;
    tr.append(name, el("td", "arrow", m.direction === "higher" ? "↑" : m.direction === "lower" ? "↓" : ""));
    for (const label of result.designs) tr.append(el("td", "", result.display[label][row]));
    tbody.append(tr);
  });
  t.append(thead, tbody);
  const wrap = el("div", "scroll", "");
  wrap.append(t);
  // A phone or a narrow window shows only the first few designs, so say that the rest are there.
  const more = el("p", "hint", `Scroll the table sideways to see all ${result.designs.length} designs.`);
  more.hidden = true;
  comparisonFit?.disconnect();
  comparisonFit = new ResizeObserver(() => {
    more.hidden = wrap.scrollWidth <= wrap.clientWidth + 1;
  });
  comparisonFit.observe(wrap);
  const designNotes = result.designs.flatMap((label) => result.notes[label].map((n) => `${label}: ${prettyTerm(n)}`));
  out.replaceChildren(
    el(
      "p",
      "",
      `Model: ${result.terms.length} terms (${result.terms.map(prettyTerm).join(", ")}). Power is for a coefficient of ` +
        `${result.effect_size} noise standard deviation at a significance level of ${result.significance_level}. ` +
        `G is searched over ${result.g_points}.`,
    ),
    more,
    wrap,
    el(
      "p",
      "",
      "No design is best on every row, and which row matters depends on what the experiment is for: A/p or E " +
        "for precise coefficients, I or G for prediction over the region, max |r| for keeping second-order " +
        "effects apart, and power for detecting an effect of a stated size. Hover over a measure for its " +
        "definition, or follow its link to the book.",
    ),
    ...noteList([...notes, ...designNotes]),
  );
}

/** Load a design from the comparison into step 1 and the run sheet, as if it had been generated there. */
function useDesign(item) {
  const simulated = state.sources.includes("simulated");
  setForm(item.design.spec);
  $("#demo-banner").hidden = true;
  showDesign(item.design);
  setStatus(`${item.design.n_runs} runs of the ${item.label} design are in step 2.`);
  if (simulated) simulateResults(); // the same process, noise and seed, on the new runs
  $("#step-design").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function drawComparison(result, k) {
  const box = $("#compare-charts");
  box.hidden = false;
  box.replaceChildren();
  const version = state.plotlyJs;
  if (!version) return;
  const fds = el("figure", "chart wide", "");
  const fdsPlot = el("div", "plot", "");
  const caption = el(
    "figcaption",
    "hint",
    "Each curve sorts a design's prediction variance, in units of the noise variance, over points spread " +
      "evenly through the cube: a low, flat curve predicts with the same precision everywhere, and its right " +
      "end is the worst case, G. Designs that cannot fit the model have no curve. ",
  );
  caption.append(link(BOOK_PAGES.fds, "Reading an FDS plot"));
  fds.append(el("h3", "", "Fraction of design space (FDS)"), fdsPlot, caption);
  box.append(fds);
  try {
    await draw(fdsPlot, fdsFigure(result.fds), version, 380);
    for (const label of result.designs) {
      const { terms, matrix } = result.correlations[label];
      const card = el("figure", "chart", "");
      const plot = el("div", "plot", "");
      card.append(el("h3", "", `Correlation map, ${label}`), plot);
      box.append(card);
      await draw(plot, correlationFigure(terms, matrix, k), version, 320);
    }
    box.append(
      el(
        "p",
        "hint wide",
        "The correlation maps show |r| between the main effects, the squares and the two-factor interactions, " +
          "in three blocks; a darker square is a pair of terms the design finds harder to tell apart, and a blank " +
          "row is a term that does not vary in the design.",
      ),
    );
  } catch (err) {
    box.append(el("p", "hint", `The charts could not be drawn (${err.message}); the table above holds the numbers.`));
  }
}

// ---------------------------------------------------------------- show the Python
function showPython(key, code) {
  const panel = $(`#py-${key}`);
  panel.querySelector("code").textContent = code;
  panel.hidden = false;
}

/** The Plotly.js version the build drew its figures with, so charts can be drawn before Python names one. */
async function readManifest() {
  try {
    const res = await fetch("wheels/manifest.json", { cache: "no-store" });
    if (res.ok) state.plotlyJs ??= (await res.json()).plotly_js ?? null;
  } catch {
    /* a plain local preview has no manifest */
  }
}

// ---------------------------------------------------------------- rendering helpers
/** A list of sentences about a result (what the design is, what it could not estimate), or nothing. */
function noteList(notes, cls = "facts") {
  if (!notes || !notes.length) return [];
  const list = el("ul", cls, "");
  list.append(...notes.map((note) => el("li", "", note)));
  return [list];
}

/** The library's warnings for a call, set apart from the result, or nothing. */
const warningList = (warnings) => noteList(warnings, "warnings");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  node.textContent = text;
  return node;
}

/** A link to the book, opening in a new tab so the page keeps its state. */
function link(href, text) {
  const a = el("a", "", text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener";
  return a;
}

function table(columns, rows) {
  const wrap = el("div", "scroll", "");
  const t = el("table", "data", "");
  renderTable(t, columns, rows);
  wrap.append(t);
  return wrap;
}

/**
 * Render a table whose column headers sort it: click once for ascending,
 * again for descending. Each header is a <button> inside the <th>, so it is
 * reachable by keyboard, and aria-sort tells a screen reader the current order.
 * `sortedBy` names the column the rows already arrive sorted on, if any; a table
 * may give `t._cell(row, col)` to draw a cell itself (the run sheet's inputs).
 */
function renderTable(t, columns, rows, sortedBy = null) {
  const head = columns
    .map((c, i) => `<th aria-sort="${c === sortedBy ? "ascending" : "none"}"><button type="button" class="sort" data-col="${i}">${escape(c)}</button></th>`)
    .join("");
  t.innerHTML = `<thead><tr>${head}</tr></thead><tbody></tbody>`;
  t._rows = rows;
  fillBody(t, rows);
  t.onclick = (e) => {
    const button = e.target.closest("button.sort");
    if (!button) return;
    const th = button.parentElement;
    const ascending = th.getAttribute("aria-sort") !== "ascending";
    for (const other of t.querySelectorAll("th")) other.setAttribute("aria-sort", "none");
    th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    const col = Number(button.dataset.col);
    const sign = ascending ? 1 : -1;
    const key = (row) => (t._cell && t._cell(row, col) !== null ? state.responses[row.index] : row[col]);
    // A stable sort over a copy, so the order the rows arrived in breaks ties.
    fillBody(t, [...t._rows].sort((a, b) => isBlank(key(a)) - isBlank(key(b)) || sign * compareCells(key(a), key(b))));
  };
}

function fillBody(t, rows) {
  const cell = (row, v, c) => {
    const own = t._cell ? t._cell(row, c) : null;
    return own ?? escape(typeof v === "number" ? +v.toPrecision(6) : v ?? "");
  };
  t.tBodies[0].innerHTML = rows.map((row) => `<tr>${row.map((v, c) => `<td>${cell(row, v, c)}</td>`).join("")}</tr>`).join("");
}

// ---------------------------------------------------------------- wiring
DEFAULT_FACTORS.forEach(addFactor);
$("#add-factor").onclick = () => {
  addFactor();
  refreshDesign();
};
$("#generate").onclick = async () => {
  // Results simulated for the old design are simulated again for the new one, from the same process, noise and
  // seed, so the true process and the analysis follow the factors as they now are.
  const simulated = state.sources.includes("simulated");
  if ((await generate()) && simulated) await simulateResults();
};
$("#factors").addEventListener("input", markStale);
$("#response").addEventListener("input", markStale);
for (const id of ["#design-type", "#alpha", "#model_type"]) $(id).onchange = refreshDesign;
$("#budget").oninput = () => {
  budgetEdited = $("#budget").value !== "";
};
$("#resolution").onchange = () => {
  resolutionEdited = true;
};
$("#run-example").onclick = runExample;
$("#simulate").onclick = simulateResults;
$("#reroll").onclick = () => {
  seed += 1;
  simulateResults();
};
$("#scenario").onchange = () => {
  showScenarioHint();
  if (state.sources.includes("simulated")) simulateResults();
};
$("#sigma").onchange = () => {
  if (state.sources.includes("simulated")) simulateResults();
};
$("#tab-simulate").onclick = () => selectTab("simulate");
$("#tab-real").onclick = () => selectTab("real");
$("#download").onclick = downloadWorkbook;
$("#model").onchange = analyseRuns;
$("#compare").onclick = compareDesigns;
for (const button of document.querySelectorAll("details.python .copy")) {
  button.onclick = async () => {
    await navigator.clipboard.writeText(button.parentElement.querySelector("code").textContent);
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = "Copy the code"), 1500);
  };
}
$("#upload").onchange = (e) => {
  const file = e.target.files[0];
  e.target.value = ""; // so choosing the same file again reads it again
  if (file) readUpload(file);
};
const drop = $("#drop");
drop.ondragover = (e) => {
  e.preventDefault();
  drop.classList.add("over");
};
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) readUpload(e.dataTransfer.files[0]);
};
showScenarioHint();
refreshDesign();
readManifest();
showBuiltExample();

call("api_catalogue").then((reply) => {
  if (!reply.ok) {
    pythonFailed(reply.error);
    return;
  }
  catalogue = reply.result;
  pythonReady = true;
  for (const id of ["#upload", "#simulate", "#reroll", "#compare"]) $(id).disabled = false;
  refreshDesign();
  resolveReady();
});
