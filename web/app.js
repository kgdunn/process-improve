// The page: builds the form, talks to the Python worker, renders results.
// All computation happens in worker.js; this file moves JSON around and draws it.
//
// The page holds one campaign at a time: a design's runs, a response per run (typed,
// simulated or read from a workbook), the process they were simulated from, and the
// analysis. Each step re-renders from that state, so the worked example drawn at first
// paint, the live run, an uploaded workbook, a saved design and a shared link all show the
// same way. Every campaign but the worked example is saved in this browser as it changes.

import { coefficientFigure, correlationFigure, draw, fdsFigure, prettyTerm, shareContourScale } from "./charts.js";
import {
  AXIAL_DISTANCE,
  CAPTIONS,
  CATEGORICAL_DESIGNS,
  CHOSEN_TERMS,
  SPARE_RUNS,
  TITLES,
  axialDistance,
  candidateTerms,
  compareCells,
  defaultResolution,
  escape,
  fitStatistics,
  fmt,
  formMatches,
  isBlank,
  isCategorical,
  levelValue,
  listNames,
  nTerms,
  namedModelTerms,
  needsCentre,
  pval,
} from "./logic.js";
import {
  CAMPAIGN_VERSION,
  LINK_KEY,
  campaignTitle,
  checkCampaign,
  packCampaign,
  progress,
  shareable,
  timeAgo,
  unpackCampaign,
} from "./campaign.js";
import { analyseCode, compareCode, designCode, enteredCode, notebook, simulateCode } from "./python.js";
import { qrCode, qrSvg } from "./qr.js";
import { forgetCampaign, saveCampaign, savedCampaigns } from "./store.js";

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
const worker = new Worker("worker.js", { type: "module" });

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
  campaign: null, // this browser's record of the campaign on the page, {id, created, title}; null for the worked example
  version: null, // the process-improve release Python runs, once it has started
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
    <td><input name="low" aria-label="Low"></td>
    <td><input name="high" aria-label="High"></td>
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
  // A blank box is sent as null, which Python reports, rather than as Number("") = 0; a label, as text.
  return [...document.querySelectorAll("#factors tbody tr")].map((tr) => ({
    name: tr.querySelector("[name=name]").value.trim(),
    low: levelValue(tr.querySelector("[name=low]").value),
    high: levelValue(tr.querySelector("[name=high]").value),
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
  if (typeof options.alpha === "number") {
    $("#alpha").value = AXIAL_DISTANCE;
    $("#alpha_value").value = options.alpha;
  } else if (options.alpha) {
    $("#alpha").value = options.alpha;
  }
  // A design made without replicates or blocks is made once, in one block: counts left from an earlier design
  // do not carry over.
  $("#n_replicates").value = options.n_replicates ?? 1;
  $("#n_blocks").value = options.n_blocks ?? 1;
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
  const labelled = readFactors().filter(isCategorical).map((f) => f.name);
  const numericOnly = (option) => labelled.length > 0 && !CATEGORICAL_DESIGNS.includes(option.value);
  for (const option of select.options) {
    option.disabled = k < Number(option.dataset.minFactors) || k > Number(option.dataset.maxFactors) || numericOnly(option);
  }
  const option = select.selectedOptions[0];
  const allowed = option.dataset.options.split(" ");
  for (const el of document.querySelectorAll("[data-option]")) el.hidden = !allowed.includes(el.dataset.option);
  const alpha = axialDistance($("#alpha").value, $("#alpha_value").value);
  $("#alpha-distance").hidden = !allowed.includes("alpha") || $("#alpha").value !== AXIAL_DISTANCE;

  // Resolution IV unless the reader chose another; three factors allow only III.
  const resolution = $("#resolution");
  for (const choice of resolution.options) choice.disabled = Number(choice.value) > k;
  if (!resolutionEdited || resolution.selectedOptions[0].disabled) resolution.value = String(defaultResolution(k));

  // A Box-Behnken design or a rotatable CCD needs a center run to fit the squares; a DSD has one built in.
  const centre = $("#n_center_points");
  const centreNeeded = needsCentre(option.value, alpha, k);
  centre.min = centreNeeded ? 1 : 0;
  if (centreNeeded && Number(centre.value) < 1) centre.value = 1;

  const p = nTerms(k, $("#model_type").value);
  if (!budgetEdited) $("#budget").value = p + SPARE_RUNS;
  $("#budget-hint").textContent = `The model has ${p} coefficients.`;

  const hint = $("#design-hint");
  if (option.disabled && numericOnly(option)) {
    hint.textContent =
      `${option.textContent} needs numeric factors, and ${listNames(labelled)} ${labelled.length === 1 ? "is" : "are"} ` +
      "categorical: choose a full factorial, fractional factorial or Plackett-Burman design.";
  } else if (option.disabled) {
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
    alpha: axialDistance($("#alpha").value, $("#alpha_value").value),
    n_replicates: Number($("#n_replicates").value),
    n_blocks: Number($("#n_blocks").value),
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

/**
 * Make a new set of runs the campaign: no results yet, nothing simulated, nothing analysed. It is
 * saved in this browser as a campaign of its own, unless it follows a design made on this page
 * that holds no result the reader typed in: trying one design after another while simulating
 * keeps one entry in Your designs, not one per try. A design opened from the list or from a
 * link is never replaced. The worked example passes ``save: false``.
 */
function showDesign(design, warnings = [], { save = true } = {}) {
  const replaces = state.campaign?.replaceable && !state.sources.includes("entered");
  state.campaign = save ? { ...(replaces ? state.campaign : newCampaign()), title: titleOf(design.spec) } : null;
  state.design = design;
  $("#share-out").hidden = true;
  state.responses = design.rows.map(() => null);
  state.sources = design.rows.map(() => null);
  state.simulation = null;
  state.analysis = null;
  renderDesign(warnings);
  renderRuns();
  renderTruth(); // the process behind the old design's results goes with them
  if ($("#model").value === CHOSEN_TERMS) setModel(CHOSEN_TERMS); // the terms of the new factors
  clearAnalysis();
  if (design.spec.options) showPython("design", designCode(design.spec));
  $("#py-run").hidden = true;
  scheduleSave();
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
    renderTruth();
    showPython("run", enteredCode(state.responses));
    scheduleAnalysis();
    scheduleSave();
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
  const sim = state.sources.includes("simulated") ? state.simulation : null; // gone with the last simulated result
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
  showPython("run", simulateCode(reply.result, state.design.spec));
  scheduleSave();
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
    model: modelChoice(), // kept in the workbook, so it is analysed the same way when it comes back
  });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const bytes = Uint8Array.from(atob(reply.result.xlsx_base64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  download(blob, `${state.design.spec.design_type}-${state.responses.length}-runs.xlsx`);
}

/** Hand the reader a file, as a download. */
function download(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
}

/**
 * Download the campaign as a Jupyter notebook: the code each step's "Show the Python" panel
 * shows, in order, so it runs to the page's design, results and analysis.
 */
function downloadNotebook() {
  if (!state.design) return;
  const book = notebook({
    title: state.campaign?.title ?? titleOf(state.design.spec),
    spec: state.design.spec,
    responses: state.responses,
    sources: state.sources,
    simulation: state.simulation,
    model: state.analysis ? (state.analysis.formula ?? state.analysis.model) : null,
    version: state.version,
    url: `${location.origin}${location.pathname}`,
  });
  const blob = new Blob([`${JSON.stringify(book, null, 1)}\n`], { type: "application/x-ipynb+json" });
  download(blob, `${state.design.spec.design_type}-${state.responses.length}-runs.ipynb`);
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
  $("#campaign-banner").hidden = true;
  setForm(r.spec);
  state.design = { spec: r.spec, columns: r.columns, rows: r.rows, model: r.spec.model, facts: null };
  state.responses = r.responses;
  state.sources = r.sources;
  state.simulation = null;
  state.campaign = { ...newCampaign(false), title: `${titleOf(r.spec)}, from ${file.name}`.slice(0, 200) };
  setModel(r.spec.analysis_model ?? $("#model").value);
  renderDesign(reply.warnings);
  renderRuns();
  renderTruth();
  showPython("design", designCode(r.spec));
  showPython("run", enteredCode(state.responses));
  scheduleSave();
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
    model: modelChoice(),
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
  showPython("analyse", analyseCode(r.spec, r.formula ?? r.model)); // the terms fitted, which the design can estimate
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
      el(
        "p",
        "hint",
        "Each effect is twice its coefficient: the change from the factor's low level to its high level. " +
          `Pseudo standard error PSE = ${fmt(r.lenth.PSE)}; margins ME = ${fmt(r.lenth.ME)} and SME = ${fmt(r.lenth.SME)}.`,
      ),
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

// ---------------------------------------------------------------- the model's terms
let modelBefore = ""; // the model chosen before the reader asked to tick terms: where the list starts

/** The model the reader asks for: "" for the model the design was made for, a named model, or the ticked terms. */
function modelChoice() {
  return $("#model").value === CHOSEN_TERMS ? { terms: tickedTerms() } : $("#model").value;
}

const tickedTerms = () => [...document.querySelectorAll("#term-list input:checked")].map((box) => JSON.parse(box.value));

/** List every term the design's factors allow, grouped, with `ticked` ticked. */
function renderTermList(ticked) {
  const chosen = new Set(ticked.map((term) => JSON.stringify(term)));
  const groups = new Map();
  for (const { term, label, group } of candidateTerms(state.design.spec.factors)) {
    const box = el("input", "", "");
    box.type = "checkbox";
    box.value = JSON.stringify(term);
    box.checked = chosen.has(box.value);
    const item = el("label", "", "");
    item.append(box, ` ${label}`);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(item);
  }
  $("#term-list").replaceChildren(
    ...[...groups].flatMap(([group, items]) => {
      const row = el("div", "group", "");
      row.append(...items);
      return [el("h4", "", group), row];
    }),
  );
}

/** Start the list of terms from the model chosen before, or the one the design was made for. */
function startTermList() {
  renderTermList(namedModelTerms(modelBefore || state.design.spec.model, state.design.spec.factors));
}

/** Put a model into step 3: a named one, "" for the design's own, or {terms} to tick. */
function setModel(model) {
  if (model && typeof model === "object") {
    $("#model").value = CHOSEN_TERMS;
    renderTermList(model.terms);
  } else {
    $("#model").value = model;
    if (model === CHOSEN_TERMS) startTermList();
  }
  $("#terms").hidden = $("#model").value !== CHOSEN_TERMS;
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
  showDesign(example.design, [], { save: false });
  state.simulation = example.simulation;
  state.responses = [...example.simulation.responses];
  state.sources = state.responses.map(() => "simulated");
  state.plotlyJs = example.plots.plotly_js;
  renderRuns();
  renderTruth();
  showPython("run", simulateCode(example.simulation, example.design.spec));
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
  setModel(example.model);
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
  if (!made.length) {
    // Nothing to score: say why each design could not be made, in place of the last comparison.
    button.disabled = false;
    setStatus("None of the designs could be generated, so there is nothing to compare.", "error");
    $("#comparison").hidden = false;
    $("#comparison").replaceChildren(...noteList(notes));
    $("#compare-charts").hidden = true;
    $("#py-compare").hidden = true;
    return;
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

// ---------------------------------------------------------------- saved designs and links
/**
 * A new campaign's record in this browser: an id of its own, when it began, and whether the next
 * design made on the page may take its place (see showDesign).
 */
function newCampaign(replaceable = true) {
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return { id, created: Date.now(), replaceable };
}

/** A design's name in the list of saved designs: its type as step 1 names it, and its factors. */
function titleOf(spec) {
  const option = [...$("#design-type").options].find((o) => o.value === spec.design_type);
  return campaignTitle(option ? option.textContent.trim() : spec.design_type, spec.factors.map((f) => f.name));
}

/** The campaign on the page, as campaign.js describes one. */
function currentCampaign() {
  const { spec, columns, rows, model, facts, outside_range: outside } = state.design;
  return {
    v: CAMPAIGN_VERSION,
    title: state.campaign?.title ?? titleOf(spec),
    design: { spec, columns, rows, model, n_runs: rows.length, facts: facts ?? null, outside_range: outside ?? [] },
    responses: state.responses,
    sources: state.sources,
    simulation: state.simulation,
    simulate: { scenario: $("#scenario").value, sigma: Number($("#sigma").value), seed },
    model: modelChoice(),
  };
}

let saveTimer = null;

/** Save the campaign on the page in this browser, a moment after its last change. The worked example is not saved. */
function scheduleSave() {
  if (!state.campaign || !state.design) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 300);
}

async function saveNow() {
  if (!state.campaign || !state.design) return;
  const { id, created } = state.campaign;
  await saveCampaign({ ...currentCampaign(), id, created, updated: Date.now() });
  await refreshLibrary();
}

/** List the designs saved in this browser, each to open or delete; the one on the page is marked. */
async function refreshLibrary() {
  const saved = await savedCampaigns();
  $("#library").hidden = saved.length === 0;
  $("#library-count").textContent = String(saved.length);
  $("#library-list").replaceChildren(...saved.map(libraryItem));
}

function libraryItem(record) {
  const item = el("li", "", "");
  const here = record.id === state.campaign?.id;
  const title = record.title || "A saved design";
  const open = el("button", "link", title);
  open.type = "button";
  open.disabled = here;
  open.onclick = () => openSaved(record);
  const about = el("span", "hint", `${progress(record.responses ?? [])}, saved ${timeAgo(record.updated)}`);
  item.append(open, about);
  if (here) {
    item.append(el("span", "hint", "on the page now"));
    return item;
  }
  const forget = el("button", "link", "Delete");
  forget.type = "button";
  forget.setAttribute("aria-label", `Delete ${title}`);
  forget.onclick = async () => {
    await forgetCampaign(record.id);
    const undo = el("button", "link", "Undo");
    undo.type = "button";
    undo.onclick = async () => {
      await saveCampaign(record);
      await refreshLibrary();
    };
    item.replaceChildren(el("span", "hint", `Deleted ${title}.`), undo);
    const left = (await savedCampaigns()).length;
    $("#library-count").textContent = String(left);
  };
  item.append(forget);
  return item;
}

async function openSaved(record) {
  let campaign;
  try {
    campaign = checkCampaign(record);
  } catch (err) {
    setStatus(err.message, "error");
    return;
  }
  await openCampaign(campaign, { id: record.id, created: record.created }, `Opened ${campaign.title}.`);
}

/** Say what was opened, or why a link could not be. */
function showBanner(text, error = false) {
  const banner = $("#campaign-banner");
  banner.textContent = text;
  banner.classList.toggle("error", error);
  banner.hidden = false;
}

/**
 * Put a saved or shared campaign on the page as it was left: the form, the runs, their results,
 * the process they were simulated from and the model. The analysis follows once Python has started.
 */
async function openCampaign(c, record, message) {
  $("#demo-banner").hidden = true;
  setForm(c.design.spec);
  state.campaign = { ...record, title: c.title || titleOf(c.design.spec) };
  state.design = { ...c.design };
  state.responses = [...c.responses];
  state.sources = [...c.sources];
  state.simulation = c.simulation;
  state.analysis = null;
  const { scenario, sigma, seed: kept } = c.simulate;
  if (scenario && [...$("#scenario").options].some((o) => o.value === scenario)) $("#scenario").value = scenario;
  if (sigma) $("#sigma").value = sigma;
  if (kept !== null) seed = kept;
  showScenarioHint();
  setModel(c.model);
  renderDesign();
  renderRuns();
  renderTruth();
  clearAnalysis();
  $("#share-out").hidden = true;
  if (c.design.spec.options) showPython("design", designCode(c.design.spec));
  const done = state.responses.some((v) => v !== null);
  if (done && state.simulation && state.sources.every((source) => source === "simulated")) {
    showPython("run", simulateCode(state.simulation, c.design.spec));
  } else if (done) {
    showPython("run", enteredCode(state.responses));
  } else {
    $("#py-run").hidden = true;
  }
  showBanner(message);
  scheduleSave(); // the design opened most recently is the one a later visit picks up
  if (!done) return;
  const mine = state.campaign.id;
  await whenReady.then(
    () => (state.campaign?.id === mine ? analyseRuns() : null),
    () => {},
  );
}

/** Resolve to the designs saved in this browser, or to none if the browser has not answered in a moment. */
const savedSoon = () => Promise.race([savedCampaigns(), new Promise((resolve) => setTimeout(() => resolve([]), 2000))]);

/**
 * Open what the address asks for: a link's design, which is saved here as a campaign of its own;
 * else the design worked on last in this browser; else the worked example. The address loses
 * its fragment, so a reload opens the saved copy rather than the link again.
 */
async function start() {
  if (location.hash.startsWith(LINK_KEY)) {
    const packed = location.hash.slice(LINK_KEY.length);
    history.replaceState(null, "", `${location.pathname}${location.search}`);
    try {
      const c = await unpackCampaign(packed);
      const title = c.title || titleOf(c.design.spec);
      await openCampaign(c, newCampaign(false), `Opened from a link: ${title}. It is saved in this browser now, under Your designs.`);
      return;
    } catch (err) {
      showBanner(`${err.message} Here is the worked example instead.`, true);
    }
  } else {
    const [last] = await savedSoon();
    if (last) {
      try {
        const c = checkCampaign(last);
        await openCampaign(
          c,
          { id: last.id, created: last.created },
          `Picked up where you left off: ${c.title}, saved ${timeAgo(last.updated)}. Your other designs are ` +
            "listed under Your designs, and Run the worked example starts afresh.",
        );
        return;
      } catch {
        /* a design this page cannot read: show the worked example */
      }
    }
  }
  await refreshLibrary();
  await showBuiltExample();
}

/** Copy a link that carries the design on the page, and its results when asked, after the address's "#". */
async function shareLink() {
  if (!state.design) return;
  const withResults = $("#share-results").checked;
  const packed = await packCampaign(shareable(currentCampaign(), withResults));
  const url = `${location.origin}${location.pathname}${LINK_KEY}${packed}`;
  $("#share-url").value = url;
  let copied = true;
  try {
    await navigator.clipboard.writeText(url);
  } catch {
    copied = false; // the browser refused the clipboard: the link is in the box, selected
  }
  $("#share-note").textContent =
    `${copied ? "Copied." : "Copy the link from the box."} Whoever opens it gets ` +
    `${withResults ? "this design and its results" : "this design, without its results"}, saved in their own ` +
    `browser. Everything is in the link itself, ${url.length} characters long; nothing is stored on a server.` +
    (url.length > 8000 ? " Some chat apps cut links this long: the workbook is the surer way to send it." : "");
  $("#share-out").hidden = false;
  $("#share-url").select();
}

/** A plain table for paper: no sorting, every cell its text. */
function paperTable(columns, rows, classes = []) {
  const t = el("table", "", "");
  const head = el("tr", "", "");
  head.append(...columns.map((c) => el("th", "", c)));
  const body = el("tbody", "", "");
  for (const row of rows) {
    const tr = el("tr", "", "");
    tr.append(...row.map((cell, i) => el("td", classes[i] ?? "", cell === null ? "" : String(cell))));
    body.append(tr);
  }
  const thead = el("thead", "", "");
  thead.append(head);
  t.append(thead, body);
  return t;
}

/**
 * Print the run sheet for the plant: the runs in their order, a box for each result, and a QR
 * code that opens the design, without results, in Factorial on a phone. There the results can be
 * typed in as the runs are made, and sent back with Copy a link.
 */
async function printSheet() {
  if (!state.design) return;
  const { spec, columns, rows } = state.design;
  const title = state.campaign?.title ?? titleOf(spec);
  const link = `${location.origin}${location.pathname}${LINK_KEY}${await packCampaign(shareable(currentCampaign(), false))}`;
  const head = el("header", "", "");
  const words = el("div", "", "");
  words.append(
    el("h1", "", title),
    el(
      "p",
      "",
      `Run sheet printed ${new Date().toLocaleDateString(undefined, { dateStyle: "long" })} from Factorial. Make the runs in this order, and ` +
        "write each result in its box.",
    ),
  );
  head.append(words);
  try {
    const code = qrCode(link, { level: "M" });
    const figure = el("figure", "qr", "");
    const image = document.createElement("img");
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(qrSvg(code))}`;
    image.alt = "A QR code that opens this design in Factorial";
    image.style.width = `${Math.max(40, Math.round((code.size + 8) * 0.42))}mm`; // modules of 0.42 mm or more
    figure.append(image, el("figcaption", "", "Scan to open this design on a phone, type the results there, and send them back with Copy a link."));
    head.append(figure);
  } catch {
    /* a design too large for one QR code is printed without it */
  }
  const response = columns.indexOf(spec.response);
  const shown = columns.map((c, i) => [c, i]).filter(([, i]) => i !== response);
  const runs = paperTable(
    [...shown.map(([c]) => c), `${spec.response}, the result`, "Notes"],
    rows.map((row) => [...shown.map(([, i]) => row[i]), "", ""]),
    [...shown.map(() => ""), "result", "notes"],
  );
  const factors = paperTable(
    ["Factor", "Low", "High", "Units"],
    spec.factors.map((f) => [f.name, f.low, f.high, f.units ?? ""]),
    ["name"],
  );
  $("#print-sheet").replaceChildren(head, el("h2", "", "Factors"), factors, el("h2", "", "Runs"), runs);
  setStatus("The run sheet is ready to print.");
  window.print();
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
$("#factors").addEventListener("input", refreshDesign); // a label typed for a level makes the factor categorical
$("#response").addEventListener("input", markStale);
for (const id of ["#design-type", "#alpha", "#model_type"]) $(id).onchange = refreshDesign;
$("#alpha_value").oninput = refreshDesign; // a typed distance can put every run but the center runs on one sphere
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
$("#notebook").onclick = downloadNotebook;
$("#print").onclick = printSheet;
$("#share").onclick = shareLink;
$("#model").onchange = () => {
  if ($("#model").value === CHOSEN_TERMS && state.design) startTermList();
  else modelBefore = $("#model").value;
  $("#terms").hidden = $("#model").value !== CHOSEN_TERMS || !state.design;
  analyseRuns();
  scheduleSave();
};
$("#term-list").onchange = () => {
  scheduleAnalysis(); // a moment after the last tick, as for typed results
  scheduleSave();
};
$("#compare").onclick = compareDesigns;
for (const button of document.querySelectorAll("details.python .copy")) {
  button.onclick = async () => {
    const code = button.parentElement.querySelector("code");
    let done = "Copied";
    try {
      await navigator.clipboard.writeText(code.textContent);
    } catch {
      // The browser refused the clipboard: select the code, for the reader to copy from the keyboard.
      getSelection().selectAllChildren(code);
      done = "Selected: copy it with Ctrl+C or ⌘C";
    }
    button.textContent = done;
    setTimeout(() => (button.textContent = "Copy the code"), done === "Copied" ? 1500 : 4000);
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
start();

call("api_catalogue").then((reply) => {
  if (!reply.ok) {
    pythonFailed(reply.error);
    return;
  }
  catalogue = reply.result;
  state.version = catalogue.version ?? null;
  pythonReady = true;
  for (const id of ["#upload", "#simulate", "#reroll", "#compare"]) $(id).disabled = false;
  refreshDesign();
  resolveReady();
});
