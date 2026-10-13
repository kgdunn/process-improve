// The page: builds the form, talks to the Python worker, renders results.
// All computation happens in worker.js; this file moves JSON around and draws it.
//
// The page holds one campaign at a time: a design's runs, a response per run (typed,
// simulated or read from a workbook), the process they were simulated from, and the
// analysis. Each step re-renders from that state, so the worked example drawn at first
// paint, the live run, an uploaded workbook, a saved design and a shared link all show the
// same way. Every campaign but the worked example is saved in this browser as it changes.

import {
  coefficientFigure,
  correlationFigure,
  draw,
  fdsFigure,
  paretoFigure,
  prettyTerm,
  redrawAll,
  shareContourScale,
} from "./charts.js";
import {
  AXIAL_DISTANCE,
  CAPTIONS,
  CATEGORICAL,
  CATEGORICAL_DESIGNS,
  CHOSEN_TERMS,
  MODES,
  NUMERIC,
  SPARE_RUNS,
  TITLES,
  axialDistance,
  candidateTerms,
  compareCells,
  defaultResolution,
  escape,
  factorType,
  fitStatistics,
  fmt,
  formMatches,
  isBlank,
  isCategorical,
  levelProblem,
  levelValue,
  listNames,
  nTerms,
  namedModelTerms,
  needsCentre,
  nextMode,
  pval,
} from "./logic.js";
import {
  BOOK_PARAM,
  EXAMPLE_PARAM,
  CAMPAIGN_VERSION,
  LINK_KEY,
  bookName,
  campaignTitle,
  checkBookExample,
  checkCampaign,
  designKey,
  indexedName,
  mergeResults,
  packCampaign,
  progress,
  sameCampaign,
  shareable,
  timeAgo,
  unpackCampaign,
} from "./campaign.js";
import { analyseCode, compareCode, cumulative, designCode, notebook, resultsCode } from "./python.js";
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
/** How long a success stays on the status line before it fades, and how long the fade takes. */
const STATUS_SHOWN = 5000;
const STATUS_FADE = 600;
/** How long a busy spell lasts before the faded status line comes back for it. */
const STATUS_WAIT = 400;
let statusTimer = null;

/**
 * Say what Python is doing on the status line. A success fades after five seconds, giving its
 * room back to the page; an error comes back at once, and a busy spell once it has lasted long
 * enough to be worth showing, so a quick analysis does not make the page jump.
 */
function setStatus(text, state = "ok") {
  const box = $("#status");
  box.querySelector(".status__text").textContent = text;
  box.dataset.state = state;
  clearTimeout(statusTimer);
  const show = () => {
    box.hidden = false;
    box.classList.remove("fading");
  };
  if (state === "ok") {
    if (box.hidden) return; // nothing to say that the page does not already show
    show();
    statusTimer = setTimeout(() => {
      box.classList.add("fading");
      statusTimer = setTimeout(() => (box.hidden = true), STATUS_FADE);
    }, STATUS_SHOWN);
  } else if (state === "error" || !box.hidden) {
    show();
  } else {
    statusTimer = setTimeout(show, STATUS_WAIT);
  }
}

let toastTimer = null;

/** Say briefly, at the foot of the screen, that something was done: a link or code copied to the clipboard. */
function toast(text, state = "ok") {
  const box = $("#toast");
  box.textContent = text;
  box.dataset.state = state;
  box.hidden = false;
  void box.offsetWidth; // lay it out unshown first, so it fades in rather than appearing
  box.classList.add("shown");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    box.classList.remove("shown");
    toastTimer = setTimeout(() => (box.hidden = true), 300);
  }, 3500);
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

/** The levels a new factor of each type starts from. */
const START_LEVELS = { [NUMERIC]: ["-1", "1"], [CATEGORICAL]: ["A", "B"] };

/** One of a factor row's boxes, by its name: name, categorical (the type's switch), low, high or units. */
const box = (tr, name) => tr.querySelector(`[name=${name}]`);

/** A factor row's type, as its switch shows it. */
const typeOf = (tr) => (box(tr, "categorical").checked ? CATEGORICAL : NUMERIC);

function addFactor(f = { name: "", low: -1, high: 1, units: "" }) {
  const tr = document.createElement("tr");
  // Each cell carries its column's heading, which a phone shows above the box (style.css).
  tr.innerHTML = `
    <td data-label="Factor name"><input name="name" aria-label="Factor name" size="8"></td>
    <td data-label="Type"><label class="switch">
      <input type="checkbox" name="categorical" role="switch" aria-label="Categorical">
      <span class="switch__track" aria-hidden="true"></span><span class="switch__text">Numeric</span>
    </label></td>
    <td data-label="Low"><input name="low" aria-label="Low"></td>
    <td data-label="High"><input name="high" aria-label="High"></td>
    <td data-label="Units"><input name="units" aria-label="Units" size="6"></td>
    <td><button type="button" class="link" aria-label="Remove factor">✕</button></td>`;
  for (const key of ["name", "low", "high", "units"]) box(tr, key).value = f[key] ?? "";
  showType(tr, factorType(box(tr, "low").value, box(tr, "high").value, NUMERIC));
  // The row settles its type before the table's own listener refreshes the design.
  box(tr, "categorical").addEventListener("input", () => switchType(tr, typeOf(tr)));
  for (const name of ["low", "high"]) {
    box(tr, name).addEventListener("input", () =>
      showType(tr, factorType(box(tr, "low").value, box(tr, "high").value, typeOf(tr))),
    );
  }
  tr.querySelector("button").onclick = () => {
    tr.remove();
    refreshDesign();
  };
  $("#factors tbody").append(tr);
  return tr;
}

/**
 * Show a factor row as its type. A categorical factor's level boxes ask for its two labels, and
 * it has no units: those typed are put aside, and come back if the factor is made numeric again.
 */
function showType(tr, type) {
  const categorical = type === CATEGORICAL;
  const units = box(tr, "units");
  box(tr, "categorical").checked = categorical;
  tr.querySelector(".switch__text").textContent = categorical ? "Categorical" : "Numeric";
  tr.classList.toggle("categorical", categorical);
  if (categorical && units.value) {
    tr.dataset.units = units.value;
    units.value = "";
  } else if (!categorical && !units.value && tr.dataset.units) {
    units.value = tr.dataset.units;
  }
  units.disabled = categorical;
  units.placeholder = categorical ? "none" : "";
  const [first, second] = categorical ? ["First label", "Second label"] : ["Low", "High"];
  for (const [name, label] of [["low", first], ["high", second]]) {
    box(tr, name).setAttribute("aria-label", label);
    box(tr, name).placeholder = label.toLowerCase();
    box(tr, name).parentElement.dataset.label = label;
  }
}

/**
 * The reader flipped a factor's type: its levels become those it last had as that type, else the
 * type's own start (-1 and 1, or A and B). A factor made categorical has its first label selected,
 * ready to be typed over; one made numeric again leaves the keyboard where it was.
 */
function switchType(tr, type) {
  const before = type === CATEGORICAL ? NUMERIC : CATEGORICAL;
  const [low, high] = [box(tr, "low"), box(tr, "high")];
  tr.dataset[before] = JSON.stringify([low.value, high.value]);
  [low.value, high.value] = tr.dataset[type] ? JSON.parse(tr.dataset[type]) : START_LEVELS[type];
  showType(tr, type);
  if (type !== CATEGORICAL) return;
  low.focus();
  low.select();
}

/** Say under the table what each factor's levels still need for its type; return whether any does. */
function showFactorProblems() {
  const problems = [...document.querySelectorAll("#factors tbody tr")].map((tr) => {
    const problem = levelProblem(
      { name: box(tr, "name").value, low: box(tr, "low").value, high: box(tr, "high").value },
      typeOf(tr),
    );
    for (const name of ["low", "high"]) box(tr, name).toggleAttribute("aria-invalid", Boolean(problem));
    return problem;
  });
  const said = problems.filter(Boolean);
  $("#factor-problems").textContent = said.join(" ");
  $("#factor-problems").hidden = said.length === 0;
  return said.length > 0;
}

const factorCount = () => document.querySelectorAll("#factors tbody tr").length;

function readFactors() {
  // A blank box is sent as null, which Python reports, rather than as Number("") = 0; a label, as text.
  // A factor's type is not sent: two labels make it categorical, as bootstrap.py reads it.
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
  const unready = showFactorProblems();
  $("#generate").disabled = !pythonReady || option.disabled || unready;
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
  flushSave();
  const replaces = state.campaign?.replaceable && !state.sources.includes("entered");
  state.campaign = save
    ? { ...(replaces ? state.campaign : newCampaign()), title: titleOf(design.spec), example: false }
    : null;
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
  hidePython("run");
  scheduleSave();
}

function renderDesign(warnings = []) {
  showExampleButton();
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
    showResultsCode();
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
  showResultsCode();
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
 * Download the campaign as a Jupyter notebook: each step's code, as its "Show the Python code"
 * panel shows it, a cell a step, so it runs to the page's design, results and analysis.
 */
function downloadNotebook() {
  if (!state.design) return;
  const book = notebook({
    title: state.campaign?.title ?? titleOf(state.design.spec),
    design: state.design,
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
  flushSave();
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
  showResultsCode();
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
  // The terms fitted, which the design can estimate, from the runs that have a result.
  showPython("analyse", analyseCode(r.formula ?? r.model, state.responses.every((v) => v !== null)));
  const parts = [
    el("p", "equation", r.equation),
    el("p", "hint", "Coefficients are in coded units: each factor runs from -1 at its low value to +1 at its high value."),
    ...noteList(fitStatistics(s, r.n_completed, r.pending_runs), "fit"),
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
  card.append(el("h3", "", TITLES[key]), plot, ...(caption ? [el("figcaption", "hint", caption)] : []));
  return { card, plot };
}

const COEFFICIENT_VIEW = "factorial-coefficients";

/** How the reader last chose to see the coefficients: their estimates, or the book's Pareto plot. */
function coefficientView() {
  try {
    return localStorage.getItem(COEFFICIENT_VIEW) === "pareto" ? "pareto" : "estimates";
  } catch {
    return "estimates";
  }
}

/**
 * The coefficients' card, with a switch between two views of the same analysis: each estimate
 * with its interval (and the true value, when simulated), or the book's Pareto plot of their sizes.
 * ``show(view, version)`` marks the view chosen, and draws it once Plotly.js is given.
 */
function coefficientCard(analysis, simulated) {
  const { card, plot } = chartCard("coefficients", null, true); // its legend says what each mark is
  const head = el("div", "chart__head", "");
  const views = el("div", "views", "");
  views.setAttribute("role", "group");
  views.setAttribute("aria-label", "Show the coefficients as");
  const buttons = [
    ["estimates", "Estimates"],
    ["pareto", "Pareto plot"],
  ].map(([view, text]) => {
    const button = el("button", "", text);
    button.type = "button";
    button.dataset.view = view;
    return button;
  });
  views.append(...buttons);
  head.append(card.querySelector("h3"), views);
  card.prepend(head);
  const figure = (view) => () =>
    view === "pareto"
      ? paretoFigure(analysis.coefficients, simulated, analysis.lenth)
      : coefficientFigure(analysis.coefficients, simulated);
  const show = async (view, version) => {
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.view === view));
    if (version) await draw(plot, figure(view), version, figure(view)().height);
  };
  for (const button of buttons) {
    button.onclick = () => {
      try {
        localStorage.setItem(COEFFICIENT_VIEW, button.dataset.view);
      } catch {
        /* storage refused: the view holds for this card */
      }
      show(button.dataset.view, state.plotlyJs);
    };
  }
  return { card, plot, show };
}

/** Draw the coefficient chart, then the library's plots for the same analysis. */
async function drawCharts(request, mine, given = null) {
  const box = $("#charts");
  box.hidden = false;
  const simulated = Boolean(request.truth);
  const coefficients = coefficientCard(state.analysis, simulated);
  box.replaceChildren(coefficients.card);
  await coefficients.show(coefficientView());

  const plots = given ?? (await libraryPlots(request, mine));
  if (mine !== ticket) return;
  const version = plots?.plotly_js ?? state.plotlyJs;
  if (!version) {
    coefficients.card.append(el("p", "hint", "The charts could not be drawn here; the tables above hold the numbers."));
    return;
  }
  state.plotlyJs = version;
  try {
    await coefficients.show(coefficientView(), version);
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
  showResultsCode();
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

/**
 * Say what the button at the top does. While the worked example is on the page it runs it again in
 * this tab. Once a design of the reader's is open (made here, or from a link, a workbook, Your
 * designs or the book) it reads as a step aside, not as a run of that design: it puts the worked
 * example in its place, and says whether that design stays under Your designs. One opened from a
 * link or the list, or with a result typed in, does; one only tried here is replaced (see showDesign).
 */
function showExampleButton() {
  const button = $("#run-example");
  const other = state.campaign && !state.campaign.example ? state.campaign.title : null;
  const kept = other && !(state.campaign.replaceable && !state.sources.includes("entered"));
  if (!button.dataset.waiting) button.textContent = other ? "Open the worked example instead" : "Run the worked example";
  button.classList.toggle("primary", !other);
  const example = "The worked example, a Box-Behnken design in three factors with results simulated from a known process,";
  $("#example-hint").textContent = !other
    ? "A Box-Behnken design in three factors, results simulated from a known process, and their analysis, " +
      "computed in this tab."
    : kept
      ? `${example} takes the place of ${other}, which stays under Your designs.`
      : `${example} takes the place of ${other}.`;
}

async function runExample() {
  const button = $("#run-example");
  if (!pythonReady) {
    button.disabled = true;
    button.dataset.waiting = "yes";
    button.textContent = "Waiting for Python to start...";
    const started = await whenReady.then(() => true, () => false);
    delete button.dataset.waiting;
    showExampleButton();
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
  if (state.campaign) state.campaign.example = true; // the worked example run here, until its design changes
  showExampleButton();
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
      await draw(plot, () => correlationFigure(terms, matrix, k), version, 320);
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
  showExampleButton(); // a result typed in keeps the design under Your designs
  if (!state.campaign || !state.design) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    saveNow();
  }, 300);
}

/**
 * Write a save that is still waiting for its moment, at once. The page calls this before another
 * campaign takes this one's place, and when the reader leaves, so a result typed just before is
 * kept. saveNow reads the campaign before it first waits, so the record is this campaign's.
 */
function flushSave() {
  if (saveTimer === null) return;
  clearTimeout(saveTimer);
  saveTimer = null;
  saveNow();
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

/** Say what was opened, or why a link could not be: a sentence, or a list of text and links. */
function showBanner(message, error = false) {
  const banner = $("#campaign-banner");
  banner.replaceChildren(...[message].flat());
  banner.classList.toggle("error", error);
  banner.hidden = false;
}

/**
 * Put a saved or shared campaign on the page as it was left: the form, the runs, their results,
 * the process they were simulated from and the model. The analysis follows once Python has started.
 */
async function openCampaign(c, record, message) {
  flushSave();
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
  if (done) showResultsCode();
  else hidePython("run");
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
 * Open what the address asks for: one of the book's examples, or a link's design, either saved
 * here as a campaign of its own; else the design worked on last in this browser; else the worked
 * example. The address loses what named it, so a reload opens the saved copy rather than the link again.
 */
async function start() {
  const { hash } = location;
  const params = new URLSearchParams(location.search);
  const book = params.get(BOOK_PARAM) ?? params.get(EXAMPLE_PARAM); // the book's examples and the published ones share one index
  if (book !== null || hash.startsWith(LINK_KEY)) {
    history.replaceState(null, "", book !== null ? location.pathname : `${location.pathname}${location.search}`);
    try {
      if (book !== null) {
        await openBookExample(bookName(book));
        return;
      }
      const c = await unpackCampaign(hash.slice(LINK_KEY.length));
      if (await resultsCameBack(c)) return;
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
            "listed under Your designs, and Open the worked example instead, at the top, starts afresh.",
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

/**
 * Open one of the book's examples, as the build made it from the book's runs and results. The
 * copy saved here is opened if it is still as the book has it; else a fresh copy, so a reader's
 * own changes to an earlier copy are kept, and the book's numbers come back all the same.
 */
async function openBookExample(name) {
  const fetched = async (path) => {
    try {
      const res = await fetch(path);
      if (res.ok) return await res.json();
    } catch {
      /* said below */
    }
    throw new Error(`This design could not be opened: the book's example ${JSON.stringify(name)} could not be fetched.`);
  };
  const file = indexedName(await fetched("book/index.json"), name);
  const example = checkBookExample(await fetched(`book/${file}.json`), name);
  const c = example.campaign;
  const kept = (await savedSoon()).find((record) => {
    try {
      return sameCampaign(checkCampaign(record), c);
    } catch {
      return false;
    }
  });
  const record = kept ? { id: kept.id, created: kept.created } : newCampaign(false);
  const note = example.note ? ` ${example.note}` : "";
  const { source } = example;
  const opened = source
    ? [
        `Opened an example from ${source.cite}, ${source.where}: ${example.title}, with its published results. `,
        link(source.url, "Read the source", "source"),
      ]
    : [`Opened the book's example: ${example.title}, with the book's results. `, link(example.book, "Read it in the book")];
  await openCampaign(c, record, [...opened, `.${note} It is saved in this browser, under Your designs.`]);
}

/**
 * Results sent back by link for a design this browser keeps, from a run sheet shared
 * here: fill them into that design rather than saving a copy. A result already here is kept.
 * Resolves to whether the link was such a return.
 */
async function resultsCameBack(c) {
  if (!c.responses.some((v) => v !== null)) return false;
  const key = designKey(c.design);
  const readable = (record) => {
    try {
      return checkCampaign(record);
    } catch {
      return null;
    }
  };
  const match = (await savedSoon()).find((record) => readable(record) && designKey(readable(record).design) === key);
  if (!match) return false;
  const here = readable(match);
  const { responses, sources, added, kept } = mergeResults(here, c);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  let message = added
    ? `Added ${plural(added, "result")} from a link to ${here.title}, kept in this browser.`
    : `The link's results are already in ${here.title}, kept in this browser.`;
  if (kept) message += ` ${plural(kept, "run")} already had a different result here, which ${kept === 1 ? "was" : "were"} kept.`;
  await openCampaign({ ...here, responses, sources }, { id: match.id, created: match.created }, message);
  return true;
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
  toast(
    copied
      ? "Link copied to the clipboard: paste it into an email or a chat."
      : "The browser would not copy the link: it is selected in the box below, to copy by hand.",
    copied ? "ok" : "error",
  );
  $("#share-note").textContent =
    `${copied ? "" : "Copy the link from the box. "}Whoever opens it gets ` +
    `${withResults ? "this design and its results" : "this design, without its results"}, saved in their own ` +
    `browser. Everything is in the link itself, ${url.length} characters long; nothing is stored on a server.` +
    (url.length > 8000 ? " Some chat apps cut links this long: the workbook is the surer way to send it." : "");
  $("#share-out").hidden = false;
  $("#share-url").select();
}

// ---------------------------------------------------------------- show the Python code
/** The steps whose panels build on each other, in order, and the code each step last showed. */
const STEPS = ["design", "run", "analyse"];
const stepCode = {};

/**
 * Show a step's code in its panel, after the code of the steps it builds on, so what is copied
 * from any panel runs as it is. Step 4's comparison stands on its own.
 */
function showPython(key, code) {
  stepCode[key] = code;
  const steps = STEPS.includes(key) ? STEPS.slice(0, STEPS.indexOf(key) + 1) : [key];
  const panel = $(`#py-${key}`);
  panel.querySelector("code").textContent = STEPS.includes(key)
    ? cumulative(...steps.map((step) => stepCode[step]).filter(Boolean))
    : code;
  panel.hidden = false;
}

/** Hide a step's panel and forget its code: its results are gone. */
function hidePython(key) {
  delete stepCode[key];
  $(`#py-${key}`).hidden = true;
}

/** Show step 2's code: the simulation while every result is as simulated, else the results on the page. */
function showResultsCode() {
  const simulated = state.simulation && state.sources.every((source) => source === "simulated");
  showPython("run", resultsCode(state.design, state.responses, simulated ? state.simulation : null));
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

/**
 * A link to the book, or to another source, opening in a tab of its own so the page keeps its
 * state: the same tab each time, rather than a new one per link.
 */
function link(href, text, target = "pid-book") {
  const a = el("a", "", text);
  a.href = href;
  a.target = target;
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

// ---------------------------------------------------------------- colour mode
const MODE_KEY = "factorial-mode";
const MODE_NAMES = { auto: "the system's", light: "light", dark: "dark" };
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

/**
 * Show the page in a colour mode: light, dark, or the system's (the head of index.html put the
 * reader's choice on the page before it was drawn). The charts are drawn again in its colours, and
 * the browser's own bar follows it.
 */
function applyMode(mode) {
  const root = document.documentElement;
  root.dataset.mode = mode;
  if (mode === "auto") delete root.dataset.theme;
  else root.dataset.theme = mode;
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    const { scheme } = meta.dataset;
    meta.media = mode === "auto" ? `(prefers-color-scheme: ${scheme})` : scheme === mode ? "all" : "not all";
  }
  $("#mode").setAttribute("aria-label", `Colour mode: ${MODE_NAMES[mode]}`);
  redrawAll();
}

$("#mode").onclick = () => {
  const mode = nextMode(document.documentElement.dataset.mode, systemDark.matches);
  try {
    if (mode === "auto") localStorage.removeItem(MODE_KEY);
    else localStorage.setItem(MODE_KEY, mode);
  } catch {
    /* storage refused: the mode holds for this visit */
  }
  applyMode(mode);
};
systemDark.addEventListener("change", () => {
  if (document.documentElement.dataset.mode === "auto") redrawAll(); // the charts follow the system
});
applyMode(MODES.includes(document.documentElement.dataset.mode) ? document.documentElement.dataset.mode : "auto");

// ---------------------------------------------------------------- wiring
DEFAULT_FACTORS.forEach(addFactor);
$("#add-factor").onclick = () => {
  addFactor();
  refreshDesign();
};
$("#add-categorical").onclick = () => {
  const [low, high] = START_LEVELS[CATEGORICAL];
  box(addFactor({ name: "", low, high, units: "" }), "name").focus();
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
      toast("Code copied to the clipboard: paste it into a script or a notebook.");
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
// A reader who closes the tab, or puts the phone away, straight after typing a result keeps it.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flushSave();
});
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
