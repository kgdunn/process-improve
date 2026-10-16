// The page: builds the form, talks to the Python worker, renders results.
// All computation happens in worker.js; this file moves JSON around and draws it.
//
// A study runs in three phases. Design: the brief (the response, the factors that might change
// it, and what the study is to learn) and the designs that can learn it, made and scored as
// the reader types. Run: the run sheet, its results typed in, read from a workbook, or
// simulated for practice. Learn: the analysis. The page holds one study at a time: a design's
// runs, a response per run, the process they were simulated from, and the analysis. Each phase
// re-renders from that state, so the worked example, the live run, an uploaded workbook, a
// saved study and a shared link all show the same way. Every study but the worked example is
// saved in this browser as it changes, and a brief not yet made into a run sheet is kept as a draft.
// In the code the first phase keeps its earlier name, plan: its tab and pane ids, and planKey,
// the form's settings a run sheet was made from.

import {
  coefficientFigure,
  correlationFigure,
  coxFigure,
  draw,
  fdsFigure,
  paretoFigure,
  prettyTerm,
  redrawAll,
  resizeWithin,
  shareContourScale,
  ternaryFigure,
  ternaryRange,
} from "./charts.js";
import {
  AXIAL_DISTANCE,
  BLOCK,
  CAPTIONS,
  CATEGORICAL,
  CATEGORICAL_DESIGNS,
  CHOSEN_TERMS,
  GIVEN,
  MODES,
  NUMERIC,
  OPTIMAL_MODEL,
  REGION_DESIGNS,
  SCREEN_FROM,
  SPARE_FOR_NOISE,
  SPARE_RUNS,
  TITLES,
  aimTerms,
  analysisModel,
  axialDistance,
  candidateRows,
  candidateTerms,
  changeWords,
  compareCells,
  componentProblem,
  defaultAim,
  defaultResolution,
  describeResponse,
  escape,
  factorType,
  fitStatistics,
  fmt,
  isBlank,
  isCategorical,
  isMixture,
  levelProblem,
  levelValue,
  listNames,
  mixtureProblem,
  nTerms,
  nameProblem,
  namedModelTerms,
  needsCentre,
  nextMode,
  percent,
  pickCandidate,
  positional,
  powerScale,
  proportionValue,
  pval,
  renameTerms,
  renaming,
  responseName,
  responseProblems,
  responseTitle,
  runBadge,
} from "./logic.js";
import {
  AIMS,
  BOOK_PARAM,
  EXAMPLE_PARAM,
  CAMPAIGN_VERSION,
  LINK_KEY,
  WORKED,
  bookName,
  campaignTitle,
  checkBookExample,
  checkCampaign,
  checkConstraints,
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
import {
  adoptLimits,
  currentLimits,
  factorsChangedForLimits,
  hasLimits,
  installLimits,
  limitsHold,
  limitsKey,
  limitsUnready,
  renameInLimits,
  setLimits,
} from "./limits.js";
import { redrawRuns } from "./picture.js";
import { analyseCode, compareCode, cumulative, designCode, notebook, resultsCode } from "./python.js";
import { forgetCampaign, saveCampaign, savedCampaigns } from "./store.js";
import { GLOSSARY, MARKED } from "./glossary.js";
import { installTerms, marked, term } from "./terms.js";

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
  simplex_lattice: `${BOOK}/mixture-experiments#doe-mixture-designs`,
  simplex_centroid: `${BOOK}/mixture-experiments#doe-mixture-designs`,
  extreme_vertices: `${BOOK}/mixture-experiments#doe-mixture-designs`,
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
 * Python could not start: say why, and stop everything that waits for it. A worked example on
 * the page stays, and every call is answered with the reason.
 */
function pythonFailed(reason) {
  if (pythonFailure) return;
  pythonFailure = reason;
  setStatus(reason, "error");
  $("#run-example").disabled = true;
  for (const resolve of pending.values()) resolve({ ok: false, error: reason, warnings: [] });
  pending.clear();
  rejectReady(new Error(reason));
  renderCandidates();
}

/** How long a success stays on the status line before it fades, and how long the fade takes. */
const STATUS_SHOWN = 5000;
const STATUS_FADE = 600;
/** How long a busy spell lasts before the faded status line comes back for it. */
const STATUS_WAIT = 400;
let statusTimer = null;

/**
 * Say what Python is doing on the status line; state is "busy" (the mark's runs light up), "ok"
 * (a tick) or "error". A success fades after five seconds, giving its room back to the page; an
 * error comes back at once, and a busy spell once it has lasted long enough to be worth showing,
 * so a quick analysis does not make the page jump.
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
  phase: "plan", // the phase on show: "plan", "run" or "learn"
  design: null, // what api_make_design returned, or the runs of an uploaded workbook
  planKey: null, // the plan (factors, design and settings) the run sheet on the page was made from
  responses: [], // one per run, in the design's row order: a number, or null while still to run
  sources: [], // one per run: "simulated", "entered" or null
  simulation: null, // what api_simulate returned: the true process, its noise and seed
  practising: false, // the reader chose to practise with simulated results
  analysis: null,
  comparison: null, // the designs that can learn the brief: what api_candidates returned, and the brief's key
  picked: null, // a design taken from the comparison, {key, design}, made again only if the plan changes
  plotlyJs: null, // the Plotly.js version the Python plotly package names
  campaign: null, // this browser's record of the study on the page, {id, created, title}; null for the worked example
  example: false, // the worked example is on the page, as the build made it or as run again here
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

/** Whether the plan is a mixture's: the factors are then the components of a blend, each a proportion. */
const mixturePlanned = () => isMixture($("#design-type").value);

// ---------------------------------------------------------------- the phases
const PANES = { plan: "#step-plan", run: "#step-run", learn: "#step-analyse" };

/**
 * Show a phase: its tab is marked, its pane shown, and on a narrow screen the brief goes with
 * Design. Charts drawn while their pane was hidden are fitted to it, and the designs that can learn
 * the brief are made once Design is on show.
 */
function showPhase(phase) {
  state.phase = phase;
  $("#workspace").dataset.phase = phase;
  for (const [name, pane] of Object.entries(PANES)) {
    const on = name === phase;
    const tab = $(`#go-${name}`);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    $(pane).hidden = !on;
  }
  resizeWithin($(PANES[phase]));
  if (phase !== "plan") cancelCandidates();
  // The top of the workspace in view, under the study bar, when the phase starts above the window's top.
  const top = $("#workspace").getBoundingClientRect().top - $("#studybar").offsetHeight - 8;
  if (top < 0) window.scrollBy({ top, behavior: "instant" });
  if (phase === "plan") scheduleCandidates();
}

/** The tabs of a tablist answer the arrow keys, Home and End, as a reader of a screen reader expects. */
function arrowKeys(list, select) {
  list.addEventListener("keydown", (e) => {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const at = tabs.indexOf(document.activeElement);
    if (at < 0) return;
    const to = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const tab = tabs[(to + tabs.length) % tabs.length];
    tab.focus();
    select(tab);
  });
}

/** Say on each tab how far its phase has got: the plan changed since the run sheet, the results in, the fit. */
function renderBadges() {
  const badge = (name, text, mark = "") => {
    const el = $(`#badge-${name}`);
    el.textContent = text;
    el.hidden = !text;
    if (mark) el.dataset.state = mark;
    else delete el.dataset.state;
  };
  const stale = planIsStale();
  badge("plan", stale ? "changed" : "", stale ? "stale" : "");
  const done = state.responses.length > 0 && state.responses.every((v) => v !== null);
  badge("run", state.design ? runBadge(state.responses) : "", done ? "done" : "");
  const r2 = state.analysis?.summary?.r_squared;
  badge("learn", r2 === null || r2 === undefined ? "" : `R² ${Number(r2).toFixed(2)}`);
}

/** Name the study in the bar: its response as the brief describes it, and the design or the factors. */
function renderStudyBar() {
  const info = readResponseInfo();
  const name = info.label ?? state.design?.spec.response ?? "";
  $("#study-title").textContent = name ? responseTitle(info, name) : "A new study";
  const k = readFactors().length;
  const noun = mixturePlanned() ? "component" : "factor";
  const about = state.design
    ? `${designLabel(state.design.spec.design_type)}, ${state.design.rows.length} runs`
    : k
      ? `${k} ${noun}${k === 1 ? "" : "s"}`
      : "";
  $("#study-about").textContent = about ? `· ${about}` : "";
  renderBadges();
}

// ---------------------------------------------------------------- the response
/** The response card's boxes as the reader left them, as text. */
function responseFields() {
  return {
    label: $("#response").value,
    units: $("#response-units").value,
    goal: document.querySelector('input[name="goal"]:checked')?.value ?? "learn",
    target: $("#response-target").value,
    typical: $("#response-typical").value,
    noise: $("#response-noise").value,
    smallest_change: $("#response-change").value,
  };
}

/** The response as the reader describes it: what a design keeps as its ``response_info``. */
const readResponseInfo = () => describeResponse(responseFields());

/** Put a design's response into the card: the reader's description, else its column's name. */
function setResponse(spec) {
  const info = spec.response_info ?? {};
  $("#response").value = info.label ?? spec.response ?? "";
  $("#response-units").value = info.units ?? "";
  const goal = document.querySelector(`input[name="goal"][value="${info.goal ?? "learn"}"]`);
  if (goal) goal.checked = true;
  $("#response-target").value = info.target ?? "";
  $("#response-typical").value = info.typical ?? "";
  $("#response-noise").value = info.noise ?? "";
  $("#response-change").value = info.smallest_change ?? "";
  $("#response-known").open = ["typical", "noise", "smallest_change"].some((key) => info[key] !== undefined);
  if (info.noise !== undefined) $("#sigma").value = info.noise;
  showResponse();
}

/** Show what the response's boxes imply: the target's box for that goal, its units after each number, and problems. */
function showResponse() {
  const fields = responseFields();
  $("#target-box").hidden = fields.goal !== "target";
  for (const unit of document.querySelectorAll(".with-unit .unit")) unit.textContent = fields.units.trim();
  const said = responseProblems(fields);
  $("#response-problems").textContent = said.join(" ");
  $("#response-problems").hidden = said.length === 0;
  for (const [id, key] of [["#response-noise", "noise"], ["#response-change", "smallest_change"]]) {
    $(id).toggleAttribute("aria-invalid", said.some((s) => s.includes(key === "noise" ? "noise" : "smallest change")));
  }
}

/**
 * The reader changed the response card. The description is the reader's, not the design's: a run
 * sheet already made keeps its column and runs, and takes the new words, units and numbers, so the
 * title, the run sheet, the charts and the saved study follow. Practice results follow a new typical
 * value or noise, and the designs are scored again for a new noise or smallest change.
 */
function responseChanged() {
  showResponse();
  const info = readResponseInfo();
  if (info.noise !== undefined) $("#sigma").value = info.noise;
  if (state.design) {
    const before = state.design.spec.response_info ?? {};
    const titled = state.campaign && state.campaign.title === titleOf(state.design.spec);
    state.design.spec = { ...state.design.spec, response_info: info };
    if (titled) state.campaign.title = titleOf(state.design.spec);
    renderRuns();
    renderTruth();
    showResultsCode();
    const changed = (keys) => keys.some((key) => before[key] !== info[key]);
    if (state.sources.includes("simulated") && changed(["typical", "noise"])) scheduleSimulation();
    else if (state.analysis && changed(["label", "units"])) scheduleAnalysis();
    scheduleSave();
  }
  scheduleCandidates();
  renderStudyBar();
  saveDraft();
}

// ---------------------------------------------------------------- the factors
/** The levels a new factor of each type starts from: a numeric one's are typed in, a categorical one's are labels. */
const START_LEVELS = { [NUMERIC]: ["", ""], [CATEGORICAL]: ["A", "B"] };

/** One of a factor row's boxes, by its name: name, categorical (the type's switch), low, high or units. */
const box = (tr, name) => tr.querySelector(`[name=${name}]`);

/** A factor row's type, as its switch shows it. */
const typeOf = (tr) => (box(tr, "categorical").checked ? CATEGORICAL : NUMERIC);

/** The factor rows of the table, and whether one is still blank: no name and no levels, which is no factor yet. */
const factorRows = () => [...document.querySelectorAll("#factors tbody tr")];
const blankRow = (tr) => ["name", "low", "high"].every((key) => box(tr, key).value.trim() === "");

function addFactor(f = { name: "", low: "", high: "", units: "" }) {
  const tr = document.createElement("tr");
  // Each cell carries its column's heading, which a narrow column shows above the box (style.css).
  tr.innerHTML = `
    <td data-label="Factor name"><input name="name" aria-label="Factor name" size="8" autocomplete="off" spellcheck="false"></td>
    <td data-label="Type"><label class="switch">
      <input type="checkbox" name="categorical" role="switch" aria-label="Categorical">
      <span class="switch__track" aria-hidden="true"></span><span class="switch__text">Numeric</span>
    </label></td>
    <td data-label="Low"><input name="low" aria-label="Low"></td>
    <td data-label="High"><input name="high" aria-label="High"></td>
    <td data-label="Units"><input name="units" aria-label="Units" size="6"></td>
    <td><button type="button" class="link" aria-label="Remove factor">✕</button></td>`;
  for (const key of ["name", "low", "high", "units"]) box(tr, key).value = f[key] ?? "";
  tr.dataset.name = box(tr, "name").value.trim(); // the name the limits know it by
  showType(tr, factorType(box(tr, "low").value, box(tr, "high").value, NUMERIC));
  // The row settles its type before the table's own listener refreshes the design.
  box(tr, "categorical").addEventListener("input", () => switchType(tr, typeOf(tr)));
  for (const name of ["low", "high"]) {
    box(tr, name).addEventListener("input", () =>
      showType(tr, factorType(box(tr, "low").value, box(tr, "high").value, typeOf(tr))),
    );
    box(tr, name).addEventListener("focusout", () => {
      if (tr.dataset.visited || blankRow(tr)) return;
      tr.dataset.visited = "yes";
      showFactorProblems();
    });
  }
  tr.querySelector("button").onclick = () => {
    tr.remove();
    factorsChanged();
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
  // A mixture's component has a lower and an upper proportion, 0 and 1 when its boxes are left blank.
  const component = !categorical && mixturePlanned();
  const [first, second] = categorical ? ["First label", "Second label"] : component ? ["Lower", "Upper"] : ["Low", "High"];
  for (const [name, label, blank] of [["low", first, "0"], ["high", second, "1"]]) {
    box(tr, name).setAttribute("aria-label", component ? `${label} proportion` : label);
    box(tr, name).placeholder = component ? blank : label.toLowerCase();
    box(tr, name).parentElement.dataset.label = label;
  }
}

/** Head the factor table, and show each row, as the plan reads them: a mixture's components, or factors with ranges. */
function showComponents() {
  const mixture = mixturePlanned();
  $("#h-low").textContent = mixture ? "Lower" : "Low";
  $("#h-high").textContent = mixture ? "Upper" : "High";
  for (const tr of factorRows()) showType(tr, typeOf(tr));
  $("#limits-mixture").hidden = !mixture;
}

/**
 * The reader flipped a factor's type: its levels become those it last had as that type, else the
 * type's own start (blank, or A and B). A factor made categorical has its first label selected,
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

/** Whether a factor still needs something before Python can take it, said or not yet said. */
let factorsUnready = false;

/**
 * Say under the table what each factor still needs, its name and its levels for its type, and
 * return whether any does. A row left blank is no factor yet, and needs nothing; a row named but
 * with both level boxes still empty is not told so until the reader has been to them.
 */
function showFactorProblems() {
  const all = factorRows();
  const rows = all.filter((tr) => !blankRow(tr));
  const names = rows.map((tr) => box(tr, "name").value.trim());
  const taken = takenNames();
  const mixture = mixturePlanned();
  let unready = false;
  const problems = rows.map((tr, i) => {
    const others = names.filter((_, j) => j !== i); // both factors that share a name are marked
    const named = nameProblem(names[i], others, all.indexOf(tr) + 1, taken);
    const typed = { name: names[i], low: box(tr, "low").value, high: box(tr, "high").value };
    const levels = mixture && typeOf(tr) !== CATEGORICAL ? componentProblem(typed) : levelProblem(typed, typeOf(tr));
    unready ||= Boolean(named || levels);
    const untouched = !tr.dataset.visited && !box(tr, "low").value.trim() && !box(tr, "high").value.trim();
    const said = [named, untouched ? "" : levels].filter(Boolean);
    box(tr, "name").toggleAttribute("aria-invalid", Boolean(named));
    for (const name of ["low", "high"]) box(tr, name).toggleAttribute("aria-invalid", Boolean(levels) && !untouched);
    return said.join(" ");
  });
  for (const tr of all.filter(blankRow)) {
    for (const name of ["name", "low", "high"]) box(tr, name).removeAttribute("aria-invalid");
  }
  // A mixture's components, each ready, must still sum as a blend can, with bounds the design type takes.
  if (mixture && !unready && rows.length && !rows.some((tr) => typeOf(tr) === CATEGORICAL)) {
    const option = $("#design-type").selectedOptions[0];
    const together = mixtureProblem(readFactors(), option.value, option.textContent.trim());
    problems.push(together);
    unready = Boolean(together);
  }
  const said = [...new Set(problems.filter(Boolean))];
  $("#factor-problems").textContent = said.join(" ");
  $("#factor-problems").hidden = said.length === 0;
  factorsUnready = unready;
  return unready;
}

function readFactors() {
  // A blank box is sent as null, which Python reports, rather than as Number("") = 0; a label, as text.
  // A factor's type is not sent: two labels make it categorical, as bootstrap.py reads it. For a mixture, a
  // blank box is the component's bound: 0 for the lower proportion, 1 for the upper.
  const mixture = mixturePlanned();
  const level = (tr, name, blank) => (mixture ? proportionValue : levelValue)(box(tr, name).value, blank);
  return factorRows()
    .filter((tr) => !blankRow(tr))
    .map((tr) => ({
      name: box(tr, "name").value.trim(),
      low: level(tr, "low", 0),
      high: level(tr, "high", 1),
      units: box(tr, "units").value.trim(),
    }));
}

const factorCount = () => readFactors().length;

/**
 * The run sheet's columns besides its factors, which a factor renamed on it cannot take: its results
 * and its blocks. None while the factors differ from the run sheet's by more than their labels: a run
 * sheet made again names its results afresh.
 */
function takenNames() {
  const d = state.design;
  if (!d || !renaming(d.spec.factors, readFactors())) return [];
  return [d.spec.response, ...(d.columns.includes(BLOCK) ? [BLOCK] : [])];
}

/** The factor table changed: the design settings, the aim, the designs that can learn it and the draft follow. */
function factorsChanged() {
  if (!aimChosen) setAim(defaultAim(factorCount()));
  factorsChangedForLimits();
  refreshDesign();
  scheduleCandidates();
  renderStudyBar();
  saveDraft();
}

/** Each rename waits for the one before it, and starts from the run sheet that one left. */
let renamed = Promise.resolve();

/**
 * The reader left a factor's name or units box. The brief takes the new label, and so does a run
 * sheet made with these factors' levels: it keeps its runs and their results, and Python writes the
 * new label into all it wrote with the old (the run sheet's columns, the design's notes, the process
 * practice results come from). A name Python would refuse is not taken: it stays marked under the
 * table, and the old name stays everywhere until it is put right.
 */
function factorRenamed(tr) {
  if (tr) limitsFollow(tr);
  factorsChanged();
  renamed = renamed.then(renameRunSheet).catch(reportError); // a rename that fails is reported, and the next still runs
}

/**
 * A factor's new name is taken into every limit that names it, at once, so the limits key the run sheet
 * as before. A name the brief cannot take yet (blank, not a name, another factor's) waits until it is
 * put right, and the limits keep the last name taken.
 */
function limitsFollow(tr) {
  const before = tr.dataset.name ?? "";
  const after = box(tr, "name").value.trim();
  const others = factorRows()
    .filter((row) => row !== tr && !blankRow(row))
    .map((row) => box(row, "name").value.trim());
  if (nameProblem(after, others, 1, takenNames())) return;
  if (before && before !== after) renameInLimits({ [before]: after });
  tr.dataset.name = after;
}

/**
 * The limits changed, or Python has read them: the designs that keep to them, the run sheet's key,
 * the designs that can learn the brief and the draft follow. A note changed on the limits a run sheet
 * was made with is a label: the run sheet takes it.
 */
function limitsChanged() {
  refreshDesign();
  scheduleCandidates();
  renderStudyBar();
  saveDraft();
  const stored = state.design?.spec.constraints;
  if (stored && !planIsStale()) {
    const typed = currentLimits();
    state.design.spec.constraints = stored.map((limit, i) => {
      const { note: _, ...rest } = limit;
      return typed[i]?.note ? { ...rest, note: typed[i].note } : rest;
    });
    scheduleSave();
  }
}

/** What Python needs of a design to rename its factors: the runs as the page holds them, and their notes. */
const renameRequest = ({ spec, columns, rows, facts, outside_range: outside }) => ({
  spec,
  columns,
  rows,
  facts: facts ?? null,
  outside_range: outside ?? [],
});

async function renameRunSheet() {
  const design = state.design;
  const factors = readFactors();
  const names = design && !factorsUnready ? renaming(design.spec.factors, factors) : null;
  if (!names) return;
  const reply = await call("api_relabel", { design: renameRequest(design), factors, simulation: state.simulation });
  if (state.design !== design) return; // another run sheet took its place meanwhile
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  showRenamed(reply.result, names);
}

/**
 * Put the renamed run sheet on the page with all that shows its factors: the summary and its notes,
 * the run sheet, the true process, the ticked terms, the Python, the study's title, and the analysis
 * and its charts, fitted again under the new names.
 */
function showRenamed({ spec, simulation, ...design }, names) {
  const titled = state.campaign && state.campaign.title === titleOf(state.design.spec);
  // Python renames the factors, and the terms a workbook was analysed with; the rest of the spec stays the page's.
  const terms = spec.analysis_model ? { analysis_model: spec.analysis_model } : {};
  const limits = spec.constraints ? { constraints: spec.constraints } : {};
  state.design = { ...state.design, ...design, spec: { ...state.design.spec, factors: spec.factors, ...terms, ...limits } };
  if (simulation) state.simulation = simulation;
  if (titled) state.campaign.title = titleOf(state.design.spec);
  if ($("#model").value === CHOSEN_TERMS) renderTermList(renameTerms(tickedTerms(), names));
  renderDesign();
  renderRuns();
  renderTruth();
  showPython("design", designCode(state.design.spec));
  showResultsCode();
  renderStudyBar();
  scheduleSave();
  if (state.responses.some((v) => v !== null)) analyseRuns();
}

// ---------------------------------------------------------------- what to learn
let aimChosen = false; // the reader chose the aim, or a study opened with one

const currentAim = () => document.querySelector('input[name="aim"]:checked')?.value ?? "main_quadratic";

/** Set the aim: an optimal design is built for its model, and the card says how many coefficients it has. */
function setAim(aim) {
  const radio = document.querySelector(`input[name="aim"][value="${aim}"]`);
  if (radio) radio.checked = true;
  $("#model_type").value = OPTIMAL_MODEL[currentAim()];
  showAim();
}

function showAim() {
  const factors = readFactors();
  const aim = currentAim();
  const p = aimTerms(factors, aim);
  const screening = !aimChosen && aim === "main_effects" && factors.length >= SCREEN_FROM;
  const box = $("#aim-terms");
  box.replaceChildren();
  // A mixture's model is a Scheffé model of its blends, which its design sets: the aims are for factors with ranges.
  const mixture = mixturePlanned();
  for (const radio of document.querySelectorAll('input[name="aim"]')) radio.disabled = mixture;
  if (mixture) {
    box.append(
      "A mixture is analysed with a Scheffé model of its blends, which its design sets: these aims are for factors with ranges.",
    );
    return;
  }
  if (!factors.length) return;
  box.append(`${p} `, term("coefficients", { key: "coefficient", alone: true }), ", the intercept included.");
  if (screening) box.append(` With ${SCREEN_FROM} factors or more, a study starts by finding which matter.`);
}

// ---------------------------------------------------------------- the design
// The design list, its hints, which settings each design takes and how many factors it accepts
// are in the HTML, so the form is right from the first paint, before Python has loaded.
let budgetEdited = false;
let resolutionEdited = false;
let designChosen = false; // the reader chose the design, or a study opened with one; else Design picks it

/** What a design type is, as the design list says under it. */
const designHint = (designType) =>
  [...$("#design-type").options].find((o) => o.value === designType)?.dataset.hint ?? "";

/** The glossary's entry for each kind of term power is reckoned for. */
const KIND_TERMS = { main: "main-effect", interaction: "interaction", quadratic: "square" };

/** A design type's name as the design list gives it. */
function designLabel(designType) {
  const option = [...$("#design-type").options].find((o) => o.value === designType);
  return option ? option.textContent.trim() : designType;
}

/** Put a design specification (a saved study's, a workbook's, the worked example's) into the brief and the plan. */
function setForm(spec) {
  $("#factors tbody").replaceChildren();
  spec.factors.forEach(addFactor);
  setLimits(spec.constraints ?? []);
  setResponse(spec);
  aimChosen = true;
  setAim(spec.aim ?? (AIMS.includes(spec.model) ? spec.model : defaultAim(spec.factors.length)));
  designChosen = true;
  setDesignForm(spec);
}

/** Put a design's type and settings into the plan, leaving the brief as it is. */
function setDesignForm(spec) {
  // Runs as given are in the list once a study of given runs is open, so the reader can make them again.
  if (spec.design_type === GIVEN) $(`#design-type option[value="${GIVEN}"]`).hidden = false;
  $("#design-type").value = spec.design_type;
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
  if (isMixture(spec.design_type)) {
    // A mixture design stores the Scheffé model it is built for: a lattice's degree, or what extreme vertices are for.
    if (options.model_type === "scheffe_linear" || options.model_type === "scheffe_quadratic") {
      $("#lattice_degree").value = options.model_type === "scheffe_linear" ? "1" : "2";
      $("#mixture_model").value = options.model_type;
    }
  } else if (options.model_type) {
    $("#model_type").value = options.model_type;
  }
  // A resolution or a number of runs other than the page's own pick is kept, as if the reader had chosen it.
  const k = factorCount();
  resolutionEdited = Boolean(options.resolution) && options.resolution !== defaultResolution(k);
  if (options.resolution) $("#resolution").value = String(options.resolution);
  budgetEdited = Boolean(options.budget) && options.budget !== nTerms(k, options.model_type) + SPARE_RUNS;
  if (options.budget) $("#budget").value = options.budget;
  refreshDesign();
}

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
  // Runs as given take categorical factors, and limits on the settings, as they are.
  const numericOnly = (option) => labelled.length > 0 && !CATEGORICAL_DESIGNS.includes(option.value) && option.value !== GIVEN;
  const limited = hasLimits();
  const outsideLimits = (option) => limited && !REGION_DESIGNS.includes(option.value) && option.value !== GIVEN;
  showComponents();
  for (const option of select.options) {
    option.disabled =
      k < Number(option.dataset.minFactors) || k > Number(option.dataset.maxFactors) || numericOnly(option) || outsideLimits(option);
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
  const one = labelled.length === 1;
  const mixture = isMixture(option.value);
  if (option.disabled && mixture && labelled.length) {
    hint.textContent =
      `A mixture's factors are its components, each a proportion, and ${listNames(labelled)} ${one ? "is" : "are"} ` +
      `categorical: remove ${one ? "it" : "them"}, or choose a design for factors with ranges.`;
  } else if (option.disabled && mixture && limited) {
    hint.textContent =
      "A mixture takes no limits on the settings on this page: give each component its lower and upper proportions " +
      "instead, or remove the limits.";
  } else if (option.disabled && mixture) {
    hint.textContent = `${option.textContent} takes ${option.dataset.minFactors} to ${option.dataset.maxFactors} components on this page; you have ${k}.`;
  } else if (option.disabled && limited && labelled.length) {
    hint.textContent =
      `Limits on the settings work with numeric factors only on this page, and ${listNames(labelled)} ${one ? "is" : "are"} ` +
      `categorical: remove the limits, or ${one ? "that factor" : "those factors"}.`;
  } else if (option.disabled && outsideLimits(option)) {
    hint.textContent = `${option.textContent} cannot keep its runs inside your limits on the settings: choose D-optimal or I-optimal.`;
  } else if (option.disabled && numericOnly(option)) {
    hint.textContent =
      `${option.textContent} needs numeric factors, and ${listNames(labelled)} ${labelled.length === 1 ? "is" : "are"} ` +
      "categorical: choose a full factorial, fractional factorial or Plackett-Burman design.";
  } else if (option.disabled) {
    hint.textContent = `${option.textContent} takes ${option.dataset.minFactors} to ${option.dataset.maxFactors} factors on this page; you have ${k}.`;
  } else {
    hint.replaceChildren(`${option.dataset.hint} `, BOOK_PAGES[option.value] ? link(BOOK_PAGES[option.value], "More in the book") : "");
  }
  hint.classList.toggle("error", option.disabled);
  showMakeable(showFactorProblems());
  $("#design-card").hidden = k < 2 && !state.design;
  showAim();
  markPicked();
  markStale();
}

/** Let the run sheet be made only once Python has started, the design takes these factors, and each is ready. */
function showMakeable(unready) {
  const fits = !$("#design-type").selectedOptions[0].disabled; // the design takes this many factors, of these kinds
  $("#generate").disabled = !pythonReady || !fits || unready || factorCount() < 2 || limitsUnready();
}

/**
 * The plan as the form holds it now: the factors' levels, the design and the settings it takes, and the
 * limits on the settings. The run sheet on the page is out of date when this differs from the plan it
 * was made from. A factor's name and units are labels, not part of the plan: a run sheet takes them as
 * they change, and the limits name each factor by its place.
 */
function planKey() {
  const option = $("#design-type").selectedOptions[0];
  const values = {
    n_center_points: $("#n_center_points").value,
    alpha: axialDistance($("#alpha").value, $("#alpha_value").value),
    resolution: $("#resolution").value,
    budget: $("#budget").value,
    n_replicates: $("#n_replicates").value,
    n_blocks: $("#n_blocks").value,
    model_type: $("#model_type").value,
    lattice_degree: $("#lattice_degree").value,
    mixture_model: $("#mixture_model").value,
  };
  const taken = option.dataset.options.split(" ").map((key) => [key, values[key]]);
  const factors = readFactors();
  const levels = factors.map((f) => [f.low, f.high]);
  return JSON.stringify([option.value, levels, Object.fromEntries(taken), limitsKey(factors.map((f) => f.name))]);
}

const planIsStale = () => Boolean(state.design) && state.planKey !== null && planKey() !== state.planKey;

/** Say when the plan has changed since the run sheet was made, in Run and beside Design's button, until it is made again. */
function markStale() {
  const stale = planIsStale();
  $("#design-stale").hidden = !stale;
  $("#plan-note").hidden = !stale;
  $("#plan-note").textContent = stale ? "The run sheet in Run is from an earlier design." : "";
  renderBadges();
  redrawRuns(); // the picture of the limits draws the runs of the run sheet made for them
}

/** The run sheet's runs as the picture of the limits draws them, each setting by its factor's name; none while it is from an earlier plan. */
function picturedRuns() {
  const d = state.design;
  if (!d || planIsStale()) return [];
  return d.rows.map((row) => Object.fromEntries(d.columns.map((name, j) => [name, Number(row[j])])));
}

/** Make the run sheet the plan describes; resolves to true when it is on the page. */
async function generate() {
  cancelCandidates(); // the reader has chosen: the designs can wait until Design is shown again
  let key = planKey();
  const info = readResponseInfo();
  const factors = readFactors();
  const response = responseName(info.label, factors.map((f) => f.name));
  const picked = state.picked?.key === key ? state.picked : null;
  const design = picked && (await pickedDesign(picked, factors));
  if (design) {
    // The design the Design tab made for this plan already: the same runs, with the brief's words on them.
    const was = design.spec.response;
    design.columns = design.columns.map((c) => (c === was ? response : c));
    design.spec = { ...design.spec, factors, response, response_info: info, aim: currentAim() };
    adoptLimits(design.spec.constraints);
    setStatus(`${design.n_runs} runs generated.`);
    madeRunSheet(design, picked.warnings ?? [], planKey());
    return true;
  }
  const button = $("#generate");
  button.disabled = true;
  setStatus("Generating design...", "busy");
  const spec = {
    design_type: $("#design-type").value,
    factors,
    response,
    response_info: info,
    aim: currentAim(),
    n_center_points: Number($("#n_center_points").value),
    alpha: axialDistance($("#alpha").value, $("#alpha_value").value),
    n_replicates: Number($("#n_replicates").value),
    n_blocks: Number($("#n_blocks").value),
    resolution: Number($("#resolution").value),
    budget: Number($("#budget").value),
    model_type: $("#model_type").value,
    lattice_degree: Number($("#lattice_degree").value),
    mixture_model: $("#mixture_model").value,
    constraints: currentLimits(),
    runs: $("#design-type").value === GIVEN ? givenRuns() : undefined,
  };
  const reply = await call("api_make_design", spec);
  button.disabled = false;
  refreshDesign();
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return false;
  }
  setStatus(`${reply.result.n_runs} runs generated.`);
  if (planKey() === key) {
    // The brief takes Python's text for the limits the run sheet was made with, so the two key alike.
    adoptLimits(reply.result.spec.constraints);
    key = planKey();
  }
  madeRunSheet(reply.result, reply.warnings, key);
  return true;
}

/** The runs a design of given runs is made again from: the run sheet's, a setting per factor in the design's order. */
function givenRuns() {
  const held = state.design;
  if (!held) return null;
  const at = held.spec.factors.map((f) => held.columns.indexOf(f.name));
  return held.rows.map((row) => at.map((i) => row[i]));
}

/**
 * The design the Design tab made for this plan, with the factors' names and units the brief has now:
 * renamed by Python if they changed since. Null if Python refuses the names on it (one is its results
 * column's name): the run sheet is then made afresh, with its results named to suit.
 */
async function pickedDesign(picked, factors) {
  const design = structuredClone(picked.design);
  if (!renaming(design.spec.factors, factors)) return design;
  const reply = await call("api_relabel", { design: renameRequest(design), factors });
  return reply.ok ? { ...design, ...reply.result } : null;
}

/** A run sheet made from the plan ``key``: it is the reader's study now, shown in Run. */
function madeRunSheet(design, warnings, key) {
  $("#demo-banner").hidden = true;
  $("#campaign-banner").hidden = true;
  if (new URLSearchParams(location.search).get(EXAMPLE_PARAM) === WORKED) history.replaceState(null, "", location.pathname);
  showDesign(design, warnings);
  state.planKey = key;
  const c = state.comparison;
  const row = c?.result && c.key === candidateKey() ? candidateRows(c.result, c.made).find((r) => r.design_type === design.spec.design_type) : null;
  setModel(analysisModel(design.model, design.spec.aim, Boolean(row?.fits), design.spec.factors));
  clearDraft();
  markStale();
  showPhase("run");
}

/**
 * Make a new set of runs the study: no results yet, nothing simulated, nothing analysed. It is
 * saved in this browser as a study of its own, unless it follows a design made on this page that
 * holds no result the reader typed in: trying one design after another while practising keeps one
 * entry in Your studies, not one per try. A study opened from the list or from a link is never
 * replaced. The worked example passes ``save: false``.
 */
function showDesign(design, warnings = [], { save = true } = {}) {
  flushSave();
  const replaces = state.campaign?.replaceable && !state.sources.includes("entered");
  state.campaign = save ? { ...(replaces ? state.campaign : newCampaign()), title: titleOf(design.spec) } : null;
  state.example = !save;
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
  renderStudyBar();
  scheduleSave();
}

/** Say what the run sheet holds, and anything about its runs to check before they are made. */
function renderDesign(warnings = state.designWarnings ?? []) {
  if (!state.design) return;
  state.designWarnings = warnings;
  const r = state.design;
  const parts = [];
  if (r.facts) {
    const centre = r.facts.center_runs ? `, ${r.facts.center_runs} of them at the center of the ranges` : "";
    const outside = r.outside_range?.length
      ? ` Some runs of ${r.outside_range.join(", ")} lie outside the ranges you entered; check they are safe to run.`
      : "";
    const order = r.spec?.design_type === GIVEN ? "in the order given" : "in random run order";
    parts.push(said("", `${r.n_runs} runs${centre}, ${order}.${outside}`), ...noteList(r.facts.notes));
  } else {
    parts.push(el("p", "", `${r.rows.length} runs, read from the workbook.`));
  }
  parts.push(...warningList(warnings));
  $("#design-summary").replaceChildren(...parts);
  $("#design-out").hidden = false;
  markStale();
}

// ---------------------------------------------------------------- the designs that can learn the brief
/** How long the brief rests before its designs are made and scored: a pause in the typing. */
const CANDIDATE_WAIT = 450;
let candidateTimer = null;
let candidateTicket = 0;
let candidatesRunning = false;

/** What api_candidates is asked for the brief as it is now. */
function candidateRequest() {
  const factors = readFactors();
  const info = readResponseInfo();
  const response = responseName(info.label, factors.map((f) => f.name));
  return { factors, model: currentAim(), response, response_info: info, constraints: currentLimits() };
}

/** What makes two briefs' designs the same: the factors, the aim, the limits, and the noise and change power is for. */
function candidateKey(request = candidateRequest()) {
  const { noise = null, smallest_change: change = null } = request.response_info;
  const limits = request.constraints.map((limit) => positional(limit.expression, request.factors.map((f) => f.name)));
  return JSON.stringify([request.factors, request.model, noise, change, limits]);
}

/** Whether the brief is ready for its designs: two factors or more, each as Python takes it. */
const briefReady = () => factorCount() >= 2 && !factorsUnready && !limitsUnready();

/**
 * Make and score the brief's designs a moment after its last change, while Design is on show, unless
 * they are on the page already. One search at a time: a brief changed during one is searched again
 * once it is back.
 */
function scheduleCandidates() {
  clearTimeout(candidateTimer);
  candidateTimer = null;
  const due =
    state.phase === "plan" && pythonReady && briefReady() && !mixturePlanned() && state.comparison?.key !== candidateKey();
  if (due && !candidatesRunning) {
    $("#candidates").setAttribute("aria-busy", "true");
    candidateTimer = setTimeout(() => findCandidates(), CANDIDATE_WAIT);
  } else if (!candidatesRunning) {
    $("#candidates").removeAttribute("aria-busy"); // a search that was waiting to start is called off
  }
  renderCandidates();
}

/** Stop a search that has not started: the reader made the run sheet, or left Design. */
function cancelCandidates() {
  clearTimeout(candidateTimer);
  candidateTimer = null;
  if (!candidatesRunning) $("#candidates").removeAttribute("aria-busy");
}

async function findCandidates({ search = false } = {}) {
  candidateTimer = null;
  const request = candidateRequest();
  const key = candidateKey(request);
  const mine = ++candidateTicket;
  candidatesRunning = true;
  $("#candidates").setAttribute("aria-busy", "true");
  renderCandidates(search ? "Searching for the optimal designs as well..." : null);
  const reply = await call("api_candidates", { ...request, search });
  candidatesRunning = false;
  $("#candidates").removeAttribute("aria-busy");
  if (mine !== candidateTicket) return;
  state.comparison = reply.ok ? { key, ...reply.result } : { key, error: reply.error };
  if (reply.ok && reply.result.result && !designChosen) {
    const picked = pickCandidate(candidateRows(reply.result.result, reply.result.made));
    if (picked) pickDesign(picked, { byReader: false });
  }
  renderCandidates();
  if ($("#compare-dialog").open) showComparison();
  if (key !== candidateKey()) scheduleCandidates(); // the brief changed while its designs were made
}

/**
 * The designs of the comparison on the page as Design lists them, in a table to pick from. While the
 * brief's own designs are made, the last list stays, dimmed and not to be picked from.
 */
function renderCandidates(busy = null) {
  const box = $("#candidates");
  const c = state.comparison;
  // The designs compared are for factors with ranges: a mixture's own design is made below, with no comparison.
  const mixture = mixturePlanned() && factorCount() >= 2;
  const current = Boolean(c) && c.key === candidateKey() && !busy && !mixture;
  $("#compare").disabled = !current || !c.result;
  $("#compare").hidden = !c?.result || mixture; // nothing to compare in detail before the brief's first designs
  $("#search").hidden = !current || !c.waiting?.length;
  let note = null;
  const holding = limitsHold();
  if (mixture) {
    note = [
      "hint",
      "The designs compared here are for factors with ranges, which a mixture's components are not: make the " +
        "mixture's run sheet below.",
    ];
  } else if (holding) note = ["hint error", holding];
  else if (limitsUnready() && factorCount() >= 2 && !factorsUnready) note = ["hint busy", "Reading the limits on the settings..."];
  else if (current && c.error) note = ["hint error", c.error];
  else if (current) note = null;
  else if (pythonFailure) note = ["hint error", "The designs are made with Python, which could not start here."];
  else if (briefReady() && !pythonReady) note = ["hint", "The designs that can learn this are listed once Python has started."];
  else if (busy || briefReady()) note = ["hint busy", busy ?? "Making and scoring the designs that can learn this..."];
  const waiting = current && !c.result && c.waiting?.length ? [waitingNote(c), ...noteList(c.notes.map(prettyTerm))] : [];
  const listed = !briefReady() || mixture ? [] : c?.result ? candidateList(c, current) : waiting;
  $("#plan-empty").hidden = Boolean(note) || listed.length > 0;
  box.hidden = !note && !listed.length;
  box.dataset.stale = String(!current && !mixture);
  box.replaceChildren(...(note ? [el("p", note[0], note[1])] : []), ...listed);
  if (current) markPicked();
}

/**
 * The comparison's designs as a list to pick from: those that fit the model in a table, sorted by
 * their runs, and which do not. The table's headings say what its numbers are; what spare runs
 * and power mean is a "?" away.
 */
function candidateList(c, current) {
  const rows = candidateRows(c.result, c.made);
  const kinds = [
    ["main", "Main effect"],
    ["interaction", "Interaction"],
    ["quadratic", "Curvature"],
  ].filter(([kind]) => rows.some((r) => r.power[kind] !== null));
  const fitting = rows.filter((r) => r.fits);
  const parts = [];
  if (fitting.length) {
    const t = el("table", "data pick", "");
    const info = readResponseInfo();
    const heading = (word, span = {}) => {
      const th = Object.assign(el("th", "", ""), span);
      th.append(word);
      return th;
    };
    // What spare runs and power are depends on the brief: the model's size, and the change worth finding.
    const spare = term("Spare runs", {
      alone: true,
      definition:
        `The runs left once the model's ${c.result.terms.length} coefficients are fitted: they estimate the ` +
        `run-to-run noise. Fewer than ${SPARE_FOR_NOISE} are marked.`,
    });
    const power = term(`Power to find ${changeWords(info)}`, {
      alone: true,
      definition:
        `The chance that the analysis finds ${changeWords(info)} from a factor's low setting to its high one, at ` +
        `the ${Math.round(100 * c.result.significance_level)}% level.` +
        (powerScale(info) ? "" : " To set your own, give the noise and the change under What you know of it already."),
    });
    const head = el("tr", "", "");
    head.append(heading("Design", { rowSpan: 2 }), heading("Runs", { rowSpan: 2 }), heading(spare, { rowSpan: 2 }));
    const sub = el("tr", "", "");
    if (kinds.length) {
      head.append(heading(power, { colSpan: kinds.length }));
      for (const [kind, words] of kinds) sub.append(heading(term(words, { key: KIND_TERMS[kind], alone: true })));
    }
    const thead = el("thead", "", "");
    thead.append(head, ...(kinds.length ? [sub] : []));
    const tbody = el("tbody", "", "");
    for (const r of fitting) {
      const tr = el("tr", "", "");
      tr.dataset.design = r.design_type;
      const name = el("td", "", "");
      const label = el("label", "", "");
      const radio = el("input", "", "");
      radio.type = "radio";
      radio.name = "candidate";
      radio.value = r.design_type;
      radio.disabled = !current;
      label.append(radio, term(r.label, { definition: designHint(r.design_type) }));
      name.append(label);
      tr.append(name, el("td", "", String(r.runs)), el("td", r.spare < SPARE_FOR_NOISE ? "low" : "", String(r.spare)));
      for (const [kind] of kinds) tr.append(el("td", "", percent(r.power[kind])));
      if (current) tr.onclick = () => pickDesign(r.design_type);
      tbody.append(tr);
    }
    t.append(thead, tbody);
    const wrap = el("div", "scroll", "");
    wrap.append(t);
    parts.push(wrap);
  }
  const unfit = rows.filter((r) => !r.fits);
  if (unfit.length) parts.push(el("p", "hint", `${listNames(unfit.map((r) => r.label))} cannot fit this model.`));
  if (c.waiting?.length) parts.push(waitingNote(c));
  parts.push(...noteList(c.notes.map(prettyTerm)));
  return parts;
}

/** Which designs wait for the reader to ask for their search, as Design says it beside the button. */
function waitingNote(c) {
  const one = c.waiting.length === 1;
  return el(
    "p",
    "hint",
    `The ${listNames(c.waiting.map((w) => w.label))} design${one ? " takes" : "s take"} a while to search for ` +
      `with ${factorCount()} factors.`,
  );
}

/** Mark the design the plan holds in the list of designs that can learn the brief. */
function markPicked() {
  const chosen = $("#design-type").value;
  for (const tr of document.querySelectorAll("#candidates tbody tr")) {
    const on = tr.dataset.design === chosen;
    tr.setAttribute("aria-selected", String(on));
    tr.querySelector("input").checked = on;
  }
}

/**
 * Take a design from the comparison into the plan: its type and settings, as Design made it. Making
 * the run sheet then uses the runs already made, unless the plan changes first.
 */
function pickDesign(designType, { byReader = true } = {}) {
  const item = state.comparison?.made?.find((m) => m.design_type === designType);
  if (!item) return;
  if (byReader) designChosen = true;
  setDesignForm(item.design.spec);
  state.picked = { key: planKey(), design: item.design, warnings: item.warnings ?? [] };
  markPicked();
  markStale();
  renderStudyBar();
  saveDraft();
}

/** Show the comparison in detail, over the page: every design on the nine measures, the FDS curves and correlation maps. */
async function showComparison() {
  const c = state.comparison;
  if (!c?.result) return;
  const dialog = $("#compare-dialog");
  if (!dialog.open) dialog.showModal();
  if (dialog.dataset.key === c.key) {
    resizeWithin(dialog);
    return;
  }
  dialog.dataset.key = c.key;
  renderComparison(c.result, c.made, c.notes);
  showPython(
    "compare",
    compareCode(
      readFactors(),
      c.made.map((m) => ({ label: m.label, design_type: m.design_type, options: m.design.spec.options })),
      c.model,
      c.made[0]?.design.spec.constraints ?? [],
    ),
  );
  await drawComparison(c.result, readFactors().length);
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
    if (m.group === "Feasibility") {
      name.append(term(m.label, { definition: m.description, alone: true }));
    } else {
      const to = link(BOOK_PAGES.measures, "");
      to.append(term(m.label, { definition: m.description }));
      name.append(to);
    }
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
  // A note several designs share, such as the same terms they cannot tell apart, is said once for all of them.
  const shared = new Map();
  for (const label of result.designs) for (const n of result.notes[label]) shared.set(n, [...(shared.get(n) ?? []), label]);
  const designNotes = [...shared].map(([n, labels]) => `${listNames(labels)}: ${prettyTerm(n)}`);
  const es = +result.effect_size.toPrecision(4);
  out.replaceChildren(
    el(
      "p",
      "",
      `Model: ${result.terms.map(prettyTerm).join(", ")}. Power is for a coefficient of ${es} noise standard ` +
        `deviation${es === 1 ? "" : "s"}, at the ${Math.round(100 * result.significance_level)}% level.`,
    ),
    more,
    wrap,
    ...noteList([...notes, ...designNotes].map(prettyTerm)),
  );
}

/** Take a design from the detailed comparison into the plan, and close the comparison. */
function useDesign(item) {
  pickDesign(item.design_type);
  $("#compare-dialog").close();
  setStatus(`The ${item.label} design is chosen: make its run sheet when it is ready.`);
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
    "A low, flat curve predicts with the same precision everywhere; its right end is the worst case, G. A design " +
      "that cannot fit the model has no curve. ",
  );
  caption.append(link(BOOK_PAGES.fds, "Reading an FDS plot"));
  const fdsTitle = el("h3", "", "");
  fdsTitle.append(term("Fraction of design space (FDS)", { key: "fds", alone: true }));
  fds.append(fdsTitle, fdsPlot, caption);
  box.append(fds);
  try {
    await draw(fdsPlot, fdsFigure(result.fds, { region: Boolean(result.region) }), version, 380);
    for (const label of result.designs) {
      const { terms, matrix } = result.correlations[label];
      const card = el("figure", "chart", "");
      const plot = el("div", "plot", "");
      const title = el("h3", "", "");
      title.append(term("Correlation map", { key: "correlation-map", alone: true }), `, ${label}`);
      card.append(title, plot);
      box.append(card);
      await draw(plot, () => correlationFigure(terms, matrix, k), version, 320);
    }
    box.append(
      el(
        "p",
        "hint wide",
        "A darker square is a pair of terms the design finds harder to tell apart; a blank row is a term that " +
          "does not vary in the design.",
      ),
    );
  } catch (err) {
    box.append(el("p", "hint", `The charts could not be drawn (${err.message}); the table above holds the numbers.`));
  }
}

// ---------------------------------------------------------------- run step
/** The response's name and units as the run sheet, the equations and the charts show them. */
function responseWords(spec = state.design.spec) {
  const info = spec.response_info ?? {};
  return { label: info.label || spec.response, units: info.units ?? "" };
}

/**
 * The run sheet, sortable by any column, with the response column as inputs: a reader
 * types each result as it comes in. Simulated values are marked until they are edited. Each
 * heading carries its units under it, and the response's is named as the reader names it.
 */
function renderRuns() {
  const d = state.design;
  showChoicesFor(d);
  $("#run-empty").hidden = true;
  $("#run-out").hidden = false;
  const t = $("#design-table");
  const response = d.spec.response;
  const { label, units } = responseWords();
  const at = d.columns.indexOf(response);
  const rows = d.rows.map((row, i) => Object.assign([...row], { index: i }));
  t._cell = (row, col) => {
    if (col !== at) return null;
    const i = row.index;
    const value = state.responses[i];
    const cls = state.sources[i] === "simulated" ? "sim" : "";
    return `<input class="${cls}" type="number" step="any" inputmode="decimal" data-index="${i}" aria-label="${escape(label)}, run ${i + 1}" value="${value ?? ""}">`;
  };
  const factorUnits = Object.fromEntries(d.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const headings = d.columns.map((c) => (c === response ? label : c));
  const colUnits = d.columns.map((c) => (c === response ? units : (factorUnits[c] ?? "")));
  renderTable(t, headings, rows, "Run order", colUnits, { "Std order": "std-order", "Run order": "run-order", Block: "blocks" });
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
    renderBadges();
    scheduleAnalysis();
    scheduleSave();
  };
  describeSource();
  renderBadges();
}

/**
 * Offer the processes to simulate from, and the models to analyse with, that the run sheet takes: a mixture's
 * (its blends, and the Scheffé models), or those for factors with ranges. A choice the run sheet does not take
 * gives way to the first that it does; the model the design was made for is every run sheet's.
 */
function showChoicesFor(design) {
  const mixture = isMixture(design.spec.design_type);
  for (const [id, fallback] of [["#scenario", mixture ? "blend" : "reactor"], ["#model", ""]]) {
    const select = $(id);
    for (const option of select.options) option.hidden = option.value !== "" && mixture !== ("mixture" in option.dataset);
    if (select.selectedOptions[0]?.hidden) select.value = fallback;
  }
  $("#terms").hidden = $("#model").value !== CHOSEN_TERMS;
  $("#truth-units").textContent = `In ${mixture ? "proportions" : "coded units"}. The analysis estimates these coefficients from the noisy results.`;
  showScenarioHint();
}

/** Say where the results in the run sheet came from. */
function describeSource() {
  const n = state.responses.length;
  const done = state.responses.filter((v) => v !== null).length;
  const simulated = state.sources.filter((s) => s === "simulated").length;
  const sim = state.simulation;
  const { label, units } = responseWords();
  let text = done ? `${done} of ${n} results in.` : `No results yet: type them into the ${label} column as the runs are made.`;
  if (simulated && sim) {
    text =
      `${simulated} of ${n} results simulated from "${sim.label}" (noise ${sim.sigma}${units ? ` ${units}` : ""}, ` +
      `seed ${sim.seed}), in italics: type over any to enter your own.`;
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
    ? `${listNames(quiet)} ${quiet.length === 1 ? "does" : "do"} not affect ${responseWords().label} in this process.`
    : "";
}

let simulationTimer = null;

/** Simulate again a moment after the response's typical value or noise was last typed, rather than on every key. */
function scheduleSimulation() {
  clearTimeout(simulationTimer);
  simulationTimer = setTimeout(simulateResults, 600);
}

async function simulateResults() {
  clearTimeout(simulationTimer);
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

/** Choose where the results come from: entered by the reader, or simulated for practice. */
function selectTab(name) {
  for (const tab of ["simulate", "real"]) {
    $(`#tab-${tab}`).setAttribute("aria-selected", String(tab === name));
    $(`#tab-${tab}`).tabIndex = tab === name ? 0 : -1;
    $(`#panel-${tab}`).hidden = tab !== name;
  }
  state.practising = name === "simulate";
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
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 10_000);
}

/**
 * Download the study as a Jupyter notebook: each step's code, as its "Show the Python code"
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
  state.planKey = planKey();
  state.responses = r.responses;
  state.sources = r.sources;
  state.simulation = null;
  state.example = false;
  state.campaign = { ...newCampaign(false), title: `${titleOf(r.spec)}, from ${file.name}`.slice(0, 200) };
  clearDraft();
  setModel(r.spec.analysis_model ?? $("#model").value);
  renderDesign(reply.warnings);
  renderRuns();
  renderTruth();
  showPython("design", designCode(r.spec));
  showResultsCode();
  renderStudyBar();
  scheduleSave();
  if (state.responses.some((v) => v !== null)) showPhase("learn");
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
  state.analysis = null;
  $("#analysis").hidden = true;
  $("#py-analyse").hidden = true;
  $("#charts").hidden = true;
  $("#charts").replaceChildren();
  $("#analyse-empty").hidden = false;
  $("#to-learn").hidden = true;
  renderBadges();
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
  $("#to-learn").hidden = !reply.ok;
  renderBadges();
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
  const mixture = isMixture(r.spec?.design_type);
  const parts = [
    el("p", "equation", r.equation),
    said("hint", mixture ? "In proportions: each component's share of the blend." : "In coded units."),
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
      {
        Coefficient: "coefficient",
        "Std. error": "std-error",
        "95% interval": "interval",
        "p-value": "p-value",
        "True value": "true-value",
      },
    ),
  ];
  if (r.lenth) {
    const heading = el("h3", "", "");
    heading.append(term("Effects", { key: "effect", alone: true }), ", judged with ", link(BOOK_PAGES.lenth, "Lenth's method"));
    parts.push(
      heading,
      said("hint", `PSE = ${fmt(r.lenth.PSE)}; ME = ${fmt(r.lenth.ME)}; SME = ${fmt(r.lenth.SME)}.`),
      table(
        ["Term", "Effect", "Beyond ME", "Beyond SME"],
        r.lenth.effects.map((e) => [e.label, fmt(e.effect), e.active_ME ? "yes" : "no", e.active_SME ? "yes" : "no"]),
        { Effect: "effect", "Beyond ME": "me", "Beyond SME": "sme" },
      ),
    );
  }
  if (mixture && r.effects?.length) {
    // A component's effect: the change in the fitted response as its share moves along its Cox direction.
    const heading = el("h3", "", "");
    heading.append("Effects along each ", term("Cox direction", { key: "cox-direction", alone: true }));
    parts.push(
      heading,
      table(
        ["Component", "Effect", "From", "To"],
        r.effects.map((e) => [e.component, fmt(e.effect), fmt(e.low), fmt(e.high)]),
      ),
      said("hint", "From and To are the component's proportion at each end of the span the effect is taken over."),
    );
  }
  if (r.anova.length) {
    const heading = el("h3", "", "");
    heading.append(term("ANOVA", { key: "anova", alone: true }));
    parts.push(
      heading,
      table(
        ["Source", "df", "Sum of squares", "F", "p-value"],
        r.anova.map((a) => [a.source, a.df, fmt(a.sum_sq), fmt(a.F), pval(a.p_value)]),
        { df: "df", "Sum of squares": "sum-of-squares", F: "f-value", "p-value": "p-value" },
      ),
    );
  }
  const lof = r.lack_of_fit;
  if (lof && lof.p_value !== null && lof.p_value !== undefined) {
    parts.push(
      said(
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

const tickedTerms = () => [...document.querySelectorAll("#term-list input:checked")].map((b) => JSON.parse(b.value));

/** List every term the design's factors allow, grouped, with `ticked` ticked. */
function renderTermList(ticked) {
  const chosen = new Set(ticked.map((term) => JSON.stringify(term)));
  const groups = new Map();
  for (const { term, label, group } of candidateTerms(state.design.spec.factors)) {
    const tick = el("input", "", "");
    tick.type = "checkbox";
    tick.value = JSON.stringify(term);
    tick.checked = chosen.has(tick.value);
    const item = el("label", "", "");
    item.append(tick, ` ${label}`);
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

/** Put a model into the analysis: a named one, "" for the design's own, or {terms} to tick. */
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
/** The glossary's entry for a chart's title, where it has one. */
const CHART_TERMS = { contour: "response-surface", true_contour: "true-process", residuals: "residuals", half_normal: "half-normal" };

function chartCard(key, caption, wide = false) {
  const card = el("figure", wide ? "chart wide" : "chart", "");
  const plot = el("div", "plot", "");
  const title = el("h3", "", CHART_TERMS[key] ? "" : TITLES[key]);
  if (CHART_TERMS[key]) title.append(term(TITLES[key], { key: CHART_TERMS[key], alone: true }));
  card.append(title, plot, ...(caption ? [el("figcaption", "hint", caption)] : []));
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
  // A mixture's coefficients are the responses of its components and blends, which no Pareto plot of effects ranks.
  const mixture = isMixture(analysis.spec?.design_type);
  const offered = mixture
    ? [["estimates", "Estimates"]]
    : [
        ["estimates", "Estimates"],
        ["pareto", "Pareto plot"],
      ];
  const buttons = offered.map(([view, text]) => {
    const button = el("button", "", "");
    button.append(view === "pareto" ? term(text, { key: "pareto" }) : text);
    button.type = "button";
    button.dataset.view = view;
    return button;
  });
  views.append(...buttons);
  head.append(card.querySelector("h3"), views);
  card.prepend(head);
  if (mixture) {
    card.append(
      el("p", "hint", "No Pareto plot: it ranks the effects of factors with ranges, and a mixture's components move together."),
    );
  }
  const figure = (view) => () =>
    view === "pareto" && !mixture
      ? paretoFigure(analysis.coefficients, simulated, analysis.lenth)
      : coefficientFigure(analysis.coefficients, simulated, undefined, { mixture });
  const show = async (asked, version) => {
    const view = mixture ? "estimates" : asked;
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
  const charts = $("#charts");
  charts.hidden = false;
  const simulated = Boolean(request.truth);
  const coefficients = coefficientCard(state.analysis, simulated);
  charts.replaceChildren(coefficients.card);
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
    // A mixture's response over its blends is drawn here, in the page's colours: a triangle for three components,
    // else the trace along each component's Cox direction.
    const words = { response: responseWords().label, units: responseWords().units };
    const { ternary, cox } = plots;
    if (ternary) {
      const range = ternaryRange(ternary);
      figures.ternary = () => ternaryFigure(ternary, { ...words, range });
      if (ternary.grid.true) figures.true_ternary = () => ternaryFigure(ternary, { ...words, range, values: "true" });
    }
    if (cox) figures.cox = () => coxFigure(cox, words);
    for (const key of ["contour", "true_contour", "ternary", "true_ternary", "cox", "residuals", "half_normal"]) {
      if (!figures[key]) continue;
      const limits = { shaded: plots.shaded, held: plots.held, factors: state.design.spec.factors };
      const caption = typeof CAPTIONS[key] === "function" ? CAPTIONS[key](plots.contour_factors, limits) : CAPTIONS[key];
      const card = chartCard(key, caption);
      charts.append(card.card);
      await draw(card.plot, figures[key], version);
    }
  } catch (err) {
    charts.append(el("p", "hint", `The charts could not be drawn (${err.message}); the tables above hold the numbers.`));
  }
}

async function libraryPlots(request, mine) {
  const reply = await call("api_plots", request);
  if (mine === ticket) setStatus(reply.ok ? "Analysis complete." : `Charts: ${reply.error}`, reply.ok ? "ok" : "error");
  return reply.ok ? reply.result : null;
}

// ---------------------------------------------------------------- the worked example
/** The worked example the build computed into demo.json, fetched once; null in a plain preview of web/. */
let builtExample;
async function fetchExample() {
  if (builtExample !== undefined) return builtExample;
  try {
    const res = await fetch("demo.json", { cache: "no-store" });
    builtExample = res.ok ? await res.json() : null;
  } catch {
    builtExample = null; // a plain local preview has no demo.json
  }
  return builtExample;
}

/**
 * Show the worked example as the build computed it, at once and before Python has started: its
 * brief, the designs that can learn it, its run sheet with results simulated from a known process,
 * and their analysis, in Learn. Without demo.json it is run here instead. Resolves to whether it shows.
 */
async function showBuiltExample() {
  const example = await fetchExample();
  if (!example) {
    if (pythonFailure) return false;
    await runExample();
    return true;
  }
  flushSave();
  $("#campaign-banner").hidden = true;
  setForm(example.inputs.design);
  $("#scenario").value = example.inputs.scenario;
  $("#sigma").value = example.inputs.sigma;
  showScenarioHint();
  showDesign(example.design, [], { save: false });
  state.planKey = planKey();
  selectTab("simulate");
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
  $("#demo-when").textContent = "when this page was built";
  $("#demo-banner").hidden = false;
  if (example.comparison) {
    state.comparison = { key: candidateKey(), ...example.comparison };
    state.picked = null;
    renderCandidates();
  }
  renderStudyBar();
  markStale();
  showPhase("learn");
  if (!pythonReady && !pythonFailure) setStatus("Showing the worked example while Python starts...", "busy");
  await drawCharts({ truth: example.simulation.truth }, ticket, example.plots);
  return true;
}

/** Run the worked example again in this tab: the design made, the results simulated and analysed, live. */
async function runExample() {
  const button = $("#run-example");
  if (!pythonReady) {
    button.disabled = true;
    button.textContent = "Waiting for Python to start...";
    const started = await whenReady.then(
      () => true,
      () => false,
    );
    button.textContent = "Run it again in your browser";
    button.disabled = !started;
    if (!started) return;
  }
  const example = catalogue.demo;
  flushSave();
  setForm(example.design);
  $("#scenario").value = example.scenario;
  $("#sigma").value = example.sigma;
  setModel(example.model);
  showScenarioHint();
  seed = example.seed;
  selectTab("simulate");
  if (!(await generate())) return;
  state.campaign = null; // the worked example, run here, is not saved
  state.example = true;
  history.replaceState(null, "", `${location.pathname}?${EXAMPLE_PARAM}=${WORKED}`);
  $("#demo-version").textContent = `process-improve ${state.version ?? ""}`.trim();
  $("#demo-when").textContent = "in this tab, just now";
  $("#demo-banner").hidden = false;
  await simulateResults();
  showPhase("learn");
}

function showScenarioHint() {
  $("#scenario-hint").textContent = $("#scenario").selectedOptions[0].dataset.hint;
}

// ---------------------------------------------------------------- a new study
/** Clear the page for a new study: an empty brief, no run sheet and no analysis, in Design, at the response. */
function newStudy() {
  flushSave();
  cancelCandidates();
  Object.assign(state, {
    design: null,
    planKey: null,
    responses: [],
    sources: [],
    simulation: null,
    analysis: null,
    comparison: null,
    picked: null,
    campaign: null,
    example: false,
  });
  clearTimeout(analysisTimer);
  setResponse({ response: "" });
  $("#response-known").open = false;
  $("#factors tbody").replaceChildren();
  addFactor();
  addFactor();
  setLimits([]);
  aimChosen = false;
  seed = 1;
  $("#sigma").value = 1;
  designChosen = false;
  setDesignForm({ design_type: "box_behnken", options: { n_center_points: 3, alpha: "face_centered" } });
  setAim(defaultAim(0));
  for (const id of ["#demo-banner", "#campaign-banner", "#run-out", "#design-out", "#share-out"]) $(id).hidden = true;
  $("#run-empty").hidden = false;
  clearAnalysis();
  setModel("");
  for (const key of ["design", "run", "analyse", "compare"]) hidePython(key);
  delete $("#compare-dialog").dataset.key;
  selectTab("real");
  if (location.search) history.replaceState(null, "", location.pathname);
  clearDraft(); // a draft left from before is not this study's
  renderStudyBar();
  refreshLibrary();
  showPhase("plan");
}

// ---------------------------------------------------------------- the draft of a brief
/** A brief not yet made into a run sheet, kept in this browser so a reload or a closed tab does not lose it. */
const DRAFT_KEY = "factorial-draft";
let draftTimer = null;

function saveDraft() {
  if (state.design || state.example) return;
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    const draft = {
      response: responseFields(),
      factors: factorRows().map((tr) => ({
        name: box(tr, "name").value,
        low: box(tr, "low").value,
        high: box(tr, "high").value,
        units: box(tr, "units").value,
      })),
      aim: aimChosen ? currentAim() : null,
      limits: currentLimits(),
      // A mixture's factor table reads as components, blank bounds and all, so the draft keeps the design with it.
      design: mixturePlanned() ? $("#design-type").value : null,
    };
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
    } catch {
      /* storage refused: the brief holds for this visit */
    }
  }, 300);
}

function clearDraft() {
  clearTimeout(draftTimer);
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch {
    /* nothing kept */
  }
}

/** The draft kept in this browser, its text checked field by field; null when there is none. */
function readDraft() {
  let draft;
  try {
    draft = JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "null");
  } catch {
    return null;
  }
  if (!draft || typeof draft !== "object") return null;
  const text = (v, most = 64) => (typeof v === "string" ? v.slice(0, most) : "");
  const r = draft.response && typeof draft.response === "object" ? draft.response : {};
  const response = Object.fromEntries(
    ["label", "units", "goal", "target", "typical", "noise", "smallest_change"].map((k) => [k, text(r[k])]),
  );
  const factors = (Array.isArray(draft.factors) ? draft.factors.slice(0, 12) : [])
    .filter((f) => f && typeof f === "object")
    .map((f) => ({ name: text(f.name), low: text(f.low), high: text(f.high), units: text(f.units, 32) }));
  let limits = [];
  try {
    limits = checkConstraints(draft.limits ?? []);
  } catch {
    /* limits this page did not write: the brief comes back without them */
  }
  const design = isMixture(draft.design) ? draft.design : null;
  return { response, factors, aim: AIMS.includes(draft.aim) ? draft.aim : null, limits, design };
}

/** Put a draft brief back on the page, in Design. */
function restoreDraft(draft) {
  const f = draft.response;
  $("#response").value = f.label;
  $("#response-units").value = f.units;
  const goal = document.querySelector(`input[name="goal"][value="${f.goal}"]`);
  if (goal) goal.checked = true;
  $("#response-target").value = f.target;
  $("#response-typical").value = f.typical;
  $("#response-noise").value = f.noise;
  $("#response-change").value = f.smallest_change;
  $("#response-known").open = Boolean(f.typical || f.noise || f.smallest_change);
  showResponse();
  if (draft.design) {
    $("#design-type").value = draft.design;
    designChosen = true;
  }
  $("#factors tbody").replaceChildren();
  (draft.factors.length ? draft.factors : [{}, {}]).forEach((factor) => addFactor(factor));
  setLimits(draft.limits);
  aimChosen = Boolean(draft.aim);
  setAim(draft.aim ?? defaultAim(factorCount()));
  refreshDesign();
  renderStudyBar();
}

// ---------------------------------------------------------------- saved studies and links
/**
 * A new study's record in this browser: an id of its own, when it began, and whether the next
 * design made on the page may take its place (see showDesign).
 */
function newCampaign(replaceable = true) {
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return { id, created: Date.now(), replaceable };
}

/** A study's name in the list of saved studies: its response when the reader named one, its design and its factors. */
function titleOf(spec) {
  const info = spec.response_info;
  const response = info?.label ? responseTitle(info, spec.response) : "";
  return campaignTitle(designLabel(spec.design_type), spec.factors.map((f) => f.name), response);
}

/** The study on the page, as campaign.js describes one. */
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

/** Save the study on the page in this browser, a moment after its last change. The worked example is not saved. */
function scheduleSave() {
  if (!state.campaign || !state.design) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    saveNow();
  }, 300);
}

/**
 * Write a save that is still waiting for its moment, at once. The page calls this before another
 * study takes this one's place, and when the reader leaves, so a result typed just before is
 * kept. saveNow reads the study before it first waits, so the record is this study's.
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

/** List the studies saved in this browser, each to open or delete; the one on the page is marked. */
async function refreshLibrary() {
  const saved = await savedCampaigns();
  $("#library").hidden = saved.length === 0;
  $("#library-count").textContent = String(saved.length);
  $("#library-list").replaceChildren(...saved.map(libraryItem));
}

function libraryItem(record) {
  const item = el("li", "", "");
  const here = record.id === state.campaign?.id;
  const title = record.title || "A saved study";
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
 * Put a saved or shared study on the page as it was left: the brief, the plan, the runs, their
 * results, the process they were simulated from and the model, in Run, or in Learn once it has
 * results. The analysis follows once Python has started.
 */
async function openCampaign(c, record, message) {
  flushSave();
  $("#demo-banner").hidden = true;
  setForm(c.design.spec);
  state.campaign = { ...record, title: c.title || titleOf(c.design.spec) };
  state.design = { ...c.design };
  state.planKey = planKey();
  state.responses = [...c.responses];
  state.sources = [...c.sources];
  state.simulation = c.simulation;
  state.analysis = null;
  state.example = false;
  state.picked = null;
  clearDraft();
  const { scenario, sigma, seed: kept } = c.simulate;
  if (scenario && [...$("#scenario").options].some((o) => o.value === scenario)) $("#scenario").value = scenario;
  if (sigma) $("#sigma").value = sigma;
  if (kept !== null) seed = kept;
  showScenarioHint();
  selectTab(state.sources.includes("simulated") ? "simulate" : "real");
  setModel(c.model);
  renderDesign([]);
  renderRuns();
  renderTruth();
  clearAnalysis();
  $("#share-out").hidden = true;
  if (c.design.spec.options) showPython("design", designCode(c.design.spec));
  const done = state.responses.some((v) => v !== null);
  if (done) showResultsCode();
  else hidePython("run");
  showBanner(message);
  renderStudyBar();
  showPhase(done ? "learn" : "run");
  scheduleSave(); // the study opened most recently is the one a later visit picks up
  if (!done) return;
  const mine = state.campaign.id;
  await whenReady.then(
    () => (state.campaign?.id === mine ? analyseRuns() : null),
    () => {},
  );
}

/** Resolve to the studies saved in this browser, or to none if the browser has not answered in a moment. */
const savedSoon = () => Promise.race([savedCampaigns(), new Promise((resolve) => setTimeout(() => resolve([]), 2000))]);

/**
 * Open what the address asks for: the worked example, one of the book's examples, or a link's
 * study, the last two saved here as a study of their own; else a brief left as a draft; else the
 * study worked on last in this browser; else a new study, at its response. The address loses a
 * book's example or a link, so a reload opens the saved copy rather than the link again.
 */
async function start() {
  const { hash } = location;
  const params = new URLSearchParams(location.search);
  const book = params.get(BOOK_PARAM) ?? params.get(EXAMPLE_PARAM); // the book's examples and the published ones share one index
  if (book === WORKED) {
    await refreshLibrary();
    if (await showBuiltExample()) return;
  } else if (book !== null || hash.startsWith(LINK_KEY)) {
    history.replaceState(null, "", book !== null ? location.pathname : `${location.pathname}${location.search}`);
    try {
      if (book !== null) {
        await openBookExample(bookName(book));
        return;
      }
      const c = await unpackCampaign(hash.slice(LINK_KEY.length));
      if (await resultsCameBack(c)) return;
      const title = c.title || titleOf(c.design.spec);
      await openCampaign(c, newCampaign(false), `Opened from a link: ${title}. It is saved in this browser now, under Your studies.`);
      return;
    } catch (err) {
      showBanner(err.message, true);
    }
  }
  await refreshLibrary();
  const draft = readDraft();
  if (draft) {
    restoreDraft(draft);
    showPhase("plan");
    return;
  }
  const [last] = await savedSoon();
  if (last) {
    try {
      const c = checkCampaign(last);
      await openCampaign(
        c,
        { id: last.id, created: last.created },
        `Picked up where you left off: ${c.title}, saved ${timeAgo(last.updated)}.`,
      );
      return;
    } catch {
      /* a study this page cannot read: start a new one */
    }
  }
  showPhase("plan");
  const touch = window.matchMedia("(pointer: coarse)").matches;
  if (!touch && !$("#campaign-banner").textContent) $("#response").focus({ preventScroll: true });
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
  await openCampaign(c, record, [...opened, `.${note} It is saved in this browser, under Your studies.`]);
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

/** Copy a link that carries the study on the page, and its results when asked, after the address's "#". */
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
    `${withResults ? "this study and its results" : "this study, without its results"}. The link holds it all: ` +
    "nothing is stored on a server." +
    (url.length > 8000 ? ` At ${url.length} characters, some chat apps cut it: the workbook is the surer way to send it.` : "");
  $("#share-out").hidden = false;
  $("#share-url").select();
}

// ---------------------------------------------------------------- show the Python code
/** The steps whose panels build on each other, in order, and the code each step last showed. */
const STEPS = ["design", "run", "analyse"];
const stepCode = {};

/**
 * Show a step's code in its panel, after the code of the steps it builds on, so what is copied
 * from any panel runs as it is. The comparison stands on its own.
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

/** Show the run's code: the simulation while every result is as simulated, else the results on the page. */
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
  for (const note of notes) {
    const item = el("li", "", "");
    item.append(...marked(note, MARKED));
    list.append(item);
  }
  return [list];
}

/** A paragraph of `text`, its glossary words marked. */
function said(cls, text) {
  const p = el("p", cls, "");
  p.append(...marked(text, MARKED));
  return p;
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

/** A table of results; `terms` names the glossary entry of a column whose heading is a term. */
function table(columns, rows, terms = {}) {
  const wrap = el("div", "scroll", "");
  const t = el("table", "data", "");
  renderTable(t, columns, rows, null, [], terms);
  wrap.append(t);
  return wrap;
}

/**
 * Render a table whose column headers sort it: click once for ascending,
 * again for descending. Each header is a <button> inside the <th>, so it is
 * reachable by keyboard, and aria-sort tells a screen reader the current order.
 * `sortedBy` names the column the rows already arrive sorted on, if any, and `units`
 * gives each column's units, shown under its heading; a table may give
 * `t._cell(row, col)` to draw a cell itself (the run sheet's inputs).
 */
function renderTable(t, columns, rows, sortedBy = null, units = [], terms = {}) {
  const head = columns
    .map((c, i) => {
      const unit = units[i] ? `<span class="col-unit">${escape(units[i])}</span>` : "";
      const name = terms[c] ? `<span class="term" data-term="${terms[c]}">${escape(c)}</span>` : escape(c);
      return `<th aria-sort="${c === sortedBy ? "ascending" : "none"}"><button type="button" class="sort" data-col="${i}">${name}</button>${unit}</th>`;
    })
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
    return own ?? escape(typeof v === "number" ? +v.toPrecision(6) : (v ?? ""));
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
installTerms(GLOSSARY);
// A press and hold is not something a page shows, so a touch screen is told of it once, on a first visit.
const TERMS_HINTED = "factorial-terms-hinted";
try {
  if (window.matchMedia("(pointer: coarse)").matches && !localStorage.getItem(TERMS_HINTED)) {
    localStorage.setItem(TERMS_HINTED, "yes");
    setTimeout(() => toast("Press and hold a dotted word to see what it means."), 1500);
  }
} catch {
  /* storage refused: no hint, rather than one on every visit */
}
// The study bar stays at the top as the page scrolls; the brief beside it starts below it.
new ResizeObserver(([entry]) => {
  document.documentElement.style.setProperty("--bar-height", `${Math.ceil(entry.borderBoxSize[0].blockSize)}px`);
}).observe($("#studybar"));

for (const name of Object.keys(PANES)) $(`#go-${name}`).onclick = () => showPhase(name);
arrowKeys($(".phases"), (tab) => showPhase(tab.id.replace("go-", "")));
arrowKeys($("#step-run .tabs"), (tab) => selectTab(tab.id.replace("tab-", "")));

$("#new-study").onclick = newStudy;
$("#show-example").onclick = () => {
  history.replaceState(null, "", `${location.pathname}?${EXAMPLE_PARAM}=${WORKED}`);
  showBuiltExample();
};
for (const id of ["#response", "#response-units", "#response-target", "#response-typical", "#response-noise", "#response-change"]) {
  $(id).addEventListener("input", responseChanged);
}
$("#goal").addEventListener("change", responseChanged);
// A factor's name and units are labels, taken when the reader leaves the box (Tab, Enter or a click elsewhere), so a
// name half typed reaches no run sheet, note or chart; what it still needs is said as it is typed. Its levels and type
// take effect at once: a label typed for a level makes the factor categorical.
const labelBox = (target) => target.matches('[name="name"], [name="units"]');
$("#factors").addEventListener("input", (e) => {
  if (labelBox(e.target)) {
    showMakeable(showFactorProblems());
    saveDraft();
  } else {
    factorsChanged();
  }
});
$("#factors").addEventListener("change", (e) => {
  if (labelBox(e.target)) factorRenamed(e.target.closest("tr"));
});
$("#add-factor").onclick = () => {
  box(addFactor(), "name").focus();
  factorsChanged();
};
$("#add-categorical").onclick = () => {
  const [low, high] = START_LEVELS[CATEGORICAL];
  box(addFactor({ name: "", low, high, units: "" }), "name").focus();
  factorsChanged();
};
for (const radio of document.querySelectorAll('input[name="aim"]')) {
  radio.onchange = () => {
    aimChosen = true;
    setAim(radio.value);
    refreshDesign();
    if (state.design) {
      state.design.spec = { ...state.design.spec, aim: radio.value };
      scheduleSave();
    }
    scheduleCandidates();
    saveDraft();
  };
}
$("#design-type").onchange = () => {
  designChosen = true;
  refreshDesign();
  // A mixture reads the factor table as components, which the designs compared for factors with ranges are not for.
  scheduleCandidates();
  renderStudyBar();
  saveDraft();
};
for (const id of ["#alpha", "#model_type", "#lattice_degree", "#mixture_model"]) $(id).onchange = refreshDesign;
$("#alpha_value").oninput = refreshDesign; // a typed distance can put every run but the center runs on one sphere
$("#budget").oninput = () => {
  budgetEdited = $("#budget").value !== "";
  markStale();
};
$("#resolution").onchange = () => {
  resolutionEdited = true;
  markStale();
};
for (const id of ["#n_center_points", "#n_replicates", "#n_blocks"]) $(id).addEventListener("input", markStale);
$("#generate").onclick = async () => {
  // While practising, the new runs are simulated from the same process, noise and seed, so the true process
  // and the analysis follow the plan as it now is.
  if ((await generate()) && state.practising) await simulateResults();
};
$("#remake").onclick = $("#generate").onclick;
$("#restore-plan").onclick = () => {
  setForm(state.design.spec);
  state.planKey = planKey();
  markStale();
  renderStudyBar();
};
$("#search").onclick = () => findCandidates({ search: true });
$("#compare").onclick = showComparison;
$("#compare-close").onclick = () => $("#compare-dialog").close();
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
$("#to-learn").onclick = () => showPhase("learn");
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
installLimits({ call, factors: readFactors, changed: limitsChanged, runs: picturedRuns });
addFactor();
addFactor();
setAim(defaultAim(0));
showResponse();
showScenarioHint();
refreshDesign();
renderStudyBar();
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
  for (const id of ["#upload", "#simulate", "#reroll"]) $(id).disabled = false;
  refreshDesign();
  scheduleCandidates();
  resolveReady();
});
