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
  chartImage,
  draw,
  fdsFigure,
  markBest,
  overlayFigure,
  paretoFigure,
  prettyTerm,
  redrawAll,
  resizeWithin,
  sensitivityFigure,
  shareContourScale,
  ternaryFigure,
  ternaryRange,
} from "./charts.js";
import {
  CONFIRMATION,
  CONFIRMATION_RUNS,
  IMPORTANCES,
  GOOD_END,
  UNCLEAR,
  balanceConfirmation,
  balanceLead,
  balanceLeverage,
  balanceRows,
  balanceWhere,
  bestContourCaption,
  confirmationRows,
  confirmationSentences,
  confirmedAt,
  dWords,
  designRows,
  extrapolationSentence,
  flatSentence,
  goalProblem,
  goalWords,
  intervalRows,
  leadSentence,
  limitingSentence,
  climb,
  mixtureSentence,
  movesOn,
  nextRanges,
  noGoalSentence,
  noIntervalSentence,
  num,
  overlayCaption,
  sensitivityCaption,
  settingWords,
  whereSentences,
  whereWords,
  withConfirmationRuns,
  withUnit,
  withoutConfirmationRuns,
} from "./best.js";
import {
  AXIAL_DISTANCE,
  BLOCK,
  CAPTIONS,
  CATEGORICAL,
  CATEGORICAL_DESIGNS,
  CHOSEN_TERMS,
  FULL,
  GIVEN,
  GOALS,
  MAX_RESPONSES,
  MODES,
  NUMERIC,
  OPTIMAL_MODEL,
  REGION_DESIGNS,
  ROMAN,
  SCREEN_FROM,
  SPARE_FOR_NOISE,
  SPARE_RUNS,
  SPACE_FILLING,
  SPACE_FILLING_RUNS,
  defaultBudget,
  TITLES,
  TRADE_OFF_FACTORS,
  TRADE_OFF_RUNS,
  aimTerms,
  analysisModel,
  anyResult,
  axialDistance,
  candidateRows,
  candidateTerms,
  cellBand,
  cellDesign,
  cellResolution,
  cellText,
  changeWords,
  compareCells,
  comparisonOrder,
  completeRuns,
  componentProblem,
  defaultAim,
  defaultResolution,
  describeResponse,
  escape,
  factorType,
  fitStatistics,
  fittedFormula,
  fmt,
  fractionRuns,
  infoOf,
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
  noResults,
  numberOrNull,
  partialRuns,
  partialSentence,
  percent,
  leftOutSentences,
  pickCandidate,
  pickSentence,
  pastedCells,
  placePasted,
  positional,
  powerScale,
  proportionValue,
  pval,
  renameTerms,
  renaming,
  joinResults,
  responseColumns,
  responseLabelProblems,
  responseName,
  responseProblems,
  responseTitle,
  responsesOf,
  responsesTitle,
  runBadge,
  showsVif,
  splitResults,
  standouts,
  STANDOUT_TIE,
  storedResponses,
  tradeOffCell,
  withResponses,
  withUnits,
} from "./logic.js";
import {
  AIMS,
  BOOK_PARAM,
  EXAMPLE_PARAM,
  CAMPAIGN_VERSION,
  LINK_KEY,
  WORKED,
  bookName,
  campaignDone,
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
import { briefHtml, canonicalJson, chartsHtml, report, runSheetHtml, sha256, stampRows } from "./report.js";
import { analyseCode, balanceCode, bestCode, compareCode, cumulative, designCode, forResponse, notebook, resultsCode } from "./python.js";
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
  results: {}, // each response's results, by its column: one per run, in the design's row order, a number or null
  marks: {}, // where each result came from, as results holds them: "simulated", "entered" or null
  shown: null, // the column of the response Learn analyses: the first, or the one its tabs show
  models: {}, // the model each response is analysed with, by its column, as modelChoice gives one
  simulation: null, // what api_simulate returned: the true process, its noise and seed
  practising: false, // the reader chose to practise with simulated results
  analysis: null,
  best: null, // the best settings for the response's goal, on the fitted model: what api_best_settings returned
  balance: null, // the settings that balance the responses with a goal: what api_balance returned
  next: null, // the ways to add runs to the run sheet, {design, offers, notes}: what api_next_runs returned for it
  goals: {}, // what the reader set of each response's goal in the balance, by its column: low, high, importance, shape
  comparison: null, // the designs that can learn the brief: what api_candidates returned, and the brief's key
  picked: null, // a design taken from the comparison, {key, design}, made again only if the plan changes
  plotlyJs: null, // the Plotly.js version the Python plotly package names
  campaign: null, // this browser's record of the study on the page, {id, created, title}; null for the worked example
  example: false, // the worked example is on the page, as the build made it or as run again here
  version: null, // the process-improve release Python runs, once it has started
  runtime: null, // the Python, packages and Pyodide it runs on, which a report names
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

/** The run sheet's responses, the first first: each its column's name and its description. */
const studyResponses = () => (state.design ? responsesOf(state.design.spec) : []);

/** The run sheet's response columns, the first first. */
const resultNames = () => studyResponses().map((r) => r.name);

/** Whether the study measures several responses. */
const several = () => studyResponses().length > 1;

/** How many runs the run sheet holds, confirmation runs too. */
const runCount = () => state.design?.rows.length ?? 0;

/** The model a response is analysed with: Learn's form for the one it shows, else the one chosen for it before. */
const modelOf = (name) => (name === state.shown ? modelChoice() : (state.models[name] ?? ""));

/** Each response's model, by its column, as a study keeps them. */
const modelsByName = () => Object.fromEntries(resultNames().map((name) => [name, modelOf(name)]));

/** The results and their sources as Python takes them: the first response's alone, as always, and the others' beside. */
const sentResults = () => ({ ...splitResults(resultNames(), state.results), ...splitResults(resultNames(), state.marks, "sources") });

/** Whether any result is simulated, and whether every result is, so all came from the known process. */
const someSimulated = () => Object.values(state.marks).some((marks) => marks.includes("simulated"));
const allSimulated = () => Object.values(state.marks).every((marks) => marks.every((m) => m === "simulated"));

/** The process a response's results were simulated from: the first's is the simulation's own, a later one's in more. */
function processOf(name) {
  if (!state.simulation) return null;
  return name === resultNames()[0] ? state.simulation : (state.simulation.more?.[name] ?? null);
}

/** A response's name and units as the run sheet, the equations and the charts show them, by its column. */
function wordsOf(name) {
  const r = studyResponses().find((one) => one.name === name) ?? studyResponses()[0] ?? { name: "y" };
  return { label: r.label || r.name, units: r.units ?? "" };
}
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
  // The designed runs: confirmation runs are checked in Learn, beside the settings they test. A run is complete when
  // every response has its result.
  const confirming = new Set(state.design ? confirmationRows(state.design) : []);
  const complete = completeRuns(state.results, resultNames());
  const designed = complete.filter((_, i) => !confirming.has(i));
  const done = designed.length > 0 && designed.every(Boolean);
  badge("run", state.design ? runBadge(designed.map((ok) => (ok ? true : null))) : "", done ? "done" : "");
  const r2 = state.analysis?.summary?.r_squared;
  badge("learn", r2 === null || r2 === undefined ? "" : `R² ${Number(r2).toFixed(2)}`);
}

/** Name the study in the bar: its responses as the brief describes them, and the design or the factors. */
function renderStudyBar() {
  const info = readResponseInfo();
  const name = info.label ?? state.design?.spec.response ?? "";
  const described = describedResponses();
  $("#study-title").textContent =
    described.length > 1
      ? responsesTitle(described.map(({ info: one }, i) => ({ name: i ? "" : name || "y", ...one })))
      : name
        ? responseTitle(info, name)
        : "A new study";
  const k = readFactors().length;
  const noun = mixturePlanned() ? "component" : "factor";
  const about = state.design
    ? `${designLabel(state.design.spec.design_type)}, ${designRows(state.design).rows.length} runs`
    : k
      ? `${k} ${noun}${k === 1 ? "" : "s"}`
      : "";
  $("#study-about").textContent = about ? `· ${about}` : "";
  renderBadges();
}

// ---------------------------------------------------------------- the responses
/** The brief's response rows: the first is the study's response, and is always there. */
const responseRows = () => [...document.querySelectorAll("#responses tbody tr")];

/** One response row's boxes as the reader left them, as text; the first's also what the reader knows of it. */
function rowFields(tr, first = false) {
  const fields = { label: box(tr, "label").value, units: box(tr, "units").value, goal: box(tr, "goal").value, target: box(tr, "target").value };
  if (!first) return fields;
  return { ...fields, typical: $("#response-typical").value, noise: $("#response-noise").value, smallest_change: $("#response-change").value };
}

/** The first response's boxes as the reader left them, as text. */
const responseFields = () => rowFields(responseRows()[0], true);

/** The first response as the reader describes it: what a design keeps as its ``response_info``. */
const readResponseInfo = () => describeResponse(responseFields());

/**
 * The brief's responses as the reader describes them: each row and its description. A later row with no name is
 * no response yet, as a blank factor row is no factor, unless the run sheet holds a column for it already.
 */
function describedResponses() {
  return responseRows().flatMap((tr, i) => {
    const blank = i > 0 && box(tr, "label").value.trim() === "" && !tr.dataset.column;
    return blank ? [] : [{ tr, info: describeResponse(rowFields(tr, i === 0)) }];
  });
}

/**
 * The brief's responses as a specification names them: each its column's name and its description. With a run
 * sheet, the responses it holds, each keeping its column; ``fresh`` names every one from its words, as a new run
 * sheet does.
 */
function readResponses({ fresh = false } = {}) {
  const described = describedResponses().filter(({ tr }, i) => fresh || !state.design || i === 0 || tr.dataset.column);
  const kept = fresh ? [] : described.map(({ tr }) => tr.dataset.column);
  const names = responseColumns(
    described.map(({ info }) => info.label),
    readFactors().map((f) => f.name),
    kept,
  );
  return described.map(({ info }, i) => ({ name: names[i], ...info }));
}

/** Tie each of the brief's responses to the run sheet's column for it, once the run sheet is made. */
function tieResponses(spec) {
  const names = responsesOf(spec).map((r) => r.name);
  describedResponses().forEach(({ tr }, i) => {
    if (names[i]) tr.dataset.column = names[i];
  });
}

/**
 * Add a response row to the brief, blank or holding a response: a copy of the first row, without its ids. A later
 * row can be removed; the first holds the study's response, and has no remove button.
 */
function addResponse(r = {}, column = null) {
  const tr = responseRows()[0].cloneNode(true);
  for (const input of tr.querySelectorAll("[id]")) input.removeAttribute("id");
  for (const input of tr.querySelectorAll("[aria-invalid]")) input.removeAttribute("aria-invalid");
  const remove = el("button", "link", "✕");
  remove.type = "button";
  remove.setAttribute("aria-label", "Remove response");
  remove.onclick = () => removeResponse(tr);
  tr.lastElementChild.replaceChildren(remove);
  $("#responses tbody").append(tr);
  fillResponse(tr, r, column);
  return tr;
}

/** Put a response into a row: its words, units, goal and target. ``column`` is the run sheet's column it fills. */
function fillResponse(tr, r, column = null) {
  box(tr, "label").value = r.label ?? column ?? "";
  box(tr, "units").value = r.units ?? "";
  box(tr, "goal").value = GOALS[r.goal] ? r.goal : "learn";
  box(tr, "target").value = r.target ?? "";
  if (column) tr.dataset.column = column;
  else delete tr.dataset.column;
}

/**
 * Put a design's responses into the brief: the reader's description of each, else its column's name, and what the
 * reader knows of the first. ``columns`` ties each row to the run sheet's column for it, which keeps its name when
 * its words change; a brief with no run sheet has none.
 */
function setResponse(spec, { columns = true } = {}) {
  const [first, ...more] = responsesOf(spec);
  for (const tr of responseRows().slice(1)) tr.remove();
  fillResponse(responseRows()[0], { label: first.label ?? first.name ?? "", ...infoOf(first) }, columns ? first.name : null);
  for (const r of more) addResponse(r, columns ? r.name : null);
  const info = infoOf(first);
  $("#response-typical").value = info.typical ?? "";
  $("#response-noise").value = info.noise ?? "";
  $("#response-change").value = info.smallest_change ?? "";
  $("#response-known").open = ["typical", "noise", "smallest_change"].some((key) => info[key] !== undefined);
  if (info.noise !== undefined) $("#sigma").value = info.noise;
  showResponse();
}

/** Whether a response still needs something before the run sheet can be made: a name another one has already. */
let responsesUnready = false;

/**
 * Show what the responses' boxes imply: each target box for a goal of hitting one, the first response's units after
 * each number it knows, the line that says the first one sizes the design, the room left for more, and problems.
 */
function showResponse() {
  const rows = responseRows();
  for (const tr of rows) {
    const aimed = box(tr, "goal").value === "target";
    box(tr, "target").disabled = !aimed;
    box(tr, "target").placeholder = aimed ? "" : "none";
  }
  const fields = responseFields();
  for (const unit of document.querySelectorAll(".with-unit .unit")) unit.textContent = fields.units.trim();
  $("#first-sizes").hidden = rows.length < 2;
  $("#known-name").textContent = rows.length < 2 ? "it" : fields.label.trim() || "the first response";
  $("#add-response").disabled = rows.length >= MAX_RESPONSES;
  $("#responses-full").hidden = rows.length < MAX_RESPONSES;
  const named = responseLabelProblems(rows.map((tr) => box(tr, "label").value));
  const said = [...responseProblems(fields), ...named];
  $("#response-problems").textContent = said.join(" ");
  $("#response-problems").hidden = said.length === 0;
  for (const [id, key] of [["#response-noise", "noise"], ["#response-change", "smallest_change"]]) {
    $(id).toggleAttribute("aria-invalid", said.some((s) => s.includes(key === "noise" ? "noise" : "smallest change")));
  }
  const shared = new Set(named.map((sentence) => sentence.split(":")[0].toLowerCase()));
  for (const tr of rows) box(tr, "label").toggleAttribute("aria-invalid", shared.has(box(tr, "label").value.trim().toLowerCase()));
  responsesUnready = named.length > 0;
}

/**
 * The reader changed a response's boxes. The descriptions are the reader's, not the design's: a run sheet already
 * made keeps its columns and runs, and takes the new words, units, goals and numbers, so the title, the run sheet,
 * the charts and the saved study follow. Practice results follow a new typical value or noise of the first response,
 * the analysis a new goal or target, and the designs are scored again for a new noise or smallest change.
 */
function responseChanged() {
  showResponse();
  const info = readResponseInfo();
  if (info.noise !== undefined) $("#sigma").value = info.noise;
  if (state.design) {
    const before = Object.fromEntries(studyResponses().map((r) => [r.name, r]));
    const titled = state.campaign && state.campaign.title === titleOf(state.design.spec);
    state.design.spec = withResponses(state.design.spec, readResponses());
    if (titled) state.campaign.title = titleOf(state.design.spec);
    renderRuns();
    renderTruth();
    renderPicker();
    showResultsCode();
    const after = studyResponses();
    const changed = (keys, which = after) => which.some((r) => keys.some((key) => before[r.name]?.[key] !== r[key]));
    if (someSimulated() && changed(["typical", "noise"], after.slice(0, 1))) scheduleSimulation();
    else if ((state.analysis || state.balance) && changed(["label", "units", "goal", "target"])) scheduleAnalysis(); // the best settings and the balance follow the goals
    scheduleSave();
  }
  scheduleCandidates();
  showMakeable(showFactorProblems());
  renderStudyBar();
  saveDraft();
}

/**
 * The reader left a response's name box. A response added to a run sheet already made takes its column now, named
 * from its words, after the other responses' columns. Its runs have no result yet; while every result is practice,
 * the results are simulated again, the new response's with them.
 */
function responseNamed(tr) {
  const d = state.design;
  const label = box(tr, "label").value.trim();
  if (!d || tr.dataset.column || !label || tr === responseRows()[0] || responsesUnready) return;
  const name = responseName(label, [...d.spec.factors.map((f) => f.name), ...d.columns]);
  tr.dataset.column = name;
  const at = Math.max(...resultNames().map((n) => d.columns.indexOf(n))) + 1;
  const insert = (list, value) => [...list.slice(0, at), value, ...list.slice(at)];
  state.design = { ...d, columns: insert(d.columns, name), rows: d.rows.map((row) => insert(row, null)) };
  state.results[name] = d.rows.map(() => null);
  state.marks[name] = d.rows.map(() => null);
  state.models[name] = modelOf(resultNames()[0]);
  const practice = anyResult(state.results) && allSimulatedBut(name);
  responseChanged();
  renderBadges();
  setStatus(`${label} is on the run sheet now, in a column of its own.`);
  if (practice) simulateResults();
}

/** Whether every result but the response ``name``'s is simulated: practice results, which a new response joins. */
const allSimulatedBut = (name) =>
  Object.entries(state.marks).every(([other, marks]) => other === name || marks.every((m) => m === "simulated"));

/**
 * The reader removed a response. Its row leaves the brief and, when the run sheet holds it, its column leaves the run
 * sheet with its results, and the status line says so. The first response stays: it is the study's own.
 */
function removeResponse(tr) {
  const name = tr.dataset.column;
  const label = box(tr, "label").value.trim() || name;
  tr.remove();
  const d = state.design;
  if (d && name && resultNames().includes(name)) {
    const at = d.columns.indexOf(name);
    const had = (state.results[name] ?? []).filter((v) => v !== null).length;
    state.design = { ...d, columns: d.columns.filter((_, j) => j !== at), rows: d.rows.map((row) => row.filter((_, j) => j !== at)) };
    for (const byName of [state.results, state.marks, state.models]) delete byName[name];
    if (state.simulation?.more) delete state.simulation.more[name];
    if (state.shown === name) {
      state.shown = state.design.spec.response;
      setModel(state.models[state.shown] ?? "");
    }
    responseChanged();
    renderBadges();
    setStatus(had ? `${label} is off the run sheet, with its ${had} result${had === 1 ? "" : "s"}.` : `${label} is off the run sheet.`);
    if (anyResult(state.results)) analyseRuns();
    else clearAnalysis();
    return;
  }
  responseChanged();
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
  return [...resultNames(), ...(d.columns.includes(BLOCK) ? [BLOCK] : [])];
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
  const others = spec.analysis_models ? { analysis_models: spec.analysis_models } : {};
  const limits = spec.constraints ? { constraints: spec.constraints } : {};
  state.design = { ...state.design, ...design, spec: { ...state.design.spec, factors: spec.factors, ...terms, ...others, ...limits } };
  if (simulation) state.simulation = simulation;
  if (titled) state.campaign.title = titleOf(state.design.spec);
  if ($("#model").value === CHOSEN_TERMS) renderTermList(renameTerms(tickedTerms(), names));
  for (const [name, model] of Object.entries(state.models)) {
    if (model && typeof model === "object") state.models[name] = { terms: renameTerms(model.terms, names) }; // each response's terms
  }
  renderDesign();
  renderRuns();
  renderTruth();
  showPython("design", designCode(state.design.spec));
  showResultsCode();
  renderStudyBar();
  scheduleSave();
  if (anyResult(state.results)) analyseRuns();
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
  $("#n_levels").value = String(options.n_levels ?? 2); // a Taguchi array at two levels stores none
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
  budgetEdited = Boolean(options.budget) && options.budget !== defaultBudget(spec.design_type, k, options.model_type);
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

  // A space-filling design is made for no model: its runs are so many a factor, and Learn fits the aim's model.
  const spread = SPACE_FILLING.includes(option.value);
  const p = nTerms(k, spread ? OPTIMAL_MODEL[currentAim()] : $("#model_type").value);
  if (!budgetEdited) $("#budget").value = defaultBudget(option.value, k, $("#model_type").value);
  $("#budget-hint").textContent = spread
    ? `${SPACE_FILLING_RUNS} runs a factor is the usual start; Learn fits the model of your aim, which has ${p} coefficients.`
    : `The model has ${p} coefficients.`;

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
      "categorical: choose a full factorial, fractional factorial, Plackett-Burman or Taguchi design.";
  } else if (option.disabled) {
    hint.textContent = `${option.textContent} takes ${option.dataset.minFactors} to ${option.dataset.maxFactors} factors on this page; you have ${k}.`;
  } else {
    // A design the book has no page on (a Taguchi array) shows its hint alone.
    const book = BOOK_PAGES[option.value];
    hint.replaceChildren(...(book ? [`${option.dataset.hint} `, link(book, "More in the book")] : [option.dataset.hint]));
  }
  hint.classList.toggle("error", option.disabled);
  showTradeOff();
  showMakeable(showFactorProblems());
  $("#design-card").hidden = k < 2 && !state.design;
  showAim();
  markPicked();
  markStale();
}

// ---------------------------------------------------------------- the trade-off grid
// Runs down, factors across, each cell the resolution of the best fraction (logic.js TRADE_OFF), shown with a
// design that takes a resolution. The column of the brief's factors takes a pick; the others are muted, there to
// show what a factor more or fewer would cost. The cell of the design the form now makes is pressed.
let tradeOffDrawn = null; // the factor count, and whether the column takes a pick, the grid was drawn for

/** Show the grid with a design that takes a resolution, drawn for the brief's factors, the form's design pressed. */
function showTradeOff() {
  const grid = $("#trade-off");
  const option = $("#design-type").selectedOptions[0];
  grid.hidden = !option.dataset.options.split(" ").includes("resolution");
  if (grid.hidden) return;
  const k = factorCount();
  const picking = !option.disabled; // a design the factors or the limits rule out takes no pick
  if (tradeOffDrawn !== `${k} ${picking}`) {
    tradeOffDrawn = `${k} ${picking}`;
    drawTradeOff(k, picking);
  }
  const made = fractionRuns(k, Number($("#resolution").value));
  for (const button of grid.querySelectorAll("button[data-runs]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.runs) === made));
  }
}

/** Draw the grid: a button in each cell of the column for k factors that a pick can make, text in every other. */
function drawTradeOff(k, picking) {
  const head = el("thead", "", "");
  const across = el("tr", "", "");
  const factors = el("th", "", "Factors");
  factors.colSpan = TRADE_OFF_FACTORS.length;
  factors.scope = "colgroup";
  across.append(el("td", "", ""), factors);
  const columns = el("tr", "", "");
  for (const [cls, text] of [["", "Runs"], ...TRADE_OFF_FACTORS.map((j) => [j === k ? "here" : "", String(j)])]) {
    columns.append(Object.assign(el("th", cls, text), { scope: "col" }));
  }
  head.append(across, columns);
  const body = el("tbody", "", "");
  for (const runs of TRADE_OFF_RUNS) {
    const tr = el("tr", "", "");
    tr.append(Object.assign(el("th", "", String(runs)), { scope: "row" }));
    for (const j of TRADE_OFF_FACTORS) {
      const cell = tradeOffCell(runs, j);
      const td = el("td", j === k ? "here" : "away", "");
      if (cell !== null) td.dataset.band = cellBand(cell);
      const design = j === k && picking ? cellDesign(runs, k) : null;
      if (design) {
        const what = cell === FULL ? "the full factorial" : `resolution ${ROMAN[cell]}`;
        const button = el("button", "", cellText(cell));
        button.type = "button";
        button.dataset.runs = runs;
        button.setAttribute("aria-label", `${runs} runs, ${k} factors: ${what}`);
        td.append(button);
      } else {
        td.textContent = cellText(cell);
        if (j === k && picking && cell !== null) {
          const asked = cellResolution(cell);
          td.title = `Not offered: the page makes resolution ${ROMAN[asked]} in ${fractionRuns(k, asked)} runs.`;
        }
      }
      tr.append(td);
    }
    body.append(tr);
  }
  $("#trade-off-table").replaceChildren(head, body);
  const move = "Add or remove a factor in the brief to move across.";
  $("#trade-off-hint").textContent = TRADE_OFF_FACTORS.includes(k) ? `The column of your ${k} factors takes a pick. ${move}` : move;
}

/** Take a cell of the grid into the plan: its resolution, or the full factorial. */
function pickCell(runs) {
  const design = cellDesign(runs, factorCount());
  if (!design) return;
  designChosen = true;
  if (design.design_type === "full_factorial") {
    $("#design-type").value = design.design_type;
    refreshDesign();
    $("#design-type").focus(); // the grid goes with the fraction: the focus stays on the choice it made
    return;
  }
  $("#resolution").value = String(design.resolution);
  resolutionEdited = true;
  refreshDesign();
}

/** Let the run sheet be made only once Python has started, the design takes these factors, and each is ready. */
function showMakeable(unready) {
  const fits = !$("#design-type").selectedOptions[0].disabled; // the design takes this many factors, of these kinds
  $("#generate").disabled = !pythonReady || !fits || unready || factorCount() < 2 || limitsUnready() || responsesUnready;
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
    n_levels: $("#n_levels").value,
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
  if (!state.design || planIsStale()) return [];
  const d = designRows(state.design); // the design's runs: a confirmation run is no run of the design
  return d.rows.map((row) => Object.fromEntries(d.columns.map((name, j) => [name, Number(row[j])])));
}

/** Make the run sheet the plan describes; resolves to true when it is on the page. */
async function generate() {
  cancelCandidates(); // the reader has chosen: the designs can wait until Design is shown again
  let key = planKey();
  const factors = readFactors();
  const responses = readResponses({ fresh: true });
  const picked = state.picked?.key === key ? state.picked : null;
  const design = picked && (await pickedDesign(picked, factors));
  if (design) {
    // The design the Design tab made for this plan already: the same runs, with the brief's words on them, and a
    // column for each response where the design had its one response's, last.
    const at = design.columns.indexOf(design.spec.response);
    const without = (list) => list.filter((_, j) => j !== at);
    design.columns = [...without(design.columns), ...responses.map((r) => r.name)];
    design.rows = design.rows.map((row) => [...without(row), ...responses.map(() => null)]);
    design.spec = { ...withResponses(design.spec, responses), factors, aim: currentAim() };
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
    ...storedResponses(responses),
    aim: currentAim(),
    n_center_points: Number($("#n_center_points").value),
    alpha: axialDistance($("#alpha").value, $("#alpha_value").value),
    n_replicates: Number($("#n_replicates").value),
    n_blocks: Number($("#n_blocks").value),
    resolution: Number($("#resolution").value),
    n_levels: Number($("#n_levels").value),
    budget: Number($("#budget").value),
    model_type: $("#model_type").value,
    lattice_degree: Number($("#lattice_degree").value),
    mixture_model: $("#mixture_model").value,
    constraints: currentLimits(),
    runs: $("#design-type").value === GIVEN ? givenRuns() : undefined,
    blocks: $("#design-type").value === GIVEN ? givenBlocks() : undefined,
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
  if (!state.design) return null;
  const held = designRows(state.design); // the confirmation runs are not made again with the design
  const at = state.design.spec.factors.map((f) => held.columns.indexOf(f.name));
  return held.rows.map((row) => at.map((i) => row[i]));
}

/** The block of each run a design of given runs is made again with, as givenRuns lists them; undefined for no blocks. */
function givenBlocks() {
  if (!state.design) return undefined;
  const held = designRows(state.design);
  const at = held.columns.indexOf(BLOCK);
  return at < 0 ? undefined : held.rows.map((row) => row[at]);
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
  tieResponses(design.spec);
  state.planKey = key;
  const c = state.comparison;
  const row = c?.result && c.key === candidateKey() ? candidateRows(c.result, c.made).find((r) => r.design_type === design.spec.design_type) : null;
  const model = analysisModel(design.model, design.spec.aim, Boolean(row?.fits), design.spec.factors);
  setModel(model);
  state.models = Object.fromEntries(resultNames().map((name) => [name, model])); // every response learns what the study aims to
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
  const replaces = state.campaign?.replaceable && !Object.values(state.marks).some((marks) => marks.includes("entered"));
  state.campaign = save ? { ...(replaces ? state.campaign : newCampaign()), title: titleOf(design.spec) } : null;
  state.example = !save;
  state.design = design;
  $("#share-out").hidden = true;
  state.results = noResults(resultNames(), design.rows.length);
  state.marks = noResults(resultNames(), design.rows.length);
  state.shown = resultNames()[0];
  state.models = {};
  state.goals = {};
  state.simulation = null;
  state.analysis = null;
  clearBalance();
  renderDesign(warnings);
  renderRuns();
  renderTruth(); // the process behind the old design's results goes with them
  if ($("#model").value === CHOSEN_TERMS) setModel(CHOSEN_TERMS); // the terms of the new factors
  clearAnalysis();
  renderPicker();
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
    // Runs as given keep the order they came in; a study Learn added runs to holds runs the page put in random order.
    const order = r.spec?.design_type === GIVEN && !r.spec.options?.grown ? "in the order given" : "in random run order";
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
 * their runs, with what each does best of them; which do not; and why the designs not listed are
 * not. The table's headings say what its numbers are; what spare runs and power mean is a "?" away.
 * Until the reader chooses a design, a sentence says why Design picked the one it holds.
 */
function candidateList(c, current) {
  const rows = candidateRows(c.result, c.made);
  const best = standouts(c.result, c.made);
  const leads = rows.some((r) => best[r.design_type]?.length);
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
    const stands = term("Stands out for", {
      alone: true,
      definition:
        `What a design does best of those listed, or within ${Math.round(100 * STANDOUT_TIE)}% of the best: its ` +
        "runs, the precision of its coefficients and predictions, how little its terms correlate, and its power. " +
        "Compare them in detail shows each measure.",
    });
    const head = el("tr", "", "");
    head.append(heading("Design", { rowSpan: 2 }), heading("Runs", { rowSpan: 2 }), heading(spare, { rowSpan: 2 }));
    const sub = el("tr", "", "");
    if (kinds.length) {
      head.append(heading(power, { colSpan: kinds.length }));
      for (const [kind, words] of kinds) sub.append(heading(term(words, { key: KIND_TERMS[kind], alone: true })));
    }
    if (leads) head.append(heading(stands, { rowSpan: 2, className: "stands" }));
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
      if (leads) tr.append(el("td", "stands", (best[r.design_type] ?? []).map((b) => b.words).join(", ")));
      if (current) tr.onclick = () => pickDesign(r.design_type);
      tbody.append(tr);
    }
    t.append(thead, tbody);
    const wrap = el("div", "scroll", "");
    wrap.append(t);
    parts.push(wrap);
  }
  const picked = designChosen ? "" : pickSentence(rows, $("#design-type").value);
  if (picked) parts.push(el("p", "hint picked", picked));
  const unfit = rows.filter((r) => !r.fits);
  if (unfit.length) parts.push(el("p", "hint", `${listNames(unfit.map((r) => r.label))} cannot fit this model.`));
  const categorical = readFactors().filter(isCategorical).map((f) => f.name);
  for (const why of leftOutSentences(c.left_out ?? [], factorCount(), categorical)) parts.push(el("p", "hint left-out", why));
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

/**
 * Mark the design the plan holds in the list of designs that can learn the brief. Once the reader
 * chooses one, the sentence on why Design picked its own goes.
 */
function markPicked() {
  const chosen = $("#design-type").value;
  const note = $("#candidates .picked");
  if (note) note.hidden = designChosen;
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
  const measureHead = el("th", "", "Measure");
  head.append(measureHead, el("th", "", "Better"));
  // The designs that can fit the model come first; those that cannot, their columns mostly blank, go to the right.
  const order = comparisonOrder(result);
  const fits = (label) => Boolean(result.values[label]?.fits);
  const firstUnfit = result.designs[order.find((i) => !fits(result.designs[i]))];
  const cellClass = (label) => (fits(label) ? "" : label === firstUnfit ? "unfit first" : "unfit");
  for (const i of order) {
    const label = result.designs[i];
    const th = el("th", cellClass(label), label);
    const use = el("button", "link", "Use this design");
    use.type = "button";
    use.onclick = () => useDesign(made[i]);
    th.append(el("br", "", ""), use);
    head.append(th);
  }
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
    for (const i of order) {
      const label = result.designs[i];
      tr.append(el("td", cellClass(label), result.display[label][row]));
    }
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
    // The arrows stay in view beside the measure names as the designs scroll: their column starts where the names end.
    t.style.setProperty("--measure-width", `${measureHead.getBoundingClientRect().width}px`);
  });
  comparisonFit.observe(wrap);
  comparisonFit.observe(measureHead);
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
/** The first response's name and units as the run sheet, the equations and the charts show them. */
function responseWords(spec = state.design.spec) {
  const info = spec.response_info ?? {};
  return { label: info.label || spec.response, units: info.units ?? "" };
}

/**
 * The run sheet, sortable by any column, with each response's column as inputs: a reader types
 * each result as it comes in. Simulated values are marked until they are edited. Each heading
 * carries its units under it, and each response's is named as the reader names it.
 */
function renderRuns() {
  const d = state.design;
  showChoicesFor(d);
  $("#run-empty").hidden = true;
  $("#run-out").hidden = false;
  const t = $("#design-table");
  const words = Object.fromEntries(resultNames().map((name) => [name, wordsOf(name)]));
  const at = new Map(resultNames().map((name) => [d.columns.indexOf(name), name]));
  const rows = d.rows.map((row, i) => Object.assign([...row], { index: i }));
  t._cell = (row, col) => {
    const name = at.get(col);
    if (name === undefined) return null;
    const i = row.index;
    const value = state.results[name]?.[i];
    const cls = state.marks[name]?.[i] === "simulated" ? "sim" : "";
    const label = escape(words[name].label);
    return `<input class="${cls}" type="number" step="any" inputmode="decimal" data-index="${i}" data-response="${escape(name)}" aria-label="${label}, run ${i + 1}" value="${value ?? ""}">`;
  };
  t._key = (row, col) => (at.has(col) ? state.results[at.get(col)]?.[row.index] : row[col]); // a result sorts by its value
  const factorUnits = Object.fromEntries(d.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const headings = d.columns.map((c) => words[c]?.label ?? c);
  const colUnits = d.columns.map((c) => words[c]?.units ?? factorUnits[c] ?? "");
  const terms = { "Std order": "std-order", "Run order": "run-order", Block: "blocks", [CONFIRMATION]: "confirmation-run" };
  renderTable(t, headings, rows, "Run order", colUnits, terms);
  t.oninput = (e) => {
    const input = e.target.closest("input[data-index]");
    if (!input) return;
    const i = Number(input.dataset.index);
    const name = input.dataset.response;
    state.results[name][i] = input.value === "" ? null : Number(input.value);
    state.marks[name][i] = input.value === "" ? null : "entered";
    input.classList.remove("sim");
    resultsChanged();
  };
  t.onpaste = pasteResults;
  describeSource();
  renderBadges();
}

/** Follow results typed or pasted into the run sheet: where they came from, the code, the badges, the analysis. */
function resultsChanged() {
  describeSource();
  renderTruth();
  showResultsCode();
  renderBadges();
  scheduleAnalysis();
  scheduleSave();
}

/** The run sheet as a pasted block is placed on it (logic.js placePasted): its run numbers, rows as shown, responses. */
function pasteSheet() {
  const d = state.design;
  const runAt = d.columns.indexOf("Run order");
  const shown = [...$("#design-table").querySelectorAll("tbody tr")].map((tr) => tr.querySelector("input[data-index]"));
  return {
    runs: d.rows.map((row, i) => (runAt >= 0 ? row[runAt] : i + 1)),
    order: shown.filter(Boolean).map((input) => Number(input.dataset.index)),
    names: resultNames(),
    labels: Object.fromEntries(resultNames().map((name) => [name, wordsOf(name).label])),
  };
}

/**
 * Take a block of results pasted into a box of the run sheet, as a spreadsheet copies it (logic.js placePasted): in
 * place, down from the box and across the responses, or matched to the runs by their run numbers or a Run order
 * heading. A block that does not fit is not pasted at all, and the status line says why.
 */
function pasteResults(e) {
  const input = e.target.closest("input[data-index]");
  if (!input) return;
  e.preventDefault(); // a number box would drop a block, or a value written 2,5
  const at = { index: Number(input.dataset.index), name: input.dataset.response };
  const placed = placePasted(pastedCells(e.clipboardData?.getData("text/plain") ?? ""), pasteSheet(), at);
  if (placed.problem) {
    setStatus(placed.problem, "error");
    return;
  }
  for (const { index, name, value } of placed.values) {
    state.results[name][index] = value;
    state.marks[name][index] = "entered";
    const box = $("#design-table").querySelector(`input[data-index="${index}"][data-response="${CSS.escape(name)}"]`);
    box.value = String(value);
    box.classList.remove("sim");
  }
  if (placed.values.length) resultsChanged();
  if (placed.values.length !== 1) setStatus(placed.said); // one value is a box's own paste, and says nothing
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

/**
 * Say where the results in the run sheet came from. With several responses a run counts once every response has its
 * result, and is simulated once every result it holds is.
 */
function describeSource() {
  const names = resultNames();
  const n = runCount();
  const sim = state.simulation;
  if (names.length === 1) {
    const [name] = names;
    const done = state.results[name].filter((v) => v !== null).length;
    const simulated = state.marks[name].filter((s) => s === "simulated").length;
    const { label, units } = responseWords();
    let text = done ? `${done} of ${n} results in.` : `No results yet: type them into the ${label} column as the runs are made, or paste them from a spreadsheet.`;
    if (simulated && sim) {
      text =
        `${simulated} of ${n} results simulated from "${sim.label}" (noise ${sim.sigma}${units ? ` ${units}` : ""}, ` +
        `seed ${sim.seed}), in italics: type over any to enter your own.`;
    }
    $("#run-source").textContent = text;
    return;
  }
  const labels = names.map((name) => wordsOf(name).label);
  const done = completeRuns(state.results, names).filter(Boolean).length;
  const simulated = Array.from({ length: n }, (_, i) => names.every((name) => state.marks[name][i] === "simulated")).filter(Boolean).length;
  let text = done
    ? `${done} of ${n} runs have every result in.`
    : anyResult(state.results)
      ? `No run has every result in yet: ${listNames(labels)} each have a column.`
      : `No results yet: type them into the ${listNames(labels)} columns as the runs are made, or paste them from a spreadsheet.`;
  if (simulated && sim) {
    text =
      `${simulated} of ${n} runs' results simulated from "${sim.label}", each response from a process of its own ` +
      `(seed ${sim.seed}), in italics: type over any to enter your own.`;
  }
  $("#run-source").textContent = text;
}

/** The true process behind the simulated results, behind a reveal: each response's, while some of its results are. */
function renderTruth() {
  const shown = resultNames().filter((name) => processOf(name) && state.marks[name]?.includes("simulated"));
  $("#truth").hidden = !shown.length; // gone with the last simulated result
  if (!shown.length) return;
  const [first, ...more] = shown.map((name) => processOf(name));
  $("#truth-equation").textContent = first.equation;
  $("#truth-more").replaceChildren(...more.map((one) => el("p", "equation", one.equation)));
  // A factor the process leaves out is still a factor of the design: say so, so a new factor visibly counts.
  const quiet = shown.flatMap((name) => {
    const left = processOf(name).inactive ?? [];
    return left.length ? [`${listNames(left)} ${left.length === 1 ? "does" : "do"} not affect ${wordsOf(name).label} in this process.`] : [];
  });
  const note = $("#truth-inactive");
  note.hidden = !quiet.length;
  note.textContent = quiet.join(" ");
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
  const names = resultNames();
  const more = Object.fromEntries(Object.entries(reply.result.more ?? {}).map(([name, one]) => [name, one.responses]));
  state.results = joinResults(names, [...reply.result.responses], more);
  state.marks = Object.fromEntries(names.map((name) => [name, state.results[name].map(() => "simulated")]));
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
  const [first, ...more] = resultNames();
  // Each response's model, unless it is the one the design was made for, is kept in the workbook, so the results are
  // analysed the same way when it comes back.
  const models = Object.fromEntries(more.map((name) => [name, modelOf(name)]).filter(([, model]) => model));
  const reply = await call("api_workbook", {
    ...held(),
    ...sentResults(),
    simulated: someSimulated() ? state.simulation : null,
    model: modelOf(first),
    ...(Object.keys(models).length ? { models } : {}),
  });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const bytes = Uint8Array.from(atob(reply.result.xlsx_base64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  download(blob, `${state.design.spec.design_type}-${runCount()}-runs.xlsx`);
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

/** The terms an analysis fitted, as a formula, when its table shows their VIFs; else null (see python.analyseCode). */
const vifTerms = (analysis) => (showsVif(analysis.coefficients) ? fittedFormula(analysis.coefficients) : null);

/**
 * Each response's analysis and best settings, as the notebook writes them: the one Learn shows as it is on the page,
 * the others fitted now, each with its own model. A response without results has none. Null for a study of one
 * response, whose notebook takes Learn's analysis as it is.
 */
async function eachAnalysis() {
  if (!several()) return null;
  const out = [];
  for (const name of resultNames()) {
    if (!state.results[name].some((v) => v !== null)) continue;
    if (name === state.shown && state.analysis) {
      out.push({ response: name, model: state.analysis.formula ?? state.analysis.model, best: state.best, vifModel: vifTerms(state.analysis) });
      continue;
    }
    const request = analysisRequest(name);
    const analysis = await call("api_analyze_runs", request);
    if (!analysis.ok) continue;
    const best = await call("api_best_settings", request);
    out.push({
      response: name,
      model: analysis.result.formula ?? analysis.result.model,
      best: best.ok ? best.result : null,
      vifModel: vifTerms(analysis.result),
    });
  }
  return out;
}

/**
 * Download the study as a Jupyter notebook: each step's code, as its "Show the Python code"
 * panel shows it, a cell a step, so it runs to the page's design, results and analysis. With
 * several responses, each response's analysis and best settings in turn.
 */
async function downloadNotebook() {
  if (!state.design) return;
  const analyses = await eachAnalysis();
  const book = notebook({
    title: state.campaign?.title ?? titleOf(state.design.spec),
    design: state.design,
    responses: state.results,
    sources: state.marks,
    simulation: state.simulation,
    model: state.analysis ? (state.analysis.formula ?? state.analysis.model) : null,
    vifModel: state.analysis ? vifTerms(state.analysis) : null,
    version: state.version,
    url: `${location.origin}${location.pathname}`,
    best: state.best,
    analyses,
    balance: state.balance,
  });
  const blob = new Blob([`${JSON.stringify(book, null, 1)}\n`], { type: "application/x-ipynb+json" });
  download(blob, `${state.design.spec.design_type}-${runCount()}-runs.ipynb`);
}

// ---------------------------------------------------------------- the study as a report
/** The steps whose code a report gives, in the order they run: each builds on those before it. */
const REPORT_STEPS = ["design", "run", "analyse", "best", "balance"];

/** What a report's copy of the page leaves out: what is there to use the page, and what the page hides. */
const NOT_REPORTED = "[hidden], button:not(.term), input, select, textarea, details.python, .actions, figure.chart";

/** The attributes a report's copy keeps: those that say how a cell spans, where a link goes, and how a part looks. */
const REPORTED_ATTRIBUTES = new Set(["class", "href", "colspan", "rowspan"]);

/**
 * A part of the page as a report holds it, as HTML: a copy without what NOT_REPORTED names, a table's heading that
 * sorts it and a dotted word each left as its text, and with none of the page's ids, ARIA or data. The page set its
 * text as text, so the HTML is safe as it is.
 */
function reportCopy(node) {
  const copy = node.cloneNode(true);
  for (const word of copy.querySelectorAll("th button, .term")) word.replaceWith(word.textContent);
  for (const part of copy.querySelectorAll(NOT_REPORTED)) part.remove();
  for (const part of [copy, ...copy.querySelectorAll("*")]) {
    for (const { name } of [...part.attributes]) if (!REPORTED_ATTRIBUTES.has(name)) part.removeAttribute(name);
  }
  return copy.innerHTML;
}

/** Whether a part of ``root`` is on show there: neither it nor a part between them is hidden. */
function shownWithin(node, root) {
  for (let part = node; part && part !== root.parentElement; part = part.parentElement) if (part.hidden) return false;
  return true;
}

/** The charts the page drew inside ``root``, as a report shows them: each card's heading, its image and its caption. */
async function chartsWithin(root) {
  const charts = [];
  for (const card of root.querySelectorAll("figure.chart")) {
    if (!shownWithin(card, root)) continue;
    const src = await chartImage(card.querySelector(".plot"));
    const title = card.querySelector("h3, h4")?.textContent ?? "";
    if (src) charts.push({ title, caption: card.querySelector("figcaption")?.textContent ?? "", src });
  }
  return charts;
}

/** The aim a design was made for, as Design words it: the choice's heading and what it fits; "" for none. */
function aimWords(aim) {
  const label = document.querySelector(`input[name=aim][value="${aim}"]`)?.closest("label");
  if (!label) return "";
  const words = (part) => label.querySelector(part)?.textContent.replace(/\s+/g, " ").trim() ?? "";
  return `${words("strong")}: ${words(".hint")}`;
}

/** What Learn shows of the response it shows, as a report section: its analysis, charts and best settings. */
async function learnSection() {
  const parts = [reportCopy($("#analysis")), chartsHtml(await chartsWithin($("#charts")))];
  if (!$("#best").hidden) {
    parts.push(`<h3>${$("#h-best").textContent}</h3>`, reportCopy($("#best-out")), chartsHtml(await chartsWithin($("#best"))));
    if (!$("#confirm").hidden) parts.push(reportCopy($("#confirm-out")));
  }
  const heading = several() ? `What the results say of ${wordsOf(state.shown).label}` : "What the results say";
  return { heading, html: parts.join("\n") };
}

/**
 * Learn's analysis of each response with results, as report sections, each as Learn shows it: with several
 * responses, Learn shows each in turn, and then the one it showed. The analysis waiting for typing to pause is
 * made first, so the report holds the results on the page.
 */
async function learnSections() {
  if (analysisTimer !== null) {
    clearTimeout(analysisTimer);
    analysisTimer = null;
    await analyseRuns();
  }
  if (!state.analysis && anyResult(state.results)) await analyseRuns();
  const shown = state.shown;
  const sections = [];
  for (const name of resultNames().filter((one) => state.results[one]?.some((v) => v !== null))) {
    if (name !== state.shown) await showAnalysisOf(name);
    if (state.analysis) sections.push(await learnSection());
  }
  if (state.shown !== shown) await showAnalysisOf(shown);
  return sections;
}

/** The sections a report ends with, after the analysis: the balance of the responses, the next step and the code. */
async function closingSections() {
  const sections = [];
  if (!$("#balance").hidden && state.balance) {
    const html = [reportCopy($("#balance-out")), chartsHtml(await chartsWithin($("#balance")))];
    sections.push({ heading: $("#h-balance").textContent, html: html.join("\n") });
  }
  if (!$("#next").hidden) sections.push({ heading: $("#h-next").textContent, html: reportCopy($("#next-out")) });
  const code = cumulative(...REPORT_STEPS.map((step) => stepCode[step]).filter(Boolean));
  sections.push({
    heading: "The Python code",
    html:
      '<p class="hint">The design, its results and what Learn made of them, so it runs as it is with the releases ' +
      `the stamp names.</p>\n<pre><code>${escape(code)}</code></pre>`,
  });
  return sections;
}

/**
 * The study's inputs, as a report's digest and link hold them: what its link carries, as Factorial reads a link,
 * so a study opened from the link gives the same digest. What the page works out again (a design's aliases, the
 * simulated results) is left out by that reading.
 */
function reportedInputs() {
  const carried = shareable(currentCampaign(), true);
  try {
    return checkCampaign(JSON.parse(JSON.stringify(carried)));
  } catch {
    return carried; // what a link could not carry is still the study's: its digest is of it as it is
  }
}

/**
 * The study as a report: one HTML page with the brief, the design and its run sheet, what Learn makes of the results,
 * the next step and the Python code, under a stamp that says how to get the same numbers again: the releases, the
 * seeds, the SHA-256 of the study's inputs, and a link that opens the study with its results. Resolves to the page and
 * the name it is saved under.
 */
async function makeReport() {
  setStatus("Making the report...", "busy");
  const learned = await learnSections();
  const closing = await closingSections();
  const inputs = reportedInputs();
  const url = `${location.origin}${location.pathname}`;
  const packed = await packCampaign(inputs).catch(() => null); // a study too large for a link has none
  const { spec, rows } = state.design;
  const notes = state.design.facts?.notes ?? [];
  const noted = notes.length ? `<ul>${notes.map((n) => `<li>${escape(n)}</li>`).join("")}</ul>\n` : "";
  const page = report({
    title: state.campaign?.title ?? titleOf(spec),
    stamp: stampRows({
      made: new Date(),
      version: state.version,
      runtime: state.runtime,
      url,
      spec,
      runs: rows.length,
      results: state.results,
      marks: state.marks,
      simulation: someSimulated() ? state.simulation : null,
      sha: await sha256(canonicalJson(inputs)),
      link: packed ? `${url}${LINK_KEY}${packed}` : null,
    }),
    sections: [
      { heading: "The brief", html: briefHtml(spec, { aim: aimWords(spec.aim), mixture: isMixture(spec.design_type) }) },
      {
        heading: `The design: ${designLabel(spec.design_type)}, ${rows.length} runs`,
        html: noted + runSheetHtml(state.design, state.results, state.marks),
      },
      ...learned,
      ...closing,
    ],
  });
  return { page, name: `${spec.design_type}-${rows.length}-runs-report.html` };
}

/** Download the study's report, to keep or to send. */
async function downloadReport() {
  if (!state.design) return;
  const { page, name } = await makeReport();
  download(new Blob([page], { type: "text/html" }), name);
  setStatus("Report downloaded: open it in a browser, or print it to PDF.");
}

/**
 * Open the study's report in a tab of its own, to read, print or share from there. The tab is opened at the click, as
 * a browser allows a new tab only then, and shows the report once it is made; where the browser opens none, the report
 * is downloaded instead.
 */
async function viewReport() {
  if (!state.design) return;
  const tab = window.open("", "_blank");
  if (tab) tab.document.title = "Making the report...";
  const { page, name } = await makeReport();
  if (!tab || tab.closed) {
    download(new Blob([page], { type: "text/html" }), name);
    setStatus("The browser opened no tab, so the report is downloaded: open it in a browser, or print it to PDF.");
    return;
  }
  // The address lasts while this page is open, so the report's tab can be reloaded.
  tab.location.href = URL.createObjectURL(new Blob([page], { type: "text/html" }));
  setStatus("The report is open in a tab of its own: print it to PDF from there, or download it to keep.");
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
  const names = resultNames();
  state.results = joinResults(names, r.responses, r.more_responses);
  state.marks = joinResults(names, r.sources, r.more_sources);
  state.shown = names[0];
  state.simulation = null;
  state.example = false;
  state.campaign = { ...newCampaign(false), title: `${titleOf(r.spec)}, from ${file.name}`.slice(0, 200) };
  clearDraft();
  setModel(r.spec.analysis_model ?? $("#model").value);
  state.models = { [names[0]]: modelChoice(), ...(r.spec.analysis_models ?? {}) };
  state.goals = {};
  clearBalance();
  renderDesign(reply.warnings);
  renderRuns();
  renderTruth();
  renderPicker();
  showPython("design", designCode(r.spec));
  showResultsCode();
  renderStudyBar();
  scheduleSave();
  if (anyResult(state.results)) showPhase("learn");
  await analyseRuns();
}

// ---------------------------------------------------------------- analyse step
let analysisTimer = null;
let ticket = 0;

/** Analyse a moment after the last keystroke, rather than on every one. */
function scheduleAnalysis() {
  clearTimeout(analysisTimer);
  analysisTimer = setTimeout(() => {
    analysisTimer = null;
    analyseRuns();
  }, 700);
}

/** What Learn says while there is nothing to analyse. */
const NOTHING_TO_LEARN = "No results yet: enter them in Run, or practise with simulated ones.";

/** Take the analysis away: its runs or their results are gone. A reply still on its way is dropped. */
function clearAnalysis() {
  ticket += 1;
  state.analysis = null;
  $("#analysis").hidden = true;
  $("#py-analyse").hidden = true;
  $("#charts").hidden = true;
  $("#charts").replaceChildren();
  $("#analyse-empty").textContent = NOTHING_TO_LEARN;
  $("#analyse-empty").hidden = false;
  $("#to-learn").hidden = true;
  $("#learn-report").hidden = true;
  clearBest();
  renderBadges();
}

/**
 * What Python is asked for one response's analysis, its charts and its best settings: the runs, every response's
 * results, the response's own model, and the process it was simulated from while some of its results are. A study
 * of one response asks as it always has; with several, ``response`` names the one analysed.
 */
function analysisRequest(name = state.shown) {
  return {
    ...held(),
    ...splitResults(resultNames(), state.results),
    model: modelOf(name),
    truth: state.marks[name]?.includes("simulated") ? (processOf(name)?.truth ?? null) : null,
    ...(several() ? { response: name } : {}),
  };
}

/**
 * Analyse the response Learn shows: its fit, its best settings and its charts. With two responses or more given a
 * goal, the balance of them follows, whichever response is shown.
 */
async function analyseRuns() {
  if (!state.design || !anyResult(state.results)) {
    clearAnalysis();
    clearBalance();
    setStatus("No results to analyse yet.");
    return;
  }
  renderPicker();
  if (!state.results[state.shown]?.some((v) => v !== null)) {
    // Some other response has results: this one waits for its own.
    clearAnalysis();
    $("#analyse-empty").textContent = `No results for ${wordsOf(state.shown).label} yet: enter them in Run.`;
    setStatus(`No results for ${wordsOf(state.shown).label} yet.`);
    await findBalance();
    return;
  }
  const mine = ++ticket;
  setStatus("Analysing...", "busy");
  const request = analysisRequest();
  const reply = await call("api_analyze_runs", request);
  if (mine !== ticket) return;
  state.analysis = reply.ok ? reply.result : null;
  renderAnalysis(reply);
  if (reply.ok) {
    const best = await findBest(request, mine);
    if (mine !== ticket) return;
    await findNext(); // the next step follows the best settings, and is on the page before the charts are drawn
    if (mine !== ticket) return;
    // The response surface is drawn through the best settings, so the star that marks them sits on the surface drawn.
    await drawCharts(best?.coded ? { ...request, best: best.coded } : request, mine);
    if (mine !== ticket) return;
  }
  await findBalance();
}

function renderAnalysis(reply) {
  const out = $("#analysis");
  out.hidden = false;
  $("#analyse-empty").hidden = true;
  $("#to-learn").hidden = !reply.ok;
  $("#learn-report").hidden = !reply.ok;
  renderBadges();
  if (!reply.ok) {
    setStatus("The results could not be analysed.", "error");
    out.replaceChildren(el("p", "error", reply.error), ...warningList(reply.warnings));
    $("#charts").hidden = true;
    $("#py-analyse").hidden = true;
    clearBest();
    return;
  }
  const r = reply.result;
  const s = r.summary;
  const simulated = r.coefficients.some((c) => c.true_value !== undefined && c.true_value !== null);
  const vif = showsVif(r.coefficients); // a column only when some term is correlated enough for the note
  setStatus("Analysis complete.");
  // The terms fitted, which the design can estimate, from the runs that have a result.
  showPython("analyse", analyseCode(r.formula ?? r.model, designComplete(), shownSpec(), vif ? fittedFormula(r.coefficients) : null));
  const mixture = isMixture(r.spec?.design_type);
  const parts = [
    el("p", "equation", r.equation),
    said("hint", mixture ? "In proportions: each component's share of the blend." : "In coded units."),
    ...noteList(fitLines(r), "fit"),
    ...noteList(r.notes),
    ...warningList(reply.warnings),
    el("h3", "", "Coefficients"),
    table(
      [
        "Term",
        "Coefficient",
        "Std. error",
        "95% interval",
        "p-value",
        ...(vif ? ["VIF"] : []),
        ...(simulated ? ["True value"] : []),
      ],
      r.coefficients.map((c) => [
        c.label,
        fmt(c.coefficient),
        fmt(c.std_error),
        c.std_error === null ? "" : `${fmt(c.ci_low)} to ${fmt(c.ci_high)}`,
        pval(c.p_value),
        ...(vif ? [fmt(c.vif)] : []),
        ...(simulated ? [fmt(c.true_value)] : []),
      ]),
      {
        Coefficient: "coefficient",
        "Std. error": "std-error",
        "95% interval": "interval",
        "p-value": "p-value",
        VIF: "vif",
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

/** The study's specification as the response Learn shows reads it: that response as its ``response``. */
const shownSpec = () => forResponse(state.design.spec, state.shown);

/**
 * The fit at a glance, a line each. With several responses the completed runs are those with every result in, as the
 * Run tab counts them, and a line says how a run with some of its results is fitted: for the responses it has.
 */
function fitLines(r) {
  if (!several()) return fitStatistics(r.summary, r.n_completed, r.pending_runs);
  const names = resultNames();
  const confirming = new Set(confirmationRows(state.design));
  const complete = completeRuns(state.results, names).filter((ok, i) => ok && !confirming.has(i)).length;
  const order = state.design.columns.indexOf("Run order");
  const runs = partialRuns(state.results, names)
    .filter((i) => !confirming.has(i))
    .map((i) => state.design.rows[i][order] ?? i + 1);
  const lines = fitStatistics(r.summary, complete, r.pending_runs);
  if (runs.length) lines.splice(1, 0, partialSentence(runs, wordsOf(state.shown).label, r.n_completed));
  return lines;
}

/**
 * Show a tab for each response above the analysis, with several responses: the one Learn shows is selected. A study
 * of one response has none, and its analysis is no tab's panel.
 */
function renderPicker() {
  const picker = $("#response-picker");
  const panel = $("#response-panel");
  const names = resultNames();
  picker.hidden = names.length < 2;
  if (names.length < 2) {
    picker.replaceChildren();
    for (const attribute of ["role", "aria-labelledby", "tabindex"]) panel.removeAttribute(attribute);
    return;
  }
  if (!names.includes(state.shown)) state.shown = names[0];
  // The tabs are made again only when the responses change, so a tab keeps the keyboard's focus as it is chosen.
  const tabs = [...picker.children];
  if (tabs.map((tab) => tab.dataset.response).join("\n") !== names.join("\n")) {
    picker.replaceChildren(
      ...names.map((name, i) => {
        const tab = el("button", "", "");
        tab.type = "button";
        tab.id = `pick-${i}`;
        tab.dataset.response = name;
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-controls", "response-panel");
        tab.onclick = () => pickResponse(name);
        return tab;
      }),
    );
  }
  for (const tab of picker.children) {
    const on = tab.dataset.response === state.shown;
    tab.textContent = wordsOf(tab.dataset.response).label;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  }
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", `pick-${names.indexOf(state.shown)}`);
}

/** Show a response's analysis in Learn: its own model, its fit, its charts and its best settings. */
function pickResponse(name) {
  if (name === state.shown || !resultNames().includes(name)) return;
  showAnalysisOf(name);
  scheduleSave();
}

/** Put a response's analysis in Learn, with its own model, keeping the model of the one it showed; resolves once it is there. */
async function showAnalysisOf(name) {
  state.models[state.shown] = modelChoice();
  state.shown = name;
  setModel(state.models[name] ?? "");
  renderPicker();
  await analyseRuns();
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
    await drawBestChart();
    if (!plots) return;
    const figures = Object.fromEntries(Object.entries(plots.figures).map(([k, v]) => [k, JSON.parse(v)]));
    if (figures.true_contour) shareContourScale(figures.contour, figures.true_contour);
    const starred = plots.held_best && starBest(figures, plots.contour_factors);
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
      const limits = { shaded: plots.shaded, held: plots.held, factors: state.design.spec.factors, starred };
      const captions = plots.held_best ? { ...CAPTIONS, contour: bestContourCaption } : CAPTIONS;
      const caption = typeof captions[key] === "function" ? captions[key](plots.contour_factors, limits) : captions[key];
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

// ---------------------------------------------------------------- the best settings for the response's goal
/**
 * Whether every run of the design has its result for the response Learn shows (or ``name``): the confirmation runs,
 * which the model is not fitted to, aside.
 */
function designComplete(name = state.shown) {
  const confirming = new Set(state.design ? confirmationRows(state.design) : []);
  return (state.results[name] ?? []).every((v, i) => confirming.has(i) || v !== null);
}

/** Take the best settings away: the analysis they were found on is gone. */
function clearBest() {
  state.best = null;
  $("#best").hidden = true;
  $("#next").hidden = true; // the next step follows the analysis; findNext shows it again with the next one
  hidePython("best");
}

/**
 * Find the best settings for the response's goal on the model just fitted, and show them. Resolves to them, or to
 * null when there are none; ``mine`` is the analysis they belong to, and a later analysis drops this reply.
 */
async function findBest(request, mine) {
  const reply = await call("api_best_settings", request);
  if (mine !== ticket) return null;
  state.best = reply.ok ? reply.result : null;
  renderBest(reply);
  return state.best;
}

/**
 * Show the best settings: what the model predicts there, the settings and where each lies, the two intervals,
 * where the settings sit against the ranges and the limits, whether the runs support the prediction there, how
 * tightly to hold each factor, and the confirmation runs. A goal of learning, or a target not yet typed, gets one line.
 */
function renderBest(reply) {
  if (!reply?.ok || !state.design) {
    clearBest();
    if (reply && !reply.ok) setStatus(`The best settings: ${reply.error}`, "error");
    return;
  }
  const best = reply.result;
  const words = wordsOf(state.shown);
  const factors = state.design.spec.factors;
  $("#best").hidden = false;
  $("#best-chart").hidden = true;
  $("#confirm").hidden = true;
  if (!best.settings) {
    hidePython("best");
    const text = best.mixture ? mixtureSentence(words) : best.flat ? flatSentence(words) : noGoalSentence(best, words);
    $("#best-out").replaceChildren(said("hint", text));
    return;
  }
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const rows = best.settings.map((s) => [
    s.name,
    s.kind === "categorical" ? s.value : withUnit(s.value, units[s.name]),
    whereWords(s, best, words),
  ]);
  const cautions = [extrapolationSentence(best), best.unclear ? UNCLEAR : ""].filter(Boolean);
  $("#best-out").replaceChildren(
    said("lead", leadSentence(best, words)),
    table(["Factor", "Best setting", "Where it lies"], rows),
    knownTable(intervalRows(best, words)),
    ...[noIntervalSentence(best)].filter(Boolean).map((text) => said("hint", text)),
    ...noteList(whereSentences(best, factors, words)),
    ...noteList(cautions, "warnings"),
  );
  renderConfirmation(best, words);
  const model = state.analysis?.formula ?? state.analysis?.model ?? "";
  showPython("best", bestCode(best, shownSpec(), model, designComplete()));
  drawBestChart();
}

/** A small table of what the model knows at the best settings: a row each, its label naming a glossary entry. */
function knownTable(rows) {
  const t = el("table", "data known-table", "");
  const body = t.createTBody();
  for (const { label, term: word, key, value } of rows) {
    const tr = body.insertRow();
    const th = el("th", "", "");
    th.scope = "row";
    th.append(...marked(label, { [word]: key }));
    tr.append(th, el("td", "", value));
  }
  const wrap = el("div", "scroll", "");
  wrap.append(t);
  return wrap;
}

/** Say what the confirmation runs on the run sheet show, and offer to add them at the best settings. */
function renderConfirmation(best, words) {
  const groups = best.confirmation ?? [];
  $("#confirm").hidden = false;
  $("#confirm-out").replaceChildren(
    ...groups.map((group) => {
      const box = el("div", "verdict", "");
      box.dataset.state = group.inside === null ? "waiting" : group.inside ? "inside" : "outside";
      box.append(...confirmationSentences(group, state.design.spec.factors, words).map((text) => said("", text)));
      return box;
    }),
  );
  const add = $("#add-confirmation");
  add.textContent = `Add ${CONFIRMATION_RUNS} confirmation runs at these settings`;
  add.hidden = confirmedAt(state.design, best.confirm_at); // runs there already: their results are what is wanted
  $("#remove-confirmation").hidden = groups.length === 0;
}

/** Draw how tightly to hold each factor, once the version of Plotly.js is known; nothing without a numeric factor. */
async function drawBestChart() {
  const best = state.best;
  const card = $("#best-chart");
  const sensitivity = best?.settings ? best.sensitivity.filter((s) => s.in_model) : [];
  if (!sensitivity.length || !state.plotlyJs) {
    card.hidden = true;
    return;
  }
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const items = best.sensitivity.map((s) => ({ ...s, units: units[s.name] }));
  const figure = () => sensitivityFigure(items, wordsOf(state.shown));
  card.hidden = false;
  const caption = card.querySelector("figcaption");
  caption.textContent = sensitivityCaption(best);
  caption.hidden = !caption.textContent; // the legend, the axis and the heading's definition say the rest
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height);
  } catch {
    card.hidden = true; // the table and the sentences above hold what it shows
  }
}

/**
 * Put a star on each response surface at the best settings, when both factors drawn are in the model, and say
 * whether it did. The surface was drawn through the best settings (Python held the other factors there).
 */
function starBest(figures, pair) {
  const best = state.best;
  if (!best?.settings || !figures.contour) return false;
  const shown = best.settings.filter((s) => pair.includes(s.name));
  if (shown.length !== 2 || !shown.every((s) => s.in_model)) return false;
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const text = `Best settings: ${shown.map((s) => settingWords(s, units[s.name])).join(", ")}`;
  for (const key of ["contour", "true_contour"]) if (figures[key]) markBest(figures[key], best.coded, pair, text);
  return true;
}

/**
 * Add confirmation runs at a setting to the end of the run sheet: the best settings for the response Learn shows, or
 * the settings that balance the responses. They never enter the fit: their average is judged against what each
 * model predicts there. While the results are practice ones, theirs are simulated from the same processes, noise and
 * seeds, which leaves every result before them as it was; else the reader makes them. ``where`` names the setting.
 */
async function addConfirmationRuns(at = state.best?.confirm_at, where = "the best settings") {
  if (!at || !state.design) return;
  const before = state.design.rows.length;
  const { columns, rows } = withConfirmationRuns(state.design, at);
  state.design = { ...state.design, columns, rows };
  const added = rows.length - before;
  for (const name of resultNames()) {
    state.results[name] = [...state.results[name], ...Array(added).fill(null)];
    state.marks[name] = [...state.marks[name], ...Array(added).fill(null)];
  }
  renderRuns();
  renderStudyBar();
  const practice = Boolean(state.simulation) && someSimulated();
  if (practice) await simulateFrom(before);
  showResultsCode();
  scheduleSave();
  await analyseRuns(); // the fit is the same; what is said of the confirmation runs follows them
  setStatus(
    practice
      ? `${added} confirmation runs added at ${where}, their results simulated.`
      : `${added} confirmation runs added at the end of the run sheet: enter their results in Run.`,
  );
}

/** Take the confirmation runs off the run sheet, with their results. */
async function removeConfirmationRuns() {
  if (!state.design) return;
  const { columns, rows, results, marks } = withoutConfirmationRuns(state.design, state.results, state.marks);
  state.design = { ...state.design, columns, rows };
  state.results = results;
  state.marks = marks;
  renderRuns();
  renderStudyBar();
  showResultsCode();
  scheduleSave();
  await analyseRuns();
  setStatus("The confirmation runs are off the run sheet.");
}

// ---------------------------------------------------------------- the settings that balance the responses
let balanceTicket = 0;
let balanceTimer = null;

/** The responses the balance can weigh: those to maximise or minimise, and those with a target typed. */
const aimedResponses = () =>
  studyResponses().filter((r) => r.goal === "maximize" || r.goal === "minimize" || (r.goal === "target" && Number.isFinite(r.target)));

/** Take the balance away: there are fewer than two responses with a goal, or no results to find it from. */
function clearBalance() {
  balanceTicket += 1;
  clearTimeout(balanceTimer);
  state.balance = null;
  $("#balance").hidden = true;
  hidePython("balance");
}

/** What the reader set of each response's goal, as Python takes it: only the responses with a goal, only what was set. */
function sentGoals() {
  const aimed = new Set(aimedResponses().map((r) => r.name));
  return Object.fromEntries(Object.entries(state.goals).filter(([name, goal]) => aimed.has(name) && Object.keys(goal).length));
}

/** Find the balance a moment after the reader last typed in its table, rather than on every key. */
function scheduleBalance() {
  clearTimeout(balanceTimer);
  balanceTimer = setTimeout(findBalance, 500);
}

/**
 * Find the settings that balance the responses with a goal, and show them: with two such responses or more, after each
 * analysis, and when the reader sets a goal's range, importance or shape. Each response is fitted with its own model.
 */
async function findBalance() {
  clearTimeout(balanceTimer);
  if (!state.design || aimedResponses().length < 2 || !anyResult(state.results)) {
    clearBalance();
    return;
  }
  const mine = ++balanceTicket;
  const reply = await call("api_balance", { ...held(), ...splitResults(resultNames(), state.results), models: modelsByName(), goals: sentGoals() });
  if (mine !== balanceTicket) return;
  renderBalance(reply);
}

/**
 * Show the balance: each response's goal in a table to change, how good the balance is, the settings, each response
 * there with its desirability, the response that holds D down most, the overlay of the acceptable ranges, and the
 * confirmation runs that test it, response by response.
 */
function renderBalance(reply) {
  if (reply.ok && !reply.result.balanced) {
    clearBalance();
    return;
  }
  $("#balance").hidden = false;
  if (!reply.ok) {
    // The responses cannot be balanced yet, as when one has no results: say why, and keep the goals as they are.
    state.balance = null;
    $("#balance-out").replaceChildren(said("hint", reply.error));
    for (const id of ["#balance-chart", "#balance-confirm"]) $(id).hidden = true;
    hidePython("balance");
    return;
  }
  const b = reply.result;
  state.balance = b;
  const factors = state.design.spec.factors;
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  renderGoals(b);
  const settings = b.settings.map((s) => [
    s.name,
    s.kind === "categorical" ? s.value : withUnit(s.value, units[s.name]),
    whereWords(s, null, { label: "" }),
  ]);
  const cautions = [extrapolationSentence(b)].filter(Boolean);
  $("#balance-out").replaceChildren(
    said("lead", balanceLead(b)),
    table(["Factor", "Balanced setting", "Where it lies"], settings),
    balanceTable(b),
    ...[limitingSentence(b)].filter(Boolean).map((text) => said("", text)),
    said("hint", balanceLeverage(b)),
    ...noteList(balanceWhere(b)),
    ...noteList(cautions, "warnings"),
  );
  renderBalanceConfirmation(b);
  showPython("balance", balanceCode(b, state.design.spec));
  drawBalanceChart();
}

/** The table of each response at the balanced settings, and the overall desirability D below it. */
function balanceTable(b) {
  const t = el("table", "data balance-table", "");
  const head = t.createTHead().insertRow();
  for (const [text, key] of [["Response", "response"], ["Predicted", null], ["95% prediction interval", "prediction-interval"], ["Desirability, d", "desirability"]]) {
    const th = el("th", "", "");
    th.scope = "col";
    th.append(...(key ? marked(text, { [text.replace(/^95% |, d$/g, "")]: key }) : [text]));
    head.append(th);
  }
  const body = t.createTBody();
  for (const row of balanceRows(b)) {
    const tr = body.insertRow();
    row.forEach((cell, j) => {
      const td = el(j ? "td" : "th", "", cell);
      if (!j) td.scope = "row";
      tr.append(td);
    });
  }
  const foot = t.createTFoot().insertRow();
  const label = el("th", "", "");
  label.scope = "row";
  label.colSpan = 3;
  label.append(...marked("Overall desirability, D", { "Overall desirability": "overall-desirability" }));
  foot.append(label, el("td", "", dWords(b.composite)));
  const wrap = el("div", "scroll", "");
  wrap.append(t);
  return wrap;
}

/**
 * The table of each response's goal in the balance: its acceptable range, its importance and how strongly its good end
 * counts, each a box or a choice. It is made again only when the responses or their goals change, so a box keeps the
 * keyboard while the reader types; each box shows what the balance used, unless the reader is typing in it.
 */
function renderGoals(b) {
  const t = $("#balance-goals");
  const key = JSON.stringify(b.responses.map((r) => [r.name, r.label, r.units, r.goal, r.target]));
  if (t.dataset.key !== key) {
    t.dataset.key = key;
    const head = `<thead><tr><th scope="col">Response</th><th scope="col">Goal</th><th scope="col">Acceptable from</th><th scope="col">to</th><th scope="col">Importance</th><th scope="col">Good end counts</th></tr></thead>`;
    const options = (pairs) => pairs.map(([value, text]) => `<option value="${escape(value)}">${escape(text)}</option>`).join("");
    const rows = b.responses.map((r) => {
      const label = escape(r.label);
      return (
        `<tr data-response="${escape(r.name)}"><th scope="row">${escape(withUnits(r.label, r.units))}</th><td>${escape(goalWords(r))}</td>` +
        `<td data-label="Acceptable from"><input type="number" step="any" name="low" aria-label="${label}: acceptable from"></td>` +
        `<td data-label="to"><input type="number" step="any" name="high" aria-label="${label}: acceptable to"></td>` +
        `<td data-label="Importance"><select name="importance" aria-label="${label}: importance">${options(IMPORTANCES.map((n) => [n, n]))}</select></td>` +
        `<td data-label="Good end counts"><select name="shape" aria-label="${label}: how strongly the good end counts">${options(Object.entries(GOOD_END))}</select></td></tr>`
      );
    });
    t.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;
  }
  for (const r of b.responses) {
    const tr = t.querySelector(`tr[data-response="${CSS.escape(r.name)}"]`);
    for (const name of ["low", "high", "importance", "shape"]) {
      const input = tr.querySelector(`[name="${name}"]`);
      if (input === document.activeElement && input.tagName === "INPUT") continue; // the reader is typing in it
      input.value = name === "low" || name === "high" ? num(r[name]) : String(r[name]);
      input.removeAttribute("aria-invalid");
    }
  }
  $("#balance-problems").hidden = true;
  $("#balance-reset").hidden = Object.keys(sentGoals()).length === 0;
}

/** The reader changed a goal in the balance's table: the balance follows, once what was typed is a goal it can take. */
function goalChanged(e) {
  const tr = e.target.closest("tr[data-response]");
  if (!tr || !state.balance) return;
  const name = tr.dataset.response;
  const r = state.balance.responses.find((one) => one.name === name);
  const typed = (key) => numberOrNull(tr.querySelector(`[name="${key}"]`).value);
  const goal = { ...(state.goals[name] ?? {}) };
  for (const key of ["low", "high"]) {
    if (e.target.name !== key) continue;
    if (typed(key) === null) delete goal[key]; // a box left blank takes the range the results give
    else goal[key] = typed(key);
  }
  if (e.target.name === "importance") goal.importance = Number(e.target.value);
  if (e.target.name === "shape") goal.shape = e.target.value;
  state.goals = { ...state.goals, [name]: goal };
  const problem = goalProblem({ ...r, default_low: r.default_low, default_high: r.default_high }, goal.low ?? null, goal.high ?? null);
  for (const key of ["low", "high"]) tr.querySelector(`[name="${key}"]`).toggleAttribute("aria-invalid", Boolean(problem));
  $("#balance-problems").textContent = problem;
  $("#balance-problems").hidden = !problem;
  scheduleSave();
  if (problem) return;
  if (e.target.tagName === "SELECT") findBalance();
  else scheduleBalance();
}

/** Say what the confirmation runs show, response by response, and offer to add them at the balanced settings. */
function renderBalanceConfirmation(b) {
  const factors = state.design.spec.factors;
  $("#balance-confirm").hidden = false;
  $("#balance-confirm-out").replaceChildren(
    ...b.confirmation.map((group) => {
      const { said: sentences, state: verdict } = balanceConfirmation(group, factors, b.responses);
      const box = el("div", "verdict", "");
      box.dataset.state = verdict;
      box.append(...sentences.map((text) => said("", text)));
      return box;
    }),
  );
  const add = $("#add-balance-confirmation");
  add.textContent = `Add ${CONFIRMATION_RUNS} confirmation runs at these settings`;
  add.hidden = confirmedAt(state.design, b.confirm_at); // runs there already: their results are what is wanted
  $("#remove-balance-confirmation").hidden = b.confirmation.length === 0;
}

/** Draw the overlay of the acceptable ranges, once the version of Plotly.js is known; nothing with one numeric factor. */
async function drawBalanceChart() {
  const b = state.balance;
  const card = $("#balance-chart");
  if (!b?.overlay || !state.plotlyJs) {
    card.hidden = true;
    return;
  }
  const factors = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f]));
  const axes = Object.fromEntries([b.overlay.x, b.overlay.y].map((name) => [name, withUnits(name, factors[name]?.units ?? "")]));
  const figure = () => overlayFigure(b.overlay, b.responses, axes);
  card.hidden = false;
  card.querySelector("figcaption").textContent = overlayCaption(b.overlay, state.design.spec.factors, b.responses);
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height);
  } catch {
    card.hidden = true; // the tables and the sentences above hold what it shows
  }
}

// ---------------------------------------------------------------- the next step
// What the analysis points to next: a new study whose ranges move the way the best settings point (best.js), and the
// ways to add runs to this study (bootstrap.next_runs), each for what its design cannot say yet. Adding runs keeps the
// runs made so far and their results, so the study grows in place, as it does with confirmation runs.

/**
 * Ask Python which ways of adding runs suit the run sheet, once for each run sheet (they follow its runs and its aim,
 * not its results), and show the next step. A mixture's blends take none of them, so it is not asked.
 */
async function findNext() {
  const design = state.design;
  if (design && !isMixture(design.spec.design_type) && state.next?.design !== design) {
    const reply = await call("api_next_runs", held());
    if (state.design !== design) return; // another run sheet took its place meanwhile
    state.next = { design, offers: reply.ok ? reply.result.offers : [], notes: reply.ok ? reply.result.notes : [] };
  }
  renderNext();
}

/** Show the next step once there is an analysis: the moved ranges when the best settings point past them, and the runs to add. */
function renderNext() {
  const design = state.design;
  const next = design && state.next?.design === design ? state.next : { offers: [], notes: [] };
  const best = state.best?.settings && !isMixture(design?.spec.design_type) ? state.best : null;
  const move = Boolean(best) && movesOn(best);
  $("#next").hidden = !design || !state.analysis || (!move && !next.offers.length && !next.notes.length);
  if ($("#next").hidden) return;
  $("#next-out").replaceChildren(...(move ? movedStudy(best) : []), ...(next.offers.length || next.notes.length ? addedRuns(next) : []));
}

/** A factor's range as the next step's table writes it: "150 to 200 degC", or a categorical factor's two levels. */
const rangeWords = (f) => (typeof f.low === "string" ? `${f.low} or ${f.high}` : `${num(f.low)} to ${withUnit(f.high, f.units ?? "")}`);

/** The new study the best settings point to: each factor's range now and moved, and the button that starts it in Design. */
function movedStudy(best) {
  const factors = state.design.spec.factors;
  const moved = nextRanges(best, factors);
  const start = el("button", "button", "Start a new study with these ranges");
  start.type = "button";
  start.id = "next-study";
  start.onclick = () => startMovedStudy(moved);
  const step = el("div", "next__part", "");
  step.append(
    el("h4", "", "A new study further along"),
    said("", `The ranges moved along the ${climb(best)}:`),
    table(["Factor", "Range now", "Next study"], factors.map((f, i) => [f.name, rangeWords(f), rangeWords(moved[i])])),
    el("p", "actions", ""),
  );
  step.lastChild.append(start);
  return [step];
}

/** The ways to add runs to this study, each what it is for and a button, and why any way is not offered. */
function addedRuns(next) {
  const step = el("div", "next__part", "");
  step.append(el("h4", "", "Add runs to this study"));
  if (next.offers.length) {
    step.append(said("hint", "The runs made so far stay as they are, with their results; the runs added go at the end of the run sheet."));
  }
  for (const offer of next.offers) {
    const box = el("div", "offer", "");
    box.dataset.kind = offer.kind;
    const blocked = offer.blocked ? " They go in a block of their own, so a shift in the process since the runs before stays out of the effects." : "";
    const add = el("button", "button", offer.label);
    add.type = "button";
    add.onclick = () => addRuns(offer.kind);
    box.append(said("", `${offer.why}${blocked}`), add);
    step.append(box);
  }
  step.append(...noteList(next.notes));
  return [step];
}

/**
 * Add one way's runs to the end of the run sheet: the runs before keep their results, the runs added have none, and the
 * plan becomes the runs as given, in their blocks, so the run sheet is not out of date. While the results are practice
 * ones, the runs added are simulated from the same processes, noise and seeds, and analysed with the rest; else the
 * reader makes them, in Run.
 */
async function addRuns(kind) {
  if (!state.design) return;
  setStatus("Adding the runs...", "busy");
  const reply = await call("api_add_runs", { ...held(), kind });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const { added, ...grown } = reply.result;
  const before = state.design.rows.length;
  state.design = { ...state.design, ...grown };
  for (const name of resultNames()) {
    state.results[name] = [...state.results[name], ...Array(added).fill(null)];
    state.marks[name] = [...state.marks[name], ...Array(added).fill(null)];
  }
  setDesignForm(grown.spec);
  state.planKey = planKey();
  state.picked = null;
  setModel(""); // the analysis follows the grown design's model
  state.models = Object.fromEntries(resultNames().map((name) => [name, ""]));
  renderDesign(reply.warnings);
  renderRuns();
  renderStudyBar();
  showPython("design", designCode(grown.spec));
  const practice = Boolean(state.simulation) && someSimulated() && (await simulateFrom(before));
  showResultsCode();
  scheduleSave();
  await analyseRuns();
  if (practice) {
    setStatus(`${added} runs added at the end of the run sheet, their results simulated.`);
    return;
  }
  showPhase("run");
  setStatus(`${added} runs added at the end of the run sheet: enter their results in Run.`);
}

/**
 * Simulate the results of the runs from place ``before`` on, from the same processes, noise and seeds, which leaves every
 * result before them as it was. Resolves to whether they were simulated.
 */
async function simulateFrom(before) {
  const { scenario, sigma, seed: drawn } = state.simulation;
  const reply = await call("api_simulate", { ...held(), scenario, sigma, seed: drawn });
  if (!reply.ok) return false;
  const [first] = resultNames();
  for (const name of resultNames()) {
    const simulated = name === first ? reply.result : reply.result.more?.[name];
    if (!simulated || !processOf(name)) continue;
    state.results[name] = state.results[name].map((v, i) => (i >= before ? simulated.responses[i] : v));
    state.marks[name] = state.marks[name].map((s, i) => (i >= before ? "simulated" : s));
  }
  renderRuns();
  return true;
}

/**
 * Start the new study the best settings point to, in Design: this study's brief and design with the moved ranges, its
 * runs not yet made. This study stays in Your studies. Runs as given are made for one study only, so the new one has
 * its design picked for it.
 */
function startMovedStudy(factors) {
  const spec = state.design.spec;
  const given = spec.design_type === GIVEN;
  newStudy();
  setForm({ ...spec, factors, ...(given ? { design_type: "box_behnken", options: { n_center_points: 3 } } : {}) });
  if (given) designChosen = false;
  showPhase("plan");
  saveDraft();
  setStatus("A new study, its ranges moved: check they are safe to run, then make its run sheet.");
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
  const [name] = resultNames(); // the worked example measures one response
  state.results = { [name]: [...example.simulation.responses] };
  state.marks = { [name]: example.simulation.responses.map(() => "simulated") };
  state.plotlyJs = example.plots.plotly_js;
  renderRuns();
  renderTruth();
  showResultsCode();
  state.analysis = example.analysis;
  renderAnalysis({ ok: true, result: example.analysis, warnings: [] });
  if (example.best) {
    state.best = example.best;
    renderBest({ ok: true, result: example.best });
  }
  renderNext(); // the moved ranges, which the best settings give; its Box-Behnken design takes no runs added
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
    results: {},
    marks: {},
    shown: null,
    models: {},
    goals: {},
    simulation: null,
    analysis: null,
    next: null,
    comparison: null,
    picked: null,
    campaign: null,
    example: false,
  });
  clearTimeout(analysisTimer);
  setResponse({ response: "" }, { columns: false });
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
  clearBalance();
  renderPicker();
  setModel("");
  for (const key of ["design", "run", "analyse", "best", "compare"]) hidePython(key);
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
      responses: responseRows()
        .slice(1)
        .map((tr) => rowFields(tr)), // the later responses: each one's name, units, goal and target, as typed
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
  const fields = (r, keys) => Object.fromEntries(keys.map((k) => [k, text(r?.[k])]));
  const r = draft.response && typeof draft.response === "object" ? draft.response : {};
  const response = fields(r, ["label", "units", "goal", "target", "typical", "noise", "smallest_change"]);
  const responses = (Array.isArray(draft.responses) ? draft.responses.slice(0, MAX_RESPONSES - 1) : [])
    .filter((one) => one && typeof one === "object")
    .map((one) => fields(one, ["label", "units", "goal", "target"]));
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
  return { response, responses, factors, aim: AIMS.includes(draft.aim) ? draft.aim : null, limits, design };
}

/** Put a draft brief back on the page, in Design. */
function restoreDraft(draft) {
  const f = draft.response;
  for (const tr of responseRows().slice(1)) tr.remove();
  fillResponse(responseRows()[0], f);
  for (const r of draft.responses) addResponse(r);
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
  const responses = responsesOf(spec);
  const response = responses.length > 1 ? responsesTitle(responses) : info?.label ? responseTitle(info, spec.response) : "";
  return campaignTitle(designLabel(spec.design_type), spec.factors.map((f) => f.name), response);
}

/**
 * The study on the page, as campaign.js describes one: the first response's results, sources and model as a study of
 * one response has always kept them, and with several, each later response's beside them, by its column.
 */
function currentCampaign() {
  const { spec, columns, rows, model, facts, outside_range: outside } = state.design;
  const [first, ...more] = resultNames();
  return {
    v: CAMPAIGN_VERSION,
    title: state.campaign?.title ?? titleOf(spec),
    design: { spec, columns, rows, model, n_runs: rows.length, facts: facts ?? null, outside_range: outside ?? [] },
    ...sentResults(),
    simulation: state.simulation,
    simulate: { scenario: $("#scenario").value, sigma: Number($("#sigma").value), seed },
    model: modelOf(first),
    ...(more.length ? { more_models: Object.fromEntries(more.map((name) => [name, modelOf(name)])) } : {}),
    ...(more.length && Object.keys(sentGoals()).length ? { goals: sentGoals() } : {}),
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
  const about = el("span", "hint", `${progress(doneRuns(record))}, saved ${timeAgo(record.updated)}`);
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

/** A saved study's runs as its progress counts them: a run is done when every response has its result. */
function doneRuns(record) {
  try {
    return campaignDone(checkCampaign(record)).map((done) => (done ? true : null));
  } catch {
    return record.responses ?? []; // a study this page cannot read: its first response's results, as kept
  }
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
  const names = resultNames();
  state.results = joinResults(names, [...c.responses], structuredClone(c.more_responses ?? {}));
  state.marks = joinResults(names, [...c.sources], structuredClone(c.more_sources ?? {}));
  state.shown = names[0];
  state.models = { [names[0]]: c.model, ...(c.more_models ?? {}) };
  state.goals = structuredClone(c.goals ?? {});
  clearBalance();
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
  selectTab(someSimulated() ? "simulate" : "real");
  setModel(c.model);
  renderDesign([]);
  renderRuns();
  renderTruth();
  clearAnalysis();
  renderPicker();
  $("#share-out").hidden = true;
  if (c.design.spec.options) showPython("design", designCode(c.design.spec));
  const done = anyResult(state.results);
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
  if (!anyResult({ first: c.responses, ...(c.more_responses ?? {}) })) return false;
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
  const { added, kept, ...merged } = mergeResults(here, c);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  let message = added
    ? `Added ${plural(added, "result")} from a link to ${here.title}, kept in this browser.`
    : `The link's results are already in ${here.title}, kept in this browser.`;
  if (kept) message += ` ${plural(kept, "run")} already had a different result here, which ${kept === 1 ? "was" : "were"} kept.`;
  await openCampaign({ ...here, ...merged }, { id: match.id, created: match.created }, message);
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
const BUILDS_ON = { design: [], run: ["design"], analyse: ["design", "run"], best: ["design", "run", "analyse"], balance: ["design", "run"] };
const stepCode = {};

/**
 * Show a step's code in its panel, after the code of the steps it builds on, so what is copied
 * from any panel runs as it is. The balance fits each response itself, so it builds on the design
 * and its results alone. The comparison stands on its own.
 */
function showPython(key, code) {
  stepCode[key] = code;
  const panel = $(`#py-${key}`);
  panel.querySelector("code").textContent = BUILDS_ON[key]
    ? cumulative(...[...BUILDS_ON[key], key].map((step) => stepCode[step]).filter(Boolean))
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
  const simulated = state.simulation && allSimulated();
  showPython("run", resultsCode(state.design, state.results, simulated ? state.simulation : null));
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
    const key = (row) => (t._key ? t._key(row, col) : row[col]);
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
// The responses' boxes take effect as they are typed; a response added to a run sheet takes its column when the reader
// leaves its name box, so a name half typed names no column.
$("#responses").addEventListener("input", responseChanged);
$("#responses").addEventListener("change", (e) => {
  if (e.target.name === "label") responseNamed(e.target.closest("tr"));
});
for (const id of ["#response-typical", "#response-noise", "#response-change"]) $(id).addEventListener("input", responseChanged);
$("#add-response").onclick = () => {
  box(addResponse(), "label").focus();
  responseChanged();
};
arrowKeys($("#response-picker"), (tab) => pickResponse(tab.dataset.response));
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
  showTradeOff();
  markStale();
};
$("#trade-off-table").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-runs]");
  if (button) pickCell(Number(button.dataset.runs));
});
$("#n_levels").onchange = markStale;
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
  if (someSimulated()) simulateResults();
};
$("#sigma").onchange = () => {
  if (someSimulated()) simulateResults();
};
$("#tab-simulate").onclick = () => selectTab("simulate");
$("#tab-real").onclick = () => selectTab("real");
$("#to-learn").onclick = () => showPhase("learn");
$("#add-confirmation").onclick = () => addConfirmationRuns();
$("#add-balance-confirmation").onclick = () => addConfirmationRuns(state.balance?.confirm_at, "the balanced settings");
$("#remove-balance-confirmation").onclick = removeConfirmationRuns;
$("#balance-goals").addEventListener("input", goalChanged);
$("#balance-reset").onclick = () => {
  state.goals = {};
  scheduleSave();
  findBalance();
};
$("#remove-confirmation").onclick = removeConfirmationRuns;
$("#download").onclick = downloadWorkbook;
$("#notebook").onclick = downloadNotebook;
$("#report").onclick = downloadReport;
$("#learn-report-button").onclick = downloadReport;
$("#report-view").onclick = viewReport;
$("#learn-report-view").onclick = viewReport;
$("#share").onclick = shareLink;
$("#model").onchange = () => {
  if ($("#model").value === CHOSEN_TERMS && state.design) startTermList();
  else modelBefore = $("#model").value;
  $("#terms").hidden = $("#model").value !== CHOSEN_TERMS || !state.design;
  if (state.shown) state.models[state.shown] = modelChoice(); // the model of the response Learn shows
  analyseRuns();
  scheduleSave();
};
$("#term-list").onchange = () => {
  if (state.shown) state.models[state.shown] = modelChoice();
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
  state.runtime = catalogue.runtime ?? null;
  pythonReady = true;
  for (const id of ["#upload", "#simulate", "#reroll"]) $(id).disabled = false;
  refreshDesign();
  scheduleCandidates();
  resolveReady();
});
