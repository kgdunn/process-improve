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
  blendOverlayFigure,
  coefficientFigure,
  correlationFigure,
  coxFigure,
  curveFigure,
  chartImage,
  draw,
  fdsFigure,
  levelsFigure,
  markBest,
  markedRun,
  onRunMarked,
  FOUND,
  FOUND_BLEND,
  IN_WINDOWS,
  SUREST_BLEND,
  WHAT_IF_SETTINGS,
  overlayFigure,
  paretoFigure,
  effectsFigure,
  mapFigure,
  campaignMapFigure,
  campaignRunsFigure,
  plotPoint,
  pictureLines,
  pictureReach,
  pictureWords,
  pressRun,
  prettyTerm,
  redrawAll,
  restOnRun,
  sensitivityFigure,
  shareContourScale,
  targetName,
  windowName,
  COLOUR_MAPS,
  COLOUR_MAP_WORDS,
  colourMap,
  colourMapFlipped,
  shownColours,
  setColourMap,
  ternaryFigure,
  ternaryRange,
  tradeOffFigure,
} from "./charts.js";
import {
  BLEND_SENSITIVITY,
  BLEND_UNCLEAR,
  CONFIRMATION,
  NO_BLEND_OVERLAY,
  CONFIRMATION_RUNS,
  apartRows,
  failedRows,
  withFailed,
  IMPORTANCES,
  GOOD_END,
  UNCLEAR,
  WHAT_IF_BACK,
  balanceConfirmation,
  balanceLead,
  balanceLeverage,
  balanceRows,
  balanceWhere,
  bestContourCaption,
  blendExtrapolation,
  blendLead,
  blendOverlayCaption,
  blendSentences,
  blendWhere,
  confirmationSentences,
  confirmedAt,
  dWords,
  designRows,
  pathRows,
  extrapolationSentence,
  flatSentence,
  goalProblem,
  goalWords,
  intervalRows,
  leadSentence,
  answerWords,
  learnChecks,
  limitingSentence,
  climb,
  movesOn,
  nextRanges,
  noGoalSentence,
  noIntervalSentence,
  num,
  overlayCaption,
  sensitivityCaption,
  settingWords,
  weighingSentence,
  whereSentences,
  whereWords,
  withConfirmationRuns,
  withPathRuns,
  withUnit,
  withoutConfirmationRuns,
  withoutPathRuns,
  columnsAt,
  curveAt,
  frontier,
  meetsLimits,
  shortfall,
  keptAs,
  predictAt,
  scenarioProblem,
  settingsFit,
  settingsWords,
  sliderStep,
  PROFILE_POINTS,
  MAP_POINTS,
  mapPair,
  surfaceOf,
  profileOf,
  profilePieces,
  profileRange,
  profileWords,
  rangeWords as intervalWords, // this file's own rangeWords writes a factor's range
  runsAt,
  insideWindow,
  dotPiles,
  runsWords,
  solveSentences,
  surestSentence,
  targetCaption,
  targetLead,
  targetRows,
  targetShare,
  windowCaption,
  windowChance,
  windowLead,
  windowLevels,
  whatIfRows,
  whatIfSentences,
  whatIfStart,
} from "./best.js";
import {
  AXIAL_DISTANCE,
  BLOCK,
  WHOLE_PLOT,
  CAPTIONS,
  CHARTS,
  MORE_CHARTS,
  chartHeight,
  correlationWords,
  THE_BLOCKS,
  termsKey,
  PICKED_WORDS,
  simplerWords,
  alsoWords,
  findingNumbers,
  findingWords,
  shareWords as wholeShare,
  runSheetCsv,
  asideOf,
  asideWords,
  fittedResults,
  toggleAside,
  CATEGORICAL,
  CATEGORICAL_DESIGNS,
  COMPARISON_TESTS,
  LEVEL_CHECKS,
  PAIRED_TESTS,
  CHOSEN_TERMS,
  FULL,
  GIVEN,
  GOALS,
  INTERACTION_MODELS,
  MAX_FACTORS,
  MAX_INTERACTION_FACTORS,
  MAX_LEVEL_RUNS,
  MAX_OPTIMAL_TERMS,
  MAX_REPLICATES,
  MAX_RESPONSES,
  MODES,
  NUMERIC,
  OPTIMAL_MODEL,
  REGION_DESIGNS,
  REPOSITORY,
  ROMAN,
  SCREEN_FROM,
  SLOW_SEARCH,
  SPARE_FOR_NOISE,
  SPARE_RUNS,
  SPACE_FILLING,
  omarsHint,
  SPACE_FILLING_RUNS,
  defaultBudget,
  TRADE_OFF_FACTORS,
  TRADE_OFF_RUNS,
  aimTerms,
  analysisModel,
  anyResult,
  axialDistance,
  candidateRows,
  candidateTerms,
  mainEffectsBrought,
  broughtWords,
  cellBand,
  cellDesign,
  cellResolution,
  cellText,
  changeWords,
  compareCells,
  csvOf,
  comparedRow,
  comparisonOrder,
  comparisonSentence,
  checkSentences,
  severalSentence,
  completeRuns,
  componentProblem,
  defaultAim,
  defaultResolution,
  describeResponse,
  escape,
  factorType,
  boxCoxSentence,
  bySignificance,
  fitStatistics,
  fittedFormula,
  fmt,
  fractionRuns,
  highestResolution,
  infoOf,
  interactionOf,
  isBlank,
  categoricalFactor,
  isCategorical,
  isMixture,
  isSeveral,
  SEVERAL_LEVEL_DESIGNS,
  LABEL_MOST,
  levelsOf,
  MAX_LEVELS,
  levelProblem,
  levelValue,
  listNames,
  mixtureProblem,
  nTerms,
  dayWords,
  momentWords,
  shortCommit,
  versionText,
  searchSeconds,
  waitWords,
  nameProblem,
  planVerdict,
  modelKey,
  modelTermsWords,
  namedModelTerms,
  printedAgrees,
  printedLike,
  printedValue,
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
  pastedCurves,
  curvesSaid,
  curveSaid,
  pastedTarget,
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
  DEFAULT_RESPONSE,
  responseTitle,
  questionLine,
  ladderStepOf,
  ladderSteps,
  offPlanWords,
  questionSentence,
  responsesOf,
  responsesTitle,
  runBadge,
  screeningSentence,
  screensOnly,
  showsVif,
  spacePoints,
  splitResults,
  spreadSentences,
  standouts,
  STANDOUT_TIE,
  storedResponses,
  tradeOffBeyond,
  tradeOffCell,
  withResponses,
  withUnits,
} from "./logic.js";
import {
  CAMPAIGN_VERSION,
  AIMS,
  BOOK_PARAM,
  BOOK_TAB,
  EXAMPLE_PARAM,
  LINK_KEY,
  REF_PARAM,
  SCENARIOS_HELD,
  SHOW_PARAM,
  WORKED,
  bookName,
  campaignDone,
  campaignTitle,
  bestResult,
  pathVerdict,
  bestSoFar,
  campaignRuns,
  campaignSoFar,
  checkBuiltOn,
  priceWords,
  roundsOf,
  checkBookExample,
  checkCampaign,
  CURVE_POINTS_HELD,
  checkConstraints,
  designKey,
  indexedName,
  mergeResults,
  packCampaign,
  partName,
  progress,
  sameCampaign,
  shareable,
  studyFormat,
  studySettings,
  timeAgo,
  unpackCampaign,
} from "./campaign.js";
import { fileStem, readStudyFile, runsDigest, studyFile } from "./study-file.js";
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
import { drawFieldChart } from "./field-chart.js";
import { RUN_ORDER_SEED, briefHtml, canonicalJson, chartsHtml, report, runSheetHtml, sha256, stampRows } from "./report.js";
import { mapCode, campaignMapCode, campaignRunsCode, analyseCode, balanceCode, bestCode, compareCode, comparisonFigureCode, correlationCode, cumulative, curveCode, designCode, figureCode, forResponse, notebook, overlayCode, ownFigureCode, resultsCode, sensitivityCode, targetCode, targetOverlayCode, tradeOffCode, whatIfCode, windowCode, windowOverlayCode } from "./python.js";
import { forgetCampaign, saveCampaign, savedCampaigns } from "./store.js";
import { CONTEXT, DESIGN_CONTEXT, GLOSSARY, MARKED } from "./glossary.js";
import { installTerms, marked, term } from "./terms.js";
import { closeOnBackdrop } from "./dialogs.js";
import { installImport, openedWords } from "./import.js";
import { pythonLine } from "./worker-rpc.js";
import { installOffline } from "./offline.js";
import { REQUEST_PARAM, installFeedback, keepError, requestWords } from "./feedback.js";

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
  a_optimal: `${BOOK}/optimal-designs#doe-optimal-designs`,
  omars: `${BOOK}/omars-designs#doe-omars-designs`,
  simplex_lattice: `${BOOK}/mixture-experiments#doe-mixture-designs`,
  simplex_centroid: `${BOOK}/mixture-experiments#doe-mixture-designs`,
  extreme_vertices: `${BOOK}/mixture-experiments#doe-mixture-designs`,
  lenth: `${BOOK}/full-factorial-designs/assessing-significance-of-main-effects-and-interactions#doe-lenth-method`,
  measures: `${BOOK}/omars-designs#doe-omars-metric-choice`,
  fds: `${BOOK}/judging-and-comparing-designs#doe-fds-plot`,
};

const $ = (sel) => document.querySelector(sel);

/**
 * A <select> that lists only some of its options at a time: the designs a study can take, the models a run
 * sheet takes. An <option> cannot be hidden in every browser, since Safari lists it all the same, as a phone's own
 * picker does, so an option the page leaves out is taken out of the list, and put back in its place when it is
 * wanted. Every option the select has, as index.html lists them, is kept here: what is said of an option, and
 * whether it can be chosen, is read and set on these. The list starts with the options index.html does not hide.
 */
function listing(select) {
  const every = [...select.options].map((option) => ({ option, group: option.closest("optgroup") }));
  const first = new Set(every.filter(({ option }) => !option.hidden).map(({ option }) => option));
  for (const { option } of every) option.hidden = false; // from here an option is listed or it is not
  let offered = (option) => first.has(option);

  /** Make the list again: the options offered, and the one chosen whatever the rule says of it. */
  function list(chosen) {
    const rows = []; // the list's own lines, in order: an option, or a group of them
    for (const { option, group } of every) {
      if (!offered(option) && option.value !== chosen) continue;
      if (!group) rows.push(option);
      else if (rows.includes(group)) group.append(option);
      else {
        group.replaceChildren(option);
        rows.push(group);
      }
    }
    select.replaceChildren(...rows);
    select.value = chosen;
  }
  list(select.value);
  return {
    /** Every option the select has, listed now or not. */
    options: every.map(({ option }) => option),
    /** The option of a value, listed now or not. */
    option: (value) => every.find(({ option }) => option.value === value)?.option,
    /** List the options `rule` offers, with `chosen` as the choice: the choice is listed whatever the rule says. */
    show(rule, chosen = select.value) {
      offered = rule;
      list(chosen);
    },
    /** Choose an option, listing it if the rule leaves it out. */
    choose: list,
  };
}
const designs = listing($("#design-type"));
const models = listing($("#model"));
/**
 * The terms of Learn's model and what stands beside them, held here: they are shown in the fold "For the
 * statistician", which is made again with each analysis, so they are off the page between one fold and the next and
 * cannot be looked up then.
 */
const TERMS = {
  box: $("#terms"),
  list: $("#term-list"),
  adder: $("#term-adder"),
  first: $("#term-first"),
  second: $("#term-second"),
  add: $("#add-term"),
  designed: $("#terms-designed"),
  clear: $("#terms-clear"),
  said: $("#terms-said"),
};
const scenarios = listing($("#scenario"));
// ---------------------------------------------------------------- the line to Python
/** The last Python errors, newest last, each with the entry point it came from: a report of a problem carries them. */
let pythonErrors = [];

/** The worker that runs Python, the calls that wait for it, and Stop (worker-rpc.js): what the page shows of them. */
const python = pythonLine({
  make: () => new Worker("worker.js", { type: "module" }),
  onStatus(status, state) {
    if (pythonFailure) return;
    setStatus(status, state);
    pythonSaid = statusTicket;
  },
  onStoppable: (offered) => ($("#stop").hidden = !offered),
  onRefusal: (fn, error) => (pythonErrors = keepError(pythonErrors, `${fn}: ${error}`)),
  onCrash: (message) => pythonFailed(`Could not start Python: ${message || "worker.js did not load"}`),
  onStopped: () => setStatus("Stopped. Python is starting again...", "busy"),
  // What a model is fitted with came after the start, so the catalogue named what Python ran on without it.
  onWarm(runtime) {
    if (runtime) state.runtime = runtime;
    resolveWarm();
  },
});

/** Call a Python ``api_*`` function; resolves to its decoded {ok, result|error, warnings}. */
const call = (fn, payload = {}) => python.call(fn, payload);

/**
 * Python could not start: say why, and stop everything that waits for it. A worked example on
 * the page stays, and every call is answered with the reason.
 */
function pythonFailed(reason) {
  if (pythonFailure) return;
  pythonFailure = reason;
  if (!pythonErrors.at(-1)?.endsWith(reason)) pythonErrors = keepError(pythonErrors, reason); // a reply may have said it
  setStatus(reason, "error");
  $("#run-example").disabled = true;
  python.fail(reason);
  rejectReady(new Error(reason));
  renderCandidates();
}

/** How long a success stays on the status line before it fades, and how long the fade takes. */
const STATUS_SHOWN = 5000;
const STATUS_FADE = 600;
/** How long a busy spell lasts before the faded status line comes back for it. */
const STATUS_WAIT = 400;
let statusTimer = null;
/** One more each time the status line is set, so a reply can tell whether the page said something after it was asked. */
let statusTicket = 0;
/** The status ticket of Python's last note on how far it has got, such as having installed plotly: the charts may replace it. */
let pythonSaid = 0;

/**
 * Say what Python is doing on the status line; state is "busy" (the mark's runs light up), "ok"
 * (a tick) or "error". The line hangs under the study's bar, over room the page leaves for it, so
 * nothing moves as it comes and goes. A success fades after five seconds; an error comes back at
 * once, and a busy spell once it has lasted long enough to be worth showing, so a quick analysis
 * does not flash a line.
 */
function setStatus(text, state = "ok") {
  statusTicket += 1;
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

/** Take the status line away at once: what it said no longer holds, and nothing takes its place. */
function clearStatus() {
  statusTicket += 1;
  clearTimeout(statusTimer);
  const box = $("#status");
  box.querySelector(".status__text").textContent = "";
  box.hidden = true;
  box.dataset.state = "ok";
}

/**
 * Whether a reply is dropped, ``left`` saying that what asked for it has left the page meanwhile (a new study, say):
 * the busy note put up as it was asked, at status ticket ``said``, then goes too, unless something was said since.
 */
function dropped(left, said) {
  if (left && statusTicket === said) clearStatus();
  return left;
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
/** The trade-offs' view of a study just opened: nothing tried, picked, limited or chosen yet. */
const newTrade = () => ({ tried: null, across: null, up: null, colour: "", limits: {}, chosen: null });

const state = {
  phase: "plan", // the phase on show: "plan", "run" or "learn"
  design: null, // what api_make_design returned, or the runs of an uploaded workbook
  planKey: null, // the plan (factors, design and settings) the run sheet on the page was made from
  results: {}, // each response's results, by its column: one per run, in the design's row order, a number or null
  marks: {}, // where each result came from, as results holds them: "simulated", "entered" or null
  curves: null, // each run's curve, when the study reads one (its spec's curve): a list per run, or null until read
  shown: null, // the column of the response Learn analyses: the first, or the one its tabs show
  models: {}, // the model each response is analysed with, by its column, as modelChoice gives one
  simulation: null, // what api_simulate returned: the true process, its noise and seed
  practising: false, // the reader chose to practise with simulated results
  practiceRound: false, // the results on the page are a practice round's: simulated by one press in Design
  analysis: null,
  contourPairs: {}, // for a response's column, the two factors the reader chose to draw its surface over: else the largest two
  moreCharts: false, // whether "More charts" is open
  setAside: {}, // for a response's column, the places on the run sheet of the runs its fit goes without
  simpler: null, // the terms the search for a simpler model proposed for the analysis on show, or null
  best: null, // the best settings for the response's goal, on the fitted model: what api_best_settings returned
  balance: null, // the settings that balance the responses with a goal: what api_balance returned
  // The trade-offs between them: what api_trade_offs returned (tried), the two responses on the chart's axes and the
  // one it is coloured by, each response's two limits, and the candidate setting or the alternative the reader chose.
  trade: newTrade(),
  // The settings that give a target: each response's target as typed, what api_target_settings returned (found),
  // and the two factors the reader chose to draw its chart over.
  target: newTarget(),
  // The settings that meet a specification: each response's window as typed, what api_specification returned
  // (found), and the two factors the reader chose to draw its chart over.
  window: newWindow(),
  whatIf: null, // the what-if panel: each response's fit from api_predictors, the settings, the limits they break
  curve: { fit: null, problem: null, target: null }, // the curves' fit from api_functional, why there is none, a target
  scenarios: [], // the settings the reader kept from the what-if panel, each {name, settings}: the study's what_if
  next: null, // the next step, {design, key, offers, notes, recommendation, waiting}: what api_next_runs returned for it
  widened: false, // the study was started with its ranges widened, nothing having stood out in the study before
  settings: {}, // the study's own settings, as the study keeps them (campaign.js studySettings): its colour map
  buildsOn: [], // the study this one was started from, as the study keeps it (its builds_on): the round before
  goals: {}, // what the reader set of each response's goal in the balance, by its column: low, high, importance, shape
  comparison: null, // the designs that can learn the brief: what api_candidates returned, and the brief's key
  picked: null, // a design taken from the comparison, {key, design}, made again only if the plan changes
  control: null, // the level Dunnett's intervals compare the others with, when the reader picks one: else the first
  plotlyJs: null, // the Plotly.js version the Python plotly package names
  charted: null, // the figures drawn for the response on show: their keys and the charts' reply they were drawn from
  commit: null, // the commit the app was built from, as its manifest names it
  committed: null, // the day of that commit, as 2026-11-13, when the build could tell
  campaign: null, // this browser's record of the study on the page, {id, created, title}; null for the worked example
  example: false, // the worked example is on the page, as the build made it or as run again here
  exampleBrief: null, // the worked example's brief, as briefKey gives it
  version: null, // the process-improve release Python runs, once it has started
  runtime: null, // the Python, packages and Pyodide it runs on, which a report names
};
/** One more each time another study takes the page: a run sheet asked for by the one before is dropped. */
let studyTicket = 0;
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
let resolveWarm;
/**
 * Resolves once the worker has what a model is fitted with, or says it could not get it: start-up leaves that out,
 * and the worker fetches it once the page is ready (worker.js).
 */
const whenWarm = new Promise((resolve) => {
  resolveWarm = resolve;
});

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
const modelsByName = () => Object.fromEntries(resultNames().map((name) => [name, asked(modelOf(name))]));

/** Each response's results as its model is fitted to them: without the runs set aside from its fit. */
const fitted = () => fittedResults(state.results, state.setAside);

/** The runs set aside from a response's fit that have a result, as places on the run sheet. */
const asideRuns = (name = state.shown) => asideOf(state.setAside, state.results, name);

/** A run's number in the order the runs were made, from its place on the run sheet. */
const runOrderOf = (place) => state.design.rows[place]?.[state.design.columns.indexOf("Run order")] ?? place + 1;

/** The results and their sources as Python takes them: the first response's alone, as always, and the others' beside. */
const sentResults = () => ({ ...splitResults(resultNames(), state.results), ...splitResults(resultNames(), state.marks, "sources") });

/** Each run's curve as Python and a saved study take them, when the study reads one. */
const sentCurves = () => (state.curves ? { curves: state.curves } : {});

/** Make room for a curve on each run added at the end of the run sheet, when the study reads one. */
function curvesFor(added) {
  if (state.curves) state.curves = [...state.curves, ...Array(added).fill(null)];
}

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
const PANES = { plan: "#step-plan", run: "#step-run", learn: "#step-analyse", next: "#step-next" };

/**
 * Show a phase: its tab is marked, its pane shown, and on a narrow screen the brief goes with
 * Design. Charts drawn while their pane was hidden are fitted to it, and the designs that can learn
 * the brief are made once Design is on show.
 */
/** Where this browser keeps the stage each of its studies was left on, by the study's id: its own bookkeeping. */
const STAGE_KEY = "factorial-stage";
/** The most studies whose stage is kept: the ones left most recently. */
const STAGES_KEPT = 50;

/** The stages kept, by study; none where storage is refused or holds something else. */
function stagesLeft() {
  try {
    const kept = JSON.parse(localStorage.getItem(STAGE_KEY) ?? "{}");
    return kept && typeof kept === "object" && !Array.isArray(kept) ? kept : {};
  } catch {
    return {};
  }
}

/** The stage a study was left on in this browser, or null: for a study opened from what this browser keeps. */
const stageLeft = (id) => (Object.hasOwn(PANES, stagesLeft()[id]) ? stagesLeft()[id] : null);

/** Keep the stage the study on the page is on, so a later visit opens it there. */
function keepStage(phase) {
  const id = state.campaign?.id;
  if (!id || !state.design) return;
  const kept = Object.entries({ ...stagesLeft(), [id]: undefined }).filter(([, stage]) => stage);
  try {
    localStorage.setItem(STAGE_KEY, JSON.stringify(Object.fromEntries([...kept.slice(1 - STAGES_KEPT), [id, phase]])));
  } catch {
    /* storage refused: the study opens on the stage its results call for */
  }
}

function showPhase(phase) {
  // On Design the question is the form itself. Over Run and Learn it is one line, closed on the way in from Design
  // and left as the reader has it between the two.
  if (phase === "plan") $("#question").open = true;
  else if ($("#workspace").dataset.phase === "plan") $("#question").open = false;
  state.phase = phase;
  keepStage(phase);
  $("#workspace").dataset.phase = phase;
  for (const [name, pane] of Object.entries(PANES)) {
    const on = name === phase;
    const tab = $(`#go-${name}`);
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    $(pane).hidden = !on;
  }
  if (phase !== "plan") {
    cancelCandidates();
    $("#welcome").hidden = true; // a first look is over once the reader has left the question
  }
  if (phase === "next") renderRounds(); // from the studies this browser keeps now
  // The top of the workspace in view, under the study bar, when the phase starts above the window's top.
  const top = $("#workspace").getBoundingClientRect().top - $("#studybar").offsetHeight - 8;
  if (top < 0) window.scrollBy({ top, behavior: "instant" });
  if (phase === "plan") scheduleCandidates();
}

/**
 * The tabs of a tablist answer the arrow keys, Home and End, as a reader of a screen reader expects: the tabs on
 * show, since a list may hold one the page does not offer now.
 */
function arrowKeys(list, select) {
  list.addEventListener("keydown", (e) => {
    const tabs = [...list.querySelectorAll('[role="tab"]:not([hidden])')];
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
  // The designed runs: confirmation runs are checked in Learn, beside the settings they test, and the runs on the path
  // in Next. A run is complete when every response has its result.
  // A run that failed has no result to wait for, so the count goes without it.
  const confirming = new Set(state.design ? [...apartRows(state.design), ...failedRows(state.design)] : []);
  const complete = completeRuns(state.results, resultNames());
  const designed = complete.filter((_, i) => !confirming.has(i));
  const done = designed.length > 0 && designed.every(Boolean);
  badge("run", state.design ? runBadge(designed.map((ok) => (ok ? true : null))) : "", done ? "done" : "");
  const r2 = state.analysis?.summary?.r_squared;
  badge("learn", r2 === null || r2 === undefined ? "" : `R² ${Number(r2).toFixed(2)}`);
}

/**
 * Name the study in the bar: its responses as the brief describes them, and the design or the factors. A response
 * known only by the column name the page gives it ("y"), as a study opened from the catalog can be, names nothing:
 * the bar then shows the study's own name, as Your studies lists it.
 */
/** What the bar calls a study with no response named yet. */
const NEW_STUDY = "A new study";

function renderStudyBar() {
  const info = readResponseInfo();
  const name = info.label ?? state.design?.spec.response ?? "";
  const described = describedResponses();
  const unnamed = name === DEFAULT_RESPONSE && state.design;
  $("#study-title").textContent =
    described.length > 1
      ? responsesTitle(described.map(({ info: one }, i) => ({ name: i ? "" : name || "y", ...one })))
      : unnamed
        ? (state.campaign?.title ?? titleOf(state.design.spec))
        : name
          ? responseTitle(info, name)
          : NEW_STUDY;
  const k = readFactors().length;
  const noun = mixturePlanned() ? "component" : "factor";
  const about = state.design
    ? `${designLabel(state.design.spec.design_type)}, ${designRows(state.design).rows.length} runs`
    : k
      ? `${k} ${noun}${k === 1 ? "" : "s"}`
      : "";
  $("#study-about").textContent = about ? `· ${about}` : "";
  renderQuestion();
  renderBadges();
}

/** The question in one line, which Run and Learn show in place of the brief: the response, the factors and the aim. */
function renderQuestion() {
  const title = $("#study-title").textContent;
  // A mixture's model is its own, so its aim is not a choice and is left out.
  const aim = document.querySelector('input[name="aim"]:checked:not(:disabled)')?.closest("label").querySelector("strong");
  $("#question-says").textContent = questionLine(
    title === NEW_STUDY ? "" : title,
    readFactors(),
    aim?.textContent.replace(/\s+/g, " ").trim(),
  );
  // Design opens with the same question read back as a sentence, once there is a response and a factor to name.
  const asked = questionSentence(describedResponses().map(({ info }) => info), readFactors(), { blend: mixturePlanned() });
  $("#asked").textContent = asked;
  $("#asked").hidden = !asked;
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
  state.models[name] = asked(modelOf(resultNames()[0]));
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
const blankRow = (tr) =>
  box(tr, "name").value.trim() === "" &&
  (typeOf(tr) === CATEGORICAL ? chipLevels(tr).length === 0 : ["low", "high"].every((key) => box(tr, key).value.trim() === ""));

function addFactor(f = { name: "", low: "", high: "", units: "" }) {
  const tr = document.createElement("tr");
  // Each cell carries its column's heading, which a narrow column shows above the box (style.css). A categorical
  // factor's levels are chips in the first level cell, which then spans the level and units columns.
  tr.innerHTML = `
    <td data-label="Factor name"><input name="name" aria-label="Factor name" size="8" autocomplete="off" spellcheck="false"></td>
    <td data-label="Type"><label class="switch">
      <input type="checkbox" name="categorical" role="switch" aria-label="Categorical">
      <span class="switch__track" aria-hidden="true"></span><span class="switch__text">Numeric</span>
    </label></td>
    <td data-label="Low"><input name="low" aria-label="Low">
      <div class="level-chips" role="list" aria-label="Levels" hidden><input name="level" class="level-chips__input" aria-label="Add a level"
        autocomplete="off" spellcheck="false" maxlength="${LABEL_MOST}"></div></td>
    <td data-label="High"><input name="high" aria-label="High"></td>
    <td data-label="Units"><input name="units" aria-label="Units" size="6"></td>
    <td><button type="button" class="link" aria-label="Remove factor">✕</button></td>`;
  for (const key of ["name", "units"]) box(tr, key).value = f[key] ?? "";
  tr.dataset.name = box(tr, "name").value.trim(); // the name the limits know it by
  if (isCategorical(f)) {
    setChips(tr, levelsOf(f).map((v) => String(v)));
    showType(tr, CATEGORICAL);
  } else {
    for (const key of ["low", "high"]) box(tr, key).value = f[key] ?? "";
    showType(tr, NUMERIC);
  }
  wireChips(tr);
  // The row settles its type before the table's own listener refreshes the design.
  box(tr, "categorical").addEventListener("input", () => switchType(tr, typeOf(tr)));
  for (const name of ["low", "high"]) {
    // Two labels typed for a numeric factor's levels make it categorical, its two labels its first chips.
    box(tr, name).addEventListener("input", () => {
      const [low, high] = [box(tr, "low").value, box(tr, "high").value];
      if (typeOf(tr) === NUMERIC && factorType(low, high, NUMERIC) === CATEGORICAL) {
        tr.dataset[NUMERIC] = JSON.stringify(["", ""]);
        setChips(tr, [low.trim(), high.trim()]);
        showType(tr, CATEGORICAL);
        box(tr, "level").focus();
      }
    });
    box(tr, name).addEventListener("focusout", () => {
      if (tr.dataset.visited || blankRow(tr)) return;
      tr.dataset.visited = "yes";
      showFactorProblems();
    });
  }
  tr.querySelector('button[aria-label="Remove factor"]').onclick = () => {
    tr.remove();
    factorsChanged();
  };
  $("#factors tbody").append(tr);
  return tr;
}

/** A categorical factor's levels, as its chips hold them, in their order. */
const chipLevels = (tr) => [...tr.querySelectorAll(".level-chip")].map((chip) => chip.dataset.level);

/** A chip for one level: its label, which a drag or Alt and an arrow key moves, and a button that removes it. */
function levelChip(level) {
  const chip = el("span", "level-chip", "");
  chip.dataset.level = level;
  chip.draggable = true;
  chip.tabIndex = 0;
  chip.setAttribute("role", "listitem");
  const remove = el("button", "level-chip__remove", "✕");
  remove.type = "button";
  remove.tabIndex = -1;
  remove.setAttribute("aria-label", `Remove level ${level}`);
  chip.append(el("span", "level-chip__text", level), remove);
  return chip;
}

/** Put these levels in a row's chips, in their order. */
function setChips(tr, levels) {
  const chips = tr.querySelector(".level-chips");
  for (const chip of chips.querySelectorAll(".level-chip")) chip.remove();
  const input = box(tr, "level");
  for (const level of levels) input.before(levelChip(level));
  showChipRoom(tr);
}

/** Say each chip's place for a screen reader, and stop the box at MAX_LEVELS levels. */
function showChipRoom(tr) {
  const chips = [...tr.querySelectorAll(".level-chip")];
  chips.forEach((chip, i) => {
    chip.setAttribute("aria-label", `${chip.dataset.level}, level ${i + 1} of ${chips.length}; Alt and an arrow key move it`);
  });
  const input = box(tr, "level");
  const full = chips.length >= MAX_LEVELS;
  input.disabled = full;
  input.placeholder = full ? `${MAX_LEVELS} levels, the most` : chips.length ? "Add a level" : "Type a level, Enter";
}

/** Tell the table a factor's levels changed, as a box typed in tells it. */
function chipsChanged(tr) {
  showChipRoom(tr);
  tr.dataset.visited = "yes";
  tr.querySelector(".level-chips").dispatchEvent(new Event("input", { bubbles: true }));
}

/** Add each level typed or pasted: a list splits at commas, semicolons, tabs and new lines, up to MAX_LEVELS. */
function addLevels(tr, text) {
  const input = box(tr, "level");
  const room = MAX_LEVELS - chipLevels(tr).length;
  const levels = String(text)
    .split(/[,;\t\n]+/)
    .map((v) => v.trim())
    .filter(Boolean)
    .slice(0, Math.max(0, room));
  if (!levels.length) return;
  for (const level of levels) input.before(levelChip(level.slice(0, LABEL_MOST)));
  input.value = "";
  chipsChanged(tr);
}

/** Move a chip one place along, or to before another chip (null: to the end), keeping the keyboard on it. */
function moveChip(tr, chip, before) {
  const input = box(tr, "level");
  tr.querySelector(".level-chips").insertBefore(chip, before ?? input);
  chip.focus();
  chipsChanged(tr);
}

/**
 * The chips' keys, clicks and drags: Enter, a comma or leaving the box adds what is typed; a pasted list adds a chip
 * per line or comma; Backspace in the empty box goes to the last chip; on a chip, Delete or Backspace removes it, an
 * arrow key goes to its neighbour, and Alt with an arrow key moves it; a chip dragged onto another goes before it.
 */
function wireChips(tr) {
  const chips = tr.querySelector(".level-chips");
  const input = box(tr, "level");
  input.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === ",") && input.value.trim()) {
      e.preventDefault();
      addLevels(tr, input.value);
    } else if (e.key === "Enter") {
      e.preventDefault();
    } else if (e.key === "Backspace" && !input.value) {
      chips.querySelector(".level-chip:last-of-type")?.focus();
    }
  });
  input.addEventListener("paste", (e) => {
    const text = e.clipboardData?.getData("text") ?? "";
    if (!/[,;\t\n]/.test(text)) return;
    e.preventDefault();
    addLevels(tr, input.value + text);
  });
  input.addEventListener("change", () => addLevels(tr, input.value));
  input.addEventListener("input", (e) => e.stopPropagation()); // a level half typed is no level yet
  chips.addEventListener("click", (e) => {
    const remove = e.target.closest(".level-chip__remove");
    if (!remove) return;
    remove.closest(".level-chip").remove();
    input.focus();
    chipsChanged(tr);
  });
  chips.addEventListener("keydown", (e) => {
    const chip = e.target.closest(".level-chip");
    if (!chip) return;
    const step = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[e.key];
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      const next = chip.nextElementSibling;
      chip.remove();
      (next?.classList.contains("level-chip") ? next : input).focus();
      chipsChanged(tr);
    } else if (step && e.altKey) {
      e.preventDefault();
      if (step < 0 && chip.previousElementSibling) moveChip(tr, chip, chip.previousElementSibling);
      else if (step > 0 && chip.nextElementSibling?.classList.contains("level-chip")) moveChip(tr, chip, chip.nextElementSibling.nextElementSibling);
    } else if (step) {
      e.preventDefault();
      const to = step < 0 ? chip.previousElementSibling : chip.nextElementSibling;
      (to ?? chip).focus();
    }
  });
  chips.addEventListener("dragstart", (e) => {
    const chip = e.target.closest(".level-chip");
    if (!chip) return;
    chip.classList.add("level-chip--dragged");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", chip.dataset.level);
  });
  chips.addEventListener("dragover", (e) => {
    if (chips.querySelector(".level-chip--dragged")) e.preventDefault();
  });
  chips.addEventListener("drop", (e) => {
    const dragged = chips.querySelector(".level-chip--dragged");
    if (!dragged) return;
    e.preventDefault();
    // A chip dropped on another takes its place, as in any list sorted by dragging: the other moves one along, away
    // from where the dragged chip came from. Dropped between chips, it goes before the first whose middle lies past.
    const all = [...chips.querySelectorAll(".level-chip")];
    const onto = e.target.closest(".level-chip");
    const past = (c) => {
      const r = c.getBoundingClientRect();
      return e.clientY < r.top || (e.clientY <= r.bottom && e.clientX < r.left + r.width / 2);
    };
    const before =
      onto && onto !== dragged
        ? all.indexOf(dragged) > all.indexOf(onto)
          ? onto
          : onto.nextElementSibling
        : all.find((c) => c !== dragged && past(c));
    dragged.classList.remove("level-chip--dragged");
    if (onto !== dragged) moveChip(tr, dragged, before ?? null);
  });
  chips.addEventListener("dragend", () => chips.querySelector(".level-chip--dragged")?.classList.remove("level-chip--dragged"));
}

/**
 * Show a factor row as its type. A categorical factor's levels are chips, in the cell of its first level, spanning
 * the level and units columns, and it has no units: those typed are put aside, and come back if the factor is made
 * numeric again.
 */
function showType(tr, type) {
  const categorical = type === CATEGORICAL;
  const units = box(tr, "units");
  box(tr, "categorical").checked = categorical;
  // Written only when they change. Safari sends no click when the words under the pointer are written again between
  // the press and the release, and leaving a factor's name does that: the switch then took a second press.
  const words = categorical ? "Categorical" : "Numeric";
  const text = tr.querySelector(".switch__text");
  if (text.textContent !== words) text.textContent = words;
  tr.classList.toggle("categorical", categorical);
  const [first, second, third] = [box(tr, "low"), box(tr, "high"), units].map((b) => b.parentElement);
  first.colSpan = categorical ? 3 : 1;
  second.hidden = categorical;
  third.hidden = categorical;
  box(tr, "low").hidden = categorical;
  tr.querySelector(".level-chips").hidden = !categorical;
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
  const [lower, upper] = component ? ["Lower", "Upper"] : ["Low", "High"];
  for (const [name, label, blank] of [["low", lower, "0"], ["high", upper, "1"]]) {
    box(tr, name).setAttribute("aria-label", component ? `${label} proportion` : label);
    box(tr, name).placeholder = component ? blank : label.toLowerCase();
    box(tr, name).parentElement.dataset.label = label;
  }
  if (categorical) first.dataset.label = "Levels";
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
 * type's own start (blank, or A and B). A factor made categorical has the keyboard in its box for
 * the next level; one made numeric again leaves the keyboard where it was.
 */
function switchType(tr, type) {
  const [low, high] = [box(tr, "low"), box(tr, "high")];
  if (type === CATEGORICAL) {
    tr.dataset[NUMERIC] = JSON.stringify([low.value, high.value]);
    setChips(tr, tr.dataset[CATEGORICAL] ? JSON.parse(tr.dataset[CATEGORICAL]) : START_LEVELS[CATEGORICAL]);
  } else {
    tr.dataset[CATEGORICAL] = JSON.stringify(chipLevels(tr));
    [low.value, high.value] = tr.dataset[NUMERIC] ? JSON.parse(tr.dataset[NUMERIC]) : START_LEVELS[NUMERIC];
  }
  showType(tr, type);
  // After the press is over: Safari gives the keyboard to the switch as the press ends, and would take it back from
  // the box for the next level.
  if (type === CATEGORICAL) setTimeout(() => box(tr, "level").focus());
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
    const categorical = typeOf(tr) === CATEGORICAL;
    const typed = { name: names[i], low: box(tr, "low").value, high: box(tr, "high").value, ...(categorical ? { levels: chipLevels(tr) } : {}) };
    const levels = mixture && !categorical ? componentProblem(typed) : levelProblem(typed, typeOf(tr));
    unready ||= Boolean(named || levels);
    const untouched = !categorical && !tr.dataset.visited && !box(tr, "low").value.trim() && !box(tr, "high").value.trim();
    const said = [named, untouched ? "" : levels].filter(Boolean);
    box(tr, "name").toggleAttribute("aria-invalid", Boolean(named));
    for (const name of ["low", "high"]) box(tr, name).toggleAttribute("aria-invalid", Boolean(levels) && !untouched && !categorical);
    tr.querySelector(".level-chips").toggleAttribute("aria-invalid", Boolean(levels) && categorical);
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
  // A blank box is sent as null, which Python reports, rather than as Number("") = 0; a label, as text. A categorical
  // factor says its type and lists its levels, any text, so a level 1 stays a label. For a mixture, a blank box is the
  // component's bound: 0 for the lower proportion, 1 for the upper.
  const mixture = mixturePlanned();
  const level = (tr, name, blank) => (mixture ? proportionValue : levelValue)(box(tr, name).value, blank);
  return factorRows()
    .filter((tr) => !blankRow(tr))
    .map((tr) =>
      typeOf(tr) === CATEGORICAL
        ? categoricalFactor(box(tr, "name").value.trim(), chipLevels(tr))
        : { name: box(tr, "name").value.trim(), low: level(tr, "low", 0), high: level(tr, "high", 1), units: box(tr, "units").value.trim() },
    );
}

const factorCount = () => readFactors().length;

/** The factor rows the reader marked as hard to change, among those that are factors; a mixture has none. */
const hardRows = () => factorRows().filter((tr) => !blankRow(tr) && tr.dataset.hard === "1");

/** The names of the factors that are hard to change, as the design is asked for them. */
const hardNames = () => hardRows().map((tr) => box(tr, "name").value.trim());

/** Mark the factors of these names as hard to change, and no other: a stored design's, as its options hold them. */
function setHard(names) {
  for (const tr of factorRows()) tr.dataset.hard = names.includes(box(tr, "name").value.trim()) ? "1" : "";
  renderHard();
}

/** Draw a tick for each factor, in the plan's settings: ticked, the factor is set once for a group of runs. */
function renderHard() {
  const rows = factorRows().filter((tr) => !blankRow(tr));
  const ticks = rows.map((tr) => {
    const label = document.createElement("label");
    label.className = "hard-to-change__factor";
    const tick = document.createElement("input");
    tick.type = "checkbox";
    tick.name = "hard_to_change";
    tick.checked = tr.dataset.hard === "1";
    tick.addEventListener("change", () => {
      tr.dataset.hard = tick.checked ? "1" : "";
      markStale();
    });
    const name = box(tr, "name").value.trim();
    tick.setAttribute("aria-label", `${name} is hard to change`); // a name alone is also a term of the model, in Learn
    label.append(tick, ` ${name}`);
    return label;
  });
  $("#hard-factors").replaceChildren(...ticks);
}

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
  renderHard();
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
  // The what-if fits and the balance, and any still on their way, name the old factors: the analysis below finds them again.
  clearWhatIf();
  clearTarget();
  clearWindow();
  clearBalance();
  if (simulation) state.simulation = simulation;
  if (titled) state.campaign.title = titleOf(state.design.spec);
  const sameTerms = (model) => ({ ...model, terms: renameTerms(model.terms, names) }); // picked or not, as they were
  if ($("#model").value === CHOSEN_TERMS) setModel(sameTerms(modelChoice()));
  for (const [name, model] of Object.entries(state.models)) {
    if (model && typeof model === "object") state.models[name] = sameTerms(model); // each response's terms
  }
  const renamed = (settings) => Object.fromEntries(Object.entries(settings).map(([factor, v]) => [names[factor] ?? factor, v]));
  state.scenarios = state.scenarios.map((s) => ({ ...s, settings: renamed(s.settings) }));
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
  renderQuestion();
}

function showAim() {
  const factors = readFactors();
  // Past MAX_INTERACTION_FACTORS factors a study aims at the main effects alone, and the aim moves there.
  const many = screensOnly(factors.length);
  if (many && currentAim() !== "main_effects") {
    setAim("main_effects");
    return;
  }
  const aim = currentAim();
  const p = aimTerms(factors, aim);
  const screening = !aimChosen && aim === "main_effects" && factors.length >= SCREEN_FROM && !many;
  const box = $("#aim-terms");
  box.replaceChildren();
  // A mixture's model is a Scheffé model of its blends, which its design sets: the aims are for factors with ranges.
  const mixture = mixturePlanned();
  for (const radio of document.querySelectorAll('input[name="aim"]')) radio.disabled = mixture || (many && radio.value !== "main_effects");
  if (mixture) {
    box.append(
      "A mixture is analysed with a Scheffé model of its blends, which its design sets: these aims are for factors with ranges.",
    );
    return;
  }
  if (!factors.length) return;
  box.append(`${p} `, term("coefficients", { key: "coefficient", alone: true }), ", the intercept included.");
  if (screening) box.append(` With ${SCREEN_FROM} factors or more, a study starts by finding which matter.`);
  if (many) box.append(` ${screeningSentence(factors.length, "brief")}`);
}

// ---------------------------------------------------------------- the design
// The design list, its hints, which settings each design takes and how many factors it accepts
// are in the HTML, so the form is right from the first paint, before Python has loaded.
let budgetEdited = false;
let resolutionEdited = false;
let designChosen = false; // the reader chose the design, or a study opened with one; else Design picks it

const alsoListed = new Set(); // listed besides the designs the page makes: runs as given, once a study has had some

/** What a design type is, as the design list says under it. */
const designHint = (designType) => designs.option(designType)?.dataset.hint ?? "";

/** The glossary's entry for each kind of term power is reckoned for. */
const KIND_TERMS = { main: "main-effect", interaction: "interaction", quadratic: "square" };

/** A design type's name as the design list gives it. */
function designLabel(designType) {
  const option = designs.option(designType);
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
  if (spec.design_type === GIVEN) alsoListed.add(GIVEN);
  listDesigns(spec.design_type);
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
  setHard(options.hard_to_change ?? []);
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
/**
 * The ladder over the designs, for a study of two to five numeric factors: each step a button that puts its design
 * and its aim in the form, with its runs in all and the runs it adds. The step the form stands on is pressed. A
 * mixture, a categorical factor or another number of factors has no ladder: the list of designs is theirs.
 */
/** The steps of the ladder for the factors as they are, and the key of the step the form is on, or null. */
function ladderNow() {
  const factors = readFactors();
  const steps = mixturePlanned() || factors.some(isCategorical) ? [] : ladderSteps(factors.length);
  const on = ladderStepOf(steps, {
    design_type: $("#design-type").value,
    n_center_points: Number($("#n_center_points").value),
    resolution: Number($("#resolution").value),
  });
  return { steps, on };
}

function renderLadder() {
  const { steps, on } = ladderNow();
  $("#ladder").hidden = !steps.length;
  if (!steps.length) {
    $("#ladder-steps").replaceChildren();
    return;
  }
  const items = steps.map((step) => {
    const button = el("button", "ladder__step", "");
    button.type = "button";
    button.dataset.step = step.key;
    button.setAttribute("aria-pressed", String(step.key === on));
    const added = step.added === step.runs ? "" : `, ${step.added} more`;
    button.append(el("span", "ladder__runs", `${step.runs} runs${added}`), el("strong", "", step.name), el("span", "ladder__learns", step.learns));
    button.onclick = () => takeStep(step);
    const item = el("li", "", "");
    item.append(button);
    return item;
  });
  $("#ladder-steps").replaceChildren(...items);
}

/** Put a step of the ladder in the form: its aim, then its design and settings, as if the reader had chosen each. */
function takeStep(step) {
  aimChosen = true;
  designChosen = true;
  setAim(step.aim);
  if (state.design) state.design.spec = { ...state.design.spec, aim: currentAim() };
  setDesignForm({ design_type: step.design_type, options: step.options });
  scheduleCandidates();
  renderStudyBar();
  saveDraft();
  document.querySelector(`#ladder-steps [data-step="${step.key}"]`)?.focus();
}

function refreshDesign() {
  renderLadder();
  renderHard();
  const k = factorCount();
  const full = factorRows().length >= MAX_FACTORS; // the add buttons stop at the most factors a study has, and say so
  for (const id of ["#add-factor", "#add-categorical"]) $(id).disabled = full;
  $("#factors-full").hidden = !full;
  const select = $("#design-type");
  const labelled = readFactors().filter(isCategorical).map((f) => f.name);
  const several = readFactors().filter(isSeveral);
  // Runs as given take categorical factors, of any number of levels, and limits on the settings, as they are.
  const numericOnly = (option) => labelled.length > 0 && !CATEGORICAL_DESIGNS.includes(option.value) && option.value !== GIVEN;
  const twoLevels = (option) => several.length > 0 && !SEVERAL_LEVEL_DESIGNS.includes(option.value) && option.value !== GIVEN;
  const limited = hasLimits();
  const outsideLimits = (option) => limited && !REGION_DESIGNS.includes(option.value) && option.value !== GIVEN;
  showComponents();
  for (const option of designs.options) {
    option.disabled =
      k < Number(option.dataset.minFactors) ||
      k > Number(option.dataset.maxFactors) ||
      numericOnly(option) ||
      twoLevels(option) ||
      outsideLimits(option);
  }
  const option = select.selectedOptions[0];
  const allowed = option.dataset.options.split(" ");
  for (const el of document.querySelectorAll("[data-option]")) el.hidden = !allowed.includes(el.dataset.option);
  const alpha = axialDistance($("#alpha").value, $("#alpha_value").value);
  $("#alpha-distance").hidden = !allowed.includes("alpha") || $("#alpha").value !== AXIAL_DISTANCE;

  // Resolution IV unless the reader chose another; three factors allow only III, and past MAX_INTERACTION_FACTORS
  // factors, whose interactions the page does not model, a fraction stops at IV.
  const resolution = $("#resolution");
  for (const choice of resolution.options) choice.disabled = Number(choice.value) > highestResolution(k);
  if (!resolutionEdited || resolution.selectedOptions[0].disabled) resolution.value = String(defaultResolution(k));

  // A design in one factor runs each level as often as its replicates say: more of them than a design of more factors.
  $("#n_replicates").max = String(k === 1 ? MAX_LEVEL_RUNS : MAX_REPLICATES);

  // A Box-Behnken design or a rotatable CCD needs a center run to fit the squares; a DSD has one built in.
  const centre = $("#n_center_points");
  const centreNeeded = needsCentre(option.value, alpha, k);
  centre.min = centreNeeded ? 1 : 0;
  if (centreNeeded && Number(centre.value) < 1) centre.value = 1;

  // An optimal design is made for a model of MAX_OPTIMAL_TERMS coefficients at most: a larger one is not offered.
  const modelType = $("#model_type");
  for (const choice of modelType.options) choice.disabled = nTerms(k, choice.value) > MAX_OPTIMAL_TERMS;
  if (modelType.selectedOptions[0].disabled) modelType.value = "main_effects";
  const tooLarge = [...modelType.options].some((choice) => choice.disabled);

  // A space-filling design is made for no model: its runs are so many a factor, and Learn fits the aim's model.
  const spread = SPACE_FILLING.includes(option.value);
  const p = nTerms(k, spread ? OPTIMAL_MODEL[currentAim()] : $("#model_type").value);
  if (!budgetEdited) $("#budget").value = defaultBudget(option.value, k, $("#model_type").value);
  const omars = option.value === "omars";
  $("#budget-hint").textContent = spread
    ? `${SPACE_FILLING_RUNS} runs a factor is the usual start; Learn fits the model of your aim, which has ${p} coefficients.`
    : omars
      ? omarsHint(k)
      : `The model has ${p} coefficients.` +
        (tooLarge ? ` With ${k} factors a larger model has more than ${MAX_OPTIMAL_TERMS}, the most an optimal design is made for here.` : "");

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
  } else if (option.disabled && outsideLimits(option)) {
    hint.textContent = `${option.textContent} cannot keep its runs inside your limits on the settings: choose D-optimal or I-optimal.`;
  } else if (option.disabled && twoLevels(option)) {
    const levels = several.map((f) => `${f.name} has ${levelsOf(f).length}`);
    hint.textContent =
      `${option.textContent} needs factors of two levels, and ${listNames(levels)}: choose a full factorial, ` +
      "Taguchi or optimal design, or give each categorical factor two levels.";
  } else if (option.disabled && numericOnly(option)) {
    hint.textContent =
      `${option.textContent} needs numeric factors, and ${listNames(labelled)} ${labelled.length === 1 ? "is" : "are"} ` +
      "categorical: choose a full factorial, fractional factorial, Plackett-Burman, Taguchi or optimal design.";
  } else if (option.disabled) {
    hint.textContent = `${option.textContent} takes ${option.dataset.minFactors} to ${option.dataset.maxFactors} factors on this page; you have ${k}.`;
  } else {
    // A design the book has no page on (a Taguchi array) shows its hint alone.
    const book = BOOK_PAGES[option.value];
    hint.replaceChildren(...(book ? [`${option.dataset.hint} `, link(book, "More in the book")] : [option.dataset.hint]));
  }
  hint.classList.toggle("error", option.disabled);
  // A search that takes long is the reader's to start: the wait is said before the button is pressed.
  const seconds = option.disabled ? 0 : searchSeconds(option.value, k, $("#model_type").value);
  const wait = $("#search-wait");
  wait.hidden = seconds < SLOW_SEARCH;
  wait.textContent = wait.hidden
    ? ""
    : `The search for this design takes up to ${waitWords(seconds)} in a browser: its model has ${p} coefficients in ` +
      `${k} factors. It starts when you make the run sheet, and you can stop it.`;
  showTradeOff();
  showMakeable(showFactorProblems());
  $("#design-card").hidden = k < 1 && !state.design;
  showAim();
  markPicked();
  markStale();
  showLikeYours();
}

// ---------------------------------------------------------------- experiments like yours
// What others in the reader's field have run, from catalog/fields.json (written by the build), with the plan marked
// on the chart: its factors, and its runs once it has a design. The field chosen is kept in this browser; a field's
// page in the catalog chooses one through ?field=.
const FIELD_KEY = "factorial-field";
let fields = null; // the fields the build wrote, once read

function keptField() {
  try {
    return localStorage.getItem(FIELD_KEY) ?? "";
  } catch {
    return ""; // storage refused
  }
}

function keepField(slug) {
  try {
    if (slug) localStorage.setItem(FIELD_KEY, slug);
    else localStorage.removeItem(FIELD_KEY);
  } catch {
    /* storage refused: the choice lasts as long as the page */
  }
}

/** Read the fields the build wrote, list them, and choose the one the address or this browser names. */
async function loadFields() {
  try {
    const response = await fetch("catalog/fields.json");
    fields = response.ok ? (await response.json()).fields : null;
  } catch {
    fields = null; // a page served without its catalog shows no panel
  }
  if (!fields) return;
  const named = Object.entries(fields).sort(([, a], [, b]) => a.label.localeCompare(b.label));
  $("#field").append(...named.map(([slug, field]) => new Option(field.label, slug)));
  const asked = new URLSearchParams(location.search).get("field") ?? "";
  if (Object.hasOwn(fields, asked)) keepField(asked);
  const chosen = keptField();
  $("#field").value = Object.hasOwn(fields, chosen) ? chosen : "";
  showLikeYours();
}

/** Show the chosen field's experiments beside the plan: a sentence, the chart with the plan, and how they compare. */
function showLikeYours() {
  $("#like-yours").hidden = !fields;
  const slug = $("#field").value;
  const field = fields?.[slug];
  $("#like-yours-body").hidden = !field;
  if (!field) return;
  const k = state.design ? state.design.spec.factors.length : factorCount();
  const plan = k ? { factors: k, runs: state.design ? runCount() : null } : null;
  $("#like-yours-sentence").textContent = field.sentence;
  drawFieldChart($("#like-yours-chart"), field.points, plan);
  $("#like-yours-verdict").textContent = planVerdict(field.points, plan);
  $("#like-yours-link").href = `catalog/fields/${encodeURIComponent(slug)}/`;
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
  $("#trade-off-hint").textContent = TRADE_OFF_FACTORS.includes(k) ? `The column of your ${k} factors takes a pick. ${move}` : tradeOffBeyond(k) || move;
}

/** Take a cell of the grid into the plan: its resolution, or the full factorial. */
function pickCell(runs) {
  const design = cellDesign(runs, factorCount());
  if (!design) return;
  designChosen = true;
  if (design.design_type === "full_factorial") {
    listDesigns(design.design_type);
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
  $("#generate").disabled = !pythonReady || !fits || unready || factorCount() < 1 || limitsUnready() || responsesUnready;
  $("#practice-round").disabled = $("#generate").disabled;
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
    // By place, as a factor's name is a label: the rows that are factors, and which of them are ticked.
    hard_to_change: factorRows()
      .filter((tr) => !blankRow(tr))
      .map((tr) => tr.dataset.hard === "1")
      .join(),
    model_type: $("#model_type").value,
    lattice_degree: $("#lattice_degree").value,
    mixture_model: $("#mixture_model").value,
  };
  const taken = option.dataset.options.split(" ").map((key) => [key, values[key]]);
  const factors = readFactors();
  const levels = factors.map((f) => (isCategorical(f) ? levelsOf(f) : [f.low, f.high]));
  return JSON.stringify([option.value, levels, Object.fromEntries(taken), limitsKey(factors.map((f) => f.name))]);
}

/** How many results on the run sheet the reader entered: typed, pasted or uploaded, not simulated for practice. */
function enteredResults() {
  if (!state.design) return 0;
  const entered = (name) => (state.results[name] ?? []).filter((value, i) => value !== null && state.marks[name]?.[i] !== "simulated");
  return resultNames().reduce((count, name) => count + entered(name).length, 0);
}

/**
 * Whether a new run sheet may take the place of the one on the page. A new run sheet starts blank, in a new random
 * order, and is a new study once the one on the page holds a result the reader entered (the study before stays in
 * Your studies). A reader who pressed the button again met a blank sheet with no word of where the results went, so:
 * - the design is as it was when the sheet was made: the sheet stays, and Run shows it;
 * - the design has changed: the reader is asked, and told where this study and its results are kept.
 * Results simulated for practice are made again with the sheet, so they are not asked about.
 */
async function mayReplaceRunSheet() {
  const entered = enteredResults();
  if (!entered) return true;
  if (!planIsStale()) {
    showPhase("run");
    setStatus("The run sheet for this design is made already: your results are kept.");
    return false;
  }
  const dialog = $("#replace-dialog");
  const kept = state.campaign?.id
    ? "This study and its results stay in Your studies, under Studies in the bar."
    : "This study's results are not kept: download the study first if you want them.";
  $("#replace-says").textContent =
    `This run sheet holds ${entered === 1 ? "1 result" : `${entered} results`} you entered. The design has changed, ` +
    `so a new run sheet starts a new study: blank, with its runs in a new order. ${kept}`;
  dialog.showModal();
  // Answered by a button at once, in the press itself, so what follows starts with no gap; Escape, which only
  // closes the question, keeps the study.
  return new Promise((resolve) => {
    answerReplace = (go) => {
      answerReplace = null;
      resolve(go);
      if (dialog.open) dialog.close();
    };
  });
}
let answerReplace = null; // the question on show, to be answered; null when none is

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

/** Whether the next run sheet made takes the place of the study on the page: one begun here, with no result typed in. */
const replacesStudy = () => Boolean(state.campaign?.replaceable) && !Object.values(state.marks).some((marks) => marks.includes("entered"));

/** A new study's seed, the moment it first needs one, in seconds; null until then, and again once a study takes it. */
let draftSeed = null;

/**
 * The seed the next run sheet's runs are put in random order with, which the comparison's designs are made with too.
 * The run sheet on the page keeps its own while the next takes its place: a study begun here, kept for its life,
 * or the worked example, while its brief is as built, so its comparison's designs are the build's. Else it is a
 * new study's, the moment it first needed one.
 */
function studySeed() {
  const keeps = replacesStudy() || (state.example && briefKey() === state.exampleBrief);
  const kept = keeps ? state.design?.spec?.options?.random_state : undefined;
  if (Number.isInteger(kept)) return kept;
  draftSeed ??= Math.floor(Date.now() / 1000);
  return draftSeed;
}

/**
 * Make the run sheet the plan describes; resolves to true when it is on the page. ``randomState`` is the seed its
 * runs are put in random order with, the study's unless it is given. A design taken from the comparison is used as
 * it was made only when it was made with that seed.
 */
async function generate({ randomState } = {}) {
  cancelCandidates(); // the reader has chosen: the designs can wait until Design is shown again
  const study = studyTicket;
  let key = planKey();
  const factors = readFactors();
  const responses = readResponses({ fresh: true });
  const seeded = randomState ?? studySeed();
  const picked = state.picked?.key === key && state.picked.design.spec.options?.random_state === seeded ? state.picked : null;
  const design = picked && (await pickedDesign(picked, factors));
  if (study !== studyTicket) return false; // another study took the page meanwhile
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
  const generating = statusTicket;
  const spec = {
    design_type: $("#design-type").value,
    factors,
    ...storedResponses(responses),
    aim: currentAim(),
    n_center_points: Number($("#n_center_points").value),
    alpha: axialDistance($("#alpha").value, $("#alpha_value").value),
    n_replicates: Number($("#n_replicates").value),
    n_blocks: Number($("#n_blocks").value),
    hard_to_change: hardNames(),
    resolution: Number($("#resolution").value),
    n_levels: Number($("#n_levels").value),
    budget: Number($("#budget").value),
    model_type: $("#model_type").value,
    lattice_degree: Number($("#lattice_degree").value),
    mixture_model: $("#mixture_model").value,
    constraints: currentLimits(),
    runs: $("#design-type").value === GIVEN ? givenRuns() : undefined,
    blocks: $("#design-type").value === GIVEN ? givenBlocks() : undefined,
    random_state: seeded,
  };
  const reply = await call("api_make_design", spec);
  button.disabled = false;
  refreshDesign();
  if (dropped(study !== studyTicket, generating)) return false; // another study took the page meanwhile
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
  if (state.widened) reply.result.spec.options = { ...reply.result.spec.options, widened: true }; // see startMovedStudy
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
  const replaces = replacesStudy();
  state.campaign = save ? { ...(replaces ? state.campaign : newCampaign()), title: titleOf(design.spec) } : null;
  if (save && !replaces) draftSeed = null; // the new study took it: the next takes a seed of its own
  state.example = !save;
  state.design = design;
  $("#share-out").hidden = true;
  state.results = noResults(resultNames(), design.rows.length);
  state.marks = noResults(resultNames(), design.rows.length);
  state.setAside = {};
  state.contourPairs = {};
  state.curves = null;
  state.curve = { fit: null, problem: null, target: null }; // the fit of the curves before goes with them
  state.shown = resultNames()[0];
  state.models = {};
  state.goals = {};
  state.trade = newTrade();
  state.target = newTarget();
  state.window = newWindow();
  state.scenarios = [];
  state.simulation = null;
  state.analysis = null;
  clearBalance();
  clearWhatIf();
  clearTarget();
  clearWindow();
  renderDesign(warnings);
  renderRuns();
  renderTruth(); // the process behind the old design's results goes with them
  showLikeYours();
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
    const given = r.spec?.design_type === GIVEN && !r.spec.options?.grown;
    // A design with a factor that is hard to change keeps the order of its whole plots, which its notes explain.
    const order = given || r.spec?.options?.hard_to_change ? "in the order given" : "in random run order";
    parts.push(said("", `${r.n_runs} runs${centre}, ${order}.${outside}`), ...noteList(r.facts.notes));
  } else {
    parts.push(el("p", "", `${r.rows.length} runs, read from the workbook.`));
  }
  parts.push(...warningList(warnings));
  $("#design-summary").replaceChildren(...parts);
  $("#design-out").hidden = false;
  offerCorrelation();
  markStale();
}

// ---------------------------------------------------------------- the correlation map of the study's runs
/** Whether the map is drawn for a design: 2 to MAX_INTERACTION_FACTORS factors, with ranges or of two levels. */
const mapped = ({ design_type: designType, factors }) =>
  !isMixture(designType) && !factors.some(isSeveral) && factors.length >= 2 && factors.length <= MAX_INTERACTION_FACTORS;

/** Offer the map for the design on the page; an open one follows the design, whose runs may have changed. */
function offerCorrelation() {
  const box = $("#correlation");
  box.hidden = !state.design || !mapped(state.design.spec);
  if (!box.hidden && box.open) showCorrelation();
}

/** List the pairs the words speak of, largest first: the model's term, the term it moves with, and how far. */
function renderPairs({ pairs }) {
  const table = $("#correlation-pairs");
  table.replaceChildren();
  if (!pairs.length) return;
  const head = table.createTHead().insertRow();
  for (const heading of ["A term of the model", "Moves with", "|r|", "Which is"]) head.append(el("th", "", heading));
  const body = table.createTBody();
  for (const { a, b, r, both, blocks } of pairs) {
    const row = body.insertRow();
    row.append(el("td", "", prettyTerm(a)), el("td", "", blocks ? THE_BLOCKS : prettyTerm(b)), el("td", "num", r.toFixed(2)));
    row.append(el("td", "", both ? "in the model too" : "left out of the model"));
  }
}

/**
 * Show how far the terms of the study's runs move together. Python is asked when the reader opens the panel, so a
 * study whose reader never opens it pays nothing for it. The sentences and the table of the largest pairs say what
 * the map draws; the map itself needs Plotly.js, which the page of a plain preview has no version of.
 */
async function showCorrelation() {
  const design = state.design;
  const box = $("#correlation");
  const key = JSON.stringify([design.spec.model, design.columns, design.rows]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  const words = $("#correlation-words");
  words.replaceChildren(el("p", "hint", "Reading the runs..."));
  const reply = await call("api_correlations", held());
  if (box.dataset.key !== key) return; // the design changed meanwhile, and its own map is on its way
  if (!reply.ok) {
    delete box.dataset.key; // asked again when the panel is next opened
    words.replaceChildren(el("p", "hint error", reply.error));
    $("#correlation-plot").replaceChildren();
    $("#correlation-pairs").replaceChildren();
    return;
  }
  const found = reply.result;
  words.replaceChildren(...correlationWords(found, prettyTerm).map((sentence) => el("p", "", sentence)));
  renderPairs(found);
  if (state.plotlyJs) {
    const python = figureAfter("run", () => correlationCode(design.spec.factors.map((f) => f.name)));
    await draw($("#correlation-plot"), () => correlationFigure(found.terms, found.matrix, found.k), state.plotlyJs, 360, { python });
  }
}
$("#correlation").addEventListener("toggle", () => {
  if ($("#correlation").open) showCorrelation();
});

// ---------------------------------------------------------------- the designs that can learn the brief
/** How long the brief rests before its designs are made and scored: a pause in the typing. */
const CANDIDATE_WAIT = 450;
let candidateTimer = null;
let candidateTicket = 0;
let candidatesRunning = false;

/** The brief as the comparison reads it: the factors, the aim, the response and what is known of it, the limits. */
function briefRequest() {
  const factors = readFactors();
  const info = readResponseInfo();
  const response = responseName(info.label, factors.map((f) => f.name));
  return { factors, model: currentAim(), response, response_info: info, constraints: currentLimits() };
}

/** What api_candidates is asked for the brief as it is now: its designs are made with the next run sheet's seed. */
const candidateRequest = () => ({ ...briefRequest(), random_state: studySeed() });

/** What makes two briefs' designs the same: the factors, the aim, the limits, and the noise and change power is for. */
function briefKey(request = briefRequest()) {
  const { noise = null, smallest_change: change = null } = request.response_info;
  const limits = request.constraints.map((limit) => positional(limit.expression, request.factors.map((f) => f.name)));
  return JSON.stringify([request.factors, request.model, noise, change, limits]);
}

/** What makes two comparisons the same: the brief, and the seed their designs' runs are put in random order with. */
const candidateKey = (request = candidateRequest()) => JSON.stringify([briefKey(request), request.random_state]);

/** Whether the brief is ready for its designs: a factor or more, each as Python takes it. */
const briefReady = () => factorCount() >= 1 && !factorsUnready && !limitsUnready();

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
  // A design is picked from the designs of the brief on the page only: a new study takes none of the brief before's.
  if (reply.ok && reply.result.result && !designChosen && key === candidateKey()) {
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
  // Nothing to compare in detail before the brief's first designs, or for one factor, whose design is its two levels;
  // nor for designs scored on their power alone (bootstrap._powers), whose list already shows all there is.
  $("#compare").hidden = !c?.result || mixture || Boolean(c.levels) || !c.result.measures.length;
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
  else if (limitsUnready() && factorCount() >= 1 && !factorsUnready) note = ["hint busy", "Reading the limits on the settings..."];
  else if (current && c.error) note = ["hint error", c.error];
  else if (current) note = null;
  else if (pythonFailure) note = ["hint error", "The designs are made with Python, which could not start here."];
  else if (briefReady() && !pythonReady) note = ["hint", "The designs that can help you learn this are listed once Python has started."];
  else if (busy || briefReady()) note = ["hint busy", busy ?? "Making and scoring the designs that can help you learn this..."];
  const waiting = current && !c.result && c.waiting?.length ? [waitingNote(c), ...noteList(c.notes.map(prettyTerm))] : [];
  const listed = !briefReady() || mixture ? [] : c?.levels && c.result ? levelPlan(c, current) : c?.result ? candidateList(c, current) : waiting;
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
  let stepped = null; // the runs of the ladder step in force, when its row shows them in place of the table's own
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
      concept: "degrees_of_freedom",
      definition:
        `The runs left once the model's ${c.result.terms.length} coefficients are fitted: they estimate the ` +
        `run-to-run noise. Fewer than ${SPARE_FOR_NOISE} are marked.`,
    });
    const power = term(`Power to find ${changeWords(info)}`, {
      alone: true,
      concept: "statistical_power",
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
    // The table compares each design as Design first makes it, with its center points. The step of the ladder the
    // reader is on has runs of its own, so its design's row shows those: the two never disagree on the runs chosen.
    const { steps, on } = ladderNow();
    const step = steps.find((one) => one.key === on);
    for (const listed of fitting) {
      const mine = step && listed.design_type === $("#design-type").value && step.runs !== listed.runs;
      const r = mine ? { ...listed, runs: step.runs, spare: listed.spare - (listed.runs - step.runs), power: {} } : listed;
      if (mine) stepped = step.runs;
      const tr = el("tr", "", "");
      tr.dataset.design = r.design_type;
      const name = el("td", "", "");
      const label = el("label", "", "");
      const radio = el("input", "", "");
      radio.type = "radio";
      radio.name = "candidate";
      radio.value = r.design_type;
      radio.disabled = !current;
      const concept = DESIGN_CONTEXT[r.design_type]; // the design's term in the concept map
      label.append(radio, term(r.label, { definition: designHint(r.design_type), concept }));
      name.append(label);
      tr.append(name, el("td", "", String(r.runs)), el("td", r.spare < SPARE_FOR_NOISE ? "low" : "", String(r.spare)));
      for (const [kind] of kinds) tr.append(el("td", "", kind in r.power ? percent(r.power[kind]) : ""));
      if (leads) tr.append(el("td", "stands", (best[r.design_type] ?? []).map((b) => b.words).join(", ")));
      if (current) tr.onclick = () => pickDesign(r.design_type);
      tbody.append(tr);
    }
    t.append(thead, tbody);
    const wrap = el("div", "scroll", "");
    wrap.append(t);
    parts.push(wrap);
    if (stepped !== null) {
      parts.push(el("p", "hint stepped", `Your design's row shows the ${stepped} runs of the step you chose above. The other designs are listed as Design first makes them, and the table has no power for your step.`));
    }
  }
  const picked = designChosen ? "" : pickSentence(rows, $("#design-type").value);
  if (picked) parts.push(el("p", "hint picked", picked));
  const unfit = rows.filter((r) => !r.fits);
  if (unfit.length) parts.push(el("p", "hint", `${listNames(unfit.map((r) => r.label))} cannot fit this model.`));
  const categorical = readFactors().filter(isCategorical).map((f) => f.name);
  const several = readFactors().filter(isSeveral);
  for (const why of leftOutSentences(c.left_out ?? [], factorCount(), categorical, several)) parts.push(el("p", "hint left-out", why));
  if (c.waiting?.length) parts.push(waitingNote(c));
  parts.push(...noteList(c.notes.map(prettyTerm)));
  return parts;
}

/**
 * A brief of one factor, its levels compared (``c.levels``, as api_candidates plans them): a row for each number of
 * runs at each level, with the power to find the change worth finding, the row the form makes marked. Two levels are
 * compared by the t-test; three or more by the one-way analysis of variance, the change between two of them and the
 * others midway, where it is hardest to find. Picking a row sets the design's replicates, its runs at each level.
 */
function levelPlan(c, current) {
  const plan = c.levels;
  const k = plan.rows[0].runs / plan.rows[0].n;
  const test = k === 2 ? "the t-test" : "the one-way analysis of variance";
  const info = readResponseInfo();
  const change = changeWords(info);
  const power = term(`Power to find ${change}`, {
    alone: true,
    concept: "statistical_power",
    definition:
      (k === 2
        ? `The chance that the t-test of the two levels finds ${change} between them, at the `
        : `The chance that the one-way analysis of variance of the ${k} levels finds ${change} between two of them, ` +
          "the others midway, at the ") +
      `${Math.round(100 * c.result.significance_level)}% level.` +
      (powerScale(info) ? "" : " To set your own, give the noise and the change under What you know of it already."),
  });
  const head = el("tr", "", "");
  for (const word of ["Runs at each level", "Runs", power]) {
    const th = el("th", "", "");
    th.append(word);
    head.append(th);
  }
  const thead = el("thead", "", "");
  thead.append(head);
  const tbody = el("tbody", "", "");
  for (const row of plan.rows) {
    const tr = el("tr", "", "");
    tr.dataset.runs = String(row.n);
    const label = el("label", "", "");
    const radio = Object.assign(el("input", "", ""), { type: "radio", name: "level-runs", value: String(row.n), disabled: !current });
    label.append(radio, String(row.n));
    const first = el("td", "", "");
    first.append(label);
    tr.append(first, el("td", "", String(row.runs)), el("td", row.power < plan.power ? "low" : "", percent(row.power)));
    if (current) tr.onclick = () => pickLevelRuns(row.n);
    tbody.append(tr);
  }
  const t = el("table", "data pick levels-plan", "");
  t.append(thead, tbody);
  const wrap = el("div", "scroll", "");
  wrap.append(t);
  const most = plan.rows.at(-1);
  const sized = plan.reached
    ? `${plan.n} runs at each level, ${k * plan.n} in all, are the fewest that give ${test} ${percent(plan.power)} ` +
      `power to find ${change}. Pick a row for more runs or fewer.`
    : `Even ${most.n} runs at each level, the most the page makes, give ${test} ${percent(most.power)} power to ` +
      `find ${change}: a change this small against this much noise needs more runs than that.`;
  const levels = k === 2 ? "two levels" : `${k} levels`;
  return [
    el("h3", "", "How many runs at each level"),
    el("p", "hint", `With one factor, the design runs its ${levels} in random order, each the same number of times.`),
    wrap,
    el("p", "hint", sized),
  ];
}

/** Make the design of one factor with this many runs at each level, as the reader picked from the plan. */
function pickLevelRuns(n) {
  designChosen = true;
  listDesigns("full_factorial");
  $("#n_center_points").value = 0;
  $("#n_replicates").value = n;
  refreshDesign();
  renderStudyBar();
  saveDraft();
}

/** Mark the plan's row of the runs at each level the form makes: a full factorial with no center runs. */
function markLevelRuns() {
  const two = $("#design-type").value === "full_factorial" && Number($("#n_center_points").value) === 0;
  for (const tr of document.querySelectorAll("#candidates .levels-plan tbody tr")) {
    const on = two && tr.dataset.runs === $("#n_replicates").value;
    tr.setAttribute("aria-selected", String(on));
    tr.querySelector("input").checked = on;
  }
}

/** Which designs wait for the reader to ask for their search, as Design says it beside the button. */
function waitingNote(c) {
  const one = c.waiting.length === 1;
  const seconds = c.waiting.reduce((sum, w) => sum + w.seconds, 0);
  return el(
    "p",
    "hint",
    `The ${listNames(c.waiting.map((w) => w.label))} design${one ? " takes" : "s take"} up to ${waitWords(seconds)} ` +
      `to search for with ${factorCount()} factors. The search starts when you ask for it, and you can stop it.`,
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
  markLevelRuns();
  for (const tr of document.querySelectorAll("#candidates table:not(.levels-plan) tbody tr")) {
    const on = tr.dataset.design === chosen;
    tr.setAttribute("aria-selected", String(on));
    tr.querySelector("input").checked = on;
  }
  listDesigns();
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
  if (dialog.dataset.key === c.key) return;
  dialog.dataset.key = c.key;
  c.hidden ??= new Set(); // the designs the reader has hidden, kept while this comparison stands
  renderComparison(c.result, c.made, c.notes, c.hidden);
  showPython(
    "compare",
    compareCode(
      readFactors(),
      c.made.map((m) => ({ label: m.label, design_type: m.design_type, options: m.design.spec.options })),
      c.model,
      c.made[0]?.design.spec.constraints ?? [],
    ),
  );
  await drawComparison(c.result, readFactors().length, c.hidden);
}

let comparisonFit = null; // watches whether the comparison table fits its box

/** A cell of one design's column in the detailed comparison, holding text or an element, and the design's label. */
function designCell(tag, label, content) {
  const cell = el(tag, "", typeof content === "string" ? content : "");
  if (typeof content !== "string") cell.append(content);
  cell.dataset.design = label;
  return cell;
}

/**
 * The detailed comparison as a table: a column per design, a row per measure. Three rows head the designs: their
 * names; a switch to show or hide each, so the reader can set designs aside for reasons the measures do not hold;
 * and the button that takes one into the plan. Every cell of a design's column carries its label, so that
 * arrangeComparison can order the columns and fill in the values.
 */
function renderComparison(result, made, notes, hidden) {
  const out = $("#comparison");
  out.hidden = false;
  const t = el("table", "data compare", "");
  const measureHead = el("th", "", "Measure");
  const names = el("tr", "", "");
  names.append(measureHead, el("th", "", "Better"));
  const shows = el("tr", "shows", "");
  shows.append(el("th", "", "Show"), el("th", "", ""));
  const uses = el("tr", "uses", "");
  uses.append(el("th", "", ""), el("th", "", ""));
  result.designs.forEach((label, i) => {
    // The page's switch, as the factors table has it for a factor's type: a checkbox drawn as a track and a knob.
    const toggle = el("input", "", "");
    toggle.type = "checkbox";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", `Show ${label}`);
    toggle.checked = !hidden.has(label);
    toggle.onchange = () => {
      if (toggle.checked) hidden.delete(label);
      else hidden.add(label);
      arrangeComparison(t, result, hidden, { slide: true });
      showComparedCharts(result, hidden);
    };
    const control = el("label", "switch", "");
    control.append(toggle, el("span", "switch__track", ""));
    const use = el("button", "link", "");
    use.type = "button";
    use.append("Use this", el("br", "", ""), "design"); // two lines in every column, however narrow
    use.setAttribute("aria-label", `Use this design: ${label}`);
    use.onclick = () => useDesign(made[i]);
    names.append(designCell("th", label, label));
    shows.append(designCell("th", label, control));
    uses.append(designCell("th", label, use));
  });
  const thead = el("thead", "", "");
  thead.append(names, shows, uses);
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
      tr.append(td);
      tbody.append(tr);
    }
    const tr = el("tr", "", "");
    tr.dataset.measure = String(row);
    const name = el("td", "", "");
    if (m.group === "Feasibility") {
      name.append(term(m.label, { definition: m.description, alone: true }));
    } else {
      const to = link(BOOK_PAGES.measures, "");
      to.append(term(m.label, { definition: m.description }));
      name.append(to);
    }
    tr.append(name, el("td", "arrow", m.direction === "higher" ? "↑" : m.direction === "lower" ? "↓" : ""));
    // "Fits the model?" is words, as the library has it; arrangeComparison writes the numbers.
    for (const label of result.designs) tr.append(designCell("td", label, m.key === "fits" ? result.display[label][row] : ""));
    tbody.append(tr);
  });
  t.append(thead, tbody);
  arrangeComparison(t, result, hidden);
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

/**
 * Put the detailed comparison's columns in order, and write and shade its values. The designs shown come first:
 * those that fit the model, then those that cannot, after a dashed line. The designs hidden follow a thick line, in
 * light grey. Each row is printed and shaded over the designs shown that fit the model (see comparedRow), so hiding
 * a row's best passes its bold to the next. With `slide`, each column moves from where it was to where it now is.
 */
function arrangeComparison(t, result, hidden, { slide = false } = {}) {
  const before = slide ? columnLefts(t) : null;
  const labels = comparisonOrder(result, hidden).map((i) => result.designs[i]);
  const fits = (label) => Boolean(result.values[label]?.fits);
  const shown = (label) => !hidden.has(label);
  const firstUnfit = labels.find((label) => shown(label) && !fits(label));
  const firstHidden = labels.find((label) => !shown(label));
  const ranked = labels.map((label) => shown(label) && fits(label));
  const hiddenCount = labels.filter((label) => !shown(label)).length;
  for (const tr of t.rows) {
    if (tr.classList.contains("group")) {
      spanGroup(tr, labels.length - hiddenCount, hiddenCount);
      continue;
    }
    const cells = new Map([...tr.querySelectorAll("[data-design]")].map((cell) => [cell.dataset.design, cell]));
    tr.append(...labels.map((label) => cells.get(label)));
    const m = tr.dataset.measure === undefined ? null : result.measures[Number(tr.dataset.measure)];
    const values = labels.map((label) => result.values[label]?.[m?.key] ?? null);
    const scores = m && m.key !== "fits" ? comparedRow(values, m.direction, ranked) : null;
    labels.forEach((label, j) => {
      const cell = cells.get(label);
      cell.classList.toggle("unfit", !fits(label));
      cell.classList.toggle("first", label === firstUnfit);
      cell.classList.toggle("set-aside", !shown(label));
      cell.classList.toggle("parted", label === firstHidden);
      if (!scores) return;
      const { text, good } = scores[j];
      cell.textContent = text;
      cell.classList.toggle("scored", good !== null);
      cell.classList.toggle("row-best", good === 1);
      if (good === null) cell.style.removeProperty("--good");
      else cell.style.setProperty("--good", good.toFixed(2));
    });
  }
  if (before) slideColumns(t, before);
}

/** A group's heading spans the designs shown; a second cell spans those hidden, so the thick line runs through it. */
function spanGroup(tr, shown, hiddenCount) {
  const [heading, aside] = tr.cells;
  heading.colSpan = 2 + shown;
  if (!hiddenCount) {
    aside?.remove();
    return;
  }
  const cell = aside ?? tr.appendChild(el("td", "parted", ""));
  cell.colSpan = hiddenCount;
}

/** Where each design's column starts on the screen, by the design's label. */
const columnLefts = (t) =>
  new Map([...t.rows[0].querySelectorAll("[data-design]")].map((cell) => [cell.dataset.design, cell.getBoundingClientRect().left]));

/** Slide each design's cells from where its column started, `before`, to where it starts now. */
function slideColumns(t, before) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const after = columnLefts(t);
  for (const cell of t.querySelectorAll("[data-design]")) {
    const dx = before.get(cell.dataset.design) - after.get(cell.dataset.design);
    if (Math.abs(dx) < 1) continue;
    cell.animate([{ transform: `translateX(${dx}px)` }, { transform: "none" }], { duration: 350, easing: "ease-in-out" });
  }
}

/** Show the designs shown, and only those, in the charts: their curves in the FDS plot, and their correlation maps. */
function showComparedCharts(result, hidden) {
  const box = $("#compare-charts");
  for (const card of box.querySelectorAll("figure[data-design]")) card.hidden = hidden.has(card.dataset.design);
  const fds = box.querySelector("figure.fds .plot");
  if (fds && state.plotlyJs) {
    draw(fds, fdsFigure(result.fds, { region: Boolean(result.region), hidden }), state.plotlyJs, 380, { python: fdsPython(result) });
  }
}

/** Take a design from the detailed comparison into the plan, and close the comparison. */
function useDesign(item) {
  pickDesign(item.design_type);
  $("#compare-dialog").close();
  setStatus(`The ${item.label} design is chosen: make its run sheet when it is ready.`);
}

/** The code of the comparison's FDS plot, for its window: every design's curve, over the settings the limits leave. */
const fdsPython = (result) =>
  figureAfter("compare", () =>
    comparisonFigureCode({ asked: result.fds_asked, limits: state.comparison?.made[0]?.design.spec.constraints ?? [] }),
  );

async function drawComparison(result, k, hidden) {
  const box = $("#compare-charts");
  box.hidden = false;
  box.replaceChildren();
  const version = state.plotlyJs;
  if (!version) return;
  const fds = el("figure", "chart wide fds", "");
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
    await draw(fdsPlot, fdsFigure(result.fds, { region: Boolean(result.region), hidden }), version, 380, { python: fdsPython(result) });
    for (const label of result.designs) {
      const { terms, matrix } = result.correlations[label];
      const card = el("figure", "chart", "");
      card.dataset.design = label;
      card.hidden = hidden.has(label);
      const plot = el("div", "plot", "");
      const title = el("h3", "", "");
      title.append(term("Correlation map", { key: "correlation-map", alone: true }), `, ${label}`);
      card.append(title, plot);
      box.append(card);
      const python = figureAfter("compare", () => comparisonFigureCode({ label }));
      await draw(plot, () => correlationFigure(terms, matrix, k), version, 320, { python });
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
  const failed = new Set(failedRows(d));
  // A run that failed holds no result, whatever filled the sheet: results simulated for every run, or pasted.
  for (const name of resultNames()) {
    for (const i of failed) {
      if (state.results[name]) state.results[name][i] = null;
      if (state.marks[name]) state.marks[name][i] = null;
    }
  }
  const rows = d.rows.map((row, i) => Object.assign([...row], { index: i }));
  // Each row is a run, by its place in the order the runs are made, as the analysis and the charts name it.
  const order = d.columns.indexOf("Run order");
  t._run = (row) => (order < 0 ? row.index + 1 : row[order]);
  if (t._design !== d) pressRun(null); // a run marked in the design before is none of this one's
  t._design = d;
  // A numeric factor's setting can be typed over with the setting the run was really made at. A mixture's cannot:
  // its proportions sum to one, so one cannot change alone.
  const typed = isMixture(d.spec.design_type) ? new Map() : new Map(d.spec.factors.filter((f) => !isCategorical(f)).map((f) => [d.columns.indexOf(f.name), f]));
  t._cell = (row, col) => {
    const name = at.get(col);
    const i = row.index;
    if (typed.has(col)) {
      const f = typed.get(col);
      const planned = d.planned?.[i]?.[f.name];
      const off = planned === undefined ? "" : ` off" title="Planned: ${escape(String(planned))}`;
      const shown = escape(String(+Number(row[col]).toPrecision(6)));
      const label = `${escape(f.name)}, run ${i + 1}: ${shown}${planned === undefined ? "" : `, planned ${escape(String(planned))}`}. Change it to the setting the run was made at`;
      // A button until it is pressed, then a box in its place (editSetting): the sheet's boxes are its results.
      return `<button type="button" class="setting${off}" data-index="${i}" data-factor="${escape(f.name)}" aria-label="${label}">${shown}</button>`;
    }
    if (name === undefined) return null;
    const value = state.results[name]?.[i];
    const cls = state.marks[name]?.[i] === "simulated" ? "sim" : "";
    const label = escape(words[name].label);
    const out = failed.has(i) ? ' disabled placeholder="failed"' : ""; // a run that failed has no result
    return `<input class="${cls}" type="number" step="any" inputmode="decimal" data-index="${i}" data-response="${escape(name)}" aria-label="${label}, run ${i + 1}" value="${value ?? ""}"${out}>`;
  };
  t._key = (row, col) => (at.has(col) ? state.results[at.get(col)]?.[row.index] : row[col]); // a result sorts by its value
  const factorUnits = Object.fromEntries(d.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const headings = d.columns.map((c) => words[c]?.label ?? c);
  const colUnits = d.columns.map((c) => words[c]?.units ?? factorUnits[c] ?? "");
  const terms = {
    "Std order": "std-order",
    "Run order": "run-order",
    Block: "blocks",
    [WHOLE_PLOT]: "whole-plot",
    [CONFIRMATION]: "confirmation-run",
  };
  renderTable(t, headings, rows, "Run order", colUnits, terms);
  // On the body, which each drawing of the table makes anew: the table's own presses sort it by a heading.
  t.tBodies[0].onclick = (e) => {
    const button = e.target.closest("button.setting");
    if (button) editSetting(button);
  };
  t.oninput = (e) => {
    const input = e.target.closest("input[data-response]");
    if (!input) return;
    const i = Number(input.dataset.index);
    const name = input.dataset.response;
    state.results[name][i] = input.value === "" ? null : Number(input.value);
    state.marks[name][i] = input.value === "" ? null : "entered";
    backInTheFit(name, i);
    input.classList.remove("sim");
    resultsChanged();
  };
  t.onpaste = pasteResults;
  renderOffPlan();
  renderBench();
  renderRunDone();
  describeSource();
  renderBadges();
  renderCurves();
}

/** Follow results typed or pasted into the run sheet: where they came from, the code, the badges, the analysis. */
/**
 * A setting typed over on the run sheet: the run was made there, so the model is fitted there. The planned setting
 * is kept beside it (``design.planned``, by the run's place on the sheet and the factor's name) so the sheet can say
 * which runs left the plan and put them back. A box left blank, or typed back to the plan, is the planned setting.
 */
function settingTyped(index, name, text) {
  const d = state.design;
  const col = d.columns.indexOf(name);
  const planned = d.planned?.[index]?.[name] ?? d.rows[index][col];
  const value = text.trim() === "" || !Number.isFinite(Number(text)) ? planned : Number(text);
  d.rows[index][col] = value;
  const kept = { ...(d.planned ?? {}) };
  const row = { ...(kept[index] ?? {}) };
  if (value === planned) delete row[name];
  else row[name] = planned;
  if (Object.keys(row).length) kept[index] = row;
  else delete kept[index];
  if (Object.keys(kept).length) d.planned = kept;
  else delete d.planned;
  renderRuns();
  resultsChanged();
}

/**
 * A setting's button pressed: a box takes its place, holding the setting, for the one the run was really made at.
 * Leaving the box, or Enter, takes what it holds; Escape leaves the setting as it was.
 */
function editSetting(button) {
  const { index, factor } = button.dataset;
  const input = document.createElement("input");
  Object.assign(input, { type: "number", step: "any", inputMode: "decimal", className: "setting", value: button.textContent });
  input.setAttribute("aria-label", `${factor}, run ${Number(index) + 1}, the setting it was made at`);
  let done = false;
  // ``back`` puts the focus on the setting's button again: after a key, not after a press somewhere else.
  const finish = (take, back) => {
    if (done) return;
    done = true;
    if (take) settingTyped(Number(index), factor, input.value);
    else renderRuns();
    if (back) document.querySelector(`#design-table button.setting[data-index="${index}"][data-factor="${CSS.escape(factor)}"]`)?.focus();
  };
  input.onblur = () => finish(true, false);
  input.onkeydown = (e) => {
    if (e.key !== "Enter" && e.key !== "Escape") return;
    e.preventDefault(); // or Enter would go on to press the button the focus returns to
    finish(e.key === "Enter", true);
  };
  button.replaceWith(input);
  input.focus();
  input.select();
}

/** Put every run back at its planned settings. */
function backOnPlan() {
  const d = state.design;
  for (const [index, row] of Object.entries(d.planned ?? {})) {
    for (const [name, planned] of Object.entries(row)) d.rows[index][d.columns.indexOf(name)] = planned;
  }
  delete d.planned;
  renderRuns();
  resultsChanged();
  $("#off-plan").focus();
}

/** Under the run sheet: that a setting can be typed over, or which runs left the plan and what that does to the fit. */
function renderOffPlan() {
  const d = state.design;
  const box = $("#off-plan");
  const numbers = runNumbers();
  const runs = Object.keys(d.planned ?? {})
    .filter((index) => index < d.rows.length) // a run taken off the sheet since is none of its runs
    .map((index) => numbers[index])
    .sort((a, b) => a - b);
  box.hidden = isMixture(d.spec.design_type) || !d.spec.factors.some((f) => !isCategorical(f));
  $("#off-plan-says").textContent = runs.length ? offPlanWords(runs) : "A run made at another setting than planned? Type the setting it was really made at over the planned one.";
  $("#off-plan-back").hidden = !runs.length;
}

function resultsChanged() {
  renderBench();
  renderRunDone();
  renderPracticeRound();
  if (state.design && pathRows(state.design).length) renderNext(); // the path's runs are read as they are typed
  describeSource();
  renderTruth();
  showResultsCode();
  renderBadges();
  scheduleAnalysis();
  scheduleSave();
}

/** Run says so at its top once every run has a result for every response: the next step is Learn. */
function renderRunDone() {
  const failed = new Set(state.design ? failedRows(state.design) : []);
  const has = (v, i) => failed.has(i) || (v !== null && v !== undefined); // a run that failed has no result to wait for
  const all = Boolean(state.design) && resultNames().every((name) => state.results[name]?.length && state.results[name].every(has));
  $("#run-done").hidden = !all;
}

// ---------------------------------------------------------------- the run sheet, a run at a time
/** Whether the reader keeps "One run at a time" open, kept in this browser: open at first on a phone. */
const BENCH_KEY = "factorial-bench";
let benchAt = null; // the place on the run sheet of the run the card shows
let benchSpec = null; // the study the card shows a run of: another study starts at its first run to make
let benchDrawn = ""; // what the card's boxes were drawn for, so typing in one does not draw them again

/** The places on the run sheet in the order the runs are made: by their run order, else as listed. */
function benchOrder(d) {
  const order = d.columns.indexOf("Run order");
  const places = d.rows.map((_, i) => i);
  return order < 0 ? places : places.sort((a, b) => (Number(d.rows[a][order]) || 0) - (Number(d.rows[b][order]) || 0));
}

/**
 * Show one run of the run sheet: its number, where it stands among the runs, each factor's setting in large type,
 * a box for each response's result, and whether the run failed. It opens at the first run still to make. Its boxes
 * are the run sheet's own: what is typed here is typed there.
 */
function renderBench() {
  const d = state.design;
  $("#bench").hidden = !d;
  if (!d) return;
  const order = benchOrder(d);
  const names = resultNames();
  const failed = new Set(failedRows(d));
  const made = (i) => failed.has(i) || names.every((name) => (state.results[name]?.[i] ?? null) !== null);
  if (benchSpec !== d.spec || benchAt === null || benchAt >= d.rows.length) benchAt = order.find((i) => !made(i)) ?? order[0];
  benchSpec = d.spec;
  const i = benchAt;
  const where = order.indexOf(i);
  const runOrder = d.columns.indexOf("Run order");
  $("#bench-run").textContent = `Run ${runOrder < 0 ? i + 1 : d.rows[i][runOrder]}`;
  const left = order.filter((place) => !made(place)).length;
  const state_ = failed.has(i) ? "failed" : made(i) ? "made" : "to make";
  $("#bench-of").textContent = `${where + 1} of ${order.length}, ${state_}. ${left ? `${left} still to make.` : "Every run is made."}`;
  $("#bench-settings").replaceChildren(
    ...d.spec.factors.flatMap((f) => [el("dt", "", f.name), el("dd", "", stepSetting(f, d.rows[i][d.columns.indexOf(f.name)]))]),
  );
  const key = `${i}|${names.join("|")}|${failed.has(i)}`;
  const boxes = $("#bench-results");
  if (benchDrawn !== key) {
    benchDrawn = key;
    boxes.replaceChildren(
      ...names.map((name) => {
        const label = el("label", "", withUnits(wordsOf(name).label, wordsOf(name).units ?? ""));
        const box = el("input", "", "");
        Object.assign(box, { type: "number", step: "any", inputMode: "decimal", disabled: failed.has(i) });
        box.dataset.bench = name;
        if (failed.has(i)) box.placeholder = "failed";
        label.append(box);
        return label;
      }),
    );
  }
  for (const box of boxes.querySelectorAll("input")) {
    if (box !== document.activeElement) box.value = state.results[box.dataset.bench]?.[i] ?? "";
  }
  const apart = new Set(apartRows(d));
  Object.assign($("#bench-failed"), { checked: failed.has(i), disabled: apart.has(i) });
  $("#bench-prev").disabled = where <= 0;
  $("#bench-next").disabled = where >= order.length - 1;
}

/** Show the run before or after the one on the card, in the order the runs are made. */
function benchStep(by) {
  const order = benchOrder(state.design);
  const to = order[order.indexOf(benchAt) + by];
  if (to === undefined) return;
  benchAt = to;
  renderBench();
  $("#bench-results input:not(:disabled)")?.focus({ preventScroll: true });
}

/**
 * Mark the run at ``place`` as failed, or as a run to make after all. A run that failed stays on the run sheet,
 * marked, with no result: the fit goes without it, as without a run still to do, and the next step does not wait.
 */
function setFailed(place, failed) {
  const { columns, rows } = withFailed(state.design, place, failed);
  state.design = { ...state.design, columns, rows };
  renderRuns();
  renderStudyBar();
  resultsChanged();
  setStatus(failed ? "The run is marked as failed: it stays on the run sheet, with no result." : "The run is to make again.");
}

/**
 * Set Learn's note for when it has nothing to show, with the stage it names as a way there: "enter them in Run".
 * The words are the note's own; "Run" in them becomes a button.
 */
function sayLearnEmpty(text) {
  const note = $("#analyse-empty");
  const at = text.indexOf(" in Run");
  if (at < 0) {
    note.textContent = text;
    return;
  }
  const run = el("button", "link", "Run");
  run.type = "button";
  run.dataset.go = "run";
  note.replaceChildren(text.slice(0, at + 4), run, text.slice(at + 7));
}

/**
 * A practice round of the plan: make its run sheet, simulate its results from the process Run's practice panel holds,
 * and open Learn on them. The reader sees what the plan can and cannot tell before a run is spent. The study is then
 * as one practised on: Learn says it is a practice round until a result is typed or the simulated ones are cleared.
 */
async function practiceRound() {
  if (!(await mayReplaceRunSheet())) return;
  if (!(await generate())) return;
  selectTab("simulate");
  await simulateResults();
  state.practiceRound = true;
  showPhase("learn");
  renderPracticeRound();
}

/** Clear a practice round's simulated results and go to the run sheet, to make the runs for real. */
function endPracticeRound() {
  const names = resultNames();
  state.results = noResults(names, state.design.rows.length);
  state.marks = noResults(names, state.design.rows.length);
  state.setAside = {};
  state.practiceRound = false;
  selectTab("real");
  clearAnalysis();
  renderRuns();
  resultsChanged();
  showPhase("run");
  $("#tab-real").focus();
}

/** Learn says it shows a practice round while every result is one the practice round simulated. */
function renderPracticeRound() {
  $("#practice-note").hidden = !(state.practiceRound && state.design && anyResult(state.results) && allSimulated());
}

/** The run sheet's run numbers, in its order: its Run order column, or 1, 2, ... for runs as given without one. */
function runNumbers() {
  const d = state.design;
  const runAt = d.columns.indexOf("Run order");
  return d.rows.map((row, i) => (runAt >= 0 ? row[runAt] : i + 1));
}

// ---------------------------------------------------------------- each run's curve
/**
 * Show the box the curves are pasted into, under the run sheet: the curve's words, and how many runs have theirs.
 * A mixture's curves are not analysed on this page, so its run sheet has no box.
 */
function renderCurves() {
  const box = $("#curves");
  box.hidden = isMixture(state.design.spec.design_type);
  const curve = state.design.spec.curve;
  if (curve) {
    // the study's words; without a curve yet, the boxes keep what the reader is typing before pasting
    $("#curve-label").value = curve.label ?? "";
    $("#curve-units").value = curve.units ?? "";
    $("#curve-along").value = curve.argument.label ?? "";
    $("#curve-along-units").value = curve.argument.units ?? "";
  }
  $("#curve-clear").hidden = !curve;
  $("#curve-status").textContent = curve ? curvesSaid(curve, state.curves) : "";
  if (curve) box.open = true;
}

/** The reader's words for the curve and for the points along it, as a study keeps them: each left out when blank. */
function curveWords() {
  const words = (label, units) => ({ ...(label ? { label } : {}), ...(units ? { units } : {}) });
  const along = words($("#curve-along").value.trim(), $("#curve-along-units").value.trim());
  return { ...words($("#curve-label").value.trim(), $("#curve-units").value.trim()), along };
}

/** Take the curves pasted into the box (logic.js pastedCurves): the study then reads a curve on each run. */
function useCurves() {
  const pasted = pastedCurves($("#curve-paste").value, runNumbers(), CURVE_POINTS_HELD);
  if (pasted.problem) {
    $("#curve-status").textContent = pasted.problem;
    return;
  }
  const { along, ...words } = curveWords();
  setCurve({ ...words, argument: { ...along, values: pasted.points } }, pasted.curves);
  $("#curve-paste").value = "";
}

/** Keep the reader's words for the curve as they type them, once the study reads one. */
function curveWordsChanged() {
  const curve = state.design?.spec.curve;
  if (!curve) return;
  const { along, ...words } = curveWords();
  setCurve({ ...words, argument: { ...along, values: curve.argument.values } }, state.curves);
}

/** Set the study's curve and each run's, or take both away (null), then show, save and analyse what changed. */
function setCurve(curve, curves) {
  const { curve: _, ...spec } = state.design.spec;
  state.design = { ...state.design, spec: curve ? { ...spec, curve } : spec };
  state.curves = curve ? curves : null;
  state.curve = { fit: null, problem: null, target: null };
  renderCurves();
  scheduleAnalysis();
  scheduleSave();
}

/** The run sheet as a pasted block is placed on it (logic.js placePasted): its run numbers, rows as shown, responses. */
function pasteSheet() {
  const shown = [...$("#design-table").querySelectorAll("tbody tr")].map((tr) => tr.querySelector("input[data-index]"));
  return {
    runs: runNumbers(),
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
  const input = e.target.closest("input[data-response]"); // a setting's box takes what is pasted as any box does
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
    backInTheFit(name, index);
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
  // Past MAX_INTERACTION_FACTORS factors, Learn offers no model with every two-factor interaction, and says why.
  const k = design.spec.factors.length;
  for (const option of models.options) option.disabled = screensOnly(k) && INTERACTION_MODELS.includes(option.value);
  $("#model-many").textContent = screeningSentence(k, "learn");
  $("#model-many").hidden = !screensOnly(k);
  const taken = (option) => option.value === "" || mixture === ("mixture" in option.dataset);
  // The reader's own terms are a model the run sheet takes, and never one the list offers: it is listed while chosen.
  const offered = (option) => taken(option) && option.value !== CHOSEN_TERMS;
  for (const [list, id, fallback] of [[scenarios, "#scenario", mixture ? "blend" : "reactor"], [models, "#model", ""]]) {
    const chosen = list.option($(id).value);
    list.show(offered, chosen && taken(chosen) && !chosen.disabled ? chosen.value : fallback);
  }
  showTerms();
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

/** What the button that simulates says: at first, and once the run sheet holds simulated results. */
const SIMULATE = "Simulate the results";
const SIMULATE_AGAIN = "Simulate again, with new noise";

/** Show the true process, or hide it, by its button: the button says which a press will do. */
function showTruth(open) {
  $("#truth").hidden = !open;
  $("#truth-toggle").setAttribute("aria-expanded", String(open));
  $("#truth-toggle").textContent = `${open ? "Hide" : "Show"} the true process`;
}

/**
 * The true process behind the simulated results, behind a reveal: each response's, while some of its results are.
 * While some are, the button that simulates says a press draws new noise: as it does (see its handler).
 */
function renderTruth() {
  const shown = resultNames().filter((name) => processOf(name) && state.marks[name]?.includes("simulated"));
  $("#simulate").textContent = shown.length ? SIMULATE_AGAIN : SIMULATE;
  $("#truth-toggle").hidden = !shown.length; // gone with the last simulated result
  if (!shown.length) showTruth(false);
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
  const asked = state.design;
  if (!asked) return;
  setStatus("Simulating the results...", "busy");
  const simulating = statusTicket;
  const reply = await call("api_simulate", {
    ...held(),
    scenario: $("#scenario").value,
    sigma: Number($("#sigma").value),
    seed,
  });
  if (dropped(state.design !== asked, simulating)) return; // another run sheet took its place meanwhile
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  state.simulation = reply.result;
  const names = resultNames();
  const more = Object.fromEntries(Object.entries(reply.result.more ?? {}).map(([name, one]) => [name, one.responses]));
  // A result the reader entered stays: practice results fill the runs that have none of the reader's own.
  const before = { results: state.results, marks: state.marks };
  const own = (name, i) => (before.results[name]?.[i] ?? null) !== null && before.marks[name]?.[i] !== "simulated";
  const simulated = joinResults(names, [...reply.result.responses], more);
  state.results = Object.fromEntries(names.map((name) => [name, simulated[name].map((v, i) => (own(name, i) ? before.results[name][i] : v))]));
  state.marks = Object.fromEntries(names.map((name) => [name, simulated[name].map((_, i) => (own(name, i) ? before.marks[name][i] : "simulated"))]));
  state.setAside = {}; // every simulated result is new
  renderRuns();
  renderTruth();
  showResultsCode();
  scheduleSave();
  await analyseRuns();
}

// ---------------------------------------------------------------- where to run: one card, a goal to choose
/**
 * The goals of the card that answers where to run, each a panel under the dials: the best settings for the
 * response's goal, the settings that give a target, and those that keep every response inside a window. Each panel
 * is shown or hidden by its own part of the page (clearBest, renderTarget, renderWindow); a goal is offered while
 * its panel has something to show.
 */
const WHERE_GOALS = ["best", "target", "window"];

/**
 * The goal the reader chose, which is shown while it is offered. Until they choose, the card opens on what the
 * study asks: a target typed for a response, else a window, else the first goal offered.
 */
let goalChosen = null;

/** Show the panel of the goal chosen, or of the one the study asks; and the card itself while it holds anything. */
function renderWhere() {
  const offered = WHERE_GOALS.filter((goal) => !document.getElementById(goal).hidden);
  const asked = state.design ? keptAsked() : {};
  const first = [asked.targets && "target", asked.windows && "window", ...offered].find((goal) => offered.includes(goal));
  const shown = offered.includes(goalChosen) ? goalChosen : (first ?? null);
  for (const goal of WHERE_GOALS) {
    const tab = document.getElementById(`goal-${goal}`);
    tab.hidden = !offered.includes(goal);
    tab.setAttribute("aria-selected", String(goal === shown));
    tab.tabIndex = goal === shown ? 0 : -1;
    document.getElementById(`goal-panel-${goal}`).hidden = goal !== shown;
  }
  $("#goals").hidden = offered.length < 2; // one goal is no choice: its panel stands under the dials
  $("#where").hidden = $("#what-if").hidden && !offered.length;
}

/** Choose what the settings are for. */
function selectGoal(goal) {
  goalChosen = goal;
  renderWhere();
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
  const models = Object.fromEntries(more.map((name) => [name, asked(modelOf(name))]).filter(([, model]) => model));
  const reply = await call("api_workbook", {
    ...held(),
    ...sentResults(),
    simulated: someSimulated() ? state.simulation : null,
    model: asked(modelOf(first)),
    ...(Object.keys(models).length ? { models } : {}),
    ...(state.scenarios.length ? { what_if: state.scenarios } : {}),
    ...keptAsked(), // the targets and the windows typed in Learn, as the study's file keeps them
    ...(state.buildsOn.length ? { builds_on: state.buildsOn } : {}),
    ...sentCurves(),
  });
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const bytes = Uint8Array.from(atob(reply.result.xlsx_base64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  download(blob, `${state.design.spec.design_type}-${runCount()}-runs.xlsx`);
}

/** The run sheet with its results as a CSV file, for any spreadsheet or program: made here, without Python. */
function downloadCsv() {
  if (!state.design) return;
  const text = runSheetCsv(state.design, state.results);
  download(new Blob([text], { type: "text/csv" }), `${state.design.spec.design_type}-${runCount()}-runs.csv`);
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
      const model = state.analysis.formula ?? state.analysis.model;
      out.push({ response: name, model, best: state.best, vifModel: vifTerms(state.analysis), ...comparedOf(state.analysis) });
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
      ...comparedOf(analysis.result),
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
    ...comparedOf(state.analysis),
    version: state.version,
    url: `${location.origin}${location.pathname}`,
    best: state.best,
    analyses,
    balance: state.balance,
    figures: figuresCharted(),
  });
  const blob = new Blob([`${JSON.stringify(book, null, 1)}\n`], { type: "application/x-ipynb+json" });
  download(blob, `${state.design.spec.design_type}-${runCount()}-runs.ipynb`);
}

// ---------------------------------------------------------------- the study as a report
/** The steps whose code a report gives, in the order they run: each builds on those before it. */
const REPORT_STEPS = ["design", "run", "analyse", "best", "balance"];

/** What a report's copy of the page leaves out: what is there to use the page, and what the page hides. */
const NOT_REPORTED = "[hidden], button:not(.term), input, select, textarea, details.python, .actions, .kept__do, .rounds__open, figure.chart, .simplify, .findings__more, svg.space, .answer__where";

/**
 * The attributes a report's copy keeps: those that say how a cell spans, where a link goes, how a part looks, and
 * what a finding's picture draws.
 */
const REPORTED_ATTRIBUTES = new Set(["class", "href", "colspan", "rowspan", "open", "viewBox", "points"]);

/**
 * A part of the page as a report holds it, as HTML: a copy without what NOT_REPORTED names, a table's heading that
 * sorts it and a dotted word each left as its text, and with none of the page's ids, ARIA or data. The page set its
 * text as text, so the HTML is safe as it is.
 */
function reportCopy(node) {
  const copy = node.cloneNode(true);
  for (const word of copy.querySelectorAll("th button, .term")) word.replaceWith(word.textContent);
  for (const part of copy.querySelectorAll(NOT_REPORTED)) part.remove();
  for (const box of copy.querySelectorAll("details")) box.setAttribute("open", ""); // paper has no heading to press
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
  // The answer first, as Learn has it; the way to its evidence is a button, which a report leaves out.
  const parts = [
    $("#answer").hidden ? "" : reportCopy($("#answer")),
    $("#checks").hidden ? "" : reportCopy($("#checks")), // each check open, with its sentence and statistic
    reportCopy($("#analysis")),
    chartsHtml(await chartsWithin($("#charts"))),
  ];
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

/**
 * The sections a report ends with, after the analysis: the balance of the responses, the settings kept with the study,
 * what the targets and the windows typed gave, the next step and the code.
 */
async function closingSections() {
  const sections = [];
  if (!$("#balance").hidden && state.balance) {
    const html = [reportCopy($("#balance-out")), chartsHtml(await chartsWithin($("#balance")))];
    sections.push({ heading: $("#h-balance").textContent, html: html.join("\n") });
  }
  if (!$("#what-if").hidden && !$("#what-if-kept").hidden) {
    sections.push({ heading: $("#h-what-if-kept").textContent, html: reportCopy($("#what-if-kept-list")) });
  }
  // What Learn was asked, with what it answered: the sentences name the targets and the windows themselves.
  for (const [id, asked] of [["target", state.target], ["window", state.window]]) {
    if ($(`#${id}`).hidden || !asked.found) continue;
    const html = [reportCopy($(`#${id}-out`)), chartsHtml(await chartsWithin($(`#${id}`)))];
    if (id === "target" && !$("#target-solve").hidden) html.push(`<h3>${$("#target-solve h4").textContent}</h3>`, reportCopy($("#solve-out")));
    sections.push({ heading: $(`#h-${id}`).textContent, html: html.join("\n") });
  }
  if (!$("#rounds").hidden) sections.push({ heading: $("#h-rounds").textContent, html: reportCopy($("#rounds-out")) });
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

// ---------------------------------------------------------------- feedback
/** The phases by the names their tabs give them, as a report of a problem names the one on show. */
const PHASE_NAMES = { plan: "Design", run: "Run", learn: "Learn", next: "Next" };

/** Learn's model as its form names it: the choice, or for terms ticked, how many. */
function modelWords() {
  if ($("#model").value === CHOSEN_TERMS) return `Your terms: ${tickedTerms().length} terms`;
  return $("#model").selectedOptions[0]?.textContent ?? "";
}

/**
 * What a report of a problem carries of the page beside the reader's words, which feedback.js lists before it is sent:
 * the commit the app was built from and the Plotly.js release; the stamp a report of the study has (when, with what,
 * the run order's seed, where the results came from, the SHA-256 of the inputs), never the link that holds the study;
 * the tab, Learn's model and the charts on show; the last Python errors; and the browser and the size of its window.
 */
async function feedbackContext() {
  const { design } = state;
  const pane = $(PANES[state.phase]);
  const stamp = stampRows({
    made: new Date(),
    version: state.version,
    runtime: state.runtime,
    url: `${location.origin}${location.pathname}`,
    spec: design?.spec ?? null,
    runs: runCount(),
    results: state.results,
    marks: state.marks,
    simulation: design && someSimulated() ? state.simulation : null,
    sha: design ? await sha256(canonicalJson(reportedInputs())) : null,
    link: null,
  });
  return {
    commit: versionText(state.commit, state.committed),
    plotly_js: state.plotlyJs,
    tab: PHASE_NAMES[state.phase],
    model: state.analysis ? modelWords() : null,
    charts: [...pane.querySelectorAll("figure.chart")]
      .filter((card) => shownWithin(card, pane))
      .map((card) => card.querySelector("h3, h4")?.textContent.trim())
      .filter(Boolean),
    stamp: stamp.map(([label, text]) => `${label}: ${text}`),
    errors: pythonErrors,
    browser: navigator.userAgent,
    window: `${window.innerWidth} x ${window.innerHeight}`,
  };
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
  studyTicket += 1;
  setForm(r.spec);
  state.design = { spec: r.spec, columns: r.columns, rows: r.rows, model: r.spec.model, facts: null };
  state.planKey = planKey();
  const names = resultNames();
  state.results = joinResults(names, r.responses, r.more_responses);
  state.marks = joinResults(names, r.sources, r.more_sources);
  state.setAside = {};
  state.contourPairs = {};
  state.curves = r.curves ?? null;
  state.shown = names[0];
  state.simulation = null;
  state.example = false;
  state.campaign = { ...newCampaign(false), title: `${titleOf(r.spec)}, from ${file.name}`.slice(0, 200) };
  clearDraft();
  setModel(r.spec.analysis_model ?? $("#model").value);
  state.models = { [names[0]]: modelChoice(), ...(r.spec.analysis_models ?? {}) };
  state.goals = {};
  state.trade = newTrade();
  // The targets and the windows the workbook keeps: what they give is worked out again, as when a study opens.
  state.target = { ...newTarget(), targets: structuredClone(r.targets ?? {}) };
  state.window = { ...newWindow(), windows: structuredClone(r.windows ?? {}) };
  state.scenarios = r.what_if ?? [];
  state.buildsOn = r.builds_on ?? []; // the round before, when the workbook is of a later round: Python checked it
  clearBalance();
  clearWhatIf();
  clearTarget();
  clearWindow();
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
/**
 * The status ticket once the analysis said it is complete: its charts say so again, or why they could not be drawn, only
 * while nothing has been said since but a note from Python on how far it has got. A refused add, say, stays.
 */
let analysisSaid = 0;

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

/** What Learn says of results that wait for Python to start: a study opened with its results is such a one. */
const RESULTS_WAITING = "Your results are in. Their analysis follows once Python has started.";

/** What Learn says with no analysis on show: that no result is in yet, or, before Python has started, that those in wait for it. */
const learnEmpty = () => (!pythonReady && anyResult(state.results) ? RESULTS_WAITING : NOTHING_TO_LEARN);

/** Take the analysis away: its runs or their results are gone. A reply still on its way is dropped. */
function clearAnalysis() {
  ticket += 1;
  state.analysis = null;
  $("#analysis").hidden = true;
  homeTerms();
  $("#printed").hidden = true;
  $("#py-analyse").hidden = true;
  $("#charts").hidden = true;
  $("#charts").replaceChildren();
  sayLearnEmpty(learnEmpty());
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
    ...splitResults(resultNames(), fitted()),
    model: asked(modelOf(name)),
    truth: state.marks[name]?.includes("simulated") ? (processOf(name)?.truth ?? null) : null,
    ...(several() ? { response: name } : {}),
    ...chosenControl(),
    ...(asideRuns(name).length ? { set_aside: asideRuns(name).map(runOrderOf) } : {}), // so Python names them apart
  };
}

/**
 * Whether an analysis compares one factor's levels, and the control Dunnett's take, and whether it sums up readings
 * repeated at a setting: its code then does the same.
 */
const comparedOf = (analysis) => ({
  compared: Boolean(analysis?.comparison?.difference || analysis?.comparison?.several),
  control: analysis?.comparison?.several?.control ?? null,
  spread: Boolean(analysis?.spread),
});

/** The control the reader picked for Dunnett's intervals, while it is a level of the study's one categorical factor. */
function chosenControl() {
  const factors = state.design?.spec.factors ?? [];
  const ok = factors.length === 1 && isSeveral(factors[0]) && levelsOf(factors[0]).includes(state.control);
  return ok ? { control: state.control } : {};
}

/**
 * Analyse the response Learn shows: its fit, its best settings and its charts. With two responses or more given a
 * goal, the balance of them follows, whichever response is shown.
 */
async function analyseRuns() {
  if (!state.design || (!anyResult(state.results) && !curvesRead())) {
    clearAnalysis();
    clearBalance();
    clearWhatIf();
    clearTarget();
    clearWindow();
    setStatus("No results to analyse yet.");
    return;
  }
  renderPicker();
  if (!state.results[state.shown]?.some((v) => v !== null)) {
    // Some other response has results, or only the curves are in: this one waits for its own.
    clearAnalysis();
    const curvesOnly = !anyResult(state.results);
    const where = curvesOnly ? "the runs' curves are drawn under What if" : "enter them in Run";
    sayLearnEmpty(`No results for ${wordsOf(state.shown).label} yet: ${where}.`);
    setStatus(`No results for ${wordsOf(state.shown).label} yet.`);
    await findBalance();
    await findCurve();
    await findWhatIf();
    findTradeOffs();
    showTarget();
    showWindow();
    return;
  }
  const mine = ++ticket;
  setStatus("Analysing...", "busy");
  const analysing = statusTicket;
  const request = analysisRequest();
  const reply = await call("api_analyze_runs", request);
  if (dropped(mine !== ticket, analysing)) return; // taken away: a new study, say
  state.analysis = reply.ok ? reply.result : null;
  renderAnalysis(reply);
  $("#printed").hidden = true;
  if (reply.ok) {
    renderPrinted(reply.result, mine); // fetched once a study; the best settings and the charts need not wait for it
    const best = await findBest(request, mine);
    if (mine !== ticket) return;
    await findNext(request); // the next step follows the best settings, and is on the page before the charts are drawn
    if (mine !== ticket) return;
    // The response surface is drawn through the best settings, so the star that marks them sits on the surface drawn.
    await drawCharts(best?.coded ? { ...request, best: best.coded } : request, mine);
    if (mine !== ticket) return;
  }
  await findBalance();
  await findCurve();
  await findWhatIf(); // after the best settings and the balance, which it opens at, and the curves it draws
  findTradeOffs(); // the settings it tries take a moment, and nothing above waits for them
  showTarget(); // on the what-if panel's fits: its search takes a moment too
  showWindow(); // last: the windows are judged through the what-if panel's settings
}

/** A button that opens Feedback on the topic of an algorithm or a model, beside a result that went wrong. */
function reportResult() {
  const button = el("button", "link", "Report this result");
  button.type = "button";
  button.dataset.feedback = "algorithm";
  const actions = el("p", "actions", "");
  actions.append(button);
  return actions;
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
    showBoxCox(null);
    homeTerms(); // out of the fold that goes, so the reader can still change the model that could not be fitted
    out.replaceChildren(...asideParts(), el("p", "error", reply.error), ...warningList(reply.warnings), reportResult());
    state.simpler = null;
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
  analysisSaid = statusTicket;
  // The terms fitted, which the design can estimate, from the runs that have a result.
  showPython("analyse", analyseCode(r.formula ?? r.model, designComplete() && !asideRuns().length, shownSpec(), vif ? fittedFormula(r.coefficients) : null, comparedOf(r)));
  const mixture = isMixture(r.spec?.design_type);
  $("#transform-choice").hidden = mixture; // a mixture's responses are analysed as measured
  state.simpler = null; // what an earlier search found goes with the analysis it was of
  $("#transform").value = r.transform ?? "";
  showBoxCox(boxCoxSentence(r.box_cox, responseWords(shownSpec()).label));
  const ratio = r.coefficients.some((c) => c.ratio !== undefined); // on the log scale, a main effect is a ratio
  const scale = r.transform ? `In coded units, fitted to ${r.equation.split(" = ")[0]}.` : "In coded units.";
  // What the results say, in sentences, with what a reader must know of the fit: its notes and its warnings.
  const shown = [
    ...asideParts(),
    // A mixture's model is chosen by name: the search for a simpler one is not offered for it.
    ...findingParts(r.findings, r.coefficients, { picked: Boolean(modelChoice().picked), simpler: !mixture }),
    ...comparisonParts(r),
    ...noteList(r.notes),
    ...warningList(reply.warnings),
  ];
  // The model itself, for the reader who wants it: its equation, its fit, its tables and each run. One fold holds
  // them, closed until it is opened.
  const parts = [
    el("p", "equation", r.equation),
    said("hint", mixture ? "In proportions: each component's share of the blend." : scale),
    ...noteList(fitLines(r), "fit"),
    el("h3", "", "Coefficients"),
    said("hint", "The most significant first, the intercept last."),
    ...coefficientsTable(r.coefficients, { ratio, vif, simulated }),
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
  parts.push(...spreadParts(r.spread), ...diagnosticParts(r.diagnostics, r.transform));
  const fold = el("details", "fold statistician", "");
  fold.id = "statistician";
  fold.open = statisticianOpen();
  fold.ontoggle = () => keepStatistician(fold.open);
  // The terms move from the fold before into this one; a term the reader has just pressed keeps the keyboard.
  const pressed = TERMS.box.contains(document.activeElement) ? document.activeElement : null;
  showTermsOfModel();
  fold.append(el("summary", "", "For the statistician: the model, its coefficients and each run"), TERMS.box, ...parts);
  out.replaceChildren(...shown, fold);
  if (pressed?.isConnected) pressed.focus();
  renderAnswer();
}

const STATISTICIAN_KEY = "factorial-statistician";

/** Whether the reader last left the fold of the model and its tables open: kept in the browser, so it holds for the next fit and the next visit. */
function statisticianOpen() {
  try {
    return localStorage.getItem(STATISTICIAN_KEY) === "open";
  } catch {
    return false;
  }
}

function keepStatistician(open) {
  try {
    if (open) localStorage.setItem(STATISTICIAN_KEY, "open");
    else localStorage.removeItem(STATISTICIAN_KEY);
  } catch {
    /* storage refused: the fold is closed at the next fit */
  }
}

/**
 * The search for a simpler model (bootstrap.simpler_model), offered among the findings: the terms the results
 * support are proposed when the reader asks, and fitted when they accept. It is made again with each analysis, so
 * what an earlier search found goes with the analysis it was of.
 */
function simplerControl() {
  const box = el("span", "simplify", "");
  box.id = "simplify";
  const ask = el("button", "link", "Simplify the model");
  const words = el("span", "", "");
  const apply = el("button", "link", "Fit these terms");
  Object.assign(ask, { type: "button", id: "simplify-ask", onclick: () => askSimpler(words, apply) });
  Object.assign(apply, { type: "button", id: "simplify-apply", hidden: true, onclick: fitSimpler });
  words.id = "simplify-words";
  words.setAttribute("role", "status");
  box.append(ask, " ", words, " ", apply);
  return box;
}

/**
 * Search for the terms of the fitted model the results support, and say them in the control's ``words``; its
 * ``apply`` button then fits them, when the reader wants that.
 */
async function askSimpler(words, apply) {
  const mine = ticket;
  apply.hidden = true;
  words.textContent = "Searching...";
  const reply = await call("api_simpler_model", analysisRequest());
  if (mine !== ticket) return; // the analysis it was asked of is gone, and its control with it
  if (!reply.ok) {
    words.textContent = reply.error;
    return;
  }
  state.simpler = reply.result.terms;
  words.textContent = simplerWords(reply.result, fmt);
  apply.hidden = !reply.result.changed || !reply.result.terms.length;
}

/** Fit the terms the search proposed: they are ticked in the list of terms, as if the reader had ticked them. */
function fitSimpler() {
  if (!state.simpler?.length) return;
  setModel({ terms: state.simpler, picked: true });
  TERMS.list.dispatchEvent(new Event("change"));
}

/** A result typed or pasted again is a new result: its run is no longer set aside from that response's fit. */
function backInTheFit(name, place) {
  if (state.setAside[name]?.includes(place)) state.setAside = toggleAside(state.setAside, name, place);
}

/**
 * Set a run aside from the fit Learn shows, or put it back, and fit the model again. The button pressed goes with
 * the analysis it was in, so the focus moves to what the page now says of the runs set aside, or to the runs' table.
 */
async function setRunAside(place) {
  state.setAside = toggleAside(state.setAside, state.shown, place);
  scheduleSave();
  showResultsCode();
  await analyseRuns();
  ($("#analysis .aside-runs button") ?? $("#analysis details.diagnostics > summary"))?.focus();
}

/**
 * The runs set aside from the fit on show, in a sentence, each with a button that puts it back: said above
 * everything else, since every number below is the model's without them.
 */
function asideParts() {
  const places = asideRuns();
  if (!places.length) return [];
  const note = el("p", "aside-runs", `${asideWords(places.map(runOrderOf))} `);
  note.id = "aside-runs";
  for (const place of places) {
    const back = el("button", "link", `Put run ${runOrderOf(place)} back`);
    back.type = "button";
    back.dataset.place = place;
    back.onclick = () => setRunAside(place);
    note.append(back, " ");
  }
  return [note];
}

const SVG = "http://www.w3.org/2000/svg";

/**
 * A word-sized picture of a finding: its ``lines`` over a level line through the center, drawn to ``reach``, the
 * one scale every picture of the list shares, so a steeper line is a larger effect. ``words`` say it to a reader
 * who does not see it.
 */
function sparkOf(lines, reach, words) {
  const svg = document.createElementNS(SVG, "svg");
  for (const [name, value] of Object.entries({ class: "spark", viewBox: "0 0 40 20", role: "img", "aria-label": words })) {
    svg.setAttribute(name, value);
  }
  const at = ([x, y]) => `${(20 + 17 * x).toFixed(1)},${(10 - (reach ? (8.5 * y) / reach : 0)).toFixed(1)}`;
  const draw = (points, cls) => {
    const line = document.createElementNS(SVG, "polyline");
    line.setAttribute("class", cls);
    line.setAttribute("points", points.map(at).join(" "));
    svg.append(line);
  };
  draw([[-1, 0], [1, 0]], "spark__base");
  lines.forEach((points, i) => draw(points, i ? "spark__line spark__line--other" : "spark__line"));
  return svg;
}

/** An element of a drawing, with its attributes. */
function svgEl(tag, attributes = {}) {
  const made = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) made.setAttribute(name, value);
  return made;
}

/**
 * A picture of the space two or three numeric factors span, as a square, or a cube seen from above its front corner:
 * the settings the runs took as rings (``made``) and those of runs not yet made as filled points (``planned``), each
 * a point of logic.js's spacePoints. Each factor runs along an edge, from the low end of its range to the high.
 * ``words`` say the picture to a reader who does not see it.
 */
function spaceOf(made, planned, factors, words) {
  const cube = factors.length === 3;
  const at = cube ? ([t, p, f]) => [85 + 38 * t - 22 * p, 80 + 8 * t + 16 * p - 36 * f] : ([t, p]) => [85 + 52 * t, 80 - 52 * p];
  const ends = [-1, 1];
  const corners = cube ? ends.flatMap((t) => ends.flatMap((p) => ends.map((f) => [t, p, f]))) : ends.flatMap((t) => ends.map((p) => [t, p]));
  const svg = svgEl("svg", { class: "space", viewBox: "0 0 170 170", role: "img", "aria-label": words });
  corners.forEach((from, i) => {
    for (const to of corners.slice(i + 1)) {
      if (from.filter((level, k) => level !== to[k]).length !== 1) continue; // an edge joins corners one factor apart
      const [[x1, y1], [x2, y2]] = [at(from), at(to)];
      const behind = cube && [from, to].some((corner) => corner.every((level) => level === -1)); // the corner the cube hides
      svg.append(svgEl("line", { class: behind ? "space__edge space__edge--back" : "space__edge", x1, y1, x2, y2 }));
    }
  });
  const draw = (cls, r) => (levels) => {
    const [cx, cy] = at(levels);
    svg.append(svgEl("circle", { class: cls, cx: cx.toFixed(1), cy: cy.toFixed(1), r }));
  };
  made.forEach(draw("space__run", 3));
  planned.forEach(draw("space__new", 4.5));
  const short = (f) => (f.name.length > 16 ? `${f.name.slice(0, 15)}\u2026` : f.name);
  const labels = cube
    ? [
        ["translate(38 146) rotate(12)", "start", `${short(factors[0])} \u2192`],
        ["translate(112 148) rotate(-36)", "start", `\u2190 ${short(factors[1])}`],
        ["translate(15 112) rotate(-90)", "start", `${short(factors[2])} \u2192`],
      ]
    : [
        ["translate(85 152)", "middle", `${short(factors[0])} \u2192`],
        ["translate(20 80) rotate(-90)", "middle", `${short(factors[1])} \u2192`],
      ];
  for (const [transform, anchor, text] of labels) {
    const label = svgEl("text", { class: "space__label", transform, "text-anchor": anchor, "aria-hidden": "true" });
    label.textContent = text;
    svg.append(label);
  }
  return svg;
}

/** The runs of a design's sheet, each as its settings by the columns' names, for spacePoints. */
const sheetRuns = (design) => design.rows.map((row) => Object.fromEntries(design.columns.map((name, j) => [name, Number(row[j])])));

/**
 * Where the best settings sit, as pictures for Learn's answer: for a study of two or three numeric factors, the
 * settings as a filled point among the runs made (spaceOf); then a line a numeric factor, from the low end of its
 * range to the high, with the setting as a point on it. Nothing for a study with no numeric factor.
 */
function answerPictures(best, design) {
  const factors = design.spec.factors;
  const numeric = best.settings.filter((s) => s.kind === "numeric" && Number.isFinite(s.coded));
  const ranged = numeric.map((s) => [s, factors.find((f) => f.name === s.name)]).filter(([, f]) => f);
  if (!ranged.length) return [];
  const parts = [];
  if ([2, 3].includes(factors.length) && ranged.length === factors.length) {
    const words = `Where these settings sit among the runs made, over ${listNames(factors.map((f) => f.name))}: the filled point.`;
    const at = factors.map((f) => ranged.find(([s]) => s.name === f.name)[0].coded);
    parts.push(spaceOf(spacePoints(sheetRuns(design), factors), [at], factors, words));
  }
  const tracks = el("ul", "tracks", "");
  for (const [setting, f] of ranged) {
    const words = `${f.name} = ${num(setting.value)}, in a range from ${num(f.low)} to ${num(f.high)}`;
    const line = svgEl("svg", { class: "track", viewBox: "0 0 120 12", role: "img", "aria-label": words });
    line.append(
      svgEl("line", { class: "space__edge", x1: 6, y1: 6, x2: 114, y2: 6 }),
      svgEl("line", { class: "space__edge", x1: 6, y1: 2, x2: 6, y2: 10 }),
      svgEl("line", { class: "space__edge", x1: 114, y1: 2, x2: 114, y2: 10 }),
      svgEl("circle", { class: "space__new", cx: (60 + 54 * setting.coded).toFixed(1), cy: 6, r: 4.5 }),
    );
    const row = el("li", "", "");
    const ends = [f.low, f.high].map((end) => el("span", "tracks__end", num(end)));
    for (const end of ends) end.setAttribute("aria-hidden", "true");
    row.append(el("span", "tracks__name", f.name), ends[0], line, ends[1]);
    tracks.append(row);
  }
  parts.push(tracks);
  return parts;
}

/** Whether the numbers behind the findings are shown beside them: the reader's choice, for as long as the page is open. */
let findingNumbersShown = false;

/** Show the numbers behind each finding of the ``list`` beside it, or take them away, as the reader last chose. */
function showFindingNumbers(list, button) {
  for (const item of list.querySelectorAll("li[data-numbers]")) {
    item.querySelector(".finding__numbers")?.remove();
    if (findingNumbersShown) item.append(el("span", "finding__numbers", item.dataset.numbers));
  }
  button.textContent = findingNumbersShown ? "Hide the numbers" : "Show the numbers";
  button.setAttribute("aria-pressed", String(findingNumbersShown));
}

/**
 * A finding as an item of the list: its sentence, with each of its pictures before the words it shows. The pictures
 * of the other real factors stand each before its factor; any other picture opens the sentence.
 */
function findingItem(finding, sentence, pictures, reach, coefficients) {
  const item = el("li", "", "");
  const before = finding.kind === "also" ? finding.factors.map(alsoWords) : pictures.map(() => "");
  let rest = sentence;
  pictures.forEach((lines, i) => {
    const at = Math.max(0, rest.indexOf(before[i]));
    item.append(rest.slice(0, at), sparkOf(lines, reach, pictureWords(finding, lines)));
    rest = rest.slice(at);
  });
  item.append(rest);
  const numbers = findingNumbers(finding, coefficients, fmt, pval);
  if (numbers) item.dataset.numbers = numbers;
  return item;
}

/**
 * What the analysis found, each finding of Python's in a sentence beside its picture: the same results give the
 * same words, and a picture is drawn from the ``coefficients`` of the terms its finding names. A model whose terms
 * were ``picked`` from these results says so last, since it bears on all that is said before it. Under the list, a
 * switch shows the numbers behind each finding; and the search for a ``simpler`` model, when it is offered, stands
 * with the terms that show no sign of an effect, or under the list when every term shows one.
 */
function findingParts(findings, coefficients, { picked, simpler }) {
  const words = responseWords(shownSpec());
  const known = (findings ?? []).filter((finding) => findingWords([finding], words).length);
  if (!known.length && !picked && !simpler) return [];
  const pictures = known.map((finding) => pictureLines(finding, coefficients));
  const reach = pictureReach(pictures);
  const list = el("ul", "findings", "");
  list.id = "findings";
  const items = known.map((finding, i) => findingItem(finding, findingWords([finding], words)[0], pictures[i], reach, coefficients));
  list.append(...items);
  if (picked) list.append(el("li", "picked", PICKED_WORDS));
  const more = el("p", "hint findings__more", "");
  if (items.some((item) => item.dataset.numbers)) {
    const numbers = el("button", "link", "");
    numbers.type = "button";
    numbers.id = "findings-numbers";
    numbers.onclick = () => {
      findingNumbersShown = !findingNumbersShown;
      showFindingNumbers(list, numbers);
    };
    showFindingNumbers(list, numbers);
    more.append(numbers);
  }
  if (simpler) {
    const unclear = items[known.findIndex((finding) => finding.kind === "no_sign")];
    if (unclear) unclear.append(" ", simplerControl());
    else more.append(more.childNodes.length ? " " : "", simplerControl());
  }
  return [el("h3", "", "What the results say"), list, ...(more.childNodes.length ? [more] : [])];
}

/**
 * Each run as the fitted model sees it, under a heading the reader opens: open already when a run is one to look
 * at again. On a transform's scale the fitted values and residuals are on that scale, which the hint says.
 */
/** How a run is marked on every chart and table that shows it, as the table of the runs says it. */
const RUN_MARKING =
  "Rest on a run here, in the run sheet or on a chart to mark it wherever it is shown; press it to keep it marked, and " +
  "again, or Escape, to let it go.";

function diagnosticParts(diagnostics, transform) {
  if (!diagnostics?.length) return [];
  const unusual = diagnostics.filter((d) => d.unusual);
  const box = el("details", "diagnostics", "");
  box.id = "diagnostics";
  box.open = unusual.length > 0;
  box.append(
    el("summary", "", unusual.length ? `Each run, as the model sees it (${unusual.length} to look at again)` : "Each run, as the model sees it"),
    said(
      "hint",
      "A residual is the result less what the model fits there" +
        (transform ? ", on the scale the model is fitted on" : "") +
        "; in standard deviations it is judged with the run itself left out of the fit. Leverage is how much the run " +
        "alone decides its own fitted value, from 0 to 1, and Cook's distance how far the whole model moves without it. " +
        RUN_MARKING,
    ),
    table(
      ["Run", "Fitted", "Residual", "In standard deviations", "Leverage", "Cook's distance", "Look again"],
      diagnostics.map((d) => [d.run, fmt(d.fitted), fmt(d.residual), fmt(d.studentized), fmt(d.leverage), fmt(d.cooks), d.unusual ? "yes" : ""]),
      { Residual: "residuals" },
      { run: (row) => row[0] },
    ),
  );
  // Each run can be set aside from this fit: its result stays on the run sheet, and the model is fitted without it.
  const order = state.design.columns.indexOf("Run order");
  box.querySelector("thead tr").append(el("th", "", "Set aside"));
  box.querySelectorAll("tbody tr").forEach((tr, i) => {
    const place = state.design.rows.findIndex((row) => row[order] === diagnostics[i].run);
    const aside = el("button", "link", "Set aside");
    aside.type = "button";
    aside.setAttribute("aria-label", `Set run ${diagnostics[i].run} aside from this fit`);
    aside.onclick = () => setRunAside(place);
    const cell = el("td", "", "");
    if (place >= 0) cell.append(aside);
    tr.append(cell);
  });
  return [box];
}

/**
 * The results read more than once at a setting (``spread``, as the analysis gives it): what they say, in sentences;
 * the response table, each summary averaged at each level of each factor; how much each factor moves each summary, and
 * at which level the spread is smallest and the signal-to-noise ratio highest; and each setting's readings summed up.
 * Nothing unless two settings were read twice or more.
 */
function spreadParts(spread) {
  if (!spread) return [];
  const { label, units } = responseWords(shownSpec());
  const factorUnits = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const ratio = spread.kind ? ["S/N"] : []; // the sentence above names the ratio: larger, smaller or nominal
  const tableHeading = el("h4", "", "");
  tableHeading.append(term("Response table", { key: "response-table", alone: true }));
  const moved = (f, key) => (f.rank[key] === null ? "" : `${fmt(f.delta[key])} (${f.rank[key]})`); // null: no setting gave it
  const best = (f, key) => (f.best[key] === undefined ? "" : String(f.best[key]));
  return [
    el("h3", "", "The spread of the readings at each setting"),
    ...spreadSentences(spread, { response: label, units, factorUnits }).map((line) => said("", line)),
    tableHeading,
    table(
      ["Factor", "Level", "Settings", "Mean", "ln s", ...ratio],
      spread.factors.flatMap((f) =>
        f.levels.map((l) => [f.name, String(l.level), l.n, fmt(l.mean), fmt(l.ln_std_dev), ...(spread.kind ? [fmt(l.sn)] : [])]),
      ),
      { "ln s": "ln-s", "S/N": "signal-to-noise" },
    ),
    el("h4", "", "Each factor's delta, and its rank"),
    table(
      ["Factor", "Mean", "ln s", ...ratio, "Least spread at", ...(spread.kind ? ["Best S/N at"] : [])],
      spread.factors.map((f) => [
        f.name,
        moved(f, "mean"),
        moved(f, "ln_std_dev"),
        ...(spread.kind ? [moved(f, "sn")] : []),
        best(f, "ln_std_dev"),
        ...(spread.kind ? [best(f, "sn")] : []),
      ]),
    ),
    said("hint", "A factor's delta is its largest average less its smallest; the factor of the largest has rank 1, in brackets."),
    el("h4", "", "Each setting's readings"),
    table(
      ["Setting", "Runs", "Readings", "Mean", "Std. dev.", "ln s", ...ratio],
      spread.settings.map((s) => [
        Object.entries(s.settings)
          .map(([name, level]) => `${name} ${level}`)
          .join(", "),
        s.runs.join(", "),
        s.n,
        fmt(s.mean),
        fmt(s.std_dev),
        fmt(s.ln_std_dev),
        ...(spread.kind ? [fmt(s.sn)] : []),
      ]),
      { "ln s": "ln-s", "S/N": "signal-to-noise" },
    ),
  ];
}

// ---------------------------------------------------------------- the numbers a catalog study's source prints
/** The numbers a catalog study's source prints that Learn shows beside its own, fetched once a study. */
const printedFetched = new Map();

function printedOf(source) {
  if (!source?.catalog || !source.printed) return Promise.resolve([]);
  if (!printedFetched.has(source.catalog)) {
    const asked = fetch(`catalog/printed/${source.catalog}.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => (Array.isArray(data?.printed) ? data.printed : []))
      .catch(() => []);
    printedFetched.set(source.catalog, asked);
  }
  return printedFetched.get(source.catalog);
}

/** A link to where a source prints a study, when the catalog gives its address: as a list, to spread into a sentence. */
const sourceLink = (source) => (source.url ? [link(source.url, "Read the source", "source"), "."] : []);

/** The terms of the model Learn fits the response shown with: those ticked, or those a named model holds. */
function fittedTerms() {
  const model = modelChoice();
  const factors = state.design.spec.factors;
  return typeof model === "object" ? model.terms : namedModelTerms(model || state.design.spec.model, factors);
}

/**
 * Show beside Learn's analysis the numbers the study's source prints for the model fitted: each with the page's own
 * number, written as the source writes it, and whether the two agree to the printed rounding, or the reason the
 * catalog knows they do not. A model the source prints other numbers for is offered, to fit in its place.
 */
async function renderPrinted(result, mine) {
  const source = state.campaign?.source;
  const numbers = source && state.shown === resultNames()[0] ? await printedOf(source) : [];
  if (mine !== ticket || !numbers.length) return; // a later analysis, or another study, has the page now
  const runs = (state.results[state.shown] ?? []).filter((v) => v !== null).length;
  const fitted = modelKey(fittedTerms());
  const rows = numbers
    .filter((n) => modelKey(n.model) === fitted)
    .map((n) => {
      // A number the catalog knows is printed wrong keeps its reason, however near it: Dataplot's single precision
      // misses the last digit by about as much as two right ways of working the same fit differ by.
      const value = printedValue(result, n.read, runs);
      const agrees = !n.reason && value !== null && printedAgrees(value, n.agree);
      const verdict = n.reason ? `differs: ${n.reason}` : agrees ? "agrees" : "differs";
      return { n, value, agrees, verdict };
    })
    .filter(({ value }) => value !== null);
  const others = new Map();
  for (const n of numbers.filter((x) => modelKey(x.model) !== fitted)) {
    const key = modelKey(n.model);
    others.set(key, { terms: n.model, count: (others.get(key)?.count ?? 0) + 1 });
  }
  const from = said("hint", rows.length ? "" : "The source prints no number for the model fitted here. ");
  from.append(`From ${source.cite}${source.where ? `, ${source.where}` : ""}. `, ...sourceLink(source));
  const parts = [];
  if (rows.length) {
    const agree = rows.filter((r) => r.agrees).length;
    const known = rows.filter((r) => !r.agrees && r.n.reason).length;
    const lead = `Factorial's analysis gives ${agree} of the ${rows.length} numbers the source prints for this model, to the printed rounding.`;
    const errata = known ? ` ${known} differ${known === 1 ? "s" : ""} for a reason the catalog records.` : "";
    parts.push(el("p", "lead", `${lead}${errata}`));
  }
  parts.push(from);
  if (rows.length) {
    parts.push(
      table(
        ["Number", "Where", "Printed", "Factorial", "Verdict"],
        rows.map(({ n, value, verdict }) => [
          `${n.what.charAt(0).toUpperCase()}${n.what.slice(1)}${n.term ? `, ${n.term}` : ""}`,
          n.where,
          n.printed,
          printedLike(value, n.printed),
          verdict,
        ]),
      ),
    );
  }
  for (const { terms, count } of [...others.values()].sort((a, b) => b.count - a.count)) {
    const line = el("p", "printed-model", `It prints ${count} number${count === 1 ? "" : "s"} for the model ${modelTermsWords(terms)}. `);
    const fit = el("button", "link", "Fit this model");
    fit.type = "button";
    fit.onclick = () => {
      setModel({ terms });
      if (state.shown) state.models[state.shown] = modelChoice();
      scheduleAnalysis();
      scheduleSave();
    };
    line.append(fit);
    parts.push(line);
  }
  $("#printed-out").replaceChildren(...parts);
  $("#printed").hidden = false;
}

/** The study's specification as the response Learn shows reads it: that response as its ``response``. */
const shownSpec = () => forResponse(state.design.spec, state.shown);

/** The words the comparison of one factor's levels is said and drawn in: the response's, and the factor's units. */
function comparisonWords() {
  const { label, units } = responseWords(shownSpec());
  return { response: label, units, factorUnits: state.design.spec.factors[0]?.units ?? "" };
}

/**
 * A study of one factor, its levels compared (``r.comparison``, as the analysis gives it): the difference between
 * the two levels and how sure it is, in a sentence; each level's runs summarised; and the difference tested four
 * ways, each test's name a word the glossary opens, or three ways within pairs for runs in blocks of two. The
 * model's equation and coefficients follow, as for any study.
 */
function comparisonParts(r) {
  const c = r.comparison;
  if (!c) return [];
  if (c.levels.length > 2 && isCategorical(state.design.spec.factors[0])) return severalParts(c);
  const d = c.difference;
  const words = comparisonWords();
  const at = (level) => (words.factorUnits ? `${level} ${words.factorUnits}` : String(level));
  const sentence = comparisonSentence(c, words);
  const parts = [
    el("h3", "", d ? `Whether ${at(d.to)} differs from ${at(d.from)}` : "Each level's results"),
    ...(sentence ? [said("", sentence)] : []),
    levelTable(c),
  ];
  if (!d) {
    if (c.levels.length === 2) parts.push(said("hint", "A difference is tested once each level has two runs or more."));
    return parts;
  }
  const named = d.paired ? PAIRED_TESTS : COMPARISON_TESTS;
  const tests = testTable(
    named,
    ["Test", "95% interval", "p-value"],
    d.tests.map((t) => [
      `${named[t.test].name}: ${named[t.test].how}`,
      t.ci_low === null || t.ci_low === undefined ? "" : `${fmt(t.ci_low)} to ${fmt(t.ci_high)}`,
      pval(t.p_value),
    ]),
    { "95% interval": "interval", "p-value": "p-value" },
  );
  const ways = d.paired ? `within each of the ${d.paired} pairs, tested three ways` : "tested four ways";
  const heading = el("h4", "", `The difference, ${d.to} less ${d.from}, ${ways}`);
  parts.push(heading, tests, ...noteList(comparisonNotes(r)));
  return parts;
}

/**
 * A table of tests, each row's first cell a test's name and what it rests on (``name: how``), as ``named`` lists
 * them: each name opens what the test is, and the words after it say what it rests on.
 */
function testTable(named, columns, rows, terms) {
  const tests = table(columns, rows, terms);
  const body = tests.querySelector("table");
  body._cell = (row, col) => {
    if (col !== 0) return null;
    const [name, how] = row[0].split(": ");
    const key = Object.values(named).find((t) => t.name === name).key;
    return `<span class="term" data-term="${key}">${escape(name)}</span>: ${escape(how)}`;
  };
  fillBody(body, body._rows);
  return tests;
}

/** The words of a comparison that open their definition. */
const SEVERAL_TERMS = { "one-way analysis of variance": "one-way-anova", "Tukey's intervals": "tukey" };

/**
 * A categorical factor's three levels or more compared (``c.several``, as the analysis gives it): whether their averages
 * differ at all, in a sentence; each level's runs summarised; every pair of levels by Tukey's intervals; and each level
 * against the control by Dunnett's, the control a level the reader picks. Each test's name opens what it is.
 */
function severalParts(c) {
  const v = c.several;
  const sentence = severalSentence(c, comparisonWords());
  const parts = [el("h3", "", `Whether the levels of ${c.factor} differ`)];
  if (sentence) {
    const line = el("p", "", "");
    line.append(...marked(sentence, SEVERAL_TERMS));
    parts.push(line);
  }
  parts.push(levelTable(c));
  if (!v) {
    parts.push(said("hint", "The levels are compared once a level has two runs or more, to measure the noise."));
    return parts;
  }
  parts.push(...checkParts(c));
  const interval = (x) => `${fmt(x.ci_low)} to ${fmt(x.ci_high)}`;
  const heading = (text, word, key) => {
    const h = el("h4", "", text);
    h.append(term(word, { key, alone: true }));
    return h;
  };
  const control = el("select", "", "");
  control.id = "control";
  for (const level of c.levels.map((l) => String(l.level))) control.append(new Option(level, level, false, level === v.control));
  control.onchange = () => {
    state.control = control.value;
    analyseRuns();
  };
  const pick = el("label", "", "Compared with ");
  pick.append(control);
  const picker = el("p", "control-pick", "");
  picker.append(pick);
  parts.push(
    heading("Every pair of levels, by ", "Tukey's intervals", "tukey"),
    table(
      ["Pair", "Difference", "95% interval", "p-value"],
      v.pairs.map((x) => [`${x.to} less ${x.from}`, fmt(x.estimate), interval(x), pval(x.p_value)]),
      { "95% interval": "interval", "p-value": "p-value" },
    ),
    heading("Each level against a control, by ", "Dunnett's intervals", "dunnett"),
    picker,
    table(
      ["Level", `Difference from ${v.control}`, "95% interval", "Differs"],
      v.versus.map((x) => [String(x.level), fmt(x.estimate), interval(x), x.differs ? "yes" : "no"]),
      { "95% interval": "interval" },
    ),
  );
  return parts;
}

/** What the one-way analysis of variance assumes, checked: a row a check (``c.several.checks``), and what each means. */
function checkParts(c) {
  const rows = (c.several.checks ?? []).map((x) => {
    const check = LEVEL_CHECKS[x.test];
    return [`${check.name}: ${check.how}`, fmt(x.statistic), x.df, pval(x.p_value)];
  });
  const terms = { df: "df", "p-value": "p-value" };
  return [
    el("h4", "", "Checks of what the analysis of variance assumes"),
    ...(rows.length ? [testTable(LEVEL_CHECKS, ["Check", "Statistic", "df", "p-value"], rows, terms)] : []),
    ...noteList(checkSentences(c)),
  ];
}

/** Each level's runs summarised, a row a level: how many, their average, standard deviation and the average's interval. */
function levelTable(c) {
  return table(
    ["Level", "Runs", "Average", "Std. dev.", "95% interval"],
    c.levels.map((l) => [String(l.level), l.n, fmt(l.mean), fmt(l.std_dev), l.ci_low === null ? "" : `${fmt(l.ci_low)} to ${fmt(l.ci_high)}`]),
    { "95% interval": "confidence-interval" },
  );
}

/** What the comparison's numbers rest on: the model's coefficient as half the difference, and the randomization test's count. */
function comparisonNotes(r) {
  const d = r.comparison.difference;
  const notes = [];
  const coefficient = r.coefficients.find((c) => c.term === r.comparison.factor);
  if (coefficient) {
    notes.push(
      `The model's coefficient of ${r.comparison.factor}, ${fmt(coefficient.coefficient)}, is half the difference: in ` +
        `coded units ${d.from} is -1 and ${d.to} is +1, two units apart.`,
    );
  }
  if (d.paired) {
    notes.push(
      `Each block holds one run at each level, so each pair's difference leaves out what its two runs share: the ` +
        "model's test of the factor, beside its blocks, is the paired t-test squared.",
    );
  }
  const shuffled = d.tests.find((t) => t.test === "randomization" || t.test === "sign_flip");
  const [ways, what] = d.paired ? ["have their levels swapped", "patterns of swaps"] : ["be split", "splits"];
  const runs = d.paired ? "pairs" : "runs";
  if (shuffled && !shuffled.exact) {
    notes.push(`The ${runs} can ${ways} more ways than the randomization test counts: it draws ${shuffled.n_arrangements} of the ${what} at random.`);
  } else if (shuffled && shuffled.ci_low === null) {
    notes.push(`The ${runs} can ${ways} ${shuffled.n_arrangements} ways, too few for the randomization test to give a 95% interval.`);
  }
  if (d.true !== null && d.true !== undefined) notes.push(`The true difference, in the process the results were simulated from: ${fmt(d.true)}.`);
  return notes;
}

/**
 * The fit at a glance, a line each. With several responses the completed runs are those with every result in, as the
 * Run tab counts them, and a line says how a run with some of its results is fitted: for the responses it has.
 */
function fitLines(r) {
  const aside = new Set(asideRuns().map(runOrderOf));
  const pending = r.pending_runs.filter((run) => !aside.has(run));
  if (!several()) return fitStatistics(r.summary, r.n_completed, pending);
  const names = resultNames();
  const confirming = new Set(apartRows(state.design));
  const complete = completeRuns(state.results, names).filter((ok, i) => ok && !confirming.has(i)).length;
  const order = state.design.columns.indexOf("Run order");
  const runs = partialRuns(state.results, names)
    .filter((i) => !confirming.has(i))
    .map((i) => state.design.rows[i][order] ?? i + 1);
  const lines = fitStatistics(r.summary, complete, pending);
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

/**
 * Analyse a response as measured ("") or on a transform's scale (a key of bootstrap.TRANSFORMS). The specification
 * keeps the choice, as ``analysis_transforms``, so every reply, saved study and workbook has it.
 */
function setTransform(name, transform) {
  const { analysis_transforms: kept = {}, ...spec } = state.design.spec;
  const { [name]: _, ...others } = kept;
  const chosen = transform ? { ...others, [name]: transform } : others;
  state.design.spec = Object.keys(chosen).length ? { ...spec, analysis_transforms: chosen } : spec;
}

/** Say what Box-Cox makes of the response, with a button that fits what it points to; hidden when it says nothing. */
function showBoxCox(hint) {
  $("#box-cox").hidden = !hint;
  if (!hint) return;
  $("#box-cox-words").textContent = hint.words;
  const apply = $("#box-cox-apply");
  apply.hidden = hint.apply === null;
  apply.textContent = hint.button;
  apply.dataset.transform = hint.apply ?? "";
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
/**
 * The model Learn fits for the response on show: a named one, or the terms ticked. Terms the search for a simpler
 * model picked from the results are marked so (``picked``) while the list holds exactly them, and the mark is
 * kept with the study: a reader who changes the terms has chosen them, and the mark goes.
 */
function modelChoice() {
  if ($("#model").value !== CHOSEN_TERMS) return $("#model").value;
  const terms = tickedTerms();
  return TERMS.list.dataset.picked === termsKey(terms) ? { terms, picked: true } : { terms };
}

/** A model as Python takes it: its name, or its terms. That the terms were picked from the results is the page's to say. */
const asked = (model) => (model && typeof model === "object" ? { terms: model.terms } : model);

const tickedTerms = () => [...TERMS.list.querySelectorAll("input:checked")].map((b) => JSON.parse(b.value));

/** List every term the design's factors allow, grouped, with `ticked` ticked. */
function renderTermList(ticked) {
  const chosen = new Set(ticked.map((term) => JSON.stringify(term)));
  const groups = new Map();
  const factors = state.design.spec.factors;
  // Past MAX_INTERACTION_FACTORS factors the list holds the interactions the reader names, with the boxes to name one.
  TERMS.adder.hidden = !screensOnly(factors.length);
  if (screensOnly(factors.length)) fillTermAdder(factors.map((f) => f.name));
  for (const { term, label, group } of candidateTerms(factors, ticked, state.design.facts?.aliases)) {
    const tick = el("input", "", "");
    tick.type = "checkbox";
    tick.value = JSON.stringify(term);
    tick.checked = chosen.has(tick.value);
    const item = el("label", "", "");
    item.append(tick, ` ${label}`);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(item);
  }
  TERMS.list.replaceChildren(
    ...[...groups].flatMap(([group, items]) => {
      const row = el("div", "group", "");
      row.append(...items);
      return [el("h4", "", group), row];
    }),
  );
}

/** List the factors in the two boxes that name an interaction, keeping each box's factor when it is still there. */
function fillTermAdder(names) {
  for (const [id, start] of [["#term-first", names[0]], ["#term-second", names[1]]]) {
    const select = $(id);
    const before = select.value;
    select.replaceChildren(...names.map((name) => Object.assign(el("option", "", name), { value: name })));
    select.value = names.includes(before) ? before : start;
  }
  showTermAdder();
}

/** A factor has no interaction with itself: the button adds one only when the two boxes name different factors. */
const showTermAdder = () => (TERMS.add.disabled = TERMS.first.value === TERMS.second.value);

/** Add the interaction the two boxes name to the terms ticked, and fit the model with it. */
function addNamedTerm() {
  const names = state.design.spec.factors.map((f) => f.name);
  const term = interactionOf(TERMS.first.value, TERMS.second.value, names);
  if (!term) return;
  const ticked = tickedTerms().filter((t) => JSON.stringify(t) !== JSON.stringify(term));
  renderTermList([...ticked, term]);
  TERMS.list.dispatchEvent(new Event("change"));
}

/** Start the list of terms from the model chosen before, or the one the design was made for. */
function startTermList() {
  renderTermList(namedModelTerms(modelBefore || state.design.spec.model, state.design.spec.factors));
}

/** Put a model into the analysis: a named one, "" for the design's own, or {terms} to tick. */
function setModel(model) {
  delete TERMS.list.dataset.picked;
  if (model && typeof model === "object") {
    models.choose(CHOSEN_TERMS);
    renderTermList(model.terms);
    if (model.picked) TERMS.list.dataset.picked = termsKey(model.terms);
  } else {
    models.choose(model);
    if (model === CHOSEN_TERMS) startTermList();
  }
  showTerms();
}

/**
 * Stand the terms above the analysis, where they are before the first one, on show while the model is the reader's
 * own. With no fold to hold them (nothing analysed yet, or a model that could not be fitted) they are still in
 * reach there, so a model too large for the runs can be made smaller.
 */
function homeTerms() {
  $("#analysis").before(TERMS.box);
  TERMS.box.hidden = $("#model").value !== CHOSEN_TERMS || !state.design;
}

/** The model changed with no analysis on show: the terms above it follow. In an analysis's fold, its next one shows them. */
function showTerms() {
  if (!TERMS.box.closest("details.statistician")) homeTerms();
}

/**
 * Show the model an analysis fitted as its terms, those in it ticked, for its fold: a named model's own, or the
 * reader's as they stand. A mixture's model is chosen by its name, and has no terms to press.
 */
function showTermsOfModel() {
  TERMS.box.hidden = blendStudy();
  if (TERMS.box.hidden) return;
  if ($("#model").value !== CHOSEN_TERMS) renderTermList(namedModelTerms($("#model").value || state.design.spec.model, state.design.spec.factors));
  TERMS.designed.disabled = $("#model").value === "";
}

/**
 * A term was pressed. A named model's terms become the reader's own. A term put in brings the main effects it is
 * made of when they are out; one taken out takes nothing with it, so that a factor nested in another can be written.
 */
function termPressed(e) {
  if ($("#model").value !== CHOSEN_TERMS) models.choose(CHOSEN_TERMS);
  TERMS.designed.disabled = false; // the model is the reader's own from here
  if (e.target === TERMS.list) return; // the terms were set as a whole (fitSimpler), not pressed
  TERMS.said.textContent = "";
  if (!e.target.checked) return;
  const term = JSON.parse(e.target.value);
  const brought = mainEffectsBrought(tickedTerms(), term);
  const boxes = [...TERMS.list.querySelectorAll("input")];
  for (const main of brought) boxes.find((box) => box.value === JSON.stringify(main)).checked = true;
  if (brought.length) TERMS.said.textContent = broughtWords(term, brought);
}

/** Search for the terms the results support (bootstrap.simpler_model) and fit them at once, saying what was kept. */
async function keepClearTerms() {
  const mine = ticket;
  TERMS.said.textContent = "Searching...";
  const reply = await call("api_simpler_model", analysisRequest());
  if (mine !== ticket) return; // another analysis has the page now
  TERMS.said.textContent = reply.ok ? simplerWords(reply.result, fmt) : reply.error;
  if (!reply.ok || !reply.result.changed || !reply.result.terms.length) return;
  state.simpler = reply.result.terms;
  fitSimpler();
}

// ---------------------------------------------------------------- charts
function chartCard(key, caption, wide = false) {
  const card = el("figure", wide ? "chart wide" : "chart", "");
  card.dataset.chart = key;
  const plot = el("div", "plot", "");
  const { title: words, glossary } = CHARTS[key];
  const title = el("h3", "", glossary ? "" : words);
  if (glossary) title.append(term(words, { key: glossary, alone: true }));
  card.append(title, plot, ...(caption ? [el("figcaption", "hint", caption)] : []));
  return { card, plot };
}

const COEFFICIENT_VIEW = "factorial-coefficients";

/** How the reader last chose to see the coefficients: their estimates, or the book's Pareto plot. */
function coefficientView() {
  try {
    const kept = localStorage.getItem(COEFFICIENT_VIEW);
    return ["pareto", "effects"].includes(kept) ? kept : "estimates";
  } catch {
    return "estimates";
  }
}

/**
 * The coefficients' card, with a switch between three views of the same analysis: each estimate
 * with its interval (and the true value, when simulated), the book's Pareto plot of their sizes, or
 * the effects in the response's own units, against the smallest change the reader said is worth finding.
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
  // On a transform's scale a coefficient is no change in the response's own units, so its effects are not drawn so.
  const inUnits = !mixture && !analysis.transform;
  const offered = [["estimates", "Estimates"], ...(mixture ? [] : [["pareto", "Pareto plot"]]), ...(inUnits ? [["effects", "Effects"]] : [])];
  const words = responseWords(shownSpec());
  const worth = Number(shownSpec().response_info?.smallest_change);
  const buttons = offered.map(([view, text]) => {
    const button = el("button", "", "");
    button.append(view === "pareto" ? term(text, { key: "pareto" }) : text);
    if (view === "effects") button.title = `How far each term moves ${words.label} over the ranges, in its own units`;
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
    view === "pareto"
      ? paretoFigure(analysis.coefficients, simulated, analysis.lenth)
      : view === "effects"
        ? effectsFigure(analysis.coefficients, { simulated, change: worth > 0 ? worth : null, ...words })
        : coefficientFigure(analysis.coefficients, simulated, undefined, { mixture });
  const show = async (asked, version) => {
    const view = offered.some(([one]) => one === asked) ? asked : "estimates"; // a view kept from a study that had it
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.view === view));
    const python = figurePython(view === "estimates" ? "coefficients" : view);
    if (version) await draw(plot, figure(view), version, figure(view)().height, { python });
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
/** One more each time the charts are drawn: a drawing still on its way when another starts stops where it is. */
let chartTicket = 0;

async function drawCharts(request, mine, given = null) {
  const drawing = ++chartTicket;
  state.charted = null; // until this analysis's figures are known
  const overtaken = () => mine !== ticket || drawing !== chartTicket;
  const charts = $("#charts");
  charts.hidden = false;
  const simulated = Boolean(request.truth);
  const coefficients = coefficientCard(state.analysis, simulated);
  charts.replaceChildren(coefficients.card);
  await coefficients.show(coefficientView());

  const plots = given ?? (await libraryPlots(request, mine));
  if (overtaken()) return;
  const version = plots?.plotly_js ?? state.plotlyJs;
  if (!version) {
    coefficients.card.append(el("p", "hint", "The charts could not be drawn here; the tables above hold the numbers."));
    return;
  }
  state.plotlyJs = version;
  if (state.whatIf) drawMap(state.whatIf); // the map waited for Plotly, which the first chart brings
  try {
    await coefficients.show(coefficientView(), version);
    await drawBestChart();
    if (!plots || overtaken()) return;
    const figures = Object.fromEntries(Object.entries(plots.figures).map(([k, v]) => [k, JSON.parse(v)]));
    if (figures.true_contour) shareContourScale(figures.contour, figures.true_contour);
    // A mixture's best blend is starred on its triangle, or where its Cox traces, drawn through it, cross.
    const blend = plots.held_best && blendStudy() && state.best?.settings ? state.best : null;
    const starred = plots.held_best && (blend !== null || starBest(figures, plots.contour_factors));
    // A mixture's response over its blends is drawn here, in the page's colours: a triangle for three components,
    // else the trace along each component's Cox direction.
    const words = { response: responseWords().label, units: responseWords().units };
    const { ternary, cox } = plots;
    if (ternary) {
      const range = ternaryRange(ternary);
      figures.ternary = () => ternaryFigure(ternary, { ...words, range, best: blend?.coded ?? null });
      if (ternary.grid.true) figures.true_ternary = () => ternaryFigure(ternary, { ...words, range, values: "true" });
    }
    if (cox) figures.cox = () => coxFigure(cox, { ...words, best: blend && cox.through === "best" ? blend.predicted : null });
    const levels = state.analysis?.comparison;
    if (levels) figures.levels = () => levelsFigure(levels, comparisonWords());
    const limits = { shaded: plots.shaded, held: plots.held, factors: state.design.spec.factors, starred };
    limits.chosen = Boolean(plots.contour_chosen);
    // The bands around the lines of each factor's effect, which its caption says how to read: none without error left.
    limits.banded = Boolean(figures.perturbation?.data.some((trace) => trace.meta === "interval"));
    const captions = plots.held_best ? { ...CAPTIONS, contour: bestContourCaption } : CAPTIONS;
    const captioned = (key) => {
      const caption = typeof captions[key] === "function" ? captions[key](plots.contour_factors, limits) : captions[key];
      return chartCard(key, caption);
    };
    const python = (key) => figurePython(key, plots);
    // The figures of this analysis, in the order the page has them: a notebook of the study ends with their code.
    state.charted = { keys: Object.keys(CHARTS).filter((key) => key === "coefficients" || figures[key]), plots };
    for (const key of ["levels", "contour", "true_contour", "ternary", "true_ternary", "cox", "residuals", "half_normal"]) {
      if (!figures[key]) continue;
      const card = captioned(key);
      if (key === "contour" && plots.axes?.length > 2) card.card.append(contourPicker(plots));
      charts.append(card.card);
      await draw(card.plot, figures[key], version, chartHeight(key), { python: python(key) });
      if (overtaken()) return;
    }
    const more = MORE_CHARTS.filter((key) => figures[key]);
    if (more.length) charts.append(moreCharts(more, figures, captioned, version, python));
  } catch (err) {
    charts.append(el("p", "hint", `The charts could not be drawn (${err.message}); the tables above hold the numbers.`));
  }
}

/**
 * What gives the Python that draws one of Learn's figures, for the figure's own window: the code of the design,
 * its results and their analysis, as the analysis's panel shows it, then the figure's own. ``plots`` is the charts'
 * reply: its ``plot_options`` hold what the page asked the library for, for a figure the library drew. A figure the
 * page drew itself has none, and its code is Plotly's, from the same analysis. The code is written when it is asked
 * for, from the analysis shown then; it is null for a figure with no code to show.
 */
function figurePython(key, plots = null) {
  return () => {
    const figure = figureAlone(key, plots);
    return figure && cumulative(...["design", "run", "analyse"].map((step) => stepCode[step]).filter(Boolean), figure);
  };
}

/** The code that draws one figure, after the analysis's: the library's call, or Plotly's for a figure of the page's. */
function figureAlone(key, plots = null) {
  const options = plots?.plot_options?.[key];
  const { formula, model, coefficients, lenth } = state.analysis;
  const spec = shownSpec();
  const complete = designComplete() && !asideRuns().length;
  if (options) return figureCode(CHARTS[key].title, options, formula ?? model, complete, spec);
  if (key === "effects") {
    const worth = Number(spec.response_info?.smallest_change);
    return ownFigureCode(key, CHARTS.coefficients.title, spec, { change: worth > 0 ? worth : null });
  }
  if (key === "pareto") {
    // Lenth's margins stand in for the intervals a fit that uses every run does not have.
    const margins = Boolean(lenth) && !coefficients.some((c) => c.ci_high !== null && c.ci_high !== undefined);
    return ownFigureCode(key, CHARTS.coefficients.title, spec, { lenth: margins, model: formula ?? model, complete });
  }
  const about = { contour: plots?.plot_options?.contour, cox: plots?.cox, process: processOf(spec.response) };
  return ownFigureCode(key, CHARTS[key].title, spec, about);
}

/**
 * The code of a figure drawn from a step other than the analysis, for its window: the step's code, after the steps
 * it builds on, and then ``lines``, which draw the figure. Nothing while the step shows no code, or with no lines.
 */
const figureAfter = (step, lines) => () => {
  const code = stepCode[step] && lines();
  return code ? cumulative(...[...(BUILDS_ON[step] ?? []), step].map((s) => stepCode[s]).filter(Boolean), code) : null;
};

/** The figures drawn for the response on show, each with the code that draws it: the last cells of its notebook. */
function figuresCharted() {
  if (!state.charted || !state.analysis) return null;
  const { keys, plots } = state.charted;
  const codes = keys.map((key) => ({ title: CHARTS[key].title, code: figureAlone(key, plots) }));
  return { response: state.shown, codes: codes.filter(({ code }) => code) };
}

/**
 * The further charts, under a heading the reader opens. Each is drawn when the heading is first opened, since a
 * chart drawn while hidden has no width to fit; the heading stays open or shut as the charts are drawn again.
 */
function moreCharts(keys, figures, captioned, version, python) {
  const box = el("details", "more-charts", "");
  box.id = "more-charts";
  box.append(el("summary", "", `More charts (${keys.length})`));
  const grid = el("div", "charts", "");
  box.append(grid);
  box.open = state.moreCharts;
  let shown = false;
  const show = async () => {
    if (!box.open || shown) return;
    shown = true;
    for (const key of keys) {
      const card = captioned(key);
      grid.append(card.card);
      await draw(card.plot, figures[key], version, chartHeight(key), { python: python(key) });
    }
  };
  box.addEventListener("toggle", () => {
    state.moreCharts = box.open;
    show();
  });
  show();
  return box;
}

/**
 * The two factors the response surface is drawn over, for the reader to change: the charts are then drawn again.
 * The choice is the response's own, since each response has its own largest effects, and a button takes it back.
 * It is a way of looking and no part of the study, so it is not saved with one.
 */
function contourPicker(plots) {
  const row = el("p", "hint contour-picker", "");
  const redraw = (focus) => {
    const request = analysisRequest();
    // The picker is drawn again with the charts: the keyboard stays where it was, or goes to the first box.
    drawCharts(state.best?.coded ? { ...request, best: state.best.coded } : request, ticket).then(() => $(focus)?.focus());
  };
  const pick = (at) => {
    const select = el("select", "", "");
    select.id = `contour-factor-${at + 1}`;
    select.setAttribute("aria-label", `${at ? "Second" : "First"} factor the response surface is drawn over`);
    for (const name of plots.axes) select.append(new Option(name, name));
    select.value = plots.contour_factors[at];
    return select;
  };
  const selects = [pick(0), pick(1)];
  selects.forEach((select, at) => {
    select.onchange = () => {
      const other = selects[1 - at];
      if (other.value === select.value) other.value = plots.contour_factors[at]; // the two change places
      state.contourPairs[state.shown] = selects.map((s) => s.value);
      redraw(`#${select.id}`);
    };
  });
  row.append("Draw it over ", selects[0], " and ", selects[1]);
  if (plots.contour_chosen) {
    const back = el("button", "link", "Back to the two largest effects");
    back.type = "button";
    back.id = "contour-largest";
    back.onclick = () => {
      delete state.contourPairs[state.shown];
      redraw(`#${selects[0].id}`); // the button goes with the choice it took back: the first box takes the keyboard
    };
    row.append(". ", back);
  }
  return row;
}

async function libraryPlots(request, mine) {
  // The reader's choice of the surface's two factors, for the response shown, holds while the study has both; a
  // refusal forgets it.
  const names = state.design.spec.factors.map((f) => f.name);
  const shown = state.shown;
  const pair = state.contourPairs[shown]?.every((name) => names.includes(name)) ? state.contourPairs[shown] : null;
  let reply = await call("api_plots", pair ? { ...request, contour_factors: pair } : request);
  if (!reply.ok && pair && !reply.stopped) {
    delete state.contourPairs[shown];
    reply = await call("api_plots", request);
  }
  if (mine === ticket && [analysisSaid, pythonSaid].includes(statusTicket)) {
    setStatus(reply.ok ? "Analysis complete." : `Charts: ${reply.error}`, reply.ok ? "ok" : "error");
  }
  return reply.ok ? reply.result : null;
}

// ---------------------------------------------------------------- the best settings for the response's goal
/**
 * Whether every run of the design has its result for the response Learn shows (or ``name``): the confirmation runs,
 * which the model is not fitted to, aside.
 */
function designComplete(name = state.shown) {
  const confirming = new Set(state.design ? apartRows(state.design) : []);
  return (state.results[name] ?? []).every((v, i) => confirming.has(i) || v !== null);
}

/** Take the best settings away: the analysis they were found on is gone. */
function clearBest() {
  state.best = null;
  renderAnswer();
  $("#best").hidden = true;
  $("#next").hidden = true; // the next step follows the analysis; findNext shows it again with the next one
  $("#to-next").hidden = true;
  $("#next-empty").hidden = false;
  renderRounds(); // the rounds are the study's, analysed or not: with no study on the page they go too
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

/** Whether the study on the page is a mixture's: its best settings are then a blend of its components. */
const blendStudy = () => isMixture(state.design?.spec?.design_type);

/**
 * Learn's first words, over the evidence. With best settings: where to run and what the model expects there, in
 * the numbers of the panel of the best settings, with that panel's cautions. With none to state (a goal of
 * learning, a target not yet typed): what matters most among the factors, the first thing the results say.
 */
function renderAnswer() {
  const box = $("#answer");
  const best = state.best;
  const shown = Boolean(state.analysis && state.design);
  const words = shown ? wordsOf(state.shown) : null;
  const blend = blendStudy();
  const said = shown ? answerWords(best, words, { blend }) : null;
  const first = !shown ? "" : best?.flat ? flatSentence(words, { blend }) : (findingWords(state.analysis.findings, words)[0] ?? "");
  box.hidden = !said && !first;
  // The checks say what Learn's findings say, in the same words: best.js stands alone, and is told them.
  const tell = {
    said: (kind) => findingWords((state.analysis.findings ?? []).filter((finding) => finding.kind === kind), words)[0] ?? "",
    number: fmt,
    p: pval,
    share: wholeShare,
  };
  renderChecks(shown ? learnChecks(state.analysis, best, tell, { blend }) : []);
  $("#answer-where").replaceChildren(...(said && !blend ? answerPictures(best, state.design) : []));
  if (box.hidden) return;
  $("#answer-run").textContent = said ? said.run : first;
  $("#answer-expect").textContent = said?.expect ?? "";
  $("#answer-expect").hidden = !said;
  $("#answer-more").hidden = !said;
}

/** What a check's state is called, for a reader who cannot see its mark. */
const CHECK_STATE = { ok: "Passed", look: "To look at", unknown: "Cannot tell" };
/** The evidence a check leads to, which the page shows already: the part, and the words of the button that shows it. */
const CHECK_EVIDENCE = {
  fit: ["analysis", "See the model and its tables"],
  runs: ["diagnostics", "See each run as the model sees it"],
  reach: ["best", "See the best settings"],
};

/**
 * The checks under the answer, each a heading that opens to its sentence, its statistic and the way to its evidence.
 * A check to look at is open, so its caution is read without a press; one the reader opened stays open through a
 * new fit.
 */
function renderChecks(checks) {
  const row = $("#checks");
  const opened = new Set([...row.querySelectorAll("details[open]")].map((box) => box.dataset.check));
  row.hidden = !checks.length;
  row.replaceChildren(
    ...checks.map((check) => {
      const box = el("details", "learn-check", "");
      box.dataset.check = check.key;
      box.dataset.state = check.state;
      box.open = check.state === "look" || opened.has(check.key);
      const heading = el("summary", "", "");
      heading.append(el("span", "sr", `${CHECK_STATE[check.state]}: `), check.title);
      box.append(heading, el("p", "", check.says));
      if (check.numbers) box.append(el("p", "hint", check.numbers));
      const [part, words] = CHECK_EVIDENCE[check.key] ?? [];
      if (part) {
        const more = el("button", "link", words);
        more.type = "button";
        more.onclick = () => showPart(part);
        box.append(more);
      }
      return box;
    }),
  );
}

/**
 * Show the best settings: what the model predicts there, the settings and where each lies, the two intervals,
 * where the settings sit against the ranges and the limits, whether the runs support the prediction there, how
 * tightly to hold each factor, and the confirmation runs. A goal of learning, or a target not yet typed, gets one line.
 * A mixture's are its best blend (see renderBlend).
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
  const blend = blendStudy();
  renderAnswer();
  $("#best").hidden = false;
  $("#best-chart").hidden = true;
  $("#confirm").hidden = true;
  $("#h-best").textContent = blend ? "The best blend for your goal" : "The best settings for your goal";
  $("#best-chart .term").textContent = blend ? "How tightly to weigh out each component" : "How tightly to hold each factor";
  if (!best.settings) {
    hidePython("best");
    const text = best.flat ? flatSentence(words, { blend }) : noGoalSentence(best, words, { blend });
    $("#best-out").replaceChildren(said("hint", text));
    return;
  }
  if (blend) {
    renderBlend(best, words);
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

/**
 * Show a mixture's best blend: what the model predicts there, each component's proportion and where it lies between its
 * bounds, the two intervals and the leverage, how tightly to weigh out each component, and the confirmation runs.
 */
function renderBlend(best, words) {
  const rows = best.settings.map((s) => [s.name, num(s.value), blendWhere(s, best, words)]);
  const cautions = [blendExtrapolation(best), best.unclear ? BLEND_UNCLEAR : ""].filter(Boolean);
  $("#best-out").replaceChildren(
    said("lead", blendLead(best, words)),
    table(["Component", "Proportion", "Where it lies"], rows),
    knownTable(intervalRows(best, words)),
    ...[noIntervalSentence(best)].filter(Boolean).map((text) => said("hint", text)),
    ...noteList(blendSentences(best)),
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

/**
 * Say what the confirmation runs on the run sheet show, and offer to add them at the best settings. A mixture's are at
 * its best blend, each proportion as it is weighed out, with no units: a proportion is a share of the blend.
 */
function renderConfirmation(best, words) {
  const groups = best.confirmation ?? [];
  const blend = blendStudy();
  const factors = blend ? state.design.spec.factors.map((f) => ({ ...f, units: "" })) : state.design.spec.factors;
  const where = blend ? "this blend" : "these settings";
  $("#confirm").hidden = false;
  $("#confirm-out").replaceChildren(
    ...(blend ? [said("hint", weighingSentence(best))] : []),
    ...groups.map((group) => {
      const box = el("div", "verdict", "");
      box.dataset.state = group.inside === null ? "waiting" : group.inside ? "inside" : "outside";
      box.append(...confirmationSentences(group, factors, words, where).map((text) => said("", text)));
      return box;
    }),
  );
  const add = $("#add-confirmation");
  add.textContent = `Add ${CONFIRMATION_RUNS} confirmation runs at ${where}`;
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
  const blend = blendStudy(); // a component's step is a share of the blend, with no units
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, blend ? "" : (f.units ?? "")]));
  const items = best.sensitivity.map((s) => ({ ...s, units: units[s.name] }));
  const figure = () => sensitivityFigure(items, wordsOf(state.shown));
  card.hidden = false;
  const caption = card.querySelector("figcaption");
  caption.textContent = blend ? BLEND_SENSITIVITY : sensitivityCaption(best);
  caption.hidden = !caption.textContent; // the legend, the axis and the heading's definition say the rest
  // The code moves each numeric factor the model holds, as the bars drawn do.
  const python = figureAfter("best", () => {
    const { formula, model } = state.analysis;
    const names = sensitivity.map((s) => s.name);
    return sensitivityCode(names, shownSpec(), formula ?? model, designComplete() && !asideRuns().length);
  });
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height, { python });
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
  curvesFor(added);
  renderRuns();
  renderStudyBar();
  const practice = Boolean(state.simulation) && someSimulated();
  const design = state.design;
  if (practice) await simulateFrom(before);
  if (state.design !== design) return; // another run sheet took its place meanwhile
  showResultsCode();
  scheduleSave();
  await analyseRuns(); // the fit is the same; what is said of the confirmation runs follows them
  if (state.design !== design) return;
  setStatus(
    practice
      ? `${added} confirmation runs added at ${where}, their results simulated.`
      : `${added} confirmation runs added at the end of the run sheet: enter their results in Run.`,
  );
}

/** Take the confirmation runs off the run sheet, with their results. */
async function removeConfirmationRuns() {
  if (!state.design) return;
  const { columns, rows, results, marks, curves } = withoutConfirmationRuns(state.design, state.results, state.marks, state.curves);
  state.design = { ...state.design, columns, rows };
  state.results = results;
  state.marks = marks;
  state.curves = curves;
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
  clearTradeOffs();
}

// ---------------------------------------------------------------- the trade-offs between the responses
let tradeTicket = 0;
let tradeTimer = null;
/** What the list of what colours the points calls the overall desirability: no response's column, which is a name. */
const OVERALL_D = "(D)";

/** Take the trade-offs away: there is no balance to draw them at, or the study is a mixture's. */
function clearTradeOffs() {
  tradeTicket += 1;
  clearTimeout(tradeTimer);
  // What was tried and the settings chosen are the balance's: with it gone they name factors and responses that may
  // be renamed or removed by the time there is one again. The limits typed stay, for the same responses.
  state.trade = { ...state.trade, tried: null, chosen: null };
  $("#trade-offs").hidden = true;
}

/** Each factor's units, by its name, as the brief gives them. */
const factorUnits = () => Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));

/**
 * Try settings on the models of the balance (bootstrap.trade_offs) and show what one response costs another. It is
 * asked after the balance is on the page, and its reply is dropped when another balance has been found meanwhile.
 */
async function findTradeOffs() {
  const b = state.balance;
  if (!b || blendStudy() || !b.settings.some((s) => s.kind !== "categorical")) {
    clearTradeOffs();
    return;
  }
  const mine = ++tradeTicket;
  const reply = await call("api_trade_offs", { ...balanceAsked(), at: b.coded });
  if (mine !== tradeTicket || state.balance !== b) return;
  if (!reply.ok) {
    clearTradeOffs(); // the balance stands without them
    return;
  }
  const names = b.responses.map((r) => r.name);
  const kept = (name, otherwise) => (names.includes(name) ? name : otherwise);
  const across = kept(state.trade.across, names[0]);
  const up = kept(state.trade.up === across ? null : state.trade.up, names.find((name) => name !== across));
  const colour = state.trade.colour === OVERALL_D || names.includes(state.trade.colour) ? state.trade.colour : "";
  const limits = Object.fromEntries(names.map((name) => [name, state.trade.limits[name] ?? [null, null]]));
  // The settings chosen stay chosen while the models say the same of the settings tried, as after confirmation runs.
  const before = state.trade.tried?.candidates.d;
  const same = before?.length === reply.result.candidates.d.length && before.every((d, i) => d === reply.result.candidates.d[i]);
  state.trade = { tried: reply.result, across, up, colour, limits, chosen: same ? state.trade.chosen : null };
  renderTradeOffs();
}

/** The study's factors with each one's kind, as the what-if panel has them, or else as the brief does. */
const tradeFactors = () =>
  state.whatIf?.factors ??
  state.design.spec.factors.map((f) => ({ ...f, kind: f.type === "categorical" || typeof f.low === "string" ? "categorical" : "numeric" }));

/** The balance's entry for a response, by its column: its label, units, goal and prediction at the balanced settings. */
const balanced = (name) => state.balance.responses.find((r) => r.name === name);

/** Show the trade-offs: what the chart shows, each response's limits, the chart, and the settings nearly as good. */
function renderTradeOffs() {
  const { tried, across, up, colour, limits } = state.trade;
  const responses = state.balance.responses;
  $("#trade-offs").hidden = false;
  const drawn = tried.candidates.d.length;
  const held = tried.held.map((h) => `${h.name} at ${h.value}`).join(", ");
  $("#trade-about").textContent =
    `Each point is one of ${drawn.toLocaleString("en")} settings tried on the models${held ? `, with ${held}` : ""}, drawn evenly ` +
    "from the ranges the limits allow. The line is the best one response can be for each value of the other. A paler " +
    "point is further from your runs than any run lies: there the models guess. Press a point to choose its settings.";
  const options = (chosen, list) =>
    list.map(([value, text]) => `<option value="${escape(value)}"${value === chosen ? " selected" : ""}>${escape(text)}</option>`).join("");
  const named = responses.map((r) => [r.name, withUnits(r.label, r.units)]);
  $("#trade-across").innerHTML = options(across, named);
  $("#trade-up").innerHTML = options(up, named);
  $("#trade-colour").innerHTML = options(colour, [["", "Nothing"], ...named, [OVERALL_D, "Overall desirability, D"]]);
  const t = $("#trade-limits");
  const key = JSON.stringify(named);
  if (t.dataset.key !== key) {
    t.dataset.key = key;
    const rows = responses.map((r) => {
      const label = escape(r.label);
      return (
        `<tr data-response="${escape(r.name)}"><th scope="row">${escape(withUnits(r.label, r.units))}</th>` +
        `<td data-label="At least"><input type="number" step="any" name="low" aria-label="${label}: keep only settings where it is at least"></td>` +
        `<td data-label="At most"><input type="number" step="any" name="high" aria-label="${label}: keep only settings where it is at most"></td></tr>`
      );
    });
    t.innerHTML = `<thead><tr><th scope="col">Keep only settings where</th><th scope="col">is at least</th><th scope="col">and at most</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  }
  // Each box shows the limit the chart keeps to, unless the reader is typing in it: a study made again starts with none.
  for (const tr of t.tBodies[0].rows) {
    for (const [i, end] of ["low", "high"].entries()) {
      const box = tr.querySelector(`[name="${end}"]`);
      if (box !== document.activeElement) box.value = limits[tr.dataset.response][i] ?? "";
    }
  }
  renderTradeNearly();
  renderTradeChosen();
  drawTradeChart();
}

/** What the trade-off chart draws of the settings tried: those that meet the limits, with the best trade-off among them. */
function tradeDrawn() {
  const { tried, across, up, colour, limits, chosen } = state.trade;
  const { settings, responses, d, far } = tried.candidates;
  const units = factorUnits();
  const meets = meetsLimits(responses, limits);
  const at = meets.flatMap((met, i) => (met && responses[across][i] !== null && responses[up][i] !== null ? [i] : []));
  const words = (i) => Object.keys(settings).map((name) => `${name} ${withUnit(settings[name][i], units[name])}`).join(", ");
  const points = at.map((i) => ({ at: i, across: responses[across][i], up: responses[up][i], far: far[i], words: words(i) }));
  const [goalAcross, goalUp] = [balanced(across), balanced(up)];
  const front = frontier(
    points.map((p) => shortfall(goalAcross, p.across)),
    points.map((p) => shortfall(goalUp, p.up)),
    points.map(() => true),
  );
  const values = colour === OVERALL_D ? d : colour ? responses[colour] : null;
  const shown = values ? at.map((i) => values[i]).filter((v) => v !== null) : [];
  const tint = values && {
    ...(colour === OVERALL_D ? { label: "Overall desirability, D", units: "" } : wordsOf(colour)),
    values,
    least: Math.min(...shown),
    most: Math.max(...shown),
  };
  // The runs made that have both results, each by its number in the order the runs were made.
  const results = fitted();
  const runs = state.design.rows.flatMap((_, place) => {
    const [x, y] = [results[across]?.[place], results[up]?.[place]];
    return Number.isFinite(x) && Number.isFinite(y) ? [{ run: runOrderOf(place), across: x, up: y }] : [];
  });
  const nearly = tried.alternatives.map((a) => ({ across: a.responses[across], up: a.responses[up] }));
  const star = { across: goalAcross.predicted, up: goalUp.predicted };
  return { across: goalAcross, up: goalUp, tried: points, front, colour: tint, runs, balanced: star, chosen: chosen && chosenPoint(chosen), nearly, met: at.length };
}

/** The settings the reader chose, on the chart's two axes: their predictions, and half of each prediction interval. */
function chosenPoint(chosen) {
  const { across, up } = state.trade;
  const w = state.whatIf;
  const half = (name) => {
    if (!w) return null;
    const at = predictedAt(w, columnsAt(w.factors, chosen.settings)).find((r) => r.name === name)?.at;
    return at?.prediction ? (at.prediction[1] - at.prediction[0]) / 2 : null;
  };
  return { across: chosen.responses[across], up: chosen.responses[up], spread: { across: half(across), up: half(up) } };
}

/** Draw the chart of the trade-offs, and say how many of the settings tried meet the limits. */
async function drawTradeChart() {
  const card = $("#trade-chart");
  if (!state.trade.tried || !state.plotlyJs) {
    card.hidden = true;
    return;
  }
  const view = tradeDrawn();
  const tried = state.trade.tried.candidates.d.length;
  const limited = Object.values(state.trade.limits).some(([low, high]) => low !== null || high !== null);
  $("#trade-count").textContent = !limited
    ? ""
    : view.met
      ? `${view.met.toLocaleString("en")} of the ${tried.toLocaleString("en")} settings tried meet your limits.`
      : "None of the settings tried meets every limit: loosen one.";
  card.hidden = false;
  card.querySelector("figcaption").textContent =
    "The star is the balanced settings. Settings on the line cannot be bettered on one response without giving up some of the other.";
  const plot = card.querySelector(".plot");
  const python = figureAfter("balance", () => {
    const { tried, across, up, colour, limits } = state.trade;
    return tradeOffCode({ asked: tried.asked, across, up, colour, limits }, state.balance, state.design.spec);
  });
  try {
    await draw(plot, () => tradeOffFigure(view), state.plotlyJs, 440, { python });
  } catch {
    card.hidden = true; // the tables above and below hold what it shows
    return;
  }
  if (!plot.dataset.listens) {
    plot.dataset.listens = "true";
    plot.on("plotly_click", (event) => {
      const point = event.points.find((p) => p.data.meta === "tried");
      if (point) chooseCandidate(point.customdata);
    });
  }
}

/** The reader picked what the chart shows: another response on an axis, or what the points are coloured by. */
function tradeViewChanged() {
  if (!state.trade.tried) return;
  let [across, up] = [$("#trade-across").value, $("#trade-up").value];
  // The two axes show two responses: choosing one axis's response for the other swaps them.
  if (across === up) [across, up] = across === state.trade.across ? [state.trade.up, up] : [across, state.trade.across];
  state.trade = { ...state.trade, across, up, colour: $("#trade-colour").value };
  renderTradeOffs();
}

/** The reader typed a limit on a response: the chart keeps the settings that meet it, a moment after the last key. */
function tradeLimitTyped(e) {
  const tr = e.target.closest("tr[data-response]");
  if (!tr || !state.trade.tried) return;
  const ends = ["low", "high"].map((end) => numberOrNull(tr.querySelector(`[name="${end}"]`).value));
  state.trade = { ...state.trade, limits: { ...state.trade.limits, [tr.dataset.response]: ends } };
  clearTimeout(tradeTimer);
  tradeTimer = setTimeout(drawTradeChart, 300);
}

/** Choose one of the settings tried, by its place among them: its settings, and each response's prediction there. */
function chooseCandidate(i) {
  const { candidates, held } = state.trade.tried;
  const settings = {
    ...Object.fromEntries(Object.entries(candidates.settings).map(([name, values]) => [name, values[i]])),
    ...Object.fromEntries(held.map((h) => [h.name, h.value])),
  };
  const responses = Object.fromEntries(Object.entries(candidates.responses).map(([name, values]) => [name, values[i]]));
  chooseSettings({ settings, responses, d: candidates.d[i], far: Boolean(candidates.far[i]) });
}

/** Take some settings as the reader's choice: say them, mark them on the chart, and move the what-if panel there. */
function chooseSettings(chosen) {
  state.trade = { ...state.trade, chosen };
  const w = state.whatIf;
  if (chosen && w && settingsFit(w.factors, chosen.settings)) {
    w.settings = { ...chosen.settings };
    renderWhatIf();
    checkWhatIfLimits();
  }
  renderTradeChosen();
  drawTradeChart();
}

/** Say the settings chosen and each response there, with what to do with them; nothing when none is chosen. */
function renderTradeChosen() {
  const box = $("#trade-chosen");
  const chosen = state.trade.chosen;
  if (!chosen) {
    box.replaceChildren();
    return;
  }
  const factors = tradeFactors();
  const rows = state.balance.responses.map((r) => [withUnits(r.label, r.units), num(chosen.responses[r.name]), num(r.predicted)]);
  const actions = el("p", "actions", "");
  for (const [act, text, cls] of [
    ["confirm", `Add ${CONFIRMATION_RUNS} confirmation runs at these settings`, "button"],
    ["what-if", "See them under What if", "link"],
    ["back", "Back to the balanced settings", "link"],
  ]) {
    const button = el("button", cls, text);
    button.type = "button";
    button.dataset.act = act;
    actions.append(button);
  }
  box.replaceChildren(
    said("lead", `Settings chosen: ${settingsWords(factors, chosen.settings, factorUnits())}.`),
    table(["Response", "Predicted at the settings chosen", "At the balanced settings"], rows),
    ...noteList(chosen.far ? ["These settings are further from your runs than any run lies: the predictions there are a guess until a run confirms them."] : [], "warnings"),
    actions,
  );
}

/** A button under the settings chosen was pressed: confirm them with runs, go to What if, or let go of them. */
function tradeChosenPressed(e) {
  const act = e.target.closest("button[data-act]")?.dataset.act;
  const chosen = state.trade.chosen;
  if (!act || !chosen) return;
  if (act === "confirm") addConfirmationRuns(chosen.settings, "the settings chosen");
  if (act === "what-if") $("#what-if").scrollIntoView({ behavior: "smooth", block: "start" });
  if (act === "back") {
    chooseSettings(null);
    if (state.whatIf) {
      state.whatIf.settings = { ...whatIfStart(state.whatIf.factors, state.balance, state.best).settings };
      renderWhatIf();
      checkWhatIfLimits();
    }
  }
}

/** List the settings nearly as good as the balanced ones, each with a button that chooses it; nothing without any. */
function renderTradeNearly() {
  const box = $("#trade-nearly");
  const { alternatives, held } = state.trade.tried;
  if (!alternatives.length) {
    box.replaceChildren();
    return;
  }
  const factors = tradeFactors();
  const units = factorUnits();
  const responses = state.balance.responses;
  const rows = alternatives.map((a, i) => {
    const use = el("button", "link", "Choose");
    use.type = "button";
    use.dataset.nearly = String(i);
    use.setAttribute("aria-label", `Choose the settings of row ${i + 1}`);
    const settings = { ...a.settings, ...Object.fromEntries(held.map((h) => [h.name, h.value])) };
    return [settingsWords(factors, settings, units), ...responses.map((r) => num(a.responses[r.name])), num(a.composite), use];
  });
  box.replaceChildren(
    el("h4", "", "Other settings nearly as good"),
    said("hint", `Each is another best of the search, far from the balanced settings, with an overall desirability within 5% of theirs (${num(state.balance.composite)}).`),
    builtTable(["Settings", ...responses.map((r) => withUnits(r.label, r.units)), "D", "Choose"], rows),
  );
}

/** "Choose" was pressed beside settings nearly as good: they become the reader's choice. */
function tradeNearlyPressed(e) {
  const i = e.target.closest("button[data-nearly]")?.dataset.nearly;
  if (i === undefined) return;
  const { alternatives, held } = state.trade.tried;
  const a = alternatives[Number(i)];
  const settings = { ...a.settings, ...Object.fromEntries(held.map((h) => [h.name, h.value])) };
  chooseSettings({ settings, responses: a.responses, d: a.composite, far: false });
}

/** What the reader set of each response's goal, as Python takes it: only the responses with a goal, only what was set. */
function sentGoals() {
  const aimed = new Set(aimedResponses().map((r) => r.name));
  return Object.fromEntries(Object.entries(state.goals).filter(([name, goal]) => aimed.has(name) && Object.keys(goal).length));
}

/** Find the balance a moment after the reader last typed in its table, rather than on every key. */
function scheduleBalance() {
  clearTimeout(balanceTimer);
  balanceTimer = setTimeout(rebalance, 500);
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
  const reply = await call("api_balance", balanceAsked());
  if (mine !== balanceTicket) return;
  renderBalance(reply);
  if (state.whatIf) renderWhatIf(); // the balanced settings it comes back to have moved
}

/** What the balance is asked with: the study, its results, each response's model and what the reader set of its goal. */
const balanceAsked = () => ({ ...held(), ...splitResults(resultNames(), fitted()), models: modelsByName(), goals: sentGoals() });

/** Find the balance again after the reader changed a goal, and then the trade-offs, which are drawn at it. */
async function rebalance() {
  await findBalance();
  findTradeOffs();
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
    clearTradeOffs(); // drawn at a balance that is no more
    $("#balance-out").replaceChildren(said("hint", reply.error));
    for (const id of ["#balance-chart", "#balance-confirm"]) $(id).hidden = true;
    hidePython("balance");
    return;
  }
  const b = reply.result;
  state.balance = b;
  const factors = state.design.spec.factors;
  const units = Object.fromEntries(factors.map((f) => [f.name, f.units ?? ""]));
  const blend = blendStudy(); // a mixture's balance is a blend: each component's proportion, between its bounds
  renderGoals(b);
  const settings = b.settings.map((s) =>
    blend
      ? [s.name, num(s.value), blendWhere(s, null, { label: "" })]
      : [s.name, s.kind === "categorical" ? s.value : withUnit(s.value, units[s.name]), whereWords(s, null, { label: "" })],
  );
  const cautions = [blend ? blendExtrapolation(b) : extrapolationSentence(b)].filter(Boolean);
  $("#balance-out").replaceChildren(
    said("lead", balanceLead(b, { blend })),
    table(blend ? ["Component", "Proportion", "Where it lies"] : ["Factor", "Balanced setting", "Where it lies"], settings),
    balanceTable(b),
    ...[limitingSentence(b, { blend })].filter(Boolean).map((text) => said("", text)),
    said("hint", balanceLeverage(b)),
    ...noteList([...balanceWhere(b), ...(blend && !b.overlay ? [NO_BLEND_OVERLAY] : [])]),
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
  if (e.target.tagName === "SELECT") rebalance();
  else scheduleBalance();
}

/**
 * Say what the confirmation runs show, response by response, and offer to add them at the balanced settings: a
 * mixture's at its balanced blend, weighed out, with no units on its proportions.
 */
function renderBalanceConfirmation(b) {
  const blend = blendStudy();
  const factors = blend ? state.design.spec.factors.map((f) => ({ ...f, units: "" })) : state.design.spec.factors;
  $("#balance-confirm").hidden = false;
  $("#balance-confirm-out").replaceChildren(
    ...(blend ? [said("hint", weighingSentence(b))] : []),
    ...b.confirmation.map((group) => {
      const { said: sentences, state: verdict } = balanceConfirmation(group, factors, b.responses);
      const box = el("div", "verdict", "");
      box.dataset.state = verdict;
      box.append(...sentences.map((text) => said("", text)));
      return box;
    }),
  );
  const add = $("#add-balance-confirmation");
  add.textContent = `Add ${CONFIRMATION_RUNS} confirmation runs at ${blend ? "this blend" : "these settings"}`;
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
  // A mixture's overlay is over its triangle of three components; a study's of factors, over two of them.
  const triangle = b.overlay.kind === "ternary";
  const factors = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f]));
  const axes = triangle ? {} : Object.fromEntries([b.overlay.x, b.overlay.y].map((name) => [name, withUnits(name, factors[name]?.units ?? "")]));
  const figure = () => (triangle ? blendOverlayFigure(b.overlay, b.responses) : overlayFigure(b.overlay, b.responses, axes));
  card.hidden = false;
  card.querySelector("figcaption").textContent = triangle
    ? blendOverlayCaption(b.overlay, b.responses)
    : overlayCaption(b.overlay, state.design.spec.factors, b.responses);
  const python = figureAfter("balance", () => overlayCode(b.overlay, state.design.spec));
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height, { python });
  } catch {
    card.hidden = true; // the tables and the sentences above hold what it shows
  }
}

// ---------------------------------------------------------------- what if: each model at the reader's settings
// Python fits every response once an analysis is done (bootstrap.predictors) and sends each fit as numbers; best.js
// predicts from them as the reader moves a setting, so every response follows at once. Only the limits the settings
// break are asked of Python, a moment after the last move.
let whatIfTicket = 0;
let whatIfLimitsTimer = null;

/** Take the what-if panel away: there are no results to fit, or the study is a mixture's. */
function clearWhatIf() {
  whatIfTicket += 1;
  clearTimeout(whatIfLimitsTimer);
  state.whatIf = null;
  state.curve.target = null; // a target belongs to the panel it was matched in
  $("#curve-target-said").textContent = "";
  $("#what-if").hidden = true;
  hidePython("what-if");
}

/**
 * Fit every response for the what-if panel, and show it: after each analysis, once the best settings and the balance
 * are found, so it can open at them. Settings the reader chose are kept while they still fit the factors.
 */
async function findWhatIf() {
  if (!state.design || (!anyResult(state.results) && !curvesRead())) {
    clearWhatIf();
    return;
  }
  const mine = ++whatIfTicket;
  const reply = await call("api_predictors", { ...held(), ...splitResults(resultNames(), fitted()), models: modelsByName() });
  if (mine !== whatIfTicket) return;
  showWhatIf(reply.ok ? reply.result : null);
}

/**
 * Show the what-if panel for the fits Python sent (bootstrap.predictors), and the curves' (findCurve); with none to
 * predict from, nor a reason the curves could not be fitted, take it away.
 */
function showWhatIf(result) {
  const curve = state.curve.fit || state.curve.problem;
  if (!result || result.mixture || (!result.responses.some((r) => r.terms) && !curve)) {
    clearWhatIf();
    return;
  }
  const { factors, responses } = result;
  const kept = state.whatIf?.settings;
  const settings = settingsFit(factors, kept) ? kept : whatIfStart(factors, state.balance, state.best).settings;
  state.whatIf = { factors, responses, settings, broken: [], keptBroken: new Map() };
  renderWhatIfControls();
  renderWhatIf();
  renderKept();
  checkWhatIfLimits();
  checkKeptLimits();
}

/**
 * Build a control for each factor, once a study and again when a factor's label or range changes: a slider and a
 * number box over a numeric factor's range, a list of a categorical one's levels.
 */
function renderWhatIfControls() {
  const box = $("#what-if-settings");
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const fitted = profiledResponses(state.whatIf).map((r) => r.name);
  const key = JSON.stringify([fitted, state.whatIf.factors.map((f) => [f.name, f.low, f.high, f.levels, units[f.name]])]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.replaceChildren(...state.whatIf.factors.map((f) => whatIfControl(f, units[f.name], fitted)));
}

/**
 * One factor's dial in the what-if panel, named by the factor: a profile for each of the ``fitted`` responses
 * (drawProfiles), one under another, each of which takes a drag, over the slider and the number box that set the
 * factor from the keyboard.
 */
function whatIfControl(f, units, fitted) {
  const row = el("div", "what-if-factor", "");
  row.dataset.factor = f.name;
  const name = el("span", "what-if-name", withUnits(f.name, units));
  const profiles = fitted.map((response) => {
    const profile = svgPart("svg", { class: "profile", role: "img" });
    profile.dataset.response = response;
    takeDrag(profile, (e) => dragProfile(e, f));
    return profile;
  });
  if (f.kind === "categorical") {
    const select = el("select", "", "");
    select.name = "level";
    select.setAttribute("aria-label", f.name);
    for (const level of f.levels) select.append(new Option(level, level));
    row.append(name, ...profiles, select);
    return row;
  }
  const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
  const span = `${num(low)} to ${withUnit(high, units)}`;
  const slider = el("input", "", "");
  Object.assign(slider, { type: "range", name: "slider", min: String(low), max: String(high), step: String(sliderStep(f)) });
  slider.setAttribute("aria-label", `${f.name}, from ${span}`);
  const box = el("input", "", "");
  Object.assign(box, { type: "number", name: "number", min: String(low), max: String(high), step: "any" });
  box.setAttribute("aria-label", withUnits(f.name, units));
  row.append(name, ...profiles, slider, box, el("span", "what-if-range", span));
  return row;
}

/**
 * Let a drawing of the what-if panel take a press and a drag: ``moved`` is given each pointer event that sets
 * something. A mouse or a pen sets where it presses. A finger that lands on the drawing may be about to scroll the
 * page past it, so a touch sets nothing until it moves sideways, or lifts where it landed: the browser takes an up
 * or down swipe for the page (the drawing's touch-action), and tells the drawing nothing more of it.
 */
function takeDrag(drawing, moved) {
  const held = (e) => drawing.hasPointerCapture(e.pointerId) && Boolean(state.whatIf);
  const set = (e) => {
    state.whatIf.dragging = true;
    moved(e);
  };
  drawing.addEventListener("pointerdown", (e) => {
    drawing.setPointerCapture(e.pointerId);
    if (e.pointerType !== "touch" && state.whatIf) set(e);
  });
  drawing.addEventListener("pointermove", (e) => held(e) && set(e));
  drawing.addEventListener("pointerup", (e) => e.pointerType === "touch" && held(e) && set(e));
  drawing.addEventListener("lostpointercapture", () => {
    if (!state.whatIf?.dragging) return;
    state.whatIf.dragging = false;
    renderWhatIf(); // the profiles' scale held still for the drag: it now fits them as they are
  });
}

/**
 * A profile's drawing, in the page's pixels: as wide as its dial (``width`` until the dial is on show), ``height``
 * tall, with ``pad`` of room about the plot, which puts its two ends over the ends of the slider's travel.
 */
const PROFILE = { width: 240, height: 132, pad: 10 };

/** A part of an SVG drawing, with its attributes. */
function svgPart(tag, attributes = {}, text = "") {
  const part = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) part.setAttribute(name, value);
  part.textContent = text;
  return part;
}

/** The responses the panel draws a row of profiles for: each whose model is fitted, in the page's order. */
const profiledResponses = (w) => w.responses.filter((r) => r.terms);

/** The response whose map the panel draws, and whose profiles its Python gives: the one Learn shows when its model is fitted, else the first that is. */
const profiledResponse = (w) => profiledResponses(w).find((r) => r.name === state.shown) ?? profiledResponses(w)[0] ?? null;

/**
 * Draw a row of profiles for each fitted response (best.js profileOf): its prediction as each factor alone moves,
 * with the 95% confidence band, dashed where the settings lie beyond the runs, and a mark where the factor is set.
 * The profiles of a response are drawn to one scale, its own. With no response fitted (a study of curves alone)
 * there is none to draw.
 */
function drawProfiles(w) {
  const fits = profiledResponses(w);
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const box = $("#what-if-settings");
  box.classList.toggle("what-if-settings--rows", fits.length > 1);
  $("#what-if-about").textContent = fits.length
    ? `Each picture is ${fits.length > 1 ? "a response" : fits[0].label} against one factor, the others as set; dashed beyond the runs. Drag it, or type a value.`
    : "Move a setting, and each response's prediction follows.";
  w.ranges ??= {};
  for (const fit of fits) {
    const profiles = w.factors.map((f) => profileOf(fit, w.factors, w.settings, f.name));
    const range = profileRange(profiles, w.dragging ? w.ranges[fit.name] : null);
    w.ranges[fit.name] = range;
    const words = { label: fit.label, units: fit.units ?? "", named: fits.length > 1 };
    w.factors.forEach((f, i) => {
      const profile = [...box.children[i].querySelectorAll("svg.profile")].find((one) => one.dataset.response === fit.name);
      const size = { width: profile.clientWidth || PROFILE.width, height: profile.clientHeight || PROFILE.height };
      profile.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
      profile.setAttribute("aria-label", profileWords(f, profiles[i], words, units[f.name]));
      profile.replaceChildren(...profileParts(f, profiles[i], range, w.settings[f.name], words, size));
    });
  }
}

/**
 * The parts of one profile's drawing: the scale's two ends, the response's name where the panel draws several, the
 * band, the line in its pieces, and the mark.
 */
function profileParts(f, points, range, value, { label, units, named }, { width, height }) {
  const { pad } = PROFILE;
  const numeric = f.kind === "numeric";
  const [low, high] = numeric ? [points[0].value, points.at(-1).value] : [0, points.length - 1];
  const across = (v) => pad + (high > low ? ((width - 2 * pad) * (v - low)) / (high - low) : (width - 2 * pad) / 2);
  const up = (v) => height - pad - ((height - 2 * pad) * (v - range[0])) / (range[1] - range[0]);
  const x = (point) => across(numeric ? point.value : points.indexOf(point));
  const at = (point, v = point.predicted) => `${x(point).toFixed(1)},${up(v).toFixed(1)}`;
  const parts = [
    svgPart("text", { class: "profile__end", x: 4, y: 12 }, withUnit(range[1], units)),
    svgPart("text", { class: "profile__end", x: 4, y: height - 4 }, withUnit(range[0], units)),
    ...(named ? [svgPart("text", { class: "profile__name", x: width - 4, y: 12, "text-anchor": "end" }, label)] : []),
  ];
  const banded = points.every((point) => point.confidence);
  if (!numeric) {
    for (const point of points) {
      if (banded) parts.push(svgPart("line", { class: "profile__bar", x1: x(point), x2: x(point), y1: up(point.confidence[0]), y2: up(point.confidence[1]) })); // prettier-ignore
      const chosen = point.value === value;
      parts.push(svgPart("circle", { class: `profile__level${chosen ? " profile__level--set" : ""}`, cx: x(point), cy: up(point.predicted), r: chosen ? 6 : 4 })); // prettier-ignore
    }
    return parts;
  }
  if (banded) {
    const edge = (end) => points.map((point) => at(point, point.confidence[end]));
    parts.push(svgPart("polygon", { class: "profile__band", points: [...edge(0), ...edge(1).reverse()].join(" ") }));
  }
  for (const piece of profilePieces(points)) {
    const cls = piece.beyond ? "profile__line profile__line--beyond" : "profile__line";
    parts.push(svgPart("polyline", { class: cls, points: piece.points.map((point) => at(point)).join(" ") }));
  }
  // The mark: where the factor is set, and what the model predicts there, read off the profile's own points.
  const after = Math.max(1, points.findIndex((point) => point.value >= value));
  const [before, next] = [points[after - 1], points[after]];
  const share = next.value > before.value ? (value - before.value) / (next.value - before.value) : 0;
  const predicted = before.predicted + share * (next.predicted - before.predicted);
  parts.push(
    svgPart("line", { class: "profile__at", x1: across(value), x2: across(value), y1: pad, y2: height - pad }),
    svgPart("circle", { class: "profile__set", cx: across(value), cy: up(predicted), r: 6 }),
  );
  return parts;
}

/**
 * Draw the response's map beside the dials (charts.js mapFigure): its fitted surface over two numeric factors, the
 * others where the dials have them, faded where the settings lie beyond the runs, with a mark where the dials are.
 * The two factors are those of the response's surface among the charts, when the reader chose them, else the
 * study's first two numeric ones; a study with fewer has no map. The best settings are starred when the map passes
 * through them: when every other factor is where they have it.
 */
function drawMap(w) {
  const fit = profiledResponse(w);
  const pair = fit && state.plotlyJs ? mapPair(w.factors, state.contourPairs[fit.name]) : null;
  $("#what-if-map").hidden = !pair;
  if (!pair) return;
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  // The surface changes with the fit and with the other factors' settings: a drag on the map itself moves neither.
  const others = w.factors.filter((f) => !pair.includes(f.name));
  const key = JSON.stringify([fit.name, fit.coefficients, pair, others.map((f) => w.settings[f.name])]);
  if (w.map?.key !== key) w.map = { key, pair, surface: surfaceOf(fit, w.factors, w.settings, pair) };
  const found = fit.name === state.shown ? Object.fromEntries((state.best?.settings ?? []).map((one) => [one.name, one.value])) : {};
  const through = pair.every((name) => Number.isFinite(found[name])) && others.every((f) => found[f.name] === w.settings[f.name]);
  const words = {
    label: fit.label,
    units: fit.units ?? "",
    axes: pair.map((name) => ({ name, units: units[name] })),
    at: pair.map((name) => w.settings[name]),
    best: through ? pair.map((name) => found[name]) : null,
  };
  $("#what-if-map-title").textContent = `${fit.label} over ${pair[0]} and ${pair[1]}`;
  const at = columnsAt(w.factors, w.settings);
  const python = figureAfter("run", () => mapCode(state.design.spec, fit.name, fit.formula, at, w.settings, pair, MAP_POINTS));
  const { surface } = w.map;
  draw($("#what-if-map-plot"), () => mapFigure(surface, words), state.plotlyJs, 300, { python });
}

/** The reader pressed or dragged on the map: its two factors are set where the pointer is, each to its slider's step. */
function dragMap(e) {
  const w = state.whatIf;
  const point = w.map && plotPoint(e.currentTarget, e.clientX, e.clientY);
  if (!point) return;
  const stepped = w.map.pair.map((name, i) => {
    const f = w.factors.find((one) => one.name === name);
    const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
    const step = sliderStep(f);
    return [name, Math.min(high, Math.max(low, Number((low + Math.round((point[i] - low) / step) * step).toPrecision(12))))];
  });
  setWhatIf(Object.fromEntries(stepped));
}

/** The reader pressed or dragged on a profile: its factor is set where the pointer is, to the slider's step. */
function dragProfile(e, f) {
  const drawing = e.currentTarget; // the plot is inside the drawing's border, PROFILE.pad from each side
  const left = drawing.getBoundingClientRect().left + drawing.clientLeft + PROFILE.pad;
  const share = Math.min(1, Math.max(0, (e.clientX - left) / (drawing.clientWidth - 2 * PROFILE.pad)));
  if (f.kind === "categorical") {
    setWhatIf({ [f.name]: f.levels[Math.round(share * (f.levels.length - 1))] });
    return;
  }
  const [low, high] = [Math.min(f.low, f.high), Math.max(f.low, f.high)];
  const step = sliderStep(f);
  const stepped = low + Math.round((share * (high - low)) / step) * step;
  setWhatIf({ [f.name]: Math.min(high, Math.max(low, Number(stepped.toPrecision(12)))) });
}

/**
 * Set factors of the panel (``changes``, each factor's new setting by its name): every response follows at once,
 * and the limits the settings break a moment later.
 */
function setWhatIf(changes) {
  if (Object.entries(changes).every(([name, value]) => state.whatIf.settings[name] === value)) return;
  state.whatIf.settings = { ...state.whatIf.settings, ...changes };
  renderWhatIf();
  clearTimeout(whatIfLimitsTimer);
  whatIfLimitsTimer = setTimeout(checkWhatIfLimits, 150);
}

/** Show the settings the panel holds in its controls; a number box the reader is typing in keeps what is typed. */
function showWhatIfSettings() {
  for (const row of $("#what-if-settings").children) {
    const value = state.whatIf.settings[row.dataset.factor];
    for (const input of row.querySelectorAll("input, select")) {
      if (input.name === "number" && input === document.activeElement) continue;
      input.value = input.name === "number" ? num(value) : String(value);
    }
  }
}

/** The what-if table's columns, and the glossary entry of each whose heading is a term. */
const WHAT_IF_COLUMNS = ["Response", "Prediction", "95% confidence interval", "95% prediction interval", "Leverage, against the runs' largest"];
const WHAT_IF_TERMS = {
  "95% confidence interval": "confidence-interval",
  "95% prediction interval": "prediction-interval",
  "Leverage, against the runs' largest": "leverage",
};

/** Each response at the columns of some settings (best.js columnsAt), as best.js predicts it from the fits Python sent. */
const predictedAt = (w, columns) => w.responses.map((r) => (r.terms ? { ...r, at: predictAt(r, columns) } : r));

/**
 * Show every response at the panel's settings, as best.js predicts it from the fits Python sent: the table, the limits
 * the settings break, the models they lean on beyond their runs, and the code that makes the same predictions, at the
 * settings kept with the study too.
 */
function renderWhatIf() {
  const w = state.whatIf;
  const at = columnsAt(w.factors, w.settings);
  const responses = predictedAt(w, at);
  const start = whatIfStart(w.factors, state.balance, state.best);
  $("#what-if").hidden = false;
  showWhatIfSettings();
  drawProfiles(w);
  drawMap(w);
  $("#what-if-back").textContent = WHAT_IF_BACK[start.from];
  $("#what-if-back").hidden = w.factors.every((f) => start.settings[f.name] === w.settings[f.name]);
  // a study whose only results so far are its curves has no response to list: the curve below is the prediction
  const fitted = w.responses.some((r) => r.terms);
  $("#what-if-out").replaceChildren(
    ...(fitted ? [table(WHAT_IF_COLUMNS, whatIfRows(responses), WHAT_IF_TERMS)] : []),
    ...noteList(whatIfSentences(fitted ? responses : [], w.broken), "warnings"),
  );
  drawRuns();
  const formulas = Object.fromEntries(w.responses.filter((r) => r.terms).map((r) => [r.name, r.formula]));
  const kept = state.scenarios.filter((s) => !keptProblem(w, s)).map((s) => ({ name: s.name, at: columnsAt(w.factors, s.settings) }));
  const profiled = profiledResponse(w);
  if (fitted) showPython("what-if", whatIfCode(state.design.spec, formulas, at, w.settings, kept, profiled && { name: profiled.name, points: PROFILE_POINTS }));
  else hidePython("what-if");
  renderCurve(at);
  scheduleSolve(); // one factor is solved for from the others, which are where this panel has them
  scheduleWindow(); // and the windows are judged through these settings
}

// ---------------------------------------------------------------- what if: a prediction as runs
/** A drawing of runs, in the page's pixels: ``width`` across with ``pad`` at each side, a dot ``dot`` across, and ``foot`` under the dots for the two ends' numbers. */
const RUNS = { width: 240, pad: 10, dot: 10, foot: 16 };

/**
 * Draw each fitted response's prediction as the runs a reader might get at the panel's settings (best.js runsAt), each
 * as likely as the next: dots that pile up where a run is likelier, the outer two at the ends of the 95% prediction
 * interval in the table above, and those inside the response's window filled. A response whose model has no interval
 * has no runs to draw.
 */
function drawRuns() {
  const w = state.whatIf;
  if (!w) return;
  const at = columnsAt(w.factors, w.settings);
  const drawn = profiledResponses(w).flatMap((fit) => {
    const runs = runsAt(fit, at);
    return runs ? [runsFigure(fit, runs, state.window.windows[fit.name])] : [];
  });
  $("#what-if-runs").hidden = !drawn.length;
  $("#what-if-runs").replaceChildren(...drawn);
}

/** One response's runs: the dots over a line, the window they are judged by shaded behind them, and what they say in words. */
function runsFigure(fit, runs, ends = [null, null]) {
  const windowed = ends.some((end) => end !== null);
  const inside = insideWindow(runs, ends);
  const { width, pad, dot, foot } = RUNS;
  const [least, most] = [runs[0], runs.at(-1)];
  const room = (most - least) * 0.1 || 1; // a little past the two ends, so the outer dots are whole
  const [low, high] = [least - room, most + room];
  const piles = dotPiles(runs, [low, high], Math.floor((width - 2 * pad) / dot));
  const tall = (Math.max(...piles.map((pile) => pile.level)) + 1) * dot;
  const across = (value) => pad + ((value - low) / (high - low)) * (width - 2 * pad);
  const drawing = svgPart("svg", { class: "runs", viewBox: `0 0 ${width} ${tall + foot}`, width, height: tall + foot, "aria-hidden": "true" });
  const [from, to] = [Math.max(low, ends[0] ?? low), Math.min(high, ends[1] ?? high)];
  if (windowed && to > from) drawing.append(svgPart("rect", { class: "runs__window", x: across(from), y: 0, width: across(to) - across(from), height: tall }));
  drawing.append(svgPart("line", { class: "runs__foot", x1: pad, x2: width - pad, y1: tall, y2: tall }));
  piles.forEach(({ column, level }, i) => {
    const run = svgPart("circle", { class: `runs__run${inside[i] ? " runs__run--inside" : ""}`, cx: pad + (column + 0.5) * dot, cy: tall - (level + 0.5) * dot, r: dot / 2 - 1.5 });
    drawing.append(run);
  });
  for (const end of [least, most]) drawing.append(svgPart("text", { class: "runs__end", x: across(end), y: tall + foot - 3, "text-anchor": "middle" }, num(end)));
  const figure = el("figure", "runs-figure", "");
  figure.dataset.response = fit.name;
  // The words say what the drawing shows, so a reader who does not see it loses nothing: the drawing is not read out.
  figure.append(drawing, el("figcaption", "hint", runsWords(runs, inside, { label: fit.label, units: fit.units ?? "" }, windowed)));
  return figure;
}

// ---------------------------------------------------------------- the curve at the what-if settings
/** Whether some run has its curve read. */
const curvesRead = () => Boolean(state.curves?.some(Boolean));

let curveTicket = 0;

/**
 * Fit the runs' curves (bootstrap.functional) once per analysis, so the what-if panel draws the curve at its settings
 * in the page's own JavaScript; or keep why they cannot be fitted yet, which the panel says in place of the curve.
 */
async function findCurve() {
  const mine = ++curveTicket;
  state.curve = { ...state.curve, fit: null, problem: null };
  const asked = state.design;
  if (!asked || !curvesRead()) return;
  const reply = await call("api_functional", curveRequest());
  if (mine !== curveTicket || state.design !== asked) return; // a later fit, or another run sheet, has the page now
  if (!reply.ok) {
    state.curve.problem = reply.error;
    return;
  }
  state.curve.fit = reply.result;
  state.plotlyJs ??= reply.result.plotly_js;
}

/** What Python fits the curves from: the runs, their curves, and the model Learn fits. */
const curveRequest = () => ({ ...held(), ...sentResults(), ...sentCurves(), model: asked(modelChoice()) });

/** Draw the curve at the panel's settings (best.js curveAt), with its bands and any target; or say why there is none. */
function renderCurve(at) {
  const { fit, problem, target } = state.curve;
  $("#what-if-curve").hidden = !fit && !problem;
  $("#what-if-curve-plot").hidden = !fit;
  $("#curve-target").hidden = !fit;
  if (!fit) {
    $("#what-if-curve-said").textContent = problem ?? "";
    return;
  }
  const { label, units, argument } = state.design.spec.curve;
  const words = { label, units, along: argument.label, alongUnits: argument.units };
  const goal = target ? { points: argument.values, values: target } : null;
  // The code fits the curves again and predicts the one drawn: it builds on the run sheet, as the panel's own does.
  const python = figureAfter("run", () => curveCode(state.design, state.curves, fit, at));
  if (state.plotlyJs) {
    draw($("#what-if-curve-plot"), () => curveFigure(fit.points, curveAt(fit, at), words, goal), state.plotlyJs, undefined, { python });
  }
  $("#what-if-curve-said").textContent = curveSaid(fit);
}

/**
 * Find the settings whose predicted curve comes closest to the target pasted (logic.js pastedTarget), searched by
 * bootstrap.functional with each categorical factor held at the panel's level, and move the panel there.
 */
async function findTarget() {
  const said = $("#curve-target-said");
  const w = state.whatIf;
  const pasted = pastedTarget($("#curve-target-values").value, state.design.spec.curve.argument.values, CURVE_POINTS_HELD);
  if (pasted.problem) {
    said.textContent = pasted.problem;
    return;
  }
  said.textContent = "Searching for the settings...";
  const held = Object.fromEntries(w.factors.filter((f) => f.kind === "categorical").map((f) => [f.name, w.settings[f.name]]));
  const reply = await call("api_functional", { ...curveRequest(), target: pasted.values, held });
  if (!reply.ok || state.whatIf !== w) {
    said.textContent = reply.ok ? "" : reply.error;
    return;
  }
  state.curve.target = pasted.values;
  w.settings = { ...w.settings, ...reply.result.match.settings };
  renderWhatIf();
  checkWhatIfLimits();
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  said.textContent = `The curve comes closest to the target at ${settingsWords(w.factors, w.settings, units)}, drawn above against it.`;
}

/** The reader moved a setting: every response follows at once, and the limits it breaks a moment later. */
function whatIfChanged(e) {
  const row = e.target.closest("[data-factor]");
  if (!row || !state.whatIf) return;
  const f = state.whatIf.factors.find((one) => one.name === row.dataset.factor);
  let value = e.target.value;
  if (f.kind === "numeric") {
    value = numberOrNull(value);
    if (value === null) return; // a number half typed: the panel waits for the rest
    value = Math.min(Math.max(value, Math.min(f.low, f.high)), Math.max(f.low, f.high)); // the models hold inside the range
  }
  setWhatIf({ [f.name]: value });
}

/** Ask Python which limits on the settings the panel's settings break, and say so; a study without limits breaks none. */
async function checkWhatIfLimits() {
  clearTimeout(whatIfLimitsTimer);
  const w = state.whatIf;
  if (!w || !(state.design.spec.constraints ?? []).length) return;
  const mine = whatIfTicket;
  const settings = w.settings;
  const reply = await call("api_outside_limits", { spec: state.design.spec, settings });
  if (mine !== whatIfTicket || state.whatIf?.settings !== settings) return; // the reader has moved on
  w.broken = reply.ok ? reply.result.broken : [];
  renderWhatIf();
}

/** Bring the panel back to where it starts: the balanced settings, the best settings, or the center of the ranges. */
function whatIfBack() {
  if (!state.whatIf) return;
  state.whatIf.settings = whatIfStart(state.whatIf.factors, state.balance, state.best).settings;
  renderWhatIf();
  checkWhatIfLimits();
}

// ---------------------------------------------------------------- what if: settings kept with the study
// The reader keeps the panel's settings under a name, and they go with the study (its what_if), through this browser,
// its file, its link and its workbook. Only the settings are kept: each response there is predicted again from the
// fits of the latest analysis whenever the list is drawn, so it follows the results.

/** Why the models do not predict at kept settings now, in a sentence, or null (best.js scenarioProblem). */
const keptProblem = (w, s, units = {}) => scenarioProblem(w.factors, s.settings, units, w.keptBroken.get(JSON.stringify(s.settings)) ?? []);

/** The settings a confirmation run at kept settings is made at: each factor's, and nothing else they hold. */
const keptAt = (s) => Object.fromEntries(state.whatIf.factors.map((f) => [f.name, s.settings[f.name]]));

/** Keep the panel's settings with the study, under the name typed or the next "Settings N", and list them. */
function keepWhatIf(e) {
  e.preventDefault();
  const w = state.whatIf;
  if (!w) return;
  const box = $("#what-if-keep-name");
  const { name, problem } = keptAs(box.value, state.scenarios, SCENARIOS_HELD);
  if (problem) {
    toast(problem, "error");
    return;
  }
  state.scenarios = [...state.scenarios, { name, settings: { ...w.settings } }];
  box.value = "";
  renderWhatIf();
  renderKept();
  checkKeptLimits();
  scheduleSave();
  toast(`Kept these settings as ${name}.`);
}

/**
 * List the settings kept with the study below the panel, side by side in one table: a row each, with each factor's
 * setting, each response as predicted there over its 95% prediction interval, or why there is none, and its buttons.
 */
function renderKept() {
  const w = state.whatIf;
  $("#what-if-kept").hidden = !w || !state.scenarios.length;
  if ($("#what-if-kept").hidden) return;
  const units = Object.fromEntries(state.design.spec.factors.map((f) => [f.name, f.units ?? ""]));
  const fits = profiledResponses(w);
  const t = el("table", "data what-if-kept", "");
  t.setAttribute("aria-labelledby", "h-what-if-kept");
  const head = t.createTHead().insertRow();
  const headings = ["Kept as", ...w.factors.map((f) => withUnits(f.name, units[f.name])), ...fits.map((r) => withUnits(r.label, r.units)), "What to do"];
  for (const [i, text] of headings.entries()) {
    const th = el("th", i === headings.length - 1 ? "kept__do" : "", text);
    th.scope = "col";
    head.append(th);
  }
  const body = t.createTBody();
  for (const s of state.scenarios) body.append(keptRow(w, s, units, fits));
  const wrap = el("div", "scroll", "");
  wrap.append(t);
  const said = state.scenarios.flatMap((s) => (keptProblem(w, s, units) ? [] : whatIfSentences(predictedAt(w, columnsAt(w.factors, s.settings))).map((text) => `${s.name}: ${text}`)));
  $("#what-if-kept-list").replaceChildren(wrap, ...noteList(said, "warnings"));
}

/** One kept setting's row: its name, each factor's setting, each response there (or, across them, why there is none), and its buttons. */
function keptRow(w, s, units, fits) {
  const tr = el("tr", "", "");
  tr.dataset.kept = s.name;
  const name = el("th", "", s.name);
  name.scope = "row";
  tr.append(name);
  for (const f of w.factors) {
    const value = s.settings[f.name];
    tr.append(el("td", "", value === undefined ? "" : f.kind === "numeric" ? num(value) : String(value)));
  }
  const actions = el("td", "kept__do", "");
  const problem = keptProblem(w, s, units);
  if (problem) {
    const why = el("td", "hint", problem);
    why.colSpan = Math.max(1, fits.length);
    tr.append(why);
  } else {
    const at = columnsAt(w.factors, s.settings);
    for (const fit of fits) {
      const { predicted, prediction } = predictAt(fit, at);
      const cell = el("td", "", withUnit(predicted, fit.units));
      if (prediction) cell.append(el("small", "", intervalWords(prediction, fit.units)));
      tr.append(cell);
    }
    if (!fits.length) tr.append(el("td", "", ""));
    actions.append(keptButton("show", s.name, "Show on the dials", "link", `Show the settings kept as ${s.name} on the dials`));
    const runs = `Add ${CONFIRMATION_RUNS} confirmation runs`;
    if (!confirmedAt(state.design, keptAt(s))) {
      actions.append(keptButton("confirm", s.name, runs, "button", `${runs} at the settings kept as ${s.name}`)); // none there yet
    }
  }
  actions.append(keptButton("delete", s.name, "Delete", "link", `Delete the settings kept as ${s.name}`));
  tr.append(actions);
  return tr;
}

/** A button of the kept settings ``name``; its accessible name, ``label``, starts with its ``text`` and names them. */
function keptButton(action, name, text, cls, label) {
  const button = el("button", cls, text);
  button.type = "button";
  Object.assign(button.dataset, { action, kept: name });
  button.setAttribute("aria-label", label);
  return button;
}

/** A button of a kept setting was pressed: set the dials there, make confirmation runs at the settings, or delete them. */
function keptPressed(e) {
  const button = e.target.closest("button[data-kept]");
  const s = button && state.whatIf ? state.scenarios.find((one) => one.name === button.dataset.kept) : null;
  if (!s) return;
  if (button.dataset.action === "confirm") {
    addConfirmationRuns(keptAt(s), `the settings kept as ${s.name}`);
    return;
  }
  if (button.dataset.action === "show") {
    setWhatIf(keptAt(s)); // every number of these settings is then in the panel's own table, its intervals and its leverage
    toast(`The dials are at the settings kept as ${s.name}.`);
    return;
  }
  state.scenarios = state.scenarios.filter((one) => one !== s);
  renderWhatIf();
  renderKept();
  scheduleSave();
  $("#what-if-keep-name").focus(); // the button went with the settings
  toast(`Deleted the settings kept as ${s.name}.`);
}

/** Ask Python, once for each, which limits on the settings the kept settings break; a study without limits breaks none. */
async function checkKeptLimits() {
  const w = state.whatIf;
  if (!w || !(state.design.spec.constraints ?? []).length) return;
  const mine = whatIfTicket;
  for (const s of state.scenarios) {
    const key = JSON.stringify(s.settings);
    if (w.keptBroken.has(key) || keptProblem(w, s)) continue;
    w.keptBroken.set(key, []); // asked, so not asked again while the reply is on its way
    const reply = await call("api_outside_limits", { spec: state.design.spec, settings: s.settings });
    if (mine !== whatIfTicket) return; // another analysis gave the panel new fits
    if (!reply.ok || !reply.result.broken.length) continue;
    w.keptBroken.set(key, reply.result.broken);
    renderWhatIf();
    renderKept();
  }
}

// ---------------------------------------------------------------- the settings that give a target
// The reader types what a response should be, or several; Python runs their models backwards
// (bootstrap.target_settings) and answers with settings, which the page says, draws and offers to confirm. One factor
// is then solved for from the others (bootstrap.solve_for), at the settings the what-if panel holds.
let targetTicket = 0;
let targetTimer = null;
let solveTicket = 0;
let solveTimer = null;

/** What the section holds for a study with no target typed yet: the targets by column, and what Python answered. */
function newTarget() {
  return { targets: {}, found: null, pair: null };
}

/** What a call that fits each response is sent: the study, its results, and each response's model. */
const fitsAsked = () => ({ ...held(), ...splitResults(resultNames(), fitted()), models: modelsByName() });

/** The responses a target can be typed for: each one with a result, with the least and the greatest of its results. */
function targetResponses() {
  const results = fitted();
  return studyResponses().flatMap((r) => {
    const values = (results[r.name] ?? []).filter(Number.isFinite);
    return values.length ? [{ name: r.name, ...wordsOf(r.name), least: Math.min(...values), most: Math.max(...values) }] : [];
  });
}

/** The study's factors with a range, by their names. */
const rangedFactors = () => tradeFactors().filter((f) => f.kind !== "categorical");

/** Take the section away: there is no result, or no factor with a range. */
function clearTarget() {
  targetTicket += 1;
  solveTicket += 1;
  clearTimeout(targetTimer);
  clearTimeout(solveTimer);
  // The settings found name the study's factors as they were: they are found again when the section comes back.
  state.target = { ...state.target, found: null };
  renderTargetFound();
  $("#target").hidden = true;
}

/**
 * Show the section, after each analysis: a box and a slider for each response's target, the runs to match, and then
 * the settings for the targets typed so far, found again on the results as they are now.
 */
function showTarget() {
  const responses = state.design ? targetResponses() : [];
  if (!responses.length || !rangedFactors().length) {
    clearTarget();
    return;
  }
  const names = responses.map((r) => r.name);
  const ranged = rangedFactors().map((f) => f.name);
  const targets = Object.fromEntries(Object.entries(state.target.targets).filter(([name]) => names.includes(name)));
  const pair = state.target.pair?.every((name) => ranged.includes(name)) ? state.target.pair : null;
  state.target = { ...state.target, targets, pair };
  $("#target").hidden = false;
  const t = $("#target-goals");
  const key = JSON.stringify(responses);
  if (t.dataset.key !== key) {
    t.dataset.key = key;
    const rows = responses.map((r) => {
      const span = `${num(r.least)} to ${withUnit(r.most, r.units)}`;
      const step = r.most > r.least ? sliderStep({ low: r.least, high: r.most }) : "any";
      return (
        `<tr data-response="${escape(r.name)}"><th scope="row">${escape(withUnits(r.label, r.units))}</th>` +
        `<td data-label="Should be"><input type="number" step="any" name="target" aria-label="${escape(r.label)}: what it should be"></td>` +
        `<td data-label="Or move it"><input type="range" name="slider" min="${r.least}" max="${r.most}" step="${step}" aria-label="${escape(r.label)}: what it should be, from ${escape(span)}"></td>` +
        `<td data-label="Your results">${escape(span)}</td></tr>`
      );
    });
    t.innerHTML = `<thead><tr><th scope="col">Response</th><th scope="col">Should be</th><th scope="col">Or move it</th><th scope="col">Your results</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  }
  showTargets();
  const results = fitted();
  const said = (place) => responses.flatMap((r) => (Number.isFinite(results[r.name][place]) ? [`${r.label} ${withUnit(results[r.name][place], r.units)}`] : []));
  const runs = state.design.rows.flatMap((_, place) => (said(place).length ? [[String(place), `Run ${runOrderOf(place)}: ${said(place).join(", ")}`]] : []));
  $("#target-match").replaceChildren(new Option("Choose a run", ""), ...runs.map(([place, text]) => new Option(text, place)));
  findSettings();
}

/** Show the targets the section holds in their boxes and on their sliders; a box the reader is typing in keeps what is typed. */
function showTargets() {
  for (const tr of $("#target-goals").tBodies[0]?.rows ?? []) {
    const target = state.target.targets[tr.dataset.response] ?? null;
    const [box, slider] = [tr.querySelector('[name="target"]'), tr.querySelector('[name="slider"]')];
    if (box !== document.activeElement) box.value = target === null ? "" : String(target);
    if (target !== null) slider.value = String(target);
  }
  const typed = Object.keys(state.target.targets).length > 0;
  $("#target-clear").hidden = !typed;
}

/** The reader typed a target or moved its slider: the settings follow a moment after the last move. */
function targetTyped(e) {
  const tr = e.target.closest("tr[data-response]");
  if (!tr) return;
  const value = numberOrNull(e.target.value);
  const { [tr.dataset.response]: _, ...others } = state.target.targets;
  state.target = { ...state.target, targets: Number.isFinite(value) ? { ...others, [tr.dataset.response]: value } : others };
  $("#target-match").value = "";
  showTargets();
  scheduleSave(); // the targets go with the study
  clearTimeout(targetTimer);
  targetTimer = setTimeout(findSettings, 400);
}

/** The reader chose a run to match: every response with a result at it takes that result as its target. */
function targetMatched() {
  const place = $("#target-match").value;
  if (place === "") return;
  const results = fitted();
  const at = targetResponses().flatMap((r) => (Number.isFinite(results[r.name][Number(place)]) ? [[r.name, results[r.name][Number(place)]]] : []));
  state.target = { ...state.target, targets: Object.fromEntries(at) };
  showTargets();
  scheduleSave();
  findSettings();
}

/** The reader cleared the targets: the boxes empty, and the settings found go. */
function targetCleared() {
  state.target = { ...state.target, targets: {} };
  $("#target-match").value = "";
  showTargets();
  scheduleSave();
  findSettings();
}

/** Find the settings that give the targets typed (bootstrap.target_settings), and show them; with none typed, nothing. */
async function findSettings() {
  clearTimeout(targetTimer);
  const { targets, pair } = state.target;
  const mine = ++targetTicket;
  const reply = Object.keys(targets).length ? await call("api_target_settings", { ...fitsAsked(), targets, ...(pair ? { pair } : {}) }) : null;
  if (mine !== targetTicket) return;
  $("#target-problems").textContent = reply && !reply.ok ? reply.error : "";
  $("#target-problems").hidden = !reply || reply.ok;
  state.target = { ...state.target, found: reply?.ok ? reply.result : null };
  renderTargetFound();
  solveFactor();
}

/** Where a numeric factor's setting lies in its range: a bar from its low end, and the range in words. */
function rangeMark(setting, factor, units) {
  if (setting.kind === "categorical") return "";
  const [low, high] = [Math.min(factor.low, factor.high), Math.max(factor.low, factor.high)];
  const span = `${num(low)} to ${withUnit(high, units)}`;
  const meter = el("meter", "range-mark", "");
  Object.assign(meter, { min: low, max: high, value: setting.value });
  meter.setAttribute("aria-label", `${setting.name} at ${withUnit(setting.value, units)}, in its range of ${span}`);
  const cell = el("span", "range-cell", "");
  cell.append(meter, ` ${span}`);
  return cell;
}

/** Show the settings found: whether they give the targets, each factor's setting, each response there, and the chart. */
function renderTargetFound() {
  const found = state.target.found;
  for (const id of ["#target-actions", "#target-solve"]) $(id).hidden = !found;
  if (!found) {
    $("#target-out").replaceChildren();
    $("#target-chart").hidden = true;
    hidePython("target");
    return;
  }
  const units = factorUnits();
  const factors = Object.fromEntries(tradeFactors().map((f) => [f.name, f]));
  const blend = blendStudy(); // a mixture's answer is a blend: each component's proportion, between its bounds
  const here = blend ? "this blend" : "these settings";
  const settings = blend
    ? table(["Component", "Proportion", "Where it lies"], found.settings.map((s) => [s.name, num(s.value), blendWhere(s, null, { label: "" })]))
    : builtTable(
        ["Factor", "Setting", "In its range"],
        found.settings.map((s) => [s.name, s.kind === "categorical" ? s.value : withUnit(s.value, units[s.name]), rangeMark(s, factors[s.name], units[s.name])]),
      );
  $("#target-out").replaceChildren(
    said("lead", targetLead(found, { limits: Boolean(state.design.spec.constraints?.length), blend })),
    settings,
    table(["Response", "Target", `Predicted at ${here}`, "95% prediction interval"], targetRows(found), { "95% prediction interval": "prediction-interval" }),
    ...[targetShare(found, { blend })].filter(Boolean).map((text) => said("hint", text)),
    said("hint", balanceLeverage(found)),
    ...noteList(blend ? (found.overlay ? [] : [NO_BLEND_OVERLAY]) : balanceWhere(found)),
    ...noteList([blend ? blendExtrapolation(found) : extrapolationSentence(found)].filter(Boolean), "warnings"),
  );
  $("#target-confirm").hidden = confirmedAt(state.design, found.confirm_at); // runs there already: their results are what is wanted
  $("#target-confirm").textContent = `Add 3 confirmation runs at ${here}`;
  $("#target-what-if").hidden = !state.whatIf; // a mixture has no what-if panel to see them under
  showPanelCode("target", targetCode(found, state.design.spec));
  drawTargetChart();
}

/** Show a section's Python, or take its panel away when the section has none: a mixture's target and specification. */
const showPanelCode = (key, code) => (code ? showPython(key, code) : hidePython(key));

/** Draw each response's line of settings that give its target over two factors; nothing with one numeric factor. */
async function drawTargetChart() {
  const found = state.target.found;
  const card = $("#target-chart");
  if (!found?.overlay || !state.plotlyJs) {
    card.hidden = true;
    return;
  }
  const { overlay } = found;
  const units = factorUnits();
  // A mixture's chart is over its triangle of three components; a study's of factors, over two the reader picks.
  const triangle = overlay.kind === "ternary";
  card.querySelector(".trade-view").hidden = triangle;
  const ranged = rangedFactors().map((f) => f.name);
  const options = (chosen) => ranged.map((name) => `<option value="${escape(name)}"${name === chosen ? " selected" : ""}>${escape(withUnits(name, units[name]))}</option>`).join("");
  if (!triangle) {
    $("#target-across").innerHTML = options(overlay.x);
    $("#target-up").innerHTML = options(overlay.y);
  }
  const axes = triangle ? {} : Object.fromEntries([overlay.x, overlay.y].map((name) => [name, withUnits(name, units[name])]));
  const figure = () =>
    triangle
      ? blendOverlayFigure(overlay, found.responses, undefined, { star: FOUND_BLEND, named: targetName })
      : overlayFigure(overlay, found.responses, axes, undefined, { star: FOUND, named: targetName });
  card.hidden = false;
  card.querySelector("figcaption").textContent = targetCaption(overlay, state.design.spec.factors, found.responses);
  const python = figureAfter("target", () => targetOverlayCode(overlay, state.design.spec));
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height, { python });
  } catch {
    card.hidden = true; // the tables above hold what it shows
  }
}

/** The reader chose the two factors the chart is drawn over: choosing one axis's factor for the other swaps them. */
function targetPairChanged() {
  const overlay = state.target.found?.overlay;
  if (!overlay) return;
  let [across, up] = [$("#target-across").value, $("#target-up").value];
  if (across === up) [across, up] = across === overlay.x ? [overlay.y, up] : [across, overlay.x];
  state.target = { ...state.target, pair: [across, up] };
  findSettings();
}

/** The settings found, by each factor's name, as the what-if panel holds settings. */
const foundSettings = () => Object.fromEntries(state.target.found.settings.map((s) => [s.name, s.value]));

/** Move the what-if panel to these settings, when they are settings of its factors, and go to it. */
function toWhatIf(settings) {
  const w = state.whatIf;
  if (!settings || !w || !settingsFit(w.factors, settings)) return;
  w.settings = { ...settings };
  renderWhatIf();
  checkWhatIfLimits();
  $("#what-if").scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Move the what-if panel to the settings found. */
const targetToWhatIf = () => toWhatIf(state.target.found && foundSettings());

/** Set the factor solved for where the reader pressed, under What if: the other factors are there already. */
function solvedToWhatIf(e) {
  const value = e.target.closest("button[data-value]")?.dataset.value;
  if (value !== undefined && state.whatIf) toWhatIf({ ...state.whatIf.settings, [$("#solve-factor").value]: Number(value) });
}

/** Solve for one factor a moment after the what-if panel's settings last moved. */
function scheduleSolve() {
  clearTimeout(solveTimer);
  if (state.target.found) solveTimer = setTimeout(solveFactor, 400);
}

/**
 * Say what one factor must be for a response to meet its target (bootstrap.solve_for), with every other factor where
 * the what-if panel has it: after the settings are found, and when the reader picks the factor or the response.
 */
async function solveFactor() {
  clearTimeout(solveTimer);
  const { found, targets } = state.target;
  const w = state.whatIf;
  const box = $("#target-solve");
  const mine = ++solveTicket;
  box.hidden = !found || !w;
  if (box.hidden) return;
  const units = factorUnits();
  const keep = (select, list) => {
    const chosen = list.some(([value]) => value === select.value) ? select.value : list[0][0];
    select.replaceChildren(...list.map(([value, text]) => new Option(text, value, false, value === chosen)));
    return chosen;
  };
  const factor = keep($("#solve-factor"), rangedFactors().map((f) => [f.name, withUnits(f.name, units[f.name])]));
  const response = keep($("#solve-response"), found.responses.map((r) => [r.name, withUnits(r.label, r.units)]));
  const asked = { ...fitsAsked(), response, target: targets[response], factor, settings: w.settings };
  const reply = await call("api_solve_for", asked);
  if (mine !== solveTicket) return;
  if (!reply.ok) {
    $("#solve-out").replaceChildren(said("hint", reply.error));
    return;
  }
  const others = Object.fromEntries(Object.entries(w.settings).filter(([name]) => name !== factor));
  const { label, units: responseUnits } = wordsOf(response);
  const words = { held: settingsWords(w.factors, others, units), units: units[factor], label, responseUnits };
  const sentences = solveSentences(reply.result, words).map((text) => said("", text));
  // Each setting that gives the target can be set under What if, to see every response there.
  reply.result.solutions.forEach((solution, place) => {
    const use = el("button", "link", `Set ${factor} to ${withUnit(solution.value, units[factor])} under What if`);
    use.type = "button";
    use.dataset.value = solution.value;
    sentences[place].append(" ", use);
  });
  $("#solve-out").replaceChildren(...sentences);
}

// ---------------------------------------------------------------- the settings that meet a specification
// The reader types a window for a response, or for several; Python judges settings against them with the library's
// feasible_region (bootstrap.specification) and answers with the share of the ranges that meets them, how surely a
// new run does, and the picture over two factors, the others where the what-if panel has them.
let windowTicket = 0;
let windowTimer = null;

/** What the section holds for a study with no window typed yet: each response's two ends, and what Python answered. */
function newWindow() {
  return { windows: {}, found: null, pair: null };
}

/** Take the section away: there is no result, no factor with a range, or no what-if panel to judge through. */
function clearWindow() {
  windowTicket += 1;
  clearTimeout(windowTimer);
  // What the windows gave names the study's factors as they were: it is found again when the section comes back.
  state.window = { ...state.window, found: null };
  renderWindow();
  $("#window").hidden = true;
}

/** The windows the reader typed, as Python takes them: only the responses with an end typed. */
const sentWindows = () => Object.fromEntries(Object.entries(state.window.windows).filter(([, ends]) => ends.some((end) => end !== null)));

/**
 * Show the section, after each analysis: two boxes for each response's window, and then what the windows typed so
 * far give on the results as they are now.
 */
function showWindow() {
  // The windows are judged through the what-if panel's settings, so the section comes with that panel. A mixture
  // has no such panel, and nothing to hold: its windows are judged over every blend.
  const responses = state.design && (state.whatIf || blendStudy()) ? targetResponses() : [];
  if (!responses.length || !rangedFactors().length) {
    clearWindow();
    return;
  }
  const names = responses.map((r) => r.name);
  const ranged = rangedFactors().map((f) => f.name);
  const windows = Object.fromEntries(Object.entries(state.window.windows).filter(([name]) => names.includes(name)));
  const pair = state.window.pair?.every((name) => ranged.includes(name)) ? state.window.pair : null;
  state.window = { ...state.window, windows, pair };
  drawRuns(); // a window that came with the study, or left with its response
  $("#window").hidden = false;
  const t = $("#window-limits");
  const key = JSON.stringify(responses);
  if (t.dataset.key !== key) {
    t.dataset.key = key;
    const rows = responses.map((r) => {
      const label = escape(r.label);
      return (
        `<tr data-response="${escape(r.name)}"><th scope="row">${escape(withUnits(r.label, r.units))}</th>` +
        `<td data-label="At least"><input type="number" step="any" name="low" aria-label="${label}: at least"></td>` +
        `<td data-label="At most"><input type="number" step="any" name="high" aria-label="${label}: at most"></td>` +
        `<td data-label="Your results">${escape(`${num(r.least)} to ${withUnit(r.most, r.units)}`)}</td></tr>`
      );
    });
    t.innerHTML = `<thead><tr><th scope="col">Response</th><th scope="col">At least</th><th scope="col">At most</th><th scope="col">Your results</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  }
  showWindows();
  findWindow();
}

/** Show the windows the section holds in their boxes; a box the reader is typing in keeps what is typed. */
function showWindows() {
  for (const tr of $("#window-limits").tBodies[0]?.rows ?? []) {
    const ends = state.window.windows[tr.dataset.response] ?? [null, null];
    for (const [i, end] of ["low", "high"].entries()) {
      const box = tr.querySelector(`[name="${end}"]`);
      if (box !== document.activeElement) box.value = ends[i] ?? "";
    }
  }
}

/** The reader typed an end of a window: what the windows give follows a moment after the last key. */
function windowTyped(e) {
  const tr = e.target.closest("tr[data-response]");
  if (!tr) return;
  const ends = ["low", "high"].map((end) => {
    const value = numberOrNull(tr.querySelector(`[name="${end}"]`).value);
    return Number.isFinite(value) ? value : null;
  });
  state.window = { ...state.window, windows: { ...state.window.windows, [tr.dataset.response]: ends } };
  scheduleSave(); // the windows go with the study
  scheduleWindow();
  drawRuns(); // the runs inside the window are the filled ones
}

/** Judge the windows again a moment from now: an end was typed, or the what-if panel's settings moved. */
function scheduleWindow() {
  clearTimeout(windowTimer);
  if ($("#window").hidden) return;
  // With no window typed nothing is asked of Python: what the last one gave, or why it was refused, goes at once.
  if (Object.keys(sentWindows()).length) windowTimer = setTimeout(findWindow, 400);
  else if (state.window.found || !$("#window-problems").hidden) findWindow();
}

/** The reader cleared the windows: the boxes empty, and what they gave goes. */
function windowCleared() {
  state.window = { ...state.window, windows: {} };
  showWindows();
  scheduleSave();
  findWindow();
  drawRuns();
}

/** Judge the windows typed (bootstrap.specification), and show what they give; with none typed, nothing. */
async function findWindow() {
  clearTimeout(windowTimer);
  const windows = sentWindows();
  const { pair } = state.window;
  const mine = ++windowTicket;
  $("#window-clear").hidden = !Object.keys(windows).length;
  // The chart holds its other factors, and every categorical factor, where the what-if panel has them.
  const asked = { ...fitsAsked(), windows, at: state.whatIf?.settings, ...(pair ? { pair } : {}) };
  const reply = Object.keys(windows).length && (state.whatIf || blendStudy()) ? await call("api_specification", asked) : null;
  if (mine !== windowTicket) return;
  $("#window-problems").textContent = reply && !reply.ok ? reply.error : "";
  $("#window-problems").hidden = !reply || reply.ok;
  state.window = { ...state.window, found: reply?.ok ? reply.result : null };
  renderWindow();
}

/** Show what the windows give: the share of the settings, how surely a new run meets them, where it is surest, the chart. */
function renderWindow() {
  const found = state.window.found;
  $("#window-actions").hidden = !found?.surest;
  if (!found) {
    $("#window-out").replaceChildren();
    $("#window-chart").hidden = true;
    hidePython("window");
    return;
  }
  const blend = blendStudy(); // a mixture's windows are met by blends
  const more = [windowChance(found, { blend }), surestSentence(found, tradeFactors(), factorUnits(), { blend })].filter(Boolean);
  // A level is chosen, not moved to: beside the level What if has, what the windows give at each of the others.
  const levels = windowLevels(found);
  $("#window-out").replaceChildren(
    said("lead", windowLead(found, { limits: Boolean(state.design.spec.constraints?.length), blend })),
    ...more.map((text) => said("", text)),
    ...(levels ? [el("h4", "", "At each level"), table(levels.columns, levels.rows)] : []),
    ...noteList(blend && !found.overlay ? [NO_BLEND_OVERLAY] : []),
  );
  if (found.surest) $("#window-confirm").hidden = confirmedAt(state.design, found.surest.confirm_at);
  $("#window-confirm").textContent = `Add 3 confirmation runs at the surest ${blend ? "blend" : "settings"}`;
  $("#window-what-if").hidden = !state.whatIf; // a mixture has no what-if panel to see them under
  showPanelCode("window", windowCode(found, state.design.spec));
  drawWindowChart();
}

/** Draw where every response is inside its window over two factors, with the lines of equal chance. */
async function drawWindowChart() {
  const found = state.window.found;
  const card = $("#window-chart");
  if (!found?.overlay || !state.plotlyJs) {
    card.hidden = true;
    return;
  }
  const { overlay } = found;
  const units = factorUnits();
  const ranged = rangedFactors().map((f) => f.name);
  const options = (chosen) => ranged.map((name) => `<option value="${escape(name)}"${name === chosen ? " selected" : ""}>${escape(withUnits(name, units[name]))}</option>`).join("");
  // A mixture's chart is over its triangle of three components, with a star at its surest blend.
  const triangle = overlay.kind === "ternary";
  card.querySelector(".trade-view").hidden = triangle;
  if (!triangle) {
    $("#window-across").innerHTML = options(overlay.x);
    $("#window-up").innerHTML = options(overlay.y);
  }
  const axes = triangle ? {} : Object.fromEntries([overlay.x, overlay.y].map((name) => [name, withUnits(name, units[name])]));
  const figure = () =>
    triangle
      ? blendOverlayFigure(overlay, found.responses, undefined, { shaded: IN_WINDOWS, star: SUREST_BLEND, named: windowName })
      : overlayFigure(overlay, found.responses, axes, undefined, { shaded: IN_WINDOWS, star: WHAT_IF_SETTINGS, named: windowName });
  card.hidden = false;
  card.querySelector("figcaption").textContent = windowCaption(overlay, state.design.spec.factors);
  const python = figureAfter("window", () => windowOverlayCode(overlay, state.design.spec, Boolean(found.search.correlation)));
  try {
    await draw(card.querySelector(".plot"), figure, state.plotlyJs, figure().height, { python });
  } catch {
    card.hidden = true; // the sentences above hold what it shows
  }
}

/** The reader chose the two factors the chart is drawn over: choosing one axis's factor for the other swaps them. */
function windowPairChanged() {
  const overlay = state.window.found?.overlay;
  if (!overlay) return;
  let [across, up] = [$("#window-across").value, $("#window-up").value];
  if (across === up) [across, up] = across === overlay.x ? [overlay.y, up] : [across, overlay.x];
  state.window = { ...state.window, pair: [across, up] };
  findWindow();
}

/** Move the what-if panel to the settings at which a new run is surest to meet the windows, and go to it. */
const windowToWhatIf = () => toWhatIf(state.window.found?.surest?.settings);

// ---------------------------------------------------------------- the next step
// What the analysis points to next: a new study whose ranges move the way the best settings point (best.js), and the
// ways to add runs to this study (bootstrap.next_runs), each for what its design cannot say yet. Adding runs keeps the
// runs made so far and their results, so the study grows in place, as it does with confirmation runs.

/**
 * Ask Python what the results call for next, and which ways of adding runs suit the run sheet, and show the next step.
 * It is asked again when the run sheet or its results change, with the analysis's own request. A mixture's blends take
 * none of the ways, and its next step is not searched, so it is not asked.
 */
async function findNext(request = analysisRequest()) {
  const design = state.design;
  const key = JSON.stringify([request.responses, request.more_responses, request.response]);
  if (design && !isMixture(design.spec.design_type) && (state.next?.design !== design || state.next.key !== key)) {
    const reply = await call("api_next_runs", request);
    if (state.design !== design) return; // another run sheet took its place meanwhile
    state.next = { design, key, ...nextOf(reply.ok ? reply.result : {}) };
  }
  renderNext();
}

/** The next step as the page holds it, from api_next_runs's reply or the worked example's. */
const nextOf = (result) => ({
  offers: result.offers ?? [],
  notes: result.notes ?? [],
  recommendation: result.recommendation ?? null,
  waiting: result.waiting ?? null,
});

/**
 * Show the next step once there is an analysis: what the results call for, and the runs to add. Before Python has
 * said what the results call for (a run without its result, or a study it does not search), the moved ranges show
 * when the best settings point past them.
 */
function renderNext() {
  const design = state.design;
  const next = design && state.next?.design === design ? state.next : nextOf({});
  const called = next.recommendation;
  const best = state.best?.settings && !isMixture(design?.spec.design_type) ? state.best : null;
  const move = !called && Boolean(best) && movesOn(best);
  const shown = called || next.waiting || move || next.offers.length || next.notes.length || pathRows(design ?? { columns: [], rows: [] }).length;
  $("#next").hidden = !design || !state.analysis || !shown;
  $("#to-next").hidden = $("#next").hidden;
  $("#next-empty").hidden = !$("#next").hidden;
  renderRounds();
  if ($("#next").hidden) return;
  $("#next-out").replaceChildren(
    ...(called ? calledFor(called) : next.waiting ? [said("hint", next.waiting)] : []),
    ...pathPart(called?.path ?? null),
    ...(move ? movedStudy(best) : []),
    ...(next.offers.length || next.notes.length ? addedRuns(next, Boolean(called)) : []),
  );
}

let roundsTicket = 0;

/**
 * Show the campaign the study on the page is a round of (campaign.js roundsOf), from the studies this browser keeps:
 * each round's study, its runs and its best result so far, and a button that opens it. A round built on runs that
 * have changed since says so. A study that stands alone, as most do, is one round, and nothing is listed.
 */
async function renderRounds() {
  const mine = ++roundsTicket;
  const id = state.campaign?.id;
  let rounds = [];
  if (id && state.design) {
    // The study on the page as it is now, which its saved copy lags by a moment.
    const here = { ...currentCampaign(), id, created: state.campaign.created };
    const records = [...(await savedSoon()).filter((record) => record.id !== id), here];
    const builtOn = new Set(records.map((record) => record.builds_on?.[0]?.id));
    const digests = new Map();
    for (const record of records) if (builtOn.has(record.id)) digests.set(record.id, await runsDigest(record));
    rounds = roundsOf(records, id, digests);
  }
  if (mine !== roundsTicket) return; // another study, or a later look, has the page
  const badge = $("#badge-next");
  const round = rounds.find((one) => one.id === id)?.round;
  $("#rounds").hidden = rounds.length < 2;
  badge.hidden = $("#rounds").hidden;
  badge.textContent = badge.hidden ? "" : `round ${round}`;
  if ($("#rounds").hidden) return;
  $("#next-empty").hidden = true;
  const rows = rounds.map((one) => {
    const study = el("span", "", one.missing ? `${one.title || "An earlier study"} (not in this browser)` : one.title);
    const open = el("button", "link", "Open");
    open.type = "button";
    open.setAttribute("aria-label", `Open round ${one.round}: ${one.title}`);
    open.onclick = () => openSaved(one.record);
    return [`Round ${one.round}`, study, one.record ? progress(one.record.responses) : "", one.record ? roundBest(one.record) : "", one.id === id ? "This study" : one.record ? open : ""];
  });
  const stale = rounds.filter((one) => one.stale).map((one) => {
    const before = rounds.find((other) => other.id === one.record.builds_on[0].id);
    return `Round ${one.round} was built on round ${before.round} as it was then: a run or a result of round ${before.round} has changed since.`;
  });
  const listed = builtTable(["Round", "Study", "Runs", "Best so far", "Open it"], rows);
  for (const row of listed.querySelectorAll("tr")) row.lastElementChild.classList.add("rounds__open"); // the page's own: no part of a report
  $("#rounds-out").replaceChildren(said("price", campaignSoFar(rounds)), listed, ...noteList(stale, "warnings"));
  listRoundsUpTo(rounds);
  drawCampaign(rounds, mine);
}

/** The rounds the campaign's charts can stop at: each round's number once, the last chosen unless the reader chose another that is still there. */
function listRoundsUpTo(rounds) {
  const numbers = [...new Set(rounds.map((one) => one.round))].sort((a, b) => a - b);
  const select = $("#rounds-upto");
  const chosen = numbers.includes(Number(select.value)) && select.dataset.rounds === String(numbers) ? select.value : String(numbers.at(-1));
  select.dataset.rounds = String(numbers);
  select.replaceChildren(...numbers.map((number) => Object.assign(el("option", "", `round ${number}`), { value: String(number) })));
  select.value = chosen;
  $("#rounds-as-of").hidden = numbers.length < 2;
}

/** Each round's fitted model for the campaign's map, by the round's id and what it was fitted to: asked of Python once. */
const roundFits = new Map();

/**
 * What Python fits the first response of a study this browser keeps from (bootstrap.predictors), as fitsAsked sends
 * the study on the page: its design, that response's results but those set aside, and its model.
 */
function fitAskedOf(record) {
  const { spec, columns, rows } = record.design;
  const aside = record.set_aside?.[spec.response] ?? [];
  const responses = record.responses.map((value, run) => (aside.includes(run) ? null : value));
  return { spec, columns, rows, responses, models: { [spec.response]: asked(record.model) } };
}

/**
 * A round's model for the campaign's map: its first response as Python fits it and how its factors are coded, or null
 * for a round with no model to draw (no results yet, a mixture, or a model that cannot be fitted). The round on the
 * page has its fit already, from the what-if panel.
 */
async function roundFit(one, here) {
  if (one.id === here && state.whatIf) {
    const fit = state.whatIf.responses.find((response) => response.name === state.design.spec.response && response.terms);
    return fit ? { fit, factors: state.whatIf.factors } : null;
  }
  const sent = fitAskedOf(one.record);
  const key = JSON.stringify([one.id, sent.rows, sent.responses, sent.models]);
  if (!roundFits.has(key)) {
    const reply = await call("api_predictors", sent);
    const fit = reply.ok ? reply.result.responses.find((response) => response.name === sent.spec.response && response.terms) : null;
    roundFits.set(key, fit ? { fit, factors: reply.result.factors } : null);
  }
  return roundFits.get(key);
}

/**
 * Draw the campaign's two charts for the rounds up to the one chosen: every run in the order they were made, with
 * the best result so far for the goal of the study on the page, and, once that study's models are fitted, every run
 * over two factors with each round's model inside that round's own ranges. ``mine`` is the look at the rounds this
 * drawing belongs to (renderRounds): a later look takes the page.
 */
async function drawCampaign(rounds, mine) {
  const upTo = Number($("#rounds-upto").value);
  const shown = rounds.filter((one) => one.record && one.round <= upTo);
  const spec = state.design.spec;
  const info = spec.response_info ?? {};
  const words = { label: info.label || spec.response, units: info.units ?? "", goal: info.goal ?? "", target: info.target ?? null };
  const w = state.whatIf;
  const pair = w ? mapPair(w.factors, state.contourPairs[state.shown]) : null;
  const runs = campaignRuns(shown, pair);
  $("#campaign-charts").hidden = !runs.length || !state.plotlyJs;
  if ($("#campaign-charts").hidden) return;
  const best = bestSoFar(runs, info);
  draw($("#campaign-runs-plot"), () => campaignRunsFigure(runs, best, words), state.plotlyJs, 300, { python: () => campaignRunsCode(runs, words) });
  $("#campaign-map").hidden = !pair;
  if (!pair) return;
  const id = state.campaign.id;
  const drawn = [];
  for (const one of shown) {
    const made = runs.filter((run) => run.round === one.round && run.at);
    const found = await roundFit(one, id);
    if (mine !== roundsTicket) return; // another study, or a later look, has the page
    // A round's model is drawn where both factors are ones it moved over a range: over that range, and no further.
    const moved = found && pair.every((name) => found.factors.some((f) => f.name === name && f.kind === "numeric"));
    const middle = moved ? Object.fromEntries(found.factors.map((f) => [f.name, f.kind === "numeric" ? (f.low + f.high) / 2 : f.levels[0]])) : null;
    drawn.push({
      round: one.round,
      x: made.map((run) => run.at[0]),
      y: made.map((run) => run.at[1]),
      surface: moved ? surfaceOf(found.fit, found.factors, middle, pair) : null,
      ...(moved ? { fit: found.fit, at: columnsAt(found.factors, middle), ranges: pair.map((name) => found.factors.find((f) => f.name === name)).map((f) => [Math.min(f.low, f.high), Math.max(f.low, f.high)]) } : { fit: null }),
    });
  }
  const units = Object.fromEntries(spec.factors.map((f) => [f.name, f.units ?? ""]));
  const axes = pair.map((name) => ({ name, units: units[name] }));
  $("#campaign-map-title").textContent = `${words.label} over ${pair[0]} and ${pair[1]}, round by round`;
  const python = () => campaignMapCode(drawn, { ...words, pair, axes, points: MAP_POINTS });
  draw($("#campaign-map-plot"), () => campaignMapFigure(drawn, { ...words, axes }), state.plotlyJs, 340, { python });
}

/** A round's best result so far for the goal of its first response, as the rounds' table writes it; "" for none. */
function roundBest(record) {
  const info = record.design.spec.response_info ?? {};
  const best = bestResult(record.responses, info);
  return best ? `${withUnit(best.value, info.units ?? "")}, in run ${best.run + 1}` : "";
}

/** A setting as the next step's table writes it: a number with its units, or a categorical factor's level. */
const stepSetting = (f, value) => (typeof value === "string" ? value : withUnit(value, f.units ?? ""));

/**
 * What the results call for (bootstrap._recommended): the step in a sentence, the settings or ranges it acts on, and
 * the button that carries it out; why, and what to do with its runs, folded below. The button adds confirmation runs,
 * adds the runs to the run sheet, or starts a new study with the ranges moved or widened.
 */
function calledFor(step) {
  const factors = state.design.spec.factors;
  const part = el("div", "next__part next__called", "");
  part.dataset.action = step.action;
  part.append(el("h4", "", "What the results call for"), said("", step.headline));
  if (step.does) part.append(step.does === "study" ? newStudyPrice() : priced(step.n_runs));
  if (step.settings) part.append(table(["Factor", "Setting"], factors.map((f) => [f.name, stepSetting(f, step.settings[f.name])])));
  if (step.ranges) {
    part.append(table(["Factor", "Range now", "Next study"], factors.map((f, i) => [f.name, rangeWords(f), rangeWords(step.ranges[i])])));
  }
  if (step.does) {
    const go = el("button", "button", step.label);
    go.type = "button";
    if (step.does === "study") go.id = "next-study";
    go.onclick = () => {
      if (step.does === "confirm") addConfirmationRuns(step.settings, "the settings the results call for");
      else if (step.does === "runs") addRuns("recommended");
      else startMovedStudy(step.ranges, step.action === "widen_ranges");
    };
    const actions = el("p", "actions", "");
    actions.append(go);
    part.append(actions);
  } else if (step.outside_limits) {
    part.append(said("hint", "Those runs would fall outside your limits on the settings, so Learn does not add them."));
  }
  const why = el("details", "next__why", "");
  why.append(el("summary", "", "Why, and what then"), ...noteList(step.reasons), said("hint", step.after), said("hint", `Judged by ${step.judged_by}.`));
  part.append(why);
  return [part];
}

/**
 * The runs along the path of steepest ascent (or descent): the step's own (``path``, bootstrap._path_runs), before
 * they are on the run sheet, with the button that adds them; then the run sheet's, with their results, where the
 * path stands (campaign.js pathVerdict) and the new study centred on the best of them. They are made in the path's
 * order, and the path is left at the first run that does worse than the run before it.
 */
function pathPart(path) {
  const design = state.design;
  const factors = design.spec.factors;
  const at = pathRows(design);
  if (!at.length && !path) return [];
  const part = el("div", "next__part next__path", "");
  part.append(el("h4", "", "The runs along the path"));
  const first = resultNames()[0];
  const info = design.spec.response_info ?? {};
  const settings = at.length ? at.map((i) => Object.fromEntries(factors.map((f) => [f.name, design.rows[i][design.columns.indexOf(f.name)]]))) : path.map((run) => run.settings);
  const predicted = path?.length === settings.length && path.every((run) => run.predicted !== null) ? path.map((run) => run.predicted) : null;
  const results = at.map((i) => state.results[first]?.[i] ?? null);
  const order = design.columns.indexOf("Run order");
  const headings = [at.length ? "Run" : "Step", ...factors.map((f) => f.name), ...(predicted ? ["The model predicts"] : []), ...(at.length ? ["Result"] : [])];
  const rows = settings.map((run, i) => [
    at.length ? (design.rows[at[i]][order] ?? at[i] + 1) : i + 1,
    ...factors.map((f) => stepSetting(f, run[f.name])),
    ...(predicted ? [withUnit(Number(predicted[i].toPrecision(4)), info.units ?? "")] : []),
    ...(at.length ? [results[i] === null ? "" : withUnit(results[i], info.units ?? "")] : []),
  ]);
  part.append(
    said("", "Make these runs one at a time, in this order. Stop at the first run that does worse than the run before it."),
    said("hint", "The path does not know your process: before you make a run, check that its settings are possible and safe."),
    ...(predicted ? [said("hint", "Past the ranges the model is a guide only: its predictions say where to look, not what you will find.")] : []),
    table(headings, rows),
  );
  const actions = el("p", "actions", "");
  const button = (id, cls, label, press) => {
    const one = el("button", cls, label);
    one.type = "button";
    one.id = id;
    one.onclick = press;
    actions.append(one);
  };
  if (!at.length) {
    part.append(priced(path.length));
    button("path-add", "button", `Add these ${path.length} runs to the run sheet`, () => addPathRuns(path));
  } else {
    const verdict = pathVerdict(results, info);
    const runOf = (i) => design.rows[at[i]][order] ?? at[i] + 1;
    part.append(said("", pathWords(verdict, runOf, results, info)));
    if (verdict.best !== null) {
      button("path-study", "button", `Start a new study centred on run ${runOf(verdict.best)}`, () => startMovedStudy(centredOn(factors, settings[verdict.best])));
    }
    button("path-remove", "link", "Take these runs off the run sheet", removePathRuns);
  }
  part.append(actions);
  return [part];
}

/** Where the path stands, in a sentence or two: the best run so far, the run that did worse, and the run to make next. */
function pathWords(verdict, runOf, results, info) {
  const value = (i) => withUnit(results[i], info.units ?? "");
  if (verdict.best === null) return "None of these runs has a result yet: enter each in Run as you make it.";
  const best = `Run ${runOf(verdict.best)} is the best of them so far (${value(verdict.best)}).`;
  if (verdict.worse !== null) return `Run ${runOf(verdict.worse)} did worse than run ${runOf(verdict.best)}: stop here. ${best}`;
  if (verdict.next !== null) return `${best} Make run ${runOf(verdict.next)} next.`;
  return `Each run did better than the one before. ${best} A new study centred there can go further.`;
}

/** The factors of a new study centred on a run: each numeric factor's range as wide as it is, with the run's setting at its middle. */
function centredOn(factors, run) {
  const clean = (x) => Number(x.toPrecision(12));
  return factors.map((f) => {
    const middle = Number(run[f.name]);
    if (isCategorical(f) || !Number.isFinite(middle)) return { ...f };
    const half = (f.high - f.low) / 2;
    return { ...f, low: clean(middle - half), high: clean(middle + half) };
  });
}

/**
 * Add the runs along the path to the end of the run sheet, in the path's order. They never enter the fit. While the
 * results are practice ones, theirs are simulated as the other runs' are; else the reader makes them, one at a time.
 */
async function addPathRuns(path) {
  if (!state.design || pathRows(state.design).length) return;
  const before = state.design.rows.length;
  const { columns, rows } = withPathRuns(state.design, path.map((run) => run.settings));
  state.design = { ...state.design, columns, rows };
  for (const name of resultNames()) {
    state.results[name] = [...state.results[name], ...Array(path.length).fill(null)];
    state.marks[name] = [...state.marks[name], ...Array(path.length).fill(null)];
  }
  curvesFor(path.length);
  renderRuns();
  renderStudyBar();
  const practice = Boolean(state.simulation) && someSimulated();
  const design = state.design;
  if (practice) await simulateFrom(before);
  if (state.design !== design) return; // another run sheet took its place meanwhile
  showResultsCode();
  scheduleSave();
  await analyseRuns(); // the fit is the same: the path's runs are no part of it
  if (state.design !== design) return;
  setStatus(
    practice
      ? `${path.length} runs along the path added, their results simulated.`
      : `${path.length} runs along the path added at the end of the run sheet: make them in order, and enter each result in Run.`,
  );
}

/** Take the runs along the path off the run sheet, with their results. */
async function removePathRuns() {
  if (!state.design) return;
  const { columns, rows, results, marks, curves } = withoutPathRuns(state.design, state.results, state.marks, state.curves);
  state.design = { ...state.design, columns, rows };
  state.results = results;
  state.marks = marks;
  state.curves = curves;
  renderRuns();
  renderStudyBar();
  showResultsCode();
  scheduleSave();
  await analyseRuns();
  setStatus("The runs along the path are off the run sheet.");
}

/** A move's price under the move: the runs it adds, beside the runs the study has already (campaign.js priceWords). */
const priced = (runs) => said("price", `${priceWords(runs, state.design.rows.length)}.`);

/** A new study's price: the runs of this study's design, its confirmation runs apart, counted on their own. */
const newStudyPrice = () =>
  said("price", `A new study: ${priceWords(state.design.rows.length - apartRows(state.design).length)} with this study's design.`);

/** A factor's range as the next step's table writes it: "150 to 200 degC", or a categorical factor's levels. */
const rangeWords = (f) =>
  isCategorical(f) ? listNames(levelsOf(f)).replace(/ and (?=[^,]*$)/, " or ") : `${num(f.low)} to ${withUnit(f.high, f.units ?? "")}`;

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
    newStudyPrice(),
    table(["Factor", "Range now", "Next study"], factors.map((f, i) => [f.name, rangeWords(f), rangeWords(moved[i])])),
    el("p", "actions", ""),
  );
  step.lastChild.append(start);
  return [step];
}

/**
 * The ways to add runs to this study, each what it is for and a button, and why any way is not offered. Below what the
 * results call for, they are the other runs the study could take.
 */
function addedRuns(next, others = false) {
  const step = el("div", "next__part", "");
  step.append(el("h4", "", others ? "Other runs this study could take" : "Add runs to this study"));
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
    box.append(said("", `${offer.why}${blocked}`), priced(offer.runs), ...offerRuns(offer), add);
    step.append(box);
  }
  step.append(...noteList(next.notes));
  return [step];
}

/**
 * A move's runs before they are made, folded under its price: each run's settings and what the run is for, and, for
 * a study of two or three numeric factors, where the runs sit among the runs made (spaceOf). Nothing is added by
 * looking.
 */
function offerRuns(offer) {
  if (!offer.added?.length) return [];
  const design = state.design;
  const factors = design.spec.factors;
  const fold = el("details", "next__runs", "");
  fold.append(el("summary", "", `See these ${offer.added.length} runs before they are made`));
  if ([2, 3].includes(factors.length) && !factors.some(isCategorical)) {
    const sheet = sheetRuns(design);
    const words = `Where these runs sit among the runs made, over ${listNames(factors.map((f) => f.name))}: a filled point is a run to add.`;
    fold.append(spaceOf(spacePoints(sheet, factors), spacePoints(offer.added, factors), factors, words));
  }
  const rows = offer.added.map((run, i) => [...factors.map((f) => stepSetting(f, run[f.name])), offer.reasons?.[i] ?? ""]);
  fold.append(table([...factors.map((f) => f.name), "What it is for"], rows));
  return [fold];
}

/**
 * Add one way's runs to the end of the run sheet: the runs before keep their results, the runs added have none, and the
 * plan becomes the runs as given, in their blocks, so the run sheet is not out of date. While the results are practice
 * ones, the runs added are simulated from the same processes, noise and seeds, and analysed with the rest; else the
 * reader makes them, in Run.
 */
async function addRuns(kind) {
  const asked = state.design;
  if (!asked) return;
  setStatus("Adding the runs...", "busy");
  const adding = statusTicket;
  const reply = await call("api_add_runs", { ...analysisRequest(), kind }); // the runs the results call for need them
  if (dropped(state.design !== asked, adding)) return; // another run sheet took its place meanwhile
  if (!reply.ok) {
    setStatus(reply.error, "error");
    return;
  }
  const { added, ...grown } = reply.result;
  const before = state.design.rows.length;
  state.design = { ...state.design, ...grown };
  const design = state.design;
  for (const name of resultNames()) {
    state.results[name] = [...state.results[name], ...Array(added).fill(null)];
    state.marks[name] = [...state.marks[name], ...Array(added).fill(null)];
  }
  curvesFor(added);
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
  if (dropped(state.design !== design, adding)) return;
  showResultsCode();
  scheduleSave();
  await analyseRuns();
  if (state.design !== design) return; // nor is a study that took the page during the analysis told of these runs
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
  const asked = state.design;
  const { scenario, sigma, seed: drawn } = state.simulation;
  const reply = await call("api_simulate", { ...held(), scenario, sigma, seed: drawn });
  if (!reply.ok || state.design !== asked) return false; // refused, or another run sheet took its place meanwhile
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
async function startMovedStudy(factors, widened = false) {
  const spec = state.design.spec;
  const given = spec.design_type === GIVEN;
  // The new study is the next round of this one's campaign, when this one is a study the browser keeps: it names
  // this study, with the digest of its runs and results as they are now. The worked example is no such study.
  const before = state.campaign?.id ? currentCampaign() : null;
  const on = before ? [{ id: state.campaign.id, sha256: await runsDigest(before), title: before.title }] : [];
  newStudy();
  state.buildsOn = on;
  state.widened = widened; // nothing stood out in the study before: the next step stops rather than widen again
  setForm({ ...spec, factors, ...(given ? { design_type: "box_behnken", options: { n_center_points: 3 } } : {}) });
  if (given) designChosen = false;
  showPhase("plan");
  saveDraft();
  setStatus(`A new study, its ranges ${widened ? "widened" : "moved"}: check they are safe to run, then make its run sheet.`);
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
  studyTicket += 1;
  $("#campaign-banner").hidden = true;
  setForm(example.inputs.design);
  scenarios.choose(example.inputs.scenario);
  $("#sigma").value = example.inputs.sigma;
  showScenarioHint();
  showDesign(example.design, [], { save: false });
  state.exampleBrief = briefKey(); // its comparison, run sheet and seed are the build's while its brief is
  state.planKey = planKey();
  selectTab("simulate");
  state.simulation = example.simulation;
  const [name] = resultNames(); // the worked example measures one response
  state.results = { [name]: [...example.simulation.responses] };
  state.marks = { [name]: example.simulation.responses.map(() => "simulated") };
  state.setAside = {};
  state.contourPairs = {};
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
  showWhatIf(example.predictors ?? null); // the build's fit, so the panel follows each move before Python starts
  showTarget(); // the boxes for a target: nothing is asked of Python until one is typed
  showWindow(); // and those for a specification's windows
  if (example.next) state.next = { design: state.design, key: null, ...nextOf(example.next) }; // asked again on a change
  renderNext(); // what its results call for; its Box-Behnken design takes no runs added
  $("#demo-version").textContent = `process-improve ${example.version}`;
  $("#demo-when").textContent = "when this page was built";
  $("#welcome").hidden = true;
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
  studyTicket += 1;
  setForm(example.design);
  takeSettings(newStudySettings()); // the example is the reader's to look at, in the colours they keep
  scenarios.choose(example.scenario);
  $("#sigma").value = example.sigma;
  setModel(example.model);
  showScenarioHint();
  seed = example.seed;
  selectTab("simulate");
  if (!(await generate({ randomState: RUN_ORDER_SEED }))) return; // the worked example's runs, as the build made them
  state.campaign = null; // the worked example, run here, is not saved
  state.example = true;
  state.exampleBrief = briefKey();
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
  studyTicket += 1;
  cancelCandidates();
  Object.assign(state, {
    design: null,
    planKey: null,
    results: {},
    marks: {},
    shown: null,
    models: {},
    goals: {},
    scenarios: [],
    simulation: null,
    analysis: null,
    next: null,
    comparison: null,
    picked: null,
    campaign: null,
    example: false,
    practiceRound: false,
    widened: false,
    buildsOn: [],
  });
  takeSettings(newStudySettings());
  draftSeed = null; // a new study, which takes a seed of its own
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
  clearWhatIf();
  clearTarget();
  clearWindow();
  renderPicker();
  setModel("");
  for (const key of ["design", "run", "analyse", "best", "compare"]) hidePython(key);
  delete $("#compare-dialog").dataset.key;
  selectTab("real");
  if (location.search) history.replaceState(null, "", location.pathname);
  clearDraft(); // a draft left from before is not this study's
  if ($("#status").dataset.state === "ok") clearStatus(); // what was said of the study before, such as its analysis
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
  draftTimer = setTimeout(writeDraft, 300);
}

function writeDraft() {
  draftTimer = null;
  if (state.design || state.example) return;
  const draft = {
    response: responseFields(),
    responses: responseRows()
      .slice(1)
      .map((tr) => rowFields(tr)), // the later responses: each one's name, units, goal and target, as typed
    factors: factorRows().map((tr) =>
      typeOf(tr) === CATEGORICAL
        ? categoricalFactor(box(tr, "name").value, chipLevels(tr))
        : { name: box(tr, "name").value, low: box(tr, "low").value, high: box(tr, "high").value, units: box(tr, "units").value },
    ),
    aim: aimChosen ? currentAim() : null,
    limits: currentLimits(),
    // A mixture's factor table reads as components, blank bounds and all, so the draft keeps the design with it.
    design: mixturePlanned() ? $("#design-type").value : null,
    ...(state.buildsOn.length ? { builds_on: state.buildsOn } : {}),
  };
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    /* storage refused: the brief holds for this visit */
  }
}

/** Write a draft still waiting for its moment, at once: the page calls this when the reader leaves, as flushSave does. */
function flushDraft() {
  if (draftTimer === null) return;
  clearTimeout(draftTimer);
  writeDraft();
}

function clearDraft() {
  clearTimeout(draftTimer);
  draftTimer = null;
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
  const factors = (Array.isArray(draft.factors) ? draft.factors.slice(0, MAX_FACTORS) : [])
    .filter((f) => f && typeof f === "object")
    .map((f) =>
      f.type === "categorical" && Array.isArray(f.levels)
        ? categoricalFactor(text(f.name), f.levels.slice(0, MAX_LEVELS).map((v) => text(v, LABEL_MOST)))
        : { name: text(f.name), low: text(f.low), high: text(f.high), units: text(f.units, 32) },
    );
  let limits = [];
  try {
    limits = checkConstraints(draft.limits ?? []);
  } catch {
    /* limits this page did not write: the brief comes back without them */
  }
  const design = isMixture(draft.design) ? draft.design : null;
  let builtOn = [];
  try {
    builtOn = checkBuiltOn(draft.builds_on);
  } catch {
    /* a round before that this page did not write: the study stands alone */
  }
  return { response, responses, factors, aim: AIMS.includes(draft.aim) ? draft.aim : null, limits, design, builds_on: builtOn };
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
    listDesigns(draft.design);
    designChosen = true;
  }
  $("#factors tbody").replaceChildren();
  (draft.factors.length ? draft.factors : [{}, {}]).forEach((factor) => addFactor(factor));
  setLimits(draft.limits);
  aimChosen = Boolean(draft.aim);
  setAim(draft.aim ?? defaultAim(factorCount()));
  state.buildsOn = draft.builds_on ?? [];
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
    v: studyFormat(spec, {
      what_if: state.scenarios,
      set_aside: keptAside(),
      ...keptAsked(),
      builds_on: state.buildsOn,
      planned: state.design.planned ?? {},
      settings: state.settings,
    }),
    title: state.campaign?.title ?? titleOf(spec),
    design: { spec, columns, rows, model, n_runs: rows.length, facts: facts ?? null, outside_range: outside ?? [] },
    ...sentResults(),
    simulation: state.simulation,
    simulate: { scenario: $("#scenario").value, sigma: Number($("#sigma").value), seed },
    model: modelOf(first),
    ...(more.length ? { more_models: Object.fromEntries(more.map((name) => [name, modelOf(name)])) } : {}),
    ...(more.length && Object.keys(sentGoals()).length ? { goals: sentGoals() } : {}),
    ...(state.campaign?.source ? { source: state.campaign.source } : {}),
    ...(state.scenarios.length ? { what_if: state.scenarios } : {}),
    ...(Object.keys(keptAside()).length ? { set_aside: keptAside() } : {}),
    ...(state.design.planned ? { planned: state.design.planned } : {}),
    ...keptAsked(),
    ...(state.buildsOn.length ? { builds_on: state.buildsOn } : {}),
    ...(Object.keys(state.settings).length ? { settings: state.settings } : {}),
    ...sentCurves(),
  };
}

/** The targets and the windows typed in Learn, as a study keeps them: those of the responses it measures, if any. */
function keptAsked() {
  const names = resultNames();
  const mine = (byName) => Object.fromEntries(Object.entries(byName).filter(([name]) => names.includes(name)));
  const [targets, windows] = [mine(state.target.targets), mine(sentWindows())];
  return { ...(Object.keys(targets).length ? { targets } : {}), ...(Object.keys(windows).length ? { windows } : {}) };
}

/** The runs set aside as a study keeps them: for each response with some, those that have a result. */
function keptAside() {
  const kept = resultNames().map((name) => [name, asideRuns(name)]);
  return Object.fromEntries(kept.filter(([, runs]) => runs.length));
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
  countLibrary(saved.length);
  $("#library-list").replaceChildren(...saved.map(libraryItem));
}

/** Say how many studies this browser keeps: on the study's menu, and in Your studies, which says so when it keeps none. */
function countLibrary(count) {
  $("#library-count").textContent = String(count);
  $("#download-all").disabled = count === 0;
  $("#library-none").hidden = count > 0;
}

/** Show Your studies, over the page. */
function openLibrary() {
  if (!$("#library").open) $("#library").showModal();
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
  // The study (its title and how far it has got), then Share, then Delete: the list's three columns (style.css).
  const study = el("span", "library__study", "");
  study.append(open, about);
  const actions = el("span", "library__actions", "");
  actions.append(shareButton(record, title));
  item.append(study, actions);
  if (here) {
    study.append(el("span", "hint", "on the page now"));
    return item;
  }
  const forget = el("button", "link library__delete", "Delete");
  forget.type = "button";
  forget.setAttribute("aria-label", `Delete ${title}`);
  forget.onclick = async () => {
    await forgetCampaign(record.id);
    const undo = el("button", "link library__delete", "Undo");
    undo.type = "button";
    undo.onclick = async () => {
      await saveCampaign(record);
      await refreshLibrary();
    };
    item.replaceChildren(el("span", "hint", `Deleted ${title}.`), undo);
    countLibrary((await savedCampaigns()).length);
  };
  actions.append(forget);
  return item;
}

// ---------------------------------------------------------------- the study file
/** Whether this browser can hand a file to the system's share sheet (an iPad's, a phone's), not only save it. */
const SHARES_FILES = (() => {
  try {
    return Boolean(navigator.canShare?.({ files: [new File(["{}"], "study.factorial.json", { type: "application/json" })] }));
  } catch {
    return false;
  }
})();

/** The study file of `records`, the studies as this browser keeps them, as a file named `name`; `opens` as studyFile takes it. */
async function studyFileOf(records, name, opens = null) {
  const app = { url: `${location.origin}${location.pathname}`, process_improve: state.version };
  const json = await studyFile(records, { app, opens });
  return new File([`${JSON.stringify(json, null, 2)}\n`], name, { type: "application/json" });
}

/**
 * A study's own file: the study, and with it the rounds before it that this browser keeps, the first round first,
 * so the file carries the campaign the study is a round of. The file says which study it is of.
 */
async function ownStudyFile(record, title) {
  const kept = new Map((await savedCampaigns()).map((one) => [one.id, one]));
  const rounds = [record];
  for (let on = record.builds_on?.[0]; on && kept.has(on.id) && !rounds.includes(kept.get(on.id)); on = rounds[0].builds_on?.[0]) {
    rounds.unshift(kept.get(on.id));
  }
  return studyFileOf(rounds, `${fileStem(title)}.factorial.json`, rounds.length > 1 ? record.id : null);
}

/**
 * A study's Share: its study file, handed to the share sheet (Mail, Messages, AirDrop, Files) where the browser can
 * share a file, else saved as a download, and the button then says Download. A browser shares only straight after
 * a tap, so the file is made as the finger or the focus reaches the button, and is ready when the tap lands.
 */
function shareButton(record, title) {
  const button = el("button", "link library__share", SHARES_FILES ? "Share" : "Download");
  button.type = "button";
  button.setAttribute("aria-label", `${button.textContent} ${title}`);
  let file = null;
  const make = () => (file ??= ownStudyFile(record, title));
  button.onpointerdown = make;
  button.onfocus = make;
  button.onclick = async () => {
    const ready = await make();
    if (!SHARES_FILES) {
      download(ready, ready.name);
      return;
    }
    try {
      await navigator.share({ files: [ready], title });
    } catch (err) {
      if (err.name !== "AbortError") download(ready, ready.name); // the sheet would not take it: save it instead
    }
  };
  return button;
}

/** Today in this browser's time zone, as a file name carries it: 2026-10-21. */
function today() {
  const now = new Date();
  return [now.getFullYear(), now.getMonth() + 1, now.getDate()].map((n) => String(n).padStart(2, "0")).join("-");
}

/** Save every study kept here as one study file: a copy to keep elsewhere, or to open in another browser. */
async function downloadAll() {
  const file = await studyFileOf(await savedCampaigns(), `factorial-studies-${today()}.json`);
  download(file, file.name);
}

/**
 * Open a study file. A file of one study puts it on the page, as a link does; a file of several adds them all to
 * Your studies. A study the very same as one kept here is not saved twice, and nothing kept here is overwritten: a
 * study with the id of a different one here comes in beside it, under an id of its own.
 */
async function openStudyFile(file) {
  let read;
  try {
    read = await readStudyFile(await file.text());
  } catch (err) {
    showBanner(err.message, true);
    return;
  }
  const kept = new Map((await savedCampaigns()).map((record) => [record.id, record]));
  const sameAs = (record, study) => {
    try {
      return canonicalJson(checkCampaign(record)) === canonicalJson(study);
    } catch {
      return false; // a study kept here that this page cannot read is not the same as any
    }
  };
  const opened = [];
  let already = 0;
  for (const { id, created, updated, study } of read.studies) {
    const here = id === null ? undefined : kept.get(id);
    if (here && sameAs(here, study)) {
      already += 1;
      opened.push(here);
      continue;
    }
    const record = { ...study, id: here || id === null ? newCampaign().id : id, created, updated };
    await saveCampaign(record);
    opened.push(record);
  }
  const refused = read.refused.map(({ title, reason }) => `${title} (${reason})`);
  const changed = read.studies.filter((s) => s.changed).map((s) => s.study.title || "A study");
  const notes = [
    changed.length ? ` Changed by hand since it was saved: ${listNames(changed)}.` : "",
    refused.length ? ` Not opened: ${refused.join("; ")}.` : "",
  ].join("");
  // A file of one study opens it; so does a file of one study and the rounds before it, which says which it is of.
  const of = read.studies.length === 1 ? 0 : read.studies.findIndex((one) => read.opens !== null && one.id === read.opens);
  if (of >= 0 && opened.length === read.studies.length) {
    const record = opened[of];
    const before = opened.length - 1;
    const where = before
      ? `It and the ${before === 1 ? "round" : `${before} rounds`} before it are saved in this browser, under Your studies.`
      : already
        ? "It was already saved in this browser."
        : "It is saved in this browser now, under Your studies.";
    await openCampaign(checkCampaign(record), { id: record.id, created: record.created }, `Opened ${record.title || "a study"} from ${file.name}. ${where}${notes}`);
    return;
  }
  await refreshLibrary();
  openLibrary();
  const added = opened.length - already;
  const counted = `${added} ${added === 1 ? "study" : "studies"} added from ${file.name}${already ? `, ${already} already here` : ""}.`;
  showBanner(`${counted}${notes}`, opened.length === 0);
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
  await openCampaign(campaign, { id: record.id, created: record.created }, `Opened ${campaign.title}.`, true);
}

/** Say what was opened, or why a link could not be: a sentence, or a list of text and links. */
function showBanner(message, error = false, passing = false) {
  const banner = $("#campaign-banner");
  banner.replaceChildren(...[message].flat());
  banner.classList.toggle("error", error);
  banner.dataset.passing = String(passing);
  banner.hidden = false;
}

// A banner that only says where the reader is ("Picked up where you left off") has done its work once the reader
// does anything: the next press or key outside it takes it away. One that says what was opened, or what failed, stays.
for (const kind of ["pointerdown", "keydown"]) {
  document.addEventListener(
    kind,
    (event) => {
      const banner = $("#campaign-banner");
      if (banner.dataset.passing === "true" && !banner.hidden && !banner.contains(event.target)) banner.hidden = true;
    },
    true,
  );
}

/**
 * Put a saved or shared study on the page as it was left: the brief, the plan, the runs, their
 * results, the process they were simulated from and the model, in Run, or in Learn once it has
 * results. The analysis follows once Python has started.
 */
async function openCampaign(c, record, message, resume = false) {
  if ($("#library").open) $("#library").close();
  $("#welcome").hidden = true;
  flushSave();
  studyTicket += 1;
  $("#demo-banner").hidden = true;
  setForm(c.design.spec);
  const left = stageLeft(record.id);
  state.practiceRound = false;
  state.campaign = { ...record, title: c.title || titleOf(c.design.spec), ...(c.source ? { source: c.source } : {}) };
  state.design = { ...c.design, ...(c.planned ? { planned: structuredClone(c.planned) } : {}) };
  state.planKey = planKey();
  const names = resultNames();
  state.results = joinResults(names, [...c.responses], structuredClone(c.more_responses ?? {}));
  state.marks = joinResults(names, [...c.sources], structuredClone(c.more_sources ?? {}));
  state.setAside = structuredClone(c.set_aside ?? {});
  state.contourPairs = {};
  state.curves = c.curves ? structuredClone(c.curves) : null;
  state.shown = names[0];
  state.models = { [names[0]]: c.model, ...(c.more_models ?? {}) };
  state.goals = structuredClone(c.goals ?? {});
  state.trade = newTrade();
  state.target = { ...newTarget(), targets: structuredClone(c.targets ?? {}) };
  state.window = { ...newWindow(), windows: structuredClone(c.windows ?? {}) };
  state.scenarios = structuredClone(c.what_if ?? []);
  state.buildsOn = structuredClone(c.builds_on ?? []);
  takeSettings(structuredClone(c.settings ?? {})); // the study's own always win over the reader's
  clearBalance();
  clearWhatIf();
  clearTarget();
  clearWindow();
  state.simulation = c.simulation;
  state.analysis = null;
  state.example = false;
  state.picked = null;
  clearDraft();
  const { scenario, sigma, seed: kept } = c.simulate;
  if (scenario && scenarios.option(scenario)) scenarios.choose(scenario);
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
  showBanner(message, false, resume);
  renderStudyBar();
  // A study this browser kept opens on the stage it was left on; one from a link or a file, on the stage its
  // results call for. Read before the page is drawn, which keeps the stage it shows.
  showPhase((resume && left) || (done ? "learn" : "run"));
  scheduleSave(); // the study opened most recently is the one a later visit picks up
  if (!done) return;
  const mine = state.campaign.id;
  await whenReady.then(
    () => (state.campaign?.id === mine ? analyseRuns() : null),
    () => {},
  );
}

/** Open the study Python made of the reader's own table (import.js), saved here as a study of its own. */
async function openImported(made) {
  const c = checkCampaign({ v: CAMPAIGN_VERSION, ...made.campaign });
  await openCampaign(c, newCampaign(false), openedWords(c.design.n_runs, made.left_out.length));
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
      const saved = "It is saved in this browser now, under Your studies.";
      const opened = c.source
        ? [`Opened from the catalog: ${title}, as ${c.source.cite} prints it${c.source.where ? `, ${c.source.where}` : ""}. `, ...sourceLink(c.source), ` ${saved}`]
        : `Opened from a link: ${title}. ${saved}`;
      await openCampaign(c, newCampaign(false), opened);
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
        true,
      );
      return;
    } catch {
      /* a study this page cannot read: start a new one */
    }
  }
  // A first visit, and any visit with nothing kept here: a blank question, under a word of welcome that says what
  // Factorial is and offers the other ways in. A link that could not be opened has said so, in place of the welcome.
  showPhase("plan");
  if ($("#campaign-banner").textContent) return;
  // The welcome has the first screen, and the keyboard starts at the top of it: the question is one press away.
  $("#welcome").hidden = false;
}

/** Put the keyboard at a blank question's first box, what the study measures. Not on a touch screen, where it would open the keys. */
function toFirstQuestion() {
  if (!window.matchMedia("(pointer: coarse)").matches) $("#response").focus({ preventScroll: true });
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
const BUILDS_ON = {
  design: [],
  run: ["design"],
  analyse: ["design", "run"],
  best: ["design", "run", "analyse"],
  balance: ["design", "run"],
  "what-if": ["design", "run"],
  target: ["design", "run"],
  window: ["design", "run"],
};
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
  // The code holds the results the models are fitted to: a run set aside has none, as on the page.
  showPython("run", resultsCode(state.design, fitted(), simulated ? state.simulation : null));
}

/**
 * The Plotly.js version the build drew its figures with, so charts can be drawn before Python names one, and the
 * commit the app was built from, which a report of a problem names.
 */
async function readManifest() {
  try {
    const res = await fetch("wheels/manifest.json", { cache: "no-store" });
    if (res.ok) {
      const manifest = await res.json();
      const { plotly_js: plotlyJs, commit, committed, committed_at: committedAt } = manifest;
      state.plotlyJs ??= plotlyJs ?? null;
      state.commit = commit ?? null;
      state.committed = committed ?? null;
      state.committedAt = committedAt ?? null;
      showVersion();
      return manifest;
    }
  } catch {
    /* a plain local preview has no manifest */
  }
  return {};
}

/**
 * Say at the foot of the page which version is on it: the commit, as a link to it, and when it was made, in the
 * reader's own time zone to the minute; the day alone from a build that names no moment.
 */
function showVersion() {
  const version = $("#version");
  version.hidden = !state.commit;
  if (!state.commit) return;
  const commit = link(`${REPOSITORY}/commit/${state.commit}`, shortCommit(state.commit), "_blank");
  commit.title = state.commit;
  commit.setAttribute("aria-label", `Commit ${shortCommit(state.commit)} on GitHub`);
  const day = momentWords(state.committedAt) ?? dayWords(state.committed);
  version.replaceChildren("Version ", commit, day ? `, ${day}.` : ".");
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
 * A link to the book, or to another source, opening in a tab of its own so the page keeps its state. The book's
 * is one tab, the same each time (campaign.tabOf says how); a link to another source opens a new one.
 */
function link(href, text, target = BOOK_TAB) {
  const a = el("a", "", text);
  a.href = href;
  a.target = target;
  a.rel = target === BOOK_TAB ? "opener" : "noopener";
  return a;
}

/**
 * A table of results; `terms` names the glossary entry of a column whose heading is a term. `sortedBy` names the
 * column the rows arrive sorted on, smallest first, and `looks` gives a column its look (style.css: dim, strong).
 */
function table(columns, rows, terms = {}, { sortedBy = null, looks = {}, run = null } = {}) {
  const wrap = el("div", "scroll", "");
  const t = el("table", "data", "");
  t._looks = columns.map((c) => looks[c] ?? "");
  t._run = run; // for a table with a row a run: the run of a row, which marks it on every chart (fillBody)
  renderTable(t, columns, rows, sortedBy, [], terms);
  wrap.append(t);
  return wrap;
}

/**
 * A table whose cells may hold an element, as a row's button is: its rows stay as given, where table()'s are text
 * a reader can sort.
 */
function builtTable(columns, rows) {
  const wrap = el("div", "scroll", "");
  const t = el("table", "data", "");
  const head = t.createTHead().insertRow();
  for (const text of columns) {
    const th = el("th", "", text);
    th.scope = "col";
    head.append(th);
  }
  const body = t.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const cell of row) tr.insertCell().append(cell instanceof Node ? cell : String(cell ?? ""));
  }
  wrap.append(t);
  return wrap;
}

/**
 * The table of a fit's coefficients, and under it the button that saves it as a file.
 *
 * - The most significant comes first and the intercept last (logic.bySignificance), which the line above the table
 *   says. No heading is marked as the one the rows are sorted on: the intercept, last whatever its p-value, would
 *   make that untrue. A reader sorts by any column as in every table.
 * - Each estimate stands between the ends of its 95% interval, the low end on its left and the high end on its
 *   right, so the three read in the order of their size. The ends are dimmed and the estimate is bold.
 * - The file is the table as it is shown, its rows in the order the reader left them in, with each number in full:
 *   the page shows four figures, and a reader who works on with the numbers wants what was computed.
 */
function coefficientsTable(coefficients, { ratio, vif, simulated }) {
  const listed = bySignificance(coefficients);
  const end = (c, which) => (c.std_error === null ? "" : fmt(c[which]));
  const columns = [
    ["Term", (c) => c.label, (c) => c.label],
    ["95% low", (c) => end(c, "ci_low"), (c) => c.ci_low],
    ["Coefficient", (c) => fmt(c.coefficient), (c) => c.coefficient],
    ["95% high", (c) => end(c, "ci_high"), (c) => c.ci_high],
    ["Std. error", (c) => fmt(c.std_error), (c) => c.std_error],
    ["p-value", (c) => pval(c.p_value), (c) => c.p_value],
    ...(ratio ? [["Low to high", (c) => (c.ratio === undefined || c.ratio === null ? "" : `×${fmt(c.ratio)}`), (c) => c.ratio]] : []),
    ...(vif ? [["VIF", (c) => fmt(c.vif), (c) => c.vif]] : []),
    ...(simulated ? [["True value", (c) => fmt(c.true_value), (c) => c.true_value]] : []),
  ];
  const names = columns.map(([name]) => name);
  const rows = listed.map((c) => columns.map(([, shown]) => shown(c)));
  const inFull = new Map(rows.map((row, i) => [row, columns.map(([, , value]) => value(listed[i]) ?? null)]));
  const shown = table(
    names,
    rows,
    {
      "95% low": "interval",
      Coefficient: "coefficient",
      "95% high": "interval",
      "Std. error": "std-error",
      "p-value": "p-value",
      VIF: "vif",
      "True value": "true-value",
    },
    { looks: { "95% low": "dim", Coefficient: "strong", "95% high": "dim" } },
  );
  const save = el("button", "button", "Download this table (.csv)");
  save.type = "button";
  save.onclick = () => {
    const text = csvOf(names, shown.querySelector("table")._shown.map((row) => inFull.get(row)));
    download(new Blob([text], { type: "text/csv" }), `coefficients-${shownSpec().response}.csv`);
  };
  const actions = el("p", "actions", "");
  actions.append(save);
  return [shown, actions];
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
      const sorted = c === sortedBy ? "ascending" : "none";
      return `<th${lookOf(t, i)} aria-sort="${sorted}"><button type="button" class="sort" data-col="${i}">${name}</button>${unit}</th>`;
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

/** The class of a table's column, as an attribute: its look, where the table gives its columns looks. */
const lookOf = (t, col) => (t._looks?.[col] ? ` class="${t._looks[col]}"` : "");

/** Fill a table's body with these rows, in this order, and note them as the rows it shows. */
function fillBody(t, rows) {
  const cell = (row, v, c) => {
    const own = t._cell ? t._cell(row, c) : null;
    return own ?? escape(typeof v === "number" ? +v.toPrecision(6) : (v ?? ""));
  };
  // A row that is a run says which, and is marked while its run is (markRows).
  const run = (row) => {
    if (!t._run) return "";
    return ` data-run="${t._run(row)}"${t._run(row) === markedRun() ? ' class="run-marked"' : ""}`;
  };
  t._shown = rows;
  t.tBodies[0].innerHTML = rows
    .map((row) => `<tr${run(row)}>${row.map((v, c) => `<td${lookOf(t, c)}>${cell(row, v, c)}</td>`).join("")}</tr>`)
    .join("");
}

// ---------------------------------------------------------------- one run, marked wherever it is shown
/** Mark a run's row in every table with a row a run: charts.js marks its point on every chart, and says so here. */
function markRows(run) {
  for (const row of document.querySelectorAll("tr[data-run]")) {
    row.classList.toggle("run-marked", run !== null && Number(row.dataset.run) === run);
  }
}
onRunMarked(markRows);

/** The run of the row an event is in, or null. */
function rowRun(target) {
  const row = target?.closest?.("tr[data-run]");
  return row ? Number(row.dataset.run) : null;
}

// A run's row marks the run as its point on a chart does: while the pointer rests on it, and for good once it is
// pressed, until it is pressed again or Escape lets it go. A press on a box to type in, or on a button, is its own.
document.addEventListener("mouseover", (event) => {
  const run = rowRun(event.target);
  if (run !== null) restOnRun(run);
});
document.addEventListener("mouseout", (event) => {
  if (rowRun(event.target) !== null && rowRun(event.relatedTarget) === null) restOnRun(null);
});
document.addEventListener("click", (event) => {
  const run = rowRun(event.target);
  if (run !== null && !event.target.closest("input, button, a, select")) pressRun(run);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.querySelector("dialog[open]")) pressRun(null);
});

// ---------------------------------------------------------------- the list of designs
/**
 * List every design the page makes in the design list, with ``chosen`` as the plan's design. "Runs as given" is
 * listed once a study has had such runs, and while it is the plan's design.
 */
function listDesigns(chosen = $("#design-type").value) {
  designs.show((option) => option.value !== GIVEN || alsoListed.has(GIVEN), chosen);
}

$("#design-type").addEventListener("change", () => listDesigns());

// ---------------------------------------------------------------- a study's settings
const SETTINGS_KEY = "factorial-settings";

/** The settings a new study starts from: those the reader asked to keep for new studies, else none. */
function newStudySettings() {
  try {
    return studySettings(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}"));
  } catch {
    return {}; // none kept, storage refused, or what is kept is not settings: a new study starts from the defaults
  }
}

/** Put a study's settings in force: the page holds them, and the charts are drawn in its colour map from here on. */
function takeSettings(settings) {
  state.settings = settings;
  setColourMap(settings.colour_map, settings.colour_map_flipped === true);
}

/**
 * Draw the settings as the study has them. Each colour map is a choice on one line: its name, then its colours
 * from a low value to a high one as the charts draw them now. The chosen map's line holds the switch that flips it.
 */
function renderSettings() {
  const choices = Object.keys(COLOUR_MAPS).map((name) => {
    const row = el("div", "settings__map", "");
    const label = el("label", "settings__choice", "");
    const choice = document.createElement("input");
    choice.type = "radio";
    choice.name = "colour_map";
    choice.value = name;
    choice.checked = name === colourMap();
    label.append(choice, el("span", "", COLOUR_MAP_WORDS[name]));
    const swatch = el("span", "settings__swatch", "");
    swatch.style.background = `linear-gradient(to right, ${shownColours(name).join(", ")})`;
    swatch.setAttribute("role", "img");
    swatch.setAttribute("aria-label", `The colours of ${COLOUR_MAP_WORDS[name]}, from a low value to a high one`);
    // Every line has the same five places, so the bars line up: the chosen map's holds the switch, the others a gap.
    const flip = choice.checked ? flipSwitch() : el("span", "settings__flip", "");
    row.append(label, flip, el("span", "settings__end", "Low"), swatch, el("span", "settings__end", "High"));
    return row;
  });
  $("#colour-maps").replaceChildren(...choices);
  $("#settings-keep").checked = localStorage.getItem(SETTINGS_KEY) !== null;
}

/** Where the switch that flips the chosen map is: the list draws it anew with each change. */
const FLIP = "#colour-maps .settings__flip input";

/** The switch that draws the chosen map from its other end, at the end of that map's line. */
function flipSwitch() {
  const label = el("label", "switch settings__flip", "");
  const flip = document.createElement("input");
  flip.type = "checkbox";
  flip.id = "colour-map-flipped";
  flip.setAttribute("role", "switch");
  flip.checked = colourMapFlipped();
  const track = el("span", "switch__track", "");
  track.setAttribute("aria-hidden", "true");
  label.append(el("span", "switch__text", "Flip"), flip, track);
  return label;
}

/** The reader changed a setting: the study takes it, the charts follow, and the study is saved with it. */
function settingsChanged(event) {
  const chosen = document.querySelector('#colour-maps input[name="colour_map"]:checked')?.value;
  const settings = { ...state.settings, ...(chosen ? { colour_map: chosen } : {}) };
  if (document.querySelector(FLIP)?.checked) settings.colour_map_flipped = true;
  else delete settings.colour_map_flipped;
  takeSettings(settings);
  try {
    if ($("#settings-keep").checked) localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
    else localStorage.removeItem(SETTINGS_KEY);
  } catch {
    /* storage refused: the settings hold for this study */
  }
  renderSettings(); // the colours beside each name, and the line the switch is on, follow the choice
  // The list is drawn anew, so the keyboard's place is put back on the control that was changed.
  const again = event.target.id === "settings-keep" ? "#settings-keep" : event.target.type === "radio" ? '#colour-maps input[name="colour_map"]:checked' : FLIP;
  document.querySelector(again)?.focus();
  redrawAll();
  if (state.design) scheduleSave();
}

$("#menu-settings").onclick = () => {
  renderSettings();
  $("#settings-dialog").showModal();
};
$("#settings-close").onclick = () => $("#settings-dialog").close();
closeOnBackdrop($("#settings-dialog"));
$("#settings-dialog").addEventListener("change", settingsChanged);

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
installTerms(GLOSSARY, CONTEXT);
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

for (const name of Object.keys(PANES)) $(`#go-${name}`).onclick = () => showPhase(name);
arrowKeys($(".phases"), (tab) => showPhase(tab.id.replace("go-", "")));
arrowKeys($("#step-run .tabs"), (tab) => selectTab(tab.id.replace("tab-", "")));
arrowKeys($("#goals"), (tab) => selectGoal(tab.id.replace("goal-", "")));
for (const goal of WHERE_GOALS) document.getElementById(`goal-${goal}`).onclick = () => selectGoal(goal);
// Each goal's panel, and the dials, come and go with the study and its results: the card follows them.
const goalsFollow = new MutationObserver(renderWhere);
for (const id of [...WHERE_GOALS, "what-if"]) goalsFollow.observe(document.getElementById(id), { attributes: true, attributeFilter: ["hidden"] });

$("#new-study").onclick = () => {
  newStudy();
  $("#welcome").hidden = true; // asked for by name: no longer a first look
  toFirstQuestion(); // the entry pressed has gone with its menu
};
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
    // The question is read back as it is typed. Were it to change only when the box is left, its line could come or
    // grow at that moment and move the page under the press that left the box: Safari then sends that press nowhere.
    renderQuestion();
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
  box(addFactor(categoricalFactor("", START_LEVELS[CATEGORICAL])), "name").focus();
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
$("#field").addEventListener("change", () => {
  keepField($("#field").value);
  showLikeYours();
});

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
for (const id of ["#design-type", "#n_center_points", "#n_replicates"]) $(id).addEventListener("input", markLevelRuns);
$("#generate").onclick = async () => {
  // While practising, the new runs are simulated from the same process, noise and seed, so the true process
  // and the analysis follow the plan as it now is.
  if (!(await mayReplaceRunSheet())) return;
  if ((await generate()) && state.practising) await simulateResults();
};
$("#replace-keep").onclick = () => answerReplace?.(false);
$("#replace-go").onclick = () => answerReplace?.(true);
$("#replace-dialog").addEventListener("close", () => answerReplace?.(false));
$("#remake").onclick = $("#generate").onclick;
$("#practice-round").onclick = practiceRound;
// The card's boxes are the run sheet's: a result typed in one is typed in the sheet's box for that run.
$("#bench-results").oninput = (e) => {
  const sheet = $("#design-table").querySelector(`input[data-index="${benchAt}"][data-response="${CSS.escape(e.target.dataset.bench)}"]`);
  if (!sheet) return;
  sheet.value = e.target.value;
  sheet.dispatchEvent(new Event("input", { bubbles: true }));
};
$("#bench-prev").onclick = () => benchStep(-1);
$("#bench-next").onclick = () => benchStep(1);
$("#bench-failed").onchange = (e) => setFailed(benchAt, e.target.checked);
{
  // Open at first on a phone, where the sheet is too wide to work from at a bench; then as the reader left it.
  let kept = null;
  try {
    kept = localStorage.getItem(BENCH_KEY);
  } catch {
    // no storage: open on a phone, closed on a desk
  }
  $("#bench").open = kept === null ? matchMedia("(max-width: 600px)").matches : kept === "open";
  $("#bench").ontoggle = () => {
    try {
      localStorage.setItem(BENCH_KEY, $("#bench").open ? "open" : "closed");
    } catch {
      // the choice lasts for this visit
    }
  };
}
$("#practice-over").onclick = endPracticeRound;
$("#restore-plan").onclick = () => {
  setForm(state.design.spec);
  state.planKey = planKey();
  markStale();
  renderStudyBar();
};
$("#search").onclick = () => findCandidates({ search: true });
$("#compare").onclick = showComparison;
$("#compare-close").onclick = () => $("#compare-dialog").close();
closeOnBackdrop($("#compare-dialog"));
$("#library-close").onclick = () => $("#library").close();
closeOnBackdrop($("#library"));

/**
 * "Studies", in the bar beside the button for a new study, opens a list: the ways in (a new study, the studies kept here, the worked example, a
 * table of the reader's own, the catalog), then what is done with the study on the page. It is a list of buttons
 * under a button, which Tab walks. A choice, Escape, a press elsewhere and Tab out of the list each close it.
 */
function studyMenu(open) {
  const menu = $("#study-menu");
  menu.hidden = !open;
  // The list is as wide as its longest line and starts under its button; where that runs past the screen's edge
  // it is moved left by as much. On a phone the stylesheet gives it the screen's width, so nothing is moved.
  menu.style.left = "";
  const past = open ? menu.getBoundingClientRect().right - (document.documentElement.clientWidth - 8) : 0;
  if (past > 0) menu.style.left = `${-past}px`;
  $("#study-button").setAttribute("aria-expanded", String(open));
  // A question with no run sheet yet has nothing to share and no report.
  if (open) for (const id of ["#menu-share", "#menu-report"]) $(id).disabled = !state.design;
}
$("#study-button").onclick = () => studyMenu($("#study-menu").hidden);
// A choice closes the list and gives the keyboard back to the study's name, before the choice is acted on: what
// it opens, such as Your studies, then hands the keyboard back there when it closes, not to an entry now hidden.
$("#study-menu").addEventListener(
  "click",
  (e) => {
    if (!e.target.closest("button, a")) return;
    studyMenu(false);
    $("#study-button").focus();
  },
  true,
);
document.addEventListener("click", (e) => {
  if (!e.target.closest("#study-menu, #study-button")) studyMenu(false);
});
$(".study").addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || $("#study-menu").hidden) return;
  studyMenu(false);
  $("#study-button").focus();
});
// Safari gives a pressed button no focus, so a press on the list is not taken for focus that left it.
$(".study").addEventListener("focusout", (e) => {
  const within = e.relatedTarget?.closest("#study-menu, #study-button");
  if (e.relatedTarget && !within) studyMenu(false);
});
$("#open-library").onclick = openLibrary;
$("#answer-more").onclick = () => showPart("best");
$("#learn-to-next").onclick = () => showPhase("next");
$("#rounds-upto").onchange = () => renderRounds();
$("#menu-share").onclick = () => showPart("download");
$("#menu-report").onclick = viewReport;
// The welcome of a first visit leads to the blank question under it, and offers the other ways in by the menu's own
// buttons.
$("#welcome-start").onclick = () => {
  $("#welcome").hidden = true;
  $("#response").focus(); // on a touch screen too: the reader asked to start, and the first box is where
};
$("#welcome-example").onclick = () => $("#show-example").click();
$("#welcome-import").onclick = () => $("#import-data").click();
$("#run-example").onclick = runExample;
$("#off-plan-back").onclick = backOnPlan;
// The ladder's pressed step follows the two settings that tell its steps apart, however they are changed.
for (const id of ["#n_center_points", "#resolution"]) for (const kind of ["input", "change"]) $(id).addEventListener(kind, renderLadder);
// A reader sent a link to the worked example lands in its results, far from the menu: the way to a study of their own
// is in the example's own banner.
$("#example-new").onclick = () => $("#new-study").click();
$("#bar-new").onclick = () => $("#new-study").click();
// A press always changes the results: the first simulates them, and one more draws new noise for the same
// process. With the seed as it was, a second press gave the same results, and looked as if it did nothing.
$("#simulate").onclick = () => {
  if (someSimulated()) seed += 1;
  simulateResults();
};
$("#truth-toggle").onclick = () => showTruth($("#truth").hidden);
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
// A stage named in a sentence is a way to it: "make it in Design", "enter them in Run", "See what the results say".
document.addEventListener("click", (event) => {
  const go = event.target.closest("button[data-go]");
  if (go) showPhase(go.dataset.go);
});
$("#add-confirmation").onclick = () => addConfirmationRuns(state.best?.confirm_at, blendStudy() ? "the best blend" : "the best settings");
$("#add-balance-confirmation").onclick = () => addConfirmationRuns(state.balance?.confirm_at, blendStudy() ? "the balanced blend" : "the balanced settings");
$("#remove-balance-confirmation").onclick = removeConfirmationRuns;
$("#balance-goals").addEventListener("input", goalChanged);
// The what-if panel follows each setting as it moves; a number box shows the setting kept once the reader leaves it.
$("#what-if-settings").addEventListener("input", whatIfChanged);
$("#what-if-settings").addEventListener("change", (e) => {
  const row = e.target.closest("[data-factor]");
  if (e.target.name === "number" && row && state.whatIf) e.target.value = num(state.whatIf.settings[row.dataset.factor]);
});
// A profile is drawn to its dial's width, which the page knows once the panel is on show, and again when it changes.
new ResizeObserver(() => {
  if (state.whatIf && !$("#what-if").hidden) drawProfiles(state.whatIf);
}).observe($("#what-if-settings"));
takeDrag($("#what-if-map-plot"), dragMap);
$("#what-if-back").onclick = whatIfBack;
$("#what-if-keep").addEventListener("submit", keepWhatIf);
$("#what-if-kept-list").addEventListener("click", keptPressed);
$("#balance-reset").onclick = () => {
  state.goals = {};
  scheduleSave();
  rebalance();
};
for (const id of ["#trade-across", "#trade-up", "#trade-colour"]) $(id).addEventListener("change", tradeViewChanged);
$("#trade-limits").addEventListener("input", tradeLimitTyped);
$("#trade-chosen").addEventListener("click", tradeChosenPressed);
$("#trade-nearly").addEventListener("click", tradeNearlyPressed);
$("#target-goals").addEventListener("input", targetTyped);
$("#target-match").addEventListener("change", targetMatched);
$("#target-clear").addEventListener("click", targetCleared);
for (const id of ["#target-across", "#target-up"]) $(id).addEventListener("change", targetPairChanged);
$("#target-confirm").addEventListener("click", () => addConfirmationRuns(state.target.found?.confirm_at, blendStudy() ? "the blend found" : "the settings found"));
$("#target-what-if").addEventListener("click", targetToWhatIf);
for (const id of ["#solve-factor", "#solve-response"]) $(id).addEventListener("change", solveFactor);
$("#solve-out").addEventListener("click", solvedToWhatIf);
$("#window-limits").addEventListener("input", windowTyped);
$("#window-clear").addEventListener("click", windowCleared);
for (const id of ["#window-across", "#window-up"]) $(id).addEventListener("change", windowPairChanged);
$("#window-confirm").addEventListener("click", () => addConfirmationRuns(state.window.found?.surest?.confirm_at, blendStudy() ? "the surest blend" : "the surest settings"));
$("#window-what-if").addEventListener("click", windowToWhatIf);
$("#remove-confirmation").onclick = removeConfirmationRuns;
$("#download").onclick = downloadWorkbook;
$("#csv").onclick = downloadCsv;
$("#curve-use").onclick = useCurves;
$("#curve-target-find").onclick = findTarget;
$("#curve-clear").onclick = () => setCurve(null, null);
for (const id of ["#curve-label", "#curve-units", "#curve-along", "#curve-along-units"]) $(id).onchange = curveWordsChanged;
$("#notebook").onclick = downloadNotebook;
$("#report").onclick = downloadReport;
$("#learn-report-button").onclick = downloadReport;
$("#report-view").onclick = viewReport;
$("#learn-report-view").onclick = viewReport;
$("#share").onclick = shareLink;
$("#model").onchange = () => {
  if ($("#model").value === CHOSEN_TERMS && state.design) startTermList();
  else modelBefore = $("#model").value;
  TERMS.said.textContent = "";
  showTerms();
  if (state.shown) state.models[state.shown] = modelChoice(); // the model of the response Learn shows
  analyseRuns();
  scheduleSave();
};
$("#transform").onchange = () => {
  if (!state.design) return;
  setTransform(state.shown ?? state.design.spec.response, $("#transform").value);
  analyseRuns();
  scheduleSave();
};
$("#box-cox-apply").onclick = () => {
  $("#transform").value = $("#box-cox-apply").dataset.transform;
  $("#transform").onchange();
};
TERMS.designed.onclick = () => {
  models.choose("");
  $("#model").onchange();
};
TERMS.clear.onclick = keepClearTerms;
TERMS.list.onchange = (e) => {
  termPressed(e);
  if (state.shown) state.models[state.shown] = modelChoice();
  scheduleAnalysis(); // a moment after the last tick, as for typed results
  scheduleSave();
};
installImport({ call, open: openImported });
$("#stop").onclick = () => python.stop();
TERMS.first.onchange = showTermAdder;
TERMS.second.onchange = showTermAdder;
TERMS.add.onclick = addNamedTerm;
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
/** A dropzone: a file dropped on `zone`, or chosen with its `input`, goes to `read`. */
function dropZone(zone, input, read) {
  input.onchange = () => {
    const [file] = input.files;
    input.value = ""; // so choosing the same file again reads it again
    if (file) read(file);
  };
  zone.ondragover = (e) => {
    e.preventDefault();
    zone.classList.add("over");
  };
  zone.ondragleave = () => zone.classList.remove("over");
  zone.ondrop = (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    if (e.dataTransfer.files[0]) read(e.dataTransfer.files[0]);
  };
}
dropZone($("#drop"), $("#upload"), readUpload); // Run: a filled-in workbook
dropZone($("#study-drop"), $("#study-upload"), openStudyFile); // Your studies: a study file
// A file dropped anywhere else does nothing, rather than replace the page with the file.
for (const type of ["dragover", "drop"]) {
  window.addEventListener(type, (e) => {
    if (e.defaultPrevented || !e.dataTransfer?.types.includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "none";
  });
}
$("#download-all").onclick = downloadAll;
// A reader who closes the tab, reloads it, or puts the phone away, straight after typing a result or a brief keeps it.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "hidden") return;
  flushSave();
  flushDraft();
});
installLimits({ call, factors: readFactors, changed: limitsChanged, runs: picturedRuns });
// A report of a problem: the study goes only when the reader ticks it, with its results, packed as a link carries it.
const feedback = installFeedback({
  context: feedbackContext,
  study: () => packCampaign(shareable(currentCampaign(), true)),
  hasStudy: () => Boolean(state.design),
});
addFactor();
addFactor();
setAim(defaultAim(0));
showResponse();
showScenarioHint();
refreshDesign();
renderStudyBar();
// The build's manifest names the Plotly.js the charts are drawn with, and says whether a copy of the page is kept
// on this device for use without a network: one is, of a built app, once Python has started and the worker has
// fetched what a model is fitted with. The browser then has every file of the copy at hand.
const whenLoaded = Promise.all([whenReady, whenWarm]);
whenLoaded.catch(() => {}); // Python did not start: the status line says so
readManifest().then((manifest) => installOffline({ manifest, started: whenLoaded, box: $("#offline") }));
/**
 * Bring a part of the page into view, as a link asks with ?show=: its step is shown, the folds it is in are opened,
 * and it is outlined for a moment. The part may not be there yet, as an analysis Python has still to make, so it is
 * waited for. A reader who has begun to scroll, press or type by then is left where they are.
 */
const SHOW_WAIT = 60_000; // ms: as long as Python takes to start and to fit, on a first visit over a slow line
function showPart(id) {
  const part = id && document.getElementById(id);
  if (!part) return;
  let left = false;
  const leave = () => {
    left = true;
  };
  const acts = ["pointerdown", "keydown", "wheel", "touchstart"];
  for (const act of acts) window.addEventListener(act, leave, { once: true, capture: true, passive: true });
  const started = Date.now();
  // Shown once nothing between it and its step is hidden: the step itself is shown here.
  const there = () => {
    for (let at = part; at && !at.dataset.pane; at = at.parentElement) if (at.hidden) return false;
    return true;
  };
  const look = () => {
    if (left || Date.now() - started > SHOW_WAIT) return;
    // A part inside a goal of the card that answers where to run: that goal is chosen, once it is offered.
    const goal = part.closest("[data-goal]")?.dataset.goal;
    if (goal && goalChosen !== goal) selectGoal(goal);
    if (part.matches("dialog")) {
      // Your studies: shown over the page, which gives it the keyboard, and outlined as any part is.
      if (!part.open) part.showModal();
      part.classList.add("shown-part");
      setTimeout(() => part.classList.remove("shown-part"), 4000);
      return;
    }
    if (!there()) {
      setTimeout(look, 200);
      return;
    }
    const pane = part.closest("[data-pane]")?.dataset.pane;
    if (pane && state.phase !== pane) showPhase(pane);
    for (let fold = part.closest("details"); fold; fold = fold.parentElement.closest("details")) fold.open = true;
    // Under the study's bar, and once more when the step's charts have taken their room.
    const under = () => {
      if (!left) window.scrollBy({ top: part.getBoundingClientRect().top - $("#studybar").offsetHeight - 12, behavior: "instant" });
    };
    under();
    setTimeout(under, 300);
    if (!part.hasAttribute("tabindex")) part.tabIndex = -1;
    part.focus({ preventScroll: true }); // a screen reader goes on from here
    part.classList.add("shown-part");
    setTimeout(() => part.classList.remove("shown-part"), 4000);
  };
  look();
}

// The part a link asks for is read before the study opens, which clears the address of some links. The address
// then loses the word, and the word for where the link was posted: a reload, or a link copied from here, opens the
// study without the jump, and the usage report does not count it as one more visit from that place.
const partAsked = partName(new URLSearchParams(location.search).get(SHOW_PARAM));
// A request for a design the app does not make opens Feedback with the request written, for the reader to send.
const requestAsked = requestWords(new URLSearchParams(location.search).get(REQUEST_PARAM));
const read = [SHOW_PARAM, REF_PARAM, REQUEST_PARAM]; // the words of an address that are read once, then dropped
if (read.some((word) => new URLSearchParams(location.search).has(word))) {
  const address = new URL(location.href);
  for (const word of read) address.searchParams.delete(word);
  history.replaceState(null, "", address);
}
start().then(() => {
  showPart(partAsked);
  if (requestAsked) feedback.open("interface", requestAsked);
});
loadFields();

call("api_catalogue").then((reply) => {
  if (!reply.ok) {
    pythonFailed(reply.error);
    return;
  }
  catalogue = reply.result;
  state.version = catalogue.version ?? null;
  state.runtime = catalogue.runtime ?? null;
  pythonReady = true;
  for (const id of ["#upload", "#simulate"]) $(id).disabled = false;
  refreshDesign();
  scheduleCandidates();
  resolveReady();
});
