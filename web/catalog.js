// The design catalog's page (catalog/index.html, at factori.al/catalog/): it reads catalog/index.json
// once, then filters, counts, maps and lists the designs as the reader chooses, and opens one in
// Factorial from the campaign its detail file holds. The rules are in catalog-logic.js; this module
// draws them. No Python runs here: the build computed every facet, so the page is ready as soon as
// the index arrives. Every address is taken from this module's own, which sits beside the app's
// index.html, so the page works wherever the app is served from.

import { LINK_KEY, packCampaign } from "./campaign.js";
import { correlationFigure, draw, fdsFigure, palette } from "./charts.js";
import {
  DEFAULT_STATE,
  FAMILY_COLOURS,
  MODELS,
  OTHER_COLOUR,
  SORTS,
  applicationLine,
  caseFacts,
  applications,
  checkIndex,
  checkLine,
  compareRefusal,
  correlationMap,
  csvText,
  decodeGlyph,
  errataCounts,
  errataShown,
  factLine,
  familyCounts,
  familyMarks,
  filtersOf,
  fromHash,
  generatorsLine,
  largestSecondOrderR,
  limitInUnits,
  limitLine,
  limitOutcome,
  minimumRuns,
  nextSteps,
  parseLimit,
  labelOf,
  leverageWords,
  levelsText,
  manyLabels,
  projectionLines,
  pythonCode,
  results,
  runLossLine,
  sameDesigns,
  settingColour,
  sharedModel,
  sourceLine,
  switchCount,
  toHash,
  toggled,
  topicCounts,
  topicOrder,
  placesAfter,
  whereParts,
  yearBins,
} from "./catalog-logic.js";
import { compareTable, drawTimeline, element, errataTable, projectionExplorer, table } from "./catalog-views.js";
import { closeOnBackdrop } from "./dialogs.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

/** An address of the app, from this module's own: the catalog page sits one folder below it. */
const app = (path) => new URL(path, import.meta.url).href;

/** How many designs the list shows at first, and adds with each press of "Show more". */
const PAGE = 48;
/** The most places a design's detail lists where the same design is printed or made. */
const TWINS_SHOWN = 15;
/** The most factors the map draws: a design with more is in the list, and the map says how many. */
const MAP_FACTORS = 16;
/** The most factors of a design whose correlation map the detail draws: 12 factors have 90 terms. */
const MAP_TERMS_FACTORS = 12;
/** How many errata the table shows at first. */
const ERRATA_SHOWN = 200;
/** How many applications the filter lists at first, those of the most designs; the rest are a click away. */
const TOPICS_SHOWN = 8;

let catalog = null;
let state = structuredClone({ ...DEFAULT_STATE });
let shown = PAGE;
let plotlyVersion = null;
/** Resolves to the Plotly.js version once the app's manifest is read, or to null when it cannot be. */
let plotlyReady = Promise.resolve(null);
let mapWired = false;
let errata = null;
const shards = new Map();
/** The limit whose outcome each design carries now (limit_outside), and the outcomes by design. */
let judgedLimit = "";
/** The judging of the limit under way, which a detail waits for, so it opens with the limit's outcome. */
let judging = Promise.resolve();
let outcomes = new Map();

// ---------------------------------------------------------------- small pieces of the document
let toastTimer = null;
function toast(text, kind = "ok") {
  const box = $("#toast");
  box.textContent = text;
  box.dataset.state = kind;
  box.hidden = false;
  void box.offsetWidth; // lay it out unshown first, so it fades in rather than appearing
  box.classList.add("shown");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    box.classList.remove("shown");
    toastTimer = setTimeout(() => (box.hidden = true), 300);
  }, 3500);
}

/**
 * Draw a design's coded runs as the book draws a run matrix: a cell per setting, blue at -1, grey at 0,
 * vermillion at +1. A card draws the runs across and the factors down; the detail draws runs down.
 */
function drawGlyph(canvas, rows, { across, width, height }) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const g = canvas.getContext("2d");
  if (!g || !rows.length) return;
  g.scale(ratio, ratio);
  const runs = rows.length;
  const k = rows[0].length;
  const [nx, ny] = across ? [runs, k] : [k, runs];
  const cw = width / nx;
  const ch = height / ny;
  rows.forEach((row, r) =>
    row.forEach((value, f) => {
      g.fillStyle = settingColour(value);
      const [x, y] = across ? [r, f] : [f, r];
      g.fillRect(x * cw, y * ch, Math.max(cw - (cw > 4 ? 1 : 0), 1), Math.max(ch - (ch > 4 ? 1 : 0), 1));
    }),
  );
}

// ---------------------------------------------------------------- the filters beside the list
const numberOrNull = (input) => (input.value.trim() === "" ? null : Math.round(Number(input.value)));

function readForm() {
  state.families = $$("#families input:checked").map((box) => box.value);
  state.topics = $$("#topics input:checked").map((box) => box.value);
  state.runs = [numberOrNull($("#runs-min")), numberOrNull($("#runs-max"))];
  state.factors = [numberOrNull($("#factors-min")), numberOrNull($("#factors-max"))];
  state.years = [numberOrNull($("#year-min")), numberOrNull($("#year-max"))];
  state.model = $("#model").value;
  state.projection = Number($("#projection").value);
  state.projectivity = Number($("#projectivity").value);
  state.categorical = { none: [null, 0], some: [1, null] }[$("#categorical").value] ?? [null, null];
  for (const box of $$("[data-switch]")) state[box.dataset.switch] = box.checked;
  state.addedSince = $("#added").value;
  state.source = $("#source").value;
  state.limit = $("#limit").value.trim();
  state.limitAll = $("#limit-all").checked;
}

function writeForm() {
  $("#query").value = state.query;
  for (const button of $$("#collection [role=radio]")) button.setAttribute("aria-checked", String(button.dataset.value === state.collection));
  for (const box of $$("#families input")) box.checked = state.families.includes(box.value);
  for (const box of $$("#topics input")) box.checked = state.topics.includes(box.value);
  if ($$("#topics-more input:checked").length) $("#topics-more").open = true; // an application chosen is in view
  const put = (selector, value) => ($(selector).value = value ?? "");
  put("#runs-min", state.runs[0]);
  put("#runs-max", state.runs[1]);
  put("#factors-min", state.factors[0]);
  put("#factors-max", state.factors[1]);
  put("#year-min", state.years[0]);
  put("#year-max", state.years[1]);
  $("#model").value = state.model;
  $("#projection").value = String(state.projection);
  $("#projectivity").value = String(state.projectivity);
  const [low, high] = state.categorical;
  $("#categorical").value = high === 0 ? "none" : low !== null && low >= 1 ? "some" : "";
  for (const box of $$("[data-switch]")) box.checked = state[box.dataset.switch];
  $("#added").value = state.addedSince;
  $("#source").value = state.source;
  $("#limit").value = state.limit;
  $("#limit-all").checked = state.limitAll;
  $("#sort").value = state.sort;
}

function buildFacets() {
  const totals = {};
  for (const d of catalog.designs) totals[d.family] = (totals[d.family] ?? 0) + 1;
  const order = Object.keys(catalog.families).filter((f) => totals[f]).sort((a, b) => totals[b] - totals[a]);
  $("#families").replaceChildren(
    ...order.map((family) => {
      const swatch = element("span", { class: "swatch" });
      swatch.style.background = FAMILY_COLOURS[family] ?? OTHER_COLOUR;
      const info = catalog.families[family];
      const label = element(
        "label",
        { class: "facets__family", title: info.reference || "" },
        element("input", { type: "checkbox", value: family }),
        swatch,
        element("span", { text: info.label }),
        element("span", { class: "count", "data-family": family }),
      );
      return label;
    }),
  );
  const topics = topicOrder(catalog).map((topic) => {
    const info = catalog.topics[topic];
    return element(
      "label",
      { class: "facets__family facets__topic", title: info.description || "" },
      element("input", { type: "checkbox", value: topic }),
      element("span", { text: info.label }),
      element("span", { class: "count", "data-topic": topic }),
    );
  });
  const rest = topics.slice(TOPICS_SHOWN);
  $("#topics").replaceChildren(
    ...topics.slice(0, TOPICS_SHOWN),
    rest.length
      ? element(
          "details",
          { class: "facets__more", id: "topics-more" },
          element("summary", { text: `${rest.length} more ${rest.length === 1 ? "application" : "applications"}` }),
          element("div", { class: "facets__list" }, ...rest),
        )
      : null,
  );
  const added = [...new Set(catalog.designs.map((d) => d.added).filter(Boolean))].sort().reverse();
  $("#added").append(...added.map((date) => element("option", { value: date, text: date })));
  $("#sort").replaceChildren(...Object.entries(SORTS).map(([value, text]) => element("option", { value, text })));
  const sources = Object.entries(catalog.sources).sort(([, a], [, b]) => a.year - b.year);
  $("#source").append(...sources.map(([key, source]) => element("option", { value: key, text: `${source.short} (${source.year})` })));
}

function renderCounts() {
  const counts = familyCounts(catalog, state);
  for (const span of $$("#families .count")) {
    const n = counts[span.dataset.family] ?? 0;
    span.textContent = String(n);
    span.closest("label").classList.toggle("empty", n === 0);
  }
  const topics = topicCounts(catalog, state);
  for (const span of $$("#topics .count")) {
    const n = topics[span.dataset.topic] ?? 0;
    span.textContent = String(n);
    span.closest("label").classList.toggle("empty", n === 0);
  }
  for (const box of $$("[data-switch]")) {
    $(".count", box.closest("label")).textContent = box.checked ? "" : String(switchCount(catalog, state, box.dataset.switch));
  }
}

function renderTimeline() {
  const bins = yearBins(catalog, state);
  const marks = familyMarks(catalog, bins);
  drawTimeline($("#year-bars"), bins, marks, state.years);
  $("#timeline-marks").replaceChildren(
    ...marks.map((mark) => {
      const swatch = element("span", { class: "swatch" });
      swatch.style.background = FAMILY_COLOURS[mark.family] ?? OTHER_COLOUR;
      return element("li", {}, swatch, `${mark.label} ${mark.year}`);
    }),
  );
}

// ---------------------------------------------------------------- the list
function card(design) {
  const canvas = element("canvas", { class: "glyph", "aria-hidden": "true" });
  const rows = decodeGlyph(design.glyph, design.n_factors);
  if (rows.length) drawGlyph(canvas, rows, { across: true, width: 132, height: Math.min(52, Math.max(18, 6 * design.n_factors)) });
  else canvas.classList.add("glyph--none");
  const checks = checkLine(design.checks);
  const badge = design.collection === "generated" ? element("span", { class: "badge", text: "generated" }) : null;
  const data = design.has_responses ? element("span", { class: "badge badge--data", text: "with results" }) : null;
  const robust = design.survives_run_loss ? element("span", { class: "badge", text: "survives a lost run" }) : null;
  const projects = design.quadratic_projection >= 2 && design.quadratic_projection < design.n_numeric
    ? element("span", { class: "badge", text: `any ${design.quadratic_projection} factors: quadratic` })
    : null;
  const twins = sameDesigns(catalog, design).length;
  const elsewhere = twins ? element("span", { class: "badge", text: `same design in ${twins} more ${twins === 1 ? "place" : "places"}` }) : null;
  const gap = design.collection === "generated" && !design.in_source ? element("span", { class: "badge", text: "no source prints it" }) : null;
  const limited = filtersOf(state).limit && design.limit_outside > 0
    ? element("span", { class: "badge badge--warn", text: `${design.limit_outside} of ${design.n_runs} runs outside the limit` })
    : null;
  return element(
    "li",
    { class: "result" },
    canvas,
    element(
      "div",
      { class: "result__body" },
      element(
        "h3",
        {},
        element("button", { type: "button", class: "link result__title", "data-id": design.id, text: design.title }),
        badge,
        data,
        robust,
        projects,
        elsewhere,
        gap,
        limited,
        ...applications(design, catalog).map((name) => element("span", { class: "badge badge--topic", text: name })),
      ),
      element("p", { class: "result__facts", text: factLine(design, catalog) }),
      element("p", { class: "result__source", text: [sourceLine(design, catalog), checks].filter(Boolean).join(" · ") }),
    ),
    element(
      "div",
      { class: "result__actions" },
      design.open
        ? element("button", { type: "button", class: "button primary", "data-open": design.id, text: "Open in Factorial" })
        : element("button", { type: "button", class: "button", "data-id": design.id, text: "Details" }),
      design.compared ? compareButton(design) : null,
    ),
  );
}

/** The button that puts a design in the comparison, or takes it out: pressed while the design is in it. */
function compareButton(design) {
  const chosen = state.compare.includes(design.id);
  return element("button", {
    type: "button",
    class: "button compare-toggle",
    "data-compare": design.id,
    "aria-pressed": String(chosen),
    text: chosen ? "In the comparison" : "Compare",
  });
}

function renderList(list) {
  $("#results").replaceChildren(...list.slice(0, shown).map(card));
  const left = list.length - shown;
  $("#more").hidden = left <= 0;
  $("#more").textContent = `Show ${Math.min(PAGE, left)} more (${left} not shown)`;
}

function render() {
  const limit = parseLimit(state.limit);
  $("#limit-error").hidden = !limit.error;
  $("#limit-error").textContent = limit.error ? `${limit.error[0].toUpperCase()}${limit.error.slice(1)}.` : "";
  if (limit.ok && judgedLimit !== limit.text) {
    $("#count").textContent = "Reading every design's runs to judge the limit...";
    judging = judgeLimit(limit);
    judging.then(render, (err) => toast(`The limit could not be judged: ${err.message}.`, "error"));
    return;
  }
  const list = results(catalog, state);
  const published = list.filter((d) => d.collection === "published").length;
  const sentence = `${list.length} of ${catalog.designs.length} designs`;
  $("#count").textContent = list.length ? `${sentence}: ${published} published, ${list.length - published} generated` : `${sentence}: change a filter or the search to see more`;
  renderCounts();
  renderTimeline();
  renderList(list);
  renderTray();
  drawMap(list);
}

function update({ keepPage = false } = {}) {
  if (!keepPage) shown = PAGE;
  history.replaceState(null, "", toHash(state) || location.pathname + location.search);
  render();
}

// ---------------------------------------------------------------- limits on the settings
/**
 * Judge every design's runs against a limit, reading the detail files the first time a limit is set,
 * and mark each design with how many of its runs are outside it (limit_outside), or null when the
 * limit cannot be judged on it.
 */
async function judgeLimit(limit) {
  const names = [...new Set(catalog.designs.map((d) => d.detail))];
  const held = await Promise.all(names.map((name) => shard(name)));
  const details = Object.assign({}, ...held.map((h) => h.designs));
  outcomes = new Map();
  for (const design of catalog.designs) {
    const detail = details[design.id];
    const kinds = (detail?.factors ?? []).map((f) => f.type ?? "numeric");
    const outcome = limitOutcome(limit, detail?.coded_runs ?? [], kinds);
    outcomes.set(design.id, outcome);
    design.limit_outside = outcome.judged ? outcome.outside.length : null;
  }
  judgedLimit = limit.text;
}

/** The campaign that opens a design in Factorial with the limit on its settings, written in its own units. */
function limitedCampaign(design, detail, limit) {
  const outcome = outcomes.get(design.id);
  const campaign = structuredClone(detail.campaign);
  const factors = campaign.design.spec.factors;
  const expression = limitInUnits(limit, factors);
  const note = `In coded units: ${limit.text}`; // a limit's note holds 80 characters at most (campaign.js)
  campaign.design.spec.constraints = [{ expression, ...(note.length <= 80 ? { note } : {}) }];
  const outside = outcome?.judged ? outcome.outside.map((i) => String(i + 1)) : [];
  const which = outside.length === 1 ? `Run ${outside[0]} is` : `Runs ${outside.slice(0, -1).join(", ")} and ${outside.at(-1)} are`;
  campaign.design.facts ??= { center_runs: 0, notes: [] };
  campaign.design.facts.notes = [
    ...campaign.design.facts.notes,
    outside.length
      ? `${which} outside your limits on the settings: choose the D-optimal or I-optimal design to make runs inside them.`
      : "Every run is inside your limits on the settings.",
  ];
  return campaign;
}

/** Whether a design can open in Factorial with the limit: it opens, its factors are numeric, and the limit fits. */
function opensWithLimit(design, detail, limit) {
  if (!detail.campaign || !limit.ok || !outcomes.get(design.id)?.judged || design.n_categorical) return false;
  return limitInUnits(limit, detail.campaign.design.spec.factors).length <= 500;
}

// ---------------------------------------------------------------- designs side by side
const comparedDesigns = () => state.compare.map((id) => catalog.designs.find((d) => d.id === id)).filter(Boolean);

function renderTray() {
  const chosen = comparedDesigns();
  $("#tray").hidden = !chosen.length;
  $("#tray-text").textContent = chosen.length < 2
    ? "Choose another design with the same factors to compare it with."
    : `${chosen.length} designs of ${chosen[0].n_factors} factors`;
  $("#tray-list").replaceChildren(
    ...chosen.map((d) =>
      element(
        "li",
        {},
        element("span", { text: d.title }),
        element("button", { type: "button", class: "link", "data-compare": d.id, "aria-label": `Take ${d.title} out of the comparison`, text: "✕" }),
      ),
    ),
  );
  $("#tray-open").disabled = chosen.length < 2;
}

function toggleCompare(id) {
  if (!state.compare.includes(id)) {
    const refusal = compareRefusal(catalog, state.compare, id);
    if (refusal) return toast(`This design cannot join the comparison: ${refusal}.`, "error");
  }
  state.compare = toggled(state.compare, id);
  update({ keepPage: true });
  for (const button of $$(`#detail-body [data-compare="${id}"]`)) button.replaceWith(compareButton(catalog.designs.find((d) => d.id === id)));
}

/** A label for each design compared, with its runs, numbered when two would read the same. */
function compareLabels(designs) {
  const labels = [];
  for (const d of designs) {
    const base = `${d.title} [n=${d.n_runs}]`;
    let label = base;
    for (let i = 2; labels.includes(label); i++) label = `${base} (${i})`;
    labels.push(label);
  }
  return labels;
}

async function openCompare() {
  const chosen = comparedDesigns();
  if (chosen.length < 2) return;
  try {
    const scored = await Promise.all(chosen.map(async (d) => (await detailOf(d.id)).detail.compare));
    let model = sharedModel(scored);
    const labels = compareLabels(chosen);
    const select = element(
      "select",
      { id: "compare-model", "aria-label": "The model the designs are compared on" },
      ...Object.entries(MODELS).map(([value, text]) => element("option", { value, text: `Model: ${text}` })),
    );
    select.value = model;
    const holder = element("div", { class: "compare__table" });
    const plot = element("div", { class: "compare__fds", "aria-hidden": "true" });
    const show = () => {
      holder.replaceChildren(compareTable(catalog.measures, chosen.map((d, i) => ({ ...d, title: labels[i] })), scored, model));
      const curves = {};
      chosen.forEach((d, i) => {
        const curve = scored[i]?.[model]?.fds;
        if (curve) curves[labels[i]] = { fraction: curve.map((_, j) => j / (curve.length - 1)), variance: curve };
      });
      plot.hidden = !Object.keys(curves).length;
      plotlyReady.then((version) => {
        if (!version) plot.hidden = true;
        else if (!plot.hidden) draw(plot, () => fdsFigure(curves), version, 320);
      });
    };
    select.addEventListener("change", () => {
      model = select.value;
      show();
    });
    const one = [];
    chosen.forEach((d, i) =>
      chosen.slice(i + 1).forEach((e, j) => {
        if (d.design_class && d.design_class === e.design_class) one.push(`${labels[i]} and ${labels[i + 1 + j]} are one design, with their runs, factors or signs in another order: their measures are the same.`);
      }),
    );
    $("#compare-body").replaceChildren(
      element("p", {
        class: "hint",
        text: "Each design is scored at its own size for the model chosen, over the cube from -1 to +1 in every factor, as compare_designs in process-improve scores it. The arrow says which direction a measure prefers; no single design is best on every row. Power is for a coefficient of one noise standard deviation, tested at the 5% level.",
      }),
      select,
      holder,
      ...one.map((text) => element("p", { class: "hint", text })),
      element("h3", { text: "Fraction of design space" }),
      element("p", {
        class: "hint",
        text: "Each curve is the prediction variance, in units of the noise variance, that a fraction of the cube does not exceed, read at 4096 points drawn uniformly from it. A curve that lies lower predicts with less variance over more of the cube.",
      }),
      plot,
    );
    show();
    if (!$("#compare").open) $("#compare").showModal();
    state.view = "compare";
    history.replaceState(null, "", toHash(state));
  } catch (err) {
    toast(`The designs could not be compared: ${err.message}.`, "error");
  }
}

// ---------------------------------------------------------------- the errata
function loadErrata() {
  errata ??= fetch(app("catalog/errata.json"), { cache: "no-cache" })
    .then((res) => {
      if (!res.ok) throw new Error(`the server answered ${res.status}`);
      return res.json();
    })
    .then((data) => data.errata)
    .catch((err) => {
      errata = null;
      throw err;
    });
  return errata;
}

async function openErrata() {
  try {
    const all = await loadErrata();
    const counts = errataCounts(all);
    const source = element(
      "select",
      { "aria-label": "The source", id: "errata-source" },
      element("option", { value: "", text: `Every source: ${all.length}` }),
      ...Object.keys(counts).map((key) => {
        const s = catalog.sources[key];
        return element("option", { value: key, text: `${s ? `${s.short} (${s.year})` : key}: ${counts[key]}` });
      }),
    );
    if (counts[state.source]) source.value = state.source;
    const words = element("input", { type: "search", id: "errata-words", placeholder: "Words to find: a page, a quantity, a reason", "aria-label": "Words to find" });
    const rounding = element("input", { type: "checkbox", id: "errata-rounding", checked: true });
    const count = element("p", { class: "hint", role: "status", id: "errata-count" });
    const holder = element("div");
    let limit = ERRATA_SHOWN;
    const more = element("button", { type: "button", class: "button", text: "Show them all" });
    const show = () => {
      const shown = errataShown(all, { source: source.value, words: words.value, rounding: rounding.checked });
      count.textContent = `${shown.length} of ${all.length} printed numbers`;
      holder.replaceChildren(errataTable(shown.slice(0, limit), catalog));
      more.hidden = shown.length <= limit;
    };
    for (const control of [source, words, rounding]) control.addEventListener("input", () => {
      limit = ERRATA_SHOWN;
      show();
    });
    more.addEventListener("click", () => {
      limit = Infinity;
      show();
    });
    $("#errata-body").replaceChildren(
      element("p", {
        text: "Every number a source prints that process-improve recomputes and does not reproduce, with the reason the catalogue records: the source's rounding, a misprint, or a method that differs from the library's. The catalogue's tests require every other recomputed number to agree with its source.",
      }),
      element("div", { class: "errata__controls" }, source, words, element("label", { class: "facets__switch" }, rounding, " Include the reasons that mention rounding")),
      count,
      holder,
      more,
    );
    show();
    if (!$("#errata").open) $("#errata").showModal();
    state.view = "errata";
    history.replaceState(null, "", toHash(state));
  } catch (err) {
    toast(`The errata could not be read: ${err.message}.`, "error");
  }
}

// ---------------------------------------------------------------- the map
/** A design's place across, nudged by a fixed amount of its own, so designs of one size do not sit on one point. */
function nudge(id) {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 9973;
  return (h / 9973 - 0.5) * 0.56;
}

function mapFigure(list) {
  const colours = palette();
  const placed = (d) => d.n_runs && d.n_factors;
  const everyone = catalog.designs.filter((d) => placed(d) && (state.collection === "both" || d.collection === state.collection));
  const kept = list.filter(placed);
  const most = MAP_FACTORS;
  const beyond = kept.filter((d) => d.n_factors > most).length;
  const background = {
    type: "scatter",
    mode: "markers",
    name: "Filtered out",
    x: everyone.map((d) => Math.min(d.n_factors, most + 0.5) + nudge(d.id)),
    y: everyone.map((d) => d.n_runs),
    marker: { size: 5, color: colours.line },
    hoverinfo: "skip",
    showlegend: false,
  };
  const byFamily = new Map();
  for (const d of kept) byFamily.set(d.family, [...(byFamily.get(d.family) ?? []), d]);
  const marks = [...byFamily].map(([family, designs]) => ({
    type: "scatter",
    mode: "markers",
    name: catalog.families[family]?.label ?? family,
    x: designs.map((d) => d.n_factors + nudge(d.id)),
    y: designs.map((d) => d.n_runs),
    customdata: designs.map((d) => d.id),
    text: designs.map((d) => `${d.title}<br>${factLine(d, catalog)}`),
    hovertemplate: "%{text}<extra></extra>",
    showlegend: false,
    marker: {
      size: 9,
      color: FAMILY_COLOURS[family] ?? OTHER_COLOUR,
      symbol: designs.map((d) => (d.collection === "generated" ? "diamond-open" : d.has_responses ? "circle" : "circle-open")),
      line: { width: 1.6, color: FAMILY_COLOURS[family] ?? OTHER_COLOUR },
    },
  }));
  const ks = Array.from({ length: most }, (_, i) => i + 1);
  const dashes = { main_effects: "dot", interactions: "dash", quadratic: "solid" };
  const lines = Object.keys(MODELS).map((model) => ({
    type: "scatter",
    mode: "lines",
    name: `Fewest runs: ${MODELS[model]}`,
    x: ks,
    y: ks.map((k) => minimumRuns(k, model)),
    line: { color: colours.muted, width: 1.2, dash: dashes[model], shape: "spline" },
    hoverinfo: "skip",
  }));
  const key = (name, symbol) => ({
    type: "scatter",
    mode: "markers",
    name,
    x: [null],
    y: [null],
    marker: { size: 9, symbol, color: colours.muted, line: { width: 1.6, color: colours.muted } },
    hoverinfo: "skip",
  });
  const keys = [key("Printed with results", "circle"), key("Published design", "circle-open"), key("Generated", "diamond-open")];
  const note = beyond
    ? [{ text: `${beyond} with more than ${most} factors: in the list`, xref: "paper", yref: "paper", x: 1, y: 0.02, xanchor: "right", yanchor: "bottom", showarrow: false, font: { size: 11, color: colours.muted } }]
    : [];
  return {
    data: [background, ...lines, ...keys, ...marks],
    layout: {
      dragmode: "select",
      hovermode: "closest",
      xaxis: { title: { text: "Factors" }, dtick: 1, range: [0.4, most + 0.6] },
      yaxis: { title: { text: "Runs" }, type: "log", tickvals: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 8 },
      annotations: note,
    },
  };
}

function drawMap(list) {
  if (!plotlyVersion) return;
  const div = $("#map");
  draw(div, () => mapFigure(list), plotlyVersion, 380).then(() => {
    if (mapWired || !div.on) return;
    mapWired = true;
    div.on("plotly_click", (event) => {
      const id = event?.points?.[0]?.customdata;
      if (id) openDetail(id);
    });
    div.on("plotly_selected", (event) => {
      const points = (event?.points ?? []).filter((p) => p.customdata);
      if (!points.length) return;
      const chosen = points.map((p) => catalog.designs.find((d) => d.id === p.customdata));
      const span = (key) => [Math.min(...chosen.map((d) => d[key])), Math.max(...chosen.map((d) => d[key]))];
      state.factors = span("n_factors");
      state.runs = span("n_runs");
      writeForm();
      update();
    });
    // A double click clears the box Plotly drew; the runs and factors it kept go with it.
    div.on("plotly_deselect", () => {
      if ([...state.factors, ...state.runs].every((v) => v === null)) return;
      state.factors = [null, null];
      state.runs = [null, null];
      writeForm();
      update();
    });
  });
}

/**
 * Put the map back as it opened: no box drawn, the whole range in view, and no filter on runs or factors.
 * Plotly keeps a drawn box and a zoom across redraws, so the map is drawn again from nothing.
 */
function resetMap() {
  clearMap();
  state.factors = [null, null];
  state.runs = [null, null];
  writeForm();
  update();
}

/** Drop the map's drawing, with any box and zoom on it, so the next render draws it afresh. */
function clearMap() {
  const div = $("#map");
  if (!window.Plotly || !div.data) return;
  window.Plotly.purge(div);
  mapWired = false;
}

function startMap() {
  plotlyReady = fetch(app("wheels/manifest.json"), { cache: "no-cache" })
    .then((res) => res.json())
    .then((manifest) => {
      if (!manifest.plotly_js) throw new Error("no Plotly.js version");
      plotlyVersion = manifest.plotly_js;
      return plotlyVersion;
    })
    .catch(() => null);
  plotlyReady.then((version) => {
    if (version) drawMap(results(catalog, state));
    else $(".catalog__map").hidden = true; // the list holds every design the map would show
  });
}

// ---------------------------------------------------------------- one design
function shard(name) {
  if (!shards.has(name)) {
    shards.set(
      name,
      fetch(app(`catalog/detail/${encodeURIComponent(name)}.json`), { cache: "no-cache" }).then((res) => {
        if (!res.ok) throw new Error(`the design's file could not be read (${res.status})`);
        return res.json();
      }),
    );
  }
  return shards.get(name);
}

async function detailOf(id) {
  const design = catalog.designs.find((d) => d.id === id);
  if (!design) throw new Error(`the catalog has no design ${id}`);
  const held = await shard(design.detail);
  return { design, detail: held.designs[id] };
}

async function openInFactorial(id, { limited = false } = {}) {
  try {
    const { design, detail } = await detailOf(id);
    if (!detail.campaign) throw new Error(detail.closed || "the page does not open this design");
    const campaign = limited ? limitedCampaign(design, detail, parseLimit(state.limit)) : detail.campaign;
    location.assign(app(`./${LINK_KEY}${await packCampaign(campaign)}`));
  } catch (err) {
    toast(`This design could not be opened: ${err.message}.`, "error");
  }
}

function copy(text, what) {
  navigator.clipboard.writeText(text).then(
    () => toast(`${what} copied.`),
    () => toast(`${what} could not be copied: select it and copy it yourself.`, "error"),
  );
}

const fmt = (v) => (v === null || v === undefined ? "" : typeof v === "number" ? String(Math.round(v * 1000) / 1000) : String(v));

/**
 * "Printed in" and where: its section, pages, tables and figures. Where the catalog knows the place's address
 * (a page of the book, at the section's anchor, or of the handbook), the first part links to it, so a reader
 * lands on the section rather than on the source's home page.
 */
function printedIn(detail) {
  const [first, ...rest] = whereParts(detail.where);
  const url = detail.where_url;
  if (!first && !url) return null;
  const place = url ? element("a", { href: url, target: "_blank", rel: "noopener", text: first ?? "the source's page" }) : first;
  return element("p", { class: "detail__where" }, "Printed in ", place, placesAfter(rest));
}

/** The longest an entry's notes may run and still be shown open; longer ones fold under a summary. */
const NOTES_SHOWN = 240;

/**
 * An entry's notes: the source's words for its design, and what the catalogue adds of it. Short notes are shown
 * as they are; long ones, which name the lines of the source each number was read from, fold under a summary,
 * so they are there for a reader who wants them and out of the way of one who reads on.
 */
function entryNotes(detail) {
  const notes = [detail.details, detail.notes].filter(Boolean).map((text) => element("p", { class: "hint", text }));
  if ([detail.details, detail.notes].join(" ").length <= NOTES_SHOWN) return notes;
  return [element("details", { class: "detail__notes" }, element("summary", { text: "Notes on this design, from its source" }), ...notes)];
}

/** One cell of a design's header: its heading, then what it holds. */
const headCell = (heading, ...parts) => element("section", { class: "detail__cell" }, element("h3", { text: heading }), ...parts);

/**
 * The top of a design's detail, the same three cells for every design: the design itself (its size, what it
 * estimates, its application), where it comes from (the source and the place it is printed, or the library), and
 * what the reader can do with it. Each says only what holds for this design; its family's is in designType.
 */
function detailHead(design, detail, source) {
  const facts = element("ul", { class: "detail__list detail__facts" }, ...caseFacts(design).map((text) => element("li", { text })));
  const application = applicationLine(design, catalog);
  const what = headCell("This design", facts, ...(application ? [element("p", { class: "detail__application", text: application })] : []));

  const from = source
    ? [
        element("p", { class: "detail__cite" }, source.url ? element("a", { href: source.url, target: "_blank", rel: "noopener", text: source.cite }) : source.cite),
        printedIn(detail),
      ]
    : [element("p", { text: "Made by process-improve's generate_design, with the call in Python, below." })];
  const where = headCell("Where it comes from", ...from.filter(Boolean));

  const actions = element("div", { class: "actions detail__actions" });
  const said = [];
  const limit = parseLimit(state.limit);
  if (detail.campaign) {
    actions.append(element("button", { type: "button", class: "button primary", "data-open": design.id, text: "Open in Factorial" }));
    if (opensWithLimit(design, detail, limit)) {
      actions.append(element("button", { type: "button", class: "button", "data-open-limited": design.id, text: "Open in Factorial with this limit" }));
    }
    if (detail.opens_as === "given") {
      const whose = detail.runs_from
        ? `the runs the library makes from the source's generators, ${generatorsLine(detail.runs_from)}`
        : source
          ? "the source's runs as it prints them"
          : "these runs as the library made them";
      said.push(element("p", { class: "hint", text: `Factorial opens ${whose}, in their order: ${detail.unmade}.` }));
    }
  } else {
    const what = design.kind === "facts" || design.kind === "sample" ? "entry" : "design"; // facts, or one sample: no design
    said.push(element("p", { class: "hint", text: `Factorial does not open this ${what}: ${detail.closed}.` }));
  }
  if (design.compared) actions.append(compareButton(design)); // append() writes a null out as the text "null"
  actions.append(
    element("button", { type: "button", class: "button", "data-copy-link": design.id, text: "Copy a link to this design" }),
    element("button", { type: "button", class: "button", "data-csv": design.id, text: "Download the runs (CSV)" }),
    element("button", { type: "button", class: "button", "data-show-code": design.id, text: "Show the Python" }),
  );
  return element("div", { class: "detail__head" }, what, where, headCell("Use it", actions, ...said));
}

/**
 * What a design's family is and where it was first published, folded under its name: it holds for every design of
 * the family, so it is there for a reader who asks and out of the way of one reading about this design.
 */
function designType(design) {
  const family = catalog.families[design.family];
  const label = family?.label ?? design.family;
  const about = [
    family?.description && element("p", { text: family.description }),
    family?.year && element("p", { class: "hint", text: `${label} designs were first published by ${family.reference}.` }),
  ].filter(Boolean);
  if (!about.length) return element("p", { class: "hint detail__family", text: `Design type: ${label}.` });
  return element("details", { class: "detail__family" }, element("summary", { text: `About this design type: ${label}` }), ...about);
}

function detailBody(design, detail) {
  const source = catalog.sources[design.source];
  const body = [detailHead(design, detail, source), designType(design), ...entryNotes(detail)];
  const limit = parseLimit(state.limit);
  const outcome = limit.ok ? outcomes.get(design.id) : null;
  if (outcome) {
    body.push(element("p", { class: outcome.judged && outcome.outside.length ? "detail__warn" : "hint", text: limitLine(limit, outcome, design.n_runs) }));
    if (opensWithLimit(design, detail, limit) && outcome.outside.length) {
      body.push(element("p", { class: "hint", text: "Factorial opens it with the limit in its factors' own units, marks the runs outside it, and makes a D-optimal or I-optimal design inside it on request." }));
    }
  }

  body.push(...measures(design, detail), ...projections(design, detail), ...steps(design));

  if (detail.coded_runs?.length) body.push(...runsTable(detail, outcome));
  if (detail.factors?.length) {
    body.push(
      element("h3", { text: "Factors" }),
      table(
        ["Name", "Levels", "Units", "Type", "Description"],
        detail.factors.map((f) => [f.name, levelsText(f), f.units ?? "", f.type ?? "", f.description ?? ""]),
        "data factors-table",
      ),
    );
  }
  const code = pythonCode(design, detail);
  body.push(
    element("h3", { id: "detail-python", text: "In Python" }),
    element("pre", { class: "detail__code" }, element("code", { text: code })),
    element("button", { type: "button", class: "link", "data-copy-code": design.id, text: "Copy the code" }),
  );
  if (source?.rights) body.push(element("p", { class: "hint detail__rights", text: source.rights }));
  return body;
}

/** The most runs a design's detail lists. */
const RUNS_SHOWN = 400;

/** A run's coded setting, tinted as the run matrix colours it: blue at -1, grey at 0, vermillion at +1. */
const settingCell = (value) =>
  element("td", { class: "setting", style: value === null || value === undefined ? null : `--tint: ${settingColour(value)}` }, fmt(value));

/** A run's leverage, as a bar from 0 to 1 beside its value; vermillion at 1, where the run cannot be lost. */
function leverageCell(h) {
  const bar = element("span", { class: "lev__bar", style: `width: ${Math.round(100 * Math.min(1, Math.max(0, h)))}%` });
  return element(
    "td",
    { class: h >= 0.999999 ? "lev lev--one" : "lev" },
    element("span", { class: "lev__track", "aria-hidden": "true" }, bar),
    h.toFixed(2),
  );
}

/**
 * The runs, one row each, in one table: each run's block as the source prints it, each coded setting tinted by its
 * value, a categorical factor of more than two labels by its label instead, the results the source prints, and each
 * run's leverage when the runs' leverages differ. Colour, settings and leverage share the row, so they line up and
 * scroll together.
 */
function runsTable(detail, outcome) {
  const factors = detail.factors ?? [];
  const blocks = detail.blocks ?? null;
  const responses = Object.keys(detail.response_values ?? {});
  const words = leverageWords(detail.leverage);
  const leverage = words.text && !words.even ? detail.leverage : null;
  const labelled = factors.map(manyLabels);
  const rows = detail.coded_runs.slice(0, RUNS_SHOWN).map((row, i) => [
    String(i + 1),
    ...(blocks ? [String(blocks[i] ?? "")] : []),
    ...row.map((v, j) => (labelled[j] ? (labelOf(factors[j], v) ?? fmt(v)) : settingCell(v))),
    ...responses.map((r) => fmt(detail.response_values[r][i])),
    ...(leverage ? [leverageCell(leverage[i])] : []),
  ]);
  const coded = labelled.some((l) => !l);
  const said = coded ? ["-1 is a factor's low level and +1 its high level, tinted blue at -1, grey at 0 and vermillion at +1."] : [];
  if (detail.runs_from) {
    said.unshift(`The source prints no runs: these are the fraction the library makes from its generators, ${generatorsLine(detail.runs_from)}, in standard order.`);
  }
  if (labelled.some(Boolean)) said.push("A categorical factor of more than two labels shows its label.");
  if (blocks) said.push("Block is the block each run is made in, as the source names it.");
  if (leverage) said.push("Leverage, from 0 to 1, is how much the model leans on a run: at 1, the model cannot be fitted without it.");
  return [
    element("h3", { text: coded ? "Runs, in coded units" : "Runs" }),
    element("p", { class: "hint", text: said.join(" ") }),
    table(
      ["Run", ...(blocks ? ["Block"] : []), ...factors.map((f) => f.name), ...responses, ...(leverage ? ["Leverage"] : [])],
      rows,
      "data runs-table",
      (i) => (outcome?.judged && outcome.outside.includes(i) ? "row--outside" : ""),
    ),
  ];
}

/** What a design withstands: losing a run, and being projected onto fewer factors; and where else it is printed. */
function measures(design, detail) {
  const out = [];
  const loss = runLossLine(design, detail.run_loss_model);
  const projections = projectionLines(design, detail.coded_runs ?? []);
  if (loss || projections.length) out.push(element("h3", { text: "What it withstands" }));
  if (loss) {
    out.push(element("p", { text: loss }));
    const leverage = leverageWords(detail.leverage);
    if (leverage.text) {
      const where = leverage.even ? "" : " Each run's leverage is in the table of runs, below.";
      out.push(element("p", { class: "detail__leverage", text: `${leverage.text}${where}` }));
    }
  }
  out.push(...projections.map((line) => element("p", { text: line })));
  const twins = sameDesigns(catalog, design);
  if (twins.length) {
    out.push(
      element("h3", { text: "The same design elsewhere" }),
      element("p", { class: "hint", text: "Its runs, with the runs, the factors or a factor's two levels in another order." }),
      element(
        "ul",
        { class: "detail__twins" },
        ...twins.slice(0, TWINS_SHOWN).map((d) =>
          element(
            "li",
            {},
            element("button", { type: "button", class: "link", "data-id": d.id, text: d.title }),
            element("span", { class: "hint", text: ` (${sourceLine(d, catalog)})` }),
          ),
        ),
      ),
    );
    if (twins.length > TWINS_SHOWN) out.push(element("p", { class: "hint", text: `And ${twins.length - TWINS_SHOWN} more.` }));
  }
  return out;
}

/** Whether a design's runs are all numeric settings, every one given: what the projections and correlations need. */
function numericRuns(design, detail) {
  const runs = detail.coded_runs ?? [];
  const complete = runs.length && runs.every((row) => row.every((v) => v !== null && v !== undefined));
  return complete && design.n_factors >= 2 && design.n_numeric === design.n_factors ? runs : null;
}

/** The projection explorer, and the correlation map, of a design whose factors are all numeric. */
function projections(design, detail) {
  const runs = numericRuns(design, detail);
  if (!runs) return [];
  const names = (detail.factors ?? []).map((f) => f.name);
  const out = [element("h3", { text: "Its projections" }), projectionExplorer(runs, names)];
  if (design.n_factors <= MAP_TERMS_FACTORS) {
    const map = correlationMap(runs, names);
    const most = largestSecondOrderR(map);
    out.push(
      element("h3", { text: "Correlations between its terms" }),
      element("p", {
        class: "hint",
        text: "The absolute correlation |r| between the main effects, squares and two-factor interactions, from white at 0 to dark at 1, as compare_designs reads them. A term that does not vary, such as the square of a two-level factor, is blank.",
      }),
    );
    if (most !== null) out.push(element("p", { text: `The largest |r| between two second-order terms is ${most.toFixed(2)}.` }));
    const plot = element("div", { class: "detail__correlations", "aria-hidden": "true" });
    out.push(plot);
    const height = Math.min(820, 160 + 15 * map.terms.length); // room for every term's label
    plotlyReady.then((version) => {
      if (version && plot.isConnected) draw(plot, () => correlationFigure(map.terms, map.matrix, design.n_factors), version, height);
      else plot.remove();
    });
  }
  return out;
}

/** The families a design can lead to, each a link to the catalog filtered to their designs. */
function steps(design) {
  const next = nextSteps(design, catalog);
  if (!next.length) return [];
  return [
    element("h3", { text: "Where it can lead" }),
    element(
      "ul",
      { class: "detail__steps" },
      ...next.map((step) => element("li", {}, element("a", { href: step.hash, text: step.label }), element("span", { class: "hint", text: `: ${step.how}` }))),
    ),
  ];
}

async function openDetail(id) {
  const dialog = $("#detail");
  try {
    await judging.catch(() => {}); // a limit that cannot be judged says so in its own toast
    const { design, detail } = await detailOf(id);
    $("#detail-title").textContent = design.title;
    $("#detail-body").replaceChildren(...detailBody(design, detail));
    if (!dialog.open) dialog.showModal();
    state.design = id;
    history.replaceState(null, "", toHash(state));
  } catch (err) {
    toast(`This design could not be shown: ${err.message}.`, "error");
  }
}

function closeDetail() {
  $("#detail").close();
}

// ---------------------------------------------------------------- the page
function wire() {
  for (const dialog of $$("dialog")) closeOnBackdrop(dialog);
  let typing = null;
  $("#query").addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(() => {
      state.query = $("#query").value;
      update();
    }, 200);
  });
  for (const button of $$(".try")) {
    button.addEventListener("click", () => {
      state.query = button.textContent;
      writeForm();
      update();
    });
  }
  for (const button of $$("#collection [role=radio]")) {
    button.addEventListener("click", () => {
      state.collection = button.dataset.value;
      writeForm();
      update();
    });
  }
  $("#facets").addEventListener("input", (event) => {
    if (event.target.id === "sort") return;
    readForm();
    update();
  });
  $("#facets").addEventListener("change", () => {
    readForm();
    update();
  });
  $("#facets").addEventListener("submit", (event) => event.preventDefault());
  $("#reset").addEventListener("click", (event) => {
    event.preventDefault();
    state = { ...structuredClone({ ...DEFAULT_STATE }), collection: state.collection, sort: state.sort };
    clearMap();
    writeForm();
    update();
  });
  $("#map-reset").addEventListener("click", resetMap);
  $("#sort").addEventListener("change", () => {
    state.sort = $("#sort").value;
    update();
  });
  $("#more").addEventListener("click", () => {
    shown += PAGE;
    update({ keepPage: true });
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest(
      "[data-id], [data-open], [data-open-limited], [data-copy-link], [data-copy-code], [data-show-code], [data-csv], [data-compare], [data-close]",
    );
    if (!target) return;
    const { id, open, openLimited, copyLink, copyCode, showCode, csv, compare, close } = target.dataset;
    if (close) $(`#${close}`).close();
    else if (showCode) $("#detail-python").scrollIntoView({ behavior: "smooth", block: "start" });
    else if (openLimited) openInFactorial(openLimited, { limited: true });
    else if (compare) toggleCompare(compare);
    else if (open) openInFactorial(open);
    else if (copyLink) copy(new URL(`#d=${copyLink}`, location.href).href, "The link");
    else if (copyCode) copy($("#detail-body .detail__code").textContent, "The code");
    else if (csv) downloadCsv(csv);
    else if (id) openDetail(id);
  });
  $("#detail-close").addEventListener("click", closeDetail);
  $("#detail").addEventListener("close", () => {
    state.design = "";
    history.replaceState(null, "", toHash(state) || location.pathname + location.search);
  });
  for (const view of ["compare", "errata"]) {
    $(`#${view}`).addEventListener("close", () => {
      if (state.view === view) state.view = "";
      history.replaceState(null, "", toHash(state) || location.pathname + location.search);
    });
  }
  $("#tray-open").addEventListener("click", openCompare);
  $("#tray-clear").addEventListener("click", () => {
    state.compare = [];
    update({ keepPage: true });
  });
  $("#errata-open").addEventListener("click", openErrata);
  wireTimeline();
  window.addEventListener("hashchange", () => {
    state = fromHash(location.hash, catalog.families, catalog.sources, catalog.topics);
    writeForm();
    render();
    for (const dialog of $$("dialog[open]")) dialog.close();
    openViews();
  });
}

/** Open what the address names: a design's detail, the comparison or the errata. */
function openViews() {
  if (state.view === "compare") openCompare();
  if (state.view === "errata") openErrata();
  if (state.design) openDetail(state.design);
}

/**
 * Let a reader choose years on the timeline: a drag across it keeps the years it spans, and a bar,
 * clicked or chosen with the keyboard, keeps its own.
 */
function wireTimeline() {
  const svg = $("#year-bars");
  let start = null;
  const yearAt = (clientX) => {
    const box = svg.getBoundingClientRect();
    const [first, last] = [Number(svg.dataset.first), Number(svg.dataset.last)];
    return Math.round(first + ((clientX - box.left) / box.width) * (last - first));
  };
  const keep = (years) => {
    state.years = years;
    writeForm();
    update();
  };
  svg.addEventListener("pointerdown", (event) => {
    start = event.clientX;
    svg.setPointerCapture?.(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    const brush = $(".timeline__brush", svg);
    if (start === null || !brush || Math.abs(event.clientX - start) < 4) return;
    const box = svg.getBoundingClientRect();
    const [a, b] = [start, event.clientX].sort((p, q) => p - q);
    const scale = 720 / box.width;
    brush.setAttribute("x", (a - box.left) * scale);
    brush.setAttribute("width", (b - a) * scale);
    brush.setAttribute("visibility", "visible");
  });
  svg.addEventListener("pointerup", (event) => {
    if (start === null) return;
    const [a, b] = [start, event.clientX].sort((p, q) => p - q);
    start = null;
    if (b - a >= 4) return keep([yearAt(a), yearAt(b)]);
    const bin = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-from]");
    if (bin) keep([Number(bin.dataset.from), Number(bin.dataset.to)]);
  });
  svg.addEventListener("keydown", (event) => {
    const bin = event.target.closest("[data-from]");
    if (!bin || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    keep([Number(bin.dataset.from), Number(bin.dataset.to)]);
  });
}

async function downloadCsv(id) {
  try {
    const { detail } = await detailOf(id);
    const link = element("a", { href: URL.createObjectURL(new Blob([csvText(detail)], { type: "text/csv" })), download: `${id}.csv` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (err) {
    toast(`The runs could not be written: ${err.message}.`, "error");
  }
}

async function start() {
  try {
    const res = await fetch(app("catalog/index.json"), { cache: "no-cache" });
    if (!res.ok) throw new Error(`the server answered ${res.status}`);
    catalog = checkIndex(await res.json());
  } catch (err) {
    $("#count").textContent = `The catalog could not be loaded: ${err.message}.`;
    return;
  }
  const published = catalog.designs.filter((d) => d.collection === "published").length;
  $("#lede").textContent =
    `${published} designed experiments printed in ${Object.keys(catalog.sources).length} textbooks, handbooks and papers, ` +
    `and ${catalog.designs.length - published} designs process-improve makes. Search them, filter them, see where each ` +
    "one sits, and open it in Factorial.";
  if (window.matchMedia("(max-width: 900px)").matches) $("#facets-box").open = false;
  const differ = catalog.designs.reduce((n, d) => n + (d.checks?.explained ?? 0), 0);
  $("#errata-open").textContent = `Errata: the ${differ} printed numbers that do not reproduce, with the reasons`;
  buildFacets();
  // A link from the book names its design in the query (?d=), which the book's theme leaves alone; the page keeps
  // what it shows in the fragment, which no server sees, so the address is rewritten that way at once.
  state = fromHash(location.hash || location.search, catalog.families, catalog.sources, catalog.topics);
  if (!location.hash && location.search) history.replaceState(null, "", location.pathname + toHash(state));
  writeForm();
  wire();
  render();
  startMap();
  openViews();
}

start();
