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
  fromHash,
  largestSecondOrderR,
  minimumRuns,
  nextSteps,
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
  yearBins,
} from "./catalog-logic.js";
import { compareTable, drawTimeline, element, errataTable, projectionExplorer, table } from "./catalog-views.js";

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

let catalog = null;
let state = structuredClone({ ...DEFAULT_STATE });
let shown = PAGE;
let plotlyVersion = null;
/** Resolves to the Plotly.js version once the app's manifest is read, or to null when it cannot be. */
let plotlyReady = Promise.resolve(null);
let mapWired = false;
let errata = null;
const shards = new Map();

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
}

function writeForm() {
  $("#query").value = state.query;
  for (const button of $$("#collection [role=radio]")) button.setAttribute("aria-checked", String(button.dataset.value === state.collection));
  for (const box of $$("#families input")) box.checked = state.families.includes(box.value);
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
  for (const box of $$("[data-switch]")) {
    $(".count", box.closest("label")).textContent = box.checked ? "" : String(switchCount(catalog, state, box.dataset.switch));
  }
}

function renderTimeline() {
  const bins = yearBins(catalog, state);
  drawTimeline($("#year-bars"), bins, familyMarks(catalog, bins), state.years);
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
    $("#compare-body").replaceChildren(
      element("p", {
        class: "hint",
        text: "Each design is scored at its own size for the model chosen, over the cube from -1 to +1 in every factor, as compare_designs in process-improve scores it. The arrow says which direction a measure prefers; no single design is best on every row. Power is for a coefficient of one noise standard deviation, tested at the 5% level.",
      }),
      select,
      holder,
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
  });
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

async function openInFactorial(id) {
  try {
    const { detail } = await detailOf(id);
    if (!detail.campaign) throw new Error(detail.closed || "the page does not open this design");
    location.assign(app(`./${LINK_KEY}${await packCampaign(detail.campaign)}`));
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

function whereText(where) {
  if (!where) return "";
  if (typeof where === "string") return where;
  const parts = [];
  if (where.section) parts.push(`section ${where.section}`);
  if (where.pages) parts.push(`${String(where.pages).match(/[-,]/) ? "pages" : "page"} ${where.pages}`);
  for (const key of ["tables", "figures"]) if (where[key]?.length) parts.push(where[key].join(", "));
  return parts.join("; ");
}

function detailBody(design, detail) {
  const family = catalog.families[design.family];
  const source = catalog.sources[design.source];
  const body = [element("p", { class: "detail__facts", text: factLine(design, catalog) })];
  if (family?.year) body.push(element("p", { class: "hint", text: `${family.label} designs were first published by ${family.reference}.` }));
  if (source) {
    const cite = element("p", { class: "detail__cite" }, source.url ? element("a", { href: source.url, rel: "noopener", text: source.cite }) : source.cite);
    body.push(cite);
    const where = whereText(detail.where);
    if (where) body.push(element("p", { text: `Printed in ${where}.` }));
    const checks = checkLine(design.checks);
    if (checks) body.push(element("p", { text: `Printed numbers: ${checks}. The library recomputes them on every change to it.` }));
    if (design.regenerable === true) body.push(element("p", { class: "detail__ok", text: "The library makes this design: the page's runs are the source's runs, to within its rounding." }));
    if (design.regenerable === false) body.push(element("p", { class: "hint", text: `The design the page makes from this description is not the source's: ${detail.closed}.` }));
  } else {
    body.push(element("p", { text: "Made by process-improve's generate_design, with the call below." }));
  }
  if (detail.details) body.push(element("p", { class: "hint", text: detail.details }));
  if (detail.notes) body.push(element("p", { class: "hint", text: detail.notes }));

  const actions = element("div", { class: "actions" });
  if (detail.campaign) {
    actions.append(element("button", { type: "button", class: "button primary", "data-open": design.id, text: "Open in Factorial" }));
  } else {
    body.push(element("p", { class: "hint", text: `Factorial does not open this design: ${detail.closed}.` }));
  }
  actions.append(
    design.compared ? compareButton(design) : null,
    element("button", { type: "button", class: "button", "data-copy-link": design.id, text: "Copy a link to this design" }),
    element("button", { type: "button", class: "button", "data-csv": design.id, text: "Download the runs (CSV)" }),
  );
  body.push(actions);

  body.push(...measures(design, detail), ...projections(design, detail), ...steps(design));

  const runs = detail.coded_runs ?? [];
  if (runs.length) {
    const k = runs[0].length;
    const canvas = element("canvas", { class: "glyph glyph--large", "aria-hidden": "true" });
    const cell = Math.max(6, Math.min(22, Math.floor(260 / k)));
    drawGlyph(canvas, runs.slice(0, 400), { across: false, width: cell * k, height: Math.min(400, runs.length) * Math.max(3, Math.min(10, Math.floor(480 / runs.length))) });
    const names = (detail.factors ?? []).map((f) => f.name);
    const responses = Object.keys(detail.response_values ?? {});
    body.push(
      element("h3", { text: "Runs, in coded units" }),
      element("p", { class: "hint", text: "-1 is a factor's low level and +1 its high level; blue is low, grey the centre, vermillion high." }),
      element(
        "div",
        { class: "detail__runs" },
        canvas,
        table(
          ["Run", ...names, ...responses],
          runs.slice(0, 400).map((row, i) => [String(i + 1), ...row.map(fmt), ...responses.map((r) => fmt(detail.response_values[r][i]))]),
        ),
      ),
    );
  }
  if (detail.factors?.length) {
    body.push(
      element("h3", { text: "Factors" }),
      table(
        ["Name", "Low", "High", "Units", "Type", "Description"],
        detail.factors.map((f) => [f.name, fmt(f.low), fmt(f.high), f.units ?? "", f.type ?? "", f.description ?? ""]),
      ),
    );
  }
  const code = pythonCode(design, detail);
  body.push(
    element("h3", { text: "In Python" }),
    element("pre", { class: "detail__code" }, element("code", { text: code })),
    element("button", { type: "button", class: "link", "data-copy-code": design.id, text: "Copy the code" }),
  );
  if (source?.rights) body.push(element("p", { class: "hint detail__rights", text: source.rights }));
  return body;
}

/**
 * Draw each run's leverage as a bar, with a line at 1: a run at 1 cannot be lost, since without it the model
 * matrix loses its rank. Those runs are drawn in vermillion, the others in the page's accent.
 */
function drawLeverage(canvas, leverage) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.min(640, Math.max(160, 12 * leverage.length));
  const height = 72;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const g = canvas.getContext("2d");
  if (!g) return;
  g.scale(ratio, ratio);
  const colours = palette();
  const step = width / leverage.length;
  const top = 6;
  leverage.forEach((h, i) => {
    g.fillStyle = h >= 0.999999 ? "#D55E00" : colours.accent;
    const bar = (height - top) * h;
    g.fillRect(i * step + 1, height - bar, Math.max(step - 2, 1), bar);
  });
  g.strokeStyle = colours.muted;
  g.setLineDash([4, 3]);
  g.beginPath();
  g.moveTo(0, top);
  g.lineTo(width, top);
  g.stroke();
}

/** What a design withstands: losing a run, and being projected onto fewer factors; and where else it is printed. */
function measures(design, detail) {
  const out = [];
  const loss = runLossLine(design, detail.run_loss_model);
  const projections = projectionLines(design);
  if (loss || projections.length) out.push(element("h3", { text: "What it withstands" }));
  if (loss) {
    out.push(element("p", { text: loss }));
    if (detail.leverage?.length) {
      const canvas = element("canvas", { class: "leverage", "aria-hidden": "true" });
      drawLeverage(canvas, detail.leverage);
      out.push(
        canvas,
        element("p", { class: "hint", text: "Each run's leverage, in run order. Losing a run scales the determinant of X'X by one less its leverage, so a run at the line, 1, cannot be lost." }),
      );
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
    const height = Math.min(560, 160 + 9 * map.terms.length);
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
    writeForm();
    update();
  });
  $("#sort").addEventListener("change", () => {
    state.sort = $("#sort").value;
    update();
  });
  $("#more").addEventListener("click", () => {
    shown += PAGE;
    update({ keepPage: true });
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-id], [data-open], [data-copy-link], [data-copy-code], [data-csv], [data-compare], [data-close]");
    if (!target) return;
    const { id, open, copyLink, copyCode, csv, compare, close } = target.dataset;
    if (close) $(`#${close}`).close();
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
    state = fromHash(location.hash, catalog.families, catalog.sources);
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
  state = fromHash(location.hash, catalog.families, catalog.sources);
  writeForm();
  wire();
  render();
  startMap();
  openViews();
}

start();
