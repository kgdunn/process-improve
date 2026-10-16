// The design catalog's page: it reads catalog/index.json once, then filters, counts, maps and lists
// the designs as the reader chooses, and opens one in Factorial from the campaign its detail file
// holds. The rules are in catalog-logic.js; this module draws them. No Python runs here: the build
// computed every facet, so the page is ready as soon as the index arrives.

import { LINK_KEY, packCampaign } from "./campaign.js";
import { draw, palette } from "./charts.js";
import {
  DEFAULT_STATE,
  FAMILY_COLOURS,
  MODELS,
  OTHER_COLOUR,
  SORTS,
  checkIndex,
  checkLine,
  csvText,
  decodeGlyph,
  factLine,
  familyCounts,
  fromHash,
  minimumRuns,
  pythonCode,
  results,
  settingColour,
  sourceLine,
  switchCount,
  toHash,
  yearBins,
} from "./catalog-logic.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

/** How many designs the list shows at first, and adds with each press of "Show more". */
const PAGE = 48;
/** The most factors the map draws: a design with more is in the list, and the map says how many. */
const MAP_FACTORS = 16;
const SWITCHES = ["blocked", "splitPlot", "mixture", "hasData", "recomputed", "differs", "opens"];

let catalog = null;
let state = structuredClone({ ...DEFAULT_STATE });
let shown = PAGE;
let plotlyVersion = null;
let mapWired = false;
const shards = new Map();

// ---------------------------------------------------------------- small pieces of the document
function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.includes("-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

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
  state.categorical = { none: [null, 0], some: [1, null] }[$("#categorical").value] ?? [null, null];
  for (const box of $$("[data-switch]")) state[box.dataset.switch] = box.checked;
  state.addedSince = $("#added").value;
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
  const [low, high] = state.categorical;
  $("#categorical").value = high === 0 ? "none" : low !== null && low >= 1 ? "some" : "";
  for (const box of $$("[data-switch]")) box.checked = state[box.dataset.switch];
  $("#added").value = state.addedSince;
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

function renderYearBars() {
  const svg = $("#year-bars");
  const bins = yearBins(catalog, state);
  const height = 44;
  const width = 240;
  svg.setAttribute("viewBox", `0 0 ${width} ${height + 14}`);
  if (!bins.length) return svg.replaceChildren();
  const most = Math.max(...bins.map((b) => b.count), 1);
  const step = width / bins.length;
  const [from, to] = state.years;
  const NS = "http://www.w3.org/2000/svg";
  const bars = bins.map((bin, i) => {
    const h = bin.count ? Math.max(2, (height * bin.count) / most) : 0;
    const rect = document.createElementNS(NS, "rect");
    const chosen = (from === null || bin.to >= from) && (to === null || bin.from <= to);
    Object.entries({ x: i * step + 1, y: height - h, width: Math.max(step - 2, 1), height: h, class: chosen ? "bar" : "bar bar--out" }).forEach(
      ([k, v]) => rect.setAttribute(k, v),
    );
    rect.setAttribute("tabindex", "0");
    rect.setAttribute("role", "button");
    rect.setAttribute("aria-label", `${bin.from} to ${bin.to}: ${bin.count} designs`);
    rect.dataset.from = bin.from;
    rect.dataset.to = bin.to;
    const title = document.createElementNS(NS, "title");
    title.textContent = `${bin.from} to ${bin.to}: ${bin.count} designs`;
    rect.append(title);
    return rect;
  });
  const label = (x, text, anchor) => {
    const t = document.createElementNS(NS, "text");
    Object.entries({ x, y: height + 12, "text-anchor": anchor, class: "bars__label" }).forEach(([k, v]) => t.setAttribute(k, v));
    t.textContent = text;
    return t;
  };
  svg.replaceChildren(...bars, label(0, bins[0].from, "start"), label(width, bins.at(-1).to, "end"));
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
  return element(
    "li",
    { class: "result" },
    canvas,
    element(
      "div",
      { class: "result__body" },
      element("h3", {}, element("button", { type: "button", class: "link result__title", "data-id": design.id, text: design.title }), badge, data),
      element("p", { class: "result__facts", text: factLine(design, catalog) }),
      element("p", { class: "result__source", text: [sourceLine(design, catalog), checks].filter(Boolean).join(" · ") }),
    ),
    element(
      "div",
      { class: "result__actions" },
      design.open
        ? element("button", { type: "button", class: "button primary", "data-open": design.id, text: "Open in Factorial" })
        : element("button", { type: "button", class: "button", "data-id": design.id, text: "Details" }),
    ),
  );
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
  renderYearBars();
  renderList(list);
  drawMap(list);
}

function update({ keepPage = false } = {}) {
  if (!keepPage) shown = PAGE;
  history.replaceState(null, "", toHash(state) || location.pathname + location.search);
  render();
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

async function startMap() {
  try {
    const manifest = await (await fetch("wheels/manifest.json", { cache: "no-cache" })).json();
    plotlyVersion = manifest.plotly_js;
    if (!plotlyVersion) throw new Error("no Plotly.js version");
    drawMap(results(catalog, state));
  } catch {
    $(".catalog__map").hidden = true; // the list holds every design the map would show
  }
}

// ---------------------------------------------------------------- one design
function shard(name) {
  if (!shards.has(name)) {
    shards.set(
      name,
      fetch(`catalog/detail/${encodeURIComponent(name)}.json`, { cache: "no-cache" }).then((res) => {
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
    location.assign(`./${LINK_KEY}${await packCampaign(detail.campaign)}`);
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

function table(head, rows, className = "data") {
  return element(
    "div",
    { class: "scroll" },
    element(
      "table",
      { class: className },
      element("thead", {}, element("tr", {}, ...head.map((h) => element("th", { scope: "col", text: h })))),
      element("tbody", {}, ...rows.map((row) => element("tr", {}, ...row.map((cell) => element("td", { text: cell ?? "" }))))),
    ),
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
    element("button", { type: "button", class: "button", "data-copy-link": design.id, text: "Copy a link to this design" }),
    element("button", { type: "button", class: "button", "data-csv": design.id, text: "Download the runs (CSV)" }),
  );
  body.push(actions);

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
  $("#year-bars").addEventListener("click", (event) => {
    const bar = event.target.closest("rect");
    if (!bar) return;
    state.years = [Number(bar.dataset.from), Number(bar.dataset.to)];
    writeForm();
    update();
  });
  $("#year-bars").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
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
    const target = event.target.closest("[data-id], [data-open], [data-copy-link], [data-copy-code], [data-csv]");
    if (!target) return;
    const { id, open, copyLink, copyCode, csv } = target.dataset;
    if (open) openInFactorial(open);
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
  window.addEventListener("hashchange", () => {
    state = fromHash(location.hash, catalog.families);
    writeForm();
    render();
    if (state.design) openDetail(state.design);
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
    const res = await fetch("catalog/index.json", { cache: "no-cache" });
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
  buildFacets();
  state = fromHash(location.hash, catalog.families);
  writeForm();
  wire();
  render();
  if (state.design) openDetail(state.design);
  startMap();
}

start();
