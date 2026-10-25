// The reproduction page (catalog/reproduced/index.html, at factori.al/catalog/reproduced/): it reads
// catalog/reproduction.json once and draws how closely the library reproduces each published case, as a
// table of tiers, a table by source, the spread of the differences and a list of the cases of each tier.
// The rules are in reproduced-logic.js; this module draws them. No Python runs here.

import { element } from "./catalog-views.js";
import {
  DEFAULT_VIEW,
  TIERS,
  byForm,
  bySource,
  errorBands,
  filterCases,
  formatError,
  fromHash,
  groupByTier,
  kindOf,
  kinds,
  percent,
  quantity,
  summarise,
  tierOf,
  toHash,
} from "./reproduced-logic.js";

const $ = (selector, root = document) => root.querySelector(selector);

/** An address of the app, from this module's own, which sits beside the app's index.html. */
const app = (path) => new URL(path, import.meta.url).href;

/** How many cases a source's group lists at first, and adds with each press of "Show more". */
const GROUP_SHOWN = 25;

let data = null;
let view = structuredClone(DEFAULT_VIEW);
let sourceOrder = "year";
/** Draws a tier's list into its box, once: the page calls it when it opens a tier itself, not waiting for the toggle. */
const fillers = new Map();

const count = (n) => n.toLocaleString("en");
const plural = (n, one, many = `${one}s`) => `${count(n)} ${n === 1 ? one : many}`;
const sourceName = (key) => {
  const source = data.sources[key] ?? {};
  return source.year ? `${source.short ?? key} (${source.year})` : (source.short ?? key);
};

/** A tier's chip: its mark, which names it apart from its colour, on its colour. */
const chip = (tier, text = TIERS.find((t) => t.key === tier).mark) =>
  element("span", { class: "tierchip", "data-tier": tier, text });

/** The cases the reader's search, source and kind keep. */
const shown = () => filterCases(data.cases, data.sources, view);

// ---------------------------------------------------------------- the headline and the table of tiers
function drawStats() {
  const all = summarise(data.cases, data.tiers);
  const sources = new Set(data.cases.filter((c) => c.tier).map((c) => c.source)).size;
  $("#stat-numbers").textContent = count(all.numbersChecked);
  $("#stat-agree").textContent = percent(all.numbers.reproduced, all.numbersChecked);
  $("#stat-cases").textContent = count(all.recomputed);
  $("#stat-cases-label").textContent = `cases from ${plural(sources, "source")}, each recomputed`;
  $("#stat-full").textContent = percent(all.cases.reproduced, all.recomputed);
  $("#stats").hidden = false;
}

function drawBreakdown(cases) {
  const s = summarise(cases, data.tiers);
  const filtered = view.query || view.source || view.form || view.kind;
  $("#scope").replaceChildren(
    filtered ? `Of the ${plural(s.recomputed, "case")} your filters keep. ` : `Of all ${plural(s.recomputed, "case")} with a recomputed number. `,
    ...(filtered ? [element("button", { type: "button", class: "link", text: "Show every case", onclick: () => setView(DEFAULT_VIEW) })] : []),
  );
  const bar = $("#tierbar");
  bar.replaceChildren(
    ...TIERS.filter((t) => s.cases[t.key]).map((t) => {
      const part = element("span", { class: "tierbar__part", "data-tier": t.key, title: `${t.label}: ${plural(s.cases[t.key], "case")}` });
      part.style.flexGrow = String(s.cases[t.key]);
      return part;
    }),
  );
  bar.setAttribute(
    "aria-label",
    TIERS.map((t) => `${t.label}: ${percent(s.cases[t.key], s.recomputed)} of the cases`).join("; "),
  );
  $("#tiers tbody").replaceChildren(
    ...TIERS.map((t) => {
      const n = s.cases[t.key];
      const meter = element("span", { class: "meter", "aria-hidden": "true" }, element("span", { "data-tier": t.key }));
      meter.firstChild.style.width = s.recomputed ? `${(100 * n) / s.recomputed}%` : "0";
      const go = n
        ? element("button", {
            type: "button",
            class: "link",
            text: t.key === "reproduced" ? "List them" : `Show the ${plural(n, "case")}`,
            onclick: () => openTier(t.key),
          })
        : null;
      return element(
        "tr",
        { "data-tier": t.key },
        element("th", { scope: "row" }, element("span", { class: "tier__name" }, chip(t.key), t.label)),
        element("td", { class: "tier__means", text: t.means }),
        element("td", { class: "num" }, element("strong", { text: count(n) }), element("span", { class: "share", text: percent(n, s.recomputed) }), meter),
        element("td", { class: "num" }, count(s.numbers[t.key]), element("span", { class: "share", text: percent(s.numbers[t.key], s.numbersChecked) })),
        element("td", {}, go),
      );
    }),
  );
}

// ---------------------------------------------------------------- by source
/** A count in the table by source: a zero is drawn muted, so the counts that are there stand out. */
const cell = (n, extra = {}) => element("td", { class: n ? "num" : "num zero", ...extra }, count(n));

/**
 * A row of the table by source: its counts by tier, its numbers recomputed and the share that agree. Only
 * a row that adds up several sources (a kind of source, or all of them) draws its cases by tier as a bar:
 * a bar beside one source would make a source of one hard case look like a source in error.
 */
function sourceRow(head, tally, { bar = false, ...attributes } = {}) {
  const parts = element("span", { class: "source__bar", "aria-hidden": "true" });
  for (const t of bar ? TIERS : []) {
    if (!tally.tiers[t.key]) continue;
    const part = element("span", { "data-tier": t.key });
    part.style.flexGrow = String(tally.tiers[t.key]);
    parts.append(part);
  }
  return element(
    "tr",
    attributes,
    element("th", { scope: "row" }, ...head),
    cell(tally.total, { class: "num bysource__cases" }),
    ...TIERS.map((t) => cell(tally.tiers[t.key], { "data-tier": t.key })),
    cell(tally.numbers, { class: "num bysource__numbers" }),
    element("td", { class: "num" }, element("strong", { text: percent(tally.agree, tally.numbers) })),
    element("td", { class: "bysource__bar" }, bar ? parts : null),
  );
}

/**
 * The table by source: a group of rows for each form a source takes (books, journal papers, the rest),
 * each with its totals, and the totals of all. It counts the cases the search and the kind keep, from
 * every source, so a reader who has chosen one source can still see and choose the others.
 */
function drawSources() {
  const kept = filterCases(data.cases, data.sources, { ...view, source: "", form: "" });
  const { groups, totals } = byForm(bySource(kept, data.sources, sourceOrder), data.forms);
  const choose = (next) => setView({ ...view, source: "", form: "", ...next, focus: "" });
  $("#bysource thead").replaceChildren(
    element(
      "tr",
      {},
      element("th", { scope: "col", text: "Source" }),
      element("th", { scope: "col", class: "num bysource__cases", text: "Cases" }),
      ...TIERS.map((t) => element("th", { scope: "col", class: "num", title: t.label }, chip(t.key), element("span", { class: "visually-hidden", text: t.label }))),
      element("th", { scope: "col", class: "num bysource__numbers", text: "Numbers" }),
      element("th", { scope: "col", class: "num", title: "The share of the printed numbers that agree with the library", text: "Agree" }),
      element("th", { scope: "col", class: "bysource__bar" }, element("span", { class: "visually-hidden", text: "Cases by tier" })),
    ),
  );
  $("#bysource").replaceChildren(
    $("#bysource caption"),
    $("#bysource thead"),
    ...groups.map((group) =>
      element(
        "tbody",
        { "data-form": group.form },
        sourceRow(
          [
            element("button", {
              type: "button",
              class: "link bysource__form",
              "aria-pressed": String(view.form === group.form),
              text: group.label,
              onclick: () => choose(view.form === group.form ? {} : { form: group.form }),
            }),
            element("span", { class: "hint", text: ` ${plural(group.rows.length, "source")}` }),
          ],
          group.totals,
          { class: "bysource__group", bar: true },
        ),
        ...group.rows.map((row) =>
          sourceRow(
            [
              element("button", {
                type: "button",
                class: "link source",
                "aria-pressed": String(view.source === row.key),
                title: row.title,
                text: sourceName(row.key),
                onclick: () => choose(view.source === row.key ? {} : { source: row.key }),
              }),
            ],
            row,
            { "data-source": row.key },
          ),
        ),
      ),
    ),
    element("tfoot", {}, sourceRow([`All ${plural(groups.reduce((n, g) => n + g.rows.length, 0), "source")}`], totals, { bar: true })),
  );
}

// ---------------------------------------------------------------- the spread of the differences
const SVG = "http://www.w3.org/2000/svg";
function shape(tag, attributes = {}, text = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}

function drawBands(cases) {
  const bands = errorBands(cases, data.tiers);
  const total = bands.reduce((sum, band) => sum + band.count, 0);
  const small = bands.filter((band) => band.upTo !== null && band.upTo <= 0.01).reduce((sum, band) => sum + band.count, 0);
  $("#bands-caption").textContent = total
    ? `Each bar counts the printed numbers that differ from the library's by more than the source's rounding, by how far they differ. ` +
      `${percent(small, total)} of the ${plural(total, "number")} that differ do so by 1% or less. The lines mark the tiers' limits, 1% and 5%.`
    : "No number of the cases your filters keep differs from the library's.";
  const svg = $("#bands");
  const width = 760;
  const height = 230;
  const pad = { left: 8, right: 8, top: 22, bottom: 52 };
  const slot = (width - pad.left - pad.right) / bands.length;
  const most = Math.max(1, ...bands.map((band) => band.count));
  const y = (n) => pad.top + (height - pad.top - pad.bottom) * (1 - n / most);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const nodes = [shape("line", { class: "bands__axis", x1: pad.left, x2: width - pad.right, y1: y(0), y2: y(0) })];
  bands.forEach((band, i) => {
    const x = pad.left + i * slot;
    const h = y(0) - y(band.count);
    const group = shape("g", { class: "bands__band", tabindex: "0", role: "img", "aria-label": `${band.label}: ${plural(band.count, "number")}` });
    group.append(shape("title", {}, `${band.label}: ${plural(band.count, "number")}`));
    group.append(shape("rect", { class: "bands__hit", x: x, y: pad.top, width: slot, height: y(0) - pad.top }));
    if (band.count) {
      group.append(shape("rect", { class: "bands__bar", "data-tier": band.tier, x: x + 5, y: y(band.count), width: slot - 10, height: Math.max(h, 1), rx: 3 }));
    }
    group.append(shape("text", { class: "bands__count", x: x + slot / 2, y: y(band.count) - 6, "text-anchor": "middle" }, count(band.count)));
    const [first, second] = band.label.split(/(?<=-)/);
    const label = shape("text", { class: "bands__label", x: x + slot / 2, y: y(0) + 16, "text-anchor": "middle" });
    label.append(shape("tspan", { x: x + slot / 2, dy: 0 }, first));
    if (second) label.append(shape("tspan", { x: x + slot / 2, dy: 14 }, second));
    group.append(label);
    nodes.push(group);
  });
  for (const [limit, text] of [[0.01, "1%"], [0.05, "5%"]]) {
    const at = bands.findIndex((band) => band.upTo === limit) + 1;
    const x = pad.left + at * slot;
    nodes.push(shape("line", { class: "bands__limit", x1: x, x2: x, y1: pad.top - 14, y2: y(0) + 4 }));
    nodes.push(shape("text", { class: "bands__limitlabel", x: x + 4, y: pad.top - 6 }, text));
  }
  svg.replaceChildren(...nodes);
}

// ---------------------------------------------------------------- the lists of cases
function numberCard(number) {
  const tier = tierOf(number.error, data.tiers);
  const printed = Array.isArray(number.printed) ? JSON.stringify(number.printed) : number.printed;
  return element(
    "li",
    { class: "number" },
    element(
      "div",
      { class: "number__head" },
      element("span", { class: "number__what", text: quantity(number) || "A printed number" }),
      element("span", { class: "number__error", "data-tier": tier, text: formatError(number.error) }),
    ),
    element(
      "dl",
      { class: "number__values" },
      element("div", {}, element("dt", { text: "Printed" }), element("dd", {}, element("code", { text: printed }))),
      element("div", {}, element("dt", { text: "process-improve" }), element("dd", {}, element("code", { text: number.found }))),
      number.where ? element("div", {}, element("dt", { text: "Where" }), element("dd", { text: number.where })) : null,
    ),
    element("p", { class: "number__reason", text: number.reason }),
  );
}

function caseItem(c) {
  const differ = c.differs.length;
  const facts = differ
    ? `${count(differ)} of ${plural(c.checked, "number")} differ${differ === 1 ? "s" : ""}`
    : `all ${plural(c.checked, "number")} agree`;
  const body = element("div", { class: "case__body" });
  const item = element(
    "details",
    { class: "case", id: `case-${c.id}`, "data-id": c.id },
    element(
      "summary",
      {},
      element("span", { class: "case__title", text: c.title }),
      element(
        "span",
        { class: "case__facts" },
        c.tier === "reproduced" ? chip("reproduced") : element("span", { class: "number__error", "data-tier": c.tier, text: `worst ${formatError(c.worst)}` }),
        element("span", { text: facts }),
        element("span", { class: "case__kind", text: kindOf(c) }),
      ),
    ),
    body,
  );
  // The numbers are drawn when the case opens: a tier can list hundreds of cases.
  item.addEventListener("toggle", () => {
    if (item.open && !body.childElementCount) {
      body.append(
        ...(differ ? [element("ul", { class: "numbers" }, ...c.differs.map(numberCard))] : []),
        element(
          "p",
          { class: "case__actions" },
          element("a", { href: app(`catalog/#d=${encodeURIComponent(c.id)}`), text: "Open its design and runs in the catalog" }),
          element("button", { type: "button", class: "link", text: "Copy a link to this case", onclick: () => copyLink(c) }),
        ),
      );
    }
    if (item.open && view.focus !== c.id) {
      view.focus = c.id;
      history.replaceState(null, "", toHash(view) || location.pathname);
    }
  });
  return item;
}

async function copyLink(c) {
  const url = new URL(location.href);
  url.hash = toHash({ ...DEFAULT_VIEW, open: [c.tier], focus: c.id });
  try {
    await navigator.clipboard.writeText(url.href);
    $("#count").textContent = `A link to "${c.title}" is copied.`;
  } catch {
    $("#count").textContent = `Copy this link: ${url.href}`;
  }
}

function sourceGroup({ source, cases }) {
  const list = element("ol", { class: "cases" });
  let upTo = 0;
  const more = element("button", { type: "button", class: "button more" });
  const fill = () => {
    list.append(...cases.slice(upTo, upTo + GROUP_SHOWN).map(caseItem));
    upTo = Math.min(cases.length, upTo + GROUP_SHOWN);
    more.hidden = upTo >= cases.length;
    more.textContent = `Show ${count(Math.min(GROUP_SHOWN, cases.length - upTo))} more of ${count(cases.length - upTo)}`;
  };
  more.onclick = fill;
  fill();
  return element(
    "div",
    { class: "group" },
    element("h3", {}, sourceName(source), element("span", { class: "badge", text: count(cases.length) })),
    list,
    more,
  );
}

function drawLists(cases) {
  // The cases of a tier are grouped by source in the order of the table by source: by form, then oldest first.
  const order = byForm(bySource(data.cases, data.sources), data.forms).groups.flatMap((g) => g.rows.map((row) => row.key));
  $("#lists").replaceChildren(
    ...TIERS.map((t) => {
      const n = cases.filter((c) => c.tier === t.key).length;
      const body = element("div", { class: "tierlist__body" });
      const box = element(
        "details",
        { class: "tierlist", id: `tier-${t.key}`, "data-tier": t.key },
        element(
          "summary",
          {},
          chip(t.key),
          element("span", { class: "tierlist__name", text: t.label }),
          element("span", { class: "badge", text: plural(n, "case") }),
          element("span", { class: "tierlist__means hint", text: t.means }),
        ),
        body,
      );
      const fill = () => {
        if (body.childElementCount) return;
        const groups = groupByTier(cases, t.key, order);
        body.append(...(groups.length ? groups.map(sourceGroup) : [element("p", { class: "hint", text: "No case your filters keep is in this tier." })]));
      };
      fillers.set(t.key, fill);
      box.addEventListener("toggle", () => {
        if (box.open) fill();
        const open = new Set(view.open);
        if (box.open) open.add(t.key);
        else open.delete(t.key);
        view.open = TIERS.map((tier) => tier.key).filter((key) => open.has(key));
        history.replaceState(null, "", toHash(view) || location.pathname);
      });
      if (view.open.includes(t.key)) {
        box.open = true;
        fill();
      }
      return box;
    }),
  );
}

function openTier(tier) {
  const box = $(`#tier-${tier}`);
  box.open = true;
  fillers.get(tier)();
  box.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  box.querySelector("summary").focus({ preventScroll: true });
}

/** Open the case the address names, with its tier, and bring it into view. */
function focusCase() {
  const c = data.cases.find((x) => x.id === view.focus && x.tier);
  if (!c) return;
  const box = $(`#tier-${c.tier}`);
  box.open = true;
  fillers.get(c.tier)();
  let item = $(`#case-${CSS.escape(c.id)}`);
  for (let i = 0; !item && i < 50; i += 1) {
    const more = [...box.querySelectorAll(".more")].find((button) => !button.hidden && button.closest(".group").querySelector("h3").textContent.startsWith(sourceName(c.source)));
    if (!more) break;
    more.click();
    item = $(`#case-${CSS.escape(c.id)}`);
  }
  if (!item) return;
  item.open = true;
  item.scrollIntoView({ block: "center" });
  item.querySelector("summary").focus({ preventScroll: true });
}

// ---------------------------------------------------------------- the reader's view
function draw() {
  const cases = shown();
  drawBreakdown(cases);
  drawSources();
  drawBands(cases);
  drawLists(cases);
  $("#query").value = view.query;
  $("#source").value = view.form ? `form:${view.form}` : view.source;
  $("#kind").value = view.kind;
  $("#clear").hidden = !(view.query || view.source || view.form || view.kind);
  const all = data.cases.filter((c) => c.tier).length;
  $("#count").textContent =
    cases.length === all ? `${plural(all, "case")}, in four tiers.` : `${plural(cases.length, "case")} of ${count(all)} kept.`;
}

function setView(next) {
  view = { ...structuredClone(DEFAULT_VIEW), ...next };
  history.replaceState(null, "", toHash(view) || location.pathname);
  draw();
}

function wire() {
  // One choice of where the cases come from: every source of a form ("form:<form>"), or one source.
  const { groups } = byForm(bySource(data.cases, data.sources, "name"), data.forms);
  $("#source").append(
    ...groups.map((group) =>
      element(
        "optgroup",
        { label: group.label },
        element("option", { value: `form:${group.form}`, text: `All ${group.label.toLowerCase()} (${count(group.totals.total)})` }),
        ...group.rows.map((row) => element("option", { value: row.key, text: sourceName(row.key) })),
      ),
    ),
  );
  $("#kind").append(...kinds(data.cases).map(([kind, n]) => element("option", { value: kind, text: `${kind[0].toUpperCase()}${kind.slice(1)} (${count(n)})` })));
  let typing = null;
  $("#query").addEventListener("input", (event) => {
    clearTimeout(typing);
    typing = setTimeout(() => setView({ ...view, query: event.target.value.trim(), focus: "" }), 200);
  });
  $("#source").addEventListener("change", (event) => {
    const [form, source] = event.target.value.startsWith("form:") ? [event.target.value.slice(5), ""] : ["", event.target.value];
    setView({ ...view, source, form, focus: "" });
  });
  $("#kind").addEventListener("change", (event) => setView({ ...view, kind: event.target.value, focus: "" }));
  $("#clear").addEventListener("click", () => setView({ ...DEFAULT_VIEW, open: view.open }));
  // A word of the reasons the catalogue records: the search keeps the cases whose numbers differ for it.
  for (const button of document.querySelectorAll(".try")) {
    button.addEventListener("click", () => {
      const open = TIERS.slice(1).map((t) => t.key);
      setView({ ...view, query: button.textContent, open, focus: "" });
    });
  }
  $("#source-order").addEventListener("change", (event) => {
    sourceOrder = event.target.value;
    drawSources();
  });
  window.addEventListener("hashchange", () => {
    view = fromHash(location.hash, data.sources, data.forms);
    draw();
    focusCase();
  });
}

async function main() {
  try {
    const response = await fetch(app("catalog/reproduction.json"));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    $("#status").textContent = `The cases could not be loaded (${error.message}). Reload the page to try again.`;
    $("#status").classList.add("error");
    return;
  }
  view = fromHash(location.hash, data.sources, data.forms);
  $("#status").hidden = true;
  for (const id of ["#breakdown", "#sources-section", "#bands-section", "#cases"]) $(id).hidden = false;
  $("#built").textContent = `Recomputed with process-improve ${data.library}.`;
  drawStats();
  wire();
  draw();
  focusCase();
}

main();
