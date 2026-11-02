// The settings marked "cannot run": a table of the corners of the factors' ranges, and the limit
// Python (api_limits) proposes for each group of corners marked. A proposal can be added to the
// limits as it is, or adjusted factor by factor from its corner: how far each factor can go, with
// the others held there, before the setting cannot be run. Nothing here talks to Python: the panel
// (limits.js) sends the marks with the limits, and hands each reply here to show.
//
// Each reply updates the table and the proposals in place, part by part, so the box the reader is
// in keeps the keyboard and what is typed in it, and a screen reader hears only what changed.

import {
  CORNERS_SHOWN,
  FILTER_CORNERS_FROM,
  alsoRuledOut,
  cornersWords,
  cutWords,
  filterCorners,
  limitEffect,
  listNames,
  settingLabel,
} from "./logic.js";

const $ = (sel) => document.querySelector(sel);

/** What the panel lends the table: what to do when the marks change, how to add a limit, and whether the list is full. */
let hooks = null;

/** The rows marked "cannot run", the filters, how many corners are shown, and the proposal being adjusted. */
let marks = new Set();
let filters = {};
let shown = CORNERS_SHOWN;
let adjusting = null;

/** The last result the table was drawn from, and the factors Python read for it. */
let drawn = null;
let factorsDrawn = [];

/** Each proposal's card names its parts by a number of its own. */
let cards = 0;

function el(tag, cls, text = "") {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

/** Show a part with this text, or hide it when there is none; text that has not changed is left, so it is not read out again. */
function say(node, text) {
  node.hidden = !text;
  if (node.textContent !== text) node.textContent = text;
}

/** The rows the reader marked "cannot run", in order. */
export const markedRows = () => [...marks].sort((a, b) => a - b);

/** The proposal being adjusted factor by factor, as api_limits takes it, or null. */
export const cutsAsked = () => adjusting;

/**
 * Forget the marks: the factors changed in number or kind, so the table's rows are other settings
 * now. The table is hidden until Python draws it again.
 */
export function clearMarks() {
  marks = new Set();
  adjusting = null;
  filters = {};
  shown = CORNERS_SHOWN;
  drawn = null;
  $("#corners").hidden = true;
}

/** The ranges changed: a proposal's corner moved with them, so the cuts given from it go. */
export function forgetCuts() {
  adjusting = null;
}

/** A setting of the table as the page names it, by its row. */
const labelOf = (row) => settingLabel(drawn.probes.rows[row], drawn.probes.columns, factorsDrawn);

/** Settings of the table named in a sentence. */
const named = (rows) => cornersWords(rows, drawn.probes, factorsDrawn);

/** The limits that rule out a row, as the list numbers them. */
const limitNumbers = (row) => drawn.probes.ruled_out_by[row].map((i) => String(i + 1));

/**
 * Draw the table of corners and the proposals from api_limits' result, for the factors Python read;
 * hide them when there is no table to mark (no numeric factor, or more than the page lists the
 * corners of).
 */
export function renderCorners(result, factors) {
  drawn = result?.probes ? result : null;
  factorsDrawn = factors;
  $("#corners").hidden = !drawn;
  if (!drawn) return;
  renderFilters(drawn.probes);
  renderTable(drawn.probes);
  renderProposals(drawn);
}

// --------------------------------------------------------------------------- the table
function renderFilters(probes) {
  const box = $("#corner-filters");
  box.hidden = probes.columns.length < FILTER_CORNERS_FROM;
  const wanted = probes.columns.join(" ");
  if (box.hidden || box.dataset.columns === wanted) return;
  box.dataset.columns = wanted;
  box.replaceChildren(
    ...probes.columns.map((name) => {
      const label = el("label", "", name);
      const select = el("select");
      select.name = name;
      for (const [value, words] of [
        ["any", "Low or high"],
        ["low", "Low"],
        ["high", "High"],
      ]) {
        const option = el("option", "", words);
        option.value = value;
        select.append(option);
      }
      select.value = filters[name] ?? "any";
      label.append(select);
      return label;
    }),
  );
}

/** The corners the filters leave, all of them, in the table's order. */
const filtered = (probes) => filterCorners(probes.rows.slice(0, probes.center), probes.columns, factorsDrawn, filters);

/** The row of the table the keyboard is in, or null. */
const focusedRow = () => {
  const tr = document.activeElement?.closest?.("#corner-table tbody tr");
  return tr ? Number(tr.dataset.row) : null;
};

/** Put the keyboard on a row's box, when the table shows that row. */
const focusRow = (row) => $(`#corner-table tr[data-row="${row}"] input`)?.focus();

/**
 * Draw the rows the filters leave, as many as are shown, and the center last. The table is built
 * again only when its rows or their values change; marking a corner, or a limit changing what is
 * ruled out, updates the rows in place.
 */
function renderTable(probes) {
  const table = $("#corner-table");
  const kept = filtered(probes);
  const rows = [...kept.slice(0, shown), probes.center];
  const units = probes.columns.map((name) => factorsDrawn.find((f) => f.name === name)?.units ?? "");
  const layout = JSON.stringify([probes.columns, units, rows, rows.map((row) => probes.rows[row])]);
  if (table.dataset.layout !== layout) {
    const at = focusedRow();
    table.dataset.layout = layout;
    const head = el("tr");
    head.append(el("th", "", "Cannot run"));
    probes.columns.forEach((name, j) => {
      const th = el("th", "", name);
      if (units[j]) th.append(el("span", "col-unit", units[j]));
      head.append(th);
    });
    head.append(el("th", "", "Ruled out by"));
    const body = el("tbody");
    body.append(...rows.map((row) => cornerRow(row, probes)));
    const thead = el("thead");
    thead.append(head);
    table.replaceChildren(thead, body);
    if (at !== null) focusRow(at);
  }
  for (const tr of table.tBodies[0].rows) markRow(tr);
  const more = kept.length - Math.min(kept.length, shown);
  $("#more-corners").hidden = more <= 0;
  $("#more-corners").textContent = `Show ${Math.min(more, CORNERS_SHOWN)} more corners`;
  renderMarkButtons(probes, kept);
}

function cornerRow(row, probes) {
  const tr = el("tr");
  tr.dataset.row = String(row);
  const cell = el("td");
  const mark = el("label", "corner__mark");
  const box = el("input");
  box.type = "checkbox";
  mark.append(box);
  if (row === probes.center) mark.append(" center");
  cell.append(mark);
  tr.append(cell);
  for (const value of probes.rows[row]) tr.append(el("td", "", String(Number(Number(value).toPrecision(4)))));
  tr.append(el("td", "corner__by"));
  return tr;
}

/** Show a row as marked, ruled out by a limit (ticked, and greyed: a limit decides it), or neither. */
function markRow(tr) {
  const row = Number(tr.dataset.row);
  const by = limitNumbers(row);
  const box = tr.querySelector("input");
  box.checked = by.length > 0 || marks.has(row);
  if (by.length) box.setAttribute("aria-disabled", "true"); // still in the tab order, so the reader hears why
  else box.removeAttribute("aria-disabled");
  const what = row === drawn.probes.center ? `the center of the ranges, ${labelOf(row)}` : labelOf(row);
  const why = by.length ? `, ruled out by limit${by.length > 1 ? "s" : ""} ${listNames(by)}` : "";
  box.setAttribute("aria-label", `Cannot run: ${what}${why}`);
  tr.classList.toggle("ruled", by.length > 0);
  tr.classList.toggle("marked", !by.length && marks.has(row));
  say(tr.querySelector(".corner__by"), listNames(by));
}

/**
 * Offer to mark every corner the filters leave, which is a face of the ranges ("the 4 corners where
 * T is 200 degC"), when a filter is set; and to clear the marks, when there are some.
 */
function renderMarkButtons(probes, kept) {
  const all = $("#mark-shown");
  all.hidden = kept.length === probes.center;
  if (!all.hidden) {
    all.textContent = `Mark ${named(kept)}`;
    const left = kept.some((row) => !marks.has(row) && !probes.ruled_out_by[row].length);
    all.setAttribute("aria-disabled", String(!left)); // focusable still, so the keyboard stays on it once pressed
  }
  $("#clear-marks").hidden = marks.size === 0;
}

/** Mark or unmark a row, show it at once, and have the proposals made again. */
function toggle(tr, on) {
  const row = Number(tr.dataset.row);
  if (on) marks.add(row);
  else marks.delete(row);
  adjusting = null;
  markRow(tr);
  renderMarkButtons(drawn.probes, filtered(drawn.probes));
  hooks.changed();
}

// --------------------------------------------------------------------------- the proposals
/**
 * Show each proposal, or why none can be made. The cards are made again when the groups of corners
 * marked change; otherwise each card is updated part by part, and the questions the reader is
 * answering are left as typed.
 */
function renderProposals(result) {
  const holder = $("#limit-proposals");
  const layout = JSON.stringify([result.proposals, result.adjusted ?? null, adjusting, hooks.full(), factorsDrawn]);
  if (holder.dataset.layout === layout) return;
  holder.dataset.layout = layout;
  const keys = result.proposals.map((proposal) => JSON.stringify(proposal.marked));
  if (JSON.stringify([...holder.children].map((card) => card.dataset.marked)) !== JSON.stringify(keys)) {
    holder.replaceChildren(...keys.map(newCard));
  }
  result.proposals.forEach((proposal, i) => fillCard(holder.children[i], proposal, result));
}

function newCard(key) {
  const id = `proposal-${++cards}`;
  const card = el("div", "proposal");
  card.dataset.marked = key;
  card.setAttribute("role", "group");
  const head = el("p", "hint proposal__for");
  head.id = `${id}-for`;
  card.setAttribute("aria-labelledby", head.id);
  const actions = el("p", "actions");
  const add = el("button", "button", "Add this limit");
  add.type = "button";
  add.dataset.action = "add";
  const adjust = el("button", "link", "Adjust it factor by factor");
  adjust.type = "button";
  adjust.dataset.action = "adjust";
  adjust.setAttribute("aria-expanded", "false");
  adjust.setAttribute("aria-controls", `${id}-cuts`);
  actions.append(add, adjust);
  const cuts = el("div", "cuts");
  cuts.id = `${id}-cuts`;
  cuts.hidden = true;
  card.append(
    head,
    el("p", "hint error proposal__problem"),
    el("p", "proposal__text"),
    el("p", "hint proposal__effect"),
    el("p", "hint warn proposal__also"),
    actions,
    cuts,
    el("p", "hint error cuts__problem"),
  );
  return card;
}

/** The answers the reader gave a proposal's questions, when they are for this proposal and its corner. */
function answersFor(proposal) {
  if (!adjusting || JSON.stringify(adjusting.marked) !== JSON.stringify(proposal.marked)) return null;
  if (adjusting.corner === proposal.corner) return adjusting;
  adjusting = null; // the limit is proposed from another corner now: the answers were for the old one
  return null;
}

function fillCard(card, proposal, result) {
  const part = (name) => card.querySelector(`.${name}`);
  say(part("proposal__for"), `For ${named(proposal.marked)}:`);
  say(part("proposal__problem"), proposal.problem ?? "");
  const answers = proposal.problem ? null : answersFor(proposal);
  const adjusted = answers ? (result.adjusted ?? null) : null;
  const limit = proposal.problem ? null : adjusted && !adjusted.problem ? adjusted : proposal;
  card.dataset.expression = limit?.expression ?? "";
  say(part("proposal__text"), limit?.expression ?? "");
  say(part("proposal__effect"), limit ? limitEffect({ ...limit, redundant: null }, result.probes) : "");
  const extra = limit ? alsoRuledOut(limit, proposal.marked, result.probes.ruled_out_by) : [];
  const one = extra.length === 1;
  say(
    part("proposal__also"),
    extra.length ? `It also rules out ${one ? "a setting" : `${extra.length} settings`} you did not mark: ${named(extra)}.` : "",
  );
  const actions = part("actions");
  actions.hidden = !limit;
  actions.querySelector('[data-action="add"]').disabled = hooks.full();
  actions.querySelector('[data-action="adjust"]').hidden = !limit || !proposal.cuts;
  fillCuts(card, limit ? proposal : { cuts: null }, answers);
  say(part("cuts__problem"), adjusted?.problem ?? "");
}

/**
 * The questions that adjust a proposal from its corner: with the others held at the corner, how far
 * each factor can go before the setting cannot be run, or that it does not limit this at all. They are
 * asked again only when the proposal's corner or cuts change, and never under the reader's keyboard.
 */
function fillCuts(card, proposal, answers) {
  const box = card.querySelector(".cuts");
  const adjust = card.querySelector('[data-action="adjust"]');
  if (!proposal.cuts) {
    box.hidden = true;
    box.replaceChildren();
    delete box.dataset.source;
    adjust.setAttribute("aria-expanded", "false");
    return;
  }
  const source = JSON.stringify([proposal.corner, proposal.cuts]);
  if (box.dataset.source === source || box.contains(document.activeElement)) return;
  box.dataset.source = source;
  const values = answers ? answers.values : proposal.cuts.map((cut) => cut.cut);
  box.replaceChildren(
    el("p", "hint", `From ${labelOf(proposal.corner)}, with the other factors held there:`),
    ...proposal.cuts.map((cut, j) => cutRow(cut, values[j])),
  );
  if (answers) box.hidden = false;
  adjust.setAttribute("aria-expanded", String(!box.hidden));
}

function cutRow(cut, value) {
  const factor = factorsDrawn.find((f) => f.name === cut.name);
  const row = el("div", "cut");
  const free = value === null || value === undefined;
  const box = el("input");
  Object.assign(box, { type: "number", step: "any", name: `cut-${cut.name}`, disabled: free });
  box.min = String(Math.min(factor.low, factor.high));
  box.max = String(Math.max(factor.low, factor.high));
  box.value = free ? "" : String(value);
  const question = el("label", "cut__question", `${cut.name} ${cutWords(cut, factor)} `);
  question.append(box, factor.units ? ` ${factor.units}` : "");
  const tick = el("input");
  Object.assign(tick, { type: "checkbox", name: `free-${cut.name}`, checked: free });
  const none = el("label", "cut__free");
  none.append(tick, " does not limit this");
  row.append(question, none);
  return row;
}

/** The cut the reader gave each factor, in the factors' order: a number, or null where it does not limit this; null while one is still being typed. */
function answered(box) {
  const values = [];
  for (const row of box.querySelectorAll(".cut")) {
    const value = row.querySelector('input[type="number"]');
    if (value.disabled) values.push(null);
    else if (value.validity.badInput || value.value.trim() === "") return null;
    else values.push(Number(value.value));
  }
  return values;
}

/** An answer changed: the limit is rebuilt through the cuts, once each factor has one. */
function cutsChanged(target) {
  const box = target.closest(".cuts");
  if (!box || !drawn) return;
  if (target.type === "checkbox") target.closest(".cut").querySelector('input[type="number"]').disabled = target.checked;
  const values = answered(box);
  const key = box.closest(".proposal").dataset.marked;
  const proposal = drawn.proposals.find((p) => JSON.stringify(p.marked) === key);
  if (!values || !proposal) return;
  adjusting = { marked: proposal.marked, corner: proposal.corner, values };
  hooks.changed();
}

// --------------------------------------------------------------------------- wiring
/**
 * Wire the table to the panel: ``changed`` asks Python again, ``add`` puts a limit into the list,
 * and ``full`` says whether the list has room for one more.
 */
export function installCorners({ changed, add, full }) {
  hooks = { changed, add, full };
  const table = $("#corner-table");
  table.addEventListener("click", (event) => {
    const box = event.target.closest("input[type=checkbox]");
    if (!box || !drawn) return;
    if (box.getAttribute("aria-disabled")) {
      event.preventDefault(); // a ruled-out setting stays shown as ruled out
      return;
    }
    toggle(box.closest("tr"), box.checked);
  });
  table.addEventListener("keydown", (event) => {
    const rows = [...table.tBodies[0].rows].map((tr) => Number(tr.dataset.row));
    const at = rows.indexOf(Number(event.target.closest("tr")?.dataset.row));
    const to = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: rows.length - 1 }[event.key];
    if (to === undefined || at < 0) return;
    event.preventDefault();
    focusRow(rows[Math.max(0, Math.min(rows.length - 1, to))]);
  });
  $("#corner-filters").addEventListener("change", (event) => {
    filters[event.target.name] = event.target.value;
    shown = CORNERS_SHOWN;
    if (drawn) renderCorners(drawn, factorsDrawn);
  });
  $("#mark-shown").onclick = (event) => {
    if (!drawn || event.currentTarget.getAttribute("aria-disabled") === "true") return;
    for (const row of filtered(drawn.probes)) if (!drawn.probes.ruled_out_by[row].length) marks.add(row);
    adjusting = null;
    renderTable(drawn.probes);
    hooks.changed();
  };
  $("#clear-marks").onclick = () => {
    marks = new Set();
    adjusting = null;
    if (drawn) renderTable(drawn.probes);
    focusRow(Number($("#corner-table tbody tr")?.dataset.row));
    hooks.changed();
  };
  $("#more-corners").onclick = () => {
    const first = filtered(drawn.probes)[shown];
    shown += CORNERS_SHOWN;
    renderTable(drawn.probes);
    if ($("#more-corners").hidden) focusRow(first); // the button is gone: the keyboard goes to the first corner it showed
  };
  const holder = $("#limit-proposals");
  holder.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button || button.disabled) return;
    const card = button.closest(".proposal");
    if (button.dataset.action === "adjust") {
      const box = card.querySelector(".cuts");
      box.hidden = !box.hidden;
      button.setAttribute("aria-expanded", String(!box.hidden));
      return;
    }
    if (!card.dataset.expression) return;
    for (const row of JSON.parse(card.dataset.marked)) marks.delete(row);
    adjusting = null;
    add(card.dataset.expression);
  });
  holder.addEventListener("input", (event) => cutsChanged(event.target));
}
