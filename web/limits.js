// The limits on the settings: the panel under the factors where the reader says which settings
// cannot be run. Each limit is typed as an inequality in the factors' own units, and Python
// (api_limits) reads it as the library will, says what it rules out, and gives back its canonical
// text. The panel holds the limits as the reader types them; the page asks it for them when it
// makes the run sheet or lists the designs, and waits while a limit cannot be read. The settings
// marked "cannot run" in the table of corners (corners.js) go with each request, and Python
// proposes a limit for them; the picture of the settings that can be run (picture.js) is drawn
// from each reply.

import {
  MAX_CONSTRAINTS,
  NOTE_LENGTH,
  insertAt,
  isCategorical,
  limitEffect,
  limitsSummary,
  positional,
  renameLimit,
  withBounds,
} from "./logic.js";
import { clearMarks, cutsAsked, forgetCuts, installCorners, markedRows, renderCorners } from "./corners.js";
import { closePicture, installPicture, pictureAsked, renameInPicture, renderPicture } from "./picture.js";

const $ = (sel) => document.querySelector(sel);

/** How long the typing rests before Python reads the limits: a pause, not each key. */
const LIMITS_WAIT = 250;

/** What the page lends the panel: Python's entry points, the brief's factors, and what to do when the limits change. */
let hooks = null;

/**
 * The last reply of api_limits; the request it answered, as JSON, whole and for the limits alone; and
 * the request in flight, with its own ticket.
 */
let reply = null;
let answered = null;
let answeredLimits = null;
let asking = null;
let ticket = 0;
let timer = null;

/** The factors' kinds, numeric or categorical, by place, and their ranges: the table of corners stands for them. */
let kinds = null;
let ranges = null;

/** The expression box the factor buttons type into: the one the reader was in last. */
let typingIn = null;

function el(tag, cls, text = "") {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

const rows = () => [...document.querySelectorAll("#limit-list .limit")];
const field = (row, name) => row.querySelector(`[name="${name}"]`);

/** The limits as the brief holds them: each one typed, with its note when it has one; a box left blank is no limit. */
export function currentLimits() {
  return rows()
    .map((row) => ({ expression: field(row, "expression").value.trim(), note: field(row, "note").value.trim() }))
    .filter((limit) => limit.expression)
    .map(({ expression, note }) => (note ? { expression, note } : { expression }));
}

export const hasLimits = () => currentLimits().length > 0;

/** What keys a run sheet to the limits: each one by its factors' places, so a rename leaves it. */
export const limitsKey = (names) => currentLimits().map((limit) => positional(limit.expression, names));

/** The factors and limits Python reads: what makes the brief ready or not. */
const limitsRequest = () => ({ factors: hooks.factors(), constraints: currentLimits() });

/**
 * The request api_limits answers: the limits; the settings marked "cannot run", with a proposal
 * being adjusted; and what the picture is asked to show.
 */
function request() {
  const marked = markedRows();
  const cuts = cutsAsked();
  const asked = limitsRequest();
  return { ...asked, ...(marked.length ? { marked } : {}), ...(cuts ? { cuts } : {}), ...pictureAsked(asked.factors) };
}

/** Whether Python has read the limits as they are now. */
const current = () => reply !== null && answeredLimits === JSON.stringify(limitsRequest());

/** Whether Python has anything to read: limits, settings marked, or the panel open on its table of corners. */
const needed = () => hasLimits() || markedRows().length > 0 || $("#limits").open;

/** Note that Python's reply holds for the request as it is now: its canonical text replaced what was typed. */
function answeredNow() {
  answered = JSON.stringify(request());
  answeredLimits = JSON.stringify(limitsRequest());
}

/**
 * The sentence that holds the brief back, when a limit cannot be read or no design can keep to the
 * limits; null when the limits are ready, or when there are none.
 */
export function limitsHold() {
  if (!hasLimits() || !current()) return null;
  return reply.ok ? reply.result.refusal : reply.error;
}

/** Whether the limits keep the brief from being made into designs: Python has still to read them, or refuses them. */
export const limitsUnready = () => hasLimits() && (!current() || limitsHold() !== null);

/**
 * Put a study's limits into the panel: a saved study's, a workbook's, a draft's. The panel opens on
 * them, and Python reads them when it is ready.
 */
export function setLimits(limits = []) {
  $("#limit-list").replaceChildren();
  for (const limit of limits) addLimit(limit);
  clearMarks();
  reply = null;
  answered = null;
  answeredLimits = null;
  $("#limits").open = limits.length > 0;
  render();
  schedule();
}

/** Rename factors inside every limit, as ``names`` maps them (old to new), all at once. */
export function renameInLimits(names) {
  for (const row of rows()) {
    const box = field(row, "expression");
    box.value = renameLimit(box.value, names);
  }
  renameInPicture(names);
  changed();
}

/**
 * The factors changed, their ranges or their kinds: the limits are read again over them. A factor
 * added, removed or made categorical changes what the table's rows stand for, so the marks go; a
 * range changed moves the corner a proposal is adjusted from, so the cuts given from it go.
 */
export function factorsChangedForLimits() {
  const factors = hooks.factors();
  const nowKinds = factors.map((f) => (isCategorical(f) ? "c" : "n")).join("");
  const nowRanges = JSON.stringify(factors.map((f) => (isCategorical(f) ? null : [f.low, f.high])));
  if (kinds !== null && nowKinds !== kinds) {
    clearMarks();
    closePicture();
  } else if (ranges !== null && nowRanges !== ranges) forgetCuts();
  kinds = nowKinds;
  ranges = nowRanges;
  renderNames();
  schedule();
}

/**
 * Take the canonical text Python gave the limits a design was made with, box by box in the order
 * the brief holds them, so the brief keys the run sheet as the run sheet keys itself.
 */
export function adoptLimits(stored = []) {
  const fresh = current();
  const typed = rows().filter((row) => field(row, "expression").value.trim());
  typed.forEach((row, i) => {
    if (stored[i]) field(row, "expression").value = stored[i].expression;
  });
  if (fresh) answeredNow(); // Python's reply holds for its canonical text as for the text typed
}

function addLimit({ expression = "", note = "" } = {}) {
  const row = el("div", "limit");
  row.setAttribute("role", "group");
  const number = el("span", "limit__number");
  const text = el("input", "limit__text");
  Object.assign(text, { name: "expression", autocomplete: "off", spellcheck: false, value: expression });
  text.setAttribute("autocapitalize", "off");
  text.placeholder = placeholder();
  const bounds = el("span", "limit__bounds");
  bounds.hidden = true;
  for (const [name, words] of [
    ["low", "at least"],
    ["high", "at most"],
  ]) {
    const label = el("label", "limit__bound");
    label.dataset.side = name;
    const box = el("input");
    Object.assign(box, { name, type: "number", step: "any" });
    label.append(`${words} `, box);
    bounds.append(label);
  }
  const why = el("input", "limit__note");
  Object.assign(why, { name: "note", maxLength: NOTE_LENGTH, autocomplete: "off", value: note });
  why.placeholder = "Why, in a few words (optional)";
  const remove = el("button", "link limit__remove", "✕");
  remove.type = "button";
  remove.onclick = () => {
    if (typingIn === text) typingIn = null;
    row.remove();
    numberRows();
    changed();
  };
  const effect = el("p", "hint limit-effect");
  const problem = el("p", "hint error limit-problem");
  problem.setAttribute("role", "status");
  problem.hidden = true;
  row.append(number, text, bounds, why, remove, effect, problem);
  text.addEventListener("focus", () => (typingIn = text));
  text.addEventListener("blur", () => {
    if (takeCanonical(row)) hooks.changed(); // the box reads as the study stores it now
  });
  for (const box of bounds.querySelectorAll("input")) box.addEventListener("input", () => fromBounds(row));
  $("#limit-list").append(row);
  numberRows();
  return row;
}

/** Number the limits 1, 2, ... as the picture of the region numbers their boundaries, and name each box for it. */
function numberRows() {
  rows().forEach((row, i) => {
    const n = i + 1;
    row.querySelector(".limit__number").textContent = String(n);
    row.setAttribute("aria-label", `Limit ${n}`);
    field(row, "expression").setAttribute("aria-label", `Limit ${n}`);
    field(row, "note").setAttribute("aria-label", `Why limit ${n}`);
    row.querySelector(".limit__remove").setAttribute("aria-label", `Remove limit ${n}`);
    for (const label of row.querySelectorAll(".limit__bound")) {
      field(label, label.dataset.side).setAttribute("aria-label", `Limit ${n}, ${label.textContent.trim()}`);
    }
  });
  const full = rows().length >= MAX_CONSTRAINTS;
  $("#add-limit").disabled = full;
  $("#limits-full").hidden = !full;
}

/** An example limit in the brief's own factors, for an empty box: the first two numeric factors, if there are two. */
function placeholder() {
  const numeric = hooks.factors().filter((f) => !isCategorical(f) && f.name);
  if (numeric.length < 2) return "such as 2*A + B <= 10";
  const [a, b] = numeric;
  return `such as ${a.name} + ${b.name} <= ...`;
}

/** A bound box changed: the limit is written again from its left side and its bounds. */
function fromBounds(row) {
  const entry = entryOf(row);
  if (!entry?.linear) return;
  const value = (side) => {
    const box = row.querySelector(`[name="${side}"]`);
    if (box.closest("label").hidden) return null;
    return box.value.trim() === "" ? entry.linear[side] : Number(box.value);
  };
  field(row, "expression").value = withBounds(entry.linear.lhs, value("low"), value("high"));
  changed();
}

/** Python's description of a limit the reply holds for this row, when the row reads as it was asked. */
function entryOf(row) {
  if (!reply?.ok) return null;
  const typed = rows().filter((r) => field(r, "expression").value.trim());
  const i = typed.indexOf(row);
  const sent = JSON.parse(answered).constraints[i];
  return sent && sent.expression === field(row, "expression").value.trim() ? reply.result.constraints[i] : null;
}

/** The reader left a limit's box: it takes the canonical text Python gave it, if Python read it as it is. Say whether it did. */
function takeCanonical(row) {
  if (!current()) return false;
  const entry = entryOf(row);
  if (!entry || entry.problem || field(row, "expression").value.trim() === entry.expression) return false;
  field(row, "expression").value = entry.expression;
  answeredNow();
  return true;
}

/** The limits changed: say so to the page, and have Python read them once the typing rests. */
function changed() {
  render();
  hooks.changed();
  schedule();
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(check, LIMITS_WAIT);
}

async function check() {
  timer = null;
  if (!needed()) {
    reply = null;
    answered = null;
    answeredLimits = null;
    render();
    renderPicture(null, [], false);
    return;
  }
  const sent = request();
  const key = JSON.stringify(sent);
  if (key === asking) return;
  if (key === answered) {
    ticket++; // back to what Python has answered: a reply on its way for something since is not wanted
    asking = null;
    return;
  }
  const mine = ++ticket;
  asking = key;
  const got = await hooks.call("api_limits", sent);
  if (mine !== ticket) return; // the limits changed again while Python read them
  asking = null;
  if (JSON.stringify(hooks.factors()) !== JSON.stringify(sent.factors)) return; // so did the factors: the next request is on its way
  const before = answeredLimits;
  reply = got;
  answered = key;
  answeredLimits = JSON.stringify({ factors: sent.factors, constraints: sent.constraints });
  for (const row of rows()) if (document.activeElement !== field(row, "expression")) takeCanonical(row);
  render();
  renderCorners(reply.ok ? reply.result : null, sent.factors);
  const pictured = reply.ok && reply.result.constraints.some((limit) => !limit.problem);
  renderPicture(reply.ok ? reply.result : null, sent.factors, pictured);
  if (answeredLimits !== before) hooks.changed(); // the marks alone change nothing the page shows
}

/** Show what Python said of each limit, and of all of them, beside the boxes. */
function render() {
  const limits = currentLimits();
  $("#limits-count").textContent = String(limits.length);
  $("#limits-count").hidden = limits.length === 0;
  renderNames();
  const fresh = current();
  for (const row of rows()) {
    const entry = fresh ? entryOf(row) : null;
    const problem = entry?.problem ?? null;
    const box = field(row, "expression");
    box.toggleAttribute("aria-invalid", Boolean(problem));
    row.querySelector(".limit-problem").textContent = problem ?? "";
    row.querySelector(".limit-problem").hidden = !problem;
    row.querySelector(".limit-effect").textContent = entry && !problem ? limitEffect(entry, reply.result.probes) : "";
    renderBounds(row, entry);
  }
  const summary = $("#limits-summary");
  summary.replaceChildren();
  if (fresh && hasLimits()) {
    const said = reply.ok ? limitsSummary(reply.result, hooks.factors()) : [{ text: reply.error, error: true }];
    for (const s of said) summary.append(el("span", s.error ? "error" : "", `${s.text} `));
  }
  summary.hidden = !summary.childNodes.length;
}

/** Show a straight limit's bound boxes, at its bounds, unless the reader is typing in one. */
function renderBounds(row, entry) {
  const linear = entry && !entry.problem ? entry.linear : null;
  const bounds = row.querySelector(".limit__bounds");
  bounds.hidden = !linear;
  if (!linear) return;
  for (const label of bounds.querySelectorAll(".limit__bound")) {
    const side = label.dataset.side;
    label.hidden = linear[side] === null;
    const box = field(label, side);
    if (document.activeElement !== box && linear[side] !== null) box.value = String(linear[side]);
  }
}

/** The buttons that type a factor's name, or at most and at least, at the caret: a phone has no easy <= to type. */
function renderNames() {
  const names = hooks.factors().filter((f) => !isCategorical(f) && f.name).map((f) => f.name);
  const holder = $("#limit-names");
  const shown = [...holder.querySelectorAll("button")].map((b) => b.dataset.insert);
  const wanted = [...names, "<=", ">="];
  holder.hidden = rows().length === 0;
  if (JSON.stringify(shown) === JSON.stringify(wanted)) return;
  holder.replaceChildren(
    ...wanted.map((piece) => {
      const button = el("button", "chip", piece === "<=" ? "≤" : piece === ">=" ? "≥" : piece);
      button.type = "button";
      button.dataset.insert = piece;
      button.setAttribute("aria-label", piece === "<=" ? "Type at most, <=" : piece === ">=" ? "Type at least, >=" : `Type ${piece}`);
      return button;
    }),
  );
}

/** Type a name or a sign at the caret of the box the reader was in, or the last box, and keep the keyboard there. */
function typeInto(piece) {
  const box = typingIn?.isConnected ? typingIn : rows().length ? field(rows().at(-1), "expression") : null;
  if (!box) return;
  const { text, caret } = insertAt(box.value, box.selectionStart ?? box.value.length, box.selectionEnd ?? box.value.length, piece);
  box.value = text;
  box.focus();
  box.setSelectionRange(caret, caret);
  changed();
}

/**
 * Wire the panel to the page: ``call`` reaches Python, ``factors`` reads the brief's, ``changed`` is
 * told of every change, and ``runs`` gives the run sheet's runs for the picture to draw.
 */
export function installLimits({ call, factors, changed: onChange, runs }) {
  hooks = { call, factors, changed: onChange };
  installPicture({ changed: schedule, runs });
  installCorners({
    changed: schedule,
    add: (expression) => {
      const row = addLimit({ expression });
      changed();
      field(row, "expression").focus(); // the limit is in the list now, where it can be edited as a typed one
    },
    full: () => rows().length >= MAX_CONSTRAINTS,
  });
  $("#limits").addEventListener("toggle", () => {
    if ($("#limits").open) schedule(); // the table of corners is drawn from Python's reply
  });
  $("#add-limit").onclick = () => {
    const row = addLimit();
    field(row, "expression").focus();
    changed();
  };
  $("#limit-list").addEventListener("input", (event) => {
    if (event.target.name === "expression" || event.target.name === "note") changed();
  });
  // A press on a button would take the focus from the box it types into; the button keeps it there instead.
  $("#limit-names").addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) event.preventDefault();
  });
  $("#limit-names").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button) typeInto(button.dataset.insert);
  });
  render();
}
