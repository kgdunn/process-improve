// The reproduction page (catalog/reproduced/index.html, at factori.al/catalog/reproduced/): it reads
// catalog/reproduction.json once and draws how many printed numbers Factorial gives as its sources print them:
// one bar of them all, the reasons the others are no match as a cascade, each with an example, and a row for
// each source. Behind "Search every case" are the cases, in a list for each state, and what the library lacks.
// The rules are in reproduced-logic.js; this module draws them. No Python runs here.

import { element } from "./catalog-views.js";
import {
  CASE_STATES,
  CAUSES,
  DEFAULT_VIEW,
  REASONS,
  STATES,
  byForm,
  bySource,
  byState,
  caseState,
  cascade,
  examples,
  missingByReason,
  filterCases,
  formatError,
  fromHash,
  groupByState,
  kindOf,
  kinds,
  percent,
  quantity,
  sizes,
  tally,
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
/** The form of source the list of sources keeps: every form while it is empty. */
let shelfForm = "";
/** The sources whose rows are open: at first the one the page has a table of its book for. */
const openRows = new Set();
/** Draws a state's list into its box, once: the page calls it when it opens a list itself, not waiting for the toggle. */
const fillers = new Map();

const count = (n) => n.toLocaleString("en");
const plural = (n, one, many = `${one}s`) => `${count(n)} ${n === 1 ? one : many}`;
const sourceName = (key) => {
  const source = data.sources[key] ?? {};
  return source.year ? `${source.short ?? key} (${source.year})` : (source.short ?? key);
};

/** A state's mark: a square in its color, hatched or empty for the two states that have no color. */
const stateKey = (state) => element("span", { class: "statekey", "data-state": state, "aria-hidden": "true" });

/** The cases the reader's search, source and kind keep. */
const shown = () => filterCases(data.cases, data.sources, view, { names: data.quantities });

/** The causes the library is right about, from the JSON (see libraryRight). */
const right = () => data.library_right;

// ---------------------------------------------------------------- the bar of every number
/** A legend's line: a state's mark, its name and how many numbers are in it. */
const stateLine = (state, n, name = STATES.find((s) => s.key === state).label) =>
  element("li", {}, stateKey(state), element("span", {}, `${name} `, element("strong", { text: count(n) })));

/**
 * The headline and its bar, for all the sources together: they never follow the filters, so no count is one
 * source's. The bar's end, past the numbers that match, opens into the reasons drawn under it.
 */
function drawWhole() {
  const all = tally(data.cases);
  const states = byState(all);
  $("#tally-value").textContent = `${count(all.match)} of ${count(all.checked)}`;
  const held = STATES.filter((state) => states[state.key]);
  const bar = $("#wholebar");
  bar.replaceChildren(
    ...held.map((state) => {
      const text = state.key === "match" ? `${percent(all.match, all.checked)} match` : null;
      const part = element("span", { class: "wholebar__part", "data-state": state.key, title: `${state.label}: ${count(states[state.key])}` }, text);
      part.style.flexGrow = String(states[state.key]);
      return part;
    }),
  );
  bar.setAttribute("aria-label", held.map((state) => `${state.label}: ${count(states[state.key])}`).join("; "));
  const from = (1000 * all.match) / (all.checked + states.gap);
  $("#zoomin polygon").setAttribute("points", `${from},0 1000,0 1000,34 0,34`);
  $("#zoomin line").setAttribute("x2", from);
}

// ---------------------------------------------------------------- the reasons, as a cascade
/** Play the turn of an example's box again, for the next example. */
function flip(box) {
  box.classList.remove("example__box--flip");
  box.getBoundingClientRect(); // the class taken off is seen, so putting it back starts the turn again
  box.classList.add("example__box--flip");
}

/**
 * One number of a reason, in a box: where it is printed, what the source prints and what Factorial gives.
 * "See another" puts the next number of the reason in the box, and after the last the first again.
 */
function exampleOf(step) {
  const list = examples(data.cases, step.cause, data.quantities);
  if (!list.length) return null;
  const box = element("div", { class: "example__box", "aria-live": "polite" });
  let at = 0;
  const show = () => {
    const eg = list[at];
    box.replaceChildren(
      element("p", { class: "example__where", text: [`${sourceName(eg.source)}, ${eg.where}`, eg.what].filter(Boolean).join(" · ") }),
      element(
        "dl",
        { class: "example__values" },
        element("div", {}, element("dt", { text: step.state === "quiz" ? "A wrong option" : "The book" }), element("dd", { text: eg.printed })),
        element("div", {}, element("dt", { text: "Factorial" }), element("dd", { text: eg.found ?? "no method yet" })),
      ),
    );
  };
  show();
  const more = element("button", {
    type: "button",
    class: "link example__more",
    text: "See another",
    onclick: () => {
      at = (at + 1) % list.length;
      show();
      flip(box);
    },
  });
  return element(
    "div",
    { class: "example" },
    element("div", { class: "example__side" }, element("span", { class: "example__label", text: "Example:" }), list.length > 1 ? more : null),
    box,
  );
}

/**
 * The reasons a number is no match, for all the sources together: a row each, with its count, an example and
 * its bar, which starts where the bar of the row before ended. Under them an axis says how many differ.
 */
function drawReasons() {
  const all = tally(data.cases);
  const states = byState(all);
  const { steps, whole, differ } = cascade(all);
  const gap = whole - differ;
  $("#reasons-title").textContent =
    `The ${plural(differ, "number")} that ${differ === 1 ? "differs" : "differ"}` +
    (gap ? `, and ${count(gap)} more that ${gap === 1 ? "is" : "are"} not in Factorial yet` : "");
  $("#reasons-key").replaceChildren(...STATES.filter((s) => s.key !== "match" && states[s.key]).map((s) => stateLine(s.key, states[s.key])));
  const rows = steps.map((step, i) => {
    const bar = element("span", { class: "reason__bar", "data-state": step.state });
    bar.style.left = `${100 * step.from}%`;
    bar.style.width = `${100 * step.share}%`;
    const link = element("span", { class: "reason__link" });
    link.style.left = `${100 * (step.from + step.share)}%`;
    return element(
      "div",
      { class: "reason", "data-state": step.state, "data-cause": step.cause },
      element("div", { class: "reason__text" }, element("h3", { class: "reason__name" }, step.name, element("span", { class: "badge", text: count(step.n) })), exampleOf(step)),
      element("div", { class: "reason__track", "aria-hidden": "true" }, bar, i < steps.length - 1 ? link : null),
    );
  });
  const mid = gap && differ ? element("span", { class: "reason__mid", text: `${count(differ)} differ` }) : null;
  if (mid) mid.style.left = `${(100 * differ) / whole}%`;
  const axis = element(
    "div",
    { class: "reason reason--axis", "aria-hidden": "true" },
    element("div", { class: "reason__text" }),
    element("div", { class: "reason__track" }, element("span", { text: "0" }), mid, element("span", { text: count(whole) })),
  );
  $("#reasons").replaceChildren(...rows, axis);
  const far = sizes(data.cases, data.tiers, right());
  const of = Object.values(far).reduce((a, b) => a + b, 0);
  $("#sizes").textContent = of
    ? `Sizes of the ${plural(of, "difference")} that ${of === 1 ? "is" : "are"} neither a misprint nor a quiz option: ` +
      `${count(far.within_1)} at 1% of the printed value or less, ${count(far.within_5)} more at 5% or less.`
    : "";
}

// ---------------------------------------------------------------- a row for each source
/**
 * A source's row: its name, its numbers recomputed, a bar of them by state and the share that match. It opens
 * onto its counts by reason, how many of its cases match in every number, and the way to its own page.
 */
function shelfRow(row) {
  const counts = tally(data.cases.filter((c) => c.source === row.key));
  const states = byState(counts);
  const facts = [["match", counts.match, STATES[0].label], ...REASONS.map((r) => [r.state, counts.causes[r.cause], r.name])].filter(([, n]) => n);
  const bar = element("span", { class: "statebar", "aria-hidden": "true" });
  for (const state of STATES) {
    if (!states[state.key]) continue;
    const part = element("span", { class: "statebar__part", "data-state": state.key });
    part.style.flexGrow = String(states[state.key]);
    bar.append(part);
  }
  const table = $("#receipt");
  const item = element(
    "details",
    { class: "shelf__row", "data-source": row.key, open: openRows.has(row.key) },
    element(
      "summary",
      { "aria-label": `${sourceName(row.key)}: ${plural(row.numbers, "number")}. ${facts.map(([, n, name]) => `${name}: ${count(n)}`).join("; ")}.` },
      element("span", {}, element("span", { class: "shelf__name", text: `${row.short} · ${row.year}` }), element("span", { class: "shelf__title", text: row.title })),
      element("span", { class: "shelf__numbers", text: count(row.numbers) }),
      bar,
      element("span", { class: "shelf__share", text: percent(counts.match, counts.checked) }),
      element("span", { class: "shelf__more", "aria-hidden": "true" }),
    ),
    element(
      "div",
      { class: "shelf__body" },
      element("ul", { class: "states" }, ...facts.map(([state, n, name]) => stateLine(state, n, name))),
      element("p", { class: "hint", text: `Cases that match in every number: ${count(row.states.match)} of ${count(row.total)}.` }),
      table?.dataset.source === row.key ? table.content.cloneNode(true) : null,
      element("p", {}, element("a", { href: `../books/${row.key}/`, text: "Every example of this source, with each difference" })),
    ),
  );
  item.addEventListener("toggle", () => (item.open ? openRows.add(row.key) : openRows.delete(row.key)));
  return item;
}

/** The sources with a recomputed case, in the order chosen, those of one form when a form is chosen. */
function drawShelf() {
  const rows = bySource(data.cases, data.sources, sourceOrder);
  const form = (key, text) =>
    element("button", {
      type: "button",
      class: "shelf__form",
      "aria-pressed": String(shelfForm === key),
      text,
      onclick: () => {
        shelfForm = key;
        drawShelf();
      },
    });
  $("#forms").replaceChildren(form("", `All ${count(rows.length)}`), ...byForm(rows, data.forms).groups.map((g) => form(g.form, `${g.label} ${count(g.rows.length)}`)));
  $("#shelf").replaceChildren(...rows.filter((row) => !shelfForm || row.form === shelfForm).map(shelfRow));
  $("#books-section").hidden = rows.length === 0;
}

/** Open the cases when the address names a search, a source, a kind, a list or a case: a book's page leads here. */
function openCases() {
  if (view.query || view.source || view.form || view.kind || view.open.length || view.focus) $("#numbers").open = true;
}

// ---------------------------------------------------------------- the lists of cases
function numberCard(number) {
  const printed = Array.isArray(number.printed) ? JSON.stringify(number.printed) : number.printed;
  const what = quantity(number, data.quantities);
  return element(
    "li",
    { class: "number" },
    element(
      "div",
      { class: "number__head" },
      element("span", { class: "number__what", text: what ? `${what[0].toUpperCase()}${what.slice(1)}` : "A printed number" }),
      element(
        "span",
        { class: "number__tags" },
        element("span", { class: "number__cause", "data-cause": number.cause, text: CAUSES.differs[number.cause] ?? "" }),
        element("span", { class: "number__error", text: formatError(number.error) }),
      ),
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
        // How far its worst number is off, leaving out those the library is right about (the case's tier).
        c.tier === "reproduced" ? null : element("span", { class: "number__error", text: `worst ${formatError(c.worst)}` }),
        element("span", { text: facts }),
        element("span", { class: "case__kind", text: kindOf(c) }),
      ),
    ),
    body,
  );
  // The numbers are drawn when the case opens: a list can hold hundreds of cases.
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
  url.hash = toHash({ ...DEFAULT_VIEW, open: [caseState(c)], focus: c.id });
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
  // The cases of a state are grouped by source in the order of the table by source: by form, then oldest first.
  const order = byForm(bySource(data.cases, data.sources), data.forms).groups.flatMap((g) => g.rows.map((row) => row.key));
  $("#lists").replaceChildren(
    ...CASE_STATES.map((s) => {
      const n = cases.filter((c) => caseState(c) === s.key).length;
      const body = element("div", { class: "statelist__body" });
      const box = element(
        "details",
        { class: "statelist", id: `state-${s.key}`, "data-state": s.key },
        element(
          "summary",
          {},
          stateKey(s.key),
          element("span", { class: "statelist__name", text: s.label }),
          element("span", { class: "badge", text: plural(n, "case") }),
          element("span", { class: "statelist__means hint", text: s.means }),
        ),
        body,
      );
      const fill = () => {
        if (body.childElementCount) return;
        const groups = groupByState(cases, s.key, order);
        body.append(...(groups.length ? groups.map(sourceGroup) : [element("p", { class: "hint", text: "No case your filters keep is in this list." })]));
      };
      fillers.set(s.key, fill);
      box.addEventListener("toggle", () => {
        if (box.open) fill();
        const open = new Set(view.open);
        if (box.open) open.add(s.key);
        else open.delete(s.key);
        view.open = CASE_STATES.map((state) => state.key).filter((key) => open.has(key));
        history.replaceState(null, "", toHash(view) || location.pathname);
      });
      if (view.open.includes(s.key)) {
        box.open = true;
        fill();
      }
      return box;
    }),
  );
}

/** Open the case the address names, with its list, and bring it into view. */
function focusCase() {
  const c = data.cases.find((x) => x.id === view.focus && x.tier);
  if (!c) return;
  const box = $(`#state-${caseState(c)}`);
  box.open = true;
  fillers.get(caseState(c))();
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

// ---------------------------------------------------------------- not in the library yet
/** How many of the reasons the list shows at first, and adds with each press of "Show more". */
const MISSING_SHOWN = 20;
let missingShown = MISSING_SHOWN;

/** One thing the library lacks: the reason, how many numbers and cases wait on it, and the cases themselves. */
function missingItem(group) {
  const cases = element(
    "ul",
    { class: "missing__cases" },
    ...group.cases.map((c) =>
      element(
        "li",
        {},
        element("a", { href: app(`catalog/#d=${encodeURIComponent(c.id)}`), text: c.title }),
        element("span", { class: "hint", text: ` ${sourceName(c.source)}` }),
      ),
    ),
  );
  return element(
    "li",
    {},
    element(
      "details",
      { class: "missing__item" },
      element(
        "summary",
        {},
        element("span", { class: "missing__reason", text: group.reason }),
        element("span", { class: "badge", text: `${plural(group.numbers, "number")}, ${plural(group.cases.length, "case")}` }),
      ),
      cases,
    ),
  );
}

/** The list of what the library lacks, for the cases the reader's filters keep, those with a recomputed number or not. */
function drawMissing() {
  const groups = missingByReason(filterCases(data.cases, data.sources, view, { all: true, names: data.quantities }));
  const numbers = groups.reduce((n, g) => n + g.numbers, 0);
  $("#missing-count").textContent = groups.length
    ? `${plural(numbers, "printed number")}, for ${plural(groups.length, "reason")}.`
    : "Every printed number of the cases your filters keep that can be recomputed is.";
  $("#missing").replaceChildren(...groups.slice(0, missingShown).map(missingItem));
  $("#missing-more").hidden = groups.length <= missingShown;
  $("#missing-more").textContent = `Show ${count(Math.min(MISSING_SHOWN, groups.length - missingShown))} more of ${count(groups.length - missingShown)}`;
}

// ---------------------------------------------------------------- the reader's view
function draw() {
  const cases = shown();
  drawLists(cases);
  drawMissing();
  $("#query").value = view.query;
  $("#source").value = view.form ? `form:${view.form}` : view.source;
  $("#kind").value = view.kind;
  $("#clear").hidden = !(view.query || view.source || view.form || view.kind);
  const all = data.cases.filter((c) => c.tier).length;
  $("#count").textContent =
    cases.length === all ? `${plural(all, "case")}, in four lists.` : `${plural(cases.length, "case")} of ${count(all)} kept.`;
}

function setView(next) {
  view = { ...structuredClone(DEFAULT_VIEW), ...next };
  missingShown = MISSING_SHOWN;
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
  $("#missing-more").addEventListener("click", () => {
    missingShown += MISSING_SHOWN;
    drawMissing();
  });
  // A word of the reasons the catalog records: the search keeps the cases whose numbers differ for it.
  for (const button of document.querySelectorAll(".try")) {
    button.addEventListener("click", () => {
      const open = CASE_STATES.slice(1).map((s) => s.key);
      setView({ ...view, query: button.textContent, open, focus: "" });
    });
  }
  $("#source-order").addEventListener("change", (event) => {
    sourceOrder = event.target.value;
    drawShelf();
  });
  window.addEventListener("hashchange", () => {
    view = fromHash(location.hash, data.sources, data.forms);
    openCases();
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
  for (const id of ["#summary", "#numbers"]) $(id).hidden = false;
  $("#built").textContent = `Recomputed with process-improve ${data.library}.`;
  const table = $("#receipt");
  if (table) openRows.add(table.dataset.source);
  drawWhole();
  drawReasons();
  drawShelf();
  openCases();
  wire();
  draw();
  focusCase();
}

main();
