// Words with a definition, shown on request rather than in a sentence under every box. Such a word
// has a dotted underline, and its definition opens in a small pop-up beside it:
//
// - on a phone, when a finger rests on it (a press and hold), which does not also press the button
//   or the choice the word is part of;
// - with a mouse, when the pointer rests on it; the pointer can move onto the pop-up to read it;
// - from the keyboard, when the control the word names is reached with Tab or an arrow key;
// - for a word that stands alone, outside any control, with a tap, a click or Enter as well: it
//   is a button whose one task is its definition.
//
// Escape, a tap or click elsewhere, and scrolling close it. A screen reader reads each definition
// as the description of its control. The words and what they mean are in glossary.js; a
// definition that depends on the study, such as a design's or a measure's, is given with the word.

/** How long a finger rests on a word before its definition opens, and how far it may drift meanwhile. */
const HOLD_MS = 450;
const SLOP_PX = 10;
/** How long a mouse rests on a word before its definition opens, and how long it stays once the mouse leaves. */
const HOVER_MS = 300;
const LEAVE_MS = 200;
/** The click a press and hold ends in, within this long of the finger lifting, is not a tap. */
const SWALLOW_MS = 600;
/** The keys a reader moves through a page with: a definition follows the focus only after one of these. */
const MOVING = new Set(["Tab", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);

let glossary = {};
let tip = null; // the one pop-up every definition opens in
let shown = null; // the word whose definition is open
let hold = null; // a finger resting on a word: { word, x, y, timer }
let swallowUntil = 0;
let keyboard = false; // whether the reader last moved with the keyboard rather than a pointer
let hoverTimer = null;
let leaveTimer = null;
let holders = null; // the hidden definitions a control's aria-describedby names
const holderIds = new Map(); // a definition's text, to the id of the element that holds it

/**
 * A word or phrase with a definition: the glossary's, by ``key``, or ``definition`` as given. A word
 * that stands alone (``alone``), outside any control, is a button; one inside a control is a span.
 */
export function term(text, { key = null, definition = null, alone = false } = {}) {
  const word = document.createElement(alone ? "button" : "span");
  if (alone) word.type = "button";
  word.className = "term";
  word.textContent = text;
  if (key) word.dataset.term = key;
  else word.dataset.definition = definition ?? "";
  return word;
}

/**
 * Text with the glossary's words in it marked, as a list of strings and words for ``append``:
 * ``words`` maps each phrase to find to its glossary key. Longer phrases are found first, and only
 * whole words match.
 */
export function marked(text, words) {
  const phrases = Object.keys(words).sort((a, b) => b.length - a.length);
  if (!phrases.length) return [text];
  const escaped = phrases.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  // Whole words only: a phrase neither follows nor precedes a letter or a digit.
  const pattern = new RegExp(`(^|[^\\p{L}\\p{N}])(${escaped.join("|")})(?![\\p{L}\\p{N}])`, "u");
  const parts = [];
  let rest = text;
  for (let found = rest.match(pattern); found; found = rest.match(pattern)) {
    const start = found.index + found[1].length;
    if (start) parts.push(rest.slice(0, start));
    parts.push(term(found[2], { key: words[found[2]], alone: true }));
    rest = rest.slice(start + found[2].length);
  }
  if (rest) parts.push(rest);
  return parts;
}

const definitionOf = (word) => (word.dataset.term ? glossary[word.dataset.term] : word.dataset.definition) ?? "";

/** The word a control that has just been reached by keyboard names: itself, inside it, or in its label. */
function wordFor(control) {
  if (control.matches?.(".term")) return control;
  const inside = control.querySelector?.(".term");
  if (inside) return inside;
  for (const label of control.labels ?? []) {
    const word = label.querySelector(".term");
    if (word) return word;
  }
  return null;
}

/**
 * Put the pop-up beside its word, inside the window: above a word under a finger, which would
 * cover it otherwise, and below a word under the mouse or the keyboard; on the other side when
 * there is no room.
 */
function place(word, above) {
  const at = word.getBoundingClientRect();
  const { offsetWidth: w, offsetHeight: h } = tip;
  const up = at.top - 8 - h;
  const down = at.bottom + 8;
  const fitsUp = up >= 8;
  const fitsDown = down + h <= innerHeight - 8;
  const top = above ? (fitsUp || !fitsDown ? Math.max(8, up) : down) : fitsDown || !fitsUp ? down : up;
  tip.style.left = `${Math.max(8, Math.min(at.left + at.width / 2 - w / 2, innerWidth - w - 8))}px`;
  tip.style.top = `${top}px`;
}

/** Open ``word``'s definition; ``how`` it was asked for decides where it opens and whether it can be hovered. */
function show(word, how) {
  const text = definitionOf(word);
  if (!text) return;
  shown?.classList.remove("shown");
  shown = word;
  word.classList.add("shown");
  // Inside an open dialog the rest of the page is inert, so the pop-up goes into the dialog.
  const home = word.closest("dialog[open]") ?? document.body;
  if (tip.parentElement !== home) {
    if (tip.matches(":popover-open")) tip.hidePopover();
    home.append(tip);
  }
  tip.textContent = text;
  tip.dataset.how = how;
  if (!tip.matches(":popover-open")) tip.showPopover();
  place(word, how === "touch");
}

function hide() {
  clearTimeout(hoverTimer);
  clearTimeout(leaveTimer);
  if (!shown) return;
  shown.classList.remove("shown");
  shown = null;
  if (tip.matches(":popover-open")) tip.hidePopover();
}

function stopHold() {
  if (!hold) return;
  clearTimeout(hold.timer);
  hold = null;
}

/** Name ``word``'s definition as the description of the control it belongs to, for a screen reader. */
function describe(word) {
  const text = definitionOf(word);
  const control = word.matches("button") ? word : (word.closest("button, a") ?? word.closest("label")?.control);
  if (!text || !control) return;
  let id = holderIds.get(text);
  if (!id) {
    id = `definition-${holderIds.size + 1}`;
    holderIds.set(text, id);
    const holder = document.createElement("span");
    holder.id = id;
    holder.textContent = text;
    holders.append(holder);
  }
  const ids = new Set((control.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean));
  ids.add(id);
  control.setAttribute("aria-describedby", [...ids].join(" "));
}

/** Describe the words already on the page and any added later, as tables and results are drawn. */
function describeAll(root) {
  if (root.matches?.(".term")) describe(root);
  for (const word of root.querySelectorAll?.(".term") ?? []) describe(word);
}

/**
 * Open definitions on request across the page, from ``entries`` (glossary.js's GLOSSARY): one
 * pop-up for every word, and the listeners that open and close it. Call once.
 */
export function installTerms(entries) {
  glossary = entries;
  tip = document.createElement("div");
  tip.className = "term-tip";
  tip.setAttribute("role", "tooltip");
  tip.popover = "manual";
  document.body.append(tip);
  holders = document.createElement("div");
  holders.hidden = true;
  document.body.append(holders);
  describeAll(document.body);
  new MutationObserver((records) => {
    for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1) describeAll(node);
    if (shown && !shown.isConnected) hide(); // its word was drawn again, as a table is after a change
  }).observe(document.body, { childList: true, subtree: true });

  // A finger resting on a word opens its definition; one that moves, or lifts first, was a scroll or a tap.
  document.addEventListener(
    "pointerdown",
    (e) => {
      keyboard = false;
      if (shown && !tip.contains(e.target)) hide();
      const word = e.target.closest?.(".term");
      if (!word || e.pointerType === "mouse") return;
      stopHold();
      hold = {
        word,
        x: e.clientX,
        y: e.clientY,
        timer: setTimeout(() => {
          hold = null;
          swallowUntil = Infinity; // until the finger lifts
          show(word, "touch");
          navigator.vibrate?.(10);
        }, HOLD_MS),
      };
    },
    true,
  );
  document.addEventListener("pointermove", (e) => {
    if (hold && Math.hypot(e.clientX - hold.x, e.clientY - hold.y) > SLOP_PX) stopHold();
  });
  for (const type of ["pointerup", "pointercancel"]) {
    document.addEventListener(type, () => {
      stopHold();
      if (swallowUntil === Infinity) swallowUntil = performance.now() + SWALLOW_MS;
    });
  }
  // The press and hold is not also a tap on the choice or button the word belongs to, nor a menu.
  document.addEventListener(
    "click",
    (e) => {
      if (performance.now() < swallowUntil) {
        swallowUntil = 0;
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      const word = e.target.closest?.("button.term");
      if (!word) return;
      e.preventDefault();
      show(word, "pinned");
    },
    true,
  );
  document.addEventListener("contextmenu", (e) => {
    if (e.pointerType && e.pointerType !== "mouse" && e.target.closest?.(".term")) e.preventDefault();
  });

  // A mouse resting on a word opens its definition, which stays while the mouse is on the word or on it.
  document.addEventListener("pointerover", (e) => {
    if (e.pointerType !== "mouse") return;
    const word = e.target.closest?.(".term");
    if (word || tip.contains(e.target)) clearTimeout(leaveTimer);
    if (!word || word === shown) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => show(word, "mouse"), HOVER_MS);
  });
  document.addEventListener("pointerout", (e) => {
    if (e.pointerType !== "mouse" || !(e.target.closest?.(".term") || tip.contains(e.target))) return;
    clearTimeout(hoverTimer);
    if (shown && tip.dataset.how === "mouse") leaveTimer = setTimeout(hide, LEAVE_MS);
  });

  // The keyboard: a control reached with Tab or an arrow key opens the definition of the word it names.
  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key === "Escape" && shown) {
        hide();
        e.preventDefault(); // the first Escape closes the definition, the next the dialog it is in
        e.stopPropagation();
      } else if (MOVING.has(e.key)) {
        keyboard = true;
      }
    },
    true,
  );
  document.addEventListener("focusin", (e) => {
    if (!keyboard) return;
    const word = wordFor(e.target);
    if (word) show(word, "keyboard");
    else if (tip.dataset.how === "keyboard") hide();
  });
  document.addEventListener("focusout", () => {
    if (shown && tip.dataset.how === "keyboard") hide();
  });

  // A pop-up stays where it opened, so scrolling closes it; one that follows the keyboard follows its word.
  document.addEventListener(
    "scroll",
    () => {
      if (!shown) return;
      if (tip.dataset.how === "keyboard") place(shown, false);
      else hide();
    },
    { capture: true, passive: true },
  );
  window.addEventListener("resize", hide);
}
