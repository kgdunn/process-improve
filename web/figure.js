// web/figure.js: a figure in a window of its own, over the page, to look at closely: as large as the window allows,
// in a shape the reader chooses, with Plotly's buttons to zoom and turn it, and saved as an image in that shape.
// charts.js loads this module once a first chart is drawn and the browser is idle, so the page's start waits for none of it.

import { chartImage, draw, erase, figureOf } from "./charts.js";
import { closeOnBackdrop } from "./dialogs.js";

/** The shapes a figure can be given at a press, each as its width over its height; "Fill" takes all the room there is. */
export const SHAPES = { Fill: null, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1 };

/** The shape a figure opens in: near the one the library itself draws its figures in, 700 by 500. */
export const FIRST_SHAPE = "4:3";

/** The smallest a figure is drawn, in pixels, and how far one press of an arrow key resizes it. */
export const SMALLEST = { width: 240, height: 200 };
const STEP = 24;

/** The width a saved image is drawn at, in CSS pixels, as a report's charts are; and how many times that for print. */
const SAVED_WIDTH = 720;
const SAVED_SCALE = 3;

/** The largest box of ``ratio`` (width over height) that fits in ``room``; all of the room for no ratio. */
export function fitted(ratio, room) {
  if (!ratio) return { ...room };
  const width = Math.min(room.width, room.height * ratio);
  return { width: Math.round(width), height: Math.round(width / ratio) };
}

/** ``size`` kept no smaller than a figure can be read at, and no larger than ``room``. */
export function within(size, room) {
  const keep = (side) => Math.round(Math.max(Math.min(SMALLEST[side], room[side]), Math.min(size[side], room[side])));
  return { width: keep("width"), height: keep("height") };
}

/** A file's name for a figure, from its title: "Predictions at the corners" gives predictions-at-the-corners.png. */
export function fileName(title, format) {
  const stem = title
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${stem || "figure"}.${format}`;
}

/** The file a data URL holds: Plotly gives a PNG in base64, and an SVG as text. */
export function blobOf(dataUrl) {
  const [head, body] = [dataUrl.slice(5, dataUrl.indexOf(",")), dataUrl.slice(dataUrl.indexOf(",") + 1)];
  const type = head.split(";")[0];
  if (!head.endsWith(";base64")) return new Blob([decodeURIComponent(body)], { type });
  return new Blob([Uint8Array.from(atob(body), (letter) => letter.charCodeAt(0))], { type });
}

const MARKUP = `
  <div class="sheet__head">
    <h2 id="figure-title"></h2>
    <button type="button" class="link" data-do="close" aria-label="Close this figure">✕</button>
  </div>
  <div class="figure__tools">
    <span class="views" role="group" aria-label="The figure's shape">
      ${Object.keys(SHAPES)
        .map((name) => `<button type="button" data-shape="${name}" aria-pressed="false">${name}</button>`)
        .join("")}
    </span>
    <button type="button" class="link" data-do="png">Save image (.png)</button>
    <button type="button" class="link" data-do="svg">Save as vector (.svg)</button>
    <button type="button" class="link" data-do="copy">Copy image</button>
    <span class="hint" role="status"></span>
  </div>
  <div class="figure__frame">
    <div class="plot" role="img"></div>
    <button type="button" class="figure__grip" aria-label="Resize the figure: drag this corner, or press the arrow keys"></button>
  </div>
  <p class="hint figure__caption"></p>
  <details class="python">
    <summary>Show the Python code of this figure</summary>
    <p class="hint">The process-improve code that draws this figure, after the code of the analysis it draws:
      copy it and it runs as it is. The page then puts the axes in your own units and its own colours.</p>
    <pre><code></code></pre>
    <button type="button" class="link" data-do="code">Copy the code</button>
  </details>`;

let dialog = null;
let opened = null; // what the window shows: {title, size}
let shape = FIRST_SHAPE; // the shape last chosen, which the next figure opens in; null once one is resized by hand
let chosen = null; // the size last dragged to, when ``shape`` is null

const part = (selector) => dialog.querySelector(selector);

/**
 * The room a figure has in the window: the window's width inside its edges, and the height the figure has now
 * with what the window can still grow by, up to the browser's own height (style.css).
 */
function room() {
  const frame = part(".figure__frame");
  const edge = getComputedStyle(dialog);
  const inside = dialog.clientWidth - parseFloat(edge.paddingLeft) - parseFloat(edge.paddingRight);
  const borders = dialog.offsetHeight - dialog.clientHeight;
  const spare = parseFloat(edge.maxHeight) - borders - dialog.scrollHeight;
  return {
    width: Math.floor(inside - (frame.offsetWidth - frame.clientWidth)),
    height: Math.floor(frame.clientHeight + spare),
  };
}

/** Give the figure a size, within the room there is: the chart inside is drawn again to fit, by charts.js. */
function resize(size) {
  opened.size = within(size, room());
  Object.assign(part(".figure__frame").style, { width: `${opened.size.width}px`, height: `${opened.size.height}px` });
  for (const button of dialog.querySelectorAll("[data-shape]")) button.setAttribute("aria-pressed", String(button.dataset.shape === shape));
}

/** Give the figure the shape last chosen, as large as the room allows; or the size last dragged to. */
const reshape = () => resize(shape ? fitted(SHAPES[shape], room()) : chosen);

/** Resize by hand: the figure keeps that size, and no named shape. */
function drag(size) {
  shape = null;
  chosen = size;
  resize(size);
  chosen = opened.size;
}

const say = (text) => (part("[role=status]").textContent = text);

/** The figure as a file, in the shape the window shows it: on white, in the light theme, with its title. */
async function image(format) {
  const { width, height } = opened.size;
  const tall = Math.round((SAVED_WIDTH * height) / width);
  return blobOf(await chartImage(part(".plot"), SAVED_WIDTH, { height: tall, title: opened.title, format, scale: SAVED_SCALE }));
}

async function save(format) {
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(await image(format));
  anchor.download = fileName(opened.title, format);
  dialog.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 10_000);
  say(`Saved as ${anchor.download}.`);
}

async function copyImage() {
  try {
    // Safari takes the image only as a promise, made while the press is being handled.
    await navigator.clipboard.write([new ClipboardItem({ "image/png": image("png") })]);
    say("Copied: paste it into a slide or a document.");
  } catch {
    say("The browser would not copy the image: save it instead.");
  }
}

async function copyCode() {
  const code = part("code");
  try {
    await navigator.clipboard.writeText(code.textContent);
    say("Code copied: paste it into a script or a notebook.");
  } catch {
    getSelection().selectAllChildren(code); // for the reader to copy from the keyboard
    say("Selected: copy it with Ctrl+C or ⌘C.");
  }
}

const DOES = { close: () => dialog.close(), png: () => save("png"), svg: () => save("svg"), copy: copyImage, code: copyCode };

/** The corner that resizes the figure: dragged with a mouse, a finger or a pen, or moved with the arrow keys. */
function installGrip(grip) {
  let from = null;
  grip.addEventListener("pointerdown", (event) => {
    from = { x: event.clientX, y: event.clientY, ...opened.size };
    grip.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  grip.addEventListener("pointermove", (event) => {
    // The window stays in the middle of the screen, so its corner moves half as far as the figure grows.
    if (from) drag({ width: from.width + 2 * (event.clientX - from.x), height: from.height + 2 * (event.clientY - from.y) });
  });
  for (const end of ["pointerup", "pointercancel"]) grip.addEventListener(end, () => (from = null));
  const KEYS = { ArrowRight: [STEP, 0], ArrowLeft: [-STEP, 0], ArrowDown: [0, STEP], ArrowUp: [0, -STEP] };
  grip.addEventListener("keydown", (event) => {
    const [wider, taller] = KEYS[event.key] ?? [];
    if (wider === undefined) return;
    event.preventDefault();
    drag({ width: opened.size.width + wider, height: opened.size.height + taller });
  });
}

function build() {
  dialog = document.createElement("dialog");
  dialog.className = "sheet figure";
  dialog.setAttribute("aria-labelledby", "figure-title");
  dialog.innerHTML = MARKUP;
  document.body.append(dialog);
  closeOnBackdrop(dialog);
  installGrip(part(".figure__grip"));
  dialog.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (button?.dataset.shape) {
      shape = button.dataset.shape;
      reshape();
    } else if (button?.dataset.do) {
      DOES[button.dataset.do]();
    }
  });
  // The chart is taken down with the window: a browser keeps only so many charts in three dimensions at once.
  dialog.addEventListener("close", () => erase(part(".plot")));
  window.addEventListener("resize", () => dialog.open && reshape());
  part("[data-do=copy]").hidden = !(window.ClipboardItem && navigator.clipboard?.write);
}

/**
 * Open the chart drawn in ``div`` in a window of its own, under ``title`` and over its ``caption``. It is drawn
 * again there from what the page drew it from, with ``python``, the code that draws it, when it has some.
 */
export async function openFigure(div, { title, caption = "" }) {
  const shown = figureOf(div);
  if (!shown) return;
  if (!dialog) build();
  opened = { title };
  part("#figure-title").textContent = title;
  part(".figure__caption").textContent = caption;
  part(".plot").setAttribute("aria-label", title);
  const code = shown.python?.() ?? "";
  part("details.python").hidden = !code;
  part("details.python").open = false;
  part("code").textContent = code;
  say("");
  dialog.showModal();
  reshape();
  await draw(part(".plot"), shown.make, shown.version, null, { toolbar: true });
}
