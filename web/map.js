// The concept map (map/index.html): the ideas of designed experiments in Process Improvement using Data,
// and how they link. Zoomed out it shows the topics; zooming in opens the larger ones into sub-topics, then names their
// terms; a tapped term shows its definition, the line of the book it comes from, and its links.
//
// map/graph.json is built and checked against the book in pid-book's knowledge-graph/ (build_kg.py, layout_kg.py),
// where every term quotes the line of the book it comes from: to update the map, copy doe_kg_map.json from there.
// This is the graph's only viewer: pid-book holds the data, and a change to how the map looks is made here alone.
import cytoscape from "./vendor/cytoscape.mjs";
import { MODES, nextMode } from "./logic.js";
import { installFeedback } from "./feedback.js";

const DATA = await fetch(new URL("map/graph.json", import.meta.url)).then((r) => {
  if (!r.ok) throw new Error(`map/graph.json: ${r.status}`);
  return r.json();
});
// The canvas draws its text in the page's own typeface.
const FONT = getComputedStyle(document.body).fontFamily;
const FAMILY = { DesignType: "design", Property: "design", Criterion: "design", Method: "method", Plot: "method", Phase: "method",
  Concept: "concept", Guideline: "guide", Diagnostic: "guide", Example: "context", Question: "context", Reference: "context" };
const FAMILIES = {
  design: ["Designs and their properties", "round-rectangle", "M3 3h14v14H3z", "Designs"],
  method: ["Methods, plots, phases", "diamond", "M10 1.5 18.5 10 10 18.5 1.5 10z", "Methods"],
  concept: ["Concepts", "ellipse", "M10 2a8 8 0 1 1 0 16a8 8 0 1 1 0-16z", "Concepts"],
  guide: ["Guidelines and warning signs", "triangle", "M10 2 18.5 17.5H1.5z", "Guides"],
  context: ["Examples, questions, sources", "hexagon", "M5.5 2.5h9L19 10l-4.5 7.5h-9L1 10z", "Examples"],
};
// How each verb reads from its source ([0]) and from its target ([1]); the verbs are those build_kg.py allows.
const READ = {
  IS_A: ["is a kind of", "kinds"], PART_OF: ["part of", "has parts"], HAS_PROPERTY: ["has property", "property of"],
  USES: ["uses", "used by"], REQUIRES: ["requires", "required by"], JUSTIFIES: ["justifies", "justified by"],
  ESTIMATES: ["estimates", "estimated by"], MEASURES: ["measures", "measured by"], SHOWS: ["shows", "shown by"],
  LOCATES: ["locates", "located by"], TESTS: ["tests", "tested by"], CAUSES: ["causes", "caused by"],
  ENABLES: ["enables", "enabled by"], LIMITS: ["limits", "limited by"], PREVENTS: ["prevents", "prevented by"],
  KEEPS: ["keeps", "kept by"], REMEDIED_BY: ["remedied by", "remedies"], INDICATES: ["indicates", "indicated by"],
  AGREES_WITH: ["agrees with", "agrees with"], CONTRASTS_WITH: ["contrasts with", "contrasts with"],
  ALTERNATIVE_TO: ["alternative to", "alternative to"], COMPLEMENTS: ["complements", "complemented by"],
  GENERALIZES: ["generalizes", "generalized by"], CAN_AUGMENT_TO: ["can be augmented to", "augments"],
  CAN_BE: ["can be", "a possible form of"], SUITED_TO: ["suited to", "served by"], APPLIES_TO: ["applies to", "addressed by"],
  PRECEDES: ["comes before", "comes after"], ILLUSTRATES: ["illustrates", "illustrated by"],
  ANSWERED_BY: ["answered by", "answers"], DESCRIBED_IN: ["described in", "describes"],
  CANNOT_SHOW: ["cannot show", "not shown by"], EQUALS: ["equals", "equals"], IS_HALF_OF: ["is half of", "is twice"],
  MAY_HAVE: ["may have", "may belong to"],
};
const read = (verb, i) => (READ[verb] || [verb.toLowerCase().replaceAll("_", " "), `${verb.toLowerCase().replaceAll("_", " ")} (from)`])[i];
// Every link bows a little to the right of the way it reads: lines that would cross at a shallow angle part, and the
// map reads as less of a tangle. Its control point sits BOW of the link's length off the straight line (the curve
// itself reaches half that), so long links bow more than short ones and every link keeps the same shape.
const BOW = 0.14;
const ARC = { "curve-style": "unbundled-bezier", "control-point-distances": "data(bow)", "control-point-weights": 0.5 };
const bowOf = (a, b) => BOW * Math.hypot(b.x - a.x, b.y - a.y);
// A name sits on the map with no box behind it: a thin halo in the map's own colour cuts the lines just short of its
// letters, so it reads where lines cross it and hides nothing else.
const HALO = px => ({ "text-outline-color": css("--map-bubble"), "text-outline-width": px(2), "text-outline-opacity": 0.7 });
const TERM_GAP = 62;   // least distance between two terms in the drawing (layout_kg.py)
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const terms = new Map(DATA.nodes.map(n => [n.id, n]));
const topics = new Map(DATA.clusters.map(c => [c.id, c]));
// Sub-topics: the second level inside the larger topics; each knows its topic.
const subtopics = new Map(DATA.clusters.flatMap(c => (c.subs || []).map(s => [s.id, { ...s, topic: c.id }])));
const groupOf = id => topics.get(id) || subtopics.get(id);   // a topic or a sub-topic, or undefined
const fam = n => FAMILY[n.type] || "concept";
const icon = (f, color) => `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="${FAMILIES[f][2]}" fill="${color}"/></svg>`;
const esc = s => s.replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
const bookUrl = n => {
  const page = `https://learnche.org/pid/${n.book_file.replace(/\.rst$/, "")}`;
  return n.book_label ? `${page}#${n.book_label.toLowerCase().replaceAll("_", "-")}` : page;
};
const linksOf = id => DATA.edges.filter(e => e.source === id || e.target === id);

// Reading level: Beginner, Intermediate or Advanced; each shows every level below it (the book's build sets each
// term's level from where chapter 5 introduces it). A first visit opens on Beginner; the reader's choice is kept.
const LEVELS = ["Beginner", "Intermediate", "Advanced"];
const leveled = DATA.nodes.some(n => n.level);
let level = leveled ? 1 : LEVELS.length;
try { const v = Number(localStorage.getItem("factorial-map-level")); if (leveled && v >= 1 && v <= LEVELS.length) level = v; } catch {}
const shown = n => (n.level || LEVELS.length) <= level;

// Kinds of term hidden from the map, remembered in this browser. The selected term always stays visible.
const hidden = new Set();
try { for (const k of JSON.parse(localStorage.getItem("factorial-map-hidden") || '["context"]')) if (k in FAMILIES) hidden.add(k); } catch {}

// The terms drawn: those at the level and of the kinds chosen, plus the selected one, less any left with no link to
// another drawn term. One pass is enough: dropping a term with no drawn link takes no link from any other. Every
// count on the map reads this same set, so a topic never counts a term it does not draw.
let onMap = new Set();
function drawnTerms(sel) {
  const kept = new Set(DATA.nodes.filter(n => n.id === sel || (shown(n) && !hidden.has(fam(n)))).map(n => n.id));
  const linked = new Set(sel ? [sel] : []);
  for (const e of DATA.edges) if (kept.has(e.source) && kept.has(e.target)) { linked.add(e.source); linked.add(e.target); }
  return new Set([...kept].filter(id => linked.has(id)));
}
onMap = drawnTerms(null);
const sizeAt = id => DATA.nodes.filter(n => (n.cluster === id || n.sub === id) && onMap.has(n.id)).length;
const drawLegend = () => {
  document.getElementById("legend").innerHTML = `<span>Show:</span>` + Object.entries(FAMILIES)
    .map(([k, [label, , , short]]) => `<button type="button" data-fam="${k}" aria-pressed="${!hidden.has(k)}"
      title="${hidden.has(k) ? "Show" : "Hide"} ${label.toLowerCase()}">${icon(k, css(`--map-${k}`))}<span class="map-full">${label}</span><span class="map-short" aria-hidden="true">${short}</span></button>`).join("");
};

// ---------- the colour mode: light, dark or the system's, in turn, from the same switch as the start page's
const MODE_KEY = "factorial-mode";
const MODE_NAMES = { auto: "the system's", light: "light", dark: "dark" };
const systemDark = matchMedia("(prefers-color-scheme: dark)");
function applyMode(mode) {
  const root = document.documentElement;
  root.dataset.mode = mode;
  if (mode === "auto") delete root.dataset.theme;
  else root.dataset.theme = mode;   // the map redraws in its colours (the observer below)
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    const { scheme } = meta.dataset;
    meta.media = mode === "auto" ? `(prefers-color-scheme: ${scheme})` : scheme === mode ? "all" : "not all";
  }
  document.getElementById("mode").setAttribute("aria-label", `Colour mode: ${MODE_NAMES[mode]}`);
}
document.getElementById("mode").addEventListener("click", () => {
  const mode = nextMode(document.documentElement.dataset.mode, systemDark.matches);
  try {
    if (mode === "auto") localStorage.removeItem(MODE_KEY);
    else localStorage.setItem(MODE_KEY, mode);
  } catch {
    /* storage refused: the mode holds for this visit */
  }
  applyMode(mode);
});
applyMode(MODES.includes(document.documentElement.dataset.mode) ? document.documentElement.dataset.mode : "auto");

// ---------- the map
const elements = [
  ...DATA.clusters.map(c => ({ group: "nodes", classes: "topic", position: { x: c.x, y: c.y },
    data: { id: c.id, label: c.short, olabel: `${c.short}\n[${sizeAt(c.id)}]`, d: 2 * c.r } })),
  ...[...subtopics.values()].map(s => ({ group: "nodes", classes: "subtopic", position: { x: s.x, y: s.y },
    data: { id: s.id, label: s.name, d: 2 * s.r } })),
  // The same names as markers of their own, drawn above terms and links while the names sit mid-circle.
  ...[...subtopics.values()].map(s => ({ group: "nodes", classes: "subname", position: { x: s.x, y: s.y },
    data: { id: `${s.id}-name`, label: `${s.name}\n[${sizeAt(s.id)}]`, w: 1.6 * s.r } })),
  ...DATA.nodes.map(n => ({ group: "nodes", classes: `term fam-${fam(n)}`, position: { x: n.x, y: n.y },
    data: { id: n.id, label: n.name, rank: n.rank, topic: n.cluster, sub: n.sub, fam: fam(n), level: n.level || LEVELS.length } })),
  ...DATA.edges.map((e, i) => ({ group: "edges", data: { id: `l${i}`, source: e.source, target: e.target, verb: read(e.type, 0), at: 0,
    bow: bowOf(terms.get(e.source), terms.get(e.target)) },
    classes: terms.get(e.source).cluster === terms.get(e.target).cluster ? "link" : "link cross" })),
  ...DATA.cluster_links.map((e, i) => ({ group: "edges", classes: "between",
    data: { id: `b${i}`, source: e.a, target: e.b, n: e.n, bow: bowOf(topics.get(e.a), topics.get(e.b)) } })),
];
const cy = cytoscape({ container: document.getElementById("cy"), elements, layout: { name: "preset" }, autoungrabify: true,
  autounselectify: true, boxSelectionEnabled: false, wheelSensitivity: 0.35, pixelRatio: "auto" });
// The map and its data, for the tests that drive this page (tests/web/e2e/test_concept_map.py) and for the console.
window.conceptMap = { cy, data: DATA };

let selected = null;   // a term id, a topic id, or null for the overview
let labelsOn = true;   // term names on the map; with them off, only the selected term is named
try { labelsOn = localStorage.getItem("factorial-map-labels") !== "off"; } catch {}
let stripWas = null;   // the strip's own text, while it reads out a link
let verbsOn = false;   // each link's verb along its line, with an arrow to show which way it reads
try { verbsOn = localStorage.getItem("factorial-map-verbs") === "on"; } catch {}
const DETAIL_GAP = 19;   // on-screen distance between terms at which they, and the sub-topics, appear
const SUB_LABEL_GAP = 45;   // ... and at which sub-topic names move to their edge, and their terms are named

// An on-screen size: `base` where the terms appear, then growing more slowly than the map as you zoom in.
function grow(z, base, power, max) {
  return Math.min(max, Math.max(base, base * Math.pow(z * TERM_GAP / DETAIL_GAP, power)));
}
// Label font, term dot and link width on screen; slow enough to keep growing up to the deepest zoom.
const sizes = z => ({ font: grow(z, 11, 0.25, 19), dot: grow(z, 7, 0.5, 22), line: grow(z, 1, 0.4, 2.6) });

function sheet(z) {
  const px = v => v / z;                 // screen pixels to drawing units
  const overview = TERM_GAP * z < DETAIL_GAP;
  const middle = !overview && TERM_GAP * z < SUB_LABEL_GAP;   // sub-topics named, their terms not yet
  const { font, dot, line } = sizes(z);
  const size = px(dot);
  // An arrowhead `length` px long on a line `width` px wide. Cytoscape draws it 0.3 x max((13.37 x width)^0.9, 29) drawing
  // units long, a floor that grows on screen with the zoom, so the scale is solved for the length wanted.
  const arrow = (width, length) => length / (0.3 * Math.max(Math.pow(13.37 * px(width), 0.9), 29) * z);
  const famRules = Object.entries(FAMILIES).map(([k, [, shape]]) =>
    ({ selector: `.fam-${k}`, style: { shape, "background-color": css(`--map-${k}`) } }));
  return [
    { selector: "node, edge", style: { "z-index-compare": "manual" } },
    { selector: ".topic", style: { shape: "ellipse", width: "data(d)", height: "data(d)", "background-color": css("--map-bubble"),
      "background-opacity": overview ? 0.9 : 0.55, "border-width": px(1), "border-color": css("--line"), "z-index": 0,
      events: "no",   // a drag on a circle must pan the map; a tap inside one is matched in the background handler

      label: overview ? "data(olabel)" : "data(label)", "font-family": FONT, "font-weight": 600,
      color: overview ? css("--fg") : css("--muted"), "font-size": px(overview ? 13 : grow(z, 12, 0.3, 17)), "text-wrap": "wrap",
      "text-max-width": overview ? "data(d)" : px(220), "text-valign": overview ? "center" : "top",
      "text-margin-y": overview ? 0 : px(-4), "text-opacity": overview ? 1 : 0.9,
      "text-background-color": css("--map-bubble"), "text-background-opacity": overview ? 0.92 : 0,
      "text-background-padding": px(3), "text-background-shape": "roundrectangle" } },
    { selector: ".topic.picked", style: { "border-width": px(2.5), "border-color": css("--accent") } },
    // Between the topics and the terms: sub-topics, named in their middle; zoom on and the names move to their edge.
    { selector: ".subtopic", style: { display: overview ? "none" : "element", shape: "ellipse", width: "data(d)",
      height: "data(d)", "background-color": css("--map-bubble"), "background-opacity": 0.5, "border-width": px(1),
      "border-style": "dashed", "border-color": css("--line"), "z-index": 1, events: "no", label: middle ? "" : "data(label)",
      "font-family": FONT, "font-weight": 600,
      color: middle ? css("--fg") : css("--muted"), "font-size": px(middle ? 12.5 : grow(z, 11, 0.3, 15)),
      "text-wrap": "wrap", "text-max-width": middle ? "data(d)" : px(180), "text-valign": middle ? "center" : "top",
      "text-margin-y": middle ? 0 : px(-3), "text-background-color": css("--map-bubble"),
      "text-background-opacity": middle ? 0.92 : 0, "text-background-padding": px(2), "text-background-shape": "roundrectangle" } },
    { selector: ".subtopic.picked", style: { "border-width": px(2.5), "border-style": "solid", "border-color": css("--accent") } },
    { selector: ".subname", style: { display: middle ? "element" : "none", width: 1, height: 1, "background-opacity": 0,
      "z-index": 40, events: "no", label: "data(label)", "font-family": FONT,
      "font-weight": 600, color: css("--fg"), "font-size": px(12.5), "text-wrap": "wrap", "text-max-width": "data(w)",
      "text-valign": "center", "text-background-color": css("--map-bubble"), "text-background-opacity": 0.92,
      "text-background-padding": px(2), "text-background-shape": "roundrectangle" } },
    // Zoomed out, the topics alone: their terms appear once there is room to tell them apart.
    { selector: ".term", style: { display: overview ? "none" : "element", width: size, height: size, "z-index": 10, label: "",
      "border-width": 0 } },
    ...famRules,
    { selector: ".term.lab", style: { label: "data(label)", color: css("--fg"), "font-family": FONT,
      "font-size": px(font), "text-wrap": "wrap", "text-max-width": px(font * LABEL_EMS),
      "text-valign": "bottom", "text-margin-y": px(3), ...HALO(px) } },
    { selector: ".term.sel", style: { width: size * 1.9, height: size * 1.9, "border-width": px(2.5), "border-color": css("--fg"),
      "font-weight": 600, "font-size": px(font * 1.12), "z-index": 30 } },
    { selector: ".term.nb", style: { "z-index": 20 } },
    { selector: ".term.held", style: { "border-width": px(3), "border-color": css("--accent"), "z-index": 35 } },
    { selector: ".link", style: { display: overview ? "none" : "element", width: px(line), "line-color": css("--muted"),
      ...ARC, "z-index": 5, "line-opacity": middle ? 0.3 : 0.55 } },   // fades the line, not its label
    // The link labels, once the terms they join are named: the verb along the line, reading the way the arrow points,
    // in italic and the accent colour, so a verb is not read as a term.
    ...(verbsOn && !middle ? [{ selector: ".link", style: { "source-label": "data(verb)", "source-text-offset": "data(at)", "font-family": FONT, "font-size": px(font * 0.85),
      color: css("--accent"), "font-style": "italic", "source-text-rotation": "autorotate", ...HALO(px), "target-arrow-shape": "triangle",
      "target-arrow-color": css("--muted"), "arrow-scale": arrow(line, dot * 0.8) } }] : []),
    { selector: ".link.cross", style: { "line-opacity": middle ? 0.12 : 0.25 } },   // links between topics: present, but behind the topic's own
    { selector: ".link.hl", style: { display: "element", width: px(line * 1.25), "line-color": css("--accent"), "target-arrow-shape": "triangle",
      "target-arrow-color": css("--accent"), "arrow-scale": arrow(line * 1.25, dot), "line-opacity": 1, "z-index": 15 } },
    { selector: ".between", style: { display: overview ? "element" : "none", ...ARC, "z-index": 1,
      width: `mapData(n, 1, 14, ${px(1)}, ${px(7)})`, "line-color": css("--muted"), opacity: 0.45 } },
    { selector: ".dim", style: { opacity: 0.18 } },
    { selector: ".topic.empty, .subtopic.empty, .subname.empty", style: { opacity: 0.35 } },
    { selector: ".link.dim", style: { opacity: 0.08, "source-label": "" } },
    { selector: ".link.mute", style: { "source-label": "" } },   // its label would land on a term or another label (settleLinkLabels)
    { selector: ".term.off", style: { display: "none" } },   // its links go with it
  ];
}

// Labels are placed greedily: the selected term and its neighbours first, then the most connected terms of
// each topic; a label that would overlap one already placed waits until zooming in makes room for it.
const LABEL_EMS = 9;   // wrap width of a label, in multiples of its font size
const widths = new Map();
const ruler = document.createElement("canvas").getContext("2d");
function textWidth(text, font) {
  const key = `${font}|${text}`;
  if (!widths.has(key)) { ruler.font = `${font}px ${FONT}`; widths.set(key, ruler.measureText(text).width); }
  return widths.get(key);
}
function placeLabels(z) {
  const gap = TERM_GAP * z;
  const show = new Set();
  placed = [];
  if (gap >= DETAIL_GAP) {
    const { font, dot } = sizes(z), wrap = font * LABEL_EMS;
    const shown = gap < 40 ? 3 : gap < 60 ? 8 : Infinity;   // per topic, by rank, before space decides
    const W = cy.width(), H = cy.height();
    const candidates = [];
    cy.nodes(".term").forEach(n => {
      if (n.hasClass("off")) return;
      const sel = n.hasClass("sel"), forced = sel || (labelsOn && n.hasClass("nb"));
      if (!sel && !labelsOn) return;
      if (!forced && n.data("rank") >= shown) return;
      if (!forced && n.data("sub") && gap < SUB_LABEL_GAP) return;   // the sub-topic's name stands for its terms
      const p = n.renderedPosition();
      if (p.x < -160 || p.x > W + 160 || p.y < -160 || p.y > H + 160) return;
      candidates.push({ n, p, sel, forced, order: sel ? -2 : forced ? -1 : n.data("rank") });
    });
    candidates.sort((a, b) => a.order - b.order || b.n.degree() - a.n.degree());
    const boxes = [];
    for (const c of candidates) {
      const f = c.sel ? font * 1.12 : font;
      const w = textWidth(c.n.data("label"), f);
      const lines = w <= wrap ? 1 : Math.ceil((w / wrap) * 1.15);
      const half = Math.min(w, wrap) / 2 + 4, top = c.p.y + (c.sel ? dot * 1.9 : dot) / 2 + 1;
      const box = { x1: c.p.x - half, x2: c.p.x + half, y1: top, y2: top + lines * f * 1.3 + 4 };
      if (c.forced || !boxes.some(o => box.x1 < o.x2 && o.x1 < box.x2 && box.y1 < o.y2 && o.y1 < box.y2)) {
        boxes.push(box); show.add(c.n.id());
        placed.push(c);
      }
    }
  }
  cy.nodes(".term").forEach(n => n.toggleClass("lab", show.has(n.id())));
}

// The first pass estimates label sizes; once they are drawn, their real boxes settle any overlap it missed.
let placed = [];
function settleOverlaps() {
  // Topic and sub-topic names are placed first: term names give way to them.
  const kept = cy.nodes(".topic, .subtopic, .subname").filter(n => n.visible() && n.style("label"))
    .map(n => n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true }));
  for (const c of placed) {
    const b = c.n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true });
    if (!c.forced && kept.some(o => b.x1 < o.x2 && o.x1 < b.x2 && b.y1 < o.y2 && o.y1 < b.y2)) c.n.removeClass("lab");
    else kept.push(b);
  }
  if (verbsOn) settleLinkLabels(kept);
}

// A link's label sits on its line, clear of the terms, their names and the other labels: in the middle where that is
// clear, else at the first clear spot further along either way, else held back until zooming in makes room for it.
// The label is drawn `at` drawing units along the line from where it leaves its source term.
const SPOTS = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8];   // fractions of the way from the source term to the target
function settleLinkLabels(kept) {
  const z = cy.zoom(), { font, dot } = sizes(z), f = font * 0.85, pad = f * 0.5;   // pad: half the text's height
  if (TERM_GAP * z < SUB_LABEL_GAP) return;   // no labels drawn yet
  const W = cy.width(), H = cy.height();
  const taken = [...kept, ...cy.nodes(".term").filter(n => n.visible()).map(n => n.renderedBoundingBox({ includeLabels: false }))];
  const hit = (x, y) => taken.some(b => x > b.x1 - pad && x < b.x2 + pad && y > b.y1 - pad && y < b.y2 + pad);
  const links = cy.edges(".link").filter(e => e.visible() && !e.hasClass("dim"));
  links.sort((a, b) => Number(b.hasClass("hl")) - Number(a.hasClass("hl"))).forEach(e => {
    const s = e.source().renderedPosition(), t = e.target().renderedPosition();
    const len = Math.hypot(t.x - s.x, t.y - s.y), w = textWidth(e.data("verb"), f) + 4;
    const from = e.source().renderedWidth() / 2, to = len - e.target().renderedWidth() / 2 - dot;   // clear of the arrowhead
    // Along the curve as Cytoscape draws it: the quadratic through its own control point, from the centre of one term
    // to the other's, close enough to the arc length for a label's few points.
    const pan = cy.pan(), [c] = e.controlPoints() || [{ x: ((s.x + t.x) / 2 - pan.x) / z, y: ((s.y + t.y) / 2 - pan.y) / z }];
    const cx = c.x * z + pan.x, cy_ = c.y * z + pan.y;
    const points = k => [-0.45, -0.2, 0, 0.2, 0.45].map(j => { const u = Math.min(1, Math.max(0, k + (w * j) / len)), v = 1 - u;
      return { x: v * v * s.x + 2 * u * v * cx + u * u * t.x, y: v * v * s.y + 2 * u * v * cy_ + u * u * t.y }; });
    const mid = points(0.5)[2], off = mid.x < 0 || mid.x > W || mid.y < 0 || mid.y > H;
    const k = off ? 0.5 : SPOTS.find(k => k * len - w / 2 >= from && k * len + w / 2 <= to && !points(k).some(p => hit(p.x, p.y)));
    e.toggleClass("mute", k === undefined);
    if (k === undefined) return;
    const at = (k * len - from) / z;
    if (Math.abs(e.data("at") - at) > 0.01) e.data("at", at);
    if (!off) taken.push(...points(k).map(p => ({ x1: p.x - pad / 2, x2: p.x + pad / 2, y1: p.y - pad / 2, y2: p.y + pad / 2 })));
  });
}

let styledZoom = null;
function refresh(force = false) {
  const z = cy.zoom();
  cy.batch(() => {
    if (force || z !== styledZoom) { cy.style(sheet(z)); styledZoom = z; }
    placeLabels(z);
  });
  cy.batch(settleOverlaps);   // after the first batch, so the labels have real sizes
}
let queued = false;
const later = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; refresh(); }); } };
cy.on("viewport", later);
cy.on("position", "node.term", later);   // labels follow a term the reader moves

function highlight(id) {
  cy.batch(() => {
    cy.elements().removeClass("sel nb hl dim picked");
    if (id && terms.has(id)) {
      const node = cy.getElementById(id);
      const near = node.closedNeighborhood();
      node.addClass("sel");
      near.nodes(".term").not(node).addClass("nb");
      near.edges(".link").addClass("hl");
      cy.nodes(".term").not(near.nodes()).addClass("dim");
      cy.edges(".link").not(near.edges()).addClass("dim");
    } else if (id && groupOf(id)) {
      cy.getElementById(id).addClass("picked");
    }
    applyHidden();
  });
  refresh(true);
}

function move(id, animate) {
  const duration = animate && !reduce ? 450 : 0;
  if (id && terms.has(id)) {
    const p = cy.getElementById(id).position();
    const z = Math.min(cy.maxZoom(), Math.max(cy.zoom(), 72 / TERM_GAP));   // close enough to read its neighbours
    const pan = { x: cy.width() / 2 - p.x * z, y: cy.height() * 0.45 - p.y * z };
    cy.stop(); cy.animate({ zoom: z, pan }, { duration });
  } else if (id && groupOf(id)) {
    // Fit the topic, but come close enough for its terms to show: on a phone a large topic's edge is cropped.
    const node = cy.getElementById(id), p = node.position(), d = node.width();
    const fit = Math.min((cy.width() - 24) / d, (cy.height() - 24) / d);
    const z = Math.min(cy.maxZoom(), Math.max(fit, (DETAIL_GAP / TERM_GAP) * 1.08));
    cy.stop(); cy.animate({ zoom: z, pan: { x: cy.width() / 2 - p.x * z, y: cy.height() / 2 - p.y * z } }, { duration });
  } else {
    cy.stop(); cy.animate({ fit: { eles: cy.nodes(".topic"), padding: 14 } }, { duration });
  }
}

// On a screen wider than the drawing, the topics spread sideways to fill it. Each topic moves whole, its sub-topics
// and terms with it, so nothing inside a topic changes; and as they move apart only across, no two circles that were
// apart can meet. The spread is as much as the map's width holds, at most SPREAD_MAX; a map taller than it is wide
// (a phone held upright) keeps the drawing as it is. A term the reader moved keeps its place in its topic.
const SPREAD_MAX = 3;
let spreadNow = 1;
const centreX = (() => { const xs = DATA.clusters.flatMap(c => [c.x - c.r, c.x + c.r]); return (Math.min(...xs) + Math.max(...xs)) / 2; })();
const topicOf = n => n.hasClass("topic") ? n.id() : n.hasClass("term") ? n.data("topic")
  : subtopics.get(n.id().replace(/-name$/, "")).topic;
function spread() {
  const ys = DATA.clusters.flatMap(c => [c.y - c.r, c.y + c.r]), tall = Math.max(...ys) - Math.min(...ys);
  // The drawing's width with the centres spread by k: the circles keep their size, so it grows less than k.
  const wideAt = k => { const xs = DATA.clusters.flatMap(c => { const x = centreX + (c.x - centreX) * k; return [x - c.r, x + c.r]; });
    return Math.max(...xs) - Math.min(...xs); };
  const room = (cy.width() / cy.height()) * tall * 0.92;   // the width that fits the map at the height's zoom
  let want = 1;
  while (want < SPREAD_MAX && wideAt(want + 0.05) <= room) want += 0.05;
  if (Math.abs(want - spreadNow) < 0.01) return;
  const shift = id => (topics.get(id).x - centreX) * (want - spreadNow);
  cy.batch(() => {
    cy.nodes().forEach(n => { const p = n.position(); n.position({ x: p.x + shift(topicOf(n)), y: p.y }); });
    cy.edges().forEach(e => e.data("bow", bowOf(e.source().position(), e.target().position())));
  });
  spreadNow = want;
}

function setLimits() {
  const z0 = cy.zoom();
  cy.fit(cy.nodes(".topic"), 14);
  const fitZoom = cy.zoom();
  cy.minZoom(fitZoom * 0.6);
  cy.maxZoom(Math.max(fitZoom * 30, 6));   // deep enough to set the closest terms at least 370 px apart, on any screen
  return { fitZoom, z0 };
}

// ---------- the card and the chips
function chipFor(id) {
  if (groupOf(id)) { const c = groupOf(id);
    return `<button class="map-chip" type="button" data-id="${id}">${esc(c.name)} <small>${sizeAt(id)}</small></button>`; }
  const n = terms.get(id);
  return `<button class="map-chip" type="button" data-id="${id}">${icon(fam(n), css(`--map-${fam(n)}`))}${esc(n.name)}</button>`;
}
const group = (title, ids) => ids.length ? `<div class="map-rel"><h3>${title}</h3><div class="map-chips">${ids.map(chipFor).join("")}</div></div>` : "";

function render() {
  const card = document.getElementById("card"), links = document.getElementById("links"), strip = document.getElementById("striptext");
  stripWas = null;
  if (selected && terms.has(selected)) {
    const n = terms.get(selected), color = css(`--map-${fam(n)}`);
    card.style.setProperty("--fam", color);
    card.innerHTML = `<span class="map-type">${n.type.replace(/([a-z])([A-Z])/g, "$1 $2")} · chapter ${n.chapter}</span><h2>${esc(n.name)}</h2><p>${esc(n.definition)}</p>
      ${n.aliases.length ? `<p class="map-aka">Also called: ${esc(n.aliases.join(", "))}</p>` : ""}
      <blockquote class="map-quote"><p>${esc(n.quote)}</p></blockquote>
      <p class="map-book"><a href="${bookUrl(n)}" target="_blank" rel="noopener">Read it in the book</a></p>`;
    const groups = new Map();
    for (const e of linksOf(selected)) {
      const [rel, other] = e.source === selected ? [read(e.type, 0), e.target] : [read(e.type, 1), e.source];
      groups.set(rel, [...(groups.get(rel) || []), other]);
    }
    links.innerHTML = group("Topic", [n.cluster]) + (n.sub ? group("Sub-topic", [n.sub]) : "") + [...groups].map(([rel, ids]) => group(rel, ids)).join("");
    strip.innerHTML = `<b>${esc(n.name)}</b> · ${esc(n.definition)}`;
  } else if (selected && groupOf(selected)) {
    const c = groupOf(selected), isSub = subtopics.has(selected);
    const members = DATA.nodes.filter(n => (isSub ? n.sub : n.cluster) === c.id && onMap.has(n.id))
      .sort((a, b) => b.degree - a.degree || a.name.localeCompare(b.name)).map(n => n.id);
    const near = DATA.cluster_links.filter(l => l.a === c.id || l.b === c.id).sort((a, b) => b.n - a.n).map(l => (l.a === c.id ? l.b : l.a));
    card.style.removeProperty("--fam");
    card.innerHTML = `<span class="map-type">${isSub ? `Sub-topic of ${esc(topics.get(c.topic).short)}` : "Topic"}</span><h2>${esc(c.name)}</h2>
      <p>${members.length} terms${level < LEVELS.length ? ` at the ${LEVELS[level - 1].toLowerCase()} level` : ""}. The most connected are ${members.slice(0, 3).map(id => esc(terms.get(id).name).toLowerCase()).join(", ")}.</p>`;
    links.innerHTML = isSub ? group("Terms, most connected first", members) + group("Topic", [c.topic])
      : group("Sub-topics", (c.subs || []).map(s => s.id)) + group("Terms, most connected first", members)
        + group("Linked topics, most links first", near);
    strip.innerHTML = `<b>${esc(c.name)}</b> · zoom in or tap a term`;
  } else {
    card.style.removeProperty("--fam");
    card.innerHTML = `<span class="map-type">Overview</span><h2>${DATA.clusters.length} topics</h2>
      <p>The terms fall into groups of closely linked ideas. Tap a topic, or zoom in: the larger topics open into sub-topics, then into terms, the most connected named first.</p>`;
    links.innerHTML = group("Topics, largest first", DATA.clusters.map(c => c.id));
    strip.innerHTML = `<b>All topics</b> · pinch or scroll to zoom, tap a topic to open it`;
  }
}

function go(id, { animate = true, hash = true } = {}) {
  if (id && !terms.has(id) && !groupOf(id)) id = null;
  selected = id;
  highlight(id);
  render();
  move(id, animate);
  if (hash) { try { history.replaceState(null, "", id ? `#${id.replaceAll("_", "-")}` : location.pathname + location.search); } catch {} }
}
const fromHash = () => location.hash.slice(1).replaceAll("-", "_");

// On a touch screen one finger scrolls the page, as it does everywhere else on it, and two fingers move and zoom the
// map, as on an embedded map: a one-finger drag that moved the map trapped a phone's reader inside it. Cytoscape
// listens for moves on the window and cancels the browser's scrolling, so one finger's moves are kept from it; and it
// moves the map with two fingers only once they spread far apart, so two fingers are followed here: the map moves with
// the point between them and zooms with their spread. A tap still opens what is under the finger; a drag that
// scrolled is not a tap, and the first such drag in a while says how to move the map.
const hint = document.createElement("p");
hint.className = "map-hint";
hint.setAttribute("aria-hidden", "true");
cy.container().parentElement.append(hint);
const say = text => {
  hint.textContent = text;
  hint.classList.add("on");
  clearTimeout(say.off);
  say.off = setTimeout(() => hint.classList.remove("on"), 1600);
};

// A term can be moved, to uncover a link's label under it: dragged with the mouse, as Cytoscape does for a grabbable
// node; on a touch screen, held still for a moment and then dragged, as one finger that moves at once scrolls the page.
// Nothing is kept: a reload restores the drawing.
cy.nodes().not(".term").ungrabify();
cy.container().addEventListener("pointerdown", ev => cy.autoungrabify(ev.pointerType !== "mouse"), { capture: true });
cy.on("grab", "node.term", ev => ev.target.addClass("held"));
cy.on("free", "node.term", ev => ev.target.removeClass("held"));
cy.on("mouseover", "node.term", () => { if (!cy.autoungrabify()) cy.container().style.cursor = "grab"; });
cy.on("mouseout", "node.term", () => { cy.container().style.cursor = ""; });
const HOLD_MS = 450;
let held = null, holding = 0;
function termAt(x, y) {   // the visible term nearest a point of the map, within a fingertip's reach
  const reach = Math.max(sizes(cy.zoom()).dot, 24) / 2 + 6;
  let best = null, gap = reach;
  cy.nodes(".term").filter(n => n.visible()).forEach(n => {
    const p = n.renderedPosition(), d = Math.hypot(p.x - x, p.y - y);
    if (d <= gap) { best = n; gap = d; }
  });
  return best;
}
const between = ([a, b]) => {
  const box = cy.container().getBoundingClientRect();
  return { x: (a.clientX + b.clientX) / 2 - box.left, y: (a.clientY + b.clientY) / 2 - box.top,
    spread: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) };
};
let finger = null, pinch = null, scrolled = false, hinted = 0;
cy.container().addEventListener("touchstart", ev => {
  finger = ev.touches.length === 1 ? [ev.touches[0].clientX, ev.touches[0].clientY] : null;
  pinch = ev.touches.length === 2 ? between(ev.touches) : null;
  if (ev.touches.length === 1) scrolled = false;
  clearTimeout(holding);
  if (ev.touches.length !== 1) return;
  const box = cy.container().getBoundingClientRect();
  const term = termAt(ev.touches[0].clientX - box.left, ev.touches[0].clientY - box.top);
  if (term) holding = setTimeout(() => {
    if (scrolled || finger === null) return;
    held = term.addClass("held");
    scrolled = true;   // what follows is a move, not a tap
    navigator.vibrate?.(15);
    say("Drag to move this term");
  }, HOLD_MS);
}, { passive: true });
window.addEventListener("touchmove", ev => {
  if (pinch && ev.touches.length === 2) {
    ev.preventDefault();
    ev.stopPropagation();
    const now = between(ev.touches);
    cy.zoom({ level: cy.zoom() * now.spread / pinch.spread, renderedPosition: { x: pinch.x, y: pinch.y } });
    cy.panBy({ x: now.x - pinch.x, y: now.y - pinch.y });
    pinch = now;
    return;
  }
  if (finger === null || ev.touches.length !== 1) return;
  ev.stopPropagation();
  if (held) {   // the term follows the finger, and the page stays where it is
    ev.preventDefault();
    const box = cy.container().getBoundingClientRect(), z = cy.zoom(), pan = cy.pan();
    held.position({ x: (ev.touches[0].clientX - box.left - pan.x) / z, y: (ev.touches[0].clientY - box.top - pan.y) / z });
    return;
  }
  if (scrolled || Math.hypot(ev.touches[0].clientX - finger[0], ev.touches[0].clientY - finger[1]) < 10) return;
  scrolled = true;
  clearTimeout(holding);
  if (Date.now() - hinted < 20000) return;
  hinted = Date.now();
  say("Two fingers move the map");
}, { capture: true, passive: false });
window.addEventListener("touchend", ev => {
  if (ev.touches.length < 2) pinch = null;
  clearTimeout(holding);
  if (held) { held.removeClass("held"); held = null; }
  // The browser is scrolling and the touch's end cannot be cancelled: Cytoscape's attempt to is then a no-op, not a
  // console error.
  if (!ev.cancelable) ev.preventDefault = () => {};
}, { capture: true });

// A link read as a sentence in the strip under the map: while the mouse is on its line, or after a tap on one of the
// selected term's links. A tap on any other link counts as one on what lies under it.
function sayLink(e) {
  const strip = document.getElementById("striptext"), name = id => esc(terms.get(id).name);
  if (stripWas === null) stripWas = strip.innerHTML;
  strip.innerHTML = `<b>${name(e.data("source"))}</b> <i>${esc(e.data("verb"))}</i> <b>${name(e.data("target"))}</b>`;
}
cy.on("mouseover", "edge.link", ev => { if (!ev.target.hasClass("dim")) sayLink(ev.target); });
cy.on("mouseout", "edge.link", () => {
  if (stripWas !== null) { document.getElementById("striptext").innerHTML = stripWas; stripWas = null; }
});
// A tap on the map brings it up the page, the filters at the top of the screen, so the card it opens below the map
// is in view: a reader who tapped with the map low on the screen would otherwise not see that anything changed. A
// page already scrolled further stays where it is.
function bringUp() {
  const top = document.querySelector(".map-find").getBoundingClientRect().top + scrollY - 8;
  if (scrollY < top - 4) scrollTo({ top, behavior: reduce ? "auto" : "smooth" });
}
cy.on("tap", "node.term", ev => { if (!scrolled) { go(ev.target.id()); bringUp(); } });
cy.on("tap", "edge.link.hl", ev => { if (!scrolled) sayLink(ev.target); });
cy.on("tap", ev => {
  if (scrolled || (ev.target !== cy && !ev.target.isEdge()) || (ev.target !== cy && ev.target.hasClass("hl"))) return;
  const { x, y } = ev.position;
  const inside = c => { const p = cy.getElementById(c.id).position(); return Math.hypot(x - p.x, y - p.y) <= c.r; };
  const overview = TERM_GAP * cy.zoom() < DETAIL_GAP;   // sub-topics are hidden there
  const hit = (!overview && [...subtopics.values()].find(inside)) || DATA.clusters.find(inside);
  if (hit && hit.id !== selected) { go(hit.id); bringUp(); }
});
document.getElementById("links").addEventListener("click", ev => {
  const b = ev.target.closest(".map-chip"); if (b) { go(b.dataset.id); document.querySelector(".map-box").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" }); }
});
const zoomBy = f => { const z = Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * f));
  cy.stop(); cy.animate({ zoom: { level: z, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: reduce ? 0 : 250 }); };
function applyHidden() {
  const sel = cy.nodes(".term.sel");
  onMap = drawnTerms(sel.nonempty() ? sel.id() : null);
  cy.nodes(".term").forEach(n => n.toggleClass("off", !onMap.has(n.id())));
  cy.nodes(".topic").forEach(n => n.data("olabel", `${topics.get(n.id()).short}\n[${sizeAt(n.id())}]`));
  cy.nodes(".subname").forEach(n => n.data("label", `${subtopics.get(n.id().replace(/-name$/, "")).name}\n[${sizeAt(n.id().replace(/-name$/, ""))}]`));
  // A topic with no term drawn at this level, or of these kinds, is dimmed: there is nothing in it to open.
  cy.nodes(".topic, .subtopic").forEach(n => n.toggleClass("empty", sizeAt(n.id()) === 0));
  cy.nodes(".subname").forEach(n => n.toggleClass("empty", sizeAt(n.id().replace(/-name$/, "")) === 0));
}
const levelSlider = document.getElementById("level"), levelName = document.getElementById("level-name");
function showLevel() {
  levelSlider.value = level;
  levelName.value = LEVELS[level - 1];
  levelSlider.setAttribute("aria-valuetext", LEVELS[level - 1]);
}
levelSlider.closest(".map-level").hidden = !leveled;
showLevel();
levelSlider.addEventListener("input", () => {
  level = Number(levelSlider.value);
  try { localStorage.setItem("factorial-map-level", String(level)); } catch {}
  showLevel();
  cy.batch(applyHidden);
  refresh(true);
  render();   // an open topic's card lists the terms at this level
});
document.getElementById("legend").addEventListener("click", ev => {
  const b = ev.target.closest("button[data-fam]"); if (!b) return;
  const k = b.dataset.fam;
  if (hidden.has(k)) hidden.delete(k); else hidden.add(k);
  try { localStorage.setItem("factorial-map-hidden", JSON.stringify([...hidden])); } catch {}
  drawLegend();
  document.querySelector(`#legend button[data-fam="${k}"]`).focus();   // keep the keyboard where it was
  cy.batch(applyHidden); refresh(true);
  render();   // an open topic's card counts the terms drawn
});
const labelsSwitch = document.getElementById("labels");
labelsSwitch.checked = labelsOn;
labelsSwitch.addEventListener("change", () => {
  labelsOn = labelsSwitch.checked;
  try { localStorage.setItem("factorial-map-labels", labelsOn ? "on" : "off"); } catch {}
  refresh(true);
});
const verbsSwitch = document.getElementById("verbs");
verbsSwitch.checked = verbsOn;
verbsSwitch.addEventListener("change", () => {
  verbsOn = verbsSwitch.checked;
  try { localStorage.setItem("factorial-map-verbs", verbsOn ? "on" : "off"); } catch {}
  refresh(true);
});
document.getElementById("zin").addEventListener("click", () => zoomBy(1.6));
document.getElementById("zout").addEventListener("click", () => zoomBy(1 / 1.6));
document.getElementById("home").addEventListener("click", () => go(null));
document.getElementById("details").addEventListener("click", () =>
  document.getElementById("card").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }));

// ---------- search
const q = document.getElementById("q"), results = document.getElementById("results");
const norm = s => s.toLowerCase().normalize("NFKD");
q.addEventListener("input", () => {
  const t = norm(q.value.trim());
  if (!t) { results.hidden = true; return; }
  const hits = DATA.nodes.filter(shown).map(n => {
    // A term's found_as names are what other texts call it: searched like its other names, never shown on its card.
    const names = [n.name, ...n.aliases, ...(n.found_as || [])].map(norm);
    const score = names.some(s => s === t) ? 0 : names.some(s => s.startsWith(t)) ? 1 : names.some(s => s.includes(t)) ? 2
      : norm(n.definition).includes(t) ? 3 : 9;
    return { n, score };
  }).filter(h => h.score < 9).sort((a, b) => a.score - b.score || a.n.name.localeCompare(b.n.name)).slice(0, 12);
  results.innerHTML = hits.length ? hits.map(({ n }) =>
    `<button type="button" role="option" data-id="${n.id}"><span>${esc(n.name)}</span><small>${n.type}</small></button>`).join("")
    : `<button type="button" disabled><span>No term matches "${esc(q.value)}"</span></button>`;
  results.hidden = false;
});
results.addEventListener("click", ev => {
  const b = ev.target.closest("button[data-id]"); if (!b) return;
  go(b.dataset.id); q.value = ""; results.hidden = true;
});
q.addEventListener("keydown", ev => {
  if (ev.key === "Enter") { const first = results.querySelector("button[data-id]"); if (first) first.click(); }
  if (ev.key === "Escape") results.hidden = true;
});
document.addEventListener("click", ev => { if (!ev.target.closest(".map-search")) results.hidden = true; });

// ---------- how the map was made. Every number comes from DATA, so it cannot drift from the map; as on the app's
// other pages, the words say what happens and the exact numbers wait behind "Show the numbers".
function renderHow() {
  const m = DATA.method, E = DATA.edges.length;
  // The how-to as it is now on the book's main branch: it holds the steps, so this page does not repeat them.
  const howto = "https://github.com/kgdunn/pid-book/blob/main/knowledge-graph/HOWTO.md";
  const verbs = new Set(DATA.edges.map(e => e.type)).size;
  // A kind of term in the words, drawn with its shape and colour on the map and in the legend.
  const kind = (f, words) => `<span class="map-kind"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="${FAMILIES[f][2]}"
    style="fill: var(--map-${f})"/></svg>${words}</span>`;
  const name = id => esc(terms.get(id).name);
  const pairs = n => (n * (n - 1)) / 2;
  const examples = [["box_behnken", "IS_A", "omars_design"], ["center_point", "KEEPS", "orthogonality"],
    ["coefficient", "IS_HALF_OF", "main_effect"]]
    .map(([s, v, t]) => DATA.edges.find(e => e.source === s && e.type === v && e.target === t)).filter(Boolean);
  const light = Object.entries(m.light).map(([v, w]) => `"${read(v, 0)}" counts ${w}`).join(", ");
  const withSubs = DATA.clusters.filter(c => c.subs.length);
  const whole = DATA.clusters.filter(c => c.moved && c.moved.length);
  const quoted = ids => ids.map(id => `"${name(id)}"`).join(" and ");
  const rows = DATA.clusters.map(c => {
    const level = c.subs.length ? `${c.subs.length} sub-topics; other seed batches agree at ${c.agreement.toFixed(2)} at worst`
      : c.size < m.sub_min_size ? `None: fewer than ${m.sub_min_size} terms`
      : `None: other seed batches agree at only ${c.agreement.toFixed(2)}`;
    return `<tr><td>${esc(c.short)}</td><td>${c.size}</td><td>${level}</td></tr>`;
  }).join("");

  document.getElementById("how").innerHTML = `
    <h2 id="how-title">How this map was made</h2>

    <h3>1. Where the ideas come from</h3>
    <p>Every term comes from <em>Process Improvement using Data</em>, the open textbook at <a href="https://learnche.org/pid" target="_blank" rel="noopener">learnche.org/pid</a>, mostly
      from its chapter on designing and analysing experiments. Each term, and each link between two terms, quotes the
      line of the book it rests on, and the map is checked against the book each time it is built.</p>
    <details><summary>How it was checked</summary>
      <p>A first list was written by hand from the chapter's text. Reviewers then checked every term and link against
        the chapter as it is now, and added what its newer sections cover, such as OMARS designs, Lenth's method and
        desirability.</p>
    </details>

    <h3>2. What the points and lines mean</h3>
    <p>Each point is one idea the book explains, and its shape says what kind:
      ${kind("design", "a design or a property of one")}, ${kind("method", "a method")}, ${kind("concept", "a concept")},
      ${kind("guide", "a guideline or warning sign")}, or ${kind("context", "an example")}. Tap one to read its
      definition and the book's own words.</p>
    <p>Each line joining the points above is a short sentence that joins two ideas, for example:</p>
    <ul>${examples.map(e => `<li><b>${name(e.source)}</b> ${read(e.type, 0)} <b>${name(e.target)}</b></li>`).join("")}</ul>
    <details><summary>Show the numbers</summary>
      <p>${E} links using ${verbs} verbs, each citing its line of the book. A definition is at most 32 words.</p>
    </details>

    <h3>3. How the topics are found</h3>
    <p>A topic is a group of ideas that are more related to each other than to the rest. A computer finds the groups,
      but each attempt depends on probabilities, so it makes hundreds of tries and keeps the grouping that is most stable
      and consistent.</p>
    <details><summary>Show the numbers</summary>
      <p>The method is <a href="https://en.wikipedia.org/wiki/Louvain_method" target="_blank" rel="noopener">Louvain clustering</a>: it moves one term at a time into a neighbour's group whenever that makes the
        groups tighter, and stops when no move helps. It visits the terms in a random order.
        Consensus clustering (<a href="https://doi.org/10.1038/srep00336" target="_blank" rel="noopener">Lancichinetti and Fortunato, 2012</a>)
        makes it repeatable: ${m.runs} runs a round, each with its own random seed; a pair of terms together in at least
        ${Math.round(m.threshold * 100)}% of runs is kept, and the rest dropped; rounds repeat
        until all ${m.runs} runs agree. A term left on its own joins the group it was most often in. Link weights:
        ${light}; every other link counts 1.</p>
      <p>Two groupings are compared pair by pair: for each pair of terms, are they together in both, or separate in both?
        The fraction of pairs that match, after correcting for what chance alone would give, is the adjusted Rand index: 1 for
        identical groupings, about 0 for unrelated ones.</p>
      <p>To test this, a computer ran ${m.single_runs} times on its own, each from a different random start. A typical
        run matched the ${DATA.clusters.length} topics on this map at ${m.single_median.toFixed(2)}, and the worst at
        ${m.single_min.toFixed(2)}, so a single run puts quite a few terms in a different topic. The full method, repeated
        twice from fresh random starts, matched them at ${m.consensus_other.map(x => x.toFixed(2)).join(" and ")}. The few
        terms that still moved are linked about equally to two topics.</p>
      <p><a href="${m.code}" target="_blank" rel="noopener">The Python code that finds the topics and sub-topics</a>.</p>
    </details>

    <h3>4. How the sub-topics are found</h3>
    <p>The larger topics are split into sub-topics in the same way, but only where the tries agree on the split; the
      others are shown
      whole. Zoom past the topics to see the sub-topics as dashed circles, each named after its most connected term.</p>
    <details><summary>Show the numbers</summary>
      <p>A topic gets sub-topics when it has at least ${m.sub_min_size} terms and two other seed batches reproduce its
        split at an adjusted Rand index of ${m.sub_min_agreement} or better. That holds for ${withSubs.length} of the
        ${DATA.clusters.length} topics.</p>
      <div class="map-tablewrap"><table>
        <thead><tr><th>Topic</th><th>Terms</th><th>Second level</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      ${whole.map((c, i) => `<p>${i ? "In" : "For example, in"} "${esc(c.short)}" (${c.size} terms), another seed batch moved ${quoted(c.moved)} from one
        sub-group to another; every other term kept its group. Agreement is counted over all ${pairs(c.size)} pairs of
        terms, and the pairs these ${c.moved.length} terms form with their old and new group-mates bring it down to
        ${c.agreement.toFixed(2)}. A split that changes with chance would change the map for no reason in the book, so
        this topic is shown whole.</p>`).join("")}
    </details>

    <h3>Build one from your own texts</h3>
    <p><a href="${howto}" target="_blank" rel="noopener">The how-to</a> gives the steps, with a prompt to paste into an
      AI coding agent for each: draft the terms and links from your texts, check every quote against its line, review
      them, and draw the map with a copy of this page.</p>`;
}
renderHow();
document.getElementById("howlink").addEventListener("click", () =>
  document.getElementById("how").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }));

// ---------- start, theme and size changes
const restyle = () => { drawLegend(); refresh(true); render(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
window.addEventListener("hashchange", () => go(fromHash(), { hash: false }));
// A phone's toolbar that hides or shows as the page scrolls fires "resize" but leaves the map's width alone: the
// reader's view stays then. Only a new width (a turned phone, a resized window) centres the map again.
let resizing, mapWidth = cy.width();
window.addEventListener("resize", () => { clearTimeout(resizing); resizing = setTimeout(() => {
  const zoom = cy.zoom(), pan = { ...cy.pan() };
  cy.resize(); spread(); setLimits();
  if (cy.width() !== mapWidth) { mapWidth = cy.width(); move(selected, false); } else cy.viewport({ zoom, pan });
}, 150); });
if (document.fonts) document.fonts.ready.then(() => { widths.clear(); refresh(true); });

// Feedback, as the app sends it: the same dialog and relay, with what is open on the map in place of a study.
installFeedback({
  context: async () => ({
    commit: await fetch(new URL("wheels/manifest.json", import.meta.url))
      .then(r => r.json()).then(m => m.commit, () => null),
    tab: `Concept map: ${document.querySelector("#card h2")?.textContent ?? ""}`,
    browser: navigator.userAgent,
    window: `${window.innerWidth} x ${window.innerHeight}`,
  }),
});

drawLegend();
spread();
setLimits();
go(fromHash() || null, { animate: false, hash: false });
