// The concept map (map/index.html): the ideas of designed experiments in Process Improvement using Data,
// and how they link. Zoomed out it shows the topics; zooming in opens the larger ones into sub-topics, then names their
// terms; a tapped term shows its definition, the line of the book it comes from, and its links.
//
// map/graph.json is built and checked against the book in pid-book's knowledge-graph/ (build_kg.py, layout_kg.py),
// where every term quotes the line of the book it comes from: to update the map, copy doe_kg_map.json from there.
import cytoscape from "./vendor/cytoscape.mjs";

const DATA = await fetch(new URL("map/graph.json", import.meta.url)).then((r) => {
  if (!r.ok) throw new Error(`map/graph.json: ${r.status}`);
  return r.json();
});
// The canvas draws its text in the page's own typeface.
const FONT = getComputedStyle(document.body).fontFamily;
const FAMILY = { DesignType: "design", Property: "design", Criterion: "design", Method: "method", Plot: "method", Phase: "method",
  Concept: "concept", Guideline: "guide", Diagnostic: "guide", Example: "context", Question: "context", Reference: "context" };
const FAMILIES = {
  design: ["Designs and their properties", "round-rectangle", "M3 3h14v14H3z"],
  method: ["Methods, plots, phases", "diamond", "M10 1.5 18.5 10 10 18.5 1.5 10z"],
  concept: ["Concepts", "ellipse", "M10 2a8 8 0 1 1 0 16a8 8 0 1 1 0-16z"],
  guide: ["Guidelines and warning signs", "triangle", "M10 2 18.5 17.5H1.5z"],
  context: ["Examples, questions, sources", "hexagon", "M5.5 2.5h9L19 10l-4.5 7.5h-9L1 10z"],
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

// Kinds of term hidden from the map, remembered in this browser. The selected term always stays visible.
const hidden = new Set();
try { for (const k of JSON.parse(localStorage.getItem("factorial-map-hidden") || '["context"]')) if (k in FAMILIES) hidden.add(k); } catch {}
const drawLegend = () => {
  document.getElementById("legend").innerHTML = `<span>Show:</span>` + Object.entries(FAMILIES)
    .map(([k, [label]]) => `<button type="button" data-fam="${k}" aria-pressed="${!hidden.has(k)}"
      title="${hidden.has(k) ? "Show" : "Hide"} ${label.toLowerCase()}">${icon(k, css(`--map-${k}`))}${label}</button>`).join("");
};

// ---------- the map
const elements = [
  ...DATA.clusters.map(c => ({ group: "nodes", classes: "topic", position: { x: c.x, y: c.y },
    data: { id: c.id, label: c.short, olabel: `${c.short}\n${c.size} terms`, d: 2 * c.r } })),
  ...[...subtopics.values()].map(s => ({ group: "nodes", classes: "subtopic", position: { x: s.x, y: s.y },
    data: { id: s.id, label: s.name, d: 2 * s.r } })),
  // The same names as markers of their own, drawn above terms and links while the names sit mid-circle.
  ...[...subtopics.values()].map(s => ({ group: "nodes", classes: "subname", position: { x: s.x, y: s.y },
    data: { id: `${s.id}-name`, label: s.name, w: 1.6 * s.r } })),
  ...DATA.nodes.map(n => ({ group: "nodes", classes: `term fam-${fam(n)}`, position: { x: n.x, y: n.y },
    data: { id: n.id, label: n.name, rank: n.rank, topic: n.cluster, sub: n.sub, fam: fam(n) } })),
  ...DATA.edges.map((e, i) => ({ group: "edges", data: { id: `l${i}`, source: e.source, target: e.target },
    classes: terms.get(e.source).cluster === terms.get(e.target).cluster ? "link" : "link cross" })),
  ...DATA.cluster_links.map((e, i) => ({ group: "edges", classes: "between", data: { id: `b${i}`, source: e.a, target: e.b, n: e.n } })),
];
const cy = cytoscape({ container: document.getElementById("cy"), elements, layout: { name: "preset" }, autoungrabify: true,
  autounselectify: true, boxSelectionEnabled: false, wheelSensitivity: 0.35, pixelRatio: "auto" });
// The map and its data, for the tests that drive this page (tests/web/e2e/test_concept_map.py) and for the console.
window.conceptMap = { cy, data: DATA };

let selected = null;   // a term id, a topic id, or null for the overview
let labelsOn = true;   // term names on the map; with them off, only the selected term is named
try { labelsOn = localStorage.getItem("factorial-map-labels") !== "off"; } catch {}
const DETAIL_GAP = 22;   // on-screen distance between terms at which they appear
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
      "text-valign": "bottom", "text-margin-y": px(3), "text-background-color": css("--panel"), "text-background-opacity": 0.8,
      "text-background-padding": px(1.5), "text-background-shape": "roundrectangle" } },
    { selector: ".term.sel", style: { width: size * 1.9, height: size * 1.9, "border-width": px(2.5), "border-color": css("--fg"),
      "font-weight": 600, "font-size": px(font * 1.12), "z-index": 30 } },
    { selector: ".term.nb", style: { "z-index": 20 } },
    { selector: ".link", style: { display: overview ? "none" : "element", width: px(line), "line-color": css("--muted"),
      "curve-style": "straight", "z-index": 5, opacity: middle ? 0.3 : 0.55 } },
    { selector: ".link.cross", style: { opacity: middle ? 0.12 : 0.25 } },   // links between topics: present, but behind the topic's own
    { selector: ".link.hl", style: { display: "element", width: px(line * 2), "line-color": css("--accent"), "target-arrow-shape": "triangle",
      "target-arrow-color": css("--accent"), "arrow-scale": 0.9, opacity: 1, "z-index": 15 } },
    { selector: ".between", style: { display: overview ? "element" : "none", "curve-style": "straight", "z-index": 1,
      width: `mapData(n, 1, 14, ${px(1)}, ${px(7)})`, "line-color": css("--muted"), opacity: 0.45 } },
    { selector: ".dim", style: { opacity: 0.18 } },
    { selector: ".link.dim", style: { opacity: 0.08 } },
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
cy.on("viewport", () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; refresh(); }); } });

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

function setLimits() {
  const z0 = cy.zoom();
  cy.fit(cy.nodes(".topic"), 14);
  const fitZoom = cy.zoom();
  cy.minZoom(fitZoom * 0.6);
  cy.maxZoom(Math.max(fitZoom * 14, 2.4));
  return { fitZoom, z0 };
}

// ---------- the card and the chips
function chipFor(id) {
  if (groupOf(id)) { const c = groupOf(id);
    return `<button class="map-chip" type="button" data-id="${id}">${esc(c.name)} <small>${c.size}</small></button>`; }
  const n = terms.get(id);
  return `<button class="map-chip" type="button" data-id="${id}">${icon(fam(n), css(`--map-${fam(n)}`))}${esc(n.name)}</button>`;
}
const group = (title, ids) => ids.length ? `<div class="map-rel"><h3>${title}</h3><div class="map-chips">${ids.map(chipFor).join("")}</div></div>` : "";

function render() {
  const card = document.getElementById("card"), links = document.getElementById("links"), strip = document.getElementById("striptext");
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
    const members = DATA.nodes.filter(n => (isSub ? n.sub : n.cluster) === c.id)
      .sort((a, b) => b.degree - a.degree || a.name.localeCompare(b.name)).map(n => n.id);
    const near = DATA.cluster_links.filter(l => l.a === c.id || l.b === c.id).sort((a, b) => b.n - a.n).map(l => (l.a === c.id ? l.b : l.a));
    card.style.removeProperty("--fam");
    card.innerHTML = `<span class="map-type">${isSub ? `Sub-topic of ${esc(topics.get(c.topic).short)}` : "Topic"}</span><h2>${esc(c.name)}</h2>
      <p>${c.size} terms. The most connected are ${members.slice(0, 3).map(id => esc(terms.get(id).name).toLowerCase()).join(", ")}.</p>`;
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

// On a phone the map fills the screen, and a drag that only ever moved the map would trap the reader in it. So one
// finger moves the map until its content would leave through the top or bottom edge, and from there, until the finger
// lifts, the drag scrolls the page. Cytoscape listens for moves on the window: this listens there first, and keeps
// those moves from it.
const EDGE = 40;   // px of content kept inside the map by a one-finger drag
let fingerY = null, paging = false;
cy.container().addEventListener("touchstart", ev => {
  fingerY = ev.touches.length === 1 ? ev.touches[0].clientY : null;
  paging = false;
}, { passive: true });
window.addEventListener("touchmove", ev => {
  if (fingerY === null || ev.touches.length !== 1) { fingerY = null; return; }
  const y = ev.touches[0].clientY, dy = y - fingerY;
  fingerY = y;
  if (!paging) {
    const bb = cy.elements(":visible").renderedBoundingBox();
    paging = (dy < 0 && bb.y2 + dy < EDGE) || (dy > 0 && bb.y1 + dy > cy.height() - EDGE);
  }
  if (paging) { ev.preventDefault(); ev.stopPropagation(); window.scrollBy(0, -dy); }
}, { capture: true, passive: false });

// A drag that went on to scroll the page ends without a tap, wherever the finger lifts.
cy.on("tap", "node.term", ev => { if (!paging) go(ev.target.id()); });
cy.on("tap", ev => {
  if (paging || (ev.target !== cy && !ev.target.isEdge())) return;   // a tap on a link counts as one on what lies under it
  const { x, y } = ev.position;
  const inside = c => Math.hypot(x - c.x, y - c.y) <= c.r;
  const overview = TERM_GAP * cy.zoom() < DETAIL_GAP;   // sub-topics are hidden there
  const hit = (!overview && [...subtopics.values()].find(inside)) || DATA.clusters.find(inside);
  if (hit && hit.id !== selected) go(hit.id);
});
document.getElementById("links").addEventListener("click", ev => {
  const b = ev.target.closest(".map-chip"); if (b) { go(b.dataset.id); document.querySelector(".map-box").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" }); }
});
const zoomBy = f => { const z = Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * f));
  cy.stop(); cy.animate({ zoom: { level: z, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: reduce ? 0 : 250 }); };
function applyHidden() {
  cy.nodes(".term").forEach(n => n.toggleClass("off", hidden.has(n.data("fam")) && !n.hasClass("sel")));
}
document.getElementById("legend").addEventListener("click", ev => {
  const b = ev.target.closest("button[data-fam]"); if (!b) return;
  const k = b.dataset.fam;
  if (hidden.has(k)) hidden.delete(k); else hidden.add(k);
  try { localStorage.setItem("factorial-map-hidden", JSON.stringify([...hidden])); } catch {}
  drawLegend();
  document.querySelector(`#legend button[data-fam="${k}"]`).focus();   // keep the keyboard where it was
  cy.batch(applyHidden); refresh(true);
});
const labelsButton = document.getElementById("labels");
const showLabelState = () => labelsButton.setAttribute("aria-pressed", String(labelsOn));
labelsButton.addEventListener("click", () => {
  labelsOn = !labelsOn;
  try { localStorage.setItem("factorial-map-labels", labelsOn ? "on" : "off"); } catch {}
  showLabelState(); refresh(true);
});
showLabelState();
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
  const hits = DATA.nodes.map(n => {
    const names = [n.name, ...n.aliases].map(norm);
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
  const m = DATA.method, N = DATA.nodes.length, E = DATA.edges.length;
  const verbs = new Set(DATA.edges.map(e => e.type)).size;
  const inChapter = k => DATA.nodes.filter(n => n.chapter === k).length;
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
    <details><summary>Show the numbers</summary>
      <p>${N} terms: ${inChapter(5)} from chapter 5 and ${inChapter(7)} from the design side of a case study in chapter 7,
        whose multivariate sections are left out. A first list was written by hand from the chapter's text; reviewers
        then checked every term and link against the chapter as it is now, and added what its newer sections cover,
        such as OMARS designs, Lenth's method and desirability.</p>
    </details>

    <h3>2. What the points and lines mean</h3>
    <p>Each point is one idea the book explains, and its shape says what kind: a design or a property of one, a method,
      a concept, a guideline or warning sign, or an example. Tap one to read its definition and the book's own words.</p>
    <p>Each line is a short sentence that joins two ideas, for example:</p>
    <ul>${examples.map(e => `<li><b>${name(e.source)}</b> ${read(e.type, 0)} <b>${name(e.target)}</b></li>`).join("")}</ul>
    <details><summary>Show the numbers</summary>
      <p>${E} links using ${verbs} verbs, each citing its line of the book. A definition is at most 32 words.</p>
    </details>

    <h3>3. How the topics are found</h3>
    <p>A topic is a group of ideas linked more to each other than to the rest. A computer finds the groups, but one try
      depends on chance, so it makes hundreds of tries and keeps the grouping they agree on. Chance then has little say
      in the topics.</p>
    <details><summary>Show the numbers</summary>
      <p>The method is <a href="https://en.wikipedia.org/wiki/Louvain_method" target="_blank" rel="noopener">Louvain clustering</a>: it moves one term at a time into a neighbour's group whenever that makes the
        groups tighter, and stops when no move helps. It visits the terms in a random order, so a term linked about
        equally to two groups lands wherever its turn falls. Consensus clustering (<a href="https://doi.org/10.1038/srep00336" target="_blank" rel="noopener">Lancichinetti and Fortunato, 2012</a>)
        makes it repeatable: ${m.runs} runs a round, each with its own random seed; a pair of terms together in at least
        ${Math.round(m.threshold * 100)}% of runs is kept, as strongly as that share, and the rest dropped; rounds repeat
        until all ${m.runs} runs agree. A term left on its own joins the group it was most often in. Link weights:
        ${light}; every other link counts 1.</p>
      <p>Two groupings are compared pair by pair: for each pair of terms, are they together in both, or apart in both?
        The share of pairs that match, corrected for what chance alone would give, is the adjusted Rand index: 1 for
        identical groupings, about 0 for unrelated ones.</p>
      <p>Single runs reproduce these ${DATA.clusters.length} topics at a median of ${m.single_median.toFixed(2)}, and as
        low as ${m.single_min.toFixed(2)}, over ${m.single_runs} seeds. Consensus from seeds it had not used reproduces
        them at ${m.consensus_other.map(x => x.toFixed(2)).join(" and ")}; the terms that still move sit between two
        topics.</p>
      <p><a href="${m.code}" target="_blank" rel="noopener">The Python code that finds the topics and sub-topics</a>.</p>
    </details>

    <h3>4. How the sub-topics are found</h3>
    <p>The larger topics are split the same way, but only where the tries agree on the split; the others are shown
      whole. Zoom past the topics to see the sub-topics as dashed circles, each named after its most connected term.</p>
    <details><summary>Show the numbers</summary>
      <p>A topic gets sub-topics when it has at least ${m.sub_min_size} terms and two other seed batches reproduce its
        split at an adjusted Rand index of ${m.sub_min_agreement} or better. That holds for ${withSubs.length} of the
        ${DATA.clusters.length} topics.</p>
      <div class="map-tablewrap"><table>
        <thead><tr><th>Topic</th><th>Terms</th><th>Second level</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      ${whole.map(c => `<p>In "${esc(c.short)}" (${c.size} terms), another seed batch moved ${quoted(c.moved)} from one
        sub-group to another; every other term kept its group. Agreement is counted over all ${pairs(c.size)} pairs of
        terms, and the pairs these ${c.moved.length} terms form with their old and new group-mates bring it down to
        ${c.agreement.toFixed(2)}. A split that changes with chance would change the map for no reason in the book, so
        this topic is shown whole.</p>`).join("")}
    </details>`;
}
renderHow();
document.getElementById("howlink").addEventListener("click", () =>
  document.getElementById("how").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }));

// ---------- start, theme and size changes
const restyle = () => { drawLegend(); refresh(true); render(); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
window.addEventListener("hashchange", () => go(fromHash(), { hash: false }));
let resizing;
window.addEventListener("resize", () => { clearTimeout(resizing); resizing = setTimeout(() => { cy.resize(); setLimits(); move(selected, false); }, 150); });
if (document.fonts) document.fonts.ready.then(() => { widths.clear(); refresh(true); });

drawLegend();
setLimits();
go(fromHash() || null, { animate: false, hash: false });
