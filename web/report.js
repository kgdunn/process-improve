// web/report.js: the study as a report, one HTML file to keep or to send. It says what the study set out to
// learn, the runs it made and their results, what the results say, and what it takes to get the same numbers
// again: its stamp names the releases that computed them, the seeds behind every random draw, and the SHA-256 of
// the study's inputs, with a link that opens the same study in Factorial.
//
// The functions here turn the study's data into HTML and know nothing of the page; app.js adds what Learn shows,
// copied from the page, and its charts as images.

import { GOALS, escape, isCategorical, isSeveral, levelsOf, listNames, responsesOf } from "./logic.js";

/**
 * The seed a design's runs are put in random order with when its specification names none: the worked example's,
 * and the studies saved before each study had a seed of its own. Mirrors bootstrap.RUN_ORDER_SEED.
 */
export const RUN_ORDER_SEED = 42;

/** The design type of runs as given, whose order is the reader's own; mirrors bootstrap.GIVEN. */
const GIVEN = "given";

/**
 * A value as JSON that does not depend on the order its keys were written in: every object's keys sorted, and a key
 * whose value is undefined left out, as JSON.stringify leaves it. Two studies with the same inputs give the same text.
 */
export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map((v) => (v === undefined ? "null" : canonicalJson(v))).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value)
      .filter((key) => value[key] !== undefined)
      .sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** The SHA-256 of a text's UTF-8 bytes, in hex; null where the browser keeps its digest from the page (no https). */
export async function sha256(text) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * A moment as the stamp gives it: the date as the reader's browser writes dates, the minute on a 24-hour clock, and
 * the time zone, since a report is read elsewhere. ``locale`` and ``timeZone`` are for the tests.
 */
export function whenWords(date, locale = undefined, timeZone = undefined) {
  const day = date.toLocaleDateString(locale, { day: "numeric", month: "long", year: "numeric", timeZone });
  const time = date.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZoneName: "short", timeZone });
  return `${day}, ${time}`;
}

/** What the numbers were computed on: Python, Pyodide when the page ran in a browser, and the numeric packages. */
export function runtimeWords(runtime) {
  if (!runtime) return "";
  const python = `Python ${runtime.python}${runtime.pyodide ? ` in Pyodide ${runtime.pyodide}` : ""}`;
  const packages = Object.entries(runtime.packages ?? {}).map(([name, release]) => `${name} ${release}`);
  return packages.length ? `${python}, with ${listNames(packages)}` : python;
}

/**
 * How the run order came about: drawn at random with the study's seed, or as the reader gave it. The runs added to
 * a study are drawn with its seed plus the number of runs before them; a study saved before studies had a seed of
 * their own used RUN_ORDER_SEED for both.
 */
export function orderWords(spec) {
  const seed = spec.options?.random_state;
  if (spec.design_type !== GIVEN) return `In random order (seed ${seed ?? RUN_ORDER_SEED}).`;
  if (!spec.options?.grown) return "As given.";
  if (seed === undefined) return `As the study had them, then the runs added in random order (seed ${RUN_ORDER_SEED}).`;
  return `As the study had them, then each set of runs added in random order (seed ${seed} plus the number of runs before it).`;
}

/**
 * Where the results came from: how many runs have one, and of those how many were entered and how many simulated,
 * from which process, with what noise and seed. ``results`` and ``marks`` are by response, a list a run each; a run
 * with any result simulated counts as simulated.
 */
export function resultWords(results, marks, simulation, runs) {
  const names = Object.keys(results);
  const given = (name, i) => results[name][i] !== null && results[name][i] !== undefined;
  const done = [...Array(runs).keys()].filter((i) => names.some((name) => given(name, i)));
  if (!done.length) return "None yet.";
  const simulated = done.filter((i) => names.some((name) => given(name, i) && marks[name]?.[i] === "simulated")).length;
  const entered = done.length - simulated;
  const from = simulation ? ` (${simulation.label}, noise ${simulation.sigma}, seed ${simulation.seed})` : "";
  const count = `${done.length} of ${runs} runs`;
  if (!simulated) return `${count}, entered.`;
  if (!entered) return `${count}, simulated for practice${from}.`;
  return `${count}, ${entered} entered and ${simulated} simulated for practice${from}.`;
}

/**
 * The stamp's rows, each [label, text, link]: when the report was made and with what, the run order's seed, where
 * the results came from, the SHA-256 of the inputs, and the link that opens the study again. A row with nothing to say
 * is left out: the runtime the page does not know, the run order and results of a page with no design yet (``spec``
 * null, as the Feedback dialog may stamp one), a digest the browser would not compute, or a link for a study too large
 * to carry in one.
 */
/** The stamp's row that holds the study's fingerprint: the SHA-256 of what the study is made of. */
export const FINGERPRINT = "Fingerprint (SHA-256)";

/** The stamp's row of the software the numbers were computed with, which the report prints at its foot. */
export const SOFTWARE = "On";

export function stampRows({ made, version, runtime, url, spec, runs, results, marks, simulation, sha, link }) {
  const rows = [
    ["Made", whenWords(made), null],
    ["With", `Factorial (${version ? `process-improve ${version}` : "process-improve, from its source"}) at ${url}`, null],
    [SOFTWARE, runtimeWords(runtime), null],
    ["Run order", spec ? orderWords(spec) : "", null],
    ["Results", spec ? resultWords(results, marks, simulation, runs) : "", null],
    [FINGERPRINT, sha ?? "", null],
    ["Open it again", link ? "This study in Factorial, with its results" : "", link],
  ];
  return rows.filter(([, text]) => text);
}

/**
 * The words a report used, each with its meaning: a list of [word, definition], in the order the report used them.
 * The word starts with a capital, as the head of its line.
 */
export function termsHtml(terms) {
  const head = (word) => word.charAt(0).toUpperCase() + word.slice(1);
  const lines = terms.map(([word, meaning]) => `<dt>${escape(head(word))}</dt><dd>${escape(meaning)}</dd>`);
  return `<dl class="terms">\n${lines.join("\n")}\n</dl>`;
}

/** An HTML table of heads and rows of text, every cell escaped; ``classes`` marks a cell, by its row and column. */
export function tableHtml(heads, rows, classes = () => "") {
  const head = `<tr>${heads.map((h) => `<th>${escape(h)}</th>`).join("")}</tr>`;
  const body = rows
    .map((row, i) => {
      const cells = row.map((cell, j) => {
        const named = classes(i, j);
        return `<td${named ? ` class="${named}"` : ""}>${escape(cell ?? "")}</td>`;
      });
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

/** A response's goal in words: a target with its value. */
const goalOf = (r) => (r.goal === "target" && r.target !== undefined ? `${GOALS.target}: ${r.target}` : (GOALS[r.goal] ?? ""));

/**
 * The brief: the factors (or a mixture's components) with their ranges, the responses with their goals, the aim in
 * the words Design gives it (``aim``, "" for none), and the limits on the settings. With a categorical factor of
 * three levels or more, each factor's levels are one column: a range, or the labels in their order.
 */
export function briefHtml(spec, { aim = "", mixture = false } = {}) {
  const factors = spec.factors.some(isSeveral)
    ? tableHtml(
        ["Factor", "Levels", "Units"],
        spec.factors.map((f) => (isCategorical(f) ? [f.name, levelsOf(f).join(", "), ""] : [f.name, `${f.low} to ${f.high}`, f.units ?? ""])),
      )
    : tableHtml(
        [mixture ? "Component" : "Factor", mixture ? "Least" : "Low", mixture ? "Most" : "High", "Units"],
        spec.factors.map((f) => [f.name, String(f.low), String(f.high), isCategorical(f) ? "" : (f.units ?? "")]),
      );
  const responses = tableHtml(
    ["Response", "Units", "Goal"],
    responsesOf(spec).map((r) => [r.label || r.name, r.units ?? "", goalOf(r)]),
  );
  const parts = [factors, responses];
  if (aim) parts.push(`<p>Aim: ${escape(aim)}.</p>`);
  const limits = spec.constraints ?? [];
  if (limits.length) {
    const items = limits.map((l) => `<li><code>${escape(l.expression)}</code>${l.note ? `: ${escape(l.note)}` : ""}</li>`);
    parts.push(`<p>Limits on the settings:</p><ul>${items.join("")}</ul>`);
  }
  return parts.join("\n");
}

/**
 * The run sheet with its results, as Run shows it: the design's columns, each response's column holding its
 * results, a simulated one in italics. ``results`` and ``marks`` are by response, a list a run each.
 */
export function runSheetHtml(design, results, marks) {
  const names = new Set(Object.keys(results));
  const simulated = (i, j) => names.has(design.columns[j]) && marks[design.columns[j]]?.[i] === "simulated";
  const rows = design.rows.map((row, i) =>
    design.columns.map((column, j) => {
      const value = names.has(column) ? results[column][i] : row[j];
      return value === null || value === undefined ? "" : String(value);
    }),
  );
  const sheet = tableHtml(design.columns, rows, (i, j) => (simulated(i, j) && rows[i][j] !== "" ? "simulated" : ""));
  const any = design.rows.some((_, i) => design.columns.some((_, j) => simulated(i, j) && rows[i][j] !== ""));
  return any ? `${sheet}\n<p class="hint">A result in italics was simulated.</p>` : sheet;
}

/**
 * The charts, each {title, caption, src, width}: a figure with its heading, its image and its caption. The image is
 * shown at the ``width`` it was drawn for, so a figure that keeps its shape is not stretched to the page's width.
 */
export function chartsHtml(charts) {
  return charts
    .map(
      (c) =>
        `<figure><h4>${escape(c.title)}</h4><img src="${escape(c.src)}" alt="${escape(c.title)}"${c.width ? ` width="${c.width}"` : ""}>` +
        `${c.caption ? `<figcaption>${escape(c.caption)}</figcaption>` : ""}</figure>`,
    )
    .join("\n");
}

/** The report's own style: a page of print, on white whatever theme the app shows, that reads on a phone too. */
const STYLE = `
:root { color-scheme: light; --fg: #1d1d1f; --muted: #5f6368; --line: #d6d9d6; --accent: #2c5f7c; --warn: #a35f00; }
* { box-sizing: border-box; }
body { margin: 0; background: #ffffff; color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 12px; }
h2 { font-size: 1.25rem; margin: 32px 0 8px; padding-top: 12px; border-top: 2px solid var(--accent); }
h3 { font-size: 1.05rem; margin: 20px 0 6px; }
h4 { font-size: 0.95rem; margin: 16px 0 4px; }
.kicker { margin: 0; color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.8rem; }
.hint, figcaption { color: var(--muted); font-size: 0.9rem; }
a { color: var(--accent); }
.scroll, section > table, .wide { overflow-x: auto; }
table { border-collapse: collapse; margin: 8px 0; font-variant-numeric: tabular-nums; display: block; overflow-x: auto; max-width: 100%; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: right; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; }
td.simulated { font-style: italic; }
.spark { display: inline-block; width: 2.8em; height: 1.4em; margin-right: 0.4em; vertical-align: -0.35em; overflow: visible; }
.spark polyline { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; vector-effect: non-scaling-stroke; }
.spark .spark__base { stroke: var(--line); stroke-width: 1; }
.spark .spark__line--other { stroke-dasharray: 3 2.5; }
.finding__numbers { display: block; color: var(--muted); font-size: 0.9em; }
.dim { color: var(--muted); }
td.strong { font-weight: 700; }
td.low { color: var(--warn); }
dl.terms { margin: 0; font-size: 0.9rem; }
dl.terms dt { font-weight: 600; margin-top: 8px; }
dl.terms dd { margin: 0; color: var(--muted); }
table.stamp { display: table; width: 100%; }
table.stamp th { width: 9.5em; text-align: left; color: var(--muted); font-weight: 500; }
table.stamp td { text-align: left; overflow-wrap: anywhere; }
table.stamp td.digest { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.85rem; }
.equation { font-family: "Cambria Math", "STIX Two Math", serif; font-size: 1.05rem; overflow-x: auto; }
figure { margin: 16px 0; break-inside: avoid; }
figure img { max-width: 100%; height: auto; display: block; }
pre { background: #f6f5f1; border: 1px solid var(--line); padding: 12px; overflow-x: auto; font-size: 0.8rem; line-height: 1.4; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; }
ul { padding-left: 1.2em; }
footer { margin-top: 40px; color: var(--muted); font-size: 0.85rem; }
@media print {
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; }
  table, pre { break-inside: avoid; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  a { color: inherit; }
}
`;

/**
 * The report, a whole HTML page: its title, the stamp's rows (see stampRows), a note on what the digest is of, and
 * each section, {heading, html}, whose HTML is the caller's to have escaped.
 */
export function report({ title, stamp, sections }) {
  const software = stamp.find(([label]) => label === SOFTWARE)?.[1];
  const rows = stamp
    .filter(([label]) => label !== SOFTWARE) // the versions are for the reader who checks the numbers: at the foot
    .map(([label, text, link]) => {
      const value = link ? `<a href="${escape(link)}">${escape(text)}</a>` : escape(text);
      const digest = label === FINGERPRINT ? ' class="digest"' : "";
      return `<tr><th>${escape(label)}</th><td${digest}>${value}</td></tr>`;
    })
    .join("\n");
  const digested = stamp.some(([label]) => label === FINGERPRINT);
  const note = digested
    ? '<p class="hint">The fingerprint is made from this study: its brief, design, results and models. Two reports ' +
      "with the same fingerprint are of the same study.</p>"
    : "";
  const versions = software ? `\n<p class="versions">Software versions: ${escape(software)}.</p>` : "";
  const body = sections.map((s) => `<section>\n<h2>${escape(s.heading)}</h2>\n${s.html}\n</section>`).join("\n");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(title)}: report</title>
<style>${STYLE}</style>
</head>
<body>
<main>
<header>
<p class="kicker">Factorial study report</p>
<h1>${escape(title)}</h1>
<table class="stamp" aria-label="How this report was made">
${rows}
</table>
${note}
</header>
${body}
<footer><p>Made with Factorial, which runs process-improve in the browser. The numbers above can be computed again
from the link at the top, or from the Python code.</p>${versions}</footer>
</main>
</body>
</html>
`;
}
