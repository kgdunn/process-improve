// What Factorial does (features.html): the page is drawn from FEATURES.md, the document the app is described
// from, so the two cannot differ. This is a reader of the Markdown that document is written in: headings,
// paragraphs, lists, tables and rules, with bold, italic, code and links inside them. It is no reader of
// Markdown at large: what the document does not use is shown as it is typed. The document's own top says how it
// is written for the page: a point in plain words, with a line of details under it, a chart's picture under its
// point, and headings that are the page's parts. This module needs no document, so it is tested under Node.js.

/** The line of the document after which the page shows it: what comes before is for those who keep it. */
export const SHOWN_FROM = "<!-- features.html shows what follows -->";

/** The part of the document the page shows: all of it, when the line that parts it is not there. */
export function shownPart(text) {
  const at = text.indexOf(SHOWN_FROM);
  return at < 0 ? text : text.slice(at + SHOWN_FROM.length);
}

const escaped = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** A heading's address, as GitHub makes it: in lower case, its punctuation dropped, a hyphen for each space. */
export const slug = (heading) =>
  heading
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_-]/gu, "")
    .trim()
    .replace(/\s/g, "-");

/** Where the app is published. The document links to it there, so that its links hold where the document is read. */
const APP = "https://factori.al/";

/**
 * A link: one to another site opens in a tab of its own, the book's in the one tab the book has, and an address
 * of any other scheme is not followed. A
 * link to the app is one within the copy this page is served from. One named "Show me" opens the app on an
 * example of its point (app.js, ?show=), and is marked for the page to style. One that requests a design opens
 * the app's Feedback with the request written (app.js, ?request=), and names its design for a screen reader.
 */
function anchor(words, to) {
  const href = to.startsWith(APP) ? `./${to.slice(APP.length)}` : to;
  const outside = /^https?:\/\//i.test(href);
  if (!outside && /^[a-z][a-z0-9+.-]*:/i.test(href)) return words;
  const show = words === "Show me" ? ' class="show-me"' : "";
  // "Request it" is said beside each design the app does not make (app.js, ?request=): a screen reader is told which.
  const asked = href.match(/^\.\/\?request=([a-z_]+)$/);
  const which = asked ? ` aria-label="${words}: ${asked[1].replaceAll("_", " ")}"` : "";
  // The book has one tab, which the app's other links to it use too (campaign.tabOf says why its tie is kept).
  const tab = /^https:\/\/learnche\.org\/pid(?:[/?#]|$)/.test(href) ? ' target="pid-book" rel="opener"' : ' target="_blank" rel="noopener"';
  return `<a${show}${which} href="${href}"${outside ? tab : ""}>${words}</a>`;
}

/** The marks inside a line: code first, since what is inside it is not Markdown. */
function inline(text) {
  const marked = (part) =>
    part
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, words, href) => anchor(words, href))
      .replace(/&lt;(https?:\/\/[^&\s]+)&gt;/g, (_, href) => anchor(href, href))
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*\s][^*]*)\*/g, "<em>$1</em>");
  return escaped(text)
    .split(/(`[^`]+`)/)
    .map((part) => (part.startsWith("`") ? `<code>${part.slice(1, -1)}</code>` : marked(part)))
    .join("");
}

const ITEM = /^(-|\d+\.) +/;
const HEADING = /^(#{1,4}) +(.+)$/;
const RULE = /^-{3,}\s*$/;
const opens = (line) => HEADING.test(line) || RULE.test(line) || ITEM.test(line) || line.startsWith("|");

const DETAIL = /^\s+- \*Details:\* +/;
/** A point's picture, on a line of its own under it: one of the app's own, in chart-pictures/, and no other address. */
const PICTURE = /^\s+!\[\]\((chart-pictures\/[a-z0-9_]+\.webp)\)\s*$/;
/** The size every picture is drawn at (tests/web/e2e/test_chart_pictures.py), so the page keeps its room before it loads. */
export const PICTURE_SIZE = [640, 420];

/**
 * A list: each line that opens with a mark is an item, and an indented line under it goes on with it. An indented
 * line that opens with "- *Details:*" starts what only some readers ask for: the page folds it under its item.
 * An indented line that is a picture puts it above its item, and makes the list a set of cards. The words beside
 * a picture say what it shows, so it has no words of its own for a screen reader to say twice.
 */
function list(lines) {
  const items = [];
  for (const line of lines) {
    const last = items.at(-1);
    if (ITEM.test(line)) items.push({ says: line.replace(ITEM, ""), details: null, picture: null });
    else if (PICTURE.test(line)) last.picture = line.match(PICTURE)[1];
    else if (DETAIL.test(line)) last.details = line.replace(DETAIL, "");
    else if (last.details === null) last.says += ` ${line.trim()}`;
    else last.details += ` ${line.trim()}`;
  }
  const more = (details) => (details === null ? "" : `<details class="more"><summary>Details</summary><p>${inline(details)}</p></details>`);
  // "Show me" is said of many points: a screen reader is told which, by the point's opening words.
  const said = (says) => {
    const opening = says.match(/^\*\*(.+?)\*\*/)?.[1].replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/\.$/, "");
    const html = inline(says);
    return opening ? html.replace('<a class="show-me" ', `<a class="show-me" aria-label="Show me: ${escaped(opening)}" `) : html;
  };
  const [width, height] = PICTURE_SIZE;
  const drawn = (picture) =>
    picture === null ? "" : `<img src="./${picture}" alt="" width="${width}" height="${height}" loading="lazy" decoding="async">`;
  const kind = /^\d/.test(lines[0]) ? "ol" : "ul";
  const cards = items.some((item) => item.picture) ? ' class="feat-pictures"' : "";
  const item = ({ says, details, picture }) => `<li>${drawn(picture)}${said(says)}${more(details)}</li>`;
  return `<${kind}${cards}>${items.map(item).join("")}</${kind}>`;
}

/** A table: its first line is its head and its second the rule under it. It scrolls sideways where it is too wide. */
function table(lines) {
  const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|");
  const row = (line, cell) => `<tr>${cells(line).map((text) => `<${cell}>${inline(text.trim())}</${cell}>`).join("")}</tr>`;
  const [head, , ...body] = lines;
  const rows = body.map((line) => row(line, "td")).join("");
  return `<div class="scroll"><table class="data"><thead>${row(head, "th")}</thead><tbody>${rows}</tbody></table></div>`;
}

/** The document as the page's HTML. Every character of it is escaped, so it can hold no element of its own. */
export function markdown(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let at = 0;
  const run = (goesOn) => {
    const taken = [lines[at++]];
    while (at < lines.length && goesOn(lines[at])) taken.push(lines[at++]);
    return taken;
  };
  while (at < lines.length) {
    const line = lines[at];
    const heading = line.match(HEADING);
    if (!line.trim()) at += 1;
    else if (heading) {
      const level = heading[1].length;
      html.push(`<h${level} id="${slug(heading[2])}">${inline(heading[2])}</h${level}>`);
      at += 1;
    } else if (RULE.test(line)) {
      html.push("<hr>");
      at += 1;
    } else if (line.startsWith("|")) html.push(table(run((next) => next.startsWith("|"))));
    else if (ITEM.test(line)) html.push(list(run((next) => ITEM.test(next) || /^\s+\S/.test(next))));
    else html.push(`<p>${inline(run((next) => next.trim() && !opens(next)).join(" "))}</p>`);
  }
  return html.join("\n");
}

const SAYS = /^\*([^*]+)\*$/;

/**
 * The document as the page: a `##` heading is a step of a study, and a `###` heading under it a part that folds,
 * closed, with the line in italics under the heading as what it says while closed. The points of a step with no
 * part of its own are the page's cards.
 */
export function page(text) {
  const [before, ...steps] = text.replace(/\r\n?/g, "\n").split(/^(?=## )/m);
  const named = (chunk) => {
    const [head, ...rest] = chunk.split("\n");
    return [head.replace(/^#+ +/, ""), rest.join("\n")];
  };
  const part = (chunk) => {
    const [name, body] = named(chunk);
    const lines = body.split("\n");
    const at = lines.findIndex((line) => line.trim());
    const says = at < 0 ? null : lines[at].trim().match(SAYS);
    if (says) lines.splice(at, 1);
    return (
      `<details class="feat-part"><summary><h3 id="${slug(name)}">${inline(name)}</h3>` +
      `${says ? `<span class="feat-says">${inline(says[1])}</span>` : ""}</summary>` +
      `<div class="feat-body">${markdown(lines.join("\n"))}</div></details>`
    );
  };
  const step = (chunk) => {
    const [name, body] = named(chunk);
    const [lead, ...parts] = body.split(/^(?=### )/m);
    const first = markdown(lead);
    return [
      `<h2 class="feat-step" id="${slug(name)}">${inline(name)}</h2>`,
      parts.length ? first : first.replace("<ul>", '<ul class="feat-cards">'),
      ...parts.map(part),
    ];
  };
  return [markdown(before), ...steps.flatMap(step)].filter(Boolean).join("\n");
}

/**
 * The addresses the page's parts had while they were numbered, each with the start of its part's address now: a
 * link kept from then still opens at its part.
 */
export const MOVED = {
  "1-in-one-minute": "in-one-minute",
  "2-designs": "choose-a-design",
  "3-planning-a-study-design": "plan-a-study",
  "4-running-it-run": "run-it",
  "5-calculations-learn": "understand-the-results",
  "6-figures": "see-it",
  "7-interaction-and-ease-of-use": "easy-to-use",
  "8-data-in-and-out-and-reproducibility": "take-it-with-you",
  "9-enterprise-privacy-security-and-operation": "your-data-and-your-trust",
  "10-learning-and-documentation": "learn-the-method",
  "11-the-other-pages-at-factorial": "more-at-factorial",
  "12-limits": "limits",
  "13-not-in-the-app": "what-it-does-not-do",
};

/** The address, of those on the page, that the part of a link after its # names: its own, or the one it moved to. */
export function addressOf(hash, ids) {
  let id = hash.replace(/^#/, "");
  try {
    id = decodeURIComponent(id);
  } catch {
    /* not an address the page made: no part */
  }
  if (ids.includes(id)) return id;
  const to = MOVED[id];
  return (to && ids.find((other) => other.startsWith(to))) ?? null;
}
