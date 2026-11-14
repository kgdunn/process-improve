// What Factorial does (features.html): the page is drawn from FEATURES.md, the document the app is described
// from, so the two cannot differ. This is a reader of the Markdown that document is written in: headings,
// paragraphs, lists, tables and rules, with bold, italic, code and links inside them. It is no reader of
// Markdown at large: what the document does not use is shown as it is typed. This module needs no document, so
// it is tested under Node.js.

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

/** A link: one to another site opens in a tab of its own, and an address of any other scheme is not followed. */
function anchor(words, href) {
  const outside = /^https?:\/\//i.test(href);
  if (!outside && /^[a-z][a-z0-9+.-]*:/i.test(href)) return words;
  return `<a href="${href}"${outside ? ' target="_blank" rel="noopener"' : ""}>${words}</a>`;
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

/** A list: each line that opens with a mark is an item, and an indented line under it goes on with it. */
function list(lines) {
  const items = [];
  for (const line of lines) {
    if (ITEM.test(line)) items.push(line.replace(ITEM, ""));
    else items[items.length - 1] += ` ${line.trim()}`;
  }
  const kind = /^\d/.test(lines[0]) ? "ol" : "ul";
  return `<${kind}>${items.map((item) => `<li>${inline(item)}</li>`).join("")}</${kind}>`;
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
