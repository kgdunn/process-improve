// The footer's line of how much factori.al is used: the readers of each of the last 90 days, from usage.json, which
// the server writes each hour (deploy/factorial/usage.py --public). It draws them as a line, with a red dot on the
// last day, and says that day's count beside it: "89 users today". A reader is a network and a browser in a day, so
// the count is approximate, and nothing of anyone's is in the file.
//
// A build carries a usage.json of no days, which factori.al's server serves its own in place of: a copy of Factorial
// at another address draws nothing, and nor does a visit that cannot fetch the file, or a file a day or more old.
// The page's footer (index.html) and the footer every other page shares (site.js) each give it an item to fill.

/** Where the counts are: beside this module, at the site's root. */
export const USAGE_URL = new URL("usage.json", import.meta.url);
/** Where the line leads: what the server counts, and how. */
export const USAGE_PAGE = new URL("technical-details-how-factorial-works.html#h-servers", import.meta.url);

/** The line's size in CSS pixels, and the room left round it for the dot. */
const WIDTH = 72;
const HEIGHT = 22;
const PAD = 3;

/** The counts in the file as ``[day, count]`` pairs, oldest first; null when it holds none, or is not as written. */
export function usageDays(data) {
  const days = data?.days;
  if (!Array.isArray(days) || !days.length) return null;
  const counted = (pair) =>
    Array.isArray(pair) && /^\d{4}-\d{2}-\d{2}$/.test(pair[0]) && Number.isInteger(pair[1]) && pair[1] >= 0;
  return days.every(counted) ? days : null;
}

/**
 * What the line says of its last day, ``day``, a date as ``YYYY-MM-DD``: "89 users today", or "1 user yesterday" in
 * the hour after midnight, before the server writes the new day. ``today`` is the date in UTC, the server's. A
 * count older than that is stale: null, and the line is left out.
 */
export function usageWords(day, count, today) {
  const before = new Date(`${today}T00:00:00Z`);
  before.setUTCDate(before.getUTCDate() - 1);
  const when = day === today ? "today" : day === before.toISOString().slice(0, 10) ? "yesterday" : null;
  return when && { count: `${count.toLocaleString("en-US")} ${count === 1 ? "user" : "users"}`, when };
}

/** The line's points, oldest first, as ``[x, y]`` in its box: the most in a day at the top, none at the foot. */
export function sparkPoints(counts, width = WIDTH, height = HEIGHT) {
  const top = Math.max(...counts, 1);
  const step = counts.length > 1 ? (width - 2 * PAD) / (counts.length - 1) : 0;
  const at = (count) => height - PAD - (count / top) * (height - 2 * PAD);
  return counts.map((count, i) => [PAD + i * step, at(count)].map((value) => Math.round(value * 10) / 10));
}

/** The line, drawn: an SVG of the counts with a dot on the last. It is a picture; the words beside it say it. */
function sparkline(counts) {
  const points = sparkPoints(counts);
  const [x, y] = points.at(-1);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [name, value] of Object.entries({ viewBox: `0 0 ${WIDTH} ${HEIGHT}`, width: WIDTH, height: HEIGHT })) {
    svg.setAttribute(name, String(value));
  }
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const line = document.createElementNS(svg.namespaceURI, "polyline");
  line.setAttribute("points", points.map((point) => point.join(",")).join(" "));
  const dot = document.createElementNS(svg.namespaceURI, "circle");
  for (const [name, value] of Object.entries({ cx: x, cy: y, r: 2.6 })) dot.setAttribute(name, String(value));
  svg.append(line, dot);
  return svg;
}

/**
 * Fill the footer's ``item`` with the line, and show it: a link to how use is counted, holding the line and its
 * words. Nothing is shown, and nothing is said, when the counts cannot be had.
 */
export async function drawUsage(item) {
  if (!item) return;
  let data = null;
  try {
    const response = await fetch(USAGE_URL, { cache: "no-cache" });
    if (response.ok) data = await response.json();
  } catch {
    return; // no network, or a file that is no JSON: the footer goes without the line
  }
  const days = usageDays(data);
  const [day, count] = days?.at(-1) ?? [];
  const words = days && usageWords(day, count, new Date().toISOString().slice(0, 10));
  if (!words) return;
  const link = document.createElement("a");
  link.className = "usage";
  link.href = USAGE_PAGE.href;
  link.title = `People who used Factorial each day, the last ${days.length} days: how it is counted`;
  const said = document.createElement("span");
  said.append(words.count, document.createElement("br"), words.when);
  link.append(sparkline(days.map(([, each]) => each)), said);
  item.replaceChildren(link);
  item.hidden = false;
}
