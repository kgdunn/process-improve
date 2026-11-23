// What every page of factori.al beside the app shares (about, the user guide, the catalog, validation, the
// textbooks' answers, the page for learning): the color-mode switch at the top, and one footer with the same links
// in the same place. A page loads this module and gets both; the app's own footer (index.html) holds the same
// links, kept in step by tests/web/js/site.test.mjs. No Python runs here.
// The footer names a page in one to three words, and one page for each question a reader has: the textbooks'
// answers are reached from "Is it accurate?", and the book, the course and its videos from the page for learning.
import { MODES, nextMode } from "./logic.js";

/** The footer's links, in groups: each a heading and its links, a path from the site's root or a whole address. */
export const SITE_LINKS = [
  {
    head: "The app",
    links: [
      ["./", "Open the app"],
      ["features.html", "User guide"],
      ["catalog/", "Design catalog"],
      ["validation.html", "Is it accurate?"],
      ["map/", "Learn experimental design"],
    ],
  },
  {
    head: "Behind it",
    links: [
      ["technical-details-how-factorial-works.html", "How it works"],
      ["https://github.com/kgdunn/process-improve", "Python package"],
    ],
  },
];

const MODE_KEY = "factorial-mode";
const MODE_NAMES = { auto: "the system's", light: "light", dark: "dark" };
/** The switch's three pictures, as the app's bar draws them: the mode now is the one shown (style.css). */
const MODE_ICONS =
  '<svg data-for="auto" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 3.5a8.5 8.5 0 0 0 0 17z" fill="currentColor"/></svg>' +
  '<svg data-for="light" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="4.5" fill="currentColor"/><g stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></g></svg>' +
  '<svg data-for="dark" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20.5 14.2A8.5 8.5 0 1 1 9.8 3.5a6.8 6.8 0 0 0 10.7 10.7z" fill="currentColor"/></svg>';

/** A link's address from this page: a path from the site's root, which is where this module is served from. */
const address = (path) => (/^https?:/.test(path) ? path : new URL(path, import.meta.url).href);

/** Show a color mode: the page's colors, the browser's own bar, and what the switch is called. */
function applyMode(mode, button) {
  const root = document.documentElement;
  root.dataset.mode = mode;
  if (mode === "auto") delete root.dataset.theme;
  else root.dataset.theme = mode;
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    const scheme = meta.dataset.scheme ?? (meta.media.includes("dark") ? "dark" : "light");
    meta.dataset.scheme = scheme;
    meta.media = mode === "auto" ? `(prefers-color-scheme: ${scheme})` : scheme === mode ? "all" : "not all";
  }
  button.setAttribute("aria-label", `Color mode: ${MODE_NAMES[mode]}`);
}

/**
 * Put the color-mode switch at the end of the page's top line, beside the way back: light, dark or the system's,
 * in turn, kept in this browser under the key the app and the concept map use. A page with a switch of its own
 * (the concept map) keeps it.
 */
function modeSwitch() {
  if (document.getElementById("mode")) return;
  const back = document.querySelector(".about__back");
  if (!back) return;
  const top = document.createElement("div");
  top.className = "site-top";
  back.replaceWith(top);
  const button = document.createElement("button");
  Object.assign(button, { type: "button", id: "mode", className: "mode-switch", title: "light/dark", innerHTML: MODE_ICONS });
  top.append(back, button);
  const systemDark = matchMedia("(prefers-color-scheme: dark)");
  // The page's head has put the mode kept in this browser on the page already, as its theme: the switch starts there.
  const kept = document.documentElement.dataset.mode ?? document.documentElement.dataset.theme;
  applyMode(MODES.includes(kept) ? kept : "auto", button);
  button.addEventListener("click", () => {
    const mode = nextMode(document.documentElement.dataset.mode, systemDark.matches);
    try {
      if (mode === "auto") localStorage.removeItem(MODE_KEY);
      else localStorage.setItem(MODE_KEY, mode);
    } catch {
      /* storage refused: the mode holds for this visit */
    }
    applyMode(mode, button);
  });
}

/** The footer every page ends with: the same groups of links, the page on show marked as the current one. */
function footer() {
  const main = document.querySelector("main");
  if (!main || document.querySelector(".site-foot")) return;
  const foot = document.createElement("footer");
  foot.className = "foot site-foot";
  const nav = document.createElement("nav");
  nav.className = "foot__nav";
  nav.setAttribute("aria-label", "Factorial's pages");
  const here = location.href.replace(/[?#].*$/, "").replace(/index\.html$/, "");
  SITE_LINKS.forEach((group, i) => {
    const box = document.createElement("div");
    const head = document.createElement("p");
    Object.assign(head, { className: "foot__head", id: `site-foot-${i}`, textContent: group.head });
    const list = document.createElement("ul");
    list.setAttribute("aria-labelledby", head.id);
    for (const [path, words] of group.links) {
      const link = document.createElement("a");
      Object.assign(link, { href: address(path), textContent: words });
      if (/^https?:/.test(path)) Object.assign(link, { target: "_blank", rel: "noopener" });
      else if (link.href === here) link.setAttribute("aria-current", "page");
      const item = document.createElement("li");
      item.append(link);
      list.append(item);
    }
    box.append(head, list);
    nav.append(box);
  });
  foot.append(nav);
  main.append(foot);
}

if (typeof document !== "undefined") {
  modeSwitch();
  footer();
}
