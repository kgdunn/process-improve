// web/features.js: the reader of the Markdown that FEATURES.md is written in, for the page that shows it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { MOVED, SHOWN_FROM, addressOf, markdown, page, shownPart, slug } from "../../../web/features.js";

const DOCUMENT = readFileSync(new URL("../../../web/FEATURES.md", import.meta.url), "utf8");

describe("the Markdown of the features document, as HTML", () => {
  test("a heading has the address GitHub gives it, so the document's own links to its parts hold on the page", () => {
    assert.equal(slug("More at factori.al"), "more-at-factorial");
    assert.equal(slug("See it: 17 charts"), "see-it-17-charts");
    assert.equal(markdown("## Design"), '<h2 id="design">Design</h2>');
    assert.equal(markdown("#### Checking the fit"), '<h4 id="checking-the-fit">Checking the fit</h4>');
  });

  test("lines with no blank line between them are one paragraph", () => {
    assert.equal(markdown("One line\nand the next.\n\nAnother."), "<p>One line and the next.</p>\n<p>Another.</p>");
  });

  test("a list's item goes on over its indented lines, and a numbered list is numbered", () => {
    assert.equal(markdown("- **First.** It goes\n  on here.\n- Second."), "<ul><li><strong>First.</strong> It goes on here.</li><li>Second.</li></ul>");
    assert.equal(markdown("1. [One](#one)\n2. Two"), '<ol><li><a href="#one">One</a></li><li>Two</li></ol>');
  });

  test("a point's line of details is folded under it, and goes on over its own indented lines", () => {
    assert.equal(
      markdown("- **Plain.** For all.\n  - *Details:* For `some`,\n    over two lines.\n- Next."),
      '<ul><li><strong>Plain.</strong> For all.<details class="more"><summary>Details</summary>' +
        "<p>For <code>some</code>, over two lines.</p></details></li><li>Next.</li></ul>",
    );
    assert.doesNotMatch(markdown("- A point.\n  It goes on."), /<details/, "a point with no such line has no fold");
  });

  test("a table has its head, and scrolls sideways where it is too wide", () => {
    const html = markdown("| Design | Factors |\n|---|---|\n| Box-Behnken | 3 to 7 |\n| D-optimal | 2 to 8 |");
    assert.equal(
      html,
      '<div class="scroll"><table class="data"><thead><tr><th>Design</th><th>Factors</th></tr></thead>' +
        "<tbody><tr><td>Box-Behnken</td><td>3 to 7</td></tr><tr><td>D-optimal</td><td>2 to 8</td></tr></tbody></table></div>",
    );
    assert.match(markdown("| A | B |\n|---|---|\n| 1 | |"), /<td>1<\/td><td><\/td>/, "a cell left empty");
  });

  test("bold, italic, code and a rule", () => {
    assert.equal(markdown("**Up to 40 factors.** In *the book*."), "<p><strong>Up to 40 factors.</strong> In <em>the book</em>.</p>");
    assert.equal(markdown("such as `T + 25*P <= 250`."), "<p>such as <code>T + 25*P &lt;= 250</code>.</p>", "code is not Markdown");
    assert.equal(markdown("---"), "<hr>");
  });

  test("a link to another site opens in a tab of its own, and one within the app in this one", () => {
    const outside = ' target="_blank" rel="noopener"';
    assert.equal(markdown("[#1225](https://github.com/x/y/issues/1225)"), `<p><a href="https://github.com/x/y/issues/1225"${outside}>#1225</a></p>`);
    assert.equal(markdown("at <https://kgdunn.github.io/process-improve/>."), `<p>at <a href="https://kgdunn.github.io/process-improve/"${outside}>https://kgdunn.github.io/process-improve/</a>.</p>`);
    const book = ' target="pid-book" rel="opener"';
    assert.equal(markdown("in <https://learnche.org/pid>."), `<p>in <a href="https://learnche.org/pid"${book}>https://learnche.org/pid</a>.</p>`, "the book's one tab");
    assert.equal(markdown("[a page](https://learnche.org/pid/design-analysis-experiments#x)"), `<p><a href="https://learnche.org/pid/design-analysis-experiments#x"${book}>a page</a></p>`);
    assert.match(markdown("[no book](https://learnche.org.example/pid)"), /target="_blank" rel="noopener"/, "an address that only begins as the book's");
    assert.equal(markdown("[the catalog](catalog/)"), '<p><a href="catalog/">the catalog</a></p>');
    assert.equal(markdown("[Limits](#12-limits)"), '<p><a href="#12-limits">Limits</a></p>');
  });

  test("a link to the app is one within this copy of it, and Show me says which point it shows", () => {
    assert.equal(markdown("[the app](https://factori.al/)"), '<p><a href="./">the app</a></p>', "in this tab, at this address");
    assert.equal(
      markdown("- **The best settings.** For your goal. [Show me](https://factori.al/?example=worked&show=best)\n  - *Details:* More."),
      '<ul><li><strong>The best settings.</strong> For your goal. <a class="show-me" aria-label="Show me: The best settings" ' +
        'href="./?example=worked&amp;show=best">Show me</a><details class="more"><summary>Details</summary><p>More.</p></details></li></ul>',
    );
    assert.match(markdown("- **[A page](map/).** It. [Show me](https://factori.al/?book=x&show=y)"), /aria-label="Show me: A page" href="\.\/\?book=x&amp;show=y"/);
    assert.doesNotMatch(markdown("- A point with no opening words. [Show me](https://factori.al/?example=worked)"), /aria-label/);
  });

  test("the document can hold no element of its own, and no address but a page's", () => {
    assert.equal(markdown('<img src=x onerror="alert(1)"> & more'), "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; more</p>");
    assert.equal(markdown("[press](javascript:alert(1))"), "<p>press)</p>", "no link is made of it");
    assert.equal(markdown('[a](https://x.y/"onclick="z)'), '<p><a href="https://x.y/&quot;onclick=&quot;z" target="_blank" rel="noopener">a</a></p>');
  });
});

describe("the document as the page: steps, parts that fold, and cards", () => {
  const text = [
    "Kept for a page with no step.",
    "## In brief",
    "- **One.** A card.",
    "## Learn",
    "### Understand it",
    "*What it says while closed.*",
    "A line inside.",
    "#### A heading inside",
    "- A point.",
    "### No line of its own",
    "- Another.",
  ].join("\n\n");
  const html = page(text);

  test("a step is a heading, and the points of a step with no part are cards", () => {
    assert.match(html, /^<p>Kept for a page with no step\.<\/p>\n<h2 class="feat-step" id="in-brief">In brief<\/h2>\n<ul class="feat-cards"><li>/);
    assert.match(html, /<h2 class="feat-step" id="learn">Learn<\/h2>\n<details/);
    assert.equal((html.match(/feat-cards/g) ?? []).length, 1, "a list inside a part is no card");
  });

  test("a part folds, closed, under its name and the line in italics that says what it holds", () => {
    assert.match(
      html,
      /<details class="feat-part"><summary><h3 id="understand-it">Understand it<\/h3><span class="feat-says">What it says while closed\.<\/span><\/summary><div class="feat-body"><p>A line inside\.<\/p>\n<h4 id="a-heading-inside">A heading inside<\/h4>\n<ul><li>A point\.<\/li><\/ul><\/div><\/details>/,
    );
    assert.match(html, /<summary><h3 id="no-line-of-its-own">No line of its own<\/h3><\/summary>/, "a part with no such line");
    assert.doesNotMatch(html, /<details[^>]* open/);
  });

  test("an address kept from when the parts were numbered leads to the part it named", () => {
    const ids = ["limits", "see-it-17-charts", "the-17-charts"];
    assert.equal(addressOf("#limits", ids), "limits");
    assert.equal(addressOf("#12-limits", ids), "limits");
    assert.equal(addressOf("#6-figures", ids), "see-it-17-charts", "whatever the count in its name is now");
    assert.equal(addressOf("#the-17-charts", ids), "the-17-charts");
    assert.equal(addressOf("#no-such-part", ids), null);
    assert.equal(addressOf("#%E0%A4%A", ids), null, "an address no page made");
    assert.equal(addressOf("", ids), null);
  });
});

describe("the features document itself, as the page shows it", () => {
  const shown = shownPart(DOCUMENT);
  const html = page(shown);
  const ids = [...html.matchAll(/<h\d[^>]* id="([^"]+)"/g)].map(([, id]) => id);

  test("the page shows the document from its headline points: what comes before is for those who keep it", () => {
    assert.equal(DOCUMENT.split(SHOWN_FROM).length, 2, "the line that parts the document is in it once");
    assert.ok(html.startsWith('<h2 class="feat-step" id="in-one-minute">In one minute</h2>\n<ul class="feat-cards"><li><strong>'));
    assert.doesNotMatch(html, /CONTRIBUTING|test_features/, "how the document is kept is not on the page");
    assert.equal(shownPart("No such line"), "No such line");
  });

  test("the steps are those of a study, and each part under them folds with a line that says what it holds", () => {
    const steps = [...html.matchAll(/<h2 class="feat-step" id="[^"]+">([^<]+)<\/h2>/g)].map(([, name]) => name);
    assert.deepEqual(steps, ["In one minute", "Design", "Run", "Learn", "Around the study", "The fine print"]);
    const parts = [...html.matchAll(/<details class="feat-part"><summary><h3 id="([^"]+)">[^<]+<\/h3>(<span class="feat-says">[^<]+<\/span>)?<\/summary>/g)];
    assert.equal(parts.length, (shown.match(/^### /gm) ?? []).length);
    assert.ok(parts.length >= 12);
    assert.ok(parts.every(([, , says]) => says), "each says what it holds while closed");
    assert.equal(new Set(ids).size, ids.length, "no heading's address is used twice");
  });

  test("every line of details is folded under its point, and none is left as it was typed", () => {
    assert.equal((html.match(/<details class="more">/g) ?? []).length, (shown.match(/^ +- \*Details:\* /gm) ?? []).length);
    assert.doesNotMatch(html, /Details:/);
    assert.ok((html.match(/<details class="more">/g) ?? []).length > 50);
  });

  test("each address the parts had while they were numbered still leads to a part, and no two to one", () => {
    const now = Object.keys(MOVED).map((old) => addressOf(`#${old}`, ids));
    assert.ok(now.every(Boolean), JSON.stringify(now));
    assert.equal(new Set(now).size, 13);
  });

  test("every link is to another site, to a part of the page or to a page of the app: none leads into the repository", () => {
    const links = [...html.matchAll(/href="([^"]+)"/g)].map(([, href]) => href);
    assert.ok(links.length > 8);
    for (const href of links) assert.match(href, /^(https:\/\/|#|about\.html$|catalog\/|map\/$|mva\/$|\.\/\?(example|book)=[a-z0-9-]+&amp;show=[a-z0-9-]+$)/, href);
    assert.ok(links.filter((to) => to.startsWith("./?")).length >= 15, "the points that end with Show me");
    const sites = links.filter((to) => to.startsWith("https://")).map((to) => new URL(to.replaceAll("&amp;", "&")).hostname);
    assert.ok(!sites.includes("factori.al"), "a link to the app stays within this copy of it");
    assert.ok(["about.html", "catalog/", "catalog/reproduced/", "map/", "mva/"].every((to) => links.includes(to)));
    for (const href of links.filter((to) => to.startsWith("#"))) assert.ok(ids.includes(href.slice(1)), href);
  });

  test("nothing of the Markdown is left as it was typed", () => {
    assert.equal((html.match(/<table/g) ?? []).length, 1, "the limits");
    assert.doesNotMatch(html, /\*\*|\]\(|^\||^#|<!--/m);
    assert.doesNotMatch(html.replace(/<code>.*?<\/code>/g, ""), /`|\*/);
  });
});
