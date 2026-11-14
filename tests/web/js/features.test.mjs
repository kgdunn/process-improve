// web/features.js: the reader of the Markdown that FEATURES.md is written in, for the page that shows it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { SHOWN_FROM, markdown, shownPart, slug } from "../../../web/features.js";

const DOCUMENT = readFileSync(new URL("../../../web/FEATURES.md", import.meta.url), "utf8");

describe("the Markdown of the features document, as HTML", () => {
  test("a heading has the address GitHub gives it, so the document's own links to its parts hold on the page", () => {
    assert.equal(slug("11. The other pages at factori.al"), "11-the-other-pages-at-factorial");
    assert.equal(slug("9. Enterprise: privacy, security and operation"), "9-enterprise-privacy-security-and-operation");
    assert.equal(markdown("## 2. Designs"), '<h2 id="2-designs">2. Designs</h2>');
    assert.equal(markdown("### The fit"), '<h3 id="the-fit">The fit</h3>');
  });

  test("lines with no blank line between them are one paragraph", () => {
    assert.equal(markdown("One line\nand the next.\n\nAnother."), "<p>One line and the next.</p>\n<p>Another.</p>");
  });

  test("a list's item goes on over its indented lines, and a numbered list is numbered", () => {
    assert.equal(markdown("- **First.** It goes\n  on here.\n- Second."), "<ul><li><strong>First.</strong> It goes on here.</li><li>Second.</li></ul>");
    assert.equal(markdown("1. [One](#one)\n2. Two"), '<ol><li><a href="#one">One</a></li><li>Two</li></ol>');
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
    assert.equal(markdown("at <https://factori.al/>."), `<p>at <a href="https://factori.al/"${outside}>https://factori.al/</a>.</p>`);
    assert.equal(markdown("[the catalog](catalog/)"), '<p><a href="catalog/">the catalog</a></p>');
    assert.equal(markdown("[Limits](#12-limits)"), '<p><a href="#12-limits">Limits</a></p>');
  });

  test("the document can hold no element of its own, and no address but a page's", () => {
    assert.equal(markdown('<img src=x onerror="alert(1)"> & more'), "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; more</p>");
    assert.equal(markdown("[press](javascript:alert(1))"), "<p>press)</p>", "no link is made of it");
    assert.equal(markdown('[a](https://x.y/"onclick="z)'), '<p><a href="https://x.y/&quot;onclick=&quot;z" target="_blank" rel="noopener">a</a></p>');
  });
});

describe("the features document itself, as the page shows it", () => {
  const html = markdown(shownPart(DOCUMENT));
  const ids = [...html.matchAll(/<h\d id="([^"]+)"/g)].map(([, id]) => id);

  test("the page shows the document from its list of contents: what comes before is for those who keep it", () => {
    assert.equal(DOCUMENT.split(SHOWN_FROM).length, 2, "the line that parts the document is in it once");
    assert.ok(html.startsWith("<ol><li><a href=\"#1-in-one-minute\">In one minute</a></li>"));
    assert.doesNotMatch(html, /CONTRIBUTING|test_features/, "how the document is kept is not on the page");
    assert.equal(shownPart("No such line"), "No such line");
  });

  test("each entry of the contents leads to a heading, and no heading's address is used twice", () => {
    const contents = [...html.slice(0, html.indexOf("</ol>")).matchAll(/href="#([^"]+)"/g)].map(([, id]) => id);
    assert.equal(contents.length, 13);
    assert.deepEqual(contents, ids.filter((id) => /^\d+-/.test(id)));
    assert.equal(new Set(ids).size, ids.length);
  });

  test("every link is to another site, to a part of the page or to a page of the app: none leads into the repository", () => {
    const links = [...html.matchAll(/href="([^"]+)"/g)].map(([, href]) => href);
    assert.ok(links.length > 13, "more than the contents");
    for (const href of links) assert.match(href, /^(https:\/\/|#|about\.html$|catalog\/|map\/$|mva\/$)/, href);
    assert.ok(["about.html", "catalog/", "catalog/reproduced/", "map/", "mva/"].every((page) => links.includes(page)));
  });

  test("nothing of the Markdown is left as it was typed", () => {
    assert.equal((html.match(/<table/g) ?? []).length, 3, "the designs, the charts and the limits");
    assert.doesNotMatch(html, /\*\*|\]\(|^\||^#|<!--/m);
    assert.doesNotMatch(html.replace(/<code>.*?<\/code>/g, ""), /`/);
  });
});
