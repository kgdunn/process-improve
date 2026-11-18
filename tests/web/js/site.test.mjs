// The footer every page shares (web/site.js), and the app's own footer, which holds the same links by hand.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { SITE_LINKS } from "../../../web/site.js";

const page = readFileSync(new URL("../../../web/index.html", import.meta.url), "utf8");
const footer = page.slice(page.indexOf('<footer class="foot">'), page.indexOf("</footer>"));
const linked = [...footer.matchAll(/<li><a href="([^"]+)"[^>]*>([^<]+)<\/a><\/li>/g)].map((found) => [found[1], found[2].replaceAll("&amp;", "&")]);

describe("one footer on every page", () => {
  test("the app's footer lists the links of the shared footer, in its order", () => {
    assert.deepEqual(linked, SITE_LINKS.flatMap((group) => group.links));
  });
  test("its groups are headed as the shared footer's are", () => {
    for (const group of SITE_LINKS) assert.ok(footer.includes(`>${group.head}</p>`), group.head);
  });
  test("every page of the site is in it, the textbooks' answers among them", () => {
    const paths = SITE_LINKS.flatMap((group) => group.links).map(([path]) => path).filter((path) => !path.startsWith("https:"));
    assert.deepEqual(paths, ["./", "features.html", "catalog/", "catalog/reproduced/", "map/", "about.html"]);
  });
});
