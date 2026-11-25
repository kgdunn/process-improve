// web/usage.js: the footer's line of how many people used Factorial each day, from the usage.json the server writes.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { USAGE_PAGE, USAGE_URL, sparkPoints, usageCounts, usageWords } from "../../../web/usage.js";

const NOW = Date.parse("2026-11-25T10:30:00Z");
const DAYS = [["2026-11-23", 81], ["2026-11-24", 89]];

describe("the counts", () => {
  test("are the file's finished days, oldest first, and the last 24 hours, as the server writes them", () => {
    const data = { updated: "2026-11-25T10:17:00Z", days: DAYS, last_24_hours: 30 };
    assert.deepEqual(usageCounts(data, NOW), { days: DAYS, recent: 30 });
    assert.deepEqual(usageCounts({ ...data, days: [] }, NOW), { days: [], recent: 30 }, "a site's first day");
  });
  test("are none for a file a day or more old, which is stale: the line is left out", () => {
    const data = { days: DAYS, last_24_hours: 30 };
    assert.ok(usageCounts({ ...data, updated: "2026-11-24T10:31:00Z" }, NOW), "23 hours 59 minutes old");
    assert.equal(usageCounts({ ...data, updated: "2026-11-24T10:30:00Z" }, NOW), null, "a day old");
    assert.equal(usageCounts({ ...data, updated: null }, NOW), null, "no time written");
  });
  test("are none for the build's file, an older server's, or one not as the server writes it", () => {
    const build = JSON.parse(readFileSync(new URL("../../../web/usage.json", import.meta.url), "utf8"));
    assert.equal(usageCounts(build, NOW), null, "a copy of Factorial elsewhere draws nothing");
    const updated = "2026-11-25T10:17:00Z";
    assert.equal(usageCounts({ updated, days: DAYS }, NOW), null, "a server that does not count the last 24 hours");
    for (const wrong of [{ days: "89" }, { days: [["2026-11-25"]] }, { days: [["25 Nov", 89]] }, { days: [["2026-11-25", -1]] }, { days: [["2026-11-25", 1.5]] }, { last_24_hours: -1 }, { last_24_hours: "30" }]) {
      const data = { updated, days: DAYS, last_24_hours: 30, ...wrong };
      assert.equal(usageCounts(data, NOW), null, JSON.stringify(wrong));
    }
    for (const data of [null, {}]) assert.equal(usageCounts(data, NOW), null, JSON.stringify(data));
  });
  test("are read beside the module, at the site's root, and lead to what the server counts", () => {
    assert.ok(USAGE_URL.href.endsWith("/web/usage.json"));
    assert.ok(USAGE_PAGE.href.endsWith("/web/technical-details-how-factorial-works.html#h-servers"));
  });
});

describe("the words beside the line", () => {
  test("say the count of the last 24 hours", () => {
    assert.deepEqual(usageWords(30), { count: "30 users", when: "last 24 hours" });
    assert.deepEqual(usageWords(1), { count: "1 user", when: "last 24 hours" });
    assert.deepEqual(usageWords(1234), { count: "1,234 users", when: "last 24 hours" });
  });
});

describe("the line", () => {
  test("runs from the oldest day at the left to the last at the right, the most in a day at the top", () => {
    const points = sparkPoints([0, 5, 10], 72, 22);
    assert.deepEqual(points.map(([x]) => x), [3, 36, 69]);
    assert.deepEqual(points.map(([, y]) => y), [19, 11, 3]);
  });
  test("is flat at the foot for days of none, and a single day sits at the left", () => {
    assert.deepEqual(sparkPoints([0, 0], 72, 22), [[3, 19], [69, 19]]);
    assert.deepEqual(sparkPoints([7], 72, 22), [[3, 3]]);
  });
});
