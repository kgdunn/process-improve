// web/usage.js: the footer's line of how many people used Factorial each day, from the usage.json the server writes.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import { USAGE_PAGE, USAGE_URL, sparkPoints, usageDays, usageWords } from "../../../web/usage.js";

describe("the counts", () => {
  test("are the file's days as the server writes them, oldest first", () => {
    const days = [["2026-11-24", 81], ["2026-11-25", 89]];
    assert.deepEqual(usageDays({ updated: "2026-11-25T10:17:00Z", days }), days);
  });
  test("are none for a file of no days, as a build carries, or one not as the server writes it", () => {
    const build = JSON.parse(readFileSync(new URL("../../../web/usage.json", import.meta.url), "utf8"));
    assert.equal(usageDays(build), null, "a copy of Factorial elsewhere draws nothing");
    for (const data of [null, {}, { days: "89" }, { days: [["2026-11-25"]] }, { days: [["25 Nov", 89]] }, { days: [["2026-11-25", -1]] }, { days: [["2026-11-25", 1.5]] }]) {
      assert.equal(usageDays(data), null, JSON.stringify(data));
    }
  });
  test("are read beside the module, at the site's root, and lead to what the server counts", () => {
    assert.ok(USAGE_URL.href.endsWith("/web/usage.json"));
    assert.ok(USAGE_PAGE.href.endsWith("/web/technical-details-how-factorial-works.html#h-servers"));
  });
});

describe("the words beside the line", () => {
  test("say the last day's count, as today", () => {
    assert.deepEqual(usageWords("2026-11-25", 89, "2026-11-25"), { count: "89 users", when: "today" });
    assert.deepEqual(usageWords("2026-11-25", 1, "2026-11-25"), { count: "1 user", when: "today" });
    assert.deepEqual(usageWords("2026-11-25", 1234, "2026-11-25"), { count: "1,234 users", when: "today" });
  });
  test("say yesterday in the hour after midnight, before the server writes the new day, across a month's end", () => {
    assert.deepEqual(usageWords("2026-11-30", 40, "2026-12-01"), { count: "40 users", when: "yesterday" });
    assert.deepEqual(usageWords("2026-12-31", 40, "2027-01-01"), { count: "40 users", when: "yesterday" });
  });
  test("are none for an older count, which is stale: the line is left out", () => {
    assert.equal(usageWords("2026-11-23", 89, "2026-11-25"), null);
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
