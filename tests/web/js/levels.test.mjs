// web/levels.js: the page's reading level, and what a level leaves out of a list.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { DEFAULT_LEVEL, FEATURES, LEVELS, LEVEL_KEY, designListed, levelOf, moreWords, readLevel } from "../../../web/levels.js";

describe("the reading level", () => {
  test("there are the concept map's three levels, under its key", () => {
    assert.deepEqual(LEVELS, ["Beginner", "Intermediate", "Adventurous"]);
    assert.equal(LEVEL_KEY, "factorial-map-level");
  });

  test("a stored choice is read, and anything else is a first visit", () => {
    assert.equal(readLevel("1"), 1);
    assert.equal(readLevel("3"), 3);
    for (const stored of [null, "", "0", "4", "2.5", "high"]) assert.equal(readLevel(stored), DEFAULT_LEVEL);
  });

  test("every part has a level of the three, and a term of the map or a reason", () => {
    for (const [feature, { level, term, because }] of Object.entries(FEATURES)) {
      assert.ok(level >= 1 && level <= LEVELS.length, feature);
      assert.ok(Boolean(term) !== Boolean(because), `${feature} names a term or says why it has none`);
    }
  });

  test("a part no level names is on every page", () => {
    assert.equal(levelOf("table:of-no-level"), 1);
    assert.equal(levelOf("design:d_optimal"), 3);
  });

  test("the list of designs offers those of the level and of the levels below it", () => {
    assert.ok(designListed("full_factorial", 1));
    assert.ok(!designListed("ccd", 1));
    assert.ok(designListed("ccd", 2) && designListed("ccd", 3));
    assert.ok(!designListed("d_optimal", 2));
  });

  test("the design the plan holds is listed at every level, and no other is for that", () => {
    assert.ok(designListed("d_optimal", 1, { chosen: "d_optimal" }));
    assert.ok(!designListed("i_optimal", 1, { chosen: "d_optimal" }));
  });

  test("runs as given are a design no level offers: a study that has them lists them", () => {
    for (const level of [1, 2, 3]) assert.ok(!designListed("given", level));
    assert.ok(designListed("given", 1, { also: ["given"] }));
    assert.ok(designListed("given", 2, { chosen: "given" }));
    assert.ok(!designListed("a design the page does not have", 3));
  });

  test("what a level leaves out is said where the list is", () => {
    assert.equal(moreWords([], "design"), "");
    assert.equal(moreWords([3], "chart"), "1 more chart at Adventurous");
    assert.equal(moreWords([3, 3, 3], "design"), "3 more designs at Adventurous");
    assert.equal(moreWords([2, 3, 2], "chart"), "2 more charts at Intermediate, 1 at Adventurous");
    assert.equal(moreWords([2, 3], "design"), "1 more designs at Intermediate, 1 at Adventurous");
  });
});
