// The decimal mark the page writes its numbers with (web/numbers.js), and what follows it across the page's modules.
import assert from "node:assert/strict";
import { afterEach, describe, test } from "node:test";

import { num, numPlain, withUnit } from "../../../web/best.js";
import { fmt, levelValue, pval } from "../../../web/logic.js";
import { DECIMAL_MARKS, decimalMark, marked, markedEquation, pointed, setDecimalMark } from "../../../web/numbers.js";

describe("the decimal mark", () => {
  afterEach(() => setDecimalMark("period"));

  test("is a period until a comma is chosen, and a period again for any other name", () => {
    assert.deepEqual(Object.keys(DECIMAL_MARKS), ["period", "comma"], "the period first: it is the default");
    assert.equal(decimalMark(), "period");
    assert.equal(marked("18.04"), "18.04");
    setDecimalMark("comma");
    assert.equal(decimalMark(), "comma");
    assert.equal(marked("18.04"), "18,04");
    assert.equal(marked("200"), "200", "a whole number has no mark");
    assert.equal(marked("< 0.0001"), "< 0,0001");
    for (const other of ["period", undefined, null, "dot", ","]) {
      setDecimalMark(other);
      assert.equal(decimalMark(), "period", String(other));
    }
  });

  test("is in every number the page writes, and never in a number a box of type number takes", () => {
    setDecimalMark("comma");
    assert.equal(fmt(1.5), "1,500");
    assert.equal(pval(0.01234), "0,0123");
    assert.equal(pval(0.00001), "< 0,0001");
    assert.equal(num(18.04), "18,04");
    assert.equal(withUnit(68.25, "%"), "68,25%");
    assert.equal(withUnit(18.04, "L/min"), "18,04 L/min");
    assert.equal(numPlain(18.04), "18.04");
  });

  test("is read in a number that is typed, while a label with a comma stays a label", () => {
    assert.equal(levelValue("1.5"), 1.5);
    assert.equal(levelValue("1,5"), "1,5", "with the period, 1,5 is a label, as it always was");
    setDecimalMark("comma");
    assert.equal(levelValue("1,5"), 1.5);
    assert.equal(levelValue(" -0,25 "), -0.25);
    assert.equal(levelValue("1.5"), 1.5, "a period is still read");
    assert.equal(levelValue("A,B"), "A,B");
    assert.equal(pointed("1,5"), "1.5");
    assert.equal(pointed("1,5,6"), "1,5,6", "not one number");
    assert.equal(pointed("old"), "old");
  });

  test("is in each number of an equation Python wrote, and not in the response's own name", () => {
    const equation = "Yield v1.2 = 61.95 + 3.95·Temperature - 0.005·Flow²";
    assert.equal(markedEquation(equation), equation);
    setDecimalMark("comma");
    assert.equal(markedEquation(equation), "Yield v1.2 = 61,95 + 3,95·Temperature - 0,005·Flow²");
    assert.equal(markedEquation("no equation"), "no equation");
  });
});
