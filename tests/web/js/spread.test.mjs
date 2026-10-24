// Results read more than once at a setting: the sentences Learn leads the response table with (web/logic.js), from
// the spread bootstrap.py's analysis gives. The table is Schmidt and Launsby's 2^(3-1), read twice a setting.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { SN_WORDS, spreadSentences } from "../../../web/logic.js";

/** Schmidt and Launsby's response table, as the analysis gives it for a target. */
function schmidt({ kind = "nominal", adjustment = "A", left_out = [] } = {}) {
  const level = (level, mean, lnS, sn) => ({ level, n: 2, mean, ln_std_dev: lnS, ...(kind ? { sn } : {}) });
  const factors = [
    {
      name: "A",
      levels: [level(1, 25, 1.1513, 17.9588), level(2, 10, 1.1513, 10)],
      delta: { mean: 15, ln_std_dev: 0, sn: 7.9588 },
      rank: { mean: 1, ln_std_dev: 2, sn: 2 },
      best: kind ? { sn: 1 } : {},
    },
    {
      name: "B",
      levels: [level(1, 17.5, 1.1513, 13.9794), level(2, 17.5, 1.1513, 13.9794)],
      delta: { mean: 0, ln_std_dev: 0, sn: 0 },
      rank: { mean: 2, ln_std_dev: 2, sn: 3 },
      best: {},
    },
    {
      name: "C",
      levels: [level(1, 17.5, 0.3466, 20.9691), level(2, 17.5, 1.956, 6.9897)],
      delta: { mean: 0, ln_std_dev: 1.6094, sn: 13.9794 },
      rank: { mean: 2, ln_std_dev: 1, sn: 1 },
      best: kind ? { ln_std_dev: 1, sn: 1 } : { ln_std_dev: 1 },
    },
  ];
  return { kind, settings: [], left_out, factors, adjustment };
}

describe("what the readings repeated at each setting say", () => {
  test("the factor that moves the spread most, its ratio's best levels and the adjustment factor", () => {
    assert.deepEqual(spreadSentences(schmidt(), { response: "the width", units: "mm", factorUnits: { C: "s" } }), [
      "C moves the spread of the width most, its ln s 0.347 at 1 s and 1.96 at 2 s: the spread is smallest with C at 1 s.",
      "The signal-to-noise ratio (nominal is best) is highest with C at 1 s and A at 1, the factor that moves it most first.",
      "To move the mean to the target, A is the adjustment factor: it moves the mean most for its effect on the spread, " +
        "by 15 mm over its levels and ln s by 0.",
    ]);
  });

  test("each ratio is named by its kind", () => {
    for (const [kind, words] of Object.entries(SN_WORDS)) {
      const [, ratio] = spreadSentences(schmidt({ kind, adjustment: null }), { response: "y" });
      assert.match(ratio, new RegExp(`\\(${words}\\)`));
    }
  });

  test("to learn there is no ratio and no adjustment factor, only the spread", () => {
    const lines = spreadSentences(schmidt({ kind: null, adjustment: null }), { response: "y" });
    assert.equal(lines.length, 1);
    assert.match(lines[0], /^C moves the spread of y most/);
  });

  test("a percent sign follows its number", () => {
    const [, , adjusts] = spreadSentences(schmidt(), { response: "y", units: "%" });
    assert.match(adjusts, /by 15% over its levels/);
  });

  test("no factor moves the spread when every delta of ln s ties", () => {
    const tied = schmidt({ adjustment: null });
    for (const f of tied.factors) delete f.best.ln_std_dev;
    assert.ok(spreadSentences(tied, { response: "y" }).every((line) => !line.includes("moves the spread")));
  });

  test("the runs read once at their setting are named", () => {
    assert.equal(spreadSentences(schmidt({ left_out: [7] }), { response: "y" }).at(-1), "Run 7, read once at its setting, is left out.");
    assert.equal(
      spreadSentences(schmidt({ left_out: [3, 7] }), { response: "y" }).at(-1),
      "Runs 3 and 7, read once at their settings, are left out.",
    );
  });
});
