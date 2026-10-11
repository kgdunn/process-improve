// web/python.js: the code each "Show the Python" panel writes, as text. That the code runs and
// reproduces the page is checked by tests/web/page/test_python_panels.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { analyseCode, compareCode, designCode, enteredCode, simulateCode } from "../../../web/python.js";

const FACTORS = [
  { name: "T", low: 150, high: 200, units: "degC" },
  { name: "P", low: 1, high: 3, units: "" },
];

describe("the design", () => {
  test("its factors, units only where given, and its options in the order the page holds them", () => {
    const code = designCode({ design_type: "ccd", factors: FACTORS, options: { random_state: 42, alpha: "rotatable" } });
    assert.match(code, /Factor\(name="T", low=150, high=200, units="degC"\),/);
    assert.match(code, /Factor\(name="P", low=1, high=3\),/);
    assert.match(code, /generate_design\(factors, design_type="ccd", random_state=42, alpha="rotatable"\)/);
  });

  test("Python's True, False and None, and no option left undefined", () => {
    const code = designCode({ design_type: "d_optimal", factors: FACTORS, options: { a: true, b: false, c: null, d: undefined } });
    assert.match(code, /a=True, b=False\)/);
  });
});

describe("the results", () => {
  test("a simulation names its process, noise and seed", () => {
    const code = simulateCode({ truth: [{ term: "T", coefficient: 4 }, { term: "I(T ** 2)", coefficient: -2.25 }], sigma: 1, seed: 3 });
    assert.match(code, /"T": 4,/);
    assert.match(code, /"I\(T \*\* 2\)": -2.25,/);
    assert.match(code, /sigma=1,/);
    assert.match(code, /random_state=3\)/);
  });

  test("a design with a categorical factor simulates at the coded runs, its labels coded -1 and +1", () => {
    const spec = { factors: [FACTORS[0], { name: "S", low: "A", high: "B", units: "" }], options: {} };
    const code = simulateCode({ truth: [{ term: "S", coefficient: 2 }], sigma: 1, seed: 3 }, spec);
    assert.match(code, /runs\["S"\] = runs\["S"\]\.map\(\{"A": -1\.0, "B": 1\.0\}\)/);
    assert.match(code, /process\.simulate\(runs, random_state=3\)/);
  });

  test("results typed in, None for a run still to do", () => {
    assert.match(enteredCode([1.5, null, -2]), /pd\.Series\(\[1\.5, None, -2\], name="y"\)/);
  });
});

describe("the analysis and the comparison", () => {
  test("the analysis names the factors and the model", () => {
    const code = analyseCode({ factors: FACTORS }, "quadratic");
    assert.match(code, /names = \["T", "P"\]/);
    assert.match(code, /runs = result\.design\[names\]\.copy\(\)/);
    assert.match(code, /analyze_experiment\(\n    runs, y, model="quadratic"/);
  });

  test("a blocked design's runs keep their Block, and a category is coded as the page codes it", () => {
    const spec = { factors: [FACTORS[0], { name: "S", low: "A", high: "B", units: "" }], options: { n_blocks: 2 } };
    const code = analyseCode(spec, "Block + T + S");
    assert.match(code, /runs = result\.design\[names \+ \["Block"\]\]\.copy\(\)/);
    assert.match(code, /runs\["S"\] = runs\["S"\]\.map/);
    assert.match(code, /model="Block \+ T \+ S"/);
  });

  test("a categorical factor's levels in the design", () => {
    const code = designCode({ design_type: "full_factorial", factors: [{ name: "S", low: "A", high: "B" }], options: {} });
    assert.match(code, /Factor\(name="S", type="categorical", levels=\["A", "B"\]\),/);
  });

  test("the comparison generates each design with its own options", () => {
    const designs = [
      { label: "Full factorial", design_type: "full_factorial", options: { n_center_points: 3 } },
      { label: "D-optimal", design_type: "d_optimal" },
    ];
    const code = compareCode(FACTORS, designs, "main_quadratic");
    assert.match(code, /"Full factorial": generate_design\(factors, design_type="full_factorial", n_center_points=3\),/);
    assert.match(code, /"D-optimal": generate_design\(factors, design_type="d_optimal"\),/);
    assert.match(code, /compare_designs\(designs, model="main_quadratic"\)/);
  });
});
