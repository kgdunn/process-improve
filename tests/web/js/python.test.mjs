// web/python.js: the code each "Show the Python code" panel writes, as text. That the code runs and
// reproduces the page is checked by tests/web/page/test_python_panels.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { analyseCode, compareCode, cumulative, designCode, resultsCode } from "../../../web/python.js";

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

  test("the limits on the settings, one to a line, which the design keeps to; a limit's note stays on the page", () => {
    const constraints = [{ expression: "T + 25 * P <= 250", type: "linear", note: "the product chars" }, { expression: "T * F <= 3600", type: "nonlinear" }];
    const code = designCode({ design_type: "d_optimal", factors: FACTORS, options: { random_state: 1 }, constraints });
    assert.match(code, /constraints = \[ {2}# the settings that can be run, in the factors' own units\n {4}"T \+ 25 \* P <= 250",\n {4}"T \* F <= 3600",\n\]/);
    assert.match(code, /generate_design\(factors, design_type="d_optimal", constraints=constraints, random_state=1\)/);
    assert.doesNotMatch(code, /product chars/);
    assert.doesNotMatch(designCode({ design_type: "ccd", factors: FACTORS, options: {} }), /constraints/);
  });

  test("Python's True, False and None, and no option left undefined", () => {
    const code = designCode({ design_type: "d_optimal", factors: FACTORS, options: { a: true, b: false, c: null, d: undefined } });
    assert.match(code, /a=True, b=False\)/);
  });

  test("a string is written as typed, even when it holds the words true, false or null", () => {
    const factors = [{ name: "S", low: "true", high: "null or false", units: "" }];
    const code = designCode({ design_type: "full_factorial", factors, options: { note: 'said "null"' } });
    assert.match(code, /levels=\["true", "null or false"\]/);
    assert.match(code, /note="said \\"null\\""\)/);
  });

  test("lists and dicts written item by item, as Python writes them", () => {
    const options = { a: [1, 2.5, null, true], b: { x: [false], y: "null", z: undefined }, c: [[]] };
    const code = designCode({ design_type: "d_optimal", factors: FACTORS, options });
    assert.match(code, /a=\[1, 2\.5, None, True\], b=\{"x": \[False\], "y": "null"\}, c=\[\[\]\]\)/);
  });
});

describe("the results", () => {
  const design = {
    spec: { factors: [FACTORS[0], { name: "S", low: "A", high: "B", units: "" }], response: "Yield", options: { n_blocks: 2 } },
    columns: ["Std order", "Run order", "T", "S", "Block", "Yield"],
    rows: [
      [2, 1, 200, "A", 1, null],
      [1, 2, 150, "B", 2, null],
    ],
  };

  test("the run sheet as the page lists it, its settings coded as the page codes them", () => {
    const code = resultsCode(design, [61.5, null]);
    assert.match(code, /"T": \[200, 150\],\n {8}"S": \["A", "B"\],\n {8}"Block": \[1, 2\],\n {8}"Yield": \[61\.5, None\],/);
    assert.match(code, /"T": \(runs\["T"\] - 175\) \/ 25,/);
    assert.match(code, /"S": runs\["S"\]\.map\(\{"A": -1\.0, "B": 1\.0\}\),/);
    assert.match(code, /"Block": runs\["Block"\],/);
    assert.match(code, /y = runs\["Yield"\]$/);
  });

  test("a simulation names its process, noise and seed, and rounds as the page does", () => {
    const simulation = { truth: [{ term: "T", coefficient: 4 }, { term: "I(T ** 2)", coefficient: -2.25 }], sigma: 0.5, seed: 3 };
    const code = resultsCode(design, [1, 2], simulation);
    assert.doesNotMatch(code, /"Yield":/, "the results come from the process");
    assert.match(code, /"I\(T \*\* 2\)": -2.25,/);
    assert.match(code, /sigma=0.5,/);
    assert.match(code, /y = process\.simulate\(X, random_state=3\)\.round\(3\)/);
  });

  test("the response is named in the reader's words and units beside y", () => {
    const described = { ...design, spec: { ...design.spec, response_info: { label: "Yield of the reactor", units: "%" } } };
    assert.match(resultsCode(described, [61.5, null]), /y = runs\["Yield"\]  # Yield of the reactor, in %$/);
    const simulated = resultsCode(described, [1, 2], { truth: [{ term: "T", coefficient: 4 }], sigma: 1, seed: 3 });
    assert.match(simulated, /# Yield of the reactor, in %, simulated from a known process\ny = process\.simulate/);
    const unitless = { ...design, spec: { ...design.spec, response_info: { label: "Hardness" } } };
    assert.match(resultsCode(unitless, [1, 2]), /y = runs\["Yield"\]  # Hardness$/);
  });

  test("a centre at zero, a negative one, and a half-range of one, written plainly", () => {
    const factors = [
      { name: "A", low: -1, high: 1 },
      { name: "B", low: -10, high: -6 },
    ];
    const plain = { spec: { factors, options: {} }, columns: ["A", "B"], rows: [[-1, -10]] };
    const code = resultsCode(plain, [1]);
    assert.match(code, /"A": runs\["A"\],/);
    assert.match(code, /"B": \(runs\["B"\] \+ 8\) \/ 2,/);
  });

  test("a long run sheet wraps its columns", () => {
    const rows = Array.from({ length: 40 }, (_, i) => [150 + i]);
    const long = { spec: { factors: [FACTORS[0]], options: {} }, columns: ["T"], rows };
    const lines = resultsCode(long, rows.map(() => 1)).split("\n");
    assert.ok(lines.every((line) => line.length <= 100));
  });
});

describe("the analysis and the comparison", () => {
  test("the analysis fits the model the page fitted, to the runs with a result", () => {
    assert.match(analyseCode("quadratic"), /analyze_experiment\(\n {4}X, y, model="quadratic"/);
    const partial = analyseCode("Block + T + S", false);
    assert.match(partial, /done = y\.notna\(\)/);
    assert.match(partial, /X\[done\], y\[done\], model="Block \+ T \+ S"/);
  });

  test("the steps as one script: their imports first, merged, then each step in turn", () => {
    const code = cumulative(
      "import pandas as pd\nfrom m import b\n\nx = 1",
      "from m import a, b\nimport numpy as np\n\ny = 2",
    );
    assert.equal(code, "import numpy as np\nimport pandas as pd\nfrom m import a, b\n\nx = 1\n\ny = 2");
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
