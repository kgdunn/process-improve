// web/python.js: the code each "Show the Python code" panel writes, as text. That the code runs and
// reproduces the page is checked by tests/web/page/test_python_panels.py.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { analyseCode, bestCode, compareCode, cumulative, designCode, notebook, resultsCode } from "../../../web/python.js";

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

  test("a Taguchi array at three levels gives each numeric factor its low, center and high values as its levels", () => {
    const factors = [...FACTORS, { name: "S", low: "A", high: "B", units: "" }];
    const code = designCode({ design_type: "taguchi", factors, options: { random_state: 42, n_levels: 3 } });
    assert.match(code, /factors = \[ {2}# each numeric factor at its low, center and high values\n/);
    assert.match(code, /Factor\(name="T", low=150, high=200, levels=\[150, 175, 200\], units="degC"\),/);
    assert.match(code, /Factor\(name="S", type="categorical", levels=\["A", "B"\]\),/);
    assert.match(code, /generate_design\(factors, design_type="taguchi", random_state=42\)/, "n_levels is the page's, not the library's");
    const two = designCode({ design_type: "taguchi", factors: FACTORS, options: { random_state: 42 } });
    assert.doesNotMatch(two, /levels=/);
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
    assert.doesNotMatch(partial, /evaluate_design/, "no VIFs unless the page shows them");
  });

  test("the VIFs the page shows, from the same runs and terms the analysis fitted", () => {
    const code = analyseCode("main_effects", true, null, "A + B + C");
    assert.match(code, /^from process_improve\.experiments import analyze_experiment, evaluate_design$/m);
    assert.match(code, /\nvif = evaluate_design\(X, model="A \+ B \+ C", metric="vif"\)\["vif"\]\nprint\(vif\)$/);
    const partial = analyseCode("Block + T + S", false, null, "Block + T + S");
    assert.match(partial, /evaluate_design\(X\[done\], model="Block \+ T \+ S", metric="vif"\)/);
  });

  test("the notebook's analysis cell is the analysis panel's, its VIFs with it", () => {
    const design = { spec: { design_type: "full_factorial", factors: FACTORS, options: {} }, columns: ["T", "P"], rows: [[150, 1]] };
    const campaign = { title: "A study", design, responses: [1], sources: ["entered"], simulation: null, url: "https://example.org/" };
    const cell = (book) => book.cells.at(-1).source.join("");
    assert.equal(cell(notebook({ ...campaign, model: "T + P", vifModel: "T + P" })), analyseCode("T + P", true, null, "T + P"));
    assert.equal(cell(notebook({ ...campaign, model: "T + P" })), analyseCode("T + P", true));
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

  test("the confirmation runs are left out of the runs the model is fitted to, and the code says so", () => {
    const sheet = {
      spec: { factors: FACTORS, response: "y", options: {} },
      columns: ["Std order", "Run order", "T", "P", "y", "Confirmation run"],
      rows: [
        [1, 1, 150, 1, null, null],
        [2, 2, 200, 3, null, null],
        [null, 3, 200, 1, null, "yes"],
      ],
    };
    const code = resultsCode(sheet, [10, 20, 30]);
    assert.match(code, /"T": \[150, 200\],/);
    assert.match(code, /"y": \[10, 20\],/);
    assert.match(code, /# The confirmation runs are left out: they test the model, so they do not enter its fit\./);
    assert.doesNotMatch(resultsCode({ ...sheet, rows: sheet.rows.slice(0, 2) }, [10, 20]), /confirmation/);
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
    assert.doesNotMatch(code, /three_levels/, "the factors at three levels only when a design takes them");
  });

  test("a Taguchi array at three levels is made from the factors at their low, center and high values", () => {
    const designs = [
      { label: "Full factorial", design_type: "full_factorial", options: { n_center_points: 3 } },
      { label: "Taguchi", design_type: "taguchi", options: { random_state: 42, n_levels: 3 } },
    ];
    const code = compareCode(FACTORS, designs, "main_quadratic");
    assert.match(code, /three_levels = \[ {2}# each numeric factor at its low, center and high values\n/);
    assert.match(code, /Factor\(name="P", low=1, high=3, levels=\[1, 2, 3\]\),/);
    assert.match(code, /"Full factorial": generate_design\(factors, design_type="full_factorial", n_center_points=3\),/);
    assert.match(code, /"Taguchi": generate_design\(three_levels, design_type="taguchi", random_state=42\),/);
  });
});

describe("the best settings", () => {
  const spec = { factors: [...FACTORS, { name: "S", low: "A", high: "B" }], response: "Yield", options: {} };
  const goal = { response: "Yield", goal: "maximize", low: 35.925, high: 87.975 };
  const best = { coded: { T: 1, P: -1, S: 1 }, search: { held: {}, goal } };

  test("the optimiser on step 3's model, with the goal's limits as the page set them", () => {
    const code = bestCode({ ...best, spec: undefined }, { ...spec, factors: FACTORS }, "quadratic");
    assert.match(code, /^import pandas as pd\nfrom process_improve\.experiments import analyze_experiment, optimize_responses\n/);
    assert.match(code, /goal = \{"response": "Yield", "goal": "maximize", "low": 35\.925, "high": 87\.975\}/);
    assert.match(code, /"coefficients": analysis\["coefficients"\],/);
    assert.match(code, /"factor_names": \["T", "P"\],/);
    assert.match(code, /factor_ranges=\{"T": \{"low": 150, "high": 200\}, "P": \{"low": 1, "high": 3\}\},/);
    assert.match(code, /print\(best\["predicted_responses"\]\["Yield"\]\)/);
    assert.match(code, /analyze_experiment\(X, y, model="quadratic", analysis_type="prediction", new_points=at\)/);
    assert.doesNotMatch(code, /math|held_at|DesignRegion/);
  });

  test("a categorical factor held at the level the page found best, as the page holds it", () => {
    const code = bestCode({ ...best, search: { held: { S: 1 }, goal } }, spec, "interactions", false);
    assert.match(code, /^import math\n/);
    assert.match(code, /# The page tried each level and found S at B best: the model is held there\./);
    assert.match(code, /def held_at\(coefficients, held\):/);
    assert.match(code, /coefficients = held_at\(analysis\["coefficients"\], \{"S": 1\}\)/);
    assert.match(code, /"coefficients": coefficients,/);
    assert.match(code, /at = pd\.DataFrame\(\[\{\*\*best\["optimal_coded"\], \*\*\{"S": 1\}\}\]\)/);
    assert.match(code, /analyze_experiment\(X\[done\], y\[done\]/, "the runs still to do left out, as in step 3");
  });

  test("the blocks held at the average block", () => {
    const code = bestCode({ ...best, search: { held: { Block1: 0 }, goal } }, { ...spec, factors: FACTORS }, "Block + T");
    assert.match(code, /# The blocks are held at the average block, as the equation is\./);
  });

  test("inside the limits on the settings", () => {
    const limited = { ...spec, factors: FACTORS, constraints: [{ expression: "T + 25 * P <= 215" }] };
    const code = bestCode(best, limited, "quadratic");
    assert.match(code, /from process_improve\.experiments import DesignRegion, analyze_experiment, optimize_responses/);
    assert.match(code, /region=DesignRegion\(factors, constraints\),/);
  });

  test("every factor in the model categorical: the prediction at the levels the page compared", () => {
    const code = bestCode({ coded: { S: 1 }, search: { held: { S: 1 }, goal: null } }, { ...spec, factors: [spec.factors[2]] }, "main_effects");
    assert.doesNotMatch(code, /optimize_responses/);
    assert.match(code, /at = pd\.DataFrame\(\[\{"S": 1\}\]\)/);
  });
});

describe("a mixture", () => {
  const spec = {
    design_type: "simplex_lattice",
    factors: [
      { name: "A", low: 0, high: 1, units: "" },
      { name: "B", low: 0.1, high: 0.9, units: "%" },
      { name: "C", low: 0, high: 1, units: "" },
    ],
    response: "y",
    options: { random_state: 42, model_type: "scheffe_quadratic" },
  };

  test("its components, each a proportion, and the library's mixture design for its model", () => {
    const code = designCode(spec);
    assert.match(code, /Factor\(name="A", type="mixture", low=0, high=1\),/);
    assert.match(code, /Factor\(name="B", type="mixture", low=0.1, high=0.9, units="%"\),/);
    assert.match(code, /generate_design\(factors, design_type="mixture", random_state=42, model_type="scheffe_quadratic"\)/);
  });

  test("its proportions are fitted as they are, not coded", () => {
    const design = { spec, columns: ["Std order", "Run order", "A", "B", "C", "y"], rows: [[1, 1, 0.5, 0.5, 0, null]] };
    const code = resultsCode(design, [31.5]);
    assert.match(code, /# The proportions, as the Scheffé model is fitted: a mixture's components are not coded\./);
    assert.match(code, / {8}"A": runs\["A"\],/);
    assert.doesNotMatch(code, /- 0.5/);
  });
});
