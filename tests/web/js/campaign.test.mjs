// web/campaign.js: a campaign packed into a link and unpacked, checked as text anyone could write.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  AIMS,
  GOAL_SHAPES,
  BOOK_PARAM,
  CAMPAIGN_VERSION,
  FACTORS_HELD,
  FEW_FACTORS,
  LEVELS_FORMAT,
  MANY_FACTORS_FORMAT,
  ONE_FACTOR_FORMAT,
  SCENARIOS_HELD,
  WHAT_IF_FORMAT,
  bookName,
  campaignDone,
  campaignTitle,
  checkBookExample,
  checkCampaign,
  checkConstraints,
  designKey,
  indexedName,
  mergeResults,
  packCampaign,
  progress,
  sameCampaign,
  shareable,
  studyFormat,
  timeAgo,
  unpackCampaign,
} from "../../../web/campaign.js";

/** A campaign as the page holds one: a 2^2 factorial in two numeric factors and a supplier, partly run. */
function campaign() {
  const factors = [
    { name: "T", low: 150, high: 200, units: "degC" },
    { name: "Supplier", low: "A", high: "B", units: "" },
  ];
  const columns = ["Std order", "Run order", "T", "Supplier", "y"];
  const rows = [
    [3, 1, 150, "B", null],
    [1, 2, 150, "A", null],
    [4, 3, 200, "B", null],
    [2, 4, 200, "A", null],
  ];
  return {
    v: 1, // a study of two factors keeps the first format
    title: "Full factorial (2-level) in T and Supplier",
    design: {
      spec: {
        version: 1,
        design_type: "full_factorial",
        factors,
        response: "y",
        model: "interactions",
        options: { random_state: 42, n_center_points: 0 },
      },
      columns,
      rows,
      model: "interactions",
      n_runs: 4,
      facts: { center_runs: 0, notes: ["No main effect or two-factor interaction is fully aliased with another."] },
      outside_range: [],
    },
    responses: [12.5, 9.75, null, 14],
    sources: ["entered", "entered", null, "entered"],
    simulation: null,
    simulate: { scenario: "reactor", sigma: 1, seed: 3 },
    model: { terms: [["T"], ["Supplier"], ["T", "Supplier"]] },
  };
}

async function packed(data) {
  return packCampaign(data);
}

describe("a link carries a campaign", () => {
  test("what is packed comes back as it was, checked", async () => {
    const original = campaign();
    const back = await unpackCampaign(await packed(original));
    assert.deepEqual(back.design.rows, original.design.rows);
    assert.deepEqual(back.responses, original.responses);
    assert.deepEqual(back.model, original.model);
    assert.deepEqual(back.design.spec.factors, original.design.spec.factors);
    assert.equal(back.title, original.title);
  });

  test("the link is URL-safe text, and short for a small design", async () => {
    const text = await packed(campaign());
    assert.match(text, /^[A-Za-z0-9_-]+$/);
    assert.ok(text.length < 1000, `${text.length} characters`);
  });

  test("a link without its results carries only the design", async () => {
    const back = await unpackCampaign(await packed(shareable(campaign(), false)));
    assert.deepEqual(back.responses, [null, null, null, null]);
    assert.deepEqual(back.sources, [null, null, null, null]);
    assert.equal(back.simulation, null);
    assert.deepEqual(back.design.rows, campaign().design.rows);
  });

  test("a link with its results carries them, and no bookkeeping of the browser it came from", async () => {
    const shared = shareable({ ...campaign(), id: "local", created: 1, updated: 2 }, true);
    assert.deepEqual(Object.keys(shared).sort(), ["design", "model", "responses", "simulate", "simulation", "sources", "title", "v"]);
    assert.deepEqual((await unpackCampaign(await packed(shared))).responses, campaign().responses);
  });
});

describe("a link that is not one of the page's is refused, saying why", () => {
  const refused = async (text, why) => {
    await assert.rejects(unpackCampaign(text), (err) => {
      assert.match(err.message, /^This design could not be opened: /);
      assert.match(err.message, why);
      return true;
    });
  };

  test("text a link cannot hold", () => refused("abc$%^", /cut or changed/));
  test("a link cut short", async () => refused((await packed(campaign())).slice(0, 40), /cut or changed/));
  test("text that unpacks to something other than JSON", async () => {
    // A raw deflate of text that is not JSON: a link someone made by hand.
    const raw = new Blob(["{not json"]).stream().pipeThrough(new CompressionStream("deflate-raw"));
    const bytes = new Uint8Array(await new Response(raw).arrayBuffer());
    const text = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    await refused(text, /cut or changed/);
  });

  test("a link that unpacks to far more than any design", async () => {
    const huge = { ...campaign(), title: "x".repeat(3 * 2 ** 20) };
    await refused(await packed(huge), /unpacks to more than a design/);
  });

  test("a later format, or none", async () => {
    await refused(await packed({ ...campaign(), v: CAMPAIGN_VERSION + 1 }), new RegExp(`format ${CAMPAIGN_VERSION + 1}`));
    await refused(await packed({ ...campaign(), v: 0 }), /format 0/);
    await refused(await packed({ ...campaign(), v: "1" }), /format "1"/);
  });

  test("results that are not one number per run", async () => {
    await refused(await packed({ ...campaign(), responses: [1, 2, 3] }), /one number, or a blank, per run/);
    await refused(await packed({ ...campaign(), responses: ['"><img src=x onerror=alert(1)>', 2, 3, 4] }), /per run/);
  });

  test("a run that does not fill the run sheet's columns", async () => {
    const data = campaign();
    data.design.rows[0] = [3, 1, 150];
    await refused(await packed(data), /does not fill the run sheet's columns/);
  });

  test("a run sheet without the factors' columns", async () => {
    const data = campaign();
    data.design.columns = ["Std order", "Run order", "T", "Vendor", "y"];
    await refused(await packed(data), /no column "Supplier"/);
  });

  test("a model with a term the design lacks", async () => {
    await refused(await packed({ ...campaign(), model: { terms: [["T"], ["Pressure"]] } }), /names a factor the design lacks/);
  });

  test("a model that is not one of the page's", async () => {
    await refused(await packed({ ...campaign(), model: "y ~ T + __import__('os')" }), /model .* is unknown/);
  });

  test("a factor whose level is neither a number nor a label", async () => {
    const data = campaign();
    data.design.spec.factors[0].low = { valueOf: 1 };
    await refused(await packed(data), /low level is neither a number nor a label/);
  });
});

describe("a campaign read back keeps only what the page uses", () => {
  test("keys the page does not know are dropped", () => {
    const data = JSON.parse(JSON.stringify(campaign()));
    data.extra = "dropped";
    data.design.unexpected = true;
    const back = checkCampaign(data);
    assert.equal(back.extra, undefined);
    assert.equal(back.design.unexpected, undefined);
  });

  test("no key of a link reaches a prototype", () => {
    // JSON.parse makes "__proto__" an own key, as a hostile link would hold it: here, the first key.
    const top = JSON.parse(`{"__proto__":{"polluted":"yes"},${JSON.stringify(campaign()).slice(1)}`);
    assert.equal(checkCampaign(top).polluted, undefined);
    const inOptions = JSON.parse(JSON.stringify(campaign()).replace('"options":{', '"options":{"__proto__":{"polluted":"yes"},'));
    assert.throws(() => checkCampaign(inOptions), /setting __proto__ is not a value/);
    assert.equal({}.polluted, undefined);
  });

  test("a study from the catalog keeps its source; a study from elsewhere has none", () => {
    const source = { cite: "Montgomery, D. C. (2013)", where: "pp. 379-380", url: "https://example.org/doe", catalog: "montgomery-problem-8-29", printed: 12 };
    assert.deepEqual(checkCampaign({ ...campaign(), source }).source, source);
    assert.deepEqual(checkCampaign({ ...campaign(), source: { cite: "A book" } }).source, { cite: "A book", where: "", url: "", catalog: "", printed: 0 });
    assert.equal("source" in checkCampaign(campaign()), false);
  });

  test("a source the page would link to or fetch from elsewhere is refused", () => {
    const refusedSource = (source, why) => assert.throws(() => checkCampaign({ ...campaign(), source: { cite: "A book", ...source } }), why);
    refusedSource({ url: "javascript:alert(1)" }, /address is not a web address/);
    refusedSource({ catalog: "../index" }, /id in the catalog is not one the catalog gives/);
    refusedSource({ printed: -1 }, /count of its printed numbers is not a count/);
    refusedSource({ cite: 7 }, /citation is not text/);
  });

  test("results without their sources are taken as entered", () => {
    const data = campaign();
    delete data.sources;
    assert.deepEqual(checkCampaign(data).sources, ["entered", "entered", null, "entered"]);
  });

  test("the reader's description of the response is kept, each part checked", () => {
    const data = campaign();
    const info = { label: "Purity", units: "%", goal: "target", target: 99.5, typical: 97, noise: 0.4, smallest_change: 1.2 };
    data.design.spec.response_info = { ...info, colour: "red" };
    data.design.spec.aim = "main_quadratic";
    const back = checkCampaign(data).design.spec;
    assert.deepEqual(back.response_info, info, "a part the page does not know is dropped");
    assert.equal(back.aim, "main_quadratic");
  });

  test("a design saved before the page asked for a description opens as it was", () => {
    const back = checkCampaign(campaign()).design.spec;
    assert.equal(back.response_info, undefined);
    assert.equal(back.aim, undefined);
  });

  test("a goal or aim this page does not know is dropped for a later page to add", () => {
    const data = campaign();
    data.design.spec.response_info = { label: "Purity", goal: "soar" };
    data.design.spec.aim = "cubic";
    const back = checkCampaign(data).design.spec;
    assert.deepEqual(back.response_info, { label: "Purity" });
    assert.equal(back.aim, undefined);
  });

  test("a description of the wrong kind is refused", () => {
    for (const [info, why] of [
      ["Purity", /description is not one/],
      [{ label: 3 }, /response's name is not text/],
      [{ units: "x".repeat(33) }, /response's units is not text/],
      [{ noise: 0 }, /response's noise is not a number/],
      [{ smallest_change: -1 }, /smallest change is not a number/],
      [{ typical: "62" }, /typical is not a number/],
    ]) {
      const data = campaign();
      data.design.spec.response_info = info;
      assert.throws(() => checkCampaign(data), why, JSON.stringify(info));
    }
  });

  test("the aims are the comparison's models", () => {
    assert.deepEqual(AIMS, ["main_effects", "interactions", "main_quadratic", "quadratic"]);
  });

  test("a simulation keeps the true process the analysis compares with", () => {
    const simulation = {
      scenario: "reactor",
      label: "Reactor yield",
      sigma: 1,
      seed: 3,
      equation: "y = 50 + 4 T",
      inactive: ["Supplier"],
      truth: [{ term: "T", coefficient: 4 }],
      responses: [1, 2, 3, 4],
    };
    const back = checkCampaign({ ...campaign(), simulation });
    assert.deepEqual(back.simulation.truth, simulation.truth);
    assert.equal(back.simulation.responses, undefined, "the results are kept once, as the campaign's");
  });
});

describe("the limits on a study's settings", () => {
  const limits = [
    { expression: "T + 25 * P <= 250", type: "linear", note: "the product chars" },
    { expression: "T * F <= 3600", type: "nonlinear" },
  ];

  test("a study's limits are kept with its design, each with its type and note", () => {
    const data = campaign();
    data.design.spec.constraints = limits;
    assert.deepEqual(checkCampaign(data).design.spec.constraints, limits);
  });

  test("a study without limits, or with none, keeps no list of them", () => {
    assert.equal(checkCampaign(campaign()).design.spec.constraints, undefined);
    const data = campaign();
    data.design.spec.constraints = [];
    assert.equal(checkCampaign(data).design.spec.constraints, undefined);
  });

  test("only the shape is checked here: Python reads each limit when it is used", () => {
    assert.deepEqual(checkConstraints([{ expression: "Z + import <= 3", type: "curved", colour: "red", note: "" }]), [
      { expression: "Z + import <= 3" },
    ]);
  });

  test("a list that is not one of limits is refused, saying which limit is wrong", () => {
    for (const [value, why] of [
      ["T <= 190", /limits on the settings is not a list/],
      [Array.from({ length: 9 }, () => ({ expression: "T <= 190" })), /limits on the settings is too long/],
      [["T <= 190"], /limit 1 is not a limit/],
      [[{ expression: 190 }], /limit 1 is not text/],
      [[{ expression: "T".repeat(501) }], /limit 1 is not text/],
      [[{ expression: "T <= 190\nP <= 2" }], /limit 1 is not one line of text/],
      [[{ expression: "T <= 190" }, { expression: "P <= 2", note: "x".repeat(81) }], /limit 2's note is not text/],
      [[{ expression: "T <= 190", note: "a\tb" }], /limit 1's note is not one line of text/],
    ]) {
      assert.throws(() => checkConstraints(value), why, JSON.stringify(value).slice(0, 60));
    }
  });
});

describe("the words for a saved design", () => {
  test("its title names the design and its factors", () => {
    assert.equal(campaignTitle("Box-Behnken", ["T", "P", "F"]), "Box-Behnken in T, P and F");
    assert.equal(campaignTitle("Full factorial (2-level)", ["T"]), "Full factorial (2-level) in T");
  });

  test("after the response it is about, when the reader described one", () => {
    assert.equal(campaignTitle("Box-Behnken", ["T", "P", "F"], "Yield (%), maximise"), "Yield (%), maximise: Box-Behnken in T, P and F");
  });

  test("when it was saved", () => {
    const now = new Date(2026, 9, 12, 9, 30).getTime();
    const minutes = (m) => now - m * 60_000;
    assert.equal(timeAgo(now - 20_000, now), "just now");
    assert.equal(timeAgo(minutes(1), now), "1 minute ago");
    assert.equal(timeAgo(minutes(45), now), "45 minutes ago");
    assert.equal(timeAgo(minutes(180), now), "3 hours ago");
    assert.equal(timeAgo(new Date(2026, 9, 11, 8).getTime(), now), "yesterday");
    assert.equal(timeAgo(new Date(2026, 9, 8, 8).getTime(), now), "4 days ago");
    assert.equal(timeAgo(new Date(2026, 8, 3, 8).getTime(), now), "3 Sep 2026");
  });

  test("how far it has got", () => {
    assert.equal(progress([null, null]), "2 runs, no results yet");
    assert.equal(progress([1, null, 3]), "3 runs, 2 results in");
    assert.equal(progress([1, 2]), "2 runs, all results in");
  });
});

describe("results that come back by link", () => {
  test("the same design is recognised whatever its results, and another is not", () => {
    const here = campaign();
    const back = { ...campaign(), responses: [1, 2, 3, 4] };
    assert.equal(designKey(here.design), designKey(back.design));
    const other = campaign();
    other.design.rows[0] = [3, 1, 175, "B", null];
    assert.notEqual(designKey(here.design), designKey(other.design));
  });

  test("a description of the response renamed since does not make it another design", () => {
    const here = campaign();
    const renamed = campaign();
    renamed.design.spec.response_info = { label: "Yield of the reactor", units: "%" };
    renamed.design.spec.aim = "interactions";
    assert.equal(designKey(here.design), designKey(renamed.design));
  });

  test("a factor renamed since does not make it another design, and a changed level does", () => {
    const here = campaign();
    const renamed = campaign();
    renamed.design.spec.factors[0] = { name: "Temperature", low: 150, high: 200, units: "K" };
    renamed.design.columns = ["Std order", "Run order", "Temperature", "Supplier", "y"];
    assert.equal(designKey(here.design), designKey(renamed.design));
    const moved = campaign();
    moved.design.spec.factors[0] = { ...moved.design.spec.factors[0], high: 210 };
    assert.notEqual(designKey(here.design), designKey(moved.design));
  });

  test("a run without a result here takes the link's, and one with a result keeps it", () => {
    const here = { responses: [12.5, null, null, 14], sources: ["entered", null, null, "entered"] };
    const incoming = { responses: [12.5, 9.5, null, 15], sources: ["entered", "entered", null, "entered"] };
    assert.deepEqual(mergeResults(here, incoming), {
      responses: [12.5, 9.5, null, 14],
      sources: ["entered", "entered", null, "entered"],
      added: 1,
      kept: 1,
    });
  });
});

describe("one of the book's examples, named in a link", () => {
  /** An example as the build writes book/<name>.json. */
  const example = () => ({
    name: "two-factor-supplier",
    title: "Temperature and supplier",
    book: "https://learnche.org/pid/design-analysis-experiments",
    note: "",
    campaign: campaign(),
  });

  const refusedBy = (fn, why) => {
    assert.throws(fn, (err) => {
      assert.match(err.message, /^This design could not be opened: /);
      assert.match(err.message, why);
      return true;
    });
  };

  test("the link names it in its query, ?book=, in words a person can read", () => {
    assert.equal(BOOK_PARAM, "book");
    assert.equal(bookName("pollutant"), "pollutant");
    assert.equal(bookName("two-level-2x2-conversion"), "two-level-2x2-conversion");
  });

  test("a name that is not one the book could print is refused before anything is fetched", () => {
    for (const text of ["", "Pollutant", "../demo", "a--b", "-a", "a b", "a/b", "a%2Fb", "x".repeat(65)]) {
      refusedBy(() => bookName(text), /not the name of an example in the book/);
    }
  });

  test("only a name the index lists is fetched, as the index gives it", () => {
    const index = { examples: [{ name: "pollutant", title: "Pollutant" }, { name: "web-sales" }] };
    assert.equal(indexedName(index, "web-sales"), "web-sales");
    refusedBy(() => indexedName(index, "no-such-example"), /the book has no example named "no-such-example"/);
    refusedBy(() => indexedName({}, "pollutant"), /no example named/);
    refusedBy(() => indexedName({ examples: [{ name: "../demo" }] }, "../demo"), /no example named/);
  });

  test("the example's campaign is checked as a link's is, and its page in the book kept", () => {
    const checked = checkBookExample(example(), "two-factor-supplier");
    assert.equal(checked.title, "Temperature and supplier");
    assert.equal(checked.book, "https://learnche.org/pid/design-analysis-experiments");
    assert.deepEqual(checked.campaign, checkCampaign(campaign()));
  });

  test("a file that is not the example the link names is refused", () => {
    refusedBy(() => checkBookExample(example(), "another"), /could not be read/);
    refusedBy(() => checkBookExample([], "two-factor-supplier"), /could not be read/);
    refusedBy(() => checkBookExample({ ...example(), book: "javascript:alert(1)" }, "two-factor-supplier"), /not a web address/);
    const later = { ...campaign(), v: CAMPAIGN_VERSION + 1 };
    refusedBy(() => checkBookExample({ ...example(), campaign: later }, "two-factor-supplier"), /could not be opened: it was saved in format/);
  });

  test("a copy saved here is the book's only while its results and model are as the book has them", () => {
    const book = checkCampaign(campaign());
    assert.ok(sameCampaign(checkCampaign(campaign()), book));
    assert.ok(!sameCampaign({ ...book, responses: [12.5, 9.75, 1, 14] }, book), "a result typed in");
    assert.ok(!sameCampaign({ ...book, model: "main_effects" }, book), "another model chosen");
    assert.ok(!sameCampaign({ ...book, title: "Renamed" }, book));
  });
});

describe("a campaign of several responses", () => {
  const COST = { name: "Cost", label: "Cost", units: "$/kg", goal: "minimize" };
  const PURITY = { name: "Purity", label: "Purity", units: "%", goal: "target", target: 99 };

  /** The 2^2 factorial measuring a yield, a cost and a purity: the first response is the study's own, y. */
  function three() {
    const c = campaign();
    c.design.spec.response_info = { label: "Yield", units: "%", goal: "maximize" };
    c.design.spec.responses = [{ name: "y", label: "Yield", units: "%", goal: "maximize" }, COST, PURITY];
    c.design.columns = [...c.design.columns, "Cost", "Purity"];
    c.design.rows = c.design.rows.map((row) => [...row, null, null]);
    c.more_responses = { Cost: [9, null, 8.5, 7], Purity: [98.9, 99.1, null, null] };
    c.more_sources = { Cost: ["entered", null, "entered", "entered"], Purity: ["entered", "entered", null, null] };
    c.more_models = { Cost: "interactions", Purity: { terms: [["T"]] } };
    return c;
  }

  test("a link carries every response's results and model, checked", async () => {
    const back = await unpackCampaign(await packCampaign(three()));
    assert.deepEqual(back.design.spec.responses, three().design.spec.responses);
    assert.deepEqual(back.responses, campaign().responses, "the first response's results, where they always were");
    assert.deepEqual(back.more_responses, three().more_responses);
    assert.deepEqual(back.more_sources, three().more_sources);
    assert.deepEqual(back.more_models, three().more_models);
  });

  test("a campaign of one response reads back as before, with nothing about other responses", () => {
    const back = checkCampaign(campaign());
    assert.deepEqual(Object.keys(back), ["v", "title", "design", "responses", "sources", "simulation", "simulate", "model"]);
    assert.equal(back.design.spec.responses, undefined);
  });

  test("a later response without results yet has none, and is analysed as designed", () => {
    const c = three();
    delete c.more_responses;
    delete c.more_sources;
    delete c.more_models;
    const back = checkCampaign(c);
    assert.deepEqual(back.more_responses, { Cost: [null, null, null, null], Purity: [null, null, null, null] });
    assert.deepEqual(back.more_models, { Cost: "", Purity: "" });
  });

  test("the list describes the first response, as response_info does", () => {
    const c = three();
    c.design.spec.response_info = { label: "Something else" };
    assert.deepEqual(checkCampaign(c).design.spec.response_info, { label: "Yield", units: "%", goal: "maximize" });
  });

  test("a link without its results keeps each response's model, and no result", async () => {
    const shared = shareable(three(), false);
    assert.equal(shared.more_responses, undefined);
    assert.deepEqual(shared.more_models, three().more_models);
    const back = await unpackCampaign(await packCampaign(shared));
    assert.deepEqual(back.more_responses.Cost, [null, null, null, null]);
  });

  test("a campaign whose responses do not hold together is refused, saying why", () => {
    const refused = (change, why) => {
      const c = three();
      change(c);
      assert.throws(() => checkCampaign(c), (err) => err.message.startsWith("This design could not be opened: ") && why.test(err.message));
    };
    refused((c) => (c.design.spec.responses = [COST, PURITY]), /its responses do not start with its response/);
    refused((c) => (c.design.spec.responses = [c.design.spec.responses[0], COST, COST]), /two of its responses share a name/);
    refused((c) => (c.design.spec.responses = [1, 2]), /response 1 is not a response/);
    refused((c) => (c.design.spec.responses = Array(6).fill(COST)), /the responses is too long/);
    refused((c) => (c.design.columns = c.design.columns.filter((name) => name !== "Purity")), /a run does not fill|no column "Purity"/);
    refused((c) => (c.more_responses = { Colour: [1, 2, 3, 4] }), /its results name a response it does not measure/);
    refused((c) => (c.more_responses.Cost = [1, 2]), /its results are not one number, or a blank, per run/);
    refused((c) => (c.more_sources.Cost = ["typed", null, null, null]), /sources are not one per run/);
    refused((c) => (c.simulation = { truth: [], sigma: 1, seed: 1, more: { Colour: { truth: [], sigma: 1, seed: 2 } } }), /simulation names a response/);
  });

  test("each response's simulated process travels with its noise and seed", () => {
    const c = three();
    const process = (seed) => ({ truth: [{ term: "Intercept", coefficient: 50 }], sigma: 1, seed, equation: "y = 50", inactive: ["T"] });
    c.simulation = { scenario: "reactor", label: "Reactor yield", ...process(1), more: { Cost: process(1001) } };
    const back = checkCampaign(c);
    assert.equal(back.simulation.more.Cost.seed, 1001);
    assert.deepEqual(back.simulation.more.Cost.inactive, ["T"]);
  });

  test("a run is done when every response has its result", () => {
    assert.deepEqual(campaignDone(checkCampaign(three())), [true, false, false, false]);
    assert.deepEqual(campaignDone(checkCampaign(campaign())), [true, true, false, true], "one response: its results");
    assert.equal(progress(campaignDone(checkCampaign(three())).map((done) => (done ? true : null))), "4 runs, 1 results in");
  });

  test("the same design is recognised whatever its results, and a study of one response keys as before", () => {
    const here = three();
    const back = three();
    back.more_responses = { Cost: [1, 2, 3, 4], Purity: [1, 2, 3, 4] };
    assert.equal(designKey(here.design), designKey(back.design));
    assert.notEqual(designKey(here.design), designKey(campaign().design), "another response is another study");
    assert.equal(
      designKey(campaign().design),
      JSON.stringify([
        "full_factorial",
        [
          [150, 200],
          ["A", "B"],
        ],
        "y",
        ["Std order", "Run order", 0, 1, "y"],
        campaign().design.rows.map((row) => row.slice(0, 4)),
      ]),
    );
  });

  test("results that come back by link fill each response's runs without a result here", () => {
    const here = checkCampaign(three());
    const incoming = checkCampaign(three());
    incoming.responses = [12.5, 9.75, 11, 14];
    incoming.sources = ["entered", "entered", "entered", "entered"];
    incoming.more_responses = { Cost: [9, 7.5, 8.5, 6], Purity: [98.9, 99.1, 99.2, null] };
    incoming.more_sources = { Cost: ["entered", "entered", "entered", "entered"], Purity: ["entered", "entered", "entered", null] };
    const merged = mergeResults(here, incoming);
    assert.deepEqual(merged.responses, [12.5, 9.75, 11, 14]);
    assert.deepEqual(merged.more_responses, { Cost: [9, 7.5, 8.5, 7], Purity: [98.9, 99.1, 99.2, null] });
    assert.equal(merged.added, 3, "one yield, one cost, one purity");
    assert.equal(merged.kept, 1, "the last run's cost here differs from the link's, and stays");
  });

  test("the book's example is the same campaign only with the same results for every response", () => {
    const a = checkCampaign(three());
    const b = checkCampaign(three());
    assert.ok(sameCampaign(a, b));
    b.more_responses.Cost[1] = 1;
    assert.ok(!sameCampaign(a, b));
  });

  test("what the reader set of each goal in the balance travels with the campaign, checked", async () => {
    const c = three();
    c.goals = { y: { low: 60, importance: 5 }, Cost: { high: 8.75, shape: "more" } };
    const back = await unpackCampaign(await packCampaign(c));
    assert.deepEqual(back.goals, c.goals);
    assert.deepEqual(shareable(c, false).goals, c.goals, "a link without results keeps them too");
    assert.equal(checkCampaign(three()).goals, undefined, "none set: none kept");
    assert.deepEqual(GOAL_SHAPES, ["less", "even", "more", "much_more"]);
    const refused = (goals, why) => {
      const d = three();
      d.goals = goals;
      assert.throws(() => checkCampaign(d), (err) => why.test(err.message));
    };
    refused({ Colour: { low: 1 } }, /its goals name a response it does not measure/);
    refused({ Cost: 3 }, /the goal of Cost is not one/);
    refused({ Cost: { high: "8" } }, /the acceptable range of Cost is not a number/);
    refused({ Cost: { importance: 6 } }, /the importance of Cost is not 1 to 5/);
    refused({ Cost: { shape: "steep" } }, /the shape of Cost's desirability is unknown/);
  });
});

describe("each study is saved in the lowest format that holds it", () => {
  /** The fixture's study with one factor: the supplier alone, at its two levels. */
  function oneFactor() {
    const data = campaign();
    const spec = { ...data.design.spec, factors: data.design.spec.factors.slice(1), model: "main_effects" };
    const columns = ["Std order", "Run order", "Supplier", "y"];
    const rows = data.design.rows.map(([std, run, , supplier, y]) => [std, run, supplier, y]);
    return { ...data, v: ONE_FACTOR_FORMAT, model: "", design: { ...data.design, spec, columns, rows, model: "main_effects" } };
  }

  /** The fixture's study with the supplier one of three: A, B or C. */
  function threeSuppliers() {
    const data = campaign();
    const supplier = { name: "Supplier", type: "categorical", levels: ["A", "B", "C"], units: "" };
    const spec = { ...data.design.spec, factors: [data.design.spec.factors[0], supplier] };
    return { ...data, v: LEVELS_FORMAT, design: { ...data.design, spec } };
  }

  test("a study of one factor is format 2, which a page of format 1 refuses in a sentence", () => {
    assert.equal(ONE_FACTOR_FORMAT, 2);
    assert.equal(studyFormat(oneFactor().design.spec), ONE_FACTOR_FORMAT);
    assert.equal(checkCampaign(oneFactor()).v, ONE_FACTOR_FORMAT);
  });

  test("a study of more factors keeps format 1, so a page of format 1 still opens it", () => {
    assert.equal(studyFormat(campaign().design.spec), 1);
    assert.equal(checkCampaign(campaign()).v, 1);
    assert.equal(checkCampaign({ ...campaign(), v: CAMPAIGN_VERSION }).v, 1, "read at any format the page knows");
  });

  test("a study of one factor saved before this format still opens, and is saved in its own", () => {
    assert.equal(checkCampaign({ ...oneFactor(), v: 1 }).v, ONE_FACTOR_FORMAT);
  });

  test("a factor that lists its levels is format 3, kept with its type and levels", () => {
    const checked = checkCampaign(threeSuppliers());
    assert.equal(checked.v, LEVELS_FORMAT);
    assert.deepEqual(checked.design.spec.factors[1], { name: "Supplier", type: "categorical", levels: ["A", "B", "C"], units: "" });
    const lots = { name: "Supplier", type: "categorical", levels: ["1", "2"], units: "" };
    const data = threeSuppliers();
    data.design.spec.factors[1] = lots;
    const two = checkCampaign(data);
    assert.deepEqual(two.design.spec.factors[1], { ...lots, low: "1", high: "2" }, "two levels are its low and high too");
    assert.equal(two.v, LEVELS_FORMAT, "labels that read as numbers need the format that keeps them labels");
  });

  test("a factor's levels are two to eight labels, each once, or the study is refused", () => {
    const withLevels = (levels, type = "categorical") => {
      const data = threeSuppliers();
      const factors = [data.design.spec.factors[0], { ...data.design.spec.factors[1], type, levels }];
      return { ...data, design: { ...data.design, spec: { ...data.design.spec, factors } } };
    };
    assert.throws(() => checkCampaign(withLevels(["A"])), /factor 2's levels are not two or more labels, each once/);
    assert.throws(() => checkCampaign(withLevels(["A", "B", "A"])), /each once/);
    assert.throws(() => checkCampaign(withLevels("ABC")), /factor 2's levels is not a list/);
    assert.throws(() => checkCampaign(withLevels([..."ABCDEFGHI"])), /factor 2's levels is too long/);
    assert.throws(() => checkCampaign(withLevels(["A", 2])), /a level of factor 2 is not text/);
    assert.throws(() => checkCampaign(withLevels(["A", "B"], "ordinal")), /factor 2's type is unknown/);
  });

  test("two studies differ when their levels do, though the run sheets read alike", () => {
    const other = threeSuppliers();
    other.design.spec.factors[1] = { ...other.design.spec.factors[1], levels: ["A", "B", "D"] };
    assert.notEqual(designKey(threeSuppliers().design), designKey(other.design));
  });
});

/**
 * A screening study of k numeric factors X1 to Xk in eight runs, with every response a study can measure, in two
 * blocks, a confirmation run, and results simulated from a process in the first three factors: the run sheet holds
 * every kind of column it can.
 */
function screening(k) {
  const names = Array.from({ length: k }, (_, i) => `X${i + 1}`);
  const responses = ["y", "y2", "y3", "y4", "y5"];
  const columns = ["Std order", "Run order", "Block", ...names, ...responses, "Confirmation run"];
  const rows = Array.from({ length: 8 }, (_, run) => [
    run + 1,
    run + 1,
    1 + (run % 2),
    ...names.map((_, j) => ((run >> j % 3) & 1 ? 1 : -1) * (j % 2 ? -1 : 1)),
    ...responses.map(() => null),
    null,
  ]);
  const simulation = {
    scenario: "main_effects",
    label: "Main effects only",
    sigma: 1,
    seed: 3,
    equation: "y = 50 + 3·X1 - 2·X2 + 1.5·X3",
    inactive: names.slice(3),
    truth: [
      { term: "Intercept", coefficient: 50 },
      { term: "X1", coefficient: 3 },
      { term: "X2", coefficient: -2 },
      { term: "X3", coefficient: 1.5 },
    ],
  };
  return {
    v: MANY_FACTORS_FORMAT,
    title: `Plackett-Burman screening in ${k} factors`,
    design: {
      spec: {
        version: 1,
        design_type: "plackett_burman",
        factors: names.map((name) => ({ name, low: 0, high: 10, units: "u" })),
        response: "y",
        responses: responses.map((name) => ({ name })),
        model: "main_effects",
        options: { random_state: 7, n_blocks: 2 },
      },
      columns,
      rows,
      model: "main_effects",
      n_runs: 8,
      facts: { center_runs: 0, notes: [] },
      outside_range: [...names],
    },
    responses: [50, 52, 48, 51, 49, 53, 47, 50],
    sources: Array(8).fill("simulated"),
    simulation,
    simulate: { scenario: "main_effects", sigma: 1, seed: 3 },
    model: { terms: [...names.map((n) => [n]), ["X1", "X2"]] },
  };
}

describe("a study of more factors than an earlier page reads", () => {
  for (const k of [13, 24, 40]) {
    test(`a screening study of ${k} factors opens whole, in the format that holds it`, async () => {
      const checked = checkCampaign(screening(k));
      assert.equal(checked.v, MANY_FACTORS_FORMAT);
      assert.equal(checked.design.spec.factors.length, k);
      assert.deepEqual(checked.simulation.inactive, screening(k).simulation.inactive, "every factor the process leaves out");
      assert.deepEqual(checked.design.outside_range, screening(k).design.outside_range);
      assert.deepEqual(checked.model, screening(k).model, "the main effects, and the interaction the reader named");
      const back = await unpackCampaign(await packCampaign(shareable(checked, true)));
      assert.deepEqual(back, checked, "a link carries it as it is");
    });
  }

  test("format 4 is later than any an earlier page reads, so that page refuses the study by its format", () => {
    assert.equal(MANY_FACTORS_FORMAT, LEVELS_FORMAT + 1);
    assert.ok(CAMPAIGN_VERSION >= MANY_FACTORS_FORMAT, "this page reads it");
    assert.equal(FEW_FACTORS, 12, "the factor list an earlier page holds");
    for (const k of [13, 24, 40]) assert.equal(studyFormat(screening(k).design.spec), MANY_FACTORS_FORMAT);
    const levels = screening(13).design.spec;
    levels.factors[0] = { name: "X1", type: "categorical", levels: ["A", "B", "C"], units: "" };
    assert.equal(studyFormat(levels), MANY_FACTORS_FORMAT, "with a factor that lists its levels too");
  });

  test("a study of 12 factors keeps the format it had, so an earlier page still opens it", () => {
    const data = screening(13);
    const spec = { ...data.design.spec, factors: data.design.spec.factors.slice(0, FEW_FACTORS) };
    assert.equal(studyFormat(spec), 1);
  });

  test("a study of more factors than a study holds is refused", () => {
    assert.equal(FACTORS_HELD, 40);
    const data = screening(40);
    data.design.spec.factors.push({ name: "X41", low: 0, high: 10, units: "u" });
    assert.throws(() => checkCampaign(data), /the factor list is too long/);
  });
});

describe("settings kept from the what-if panel", () => {
  /** The study, with two sets of settings kept under their names. */
  const withKept = () => ({
    ...campaign(),
    what_if: [
      { name: "Cooler, more pressure", settings: { T: 175, Supplier: "B" } },
      { name: "Hot", settings: { T: 200, Supplier: "A" } },
    ],
  });

  test("are kept with the study, in the format that holds them, and only then", () => {
    const checked = checkCampaign(withKept());
    assert.deepEqual(checked.what_if, withKept().what_if);
    assert.equal(checked.v, WHAT_IF_FORMAT);
    assert.equal(WHAT_IF_FORMAT, MANY_FACTORS_FORMAT + 1, "later than any an earlier page reads");
    assert.equal(CAMPAIGN_VERSION, WHAT_IF_FORMAT);
    assert.equal(studyFormat(campaign().design.spec, withKept().what_if), WHAT_IF_FORMAT);
    assert.equal(checkCampaign({ ...campaign(), what_if: [] }).v, 1, "none kept: the format the study had");
    assert.equal(checkCampaign({ ...campaign(), what_if: [] }).what_if, undefined);
    assert.equal(checkCampaign({ ...campaign(), what_if: null }).what_if, undefined);
  });

  test("go out and back through a link, with its results or without", async () => {
    const checked = checkCampaign(withKept());
    for (const withResults of [true, false]) {
      const carried = shareable(checked, withResults);
      assert.deepEqual(carried.what_if, withKept().what_if, "they are settings: what they predict comes from the results");
      const back = await unpackCampaign(await packCampaign(carried));
      assert.deepEqual(back.what_if, withKept().what_if);
      assert.equal(back.v, WHAT_IF_FORMAT);
    }
    assert.equal(shareable(checkCampaign(campaign()), true).what_if, undefined);
  });

  test("are checked for their shape: a setting the factors no longer have still opens, to be judged where it is listed", () => {
    const edited = withKept();
    edited.what_if[0].settings = { T: 260, Supplier: "C", Gone: 1 };
    assert.deepEqual(checkCampaign(edited).what_if[0].settings, { T: 260, Supplier: "C", Gone: 1 });
  });

  test("that are not settings are refused, saying why", () => {
    const refused = (whatIf, why) => assert.throws(() => checkCampaign({ ...campaign(), what_if: whatIf }), why);
    refused({ name: "Hot" }, /the kept settings is not a list/);
    refused([{ name: "Hot" }], /kept settings 1 are not a name and settings/);
    refused([{ name: "", settings: { T: 175 } }], /the name of kept settings 1 is not one line/);
    refused([{ name: "Two\nlines", settings: { T: 175 } }], /not one line of text/);
    refused([{ name: "x".repeat(SCENARIOS_HELD.name + 1), settings: {} }], /the name of kept settings 1 is not text/);
    refused([{ name: "Hot", settings: { T: [1] } }], /the setting of T is neither a number nor a label/);
    refused([{ name: "Hot", settings: { T: Infinity } }], /the setting of T is neither a number nor a label/);
    refused(
      [
        { name: "Hot", settings: { T: 200 } },
        { name: "Hot", settings: { T: 150 } },
      ],
      /two of its kept settings share a name/,
    );
    const many = Array.from({ length: SCENARIOS_HELD.count + 1 }, (_, i) => ({ name: `Settings ${i + 1}`, settings: { T: 175 } }));
    refused(many, /the kept settings is too long/);
    assert.equal(checkCampaign({ ...campaign(), what_if: many.slice(1) }).what_if.length, SCENARIOS_HELD.count);
  });
});
