// web/campaign.js: a campaign packed into a link and unpacked, checked as text anyone could write.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  AIMS,
  BOOK_PARAM,
  CAMPAIGN_VERSION,
  bookName,
  campaignTitle,
  checkBookExample,
  checkCampaign,
  designKey,
  indexedName,
  mergeResults,
  packCampaign,
  progress,
  sameCampaign,
  shareable,
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
    v: CAMPAIGN_VERSION,
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

  test("another format", async () => refused(await packed({ ...campaign(), v: 2 }), /format 2/));

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
    refusedBy(() => checkBookExample({ ...example(), campaign: { ...campaign(), v: 2 } }, "two-factor-supplier"), /format 2/);
  });

  test("a copy saved here is the book's only while its results and model are as the book has them", () => {
    const book = checkCampaign(campaign());
    assert.ok(sameCampaign(checkCampaign(campaign()), book));
    assert.ok(!sameCampaign({ ...book, responses: [12.5, 9.75, 1, 14] }, book), "a result typed in");
    assert.ok(!sameCampaign({ ...book, model: "main_effects" }, book), "another model chosen");
    assert.ok(!sameCampaign({ ...book, title: "Renamed" }, book));
  });
});
