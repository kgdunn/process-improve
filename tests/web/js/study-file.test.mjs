// web/study-file.js: studies written to a file and read back, the file checked as text anyone could write.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { pathToFileURL } from "node:url";

import {
  CAMPAIGN_VERSION,
  CURVE_FORMAT,
  LEVELS_FORMAT,
  MANY_FACTORS_FORMAT,
  ROUNDS_FORMAT,
  SET_ASIDE_FORMAT,
  TARGETS_FORMAT,
  WHAT_IF_FORMAT,
  checkCampaign,
  bestResult,
  bestSoFar,
  campaignRuns,
  packCampaign,
  roundsOf,
  shareable,
} from "../../../web/campaign.js";
import { canonicalJson, sha256 } from "../../../web/report.js";
import { FILE_FORMAT, FILE_VERSION, fileStem, readStudyFile, runsDigest, studyFile, studyOf } from "../../../web/study-file.js";

/** A study as this browser keeps one: a 2^2 factorial in a numeric factor and a supplier, partly run. */
function record(id = "study-1", title = "Full factorial (2-level) in T and Supplier") {
  return {
    id,
    created: Date.UTC(2026, 9, 19, 12, 0),
    updated: Date.UTC(2026, 9, 21, 8, 29, 41),
    v: 1, // the format a study of this design is saved in
    title,
    design: {
      spec: {
        version: 1,
        design_type: "full_factorial",
        factors: [
          { name: "T", low: 150, high: 200, units: "degC" },
          { name: "Supplier", low: "A", high: "B", units: "" },
        ],
        response: "y",
        model: "interactions",
        options: { random_state: 1792419802, n_center_points: 0 },
      },
      columns: ["Std order", "Run order", "T", "Supplier", "y"],
      rows: [
        [3, 1, 150, "B", null],
        [1, 2, 150, "A", null],
        [4, 3, 200, "B", null],
        [2, 4, 200, "A", null],
      ],
      model: "interactions",
      n_runs: 4,
      facts: { center_runs: 0, notes: [] },
      outside_range: [],
    },
    responses: [12.5, 9.75, null, 14],
    sources: ["entered", "entered", null, "simulated"],
    simulation: null,
    simulate: { scenario: "reactor", sigma: 1, seed: 3 },
    model: "",
  };
}

const exported = new Date(Date.UTC(2026, 9, 21, 8, 30));
const app = { url: "https://factori.al/", process_improve: "1.4.0" };
const written = (records) => studyFile(records, { app, exported });
const asText = async (records) => JSON.stringify(await written(records), null, 2);

describe("the study file", () => {
  test("says what it is, its version, when and by what it was written", async () => {
    const file = await written([record()]);
    assert.equal(file.format, FILE_FORMAT);
    assert.equal(file.version, FILE_VERSION);
    assert.equal(file.exported, "2026-10-21T08:30:00.000Z");
    assert.deepEqual(file.app, { name: "Factorial", url: "https://factori.al/", process_improve: "1.4.0" });
  });

  test("holds each study whole, its bookkeeping beside it, its times readable, and the digest of the study", async () => {
    const { studies } = await written([record("a"), record("b", "Another")]);
    assert.deepEqual(studies.map((s) => s.id), ["a", "b"]);
    assert.equal(studies[0].created, "2026-10-19T12:00:00.000Z");
    assert.equal(studies[0].updated, "2026-10-21T08:29:41.000Z");
    const { id, created, updated, ...study } = record("a");
    assert.deepEqual(studies[0].study, study, "the study as kept, without id, created and updated");
    assert.deepEqual(studyOf(record("a")), study);
    assert.equal(studies[0].sha256, await sha256(canonicalJson(study)));
  });

  test("reads back what it wrote: each study checked, with its id and times", async () => {
    const { studies, refused } = await readStudyFile(await asText([record("a"), record("b", "Another")]));
    assert.deepEqual(refused, []);
    assert.deepEqual(studies.map((s) => [s.id, s.study.title, s.changed]), [
      ["a", "Full factorial (2-level) in T and Supplier", false],
      ["b", "Another", false],
    ]);
    assert.deepEqual(studies[0].study, checkCampaign(studyOf(record())));
    assert.equal(studies[0].created, record().created);
    assert.equal(studies[0].updated, record().updated);
  });

  test("ignores the keys it does not know, at any level, so a later Factorial can add to it", async () => {
    const file = await written([record()]);
    file.stacked_on = ["an earlier file"];
    file.studies[0].builds_on = [{ id: "study-0", sha256: "ab" }];
    file.studies[0].study.inverse = { target: 63 };
    const { studies, refused } = await readStudyFile(JSON.stringify(file));
    assert.deepEqual(refused, []);
    assert.equal(studies.length, 1);
    assert.equal(studies[0].study.inverse, undefined, "what this page does not know, it does not keep");
    assert.equal(studies[0].changed, true, "the study holds a key it was not saved with");
  });

  test("says so of a study changed since it was saved, and still reads it", async () => {
    const file = await written([record()]);
    file.studies[0].study.responses[2] = 11;
    const { studies } = await readStudyFile(JSON.stringify(file));
    assert.equal(studies[0].changed, true);
    assert.equal(studies[0].study.responses[2], 11);
    delete file.studies[0].sha256;
    assert.equal((await readStudyFile(JSON.stringify(file))).studies[0].changed, false, "no digest, nothing to say");
  });

  test("refuses a study it cannot read, saying which and why, and reads the others", async () => {
    const file = await written([record("a", "Good"), record("b", "Broken"), record("c", "Later")]);
    file.studies[1].study.design.rows = [];
    file.studies[2].study.v = CAMPAIGN_VERSION + 1;
    const { studies, refused } = await readStudyFile(JSON.stringify(file));
    assert.deepEqual(studies.map((s) => s.id), ["a"]);
    assert.deepEqual(refused, [
      { title: "Broken", reason: "it has no runs" },
      { title: "Later", reason: `it was saved by a later Factorial, in study format ${CAMPAIGN_VERSION + 1}; reload the page for the latest Factorial` },
    ]);
  });

  test("refuses a file that is not a study file, or is from a later Factorial, in a sentence", async () => {
    const refusal = (text) => readStudyFile(text).then(() => null, (err) => err.message);
    assert.equal(await refusal("{"), "This file could not be opened: it is not JSON.");
    assert.equal(await refusal('{"format": "something else"}'), "This file could not be opened: it is not a Factorial study file.");
    assert.match(await refusal(JSON.stringify({ format: FILE_FORMAT, version: FILE_VERSION + 1, studies: [] })), /from a later Factorial/);
    assert.match(await refusal(JSON.stringify({ format: FILE_FORMAT, version: FILE_VERSION })), /no list of studies/);
  });

  test("gives a study with no id, or no times, none to keep: the page makes them", async () => {
    const file = await written([record()]);
    delete file.studies[0].id;
    file.studies[0].created = "yesterday";
    delete file.studies[0].updated;
    const now = Date.UTC(2026, 9, 22);
    const [study] = (await readStudyFile(JSON.stringify(file), now)).studies;
    assert.equal(study.id, null);
    assert.equal(study.created, now);
    assert.equal(study.updated, now, "updated falls back to created");
  });

  for (const k of [13, 24, 40]) {
    test(`writes a screening study of ${k} factors and reads it back whole, in the format that holds it`, async () => {
      const names = Array.from({ length: k }, (_, i) => `X${i + 1}`);
      const study = record("many", `Plackett-Burman screening in ${k} factors`);
      study.design.spec = { ...study.design.spec, design_type: "plackett_burman", model: "main_effects" };
      study.design.spec.factors = names.map((name) => ({ name, low: 0, high: 10, units: "u" }));
      study.design.columns = ["Std order", "Run order", ...names, "y"];
      study.design.rows = [1, 2, 3, 4].map((run) => [run, run, ...names.map((_, j) => ((run + j) % 2 ? 10 : 0)), null]);
      study.design.model = "main_effects";
      study.v = MANY_FACTORS_FORMAT;
      const { studies, refused } = await readStudyFile(await asText([study]));
      assert.deepEqual(refused, []);
      assert.equal(studies[0].study.v, MANY_FACTORS_FORMAT);
      assert.deepEqual(studies[0].study.design.spec.factors.map((f) => f.name), names);
      assert.equal(studies[0].changed, false);
      assert.ok(MANY_FACTORS_FORMAT > LEVELS_FORMAT, "a page of format 3 refuses it as saved by a later Factorial");
    });
  }

  test("writes the settings kept from the what-if panel and reads them back, in the format that holds them", async () => {
    const kept = [
      { name: "Cooler, more pressure", settings: { T: 175, Supplier: "B" } },
      { name: "Hot", settings: { T: 200, Supplier: "A" } },
    ];
    const study = { ...record("kept"), v: WHAT_IF_FORMAT, what_if: kept };
    const { studies, refused } = await readStudyFile(await asText([study, record("none", "Without")]));
    assert.deepEqual(refused, []);
    assert.deepEqual(studies[0].study.what_if, kept, "no prediction is kept: each is worked out from the results");
    assert.equal(studies[0].study.v, WHAT_IF_FORMAT);
    assert.equal(studies[0].changed, false);
    assert.equal(studies[1].study.what_if, undefined);
    assert.equal(studies[1].study.v, 1, "a study without kept settings keeps the format it had");
  });

  /**
   * The page as it was before format ``format`` + 1: these modules, reading formats 1 to ``format``. Its study file
   * and its links refuse a study by its format, so it never opens one only to drop what it cannot hold on its next
   * save; the other studies in the file open.
   */
  async function refusedByAnEarlierPage(format, study) {
    const dir = mkdtempSync(join(tmpdir(), `factorial-format-${format}-`));
    try {
      const web = new URL("../../../web/", import.meta.url);
      for (const name of ["study-file.js", "report.js", "logic.js", "numbers.js"]) copyFileSync(new URL(name, web), join(dir, name));
      const source = readFileSync(new URL("campaign.js", web), "utf8");
      const earlier = source.replace(/export const CAMPAIGN_VERSION = \d+;/, `export const CAMPAIGN_VERSION = ${format};`);
      assert.notEqual(earlier, source);
      writeFileSync(join(dir, "campaign.js"), earlier);
      const before = await import(pathToFileURL(join(dir, "study-file.js")).href);
      const linked = await import(pathToFileURL(join(dir, "campaign.js")).href);
      const { studies, refused } = await before.readStudyFile(await asText([study, record("none", "Without")]));
      const saved = format + 1;
      assert.deepEqual(refused, [
        { title: study.title, reason: `it was saved by a later Factorial, in study format ${saved}; reload the page for the latest Factorial` },
      ]);
      assert.deepEqual(studies.map((s) => s.id), ["none"], "the other studies in the file open");
      const link = await packCampaign(shareable(checkCampaign(studyOf(study)), true));
      await assert.rejects(linked.unpackCampaign(link), new RegExp(`it was saved in format ${saved}, and this page reads formats 1 to ${format}`));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  test("an earlier Factorial, of format 4, refuses a study with kept settings as saved by a later one", async () => {
    const study = { ...record("kept", "Kept"), v: WHAT_IF_FORMAT, what_if: [{ name: "Hot", settings: { T: 200, Supplier: "A" } }] };
    await refusedByAnEarlierPage(WHAT_IF_FORMAT - 1, study);
  });

  test("an earlier Factorial, of format 5, refuses a study that reads a curve on each run as saved by a later one", async () => {
    const study = { ...record("curved", "Curved"), v: CURVE_FORMAT };
    study.design.spec.curve = { label: "Colour", argument: { label: "Time", values: [0, 5, 10] } };
    study.curves = [[0, 1, 2], null, null, null];
    await refusedByAnEarlierPage(CURVE_FORMAT - 1, study);
  });

  test("an earlier Factorial, of format 6, refuses a study with a run set aside as saved by a later one", async () => {
    const study = { ...record("aside", "Aside"), v: SET_ASIDE_FORMAT, set_aside: { y: [1] } };
    await refusedByAnEarlierPage(SET_ASIDE_FORMAT - 1, study);
  });

  test("writes the runs set aside from a fit and reads them back, in the format that holds them", async () => {
    const study = { ...record("aside"), v: SET_ASIDE_FORMAT, set_aside: { y: [3, 0, 3] } };
    const { studies, refused } = await readStudyFile(await asText([study, record("none", "Without")]));
    assert.deepEqual(refused, []);
    assert.deepEqual(studies[0].study.set_aside, { y: [0, 3] }, "each place once, in order");
    assert.equal(studies[0].study.v, SET_ASIDE_FORMAT);
    assert.deepEqual(studies[0].study.responses, [12.5, 9.75, null, 14], "a run set aside keeps its result");
    assert.equal(studies[1].study.set_aside, undefined);
    assert.equal(studies[1].study.v, 1, "a study with no run set aside keeps the format it had");
    assert.deepEqual(shareable(studies[0].study, true).set_aside, { y: [0, 3] });
    assert.equal(shareable(studies[0].study, false).set_aside, undefined, "a link without results has none to set aside");
  });

  test("a run set aside must be one of the run sheet's, for a response the study measures", () => {
    const { id, created, updated, ...study } = record();
    assert.throws(() => checkCampaign({ ...study, v: SET_ASIDE_FORMAT, set_aside: { y: [4] } }), /a run set aside is not one of the run sheet's/);
    assert.throws(() => checkCampaign({ ...study, v: SET_ASIDE_FORMAT, set_aside: { cost: [0] } }), /which the study does not measure/);
    assert.throws(() => checkCampaign({ ...study, v: SET_ASIDE_FORMAT, set_aside: [0] }), /not listed by response/);
    assert.equal(checkCampaign({ ...study, set_aside: { y: [] } }).set_aside, undefined, "none set aside is not kept");
  });

  test("a study that builds on another goes out and back with the link to it, in the format that holds it", async () => {
    const first = record("round-1", "The screen");
    const digest = await runsDigest(studyOf(first));
    assert.match(digest, /^[0-9a-f]{64}$/);
    const second = { ...record("round-2", "The response surface"), v: ROUNDS_FORMAT, builds_on: [{ id: "round-1", sha256: digest, title: "The screen" }] };
    const { studies, refused } = await readStudyFile(await asText([first, second]));
    assert.deepEqual(refused, []);
    assert.deepEqual(studies[1].study.builds_on, [{ id: "round-1", sha256: digest, title: "The screen" }]);
    assert.equal(studies[1].study.v, ROUNDS_FORMAT);
    assert.equal(studies[0].study.builds_on, undefined);
    assert.equal(studies[0].study.v, 1, "a study that stands alone keeps the format it had");
    assert.deepEqual(shareable(studies[1].study, false).builds_on, studies[1].study.builds_on, "a link names the round before, with or without results");
    // The digest is of the runs and their results: a result changed changes it, and what Learn was asked does not.
    assert.equal(await runsDigest({ ...studyOf(first), what_if: [{ name: "Hot", settings: { T: 200 } }], model: "main_effects" }), digest);
    assert.notEqual(await runsDigest({ ...studyOf(first), responses: [12.5, 9.75, 11, 14] }), digest);
  });

  test("the study a study builds on must be named, with a digest that is one", () => {
    const { id, created, updated, ...study } = record();
    const on = (entry) => () => checkCampaign({ ...study, v: ROUNDS_FORMAT, builds_on: [entry] });
    assert.throws(on("round-1"), /study 1 it builds on is not named/);
    assert.throws(on({ id: "" }), /the name of study 1 it builds on is not one line of text/);
    assert.throws(on({ id: "round-1", sha256: "abc" }), /the digest of study 1 it builds on is not a SHA-256/);
    assert.throws(() => checkCampaign({ ...study, v: ROUNDS_FORMAT, builds_on: "round-1" }), /the studies it builds on/);
    assert.deepEqual(checkCampaign({ ...study, v: ROUNDS_FORMAT, builds_on: [{ id: "round-1" }] }).builds_on, [{ id: "round-1", sha256: null }]);
    assert.equal(checkCampaign({ ...study, builds_on: [] }).builds_on, undefined, "building on none is not kept");
  });

  test("an earlier Factorial, of format 8, refuses a study that builds on another as saved by a later one", async () => {
    await refusedByAnEarlierPage(ROUNDS_FORMAT - 1, { ...record("later", "Later"), v: ROUNDS_FORMAT, builds_on: [{ id: "round-1", sha256: null }] });
  });

  test("the rounds of a campaign: the studies before, the study itself, and those built on any of them", () => {
    const study = (id, created, on = null, sha = null) => ({ id, title: `Study ${id}`, created, ...(on ? { builds_on: [{ id: on, sha256: sha, title: `Study ${on}` }] } : {}) });
    const [a, b, c, d] = [study("a", 1), study("b", 2, "a", "1".repeat(64)), study("c", 3, "b"), study("d", 4, "a")];
    const said = (rounds) => rounds.map((r) => `${r.round} ${r.id}${r.stale ? " stale" : ""}${r.missing ? " missing" : ""}`);
    assert.deepEqual(said(roundsOf([d, c, b, a], "c")), ["1 a", "2 b", "3 c", "2 d"], "from the first round, whichever study is on the page");
    assert.deepEqual(said(roundsOf([a, b, c, d], "a")), ["1 a", "2 b", "3 c", "2 d"]);
    assert.equal(roundsOf([a, b], "b")[1].record, b);
    assert.deepEqual(said(roundsOf([a], "a")), ["1 a"], "a study that stands alone is one round");
    assert.deepEqual(roundsOf([a], "nowhere"), []);
    // The first round's runs changed after the second was built on them: the second is stale. A digest unknown says nothing.
    assert.deepEqual(said(roundsOf([a, b, c], "c", new Map([["a", "2".repeat(64)], ["b", "3".repeat(64)]]))), ["1 a", "2 b stale", "3 c"]);
    assert.deepEqual(said(roundsOf([a, b], "b", new Map([["a", "1".repeat(64)]]))), ["1 a", "2 b"], "unchanged");
    // A study that came by a link, without the round before: that round is named from the link.
    const lost = roundsOf([b, c], "c");
    assert.deepEqual(said(lost), ["1 a missing", "2 b", "3 c"]);
    assert.deepEqual([lost[0].title, lost[0].record], ["Study a", null]);
    // Two studies written as each other's parent, by hand: each is listed once.
    assert.equal(roundsOf([study("x", 1, "y"), study("y", 2, "x")], "x").length, 2);
  });

  test("the runs of a campaign, round by round, and the best result so far for the goal", () => {
    const sheet = (id, rows, responses) => ({ id, design: { columns: ["Std order", "Run order", "T", "Supplier", "y"], rows }, responses });
    const rounds = [
      { round: 1, record: sheet("a", [[1, 1, 150, "A", null], [2, 2, 200, "B", null], [3, 3, 175, "A", null]], [5, null, 9]) },
      { round: 2, record: null, missing: true }, // named by a link, with no study at hand
      { round: 2, record: sheet("b", [[1, 1, 225, "A", null], [2, 2, 250, "B", null]], [7, 8]) },
    ]; // prettier-ignore
    const runs = campaignRuns(rounds);
    assert.deepEqual(runs.map((run) => [run.round, run.run, run.value]), [[1, 0, 5], [1, 2, 9], [2, 0, 7], [2, 1, 8]], "a run with no result yet is not one"); // prettier-ignore
    assert.ok(runs.every((run) => run.at === null), "with no two factors asked for, no place");
    assert.deepEqual(campaignRuns(rounds, ["T", "Std order"]).map((run) => run.at), [[150, 1], [175, 3], [225, 1], [250, 2]]); // prettier-ignore
    assert.deepEqual(campaignRuns(rounds, ["T", "Supplier"])[0].at, null, "a factor with levels has no place on an axis");
    assert.deepEqual(campaignRuns(rounds, ["T", "Flow"])[0].at, null, "nor has a factor the round did not have");

    assert.deepEqual(bestSoFar(runs, { goal: "maximize" }), [5, 9, 9, 9]);
    assert.deepEqual(bestSoFar(runs, { goal: "minimize" }), [5, 5, 5, 5]);
    assert.deepEqual(bestSoFar(runs, { goal: "target", target: "7.4" }), [5, 9, 7, 7], "the nearest to the target");
    assert.deepEqual(bestSoFar(runs, { goal: "target", target: "" }), [null, null, null, null], "a target not typed names no best");
    assert.deepEqual(bestSoFar(runs, { goal: "learn" }), [null, null, null, null]);
    assert.deepEqual(bestSoFar([], { goal: "maximize" }), []);

    assert.deepEqual(bestResult([5, null, 9], { goal: "maximize" }), { value: 9, run: 2 });
    assert.deepEqual(bestResult([5, null, 9], { goal: "minimize" }), { value: 5, run: 0 });
    assert.deepEqual(bestResult([5, null, 9], { goal: "target", target: 8 }), { value: 9, run: 2 });
    assert.equal(bestResult([null, null], { goal: "maximize" }), null, "no result yet");
    assert.equal(bestResult([5], {}), null);
  });

  test("an earlier Factorial, of format 7, refuses a study with a target or a window as saved by a later one", async () => {
    await refusedByAnEarlierPage(TARGETS_FORMAT - 1, { ...record("aimed", "Aimed"), v: TARGETS_FORMAT, targets: { y: 12 } });
    await refusedByAnEarlierPage(TARGETS_FORMAT - 1, { ...record("held", "Held"), v: TARGETS_FORMAT, windows: { y: [10, null] } });
  });

  test("writes the targets and the windows typed in Learn and reads them back, in the format that holds them", async () => {
    const study = { ...record("asked"), v: TARGETS_FORMAT, targets: { y: 12 }, windows: { y: [10, null] } };
    const { studies, refused } = await readStudyFile(await asText([study, record("none", "Without")]));
    assert.deepEqual(refused, []);
    assert.deepEqual(studies[0].study.targets, { y: 12 });
    assert.deepEqual(studies[0].study.windows, { y: [10, null] }, "an end left open stays open");
    assert.equal(studies[0].study.v, TARGETS_FORMAT);
    assert.equal(TARGETS_FORMAT, SET_ASIDE_FORMAT + 1, "later than any an earlier page reads");
    assert.deepEqual([studies[1].study.targets, studies[1].study.windows, studies[1].study.v], [undefined, undefined, 1]);
    for (const withResults of [true, false]) {
      const carried = shareable(studies[0].study, withResults);
      assert.deepEqual([carried.targets, carried.windows], [{ y: 12 }, { y: [10, null] }], "a link asks the same of its results");
    }
  });

  test("a target is a number and a window two ends, for a response the study measures", () => {
    const { id, created, updated, ...study } = record();
    const refused = (asked, why) => assert.throws(() => checkCampaign({ ...study, v: TARGETS_FORMAT, ...asked }), why);
    refused({ targets: { cost: 3 } }, /its targets name "cost", which the study does not measure/);
    refused({ windows: { cost: [1, 2] } }, /its windows name "cost", which the study does not measure/);
    refused({ targets: [12] }, /its targets are not listed by response/);
    for (const target of ["12", null, Infinity]) refused({ targets: { y: target } }, /the target of y is not a number/);
    for (const ends of [[1], [1, 2, 3], ["1", 2], 5, null]) refused({ windows: { y: ends } }, /the window of y is not its two ends/);
    const empty = checkCampaign({ ...study, targets: {}, windows: { y: [null, null] } });
    assert.deepEqual([empty.targets, empty.windows, empty.v], [undefined, undefined, 1], "nothing typed is not kept");
    assert.deepEqual(checkCampaign({ ...study, windows: { y: [14, 10] } }).windows, { y: [14, 10] }, "ends past each other open: Python says so");
  });

  test("names its file from the study's title", () => {
    assert.equal(fileStem("Yield (%), maximize: CCD in T, P and F"), "yield-maximize-ccd-in-t-p-and-f");
    assert.equal(fileStem("Débit à 25 °C"), "debit-a-25-c");
    assert.equal(fileStem("---"), "study");
    assert.equal(fileStem("x".repeat(100)), "x".repeat(60));
    // A long title ends on a whole word: 61 characters to "pressure", so the name stops at the word before it.
    const long = "Yield (%), maximize: Full factorial (2-level) in Temperature, Pressure and Flow";
    assert.equal(fileStem(long), "yield-maximize-full-factorial-2-level-in-temperature");
    assert.equal(fileStem(`${"a".repeat(58)} b c`), `${"a".repeat(58)}-b`); // a word that ends on the 60th character stays
    assert.equal(fileStem(`${"a".repeat(59)} b c`), "a".repeat(59)); // and one that would end on the 61st goes
  });
});
