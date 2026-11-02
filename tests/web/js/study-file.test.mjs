// web/study-file.js: studies written to a file and read back, the file checked as text anyone could write.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { pathToFileURL } from "node:url";

import { CAMPAIGN_VERSION, LEVELS_FORMAT, MANY_FACTORS_FORMAT, WHAT_IF_FORMAT, checkCampaign, packCampaign, shareable } from "../../../web/campaign.js";
import { canonicalJson, sha256 } from "../../../web/report.js";
import { FILE_FORMAT, FILE_VERSION, fileStem, readStudyFile, studyFile, studyOf } from "../../../web/study-file.js";

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

  test("an earlier Factorial, of format 4, refuses a study with kept settings as saved by a later one", async () => {
    // The page as it was before format 5: these modules, reading formats 1 to 4. Its study file and its links
    // refuse a study by its format, so it never opens one only to drop the kept settings on its next save.
    const dir = mkdtempSync(join(tmpdir(), "factorial-format-4-"));
    try {
      const web = new URL("../../../web/", import.meta.url);
      for (const name of ["study-file.js", "report.js", "logic.js"]) copyFileSync(new URL(name, web), join(dir, name));
      const source = readFileSync(new URL("campaign.js", web), "utf8");
      const earlier = source.replace("export const CAMPAIGN_VERSION = 5;", "export const CAMPAIGN_VERSION = 4;");
      assert.notEqual(earlier, source);
      writeFileSync(join(dir, "campaign.js"), earlier);
      const before = await import(pathToFileURL(join(dir, "study-file.js")).href);
      const linked = await import(pathToFileURL(join(dir, "campaign.js")).href);
      const study = { ...record("kept", "Kept"), v: WHAT_IF_FORMAT, what_if: [{ name: "Hot", settings: { T: 200, Supplier: "A" } }] };
      const { studies, refused } = await before.readStudyFile(await asText([study, record("none", "Without")]));
      assert.deepEqual(refused, [
        { title: "Kept", reason: "it was saved by a later Factorial, in study format 5; reload the page for the latest Factorial" },
      ]);
      assert.deepEqual(studies.map((s) => s.id), ["none"], "the other studies in the file open");
      const link = await packCampaign(shareable(checkCampaign(studyOf(study)), true));
      await assert.rejects(linked.unpackCampaign(link), /it was saved in format 5, and this page reads formats 1 to 4/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("names its file from the study's title", () => {
    assert.equal(fileStem("Yield (%), maximise: CCD in T, P and F"), "yield-maximise-ccd-in-t-p-and-f");
    assert.equal(fileStem("Débit à 25 °C"), "debit-a-25-c");
    assert.equal(fileStem("---"), "study");
    assert.ok(fileStem("x".repeat(100)).length <= 60);
  });
});
