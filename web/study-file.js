// The study file: Factorial's studies as JSON, to keep a copy of every study, to move them to another browser, or
// to send one to someone. The page writes it and reads it back; nothing here touches the document, so tests/web/js
// runs it under Node.
//
// The file, with one study or many:
//
//   {
//     "format": "factorial-studies",            what the file is
//     "version": 1,                             the file's own version, FILE_VERSION
//     "exported": "2026-10-21T08:30:00.000Z",
//     "app": { "name": "Factorial", "url": "https://factori.al/", "process_improve": "1.4.0" },
//     "studies": [
//       {
//         "id": "8f1c…",                        names the study for good
//         "created": "2026-10-19T12:00:00.000Z",
//         "updated": "2026-10-21T08:29:41.000Z",
//         "sha256": "ab12…",                    of the study's canonical JSON (report.js's canonicalJson)
//         "study": { "v": 1, "title": "…", "design": { … }, "responses": [ … ], … }
//       }
//     ]
//   }
//
// A study is the object a link carries and this browser keeps (campaign.js), whole: its brief, design, run sheet,
// every response's results and where each came from, the process they were simulated from, and the models.
//
// What lets a later Factorial add to the file, and an earlier one read what it can:
//   - A reader ignores a key it does not know, at any level. New things are new keys, beside the old ones.
//   - `version` goes up only for a change an earlier Factorial would misread. A file of a later version is refused
//     whole, with a sentence that says so.
//   - Each study carries its own format, `study.v` (CAMPAIGN_VERSION). It goes up when a study holds something an
//     earlier Factorial could not keep, so that one refuses that study, saying why, and opens the others.
//   - `id` stays with a study through copies and files, so a later study can point to an earlier one by its id and
//     its sha256: a design that follows another and its model, with factors added or dropped.
//   - `sha256` lets a reader tell a study changed by hand from one as Factorial saved it. The study still opens.

import { CAMPAIGN_VERSION, checkCampaign } from "./campaign.js";
import { canonicalJson, sha256 } from "./report.js";

/** What the file says it is. */
export const FILE_FORMAT = "factorial-studies";

/** The file's own version: it goes up only for a change an earlier Factorial would misread. */
export const FILE_VERSION = 1;

/** A study's fields that are this browser's bookkeeping, not the study: the file holds them beside it. */
const BOOKKEEPING = ["id", "created", "updated"];

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

/** A moment as the file writes it, from the milliseconds this browser keeps; null for none. */
const isoTime = (ms) => (Number.isFinite(ms) ? new Date(ms).toISOString() : null);

/** A moment the file holds, as the milliseconds this browser keeps; `fallback` for one that is missing or not a time. */
function msTime(text, fallback) {
  const ms = typeof text === "string" ? Date.parse(text) : NaN;
  return Number.isFinite(ms) ? ms : fallback;
}

/** A study as the file holds it: the study this browser keeps, without its bookkeeping. */
export const studyOf = (record) => Object.fromEntries(Object.entries(record).filter(([key]) => !BOOKKEEPING.includes(key)));

/**
 * The study file for `records`, the studies as this browser keeps them, as JSON to write out. `app` says what
 * wrote it (the page's address, the process-improve release); `exported` is when.
 */
export async function studyFile(records, { app = {}, exported = new Date() } = {}) {
  const studies = [];
  for (const record of records) {
    const study = studyOf(record);
    studies.push({
      id: record.id,
      created: isoTime(record.created),
      updated: isoTime(record.updated),
      sha256: await sha256(canonicalJson(study)),
      study,
    });
  }
  return { format: FILE_FORMAT, version: FILE_VERSION, exported: exported.toISOString(), app: { name: "Factorial", ...app }, studies };
}

/** Refuse a whole file, saying why in a sentence for the page. */
function refuse(why) {
  throw new Error(`This file could not be opened: ${why}.`);
}

/**
 * Read a study file's text. Each study is checked as a link's study is (campaign.js's checkCampaign); one that
 * fails is refused, with its title and the reason ("it has no runs"), and the others are read. Each study read comes with its `id` (null when
 * the file gives none), its `created` and `updated` times in milliseconds, and `changed`, true when its sha256 does
 * not match it. Throws, with a sentence for the page, for a file that is not a study file or is of a later version.
 */
export async function readStudyFile(text, now = Date.now()) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    refuse("it is not JSON");
  }
  if (!isObject(data) || data.format !== FILE_FORMAT) refuse("it is not a Factorial study file");
  if (!Number.isInteger(data.version) || data.version < 1) refuse("it does not say which version of the file it is");
  if (data.version > FILE_VERSION) {
    refuse(`it is a version ${data.version} study file, from a later Factorial, and this page reads version ${FILE_VERSION}. Reload the page for the latest Factorial`);
  }
  if (!Array.isArray(data.studies)) refuse("it holds no list of studies");
  const studies = [];
  const refused = [];
  for (const entry of data.studies) {
    const raw = isObject(entry) ? entry.study : undefined;
    const title = isObject(raw) && typeof raw.title === "string" && raw.title ? raw.title : "A study";
    let study;
    try {
      if (isObject(raw) && Number.isInteger(raw.v) && raw.v > CAMPAIGN_VERSION) {
        throw new Error(`it was saved by a later Factorial, in study format ${raw.v}; reload the page for the latest Factorial`);
      }
      study = checkCampaign(raw);
    } catch (err) {
      // The reason alone, as "it has no runs", whichever check found it.
      refused.push({ title, reason: err.message.replace(/^This design could not be opened: /, "").replace(/\.$/, "") });
      continue;
    }
    const digest = typeof entry.sha256 === "string" ? await sha256(canonicalJson(raw)) : null;
    const id = typeof entry.id === "string" && entry.id && entry.id.length <= 100 ? entry.id : null;
    const created = msTime(entry.created, now);
    studies.push({ id, created, updated: msTime(entry.updated, created), study, changed: digest !== null && digest !== entry.sha256 });
  }
  return { studies, refused };
}

/** A file name from a study's title: "Yield (%), maximise: CCD in T, P and F" becomes "yield-maximise-ccd-in-t-p-and-f". */
export function fileStem(title) {
  const stem = String(title)
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/, "");
  return stem || "study";
}
