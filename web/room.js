// A room: one study that two or more browsers keep in step through the site's room relay
// (deploy/factorial/room_relay.py), which keeps it sealed and cannot read it.
//
// The room's link carries one secret after "#room=", which a browser sends to no server. The page derives three
// things from it, each one way:
//
//   secret ──HKDF──► the key that seals the study (AES-GCM)    never leaves the browser
//          └─HKDF──► the write token                            sent with a change
//                      └─SHA-256──► the room's id               sent with a read
//
// The relay checks a change by hashing its token to the room's id. The id leads back to neither the token nor the
// key, so the relay, and anyone who reads what it keeps, holds bytes it cannot open.
//
// A study is deflated (campaign.js), padded to one of PADDED sizes, so its size does not say how many runs it has,
// and sealed with a fresh nonce. The relay keeps the 12-byte nonce, the sealed study and AES-GCM's 16-byte tag.
//
// Each browser keeps the whole study as it always has (store.js): the room is a mailbox. Results travel both ways,
// merged three ways against the room's study as this browser last saw it (mergeRoom), so a result corrected in one
// browser reaches the others. Nothing here touches the document, so tests/web/js runs it under Node.

import { deflateCampaign, fromBase64Url, inflateCampaign, toBase64Url } from "./campaign.js";

/** Where a room's link carries its secret: the address's fragment, after this. */
export const ROOM_KEY = "#room=";

/** The relay, from the app's own address, so a copy served elsewhere asks its own server. */
export const ROOM_PATH = "api/room";

/** The sizes a study is padded to, its length first; room_relay.py takes these, sealed, and nothing else. */
export const PADDED = [4096, 16384, 65536];

const NONCE = 12;
const LENGTH = 4;
const SECRET = /^[A-Za-z0-9_-]{43}$/;
const LABELS = { key: "factorial room: key", write: "factorial room: write" };
/** What each seal is bound to besides the study: this format of a room, so a later one cannot be mistaken for it. */
const BOUND = new TextEncoder().encode("factorial room 1");

/** A room's error, with the relay's HTTP status, or 0 when the relay could not be reached. */
export class RoomError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** A new room's secret: 32 random bytes, in URL-safe base64. */
export const newSecret = () => toBase64Url(crypto.getRandomValues(new Uint8Array(32)));

/** The secret an address's fragment carries, or null when it is not a room's; throws for a link cut short. */
export function secretOf(hash) {
  if (!hash.startsWith(ROOM_KEY)) return null;
  const secret = hash.slice(ROOM_KEY.length);
  if (!SECRET.test(secret)) throw new RoomError(0, "This room could not be opened: the link has been cut or changed.");
  return secret;
}

/** What the page needs of a room, from its secret: the key, which cannot be read back out, the token and the id. */
export async function roomOf(secret) {
  const material = await crypto.subtle.importKey("raw", fromBase64Url(secret), "HKDF", false, ["deriveBits", "deriveKey"]);
  const hkdf = (label) => ({ name: "HKDF", hash: "SHA-256", salt: new Uint8Array(0), info: new TextEncoder().encode(label) });
  const key = await crypto.subtle.deriveKey(hkdf(LABELS.key), material, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
  const token = new Uint8Array(await crypto.subtle.deriveBits(hkdf(LABELS.write), material, 256));
  const id = new Uint8Array(await crypto.subtle.digest("SHA-256", token)).slice(0, 16);
  return { secret, key, token: toBase64Url(token), id: toBase64Url(id) };
}

/** A deflated study, its length first, padded with zeros to the smallest of PADDED it fits; throws when none does. */
export function pad(bytes) {
  const size = PADDED.find((s) => LENGTH + bytes.length <= s);
  if (!size) throw new RoomError(413, tooLarge());
  const out = new Uint8Array(size);
  new DataView(out.buffer).setUint32(0, bytes.length);
  out.set(bytes, LENGTH);
  return out;
}

/** The study a padded block holds. */
export function unpad(padded) {
  const n = PADDED.includes(padded.length) ? new DataView(padded.buffer, padded.byteOffset).getUint32(0) : -1;
  if (n < 0 || n > padded.length - LENGTH) throw new RoomError(0, "This room could not be opened: what it holds is not a study.");
  return padded.slice(LENGTH, LENGTH + n);
}

/** Seal a padded study with the room's key and a fresh nonce: the nonce, then the sealed study and its tag. */
export async function seal(key, padded) {
  const nonce = crypto.getRandomValues(new Uint8Array(NONCE));
  const sealed = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce, additionalData: BOUND }, key, padded));
  const blob = new Uint8Array(NONCE + sealed.length);
  blob.set(nonce);
  blob.set(sealed, NONCE);
  return blob;
}

/** Open what seal made; throws when the key is another room's, or a byte was changed on the way. */
export async function unseal(key, blob) {
  try {
    const iv = blob.subarray(0, NONCE);
    return new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv, additionalData: BOUND }, key, blob.subarray(NONCE)));
  } catch {
    throw new RoomError(0, "This room could not be opened: what the server holds was not sealed with this link's key, or was changed.");
  }
}

/** A study, deflated, padded and sealed, ready for the relay. */
export async function sealStudy(room, campaign) {
  return seal(room.key, pad(await deflateCampaign(campaign)));
}

/** The study a room's blob holds, checked as a link's is (campaign.js checkCampaign). */
export async function openStudy(room, blob) {
  return inflateCampaign(unpad(await unseal(room.key, blob)));
}

// ---------------------------------------------------------------- the relay
const credential = (value) => ({ Authorization: `Room ${value}` });
const versionOf = (res) => (res.headers.get("ETag") ?? "").replace(/^"|"$/g, "") || null;

/** What a reader is told when the relay refused, or could not be reached: ``status`` 0 for no answer. */
export function roomTrouble(status) {
  if (status === 413) return tooLarge();
  if (status === 429) return "This network has changed or read rooms often in the last hour: Factorial tries again later, and keeps your changes here meanwhile.";
  if (status === 0 || status >= 500) return "The room could not be reached just now: your changes are kept here, and go when it can be.";
  return `The room refused a change (${status}): your copy here is unchanged.`;
}

function tooLarge() {
  return `This study is too large for a room, which holds ${PADDED.at(-1) / 1024} KiB of it once it is packed. A link or the study file carries it.`;
}

/** Call the relay; a failed fetch is status 0. */
async function call(ask, init) {
  try {
    return await ask(ROOM_PATH, { cache: "no-store", ...init });
  } catch {
    throw new RoomError(0, roomTrouble(0));
  }
}

/**
 * Read a room: ``{ blob, version }``; ``{ same: true }`` when it is still the ``version`` this browser has; or
 * ``{ gone: true }`` when there is no such room. Throws a RoomError for anything else. ``ask`` stands in for fetch.
 */
export async function readRoom(room, version = null, ask = fetch) {
  const headers = { ...credential(room.id), ...(version ? { "If-None-Match": `"${version}"` } : {}) };
  const res = await call(ask, { headers });
  if (res.status === 304) return { same: true };
  if (res.status === 404) return { gone: true };
  if (!res.ok) throw new RoomError(res.status, roomTrouble(res.status));
  return { blob: new Uint8Array(await res.arrayBuffer()), version: versionOf(res) };
}

/**
 * Write a sealed study to the room, if it is still at ``version``, or make the room, for a version of null:
 * ``{ version }`` of what it now holds, or ``{ moved: true }`` when someone wrote first, to read it again.
 */
export async function writeRoom(room, blob, version, ask = fetch) {
  const match = version ? { "If-Match": `"${version}"` } : { "If-None-Match": "*" };
  const headers = { ...credential(room.token), "Content-Type": "application/octet-stream", ...match };
  const res = await call(ask, { method: "PUT", headers, body: blob });
  if (res.status === 412) return { moved: true };
  if (!res.ok) throw new RoomError(res.status, roomTrouble(res.status));
  return { version: versionOf(res) };
}

/** Remove the room from the relay; a room already gone is no error. */
export async function removeRoom(room, ask = fetch) {
  const res = await call(ask, { method: "DELETE", headers: credential(room.token) });
  if (!res.ok && res.status !== 404) throw new RoomError(res.status, roomTrouble(res.status));
}

// ---------------------------------------------------------------- results, merged three ways
/** The parts of a study a room keeps in step: each response's results and their sources, and each run's curve. */
export function resultsOf(c) {
  const out = { responses: [...c.responses], sources: [...c.sources] };
  if (c.more_responses) {
    out.more_responses = structuredClone(c.more_responses);
    out.more_sources = structuredClone(c.more_sources ?? {});
  }
  if (c.curves) out.curves = structuredClone(c.curves);
  return out;
}

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * Merge the results ``there`` in the room into those ``here``, against ``base``, the room's results as this browser
 * last saw them (null when it never has). A result changed on one side alone takes that side's value. One changed on
 * both sides takes the room's, which reached it first, unless the room's is blank: a value is never lost to a blank.
 * Every browser applies the same rule, so all come to hold the same results.
 *
 * ``first`` names the first response's results in what is returned. Returns ``{ results, took, replaced, ahead }``:
 * the merged results; the results taken from the room, ``{ name, run }``; those here the room's replaced,
 * ``{ name, run, mine, theirs }``; and whether the merged results hold something the room does not yet.
 */
export function mergeRoom(here, base, there, first = "") {
  const took = [];
  const replaced = [];
  let ahead = false;
  const one = (name, values, sources, before, theirs, theirSources) => {
    const merged = [...values];
    const marks = sources ? [...sources] : null;
    values.forEach((mine, run) => {
      const now = theirs?.[run] ?? null;
      const was = before?.[run] ?? null;
      if (same(mine, now)) return;
      const keep = same(now, was) || (!same(mine, was) && now === null);
      if (keep) {
        ahead = true; // a change here the room has not had
        return;
      }
      if (!same(mine, was) && mine !== null) replaced.push({ name, run, mine, theirs: now });
      else took.push({ name, run });
      merged[run] = now;
      if (marks) marks[run] = theirSources?.[run] ?? (now === null ? null : "entered");
    });
    return [merged, marks];
  };
  const out = {};
  [out.responses, out.sources] = one(first, here.responses, here.sources, base?.responses, there.responses, there.sources);
  if (here.more_responses) {
    out.more_responses = {};
    out.more_sources = {};
    for (const name of Object.keys(here.more_responses)) {
      [out.more_responses[name], out.more_sources[name]] = one(
        name,
        here.more_responses[name],
        here.more_sources?.[name] ?? here.more_responses[name].map(() => null),
        base?.more_responses?.[name],
        there.more_responses?.[name],
        there.more_sources?.[name],
      );
    }
  }
  if (here.curves || there.curves) {
    const curves = here.curves ?? here.responses.map(() => null);
    [out.curves] = one("curve", curves, null, base?.curves, there.curves, null);
  }
  return { results: out, took, replaced, ahead };
}
