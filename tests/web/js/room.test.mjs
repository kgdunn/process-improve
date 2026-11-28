// web/room.js: a room's keys from its secret, a study padded and sealed, the locker's answers, and results merged.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { checkCampaign, designKey } from "../../../web/campaign.js";
import {
  PADDED,
  ROOM_KEY,
  NO_LOCKER,
  ROOM_PATH,
  RoomError,
  mergeRoom,
  newSecret,
  openStudy,
  pad,
  readRoom,
  removeRoom,
  resultsOf,
  roomOf,
  sealStudy,
  secretOf,
  seal,
  unpad,
  unseal,
  writeRoom,
} from "../../../web/room.js";

/** Bytes 0 to 31 as a room's secret, and what the locker derives from it: tests/deploy/test_encrypted_study_locker.py holds the same. */
const SECRET = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
const TOKEN = "-znuuYT_caimAH2REqvREL_BuEwrsp7hbDyTx3TI6pw";
const ROOM = "GD5GAO3nLHTR_pA7bHTdvg";
/** The sizes the locker takes (encrypted_study_locker.py BLOB_SIZES): the nonce, the padded study, the tag. */
const BLOB_SIZES = PADDED.map((size) => 12 + size + 16);

/** A study as the page holds one: a 2^2 factorial in a temperature and a supplier, partly run. */
function study() {
  return checkCampaign({
    v: 1,
    title: "Full factorial (2-level) in T and Supplier",
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
        options: { random_state: 42, n_center_points: 0 },
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
    sources: ["entered", "entered", null, "entered"],
    simulation: null,
    model: { terms: [["T"], ["Supplier"], ["T", "Supplier"]] },
  });
}

describe("a room's secret", () => {
  test("gives the write token and the room's id the locker derives", async () => {
    const room = await roomOf(SECRET);
    assert.equal(room.token, TOKEN);
    assert.equal(room.id, ROOM);
    assert.equal(room.secret, SECRET);
  });

  test("gives a key the page cannot read back out", async () => {
    const { key } = await roomOf(SECRET);
    assert.equal(key.extractable, false);
    assert.deepEqual(key.usages.toSorted(), ["decrypt", "encrypt"]);
    await assert.rejects(crypto.subtle.exportKey("raw", key));
  });

  test("is new each time, and each gives a room of its own", async () => {
    const [a, b] = [newSecret(), newSecret()];
    assert.match(a, /^[A-Za-z0-9_-]{43}$/);
    assert.notEqual(a, b);
    assert.notEqual((await roomOf(a)).id, (await roomOf(b)).id);
  });

  test("is read from a room's link, and a link cut short is refused", () => {
    assert.equal(secretOf(`${ROOM_KEY}${SECRET}`), SECRET);
    assert.equal(secretOf("#c=abc"), null, "a study's link is not a room's");
    assert.equal(secretOf(""), null);
    assert.throws(() => secretOf(`${ROOM_KEY}${SECRET.slice(0, 40)}`), /the link has been cut or changed/);
    assert.throws(() => secretOf(`${ROOM_KEY}${SECRET}!`), RoomError);
  });
});

describe("a study, padded and sealed", () => {
  test("is padded to the smallest size it fits, and comes back as it was", () => {
    for (const [n, size] of [[0, 4096], [4092, 4096], [4093, 16384], [65532, 65536]]) {
      const bytes = Uint8Array.from({ length: n }, (_, i) => i % 251);
      const padded = pad(bytes);
      assert.equal(padded.length, size, `${n} bytes`);
      assert.deepEqual(unpad(padded), bytes);
    }
  });

  test("too large for a room is refused, saying so", () => {
    assert.throws(() => pad(new Uint8Array(65533)), (err) => err.status === 413 && /too large for a room/.test(err.message));
  });

  test("a block that is not a padded study is refused", () => {
    assert.throws(() => unpad(new Uint8Array(100)), /not a study/);
    const lying = new Uint8Array(4096);
    new DataView(lying.buffer).setUint32(0, 5000);
    assert.throws(() => unpad(lying), /not a study/);
  });

  test("is sealed to one of the sizes the locker takes, never the same bytes twice", async () => {
    const room = await roomOf(SECRET);
    const [a, b] = [await sealStudy(room, study()), await sealStudy(room, study())];
    assert.ok(BLOB_SIZES.includes(a.length), `${a.length} bytes`);
    assert.equal(a.length, BLOB_SIZES[0], "a study of a few runs fits the smallest");
    assert.notDeepEqual(a, b, "a fresh nonce each time");
  });

  test("opens with the room's key, and comes back checked", async () => {
    const room = await roomOf(SECRET);
    const back = await openStudy(room, await sealStudy(room, study()));
    assert.deepEqual(back.responses, [12.5, 9.75, null, 14]);
    assert.equal(designKey(back.design), designKey(study().design));
  });

  test("does not open with another room's key, nor once a byte is changed", async () => {
    const room = await roomOf(SECRET);
    const blob = await seal(room.key, pad(new Uint8Array([1, 2, 3])));
    await assert.rejects(unseal((await roomOf(newSecret())).key, blob), /not sealed with this link's key/);
    for (const at of [0, 20, blob.length - 1]) {
      const changed = blob.slice();
      changed[at] ^= 1;
      await assert.rejects(unseal(room.key, changed), RoomError, `byte ${at}`);
    }
  });
});

describe("the locker, as the page asks it", () => {
  /** A stand-in for fetch that answers each call with ``answers``' next, and keeps what it was asked. */
  function locker(...answers) {
    const asked = [];
    const ask = async (url, init) => {
      asked.push({ url, ...init });
      const next = answers.shift();
      if (next instanceof Error) throw next;
      return new Response(next.body ?? null, { status: next.status, headers: next.headers ?? {} });
    };
    return { ask, asked };
  }

  test("a read names the room by its id, and the version it has", async () => {
    const room = await roomOf(SECRET);
    const { ask, asked } = locker({ status: 200, body: new Uint8Array([7, 8]), headers: { ETag: '"v1"' } }, { status: 304 });
    assert.deepEqual(await readRoom(room, null, ask), { blob: new Uint8Array([7, 8]), version: "v1" });
    assert.deepEqual(await readRoom(room, "v1", ask), { same: true });
    assert.equal(asked[0].url, ROOM_PATH);
    assert.deepEqual(asked[0].headers, { Authorization: `Room ${ROOM}` }, "the id, never the token");
    assert.equal(asked[1].headers["If-None-Match"], '"v1"');
    assert.equal(asked[0].cache, "no-store");
  });

  test("a room that is gone says so", async () => {
    const { ask } = locker({ status: 404 });
    assert.deepEqual(await readRoom(await roomOf(SECRET), null, ask), { gone: true });
  });

  test("a write carries the token and the version it replaces, or makes the room", async () => {
    const room = await roomOf(SECRET);
    const { ask, asked } = locker({ status: 201, headers: { ETag: '"v1"' } }, { status: 200, headers: { ETag: '"v2"' } }, { status: 412 });
    const blob = new Uint8Array(BLOB_SIZES[0]);
    assert.deepEqual(await writeRoom(room, blob, null, ask), { version: "v1" });
    assert.deepEqual(await writeRoom(room, blob, "v1", ask), { version: "v2" });
    assert.deepEqual(await writeRoom(room, blob, "v1", ask), { moved: true }, "someone wrote first");
    assert.equal(asked[0].method, "PUT");
    assert.equal(asked[0].headers.Authorization, `Room ${TOKEN}`);
    assert.equal(asked[0].headers["If-None-Match"], "*");
    assert.equal(asked[1].headers["If-Match"], '"v1"');
    assert.equal(asked[1].headers["Content-Type"], "application/octet-stream");
    assert.equal(asked[1].body, blob);
  });

  test("a refusal or no answer is a RoomError with a sentence for the page", async () => {
    const room = await roomOf(SECRET);
    const { ask } = locker({ status: 429 }, new TypeError("Failed to fetch"), { status: 502 }, { status: 413 });
    await assert.rejects(readRoom(room, null, ask), (err) => err.status === 429 && /tries again later/.test(err.message));
    await assert.rejects(readRoom(room, null, ask), (err) => err.status === 0 && /could not be reached/.test(err.message));
    await assert.rejects(writeRoom(room, new Uint8Array(1), "v", ask), (err) => err.status === 502 && /kept here/.test(err.message));
    await assert.rejects(writeRoom(room, new Uint8Array(1), "v", ask), /too large for a room/);
  });

  test("a server without the locker, as a copy of the app served elsewhere, says so", async () => {
    const room = await roomOf(SECRET);
    for (const status of [404, 405, 501]) {
      const { ask } = locker({ status });
      await assert.rejects(writeRoom(room, new Uint8Array(1), null, ask), (err) => err.message === NO_LOCKER, `${status}`);
    }
  });

  test("a room is removed with its token, and one already gone is no error", async () => {
    const room = await roomOf(SECRET);
    const { ask, asked } = locker({ status: 204 }, { status: 404 }, { status: 403 });
    await removeRoom(room, ask);
    await removeRoom(room, ask);
    await assert.rejects(removeRoom(room, ask), RoomError);
    assert.deepEqual([asked[0].method, asked[0].headers.Authorization], ["DELETE", `Room ${TOKEN}`]);
  });
});

describe("results merged three ways", () => {
  const results = (responses, sources = responses.map((v) => (v === null ? null : "entered"))) => ({ responses, sources });

  test("a result changed in the room alone is taken, with where it came from", () => {
    const base = results([1, 2, null]);
    const there = results([1, 2.5, 3], ["entered", "entered", "simulated"]);
    const merged = mergeRoom(results([1, 2, null]), base, there, "y");
    assert.deepEqual(merged.results, there);
    assert.deepEqual(merged.took, [{ name: "y", run: 1 }, { name: "y", run: 2 }]);
    assert.deepEqual([merged.replaced, merged.ahead], [[], false]);
  });

  test("a result changed here alone is kept, and the room is behind", () => {
    const base = results([1, 2, 3]);
    const merged = mergeRoom(results([1, 2.2, null]), base, results([1, 2, 3]));
    assert.deepEqual(merged.results.responses, [1, 2.2, null], "a correction, and a result cleared");
    assert.deepEqual([merged.took, merged.ahead], [[], true]);
  });

  test("a result corrected in another browser reaches this one", () => {
    // What the link-return merge (campaign.js mergeResults) cannot do: there, a result here is always kept.
    const merged = mergeRoom(results([5.1]), results([5.1]), results([5.2]));
    assert.deepEqual(merged.results.responses, [5.2]);
  });

  test("a result changed on both sides takes the room's, and says which it replaced", () => {
    const merged = mergeRoom(results([1, 7, null]), results([1, 2, 3]), results([1, 8, 9]), "y");
    assert.deepEqual(merged.results.responses, [1, 8, 9]);
    assert.deepEqual(merged.replaced, [{ name: "y", run: 1, mine: 7, theirs: 8 }]);
    assert.deepEqual(merged.took, [{ name: "y", run: 2 }], "a result cleared here and changed there is taken");
    assert.equal(merged.ahead, false);
  });

  test("a value is never lost to a blank", () => {
    const merged = mergeRoom(results([7]), results([2]), results([null]));
    assert.deepEqual([merged.results.responses, merged.ahead], [[7], true]);
  });

  test("a browser that never saw the room keeps its own results where the room has none", () => {
    const merged = mergeRoom(results([1, null, 3]), null, results([null, 2, 4]));
    assert.deepEqual(merged.results.responses, [1, 2, 4]);
    assert.deepEqual(merged.replaced, [{ name: "", run: 2, mine: 3, theirs: 4 }]);
    assert.equal(merged.ahead, true);
  });

  test("every response, and each run's curve, is merged", () => {
    const here = { ...results([1, 2]), more_responses: { Cost: [5, null] }, more_sources: { Cost: ["entered", null] }, curves: [[1, 2], null] };
    const base = { ...results([1, 2]), more_responses: { Cost: [5, null] }, curves: [[1, 2], null] };
    const there = { ...results([1, 2]), more_responses: { Cost: [5, 6] }, more_sources: { Cost: ["entered", "entered"] }, curves: [[1, 2], [3, 4]] };
    const merged = mergeRoom(here, base, there, "y");
    assert.deepEqual(merged.results.more_responses, { Cost: [5, 6] });
    assert.deepEqual(merged.results.more_sources, { Cost: ["entered", "entered"] });
    assert.deepEqual(merged.results.curves, [[1, 2], [3, 4]]);
    assert.deepEqual(merged.took, [{ name: "Cost", run: 1 }, { name: "curve", run: 1 }]);
  });

  test("resultsOf takes a study's results and leaves its design", () => {
    const kept = resultsOf(study());
    assert.deepEqual(Object.keys(kept), ["responses", "sources"]);
    kept.responses[0] = 99;
    assert.equal(study().responses[0], 12.5, "a copy");
  });

  test("browsers that edit at once, and keep in step in turn, come to hold the same results", () => {
    // Three browsers each keep a base (the room as they last saw it) and write as the page does: read, merge, and
    // write if ahead, the write refused when someone wrote first. Random edits, then rounds with none.
    let seed = 7;
    const random = () => ((seed = (seed * 48271) % 2147483647) / 2147483647);
    const runs = 6;
    let room = { results: results(Array(runs).fill(null)), version: 0 };
    const browsers = [0, 1, 2].map(() => ({ here: results(Array(runs).fill(null)), base: null, version: -1 }));
    const sync = (b) => {
      const merged = mergeRoom(b.here, b.base, room.results);
      b.here = merged.results;
      b.base = room.results;
      b.version = room.version;
      if (merged.ahead) {
        room = { results: structuredClone(b.here), version: room.version + 1 };
        b.base = room.results;
        b.version = room.version;
      }
    };
    for (let step = 0; step < 300; step += 1) {
      const b = browsers[Math.floor(random() * 3)];
      if (random() < 0.6) {
        const run = Math.floor(random() * runs);
        b.here.responses[run] = random() < 0.2 ? null : Math.round(random() * 100);
        b.here.sources[run] = b.here.responses[run] === null ? null : "entered";
      } else {
        sync(b);
      }
    }
    for (let round = 0; round < 3; round += 1) browsers.forEach(sync);
    for (const b of browsers) assert.deepEqual(b.here, room.results);
  });
});
