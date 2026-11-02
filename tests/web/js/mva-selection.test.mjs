// web/mva/selection.js: the store every linked view of the multivariate app reads and writes.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { SET_COLOURS, Selection, combine, idOf } from "../../../web/mva/selection.js";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("selecting", () => {
  test("ids are held as text, whatever the table's index holds", () => {
    const store = new Selection();
    store.select("rows", [52, "53"]);
    assert.deepEqual([...store.selected("rows")], ["52", "53"]);
    assert.ok(store.has("rows", 52));
    assert.equal(idOf(7), "7");
  });

  test("replace, add, remove and toggle combine a pick with the selection", () => {
    const store = new Selection();
    store.select("rows", ["a", "b"]);
    store.select("rows", ["c"], "add");
    assert.deepEqual([...store.selected("rows")].sort(), ["a", "b", "c"]);
    store.select("rows", ["a"], "remove");
    store.select("rows", ["b", "d"], "toggle");
    assert.deepEqual([...store.selected("rows")].sort(), ["c", "d"]);
    store.select("rows", ["x"]);
    assert.deepEqual([...store.selected("rows")], ["x"]);
  });

  test("rows and columns are selected apart", () => {
    const store = new Selection();
    store.select("rows", ["1"]);
    store.select("cols", ["Tin"]);
    store.clear("rows");
    assert.equal(store.selected("rows").size, 0);
    assert.deepEqual([...store.selected("cols")], ["Tin"]);
  });

  test("a selection handed out is a copy the caller cannot change the store through", () => {
    const store = new Selection();
    store.select("rows", ["1"]);
    store.selected("rows").add("2");
    assert.deepEqual([...store.selected("rows")], ["1"]);
  });

  test("an unknown dimension or mode is refused", () => {
    const store = new Selection();
    assert.throws(() => store.select("cells", ["1"]), /Unknown dimension/);
    assert.throws(() => store.select("rows", ["1"], "xor"), /Unknown mode/);
  });
});

describe("telling the views", () => {
  test("changes made together arrive together: a lasso of 5,000 points is one call", async () => {
    const store = new Selection();
    const calls = [];
    store.subscribe((changes) => calls.push([...changes].sort()));
    store.select(
      "rows",
      Array.from({ length: 5000 }, (_, i) => i),
    );
    store.hover("cols", "Tin");
    store.exclude(["3"]);
    await tick();
    assert.deepEqual(calls, [["excluded", "hover", "rows"]]);
  });

  test("a change that changes nothing tells no one", async () => {
    const store = new Selection();
    store.select("rows", ["1"]);
    await tick();
    let calls = 0;
    store.subscribe(() => (calls += 1));
    store.select("rows", ["1"]);
    store.hover("rows", null);
    store.include(["9"]);
    await tick();
    assert.equal(calls, 0);
  });

  test("a view that unsubscribes hears nothing more", async () => {
    const store = new Selection();
    let calls = 0;
    const stop = store.subscribe(() => (calls += 1));
    stop();
    store.select("rows", ["1"]);
    await tick();
    assert.equal(calls, 0);
  });
});

describe("excluding", () => {
  test("excluded rows are kept apart from the selection", () => {
    const store = new Selection();
    store.select("rows", ["52", "53"]);
    store.exclude(store.selected("rows"));
    assert.ok(store.isExcluded(52));
    store.include(["52"]);
    assert.deepEqual([...store.excluded()], ["53"]);
    assert.equal(store.selected("rows").size, 2, "excluding does not deselect");
  });
});

describe("saved sets", () => {
  test("a lasso saved under a name keeps its rows and takes the next colour", () => {
    const store = new Selection();
    store.select("rows", ["1", "2"]);
    store.saveSet("Upset");
    store.saveSet("Start-up", ["10"]);
    const sets = store.sets();
    assert.deepEqual(
      sets.map((s) => [s.name, [...s.rows], s.colour]),
      [
        ["Upset", ["1", "2"], SET_COLOURS[0]],
        ["Start-up", ["10"], SET_COLOURS[1]],
      ],
    );
  });

  test("two sets combine by union, intersection, difference and symmetric difference", () => {
    const a = new Set(["1", "2", "3"]);
    const b = new Set(["3", "4"]);
    assert.deepEqual([...combine(a, "union", b)], ["1", "2", "3", "4"]);
    assert.deepEqual([...combine(a, "intersection", b)], ["3"]);
    assert.deepEqual([...combine(a, "difference", b)], ["1", "2"]);
    assert.deepEqual([...combine(a, "symmetric", b)], ["1", "2", "4"]);
    assert.throws(() => combine(a, "xor", b), /Unknown set operation/);
  });

  test("a set, or two sets combined, can be selected again", () => {
    const store = new Selection();
    store.saveSet("A", ["1", "2", "3"]);
    store.saveSet("B", ["3", "4"]);
    store.selectSet("A", { operation: "difference", other: "B" });
    assert.deepEqual([...store.selected("rows")], ["1", "2"]);
  });

  test("renaming keeps a set's rows, colour and place; a clash or an empty name is refused", () => {
    const store = new Selection();
    store.saveSet("A", ["1"]);
    store.saveSet("B", ["2"]);
    store.renameSet("A", "Upset");
    assert.deepEqual(
      store.sets().map((s) => s.name),
      ["Upset", "B"],
    );
    assert.equal(store.sets()[0].colour, SET_COLOURS[0]);
    assert.throws(() => store.renameSet("Upset", "B"), /exists already/);
    assert.throws(() => store.renameSet("B", "  "), /needs a name/);
    assert.throws(() => store.setRows("C"), /no saved set/);
  });
});

describe("the project file", () => {
  test("the sets and the excluded rows round-trip through JSON", () => {
    const store = new Selection();
    store.saveSet("Upset", [52, 53]);
    store.saveSet("Mine", ["B7"], "#123456");
    store.exclude([54]);
    const again = Selection.fromJSON(JSON.parse(JSON.stringify(store)));
    assert.deepEqual(again.toJSON(), store.toJSON());
  });

  test("what a file holds is read as data: a bad colour or a non-list is ignored", () => {
    const store = Selection.fromJSON({
      selections: { A: ["1"], B: "not a list" },
      colours: { A: "red; background: url(x)" },
      excluded: "nope",
    });
    assert.deepEqual(
      store.sets().map((s) => [s.name, s.colour]),
      [["A", SET_COLOURS[0]]],
    );
    assert.equal(store.excluded().size, 0);
  });
});

test("a snapshot is frozen and only rows carry exclusions and sets", () => {
  const store = new Selection();
  store.exclude(["1"]);
  store.saveSet("A", ["2"]);
  const rows = store.snapshot("rows");
  const cols = store.snapshot("cols");
  assert.ok(Object.isFrozen(rows));
  assert.equal(rows.excluded.size, 1);
  assert.equal(cols.excluded.size, 0);
  assert.equal(cols.sets.length, 0);
});
