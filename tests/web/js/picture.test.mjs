// web/picture.js: tracing the settings that can be run through where a curved boundary crosses a grid.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { chains, traced } from "../../../web/picture.js";

/** The area of the polygons, by the shoelace formula; each runs counter-clockwise, so each counts positive. */
const area = (polygons) =>
  polygons.reduce((sum, points) => sum + points.reduce((s, [x1, y1], k) => {
    const [x2, y2] = points[(k + 1) % points.length];
    return s + (x1 * y2 - x2 * y1) / 2;
  }, 0), 0);

describe("tracing the settings that can be run", () => {
  test("a corner cut by the boundary is cut where the boundary crosses the grid", () => {
    // x + y <= 3.5 over [0, 2] x [0, 2] on a grid of 0, 1, 2: only (2, 2) is out, and the boundary
    // crosses the grid's lines halfway from (1, 2) to (2, 2) and from (2, 1) to (2, 2).
    const z = [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 0],
    ];
    const { fill, edge } = traced({ x: [0, 1, 2], y: [0, 1, 2], z, crossings: [[2, 1, 0, 0.5], [1, 2, 1, 0.5]] });
    assert.equal(area(fill), 4 - 0.125, "the square less the triangle cut off its corner");
    assert.deepEqual(edge, [[[2, 1.5], [1.5, 2]]]);
  });

  test("cells wholly inside are drawn as one rectangle a row", () => {
    const z = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ];
    const { fill, edge } = traced({ x: [0, 1, 2, 3], y: [0, 1], z, crossings: [] });
    assert.deepEqual(fill, [[[0, 0], [3, 0], [3, 1], [0, 1]]]);
    assert.deepEqual(edge, []);
  });

  test("a cell whose opposite corners can be run is joined through its middle", () => {
    const z = [
      [1, 0],
      [0, 1],
    ];
    const crossings = [[0, 0, 0, 0.5], [0, 1, 1, 0.5], [1, 0, 0, 0.5], [0, 0, 1, 0.5]];
    const { fill, edge } = traced({ x: [0, 1], y: [0, 1], z, crossings });
    assert.equal(fill.length, 1);
    assert.equal(area(fill), 0.75, "the square less the two corners that cannot be run");
    assert.equal(edge.length, 2);
  });

  test("a crossing the grid does not give is taken halfway along", () => {
    const { fill } = traced({ x: [0, 1], y: [0, 1], z: [[1, 0], [1, 0]], crossings: [] });
    assert.equal(area(fill), 0.5);
  });
});

describe("joining a traced boundary into lines", () => {
  const ranges = [[0, 10], [0, 10]];

  test("pieces that share an end are one line, from one open end to the other, whichever way each runs", () => {
    const pieces = [[[0, 5], [1, 4]], [[2, 3], [1, 4]], [[2, 3], [3, 2.5]]];
    assert.deepEqual(chains(pieces, ...ranges), [[[0, 5], [1, 4], [2, 3], [3, 2.5]]]);
  });

  test("a loop is walked once round", () => {
    const pieces = [[[4, 4], [6, 4]], [[6, 4], [5, 6]], [[5, 6], [4, 4]]];
    assert.deepEqual(chains(pieces, ...ranges), [[[4, 4], [6, 4], [5, 6], [4, 4]]]);
  });

  test("ends the same to a millionth of the range are one end, and a piece of no length is dropped", () => {
    const pieces = [[[0, 5], [1, 4]], [[1, 4], [1, 4]], [[1 + 1e-9, 4], [2, 3]]];
    assert.equal(chains(pieces, ...ranges).length, 1);
  });

  test("a curve traced on a grid comes back as one line across the rectangle", () => {
    // x * y <= 2 on [0, 2] x [0, 2], traced on a grid of 0, 0.5, ..., 2 with exact crossings.
    const x = [0, 0.5, 1, 1.5, 2];
    const y = [0, 0.5, 1, 1.5, 2];
    const z = y.map((yi) => x.map((xj) => Number(xj * yi <= 2)));
    const crossings = [];
    y.forEach((yi, i) => x.slice(0, -1).forEach((xj, j) => {
      if (z[i][j] !== z[i][j + 1]) crossings.push([i, j, 0, (2 / yi - xj) / 0.5]);
    }));
    y.slice(0, -1).forEach((yi, i) => x.forEach((xj, j) => {
      if (z[i][j] !== z[i + 1][j]) crossings.push([i, j, 1, (2 / xj - yi) / 0.5]);
    }));
    const [line, ...more] = chains(traced({ x, y, z, crossings }).edge, [0, 2], [0, 2]);
    assert.deepEqual(more, []);
    for (const [px, py] of line) assert.ok(Math.abs(px * py - 2) < 0.15, `${px}, ${py} lies near the curve`);
    assert.deepEqual([line[0], line.at(-1)].map(([px, py]) => Math.round(px * py * 100) / 100).sort(), [2, 2]);
  });
});
