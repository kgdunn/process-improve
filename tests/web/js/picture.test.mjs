// web/picture.js: tracing the settings that can be run through where a curved boundary crosses a grid.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { traced } from "../../../web/picture.js";

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
