// web/figure.js: a figure in a window of its own. What it works out without a browser: the figure's size in the
// room it has, the name of the file it is saved as, and the file a data URL holds.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { FIRST_SHAPE, SHAPES, SMALLEST, blobOf, fileName, fitted, within } from "../../../web/figure.js";

describe("the shapes a figure can be given", () => {
  test("each is a width over a height, and one fills the room", () => {
    assert.deepEqual(Object.keys(SHAPES), ["Fill", "16:9", "4:3", "1:1"]);
    assert.equal(SHAPES.Fill, null);
    assert.equal(SHAPES["16:9"], 16 / 9);
    assert.ok(FIRST_SHAPE in SHAPES);
  });

  test("the largest box of a shape that fits the room", () => {
    assert.deepEqual(fitted(4 / 3, { width: 900, height: 500 }), { width: 667, height: 500 }, "a wide room: its height");
    assert.deepEqual(fitted(4 / 3, { width: 600, height: 900 }), { width: 600, height: 450 }, "a tall room: its width");
    assert.deepEqual(fitted(1, { width: 600, height: 900 }), { width: 600, height: 600 });
    assert.deepEqual(fitted(null, { width: 600, height: 900 }), { width: 600, height: 900 }, "no shape: all of the room");
  });
});

describe("a size the reader drags to", () => {
  const room = { width: 800, height: 600 };

  test("is kept within the room, and no smaller than a figure can be read at", () => {
    assert.deepEqual(within({ width: 500, height: 400 }, room), { width: 500, height: 400 });
    assert.deepEqual(within({ width: 5000, height: 5000 }, room), room);
    assert.deepEqual(within({ width: 10, height: -40 }, room), SMALLEST);
    assert.deepEqual(within({ width: 500.4, height: 399.6 }, room), { width: 500, height: 400 }, "in whole pixels");
  });

  test("a room smaller than the smallest figure gives the room", () => {
    assert.deepEqual(within({ width: 500, height: 400 }, { width: 200, height: 150 }), { width: 200, height: 150 });
  });
});

describe("the file a figure is saved as", () => {
  test("is named from its title", () => {
    assert.equal(fileName("Each factor's effect on the prediction", "png"), "each-factors-effect-on-the-prediction.png");
    assert.equal(fileName("Fitted response surface, in three dimensions", "svg"), "fitted-response-surface-in-three-dimensions.svg");
    assert.equal(fileName("Correlation map, D-optimal (12 runs)", "png"), "correlation-map-d-optimal-12-runs.png");
    assert.equal(fileName("  ", "png"), "figure.png");
  });

  test("holds what the data URL holds: an image in base64, or a drawing as text", async () => {
    const png = blobOf(`data:image/png;base64,${btoa("\x89PNG")}`);
    assert.equal(png.type, "image/png");
    assert.deepEqual([...new Uint8Array(await png.arrayBuffer())], [0x89, 0x50, 0x4e, 0x47]);
    const svg = blobOf(`data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg">Yield, 50%</svg>')}`);
    assert.equal(svg.type, "image/svg+xml");
    assert.equal(await svg.text(), '<svg xmlns="http://www.w3.org/2000/svg">Yield, 50%</svg>');
  });
});
