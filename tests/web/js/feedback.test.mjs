// web/feedback.js: the report the Feedback dialog sends, what it may hold, and where it may go.

import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  CONTEXT,
  MAX_PIXELS,
  MAX_SCREENSHOT,
  MAX_STUDY,
  MESSAGE_MAX,
  RELAY,
  TOPICS,
  buildReport,
  characters,
  contextLines,
  cut,
  decodedSize,
  emailProblem,
  fitContext,
  keepError,
  messageCount,
  messageProblem,
  relayUrl,
  screenshotProblem,
  screenshotScale,
  sentLines,
  studyProblem,
} from "../../../web/feedback.js";

const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, ...Array(100).fill(0), 0xff, 0xd9]);
const SHOT = `data:image/jpeg;base64,${JPEG.toString("base64")}`;
const CONTEXT_SENT = {
  commit: "9f3827d1c0ffee",
  plotly_js: "3.1.0",
  tab: "Learn",
  model: "As designed",
  charts: ["Coefficients", "Response surface"],
  stamp: ["Made: 2026-10-18 21:04 UTC", "Inputs, SHA-256: 5e2a"],
  errors: ["api_analyze_runs: Python stopped with LinAlgError: Singular matrix"],
  browser: "Mozilla/5.0 (X11; Linux x86_64)",
  window: "1280 x 900",
};
const MESSAGE = "The contour plot shows a ridge, and the summary calls it a maximum.";

describe("the report", () => {
  test("holds the topic, the message and the context, and only what the reader chose to add", () => {
    const report = buildReport({ topic: "algorithm", message: MESSAGE, context: CONTEXT_SENT });
    assert.deepEqual(Object.keys(report), ["topic", "message", "context"]);
    assert.deepEqual(report.context, CONTEXT_SENT);
    const whole = buildReport({ topic: "general", message: MESSAGE, email: " reader@example.com ", screenshot: SHOT, study: "jZJN" });
    assert.deepEqual(Object.keys(whole), ["topic", "message", "email", "screenshot", "study", "context"]);
    assert.equal(whole.email, "reader@example.com", "the address without the spaces around it");
  });

  test("keeps the message's line breaks, and drops a control character the relay would refuse", () => {
    const { message } = buildReport({ topic: "general", message: "Line one\r\nline two\tand\u0000 three\u000c." });
    assert.equal(message, "Line one\r\nline two\tand three.");
    assert.equal(buildReport({ topic: "general", message: "half a pair \ud83d." }).message, "half a pair �.");
  });

  test("names each topic as the relay does", () => {
    assert.deepEqual(TOPICS, { algorithm: "Algorithm or model", interface: "Interface suggestion", general: "General" });
  });
});

describe("the message", () => {
  test("has at least ten characters once trimmed, and at most five thousand", () => {
    assert.equal(messageProblem("Ten chars!"), "");
    assert.equal(messageProblem("    short    "), "Write at least 10 characters.");
    assert.equal(messageProblem("x".repeat(MESSAGE_MAX)), "");
    assert.equal(messageProblem("x".repeat(MESSAGE_MAX + 1)), "Write at most 5000 characters.");
  });

  test("counts an emoji once, as the relay does", () => {
    const smiles = "😀".repeat(10);
    assert.equal(smiles.length, 20, "JavaScript counts its halves");
    assert.equal(characters(smiles), 10);
    assert.equal(messageProblem(smiles), "");
    assert.equal(messageCount(smiles), "10 of 5000 characters");
    assert.equal(cut("ab😀cd", 3), "ab😀", "never through the middle of a character");
  });
});

describe("the reply address", () => {
  test("is one address, or none", () => {
    for (const address of ["", "   ", "reader@example.com", " first.last+tag@sub.example.co.uk ", "o'brien@example.ie"]) {
      assert.equal(emailProblem(address), "", address);
    }
  });

  test("is refused when it is not one address the relay would take", () => {
    const refused = [
      "reader",
      "reader@localhost",
      "Reader <reader@example.com>",
      "one@example.com, two@example.com",
      "reader@example.com\nBcc: victim@example.com",
      "réader@example.com",
      ".reader@example.com",
      "reader.@example.com",
      "re..ader@example.com",
      "reader@[127.0.0.1]",
      `${"r".repeat(250)}@example.com`,
    ];
    for (const address of refused) {
      assert.equal(emailProblem(address), "Give one email address, such as name@example.com, or leave the box empty.", address);
    }
  });
});

describe("the screenshot", () => {
  test("is a JPEG of at most 2 MB, as a data: URL", () => {
    assert.equal(screenshotProblem(SHOT), "");
    assert.match(screenshotProblem("data:image/png;base64,iVBORw0KGgo="), /no JPEG of the tab/);
    const large = `data:image/jpeg;base64,${"A".repeat(Math.ceil((MAX_SCREENSHOT * 4) / 3) + 4)}`;
    assert.equal(screenshotProblem(large), "The screenshot is larger than 2 MB, so the report goes without it.");
  });

  test("is measured as the bytes its base64 decodes to", () => {
    for (const n of [0, 1, 2, 3, 100, 101, 102]) {
      const base64 = Buffer.alloc(n, 7).toString("base64");
      assert.equal(decodedSize(base64), n, base64);
    }
  });

  test("is taken at scale 1, or smaller to hold about 4 megapixels", () => {
    assert.equal(screenshotScale(1280, 900), 1);
    const scale = screenshotScale(1280, 8000);
    assert.ok(scale < 1);
    assert.ok(Math.abs(1280 * scale * 8000 * scale - MAX_PIXELS) < 1, "the long pane at 4 MP, its shape kept");
  });
});

describe("the study", () => {
  test("goes as a link packs it, if it is not too large", () => {
    assert.equal(studyProblem("jZJNbsIwEIX-v4rU3wLfYxRk1_"), "");
    assert.match(studyProblem("a".repeat(MAX_STUDY + 1)), /too large to go with a report/);
    assert.match(studyProblem("not packed!"), /could not be packed/);
  });
});

describe("the context", () => {
  test("holds only the keys the relay knows, each cut to what it takes", () => {
    const { tab: _tab, ...rest } = CONTEXT_SENT;
    assert.deepEqual(fitContext({ ...CONTEXT_SENT, cookie: "x", tab: "" }), rest, "a key it does not know, and an empty text, left out");
    const fitted = fitContext({ browser: "b".repeat(500), charts: Array(40).fill("Chart"), errors: ["e".repeat(400)] });
    assert.equal(fitted.browser.length, CONTEXT.browser);
    assert.equal(fitted.charts.length, CONTEXT.charts[0]);
    assert.equal(fitted.errors[0].length, CONTEXT.errors[1]);
  });

  test("puts each text on one line, with no control character", () => {
    assert.deepEqual(fitContext({ model: "As designed\n\tor\u0007 not", charts: "Coefficients" }), {
      model: "As designed or not",
      charts: ["Coefficients"],
    });
  });

  test("keeps the last five Python errors, each cut to 300 characters", () => {
    let errors = [];
    for (let n = 1; n <= 7; n += 1) errors = keepError(errors, `api_plots: error ${n}`);
    assert.deepEqual(errors, [3, 4, 5, 6, 7].map((n) => `api_plots: error ${n}`));
    assert.equal(keepError([], `api_plots: ${"x".repeat(400)}\nTraceback`)[0].length, 300);
  });

  test("reads as the relay's email lists it", () => {
    assert.deepEqual(contextLines(CONTEXT_SENT), [
      "App commit: 9f3827d1c0ffee",
      "Plotly.js: 3.1.0",
      "Tab on show: Learn",
      "Model: As designed",
      "Charts on show: Coefficients; Response surface",
      "Browser: Mozilla/5.0 (X11; Linux x86_64)",
      "Window: 1280 x 900",
      "Made: 2026-10-18 21:04 UTC",
      "Inputs, SHA-256: 5e2a",
      "Python error: api_analyze_runs: Python stopped with LinAlgError: Singular matrix",
    ]);
    assert.deepEqual(contextLines({ tab: "Design" }), ["Tab on show: Design", "Last Python errors: none"]);
  });
});

describe("what the dialog says will be sent", () => {
  test("names every part of the report, and what it leaves out", () => {
    const all = sentLines({ topic: "algorithm", message: ` ${MESSAGE} `, email: "reader@example.com", screenshot: SHOT, studied: true });
    assert.deepEqual(all, [
      "Topic: Algorithm or model",
      `Your message: ${MESSAGE.length} characters`,
      "Your email address, to reply to: reader@example.com",
      "A screenshot of this tab: a JPEG of 1 KB",
      "Your study, as a link that opens it with its results",
      "Last Python errors: none",
    ]);
    const none = sentLines({ topic: "general", message: MESSAGE, context: { tab: "Run" } });
    assert.deepEqual(none.slice(2), [
      "No email address: the maintainer cannot reply",
      "No screenshot",
      "Not your study",
      "Tab on show: Run",
      "Last Python errors: none",
    ]);
  });
});

describe("where a report goes", () => {
  test("to the relay from factori.al, and from this machine for the tests", () => {
    for (const address of ["https://factori.al/", "https://factori.al/?example=worked", "http://127.0.0.1:8000/", "http://localhost:5173/", "http://[::1]:8000/"]) {
      assert.equal(relayUrl(new URL(address)), RELAY, address);
    }
    assert.equal(RELAY, "/api/feedback");
  });

  test("nowhere from a copy at any other address, so a reader's screenshot reaches no other server", () => {
    const copies = [
      "http://factori.al/",
      "https://www.factori.al/",
      "https://factori.al.example.com/",
      "https://kgdunn.github.io/process-improve/app/",
      "https://doe.example.com/",
      "file:///home/reader/site/index.html",
    ];
    for (const address of copies) assert.equal(relayUrl(new URL(address)), null, address);
  });
});
