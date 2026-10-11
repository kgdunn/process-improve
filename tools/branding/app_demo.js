// Drive the in-browser designed-experiments app through its worked example and capture one frame per step.
//
//   node tools/branding/app_demo.js OUT_DIR
//
// Writes demo-1-design.png, demo-2-runs.png, demo-3-analysis.png and demo-4-compare.png to OUT_DIR.
// It runs the live app at https://kgdunn.github.io/process-improve/app/, or the one APP_URL names (a
// local build, before a change is deployed). Needs Playwright. PLAYWRIGHT_PROXY, if set, routes the
// browser through that proxy; the local server is reached directly.
const path = require("path");
const { chromium } = require("playwright");

const APP = process.env.APP_URL || "https://kgdunn.github.io/process-improve/app/";
const out = process.argv[2] || ".";
const args = process.env.PLAYWRIGHT_PROXY
  ? [`--proxy-server=${process.env.PLAYWRIGHT_PROXY}`, "--proxy-bypass-list=127.0.0.1;localhost"]
  : [];

const idle = (page) =>
  page.waitForFunction(() => document.querySelector("#status").dataset.state !== "busy", null, { timeout: 600000 });

/** Scroll a step to the top of the window, below the sticky step bar, and capture the window. */
async function capture(page, selector, name) {
  await page.evaluate((sel) => {
    const top = document.querySelector(sel).getBoundingClientRect().top + window.scrollY;
    window.scrollTo(0, top - document.querySelector(".stepper").offsetHeight - 8);
  }, selector);
  await page.mouse.move(0, 0); // so no chart shows its hover toolbar
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(out, name) });
}

(async () => {
  const browser = await chromium.launch({ args });
  const page = await browser.newPage({ viewport: { width: 860, height: 900 }, deviceScaleFactor: 2 });
  await page.goto(APP, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.querySelector("#generate").disabled, null, { timeout: 400000 });

  // The worked example, live: a Box-Behnken design, simulated from a known process, analysed.
  await page.click("#run-example");
  await page.waitForFunction(() => document.querySelector("#demo-banner").hidden, null, { timeout: 60000 });
  await idle(page);
  await page.waitForFunction(() => document.querySelectorAll("#charts .main-svg").length >= 4, null, { timeout: 120000 });
  await capture(page, "#step-design", "demo-1-design.png");

  await page.click("#truth summary");
  await capture(page, "#step-run", "demo-2-runs.png");
  await capture(page, "#charts", "demo-3-analysis.png");

  await page.click("#compare");
  await page.waitForFunction(() => document.querySelectorAll("#compare-charts .main-svg").length > 0, null, { timeout: 600000 });
  await idle(page);
  // On a wide screen step 4 is wider than the other steps: take its frame at a width where every design fits, at
  // the same aspect ratio, so the frames scale to the same size.
  await page.setViewportSize({ width: 1100, height: Math.round((900 * 1100) / 860) });
  await capture(page, "#comparison", "demo-4-compare.png");
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
