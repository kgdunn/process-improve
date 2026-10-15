// Drive the in-browser designed-experiments app through its worked example and capture one frame per phase.
//
//   node tools/branding/app_demo.js OUT_DIR
//
// Writes demo-1-plan.png, demo-2-compare.png, demo-3-run.png and demo-4-learn.png to OUT_DIR. It runs the
// live app at https://factori.al/, or the one APP_URL names (a local build, before a change is deployed).
// Needs Playwright. PLAYWRIGHT_PROXY, if set, routes the browser through that proxy; the local server is
// reached directly.
const path = require("path");
const { chromium } = require("playwright");

const APP = process.env.APP_URL || "https://factori.al/";
const out = process.argv[2] || ".";
const args = process.env.PLAYWRIGHT_PROXY
  ? [`--proxy-server=${process.env.PLAYWRIGHT_PROXY}`, "--proxy-bypass-list=127.0.0.1;localhost"]
  : [];

/** Show a phase, scroll its workspace to the top of the window, below the sticky study bar, and capture the window. */
async function capture(page, phase, name) {
  await page.click(`#go-${phase}`);
  await page.evaluate(() => {
    const top = document.querySelector("#workspace").getBoundingClientRect().top + window.scrollY;
    window.scrollTo(0, top - document.querySelector("#studybar").offsetHeight - 8);
  });
  await page.mouse.move(0, 0); // so no chart shows its hover toolbar
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(out, name) });
}

(async () => {
  const browser = await chromium.launch({ args });
  // Wide enough for the brief's column beside the phase on show.
  const page = await browser.newPage({ viewport: { width: 1180, height: 860 }, deviceScaleFactor: 2 });
  await page.goto(`${APP}?example=worked`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.querySelector("#generate").disabled, null, { timeout: 400000 });

  // The worked example, live: the reactor's yield, the designs that can learn its curvature, a Box-Behnken design
  // simulated from a known process, and its analysis.
  await page.click("#run-example");
  await page.waitForFunction(
    () =>
      document.querySelector("#demo-when").textContent.includes("in this tab") &&
      document.querySelector("#status .status__text").textContent === "Analysis complete." &&
      document.querySelectorAll("#charts .main-svg").length >= 4,
    null,
    { timeout: 600000 },
  );
  // The status line fades a few seconds after a success, so the frames show the study alone.
  await page.waitForFunction(() => document.querySelector("#status").hidden, null, { timeout: 20000 });
  await capture(page, "plan", "demo-1-plan.png");

  await page.click("#compare");
  await page.waitForFunction(() => document.querySelectorAll("#compare-charts .main-svg").length > 0, null, { timeout: 600000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(out, "demo-2-compare.png") });
  await page.click("#compare-close");

  await page.click("#go-run");
  await page.click("#truth summary");
  await capture(page, "run", "demo-3-run.png");
  await capture(page, "learn", "demo-4-learn.png");
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
