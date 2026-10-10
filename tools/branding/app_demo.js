// Drive the in-browser designed-experiments app end to end and capture one frame per stage.
//
//   node tools/branding/app_demo.js OUT_DIR
//
// Writes demo-1-design.png, demo-2-runs.png and demo-3-analysis.png to OUT_DIR, using the live app
// at https://kgdunn.github.io/process-improve/app/. Needs Playwright. PYTHON names the interpreter
// that fills in the downloaded workbook (default "python"); PLAYWRIGHT_PROXY, if set, routes the
// browser through that proxy.
const path = require("path");
const { execFileSync } = require("child_process");
const { chromium } = require("playwright");

const APP = "https://kgdunn.github.io/process-improve/app/";
const out = process.argv[2] || ".";
const python = process.env.PYTHON || "python";
const proxy = process.env.PLAYWRIGHT_PROXY ? { server: process.env.PLAYWRIGHT_PROXY, bypass: "localhost,127.0.0.1" } : undefined;

(async () => {
  const browser = await chromium.launch({ proxy });
  const context = await browser.newContext({ viewport: { width: 860, height: 900 }, deviceScaleFactor: 2, acceptDownloads: true });
  const page = await context.newPage();
  await page.goto(APP, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.querySelector("#generate").disabled, null, { timeout: 300000 });
  await page.fill("#response", "Yield");
  await page.screenshot({ path: path.join(out, "demo-1-design.png") });

  await page.click("#generate");
  await page.waitForFunction(() => document.querySelectorAll("#design-table tr").length > 3, null, { timeout: 120000 });
  await page.evaluate(() => document.querySelector("#design-out").scrollIntoView({ block: "start" }));
  await page.screenshot({ path: path.join(out, "demo-2-runs.png") });

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#download")]);
  const blank = path.join(out, "app-run.xlsx");
  const filled = path.join(out, "app-run-filled.xlsx");
  await download.saveAs(blank);
  execFileSync(python, [path.join(__dirname, "fill_workbook.py"), blank, filled], { stdio: "inherit" });

  await page.setInputFiles("#upload", filled);
  await page.waitForFunction(() => document.querySelector("#analysis").innerText.length > 200, null, { timeout: 180000 });
  await page.evaluate(() => document.querySelector("#h-analyse").scrollIntoView({ block: "start" }));
  await page.screenshot({ path: path.join(out, "demo-3-analysis.png") });
  await browser.close();
})();
