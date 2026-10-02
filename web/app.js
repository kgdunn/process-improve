// The page: builds the form, talks to the Python worker, renders results.
// All computation happens in worker.js; this file only moves JSON around.

const $ = (sel) => document.querySelector(sel);
const worker = new Worker("worker.js");

// ---------------------------------------------------------------- worker RPC
let nextId = 0;
const pending = new Map();

worker.onmessage = ({ data }) => {
  if (data.status) {
    setStatus(data.status, data.failed);
    return;
  }
  const resolve = pending.get(data.id);
  pending.delete(data.id);
  resolve(JSON.parse(data.reply));
};

/** Call a Python ``api_*`` function; resolves to its decoded {ok, result|error}. */
function call(fn, payload = {}) {
  const id = nextId++;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker.postMessage({ id, fn, payload: JSON.stringify(payload) });
  });
}

function setStatus(text, failed = false) {
  const el = $("#status");
  el.textContent = text;
  el.classList.toggle("error", failed);
}

// ---------------------------------------------------------------- factor table
const DEFAULT_FACTORS = [
  { name: "T", low: 150, high: 200, units: "degC" },
  { name: "P", low: 1, high: 3, units: "bar" },
  { name: "F", low: 10, high: 20, units: "L/min" },
];

function addFactor(f = { name: "", low: -1, high: 1, units: "" }) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input name="name" aria-label="Factor name" size="8"></td>
    <td><input name="low" type="number" step="any" aria-label="Low"></td>
    <td><input name="high" type="number" step="any" aria-label="High"></td>
    <td><input name="units" aria-label="Units" size="6"></td>
    <td><button type="button" class="link" aria-label="Remove factor">✕</button></td>`;
  for (const key of ["name", "low", "high", "units"]) tr.querySelector(`[name=${key}]`).value = f[key];
  tr.querySelector("button").onclick = () => tr.remove();
  $("#factors tbody").append(tr);
}

function readFactors() {
  return [...document.querySelectorAll("#factors tbody tr")].map((tr) => ({
    name: tr.querySelector("[name=name]").value.trim(),
    low: Number(tr.querySelector("[name=low]").value),
    high: Number(tr.querySelector("[name=high]").value),
    units: tr.querySelector("[name=units]").value.trim(),
  }));
}

// ---------------------------------------------------------------- design step
// The design list, its hints and which settings each design takes are in the
// HTML, so the form is right from the first paint, before Python has loaded.
function showOptions() {
  const option = $("#design-type").selectedOptions[0];
  const allowed = option.dataset.options.split(" ");
  $("#design-hint").textContent = option.dataset.hint;
  for (const el of document.querySelectorAll("[data-option]")) {
    el.hidden = !allowed.includes(el.dataset.option);
  }
}

async function generate() {
  const button = $("#generate");
  button.disabled = true;
  setStatus("Generating design...");
  const spec = {
    design_type: $("#design-type").value,
    factors: readFactors(),
    response: $("#response").value.trim(),
    n_center_points: Number($("#n_center_points").value),
    alpha: $("#alpha").value,
    budget: Number($("#budget").value),
    model_type: $("#model_type").value,
  };
  const reply = await call("api_make_design", spec);
  button.disabled = false;
  if (!reply.ok) {
    setStatus(reply.error, true);
    return;
  }
  const r = reply.result;
  setStatus(`${r.n_runs} runs generated.`);
  const warn = r.outside_range.length
    ? ` Some runs of ${r.outside_range.join(", ")} lie outside the ranges you entered; check they are safe to run.`
    : "";
  $("#design-summary").textContent =
    `${r.n_runs} runs, listed in randomised run order. The workbook will be analysed with the ` +
    `${r.model.replace("_", " ")} model unless you choose another.${warn}`;
  renderTable($("#design-table"), r.columns, r.rows, "Run order");
  const blob = new Blob([Uint8Array.from(atob(r.xlsx_base64), (c) => c.charCodeAt(0))], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const link = $("#download");
  URL.revokeObjectURL(link.href);
  link.href = URL.createObjectURL(blob);
  link.download = `${spec.design_type}-${r.n_runs}-runs.xlsx`;
  $("#design-out").hidden = false;
}

// ---------------------------------------------------------------- analyse step
let lastWorkbook = null;

async function analyse() {
  if (!lastWorkbook) return;
  setStatus("Analysing...");
  const reply = await call("api_analyze", { xlsx_base64: lastWorkbook, model: $("#model").value });
  const out = $("#analysis");
  out.hidden = false;
  if (!reply.ok) {
    setStatus("The workbook could not be analysed.", true);
    out.replaceChildren(el("p", "error", reply.error));
    return;
  }
  const r = reply.result;
  const s = r.summary;
  setStatus("Analysis complete.");
  const parts = [
    el("p", "equation", r.equation),
    el("p", "hint", "Coefficients are in coded units: each factor runs from -1 at its low value to +1 at its high value."),
    el(
      "p",
      "",
      `${r.n_completed} completed runs. R² = ${fmt(s.r_squared)}, adjusted R² = ${fmt(s.r_squared_adj)}, ` +
        `predicted R² = ${fmt(s.r_squared_pred)}, residual degrees of freedom = ${s.df_residual}.` +
        (r.pending_runs.length ? ` Still to run: ${r.pending_runs.join(", ")}.` : ""),
    ),
    el("h3", "", "Coefficients"),
    table(
      ["Term", "Coefficient", "Std. error", "95% interval", "p-value"],
      r.coefficients.map((c) => [
        c.term, fmt(c.coefficient), fmt(c.std_error), `${fmt(c.ci_low)} to ${fmt(c.ci_high)}`, pval(c.p_value),
      ]),
    ),
    el("h3", "", "ANOVA"),
    table(
      ["Source", "df", "Sum of squares", "F", "p-value"],
      r.anova.map((a) => [a.source, a.df, fmt(a.sum_sq), fmt(a.F), pval(a.p_value)]),
    ),
  ];
  const lof = r.lack_of_fit;
  if (lof && lof.p_value !== null && lof.p_value !== undefined) {
    parts.push(
      el(
        "p",
        "",
        `Lack of fit: F = ${fmt(lof.f_statistic)}, p = ${pval(lof.p_value)} ` +
          `(pure error from the replicated runs: ${lof.df_pure_error} degree${lof.df_pure_error === 1 ? "" : "s"} of freedom).`,
      ),
    );
  }
  out.replaceChildren(...parts);
}

async function readUpload(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  lastWorkbook = btoa(binary);
  await analyse();
}

// ---------------------------------------------------------------- rendering
const fmt = (x) => (x === null || x === undefined ? "" : Number(x).toPrecision(4));
const pval = (p) => (p === null || p === undefined ? "" : p < 0.0001 ? "< 0.0001" : Number(p).toFixed(4));

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  node.textContent = text;
  return node;
}

function table(columns, rows) {
  const wrap = el("div", "scroll", "");
  const t = el("table", "data", "");
  renderTable(t, columns, rows);
  wrap.append(t);
  return wrap;
}

/**
 * Render a table whose column headers sort it: click once for ascending,
 * again for descending. Each header is a <button> inside the <th>, so it is
 * reachable by keyboard, and aria-sort tells a screen reader the current order.
 * `sortedBy` names the column the rows already arrive sorted on, if any.
 */
function renderTable(t, columns, rows, sortedBy = null) {
  const head = columns
    .map((c, i) => `<th aria-sort="${c === sortedBy ? "ascending" : "none"}"><button type="button" class="sort" data-col="${i}">${escape(c)}</button></th>`)
    .join("");
  t.innerHTML = `<thead><tr>${head}</tr></thead><tbody></tbody>`;
  t._rows = rows;
  fillBody(t, rows);
  t.onclick = (e) => {
    const button = e.target.closest("button.sort");
    if (!button) return;
    const th = button.parentElement;
    const ascending = th.getAttribute("aria-sort") !== "ascending";
    for (const other of t.querySelectorAll("th")) other.setAttribute("aria-sort", "none");
    th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    const col = Number(button.dataset.col);
    const sign = ascending ? 1 : -1;
    // A stable sort over a copy, so the order the rows arrived in breaks ties.
    fillBody(
      t,
      [...t._rows].sort((a, b) => isBlank(a[col]) - isBlank(b[col]) || sign * compareCells(a[col], b[col])),
    );
  };
}

function fillBody(t, rows) {
  t.tBodies[0].innerHTML = rows
    .map((row) => `<tr>${row.map((v) => `<td>${escape(typeof v === "number" ? +v.toPrecision(6) : v ?? "")}</td>`).join("")}</tr>`)
    .join("");
}

/** Blank cells (an unfilled response) sort last in either direction. */
const isBlank = (v) => v === null || v === undefined || v === "";

/**
 * Order two non-blank cells: numbers numerically, including text that leads
 * with one ("< 0.0001", "4.623 to 5.300"), then everything else alphabetically.
 */
function compareCells(a, b) {
  const na = sortNumber(a);
  const nb = sortNumber(b);
  if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

const sortNumber = (v) => (typeof v === "number" ? v : parseFloat(String(v).replace(/^[<>≤≥]\s*/, "")));

const escape = (v) => String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------- wiring
DEFAULT_FACTORS.forEach(addFactor);
$("#add-factor").onclick = () => addFactor();
$("#generate").onclick = generate;
$("#design-type").onchange = showOptions;
$("#model").onchange = analyse;
$("#upload").onchange = (e) => e.target.files[0] && readUpload(e.target.files[0]);
const drop = $("#drop");
drop.ondragover = (e) => {
  e.preventDefault();
  drop.classList.add("over");
};
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) readUpload(e.dataTransfer.files[0]);
};

showOptions();

// Python is ready once the catalogue answers. Any design the loaded package
// does not offer is dropped from the list rather than failing on Generate.
call("api_catalogue").then((reply) => {
  if (!reply.ok) return;
  const select = $("#design-type");
  for (const option of [...select.options]) {
    if (!(option.value in reply.result.designs)) option.remove();
  }
  showOptions();
  $("#generate").disabled = false;
  $("#upload").disabled = false;
});
