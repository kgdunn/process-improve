// The multivariate page: the building blocks wired together. Open a table, refine it with recipe steps that
// write their own Python, fit PCA or PLS, and work with linked views: a lasso on the score plot selects rows in
// the table, the SPE and T² charts and every header; excluding rows refits and keeps them on every plot, hollow,
// projected through the model that never saw them; and "Why are these rows different?" ranks the variables
// that move the group, with the raw data beside the bars. "Cross-validate" scores every number of components on
// values the model never saw, only when asked, since it fits the model many times over.

import { bindView } from "./link.js";
import { Selection } from "./selection.js";
import { DataTable } from "./table.js";
import {
  barFigure,
  componentsFigure,
  componentsSentence,
  componentTitle,
  scatterFigure,
  sequenceFigure,
} from "./views.js";

/** The Plotly.js that Python's plotly ships, unless the build names another in the app's wheels/manifest.json. */
const PLOTLY_JS = "4.1.1";

const $ = (id) => document.getElementById(id);
const store = new Selection();
const state = {
  table: null,
  recipe: null,
  variant: { kind: "PCA", n_components: 3, y_columns: [] },
  fit: null,
  views: [],
  cv: null,
  cvKey: null,
};
let dataTable = null;
let explanation = null;

// ---------------------------------------------------------------------------
// Python: the worker, or the test's native answers
// ---------------------------------------------------------------------------

let worker = null;
let next = 0;
const waiting = new Map();

function status(text, kind = "busy") {
  $("status").textContent = text;
  $("status").dataset.state = kind;
}

function startWorker() {
  const started = new Worker("worker.js", { type: "module" }); // Pyodide refuses a classic worker
  started.onmessage = ({ data }) => {
    if (data.status) return status(data.status, data.state);
    waiting.get(data.id)?.(data.reply);
    waiting.delete(data.id);
  };
  return started;
}

async function call(fn, payload = {}) {
  const text = window.__mvaNative
    ? await window.__mvaNative(fn, JSON.stringify(payload))
    : await new Promise((resolve) => {
        worker ??= startWorker();
        const id = next++;
        waiting.set(id, resolve);
        worker.postMessage({ id, fn, payload: JSON.stringify(payload) });
      });
  const reply = JSON.parse(text);
  if (!reply.ok) throw new Error(reply.error);
  return reply;
}

/** Run an action, saying what it does, and say what went wrong if it fails. */
async function act(what, action) {
  status(`${what}...`);
  try {
    await action();
    status("Ready", "ok");
  } catch (err) {
    status(String(err.message ?? err), "error");
  }
}

// ---------------------------------------------------------------------------
// Plotly, loaded once
// ---------------------------------------------------------------------------

let plotlyLoading = null;
function plotly() {
  plotlyLoading ??= fetch("../wheels/manifest.json", { cache: "no-store" })
    .then((res) => (res.ok ? res.json() : {}))
    .catch(() => ({}))
    .then(
      ({ plotly_js: version = PLOTLY_JS }) =>
        new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = `https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${version}/plotly.min.js`;
          script.onload = () => resolve(window.Plotly);
          script.onerror = () => reject(new Error(`Plotly.js ${version} could not be loaded`));
          document.head.append(script);
        }),
    );
  return plotlyLoading;
}

// ---------------------------------------------------------------------------
// The table and the recipe
// ---------------------------------------------------------------------------

function showTable(reply) {
  // Another file's cross-validation says nothing about this one; a new recipe step only makes it stale.
  if (reply.recipe.source.sha256 !== state.recipe?.source.sha256) forgetCrossValidation();
  state.table = reply;
  state.recipe = reply.recipe;
  const ids = new Set(reply.rowIds);
  store.include([...store.excluded()].filter((id) => !ids.has(id)));
  store.clear();
  dataTable?.destroy();
  dataTable = new DataTable($("table"), {
    rowIds: reply.rowIds,
    columns: reply.columns.map((c) => ({ id: c.id, values: c.values })),
    selection: store,
  });
  $("steps").replaceChildren(
    ...reply.steps.map((text) => Object.assign(document.createElement("li"), { textContent: text })),
  );
  $("code").textContent = reply.code;
  $("undo").disabled = !reply.recipe.steps.length;
  $("save-project").disabled = false;
}

async function addStep(step) {
  const recipe = { ...state.recipe, steps: [...state.recipe.steps, step] };
  showTable(await call("api_apply", { recipe }));
  await refit();
}

// ---------------------------------------------------------------------------
// The model and its linked views
// ---------------------------------------------------------------------------

function variant() {
  const numeric = state.table.columns.filter((c) => c.numeric).map((c) => c.id);
  const y = state.variant.y_columns.filter((c) => numeric.includes(c));
  return {
    kind: state.variant.kind,
    n_components: state.variant.n_components,
    x_columns: numeric.filter((c) => !y.includes(c)),
    y_columns: y,
    excluded: [...store.excluded()],
  };
}

async function refit() {
  if (!state.table) return;
  state.fit = await call("api_fit", { recipe: state.recipe, variant: variant() });
  const fit = state.fit;
  $("code").textContent = fit.code + (state.cv && !cvStale() ? `\n${state.cv.code}` : "");
  $("cross-validate").disabled = false;
  $("model-summary").textContent =
    `${fit.rowIds.length - fit.excluded.length} rows fitted, ${fit.excluded.length} excluded; R²X ${fit.r2x.map((r) => (100 * r).toFixed(1)).join(" + ")}%`;
  if (!state.views.length) await bindViews();
  else state.views.forEach((view) => view.redraw());
  await drawComponents();
}

const limitsOf = (name) => [
  { value: state.fit.limits["0.95"][name], label: "95%" },
  { value: state.fit.limits["0.99"][name], label: "99%" },
];

async function bindViews() {
  const Plotly = await plotly();
  const fit = () => state.fit;
  // A point is hollow when the fit on screen projected it, not the moment the store excludes it: until the
  // refit arrives the plots still show the old fit, and its points were all fitted.
  const fitted = (s) => ({ ...s, excluded: new Set(fit().excluded) });
  state.views = [
    bindView(
      $("scores"),
      store,
      "rows",
      (s) =>
        scatterFigure(
          fit().rowIds.map((id, i) => ({ id, x: fit().scores[0][i], y: fit().scores[1]?.[i] ?? 0, label: id })),
          fitted(s),
          {
            xTitle: componentTitle("t", 1, fit().r2x[0]),
            yTitle: componentTitle("t", 2, fit().r2x[1]),
            ellipse: fit().ellipse.x ? { ...fit().ellipse, name: "T² limit, 95%" } : null,
          },
        ),
      { Plotly },
    ),
    bindView(
      $("loadings"),
      store,
      "cols",
      (s) =>
        scatterFigure(
          fit().variables.map((id, k) => ({ id, x: fit().loadings[0][k], y: fit().loadings[1]?.[k] ?? 0 })),
          s,
          { xTitle: componentTitle("p", 1, fit().r2x[0]), yTitle: componentTitle("p", 2, fit().r2x[1]), labels: "all" },
        ),
      { Plotly },
    ),
    bindView(
      $("spe"),
      store,
      "rows",
      (s) =>
        sequenceFigure([{ name: "SPE", ids: fit().rowIds, y: fit().SPE }], fitted(s), {
          limits: limitsOf("SPE"),
          yTitle: "SPE",
        }),
      { Plotly },
    ),
    bindView(
      $("t2"),
      store,
      "rows",
      (s) =>
        sequenceFigure([{ name: "T²", ids: fit().rowIds, y: fit().T2 }], fitted(s), {
          limits: limitsOf("T2"),
          yTitle: "T²",
        }),
      { Plotly },
    ),
  ];
  await Promise.all(state.views.map((view) => view.ready));
}

// ---------------------------------------------------------------------------
// How many components? On request only
// ---------------------------------------------------------------------------

/** What a cross-validation depends on: the table and the model, but not the model's number of components. */
const cvKey = () => JSON.stringify({ recipe: state.recipe, variant: { ...variant(), n_components: 1 } });
const cvStale = () => state.cvKey !== cvKey();

function forgetCrossValidation() {
  state.cv = null;
  state.cvKey = null;
  $("cv-panel").hidden = true;
}

async function crossValidate() {
  const key = cvKey();
  state.cv = await call("api_cross_validate", { recipe: state.recipe, variant: variant() });
  state.cvKey = key;
  $("cv-panel").hidden = false;
  $("code").textContent = `${state.fit.code}\n${state.cv.code}`;
  await drawComponents();
}

async function drawComponents() {
  if (!state.cv) return;
  const Plotly = await plotly();
  const stale = cvStale();
  const figure = componentsFigure(state.cv, { current: state.variant.n_components, stale });
  $("cv-sentence").textContent =
    (stale ? "The model has changed since this cross-validation: cross-validate again. " : "") +
    componentsSentence(state.cv, state.variant.n_components);
  $("cv-notes").textContent = state.cv.notes.length ? ` Some fits warned: ${state.cv.notes.join(" ")}` : "";
  const div = $("cv");
  await Plotly.react(div, figure.data, figure.layout, { displaylogo: false, responsive: true });
  if (!div.dataset.bound) {
    div.dataset.bound = "yes";
    div.on("plotly_click", ({ points }) => points[0] && useComponents(points[0].x));
  }
}

function useComponents(n) {
  state.variant.n_components = Math.max(1, Math.min(20, Number(n) || 1));
  $("components").value = state.variant.n_components;
  act("Refitting", refit);
}

// ---------------------------------------------------------------------------
// Why are these rows different?
// ---------------------------------------------------------------------------

async function explain() {
  const group = [...store.selected("rows")];
  const reply = await call("api_explain", {
    recipe: state.recipe,
    variant: variant(),
    group,
    components: [1, 2],
    top: 3,
  });
  explanation?.forEach((view) => view.destroy());
  const ranked = Object.entries(reply.scoreDifference).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const component = ranked[0][0];
  const Plotly = await plotly();
  $("explanation").hidden = false;
  $("explanation-sentence").textContent = sentence(reply, group.length);
  const items = reply.variables.map((id, k) => ({ id, value: reply.contributions[component][k] }));
  const views = [
    bindView(
      $("contributions"),
      store,
      "cols",
      (s) =>
        barFigure(items, s, { sort: true, yTitle: `Contribution to the shift in ${component.replace("PC", "t")}` }),
      { Plotly },
    ),
  ];
  const summary = reply.summary;
  const rows = reply.top.map((name) => {
    const k = reply.variables.indexOf(name);
    const cells = [
      name,
      summary["group mean"][k],
      summary["reference mean"][k],
      summary["difference in reference SDs"][k],
    ];
    return `<tr>${cells.map((c, i) => `<td>${i ? Number(c).toPrecision(4) : escapeHtml(c)}</td>`).join("")}</tr>`;
  });
  $("summary").innerHTML =
    "<tr><th>Variable</th><th>Group mean</th><th>Rest mean</th><th>Difference, in SDs of the rest</th></tr>" +
    rows.join("");
  // Every container first, then the charts: a chart drawn while it is the grid's only child takes the whole width.
  const group_ = new Set(reply.evidence.group);
  const boxes = reply.evidence.series.map((series) => {
    const div = document.createElement("div");
    div.setAttribute("role", "img");
    div.setAttribute("aria-label", `${series.name} in run order, the selected rows marked`);
    return div;
  });
  $("evidence").replaceChildren(...boxes);
  reply.evidence.series.forEach((series, k) => {
    views.push(
      bindView(
        boxes[k],
        store,
        "rows",
        (s) =>
          sequenceFigure(
            [{ name: series.name, ids: reply.evidence.rowIds, y: series.values }],
            { ...s, excluded: new Set(state.fit.excluded) },
            { group: group_, yTitle: series.name },
          ),
        { Plotly },
      ),
    );
  });
  explanation = views;
  await Promise.all(views.map((view) => view.ready));
}

function sentence(reply, n) {
  const moves = Object.entries(reply.scoreDifference)
    .map(([pc, d]) => `${Math.abs(d).toFixed(2)} ${d < 0 ? "lower" : "higher"} on ${pc.replace("PC", "t")}`)
    .join(" and ");
  const top = reply.top[0];
  const k = reply.variables.indexOf(top);
  const sds = reply.summary["difference in reference SDs"][k];
  return (
    `The ${n} selected rows sit ${moves} than the other ${reply.reference.length} fitted rows, on average. ` +
    `${top} contributes most: its average is ${Math.abs(sds).toFixed(1)} standard deviations ${sds < 0 ? "below" : "above"} the rest's.`
  );
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

const base64Of = (buffer) => {
  const bytes = new Uint8Array(buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
};

async function openFile(file) {
  const buffer = await file.arrayBuffer();
  const options = {};
  if (!/\.xlsx?$/i.test(file.name)) {
    // A CSV whose first header is empty, as pandas writes an index, has its row ids in the first column.
    const first = new TextDecoder().decode(buffer.slice(0, 4096)).split(/\r?\n/, 1)[0];
    if (/^\s*[,;\t]/.test(first)) options.index_col = 0;
    if (first.split(";").length > first.split(",").length) options.sep = ";";
  }
  showTable(await call("api_open", { name: file.name, content: base64Of(buffer), options }));
  await refit();
}

async function saveProject() {
  const reply = await call("api_save", {
    recipe: state.recipe,
    variants: [{ ...variant(), name: "Model 1" }],
    selections: store.toJSON().selections,
  });
  const bytes = Uint8Array.from(atob(reply.content), (c) => c.charCodeAt(0));
  const link = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([bytes], { type: "application/zip" })),
    download: `${state.recipe.source.name.replace(/\.[^.]+$/, "")}.mvproj`,
  });
  link.click();
  URL.revokeObjectURL(link.href);
}

async function openProject(file) {
  const reply = await call("api_load", { content: base64Of(await file.arrayBuffer()) });
  showTable(reply);
  const [first] = reply.variants;
  if (first) {
    state.variant = { kind: first.kind, n_components: first.n_components, y_columns: first.y_columns };
    $("kind").value = first.kind;
    $("components").value = first.n_components;
    store.exclude(first.excluded);
  }
  for (const [name, rows] of Object.entries(reply.selections)) store.saveSet(name, rows);
  await refit();
  if (!reply.reproduces)
    throw new Error(`Opened, but the fits stored with it differ from the fits made now (written by ${reply.library}).`);
}

// ---------------------------------------------------------------------------
// The toolbar follows the store
// ---------------------------------------------------------------------------

store.subscribe((changes) => {
  const rows = store.selected("rows");
  const cols = store.selected("cols");
  $("selection-count").textContent =
    rows.size || cols.size ? `${rows.size} rows and ${cols.size} columns selected` : "Nothing selected";
  const anyExcluded = [...rows].some((id) => store.isExcluded(id));
  $("exclude").disabled = !rows.size || [...rows].every((id) => store.isExcluded(id));
  $("include").disabled = !anyExcluded;
  $("save-set").disabled = !rows.size;
  $("explain").disabled = !rows.size || !state.fit;
  $("drop-rows").disabled = !rows.size;
  $("drop-columns").disabled = !cols.size;
  $("use-y").disabled = !cols.size;
  if (changes.has("sets")) drawSets();
  if (changes.has("excluded") && state.fit) act("Refitting", refit);
});

function drawSets() {
  $("sets").replaceChildren(
    ...store.sets().map(({ name, rows, colour }) => {
      const button = document.createElement("button");
      button.type = "button";
      const swatch = Object.assign(document.createElement("span"), { className: "swatch" });
      swatch.style.background = colour;
      button.append(swatch, `${name} (${rows.size})`);
      button.addEventListener("click", () => store.selectSet(name));
      return button;
    }),
  );
}

// ---------------------------------------------------------------------------
// The page's controls
// ---------------------------------------------------------------------------

$("open-example").addEventListener("click", () =>
  act("Opening the LDPE example", async () => {
    showTable(await call("api_open", { example: "ldpe" }));
    await refit();
  }),
);
$("open-file").addEventListener(
  "change",
  (e) => e.target.files[0] && act("Opening the file", () => openFile(e.target.files[0])),
);
$("open-project").addEventListener(
  "change",
  (e) => e.target.files[0] && act("Opening the project", () => openProject(e.target.files[0])),
);
$("save-project").addEventListener("click", () => act("Saving the project", saveProject));
$("exclude").addEventListener("click", () => store.exclude(store.selected("rows")));
$("include").addEventListener("click", () => store.include(store.selected("rows")));
$("save-set").addEventListener("click", () => {
  const name = prompt("Name this set of rows", `Set ${store.sets().length + 1}`);
  if (name) store.saveSet(name);
});
$("explain").addEventListener("click", () => act("Explaining the selection", explain));
$("drop-rows").addEventListener("click", () => {
  const reason = prompt("Why are these rows dropped? The reason is kept with the recipe.", "");
  if (reason === null) return;
  act("Dropping rows", () => addStep({ kind: "drop_rows", rows: rowLabels(store.selected("rows")), reason }));
});
$("drop-columns").addEventListener("click", () =>
  act("Dropping columns", () => addStep({ kind: "drop_columns", columns: [...store.selected("cols")] })),
);
$("use-y").addEventListener("click", () => {
  state.variant.y_columns = [...store.selected("cols")];
  store.clear("cols");
  act("Refitting", refit);
});
$("kind").addEventListener("change", (e) => {
  state.variant.kind = e.target.value;
  act("Refitting", refit);
});
$("components").addEventListener("change", (e) => useComponents(e.target.value));
$("cross-validate").addEventListener("click", () => act("Cross-validating every number of components", crossValidate));
$("add-filter").addEventListener("click", () =>
  act("Filtering rows", () => addStep({ kind: "filter_rows", expression: $("filter").value })),
);
$("add-column").addEventListener("click", () =>
  act("Adding the column", () =>
    addStep({ kind: "add_column", name: $("new-name").value, expression: $("new-expression").value }),
  ),
);
$("undo").addEventListener("click", () =>
  act("Undoing the last step", async () => {
    showTable(await call("api_apply", { recipe: { ...state.recipe, steps: state.recipe.steps.slice(0, -1) } }));
    await refit();
  }),
);

/** The table's own row labels for ids the store holds as text, as the recipe must name them. */
const rowLabels = (ids) => {
  const labels = new Map(state.table.rowIds.map((id, i) => [id, state.table.rowLabels[i]]));
  return [...ids].map((id) => labels.get(id));
};

window.__mva = { store, state, call };
status(window.__mvaNative ? "Ready" : "Starting Python...", window.__mvaNative ? "ok" : "busy");
if (!window.__mvaNative)
  call("api_version").then(
    ({ version }) => status(`Ready: process-improve ${version}`, "ok"),
    () => {},
  );
