// A table of the reader's own, made into a study: a design run elsewhere, or records of a process. The reader
// pastes it or chooses a .csv or .xlsx file; Python reads it and guesses what each column is (api_read_table);
// the reader corrects the guesses here; and Python makes the study of it (api_import_study), which the page opens
// as it opens any other. The table is read in the browser and goes nowhere.
//
// The helpers above installImport need no document, so they are tested under Node.js.

import { nameProblem } from "./logic.js";

/** What a column of the table can be in the study, and how the list names each. */
export const ROLES = [
  ["factor", "A factor"],
  ["response", "A response"],
  ["ignore", "Leave it out"],
];

/** The most responses a study measures: bootstrap.MAX_RESPONSES. */
export const MAX_RESPONSES = 5;

/** The files the page reads a table from, and which of them are workbooks. */
export const TABLE_FILES = ".csv,.tsv,.txt,.xlsx";
export const isWorkbook = (name) => /\.xlsx$/i.test(name);

/** What a column holds, in a few words: "numbers, 150 to 200, at 2 settings" or "labels: A, B". */
export function holdsWords(column) {
  if (column.kind === "number") {
    const range = column.low === column.high ? `${column.low}` : `${column.low} to ${column.high}`;
    return `numbers, ${range}, at ${column.settings} setting${column.settings === 1 ? "" : "s"}`;
  }
  if (column.kind === "text") {
    const more = column.settings > column.levels.length ? ", ..." : "";
    return `labels: ${column.levels.join(", ")}${more}`;
  }
  return "empty";
}

/** A numeric factor's range is typed; a categorical one's levels are its labels. */
export const takesRange = (column) => column.role === "factor" && column.kind === "number";

/**
 * What stops the table from being a study, each in a sentence, in the order the reader would mend them: a study
 * needs a factor and a response, each with a name a model can hold, and a numeric factor a low below its high.
 */
export function importProblems(columns) {
  const factors = columns.filter((c) => c.role === "factor");
  const responses = columns.filter((c) => c.role === "response");
  const problems = [];
  if (!factors.length) problems.push("Choose at least one column as a factor.");
  if (!responses.length) problems.push("Choose at least one column as a response.");
  if (responses.length > MAX_RESPONSES) problems.push(`A study measures at most ${MAX_RESPONSES} responses; ${responses.length} are chosen.`);
  const used = [...factors, ...responses];
  used.forEach((column, at) => {
    const others = used.filter((_, other) => other !== at).map((c) => String(c.name ?? "").trim());
    const said = nameProblem(column.name, others, at + 1).replace(/^Factor \d+ needs a name\.$/, `${column.heading} needs a name.`);
    if (said) problems.push(said.replace("two factors cannot share a name", "two columns of the study cannot share a name"));
  });
  for (const column of responses) {
    if (column.kind !== "number") problems.push(`${column.heading} holds labels, and a response is a number.`);
  }
  for (const column of factors.filter(takesRange)) {
    const typed = (value) => (value === null || value === undefined || value === "" ? NaN : Number(value)); // a blank is no number
    const [low, high] = [typed(column.low), typed(column.high)];
    if (!(low < high)) problems.push(`${column.name || column.heading}: the low value must be below the high value.`);
  }
  return [...new Set(problems)];
}

/** What the page says once the study is open: how many runs it has, and how many of the table's it left out. */
export function openedWords(runs, leftOut) {
  const left = leftOut ? ` ${leftOut} run${leftOut === 1 ? "" : "s"} with no setting for some factor ${leftOut === 1 ? "was" : "were"} left out.` : "";
  return `Opened your table as a study of ${runs} runs, taken as they are.${left} It is saved in this browser, under Your studies.`;
}

// ---------------------------------------------------------------- the dialog
const $ = (selector) => document.querySelector(selector);

function make(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

/** A file's bytes in base64, as Python takes a workbook. */
async function base64Of(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

/**
 * Wire the dialog that makes a study of the reader's own table. ``call`` asks Python, as the page does; ``open``
 * takes what api_import_study returned and opens its study.
 */
export function installImport({ call, open }) {
  const dialog = $("#import-dialog");
  let table = null; // what api_read_table returned, its columns as the reader has left them
  let source = null; // what was read: {text} or {xlsx_base64}, to read another sheet of

  const say = (words, error = false) => {
    $("#import-status").textContent = words;
    $("#import-status").classList.toggle("error", error);
  };

  function showProblems() {
    const problems = table ? importProblems(table.columns) : [];
    $("#import-problems").textContent = problems.join(" ");
    $("#import-problems").hidden = !problems.length;
    $("#import-open").disabled = problems.length > 0;
  }

  /** One line of the list for each column: what it holds, what it is, and its name, units and range in the study. */
  function columnRow(column) {
    const field = (key, label, props = {}) => {
      const input = make("input", { value: column[key] ?? "", ...props });
      input.setAttribute("aria-label", `${label} of ${column.heading}`);
      input.oninput = () => {
        column[key] = input.type === "number" ? (input.value === "" ? null : Number(input.value)) : input.value;
        showProblems();
      };
      return input;
    };
    const role = make("select", {}, ...ROLES.map(([value, words]) => make("option", { value }, words)));
    role.value = column.role;
    role.setAttribute("aria-label", `What ${column.heading} is`);
    const name = field("name", "Name", { type: "text", maxLength: 64 });
    const units = field("units", "Units", { type: "text", maxLength: 32 });
    const low = field("low", "Low value", { type: "number", step: "any" });
    const high = field("high", "High value", { type: "number", step: "any" });
    const range = make("span", { className: "import-range" }, low, " to ", high);
    const show = () => {
      const used = column.role !== "ignore";
      name.hidden = !used;
      units.hidden = !used || column.kind !== "number";
      range.hidden = !takesRange(column);
    };
    role.onchange = () => {
      column.role = role.value;
      show();
      showProblems();
    };
    show();
    const cell = (label, ...parts) => {
      const td = make("td", {}, ...parts);
      td.dataset.label = label;
      return td;
    };
    const row = make(
      "tr",
      {},
      make("th", { scope: "row" }, column.heading, make("span", { className: "hint import-holds" }, holdsWords(column))),
      cell("It is", role),
      cell("Name", name),
      cell("Units", units),
      cell("Range", range),
    );
    row.dataset.column = column.heading;
    return row;
  }

  function showTable(result, title) {
    table = result;
    $("#import-rows").replaceChildren(...table.columns.map(columnRow));
    $("#import-title").value = title ?? $("#import-title").value;
    const sheets = $("#import-sheet");
    sheets.replaceChildren(...table.sheets.map((name) => make("option", { value: name }, name)));
    sheets.value = table.sheet ?? "";
    $("#import-sheet-choice").hidden = table.sheets.length < 2;
    $("#import-columns").hidden = false;
    say(`Read ${table.n_rows} runs in ${table.columns.length} columns. Check what each column is, then open the study.`);
    showProblems();
  }

  async function read(payload, title) {
    source = payload;
    say("Reading the table...");
    $("#import-columns").hidden = true;
    const reply = await call("api_read_table", payload);
    if (!reply.ok) {
      table = null;
      say(reply.error, true);
      return;
    }
    showTable(reply.result, title);
  }

  $("#import-data").onclick = () => {
    if (!dialog.open) dialog.showModal();
  };
  $("#import-close").onclick = () => dialog.close();
  $("#import-file").onchange = async (event) => {
    const [file] = event.target.files;
    event.target.value = ""; // the same file can be chosen again
    if (!file) return;
    const title = file.name.replace(/\.[^.]+$/, "");
    await read(isWorkbook(file.name) ? { xlsx_base64: await base64Of(file) } : { text: await file.text() }, title);
  };
  $("#import-read").onclick = () => read({ text: $("#import-text").value });
  $("#import-sheet").onchange = () => read({ ...source, sheet: $("#import-sheet").value });
  $("#import-open").onclick = async () => {
    if (!table || importProblems(table.columns).length) return;
    say("Making the study...");
    const reply = await call("api_import_study", { columns: table.columns, rows: table.rows, title: $("#import-title").value.trim() });
    if (!reply.ok) {
      say(reply.error, true);
      return;
    }
    dialog.close();
    say("");
    await open(reply.result);
  };
}
