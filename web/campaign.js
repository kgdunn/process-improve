// A campaign as the page keeps it: one design, its results, where they came from, and the model
// they are analysed with. The page saves each one in this browser (store.js) and can carry one
// in a link, compressed into the part of the address after "#", which a browser never sends to
// any server. Nothing here touches the document, so tests/web/js runs it under Node.
//
// A campaign read back, from this browser or from a link someone sent, is checked field by
// field before the page uses any of it: a link is text anyone can write.

/**
 * The latest format of a campaign this page reads; one of a later format is refused, not guessed at. Each study is
 * saved in the lowest format that holds it (studyFormat), so a page from before a format opens every study it can.
 * Format 2 holds a study of one factor, which a page of format 1 could not analyse. Format 3 holds a categorical
 * factor that lists its levels: three or more, or two that read as numbers, which an earlier page would misread.
 */
export const CAMPAIGN_VERSION = 3;

/** The format of a study of one factor; a study of more factors keeps format 1. */
export const ONE_FACTOR_FORMAT = 2;

/** The format of a study with a categorical factor that lists its levels; two labels alone keep an earlier format. */
export const LEVELS_FORMAT = 3;

/** The lowest format that holds a study of this specification, which the study is saved in. */
export const studyFormat = (spec) =>
  spec.factors.some((f) => Array.isArray(f.levels)) ? LEVELS_FORMAT : spec.factors.length === 1 ? ONE_FACTOR_FORMAT : 1;

/** Where a link carries its campaign: the address's fragment, after this. */
export const LINK_KEY = "#c=";

/** The most a link may unpack to: a campaign of the largest designs the page makes is far smaller. */
const MAX_BYTES = 2 ** 20;

/** The named models the analysis offers, a mixture's Scheffé models last; "" is the model the design was made for. */
export const MODELS = ["", "main_effects", "interactions", "quadratic", "scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic"];

/** What a study may aim to learn: the models designs are compared on; mirrors process_improve.experiments.compare.MODELS. */
export const AIMS = ["main_effects", "interactions", "main_quadratic", "quadratic"];

/** What the reader can want of the response; mirrors bootstrap.GOALS. */
const GOAL_VALUES = ["maximize", "minimize", "target", "learn"];

const SOURCES = ["simulated", "entered", null];

/** Refuse a campaign, saying what is wrong with it. */
function refuse(what) {
  throw new Error(`This design could not be opened: ${what}.`);
}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const isText = (v, most = 200) => typeof v === "string" && v.length <= most;
const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

function text(value, what, most = 200) {
  if (!isText(value, most)) refuse(`${what} is not text`);
  return value;
}

function list(value, what, most) {
  if (!Array.isArray(value)) refuse(`${what} is not a list`);
  if (value.length > most) refuse(`${what} is too long`);
  return value;
}

/** A level as the page holds it: a number, or a label for a categorical factor. */
function level(value, what) {
  if (isFiniteNumber(value) || isText(value, 64)) return value;
  return refuse(`${what} is neither a number nor a label`);
}

/** The most levels a categorical factor takes, and the longest label; mirror logic.js MAX_LEVELS and LABEL_MOST. */
export const MAX_LEVELS = 8;
export const LABEL_MOST = 64;

/** A categorical factor's levels: two to MAX_LEVELS labels, each once; two of them are its low and high too. */
function levels(f, i) {
  const all = list(f.levels, `factor ${i + 1}'s levels`, MAX_LEVELS).map((v) => text(v, `a level of factor ${i + 1}`, LABEL_MOST));
  if (all.length < 2 || new Set(all).size < all.length) refuse(`factor ${i + 1}'s levels are not two or more labels, each once`);
  return { type: "categorical", levels: all, ...(all.length === 2 ? { low: all[0], high: all[1] } : {}), units: "" };
}

function factor(f, i) {
  if (!isObject(f)) refuse(`factor ${i + 1} is not a factor`);
  const name = text(f.name, `factor ${i + 1}'s name`, 64);
  if (f.type !== undefined || f.levels !== undefined) {
    if (f.type !== "categorical") refuse(`factor ${i + 1}'s type is unknown`);
    return { name, ...levels(f, i) };
  }
  return {
    name,
    low: level(f.low, `factor ${i + 1}'s low level`),
    high: level(f.high, `factor ${i + 1}'s high level`),
    units: f.units === undefined ? "" : text(f.units, `factor ${i + 1}'s units`, 32),
  };
}

/** The settings the design was made with: numbers, short text, or the true and false Python reads back. */
function options(value) {
  if (value === undefined || value === null) return {};
  if (!isObject(value)) refuse("the design's settings are not settings");
  const out = {};
  for (const [key, v] of Object.entries(value)) {
    if (!/^[a-z_]{1,32}$/.test(key)) refuse(`the design's setting ${JSON.stringify(key).slice(0, 40)} is unknown`);
    if (!(isFiniteNumber(v) || isText(v, 64) || typeof v === "boolean")) refuse(`the design's setting ${key} is not a value`);
    out[key] = v;
  }
  return out;
}

/**
 * What a design's specification says of its response besides its column's name: the reader's own
 * name for it, its units, the goal, and the numbers the reader knows of it. A field of the wrong
 * kind is refused; a goal this page does not know is dropped, so a later page can add one.
 */
function responseInfo(value) {
  if (!isObject(value)) refuse("the response's description is not one");
  const out = {};
  if (value.label !== undefined) out.label = text(value.label, "the response's name", 64);
  if (value.units !== undefined) out.units = text(value.units, "the response's units", 32);
  if (GOAL_VALUES.includes(value.goal)) out.goal = value.goal;
  for (const key of ["target", "typical", "noise", "smallest_change"]) {
    if (value[key] === undefined || value[key] === null) continue;
    const positive = key === "noise" || key === "smallest_change";
    if (!isFiniteNumber(value[key]) || (positive && value[key] <= 0)) refuse(`the response's ${key.replace("_", " ")} is not a number`);
    out[key] = value[key];
  }
  return out;
}

/**
 * The most limits a study can have on its settings, and the most characters of a limit and of its note:
 * as bootstrap.py holds them, and logic.js (tests/web/page/test_parity.py keeps the three in step).
 */
export const LIMITS_HELD = { count: 8, expression: 500, note: 80 };

/** A control character, which no limit or note typed on the page holds. */
const CONTROL = /[\u0000-\u001f\u007f]/;

/**
 * The limits on a study's settings, checked for their shape only: Python reads each one when it is used.
 * Each is its text, its type when that is one Python writes, and its note when it has one; anything else
 * is dropped, and a list that is not one of limits is refused.
 */
export function checkConstraints(value) {
  const limits = list(value, "the limits on the settings", LIMITS_HELD.count);
  return limits.map((c, i) => {
    if (!isObject(c)) refuse(`limit ${i + 1} is not a limit`);
    const expression = text(c.expression, `limit ${i + 1}`, LIMITS_HELD.expression);
    if (CONTROL.test(expression)) refuse(`limit ${i + 1} is not one line of text`);
    const out = { expression };
    if (c.type === "linear" || c.type === "nonlinear") out.type = c.type;
    if (c.note !== undefined && c.note !== "") {
      out.note = text(c.note, `limit ${i + 1}'s note`, LIMITS_HELD.note);
      if (CONTROL.test(out.note)) refuse(`limit ${i + 1}'s note is not one line of text`);
    }
    return out;
  });
}

/** The most responses a study measures; mirrors bootstrap.MAX_RESPONSES, as logic.js does. */
export const RESPONSES_HELD = 5;

/**
 * A study's responses, when it lists several: each its column's name and its description, the first being the
 * study's ``response``. A study of one response lists none, as before the page measured several.
 */
function responseList(value, first) {
  const listed = list(value, "the responses", RESPONSES_HELD);
  const out = listed.map((r, i) => {
    if (!isObject(r)) refuse(`response ${i + 1} is not a response`);
    return { name: text(r.name, `response ${i + 1}'s name`, 64), ...responseInfo(Object.fromEntries(Object.entries(r).filter(([k]) => k !== "name"))) };
  });
  if (out.length < 2 || out[0].name !== first) refuse("its responses do not start with its response");
  if (new Set(out.map((r) => r.name)).size !== out.length) refuse("two of its responses share a name");
  return out;
}

/** The names of a checked specification's responses, the first first. */
const responseNames = (s) => (s.responses ? s.responses.map((r) => r.name) : [s.response]);

function spec(value) {
  if (!isObject(value)) refuse("it holds no design");
  const factors = list(value.factors, "the factor list", 12).map(factor);
  if (factors.length < 1) refuse("it has no factors");
  const out = {
    design_type: text(value.design_type, "the design type", 40),
    factors,
    response: text(value.response ?? "y", "the response's name", 64),
    model: text(value.model ?? "", "the design's model", 40),
    options: options(value.options),
  };
  if (value.version !== undefined) out.version = value.version;
  if (value.response_info !== undefined && value.response_info !== null) out.response_info = responseInfo(value.response_info);
  if (value.responses !== undefined && value.responses !== null) {
    out.responses = responseList(value.responses, out.response);
    const { name: _, ...first } = out.responses[0]; // the list describes the first response, as response_info does
    if (Object.keys(first).length) out.response_info = first;
    else delete out.response_info;
  }
  if (AIMS.includes(value.aim)) out.aim = value.aim;
  if (value.constraints !== undefined && value.constraints !== null) {
    const limits = checkConstraints(value.constraints);
    if (limits.length) out.constraints = limits;
  }
  if (value.analysis_transforms !== undefined && value.analysis_transforms !== null) {
    const kept = transforms(value.analysis_transforms, responseNames(out));
    if (Object.keys(kept).length) out.analysis_transforms = kept;
  }
  return out;
}

/** The transforms a study's responses are analysed with, by their columns (keys of bootstrap.TRANSFORMS). */
export const TRANSFORMS = ["log", "sqrt", "inverse"];

/**
 * Each response's transform, as a study keeps it in its design's specification: one of TRANSFORMS, by a response's
 * column. A transform the page does not offer is refused; one of a response the study no longer measures is dropped,
 * as bootstrap._checked_transforms drops it.
 */
function transforms(value, names) {
  if (!isObject(value)) refuse("its responses' transforms are not a list by response");
  const out = {};
  for (const [name, transform] of Object.entries(value)) {
    if (!TRANSFORMS.includes(transform)) refuse(`the transform ${JSON.stringify(transform).slice(0, 40)} is unknown`);
    if (names.includes(name)) out[name] = transform;
  }
  return out;
}

/** The model to analyse with: "" (the design's own), a named model, or {terms}, each a list of factor names. */
export function checkModel(value, factorNames) {
  if (typeof value === "string") {
    if (!MODELS.includes(value)) refuse(`the model ${JSON.stringify(value).slice(0, 40)} is unknown`);
    return value;
  }
  if (!isObject(value)) refuse("the model is neither a name nor a list of terms");
  const terms = list(value.terms, "the model's terms", 200).map((term) => {
    const names = list(term, "a term", 3);
    if (names.length < 1 || !names.every((n) => factorNames.includes(n))) refuse("a term names a factor the design lacks");
    return [...names];
  });
  return { terms };
}

/** What the design is, as Python described it when it was made: its center runs and a sentence per fact. */
function facts(value) {
  if (value === undefined || value === null) return null;
  if (!isObject(value)) refuse("the design's facts are not facts");
  return {
    center_runs: Number.isInteger(value.center_runs) && value.center_runs >= 0 ? value.center_runs : 0,
    notes: list(value.notes ?? [], "the design's notes", 50).map((note) => text(note, "a note on the design", 2000)),
  };
}

/** One response's simulated process: its terms, its noise and seed, its equation and the factors it leaves out. */
function process(value) {
  if (!isObject(value)) refuse("its simulation is not one");
  const truth = list(value.truth, "the true process", 100).map((t) => {
    if (!isObject(t) || !isText(t.term, 100) || !isFiniteNumber(t.coefficient)) refuse("a term of the true process is not one");
    return { term: t.term, coefficient: t.coefficient };
  });
  if (!isFiniteNumber(value.sigma) || !Number.isInteger(value.seed)) refuse("its simulation's noise or seed is not a number");
  return {
    sigma: value.sigma,
    seed: value.seed,
    equation: text(value.equation ?? "", "the true process's equation", 4000),
    inactive: list(value.inactive ?? [], "the inactive factors", 12).map((n) => text(n, "an inactive factor", 64)),
    truth,
  };
}

/**
 * The process the results were simulated from: the first response's, as for a study of one response, and with
 * several, each later response's in ``more``, by its column (``later`` names them).
 */
function simulation(value, later = []) {
  if (value === undefined || value === null) return null;
  const first = process(value);
  const out = {
    scenario: text(value.scenario ?? "", "the simulated process", 40),
    label: text(value.label ?? "", "the simulated process's name", 100),
    sigma: first.sigma,
    seed: first.seed,
    equation: first.equation,
    inactive: first.inactive,
    truth: first.truth,
  };
  if (value.more !== undefined && value.more !== null) {
    if (!isObject(value.more) || !Object.keys(value.more).every((name) => later.includes(name))) refuse("its simulation names a response it does not measure");
    out.more = Object.fromEntries(Object.entries(value.more).map(([name, one]) => [name, process(one)]));
  }
  return out;
}

/**
 * Check a campaign read back from this browser or from a link, and return a copy holding only
 * what the page uses, or throw an Error saying, in a sentence the page can show, what is wrong.
 */
export function checkCampaign(data) {
  if (!isObject(data)) refuse("it is not a design");
  if (!Number.isInteger(data.v) || data.v < 1 || data.v > CAMPAIGN_VERSION) {
    refuse(`it was saved in format ${JSON.stringify(data.v)}, and this page reads formats 1 to ${CAMPAIGN_VERSION}`);
  }
  if (!isObject(data.design)) refuse("it holds no design");
  const design = data.design;
  const s = spec(design.spec);
  const columns = list(design.columns, "the run sheet's columns", 40).map((c) => text(c, "a column's name", 64));
  const names = responseNames(s);
  for (const name of [...s.factors.map((f) => f.name), ...names]) {
    if (!columns.includes(name)) refuse(`the run sheet has no column ${JSON.stringify(name)}`);
  }
  const rows = list(design.rows, "the run sheet", 2000).map((row) => {
    if (list(row, "a run", columns.length).length !== columns.length) refuse("a run does not fill the run sheet's columns");
    return row.map((cell) => (cell === null || isFiniteNumber(cell) || isText(cell, 64) ? cell : refuse("a cell of the run sheet is not a value")));
  });
  if (rows.length < 1) refuse("it has no runs");
  const n = rows.length;
  const [responses, sources] = results(data.responses, data.sources, n);
  const factorNames = s.factors.map((f) => f.name);
  const out = {
    v: studyFormat(s),
    title: data.title === undefined ? "" : text(data.title, "its title", 200),
    design: {
      spec: s,
      columns,
      rows,
      model: text(design.model ?? s.model, "the design's model", 40),
      n_runs: n,
      facts: facts(design.facts),
      outside_range: list(design.outside_range ?? [], "the factors outside their ranges", 12).map((name) =>
        text(name, "a factor outside its range", 64),
      ),
    },
    responses,
    sources,
    simulation: simulation(data.simulation, names.slice(1)),
    simulate: simulateSettings(data.simulate),
    model: checkModel(data.model ?? "", factorNames),
  };
  if (names.length > 1) Object.assign(out, laterResults(data, names.slice(1), n, factorNames, names[0]));
  if (data.source !== undefined && data.source !== null) out.source = studySource(data.source);
  return out;
}

/**
 * Where a study from the design catalog comes from, as the catalog wrote it (build_web_app.study_source): the
 * source's citation, where it prints the runs, that place's web address, the study's id in the catalog and how many
 * of the numbers the source prints Learn shows beside its own. The address must be a web page's and the id a file
 * name's, since the page links to the one and fetches catalog/printed/<id>.json by the other.
 */
function studySource(data) {
  if (!isObject(data)) refuse("its source is not a citation");
  const url = text(data.url ?? "", "its source's address", 500);
  if (url && !/^https?:\/\/[^\s]+$/.test(url)) refuse("its source's address is not a web address");
  const catalog = text(data.catalog ?? "", "its id in the catalog", 120);
  if (catalog && !/^[\w.-]+$/.test(catalog)) refuse("its id in the catalog is not one the catalog gives");
  const printed = data.printed ?? 0;
  if (!Number.isInteger(printed) || printed < 0) refuse("the count of its printed numbers is not a count");
  return { cite: text(data.cite, "its source's citation", 500), where: text(data.where ?? "", "where its source prints it"), url, catalog, printed };
}

/** One response's results and their sources: a number or a blank per run, and where each came from. */
function results(values, marks, n) {
  const responses = values === undefined ? Array(n).fill(null) : list(values, "the results", n);
  if (responses.length !== n || !responses.every((v) => v === null || isFiniteNumber(v))) refuse("its results are not one number, or a blank, per run");
  const sources = marks === undefined ? responses.map((v) => (v === null ? null : "entered")) : list(marks, "the results' sources", n);
  if (sources.length !== n || !sources.every((v) => SOURCES.includes(v))) refuse("its results' sources are not one per run");
  return [[...responses], [...sources]];
}

/**
 * The later responses' results, their sources and the model each is analysed with, by their columns, as a
 * campaign of several responses keeps them beside the first's: ``more_responses``, ``more_sources`` and
 * ``more_models``. A response without results yet has none, and one without a model is analysed as designed. What
 * the reader set of each response's goal in the balance, ``goals``, is kept when there is some.
 */
function laterResults(data, names, n, factorNames, first) {
  const keyed = (value, what) => {
    if (value === undefined || value === null) return {};
    if (!isObject(value) || !Object.keys(value).every((name) => names.includes(name))) refuse(`${what} name a response it does not measure`);
    return value;
  };
  const values = keyed(data.more_responses, "its results");
  const marks = keyed(data.more_sources, "its results' sources");
  const models = keyed(data.more_models, "its models");
  const out = { more_responses: {}, more_sources: {}, more_models: {} };
  for (const name of names) {
    [out.more_responses[name], out.more_sources[name]] = results(values[name], marks[name], n);
    out.more_models[name] = checkModel(models[name] ?? "", factorNames);
  }
  if (data.goals !== undefined && data.goals !== null) out.goals = balanceGoals(data.goals, [first, ...names]);
  return out;
}

/** The shapes a response's desirability can have in the balance; mirrors bootstrap.SHAPES, as best.js does. */
export const GOAL_SHAPES = ["less", "even", "more", "much_more"];

/**
 * What the reader set of each response's goal in the balance, by its column: its acceptable range (``low``,
 * ``high``), its importance from 1 to 5 and its shape. Each is checked for its kind; Python checks they hold together.
 */
function balanceGoals(value, names) {
  if (!isObject(value) || !Object.keys(value).every((name) => names.includes(name))) refuse("its goals name a response it does not measure");
  return Object.fromEntries(
    Object.entries(value).map(([name, goal]) => {
      if (!isObject(goal)) refuse(`the goal of ${name} is not one`);
      const out = {};
      for (const key of ["low", "high"]) {
        if (goal[key] === undefined) continue;
        if (!isFiniteNumber(goal[key])) refuse(`the acceptable range of ${name} is not a number`);
        out[key] = goal[key];
      }
      if (goal.importance !== undefined) {
        if (![1, 2, 3, 4, 5].includes(goal.importance)) refuse(`the importance of ${name} is not 1 to 5`);
        out.importance = goal.importance;
      }
      if (goal.shape !== undefined) {
        if (!GOAL_SHAPES.includes(goal.shape)) refuse(`the shape of ${name}'s desirability is unknown`);
        out.shape = goal.shape;
      }
      return [name, out];
    }),
  );
}

/** How the reader last simulated results: the process, the noise and the seed, each only when it is one. */
function simulateSettings(value) {
  const simulate = isObject(value) ? value : {};
  return {
    scenario: isText(simulate.scenario, 40) ? simulate.scenario : null,
    sigma: isFiniteNumber(simulate.sigma) && simulate.sigma > 0 ? simulate.sigma : null,
    seed: Number.isInteger(simulate.seed) ? simulate.seed : null,
  };
}

// ---------------------------------------------------------------- links
function toBase64Url(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(text) {
  if (!/^[A-Za-z0-9_-]*$/.test(text)) refuse("the link has been cut or changed");
  const binary = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

/** Read a stream to its end, refusing more than ``most`` bytes: a link must not unpack to anything larger. */
async function readAll(stream, most) {
  const reader = stream.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.length;
    if (total > most) {
      await reader.cancel();
      refuse("the link unpacks to more than a design");
    }
    parts.push(value);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

/** Pack a campaign into the text a link carries after LINK_KEY: JSON, deflated, in URL-safe base64. */
export async function packCampaign(campaign) {
  const json = new Blob([JSON.stringify(campaign)]);
  const zipped = await readAll(json.stream().pipeThrough(new CompressionStream("deflate-raw")), MAX_BYTES);
  return toBase64Url(zipped);
}

/** Unpack and check the campaign a link carries; throws an Error with a sentence for the page. */
export async function unpackCampaign(text) {
  let json;
  try {
    const bytes = new Blob([fromBase64Url(text)]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    json = new TextDecoder("utf-8", { fatal: true }).decode(await readAll(bytes, MAX_BYTES));
  } catch (err) {
    if (err.message.startsWith("This design could not be opened")) throw err;
    refuse("the link has been cut or changed");
  }
  let data;
  try {
    data = JSON.parse(json);
  } catch {
    refuse("the link has been cut or changed");
  }
  return checkCampaign(data);
}

/** What a link carries: the design, and its results only if asked; never this browser's own bookkeeping. */
export function shareable(campaign, withResults) {
  const { v, title, design, model } = campaign;
  const { spec, columns, rows, facts, outside_range } = design;
  const out = { v, title, design: { spec, columns, rows, model: design.model, facts, outside_range }, model };
  if (campaign.more_models) out.more_models = campaign.more_models;
  if (campaign.goals) out.goals = campaign.goals;
  if (withResults) {
    Object.assign(out, { responses: campaign.responses, sources: campaign.sources, simulation: campaign.simulation });
    if (campaign.more_responses) Object.assign(out, { more_responses: campaign.more_responses, more_sources: campaign.more_sources });
    out.simulate = campaign.simulate;
  }
  return out;
}

// ---------------------------------------------------------------- the book's examples
/**
 * The query parameter a link names one of the book's examples by: https://factori.al/?book=pollutant.
 * A query rather than a fragment, as a design's link uses: the book's pages read a link's fragment
 * as an anchor of their own, and the name of the book's example is nothing of the reader's.
 */
export const BOOK_PARAM = "book";

/** The query parameter a link names an example of another published source by: https://factori.al/?example=<name>. */
export const EXAMPLE_PARAM = "example";

/** The name ``?example=`` gives the worked example the page was built with; the build lets no other example take it. */
export const WORKED = "worked";

/** A book example's name: lowercase words and numbers joined by hyphens, such as two-level-pollutant. */
const BOOK_NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** Check the name of the book's example a link carries, before anything is fetched for it. */
export function bookName(text) {
  const name = String(text);
  if (name.length > 64 || !BOOK_NAME.test(name)) refuse(`${JSON.stringify(name.slice(0, 64))} is not the name of an example in the book`);
  return name;
}

/**
 * Find the book's example ``name`` in book/index.json, and return the name as the index gives it:
 * the page fetches only a file the index lists, never one a link makes up.
 */
export function indexedName(index, name) {
  const listed = isObject(index) && Array.isArray(index.examples) ? index.examples : [];
  const found = listed.find((example) => isObject(example) && example.name === name);
  if (!found || !BOOK_NAME.test(found.name)) refuse(`the book has no example named ${JSON.stringify(name)}`);
  return found.name;
}

/** A web address an example links to, which must be https. */
function address(value, what) {
  const url = text(value, what, 300);
  if (!url.startsWith("https://")) refuse(`${what} is not a web address`);
  return url;
}

/**
 * Check one of the examples as the build writes book/<name>.json: its name, its title, the page
 * of the book it is on (or, for another published source, its citation, where in the source its
 * runs are printed and the source's address), a note on how the page holds it, and the campaign it opens.
 */
export function checkBookExample(data, name) {
  if (!isObject(data) || data.name !== name) refuse(`the book's example ${JSON.stringify(name)} could not be read`);
  const cited = data.source !== undefined;
  if (cited && !isObject(data.source)) refuse("its source could not be read");
  return {
    name,
    title: text(data.title, "its title"),
    book: cited ? null : address(data.book, "its page in the book"),
    source: cited
      ? {
          cite: text(data.source.cite, "its source", 300),
          where: text(data.source.where, "where its source prints it", 300),
          url: address(data.source.url, "its source"),
        }
      : null,
    note: text(data.note ?? "", "its note", 1000),
    campaign: checkCampaign(data.campaign),
  };
}

/** Whether two campaigns are the same design with the same title, results and model: a book's example as it was left. */
export function sameCampaign(a, b) {
  const held = (c) => JSON.stringify([c.title, c.responses, c.model, c.more_responses ?? null, c.more_models ?? null]);
  return designKey(a.design) === designKey(b.design) && held(a) === held(b);
}

// ---------------------------------------------------------------- words
/**
 * A campaign's name in the list of saved studies: the design and the factors it varies, after the
 * response it is about when the reader described one ("Yield (%), maximise: Box-Behnken in T, P and F").
 */
export function campaignTitle(designLabel, factorNames, response = "") {
  const names = factorNames.length > 1 ? `${factorNames.slice(0, -1).join(", ")} and ${factorNames.at(-1)}` : factorNames.join("");
  return `${response ? `${response}: ` : ""}${designLabel} in ${names}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** When a campaign was saved, as a person says it: "just now", "5 minutes ago", "yesterday", "3 Oct 2026". */
export function timeAgo(then, now = Date.now()) {
  const minutes = Math.floor((now - then) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const day = (t) => {
    const d = new Date(t);
    return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  };
  const days = Math.round((day(now) - day(then)) / 86_400_000);
  if (days <= 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const d = new Date(then);
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}

/**
 * How far a campaign has got: "15 runs, 9 results in", "15 runs, all results in", or "15 runs, no results yet".
 * ``responses`` holds a value per run, null for a run still to do: for a study of several responses, a run is
 * done when every response has its result (see campaignDone).
 */
export function progress(responses) {
  const n = responses.length;
  const done = responses.filter((v) => v !== null).length;
  const runs = `${n} run${n === 1 ? "" : "s"}`;
  if (done === 0) return `${runs}, no results yet`;
  return done === n ? `${runs}, all results in` : `${runs}, ${done} results in`;
}

// ---------------------------------------------------------------- results coming back
/**
 * What makes two campaigns the same design: its type, its factors' levels, its response and its runs,
 * whatever their results. A factor's name and units are labels, which the reader can change on a run
 * sheet already made, so results sent back from a copy shared before a rename still find their design.
 */
export function designKey(design) {
  const { spec, columns, rows } = design;
  const names = responseNames(spec);
  const at = new Set(names.map((name) => columns.indexOf(name)));
  const place = new Map(spec.factors.map((f, i) => [f.name, i]));
  return JSON.stringify([
    spec.design_type,
    spec.factors.map((f) => f.levels ?? [f.low, f.high]),
    names.length > 1 ? names : spec.response, // a study of one response keys as it always has
    columns.map((c) => place.get(c) ?? c),
    rows.map((row) => row.filter((_, i) => !at.has(i))),
  ]);
}

/** Whether each run of a checked campaign is done: it has a result for every response. */
export function campaignDone(c) {
  const later = Object.values(c.more_responses ?? {});
  return c.responses.map((v, i) => v !== null && later.every((values) => values[i] !== null));
}

/**
 * Bring results that came back by link into the same design saved here: each run without a
 * result here takes the link's; a run with a result here keeps it, even where the link's differs.
 * Returns the results and their sources, how many were added, and how many differed and were kept.
 * A study of several responses merges each response's in turn, as ``more_responses`` and ``more_sources``.
 */
export function mergeResults(here, incoming) {
  const merged = mergeOne(here.responses, here.sources, incoming.responses, incoming.sources);
  const out = { responses: merged.responses, sources: merged.sources, added: merged.added, kept: merged.kept };
  if (here.more_responses) {
    out.more_responses = {};
    out.more_sources = {};
    for (const name of Object.keys(here.more_responses)) {
      const blank = here.more_responses[name].map(() => null);
      const one = mergeOne(
        here.more_responses[name],
        here.more_sources[name],
        incoming.more_responses?.[name] ?? blank,
        incoming.more_sources?.[name] ?? blank,
      );
      out.more_responses[name] = one.responses;
      out.more_sources[name] = one.sources;
      out.added += one.added;
      out.kept += one.kept;
    }
  }
  return out;
}

/** Merge one response's results that came back by link into those here, as mergeResults says. */
function mergeOne(hereValues, hereSources, values, sources) {
  const responses = [...hereValues];
  const merged = [...hereSources];
  let added = 0;
  let kept = 0;
  values.forEach((value, i) => {
    if (value === null) return;
    if (responses[i] === null) {
      responses[i] = value;
      merged[i] = sources[i] ?? "entered";
      added += 1;
    } else if (responses[i] !== value) {
      kept += 1;
    }
  });
  return { responses, sources: merged, added, kept };
}
