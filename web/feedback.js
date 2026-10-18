// web/feedback.js: the Feedback dialog, where a reader reports a problem with a result, suggests a change to the
// page, or says anything else, to the maintainer. From factori.al the page posts the report to the site's relay
// (deploy/factorial/feedback_relay.py), which emails it and keeps nothing.
//
// The functions here before installFeedback know nothing of the page: they build the report, check it against what
// the relay takes, and say what goes with it. tests/web/js/feedback.test.mjs calls them as they are. The page lends
// the dialog what it sends of itself (the context, and the study when the reader ticks it), so this file imports
// nothing.

/** The topics, by the name a report gives, as the relay names them (feedback_relay.TOPICS). */
export const TOPICS = { algorithm: "Algorithm or model", interface: "Interface suggestion", general: "General" };

/** A message's length in characters: at least MESSAGE_MIN once its spaces are trimmed, and at most MESSAGE_MAX. */
export const MESSAGE_MIN = 10;
export const MESSAGE_MAX = 5000;

/** The longest reply address the relay takes. */
export const EMAIL_MAX = 254;

/** The most a screenshot may hold once decoded, and the most characters a study's packed text may run to. */
export const MAX_SCREENSHOT = 2 * 2 ** 20;
export const MAX_STUDY = 2 ** 18;

/**
 * What the page sends of itself with a report, as the relay takes it (feedback_relay.CONTEXT): each key with the most
 * characters its text may hold, or, for a list, the most texts it may hold and the most characters each.
 */
export const CONTEXT = {
  commit: 64,
  plotly_js: 40,
  tab: 40,
  model: 200,
  charts: [30, 200],
  stamp: [12, 400],
  errors: [5, 300],
  browser: 400,
  window: 40,
};

/** How a report names each of the context's texts, as the relay's email does; the stamp's rows name themselves. */
const CONTEXT_LABELS = {
  commit: "App commit",
  plotly_js: "Plotly.js",
  tab: "Tab on show",
  model: "Model",
  charts: "Charts on show",
  browser: "Browser",
  window: "Window",
};

/** A screenshot: the most pixels it holds, and the quality of its JPEG. */
export const MAX_PIXELS = 4_000_000;
export const JPEG_QUALITY = 0.8;

/** A JPEG as a data: URL, which is how a report carries its screenshot. */
const JPEG_URL = "data:image/jpeg;base64,";

/** Where a report is posted, and from which hosts: the app's own, and this machine, where the tests run it. */
export const RELAY = "/api/feedback";
const SITE = "factori.al";
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);

/** Control characters the relay refuses in any text, and the line breaks and tabs it takes only in the message. */
const CONTROL = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;
const LINE_BREAKS = /[\t\n\r]+/g;

/** The characters of a text as the relay counts them: one for each code point, so an emoji counts once. */
export const characters = (text) => [...text].length;

/** A text cut to at most ``most`` characters, never through the middle of one. */
export function cut(text, most) {
  const chars = [...text];
  return chars.length > most ? chars.slice(0, most).join("") : text;
}

/** A text that UTF-8 can write: half of a surrogate pair, which a cut or a paste can leave, becomes U+FFFD. */
const wellFormed = (text) => String(text).toWellFormed?.() ?? String(text);

/** A text as the relay takes it on one line: its line breaks and tabs a space, and no control character. */
const oneLine = (text) => wellFormed(text).replace(LINE_BREAKS, " ").replace(CONTROL, "");

/** The message as the relay takes it: as the reader wrote it, less any control character a paste brought in. */
const cleanMessage = (text) => wellFormed(text).replace(CONTROL, "");

/** What stops a message from going, as a sentence; "" when it can go. */
export function messageProblem(message) {
  if (characters(message.trim()) < MESSAGE_MIN) return `Write at least ${MESSAGE_MIN} characters.`;
  if (characters(message) > MESSAGE_MAX) return `Write at most ${MESSAGE_MAX} characters.`;
  return "";
}

/** The count under the message box. */
export const messageCount = (message) => `${characters(message)} of ${MESSAGE_MAX} characters`;

/** One address and nothing else: a dot-atom name, an at sign, and a domain with a dot, as the relay reads one. */
const ADDRESS = /^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$/;

/** What stops a reply address from going, as a sentence; "" for one that can go, or for none. */
export function emailProblem(email) {
  const address = email.trim();
  if (!address || (address.length <= EMAIL_MAX && ADDRESS.test(address))) return "";
  return "Give one email address, such as name@example.com, or leave the box empty.";
}

/** How many bytes a base64 text decodes to. */
export const decodedSize = (base64) =>
  Math.floor((base64.length * 3) / 4) - (base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0);

/** What stops a screenshot from going with a report, as a sentence; "" when it can go. */
export function screenshotProblem(dataUrl) {
  if (!dataUrl.startsWith(JPEG_URL)) return "The browser made no JPEG of the tab, so the report goes without it.";
  if (decodedSize(dataUrl.slice(JPEG_URL.length)) > MAX_SCREENSHOT) {
    return `The screenshot is larger than ${MAX_SCREENSHOT / 2 ** 20} MB, so the report goes without it.`;
  }
  return "";
}

/** What stops a study from going with a report, as a sentence; "" when it can go. */
export function studyProblem(packed) {
  if (packed.length > MAX_STUDY) {
    return "Your study is too large to go with a report, so the report goes without it: its workbook, from Run, can go by email.";
  }
  return /^[A-Za-z0-9_-]+$/.test(packed) ? "" : "Your study could not be packed, so the report goes without it.";
}

/** The scale a screenshot of a box this size is taken at: 1, or less to hold it to MAX_PIXELS. */
export const screenshotScale = (width, height) =>
  width * height > MAX_PIXELS ? Math.sqrt(MAX_PIXELS / (width * height)) : 1;

/** The Python errors a report carries, after one more: the last few, each cut to the length the relay takes. */
export function keepError(errors, error) {
  const [count, most] = CONTEXT.errors;
  return [...errors, cut(oneLine(error), most)].slice(-count);
}

/** The context as the relay takes it: only the keys it knows, each text cut to its length and each list to its count. */
export function fitContext(context = {}) {
  const fitted = {};
  for (const [key, limit] of Object.entries(CONTEXT)) {
    const value = context[key];
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(limit)) {
      const [count, most] = limit;
      fitted[key] = [value].flat().slice(0, count).map((item) => cut(oneLine(item), most));
    } else {
      fitted[key] = cut(oneLine(value), limit);
    }
  }
  return fitted;
}

/** The context as lines of text, each labelled as the relay's email labels it, the stamp's rows and the errors last. */
export function contextLines(context = {}) {
  const fitted = fitContext(context);
  const lines = [];
  for (const [key, label] of Object.entries(CONTEXT_LABELS)) {
    if (key in fitted) lines.push(`${label}: ${[fitted[key]].flat().join("; ")}`);
  }
  lines.push(...(fitted.stamp ?? []));
  const errors = fitted.errors ?? [];
  if (!errors.length) lines.push("Last Python errors: none");
  for (const error of errors) lines.push(`Python error: ${error}`);
  return lines;
}

/**
 * The report the relay takes: the topic, the message, and only what the reader chose to send with it, a reply address,
 * a screenshot and the study, beside the page's context.
 */
export function buildReport({ topic, message, email = "", screenshot = null, study = null, context = {} }) {
  const report = { topic, message: cleanMessage(message) };
  if (email.trim()) report.email = email.trim();
  if (screenshot) report.screenshot = screenshot;
  if (study) report.study = study;
  report.context = fitContext(context);
  return report;
}

/**
 * What the dialog says will be sent, a line for each part of the report: every field it holds, and what it leaves out
 * of what it could hold. ``studied`` says the study goes, before it is packed for the report.
 */
export function sentLines({ topic, message, email = "", screenshot = null, studied = false, context = {} }) {
  const address = email.trim();
  const size = screenshot ? Math.ceil(decodedSize(screenshot.slice(JPEG_URL.length)) / 1024) : 0;
  return [
    `Topic: ${TOPICS[topic]}`,
    `Your message: ${characters(message.trim())} characters`,
    address ? `Your email address, to reply to: ${address}` : "No email address: the maintainer cannot reply",
    screenshot ? `A screenshot of this tab: a JPEG of ${size} KB` : "No screenshot",
    studied ? "Your study, as a link that opens it with its results" : "Not your study",
    ...contextLines(context),
  ];
}

/**
 * Where the page posts a report from ``location``: the relay, on factori.al and on this machine; null anywhere else,
 * such as the copy on GitHub Pages or one a reader serves, so that nothing a reader sends reaches another server.
 */
export function relayUrl(location) {
  const { protocol, hostname } = location;
  if (protocol === "https:" && hostname === SITE) return RELAY;
  if ((protocol === "http:" || protocol === "https:") && LOOPBACK.has(hostname)) return RELAY;
  return null;
}

/** How long a report may take to reach the relay and be answered, in milliseconds. */
const SEND_TIMEOUT = 30_000;

/** Why a report did not go, as the dialog says it: by the relay's answer, or with none (null) when it was never reached. */
export function failureWords(status) {
  if (status === null) return "The report could not be sent: the page could not reach factori.al.";
  if (status === 429) return "The report could not be sent: your network has sent as many reports as the site takes in an hour.";
  if (status === 413) return "The report could not be sent: it is larger than the site takes.";
  if (status >= 500) return "The report could not be sent: the site's mail relay is busy or not running.";
  return `The report could not be sent: the site refused it (HTTP ${status}).`;
}

/** Post a report to the relay; resolve to whether it went, and why not as a sentence. */
async function post(url, report) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(report),
      signal: AbortSignal.timeout(SEND_TIMEOUT),
    });
    return res.ok ? { ok: true } : { ok: false, why: failureWords(res.status) };
  } catch {
    return { ok: false, why: failureWords(null) }; // offline, refused, or too slow
  }
}

/**
 * A screenshot of ``root``, the page's main element, as a JPEG data: URL. modern-screenshot, imported only now, draws
 * the page through SVG's foreignObject, so the browser itself applies the page's CSS, color-mix() among it. A dialog is
 * left out, the page's own background fills what is transparent, and a page longer than MAX_PIXELS allows is scaled
 * down whole. No font is fetched: the page's are the system's.
 */
async function takeScreenshot(root) {
  const { domToJpeg } = await import("./vendor/modern-screenshot.mjs");
  const { width, height } = root.getBoundingClientRect();
  return domToJpeg(root, {
    scale: screenshotScale(width, height),
    quality: JPEG_QUALITY,
    backgroundColor: getComputedStyle(document.body).backgroundColor,
    filter: (node) => node.nodeName !== "DIALOG",
    font: false,
    timeout: 10_000,
  });
}

/** The dialog's form: built once, from fixed text, the first time a Feedback button is pressed. */
const FORM = `
<form class="feedback__form" novalidate>
  <div class="sheet__head">
    <h2 id="h-feedback">Feedback</h2>
    <button type="button" class="link" data-close aria-label="Close the feedback">✕</button>
  </div>
  <div data-part="write">
    <p class="hint">A problem with a result, a change you would like to the page, or anything else: it goes to the
      maintainer of Factorial.</p>
    <fieldset class="feedback__topics">
      <legend>Topic</legend>
      ${Object.entries(TOPICS)
        .map(([value, label]) => `<label class="check"><input type="radio" name="topic" value="${value}"> ${label}</label>`)
        .join("\n      ")}
    </fieldset>
    <label class="feedback__field">Your message
      <textarea id="feedback-message" name="message" rows="6" maxlength="${MESSAGE_MAX}"
        aria-describedby="feedback-count"></textarea></label>
    <p id="feedback-count" class="hint"></p>
    <label class="feedback__field">Your email address, if you would like a reply
      <input id="feedback-email" name="email" type="email" maxlength="${EMAIL_MAX}" autocomplete="email"
        spellcheck="false"></label>
    <div class="feedback__extras" data-part="extras">
      <label class="check"><input type="checkbox" id="feedback-screenshot" name="screenshot" checked> Include a
        screenshot of this tab</label>
      <p class="hint">The screenshot shows what is on screen, including your study's data.</p>
      <p class="hint" data-note="shot" role="status" hidden></p>
      <img id="feedback-shot" class="feedback__shot" alt="The screenshot that goes with the report" hidden>
      <label class="check"><input type="checkbox" id="feedback-study" name="study"> Include my study, so the result
        can be reproduced</label>
      <p class="hint" data-note="study"></p>
    </div>
    <details class="feedback__sent"><summary>What will be sent</summary><ul id="feedback-sent"></ul></details>
    <p class="hint" data-note="status" role="status" hidden></p>
    <p class="actions">
      <button type="submit" id="feedback-send" class="primary">Send</button>
      <button type="button" class="link" data-close>Cancel</button>
    </p>
  </div>
  <div data-part="sent" hidden>
    <p data-note="thanks" role="status"></p>
    <p class="actions"><button type="button" class="primary" data-close>Close</button></p>
  </div>
</form>`;

/** What the box under the study's checkbox says, with a study on the page and without one. */
const STUDY_NOTE = "It goes as a link that opens your study, with its results, in Factorial.";
const NO_STUDY = "There is no run sheet yet, so there is no study to send.";

/**
 * Make the Feedback dialog, and open it from every button with a ``data-feedback`` attribute, on the topic that names
 * (General when it names none). The page lends it ``context()``, which resolves to what the page sends of itself (the
 * keys of CONTEXT), ``study()``, which resolves to the study's packed text, and ``hasStudy()``, whether there is one.
 * Returns ``open(topic)``.
 */
export function installFeedback({ context, study, hasStudy }) {
  let dialog = null;
  let form = null;
  let about = {}; // the page's context, as it was when the dialog opened
  let shot = null; // the screenshot's data: URL, once taken and fit to go
  let taking = null; // the screenshot being taken: a promise, replaced by a newer one, and null once it is done
  let sending = false;
  const relay = relayUrl(location);

  const field = (name) => form.elements[name];
  const part = (name) => dialog.querySelector(`[data-part="${name}"]`);
  const note = (name) => dialog.querySelector(`[data-note="${name}"]`);
  const topic = () => field("topic").value || "general";

  /** Say what is happening, or what is wrong, under the form; "" says nothing. */
  function say(text, problem = false) {
    note("status").textContent = text;
    note("status").classList.toggle("error", problem);
    note("status").hidden = !text;
  }

  /** The count under the message, and the list of what will be sent, as the form now stands. */
  function render() {
    dialog.querySelector("#feedback-count").textContent = messageCount(field("message").value);
    const lines = sentLines({
      topic: topic(),
      message: field("message").value,
      email: field("email").value,
      screenshot: field("screenshot").checked ? shot : null,
      studied: field("study").checked,
      context: about,
    });
    dialog.querySelector("#feedback-sent").replaceChildren(
      ...lines.map((line) => Object.assign(document.createElement("li"), { textContent: line })),
    );
  }

  /** Show the screenshot under its box, and say what is happening to it; "" says nothing. */
  function showShot(text = "") {
    const image = dialog.querySelector("#feedback-shot");
    if (shot) image.src = shot;
    else image.removeAttribute("src");
    image.hidden = !shot;
    note("shot").textContent = text;
    note("shot").hidden = !text;
  }

  /** Take the screenshot of the page and show it; or say why there is none, and the report goes without it. */
  function capture() {
    shot = null;
    showShot("Taking the screenshot...");
    const mine = takeScreenshot(document.querySelector("main")).then(
      (dataUrl) => ({ dataUrl, why: screenshotProblem(dataUrl) }),
      () => ({ dataUrl: null, why: "The screenshot could not be taken, so the report goes without it." }),
    );
    taking = mine;
    mine.then(({ dataUrl, why }) => {
      if (taking !== mine) return; // unticked, or taken again, meanwhile
      taking = null;
      shot = why ? null : dataUrl;
      showShot(why);
      render();
    });
  }

  function build() {
    dialog = document.createElement("dialog");
    dialog.id = "feedback-dialog";
    dialog.className = "sheet feedback";
    dialog.setAttribute("aria-labelledby", "h-feedback");
    dialog.innerHTML = FORM;
    document.body.append(dialog);
    form = dialog.querySelector("form");
    form.addEventListener("input", () => {
      say("");
      render();
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send();
    });
    field("screenshot").addEventListener("change", () => {
      if (field("screenshot").checked) {
        capture();
        return;
      }
      taking = null;
      shot = null;
      showShot();
    });
    for (const button of dialog.querySelectorAll("[data-close]")) button.onclick = () => dialog.close();
  }

  /** Open the dialog on ``chosen``, a topic; a message written before and not sent is kept. */
  async function open(chosen) {
    if (!dialog) build();
    if (!part("sent").hidden) {
      part("sent").hidden = true;
      part("write").hidden = false;
    }
    field("topic").value = chosen in TOPICS ? chosen : "general";
    const can = hasStudy();
    field("study").disabled = !can;
    if (!can) field("study").checked = false;
    note("study").textContent = can ? STUDY_NOTE : NO_STUDY;
    say("");
    about = {};
    render();
    if (!dialog.open) dialog.showModal();
    field("message").focus();
    if (relay && field("screenshot").checked) capture();
    about = (await Promise.resolve().then(context).catch(() => null)) ?? {};
    render();
  }

  /** Send the report: check it, add the study if the reader asked for it, and post it. */
  async function send() {
    if (sending) return;
    const problem = messageProblem(field("message").value) || emailProblem(field("email").value);
    if (problem) {
      say(problem, true);
      field(messageProblem(field("message").value) ? "message" : "email").focus();
      return;
    }
    sending = true;
    dialog.querySelector("#feedback-send").disabled = true;
    while (taking) {
      say("Waiting for the screenshot...");
      await taking;
    }
    say("Sending...");
    const left = []; // what the reader asked for that cannot go, as sentences
    let packed = null;
    if (field("study").checked) {
      packed = await Promise.resolve().then(study).catch(() => null);
      const why = packed ? studyProblem(packed) : "Your study could not be packed, so the report goes without it.";
      if (why) {
        left.push(why);
        packed = null;
      }
    }
    const report = buildReport({
      topic: topic(),
      message: field("message").value,
      email: field("email").value,
      screenshot: field("screenshot").checked ? shot : null,
      study: packed,
      context: about,
    });
    const sent = relay
      ? await post(relay, report)
      : { ok: false, why: "This copy of Factorial is not at factori.al, so it sends no report itself." };
    sending = false;
    dialog.querySelector("#feedback-send").disabled = false;
    if (!sent.ok) {
      say(sent.why, true);
      return;
    }
    say("");
    field("message").value = "";
    field("study").checked = false;
    field("screenshot").checked = true;
    shot = null;
    showShot();
    note("thanks").textContent = ["Thank you: your report is on its way to the maintainer.", ...left].join(" ");
    part("write").hidden = true;
    part("sent").hidden = false;
    render();
  }

  document.addEventListener("click", (event) => {
    const opener = event.target.closest?.("[data-feedback]");
    if (opener && !dialog?.contains(opener)) open(opener.dataset.feedback);
  });
  return { open };
}
