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
