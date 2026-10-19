- **Factorial takes feedback.** A Feedback button in the header and the footer, and "Report this
  result" on Learn and beside an analysis error, open a dialog for a problem with a result, a
  suggestion, or anything else. A screenshot of the tab goes unless the reader unticks it (taken by a
  vendored modern-screenshot, MIT, loaded only then), and the study, as a link with its results, only
  if they tick it, beside the page's build, a report's stamp, the tab and charts on show and the last
  Python errors; a list names all of it. From factori.al the report goes to
  `deploy/factorial/feedback_relay.py`, a standard-library relay behind Caddy that checks it, limits
  each network to 10 reports an hour, emails the maintainer and keeps nothing. The dialog then shows
  the report's reference (such as FB-7K3Q2M, also in the email's subject), and a reader who gave an
  address gets a receipt of fixed text: the reference, the topic and the day, and nothing they
  wrote, at most one a day to an address. Elsewhere, or when the
  relay cannot take it, the dialog offers a GitHub issue from `.github/ISSUE_TEMPLATE/web_app.yml`
  (without the screenshot, the study or the address), a copy, and an email.
