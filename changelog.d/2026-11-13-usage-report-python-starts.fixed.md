- **`factorial-usage` counts Python started on factori.al from the request every start makes.**
  The report counted a fetch of `bootstrap.py`. A build now ships the page's own Python in an
  archive the browser keeps for a year, so a returning visitor no longer asks for it, and the
  column would have read close to zero. It now counts the page's worker reading `runtime.json`,
  which it does past every cache each time it starts Python; the same request was logged before
  this change too, so the column reads the same over the logs kept. Run `setup.sh` on the server
  to install the new report.
