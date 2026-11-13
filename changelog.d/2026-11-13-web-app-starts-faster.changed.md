- **factori.al starts Python in about a third of the time.** Measured in Chromium, the wait until the
  page says Ready fell from about 26 s to about 11 s on a first visit, and then to about 8 s. Four
  changes do it. Pyodide's packages are loaded from the release's `pyc` folder, already compiled to
  bytecode, and download while the interpreter starts. The build compiles the app's own Python, the
  wheel and `web/bootstrap.py`, for the Python that Pyodide runs, and the wheel's browser copy
  leaves out the reference catalogue's 29 MB of sources, which the page never reads. openpyxl and
  plotly are installed when a call first needs them, a workbook or a chart, and no longer at
  start-up, so a visit that makes a run sheet downloads neither, and the page starts on a network
  that cannot reach PyPI. The build asks PyPI once for the exact files of those packages and writes
  their addresses into its manifest: the page installs those files, asks no package index, and
  chooses no version in the browser. And statsmodels and scikit-learn, which a model is fitted
  with, are left out of the start: the page is ready once a design can be made, and fetches the two
  in the background as soon as it is. That last step alone took Ready from 11.4 s to 8.3 s on a
  first visit and from 10.2 s to 7.7 s on a later one, measured in one session, while what a model
  is fitted with is in at the same moment as before, so a first analysis is no later. A fit asked
  for before then waits for the same install, and the status line says so. (#1233, #1235)
