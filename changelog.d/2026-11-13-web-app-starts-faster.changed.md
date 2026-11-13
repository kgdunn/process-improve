- **factori.al starts Python in well under half the time.** Measured in Chromium, the wait until the
  page says Ready fell from about 26 s to about 11 s on a first visit, and from 25 s to 10 s on a
  later one. Three changes do it. Pyodide's packages are loaded from the release's `pyc` folder,
  already compiled to bytecode, and download while the interpreter starts. The build compiles the
  app's own Python, the wheel and `web/bootstrap.py`, for the Python that Pyodide runs, and the
  wheel's browser copy leaves out the reference catalogue's 29 MB of sources, which the page never
  reads. openpyxl and plotly are installed when a call first needs them, a workbook or a chart, and
  no longer at start-up, so a visit that makes a run sheet downloads neither, and the page starts
  on a network that cannot reach PyPI. The build asks PyPI once for the exact files of those
  packages and writes their addresses into its manifest: the page installs those files, asks no
  package index, and chooses no version in the browser. (#1233)
