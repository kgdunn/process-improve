- **Curve responses in the reference catalogue.** A response measured as a curve (a time course, a
  profile or a spectrum per run) now carries a `curve` key: its columns in the run table, in order,
  and where along the curve each is read, either points every run shares or each run's own times.
  The colour-development case of the book uses it, and a check that names no response is about the
  first response that is not a curve.
- **`experiments.datasets.biscuit()`.** The NIR spectra of 72 biscuit doughs whose fat, sucrose,
  dry flour and water were varied (Osborne et al., 1984): a mixture of four ingredients with a
  700-point spectrum as each run's response, also reachable as `data("biscuit")`. The file keeps
  the GPL of the R package it comes from; see the `NOTICE` beside it.
