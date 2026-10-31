- **Curve responses in the reference catalogue.** A response measured as a curve (a time course, a
  profile or a spectrum per run) now carries a `curve` key: its columns in the run table, in order,
  and where along the curve each is read, either points every run shares or each run's own times.
  The colour-development case of the book uses it, and a check that names no response is about the
  first response that is not a curve.
- **`experiments.datasets.biscuit()`.** The NIR spectra of 72 biscuit doughs whose fat, sucrose,
  dry flour and water were varied (Osborne et al., 1984): a mixture of four ingredients with a
  700-point spectrum as each run's response, also reachable as `data("biscuit")`. The file keeps
  the GPL of the R package it comes from; see the `NOTICE` beside it.
- **The reference catalogue holds McMullen and Jurica (2024), the first case whose response is a time course
  sampled at different times in each run.** `mcmullen-jurica-2024` holds the methoxylation optimisation of the
  article's supplementary information: 40 reactions with 9 to 12 samples each, and the 15 nearest the optimum
  modelled by functional PCA and a quadratic model of their scores. The library reproduces 21 of the 22 numbers
  of that model; one coefficient is 0.07% off for a reason not yet known, and the 16 FPCA numbers wait for a
  functional PCA in the library. Table S10 reproduces only from Table S7's settings, which shows that Table S9
  swaps the DMF values of experiments 12 and 25.
