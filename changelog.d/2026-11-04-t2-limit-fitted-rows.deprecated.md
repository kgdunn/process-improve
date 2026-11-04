- **The Hotelling's T² limit for the rows a model was fitted to changes in 3.0, from the F law to the beta
  law; announced in 2.4.0.** A fitted row helped estimate the centre and the spread of the scores, so its
  T² can never exceed (N - 1)²/N, and its law is ((N - 1)²/N)·Beta(A/2, (N - A - 1)/2). The F limit,
  A(N² - 1)/(N(N - A))·F(A, N - A), is the law of a new row, and it is too wide for fitted rows: at N = 20
  and A = 2 it flags 0.8% of them where it states 5%, and below N = 12 it can flag none. (#978)
  - `judged_rows="fitted"` or `"new"` says which rows a limit judges. `hotellings_t2_limit` and
    `ellipse_coordinates` take it, as do the methods of the same names on every model that has them (`PCA`,
    `PLS` and its subclasses, `OPLS`, `TPLS`, `MBPLS`, `MBPCA`, the adaptive models, `BatchPCA` and
    `BatchPLS`), and `score_plot`, `t2_plot` and `detect_outliers`; `find_reference_batch` takes it as a
    `settings` key.
  - Until 3.0 the default, `None`, gives the F limit, as before, and a call whose number would change
    issues a `FutureWarning` at the caller's line, with both numbers. `detect_outliers` also puts the
    warning's text in each record, as `hotellings_t2_limit_note`. Passing either value silences it; a call
    with no components, or as many components as rows, does not warn. From 3.0 `None` means `"fitted"`.
  - Code that judges new rows already says so and is unchanged: `BatchMonitor`, the adaptive models, the
    batch control functions, the held-out rows of `compare_cv_criteria`, optimal inversion and the
    `pca_predict` and `pls_predict` tools. `MidCourseCorrector` uses the shared limit in place of its own
    copy of the F formula.
  - Two results change now, since a tool cannot see a warning: the `detect_multivariate_outliers` tool
    judges its fitted rows by the beta limit, and the multivariate workbench (the `mva_` tools and the
    browser's model page) judges its fitted rows, and draws its score-plot ellipse, by the beta limit, and
    the rows a model left out by the F limit, which it reports as `"T2 new rows"`.
