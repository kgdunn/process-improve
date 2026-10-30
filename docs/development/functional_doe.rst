Functional responses in designed experiments (investigation)
============================================================

Status: investigation. Nothing here is implemented yet. This note records how functional
principal component analysis (FPCA) could be added to ``process_improve`` for designed
experiments, and to the browser app, and what a prototype on the book's case study showed.

The idea
--------

Some experiments return a curve per run, not a number: a colour-development profile, a
dissolution profile, a temperature trajectory, a spectrum. The functional-data route to such a
response has three steps:

1. Smooth each measured curve with a basis expansion (B-splines, penalised B-splines (P-splines),
   a Fourier basis, or wavelets). A curve becomes a short coefficient vector.
2. Run PCA on those curves. The result is a mean function, a few orthonormal *eigenfunctions*
   (shape functions), and per-run *FPC scores*.
3. Model each FPC score against the design factors with an ordinary DOE model. Because the
   eigenfunctions rebuild the curve, a prediction of the scores is a prediction of the whole
   curve: :math:`\hat{y}(t \mid x) = \mu(t) + \sum_k \hat{s}_k(x)\,\phi_k(t)`.

The companion book already discusses this route, as an alternative to PLS on the curve, in the
mixed-level profile case study (``product-development-product-improvement/mixed-level-profile-case-study.rst``,
the paragraph just before "Model diagnostics: SPE and Hotelling's T2"). It is prose only today; there
is no code for it in the book or in the library.

What the prototype showed
-------------------------

The prototype rebuilds the book's 60-run split-plot design and its simulated ten-point curves
(same seed, same ground truth), then runs the three steps above with ``scipy`` and ``numpy`` only.
The results:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    *   - Check
        - Result
    *   - P-spline smoothing, one lambda for all curves, chosen by generalised cross-validation
        - lambda 0.178, 4.7 effective degrees of freedom out of 10 time points
    *   - Variance explained by FPC 1, FPCs 1 and 2
        - 98.0%, 99.9%
    *   - Do two eigenfunctions span the two shapes the simulation was built from?
        - Yes: the plateau shape to within 0.005, the late-drift shape to within 0.046 (its kink is
          smoothed)
    *   - ANOVA of FPC 1 (amplitude), p < 0.01
        - Every term: compound, the three continuous factors, their interactions with compound,
          concentration
    *   - ANOVA of FPC 2 (late drift), p < 0.01
        - Compound only, which is how the simulation generated the drift

So the per-score ANOVA reads the mechanism off directly: the process factors change how much colour
develops; only the choice of compound changes whether it keeps drifting. The scores are uncorrelated
across runs by construction, so one ANOVA per score is a fair decomposition.

Ten-fold cross-validated prediction of the whole curve, root mean square error against the
noise-free truth (the measurement noise standard deviation is 0.03):

.. list-table::
    :header-rows: 1
    :widths: 60 40

    *   - Model of the curve, all with the book's compound-by-factor interaction terms
        - CV RMSE against the truth
    *   - FPCA, 2 FPCs, least squares on each score
        - 0.046
    *   - Least squares on each of the ten time points, no FPCA
        - 0.051
    *   - PLS, 3 components (the book's final model)
        - 0.118

The reading: most of the gap to PLS comes from the regression model, not from FPCA. Least squares on
all 24 interaction terms fits this simulation, which was generated from that model, and PLS with 3
components compresses it. FPCA adds about a 10% further reduction, from the smoothing and the
rank-2 truncation. With ten equally spaced points the functional machinery adds little over PCA of
the raw columns; it pays when the grid is dense and noisy, irregular, or differs from run to run.

Two identities make FPCA useful for the questions the book's case study asks:

* **Curve matching is score matching.** For curves in the span of the eigenfunctions, the L2
  distance between two curves equals the Euclidean distance between their score vectors (the
  prototype checks this to five decimals). Finding settings that reproduce a target curve, which
  the book does by PLS inversion, becomes a least-squares problem in two or three score
  dimensions, which ``experiments.optimization`` and the desirability code can already solve.
* **A pointwise band needs the full score covariance.** The scores are uncorrelated across runs,
  but their residuals after the factor fit are not: in the prototype the two score residuals
  correlate at -0.66, and a band built from the per-score variances alone is off by up to 71%.
  The variance of the predicted mean curve at a new point is
  :math:`h_0\,\phi(t)^T \Sigma_\eta\,\phi(t)`, with :math:`h_0` the leverage of the point and
  :math:`\Sigma_\eta` the residual covariance of the scores. A band for a new run adds
  :math:`\phi(t)^T \Sigma_\eta\,\phi(t)` and the variance of the truncation residual. Neither
  includes the uncertainty in the estimated eigenfunctions, which a bootstrap adds.

One detail matters for correctness. PCA of the basis coefficients must use the L2 metric of the
basis, the Gram matrix :math:`W = \int \phi(t)\phi(t)^T\,dt`, not the identity. Otherwise the
eigenfunctions depend on where the knots happen to be. The core of the prototype:

.. code-block:: python

    W = (B_fine.T * np.gradient(fine)) @ B_fine        # Gram matrix by quadrature
    W_half = np.real(scipy.linalg.sqrtm(W))
    Z = (coef - coef.mean(axis=0)) @ W_half             # coefficients in the L2 metric
    evals, U = np.linalg.eigh(Z.T @ Z / (len(Z) - 1))
    evals, U = evals[::-1], U[:, ::-1]                  # largest variance first
    eigenfunction_coef = np.linalg.solve(W_half, U)     # back to basis coefficients
    scores = Z @ U

Proposed library surface
------------------------

Smallest change that covers the case study, with no new runtime dependency (``scipy`` has
``BSpline.design_matrix`` and ``make_smoothing_spline``):

* ``multivariate/_fpca.py``: an ``FPCA`` estimator on ``BaseEstimator`` and ``TransformerMixin``,
  like ``PCA``. ``fit(curves, grid=...)`` takes a wide frame (one row per run, one column per
  time point) or a long frame (run, time, value) for irregular grids. Fitted attributes:
  ``mean_function_``, ``eigenfunctions_``, ``scores_`` (a ``_LazyFrame`` descriptor, following the
  pattern in ``multivariate/_base.py``), ``explained_variance_ratio_``. Methods ``transform``,
  ``inverse_transform`` (scores to curves on any grid) and ``derivative`` (rates come from the basis
  analytically, which gives an initial-rate response for free).
  Basis choice to start with: P-splines with lambda by GCV. Fourier and wavelets wait until a data
  set needs them.
* ``experiments``: one function, ``analyze_functional_experiment(design, curves, model=...)``, that
  fits ``FPCA``, calls ``analyze_experiment`` once per score (so blocking, split-plot whole plots
  and coding behave as they do today), and returns the per-score ANOVA tables plus a
  ``predict_curve(new_points)`` that returns the curve and its pointwise band.
* Registration (alignment of curves that shift in time) reuses the DTW code in ``batch/``. It is
  out of scope for the first version; the case study's curves need none.

Browser app
-----------

* **Entry.** A response of kind *curve*: one response name, many columns (``colour[t0]`` to
  ``colour[t9]``) in the run sheet and the workbook. ``MAX_RESPONSES`` stays at 5 by counting a
  curve as one response.
* **Analysis.** One worker call, ``api_functional``, returns the mean function and eigenfunctions
  on a plotting grid, the score coefficients, :math:`(X^TX)^{-1}` and the residual covariance of
  the scores. These are a few kilobytes.
* **Profiler.** Prediction is linear in the coefficients, so the page draws the predicted curve and
  its band in JavaScript as the factor sliders move, with no round trip to Pyodide. The same
  numbers drive a second view: drag one FPC score and watch the curve shape change, which is how
  a user learns what each eigenfunction means.
* **Target curve.** Draw or paste a target curve; the page projects it to scores and the existing
  best-settings search finds the factor settings closest to it.

Open questions
--------------

* Whether ``FPCA`` belongs in ``multivariate`` or in a new ``functional`` subpackage.
* Whether the book's case study gains a runnable FPCA comparison once the release is on PyPI.
  The book runs its code against the PyPI release, so that change follows the library release.
* How many FPCs by default: a variance threshold (for example 99%), or cross-validated curve
  prediction error as in the table above.
