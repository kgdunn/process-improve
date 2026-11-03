- **`multivariate.FPCA`: functional PCA of curves.** Each curve (a time course, a profile or a
  spectrum per sample) is smoothed by one penalised B-spline smoother, its penalty chosen by
  generalised cross-validation pooled over the curves, and the components are found in the L2
  metric of the basis, so they are orthonormal functions whatever the number of B-splines.
  Missing points are `NaN`, so curves sampled at their own times fit too. `transform` scores new
  curves, `inverse_transform` rebuilds curves or their derivatives from scores, and `pca_` holds
  the underlying PCA, whose Hotelling's T² and SPE flag an unusual curve. On the methoxylation case
  of McMullen and Jurica (2024) in the reference catalogue, the first component's scores correlate
  0.999 with the source's.
