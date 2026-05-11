#!/usr/bin/env Rscript
#
# Generate the mixOmics::nipals reference loadings and scores against which
# process_improve's PCA(algorithm="nipals") is cross-checked.
#
# Contract:
# - Reads linnerud_centered_with_nan.csv (produced by prepare_fixture.py).
#   The matrix is already column-mean-centered using nanmean, so no further
#   preprocessing is applied here. NaN entries are preserved.
# - Runs mixOmics::nipals with n.components = N_COMPONENTS (set below).
# - Writes:
#     reference_loadings.csv  (K x A; rows = columns of X)
#     reference_scores.csv    (N x A; rows = samples)
#     reference_metadata.json (n.components, max.iter, tol, mixOmics
#                              version, R version, sessionInfo summary)
#
# Run from the repository root:
#
#   Rscript tests/fixtures/mixomics_nipals/run_reference.R
#
# Requires mixOmics (Bioconductor):
#
#   if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#   BiocManager::install("mixOmics")

suppressPackageStartupMessages({
  library(mixOmics)
  library(jsonlite)
})

here <- normalizePath(dirname(sys.frame(1)$ofile %||% "tests/fixtures/mixomics_nipals"))
if (is.null(here) || !dir.exists(here)) {
  here <- file.path(getwd(), "tests", "fixtures", "mixomics_nipals")
}

input_csv <- file.path(here, "linnerud_centered_with_nan.csv")
stopifnot(file.exists(input_csv))

X <- as.matrix(read.csv(input_csv, check.names = FALSE))

N_COMPONENTS <- 3L
MAX_ITER <- 1000L
TOL <- 1e-09

fit <- nipals(X, ncomp = N_COMPONENTS, max.iter = MAX_ITER, tol = TOL)

# mixOmics::nipals returns:
#   $p     loadings, K x ncomp
#   $t     scores,   N x ncomp
#   $eig   singular values, length ncomp
loadings_df <- as.data.frame(fit$p)
scores_df <- as.data.frame(fit$t)
colnames(loadings_df) <- paste0("PC", seq_len(N_COMPONENTS))
colnames(scores_df) <- paste0("PC", seq_len(N_COMPONENTS))
loadings_df <- cbind(variable = colnames(X), loadings_df)

write.csv(
  loadings_df,
  file = file.path(here, "reference_loadings.csv"),
  row.names = FALSE
)
write.csv(
  scores_df,
  file = file.path(here, "reference_scores.csv"),
  row.names = FALSE
)

metadata <- list(
  n_components = N_COMPONENTS,
  max_iter = MAX_ITER,
  tol = TOL,
  eigenvalues = as.numeric(fit$eig),
  mixOmics_version = as.character(packageVersion("mixOmics")),
  R_version = R.version.string,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)
write_json(
  metadata,
  path = file.path(here, "reference_metadata.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

cat("wrote reference loadings, scores, metadata to", here, "\n")
