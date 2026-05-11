#!/usr/bin/env Rscript
#
# Generate the mixOmics::nipals reference loadings and scores against which
# process_improve's PCA(algorithm="nipals") is cross-checked.
#
# Contract:
# - Reads linnerud_centered_with_nan.csv (produced by prepare_fixture.py).
#   The matrix is already column-mean-centered using nanmean, so no further
#   preprocessing is applied here. NaN entries are preserved.
# - Runs mixOmics::nipals with n.components = N_COMPONENTS.
# - Writes (next to this script):
#     reference_loadings.csv  (K x A; rows = columns of X)
#     reference_scores.csv    (N x A; rows = samples)
#     reference_metadata.json (n.components, max.iter, tol, mixOmics
#                              version, R version, eigenvalues)
#
# Run from the repository root:
#
#   Rscript tests/fixtures/mixomics_nipals/run_reference.R
#
# Or from anywhere by passing the fixture directory as a single argument:
#
#   Rscript /path/to/run_reference.R /path/to/tests/fixtures/mixomics_nipals
#
# One-time mixOmics install (it lives on Bioconductor, not CRAN):
#
#   if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#   BiocManager::install("mixOmics")
#   install.packages("jsonlite")

suppressPackageStartupMessages({
  library(mixOmics)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Locate the fixture directory: explicit arg, script-relative under Rscript,
# or repo-relative from the current working directory.
# ---------------------------------------------------------------------------

locate_fixture_dir <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1L && nzchar(args[[1L]])) {
    return(normalizePath(args[[1L]], mustWork = TRUE))
  }
  argv <- commandArgs(trailingOnly = FALSE)
  file_arg <- argv[grepl("^--file=", argv)]
  if (length(file_arg) == 1L) {
    script_path <- sub("^--file=", "", file_arg)
    return(normalizePath(dirname(script_path), mustWork = TRUE))
  }
  candidate <- file.path(getwd(), "tests", "fixtures", "mixomics_nipals")
  if (dir.exists(candidate)) {
    return(normalizePath(candidate, mustWork = TRUE))
  }
  stop(
    "Could not locate the fixture directory. Pass it as the first argument, ",
    "or run from the repository root."
  )
}

here <- locate_fixture_dir()
input_csv <- file.path(here, "linnerud_centered_with_nan.csv")
if (!file.exists(input_csv)) {
  stop(
    "Missing canonical input: ", input_csv, ". ",
    "Run `python tests/fixtures/mixomics_nipals/prepare_fixture.py` first."
  )
}

# ---------------------------------------------------------------------------
# Settings (kept in sync with the Python-side tolerance in
# tests/test_pca_nipals_mixomics_crosscheck.py).
# ---------------------------------------------------------------------------

N_COMPONENTS <- 3L
MAX_ITER <- 1000L
TOL <- 1e-09

# ---------------------------------------------------------------------------
# Fit mixOmics::nipals on the already-centered matrix.
# ---------------------------------------------------------------------------

X <- as.matrix(read.csv(input_csv, check.names = FALSE))

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
