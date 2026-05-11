# mixOmics::nipals cross-check fixture

Reference numerical fixture for validating
`process_improve.PCA(algorithm="nipals")` on data with missing values
against the R package `mixOmics`, which implements the canonical
NIPALS skip-NaN variant from Wold (1966).

## Files

| File | Role |
| --- | --- |
| `linnerud.csv` | Source matrix (20 samples; 3 exercise + 3 physiological columns; vendored from `sklearn.datasets.load_linnerud`). |
| `prepare_fixture.py` | Single source of truth for the canonical matrix. Injects MCAR NaN with a fixed seed and mean-centers using `nanmean`. |
| `linnerud_centered_with_nan.csv` | Canonical centered-with-NaN matrix; both Python and R fit on this. Regenerable from `prepare_fixture.py`. |
| `run_reference.R` | R script that fits `mixOmics::nipals` on the canonical matrix and writes loadings, scores, and metadata. |
| `reference_loadings.csv` | Loadings P (K x A) emitted by `mixOmics::nipals`. **Committed by the project owner once generated.** |
| `reference_scores.csv` | Scores T (N x A) emitted by `mixOmics::nipals`. **Committed by the project owner once generated.** |
| `reference_metadata.json` | Capture of the R session info (mixOmics + R version, tolerance, max.iter, eigenvalues). |

The pytest cross-check
(`tests/test_pca_nipals_mixomics_crosscheck.py`) is auto-skipped while
`reference_loadings.csv` / `reference_scores.csv` are absent, so the
fixture lands incrementally without breaking CI.

## Regenerating the fixture

```bash
# 1. Recreate the canonical centered-with-NaN matrix
uv run python tests/fixtures/mixomics_nipals/prepare_fixture.py

# 2. Generate the R-side reference
Rscript tests/fixtures/mixomics_nipals/run_reference.R
```

`mixOmics` is on Bioconductor:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("mixOmics")
```

## Source data attribution

`linnerud.csv` is the Linnerud physical-fitness dataset (20 athletes;
Tenenhaus, *La regression PLS: theorie et pratique*, 1998). It is
public-domain and ships unmodified inside scikit-learn under their
BSD-3-Clause licence. No third-party data is redistributed by this
fixture beyond what scikit-learn already ships.

## Provenance and licensing of the reference numbers

`reference_loadings.csv`, `reference_scores.csv`, and
`reference_metadata.json` are *outputs* produced by running
`mixOmics::nipals` on user-supplied input data
(`linnerud_centered_with_nan.csv`, derived from the BSD-3-Clause
linnerud matrix above). They are not derivative works of the
mixOmics source code:

- No source code, documentation, vignettes, or examples from the
  mixOmics package are copied into this repository. `run_reference.R`
  is original code written from scratch that calls into mixOmics
  via its public API.
- The reference numbers are a mathematical decomposition of
  user-owned input data. They are facts, not creative expression of
  the program (Feist v. Rural, 499 U.S. 340; analogous treatment
  under EU sui generis database protection does not apply to small
  computed result sets like this).
- The relevant FSF position is that program output is not, in
  general, covered by the copyright on the program. See
  https://www.gnu.org/licenses/gpl-faq.html#WhatCaseIsOutputGPL.

The mixOmics package itself is GPL-2 and remains a separately
installed runtime dependency of `run_reference.R`. It is not
redistributed by `process-improve`. Re-running the reference script
on a different machine requires the user to install mixOmics from
Bioconductor under its own licence terms.

Attribution for the reference implementation:

Rohart, F., Gautier, B., Singh, A. and Le Cao, K.-A. *mixOmics:
An R package for omics feature selection and multiple data
integration.* PLoS Computational Biology 13(11): e1005752 (2017).
https://doi.org/10.1371/journal.pcbi.1005752

This provenance note is informational and does not constitute legal
advice. If the project has formally vetted licensing policy that
contradicts the analysis above, that policy controls.
