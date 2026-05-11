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
