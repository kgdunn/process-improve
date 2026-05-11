"""Cross-check process_improve's NIPALS-with-NaN PCA against mixOmics::nipals.

mixOmics is the reference implementation of skip-NaN NIPALS in R; it
implements Wold's original 1966 algorithm directly and is widely used
in chemometrics and bioinformatics. Locking process_improve's
output to mixOmics on a small public matrix gives an independent,
implementation-different check of the inner loop now shared by PCA
and MBPLS.

The cross-check fixture lives in ``tests/fixtures/mixomics_nipals/``.
See ``tests/fixtures/mixomics_nipals/README.md`` for the regeneration
recipe. Until the R-side reference CSVs land, this test module is
auto-skipped (we only assert sign-agnostic numerical agreement once
both sides have been computed on the same canonical centered matrix).

References
----------
Wold, H. "Estimation of principal components and related models by
iterative least squares." In Krishnaiah, P. R. (ed.), *Multivariate
Analysis*, Academic Press, 1966, pp. 391-420.

Rohart, F., Gautier, B., Singh, A. and Le Cao, K.-A. "mixOmics: An R
package for omics feature selection and multiple data integration."
PLoS Computational Biology 13(11): e1005752 (2017).
https://doi.org/10.1371/journal.pcbi.1005752
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "mixomics_nipals"
CANONICAL_CSV = FIXTURE_DIR / "linnerud_centered_with_nan.csv"
REFERENCE_LOADINGS_CSV = FIXTURE_DIR / "reference_loadings.csv"
REFERENCE_SCORES_CSV = FIXTURE_DIR / "reference_scores.csv"

# Tolerance: mixOmics::nipals default tol is 1e-09 and process_improve's
# PCA NIPALS inner loop uses epsqrt (about 1.49e-08). After both have
# converged the per-element agreement is typically much tighter than
# 1e-6, but we pick a conservative bound to absorb the small slack from
# two independent termination criteria.
COMPONENT_TOLERANCE = 1e-6


pytestmark = pytest.mark.skipif(
    not (REFERENCE_LOADINGS_CSV.exists() and REFERENCE_SCORES_CSV.exists()),
    reason=(
        "mixOmics reference fixture not yet generated; run "
        "`Rscript tests/fixtures/mixomics_nipals/run_reference.R` "
        "after `python tests/fixtures/mixomics_nipals/prepare_fixture.py`"
    ),
)


def _align_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Flip the sign of each candidate column to match the reference column.

    NIPALS components are sign-ambiguous; mixOmics and process_improve
    can converge to columns that differ by an overall sign. We pick the
    sign that minimises the column residual against the reference.
    """
    flipped = candidate.copy()
    for j in range(candidate.shape[1]):
        if np.dot(reference[:, j], candidate[:, j]) < 0:
            flipped[:, j] = -candidate[:, j]
    return flipped


@pytest.fixture(scope="module")
def canonical_matrix() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_CSV)


@pytest.fixture(scope="module")
def reference() -> tuple[np.ndarray, np.ndarray]:
    loadings = pd.read_csv(REFERENCE_LOADINGS_CSV)
    # The R script prefixes the loadings table with a 'variable' name
    # column; strip it before turning into a numeric array.
    if loadings.columns[0] == "variable":
        loadings = loadings.drop(columns="variable")
    scores = pd.read_csv(REFERENCE_SCORES_CSV)
    return loadings.to_numpy(), scores.to_numpy()


def test_loadings_match_mixomics(canonical_matrix, reference) -> None:
    ref_p, _ = reference
    model = PCA(n_components=ref_p.shape[1], algorithm="nipals").fit(canonical_matrix)
    candidate = _align_sign(ref_p, model.loadings_.to_numpy())
    np.testing.assert_allclose(candidate, ref_p, atol=COMPONENT_TOLERANCE)


def test_scores_match_mixomics(canonical_matrix, reference) -> None:
    _, ref_t = reference
    model = PCA(n_components=ref_t.shape[1], algorithm="nipals").fit(canonical_matrix)
    # Use the same sign alignment derived from loadings; scores carry
    # the matching sign flip.
    ref_p, _ = reference
    raw_p = model.loadings_.to_numpy()
    flip = np.sign(np.einsum("ij,ij->j", ref_p, raw_p))
    flip[flip == 0] = 1.0
    candidate = model.scores_.to_numpy() * flip
    np.testing.assert_allclose(candidate, ref_t, atol=COMPONENT_TOLERANCE)


def test_fixture_matrix_is_centered(canonical_matrix) -> None:
    """The canonical input must be column-mean-centered using nanmean."""
    nan_means = np.nanmean(canonical_matrix.to_numpy(), axis=0)
    np.testing.assert_allclose(nan_means, 0.0, atol=1e-9)
