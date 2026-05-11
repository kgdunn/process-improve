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
auto-skipped.

Normalisation conventions
-------------------------
The two implementations use unit-norm loadings (the standard
chemometrics convention) but differ in how they scale the scores:

- ``process_improve.PCA`` returns ``T = X @ P`` directly, so column
  ``j`` of ``T`` has L2 norm equal to the singular value of the
  ``j``-th component. The squared norm equals the eigenvalue that
  ``mixOmics::nipals`` stores in ``$eig``.
- ``mixOmics::nipals`` returns scores normalised to unit L2 norm per
  column and a separate ``$eig`` vector with the singular values.

The two are therefore identical up to a per-column scalar
(the singular value) and a per-column sign (NIPALS sign ambiguity).
The score-agreement test below normalises both sides to unit L2 norm
per column before comparing.

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

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "mixomics_nipals"
CANONICAL_CSV = FIXTURE_DIR / "linnerud_centered_with_nan.csv"
REFERENCE_LOADINGS_CSV = FIXTURE_DIR / "reference_loadings.csv"
REFERENCE_SCORES_CSV = FIXTURE_DIR / "reference_scores.csv"
REFERENCE_METADATA_JSON = FIXTURE_DIR / "reference_metadata.json"

# Tolerance: mixOmics::nipals and process_improve both run to a
# per-component convergence floor of ~1e-9, but they differ in how
# they deflate the residual and in which quantity they compare
# between inner-loop iterations. On a real NaN-containing matrix the
# two converge to loadings that agree to roughly five decimal places
# rather than to either implementation's own tolerance. 1e-4 is the
# defensible cross-implementation bound.
COMPONENT_TOLERANCE = 1e-4

# Eigenvalues come from the reference_metadata.json file as a printed
# JSON number with at most four decimal digits, so the singular-value
# tolerance is correspondingly looser.
EIGENVALUE_TOLERANCE = 1e-3


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


def _unit_norm_columns(a: np.ndarray) -> np.ndarray:
    """Return a copy of ``a`` with each column scaled to unit L2 norm."""
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    return a / norms


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
    """Score columns must agree up to sign and per-column scale.

    process_improve returns ``T = X @ P`` (column norm = singular value)
    while mixOmics returns unit-norm score columns. Normalise both to
    unit-norm per column, then compare with the loadings-derived sign.
    """
    ref_p, ref_t = reference
    model = PCA(n_components=ref_t.shape[1], algorithm="nipals").fit(canonical_matrix)

    raw_p = model.loadings_.to_numpy()
    flip = np.sign(np.einsum("ij,ij->j", ref_p, raw_p))
    flip[flip == 0] = 1.0
    candidate = _unit_norm_columns(model.scores_.to_numpy()) * flip
    np.testing.assert_allclose(candidate, _unit_norm_columns(ref_t), atol=COMPONENT_TOLERANCE)


def test_singular_values_match_mixomics_eigenvalues(canonical_matrix) -> None:
    """``mixOmics::nipals$eig`` stores the per-component singular values.

    process_improve's score-column L2 norm equals that same singular
    value (because loadings are unit-norm and ``T = X @ P``). Pinning
    this relationship closes the loop: the loadings agree, the scores
    agree up to sign and scale, and the scale factor itself is what
    mixOmics calls the eigenvalue.
    """
    metadata = json.loads(REFERENCE_METADATA_JSON.read_text())
    ref_eig = np.asarray(metadata["eigenvalues"], dtype=float)

    model = PCA(n_components=len(ref_eig), algorithm="nipals").fit(canonical_matrix)
    pi_score_norms = np.linalg.norm(model.scores_.to_numpy(), axis=0)
    np.testing.assert_allclose(pi_score_norms, ref_eig, atol=EIGENVALUE_TOLERANCE)


def test_fixture_matrix_is_centered(canonical_matrix) -> None:
    """The canonical input must be column-mean-centered using nanmean."""
    nan_means = np.nanmean(canonical_matrix.to_numpy(), axis=0)
    np.testing.assert_allclose(nan_means, 0.0, atol=1e-9)
