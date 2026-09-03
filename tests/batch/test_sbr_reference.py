"""Cross-check a batch PCA on the SBR data against the legacy MATLAB toolkit's output.

The fixture in ``tests/fixtures/sbr_batch_pca/`` holds the scores and loadings
that ``unit_tests.m::PCA_batch_data`` of the 2011 toolkit asserted: a
two-component PCA on all nine trajectories of the 53 batches, every unfolded
column mean-centred and scaled to unit variance. MATLAB unfolds time-major;
the fixture is labelled ``(tag, sequence)`` so it can be reindexed onto the
tag-major layout that :func:`process_improve.batch.dict_to_wide` produces.
Principal components are defined up to their sign, so each column is
sign-aligned before comparing, as the legacy ``assertEAE`` did.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import dict_to_wide, load_sbr
from process_improve.multivariate import PCA, MCUVScaler
from tests._case_study_scripts import SBR_URL_OVERRIDE, load_or_skip

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "sbr_batch_pca"

pytestmark = [pytest.mark.dataset, pytest.mark.slow]


@pytest.fixture(scope="module")
def fitted() -> PCA:
    """PCA(2) on the MCUV-scaled unfolded trajectories, all nine tags, columns re-labelled."""
    sbr = load_or_skip(lambda: load_sbr(url=SBR_URL_OVERRIDE))
    wide = dict_to_wide(sbr.X)
    scaled = MCUVScaler().fit_transform(wide)
    scaled.columns = wide.columns  # MCUVScaler drops the 2-level labels
    return PCA(n_components=2).fit(scaled)


def _sign_aligned(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each column of ``candidate`` so it points the same way as ``reference``."""
    signs = np.sign(np.nansum(candidate * reference, axis=0))
    signs[signs == 0] = 1.0
    return candidate * signs


def test_r2_matches_legacy_matlab(fitted: PCA) -> None:
    """The per-component R2 the legacy unit test asserts to five decimals."""
    metadata = json.loads((FIXTURE_DIR / "reference_metadata.json").read_text())
    np.testing.assert_allclose(fitted.r2_per_component_.to_numpy(), metadata["legacy_r2_per_component"], atol=1e-5)


def test_scores_match_legacy_matlab(fitted: PCA) -> None:
    """Batch scores agree with the MATLAB scores (sign-agnostic, absolute 1e-2 on scores of size 10-40)."""
    reference = pd.read_csv(FIXTURE_DIR / "reference_scores.csv", index_col="batch_id")
    scores = fitted.scores_.reindex(reference.index).to_numpy()
    np.testing.assert_allclose(_sign_aligned(scores, reference.to_numpy()), reference.to_numpy(), atol=1e-2)


def test_loadings_match_legacy_matlab(fitted: PCA) -> None:
    """Loadings agree cell by cell once the time-major fixture is reindexed onto the tag-major layout."""
    reference = pd.read_csv(FIXTURE_DIR / "reference_loadings.csv").set_index(["tag", "sequence"])
    reference = reference.reindex(fitted.loadings_.index)
    assert not reference.isna().any().any(), "every fitted (tag, sequence) cell must exist in the fixture"
    loadings = _sign_aligned(fitted.loadings_.to_numpy(), reference.to_numpy())
    np.testing.assert_allclose(loadings, reference.to_numpy(), atol=1e-4)
    # The two constant feed-flow cells at sample 0 have exactly zero loadings on both sides.
    zero_cells = reference.index[(reference == 0).all(axis=1)]
    assert set(zero_cells) == {("StyreneFlow", 0), ("ButadieneFlow", 0)}
