"""The missing-data machinery, run on data without missing values, reproduces the complete-data results.

Every estimator that exists for rows with missing cells (the NIPALS and TSR fit paths of PCA; the TSR, SCP and
PMP score operators behind ``project``, ``predict_online`` and the contribution helpers) is run here with nothing
missing and compared with the direct computation, so that a row with one missing cell cannot land far from where
the same row lands with none. The data are the bundled dryer batches (71 complete batches), the DuPont batches
(remote) and the SBR batches (remote or a local copy through ``PROCESS_IMPROVE_SBR_URL``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import BatchPCA, BatchPLS, dict_to_wide
from process_improve.batch.datasets import load_dryer
from process_improve.batch.preprocessing import resample_to_reference
from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._projection import operator_for_pattern
from tests._case_study_scripts import SBR_URL_OVERRIDE, load_or_skip, load_script

pytestmark = pytest.mark.dataset

METHODS = ("tsr", "scp", "pmp")


def _align_signs(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the columns of ``values`` whose sign disagrees with ``reference`` (a loading's sign is arbitrary)."""
    signs = np.sign(np.sum(values * reference, axis=0))
    signs[signs == 0] = 1.0
    return values * signs


def _full_mask_scores(model, x: np.ndarray, method: str) -> np.ndarray:
    """Scores from the missing-data operator built for the pattern in which every feature is observed."""
    is_pls = hasattr(model, "direct_weights_")
    loadings = np.asarray(model.x_loadings_ if is_pls else model.loadings_, dtype=float)
    guide = np.asarray(model.direct_weights_ if is_pls else model.loadings_, dtype=float)
    weights = np.asarray(model.x_weights_, dtype=float) if is_pls else None
    everything = np.ones(loadings.shape[0], dtype=bool)
    variances = np.asarray(model.explained_variance_, dtype=float)
    op = operator_for_pattern(loadings, guide, variances, everything, method=method, x_weights=weights)
    return x @ op.matrix.T


@pytest.fixture(scope="module")
def dryer_scaled() -> pd.DataFrame:
    """Return the bundled dryer batches, aligned and unfolded batchwise, centred and scaled: no missing cells."""
    batches = load_dryer()
    tags = [c for c in next(iter(batches.values())).columns if c != "ClockTime"]
    aligned = resample_to_reference({k: v[tags] for k, v in batches.items()}, columns_to_align=tags, reference_batch=1)
    wide = dict_to_wide(aligned)
    scaled = MCUVScaler().fit_transform(wide)
    scaled.columns = wide.columns
    assert not scaled.isna().any().any()
    return scaled


@pytest.fixture(scope="module")
def dryer_quality(dryer_scaled: pd.DataFrame) -> pd.DataFrame:
    """Return a two-column quality block driven by the trajectories, centred and scaled."""
    rng = np.random.default_rng(3)
    x = dryer_scaled.to_numpy()
    y = x @ rng.normal(size=(x.shape[1], 2)) / np.sqrt(x.shape[1]) + 0.1 * rng.normal(size=(x.shape[0], 2))
    return MCUVScaler().fit_transform(pd.DataFrame(y, index=dryer_scaled.index, columns=["y1", "y2"]))


@pytest.fixture(scope="module")
def pca_svd(dryer_scaled: pd.DataFrame) -> PCA:
    return PCA(n_components=3, algorithm="svd").fit(dryer_scaled)


@pytest.fixture(scope="module")
def pls(dryer_scaled: pd.DataFrame, dryer_quality: pd.DataFrame) -> PLS:
    return PLS(n_components=3, scale=False).fit(dryer_scaled, dryer_quality)


@pytest.mark.parametrize(("algorithm", "tol"), [("nipals", 1e-4), ("tsr", 1e-9)])
def test_pca_missing_data_fit_paths_reproduce_svd(
    dryer_scaled: pd.DataFrame, pca_svd: PCA, algorithm: str, tol: float
) -> None:
    """Fitted by NIPALS or TSR on complete data, a PCA is the SVD model (NIPALS to its convergence tolerance)."""
    model = PCA(n_components=3, algorithm=algorithm).fit(dryer_scaled)
    np.testing.assert_allclose(model.r2_cumulative_, pca_svd.r2_cumulative_, atol=tol)
    np.testing.assert_allclose(model.r2_per_variable_, pca_svd.r2_per_variable_, atol=tol)
    np.testing.assert_allclose(model.spe_, pca_svd.spe_, atol=tol)
    np.testing.assert_allclose(model.hotellings_t2_, pca_svd.hotellings_t2_, atol=tol)
    loadings, scores = model.loadings_.to_numpy(), model.scores_.to_numpy()
    np.testing.assert_allclose(_align_signs(loadings, pca_svd.loadings_.to_numpy()), pca_svd.loadings_, atol=tol)
    np.testing.assert_allclose(_align_signs(scores, pca_svd.scores_.to_numpy()), pca_svd.scores_, atol=tol)


@pytest.mark.parametrize("method", METHODS)
def test_pca_operators_with_everything_observed_give_the_scores(
    dryer_scaled: pd.DataFrame, pca_svd: PCA, method: str
) -> None:
    np.testing.assert_allclose(_full_mask_scores(pca_svd, dryer_scaled.to_numpy(), method), pca_svd.scores_, atol=1e-10)


@pytest.mark.parametrize("method", ["tsr", "scp"])
def test_pls_operators_with_everything_observed_give_the_scores(
    dryer_scaled: pd.DataFrame, pls: PLS, method: str
) -> None:
    """TSR reduces to the score map exactly; SCP is NIPALS's own score step, projecting onto the weights."""
    np.testing.assert_allclose(_full_mask_scores(pls, dryer_scaled.to_numpy(), method), pls.scores_, atol=1e-9)


def test_pls_pmp_is_the_least_squares_fit_onto_the_loadings(dryer_scaled: pd.DataFrame, pls: PLS) -> None:
    """PMP fits the observed columns onto the loading plane, which for PLS is not the score map (documented)."""
    x = dryer_scaled.to_numpy()
    loadings = pls.x_loadings_.to_numpy()
    expected = np.linalg.solve(loadings.T @ loadings, loadings.T @ x.T).T
    np.testing.assert_allclose(_full_mask_scores(pls, x, "pmp"), expected, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_project_on_complete_rows_matches_the_fit(
    dryer_scaled: pd.DataFrame, pca_svd: PCA, pls: PLS, method: str
) -> None:
    for model in (pca_svd, pls):
        result = model.project(dryer_scaled, method=method)
        np.testing.assert_allclose(result.scores, model.scores_, atol=1e-10)
        np.testing.assert_allclose(result.spe, model.spe_.iloc[:, -1], atol=1e-10)
        np.testing.assert_allclose(result.hotellings_t2, model.hotellings_t2_.iloc[:, -1], atol=1e-10)
        assert (result.n_observed == dryer_scaled.shape[1]).all()
    np.testing.assert_allclose(pls.project(dryer_scaled, method=method).y_hat, pls.predict(dryer_scaled), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_contributions_ignore_the_method_on_complete_rows(
    dryer_scaled: pd.DataFrame, pca_svd: PCA, pls: PLS, method: str
) -> None:
    for model in (pca_svd, pls):
        for name in ("score_contributions", "spe_contributions", "t2_contributions"):
            direct = getattr(model, name)(dryer_scaled)
            np.testing.assert_array_equal(getattr(model, name)(dryer_scaled, method=method), direct)


def test_pls_scp_reproduces_the_nipals_scores_of_incomplete_rows(
    dryer_scaled: pd.DataFrame, dryer_quality: pd.DataFrame
) -> None:
    """With a few cells missing, ``project(method="scp")`` returns the scores the NIPALS fit stored for those rows."""
    x = dryer_scaled.copy()
    x.iloc[3, 5] = np.nan
    x.iloc[10, [100, 101]] = np.nan
    model = PLS(n_components=3, scale=False).fit(x, dryer_quality)
    gap = np.abs(model.project(x, method="scp").scores.to_numpy() - model.scores_.to_numpy())
    incomplete = np.zeros(len(x), dtype=bool)
    incomplete[[3, 10]] = True
    assert gap[incomplete].max() <= 10 * gap[~incomplete].max() + 1e-9  # as close as the complete rows, which differ
    assert gap.max() < 1e-2  # only by the convergence tolerance of the final weights and loadings


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["nipals", "tsr"])
def test_dupont_batch_pca_paths_agree(algorithm: str) -> None:
    """The DuPont case (complete data) gives the same batch PCA on every fit path."""
    script = load_script("dupont_batch_pca")
    batches = load_or_skip(script.load_data)
    reference = BatchPCA(n_components=3).fit(batches)
    model = BatchPCA(n_components=3, algorithm=algorithm).fit(batches)
    tol = 1e-4 if algorithm == "nipals" else 1e-9
    np.testing.assert_allclose(model.r2_cumulative_, reference.r2_cumulative_, atol=tol)
    np.testing.assert_allclose(model.spe_, reference.spe_, atol=tol)
    np.testing.assert_allclose(model.hotellings_t2_, reference.hotellings_t2_, atol=tol)
    scores = _align_signs(model.scores_.to_numpy(), reference.scores_.to_numpy())
    np.testing.assert_allclose(scores, reference.scores_, atol=tol)


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_sbr_online_prediction_of_a_complete_batch_is_the_prediction(method: str) -> None:
    """The SBR case (complete data): every estimator, given the whole batch, returns the model's own prediction."""
    script = load_script("sbr_batch_pls")
    batches, quality = load_or_skip(lambda: script.load_data(SBR_URL_OVERRIDE))
    model = BatchPLS(n_components=2).fit(batches, quality)
    batch_id = next(iter(batches))
    whole = model.predict_online(batches[batch_id], upto_k=model.n_timesteps_, method=method)
    np.testing.assert_allclose(whole.y_hat, model.predictions_.loc[batch_id], rtol=1e-10)
    np.testing.assert_allclose(whole.scores, model.scores_.loc[batch_id], rtol=1e-10)
    if method != "pmp":  # the operator itself, with everything observed, is the score map for TSR and SCP
        x = model.unfold_and_scale({batch_id: batches[batch_id]}).to_numpy()
        np.testing.assert_allclose(_full_mask_scores(model._pls, x, method), model.scores_.loc[[batch_id]], atol=1e-9)
