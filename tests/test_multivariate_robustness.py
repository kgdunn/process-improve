"""Robustness regression tests for SEC-05 and SEC-06 (multivariate methods).

SEC-05: NIPALS / multiblock divisions are guarded so a degenerate or
fully-deflated block yields finite (~0) projections instead of silently
producing ``inf``/``nan``.

SEC-06: MBPLS / MBPCA record per-component convergence in ``fitting_info_``
and warn when ``max_iter`` is hit; ``Resampler.fractional`` re-validates
``fraction_excluded`` so a mutated-to-zero value raises rather than dividing
by zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

# ``_nz`` is a private helper; import from the implementation module directly
# rather than via the ``methods.py`` re-exporter (which only exposes public
# names via ``import *``).
from process_improve.multivariate._pca_pls import _nz
from process_improve.multivariate.methods import (
    MBPCA,
    MBPLS,
    DataFrameDict,
    Resampler,
    SpecificationWarning,
)

_DENOM_FLOOR = float(np.finfo(float).tiny)


# ---------------------------------------------------------------------------
# _nz helper
# ---------------------------------------------------------------------------


class TestNzHelper:
    def test_passes_through_normal_values(self) -> None:
        assert _nz(5.0) == 5.0
        assert _nz(1e-6) == 1e-6

    def test_floors_zero(self) -> None:
        assert _nz(0.0) == _DENOM_FLOOR

    def test_unguarded_zero_division_is_nan(self) -> None:
        # Documents the bug this guard prevents.
        v = np.zeros(3)
        with np.errstate(invalid="ignore"):
            assert np.all(np.isnan(v / np.linalg.norm(v)))
        # Guarded form is finite.
        assert np.all(np.isfinite(v / _nz(np.linalg.norm(v))))


# ---------------------------------------------------------------------------
# SEC-05: degenerate block must not poison the model with inf/nan
# ---------------------------------------------------------------------------


def _blocks_with_constant_block(n: int = 15) -> dict[str, pd.DataFrame]:
    """One normal block plus one entirely-constant block.

    A constant block centres to exactly zero, so its loading vector collapses
    and the normalisation ``p_b / norm(p_b)`` would be ``0/0`` without the guard.
    """
    rng = np.random.default_rng(0)
    return {
        "A": pd.DataFrame(rng.normal(size=(n, 3)), columns=["a1", "a2", "a3"]),
        "B": pd.DataFrame(np.full((n, 2), 7.0), columns=["b1", "b2"]),
    }


class TestDegenerateBlockStaysFinite:
    def test_mbpca_constant_block_finite(self) -> None:
        model = MBPCA(n_components=2).fit(_blocks_with_constant_block())
        assert np.all(np.isfinite(model.super_scores_.values))
        for loadings in model.block_loadings_.values():
            assert np.all(np.isfinite(loadings.values))
        for scores in model.block_scores_.values():
            assert np.all(np.isfinite(scores.values))

    def test_mbpls_constant_block_finite(self) -> None:
        blocks = _blocks_with_constant_block()
        rng = np.random.default_rng(1)
        y = pd.DataFrame(rng.normal(size=(15, 1)), columns=["y"])
        model = MBPLS(n_components=2).fit(blocks, y)
        assert np.all(np.isfinite(model.super_scores_.values))
        assert np.all(np.isfinite(model.predictions_.values))
        for loadings in model.block_loadings_.values():
            assert np.all(np.isfinite(loadings.values))


# ---------------------------------------------------------------------------
# SEC-06: convergence reporting
# ---------------------------------------------------------------------------


def _clean_blocks(n: int = 20) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(n, 2))
    a = latent @ rng.normal(size=(2, 3)) + 0.05 * rng.normal(size=(n, 3))
    b = latent @ rng.normal(size=(2, 2)) + 0.05 * rng.normal(size=(n, 2))
    return {
        "A": pd.DataFrame(a, columns=["a1", "a2", "a3"]),
        "B": pd.DataFrame(b, columns=["b1", "b2"]),
    }


class TestConvergenceFlag:
    def test_mbpca_reports_converged_on_clean_fit(self) -> None:
        model = MBPCA(n_components=2).fit(_clean_blocks())
        converged = model.fitting_info_["converged"]
        assert converged.dtype == bool
        assert len(converged) == 2
        assert np.all(converged)

    def test_mbpca_flags_non_convergence_and_warns(self) -> None:
        with pytest.warns(SpecificationWarning, match="did not converge"):
            model = MBPCA(n_components=2, max_iter=1).fit(_clean_blocks())
        assert not np.all(model.fitting_info_["converged"])

    def test_mbpls_reports_converged_on_clean_fit(self) -> None:
        rng = np.random.default_rng(5)
        y = pd.DataFrame(rng.normal(size=(20, 1)), columns=["y"])
        model = MBPLS(n_components=2).fit(_clean_blocks(), y)
        assert "converged" in model.fitting_info_
        assert np.all(model.fitting_info_["converged"])

    def test_mbpls_flags_non_convergence_and_warns(self) -> None:
        rng = np.random.default_rng(6)
        y = pd.DataFrame(rng.normal(size=(20, 1)), columns=["y"])
        with pytest.warns(SpecificationWarning, match="did not converge"):
            model = MBPLS(n_components=2, max_iter=1).fit(_clean_blocks(), y)
        assert not np.all(model.fitting_info_["converged"])


# ---------------------------------------------------------------------------
# SEC-06: Resampler.fractional re-validates fraction_excluded
# ---------------------------------------------------------------------------


class _StubEstimator(BaseEstimator):
    def __init__(self, dummy: int = 1) -> None:
        self.dummy = dummy

    def fit(self, X: DataFrameDict, y: object | None = None) -> _StubEstimator:  # noqa: ARG002
        self.n_train_samples_ = len(X)
        return self


def _accessor(estimator: _StubEstimator) -> dict[str, float]:
    return {"n": float(estimator.n_train_samples_)}


@pytest.fixture
def tiny_dfd() -> DataFrameDict:
    n = 10
    rng = np.random.default_rng(0)
    return DataFrameDict(
        {
            "F": {"main": pd.DataFrame(rng.standard_normal((n, 2)), columns=["f1", "f2"])},
            "Z": {"conds": pd.DataFrame(rng.standard_normal((n, 1)), columns=["z1"])},
            "Y": {"out": pd.DataFrame(rng.standard_normal((n, 1)), columns=["y1"])},
        }
    )


class TestFractionalGuard:
    def test_fraction_excluded_zero_raises(self, tiny_dfd: DataFrameDict) -> None:
        # Default fraction_excluded is 0.0; calling fractional() directly must
        # not divide by zero.
        r = Resampler(estimator=_StubEstimator(), x=tiny_dfd, accessor=_accessor, use_jackknife=False)
        with pytest.raises(ValueError, match="fraction_excluded"):
            r.fractional(show_progress=False)

    def test_mutated_to_zero_after_init_raises(self, tiny_dfd: DataFrameDict) -> None:
        # The __init__ guard can be bypassed by mutating the attribute; the
        # method-level guard catches it.
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
            fraction_excluded=0.5,
        )
        r.fraction_excluded = 0.0
        with pytest.raises(ValueError, match="fraction_excluded"):
            r.fractional(show_progress=False)
