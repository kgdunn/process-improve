"""Perf baselines for the ENG-18 ndarray-backed fitted attributes.

After ENG-18 (#300) the hot-path fitted attributes (``loadings_`` / ``scores_`` /
``spe_`` / ``x_loadings_`` / ``x_weights_``) are stored as private numpy ndarrays,
and the internal math (``transform`` / ``predict`` / ``score_contributions``)
reads those arrays directly instead of paying a ``DataFrame.values`` conversion
on every call. These baselines track regressions on those paths, per the ENG-15
perf pattern (one test per hot path, smallest representative input,
``test_<func>_baseline`` naming).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from process_improve.multivariate.methods import PCA, PLS, MCUVScaler


def _scaled_x(n: int = 200, k: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((n, k))))


def test_pca_transform_baseline(benchmark) -> None:  # type: ignore[no-untyped-def]
    """``transform`` projects via the private ``_loadings`` ndarray (no per-call .values)."""
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)
    benchmark(model.transform, x)


def test_pca_predict_baseline(benchmark) -> None:  # type: ignore[no-untyped-def]
    """PCA ``predict`` reconstructs via the private ``_loadings`` ndarray."""
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)
    benchmark(model.predict, x)


def test_pls_predict_baseline(benchmark) -> None:  # type: ignore[no-untyped-def]
    """PLS ``predict`` reconstructs via the private ``_x_loadings`` ndarray."""
    rng = np.random.default_rng(1)
    x = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((200, 20))))
    y = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((200, 1))))
    model = PLS(n_components=5).fit(x, y)
    benchmark(model.predict, x)
