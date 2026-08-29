"""Deterministic performance-shape assertions for the ENG-18 fitted attributes.

After ENG-18 (#300) the hot-path fitted attributes (``loadings_`` / ``scores_``
/ ``spe_`` / ``x_loadings_`` / ``x_weights_``) are stored as private numpy
ndarrays behind ``_LazyFrame`` descriptors, and the internal math (``transform``
/ ``predict``) reads those arrays directly instead of paying a
``DataFrame.values`` conversion on every call.

These tests protect that design with assertions that fail deterministically on
any CI runner (#511): the public DataFrame view is built exactly once and then
cached (repeated access returns the identical object), pickling excludes the
cache, and ``transform`` / ``predict`` never touch the public frame views at
all. They are not wall-clock benchmarks; timing-based regression gates remain
the planned ENG-15 CI job (see CONTRIBUTING.md, "Performance-regression
policy").
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from process_improve.multivariate import _base
from process_improve.multivariate.methods import PCA, PLS, MCUVScaler


def _scaled_x(n: int = 200, k: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((n, k))))


def _fitted_pls() -> tuple[PLS, pd.DataFrame]:
    rng = np.random.default_rng(1)
    x = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((200, 20))))
    y = MCUVScaler().fit_transform(pd.DataFrame(rng.standard_normal((200, 1))))
    return PLS(n_components=5).fit(x, y), x


def _count_lazyframe_builds(monkeypatch, calls: list[str]) -> None:
    """Patch ``pd.DataFrame`` as seen by ``_base`` to record every lazy-frame build."""
    real_frame = pd.DataFrame

    def counting(*args, **kwargs):
        calls.append("build")
        return real_frame(*args, **kwargs)

    monkeypatch.setattr(_base.pd, "DataFrame", counting)


def test_pca_scores_frame_built_once_and_cached(monkeypatch) -> None:
    """``scores_`` builds its DataFrame on first access only; repeats hit the cache.

    Losing the ``_LazyFrame`` cache (a rebuild per access) is the regression
    the old wall-clock benchmark was meant to catch; this asserts it directly.
    """
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)

    calls: list[str] = []
    _count_lazyframe_builds(monkeypatch, calls)

    first = model.scores_
    second = model.scores_
    assert first is second, "repeated scores_ access must return the cached object"
    assert len(calls) == 1, f"scores_ DataFrame built {len(calls)} times; expected exactly once"


def test_pca_loadings_frame_built_once_and_cached(monkeypatch) -> None:
    """``loadings_`` is cached identically to ``scores_``."""
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)

    calls: list[str] = []
    _count_lazyframe_builds(monkeypatch, calls)

    assert model.loadings_ is model.loadings_
    assert len(calls) == 1


def _record_frame_rebuilds(monkeypatch, builds: list[str]) -> None:
    """Record every ``_LazyFrame`` access that misses the cache (a frame build).

    Cache *hits* are free and not recorded: fittedness probes such as
    ``check_is_fitted(self, "scores_")`` legitimately go through the
    descriptor. What ENG-18 forbids is rebuilding a DataFrame per call.
    """
    original_get = _base._LazyFrame.__get__

    def recording_get(self, obj, objtype=None):
        if obj is not None:
            cache = obj.__dict__.get("_frame_cache") or {}
            if self._public_name not in cache:
                builds.append(self._public_name)
        return original_get(self, obj, objtype)

    monkeypatch.setattr(_base._LazyFrame, "__get__", recording_get)


def test_pca_transform_and_diagnose_build_no_frames_per_call(monkeypatch) -> None:
    """Repeated ``transform`` / ``diagnose`` calls trigger zero lazy-frame builds.

    The ENG-18 point is that the hot paths read the private ndarrays and skip
    the per-call ``DataFrame`` conversion. After one warm-up call (which may
    lazily build a view once, e.g. via ``check_is_fitted``), further calls
    must cause no cache misses; a regression that routes the math back
    through ``scores_`` / ``loadings_`` rebuilds shows up here.
    """
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)
    model.transform(x)
    model.diagnose(x)  # warm-up: any one-time lazy builds happen here

    builds: list[str] = []
    _record_frame_rebuilds(monkeypatch, builds)

    model.transform(x)
    model.diagnose(x)
    model.transform(x)
    assert builds == [], f"hot paths rebuilt public frame views per call: {builds}"


def test_pls_predict_builds_no_frames_per_call(monkeypatch) -> None:
    """Repeated PLS ``predict`` calls reconstruct via the private ndarrays only."""
    model, x = _fitted_pls()
    model.predict(x)  # warm-up: any one-time lazy builds happen here

    builds: list[str] = []
    _record_frame_rebuilds(monkeypatch, builds)

    model.predict(x)
    model.predict(x)
    assert builds == [], f"PLS.predict rebuilt public frame views per call: {builds}"


def test_pickling_excludes_the_frame_cache() -> None:
    """The lazily-built DataFrame cache never travels through pickle.

    ``__getstate__`` drops ``_frame_cache`` so pickles stay small and the
    ndarrays remain the single source of truth; the views rebuild on demand
    after unpickling and match the originals.
    """
    x = _scaled_x()
    model = PCA(n_components=5).fit(x)
    _ = model.scores_  # populate the cache
    assert "_frame_cache" in model.__dict__

    state = model.__getstate__()
    assert "_frame_cache" not in state

    clone = pickle.loads(pickle.dumps(model))  # noqa: S301 - round-tripping our own object
    assert "_frame_cache" not in clone.__dict__
    pd.testing.assert_frame_equal(clone.scores_, model.scores_)
    pd.testing.assert_frame_equal(clone.loadings_, model.loadings_)
