"""ENG-05 (#287): the convenience plot / limit / diagnostic bindings on the
multivariate estimators must be real, introspectable methods rather than
``functools.partial`` instances.

Acceptance: ``help`` / ``inspect.signature`` report the underlying function
(minus the model parameter), fitted models pickle round-trip, and subclasses can
override the methods.
"""

from __future__ import annotations

import functools
import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import plots as _plots
from process_improve.multivariate.methods import MBPCA, MBPLS, PCA, PLS, MCUVScaler


def _fitted_pca() -> PCA:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=list("abcde"))
    return PCA(n_components=2).fit(MCUVScaler().fit_transform(X))


def _fitted_pls() -> PLS:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=list("abcde"))
    Y = pd.DataFrame(rng.standard_normal((20, 2)), columns=["y1", "y2"])
    return PLS(n_components=2).fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))


def _two_block() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((30, 2))
    a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((30, 6))
    b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((30, 4))
    return {
        "A": pd.DataFrame(a, columns=[f"a{i}" for i in range(6)]),
        "B": pd.DataFrame(b, columns=[f"b{i}" for i in range(4)]),
    }


def _fitted_mbpca() -> MBPCA:
    return MBPCA(n_components=2).fit(_two_block())


def _fitted_mbpls() -> MBPLS:
    rng = np.random.default_rng(3)
    y = pd.DataFrame(rng.standard_normal((30, 1)), columns=["y"])
    return MBPLS(n_components=2).fit(_two_block(), y)


@pytest.mark.parametrize("factory", [_fitted_pca, _fitted_pls, _fitted_mbpca, _fitted_mbpls])
def test_fitted_model_pickle_round_trips(factory) -> None:
    """A fitted model pickles and the reloaded model's limit method agrees."""
    model = factory()
    reloaded = pickle.loads(pickle.dumps(model))  # noqa: S301 - trusted round-trip of our own object
    assert float(reloaded.hotellings_t2_limit()) == pytest.approx(float(model.hotellings_t2_limit()))


def test_convenience_bindings_are_not_partial() -> None:
    """None of the bound convenience callables is a functools.partial."""
    model = _fitted_pca()
    for name in ("score_plot", "spe_plot", "vip", "spe_limit", "hotellings_t2_limit", "ellipse_coordinates"):
        assert not isinstance(getattr(model, name), functools.partial), name


def test_score_plot_signature_and_doc() -> None:
    """``score_plot`` exposes the underlying signature (minus model) and docstring."""
    model = _fitted_pca()
    sig = inspect.signature(model.score_plot)
    assert "model" not in sig.parameters
    assert "self" not in sig.parameters
    assert next(iter(sig.parameters)) == "pc_horiz"
    assert model.score_plot.__name__ == "score_plot"
    assert model.score_plot.__doc__ == _plots.score_plot.__doc__
    assert "partial(" not in (model.score_plot.__doc__ or "")


def test_hotellings_t2_limit_signature() -> None:
    """The explicit ``hotellings_t2_limit`` method exposes only ``conf_level``."""
    model = _fitted_pca()
    assert list(inspect.signature(model.hotellings_t2_limit).parameters) == ["conf_level"]


def test_subclass_can_override_method() -> None:
    """A subclass can override a convenience method via normal MRO."""

    class SubPCA(PCA):
        def score_plot(self, *args, **kwargs):  # noqa: ARG002
            return "overridden"

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=list("abcde"))
    model = SubPCA(n_components=2).fit(MCUVScaler().fit_transform(X))
    assert model.score_plot() == "overridden"
