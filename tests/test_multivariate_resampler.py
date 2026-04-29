"""Tests for ``Resampler`` in ``process_improve.multivariate.methods``.

The Resampler class supports jackknife, bootstrap, and fractional
resampling against any ``BaseEstimator``-derived estimator that takes a
``DataFrameDict`` as its training data. We use a tiny stub estimator
here so the resampling loops finish in a fraction of a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from process_improve.multivariate.methods import DataFrameDict, Resampler

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _StubEstimator(BaseEstimator):
    """Trivial sklearn-compatible estimator that records the train-set size."""

    def __init__(self, dummy: int = 1) -> None:
        # Required by sklearn's clone(): every constructor parameter must be
        # an instance attribute with the same name.
        self.dummy = dummy

    def fit(self, X: DataFrameDict, y: object | None = None) -> _StubEstimator:  # noqa: ARG002
        self.n_train_samples_ = len(X)
        return self


def _accessor(estimator: _StubEstimator) -> dict[str, float]:
    """Return a small dict of numeric parameters for one resample."""
    rng = np.random.default_rng(estimator.n_train_samples_)
    return {
        "n": float(estimator.n_train_samples_),
        "noise_a": float(rng.normal()),
        "noise_b": float(rng.normal()),
    }


def _noisy_accessor(estimator: _StubEstimator) -> dict[str, float]:
    """Variant of _accessor that always returns positive-variance noise -
    needed for ridgeplot's KDE which fails when any column is constant.
    """
    rng = np.random.default_rng()
    return {
        "noise_a": float(rng.normal()),
        "noise_b": float(rng.normal()),
        "noise_c": float(rng.normal()),
    }


@pytest.fixture
def tiny_dfd() -> DataFrameDict:
    """Build a DataFrameDict with 10 samples across F, Z, and Y blocks."""
    n = 10
    rng = np.random.default_rng(0)
    f_block = {"main": pd.DataFrame(rng.standard_normal((n, 2)), columns=["f1", "f2"])}
    z_block = {"conds": pd.DataFrame(rng.standard_normal((n, 1)), columns=["z1"])}
    y_block = {"out": pd.DataFrame(rng.standard_normal((n, 1)), columns=["y1"])}
    return DataFrameDict({"F": f_block, "Z": z_block, "Y": y_block})


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestResamplerInit:
    def test_estimator_must_be_base_estimator(self, tiny_dfd: DataFrameDict) -> None:
        with pytest.raises(TypeError, match="estimator must be a BaseEstimator"):
            Resampler(estimator="not-an-estimator", x=tiny_dfd, accessor=_accessor)  # type: ignore[arg-type]

    def test_x_must_be_dataframe_dict(self) -> None:
        with pytest.raises(TypeError, match="x must be a DataFrameDict"):
            Resampler(estimator=_StubEstimator(), x={}, accessor=_accessor)  # type: ignore[arg-type]

    def test_accessor_must_be_callable(self, tiny_dfd: DataFrameDict) -> None:
        with pytest.raises(TypeError, match="accessor must be a callable"):
            Resampler(estimator=_StubEstimator(), x=tiny_dfd, accessor="not-a-callable")  # type: ignore[arg-type]

    def test_mutually_exclusive_flags_rejected(self, tiny_dfd: DataFrameDict) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            Resampler(
                estimator=_StubEstimator(),
                x=tiny_dfd,
                accessor=_accessor,
                use_jackknife=True,
                bootstrap_rounds=2,
                fraction_excluded=0.2,
            )


# ---------------------------------------------------------------------------
# resample() dispatch
# ---------------------------------------------------------------------------


class TestResamplerDispatch:
    def test_resample_default_is_jackknife(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(estimator=_StubEstimator(), x=tiny_dfd, accessor=_accessor)
        r.resample(show_progress=False)
        # Jackknife runs once per sample.
        assert r.n_resamples == len(tiny_dfd)

    def test_resample_dispatch_to_bootstrap(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
            bootstrap_rounds=4,
        )
        r.resample(show_progress=False)
        assert r.n_resamples == 4

    def test_resample_dispatch_to_fractional(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
            fraction_excluded=0.5,
        )
        r.resample(show_progress=False)
        # fractional runs len(x) iterations.
        assert r.n_resamples == len(tiny_dfd)

    def test_resample_with_no_method_raises(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
        )
        with pytest.raises(ValueError, match="use_jackknife or bootstrap_rounds"):
            r.resample(show_progress=False)


# ---------------------------------------------------------------------------
# Individual resampling methods
# ---------------------------------------------------------------------------


class TestResamplerMethods:
    def test_jackknife_size_n_minus_1(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(estimator=_StubEstimator(), x=tiny_dfd, accessor=_accessor)
        r.jackknife(show_progress=False)
        # Each jackknife fit holds out exactly one sample.
        assert all(p["n"] == len(tiny_dfd) - 1 for p in r.parameters)

    def test_bootstrap_uses_full_sample_size(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
            bootstrap_rounds=3,
        )
        r.bootstrap(show_progress=False)
        # Each bootstrap fit sees a sample of size n (with replacement).
        assert all(p["n"] == len(tiny_dfd) for p in r.parameters)
        assert r.n_resamples == 3

    def test_fractional_excludes_correct_fraction(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_accessor,
            use_jackknife=False,
            fraction_excluded=0.5,
        )
        r.fractional(show_progress=False)
        # n_groups = int(1 / 0.5) = 2 -> drop ~half of the rows each iter.
        # Train size should be ~n/2.
        for p in r.parameters:
            assert 4 <= p["n"] <= 6


# ---------------------------------------------------------------------------
# plot_results
# ---------------------------------------------------------------------------


class TestPlotResults:
    def test_plot_results_returns_figure(self, tiny_dfd: DataFrameDict) -> None:
        import plotly.graph_objects as go

        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_noisy_accessor,
            use_jackknife=False,
            bootstrap_rounds=8,
        )
        r.bootstrap(show_progress=False)
        fig = r.plot_results()
        assert isinstance(fig, go.Figure)

    def test_plot_results_with_cutoff_adds_vline(self, tiny_dfd: DataFrameDict) -> None:
        r = Resampler(
            estimator=_StubEstimator(),
            x=tiny_dfd,
            accessor=_noisy_accessor,
            use_jackknife=False,
            bootstrap_rounds=8,
        )
        r.bootstrap(show_progress=False)
        fig = r.plot_results(cutoff=0.5)
        # The added vertical line should appear in the figure layout shapes.
        shapes = fig.layout.shapes
        assert len(shapes) >= 1
