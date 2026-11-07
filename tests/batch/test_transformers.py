"""The scikit-learn transformers over the batch alignment and feature functions (#1105)."""

from __future__ import annotations

import doctest
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import InvalidParameterError

from process_improve.batch import (
    BatchFeatureExtractor,
    BatchPCA,
    BatchPLS,
    DTWAligner,
    ResampleAligner,
    batch_dtw,
    f_agemax,
    f_mean,
    find_reference_batch,
    load_nylon,
    resample_to_reference,
)
from process_improve.batch import _transformers as transformers_module
from process_improve.multivariate import PLS, MCUVScaler

#: Builds a ResampleAligner or a quiet DTWAligner from keyword arguments.
MakeAligner = Callable[..., ResampleAligner | DTWAligner]

#: DTW settings that keep the tests quiet and quick.
QUIET = {"show_progress": False}


def _warped(n_samples: int, *, power: float = 1.2, level: float = 0.0) -> pd.DataFrame:
    """Two smooth tags on a time axis bent by ``power``, so batches differ in length and in pace."""
    t = np.linspace(0.0, 1.0, n_samples) ** power
    return pd.DataFrame({"a": np.sin(2 * np.pi * t) + level, "b": np.cos(np.pi * t) + 2.0 * t})


def _campaign(n_batches: int = 8) -> dict[str, pd.DataFrame]:
    """Batches of 25 to 44 samples, each at its own pace and level."""
    rng = np.random.default_rng(1105)
    lengths = rng.integers(25, 45, n_batches)
    powers = rng.uniform(0.8, 1.4, n_batches)
    levels = rng.normal(0.0, 0.2, n_batches)
    return {
        f"B{i}": _warped(int(n), power=float(p), level=float(lv))
        for i, (n, p, lv) in enumerate(zip(lengths, powers, levels, strict=True))
    }


def _quality(batches: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a final quality that follows each batch's level, indexed by batch."""
    return pd.DataFrame({"quality": [float(b["a"].mean()) for b in batches.values()]}, index=list(batches))


@pytest.fixture(scope="module")
def nylon() -> dict:
    return load_nylon()


# ---------------------------------------------------------------------------------------- ResampleAligner
class TestResampleAligner:
    def test_it_matches_the_function_it_wraps(self) -> None:
        batches = _campaign()
        aligned = ResampleAligner(reference_batch="B2").fit_transform(batches)
        expected = resample_to_reference(batches, ["a", "b"], reference_batch="B2")
        assert list(aligned) == list(batches)
        for batch_id, frame in expected.items():
            pd.testing.assert_frame_equal(aligned[batch_id], frame)

    def test_a_new_batch_takes_the_reference_length_without_the_reference(self) -> None:
        """The length is learnt in fit; transform needs neither the reference nor the training batches."""
        batches = _campaign()
        aligner = ResampleAligner(reference_batch="B2").fit(batches)
        new = {"short": _warped(12), "long": _warped(90, power=0.9)}
        aligned = aligner.transform(new)

        assert aligner.n_timesteps_ == len(batches["B2"])
        assert {batch_id: frame.shape for batch_id, frame in aligned.items()} == {
            "short": (len(batches["B2"]), 2),
            "long": (len(batches["B2"]), 2),
        }
        expected = resample_to_reference({**new, "B2": batches["B2"]}, ["a", "b"], reference_batch="B2")
        pd.testing.assert_frame_equal(aligned["long"], expected["long"])

    def test_the_settings_reach_the_interpolation(self) -> None:
        batches = _campaign()
        settings = {"interpolate_kind": "linear"}
        aligned = ResampleAligner(reference_batch="B0", settings=settings).fit_transform(batches)
        expected = resample_to_reference(batches, ["a", "b"], reference_batch="B0", settings=settings)
        pd.testing.assert_frame_equal(aligned["B3"], expected["B3"])

    def test_one_batch_is_its_own_reference(self) -> None:
        only = {"only": _warped(30)}
        aligner = ResampleAligner().fit(only)
        assert aligner.reference_batch_ == "only"
        pd.testing.assert_frame_equal(aligner.transform(only)["only"], only["only"], rtol=1e-12)

    def test_the_columns_come_back_in_the_fitted_order(self) -> None:
        batches = _campaign()
        aligner = ResampleAligner(columns_to_align=["b", "a"], reference_batch="B0").fit(batches)
        assert list(aligner.transform({"new": _warped(30)})["new"].columns) == ["b", "a"]

    @pytest.mark.dataset
    def test_without_a_reference_it_takes_the_most_typical_batch_quietly(self, nylon: dict) -> None:
        """find_reference_batch judges the batches it fitted, so no FutureWarning about judged_rows."""
        tags = list(nylon[1].columns)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aligner = ResampleAligner().fit(nylon)
        assert aligner.reference_batch_ == find_reference_batch(nylon, tags, settings={"judged_rows": "fitted"})
        assert aligner.columns_to_align_ == tags
        assert aligner.n_features_in_ == 10


# -------------------------------------------------------------------------------------------- DTWAligner
class TestDTWAligner:
    def test_the_training_batches_come_back_as_batch_dtw_aligned_them(self) -> None:
        """Seen as new batches, the training batches are aligned exactly as batch_dtw aligned them."""
        batches = _campaign()
        aligner = DTWAligner(reference_batch="B0", settings=QUIET).fit(batches)
        aligned = aligner.transform(batches)
        outputs = batch_dtw(batches, ["a", "b"], reference_batch="B0", settings=QUIET)

        assert len(outputs["weight_history"]) > 3, "the last iteration must align to an average, not to B0"
        assert list(aligned) == list(batches)
        for batch_id, frame in aligned.items():
            np.testing.assert_allclose(
                frame.to_numpy(), outputs["aligned_batch_dfdict"][batch_id][["a", "b"]].to_numpy(), rtol=1e-12
            )
        pd.testing.assert_series_equal(
            aligner.weights_, outputs["weight_history"].iloc[-1].rename("weight"), check_exact=True
        )

    def test_a_new_batch_is_aligned_as_the_same_batch_seen_in_fit(self) -> None:
        """Alignment depends on the batch, not on its key: the old facade looked the key up instead."""
        batches = _campaign()
        aligner = DTWAligner(reference_batch="B0", settings=QUIET).fit(batches)
        renamed = aligner.transform({"renamed": batches["B3"]})["renamed"]
        pd.testing.assert_frame_equal(renamed, aligner.transform(batches)["B3"])

    @pytest.mark.parametrize("n_samples", [12, 90], ids=["shorter", "longer"])
    def test_a_new_batch_shorter_or_longer_than_the_reference(self, n_samples: int) -> None:
        aligner = DTWAligner(reference_batch="B0", settings=QUIET).fit(_campaign())
        aligned = aligner.transform({"new": _warped(n_samples, power=1.1)})["new"]
        assert aligned.shape == (100, 2)
        assert np.isfinite(aligned.to_numpy()).all()

    def test_the_settings_reach_the_time_axis_and_the_subsampling(self) -> None:
        batches = _campaign()
        settings = {**QUIET, "interpolate_time_axis_delta": 0.5, "subsample": 2}
        aligner = DTWAligner(reference_batch="B0", settings=settings).fit(batches)
        outputs = batch_dtw(batches, ["a", "b"], reference_batch="B0", settings=settings)

        assert aligner.n_timesteps_ == 200
        assert len(aligner.reference_trajectory_) == len(batches["B0"].iloc[::2])
        aligned = aligner.transform(batches)
        np.testing.assert_allclose(aligned["B5"].to_numpy(), outputs["aligned_batch_dfdict"]["B5"][["a", "b"]])

    def test_one_batch_is_its_own_reference(self) -> None:
        aligner = DTWAligner(settings=QUIET).fit({"only": _warped(30)})
        assert aligner.reference_batch_ == "only"
        assert aligner.transform({"new": _warped(40)})["new"].shape == (100, 2)

    def test_only_the_aligned_tags_reach_batch_dtw(self) -> None:
        """A text column, a phase label say, is left out rather than failing inside the interpolation."""
        batches = {batch_id: frame.assign(phase="heat") for batch_id, frame in _campaign(4).items()}
        aligned = DTWAligner(reference_batch="B0", settings=QUIET).fit_transform(batches)
        assert list(aligned["B1"].columns) == ["a", "b"]

    def test_a_named_constant_tag_is_kept_with_zero_weight_and_warned_about(self) -> None:
        """batch_dtw gives a tag with no variance zero weight (#1184); the aligner keeps that weight."""
        batches = {batch_id: frame.assign(setpoint=50.0) for batch_id, frame in _campaign(4).items()}
        aligner = DTWAligner(columns_to_align=["a", "b", "setpoint"], reference_batch="B0", settings=QUIET)
        with pytest.warns(UserWarning, match=r"so they get zero weight: 'setpoint'"):
            aligned = aligner.fit_transform(batches)
        assert aligner.weights_["setpoint"] == 0.0
        assert aligner.weights_[["a", "b"]].sum() == pytest.approx(2.0)
        assert aligned["B1"]["setpoint"].to_numpy() == pytest.approx(50.0)

    @pytest.mark.dataset
    def test_it_aligns_the_nylon_batches(self, nylon: dict) -> None:
        training = {batch_id: nylon[batch_id] for batch_id in range(1, 9)}
        settings = {**QUIET, "subsample": 2}
        aligner = DTWAligner(reference_batch=1, settings=settings).fit(training)
        outputs = batch_dtw(training, list(nylon[1].columns), reference_batch=1, settings=settings)
        for batch_id, frame in aligner.transform(training).items():
            np.testing.assert_allclose(frame.to_numpy(), outputs["aligned_batch_dfdict"][batch_id].iloc[:, 1:])
        assert aligner.transform({57: nylon[57]})[57].shape == (100, 10)


# ------------------------------------------------------------------------------ what both aligners refuse
ALIGNERS = [
    pytest.param(ResampleAligner, id="resample"),
    pytest.param(lambda **kw: DTWAligner(settings=QUIET, **kw), id="dtw"),
]


@pytest.mark.parametrize("make", ALIGNERS)
class TestBothAligners:
    def test_a_missing_value_is_refused_by_batch_and_tag(self, make: MakeAligner) -> None:
        """In fit and in transform, in the words of resample_to_reference (#1184, #1169)."""
        batches = _campaign(4)
        gappy = {**batches, "B1": batches["B1"].assign(b=lambda f: f["b"].where(f.index != 5))}
        name = type(make()).__name__
        with pytest.raises(
            ValueError, match=rf"^{name}\.fit cannot resample a tag with missing values; .*\{{'B1': \['b'\]\}}\. Fill"
        ):
            make(reference_batch="B0").fit(gappy)
        aligner = make(reference_batch="B0").fit(batches)
        with pytest.raises(
            ValueError, match=rf"^{name}\.transform cannot resample a tag with missing values; .*\{{'new': \['b'\]\}}"
        ):
            aligner.transform({"new": gappy["B1"]})

    def test_a_new_batch_without_a_fitted_tag_is_refused(self, make: MakeAligner) -> None:
        aligner = make(reference_batch="B0").fit(_campaign(4))
        with pytest.raises(ValueError, match=r"missing \(batch: tags\): \{'new': \['b'\]\}"):
            aligner.transform({"new": _warped(30)[["a"]]})

    def test_a_constant_tag_is_left_out_unless_named(self, make: MakeAligner) -> None:
        batches = {batch_id: frame.assign(setpoint=50.0) for batch_id, frame in _campaign(4).items()}
        aligner = make(reference_batch="B0").fit(batches)
        assert aligner.columns_to_align_ == ["a", "b"]
        assert list(aligner.transform(batches)["B2"].columns) == ["a", "b"]

    def test_an_unknown_reference_batch_is_refused(self, make: MakeAligner) -> None:
        with pytest.raises(KeyError, match=r"`reference_batch` was not found in the dict of batches; got 'nope'"):
            make(reference_batch="nope").fit(_campaign(4))

    def test_a_single_frame_is_refused(self, make: MakeAligner) -> None:
        with pytest.raises(TypeError, match=r"expects `batches` as a dict of per-batch DataFrames"):
            make().fit(_warped(30))

    def test_it_follows_the_estimator_contract(self, make: MakeAligner) -> None:
        aligner = make(reference_batch="B0")
        with pytest.raises(NotFittedError):
            aligner.transform(_campaign(4))
        fitted = aligner.fit(_campaign(4))
        assert sorted(fitted.get_params()) == ["columns_to_align", "reference_batch", "settings"]
        cloned = clone(fitted)
        assert not hasattr(cloned, "n_timesteps_")
        assert cloned.get_params() == fitted.get_params()
        with pytest.raises(InvalidParameterError, match=r"'settings' parameter"):
            clone(fitted).set_params(settings="cubic").fit(_campaign(4))


def test_resample_aligner_refuses_a_batch_of_one_row() -> None:
    with pytest.raises(ValueError, match=r"ResampleAligner\.transform needs at least 2 rows.*\{'one': 1\}"):
        ResampleAligner(reference_batch="B0").fit(_campaign(4)).transform({"one": _warped(1)})


# ------------------------------------------------------------------------------- in a Pipeline, with a model
@pytest.mark.integration
@pytest.mark.parametrize(
    "aligner",
    [ResampleAligner(reference_batch="B0"), DTWAligner(reference_batch="B0", settings=QUIET)],
    ids=["resample", "dtw"],
)
def test_an_aligner_and_batch_pls_make_one_pipeline(aligner: object) -> None:
    """Fit and predict in one object: a new batch is aligned as the training batches were, then predicted."""
    batches = _campaign(10)
    training = {batch_id: batches[batch_id] for batch_id in list(batches)[:8]}
    new = {batch_id: batches[batch_id] for batch_id in list(batches)[8:]}
    pipeline = Pipeline([("align", aligner), ("pls", BatchPLS(n_components=1))]).fit(training, _quality(training))

    by_hand = BatchPLS(n_components=1).fit(clone(aligner).fit_transform(training), _quality(training))
    predicted = pipeline.predict(new).y_hat
    expected = by_hand.predict(pipeline.named_steps["align"].transform(new)).y_hat
    assert list(predicted.index) == list(new)
    pd.testing.assert_frame_equal(predicted, expected)


@pytest.mark.integration
@pytest.mark.dataset
def test_resample_aligner_and_batch_pca_make_one_pipeline(nylon: dict) -> None:
    pipeline = Pipeline([("align", ResampleAligner(reference_batch=1)), ("pca", BatchPCA(n_components=3))])
    scores = pipeline.fit(nylon).transform({57: nylon[57], 1: nylon[1]})
    model = BatchPCA(n_components=3).fit(resample_to_reference(nylon, list(nylon[1].columns), reference_batch=1))
    assert pipeline.named_steps["pca"].scores_.to_numpy() == pytest.approx(model.scores_.to_numpy())
    assert scores.loc[[57, 1]].to_numpy() == pytest.approx(model.scores_.loc[[57, 1]].to_numpy())


# ------------------------------------------------------------------------------------ BatchFeatureExtractor
class TestBatchFeatureExtractor:
    def test_one_row_per_batch_in_the_order_given(self) -> None:
        """Keys that sort differently from their order keep their rows: the old facade relabelled them."""
        batches = {"10": _warped(30), "2": _warped(40, level=1.0), "1": _warped(35, level=2.0)}
        table = BatchFeatureExtractor(features=("mean", "count")).fit_transform(batches)

        assert list(table.index) == ["10", "2", "1"]
        assert table.index.name == "batch_id"
        assert list(table.columns) == ["a_mean", "b_mean", "a_count", "b_count"]
        for batch_id, batch in batches.items():
            assert table.loc[batch_id, "a_mean"] == pytest.approx(batch["a"].mean())
            assert table.loc[batch_id, "b_count"] == len(batch)

    def test_the_values_are_the_extractors(self) -> None:
        batches = _campaign(4)
        stacked = pd.concat(batches.values(), keys=list(batches), names=["batch", None]).reset_index(level=0)
        table = BatchFeatureExtractor(features=("mean",)).fit_transform(batches)
        expected = f_mean(stacked, tags=["a", "b"], batch_col="batch").droplevel(1)
        np.testing.assert_allclose(table.to_numpy(), expected.loc[list(batches)].to_numpy())

    def test_agemax_is_a_label_of_the_batchs_own_index(self) -> None:
        """Each batch keeps its own index, so the age is a time stamp of that batch, not a stacked row number."""
        start = pd.Timestamp("2026-01-01")
        batches = {
            name: _warped(n).set_index(pd.date_range(start, periods=n, freq="min"))
            for name, n in [("x", 30), ("y", 50)]
        }
        table = BatchFeatureExtractor(features=("agemax",), tags=["a"]).fit_transform(batches)
        for name, batch in batches.items():
            assert table.loc[name, "a_agemax"] == batch["a"].idxmax()
        assert (
            table.loc["y", "a_agemax"]
            == f_agemax(batches["y"].assign(batch=1), tags=["a"], batch_col="batch").iloc[0, 0]
        )

    def test_a_tag_with_missing_values_is_summarised_from_what_was_measured(self) -> None:
        batches = _campaign(3)
        batches["B1"] = batches["B1"].assign(a=lambda f: f["a"].where(f.index % 4 != 0))
        table = BatchFeatureExtractor(features=("mean", "count", "last")).fit_transform(batches)
        measured = batches["B1"]["a"].dropna()
        assert table.loc["B1", "a_mean"] == pytest.approx(measured.mean())
        assert table.loc["B1", "a_count"] == len(measured)
        assert table.loc["B1", "a_endpoint"] == measured.iloc[-1]

    def test_a_constant_tag_and_a_single_batch(self) -> None:
        table = BatchFeatureExtractor().fit_transform({"only": _warped(30).assign(setpoint=50.0)})
        assert table.shape == (1, 15)
        assert table.loc["only", "setpoint_mean"] == 50.0
        assert table.loc["only", "setpoint_std"] == 0.0

    def test_new_batches_get_the_fitted_columns(self) -> None:
        extractor = BatchFeatureExtractor(tags=["b"]).fit(_campaign(4))
        table = extractor.transform({"new": _warped(60).assign(extra=1.0)})
        assert list(table.columns) == list(extractor.get_feature_names_out())
        assert list(table.columns) == ["b_mean", "b_std", "b_min", "b_max", "b_endpoint"]
        with pytest.raises(ValueError, match=r"missing \(batch: tags\): \{'new': \['b'\]\}"):
            extractor.transform({"new": _warped(60)[["a"]]})

    @pytest.mark.parametrize(
        ("features", "match"),
        [(("mean", "banana"), r"got \['mean', 'banana'\]"), ((), r"at least one of")],
        ids=["unknown", "none"],
    )
    def test_the_features_must_be_known(self, features: tuple, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BatchFeatureExtractor(features=features).fit(_campaign(3))

    @pytest.mark.parametrize(
        ("batches", "tags", "exc", "match"),
        [
            (_warped(30), None, TypeError, r"expects a dict of per-batch DataFrames.*got DataFrame"),
            ({}, None, ValueError, r"needs at least one batch"),
            ({"x": _warped(30).assign(phase="heat")}, ["phase"], ValueError, r"only summarise numeric.*\['phase'\]"),
        ],
        ids=["frame", "empty", "text-tag"],
    )
    def test_what_it_refuses(self, batches: dict, tags: list | None, exc: type, match: str) -> None:
        with pytest.raises(exc, match=match):
            BatchFeatureExtractor(tags=tags).fit(batches)

    def test_it_follows_the_estimator_contract(self) -> None:
        extractor = BatchFeatureExtractor(tags=["a"])
        with pytest.raises(NotFittedError):
            extractor.transform(_campaign(3))
        with pytest.raises(NotFittedError):
            extractor.get_feature_names_out()
        fitted = extractor.fit(_campaign(3))
        assert sorted(fitted.get_params()) == ["features", "tags"]
        assert not hasattr(clone(fitted), "tags_")
        with pytest.raises(InvalidParameterError, match=r"'features' parameter"):
            BatchFeatureExtractor(features="mean").fit(_campaign(3))

    @pytest.mark.integration
    @pytest.mark.dataset
    def test_the_table_is_the_x_block_of_a_quality_model(self, nylon: dict) -> None:
        tags = list(nylon[1].columns)
        quality = pd.DataFrame({"q": [float(b["Tag01"].mean()) for b in nylon.values()]}, index=list(nylon))
        pipeline = Pipeline(
            [
                ("features", BatchFeatureExtractor(features=("mean", "std"), tags=tags)),
                ("scale", MCUVScaler()),
                ("pls", PLS(n_components=2)),
            ]
        ).fit(nylon, quality)

        assert pipeline.named_steps["pls"].r2_cumulative_.iloc[-1] > 0.5
        predicted = pipeline.predict({57: nylon[57], 3: nylon[3]})
        assert predicted.shape == (2, 1)
        assert list(pipeline.named_steps["features"].get_feature_names_out()[:2]) == ["Tag01_mean", "Tag02_mean"]


@pytest.mark.dataset
@pytest.mark.slow
def test_the_docstring_examples_print_what_they_show() -> None:
    result = doctest.testmod(transformers_module, report=False)
    assert result.attempted > 0
    assert result.failed == 0
