"""Generalized Procrustes analysis (#180).

Three kinds of check. Two configurations reduce GPA to ordinary Procrustes, which
scipy implements independently, so the loss must match scipy's exactly. Similarity
transforms of one configuration have a known answer, a perfect consensus. And the
identities that define the method hold on any data: the total splits into consensus
and residual, rotations are orthogonal, and moving, turning or resizing an input
changes nothing. A synthetic sensory panel with planted faults then checks that the
diagnostics point at the right assessors.
"""

from __future__ import annotations

import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from sklearn.base import clone

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._procrustes import _optimal_scaling
from process_improve.multivariate.methods import GPA

PRODUCTS = [f"prod{i}" for i in range(10)]
ATTRIBUTES = ["sweet", "sour", "bitter", "salty"]


def _orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw a random orthogonal matrix; about half of them include a reflection."""
    q, r = np.linalg.qr(rng.normal(size=(size, size)))
    return q * np.sign(np.diag(r))


def _moved(rng: np.random.Generator, values: np.ndarray, size: float) -> np.ndarray:
    """Turn, resize and shift a configuration: the three differences GPA exists to remove."""
    return size * values @ _orthogonal(rng, values.shape[1]) + rng.normal(scale=5.0, size=values.shape[1])


def _panel(seed: int = 0) -> pd.DataFrame:
    """Eight assessors, ten products, four attributes, two replicates, as ``descriptive_long``.

    The products differ along two directions. P3 uses 40% of the range the others use,
    P5 swaps sour and bitter, and P6 scores at random.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(len(PRODUCTS), 2)) @ rng.normal(size=(2, len(ATTRIBUTES)))
    rows = []
    for number in range(1, 9):
        assessor = f"P{number}"
        spread = 0.4 if assessor == "P3" else rng.uniform(0.8, 1.2)
        order = [0, 2, 1, 3] if assessor == "P5" else [0, 1, 2, 3]
        offset = rng.normal(scale=0.5, size=len(ATTRIBUTES))
        for replicate in (1, 2):
            if assessor == "P6":
                scores = rng.normal(5, 1.5, size=truth.shape)
            else:
                scores = 5 + offset + spread * truth[:, order] + rng.normal(scale=0.3, size=truth.shape)
            rows.extend(
                {
                    "panelist_id": assessor,
                    "session": 1,
                    "product": product,
                    "attribute": attribute,
                    "replicate": replicate,
                    "score": scores[i, j],
                }
                for i, product in enumerate(PRODUCTS)
                for j, attribute in enumerate(ATTRIBUTES)
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _panel()


@pytest.fixture
def copies() -> tuple[np.ndarray, dict[str, pd.DataFrame], list[float]]:
    """Four exact similarity transforms of one configuration, sizes 1, 3, 0.2 and 10."""
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(9, 3))
    sizes = [1.0, 3.0, 0.2, 10.0]
    return truth, {f"c{k}": pd.DataFrame(_moved(rng, truth, size)) for k, size in enumerate(sizes)}, sizes


class TestAgainstScipy:
    """With two configurations GPA is ordinary Procrustes, which scipy implements on its own."""

    def test_scaled_loss_matches_scipy_procrustes(self) -> None:
        """Both scaled to unit size, scipy's disparity is 1 - t^2 and GPA's residual 1 - t."""
        rng = np.random.default_rng(0)
        first, second = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        gpa = GPA().fit({"a": pd.DataFrame(first), "b": pd.DataFrame(second)})
        _, _, disparity = procrustes(first, second)
        total = sum(float((aligned**2).to_numpy().sum()) for aligned in gpa.aligned_.values())
        relative_loss = 2 * float(gpa.residuals_.to_numpy().sum()) / total
        assert relative_loss == pytest.approx(1 - np.sqrt(1 - disparity), rel=1e-10)

    def test_unscaled_loss_matches_scipy_orthogonal_procrustes(self) -> None:
        rng = np.random.default_rng(1)
        first, second = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        gpa = GPA(scale=False).fit({"a": pd.DataFrame(first), "b": pd.DataFrame(second)})
        first, second = first - first.mean(axis=0), second - second.mean(axis=0)
        _, trace = orthogonal_procrustes(second, first)
        expected = (np.sum(first**2) + np.sum(second**2) - 2 * trace) / 2
        assert float(gpa.residuals_.to_numpy().sum()) == pytest.approx(expected, rel=1e-10)


class TestRecovery:
    def test_similar_configurations_align_perfectly(self, copies: tuple) -> None:
        truth, configs, sizes = copies
        gpa = GPA().fit(configs)
        # The iterations stop once the residual settles to ``tol`` of the total, so an exact
        # fit shows as a residual at that scale rather than at machine precision.
        assert gpa.consensus_share_ == pytest.approx(1.0, abs=1e-12)
        assert gpa.residuals_.to_numpy().sum() < 1e-12 * float((gpa.consensus_**2).to_numpy().sum())
        # Each copy is scaled back by the inverse of the size it was given.
        np.testing.assert_allclose(gpa.scale_factors_ * sizes, (gpa.scale_factors_ * sizes).iloc[0])
        _, _, disparity = procrustes(truth, gpa.consensus_.to_numpy())
        assert disparity < 1e-12

    def test_the_consensus_is_nearer_the_truth_than_any_one_configuration(self) -> None:
        """Averaging aligned configurations averages their noise away."""
        rng = np.random.default_rng(2)
        truth = rng.normal(size=(12, 3))
        configs = {
            f"c{k}": pd.DataFrame(_moved(rng, truth + rng.normal(scale=0.25, size=truth.shape), rng.uniform(0.5, 2)))
            for k in range(6)
        }
        consensus = GPA().fit(configs).consensus_.to_numpy()
        best_single = min(procrustes(truth, config.to_numpy())[2] for config in configs.values())
        assert procrustes(truth, consensus)[2] < best_single / 2


class TestInvariance:
    """GPA exists to ignore position, orientation and size; so moving any input changes nothing."""

    def test_moving_one_configuration_changes_nothing(self, panel: pd.DataFrame) -> None:
        """Fitted to a tight ``tol``, so the comparison measures the invariance, not the stopping rule."""
        configs = GPA.configurations_from_long(panel)
        before = GPA(tol=1e-13).fit(configs)
        moved = dict(configs)
        moved["P2"] = pd.DataFrame(
            _moved(np.random.default_rng(7), configs["P2"].to_numpy(), 25.0), index=configs["P2"].index
        )
        after = GPA(tol=1e-13).fit(moved)
        assert after.consensus_share_ == pytest.approx(before.consensus_share_, rel=1e-12)
        # The residuals are in data units, which moved["P2"] changed; their shares are not.
        np.testing.assert_allclose(
            after.residuals_ / after.residuals_.to_numpy().sum(),
            before.residuals_ / before.residuals_.to_numpy().sum(),
            atol=1e-9,
        )

    def test_unscaled_ignores_position_and_orientation_but_not_size(self, copies: tuple) -> None:
        _, configs, _ = copies
        rng = np.random.default_rng(8)
        turned = {name: pd.DataFrame(_moved(rng, config.to_numpy(), 1.0)) for name, config in configs.items()}
        np.testing.assert_allclose(
            GPA(scale=False).fit(turned).residuals_.sum(), GPA(scale=False).fit(configs).residuals_.sum(), atol=1e-9
        )
        assert GPA(scale=False).fit(configs).consensus_share_ < 0.9


class TestPartsOfVariation:
    """Gower's decomposition: what the configurations share and where they differ."""

    def test_the_total_splits_into_consensus_and_residual(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        total = sum(float((aligned**2).to_numpy().sum()) for aligned in gpa.aligned_.values())
        consensus = len(gpa.aligned_) * float((gpa.consensus_**2).to_numpy().sum())
        assert total == pytest.approx(consensus + float(gpa.residuals_.to_numpy().sum()))
        assert gpa.consensus_share_ == pytest.approx(consensus / total)

    def test_scaling_keeps_the_total_sum_of_squares(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        centred = sum(float(((config - config.mean()) ** 2).to_numpy().sum()) for config in configs.values())
        aligned = sum(float((values**2).to_numpy().sum()) for values in gpa.aligned_.values())
        assert aligned == pytest.approx(centred)

    def test_rotations_are_orthogonal(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        for rotation in gpa.rotations_.values():
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(gpa.n_dimensions_), atol=1e-12)

    def test_the_consensus_is_on_its_principal_axes(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        consensus = gpa.consensus_.to_numpy()
        cross = consensus.T @ consensus
        np.testing.assert_allclose(cross - np.diag(np.diag(cross)), 0, atol=1e-10)
        assert np.all(np.diff(gpa.explained_variance_ratio_) <= 1e-12)
        assert gpa.explained_variance_ratio_.sum() == pytest.approx(1.0)

    def test_more_rounds_never_raise_the_residual(self, panel: pd.DataFrame) -> None:
        """Each rotation and each scaling step can only lower the residual (Gower, 1975)."""
        configs = GPA.configurations_from_long(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpecificationWarning)  # the early stops warn, by design
            losses = [float(GPA(max_iter=n).fit(configs).residuals_.to_numpy().sum()) for n in (1, 2, 3, 5, 50)]
        assert all(later <= earlier * (1 + 1e-12) for earlier, later in itertools.pairwise(losses))

    def test_stopping_before_the_residual_settles_warns(self, panel: pd.DataFrame) -> None:
        with pytest.warns(SpecificationWarning, match="raise max_iter"):
            GPA(max_iter=1).fit(GPA.configurations_from_long(panel))


class TestPanel:
    """The sensory use: the diagnostics must find the faults planted in the panel."""

    def test_configurations_from_long_averages_the_replicates(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        assert list(configs) == [f"P{i}" for i in range(1, 9)]
        assert configs["P1"].shape == (len(PRODUCTS), len(ATTRIBUTES))
        rows = panel.query("panelist_id == 'P1' and product == 'prod3' and attribute == 'sour'")
        assert configs["P1"].loc["prod3", "sour"] == pytest.approx(rows["score"].mean())

    def test_free_choice_profiling_keeps_each_assessors_own_words(self, panel: pd.DataFrame) -> None:
        own_words = panel.assign(
            attribute=panel["attribute"].where(panel["panelist_id"] != "P1", "P1-" + panel["attribute"])
        )
        configs = GPA.configurations_from_long(own_words)
        assert list(configs["P1"].columns) == sorted(f"P1-{attribute}" for attribute in ATTRIBUTES)
        assert list(configs["P2"].columns) == sorted(ATTRIBUTES)
        assert GPA().fit(configs).n_dimensions_ == len(ATTRIBUTES)

    def test_the_random_scorer_disagrees_most(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        assert gpa.residuals_.sum().idxmax() == "P6"

    def test_the_narrow_range_assessor_is_stretched_most(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        assert gpa.scale_factors_.idxmax() == "P3"

    def test_a_swapped_pair_of_attributes_is_absorbed_by_the_rotation(self, panel: pd.DataFrame) -> None:
        """P5 swapped two words, not the products' order: a reflection, which GPA allows."""
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        others = gpa.residuals_.sum().drop(["P5", "P6"])
        assert gpa.residuals_.sum()["P5"] < others.max()


class TestConsensusTest:
    def test_a_real_consensus_beats_every_shuffle(self, panel: pd.DataFrame) -> None:
        result = GPA().fit(GPA.configurations_from_long(panel)).consensus_test(99, random_state=0)
        assert result.p_value == pytest.approx(1 / 100)
        assert result.consensus_share > result.null_shares.max()

    def test_noise_has_a_consensus_but_not_a_significant_one(self) -> None:
        """GPA finds a consensus of about half the variation in pure noise; the test is not fooled."""
        rng = np.random.default_rng(3)
        gpa = GPA().fit({f"c{k}": pd.DataFrame(rng.normal(size=(10, 3))) for k in range(4)})
        assert gpa.consensus_share_ > 0.4
        assert gpa.consensus_test(99, random_state=0).p_value > 0.05

    def test_the_same_seed_gives_the_same_answer(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        first = gpa.consensus_test(20, random_state=5)
        again = gpa.consensus_test(20, random_state=np.random.default_rng(5))
        np.testing.assert_array_equal(first.null_shares, again.null_shares)

    def test_n_permutations_must_be_positive(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            GPA().fit(GPA.configurations_from_long(panel)).consensus_test(0)


class TestTransform:
    def test_the_fitted_configurations_give_the_consensus(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        np.testing.assert_allclose(gpa.transform(configs), gpa.consensus_, atol=1e-12)

    def test_new_objects_use_the_fitted_alignment(self, panel: pd.DataFrame) -> None:
        """Two products on their own land where they did in the fit, not re-centred on themselves."""
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        pair = {name: config.iloc[[2, 7]] for name, config in configs.items()}
        np.testing.assert_allclose(gpa.transform(pair), gpa.consensus_.iloc[[2, 7]], atol=1e-12)

    def test_the_configurations_must_match_the_fit(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        with pytest.raises(ValueError, match="Expected the configurations"):
            gpa.transform({name: configs[name] for name in ("P1", "P2")})
        renamed = dict(configs)
        renamed["P1"] = configs["P1"].rename(columns={"sweet": "sugary"})
        with pytest.raises(ValueError, match="expected"):
            gpa.transform(renamed)


class TestInputs:
    def test_configurations_of_different_widths_are_padded(self) -> None:
        rng = np.random.default_rng(5)
        truth = rng.normal(size=(8, 3))
        gpa = GPA().fit({"wide": pd.DataFrame(truth), "narrow": pd.DataFrame(truth[:, :2])})
        assert gpa.n_dimensions_ == 3
        assert gpa.consensus_.shape == (8, 3)

    def test_tiny_units_are_not_mistaken_for_no_spread(self, copies: tuple) -> None:
        _, configs, _ = copies
        tiny = {name: config * 1e-9 for name, config in configs.items()}
        assert GPA().fit(tiny).consensus_share_ == pytest.approx(1.0, abs=1e-12)

    def test_not_a_dict_is_refused(self, copies: tuple) -> None:
        _, configs, _ = copies
        with pytest.raises(TypeError, match="dict of configurations"):
            GPA().fit(list(configs.values()))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda c: {"c0": c["c0"]}, "at least two configurations"),
            (lambda c: {**c, "c1": c["c1"].iloc[::-1]}, "same objects"),
            (lambda c: {name: config.iloc[:2] for name, config in c.items()}, "at least three objects"),
            (lambda c: {**c, "c1": c["c1"].mask(c["c1"] > 2)}, "missing values"),
            (lambda c: {**c, "c1": c["c1"].assign(label="x")}, "non-numeric"),
            (lambda c: {**c, "c1": c["c1"] * 0 + 4.0}, "same point"),
        ],
    )
    def test_invalid_configurations_are_refused(self, copies: tuple, change: object, message: str) -> None:
        _, configs, _ = copies
        with pytest.raises(ValueError, match=message):
            GPA().fit(change(configs))  # type: ignore[operator]

    def test_configurations_from_long_needs_its_columns(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="lacks the columns"):
            GPA.configurations_from_long(panel.drop(columns="score"))

    def test_clone_keeps_the_settings(self) -> None:
        assert clone(GPA(scale=False, tol=1e-6)).get_params() == GPA(scale=False, tol=1e-6).get_params()

    def test_map_plot(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        fig = gpa.map_plot()
        assert len(fig.data) == 8 + 1
        assert fig.layout.title.text == "GPA consensus"
        with pytest.raises(ValueError, match="Axes run from 1"):
            gpa.map_plot(1, 9)


def _centred(configs: dict[str, pd.DataFrame], width: int) -> list[np.ndarray]:
    """Centre each configuration and pad it with zero columns to ``width``, as GPA does before aligning."""
    out = []
    for config in configs.values():
        values = config.to_numpy(dtype=float) - config.to_numpy(dtype=float).mean(axis=0)
        out.append(np.hstack([values, np.zeros((len(values), width - values.shape[1]))]))
    return out


def _noisy(seed: int, noise: list[float], sizes: list[float]) -> dict[str, pd.DataFrame]:
    """Similarity transforms of one configuration, each with its own noise level and size."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(10, 3))
    return {
        f"c{k}": pd.DataFrame(_moved(rng, truth + rng.normal(scale=level, size=truth.shape), size))
        for k, (level, size) in enumerate(zip(noise, sizes, strict=True))
    }


class TestOptimalScalingByHand:
    """ten Berge's factors and signs, from unit configurations whose inner products are set by hand."""

    # Three unit vectors with inner products 0.5 (first and second) and -0.5 (each with the third):
    # the inner-product matrix is 0.5 I + 0.5 s s' with s = (1, 1, -1), whose leading eigenvector is
    # s / sqrt(3), eigenvalue 2. So every factor is sqrt(3) / sqrt(3) = 1, and the third is turned.
    _A = np.array([[1.0], [0.0], [0.0]])
    _B = np.array([[0.5], [np.sqrt(3) / 2], [0.0]])
    _C = np.array([[-0.5], [-1 / (2 * np.sqrt(3))], [np.sqrt(2 / 3)]])

    @pytest.mark.parametrize(
        ("order", "signs"),
        [((0, 1, 2), [1, 1, -1]), ((2, 0, 1), [-1, 1, 1]), ((1, 2, 0), [1, -1, 1])],
        ids=["third-turned", "first-turned", "second-turned"],
    )
    def test_the_configuration_against_the_others_is_turned(self, order: tuple, signs: list[int]) -> None:
        """The leading eigenvector is taken in the orientation whose entries sum to more than 0.

        eigh returns it with a negative sum for the first two orders and a positive sum for the third.
        """
        configs = [self._A, self._B, self._C]
        factors, turned = _optimal_scaling(np.array([configs[k] for k in order]))
        np.testing.assert_allclose(factors, [1.0, 1.0, 1.0], rtol=1e-12)
        np.testing.assert_array_equal(turned, signs)

    def test_a_configuration_sharing_nothing_gets_no_weight(self) -> None:
        """With the third orthogonal to the others, the leading eigenvector is (1, 1, 0) / sqrt(2), eigenvalue 1.5."""
        factors, signs = _optimal_scaling(np.array([self._A, self._B, np.array([[0.0], [0.0], [1.0]])]))
        np.testing.assert_allclose(factors, [np.sqrt(1.5), np.sqrt(1.5), 0.0], rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(signs, [1, 1, 1])


class TestIterations:
    def test_a_tolerance_of_the_whole_total_stops_after_two_rounds(self, copies: tuple) -> None:
        """The first round cannot stop (the previous residual is infinite), and the second gains less than the total.

        Fitted in units a thousand times larger, so the tolerance is read relative to the total.
        """
        _, configs, _ = copies
        large = {name: config * 1000 for name, config in configs.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for scale in (True, False):
                assert GPA(tol=1.0, scale=scale).fit(large).n_iter_ == 2

    def test_max_iter_rounds_and_no_more(self, panel: pd.DataFrame) -> None:
        with pytest.warns(SpecificationWarning) as record:
            gpa = GPA(max_iter=1).fit(GPA.configurations_from_long(panel))
        assert gpa.n_iter_ == 1
        assert str(record[0].message) == (
            "GPA stopped after max_iter=1 rounds before the residual settled; raise max_iter."
        )
        assert pathlib.Path(record[0].filename).name != "_procrustes.py"

    def test_a_fit_that_settles_does_not_warn(self, panel: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gpa = GPA().fit(GPA.configurations_from_long(panel))
        assert 2 <= gpa.n_iter_ < 100

    def test_the_defaults(self) -> None:
        assert GPA().get_params() == {"scale": True, "tol": 1e-10, "max_iter": 100}

    @pytest.mark.parametrize("scale", [True, False])
    def test_each_rotation_is_the_procrustes_rotation_onto_the_others(self, scale: bool) -> None:
        """Gower's fixed point: each configuration's rotation is scipy's orthogonal Procrustes onto the sum of the rest.

        Sizes of 1, 4 and 0.3 keep the configurations' own sizes in play when not scaling.
        """
        configs = _noisy(11, [0.2, 0.4, 0.6], [1.0, 4.0, 0.3])
        gpa = GPA(scale=scale, tol=1e-15, max_iter=1000).fit(configs)
        aligned = [frame.to_numpy() for frame in gpa.aligned_.values()]
        for k, (name, centred) in enumerate(zip(gpa.aligned_, _centred(configs, 3), strict=True)):
            others = sum(aligned[j] for j in range(len(aligned)) if j != k)
            expected, _ = orthogonal_procrustes(centred, others)
            np.testing.assert_allclose(gpa.rotations_[name], expected, atol=1e-6)

    def test_scaling_beats_equal_sizes(self) -> None:
        """For a total size, ten Berge's factors maximise the consensus share; equal sizes are one choice of factors."""
        configs = _noisy(12, [0.1, 0.1, 0.8, 1.5], [1.0, 1.0, 1.0, 1.0])
        unit = {name: config / np.linalg.norm(config - config.mean()) for name, config in configs.items()}
        scaled = GPA().fit(configs).consensus_share_
        equal = GPA(scale=False).fit(unit).consensus_share_
        assert scaled > equal + 1e-3


class TestFittedLabels:
    def test_every_result_is_labelled_by_object_dimension_and_configuration(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        objects = configs["P1"].index
        for frame in (gpa.consensus_, *gpa.aligned_.values()):
            pd.testing.assert_index_equal(frame.index, objects)
            assert list(frame.columns) == [1, 2, 3, 4]
        pd.testing.assert_index_equal(gpa.residuals_.index, objects)
        assert list(gpa.residuals_.columns) == list(configs)
        assert gpa.scale_factors_.name == "scale factor"
        assert list(gpa.scale_factors_.index) == list(configs)

    def test_the_residuals_are_the_squared_distances_from_the_consensus(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        for name, aligned in gpa.aligned_.items():
            expected = ((aligned - gpa.consensus_) ** 2).sum(axis=1)
            np.testing.assert_allclose(gpa.residuals_[name], expected, rtol=1e-12)

    def test_the_axes_shares_and_signs_are_read_off_the_consensus(self, panel: pd.DataFrame) -> None:
        """Each axis's share is its column's sum of squares over the total; each points at its furthest object."""
        consensus = GPA().fit(GPA.configurations_from_long(panel)).consensus_
        squares = (consensus**2).sum()
        np.testing.assert_allclose(
            GPA().fit(GPA.configurations_from_long(panel)).explained_variance_ratio_,
            squares / squares.sum(),
            rtol=1e-10,
        )
        for column in consensus:
            values = consensus[column].to_numpy()
            assert values[np.abs(values).argmax()] > 0

    def test_new_objects_keep_their_labels(self, panel: pd.DataFrame) -> None:
        configs = GPA.configurations_from_long(panel)
        gpa = GPA().fit(configs)
        placed = gpa.transform({name: config.iloc[[7, 2]] for name, config in configs.items()})
        assert list(placed.index) == ["prod7", "prod2"]
        assert list(placed.columns) == [1, 2, 3, 4]


class TestInputsInFull:
    def test_three_objects_are_enough(self) -> None:
        rng = np.random.default_rng(6)
        gpa = GPA().fit({"a": pd.DataFrame(rng.normal(size=(3, 2))), "b": pd.DataFrame(rng.normal(size=(3, 2)))})
        assert gpa.consensus_.shape == (3, 2)

    def test_the_messages_in_full(self, copies: tuple) -> None:
        _, configs, _ = copies
        with pytest.raises(
            TypeError,
            match=r"^GPA takes a dict of configurations, name to DataFrame \(as MBPCA takes blocks\); got list\.$",
        ):
            GPA().fit(list(configs.values()))
        with pytest.raises(ValueError, match=r"^GPA needs at least two configurations to compare\.$"):
            GPA().fit({"c0": configs["c0"]})
        with pytest.raises(
            ValueError,
            match=r"^GPA needs at least three objects: two points always lie on a line, and any two lines "
            r"can be turned onto each other, so there would be nothing to compare\.$",
        ):
            GPA().fit({name: config.iloc[:2] for name, config in configs.items()})

    @pytest.mark.parametrize(
        "flat",
        [np.zeros((9, 3)), np.tile([1.0, 5.0, -2.0], (9, 1))],
        ids=["all-zero", "each-column-constant"],
    )
    def test_a_configuration_with_no_spread_is_refused(self, copies: tuple, flat: np.ndarray) -> None:
        _, configs, _ = copies
        with pytest.raises(ValueError, match=r"^Configuration 'c1' puts every object at the same point\.$"):
            GPA().fit({**configs, "c1": pd.DataFrame(flat)})

    def test_a_yes_no_column_is_read_as_one_and_zero(self) -> None:
        rng = np.random.default_rng(9)
        first = pd.DataFrame(rng.normal(size=(6, 2)), columns=["x", "y"])
        flags = pd.DataFrame({"x": rng.normal(size=6), "flag": [True, False, True, True, False, False]})
        as_bools = GPA().fit({"a": first, "b": flags})
        as_floats = GPA().fit({"a": first, "b": flags.astype(float)})
        np.testing.assert_allclose(as_bools.consensus_, as_floats.consensus_, atol=1e-12)
        np.testing.assert_allclose(as_bools.transform({"a": first, "b": flags}), as_floats.consensus_, atol=1e-12)

    def test_an_attribute_scored_for_some_objects_only_is_kept_and_refused(self, panel: pd.DataFrame) -> None:
        """A column with a gap stays in its configuration, so fit names the gap rather than losing the attribute."""
        gappy = panel[
            ~((panel["panelist_id"] == "P2") & (panel["product"] == "prod4") & (panel["attribute"] == "sour"))
        ]
        configs = GPA.configurations_from_long(gappy)
        assert list(configs["P2"].columns) == sorted(ATTRIBUTES)
        assert np.isnan(configs["P2"].loc["prod4", "sour"])
        with pytest.raises(ValueError, match=r"^Configuration 'P2' has missing values"):
            GPA().fit(configs)


class TestConsensusTestByHand:
    def test_each_shuffle_is_a_refit_with_the_objects_shuffled_in_all_but_the_first(self, copies: tuple) -> None:
        """Shuffle the rows of every configuration but the first, as numpy's generator for the seed does, and refit."""
        _, configs, _ = copies
        names = list(configs)
        for scale in (True, False):
            gpa = GPA(scale=scale).fit(configs)
            result = gpa.consensus_test(3, random_state=7)
            rng = np.random.default_rng(7)
            for share in result.null_shares:
                shuffled = {names[0]: configs[names[0]]}
                for name in names[1:]:
                    order = rng.permutation(len(configs[name]))
                    shuffled[name] = pd.DataFrame(configs[name].to_numpy()[order], index=configs[name].index)
                assert share == pytest.approx(GPA(scale=scale).fit(shuffled).consensus_share_, rel=1e-8)
            assert result.p_value == (1 + int(np.sum(result.null_shares >= result.consensus_share))) / 4

    def test_one_shuffle_is_enough_and_199_the_default(self, copies: tuple) -> None:
        _, configs, _ = copies
        gpa = GPA().fit(configs)
        assert len(gpa.consensus_test(1, random_state=0).null_shares) == 1
        assert len(gpa.consensus_test(random_state=0).null_shares) == 199


class TestMapPlotInFull:
    def test_the_spokes_the_consensus_and_the_layout(self, panel: pd.DataFrame) -> None:
        from process_improve.visualization.themes import DEFAULT_THEME

        gpa = GPA().fit(GPA.configurations_from_long(panel))
        fig = gpa.map_plot()
        centre = gpa.consensus_[[1, 2]].to_numpy()
        for trace, (name, aligned) in zip(fig.data[:-1], gpa.aligned_.items(), strict=True):
            ends = aligned[[1, 2]].to_numpy()
            assert (trace.name, trace.mode, trace.opacity) == (name, "lines", 0.6)
            for axis, drawn in ((0, trace.x), (1, trace.y)):
                values = np.asarray(drawn, dtype=float)
                np.testing.assert_array_equal(values[0::3], centre[:, axis])
                np.testing.assert_array_equal(values[1::3], ends[:, axis])
                assert np.isnan(values[2::3]).all()
        points = fig.data[-1]
        np.testing.assert_array_equal(points.x, centre[:, 0])
        np.testing.assert_array_equal(points.y, centre[:, 1])
        assert (points.name, points.mode, points.textposition) == ("consensus", "markers+text", "top center")
        assert list(points.text) == [f"prod{i}" for i in range(10)]
        assert (points.marker.size, points.marker.color) == (10, "black")

        share = 100 * (gpa.consensus_**2).sum() / (gpa.consensus_**2).sum().sum()
        layout = fig.layout
        assert layout.xaxis.title.text == f"Axis 1 ({share[1]:.1f}% of the consensus)"
        assert layout.yaxis.title.text == f"Axis 2 ({share[2]:.1f}% of the consensus)"
        assert (layout.xaxis.zeroline, layout.yaxis.zeroline) == (True, True)
        assert (layout.yaxis.scaleanchor, layout.yaxis.scaleratio) == ("x", 1)
        assert (layout.width, layout.height) == (600, 600)
        assert layout.template == go.Figure(layout={"template": DEFAULT_THEME}).layout.template

    def test_other_axes_and_settings(self, panel: pd.DataFrame) -> None:
        gpa = GPA().fit(GPA.configurations_from_long(panel))
        fig = gpa.map_plot(
            4, 3, settings={"title": "Panel", "html_image_height": 400, "html_aspect_ratio_w_over_h": 1.5}
        )
        np.testing.assert_array_equal(fig.data[-1].x, gpa.consensus_[4].to_numpy())
        np.testing.assert_array_equal(fig.data[-1].y, gpa.consensus_[3].to_numpy())
        assert fig.layout.xaxis.title.text.startswith("Axis 4 (")
        assert (fig.layout.title.text, fig.layout.width, fig.layout.height) == ("Panel", 600, 400)
        with pytest.raises(ValueError, match=r"^Axes run from 1 to 4; got 0\.$"):
            gpa.map_plot(0, 1)
