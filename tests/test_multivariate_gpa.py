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
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from sklearn.base import clone

from process_improve.multivariate._common import SpecificationWarning
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
