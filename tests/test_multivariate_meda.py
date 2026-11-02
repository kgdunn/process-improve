"""MEDA and oMEDA: variable relationships and group differences through a fitted model (#373).

The synthetic data has a known structure: variables a1, a2 and a3 follow one latent
factor (a3 in opposition), b1 and b2 follow a second, and n1 is noise. MEDA must show
the two blocks and nothing between them, and oMEDA must find a shift planted in a
known subset of variables, with its sign.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from process_improve.multivariate._diagnostics import _seriation_order
from process_improve.multivariate.methods import PCA, PLS, MCUVScaler, meda, omeda

BLOCK_A, BLOCK_B = ["a1", "a2", "a3"], ["b1", "b2"]


def _blocks(n_rows: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1, f2 = rng.standard_normal(n_rows), rng.standard_normal(n_rows)

    def noise() -> np.ndarray:
        return 0.2 * rng.standard_normal(n_rows)

    raw = pd.DataFrame(
        {
            "a1": f1 + noise(),
            "a2": f1 + noise(),
            "a3": -f1 + noise(),
            "b1": f2 + noise(),
            "b2": f2 + noise(),
            "n1": rng.standard_normal(n_rows),
        }
    )
    return MCUVScaler().fit_transform(raw)


def _ldpe() -> pd.DataFrame:
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE"
    return MCUVScaler().fit_transform(pd.read_csv(folder / "LDPE.csv", index_col=0).iloc[:, :14])


def _fit_pca(X: pd.DataFrame, n_components: int) -> PCA:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PCA(n_components=n_components).fit(X)


class TestMeda:
    def test_shows_the_two_blocks_and_nothing_between_them(self) -> None:
        X = _blocks()
        m = meda(_fit_pca(X, 2), X)
        assert (m.loc[["a1", "a2"], ["a1", "a2"]].to_numpy() > 0.8).all()
        assert (m.loc[["a1", "a2"], "a3"] < -0.8).all()  # a3 moves in opposition
        assert m.loc["b1", "b2"] > 0.8
        assert np.abs(m.loc[BLOCK_A, BLOCK_B].to_numpy()).max() < 0.1
        assert np.abs(m.loc["n1", BLOCK_A + BLOCK_B]).max() < 0.1

    def test_squared_index_is_the_goodness_of_prediction_from_one_variable(self) -> None:
        """Unsigned MEDA is exactly 1 - ||x_k - x_hat||^2 / ||x_k||^2, with x_hat from x_j alone."""
        X = _blocks()
        pca = _fit_pca(X, 2)
        q2 = meda(pca, X, signed=False).to_numpy()
        x, loadings = X.to_numpy(), pca.loadings_.to_numpy()
        cross = x.T @ x
        modelled = cross @ loadings @ loadings.T
        for j in range(x.shape[1]):
            for k in range(x.shape[1]):
                x_hat = x[:, j] * modelled[j, k] / cross[j, j]
                expected = 1 - np.sum((x[:, k] - x_hat) ** 2) / np.sum(x[:, k] ** 2)
                assert q2[j, k] == pytest.approx(expected, abs=1e-12)

    def test_signed_index_keeps_the_size_and_adds_the_sign(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        signed, q2 = meda(pca, X), meda(pca, X, signed=False)
        related = np.abs(signed.to_numpy()) > 0.5
        np.testing.assert_allclose(np.abs(signed.to_numpy())[related], np.abs(q2.to_numpy())[related])

    def test_diagonal_is_the_share_of_each_variable_the_model_reproduces(self) -> None:
        X = _ldpe()  # real data
        pca = _fit_pca(X, 3)
        m = meda(pca, X)
        x, loadings = X.to_numpy(), pca.loadings_.to_numpy()
        share = np.sum((x @ loadings @ loadings.T) ** 2, axis=0) / np.sum(x**2, axis=0)
        np.testing.assert_allclose(np.diag(m.to_numpy()), share * (2 - share), atol=1e-8)
        np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T, atol=1e-8)  # symmetric for PCA
        assert np.abs(m.to_numpy()).max() <= 1 + 1e-12

    def test_each_component_brings_in_its_own_relationships(self) -> None:
        """The first component carries one block only; the second adds the other."""
        X = _blocks()
        one, two = meda(_fit_pca(X, 1), X), meda(_fit_pca(X, 2), X)
        first = BLOCK_A if abs(one.loc["a1", "a2"]) > abs(one.loc["b1", "b2"]) else BLOCK_B
        second = BLOCK_B if first == BLOCK_A else BLOCK_A
        assert abs(one.loc[first[0], first[1]]) > 0.8
        assert abs(one.loc[second[0], second[1]]) < 0.1
        assert abs(two.loc[second[0], second[1]]) > 0.8

    def test_seriation_makes_each_block_contiguous(self) -> None:
        X = _blocks().iloc[:, [0, 3, 5, 1, 4, 2]]  # shuffle the blocks apart
        pca = _fit_pca(X, 2)
        plain, seriated = meda(pca, X), meda(pca, X, seriate=True)
        order = list(seriated.index)
        for block in (BLOCK_A, BLOCK_B):
            positions = sorted(order.index(name) for name in block)
            assert positions == list(range(positions[0], positions[0] + len(block)))
        # The labels travel with the values.
        pd.testing.assert_frame_equal(seriated, plain.loc[order, order])

    def test_seriating_two_variables_keeps_their_order(self) -> None:
        X = _blocks()[["a1", "b1"]]
        assert list(meda(_fit_pca(X, 1), X, seriate=True).index) == ["a1", "b1"]

    def test_pls_uses_the_direct_weights(self) -> None:
        """For PLS the scores come from W(P'W)^-1 and the reconstruction from P."""
        X = _blocks()
        y = pd.DataFrame({"y": X["a1"] + X["a2"] - X["a3"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pls = PLS(n_components=1).fit(X, y)
        m = meda(pls, X)
        x = X.to_numpy()
        cross = x.T @ x
        modelled = cross @ pls.direct_weights_.to_numpy() @ pls.x_loadings_.to_numpy().T
        expected = (2 * cross - modelled) * np.abs(modelled) / np.outer(np.diag(cross), np.diag(cross))
        np.testing.assert_allclose(m.to_numpy(), expected, atol=1e-12)
        assert m.loc["a1", "a2"] > 0.8  # the block that predicts y
        assert abs(m.loc["b1", "b2"]) < 0.1

    def test_does_not_depend_on_the_units(self) -> None:
        """S and S_A scale by c^2, so numerator and denominator both by c^4; c = 0.01 puts S_jj S_kk below 1."""
        X = _blocks()
        pca = _fit_pca(X, 2)
        np.testing.assert_allclose(meda(pca, 0.01 * X), meda(pca, X), rtol=1e-10, atol=1e-14)

    def test_a_variable_with_no_sum_of_squares_reads_as_0_without_a_warning(self) -> None:
        X = _blocks().assign(n1=0.0)
        pca = _fit_pca(_blocks(), 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = meda(pca, X)
        assert (m["n1"] == 0.0).all()
        assert (m.loc["n1"] == 0.0).all()

    def test_reads_object_data_as_numbers(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        pd.testing.assert_frame_equal(meda(pca, X.astype(object)), meda(pca, X))
        group = X.index[:10]
        pd.testing.assert_series_equal(omeda(pca, X.astype(object), group), omeda(pca, X, group))

    def test_missing_cells_are_filled_through_the_model(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        gappy = X.copy()
        gappy.iloc[[3, 17, 40], [0, 3, 5]] = np.nan
        np.testing.assert_allclose(meda(pca, gappy).to_numpy(), meda(pca, X).to_numpy(), atol=0.05)


class TestOmeda:
    @staticmethod
    def _shifted() -> tuple[pd.DataFrame, PCA, pd.Index, pd.Index]:
        """Return data whose first 15 rows have b1 and b2 raised, a model, and the two groups."""
        rng = np.random.default_rng(3)
        n_rows = 60
        f1, f2 = rng.standard_normal(n_rows), rng.standard_normal(n_rows)
        raw = pd.DataFrame(
            {
                "a1": f1 + 0.2 * rng.standard_normal(n_rows),
                "a2": f1 + 0.2 * rng.standard_normal(n_rows),
                "b1": f2 + 0.2 * rng.standard_normal(n_rows),
                "b2": f2 + 0.2 * rng.standard_normal(n_rows),
                "n1": rng.standard_normal(n_rows),
            }
        )
        raw.loc[raw.index[:15], ["b1", "b2"]] += 2.5
        X = MCUVScaler().fit_transform(raw)
        return X, _fit_pca(X, 2), X.index[:15], X.index[15:]

    def test_ranks_the_shifted_variables_first_with_their_sign(self) -> None:
        X, pca, group, rest = self._shifted()
        o = omeda(pca, X, group=group, reference=rest)
        assert set(o.abs().nlargest(2).index) == {"b1", "b2"}
        assert (o[["b1", "b2"]] > 0).all()  # higher in the group
        assert o[["b1", "b2"]].min() > 5 * o.drop(["b1", "b2"]).abs().max()  # by a wide margin

    def test_swapping_the_sides_flips_every_sign(self) -> None:
        X, pca, group, rest = self._shifted()
        pd.testing.assert_series_equal(omeda(pca, X, group=rest, reference=group), -omeda(pca, X, group, rest))

    def test_matches_the_formula(self) -> None:
        X, pca, group, rest = self._shifted()
        d = np.where(X.index.isin(group), 1.0, -1.0)
        x, loadings = X.to_numpy(), pca.loadings_.to_numpy()
        modelled = (x @ loadings @ loadings.T).T @ d
        expected = (2 * x.T @ d - modelled) * np.abs(modelled) / np.sqrt(d @ d)
        np.testing.assert_allclose(omeda(pca, X, group, rest).to_numpy(), expected, atol=1e-10)

    def test_weights_are_rescaled_so_each_side_peaks_at_one(self) -> None:
        X, pca, group, rest = self._shifted()
        weights = np.where(X.index.isin(group), 7.0, -0.5)
        np.testing.assert_allclose(omeda(pca, X, weights=weights), omeda(pca, X, group, rest))
        np.testing.assert_allclose(omeda(pca, X, weights=weights / 14.0), omeda(pca, X, group, rest))  # 0.5, -1/28
        integers = np.where(X.index.isin(group), 7, -2)
        pd.testing.assert_series_equal(omeda(pca, X, weights=integers), omeda(pca, X, group, rest))
        assert omeda(pca, X, group, rest).name == "omeda"

    def test_positive_weights_with_zeros_mirror_the_group_alone(self) -> None:
        X, pca, group, _ = self._shifted()
        weights = np.where(X.index.isin(group), 3.0, 0.0)
        pd.testing.assert_series_equal(omeda(pca, X, weights=weights), omeda(pca, X, group=group))

    def test_one_sided_weights_mirror_the_group_alone(self) -> None:
        """Weights of -1 on the group, and nothing positive, give the group-vs-centre result negated."""
        X, pca, group, _ = self._shifted()
        weights = -X.index.isin(group).astype(float)
        pd.testing.assert_series_equal(omeda(pca, X, weights=weights), -omeda(pca, X, group=group))

    def test_group_alone_compares_against_the_model_centre(self) -> None:
        X, pca, group, _ = self._shifted()
        o = omeda(pca, X, group=group)
        assert set(o.abs().nlargest(2).index) == {"b1", "b2"}
        assert (o[["b1", "b2"]] > 0).all()

    def test_boolean_mask_selects_like_labels(self) -> None:
        X, pca, group, rest = self._shifted()
        mask = list(X.index.isin(group))
        pd.testing.assert_series_equal(
            omeda(pca, X, group=mask, reference=[not m for m in mask]), omeda(pca, X, group, rest)
        )

    def test_real_data_runs_and_is_finite(self) -> None:
        X = _ldpe()
        pca = _fit_pca(X, 3)
        o = omeda(pca, X, group=X.index[-4:], reference=X.index[:-4])
        assert list(o.index) == list(X.columns)
        assert np.isfinite(o).all()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"group": [0], "weights": np.ones(60)}, "Pass either weights, or group/reference, not both."),
            ({}, "Pass group (optionally with reference), or weights."),
            (
                {"group": [0, 1], "reference": [1, 2]},
                "group and reference share observations; each observation belongs to one side.",
            ),
            ({"weights": np.zeros(60)}, "The comparison selects no observations: every weight is zero."),
            ({"weights": np.ones(3)}, "weights must have one entry per row of X (60), got shape (3,)."),
            ({"group": [True, False]}, "group is a boolean mask of length 2, but X has 60 rows."),
            (
                {"group": [0], "reference": [999]},
                (
                    "reference contains entries that are not index labels of X: [999]. Selection is by index "
                    "label or boolean mask; to select by position, pass X.index[...] instead."
                ),
            ),
        ],
    )
    def test_rejects_a_malformed_comparison(self, kwargs: dict, message: str) -> None:
        X, pca, _, _ = self._shifted()
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            omeda(pca, X, **kwargs)


class TestSeriation:
    """The order of a seriated map: average-linkage clusters of the distance ``1 - |m|``, kept together."""

    @staticmethod
    def _symmetric(n: int, entries: dict[tuple[int, int], float]) -> np.ndarray:
        m = np.eye(n)
        for (i, j), value in entries.items():
            m[i, j] = m[j, i] = value
        return m

    @staticmethod
    def _adjacent(order: np.ndarray, i: int, j: int) -> bool:
        positions = list(order)
        return abs(positions.index(i) - positions.index(j)) == 1

    def test_three_variables_put_the_related_pair_together(self) -> None:
        order = _seriation_order(self._symmetric(3, {(0, 1): 0.0, (0, 2): 0.9, (1, 2): 0.1}))
        assert self._adjacent(order, 0, 2)

    def test_clusters_by_the_mean_similarity_not_the_nearest_neighbour(self) -> None:
        """Average linkage joins b-d (0.9), then a-c (0.5): c's mean similarity to {b, d} is (0.7 + 0.2) / 2 = 0.45.

        Single linkage would join c to {b, d} through b (0.7) instead. Every leaf order of the
        tree keeps its clusters together, so a sits next to c, and b next to d.
        """
        entries = {(0, 1): 0.1, (0, 2): 0.5, (0, 3): 0.6, (1, 2): 0.7, (1, 3): 0.9, (2, 3): 0.2}
        order = _seriation_order(self._symmetric(4, entries))
        assert self._adjacent(order, 0, 2)
        assert self._adjacent(order, 1, 3)

    def test_moderate_links_between_two_blocks_keep_the_blocks_apart(self) -> None:
        """Pairs a-c (0.9) and b-d (0.8), with links of 0.3 to 0.45 between them, come out as two blocks."""
        entries = {(0, 1): 0.3, (0, 2): 0.9, (0, 3): 0.35, (1, 2): 0.4, (1, 3): 0.8, (2, 3): 0.45}
        order = _seriation_order(self._symmetric(4, entries))
        assert self._adjacent(order, 0, 2)
        assert self._adjacent(order, 1, 3)

    def test_an_entry_above_1_counts_as_1(self) -> None:
        """A PLS map, or one of new data, can exceed 1 in size; the distance stops at 0, as for an exact copy."""
        entries = {(0, 1): 0.1, (0, 2): 0.3, (0, 3): 0.6, (1, 2): 0.8, (1, 3): 1.5, (2, 3): 3.0}
        strong = self._symmetric(4, entries)
        np.testing.assert_array_equal(_seriation_order(strong), _seriation_order(np.clip(strong, None, 1.0)))


class TestModelMethodsAndPlots:
    def test_methods_forward_to_the_functions(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        pd.testing.assert_frame_equal(pca.meda(X, seriate=True), meda(pca, X, seriate=True))
        pd.testing.assert_series_equal(pca.omeda(X, group=X.index[:10]), omeda(pca, X, group=X.index[:10]))
        assert list(inspect.signature(pca.meda).parameters) == ["X", "signed", "seriate"]
        assert list(inspect.signature(pca.omeda).parameters) == ["X", "group", "reference", "weights"]
        assert hasattr(PLS, "meda")
        assert hasattr(PLS, "omeda_plot")

    def test_meda_plot_is_a_seriated_heat_map(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        fig = pca.meda_plot(X)
        assert isinstance(fig, go.Figure)
        (trace,) = fig.data
        assert trace.type == "heatmap"
        expected = meda(pca, X, seriate=True)
        assert list(trace.x) == list(expected.columns)
        np.testing.assert_allclose(np.asarray(trace.z), expected.to_numpy())
        assert (trace.zmin, trace.zmax) == (-1.0, 1.0)

    def test_meda_plot_draws_onto_an_existing_figure(self) -> None:
        X = _blocks()
        existing = go.Figure()
        fig = _fit_pca(X, 2).meda_plot(X, fig=existing)
        assert fig is existing
        assert [trace.type for trace in fig.data] == ["heatmap"]

    def test_meda_plot_settings(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        fig = pca.meda_plot(X, {"signed": False, "seriate": False, "show_values": True})
        (trace,) = fig.data
        assert list(trace.x) == list(X.columns)
        assert trace.colorbar.title.text == "q2"
        assert trace.text is not None

    def test_omeda_plot_draws_one_bar_per_variable(self) -> None:
        X = _blocks()
        pca = _fit_pca(X, 2)
        fig = pca.omeda_plot(X, group=X.index[:10], reference=X.index[10:])
        (trace,) = fig.data
        assert trace.type == "bar"
        np.testing.assert_allclose(
            np.asarray(trace.y), omeda(pca, X, group=X.index[:10], reference=X.index[10:]).to_numpy()
        )
