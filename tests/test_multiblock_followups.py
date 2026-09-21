"""Multi-block follow-ups from #193: batchwise unfolding into blocks, and resampling them.

Two gaps are closed here, and the tests are written around what each one is *for*
rather than around its signature:

* ``unfold_blocks`` turns several aligned batch dictionaries into the
  ``dict[str, pd.DataFrame]`` that ``MBPCA`` and ``MBPLS`` accept. The property
  that matters is that a row means the same batch in every block.
* ``BlockSet`` gives that dict a row axis, which is the only thing standing
  between the multi-block models and ``Resampler``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.batch.data_input import dict_to_wide
from process_improve.batch.preprocessing import unfold_blocks
from process_improve.multivariate.methods import MBPCA, BlockSet, Resampler

N_BATCHES = 10


def _batches(n_rows: int, columns: list[str], *, ids: list[str], seed: int) -> dict[str, pd.DataFrame]:
    """Build a standard batch dictionary: one aligned, all-numeric frame per batch."""
    rng = np.random.default_rng(seed)
    return {b: pd.DataFrame(rng.normal(size=(n_rows, len(columns))), columns=columns) for b in ids}


@pytest.fixture
def ids() -> list[str]:
    return [f"batch_{i:02d}" for i in range(N_BATCHES)]


@pytest.fixture
def two_blocks(ids: list[str]) -> dict[str, dict]:
    """Two trajectory blocks of deliberately different width *and* different length."""
    return {
        "spectra": _batches(8, ["s1", "s2", "s3"], ids=ids, seed=0),
        "process": _batches(6, ["T", "P"], ids=ids, seed=1),
    }


# ---------------------------------------------------------------------------
# unfold_blocks
# ---------------------------------------------------------------------------


class TestUnfoldBlocks:
    def test_each_block_matches_dict_to_wide(self, two_blocks: dict[str, dict]) -> None:
        """Per block, this is exactly the existing unfolding; only the packaging is new."""
        unfolded = unfold_blocks(two_blocks)

        assert set(unfolded) == {"spectra", "process"}
        for name, batches in two_blocks.items():
            expected = dict_to_wide(batches).reindex(unfolded[name].index)
            pd.testing.assert_frame_equal(unfolded[name], expected)

    def test_blocks_keep_their_own_widths(self, two_blocks: dict[str, dict]) -> None:
        """8 samples x 3 tags and 6 x 2: the blocks are not forced to a common shape."""
        unfolded = unfold_blocks(two_blocks)
        assert unfolded["spectra"].shape == (N_BATCHES, 8 * 3)
        assert unfolded["process"].shape == (N_BATCHES, 6 * 2)

    def test_row_i_is_the_same_batch_in_every_block(self, ids: list[str]) -> None:
        """The contract: one row order across all blocks, aligned by batch id.

        The blocks are built in opposite insertion orders. That does not by itself
        break anything today, because `dict_to_wide` pivots on batch_id and returns
        a sorted index either way; the test pins the guarantee rather than the
        mechanism, so it still holds if that sorting ever stops.
        `test_initial_conditions_are_reordered_to_match` is the case where the
        reindex is doing real work.
        """
        spectra = _batches(5, ["s1", "s2"], ids=ids, seed=2)
        shuffled = list(reversed(ids))
        process = {b: _batches(4, ["T"], ids=ids, seed=3)[b] for b in shuffled}
        assert list(process) != list(spectra)

        unfolded = unfold_blocks({"spectra": spectra, "process": process})

        first = unfolded["spectra"].index
        for wide in unfolded.values():
            assert wide.index.equals(first)
        # And the alignment is by batch id, not by position: the row labelled
        # `ids[0]` in one block carries `ids[0]`'s data in the other too.
        expected_process = dict_to_wide(process).loc[first]
        pd.testing.assert_frame_equal(unfolded["process"], expected_process)

    def test_initial_conditions_become_their_own_block(self, two_blocks: dict[str, dict], ids: list[str]) -> None:
        """Z is a block, not a prefix glued onto a trajectory block.

        ``BatchPCA`` concatenates Z onto the unfolded X because the model beneath
        it is single-block. Keeping them apart is the whole reason to reach for a
        multi-block model.
        """
        z = pd.DataFrame(
            np.arange(N_BATCHES * 2, dtype=float).reshape(N_BATCHES, 2),
            index=ids,
            columns=["charge", "purity"],
        )
        unfolded = unfold_blocks(two_blocks, initial_conditions=z)

        assert set(unfolded) == {"spectra", "process", "initial_conditions"}
        assert unfolded["initial_conditions"].shape == (N_BATCHES, 2)
        # Unchanged width on the trajectory blocks: nothing was glued on.
        assert unfolded["spectra"].shape[1] == 8 * 3

    def test_initial_conditions_are_reordered_to_match(self, two_blocks: dict[str, dict], ids: list[str]) -> None:
        """Z given in a different row order still lines up with the trajectories."""
        z = pd.DataFrame({"charge": np.arange(N_BATCHES, dtype=float)}, index=ids)
        unfolded = unfold_blocks(two_blocks, initial_conditions=z.iloc[::-1])
        assert (
            unfolded["initial_conditions"]["charge"].to_list()
            == z.reindex(unfolded["spectra"].index)["charge"].to_list()
        )

    def test_blocks_must_cover_the_same_batches(self, two_blocks: dict[str, dict]) -> None:
        short = dict(list(two_blocks["process"].items())[:-1])
        with pytest.raises(ValueError, match="must cover the same batches"):
            unfold_blocks({"spectra": two_blocks["spectra"], "process": short})

    def test_empty_blocks_rejected(self) -> None:
        with pytest.raises(ValueError, match="At least one block"):
            unfold_blocks({})

    def test_reserved_block_name_rejected(self, two_blocks: dict[str, dict], ids: list[str]) -> None:
        """Silently overwriting the caller's block would lose data."""
        blocks = {**two_blocks, "initial_conditions": two_blocks["process"]}
        z = pd.DataFrame({"charge": np.zeros(N_BATCHES)}, index=ids)
        with pytest.raises(ValueError, match="reserved"):
            unfold_blocks(blocks, initial_conditions=z)

    def test_feeds_mbpca_directly(self, two_blocks: dict[str, dict]) -> None:
        """The acceptance criterion: aligned batches into a multi-block model, no reshaping."""
        model = MBPCA(n_components=2).fit(unfold_blocks(two_blocks))
        assert model.super_scores_.shape == (N_BATCHES, 2)
        assert model.block_names_ == ["spectra", "process"]


# ---------------------------------------------------------------------------
# BlockSet
# ---------------------------------------------------------------------------


class TestBlockSet:
    def test_len_is_rows_not_blocks(self, two_blocks: dict[str, dict]) -> None:
        """Documented and deliberate, matching DataFrameDict; pinned so it cannot drift."""
        blocks = BlockSet(unfold_blocks(two_blocks))
        assert len(blocks) == N_BATCHES
        assert len(blocks.keys()) == 2

    def test_is_still_a_plain_dict_to_its_consumers(self, two_blocks: dict[str, dict]) -> None:
        """Subclassing dict is what lets existing code keep working untouched."""
        wide = unfold_blocks(two_blocks)
        blocks = BlockSet(wide)
        assert isinstance(blocks, dict)
        pd.testing.assert_frame_equal(blocks["spectra"], wide["spectra"])
        assert MBPCA(n_components=2).fit(blocks).super_scores_.shape == (N_BATCHES, 2)

    @pytest.mark.parametrize("lookup", [[0, 1, 2], np.array([0, 1, 2])])
    def test_row_slicing_hits_every_block(self, two_blocks: dict[str, dict], lookup: object) -> None:
        blocks = BlockSet(unfold_blocks(two_blocks))
        taken = blocks[lookup]
        assert isinstance(taken, BlockSet)
        assert len(taken) == 3
        for name, block in taken.items():
            assert block.shape[1] == blocks[name].shape[1]
            assert block.index.to_list() == blocks[name].index[:3].to_list()

    def test_single_row_stays_two_dimensional(self, two_blocks: dict[str, dict]) -> None:
        """A one-row resample must still arrive at fit() as a frame, not a Series."""
        taken = BlockSet(unfold_blocks(two_blocks))[0]
        for block in taken.values():
            assert isinstance(block, pd.DataFrame)
            assert block.shape[0] == 1

    def test_unequal_row_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            BlockSet({"a": pd.DataFrame(np.zeros((4, 2))), "b": pd.DataFrame(np.zeros((3, 2)))})

    def test_non_frame_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            BlockSet({"a": pd.DataFrame(np.zeros((4, 2))), "b": np.zeros((4, 2))})  # type: ignore[dict-item]

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="At least one block"):
            BlockSet({})

    def test_equal_content_compares_equal(self, two_blocks: dict[str, dict]) -> None:
        """The inherited dict.__eq__ raised here: it asked a DataFrame for its truth value."""
        left = BlockSet(unfold_blocks(two_blocks))
        right = BlockSet(unfold_blocks(two_blocks))
        assert left is not right
        assert left == right
        assert (left != right) is False

    def test_differing_content_compares_unequal(self, two_blocks: dict[str, dict]) -> None:
        wide = unfold_blocks(two_blocks)
        changed = {name: block.copy() for name, block in wide.items()}
        changed["spectra"].iloc[0, 0] += 1.0
        assert BlockSet(wide) != BlockSet(changed)
        assert BlockSet(wide) != BlockSet({"spectra": wide["spectra"]})

    def test_compares_against_the_plain_dict_it_came_from(self, two_blocks: dict[str, dict]) -> None:
        """Symmetric: Python tries the subclass's __eq__ first, so both orders answer."""
        wide = unfold_blocks(two_blocks)
        assert BlockSet(wide) == wide
        assert wide == BlockSet(wide)
        assert BlockSet(wide) != {"spectra": 1, "process": 2}

    def test_foreign_types_compare_unequal_rather_than_raising(self, two_blocks: dict[str, dict]) -> None:
        blocks = BlockSet(unfold_blocks(two_blocks))
        assert blocks != 7
        assert (blocks == 7) is False

    def test_unhashable_like_a_dict(self, two_blocks: dict[str, dict]) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            hash(BlockSet(unfold_blocks(two_blocks)))

    def test_bad_row_lookup_is_rejected(self, two_blocks: dict[str, dict]) -> None:
        """Not a str (a block name) and not a row lookup: say so instead of failing deeper in pandas."""
        with pytest.raises(TypeError, match="must be an int, a list of ints, or an ndarray"):
            BlockSet(unfold_blocks(two_blocks))[{0, 1}]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Resampler over multi-block data: the #193 item itself
# ---------------------------------------------------------------------------


class TestMultiBlockResampling:
    def test_jackknife_over_a_multi_block_model(self, two_blocks: dict[str, dict]) -> None:
        """Before #193 this was simply not expressible: Resampler took DataFrameDict only."""
        blocks = BlockSet(unfold_blocks(two_blocks))
        resampler = Resampler(
            estimator=MBPCA(n_components=2),
            x=blocks,
            accessor=lambda model: model.super_loadings_[1].to_numpy(),
            use_jackknife=True,
        ).resample(show_progress=False)

        # Leave-one-out: one refit per batch, each giving one loading per block.
        assert resampler.n_resamples == N_BATCHES
        assert np.asarray(resampler.parameters).shape == (N_BATCHES, 2)
        assert np.isfinite(np.asarray(resampler.parameters)).all()

    def test_bootstrap_is_reproducible(self, two_blocks: dict[str, dict]) -> None:
        """random_state reaches the multi-block path too (reproducibility.rst)."""
        blocks = BlockSet(unfold_blocks(two_blocks))

        def run(seed: int) -> np.ndarray:
            return np.asarray(
                Resampler(
                    estimator=MBPCA(n_components=2),
                    x=blocks,
                    accessor=lambda model: model.super_loadings_[1].to_numpy(),
                    use_jackknife=False,
                    bootstrap_rounds=4,
                    random_state=seed,
                )
                .resample(show_progress=False)
                .parameters
            )

        np.testing.assert_allclose(run(0), run(0))
        assert not np.allclose(run(0), run(1))

    def test_plain_dict_still_refused(self, two_blocks: dict[str, dict]) -> None:
        """A dict has no row axis; guessing one is how a resample ends up misaligned."""
        with pytest.raises(TypeError, match="BlockSet"):
            Resampler(
                estimator=MBPCA(n_components=2),
                x=unfold_blocks(two_blocks),  # type: ignore[arg-type]
                accessor=lambda model: model,
                use_jackknife=True,
            )
