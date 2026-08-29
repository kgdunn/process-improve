"""Tests for the product-by-compound preprocessing pipeline.

The pipeline is trim, transform, centre, scale, in that order, with an
``apply_fitted_*`` partner for each fitting step so a held-out row can be
preprocessed without having seen its own values. Several of the tests below pin
choices that are easy to reverse and quiet when reversed: zeros default to
``unknown`` rather than censored, trimmed compounds come back as a presence
layer, and ``detected_only`` defaults to ``False``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.chemistry import (
    apply_fitted_center_scale,
    apply_fitted_transform,
    apply_transform,
    center_and_scale,
    choose_transform,
    classify_zero_states,
    normalisation_check,
    trim_by_prevalence,
)


@pytest.fixture
def chem() -> pd.DataFrame:
    """Build a small block: one wide-range compound, one narrow, one rare, one absent."""
    return pd.DataFrame(
        {
            "wide": [0.0, 1.0, 12.0, 400.0, 3.0, 55.0],
            "narrow": [10.0, 12.0, 11.0, 13.0, 12.5, 11.5],
            "rare": [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
            "absent": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=[f"product_{i}" for i in range(6)],
    )


class TestClassifyZeroStates:
    def test_defaults_to_unknown(self, chem: pd.DataFrame) -> None:
        """Never default to censored: a zero nobody has spoken for is unknown."""
        states = classify_zero_states(chem).set_index("compound")
        assert set(states["zero_state"]) == {"unknown"}
        assert set(states["source"]) == {"default"}

    def test_declared_states_win(self, chem: pd.DataFrame) -> None:
        states = classify_zero_states(chem, declared={"rare": "essential"}).set_index("compound")
        assert states.loc["rare", "zero_state"] == "essential"
        assert states.loc["rare", "source"] == "declared"
        assert states.loc["wide", "zero_state"] == "unknown"

    def test_a_detection_limit_is_a_declaration_that_zeros_are_censored(self, chem: pd.DataFrame) -> None:
        states = classify_zero_states(chem, lod={"wide": 0.5}).set_index("compound")
        assert states.loc["wide", "zero_state"] == "rounded"
        assert states.loc["wide", "source"] == "lod"
        assert states.loc["wide", "lod"] == pytest.approx(0.5)

    def test_declared_overrides_a_detection_limit(self, chem: pd.DataFrame) -> None:
        states = classify_zero_states(chem, declared={"wide": "essential"}, lod={"wide": 0.5}).set_index("compound")
        assert states.loc["wide", "zero_state"] == "essential"
        assert states.loc["wide", "source"] == "declared"

    def test_zero_counts_show_where_the_question_matters(self, chem: pd.DataFrame) -> None:
        states = classify_zero_states(chem).set_index("compound")
        assert states.loc["narrow", "n_zero"] == 0
        assert states.loc["rare", "n_zero"] == 4
        assert states.loc["rare", "n_nonzero"] == 2

    def test_a_typo_in_declared_is_an_error(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not columns of the block"):
            classify_zero_states(chem, declared={"widee": "essential"})

    def test_an_unknown_state_name_is_an_error(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="declared states must be one of"):
            classify_zero_states(chem, declared={"wide": "censored"})

    def test_empty_block_raises(self) -> None:
        with pytest.raises(ValueError, match="no products"):
            classify_zero_states(pd.DataFrame(columns=["a", "b"]))


class TestTrimByPrevalence:
    def test_splits_on_the_nonzero_count(self, chem: pd.DataFrame) -> None:
        kept, dropped, _presence = trim_by_prevalence(chem, min_nonzero=3)
        assert list(kept.columns) == ["wide", "narrow"]
        assert list(dropped.columns) == ["rare", "absent"]

    def test_presence_covers_every_compound_not_only_the_kept_ones(self, chem: pd.DataFrame) -> None:
        """A rare compound's binary fingerprint often says more than its concentration."""
        _kept, dropped, presence = trim_by_prevalence(chem, min_nonzero=3)
        assert list(presence.columns) == list(chem.columns)
        assert presence.shape == chem.shape
        for column in dropped.columns:
            assert column in presence.columns
        assert presence["rare"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    def test_rows_are_preserved_in_both_halves(self, chem: pd.DataFrame) -> None:
        kept, dropped, presence = trim_by_prevalence(chem)
        for frame in (kept, dropped, presence):
            assert list(frame.index) == list(chem.index)

    def test_a_missing_measurement_is_not_an_absence(self) -> None:
        block = pd.DataFrame({"a": [1.0, np.nan, 0.0]})
        _kept, _dropped, presence = trim_by_prevalence(block, min_nonzero=1)
        assert presence["a"].tolist()[0] == 1.0
        assert np.isnan(presence["a"].tolist()[1])
        assert presence["a"].tolist()[2] == 0.0

    def test_min_nonzero_zero_keeps_everything(self, chem: pd.DataFrame) -> None:
        kept, dropped, _presence = trim_by_prevalence(chem, min_nonzero=0)
        assert list(kept.columns) == list(chem.columns)
        assert dropped.shape[1] == 0

    def test_negative_min_nonzero_raises(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="min_nonzero"):
            trim_by_prevalence(chem, min_nonzero=-1)


class TestNormalisationCheck:
    def test_constant_sum_data_has_nothing_outside(self) -> None:
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(rng.uniform(1, 10, size=(8, 5)))
        closed = raw.div(raw.sum(axis=1), axis=0) * 100.0
        totals, outside = normalisation_check(closed)
        assert np.allclose(totals.to_numpy(), 100.0)
        assert outside.empty

    def test_a_diluted_row_is_reported(self) -> None:
        block = pd.DataFrame({"a": [10.0, 10.0, 10.0, 10.0, 1.0], "b": [5.0, 5.0, 5.0, 5.0, 0.5]})
        totals, outside = normalisation_check(block, factor=1.8)
        assert list(outside.index) == [4]
        assert totals.iloc[4] == pytest.approx(1.5)

    def test_the_band_is_symmetric_in_fold_change(self) -> None:
        block = pd.DataFrame({"a": [10.0, 10.0, 10.0, 20.0, 5.0]})
        _totals, outside = normalisation_check(block, factor=1.8)
        assert sorted(outside.index) == [3, 4]

    def test_an_all_missing_row_totals_to_nan_rather_than_zero(self) -> None:
        block = pd.DataFrame({"a": [10.0, np.nan], "b": [10.0, np.nan]})
        totals, _outside = normalisation_check(block)
        assert np.isnan(totals.iloc[1])

    def test_factor_must_exceed_one(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="factor"):
            normalisation_check(chem, factor=1.0)

    def test_a_non_positive_median_total_raises(self) -> None:
        block = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match="median row total"):
            normalisation_check(block)


class TestChooseTransform:
    def test_orders_of_magnitude_choose_log(self) -> None:
        assert choose_transform(pd.Series([1.0, 10.0, 500.0])) == "log"

    def test_a_narrow_range_chooses_linear(self) -> None:
        assert choose_transform(pd.Series([10.0, 12.0, 13.0])) == "linear"

    def test_the_middle_is_ambiguous_rather_than_a_coin_toss(self) -> None:
        assert choose_transform(pd.Series([1.0, 5.0])) == "ambiguous"

    def test_zeros_do_not_enter_the_range_ratio(self) -> None:
        """Otherwise every column with a zero would have an infinite ratio."""
        assert choose_transform(pd.Series([0.0, 10.0, 12.0])) == "linear"

    def test_a_negative_value_makes_a_log_inapplicable(self) -> None:
        assert choose_transform(pd.Series([-1.0, 10.0, 5000.0])) == "linear"

    def test_fewer_than_two_detected_values_gives_linear(self) -> None:
        assert choose_transform(pd.Series([0.0, 0.0, 7.0])) == "linear"

    def test_overlapping_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            choose_transform(pd.Series([1.0, 2.0]), ratio_log=2.0, ratio_linear=3.0)


class TestApplyTransform:
    def test_rule_and_offset_are_recorded_per_compound(self, chem: pd.DataFrame) -> None:
        _transformed, applied = apply_transform(chem)
        table = applied.set_index("compound")
        assert table.loc["wide", "rule"] == "log"
        assert table.loc["narrow", "rule"] == "linear"
        assert table.loc["narrow", "offset"] == 0.0

    def test_a_linear_compound_passes_through_untouched(self, chem: pd.DataFrame) -> None:
        transformed, _applied = apply_transform(chem)
        pd.testing.assert_series_equal(transformed["narrow"], chem["narrow"], check_names=False)

    def test_a_declared_detection_limit_sets_the_substitution(self, chem: pd.DataFrame) -> None:
        transformed, applied = apply_transform(chem, lod={"wide": 0.4})
        table = applied.set_index("compound")
        assert table.loc["wide", "offset"] == pytest.approx(0.2)
        assert transformed["wide"].iloc[0] == pytest.approx(np.log10(0.2))

    def test_without_a_limit_the_substitution_is_half_the_smallest_seen(self, chem: pd.DataFrame) -> None:
        transformed, applied = apply_transform(chem)
        table = applied.set_index("compound")
        assert table.loc["wide", "offset"] == pytest.approx(0.5)
        assert transformed["wide"].iloc[0] == pytest.approx(np.log10(0.5))

    def test_detected_values_survive_the_log_exactly(self, chem: pd.DataFrame) -> None:
        transformed, _applied = apply_transform(chem)
        assert transformed["wide"].iloc[3] == pytest.approx(np.log10(400.0))

    def test_ambiguous_default_is_recorded(self) -> None:
        block = pd.DataFrame({"a": [1.0, 5.0, 3.0]})
        _transformed, applied = apply_transform(block, ambiguous="log")
        table = applied.set_index("compound")
        assert table.loc["a", "rule"] == "log"
        assert table.loc["a", "chosen_by"] == "ambiguous_default"

    def test_ambiguous_defaults_to_linear(self) -> None:
        block = pd.DataFrame({"a": [1.0, 5.0, 3.0]})
        _transformed, applied = apply_transform(block)
        assert applied.set_index("compound").loc["a", "rule"] == "linear"

    def test_missing_cells_stay_missing_through_a_log(self) -> None:
        block = pd.DataFrame({"a": [1.0, np.nan, 0.0, 900.0]})
        transformed, _applied = apply_transform(block)
        assert np.isnan(transformed["a"].iloc[1])
        assert not np.isnan(transformed["a"].iloc[2])

    def test_a_bad_ambiguous_value_raises(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="ambiguous must be"):
            apply_transform(chem, ambiguous="whatever")

    def test_a_typo_in_lod_raises(self, chem: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not columns of the block"):
            apply_transform(chem, lod={"widee": 1.0})


class TestApplyFittedTransform:
    def test_replays_the_training_decisions(self, chem: pd.DataFrame) -> None:
        train, test = chem.iloc[:4], chem.iloc[4:]
        train_t, applied = apply_transform(train)
        test_t = apply_fitted_transform(test, applied)
        # The rule and the offset both came from the training rows only.
        assert applied.set_index("compound").loc["wide", "rule"] == "log"
        assert test_t["wide"].iloc[0] == pytest.approx(np.log10(3.0))
        assert train_t.shape == (4, 4)

    def test_a_test_row_does_not_change_its_own_offset(self, chem: pd.DataFrame) -> None:
        """The whole point: a smaller value in the test rows must not move the offset."""
        train = chem.iloc[:4]
        test = pd.DataFrame({"wide": [0.0], "narrow": [11.0], "rare": [0.0], "absent": [0.0]}, index=["held_out"])
        _train_t, applied = apply_transform(train)
        offset = float(applied.set_index("compound").loc["wide", "offset"])
        test_t = apply_fitted_transform(test, applied)
        assert test_t["wide"].iloc[0] == pytest.approx(np.log10(offset))

    def test_round_trips_with_apply_transform_on_the_same_rows(self, chem: pd.DataFrame) -> None:
        transformed, applied = apply_transform(chem)
        pd.testing.assert_frame_equal(apply_fitted_transform(chem, applied), transformed)

    def test_a_missing_compound_raises(self, chem: pd.DataFrame) -> None:
        _transformed, applied = apply_transform(chem[["wide"]])
        with pytest.raises(ValueError, match="no entry for compound"):
            apply_fitted_transform(chem, applied)

    def test_a_missing_column_in_the_table_raises(self, chem: pd.DataFrame) -> None:
        _transformed, applied = apply_transform(chem)
        with pytest.raises(ValueError, match="must carry the columns"):
            apply_fitted_transform(chem, applied.drop(columns=["offset"]))

    def test_an_unknown_rule_raises(self, chem: pd.DataFrame) -> None:
        _transformed, applied = apply_transform(chem)
        applied.loc[0, "rule"] = "sqrt"
        with pytest.raises(ValueError, match="unknown transform rule"):
            apply_fitted_transform(chem, applied)


class TestCenterAndScale:
    @staticmethod
    def _blocks(chem: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        _kept, _dropped, presence = trim_by_prevalence(chem, min_nonzero=0)
        transformed, _applied = apply_transform(chem)
        return transformed, presence

    def test_autoscale_gives_unit_variance(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        scaled, _constants = center_and_scale(transformed, presence)
        assert scaled["wide"].mean() == pytest.approx(0.0, abs=1e-12)
        assert scaled["wide"].std(ddof=1) == pytest.approx(1.0)

    def test_pareto_keeps_the_large_columns_larger(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        auto, _ = center_and_scale(transformed, presence, method="autoscale")
        pareto, _ = center_and_scale(transformed, presence, method="pareto")
        # Pareto divides by sqrt(sd), so a column with sd > 1 keeps more of its spread.
        assert transformed["wide"].std(ddof=1) > 1
        assert pareto["wide"].std(ddof=1) > auto["wide"].std(ddof=1)

    def test_constants_record_a_divisor_not_a_multiplier(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        _scaled, constants = center_and_scale(transformed, presence)
        table = constants.set_index("compound")
        assert table.loc["narrow", "divisor"] == pytest.approx(transformed["narrow"].std(ddof=1))
        assert table.loc["narrow", "center"] == pytest.approx(transformed["narrow"].mean())

    def test_detected_only_defaults_to_false(self, chem: pd.DataFrame) -> None:
        """The zeros here were never imputed, so they are real observations."""
        transformed, presence = self._blocks(chem)
        _scaled, constants = center_and_scale(transformed, presence)
        assert constants.set_index("compound").loc["rare", "n_used"] == len(chem)

    def test_detected_only_uses_only_the_detected_cells(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        _scaled, constants = center_and_scale(transformed, presence, detected_only=True)
        assert constants.set_index("compound").loc["rare", "n_used"] == 2

    def test_detected_only_pushes_undetected_cells_far_from_the_centre(self, chem: pd.DataFrame) -> None:
        """Why it is off by default: on un-imputed zeros the column becomes a huge binary."""
        transformed, presence = self._blocks(chem)
        default, _ = center_and_scale(transformed, presence)
        opted_in, _ = center_and_scale(transformed, presence, detected_only=True)
        assert abs(opted_in["rare"].iloc[0]) > 5 * abs(default["rare"].iloc[0])

    def test_a_constant_column_passes_through(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        scaled, constants = center_and_scale(transformed, presence)
        assert constants.set_index("compound").loc["absent", "divisor"] == 1.0
        assert np.allclose(scaled["absent"].to_numpy(), 0.0)

    def test_a_bad_method_raises(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        with pytest.raises(ValueError, match="method must be one of"):
            center_and_scale(transformed, presence, method="range")

    def test_a_mismatched_detected_layer_raises(self, chem: pd.DataFrame) -> None:
        transformed, presence = self._blocks(chem)
        with pytest.raises(ValueError, match="exactly the rows and columns"):
            center_and_scale(transformed, presence[["wide"]])


class TestApplyFittedCenterScale:
    def test_round_trips_on_the_same_rows(self, chem: pd.DataFrame) -> None:
        _kept, _dropped, presence = trim_by_prevalence(chem, min_nonzero=0)
        transformed, _applied = apply_transform(chem)
        scaled, constants = center_and_scale(transformed, presence)
        pd.testing.assert_frame_equal(apply_fitted_center_scale(transformed, constants), scaled)

    def test_divides_rather_than_multiplies(self) -> None:
        block = pd.DataFrame({"a": [0.0, 2.0, 4.0, 6.0]})
        presence = pd.DataFrame({"a": [0.0, 1.0, 1.0, 1.0]})
        _scaled, constants = center_and_scale(block, presence)
        divisor = float(constants.set_index("compound").loc["a", "divisor"])
        centre = float(constants.set_index("compound").loc["a", "center"])
        held_out = pd.DataFrame({"a": [10.0]})
        assert apply_fitted_center_scale(held_out, constants)["a"].iloc[0] == pytest.approx((10.0 - centre) / divisor)

    def test_a_missing_compound_raises(self, chem: pd.DataFrame) -> None:
        _kept, _dropped, presence = trim_by_prevalence(chem, min_nonzero=0)
        transformed, _applied = apply_transform(chem)
        _scaled, constants = center_and_scale(transformed[["wide"]], presence[["wide"]])
        with pytest.raises(ValueError, match="no entry for compound"):
            apply_fitted_center_scale(transformed, constants)

    def test_a_hand_edited_divisor_of_zero_raises(self, chem: pd.DataFrame) -> None:
        _kept, _dropped, presence = trim_by_prevalence(chem, min_nonzero=0)
        transformed, _applied = apply_transform(chem)
        _scaled, constants = center_and_scale(transformed, presence)
        constants.loc[0, "divisor"] = 0.0
        with pytest.raises(ValueError, match="non-positive or non-finite divisor"):
            apply_fitted_center_scale(transformed, constants)


class TestNestedPipeline:
    """The reason the ``apply_fitted_*`` pair exist: preprocessing a fold honestly."""

    def test_held_out_rows_never_influence_their_own_constants(self, chem: pd.DataFrame) -> None:
        train_rows = [0, 1, 2, 3]
        test_rows = [4, 5]
        train, test = chem.iloc[train_rows], chem.iloc[test_rows]

        kept, _dropped, presence = trim_by_prevalence(train, min_nonzero=2)
        train_t, applied = apply_transform(kept)
        train_s, constants = center_and_scale(train_t, presence[kept.columns])

        test_t = apply_fitted_transform(test[kept.columns], applied)
        test_s = apply_fitted_center_scale(test_t, constants)

        assert list(test_s.columns) == list(train_s.columns)
        assert list(test_s.index) == list(test.index)

        # Perturbing a test row cannot move any training constant.
        perturbed = chem.copy()
        perturbed.iloc[test_rows] *= 1000.0
        kept2, _d2, presence2 = trim_by_prevalence(perturbed.iloc[train_rows], min_nonzero=2)
        _train_t2, applied2 = apply_transform(kept2)
        _train_s2, constants2 = center_and_scale(_train_t2, presence2[kept2.columns])
        pd.testing.assert_frame_equal(applied, applied2)
        pd.testing.assert_frame_equal(constants, constants2)

    def test_the_fixed_order_is_trim_transform_centre_scale(self, chem: pd.DataFrame) -> None:
        """Transforming before trimming would derive offsets from compounds about to go."""
        kept, _dropped, presence = trim_by_prevalence(chem, min_nonzero=3)
        transformed, applied = apply_transform(kept)
        scaled, constants = center_and_scale(transformed, presence[kept.columns])
        assert list(applied["compound"]) == list(kept.columns)
        assert list(constants["compound"]) == list(kept.columns)
        assert scaled.shape == kept.shape
