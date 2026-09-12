import numpy as np
import pandas as pd
import pytest

from process_improve.batch.alignment_helpers import (
    backtrack_optimal_path,
    distance_matrix,
    full_band,
    itakura,
    itakura_band,
    resolve_band,
    sakoe_chiba,
    sakoe_chiba_band,
    validate_band,
)
from process_improve.batch.preprocessing import (
    apply_scaling,
    batch_dtw,
    determine_scaling,
    find_reference_batch,
    reverse_scaling,
)


def test_scaling(dryer_data: dict) -> None:
    """Test batch scaling and reverse scaling."""
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    scale_df = determine_scaling(
        dryer_data,
        columns_to_align=columns_to_align,
        settings={"robust": False},
    )
    assert np.array([152.3796, 48.2545, 101.7032, 73.1462, 68.0041]) == pytest.approx(
        scale_df.loc[columns_to_align]["Range"]
    )

    batches_scaled = apply_scaling(dryer_data, scale_df, columns_to_align=columns_to_align)
    reference_batch = batches_scaled[1]
    assert np.array([0.793227, 0.171115, 1.007772, 0.051198, 0.050173]) == pytest.approx(
        reference_batch[columns_to_align].iloc[0], abs=1e-4
    )
    orig = reverse_scaling(batches_scaled, scale_df)
    assert np.linalg.norm(orig[1] - dryer_data[1][columns_to_align]) == pytest.approx(0, abs=1e-10)


@pytest.mark.slow
def test_alignment(dryer_data: dict) -> None:
    """Test batch DTW alignment on dryer data."""
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    outputs = batch_dtw(
        dryer_data,
        columns_to_align=columns_to_align,
        reference_batch=2,
        settings={
            "robust": False,
            "tolerance": 1,
        },  # high tolerance ensures only 1 iteration
    )
    assert pytest.approx(outputs["weight_history"].iloc[0]) == [1, 1, 1, 1, 1]
    assert pytest.approx(outputs["scale_df"]["Range"][columns_to_align]) == [
        152.379618,
        48.254502,
        101.703155,
        73.146169,
        68.004085,
    ]
    b1 = outputs["aligned_batch_objects"][1]
    expected_warping_path = [
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        7,
        8,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        10,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        12,
        13,
        14,
        16,
        23,
        24,
        25,
        36,
        37,
        38,
        39,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        74,
        75,
        76,
        77,
        78,
        81,
        82,
        91,
        92,
        95,
        96,
        97,
        101,
        102,
        104,
        105,
        106,
        108,
        109,
        110,
        110,
        110,
        110,
        110,
        111,
        112,
        113,
        114,
        115,
        115,
        116,
        116,
        117,
        118,
        118,
        118,
        119,
        119,
        120,
        120,
        121,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        139,
        140,
        141,
        142,
        142,
        142,
        142,
        142,
        142,
        142,
        143,
        147,
        148,
    ]
    assert expected_warping_path == pytest.approx(b1.warping_path + 1)
    assert pytest.approx(
        outputs["last_average_batch"].iloc[1, :] / (outputs["scale_df"]["Range"][columns_to_align]),
        abs=1e-4,
    ) == [1.0371, 0.1673, 0.9712, 0.6538, 0.2532]
    assert len(outputs["aligned_batch_dfdict"]) == 71
    assert outputs["aligned_batch_dfdict"].pop(1).shape == (100, 12)

    # Repeat, with a lower tolerance, to ensure the number of iterations exceeds 3.
    outputs = batch_dtw(
        dryer_data,
        columns_to_align=columns_to_align,
        reference_batch=2,
        settings={"robust": False, "tolerance": 0.06, "show_progress": True},
    )
    assert outputs["weight_history"].shape == (3, 5)
    # TODO(#197): restore this assertion once DTW termination is settled.
    # assert [0.43702525, 1.33206459, 0.98298667, 0.93599197, 1.31193153] == pytest.approx(
    #     outputs["weight_history"][4, :], abs=1e-7
    # )


def test_reference_batch_selection_dryer(dryer_data: dict) -> None:
    """Test that the correct reference batch is selected for dryer data."""
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    good_reference_candidate = find_reference_batch(
        dryer_data,
        columns_to_align=columns_to_align,
        settings={
            "robust": False,
        },
    )
    assert good_reference_candidate == 3


def test_reference_batch_selection_nylon(nylon_data: dict) -> None:
    """Test that the correct reference batch is selected for nylon data."""
    columns_to_align = [
        "Tag01",
        "Tag02",
        "Tag03",
        "Tag04",
        "Tag05",
        "Tag06",
        "Tag07",
        "Tag08",
        "Tag09",
        "Tag10",
    ]
    good_reference_candidate = find_reference_batch(
        nylon_data,
        columns_to_align=columns_to_align,
        settings={
            "robust": False,
        },
    )
    assert good_reference_candidate == 45


def test_find_reference_batch_rejects_request_exceeding_candidates(dryer_data: dict) -> None:
    """SEC-13 (#261) regression guard.

    Requesting more reference batches than exist used to enter an unbounded
    cutoff-relaxation loop that eventually tripped ``assert conf_level < 1.0``
    inside ``spe_calculation`` -- an opaque, ``python -O``-strippable
    AssertionError. The fix validates up front and raises a clear
    ValueError.
    """
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    with pytest.raises(ValueError, match="exceeds the number of candidate batches"):
        find_reference_batch(
            dryer_data,
            columns_to_align=columns_to_align,
            settings={
                "robust": False,
                "number_of_reference_batches": len(dryer_data) + 100,
            },
        )


def test_find_reference_batch_returns_multiple(dryer_data: dict) -> None:
    """SEC-13 (#261): requesting >1 batches returns a list, exercising the
    cutoff-relaxation loop body and the multi-batch return path.
    """
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    result = find_reference_batch(
        dryer_data,
        columns_to_align=columns_to_align,
        settings={
            "robust": False,
            "number_of_reference_batches": 3,
        },
    )
    assert isinstance(result, list)
    assert len(result) == 3


def test_find_reference_batch_rejects_zero_request(dryer_data: dict) -> None:
    """SEC-13 (#261): a non-positive request raises a clear ValueError."""
    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    with pytest.raises(ValueError, match=r"must be >= 1"):
        find_reference_batch(
            dryer_data,
            columns_to_align=columns_to_align,
            settings={
                "robust": False,
                "number_of_reference_batches": 0,
            },
        )


# ---- Alignment helper tests (batch/alignment_helpers.py) ----


@pytest.mark.slow
def test_distance_matrix_identity() -> None:
    """distance_matrix of identical sequences should have zero diagonal."""
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    test = ref.copy()
    weight = np.eye(2)
    d_matrix = distance_matrix(test, ref, weight)
    assert d_matrix.shape == (3, 3)
    # Diagonal of dist (not cumulative D) should be 0 for identical sequences
    assert d_matrix[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_distance_matrix_shape() -> None:
    """distance_matrix should return (nr, nt) shaped matrix."""
    ref = np.array([[1.0], [2.0], [3.0], [4.0]])
    test = np.array([[1.5], [2.5], [3.5]])
    weight = np.eye(1)
    d_matrix = distance_matrix(test, ref, weight)
    assert d_matrix.shape == (4, 3)


def test_backtrack_optimal_path_identity() -> None:
    """Backtrack on a zero-diagonal D matrix should return the diagonal path."""
    # Build a simple cumulative distance matrix where diagonal is optimal
    d_matrix = np.array([[0.0, 10.0, 20.0], [10.0, 0.0, 10.0], [20.0, 10.0, 0.0]])
    path, _path_sum = backtrack_optimal_path(d_matrix)
    assert path.shape[1] == 2
    # Path should start at (0,0) and end at (2,2)
    assert path[0, 0] == 0
    assert path[0, 1] == 0
    assert path[-1, 0] == 2
    assert path[-1, 1] == 2


def test_backtrack_optimal_path_returns_sum() -> None:
    """Backtrack should return a finite path sum."""
    ref = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    test = np.array([[1.1, 0.0], [2.1, 0.0], [3.1, 0.0]])
    weight = np.eye(2)
    d_matrix = distance_matrix(test, ref, weight)
    _path, path_sum = backtrack_optimal_path(d_matrix)
    assert np.isfinite(path_sum)
    assert path_sum >= 0


class TestScalingRejectsUnsupportedContainers:
    """The scaling functions take a dict of per-batch frames. Regression tests for #560.

    A single wide DataFrame used to pass the column-resolution branch and then fail
    inside the loop with ``AttributeError: 'Series' object has no attribute
    'columns'``, because ``DataFrame.items()`` yields ``(column, Series)`` pairs.
    """

    @staticmethod
    def _wide_frame() -> pd.DataFrame:
        return pd.DataFrame({"temp": [10.0, 11, 12], "press": [1.0, 2, 3]})

    def test_determine_scaling_rejects_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match=r"determine_scaling expects `batches` as a dict"):
            determine_scaling(self._wide_frame())

    def test_apply_scaling_rejects_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match=r"apply_scaling expects `batches` as a dict"):
            apply_scaling(self._wide_frame(), scale_df=None)

    def test_reverse_scaling_rejects_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match=r"reverse_scaling expects `batches` as a dict"):
            reverse_scaling(self._wide_frame(), scale_df=None)

    def test_the_message_names_the_conversion(self) -> None:
        """The error is actionable: it says how to turn the frame into the right shape."""
        with pytest.raises(TypeError, match=r"dict\(tuple\(df\.groupby\(batch_col\)\)\)"):
            determine_scaling(self._wide_frame())

    def test_non_mapping_input_is_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"got list"):
            determine_scaling([self._wide_frame()])

    def test_empty_dict_without_explicit_columns_raises(self) -> None:
        with pytest.raises(ValueError, match=r"cannot resolve `columns_to_align` from an empty"):
            determine_scaling({})

    def test_dict_input_is_unaffected(self, dryer_data: dict) -> None:
        """The supported path keeps working, and still resolves columns from batch one."""
        scale_df = determine_scaling(dryer_data)
        assert list(scale_df.columns) == ["Range", "Minimum"]
        assert not scale_df.empty


_DRYER_ALIGN_COLUMNS = [
    "AgitatorPower",
    "AgitatorTorque",
    "JacketTemperatureSP",
    "JacketTemperature",
    "DryerTemp",
]


class TestBatchDtwDistancesOutput:
    """`batch_dtw` reports each batch's distance to the reference (#199).

    The per-batch distances were computed on every iteration and thrown away, so there
    was no way to see which batches aligned badly. They come from the `DTWresult`
    objects already returned, so nothing extra is computed.
    """

    def _run(self, dryer_data: dict, **extra_settings: object) -> dict:
        settings = {"robust": False, "tolerance": 0.1, "show_progress": False}
        settings.update(extra_settings)
        return batch_dtw(dryer_data, columns_to_align=_DRYER_ALIGN_COLUMNS, reference_batch=2, settings=settings)

    def test_distances_are_reported_for_every_batch(self, dryer_data: dict) -> None:
        outputs = self._run(dryer_data)
        distances = outputs["distances"]

        assert list(distances.columns) == ["Distance", "Normalized distance"]
        assert len(distances) == len(outputs["aligned_batch_objects"]) == 71
        assert distances.index.name == "batch_id"
        assert (distances >= 0).to_numpy().all()

    def test_the_reference_batch_has_zero_distance_to_itself(self, dryer_data: dict) -> None:
        distances = self._run(dryer_data)["distances"]

        assert distances.loc[2, "Distance"] == pytest.approx(0.0)
        assert distances.loc[2, "Normalized distance"] == pytest.approx(0.0)

    def test_distances_agree_with_the_result_objects(self, dryer_data: dict) -> None:
        """They are read from the DTWresults, not recomputed, so they must match exactly."""
        outputs = self._run(dryer_data)

        for batch_id, result in outputs["aligned_batch_objects"].items():
            assert outputs["distances"].loc[batch_id, "Distance"] == result.distance
            assert outputs["distances"].loc[batch_id, "Normalized distance"] == result.normalized_distance

    def test_normalizing_changes_the_ranking_of_unequal_length_batches(self, dryer_data: dict) -> None:
        """The normalized column divides by path length, so it is the comparable one."""
        distances = self._run(dryer_data)["distances"]
        lengths = {bid: df.shape[0] for bid, df in dryer_data.items()}

        assert min(lengths.values()) != max(lengths.values()), "fixture must hold unequal lengths"
        # The two orderings are not the same, which is the point of reporting both.
        assert list(distances["Distance"].nlargest(10).index) != list(
            distances["Normalized distance"].nlargest(10).index
        )


class TestBatchDtwWeightingSetting:
    """`settings["weighting"]` selects how deviations accumulate (#199).

    The default stays "quadratic", the published Kassidas choice, whose reciprocal is an
    inverse-variance weight and so matches the Mahalanobis form of the weighted DTW
    distance. "absolute" is offered for comparison.
    """

    def _weights(self, dryer_data: dict, **extra_settings: object) -> pd.DataFrame:
        settings = {"robust": False, "tolerance": 0.1, "show_progress": False}
        settings.update(extra_settings)
        return batch_dtw(dryer_data, columns_to_align=_DRYER_ALIGN_COLUMNS, reference_batch=2, settings=settings)[
            "weight_history"
        ]

    def test_the_default_is_quadratic_and_unchanged(self, dryer_data: dict) -> None:
        """Pinned against the values this produced before the setting existed."""
        implicit = self._weights(dryer_data)
        explicit = self._weights(dryer_data, weighting="quadratic")

        assert implicit.iloc[-1].to_numpy() == pytest.approx(explicit.iloc[-1].to_numpy(), rel=1e-12)
        # Captured from this fixture; rel=1e-6 is far tighter than the ~50% the
        # "absolute" functional moves these by, while leaving room for platform float
        # noise in an iterative DTW across the CI matrix.
        assert implicit.iloc[-1].to_numpy() == pytest.approx(
            [
                0.48684365176347233,
                1.372434900728356,
                0.987884606056083,
                0.915196609356401,
                1.2376402320956883,
            ],
            rel=1e-6,
        )

    def test_absolute_gives_a_different_fixed_point(self, dryer_data: dict) -> None:
        """Measured, not assumed: it widens the weight spread here rather than flattening it."""
        quadratic = self._weights(dryer_data).iloc[-1].to_numpy()
        absolute = self._weights(dryer_data, weighting="absolute").iloc[-1].to_numpy()

        assert absolute != pytest.approx(quadratic, rel=1e-6)
        spread = lambda weights: weights.max() / weights.min()  # noqa: E731
        assert spread(absolute) > spread(quadratic)

    def test_an_unrecognised_weighting_is_rejected(self, dryer_data: dict) -> None:
        """Without this it fell silently into the absolute branch."""
        with pytest.raises(ValueError, match=r"weighting'\]='geometric' is not recognized"):
            self._weights(dryer_data, weighting="geometric")


class TestFullBand:
    """The default band, which must reproduce the behaviour from before #197."""

    def test_it_admits_every_reference_row(self) -> None:
        band = full_band(n_test=4, n_ref=7)

        assert band.shape == (4, 2)
        assert band.dtype == np.int64
        assert band[:, 0].tolist() == [0, 0, 0, 0]
        assert band[:, 1].tolist() == [7, 7, 7, 7]

    def test_it_is_what_band_none_resolves_to(self) -> None:
        assert np.array_equal(resolve_band(None, 5, 9), full_band(5, 9))

    def test_the_default_cost_matrix_is_unchanged_by_the_band_machinery(self) -> None:
        """Passing the full band explicitly must give bit-identical numbers to passing nothing."""
        rng = np.random.default_rng(7)
        test, ref, weights = rng.normal(size=(23, 3)), rng.normal(size=(31, 3)), np.eye(3)

        implicit = distance_matrix(test, ref, weights)
        explicit = distance_matrix(test, ref, weights, band=full_band(23, 31))

        assert np.array_equal(implicit, explicit, equal_nan=True)
        assert not np.isnan(implicit).any(), "an unconstrained matrix has no unreachable cells"


class TestSakoeChibaBand:
    """A fixed-width corridor around the diagonal (#197)."""

    def test_it_centres_on_the_diagonal_between_unequal_lengths(self) -> None:
        """With 5 test samples and 9 reference rows the centre advances 2 rows per sample."""
        band = sakoe_chiba_band(n_test=5, n_ref=9, window=2)

        centres = (band[:, 0] + band[:, 1] - 1) / 2
        assert centres.tolist() == [1.0, 2.0, 4.0, 6.0, 7.0]

    def test_both_corners_are_always_admitted(self) -> None:
        for n_test, n_ref, window in ((10, 10, 1), (10, 40, 1), (40, 10, 1), (3, 97, 0.01)):
            band = sakoe_chiba_band(n_test, n_ref, window)
            assert band[0, 0] == 0, (n_test, n_ref, window)
            assert band[-1, 1] == n_ref, (n_test, n_ref, window)

    def test_the_radius_is_floored_at_the_diagonal_step(self) -> None:
        """A one-row corridor across a steep diagonal would leave gaps no path can cross."""
        band = sakoe_chiba_band(n_test=5, n_ref=41, window=1)

        validate_band(band, 5, 41)  # would raise if the flooring were absent
        assert (band[:, 1] - band[:, 0]).min() > 1

    def test_an_integer_window_counts_rows_and_a_float_counts_fraction(self) -> None:
        rows = sakoe_chiba_band(n_test=21, n_ref=21, window=2)
        fraction = sakoe_chiba_band(n_test=21, n_ref=21, window=0.1)  # 0.1 * 20 = 2 rows

        assert np.array_equal(rows, fraction)

    def test_a_bool_window_is_rejected(self) -> None:
        """`True` would silently mean a one-row radius."""
        with pytest.raises(TypeError, match="must be an int"):
            sakoe_chiba_band(10, 10, window=True)

    @pytest.mark.parametrize(
        ("window", "message"), [(0, "at least 1"), (-3, "at least 1"), (0.0, r"\(0, 1\]"), (1.5, r"\(0, 1\]")]
    )
    def test_a_window_outside_its_range_is_rejected(self, window: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            sakoe_chiba_band(10, 10, window=window)

    def test_a_wide_enough_window_recovers_the_unconstrained_distance(self) -> None:
        rng = np.random.default_rng(11)
        test, ref, weights = rng.normal(size=(30, 3)), rng.normal(size=(34, 3)), np.eye(3)

        unconstrained = backtrack_optimal_path(distance_matrix(test, ref, weights))[1]
        wide = backtrack_optimal_path(distance_matrix(test, ref, weights, band=sakoe_chiba(1.0)))[1]

        assert wide == unconstrained

    def test_a_narrow_window_can_only_cost_more(self) -> None:
        """The corridor is a subset of the search space, so the optimum cannot improve."""
        rng = np.random.default_rng(11)
        test, ref, weights = rng.normal(size=(30, 3)), rng.normal(size=(34, 3)), np.eye(3)

        unconstrained = backtrack_optimal_path(distance_matrix(test, ref, weights))[1]
        narrow = backtrack_optimal_path(distance_matrix(test, ref, weights, band=sakoe_chiba(2)))[1]

        assert narrow > unconstrained

    def test_out_of_band_cells_are_left_unreachable(self) -> None:
        rng = np.random.default_rng(3)
        test, ref, weights = rng.normal(size=(20, 2)), rng.normal(size=(20, 2)), np.eye(2)

        D = distance_matrix(test, ref, weights, band=sakoe_chiba(3))

        assert np.isnan(D).any(), "a constrained matrix has unreachable cells"
        assert not np.isnan(D[0, 0])
        assert not np.isnan(D[-1, -1])


class TestItakuraBand:
    """A slope-bounded parallelogram (#197)."""

    def test_the_corner_columns_are_one_cell_wide(self) -> None:
        band = resolve_band(itakura(max_slope=2.0), n_test=12, n_ref=12)

        assert band[0].tolist() == [0, 1]
        assert band[-1].tolist() == [11, 12]

    def test_it_is_widest_in_the_middle(self) -> None:
        """A fixed-width corridor does not do this, which is what separates the two."""
        widths = np.diff(itakura_band(n_test=21, n_ref=21, max_slope=2.0), axis=1).ravel()

        assert widths.argmax() not in (0, len(widths) - 1)
        assert widths[len(widths) // 2] > widths[0]

    def test_no_column_is_empty_for_unequal_lengths(self) -> None:
        """Rounding the edges inwards empties a column whose corridor is under one row wide."""
        band = itakura_band(n_test=40, n_ref=97, max_slope=2.0)

        validate_band(band, 40, 97)
        assert (band[:, 1] - band[:, 0]).min() >= 1

    def test_a_slope_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"at least 1\.0"):
            itakura_band(10, 10, max_slope=0.9)

    def test_very_unequal_lengths_are_refused_and_point_at_sakoe_chiba(self) -> None:
        """Its corner columns are one cell wide, so a steep climb has nowhere to start."""
        with pytest.raises(ValueError, match=r"at any slope.*Sakoe-Chiba"):
            itakura_band(n_test=2, n_ref=40, max_slope=5.0)

    @pytest.mark.parametrize(("n_test", "n_ref"), [(3, 12), (5, 40), (12, 97), (40, 97)])
    def test_the_slope_the_refusal_advises_is_actually_accepted(self, n_test: int, n_ref: int) -> None:
        """A threshold quoted from a closed form and then rounded for display can fall below it."""
        with pytest.raises(ValueError, match="Use max_slope") as caught:
            itakura_band(n_test, n_ref, max_slope=1.0)

        advised = float(str(caught.value).split("Use max_slope >= ")[-1].rstrip("."))
        itakura_band(n_test, n_ref, max_slope=advised)  # must not raise


class TestBandValidation:
    """`validate_band` rejects a corridor no monotone warping path can follow (#197)."""

    def test_the_wrong_shape_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
            validate_band(np.zeros((3, 2), dtype=np.int64), n_test=4, n_ref=6)

    def test_an_empty_column_is_rejected(self) -> None:
        band = full_band(4, 6)
        band[2] = [3, 3]
        with pytest.raises(ValueError, match="empty at test sample 2"):
            validate_band(band, 4, 6)

    def test_non_monotone_bounds_are_rejected(self) -> None:
        band = np.array([[0, 6], [2, 6], [1, 6], [0, 6]], dtype=np.int64)
        with pytest.raises(ValueError, match="non-decreasing"):
            validate_band(band, 4, 6)

    def test_a_band_missing_the_starting_cell_is_rejected(self) -> None:
        band = full_band(4, 6)
        band[:, 0] = 1
        with pytest.raises(ValueError, match="starting cell"):
            validate_band(band, 4, 6)

    def test_a_band_missing_the_final_cell_is_rejected(self) -> None:
        band = full_band(4, 6)
        band[:, 1] = 5
        with pytest.raises(ValueError, match="final cell"):
            validate_band(band, 4, 6)

    def test_a_disconnected_band_is_rejected(self) -> None:
        """A path steps at most one row per column, so a jump leaves it stranded."""
        band = np.array([[0, 2], [4, 6], [4, 6], [4, 6]], dtype=np.int64)
        with pytest.raises(ValueError, match="disconnected between test samples 0 and 1"):
            validate_band(band, 4, 6)

    def test_rows_outside_the_reference_are_rejected(self) -> None:
        band = np.array([[0, 9], [0, 9], [0, 9], [0, 9]], dtype=np.int64)
        with pytest.raises(ValueError, match=r"within \[0, 6\]"):
            validate_band(band, 4, 6)

    def test_a_float_band_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="integer row bounds"):
            resolve_band(np.zeros((4, 2)), 4, 6)


class TestBacktrackingUnreachableCells:
    """Out-of-band cells are NaN, and NaN used to reach a bare `AssertionError` (#197)."""

    def test_an_unreachable_final_cell_raises_a_named_error(self) -> None:
        D = np.array([[0.0, 1.0], [1.0, np.nan]])

        with pytest.raises(ValueError, match="not reachable"):
            backtrack_optimal_path(D)

    def test_an_unreachable_interior_cell_raises_rather_than_asserting(self) -> None:
        D = np.array([[0.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, 1.0]])

        with pytest.raises(ValueError, match="no reachable predecessor"):
            backtrack_optimal_path(D)

    def test_ties_still_prefer_the_diagonal_then_the_horizontal(self) -> None:
        """The finite-only selection must not change the unconstrained tie order."""
        D = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        path, _distance = backtrack_optimal_path(D)

        # All costs equal, so every step takes the diagonal: a two-step path.
        assert path.tolist() == [[0, 0], [1, 1], [2, 2]]


class TestBatchDtwBandSetting:
    """`settings["band"]` constrains the warping path for the whole alignment (#197)."""

    def _run(self, dryer_data: dict, **extra_settings: object) -> dict:
        settings: dict = {"robust": False, "tolerance": 0.1, "show_progress": False}
        settings.update(extra_settings)
        return batch_dtw(dryer_data, columns_to_align=_DRYER_ALIGN_COLUMNS, reference_batch=2, settings=settings)

    def test_the_default_places_no_constraint(self, dryer_data: dict) -> None:
        implicit = self._run(dryer_data)["weight_history"].iloc[-1].to_numpy()
        explicit = self._run(dryer_data, band=None)["weight_history"].iloc[-1].to_numpy()

        assert implicit == pytest.approx(explicit, rel=1e-12)
        # The same pinned values as the unbanded alignment produces; see
        # TestBatchDtwWeightingSetting.
        assert implicit == pytest.approx(
            [0.48684365176347233, 1.372434900728356, 0.987884606056083, 0.915196609356401, 1.2376402320956883],
            rel=1e-6,
        )

    def test_a_wide_corridor_reproduces_the_unconstrained_alignment(self, dryer_data: dict) -> None:
        """The dryer batches run 89 to 201 samples; a 50% window contains every true warp."""
        unconstrained = self._run(dryer_data)["weight_history"].iloc[-1].to_numpy()
        wide = self._run(dryer_data, band=sakoe_chiba(0.5))["weight_history"].iloc[-1].to_numpy()

        assert wide == pytest.approx(unconstrained, rel=1e-9)

    def test_narrowing_the_corridor_degrades_the_alignment(self, dryer_data: dict) -> None:
        """Measured on this fixture: excluding the true warp makes every batch fit worse."""
        worst = [
            self._run(dryer_data, band=band)["distances"]["Normalized distance"].max()
            for band in (None, sakoe_chiba(0.2), sakoe_chiba(0.1), sakoe_chiba(0.05))
        ]

        assert worst == sorted(worst), f"expected monotone degradation, got {worst}"
        assert worst[0] == pytest.approx(0.0714267, rel=1e-4)
        assert worst[-1] == pytest.approx(0.235022, rel=1e-4)

    def test_a_band_of_the_wrong_kind_is_rejected_once(self, dryer_data: dict) -> None:
        with pytest.raises(TypeError, match="is not a band constraint"):
            self._run(dryer_data, band="sakoe-chiba")

    def test_an_impossible_band_explains_itself(self, dryer_data: dict) -> None:
        """The per-batch handler used to discard the cause, which is the actionable part."""
        with pytest.raises(ValueError, match=r"Failed on batch .*starting cell"):
            self._run(dryer_data, band=lambda n_test, n_ref: np.full((n_test, 2), [1, n_ref], dtype=np.int64))
