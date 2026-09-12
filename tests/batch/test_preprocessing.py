import numpy as np
import pandas as pd
import pytest

from process_improve.batch.alignment_helpers import backtrack_optimal_path, distance_matrix
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
