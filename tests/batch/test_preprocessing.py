import numpy as np
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
    # TODO: still work on this, depending on how you terminate DTW.
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
