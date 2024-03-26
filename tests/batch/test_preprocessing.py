import numpy as np
import pytest

from process_improve.batch.preprocessing import (
    apply_scaling,
    batch_dtw,
    determine_scaling,
    find_reference_batch,
    reverse_scaling,
)


def test_scaling(dryer_data):
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


def test_alignment(dryer_data):
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
    assert [1, 1, 1, 1, 1] == pytest.approx(outputs["weight_history"].iloc[0])
    assert [152.379618, 48.254502, 101.703155, 73.146169, 68.004085] == pytest.approx(
        outputs["scale_df"]["Range"][columns_to_align]
    )
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
    assert [1.0371, 0.1673, 0.9712, 0.6538, 0.2532] == pytest.approx(
        outputs["last_average_batch"].iloc[1, :] / (outputs["scale_df"]["Range"][columns_to_align]),
        abs=1e-4,
    )
    len(outputs["aligned_batch_dfdict"]) == 71
    outputs["aligned_batch_dfdict"].pop(1).shape == (100, 12)

    # Repeat, with a lower tolerance, to ensure the number of iterations exceeds 3.
    outputs = batch_dtw(
        dryer_data,
        columns_to_align=columns_to_align,
        reference_batch=2,
        settings={"robust": False, "tolerance": 0.06, "show_progress": True},
    )
    assert (3, 5) == outputs["weight_history"].shape
    # TODO: still work on this, depending on how you terminate DTW.
    # assert [0.43702525, 1.33206459, 0.98298667, 0.93599197, 1.31193153] == pytest.approx(
    #     outputs["weight_history"][4, :], abs=1e-7
    # )


def test_reference_batch_selection_dryer(dryer_data):
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


def test_reference_batch_selection_nylon(nylon_data):
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
