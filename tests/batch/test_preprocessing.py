import numpy as np
import matplotlib.pylab as plt
from pytest import approx
from dtwalign import dtw

from process_improve.batch.preprocessing import (
    batch_dtw,
    determine_scaling,
    apply_scaling,
    reverse_scaling,
)


def dtw_understanding():
    x = np.linspace(0, 2 * np.pi, num=50)
    reference = np.reshape(np.sin(x), (50, 1))
    query = np.reshape(np.cos(x), (50, 1))

    # Template is shifted forward in time.

    plt.plot(x, reference, ".-", c="blue")
    plt.plot(x, query, ".-", c="red")
    plt.grid()
    res = dtw(
        query, y=reference, window_type="sakoechiba", window_size=int(0.2 * len(x))
    )
    warping_path = res.get_warping_path(target="query")
    plt.plot(x, query[warping_path], ".-", c="purple")
    plt.title("The query signal (red), aligned (purple) with the reference (blue)")
    # plt.savefig("demo-of-DTW.png")
    print(res.distance)

    # Try other distance estimate:
    # ref_steps = 50
    # qry_steps = 50
    # covariance = np.zeros((ref_steps, qry_steps))
    #
    # for j in range(qry_steps):
    #     B = np.ones((ref_steps, 1)) * query[j]
    #     covariance[:, j] = np.diag(np.matmul((B - reference), (B - reference).T))
    #
    # from scipy.spatial.distance import mahalanobis
    #
    # def distance_metric(x, y):
    #     return mahalanobis(x, y, covariance)
    #
    # plt.plot(x, reference, ".-", c="blue")
    # plt.plot(x, query, ".-", c="red")
    # plt.grid()
    # res = dtw(query, y=reference, dist=lambda x, y: distance_metric(x, y))
    # warping_path = res.get_warping_path(target="query")
    # plt.plot(x, query[warping_path], ".-", c="purple")
    # plt.title("The query signal (red), aligned (purple) with the reference (blue)")
    # plt.savefig("demo-of-DTW.png")
    # print(res.distance)

    # Another case: different lengths
    plt.clf()
    x_ref = np.linspace(0, 2 * np.pi, num=50)
    reference = np.sin(x_ref)
    x_query = np.linspace(0, 2 * np.pi, num=25)
    query = np.sin(x_query) + np.random.randn(25) * 0.1
    plt.plot(x_ref, reference, ".-", c="blue")
    plt.plot(x_query, query, ".-", c="red")
    plt.grid()
    res = dtw(query, y=reference)
    warping_path = res.get_warping_path(target="query")
    plt.plot(x_ref, query[warping_path], ".-", c="purple")
    plt.title("The query signal (red), aligned (purple) with the reference (blue)")
    # plt.savefig("demo-of-DTW-varying-lengths.png")
    print(res.distance)

    plt.clf()
    plt.plot(x_ref, x_query[warping_path])
    # plt.savefig("demo-of-DTW-x-axis-distortion.png")

    plt.clf()
    plt.plot(reference, query[warping_path], ".")
    plt.xlabel("Reference")
    plt.ylabel("Query (corrected)")
    plt.grid()
    # plt.savefig("demo-of-DTW-corrected-signal.png")

    # Try


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
    assert np.array([152.3796, 48.2545, 101.7032, 73.1462, 68.0041]) == approx(
        scale_df.loc[columns_to_align]["Range"]
    )

    batches_scaled = apply_scaling(
        dryer_data, scale_df, columns_to_align=columns_to_align
    )
    reference_batch = batches_scaled["1"]
    assert np.array([0.8354, 0.1660, 0.9833, 0.2556, 0.2884]) == approx(
        reference_batch[columns_to_align].iloc[0], abs=1e-4
    )
    orig = reverse_scaling(batches_scaled, scale_df)
    assert np.linalg.norm(orig["1"] - dryer_data["1"][columns_to_align]) == approx(
        0, abs=1e-10
    )


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
        reference_batch="2",
        settings={
            "robust": False,
            "tolerance": 1,
        },  # high tolerance ensures only 1 iteration
    )
    assert [1, 1, 1, 1, 1] == approx(outputs["weight_history"][0])
    assert [152.379618, 48.254502, 101.703155, 73.146169, 68.004085] == approx(
        outputs["scale_df"]["Range"][columns_to_align]
    )
    b1 = outputs["aligned_batch_objects"]["1"]
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
    assert expected_warping_path == approx(b1.warping_path + 1)
    assert [1.0371, 0.1673, 0.9712, 0.6538, 0.2532] == approx(
        outputs["last_average_batch"].iloc[1, :]
        / (outputs["scale_df"]["Range"][columns_to_align]),
        abs=1e-4,
    )
    outputs["aligned_wide_df"].shape[0] == 71
    outputs["aligned_wide_df"].shape[1] == 11 * 129

    # Repeat, with a lower tolerance, to ensure the number of iterations exceeds 3.
    outputs = batch_dtw(
        dryer_data,
        columns_to_align=columns_to_align,
        reference_batch="2",
        settings={"robust": False, "tolerance": 0.01},
    )
    assert (5, 5) == outputs["weight_history"].shape
    assert [0.43702525, 1.33206459, 0.98298667, 0.93599197, 1.31193153] == approx(
        outputs["weight_history"][4, :], abs=1e-7
    )
