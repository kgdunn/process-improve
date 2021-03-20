import numpy as np
import matplotlib.pylab as plt
from pytest import approx

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
    res = dtw(query, y=reference, window_type="sakoechiba", window_size=int(0.2 * len(x)))
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

    #
    #

    columns_to_align = [
        "AgitatorPower",
        "AgitatorTorque",
        "JacketTemperatureSP",
        "JacketTemperature",
        "DryerTemp",
    ]
    scale_df = determine_scaling(
        dryer_data, columns_to_align=columns_to_align, settings={"robust": False},
    )
    assert np.array([152.3796, 48.2545, 101.7032, 73.1462, 68.0041]) == approx(
        scale_df.loc[columns_to_align]["Range"]
    )

    batches_scaled = apply_scaling(dryer_data, scale_df, columns_to_align=columns_to_align)
    reference_batch = batches_scaled["1"]
    assert np.array([0.8354, 0.1660, 0.9833, 0.2556, 0.2884]) == approx(
        reference_batch[columns_to_align].iloc[0], abs=1e-4
    )
    orig = reverse_scaling(batches_scaled, scale_df)
    assert np.linalg.norm(orig["1"] - dryer_data["1"][columns_to_align]) == approx(0, abs=1e-10)


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
        settings={"robust": False},
    )
