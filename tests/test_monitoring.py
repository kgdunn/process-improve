import pathlib
import pytest

from pytest import approx
import numpy as np
import pandas as pd

from process_improve.monitoring.control_charts import ControlChart


class test_validate_against_R_qcc_xbar_one:
    """
    Uses testing data (on actual and simulated data) to verify the results agree with other
    another software package, the "qcc" library in R.

    R commands
    ----------
    data <- read.csv('https://openmv.net/file/rubber-colour.csv')
    chart <- qcc(data=data$Colour, type="xbar.one")
    target = chart$center   # 238.78
    s = chart$std.dev       # 10.43234
    """

    folder = (
        pathlib.Path(__file__).parents[1]
        / "process_improve"
        / "datasets"
        / "monitoring"
    )
    cc_values = pd.read_csv(folder / "rubber-colour.csv")
    y = cc_values["Colour"]

    # Do we get similar results to the "chart <- qcc(data=data$Colour, type="xbar.one")" from R?
    cc = ControlChart(variant="xbar.no.subgroup", style="regular")
    cc.calculate_limits(y)
    assert cc.target == approx(238.78, abs=1e-2)
    # cannot get this more precise, since the R code used in QCC follows a different approach:
    # calculates the std dev from a sequence of differences from point to point.
    assert round(cc.s) == 11

    cc = ControlChart(variant="xbar.no.subgroup", style="robust")
    cc.calculate_limits(y)
    assert cc.target == approx(239.5, abs=1e-1)
    # cannot get this more precise, since the R code used in QCC follows a different approach:
    # calculates the std dev from a sequence of differences from point to point.
    assert cc.s == approx(14.0847, abs=1e-3)


class test_control_chart:
    """
    Generic control chart tests.
    """

    y = np.array([33, 30, 29, 30, 27, 32, 17])
    # Simulated in R: loc=50, sd=4
    known_s = np.array(
        [41.5, 43.0, 51.8, 50.4, 45.6, 49.5, 45.3, 49.0, 41.4, 51.8, 58.4]
    )

    cc = ControlChart(
        variant="xbar.no.subgroup",
        style="robust",
    )
    cc.calculate_limits(y)
    assert cc.target == np.median(y[0:-1])
    assert np.all(y[cc.idx_outside_3S] == [17])

    # test_with_known_s(self):
    cc = ControlChart(variant="hw")
    cc.calculate_limits(known_s, target=50, s=4)
    assert cc.target == 50
    assert cc.s == 4


class test_holt_winters_control_chart:
    """
    Testing the Holt-Winters control chart settings.
    """

    y = np.array([10.09, 9.08, 3.14, 7.00, 11.47, 11.95, 3.96, 8.18, 1.87, 8.72])

    medium_length = np.array(
        [
            98.235,
            93.380,
            98.345,
            92.535,
            93.635,
            92.595,
            98.715,
            98.715,
            103.365,
            93.945,
            91.185,
        ]
    )

    short_length = np.array([86.115, 91.615])
    with_missing = np.array(
        [
            101.613,
            94.355,
            100.806,
            100.269,
            100.269,
            102.419,
            106.452,
            102.688,
            111.828,
            100.538,
            147.043,
            104.301,
            101.613,
            98.118,
            95.161,
            np.NaN,
            95.968,
            96.505,
            101.075,
            96.505,
            99.731,
            94.892,
            98.118,
            100.000,
            99.462,
        ]
    )

    # TODO: an example with NaN right at the start
    # TODO: an example with NaN in the first few samples (warm up)

    # def test_asserts:
    # Checks that the required asserts are raised

    cc = ControlChart()
    with pytest.raises(
        AssertionError, match="Lambda_1 must be less than or equal to 1.0."
    ):
        cc.calculate_limits(y, ld_1=1.2, ld_2=0.5)

    with pytest.raises(
        AssertionError, match="Lambda_1 must be greater than or equal to zero."
    ):
        cc.calculate_limits(y, ld_1=-1, ld_2=0.5)

    # test_hw_chart_missing_values(self):
    cc = ControlChart()
    cc.calculate_limits(with_missing, ld_1=0.4, ld_2=0.7)
    assert cc.target == approx(100.3, abs=1e-1)
    assert cc.s == approx(6, abs=1)

    # test_hw_short_length:
    # Ensures that short length sequences are also handled well.

    cc = ControlChart()
    cc.calculate_limits(short_length, ld_1=0.2, ld_2=0.5)
    assert cc.target == approx((86.115 + 91.615) / 2.0, abs=1e-3)
    assert cc.s == approx(np.std([86.115, 91.615], ddof=1))

    # test_hw_medium_length
    # Ensures that short length sequences are also handled well.
    cc = ControlChart()
    cc.calculate_limits(medium_length, ld_1=0.2, ld_2=0.5)
    assert cc.target == approx(93.945, abs=1e-3)
    assert cc.s == approx(4.493, abs=1e-3)
    assert len(cc.idx_outside_3S) == 0


class test_holt_winters_control_chart_BatchYield:
    """
    Testing the Holt-Winters control chart settings on a different dataset: batch yields
    http://openmv.net/info/batch-yield-and-purity (Kevin Dunn, personal data)
    """

    folder = (
        pathlib.Path(__file__).parents[1]
        / "process_improve"
        / "datasets"
        / "monitoring"
    )
    cc_values = pd.read_csv(folder / "batch-yield-and-purity.csv")
    subgroupsN = 3
    rounder = int(np.floor(cc_values.shape[0] / 3))
    subgroups = (
        cc_values["yield"].values[0 : (rounder * subgroupsN)].reshape(rounder, 3)
    )
    y = subgroups.mean(axis=1)

    cc = ControlChart()
    cc.calculate_limits(y)  # ld_1=0.5, ld_2=0.5
    assert cc.target == approx(75.1, abs=1e-1)
    assert cc.s == approx(4.16, abs=1e-3)
