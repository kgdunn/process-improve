import numpy as np
from pytest import approx

from process_improve import robust


def test_robust_robust_scale():
    """
    A scale estimator which is robust to outliers

    Testing against R code [R version 3.6.0 (2019-04-26)]
    > library(robustbase)
    # All code has the default argument: "finite.corr=TRUE"
    > robustbase::Sn(c(0, 1))                   # 0.8861018: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2))                # 2.207503
    > robustbase::Sn(c(0, 1, 2, 3))             # 1.13774: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2, 3, 4))          # 1.611203
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5))       # 2.368504: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5, 6))    # 2.85747
    > robustbase::Sn(c(0, 1, 2, 3, 4, 50, 6))   # 2.85747
    > robustbase::Sn(c(0, 1, 20, 3, 4, 50, 6))  # 5.714939: FAILS in our implementation
    > robustbase::Sn(c(0, 10, 20, 3, 4, 50, 6)) # 8.572409
    > robustbase::Sn(seq(1, 10))                # 3.5778: FAILS in our implementation
    > robustbase::Sn(seq(1, 11))                # 3.896614
    > robustbase::Sn(seq(1, 19))                # 6.259503
    > robustbase::Sn(seq(1, 1500))              # 447.225

    TODO: found this weird sequence that gives Sn of zero, even though there is variability:
    99, 95, 95, 100, 100, 100, 100, 95, 100, 100, 100, 100, 105, 105, 100, 95, 105, 100, 95, 100
    How to make it robust to this weird situation?
    """

    # Tests with an even number of samples, and small sample sizes do not agree with R.
    # This is because the R implementation aims for efficiency of calculation, and does not
    # follow the formula presented in the original paper.
    # Since we aim to be using this on medium/larger data sets, it should not matter.

    assert robust.Sn(list(range(3))) == approx(2.207503, rel=1e-6)
    assert robust.Sn(list(range(5))) == approx(1.611203, rel=1e-6)
    assert robust.Sn(list(range(7))) == approx(2.85747, rel=1e-6)
    assert robust.Sn([0, 10, 20, 3, 4, 50, 6]) == approx(8.572409, rel=1e-7)
    assert robust.Sn(list(range(1, 12))) == approx(3.896614, rel=1e-6)
    assert robust.Sn(list(range(1, 19))) == approx(5.3667, rel=1e-6)
    assert robust.Sn(list(range(1, 20))) == approx(6.259503, rel=1e-6)
    # Corner cases:
    assert np.isnan(robust.Sn([]))
    assert robust.Sn([13]) == 0.0


def test_summary_stats_corner_case_with_robust_scale():
    x = [
        99,
        95,
        95,
        100,
        100,
        100,
        100,
        95,
        100,
        100,
        100,
        100,
        105,
        105,
        100,
        95,
        105,
        100,
        95,
        100,
    ]
    out = robust.summary_stats(np.array(x), method="robust")
    assert out["center"] == np.mean(x)
    assert out["center"] != np.median(x)

    out = robust.summary_stats(np.array(x), method="something-else")
    assert out["center"] == np.mean(x)
    assert out["center"] != np.median(x)
