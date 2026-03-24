import pathlib
import unittest

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.models import lm, predict, summary
from process_improve.experiments.structures import c, create_names, expand_grid, gather, supplement


class TestStructures(unittest.TestCase):
    """Test the data structures."""

    def setUp(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0, 0]
        self.C = [4, 5, 6, 4, 6]
        self.D = [0, 1, "green"]
        self.y = [52, 74, 62, 80, 50, 65]

    def test_create_names(self):
        """Create factor names."""

        self.assertListEqual(create_names(5), ["A", "B", "C", "D", "E"])

        self.assertListEqual(create_names(3, letters=False), ["X1", "X2", "X3"])

        self.assertListEqual(
            create_names(3, letters=False, prefix="Q", start_at=9, padded=True),
            ["Q09", "Q10", "Q11"],
        )

        self.assertListEqual(
            create_names(4, letters=False, prefix="Z", start_at=99, padded=True),
            ["Z099", "Z100", "Z101", "Z102"],
        )

    def test_create_factors(self):
        A1 = c(*(self.A))
        A2 = c(*(self.A), index=["lo", "hi", "lo", "hi", "cp", "hi"])
        B = c(*(self.B), name="B")
        C1 = c(*(self.C), range=(4, 6))
        C2 = c(*(self.C), center=5, range=(4, 6))
        C3 = c(*(self.C), lo=4, hi=6)
        C4 = c(*(self.C), lo=4, hi=6, name="C4")
        C5 = c([4, 5, 6, 4, 6], lo=4, hi=6, name="C5")
        C6 = c(*(self.C), lo=5, hi=6, name="C6")
        y = c(*(self.y), name="conversion", units="%")

        self.assertTrue(isinstance(A1, pd.Series))
        self.assertTrue(A1.shape == (6,))
        self.assertTrue(hasattr(A1, "pi_index"))
        self.assertTrue(hasattr(A1, "name"))

        self.assertTrue(A1.name == "Unnamed [coded]")
        self.assertTrue(hasattr(A1, "pi_lo"))
        self.assertTrue(A1.pi_lo == -1)
        self.assertTrue(hasattr(A1, "pi_hi"))
        self.assertTrue(A1.pi_hi == +1)
        self.assertTrue(hasattr(A1, "pi_range"))
        self.assertTrue(A1.pi_range[0] == -1)
        self.assertTrue(A1.pi_range[1] == +1)
        self.assertTrue(hasattr(A1, "pi_center"))
        self.assertTrue(A1.pi_center == 0)

        self.assertTrue(isinstance(A2.index, pd.Index))
        self.assertTrue(hasattr(A2, "pi_index"))
        self.assertTrue(A2.name == "Unnamed [coded]")

        with self.assertRaises(IndexError):
            A2 = c(*(self.A), index=["lo", "hi", "lo", "hi", "cp"])

        self.assertTrue(B.shape == (6,))
        self.assertTrue(B.name == "B [coded]")

        self.assertTrue(C1.pi_range == (4, 6))
        self.assertTrue(C2.pi_center == 5)
        self.assertTrue(C2.pi_range == (4, 6))
        self.assertTrue(C3.pi_lo == 4)
        self.assertTrue(C3.pi_hi == 6)
        self.assertTrue(C4.pi_lo == 4)
        self.assertTrue(C4.pi_hi == 6)
        self.assertTrue(C5.pi_hi == 6)
        self.assertTrue(C5.name == "C5")

        # User says the low is 5, but the minimum is actually different
        self.assertTrue(C6.pi_lo == 5)
        self.assertTrue(C6.pi_range == (5, 6))

        D = c(*(self.D))
        self.assertTrue(D.pi_numeric is True)

        self.assertTrue(len(y) == 6)
        self.assertTrue(y.name == "conversion [%]")

    def test_column_math(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0, 0]
        A = c(*(self.A))
        B = c(*(self.B), name="B")
        C1 = A * B
        C2 = B * A

        self.assertTrue(np.all(C1.values == [+1, -1, -1, +1, 0, 0]))
        self.assertTrue(np.all(C1.index == A.index))
        self.assertTrue(np.all(C2.values == [+1, -1, -1, +1, 0, 0]))
        self.assertTrue(np.all(C2.index == A.index))

    def test_gather(self):
        A = c(*(self.A))
        B = c(*(self.B))
        y = c(*(self.y), name="conversion")

        expt = gather(A=A, B=B, y=y)
        self.assertTrue(expt.shape == (6, 3))

        expt = gather(A=A, B=B, y=y, title="Testing expt name")

        # Eventually this method must go to the "DF" class; currently in the
        # model class; not really appropriate there.
        self.assertTrue(expt.get_title() == "Testing expt name")


class TestModels(unittest.TestCase):
    def setUp(self):
        A = c(-1, +1, -1, +1)
        B = c(-1, -1, +1, +1)
        C = A * B
        y = c(41, 27, 35, 20, name="Stability", units="days")
        self.expt = gather(A=A, B=B, C=C, y=y, title="Half-fraction, using C = A*B")
        self.model_stability_poshalf = lm("y ~ A*B*C", self.expt)

        """
        Results from R:
        A = c(-1, +1, -1, +1)
        B = c(-1, -1, +1, +1)
        C = A*B
        y = c(41, 27, 35, 20)
        data = data.frame(A=A, B=B, C=C, y=y)
        model = lm(y ~ A*B*C, data=data)
        summary(model)

        Call:
            lm(formula = y ~ A * B * C, data = data)

            Residuals:
            ALL 4 residuals are 0: no residual degrees of freedom!

            Coefficients: (4 not defined because of singularities)
                        Estimate Std. Error t value Pr(>|t|)
            (Intercept)    30.75         NA      NA       NA
            A              -7.25         NA      NA       NA
            B              -3.25         NA      NA       NA
            C              -0.25         NA      NA       NA
            A:B               NA         NA      NA       NA
            A:C               NA         NA      NA       NA
            B:C               NA         NA      NA       NA
            A:B:C             NA         NA      NA       NA

            Residual standard error: NaN on 0 degrees of freedom
            Multiple R-squared:      1,	Adjusted R-squared:    NaN
            F-statistic:   NaN on 3 and 0 DF,  p-value: NA
        """

    def test_half_fraction(self):
        """Testing attributes for the half-fraction model."""
        self.assertTrue(self.model_stability_poshalf.nobs == 4)
        self.assertTrue(self.model_stability_poshalf.df_resid == 0)
        beta = self.model_stability_poshalf.get_parameters(drop_intercept=False)

        self.assertAlmostEqual(beta["A"], -7.25)
        self.assertAlmostEqual(beta["B"], -3.25)
        self.assertAlmostEqual(beta["C"], -0.25)
        self.assertAlmostEqual(beta["Intercept"], 30.75)
        self.assertTrue(self.model_stability_poshalf.get_aliases() == ["A + B:C", "B + A:C", "C + A:B"])

        for resid in self.model_stability_poshalf.residuals:
            self.assertAlmostEqual(resid, 0.0)


class Test_API_usage(unittest.TestCase):
    def setUp(self):
        # TODO: replace unittest -> pytest
        # TODO: use path
        folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "experiments"
        self.df1 = pd.read_csv(folder / "test_doe1.csv").set_index("Run order")

    @pytest.mark.skip(reason="Figure out StringArray in pandas")
    def test_case_1(self):
        index = self.df1.index
        C = c(
            self.df1["C"],
            lo=self.df1["C"].min(),
            hi=self.df1["C"].max(),
            index=index,
            name="C",
        )
        M = c(self.df1["M"], levels=self.df1["M"].unique(), name="M")
        V = c(self.df1["V"], lo=self.df1["V"].min(), hi=self.df1["V"].max(), name="V")
        B = c(self.df1["B"], name="B")
        self.assertTrue(B.pi_levels["B"] == ["Ard", "Eme"])

        y = self.df1["y"]

        expt = gather(C=C, M=M, V=V, B=B, y=y)
        self.assertTrue(np.all(C.index == M.index))

        _ = lm("np.log10(y) ~ C*M*B*V", expt)
        # summary(model)
        # pareto_plot(model)
        # contour_plot(model, C, M)
        # contour_plot(model, "C", "M")
        # predict_plot(model)

    def test_aliasing(self):
        d4 = c(24, 48, 36, 36, 60, units="hours", lo=24, high=48)
        y4 = c(31, 65, 52, 54, 69)
        expt4 = gather(d=d4, y=y4, title="RW units")
        model4 = lm("y ~ d + I(np.power(d, 2))", data=expt4)
        self.assertEqual(len(model4.aliasing), 0)

        model5 = lm("y ~ d + I(np.power(d, 2))", data=expt4, alias_threshold=0.99)
        self.assertEqual(len(model5.aliasing), 2)

    def test_realworld_coded(self):
        """Test conversion between real-world and coded units."""
        c1 = c(2.5, 3, 2.5, 3, center=2.75, range=[2.5, 3], name="cement")
        c1_coded = c1.to_coded()
        self.assertListEqual(c1_coded.to_list(), [-1.0, 1.0, -1.0, 1.0])

        c2 = c(
            [-1.0, -1.0, +1.0, +1.0],
            center=2.75,
            range=[2.5, 3],
            name="cement",
            coded=True,
        )
        c2_rw = c2.to_realworld()
        self.assertListEqual(c2_rw.to_list(), [2.5, 2.5, 3.0, 3.0])


# ---- Model tests (improving experiments/models.py coverage) ----


def test_model_summary_output():
    """Model.summary() should return a summary object with tables."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y, title="Summary test")
    model = lm("y ~ A + B", expt)
    smry = model.summary(print_to_screen=False)
    assert smry is not None
    assert len(smry.tables) >= 2


def test_model_summary_with_name():
    """Model.summary() with a model name should include it in the title."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y, title="Named model")
    model = lm("y ~ A + B", expt, name="CustomName")
    smry = model.summary(print_to_screen=False)
    # The summary title should contain the custom name
    assert "CustomName" in str(smry)


def test_model_get_parameters():
    """get_parameters should return coefficients, optionally without intercept."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)

    params_no_intercept = model.get_parameters(drop_intercept=True)
    assert "Intercept" not in params_no_intercept.index

    params_with_intercept = model.get_parameters(drop_intercept=False)
    assert "Intercept" in params_with_intercept.index
    # For y = [52, 74, 62, 80], A = [-1,1,-1,1], B = [-1,-1,1,1]:
    # intercept = mean = 67, A effect = (74+80-52-62)/4 = 10, B effect = (62+80-52-74)/4 = 4
    assert params_with_intercept["A"] == pytest.approx(10.0, abs=1e-6)
    assert params_with_intercept["B"] == pytest.approx(4.0, abs=1e-6)


def test_model_get_factor_names():
    """get_factor_names should return factors at the requested interaction level."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A*B", expt)

    level1 = model.get_factor_names(level=1)
    assert "A" in level1
    assert "B" in level1

    level2 = model.get_factor_names(level=2)
    assert len(level2) == 1  # A:B interaction


def test_model_get_response_name():
    """get_response_name should return the response variable name."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    assert model.get_response_name() == "y"


def test_model_str():
    """str(model) should return the formula description."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    desc = str(model)
    assert "A" in desc
    assert "B" in desc


def test_predict_function():
    """predict() should make predictions from the model."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)

    pred = predict(model, A=[0], B=[0])
    assert pred[0] == pytest.approx(67.0, abs=1e-6)

    pred_hi = predict(model, A=[1], B=[1])
    assert pred_hi[0] == pytest.approx(81.0, abs=1e-6)


def test_summary_function_with_aliasing():
    """The standalone summary() function should include aliasing info."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    C = A * B
    y = c(41, 27, 35, 20, name="Stability", units="days")
    expt = gather(A=A, B=B, C=C, y=y, title="Half-fraction")
    model = lm("y ~ A*B*C", expt)

    smry = summary(model, show=False, aliasing_up_to_level=2)
    smry_str = str(smry)
    assert "Aliasing pattern" in smry_str


def test_model_get_aliases_websafe():
    """get_aliases with websafe=True should return HTML-formatted strings."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    C = A * B
    y = c(41, 27, 35, 20, name="Stability")
    expt = gather(A=A, B=B, C=C, y=y)
    model = lm("y ~ A*B*C", expt)

    aliases = model.get_aliases(websafe=True)
    for alias_str in aliases:
        assert "<span" in alias_str


def test_model_get_aliases_empty():
    """get_aliases should return empty list when there is no aliasing."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    assert model.get_aliases() == []
