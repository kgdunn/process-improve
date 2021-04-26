import pathlib
import unittest
import numpy as np
import pandas as pd


from process_improve.experiments.structures import gather, create_names, c
from process_improve.experiments.models import lm


class TestStructures(unittest.TestCase):
    """
    Test the data structures.
    """

    def setUp(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0, 0]
        self.C = [4, 5, 6, 4, 6]
        self.D = [0, 1, "green"]
        self.y = [52, 74, 62, 80, 50, 65]

    def test_create_names(self):
        """Creating factor names." """

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
        """
        Testing attributes for the half-fraction model.
        """
        self.assertTrue(self.model_stability_poshalf.nobs == 4)
        self.assertTrue(self.model_stability_poshalf.df_resid == 0)
        beta = self.model_stability_poshalf.get_parameters(drop_intercept=False)

        self.assertAlmostEqual(beta["A"], -7.25)
        self.assertAlmostEqual(beta["B"], -3.25)
        self.assertAlmostEqual(beta["C"], -0.25)
        self.assertAlmostEqual(beta["Intercept"], 30.75)
        self.assertTrue(
            self.model_stability_poshalf.get_aliases()
            == ["A + B:C", "B + A:C", "C + A:B"]
        )

        for resid in self.model_stability_poshalf.residuals:
            self.assertAlmostEqual(resid, 0.0)


class Test_API_usage(unittest.TestCase):
    def setUp(self):
        # TODO: replace unittest -> pytest
        # TODO: use path
        folder = (
            pathlib.Path(__file__).parents[1]
            / "process_improve"
            / "datasets"
            / "experiments"
        )
        self.df1 = pd.read_csv(folder / "test_doe1.csv")
        self.df1.set_index("Run order", inplace=True)

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
        """
        Test conversion between real-world and coded units.
        """
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


if __name__ == "__main__":
    unittest.main()
