import pathlib
import unittest

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.designs_factorial import full_factorial
from process_improve.experiments.models import lm, predict, summary
from process_improve.experiments.optimal import optimization_function, point_exchange
from process_improve.experiments.structures import c, create_names, expand_grid, gather, supplement


class TestStructures(unittest.TestCase):
    """Test the data structures."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0, 0]
        self.C = [4, 5, 6, 4, 6]
        self.D = [0, 1, "green"]
        self.y = [52, 74, 62, 80, 50, 65]

    def test_create_names(self) -> None:
        """Create factor names."""
        assert create_names(5) == ["A", "B", "C", "D", "E"]

        assert create_names(3, letters=False) == ["X1", "X2", "X3"]

        assert create_names(3, letters=False, prefix="Q", start_at=9, padded=True) == ["Q09", "Q10", "Q11"]

        assert create_names(4, letters=False, prefix="Z", start_at=99, padded=True) == [
            "Z099",
            "Z100",
            "Z101",
            "Z102",
        ]

    def test_create_factors(self) -> None:
        """Test creating factors with various options."""
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

        assert isinstance(A1, pd.Series)
        assert A1.shape == (6,)
        assert hasattr(A1, "pi_index")
        assert hasattr(A1, "name")

        assert A1.name == "Unnamed [coded]"
        assert hasattr(A1, "pi_lo")
        assert A1.pi_lo == -1
        assert hasattr(A1, "pi_hi")
        assert A1.pi_hi == +1
        assert hasattr(A1, "pi_range")
        assert A1.pi_range[0] == -1
        assert A1.pi_range[1] == +1
        assert hasattr(A1, "pi_center")
        assert A1.pi_center == 0

        assert isinstance(A2.index, pd.Index)
        assert hasattr(A2, "pi_index")
        assert A2.name == "Unnamed [coded]"

        with pytest.raises(IndexError):
            A2 = c(*(self.A), index=["lo", "hi", "lo", "hi", "cp"])

        assert B.shape == (6,)
        assert B.name == "B [coded]"

        assert C1.pi_range == (4, 6)
        assert C2.pi_center == 5
        assert C2.pi_range == (4, 6)
        assert C3.pi_lo == 4
        assert C3.pi_hi == 6
        assert C4.pi_lo == 4
        assert C4.pi_hi == 6
        assert C5.pi_hi == 6
        assert C5.name == "C5"

        # User says the low is 5, but the minimum is actually different
        assert C6.pi_lo == 5
        assert C6.pi_range == (5, 6)

        D = c(*(self.D))
        assert D.pi_numeric is True

        assert len(y) == 6
        assert y.name == "conversion [%]"

    def test_column_math(self) -> None:
        """Test column multiplication."""
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0, 0]
        A = c(*(self.A))
        B = c(*(self.B), name="B")
        C1 = A * B
        C2 = B * A

        assert np.all(C1.values == [+1, -1, -1, +1, 0, 0])
        assert np.all(C1.index == A.index)
        assert np.all(C2.values == [+1, -1, -1, +1, 0, 0])
        assert np.all(C2.index == A.index)

    def test_gather(self) -> None:
        """Test gathering factors into an experiment."""
        A = c(*(self.A))
        B = c(*(self.B))
        y = c(*(self.y), name="conversion")

        expt = gather(A=A, B=B, y=y)
        assert expt.shape == (6, 3)

        expt = gather(A=A, B=B, y=y, title="Testing expt name")

        # Eventually this method must go to the "DF" class; currently in the
        # model class; not really appropriate there.
        assert expt.get_title() == "Testing expt name"


class TestModels(unittest.TestCase):
    """Test model fitting and results."""

    def setUp(self) -> None:
        """Set up test fixtures for model tests."""
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

    def test_half_fraction(self) -> None:
        """Testing attributes for the half-fraction model."""
        assert self.model_stability_poshalf.nobs == 4
        assert self.model_stability_poshalf.df_resid == 0
        beta = self.model_stability_poshalf.get_parameters(drop_intercept=False)

        assert beta["A"] == pytest.approx(-7.25)
        assert beta["B"] == pytest.approx(-3.25)
        assert beta["C"] == pytest.approx(-0.25)
        assert beta["Intercept"] == pytest.approx(30.75)
        assert self.model_stability_poshalf.get_aliases() == ["A + B:C", "B + A:C", "C + A:B"]

        for resid in self.model_stability_poshalf.residuals:
            assert resid == pytest.approx(0.0)


class TestAPIUsage(unittest.TestCase):
    """Test API usage patterns."""

    def setUp(self) -> None:
        """Set up test fixtures for API usage tests."""
        # TODO: replace unittest -> pytest
        # TODO: use path
        folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "experiments"
        self.df1 = pd.read_csv(folder / "test_doe1.csv").set_index("Run order")

    @pytest.mark.skip(reason="Figure out StringArray in pandas")
    def test_case_1(self) -> None:
        """Test case 1 with real-world data."""
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
        assert B.pi_levels["B"] == ["Ard", "Eme"]

        y = self.df1["y"]

        expt = gather(C=C, M=M, V=V, B=B, y=y)
        assert np.all(C.index == M.index)

        _ = lm("np.log10(y) ~ C*M*B*V", expt)
        # summary(model)
        # pareto_plot(model)
        # contour_plot(model, C, M)
        # contour_plot(model, "C", "M")
        # predict_plot(model)

    def test_aliasing(self) -> None:
        """Test aliasing detection in models."""
        d4 = c(24, 48, 36, 36, 60, units="hours", lo=24, high=48)
        y4 = c(31, 65, 52, 54, 69)
        expt4 = gather(d=d4, y=y4, title="RW units")
        model4 = lm("y ~ d + I(np.power(d, 2))", data=expt4)
        assert len(model4.aliasing) == 0

        model5 = lm("y ~ d + I(np.power(d, 2))", data=expt4, alias_threshold=0.99)
        assert len(model5.aliasing) == 2

    def test_realworld_coded(self) -> None:
        """Test conversion between real-world and coded units."""
        c1 = c(2.5, 3, 2.5, 3, center=2.75, range=[2.5, 3], name="cement")
        c1_coded = c1.to_coded()
        assert c1_coded.to_list() == [-1.0, 1.0, -1.0, 1.0]

        c2 = c(
            [-1.0, -1.0, +1.0, +1.0],
            center=2.75,
            range=[2.5, 3],
            name="cement",
            coded=True,
        )
        c2_rw = c2.to_realworld()
        assert c2_rw.to_list() == [2.5, 2.5, 3.0, 3.0]


# ---- Model tests (improving experiments/models.py coverage) ----


def test_model_summary_output() -> None:
    """Model.summary() should return a summary object with tables."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y, title="Summary test")
    model = lm("y ~ A + B", expt)
    smry = model.summary(print_to_screen=False)
    assert smry is not None
    assert len(smry.tables) >= 2


def test_model_summary_with_name() -> None:
    """Model.summary() with a model name should include it in the title."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y, title="Named model")
    model = lm("y ~ A + B", expt, name="CustomName")
    smry = model.summary(print_to_screen=False)
    # The summary title should contain the custom name
    assert "CustomName" in str(smry)


def test_model_get_parameters() -> None:
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


def test_model_get_factor_names() -> None:
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


def test_model_get_response_name() -> None:
    """get_response_name should return the response variable name."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    assert model.get_response_name() == "y"


def test_model_str() -> None:
    """str(model) should return the formula description."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    desc = str(model)
    assert "A" in desc
    assert "B" in desc


def test_predict_function() -> None:
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


def test_summary_function_with_aliasing() -> None:
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


def test_model_get_aliases_websafe() -> None:
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


def test_model_get_aliases_empty() -> None:
    """get_aliases should return empty list when there is no aliasing."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80, name="yield")
    expt = gather(A=A, B=B, y=y)
    model = lm("y ~ A + B", expt)
    assert model.get_aliases() == []


# ---- Structure tests (improving experiments/structures.py coverage) ----


def test_expand_grid_basic() -> None:
    """expand_grid should create all combinations of factor levels."""
    A = c(-1, +1)
    B = c(-1, +1)
    result = expand_grid(A=A, B=B)
    assert len(result) == 2  # 2 columns
    assert len(result[0]) == 4  # 2^2 = 4 rows


def test_expand_grid_three_factors() -> None:
    """expand_grid with 3 factors should produce 2^3 = 8 rows."""
    A = c(-1, +1)
    B = c(-1, +1)
    C_factor = c(-1, +1)
    result = expand_grid(A=A, B=B, C=C_factor)
    assert len(result) == 3
    assert len(result[0]) == 8


def test_supplement_function() -> None:
    """Supplement should carry over kwargs to a new Column from existing values."""
    A = c(-1, +1, -1, +1)
    A_supp = supplement(A, name="Feed rate", units="g/min", lo=-1, hi=1)
    assert A_supp.pi_name == "Feed rate"
    assert A_supp.pi_units == "g/min"
    assert len(A_supp) == 4


def test_full_factorial_default_names() -> None:
    """full_factorial should create a 2^k design with default factor names."""
    result = full_factorial(3)
    assert len(result) == 3  # 3 factors
    assert len(result[0]) == 8  # 2^3 = 8 runs


def test_full_factorial_custom_names() -> None:
    """full_factorial with custom names should use provided names."""
    result = full_factorial(2, names=["Temp", "Pressure"])
    assert len(result) == 2
    assert result[0].pi_name == "Temp"
    assert result[1].pi_name == "Pressure"
    assert len(result[0]) == 4  # 2^2 = 4 runs


def test_column_division() -> None:
    """Column division should work element-wise."""
    A = c(2.0, 4.0, 6.0)
    B = c(1.0, 2.0, 3.0)
    result = A / B
    assert np.allclose(result.values, [2.0, 2.0, 2.0])


def test_column_addition() -> None:
    """Column addition should work element-wise."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    result = A + B
    assert np.allclose(result.values, [-2, 0, 0, 2])


def test_column_to_coded_already_coded() -> None:
    """to_coded on already-coded column should return the same values."""
    A = c(-1, +1, 0, name="A")
    coded = A.to_coded()
    assert np.allclose(coded.values, A.values)


def test_column_to_realworld_not_coded() -> None:
    """to_realworld on a real-world column should return the same values."""
    A = c(100, 200, 150, lo=100, hi=200, name="Temp", units="C")
    rw = A.to_realworld()
    assert np.allclose(rw.values, A.values)


def test_column_with_units_name() -> None:
    """Column with units should format name correctly."""
    A = c(100, 200, lo=100, hi=200, name="Temp", units="C")
    assert "C" in A.name
    assert "Temp" in A.name


def test_column_categorical_with_levels() -> None:
    """Column with explicit levels should store them."""
    D = c(0, 1, 0, 1, levels=(0, 1))
    assert hasattr(D, "pi_levels")


def test_gather_drops_missing_values() -> None:
    """Gather should drop rows with any NaN values."""
    A = c(-1, +1, -1, +1, float("nan"))
    B = c(-1, -1, +1, +1, 0)
    expt = gather(A=A, B=B)
    assert expt.shape[0] == 4  # NaN row dropped


def test_expt_repr() -> None:
    """Expt repr should include title and dimensions."""
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    expt = gather(A=A, B=B, title="My experiment")
    r = repr(expt)
    assert "My experiment" in r
    assert "4 experiments" in r


# ---- Optimal design tests (experiments/optimal.py) ----


def test_optimization_function_basic() -> None:
    """optimization_function should return log determinant of (X'X)^-1."""
    X = pd.DataFrame([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    result = optimization_function(X)
    assert np.isfinite(result)


def test_optimization_function_singular() -> None:
    """optimization_function should return inf for singular designs."""
    X = pd.DataFrame([[1, 1], [1, 1], [1, 1]])
    result = optimization_function(X)
    assert result == float(np.inf)


def test_point_exchange_simple() -> None:
    """point_exchange should select a near-optimal subset of candidate points."""
    # Create a full factorial as the candidate set
    rng = np.random.default_rng(42)
    candidates = pd.DataFrame(rng.choice([-1, 0, 1], size=(20, 2)), columns=["A", "B"])
    candidates = pd.concat([candidates, pd.DataFrame([[-1, -1], [1, -1], [-1, 1], [1, 1]], columns=["A", "B"])])
    design, d_opt = point_exchange(candidates, number_points=4)
    assert design.shape[0] == 4
    assert design.shape[1] == 2
    assert np.isfinite(d_opt)
