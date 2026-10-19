"""The components of interactions of three-level factors, with Yates's names (#803 H2).

Davies (Example 9-4) splits the three-factor interaction of the cotton dyestuff 3^3, run twice in
blocks of nine, into Yates's W, X, Y and Z: X = 6191 and Z = 1002 from both replicates, W = 2939
from the second replicate and Y = 664 from the first, each from the replicate in which it is not
confounded with the blocks. Montgomery (Example 9.1) names the syrup loss components the same way.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.experiments.three_level_components import interaction_components


def catalogue(source: str, name: str) -> pd.DataFrame:
    [entry] = [e for e in reference_experiments(source) if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def dyestuff() -> pd.DataFrame:
    runs = catalogue("davies-1956", "davies-cotton-dyestuff-3-3")
    return runs.assign(replicate=runs["Group"].str[-1])


def components(table: pd.DataFrame) -> dict[str, float]:
    return dict(zip(table["yates"], table["sum_of_squares"], strict=True))


# --------------------------------------------------------------------------- the sources
@pytest.mark.dataset
def test_davies_three_factor_components_are_yates_w_x_y_z(dyestuff):
    both = components(interaction_components(dyestuff, "Yield", ["C", "V", "N"]))
    first = components(interaction_components(dyestuff[dyestuff["replicate"] == "1"], "Yield", ["C", "V", "N"]))
    second = components(interaction_components(dyestuff[dyestuff["replicate"] == "2"], "Yield", ["C", "V", "N"]))
    # X and Z from both replicates; the book's are 0.7 above, its own rounding.
    assert both["X(CVN)"] == pytest.approx(6191, abs=1)
    assert both["Z(CVN)"] == pytest.approx(1002, abs=1)
    # W and Y from the replicate in which each is not confounded with the blocks.
    assert round(second["W(CVN)"]) == 2939
    assert round(first["Y(CVN)"]) == 664
    unconfounded = both["X(CVN)"] + both["Z(CVN)"] + second["W(CVN)"] + first["Y(CVN)"]
    assert unconfounded == pytest.approx(10796, abs=2)


@pytest.mark.dataset
def test_the_confounded_component_is_constant_within_each_block(dyestuff):
    """W is confounded with the blocks of the first replicate and Y with those of the second."""
    levels = dyestuff[["C", "V", "N"]].to_numpy() + 1
    for word, replicate in (((1, 2, 2), "1"), ((1, 1, 2), "2")):
        runs = dyestuff["replicate"] == replicate
        contrast = pd.Series(levels[runs] @ np.array(word) % 3)
        assert contrast.groupby(dyestuff.loc[runs, "Group"].to_numpy()).nunique().max() == 1


@pytest.mark.dataset
def test_montgomery_s_syrup_loss_components():
    """W = 3804.11 and X = 221.78; the book's Y and Z are each other's, as the catalogue records."""
    runs = catalogue("montgomery-2013", "montgomery-syrup-loss")
    table = interaction_components(runs, "y", ["nozzle", "speed", "pressure"])
    found = components(table)
    assert round(found["W(nozzle*speed*pressure)"], 2) == 3804.11
    assert round(found["X(nozzle*speed*pressure)"], 3) == 221.778
    assert round(found["Y(nozzle*speed*pressure)"], 3) == 584.111  # printed 18.77
    assert round(found["Z(nozzle*speed*pressure)"], 3) == 18.778  # printed 584.11
    assert table.loc[table["yates"] == "W(nozzle*speed*pressure)", "component"].item() == "nozzle*speed^2*pressure^2"


# --------------------------------------------------------------------------- what the components are
@pytest.mark.dataset
def test_an_interaction_s_components_add_up_to_its_sum_of_squares(dyestuff):
    """In the full factorial, run twice, the components split the ANOVA's sums of squares exactly."""
    runs = dyestuff.rename(columns={"C": "c", "V": "v", "N": "n"})  # patsy's C() would read a column named C
    table = interaction_components(runs, "Yield", ["c", "v", "n"])
    anova = anova_lm(smf.ols("Yield ~ C(c) * C(v) * C(n)", data=runs).fit(), typ=1)
    for effect, row in (("c", "C(c)"), ("vn", "C(v):C(n)"), ("cvn", "C(c):C(v):C(n)")):
        assert table.loc[table["effect"] == effect, "sum_of_squares"].sum() == pytest.approx(anova.loc[row, "sum_sq"])
    assert table["sum_of_squares"].sum() == pytest.approx(anova["sum_sq"].drop("Residual").sum())


def test_yates_names_follow_the_exponents():
    runs = pd.DataFrame(list(itertools.product(range(3), repeat=3)), columns=list("ABC"))
    runs["y"] = np.random.default_rng(0).normal(size=len(runs))
    table = interaction_components(runs, "y", list("ABC")).set_index("component")
    assert table["yates"].to_dict() == {
        "A": "", "B": "", "C": "",
        "AB": "J(AB)", "AB2": "I(AB)", "AC": "J(AC)", "AC2": "I(AC)", "BC": "J(BC)", "BC2": "I(BC)",
        "ABC": "Z(ABC)", "ABC2": "Y(ABC)", "AB2C": "X(ABC)", "AB2C2": "W(ABC)",
    }  # fmt: skip
    assert set(table["df"]) == {2}


def test_four_factors_have_eight_components_of_their_interaction_and_no_yates_names():
    runs = pd.DataFrame(list(itertools.product(range(3), repeat=4)), columns=list("ABCD"))
    runs["y"] = np.random.default_rng(1).normal(size=len(runs))
    table = interaction_components(runs, "y", list("ABCD"))
    assert len(table) == 4 + 6 * 2 + 4 * 4 + 8
    assert set(table.loc[table["effect"] == "ABCD", "yates"]) == {""}
    # One run at each of the 81 settings: the 40 components hold every one of its 80 degrees of freedom.
    total = ((runs["y"] - runs["y"].mean()) ** 2).sum()
    assert table["sum_of_squares"].sum() == pytest.approx(total)


def test_reversing_a_factor_s_levels_exchanges_the_i_and_j_components():
    runs = pd.DataFrame(list(itertools.product(["lo", "mid", "hi"], repeat=2)), columns=["A", "B"])
    runs["y"] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0]
    order = pd.CategoricalDtype(["lo", "mid", "hi"], ordered=True)
    forward = components(interaction_components(runs.astype({"A": order, "B": order}), "y", ["A", "B"]))
    backward_order = pd.CategoricalDtype(["hi", "mid", "lo"], ordered=True)
    backward = components(interaction_components(runs.astype({"A": order, "B": backward_order}), "y", ["A", "B"]))
    assert (backward["I(AB)"], backward["J(AB)"]) == pytest.approx((forward["J(AB)"], forward["I(AB)"]))


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("data", "factors", "message"),
    [
        (pd.DataFrame({"A": [0, 1, 2], "y": [1.0, 2.0, 3.0]}), ["B"], r"missing \['B'\]"),
        (pd.DataFrame({"A": [0, 1, 2], "y": [1.0, 2.0, 3.0]}), ["A", "A"], "distinct factors"),
        (pd.DataFrame({"A": [0, 1, 0], "y": [1.0, 2.0, 3.0]}), ["A"], "exactly three settings"),
        (pd.DataFrame({"A": [0, 1, 2], "y": [1.0, np.nan, 3.0]}), ["A"], "finite number in every run"),
    ],
)
def test_data_that_cannot_be_split_are_refused(data, factors, message):
    with pytest.raises(ValueError, match=message):
        interaction_components(data, "y", factors)
