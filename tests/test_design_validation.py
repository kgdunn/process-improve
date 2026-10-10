"""Inputs that ``generate_design`` refuses rather than turning into a wrong design."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import generate_design
from process_improve.experiments.designs_utils import _numeric_codes, categorical_labels
from process_improve.experiments.factor import Factor


def test_duplicate_factor_names_raise() -> None:
    """Two factors named 'A' collapsed into one column of a two-factor design."""
    factors = [Factor(name="A", low=0, high=10), Factor(name="A", low=5, high=9)]
    with pytest.raises(ValueError, match="unique"):
        generate_design(factors, "full_factorial")


@pytest.mark.parametrize(
    "design_type", ["full_factorial", "ccd", "box_behnken", "plackett_burman", "dsd", "fractional_factorial", "omars"]
)
def test_mixture_components_in_a_box_family_raise(design_type: str) -> None:
    """These families returned coded +/-1 rows labelled as proportions (row sums from -3 to 3)."""
    factors = [Factor(name=f"x{i}", type="mixture", low=0.1, high=0.6) for i in range(3)]
    with pytest.raises(ValueError, match="sum to 1"):
        generate_design(factors, design_type)


@pytest.mark.parametrize("design_type", ["mixture", "d_optimal", "maximin"])
def test_mixture_components_in_a_mixture_family_sum_to_one(design_type: str) -> None:
    factors = [Factor(name=f"x{i}", type="mixture", low=0.1, high=0.6) for i in range(3)]
    actual = generate_design(factors, design_type, budget=10).design_actual[["x0", "x1", "x2"]]
    assert actual.sum(axis=1).round(9).eq(1).all()


def _factors(n: int) -> list[Factor]:
    return [Factor(name=chr(65 + i), low=0, high=10) for i in range(n)]


@pytest.mark.parametrize("name", ["RunOrder", "Block"])
def test_reserved_factor_names_raise(name: str) -> None:
    """A factor named 'Block' was overwritten by the block labels; 'RunOrder' failed inside pandas."""
    factors = [Factor(name=name, low=0, high=1), Factor(name="B", low=5, high=9)]
    with pytest.raises(ValueError, match="reserved"):
        generate_design(factors, "full_factorial", n_center_points=0, n_blocks=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_replicates": 0}, "n_replicates"),
        ({"n_replicates": -1}, "n_replicates"),
        ({"n_center_points": -3}, "n_center_points"),
        ({"n_blocks": -2}, "n_blocks"),
        ({"n_blocks": 0}, "n_blocks"),
    ],
)
def test_counts_below_their_minimum_raise(kwargs: dict, match: str) -> None:
    """These were treated as the defaults (or reached numpy as a negative dimension)."""
    with pytest.raises(ValueError, match=match):
        generate_design(_factors(3), "full_factorial", **kwargs)


_TWO_LEVELS = Factor(name="C", type="categorical", levels=["lo", "hi"])


def test_a_categorical_factor_cannot_take_a_centre_setting() -> None:
    """A two-level categorical factor is coded -1 and +1; the 0 of a centre point names no level."""
    with pytest.raises(
        ValueError,
        match=(
            r"^Categorical factor 'C' has 2 levels, coded \[-1\.0, 1\.0\], but the design asks for the setting 0\.0; "
            r"this design type cannot place a categorical factor there\.$"
        ),
    ):
        categorical_labels(np.array([-1.0, 0.0]), _TWO_LEVELS)


def test_labelled_design_columns_are_coded_back_to_numbers() -> None:
    """A design holding level labels (an object array) is turned back into codes column by column."""
    matrix = np.array([["lo", 1.0], ["hi", -1.0], [-1.0, 0.0]], dtype=object)
    codes = _numeric_codes(matrix, [_TWO_LEVELS, Factor(name="A", low=0, high=1)])
    np.testing.assert_array_equal(codes, [[-1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
    assert codes.dtype == float


def test_negative_center_points_for_box_behnken_raise_a_clear_error() -> None:
    with pytest.raises(ValueError, match="n_center_points must be a whole number"):
        generate_design(_factors(3), "box_behnken", n_center_points=-2)


@pytest.mark.parametrize(
    ("design_type", "kwargs"),
    [
        ("full_factorial", {"resolution": 3}),
        ("plackett_burman", {"resolution": 5}),
        ("box_behnken", {"generators": ["C=AB"]}),
        ("ccd", {"generators": ["E=AB"], "resolution": 3}),
    ],
)
def test_resolution_and_generators_refused_where_unused(design_type: str, kwargs: dict) -> None:
    """They were echoed into the result as the design's own properties (a PB claiming resolution V)."""
    with pytest.raises(ValueError, match="define a fractional factorial"):
        generate_design(_factors(5), design_type, **kwargs)


def test_result_reports_only_the_measured_resolution() -> None:
    result = generate_design(_factors(3), "full_factorial")
    assert result.resolution is None
    assert result.generators is None


def test_resolution_without_design_type_chooses_a_fraction() -> None:
    result = generate_design(_factors(5), resolution=3, n_center_points=0)
    assert result.design_type == "fractional_factorial"
    assert result.n_runs == 8


@pytest.mark.parametrize(
    ("design_type", "k", "budget"), [("fractional_factorial", 7, 16), ("ccd", 5, 20), ("full_factorial", 3, 10)]
)
def test_fixed_size_design_above_the_budget_raises(design_type: str, k: int, budget: int) -> None:
    """The budget was ignored: a 7-factor half fraction came back with 67 runs for a budget of 16."""
    with pytest.raises(ValueError, match=f"more than budget={budget}"):
        generate_design(_factors(k), design_type, budget=budget)


def test_fixed_size_design_within_the_budget_is_built() -> None:
    assert generate_design(_factors(3), "full_factorial", budget=11).n_runs == 11


def _center_rows(result, names: str) -> int:
    return int((result.design[list(names)] == 0).all(axis=1).sum())


@pytest.mark.parametrize(("n_center_points", "expected"), [(None, 1), (0, 1), (1, 1), (4, 4)])
def test_dsd_honours_n_center_points_as_the_total(n_center_points: int | None, expected: int) -> None:
    """A DSD always came back with its one built-in centre run, whatever was asked."""
    result = generate_design(_factors(6), "dsd", n_center_points=n_center_points)
    assert _center_rows(result, "ABCDEF") == expected
    assert result.n_runs == 13 + expected - 1


def test_omars_with_a_budget_counts_its_centre_runs() -> None:
    """The ILP path hard-coded one centre run."""
    result = generate_design(_factors(3), "omars", budget=21, n_center_points=3, random_state=1)
    assert result.n_runs == 21
    assert _center_rows(result, "ABC") == 3


@pytest.mark.parametrize("budget", [16, 30])
def test_omars_treats_the_budget_as_an_upper_bound(budget: int) -> None:
    """An even budget raised, suggesting n_runs=budget+1, above the budget."""
    result = generate_design(_factors(3), "omars", budget=budget, random_state=1)
    assert result.n_runs == budget - 1


def test_omars_budget_too_small_names_the_budget() -> None:
    with pytest.raises(ValueError, match="budget=8"):
        generate_design(_factors(3), "omars", budget=8, random_state=1)


@pytest.mark.parametrize(
    ("design_type", "factors", "kwargs"),
    [
        ("mixture", [Factor(name=f"x{i}", type="mixture") for i in range(3)], {}),
        ("d_optimal", _factors(3), {"budget": 8}),
        ("taguchi", _factors(3), {}),
        ("latin_hypercube", _factors(3), {"budget": 8}),
    ],
)
def test_positive_center_points_refused_where_the_design_adds_none(
    design_type: str, factors: list[Factor], kwargs: dict
) -> None:
    """n_center_points was silently ignored by these families."""
    with pytest.raises(ValueError, match="does not add centre points"):
        generate_design(factors, design_type, n_center_points=4, **kwargs)
    generate_design(factors, design_type, n_center_points=0, **kwargs)


def test_full_factorial_respects_the_combinatorial_cap() -> None:
    from process_improve.config import settings

    k = settings.max_factors_combinatorial + 1
    factors = [Factor(name=f"X{i}", low=0, high=1) for i in range(k)]
    with pytest.raises(ValueError, match="combinatorial cap"):
        generate_design(factors, "full_factorial")


@pytest.mark.parametrize(("k", "budget"), [(3, 8), (3, 6), (6, 8), (8, 11), (10, 12), (5, 12), (5, 15), (4, 6), (2, 3)])
def test_auto_selected_design_fits_the_budget(k: int, budget: int) -> None:
    """Centre points were left out of the comparison, and a too-large PB or D-optimal model was chosen."""
    factors = [Factor(name=f"X{i}", low=0, high=1) for i in range(k)]
    result = generate_design(factors, budget=budget)
    assert result.n_runs <= budget


def test_auto_selection_counts_replicates() -> None:
    """Two replicates of the 2^3 factorial and 3 centre runs (not replicated, #513) are 19 runs."""
    result = generate_design(_factors(3), budget=19, n_replicates=2)
    assert result.design_type == "full_factorial"
    assert result.n_runs == 19
    smaller = generate_design(_factors(3), budget=18, n_replicates=2)
    assert smaller.design_type != "full_factorial"
    assert smaller.n_runs <= 18


def test_auto_selection_routes_many_level_categorical_factors_to_designs_that_hold_them() -> None:
    """Six continuous factors and a three-level categorical went to a fraction that refuses it."""
    factors = [*[Factor(name=f"X{i}", low=0, high=1) for i in range(6)]]
    factors.append(Factor(name="C", type="categorical", levels=["a", "b", "c"]))
    assert generate_design(factors).design_type == "full_factorial"
    result = generate_design(factors, budget=20)
    assert result.design_type == "d_optimal"
    assert result.n_runs <= 20


def test_none_seed_draws_a_random_order() -> None:
    """random_state=None returned the standard order with every centre point last."""
    orders = {tuple(generate_design(_factors(3), "full_factorial", random_state=None).run_order) for _ in range(5)}
    assert orders != {tuple(range(1, 12))}
    assert len(orders) > 1


def test_optimal_design_without_split_plot_is_randomised() -> None:
    """The pyoptex backend skipped randomisation even without hard_to_change factors."""
    pytest.importorskip("pyoptex")
    orders = [generate_design(_factors(3), "d_optimal", budget=10, random_state=seed).run_order for seed in (1, 2)]
    assert orders[0] != list(range(1, 11)) or orders[1] != list(range(1, 11))
    assert orders[0] != orders[1]


def test_fixed_runs_with_replicates_raise() -> None:
    """Replicating the design repeated runs already made and shuffled them among the new ones."""
    import numpy as np

    from process_improve.experiments.designs_utils import build_design_result

    matrix = np.array([[0.0, 0.0], [0.5, -0.5], [1, 1], [-1, 1], [1, -1], [-1, -1]])
    factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
    with pytest.raises(ValueError, match="n_replicates cannot be combined with fixed_runs"):
        build_design_result(matrix, factors, "d_optimal", n_replicates=2, random_state=1, n_leading_fixed=2)
    result = build_design_result(matrix, factors, "d_optimal", random_state=1, n_leading_fixed=2)
    assert result.design[["A", "B"]].to_numpy()[:2].tolist() == [[0.0, 0.0], [0.5, -0.5]]
