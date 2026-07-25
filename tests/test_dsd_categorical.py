"""Tests for two-level categorical factors in definitive screening designs.

Every oracle here comes from Jones, B. and Nachtsheim, C. J. (2013),
"Definitive screening designs with added two-level categorical factors",
*Journal of Quality Technology*, 45(2):121-129:

- **Table 4** lists run sizes for both procedures over 4 to 12 continuous
  factors and 1 to 4 categorical factors: 36 published values.
- **Table 4** also lists the largest entry of the constant-term row of the
  alias matrix for the DSD-augment designs, which works out to ``2 / n``.
- **Table 3** gives that alias matrix in full for four continuous and two
  categorical factors: zero against every continuous-continuous interaction,
  0.1429 against every interaction involving a categorical factor.
- **Section 2** defines the DSD-augment procedure and its defining property,
  that main effects stay unbiased by every second-order effect.
- **Section 3** defines the ORTH-augment procedure, which gives an orthogonal
  linear main-effects plan for up to four categorical factors.

Two entries in the paper's Figure 1 disagree with the rest of the paper and are
not used as oracles; see ``TestFigure1Errata`` for the evidence.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_response_surface import (
    dispatch_dsd,
    dsd_centre_runs,
    dsd_conference_order,
    dsd_run_count,
)
from process_improve.experiments.factor import Factor

# Jones & Nachtsheim (2013) Table 4: (n_DSD, n_ORTH, m continuous, c categorical).
TABLE_4 = [
    (14, 14, 4, 1), (14, 16, 4, 2), (18, 20, 4, 3), (18, 20, 4, 4),
    (14, 14, 5, 1), (18, 20, 5, 2), (18, 20, 5, 3), (22, 24, 5, 4),
    (18, 18, 6, 1), (18, 20, 6, 2), (22, 24, 6, 3), (22, 24, 6, 4),
    (18, 18, 7, 1), (22, 24, 7, 2), (22, 24, 7, 3), (26, 28, 7, 4),
    (22, 22, 8, 1), (22, 24, 8, 2), (26, 28, 8, 3), (26, 28, 8, 4),
    (22, 22, 9, 1), (26, 28, 9, 2), (26, 28, 9, 3), (30, 32, 9, 4),
    (26, 26, 10, 1), (26, 28, 10, 2), (30, 32, 10, 3), (30, 32, 10, 4),
    (26, 26, 11, 1), (30, 32, 11, 2), (30, 32, 11, 3), (34, 36, 11, 4),
    (30, 30, 12, 1), (30, 32, 12, 2), (34, 36, 12, 3), (34, 36, 12, 4),
]  # fmt: skip

_TABLE_4_IDS = [f"m{m}c{c}" for _, _, m, c in TABLE_4]


def _factors(n_continuous: int, n_categorical: int, *, interleave: bool = False) -> list[Factor]:
    """Build a factor list, optionally with the categorical factors not last."""
    continuous = [Factor(name=f"X{i + 1}", low=-1.0, high=1.0) for i in range(n_continuous)]
    categorical = [
        Factor(name=f"C{j + 1}", type="categorical", levels=[f"lo{j + 1}", f"hi{j + 1}"]) for j in range(n_categorical)
    ]
    if not interleave:
        return continuous + categorical
    merged: list[Factor] = []
    for i in range(max(len(continuous), len(categorical))):
        if i < len(categorical):
            merged.append(categorical[i])
        if i < len(continuous):
            merged.append(continuous[i])
    return merged


def _model_matrices(design: np.ndarray, n_continuous: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the first-order matrix, the interaction matrix, and interaction labels.

    The model is Equation (1) of the paper: intercept, linear main effects for
    every factor, all two-factor interactions, and quadratics for the continuous
    factors only (a two-level categorical factor has no square).
    """
    n, k = design.shape
    first_order = np.column_stack([np.ones(n), design])
    labels = [f"{a + 1}{b + 1}" for a, b in itertools.combinations(range(k), 2)]
    interactions = np.column_stack([design[:, a] * design[:, b] for a, b in itertools.combinations(range(k), 2)])
    return first_order, interactions, labels


def _second_order(design: np.ndarray, n_continuous: int) -> np.ndarray:
    """All second-order terms: quadratics on the continuous factors, plus every interaction."""
    _n, k = design.shape
    quadratics = [design[:, j] ** 2 for j in range(n_continuous)]
    interactions = [design[:, a] * design[:, b] for a, b in itertools.combinations(range(k), 2)]
    return np.column_stack(quadratics + interactions)


def _alias_matrix(first_order: np.ndarray, terms: np.ndarray) -> np.ndarray:
    """Return the alias matrix ``(X1' X1)^-1 X1' X2``."""
    return np.linalg.solve(first_order.T @ first_order, first_order.T @ terms)


class TestTable4RunSizes:
    """The 36 published run sizes, for both procedures."""

    @pytest.mark.parametrize(("n_dsd", "_n_orth", "m", "c"), TABLE_4, ids=_TABLE_4_IDS)
    def test_dsd_augment_run_size(self, n_dsd: int, _n_orth: int, m: int, c: int) -> None:
        design, meta = dispatch_dsd(_factors(m, c), categorical_method="dsd")
        assert design.shape == (n_dsd, m + c)
        assert meta["categorical_method"] == "dsd"
        assert meta["n_categorical"] == c

    @pytest.mark.parametrize(("_n_dsd", "n_orth", "m", "c"), TABLE_4, ids=_TABLE_4_IDS)
    def test_orth_augment_run_size(self, _n_dsd: int, n_orth: int, m: int, c: int) -> None:
        design, meta = dispatch_dsd(_factors(m, c), categorical_method="orth")
        assert design.shape == (n_orth, m + c)
        assert meta["categorical_method"] == "orth"

    @pytest.mark.parametrize(("n_dsd", "n_orth", "m", "c"), TABLE_4, ids=_TABLE_4_IDS)
    def test_run_count_helper_predicts_the_generated_size(self, n_dsd: int, n_orth: int, m: int, c: int) -> None:
        """The planning helper must agree with the constructor, not approximate it."""
        assert dsd_run_count(m + c, n_categorical=c, categorical_method="dsd") == n_dsd
        assert dsd_run_count(m + c, n_categorical=c, categorical_method="orth") == n_orth

    def test_orth_costs_two_extra_runs_only_beyond_one_categorical(self) -> None:
        """Table 4 shows n_ORTH == n_DSD for every single-categorical row."""
        for n_dsd, n_orth, _m, c in TABLE_4:
            if c == 1:
                assert n_orth == n_dsd
            else:
                assert n_orth == n_dsd + 2

    def test_centre_run_counts(self) -> None:
        assert dsd_centre_runs(0) == 1
        assert dsd_centre_runs(1) == 2
        assert dsd_centre_runs(1, "orth") == 2
        assert dsd_centre_runs(3) == 2
        assert dsd_centre_runs(3, "orth") == 4

    def test_a_categorical_factor_costs_a_conference_column(self) -> None:
        """Nine continuous plus three categorical is a twelve-factor design, not a nine-factor one."""
        assert dsd_conference_order(9 + 3) == dsd_conference_order(12)
        design, meta = dispatch_dsd(_factors(9, 3))
        assert design.shape[1] == 12
        assert meta["conference_order"] == dsd_conference_order(12)


class TestDsdAugmentProperties:
    """Section 2: the design stays *definitive*."""

    @pytest.mark.parametrize(("m", "c"), [(4, 1), (4, 2), (4, 3), (4, 4), (5, 2), (6, 3), (8, 4), (12, 2)])
    def test_main_effects_unbiased_by_every_second_order_effect(self, m: int, c: int) -> None:
        """The defining property: main-effect rows of the alias matrix are exactly zero.

        This is what "definitive" means, and it is what separates DSD-augment
        from ORTH-augment.
        """
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="dsd")
        first_order, _inter, _labels = _model_matrices(design, m)
        alias = _alias_matrix(first_order, _second_order(design, m))
        assert np.abs(alias[1:, :]).max() < 1e-12

    @pytest.mark.parametrize(("n_dsd", "_n_orth", "m", "c"), TABLE_4, ids=_TABLE_4_IDS)
    def test_largest_constant_term_alias_is_two_over_n(self, n_dsd: int, _n_orth: int, m: int, c: int) -> None:
        """Table 4's rightmost column: the largest entry of the alias matrix's first row.

        Across all 36 published rows this equals ``2 / n`` exactly.  It is the
        check that is sensitive to the sign vectors the search picks, which the
        run-size check is not.
        """
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="dsd")
        first_order, interactions, _labels = _model_matrices(design, m)
        alias = _alias_matrix(first_order, interactions)
        assert np.abs(alias[0]).max() == pytest.approx(2 / n_dsd, abs=5e-5)

    def test_table_3_alias_row_reproduced_entry_by_entry(self) -> None:
        """Table 3: the full constant-term alias row for four continuous, two categorical.

        Zero against each of 12, 13, 14, 23, 24 and 34; 0.1429 against each of
        the nine interactions that involve a categorical factor, including the
        categorical-by-categorical 56.
        """
        design, _meta = dispatch_dsd(_factors(4, 2), categorical_method="dsd")
        first_order, interactions, labels = _model_matrices(design, 4)
        alias = dict(zip(labels, np.abs(_alias_matrix(first_order, interactions)[0]), strict=True))

        for continuous_pair in ("12", "13", "14", "23", "24", "34"):
            assert alias[continuous_pair] == pytest.approx(0.0, abs=1e-12), continuous_pair
        for categorical_pair in ("15", "16", "25", "26", "35", "36", "45", "46", "56"):
            assert alias[categorical_pair] == pytest.approx(0.1429, abs=5e-5), categorical_pair

    def test_information_matrix_matches_equation_2(self) -> None:
        """Equation (2): X'X for four continuous and two categorical factors.

        14 on the intercept, 10 on each continuous main effect (four of the
        fourteen runs are at that factor's centre), 14 on each categorical main
        effect, zero between the intercept and every main effect, and entries of
        magnitude 2 coupling the categorical columns to the rest.
        """
        design, _meta = dispatch_dsd(_factors(4, 2), categorical_method="dsd")
        first_order, _inter, _labels = _model_matrices(design, 4)
        info = first_order.T @ first_order

        assert info[0, 0] == 14
        assert list(np.diag(info)[1:5]) == [10] * 4
        assert list(np.diag(info)[5:]) == [14] * 2
        assert np.abs(info[0, 1:]).max() == 0  # intercept orthogonal to every main effect
        coupling = np.abs(info[1:, 1:] - np.diag(np.diag(info[1:, 1:])))
        assert set(np.unique(coupling)) <= {0.0, 2.0}

    @pytest.mark.parametrize(("m", "c"), [(4, 1), (4, 2), (5, 3), (6, 4), (8, 2)])
    def test_categorical_columns_are_balanced(self, m: int, c: int) -> None:
        """z_{j,1} = b_j and z_{j,2} = -b_j keeps the foldover pair balanced."""
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="dsd")
        for column in design[:, m:].T:
            assert (column == 1).sum() == (column == -1).sum()
            assert set(np.unique(column)) == {-1.0, 1.0}

    @pytest.mark.parametrize(("m", "c"), [(4, 2), (5, 3), (8, 2)])
    def test_information_matrix_is_not_diagonal(self, m: int, c: int) -> None:
        """The documented cost of DSD-augment.

        The paper states plainly that "the main effects columns for categorical
        factors exhibit small correlations so that the information matrix is not
        diagonal".  Asserting it keeps the trade-off honest rather than letting a
        future change quietly claim orthogonality it does not have.
        """
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="dsd")
        first_order, _inter, _labels = _model_matrices(design, m)
        info = first_order.T @ first_order
        off_diagonal = np.abs(info - np.diag(np.diag(info)))
        assert off_diagonal.max() > 0
        assert off_diagonal.max() == 2  # never larger than 2, per Section 2


class TestOrthAugmentProperties:
    """Section 3: an orthogonal linear main-effects plan for up to four categorical factors."""

    @pytest.mark.parametrize(("m", "c"), [(4, 1), (4, 2), (4, 3), (4, 4), (5, 2), (6, 3), (8, 4), (12, 4)])
    def test_main_effects_plan_is_exactly_orthogonal(self, m: int, c: int) -> None:
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="orth")
        first_order, _inter, _labels = _model_matrices(design, m)
        info = first_order.T @ first_order
        assert np.abs(info - np.diag(np.diag(info))).max() < 1e-9

    @pytest.mark.parametrize(("m", "c"), [(4, 2), (5, 3), (6, 4)])
    def test_main_effects_are_partially_aliased_with_categorical_interactions(self, m: int, c: int) -> None:
        """The documented cost of ORTH-augment, and the reason it is not the default.

        The paper: "all of the main effects may be biased by potential two-factor
        interactions involving categorical factors", with entries ranging up to
        about 0.4.
        """
        design, _meta = dispatch_dsd(_factors(m, c), categorical_method="orth")
        first_order, interactions, _labels = _model_matrices(design, m)
        alias = _alias_matrix(first_order, interactions)
        assert np.abs(alias[1:, :]).max() > 0
        assert np.abs(alias[1:, :]).max() <= 0.4 + 1e-9

    def test_beyond_four_categorical_factors_orthogonality_is_only_approximate(self) -> None:
        """Section 3 promises exact orthogonality only up to c = 4, "nearly" beyond.

        The design must still be usable, so the correlation is bounded rather
        than absent.
        """
        design, _meta = dispatch_dsd(_factors(6, 6), categorical_method="orth")
        first_order, _inter, _labels = _model_matrices(design, 6)
        info = first_order.T @ first_order
        scale = np.sqrt(np.outer(np.diag(info), np.diag(info)))
        correlation = np.abs(info / scale)
        np.fill_diagonal(correlation, 0.0)
        assert correlation.max() < 0.35


class TestFigure1Errata:
    """Two entries of the paper's Figure 1 disagree with the rest of the paper.

    Recorded here so a future reader who checks our output against the printed
    figure knows why the c = 2 panels differ, and so the reasoning is testable
    rather than asserted in prose.
    """

    def test_dsd_c2_panel_is_not_d_optimal(self) -> None:
        """Figure 1(a), c = 2 contradicts Table 3, Equation (2), and Step 3.

        As printed, the two categorical columns have inner product 6, giving a
        constant-term alias of 6/14 = 0.4286 against the 56 interaction.  Table 3
        gives 0.1429 there and Equation (2) gives an inner product of magnitude
        2.  Section 2, Step 3 resolves it: the sign vectors are chosen to
        maximise the first-order information determinant, and the choice that
        does so is the one matching Table 3, not the one printed.
        """
        design, _meta = dispatch_dsd(_factors(4, 2), categorical_method="dsd")
        ours = float(design[:, 4] @ design[:, 5])
        assert abs(ours) == 2  # Equation (2), not the 6 the figure implies

        # The printed alternative is a valid design, just a worse one: same run
        # size, larger aliasing, smaller determinant.
        printed = design.copy()
        printed[:, 5] = np.where(np.abs(printed[:, 4] - printed[:, 5]) > 0, -printed[:, 5], printed[:, 5])
        model_ours = np.column_stack([np.ones(design.shape[0]), design])
        model_printed = np.column_stack([np.ones(printed.shape[0]), printed])
        assert np.linalg.slogdet(model_ours.T @ model_ours)[1] >= np.linalg.slogdet(model_printed.T @ model_printed)[1]

    def test_orth_c2_panel_breaks_its_own_step_3(self) -> None:
        """Figure 1(b), c = 2, run 10 prints a minus in the first categorical column.

        Runs 9 and 10 are the foldover pair carrying z_{1,1} and z_{1,2}, and
        Section 3, Step 3 requires both to be +1.  With the minus as printed the
        column is unbalanced 7 to 9 and the main-effects plan is not orthogonal,
        contradicting the procedure's whole purpose.  Following Step 3 gives a
        balanced column and exact orthogonality.
        """
        design, _meta = dispatch_dsd(_factors(4, 2), categorical_method="orth")
        for column in design[:, 4:].T:
            assert (column == 1).sum() == (column == -1).sum() == 8
        first_order, _inter, _labels = _model_matrices(design, 4)
        info = first_order.T @ first_order
        assert np.abs(info - np.diag(np.diag(info))).max() < 1e-9


class TestFactorOrderAndValidation:
    """API behaviour around the construction."""

    @pytest.mark.parametrize("method", ["dsd", "orth"])
    def test_categorical_factors_need_not_be_last(self, method: str) -> None:
        """The construction needs them trailing; the caller should not have to care."""
        factors = _factors(4, 2, interleave=True)
        design, _meta = dispatch_dsd(factors, categorical_method=method)
        assert [f.name for f in factors] == ["C1", "X1", "C2", "X2", "X3", "X4"]
        for position, factor in enumerate(factors):
            column = design[:, position]
            if factor.type.value == "categorical":
                assert set(np.unique(column)) == {-1.0, 1.0}, f"{factor.name} should be two-level"
            else:
                assert 0.0 in set(np.unique(column)), f"{factor.name} should carry a centre level"

    @pytest.mark.parametrize("method", ["dsd", "orth"])
    def test_reordering_preserves_the_design(self, method: str) -> None:
        """Permuting the factor list permutes the columns and nothing else."""
        ordered, _ = dispatch_dsd(_factors(4, 2), categorical_method=method)
        interleaved, _ = dispatch_dsd(_factors(4, 2, interleave=True), categorical_method=method)
        # Interleaved order is C1, X1, C2, X2, X3, X4 -> map back to X1..X4, C1, C2.
        remapped = interleaved[:, [1, 3, 4, 5, 0, 2]]
        assert np.array_equal(np.sort(ordered, axis=0), np.sort(remapped, axis=0))

    def test_three_level_categorical_is_rejected_with_a_pointer(self) -> None:
        factors = [
            *[Factor(name=f"X{i}", low=-1, high=1) for i in range(3)],
            Factor(name="Catalyst", type="categorical", levels=["A", "B", "C"]),
        ]
        with pytest.raises(ValueError, match="two-level categorical factors only"):
            dispatch_dsd(factors)
        with pytest.raises(ValueError, match="d_optimal"):
            dispatch_dsd(factors)

    def test_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="categorical_method"):
            dispatch_dsd(_factors(4, 2), categorical_method="orthogonal")
        with pytest.raises(ValueError, match="categorical_method"):
            dsd_centre_runs(2, "nonsense")

    def test_method_is_ignored_without_categorical_factors(self) -> None:
        """An all-continuous DSD is unchanged by the argument."""
        a, meta_a = dispatch_dsd(_factors(6, 0), categorical_method="dsd")
        b, meta_b = dispatch_dsd(_factors(6, 0), categorical_method="orth")
        assert np.array_equal(a, b)
        assert "categorical_method" not in meta_a
        assert meta_b["centre_runs"] == 1


class TestGenerateDesignIntegration:
    """The user-facing path."""

    def test_categorical_levels_appear_in_the_actual_design(self) -> None:
        factors = [
            Factor(name="Temp", low=150, high=200, units="degC"),
            Factor(name="Press", low=1, high=5, units="bar"),
            Factor(name="Time", low=10, high=60, units="min"),
            Factor(name="Solvent", type="categorical", levels=["MeOH", "EtOH"]),
        ]
        result = generate_design(factors, design_type="dsd", center_points=0)
        assert result.n_runs == dsd_run_count(4, n_categorical=1)
        solvent = result.design_actual["Solvent"]
        assert set(solvent.unique()) == {"MeOH", "EtOH"}
        assert set(result.design_actual["Temp"].unique()) == {150.0, 175.0, 200.0}

    def test_method_is_threaded_through_and_recorded(self) -> None:
        factors = _factors(4, 2)
        default = generate_design(factors, design_type="dsd", center_points=0)
        orth = generate_design(factors, design_type="dsd", center_points=0, categorical_method="orth")
        assert default.metadata["categorical_method"] == "dsd"
        assert orth.metadata["categorical_method"] == "orth"
        assert default.n_runs == 14
        assert orth.n_runs == 16

    def test_three_level_categorical_message_reaches_the_caller(self) -> None:
        factors = [
            *[Factor(name=f"X{i}", low=-1, high=1) for i in range(3)],
            Factor(name="Catalyst", type="categorical", levels=["A", "B", "C"]),
        ]
        with pytest.raises(ValueError, match="two-level categorical factors only"):
            generate_design(factors, design_type="dsd")


class TestManyCategoricalFactors:
    """Beyond seven categorical factors the exhaustive sign search gives way to a heuristic."""

    def test_heuristic_search_still_produces_a_valid_design(self) -> None:
        """2 ** (2c) becomes too many determinants, so the search is seeded and probed.

        The paper switches to a coordinate-exchange algorithm at large c for the
        same reason.  The design must still be definitive; only optimality of the
        sign choice is given up.
        """
        design, meta = dispatch_dsd(_factors(4, 8), categorical_method="dsd")
        assert meta["n_categorical"] == 8
        assert design.shape[1] == 12
        assert design.shape[0] == dsd_run_count(12, n_categorical=8)

        first_order, _inter, _labels = _model_matrices(design, 4)
        alias = _alias_matrix(first_order, _second_order(design, 4))
        assert np.abs(alias[1:, :]).max() < 1e-12  # still definitive

        for column in design[:, 4:].T:
            assert set(np.unique(column)) == {-1.0, 1.0}
            assert (column == 1).sum() == (column == -1).sum()

    def test_orth_centre_block_cycles_beyond_four(self) -> None:
        """The ORTH centre pattern repeats every four columns for c > 4."""
        design, meta = dispatch_dsd(_factors(4, 6), categorical_method="orth")
        assert meta["centre_runs"] == 4
        centre = design[-4:, 4:]
        assert set(np.unique(centre)) == {-1.0, 1.0}
        # Each of the four centre runs raises exactly one or two categorical factors.
        assert all((row == 1).sum() >= 1 for row in centre)


class TestCategoricalLabelMapping:
    """``matrix_to_columns`` translates coded categorical columns into level labels."""

    def test_labels_pass_through_untouched(self) -> None:
        """Families that already emit labels (the optimal designs) must not be re-mapped."""
        from process_improve.experiments.designs_utils import matrix_to_columns

        factors = [
            Factor(name="X1", low=0, high=10),
            Factor(name="Cat", type="categorical", levels=["red", "blue"]),
        ]
        matrix = np.array([[0.0, "red"], [10.0, "blue"]], dtype=object)
        columns = matrix_to_columns(matrix, factors)
        assert list(columns[1]) == ["red", "blue"]

    def test_taguchi_with_a_categorical_factor(self) -> None:
        """Regression: this raised before index-coded categorical columns were mapped.

        ``dispatch_taguchi`` codes a categorical factor as level indices 0..n-1,
        which the ``Column`` constructor rejected because the labels were never
        substituted.
        """
        factors = [
            Factor(name="A", low=0, high=1),
            Factor(name="B", low=0, high=1),
            Factor(name="Cat", type="categorical", levels=["x", "y"]),
        ]
        result = generate_design(factors, design_type="taguchi", center_points=0)
        assert set(result.design_actual["Cat"].unique()) == {"x", "y"}

    def test_three_level_categorical_indices_are_mapped(self) -> None:
        """The index branch is not limited to two levels."""
        from process_improve.experiments.designs_utils import matrix_to_columns

        factors = [Factor(name="Cat", type="categorical", levels=["a", "b", "c"])]
        columns = matrix_to_columns(np.array([[0.0], [2.0], [1.0], [0.0]]), factors)
        assert list(columns[0]) == ["a", "c", "b", "a"]

    def test_unrecognised_numeric_coding_is_left_for_the_column_to_reject(self) -> None:
        """A coding that matches neither convention must not be silently reinterpreted."""
        from process_improve.experiments.designs_utils import matrix_to_columns

        factors = [Factor(name="Cat", type="categorical", levels=["a", "b"])]
        with pytest.raises(ValueError, match="All values must be present in `levels`"):
            matrix_to_columns(np.array([[7.0], [9.0]]), factors)
