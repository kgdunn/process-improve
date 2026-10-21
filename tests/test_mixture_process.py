"""Tests for mixture-process models and designs: mixture components and process factors together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from process_improve.experiments import DesignRegion, Factor
from process_improve.experiments.designs_mixture_process import mixture_process_model

#: Three bounded components and two process factors, used for the region and the designs.
BLENDS = [
    Factor(name="x1", type="mixture", low=0.1, high=0.6),
    Factor(name="x2", type="mixture", low=0.1, high=0.6),
    Factor(name="x3", type="mixture", low=0.1, high=0.6),
]
SETTINGS = [Factor(name="T", low=20, high=80, units="degC"), Factor(name="S", low=100, high=300, units="rpm")]


def _points(q: int, m: int, n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Random blends of ``q`` components, which sum to 1, with ``m`` process factors in coded units."""
    rng = np.random.default_rng(seed)
    blends = rng.dirichlet(np.ones(q), size=n)
    data = {f"x{i + 1}": blends[:, i] for i in range(q)}
    data.update({f"z{k + 1}": rng.uniform(-1, 1, size=n) for k in range(m)})
    return pd.DataFrame(data)


def _names(q: int, m: int) -> tuple[list[str], list[str]]:
    return [f"x{i + 1}" for i in range(q)], [f"z{k + 1}" for k in range(m)]


def _rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=1e-9))


class TestModelSize:
    """The parameter counts that Goos and Jones (2011, section 6.3) give for each model."""

    @pytest.mark.parametrize(("q", "m"), [(2, 1), (3, 2), (4, 3), (5, 2)])
    def test_kowalski_cornell_vining(self, q: int, m: int) -> None:
        expected = q + q * (q - 1) // 2 + q * m + m * (m - 1) // 2 + m
        quadratic = mixture_process_model(*_names(q, m), "quadratic")
        assert quadratic.n_parameters == expected
        assert mixture_process_model(*_names(q, m), "kcv").terms == quadratic.terms

    @pytest.mark.parametrize(("q", "m"), [(2, 1), (3, 2), (4, 3)])
    def test_crossed(self, q: int, m: int) -> None:
        expected = (q + q * (q - 1) // 2) * (1 + m + m * (m - 1) // 2)
        assert mixture_process_model(*_names(q, m), "scheffe_quadratic * interactions").n_parameters == expected

    def test_crossed_three_components_two_process_factors(self) -> None:
        """The 24 parameters the book quotes for three ingredients and two process variables."""
        assert mixture_process_model(*_names(3, 2), "scheffe_quadratic * interactions").n_parameters == 24

    def test_additive_table_6_2(self) -> None:
        """Three mills' second-order Scheffé model plus two process main effects: 8 terms."""
        assert mixture_process_model(*_names(3, 2), "scheffe_quadratic + main_effects").n_parameters == 8

    @pytest.mark.parametrize(("q", "m"), [(3, 1), (4, 3)])
    def test_first_order(self, q: int, m: int) -> None:
        assert mixture_process_model(*_names(q, m), "main_effects").n_parameters == q + m


class TestOrderBasedModels:
    """The order-based models are the full polynomial in every factor, less what sum(x) = 1 makes redundant."""

    @staticmethod
    def _full_polynomial(data: pd.DataFrame, squares: bool) -> np.ndarray:
        names = list(data.columns)
        rhs = f"({' + '.join(names)}) ** 2"
        if squares:
            rhs += "".join(f" + I({n} ** 2)" for n in names)
        return np.asarray(dmatrix(rhs, data))

    @pytest.mark.parametrize(("order", "squares"), [("interactions", False), ("quadratic", True)])
    @pytest.mark.parametrize(("q", "m"), [(3, 2), (4, 3)])
    def test_same_span_as_the_full_polynomial(self, order: str, squares: bool, q: int, m: int) -> None:
        data = _points(q, m)
        model = mixture_process_model(*_names(q, m), order).matrix(data)
        full = self._full_polynomial(data, squares)
        # On blends that sum to 1 the full polynomial loses exactly the redundant terms, and the model
        # holds the rest: its columns are independent and span every column of the full polynomial.
        assert _rank(model) == model.shape[1] == _rank(full)
        assert _rank(np.hstack([model, full])) == model.shape[1]


class TestMatrixAndFormulaAgree:
    """The model matrix the design engine optimises is the one the recorded formula builds."""

    @pytest.mark.parametrize(
        "model_type",
        [
            "main_effects",
            "interactions",
            "quadratic",
            "scheffe_quadratic + main_effects",
            "scheffe_special_cubic * interactions",
            "scheffe_cubic * main_effects",
            "scheffe_linear + quadratic",
        ],
    )
    def test_columns_match(self, model_type: str) -> None:
        data = _points(3, 2)
        model = mixture_process_model(*_names(3, 2), model_type)
        ours = model.matrix(data)
        theirs = np.asarray(dmatrix(model.formula_rhs(), data))
        assert theirs.shape == ours.shape
        # Each column appears once on each side, though patsy orders the terms by degree.
        matches = np.isclose(ours[:, :, None], theirs[:, None, :]).all(axis=0)
        assert (matches.sum(axis=1) == 1).all()
        assert (matches.sum(axis=0) == 1).all()

    def test_cubic_difference_values(self) -> None:
        data = _points(3, 1)
        model = mixture_process_model(*_names(3, 1), "scheffe_cubic + main_effects")
        names = [":".join(a.code for a in term) for term in model.terms]
        column = model.matrix(data)[:, names.index("I(x1 * x2 * (x1 - x2))")]
        np.testing.assert_allclose(column, data.x1 * data.x2 * (data.x1 - data.x2))

    def test_formula_has_no_intercept(self) -> None:
        assert mixture_process_model(*_names(3, 2), "interactions").formula_rhs().startswith("-1 + x1 + x2 + x3")


class TestModelNames:
    @pytest.mark.parametrize("model_type", ["cubic", "scheffe_quadratic", "linear * interactions x", "full"])
    def test_unknown_names_are_refused(self, model_type: str) -> None:
        with pytest.raises(ValueError, match=r"not a mixture-process model|Unknown"):
            mixture_process_model(*_names(3, 2), model_type)

    def test_unknown_process_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown process model 'cubic'"):
            mixture_process_model(*_names(3, 2), "scheffe_quadratic * cubic")

    def test_combined_name_is_normalised(self) -> None:
        model = mixture_process_model(*_names(3, 2), "quadratic*main_effects")
        assert model.name == "scheffe_quadratic * main_effects"


class TestRegion:
    """A mixture-process region is the product of a mixture region and a box region."""

    def test_kind_columns_and_bounds(self) -> None:
        region = DesignRegion([BLENDS[0], SETTINGS[0], *BLENDS[1:], SETTINGS[1]])
        assert region.kind == "mixture_process"
        assert region.names == ["x1", "T", "x2", "x3", "S"]
        assert region.bounds == [(0.1, 0.6), (-1.0, 1.0), (0.1, 0.6), (0.1, 0.6), (-1.0, 1.0)]

    def test_feasible_needs_blends_summing_to_one_and_settings_in_the_box(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 + x2 <= 0.7", "T + S <= 300"])
        points = np.array(
            [
                [0.3, 0.3, 0.4, 0.0, 0.0],  # feasible
                [0.3, 0.3, 0.3, 0.0, 0.0],  # the blend sums to 0.9
                [0.4, 0.4, 0.2, 0.0, 0.0],  # x1 + x2 = 0.8
                [0.3, 0.3, 0.4, 1.2, 0.0],  # T beyond its range
                [0.3, 0.3, 0.4, 1.0, 1.0],  # T + S = 380 in actual units
            ]
        )
        np.testing.assert_array_equal(region.feasible(points), [True, False, False, False, False])

    def test_sample_is_feasible_and_spread_over_both_parts(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 <= 0.4"])
        points = region.sample(4000, np.random.default_rng(1))
        assert region.feasible(points).all()
        np.testing.assert_allclose(points[:, :3].sum(axis=1), 1.0)
        np.testing.assert_allclose(points[:, 3:].mean(axis=0), 0.0, atol=0.05)

    def test_support_points_cross_the_parts(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS])
        blends = DesignRegion(BLENDS).support_points()
        settings = DesignRegion(SETTINGS).support_points()
        points = region.support_points()
        assert len(points) == len(blends) * len(settings)
        assert region.feasible(points).all()

    def test_volume_fraction_is_the_product_of_the_parts(self) -> None:
        region = DesignRegion([Factor(name=n, type="mixture") for n in "ABC"] + SETTINGS[:1], ["A <= 0.5", "T <= 50"])
        assert region.volume_fraction() == pytest.approx(0.75 * 0.5)

    def test_round_trip_keeps_the_kind(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 <= 0.4"])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.kind == "mixture_process"
        assert again.is_constrained

    def test_a_constraint_joining_both_parts_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"names both mixture components \['x1'\] and process factors \['T'\]"):
            DesignRegion([*BLENDS, *SETTINGS], ["x1 + 0.01*T <= 1"])

    def test_box_diagnostics_are_refused(self) -> None:
        with pytest.raises(ValueError, match="describes a box of factor settings"):
            DesignRegion([*BLENDS, *SETTINGS]).centroid()
