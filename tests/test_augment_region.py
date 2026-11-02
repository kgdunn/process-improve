"""augment_design inside a region: new runs keep to the limits on the settings (#809)."""

from __future__ import annotations

import doctest

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.augment import augment_design
from process_improve.experiments.region import DesignRegion
from process_improve.tool_spec import execute_tool_call

#: A reactor whose temperature and dosing rate cannot both be high: 3*T + 5*D <= 600 cuts the (+, +) corner.
FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
LIMIT = "3*T + 5*D <= 600"
#: The 2^2 factorial in coded units; its run at index 3, (150, 60), breaks LIMIT.
CUBE = pd.DataFrame({"T": [-1.0, 1.0, -1.0, 1.0], "D": [-1.0, -1.0, 1.0, 1.0]})


def _new_runs(result: dict) -> np.ndarray:
    return pd.DataFrame(result["new_runs"])[["T", "D"]].to_numpy(dtype=float)


class TestBrokenLimits:
    def test_each_point_names_the_ranges_it_leaves_and_the_limits_it_fails(self) -> None:
        region = DesignRegion(FACTORS, [LIMIT, "100 <= T + D <= 190"])
        points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.5, -1.0], [2.0, 1.0]])
        assert region.broken_limits(points) == [
            [LIMIT, "100 <= T + D <= 190"],  # a chained limit is named once
            [],
            ["100 <= T <= 150"],
            ["100 <= T <= 150", LIMIT, "100 <= T + D <= 190"],
        ]

    def test_a_point_breaks_no_limit_exactly_when_it_is_feasible(self) -> None:
        region = DesignRegion(FACTORS, [LIMIT, "T*D <= 7000", "sqrt(T - 110) >= 0"])
        points = np.random.default_rng(3).uniform(-1.3, 1.3, size=(400, 2))
        with np.errstate(invalid="ignore"):
            feasible = region.feasible(points)
        assert [not broken for broken in region.broken_limits(points)] == feasible.tolist()

    def test_a_curved_limit_that_cannot_be_evaluated_is_broken(self) -> None:
        """``sqrt`` of a negative number: :meth:`feasible` cannot run the setting either."""
        region = DesignRegion(FACTORS, ["sqrt(T - 110) >= 1"])
        assert region.broken_limits(np.array([[-1.0, 0.0]])) == [["sqrt(T - 110) >= 1"]]

    def test_a_mixture_region_is_refused(self) -> None:
        blend = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"]
        with pytest.raises(ValueError, match=r"DesignRegion.broken_limits\(\) describes a box"):
            DesignRegion(blend).broken_limits(np.array([[0.2, 0.3, 0.5]]))

    def test_the_example_runs(self) -> None:
        finder = doctest.DocTestFinder()
        [test] = finder.find(
            DesignRegion.broken_limits,
            "broken_limits",
            globs={"DesignRegion": DesignRegion, "Factor": Factor, "np": np},
        )
        result = doctest.DocTestRunner().run(test)
        assert (result.failed, result.attempted) == (0, 2)


class TestOptimalRuns:
    def test_new_runs_keep_to_the_limits_and_a_run_outside_is_kept_with_a_warning(self) -> None:
        """The run that could not be made stays; the six new runs avoid the corner the limit cuts off."""
        region = DesignRegion(FACTORS, [LIMIT])
        message = (
            r"1 run\(s\) of the existing design lie outside the region \(index \[3\]\), breaking 3\*T \+ 5\*D <= 600"
        )
        with pytest.warns(UserWarning, match=message):
            result = augment_design(
                CUBE, "add_runs_optimal", n_additional_runs=6, target_model="quadratic", region=region
            )
        assert region.feasible(_new_runs(result)).all()
        assert result["existing_runs_outside_region"] == [3]
        assert "The new runs keep to the region's limits: 3*T + 5*D <= 600." in result["explanation"]
        free = augment_design(CUBE, "add_runs_optimal", n_additional_runs=6, target_model="quadratic")
        assert not region.feasible(_new_runs(free)).all(), "without the region, runs go where the limit forbids"

    @pytest.mark.parametrize("seed", [1, 2])
    def test_the_runs_match_the_constrained_engine_with_the_existing_runs_fixed(self, seed: int) -> None:
        """A straight and a curved limit on three factors: the D-criterion of generate_design with fixed_runs."""
        factors = [*FACTORS, Factor(name="P", low=1, high=3)]
        limits = [LIMIT, "T*P <= 400"]
        region = DesignRegion(factors, limits)
        cube = pd.DataFrame(
            [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], columns=["T", "D", "P"], dtype=float
        )
        inside = cube[region.feasible(cube.to_numpy())].reset_index(drop=True)
        result = augment_design(
            inside,
            "add_runs_optimal",
            n_additional_runs=8,
            target_model="quadratic",
            region=region.to_dict(),
            random_state=seed,
        )
        made = generate_design(
            factors,
            "d_optimal",
            budget=len(inside) + 8,
            constraints=limits,
            fixed_runs=inside,
            model_type="quadratic",
            random_state=seed,
        )

        def log_det(runs: pd.DataFrame) -> float:
            x = dmatrix("(T + D + P) ** 2 + I(T ** 2) + I(D ** 2) + I(P ** 2)", runs, return_type="dataframe")
            return float(np.linalg.slogdet(x.T @ x)[1])

        augmented = pd.DataFrame(result["augmented_design"])
        assert region.feasible(augmented.to_numpy()).all()
        assert log_det(augmented) == pytest.approx(log_det(made.design[["T", "D", "P"]].astype(float)), abs=1e-6)

    def test_a_model_the_allowed_settings_cannot_support_is_refused(self) -> None:
        """Two limits hold T and D on a line, T = 75 + 1.25*D, where no setting separates them."""
        region = DesignRegion(FACTORS, ["T - 1.25*D <= 75", "T - 1.25*D >= 75"])
        with pytest.raises(ValueError, match="The settings the region allows cannot support the 'interactions' model"):
            augment_design(CUBE.iloc[[0]], "add_runs_optimal", n_additional_runs=4, region=region)


class TestCentreRuns:
    def test_the_centre_runs_move_to_the_region_when_the_centre_breaks_a_limit(self) -> None:
        region = DesignRegion(FACTORS, ["T + D >= 190"])  # the triangle (150, 40), (150, 60), (130, 60)
        corner = pd.DataFrame({"T": [1.0, 1.0, 0.2], "D": [0.0, 1.0, 1.0]})
        result = augment_design(corner, "add_center_points", n_additional_runs=2, region=region)
        new = _new_runs(result)
        assert len(new) == 2
        assert np.allclose(new[0], new[1])
        assert region.feasible(new).all()
        actual = [125 + 25 * new[0][0], 40 + 20 * new[0][1]]
        assert actual == pytest.approx([430 / 3, 160 / 3], abs=0.1), "the triangle's centroid"
        assert "the centre of the box breaks T + D >= 190" in result["explanation"]
        assert "fit a model with squared terms instead" in result["explanation"]

    def test_the_centre_runs_stay_at_the_centre_when_it_is_allowed(self) -> None:
        result = augment_design(
            CUBE.iloc[:3], "add_center_points", n_additional_runs=3, region=DesignRegion(FACTORS, [LIMIT])
        )
        assert (_new_runs(result) == 0).all()
        assert result["existing_runs_outside_region"] == []

    def test_a_region_too_thin_to_find_its_centre_in_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        region = DesignRegion(FACTORS, ["T + D >= 190"])
        monkeypatch.setattr(DesignRegion, "centroid", lambda _self: None)
        corner = pd.DataFrame({"T": [1.0, 1.0, 0.5], "D": [0.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match="No setting of the region was found to place the centre runs at"):
            augment_design(corner, "add_center_points", region=region)


class TestRunsTheLimitsRuleOut:
    def test_axial_runs_sit_on_the_faces_and_one_that_breaks_a_limit_is_named(self) -> None:
        region = DesignRegion(FACTORS, [LIMIT])
        with pytest.raises(ValueError, match="add_axial_points would add 2 run") as refused:
            augment_design(CUBE.iloc[:3], "add_axial_points", region=region)
        assert "(T = 150, D = 40) breaks 3*T + 5*D <= 600" in str(refused.value)
        assert "(T = 125, D = 60) breaks 3*T + 5*D <= 600" in str(refused.value)
        assert "Use augmentation_type='add_runs_optimal'" in str(refused.value)
        roomy = augment_design(CUBE.iloc[:3], "add_axial_points", region=DesignRegion(FACTORS, ["3*T + 5*D <= 700"]))
        assert roomy["alpha"] == 1.0
        assert "The axial runs sit on the faces of the cube: the region's ranges are limits." in roomy["explanation"]

    def test_an_axial_distance_past_the_region_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"alpha = 1.\d+ puts axial runs past the region's ranges"):
            augment_design(CUBE, "upgrade_to_rsm", alpha="rotatable", region=DesignRegion(FACTORS))

    def test_a_foldover_run_that_breaks_a_limit_is_refused(self) -> None:
        half = pd.DataFrame({"T": [-1.0, 1.0], "D": [-1.0, -1.0]})
        with pytest.raises(ValueError, match=r"foldover would add 1 run\(s\) outside the region: \(T = 150, D = 60\)"):
            augment_design(half, "foldover", region=DesignRegion(FACTORS, [LIMIT]))
        folded = augment_design(half, "foldover", region=DesignRegion(FACTORS, ["3*T + 5*D <= 750"]))
        assert len(folded["new_runs"]) == 2

    def test_copies_of_a_run_outside_the_region_are_refused(self) -> None:
        with (
            pytest.warns(UserWarning, match="lie outside the region"),
            pytest.raises(ValueError, match=r"replicate would add 1 run\(s\) outside the region"),
        ):
            augment_design(CUBE, "replicate", region=DesignRegion(FACTORS, [LIMIT]))

    def test_the_upgraded_design_s_centre_runs_are_checked_too(self) -> None:
        """Three axial runs and the five centre runs fall in the corner the limit cuts off; each setting once."""
        corner = pd.DataFrame({"T": [1.0, 1.0, 0.5], "D": [0.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match=r"upgrade_to_rsm would add 8 run\(s\) outside the region") as refused:
            augment_design(corner, "upgrade_to_rsm", region=DesignRegion(FACTORS, ["T + D >= 190"]))
        assert str(refused.value).count("breaks T + D >= 190") == 4
        assert "(T = 125, D = 40) breaks" in str(refused.value), "the centre"

    def test_many_settings_are_listed_five_at_a_time(self) -> None:
        """Eight runs at T of 0.25 to 1 (coded), all inside T >= 125; their eight mirror images all outside."""
        runs = pd.DataFrame([[t, d] for t in (0.25, 0.5, 0.75, 1.0) for d in (-1.0, 1.0)], columns=["T", "D"])
        with pytest.raises(ValueError, match=r"foldover would add 8 run\(s\).*; and 3 more setting\(s\)\."):
            augment_design(runs, "foldover", region=DesignRegion(FACTORS, ["T >= 125"]))

    def test_blocks_add_no_runs_to_check(self) -> None:
        twice = pd.concat([CUBE.iloc[:3], CUBE.iloc[:3]], ignore_index=True)
        result = augment_design(twice, "add_blocks", n_additional_runs=2, region=DesignRegion(FACTORS, [LIMIT]))
        assert result["new_runs"] == []
        assert result["existing_runs_outside_region"] == []


class TestTheRegionIsChecked:
    @pytest.mark.parametrize(
        ("region", "message"),
        [
            (
                DesignRegion([Factor(name=n, type="mixture", low=0, high=1) for n in ("T", "D")]),
                "this region holds mixture components",
            ),
            (
                DesignRegion([*FACTORS, Factor(name="K", type="categorical", levels=["A", "B"])]),
                r"the region's \['K'\] are categorical",
            ),
            (DesignRegion(FACTORS[:1]), r"it lacks \['D'\]\."),
            (DesignRegion([*FACTORS, Factor(name="P", low=1, high=3)]), r"it also holds \['P'\]\."),
            (DesignRegion(FACTORS, ["T >= 160"]), "The region's limits leave no setting to run: T >= 160."),
        ],
    )
    def test_a_region_that_does_not_fit_the_design_is_refused(self, region: DesignRegion, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            augment_design(CUBE, "add_center_points", region=region)

    def test_the_region_s_factors_may_come_in_any_order(self) -> None:
        region = DesignRegion(FACTORS[::-1], [LIMIT])
        result = augment_design(CUBE.iloc[:3], "add_runs_optimal", n_additional_runs=3, region=region)
        assert DesignRegion(FACTORS, [LIMIT]).feasible(_new_runs(result)).all()

    def test_without_a_region_nothing_changes(self) -> None:
        result = augment_design(CUBE, "add_center_points", n_additional_runs=2)
        assert "existing_runs_outside_region" not in result
        assert (_new_runs(result) == 0).all()


class TestTool:
    def test_the_tool_takes_the_region_as_generate_design_records_it(self) -> None:
        made = generate_design(FACTORS, "d_optimal", budget=6, constraints=[LIMIT], model_type="interactions")
        existing = made.design[["T", "D"]].astype(float).to_dict(orient="records")
        result = execute_tool_call(
            "augment_design",
            {
                "existing_design": existing,
                "augmentation_type": "add_runs_optimal",
                "n_additional_runs": 4,
                "target_model": "quadratic",
                "region": made.metadata["region"],
            },
        )
        assert "error" not in result
        assert DesignRegion(FACTORS, [LIMIT]).feasible(_new_runs(result)).all()
        assert result["existing_runs_outside_region"] == []

    def test_the_tool_reports_a_run_the_limits_rule_out(self) -> None:
        result = execute_tool_call(
            "augment_design",
            {
                "existing_design": CUBE.iloc[:3].to_dict(orient="records"),
                "augmentation_type": "add_axial_points",
                "region": DesignRegion(FACTORS, [LIMIT]).to_dict(),
            },
        )
        assert "breaks 3*T + 5*D <= 600" in result["error"]
