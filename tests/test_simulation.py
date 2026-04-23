"""Tests for the fake-data simulator (process_improve.simulation)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from process_improve.simulation.model import materialize_model
from process_improve.simulation.tools import (
    create_simulator,
    get_simulation_tool_specs,
    reveal_simulator,
    simulate_process,
)
from process_improve.tool_spec import execute_tool_call, get_tool_specs

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _default_factors() -> list[dict]:
    return [
        {"name": "flow", "low": 100.0, "high": 300.0, "units": "L/min"},
        {"name": "pH", "low": 7.0, "high": 11.0},
        {"name": "surfactant", "low": 10.0, "high": 80.0, "units": "ppm"},
    ]


def _default_outputs() -> list[dict]:
    return [
        {"name": "recovery", "units": "%", "direction": "maximize"},
        {"name": "grade", "units": "%", "direction": "maximize"},
    ]


def _mid_settings() -> dict[str, float]:
    return {"flow": 200.0, "pH": 9.0, "surfactant": 45.0}


def _make_sim(seed: int = 42, **overrides) -> dict:
    payload = {
        "process_description": "Ni flotation vessel",
        "factors": _default_factors(),
        "outputs": _default_outputs(),
        "noise_level": "medium",
        "time_drift": False,
        "seed": seed,
    }
    payload.update(overrides)
    return create_simulator(**payload)


# ---------------------------------------------------------------------------
# create_simulator: validation + shape
# ---------------------------------------------------------------------------


class TestCreateSimulator:
    def test_returns_sim_id_public_and_private(self):
        sim = _make_sim()
        assert isinstance(sim["sim_id"], str)
        assert len(sim["sim_id"]) > 10
        assert "public" in sim
        assert "_private" in sim

    def test_public_does_not_leak_seed_or_coefficients(self):
        sim = _make_sim()
        pub = sim["public"]
        # The public summary is what the LLM sees after the host strips
        # _private. It must not contain the seed or any coefficients.
        flat = str(pub)
        assert "seed" not in pub
        assert "intercept" not in flat
        assert "coefficient" not in flat

    def test_private_carries_everything_needed_to_reproduce(self):
        sim = _make_sim(seed=777)
        priv = sim["_private"]
        assert priv["seed"] == 777
        assert [f["name"] for f in priv["factors"]] == ["flow", "pH", "surfactant"]
        assert [o["name"] for o in priv["outputs"]] == ["recovery", "grade"]
        assert priv["noise_level"] == "medium"
        assert priv["time_drift"] is False

    def test_duplicate_factor_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate factor"):
            create_simulator(
                process_description="x",
                factors=[
                    {"name": "A", "low": 0.0, "high": 1.0},
                    {"name": "A", "low": 0.0, "high": 2.0},
                ],
                outputs=_default_outputs(),
            )

    def test_low_must_be_less_than_high(self):
        with pytest.raises(ValueError, match=r"low .* must be"):
            create_simulator(
                process_description="x",
                factors=[{"name": "A", "low": 5.0, "high": 5.0}],
                outputs=_default_outputs(),
            )

    def test_invalid_noise_level_rejected(self):
        with pytest.raises(ValueError, match="noise_level"):
            _make_sim(noise_level="enormous")

    def test_empty_outputs_rejected(self):
        with pytest.raises(ValueError, match="outputs"):
            create_simulator(
                process_description="x",
                factors=_default_factors(),
                outputs=[],
            )


# ---------------------------------------------------------------------------
# simulate_process: noise semantics, clipping, drift
# ---------------------------------------------------------------------------


class TestSimulateProcess:
    def test_requires_simulator_state(self):
        result = simulate_process(sim_id="abc", settings=_mid_settings())
        assert result["error"] == "simulator_state_missing"

    def test_returns_one_value_per_output(self):
        sim = _make_sim()
        result = simulate_process(
            sim_id=sim["sim_id"],
            settings=_mid_settings(),
            simulator_state=sim["_private"],
        )
        assert set(result["outputs"]) == {"recovery", "grade"}
        for v in result["outputs"].values():
            assert isinstance(v, float)

    def test_fresh_noise_each_call(self):
        sim = _make_sim()
        calls = [
            simulate_process(
                sim_id=sim["sim_id"],
                settings=_mid_settings(),
                simulator_state=sim["_private"],
            )
            for _ in range(20)
        ]
        recovery = [c["outputs"]["recovery"] for c in calls]
        # Variance must be non-trivial: fresh-noise policy, not sticky.
        assert float(np.std(recovery)) > 1e-6

    def test_mean_converges_to_deterministic_surface(self):
        """Many noisy draws at the same point should average to the noiseless surface."""
        sim = _make_sim(seed=123, noise_level="medium")
        priv = sim["_private"]
        samples = 400
        recovery_sum = 0.0
        for _ in range(samples):
            r = simulate_process(
                sim_id=sim["sim_id"],
                settings=_mid_settings(),
                simulator_state=priv,
            )
            recovery_sum += r["outputs"]["recovery"]
        empirical_mean = recovery_sum / samples

        # The revealed model lets us compute the noiseless prediction.
        model = materialize_model(priv)
        # At mid-range, all coded values are 0 ==> y = intercept.
        expected = model["per_output"]["recovery"]["intercept"]
        sigma = model["per_output"]["recovery"]["noise_sigma"]
        # 99.9 % band of the sample mean is ~3.3 * sigma / sqrt(n).
        tol = 3.5 * sigma / np.sqrt(samples)
        assert abs(empirical_mean - expected) < max(tol, 0.05)

    def test_out_of_range_settings_clipped(self):
        sim = _make_sim()
        result = simulate_process(
            sim_id=sim["sim_id"],
            settings={"flow": 9999.0, "pH": 9.0, "surfactant": 45.0},
            simulator_state=sim["_private"],
        )
        assert result["settings"]["flow"] == 300.0
        assert any("clipped" in w for w in result["warnings"])

    def test_missing_factor_uses_midrange(self):
        sim = _make_sim()
        result = simulate_process(
            sim_id=sim["sim_id"],
            settings={"pH": 9.0, "surfactant": 45.0},  # no flow
            simulator_state=sim["_private"],
        )
        assert result["settings"]["flow"] == 200.0
        assert any("mid-range" in w for w in result["warnings"])

    def test_time_drift_off_ignores_offset(self):
        sim = _make_sim(time_drift=False, noise_level="low")
        priv = sim["_private"]

        def mean_recovery_at(t: float, n: int = 200) -> float:
            return float(
                np.mean(
                    [
                        simulate_process(
                            sim_id=sim["sim_id"],
                            settings=_mid_settings(),
                            simulator_state=priv,
                            timestamp_offset_days=t,
                        )["outputs"]["recovery"]
                        for _ in range(n)
                    ]
                )
            )

        m0 = mean_recovery_at(0.0)
        m30 = mean_recovery_at(30.0)
        # With noise_level=low and n=200, |mean diff| << 1 % of intercept.
        assert abs(m0 - m30) < 1.0

    def test_time_drift_on_shifts_mean(self):
        # Find a seed whose drift rate is non-negligible; most seeds qualify,
        # but the draw comes from a N(0, 1%*|intercept|) prior so a few are
        # near zero. Search deterministically to keep the test stable.
        for seed in range(1, 20):
            sim = _make_sim(seed=seed, time_drift=True, noise_level="low")
            model = materialize_model(sim["_private"])
            drift = model["per_output"]["recovery"]["drift_rate_per_day"]
            if abs(drift) >= 0.05:
                break
        else:
            pytest.skip("No seed produced a meaningful drift rate.")

        priv = sim["_private"]
        sigma = model["per_output"]["recovery"]["noise_sigma"]
        n = 400
        m0 = float(
            np.mean(
                [
                    simulate_process(
                        sim_id=sim["sim_id"],
                        settings=_mid_settings(),
                        simulator_state=priv,
                        timestamp_offset_days=0.0,
                    )["outputs"]["recovery"]
                    for _ in range(n)
                ]
            )
        )
        m30 = float(
            np.mean(
                [
                    simulate_process(
                        sim_id=sim["sim_id"],
                        settings=_mid_settings(),
                        simulator_state=priv,
                        timestamp_offset_days=30.0,
                    )["outputs"]["recovery"]
                    for _ in range(n)
                ]
            )
        )
        expected_shift = drift * 30.0
        tol = 5.0 * sigma / np.sqrt(n)
        assert abs((m30 - m0) - expected_shift) < max(tol, 0.2)


# ---------------------------------------------------------------------------
# reveal_simulator: confirmation gating
# ---------------------------------------------------------------------------


class TestRevealSimulator:
    def test_pending_when_not_confirmed(self):
        sim = _make_sim()
        out = reveal_simulator(sim_id=sim["sim_id"], simulator_state=sim["_private"])
        assert out["status"] == "confirmation_needed"
        assert "model" not in out

    def test_returns_full_model_when_confirmed(self):
        sim = _make_sim(seed=9)
        out = reveal_simulator(
            sim_id=sim["sim_id"],
            simulator_state=sim["_private"],
            confirmed=True,
        )
        assert out["status"] == "revealed"
        model = out["model"]
        assert "per_output" in model
        assert "recovery" in model["per_output"]
        assert "intercept" in model["per_output"]["recovery"]

    def test_same_seed_reveals_identical_model(self):
        a = _make_sim(seed=55)
        b = _make_sim(seed=55)
        ra = reveal_simulator(
            sim_id=a["sim_id"],
            simulator_state=a["_private"],
            confirmed=True,
        )
        rb = reveal_simulator(
            sim_id=b["sim_id"],
            simulator_state=b["_private"],
            confirmed=True,
        )
        assert ra["model"] == rb["model"]

    def test_different_seeds_different_model(self):
        a = _make_sim(seed=55)
        b = _make_sim(seed=56)
        ma = reveal_simulator(
            sim_id=a["sim_id"],
            simulator_state=a["_private"],
            confirmed=True,
        )["model"]
        mb = reveal_simulator(
            sim_id=b["sim_id"],
            simulator_state=b["_private"],
            confirmed=True,
        )["model"]
        assert ma != mb


# ---------------------------------------------------------------------------
# Structural hints
# ---------------------------------------------------------------------------


class TestStructuralHints:
    def test_negative_interaction_hint_is_honoured(self):
        """At least 8/10 seeds should produce a negative pH*surfactant coef."""
        negatives = 0
        for seed in range(10):
            sim = _make_sim(
                seed=seed,
                structural_hints=["negative interaction between pH and surfactant"],
            )
            model = materialize_model(sim["_private"])
            for out_coefs in model["per_output"].values():
                for inter in out_coefs["interactions"]:
                    if sorted(inter["factors"]) == sorted(["pH", "surfactant"]):
                        if inter["coefficient"] < 0:
                            negatives += 1
                        break
        assert negatives >= 8, f"Only {negatives}/10 interactions were negative"

    def test_positive_main_effect_hint(self):
        sim = _make_sim(
            seed=1,
            structural_hints=["positive effect of flow on recovery"],
        )
        model = materialize_model(sim["_private"])
        coef = model["per_output"]["recovery"]["main"]["flow"]
        assert coef > 0

    def test_quadratic_hint_creates_quadratic_term(self):
        sim = _make_sim(
            seed=2,
            structural_hints=["flow has a quadratic effect on recovery"],
        )
        model = materialize_model(sim["_private"])
        assert "flow" in model["per_output"]["recovery"]["quadratic"]
        assert abs(model["per_output"]["recovery"]["quadratic"]["flow"]) >= 1.0


# ---------------------------------------------------------------------------
# Determinism across processes
# ---------------------------------------------------------------------------


def _materialize_in_child(private_state: dict) -> dict:
    """Pickled worker target for the cross-process determinism test."""
    return materialize_model(private_state)


class TestCrossProcessDeterminism:
    def test_materialize_model_is_bit_identical_in_subprocess(self):
        sim = _make_sim(seed=101)
        priv = sim["_private"]
        local = materialize_model(priv)
        with ProcessPoolExecutor(max_workers=1) as pool:
            remote = pool.submit(_materialize_in_child, priv).result()
        # Dict comparison: every intercept, every coefficient, equal.
        assert local == remote


# ---------------------------------------------------------------------------
# Tool registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_tools_registered(self):
        names = {s["name"] for s in get_tool_specs()}
        assert {"create_simulator", "simulate_process", "reveal_simulator"} <= names

    def test_simulation_helper_lists_all_three(self):
        specs = get_simulation_tool_specs()
        assert {s["name"] for s in specs} == {
            "create_simulator",
            "simulate_process",
            "reveal_simulator",
        }
        assert len(specs) == 3

    def test_dispatch_via_execute_tool_call(self):
        out = execute_tool_call(
            "create_simulator",
            {
                "process_description": "toy process",
                "factors": [{"name": "A", "low": 0.0, "high": 1.0}],
                "outputs": [{"name": "y"}],
                "seed": 1,
            },
        )
        assert "sim_id" in out
        assert "public" in out
        assert "_private" in out

    def test_simulator_state_not_in_public_schema(self):
        """The LLM must not be able to forge state: simulator_state is hidden."""
        specs = {s["name"]: s for s in get_simulation_tool_specs()}
        for name in ("simulate_process", "reveal_simulator"):
            props = specs[name]["input_schema"].get("properties", {})
            assert "simulator_state" not in props
        # ``confirmed`` is also server-injected.
        assert "confirmed" not in specs["reveal_simulator"]["input_schema"].get(
            "properties", {}
        )
