"""Tests for the fake-data simulator (process_improve.simulation)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from process_improve.simulation.context import simulator_host_context
from process_improve.simulation.model import materialize_model
from process_improve.simulation.tools import (
    CreateSimulatorInput,
    RevealSimulatorInput,
    SimulateProcessInput,
    create_simulator,
    get_simulation_tool_specs,
    reveal_simulator,
    simulate_process,
)
from process_improve.tool_safety import ToolInputInvalidError, safe_execute_tool_call
from process_improve.tool_spec import execute_tool_call, get_tool_specs

# ---------------------------------------------------------------------------
# Host-injection helpers
# ---------------------------------------------------------------------------
# ``simulator_state`` and ``confirmed`` are no longer kwargs (SEC-15): the host
# supplies them out of band through ``simulator_host_context``. These thin
# wrappers let the tests drive that side channel with the old call shape and
# build the pydantic input models on behalf of callers.


def _simulate_process(
    *,
    sim_id: str,
    settings: dict[str, float],
    timestamp_offset_days: float = 0.0,
    simulator_state: dict | None = None,
) -> dict:
    with simulator_host_context(simulator_state=simulator_state):
        return simulate_process(
            SimulateProcessInput(
                sim_id=sim_id,
                settings=settings,
                timestamp_offset_days=timestamp_offset_days,
            )
        )


def _reveal_simulator(
    *,
    sim_id: str,
    simulator_state: dict | None = None,
    confirmed: bool = False,
) -> dict:
    with simulator_host_context(simulator_state=simulator_state, confirmed=confirmed):
        return reveal_simulator(RevealSimulatorInput(sim_id=sim_id))


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
    return create_simulator(CreateSimulatorInput(**payload))


# ---------------------------------------------------------------------------
# create_simulator: validation + shape
# ---------------------------------------------------------------------------


class TestDrawInitialSeed:
    """SEC-28 (#277): seeds must carry at least 63 bits of entropy.

    The previous implementation truncated to 31 bits, which combined with
    observable simulator outputs allowed brute-forcing the hidden model. The
    fix uses ``secrets.randbits(63)`` for a JSON-int-safe 63-bit seed.
    """

    def test_seed_uses_at_least_63_bits(self):
        from process_improve.simulation.model import draw_initial_seed

        # Sample many seeds and assert the spread occupies the full 63-bit range.
        # 200 samples is plenty: P(all-bits-below-2**62) when sampling from a
        # uniform 63-bit space is (1/2)**200, i.e. statistically impossible.
        samples = [draw_initial_seed() for _ in range(200)]
        assert all(s >= 0 for s in samples)
        # At least one sample must exceed 2**31 to prove width > 31 bits.
        assert max(samples) > 2**31, (
            f"draw_initial_seed appears truncated to <=31 bits; max={max(samples)}"
        )
        # At least one sample must exceed 2**62 to prove width >= 63 bits.
        assert max(samples) > 2**62, (
            f"draw_initial_seed appears truncated to <=62 bits; max={max(samples)}"
        )


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
                CreateSimulatorInput(
                    process_description="x",
                    factors=[
                        {"name": "A", "low": 0.0, "high": 1.0},
                        {"name": "A", "low": 0.0, "high": 2.0},
                    ],
                    outputs=_default_outputs(),
                )
            )

    def test_low_must_be_less_than_high(self):
        with pytest.raises(ValueError, match=r"low .* must be"):
            create_simulator(
                CreateSimulatorInput(
                    process_description="x",
                    factors=[{"name": "A", "low": 5.0, "high": 5.0}],
                    outputs=_default_outputs(),
                )
            )

    def test_invalid_noise_level_rejected(self):
        # Pydantic ``Literal`` rejects the bad noise_level via ValidationError
        # (a ValueError subclass).
        with pytest.raises(ValueError, match="noise_level"):
            _make_sim(noise_level="enormous")

    def test_empty_outputs_rejected(self):
        # Pydantic ``min_length=1`` rejects an empty outputs list via
        # ValidationError (a ValueError subclass).
        with pytest.raises(ValueError, match="outputs"):
            create_simulator(
                CreateSimulatorInput(
                    process_description="x",
                    factors=_default_factors(),
                    outputs=[],
                )
            )


# ---------------------------------------------------------------------------
# simulate_process: noise semantics, clipping, drift
# ---------------------------------------------------------------------------


class TestSimulateProcess:
    def test_requires_simulator_state(self):
        result = _simulate_process(sim_id="abc", settings=_mid_settings())
        assert result["error"] == "simulator_state_missing"

    def test_returns_one_value_per_output(self):
        sim = _make_sim()
        result = _simulate_process(
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
            _simulate_process(
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
            r = _simulate_process(
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
        result = _simulate_process(
            sim_id=sim["sim_id"],
            settings={"flow": 9999.0, "pH": 9.0, "surfactant": 45.0},
            simulator_state=sim["_private"],
        )
        assert result["settings"]["flow"] == 300.0
        assert any("clipped" in w for w in result["warnings"])

    def test_missing_factor_uses_midrange(self):
        sim = _make_sim()
        result = _simulate_process(
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
                        _simulate_process(
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
                    _simulate_process(
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
                    _simulate_process(
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
        out = _reveal_simulator(sim_id=sim["sim_id"], simulator_state=sim["_private"])
        assert out["status"] == "confirmation_needed"
        assert "model" not in out

    def test_confirmed_but_missing_state_returns_error(self):
        """The reveal gate clears via context but no state was injected."""
        out = _reveal_simulator(sim_id="abc", simulator_state=None, confirmed=True)
        assert out.get("error") == "simulator_state_missing"
        assert "model" not in out

    def test_returns_full_model_when_confirmed(self):
        sim = _make_sim(seed=9)
        out = _reveal_simulator(
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
        ra = _reveal_simulator(
            sim_id=a["sim_id"],
            simulator_state=a["_private"],
            confirmed=True,
        )
        rb = _reveal_simulator(
            sim_id=b["sim_id"],
            simulator_state=b["_private"],
            confirmed=True,
        )
        assert ra["model"] == rb["model"]

    def test_different_seeds_different_model(self):
        a = _make_sim(seed=55)
        b = _make_sim(seed=56)
        ma = _reveal_simulator(
            sim_id=a["sim_id"],
            simulator_state=a["_private"],
            confirmed=True,
        )["model"]
        mb = _reveal_simulator(
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


# ---------------------------------------------------------------------------
# SEC-15: host-only params cannot be injected through the dispatch path
# ---------------------------------------------------------------------------


class TestKwargInjectionGate:
    """Forging ``simulator_state`` / pre-clearing ``confirmed`` must not work.

    A prompt-injected agent must not be able to smuggle either through the
    tool input and reach the function.
    """

    def test_execute_tool_call_rejects_injected_confirmed(self):
        # The pydantic ``extra="forbid"`` boundary rejects forged
        # ``confirmed`` / ``simulator_state`` kwargs before the function ever
        # runs. This is the structural closure of SEC-15: the reveal gate
        # cannot be smuggled.
        with pytest.raises(ToolInputInvalidError):
            execute_tool_call(
                "reveal_simulator",
                {"sim_id": "x", "confirmed": True, "simulator_state": {"factors": []}},
            )

    def test_execute_tool_call_rejects_injected_state_for_simulate(self):
        forged = _make_sim()["_private"]
        with pytest.raises(ToolInputInvalidError):
            execute_tool_call(
                "simulate_process",
                {"sim_id": "x", "settings": _mid_settings(), "simulator_state": forged},
            )

    def test_injected_state_rejected_even_when_host_confirms(self):
        # Even with a legitimate host context, a forged ``simulator_state``
        # in the tool input is rejected at the pydantic boundary.
        sim = _make_sim(seed=7)
        forged_sim = _make_sim(seed=999)
        with (
            simulator_host_context(simulator_state=sim["_private"], confirmed=True),
            pytest.raises(ToolInputInvalidError),
        ):
            execute_tool_call(
                "reveal_simulator",
                {
                    "sim_id": sim["sim_id"],
                    "simulator_state": forged_sim["_private"],
                    "confirmed": True,
                },
            )

    def test_safe_execute_tool_call_rejects_injected_kwargs(self):
        # The safe (untrusted-transport) path rejects unknown keys outright
        # rather than silently dropping them.
        with pytest.raises(ToolInputInvalidError):
            safe_execute_tool_call(
                "reveal_simulator",
                {"sim_id": "x", "confirmed": True},
                timeout=10,
            )

    def test_gate_still_fires_for_legitimate_host_calls(self):
        # The normal two-step gate is unaffected: first call asks for
        # confirmation, the host-confirmed second call reveals.
        sim = _make_sim()
        first = execute_tool_call("reveal_simulator", {"sim_id": sim["sim_id"]})
        assert first["status"] == "confirmation_needed"
        with simulator_host_context(simulator_state=sim["_private"], confirmed=True):
            second = execute_tool_call("reveal_simulator", {"sim_id": sim["sim_id"]})
        assert second["status"] == "revealed"
