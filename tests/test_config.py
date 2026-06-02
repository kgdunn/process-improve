"""Tests for :mod:`process_improve.config`.

Pin the contract documented in the module: every knob has a
documented default, is overrideable via the corresponding
``PROCESS_IMPROVE_*`` env var, can be re-read after ``reload()``,
and surfaces to ``tool_safety`` / ``mcp_server`` at call time
(not at import time). Tracks ENG-09 (#291) and ENG-27 (#309).
"""

from __future__ import annotations

import importlib

import pytest

from process_improve.config import DEFAULTS, ENV_VAR_NAMES, Settings, settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Drop the cache before each test so env-var changes are picked up."""
    settings.reload()
    yield
    settings.reload()


class TestDefaults:
    """Every knob has a documented default and is reachable on first access."""

    @pytest.mark.parametrize("name", list(DEFAULTS))
    def test_default_value_matches_registry(self, name: str) -> None:
        assert getattr(settings, name) == DEFAULTS[name]

    def test_as_dict_returns_every_knob(self) -> None:
        snapshot = settings.as_dict()
        assert set(snapshot) == set(DEFAULTS)
        for name, expected in DEFAULTS.items():
            assert snapshot[name] == expected


class TestEnvVarOverride:
    """Each knob reads its env var on first access."""

    def test_tool_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["tool_timeout"], "30")
        settings.reload()
        assert settings.tool_timeout == 30.0

    def test_max_cells_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["max_cells"], "2500000")
        settings.reload()
        assert settings.max_cells == 2_500_000

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1", True), ("true", True), ("yes", True), ("on", True),
         ("0", False), ("false", False), ("no", False), ("", False)],
    )
    def test_mcp_safe_mode_truthy_values(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["mcp_safe_mode"], raw)
        settings.reload()
        assert settings.mcp_safe_mode is expected

    def test_invalid_int_raises_on_first_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["max_cells"], "not-an-int")
        settings.reload()
        with pytest.raises(ValueError, match="not a valid integer"):
            _ = settings.max_cells


class TestCacheBehaviour:
    """First read pins the value; subsequent reads return the cache."""

    def test_env_change_after_first_read_does_not_take_effect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["tool_timeout"], "20")
        settings.reload()
        first = settings.tool_timeout
        monkeypatch.setenv(ENV_VAR_NAMES["tool_timeout"], "40")
        # No reload: the cache still holds the old value.
        assert settings.tool_timeout == first == 20.0

    def test_reload_picks_up_new_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_VAR_NAMES["tool_timeout"], "20")
        settings.reload()
        assert settings.tool_timeout == 20.0
        monkeypatch.setenv(ENV_VAR_NAMES["tool_timeout"], "40")
        settings.reload()
        assert settings.tool_timeout == 40.0


class TestCodeOverride:
    """Tests can override a knob from code without touching the environment."""

    def test_attribute_setter_writes_through_cache(self) -> None:
        settings.tool_timeout = 7.5
        assert settings.tool_timeout == 7.5

    def test_setter_coerces_to_declared_type(self) -> None:
        settings.tool_timeout = "12"  # type: ignore[assignment]
        assert settings.tool_timeout == 12.0
        settings.max_cells = "5000"  # type: ignore[assignment]
        assert settings.max_cells == 5_000

    def test_setter_observed_by_tool_safety(self) -> None:
        """A code-override surfaces through the legacy module-level shim."""
        tool_safety = importlib.import_module("process_improve.tool_safety")
        settings.tool_timeout = 99.0
        # The legacy ``DEFAULT_TIMEOUT_S`` name still resolves -- but it
        # now resolves to whatever ``settings.tool_timeout`` is right now,
        # not whatever it was at import time.
        assert tool_safety.DEFAULT_TIMEOUT_S == 99.0


class TestLegacyShim:
    """The five ``DEFAULT_*`` names that ``tool_safety`` used to export
    forward to the new settings, so existing imports keep working
    (ENG-22 deprecation candidate for v2.0).
    """

    @pytest.mark.parametrize(
        ("legacy", "knob"),
        [
            ("DEFAULT_TIMEOUT_S", "tool_timeout"),
            ("DEFAULT_MAX_CELLS", "max_cells"),
            ("DEFAULT_MAX_STRING", "max_string"),
            ("DEFAULT_MAX_DEPTH", "max_depth"),
            ("DEFAULT_MEMORY_MB", "max_memory_mb"),
        ],
    )
    def test_legacy_name_forwards_to_settings(
        self, legacy: str, knob: str
    ) -> None:
        tool_safety = importlib.import_module("process_improve.tool_safety")
        assert getattr(tool_safety, legacy) == getattr(settings, knob)

    def test_unknown_attribute_still_raises(self) -> None:
        tool_safety = importlib.import_module("process_improve.tool_safety")
        with pytest.raises(AttributeError, match="DEFAULT_NONEXISTENT"):
            _ = tool_safety.DEFAULT_NONEXISTENT


class TestSettingsIsolation:
    """A fresh ``Settings()`` is independent of the module-level singleton."""

    def test_independent_instance(self) -> None:
        fresh = Settings()
        settings.tool_timeout = 1.0
        # The new instance does not see the singleton's overrides.
        assert fresh.tool_timeout == DEFAULTS["tool_timeout"]
