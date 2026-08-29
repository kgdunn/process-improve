"""Tests for the registered Plotly base themes."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from process_improve.multivariate.methods import PCA, MCUVScaler
from process_improve.multivariate.plots import score_plot, spe_plot
from process_improve.visualization.themes import (
    DEFAULT_THEME,
    THEME_NAMES,
    register_themes,
    set_theme,
)


@pytest.fixture
def fitted_pca() -> PCA:
    """Return a small fitted PCA model on synthetic data."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(40, 6)), columns=[f"v{i}" for i in range(6)])
    return PCA(n_components=2).fit(MCUVScaler().fit_transform(x))


def test_all_themes_registered() -> None:
    """Every theme name resolves to a registered Plotly template."""
    register_themes()
    for name in THEME_NAMES:
        assert name in pio.templates
        assert isinstance(pio.templates[name], go.layout.Template)


@pytest.mark.slow
def test_import_does_not_change_global_default() -> None:
    """Importing the package registers themes but leaves the global default alone.

    Run in a subprocess so the check is immune to other tests (or this
    module's own imports) having already touched ``pio.templates.default``.
    """
    code = (
        "import plotly.io as pio;"
        "before = pio.templates.default;"
        "import process_improve;"  # triggers the visualization package import
        "import process_improve.visualization;"
        "assert pio.templates.default == before, ("
        "    f'import changed default: {before!r} -> {pio.templates.default!r}');"
        # registration is still additive and must have happened
        "assert 'pi_journal' in pio.templates"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_set_theme_changes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_theme switches the global default; the pre-test value is restored.

    Restoring the value that was in effect *before* this test (rather than
    the package's DEFAULT_THEME) keeps ``pio.templates.default`` untouched
    for every test that runs afterwards in the same worker.
    """
    monkeypatch.setattr(pio.templates, "default", pio.templates.default)
    for name in THEME_NAMES:
        set_theme(name)
        assert pio.templates.default == name


def test_set_theme_rejects_unknown() -> None:
    """An unknown theme name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown theme"):
        set_theme("not_a_theme")


def test_theme_plot_backgrounds_are_pinned() -> None:
    """Pin every theme's plot-area background colour.

    The colours are not all distinct: pi_tufte and pi_journal deliberately
    share a white plot area, so this pins all four values rather than
    claiming uniqueness.
    """
    register_themes()
    backgrounds = {name: pio.templates[name].layout.plot_bgcolor for name in THEME_NAMES}
    assert backgrounds == {
        "pi_tufte": "#FFFFFF",
        "pi_economist": "#D9E9F1",
        "pi_journal": "#FFFFFF",
        "pi_brand": "#FAFAFA",
    }


def test_score_plot_uses_default_theme(fitted_pca: PCA) -> None:
    """A score plot carries the default theme by default."""
    fig = score_plot(fitted_pca)
    assert fig.layout.template.layout.plot_bgcolor == pio.templates[DEFAULT_THEME].layout.plot_bgcolor


def test_plot_template_is_overridable(fitted_pca: PCA) -> None:
    """The `template` setting overrides the default per call."""
    fig = spe_plot(fitted_pca, settings={"template": "pi_economist"})
    assert fig.layout.template.layout.plot_bgcolor == "#D9E9F1"
