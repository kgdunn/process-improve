"""Each British public name still works, warns that it is deprecated, and names the caller's line (#1232).

The package spells its names the American way. The former British names stay until 3.0, each as an
alias that gives the policy's ``DeprecationWarning`` (``docs/development/deprecation_policy.rst``)
and then does what its replacement does.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import plotting
from process_improve.chemistry import normalization_check
from process_improve.experiments import design_catalog, reference_catalogue
from process_improve.multivariate.methods import PLS, PLSDA, PRM, UncenteredDataWarning
from process_improve.univariate import least_favorable_means


def _caught(call: Callable[[], Any], pattern: str) -> tuple[Any, list[warnings.WarningMessage]]:
    """Return what ``call`` returns, and the deprecation warnings it gave whose message matches ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    found = [w for w in caught if issubclass(w.category, DeprecationWarning) and pattern in str(w.message)]
    assert found, [str(w.message) for w in caught]
    assert {w.filename for w in found} == {__file__}, "the warning names the caller's line"
    return result, found


def test_colours_per_batch_id_warns_and_gives_the_colors() -> None:
    """``colours_per_batch_id`` and its British keywords give what ``colors_per_batch_id`` gives."""
    expected = plotting.colors_per_batch_id([1, 2], {}, 2, use_default_color=True)
    found, _ = _caught(
        lambda: plotting.colours_per_batch_id([1, 2], {}, 2, use_default_colour=True), "colours_per_batch_id"
    )
    assert found == expected


def test_incolour_keyword_warns_and_converts() -> None:
    """``get_rgba_from_triplet(incolour=...)`` converts as ``incolor`` does."""
    found, _ = _caught(lambda: plotting.get_rgba_from_triplet(incolour=[0.5, 0.5, 0.5], as_string=True), "incolour")
    assert found == plotting.get_rgba_from_triplet([0.5, 0.5, 0.5], as_string=True) == "rgba(127,127,127,1.0)"
    with pytest.raises(TypeError, match="missing 1 required argument: 'incolor'"):
        plotting.get_rgba_from_triplet()


def test_colour_map_setting_warns_and_is_used() -> None:
    """``plot_multitags(settings={"colour_map": ...})`` draws with that map, as ``color_map`` does."""
    pytest.importorskip("plotly")
    batches = {name: pd.DataFrame({"temp": np.arange(4.0) + i}) for i, name in enumerate(["A", "B"])}

    def palette(n: int) -> list[tuple[float, float, float]]:
        return [(0.2, 0.4, 0.6)] * n

    fig, _ = _caught(lambda: plotting.plot_multitags(batches, settings={"colour_map": palette}), "'colour_map'")
    expected = plotting.plot_multitags(batches, settings={"color_map": palette})
    assert [t.line.color for t in fig.data] == [t.line.color for t in expected.data] == ["rgba(51,102,153,1.0)"] * 2


def test_settings_colour_map_attribute_warns_and_reads_color_map() -> None:
    """``MultiTagPlotSettings`` takes ``colour_map`` as ``color_map``, and its ``colour_map`` reads that map."""

    def palette(n: int) -> list[tuple[float, float, float]]:
        return [(0.2, 0.4, 0.6)] * n

    settings = plotting.MultiTagPlotSettings(colour_map=palette)
    assert settings.color_map is palette
    found, _ = _caught(lambda: settings.colour_map, "MultiTagPlotSettings.colour_map")
    assert found is palette


def test_animation_colour_assignment_keyword_warns_and_draws() -> None:
    """``generate_one_frame(animation_colour_assignment=...)`` draws the frame ``animation_color_assignment`` does."""
    subplots = pytest.importorskip("plotly.subplots")
    batches = {"A": pd.DataFrame({"temp": np.arange(4.0)})}
    lines = {"A": {"width": 2, "color": "red"}}
    expected = plotting.generate_one_frame(
        batches, ["temp"], subplots.make_subplots(rows=1, cols=1), 2, None, ["A"], lines
    )
    found, _ = _caught(
        lambda: plotting.generate_one_frame(
            batches, ["temp"], subplots.make_subplots(rows=1, cols=1), 2, None, ["A"], animation_colour_assignment=lines
        ),
        "animation_colour_assignment",
    )
    drawn = [(list(t.x), list(t.y), t.line.color) for t in found]
    assert drawn == [(list(t.x), list(t.y), t.line.color) for t in expected] == [([0, 1], [0.0, 1.0], "red")]
    with pytest.raises(TypeError, match="'animation_color_assignment'"):
        plotting.generate_one_frame(batches, ["temp"], subplots.make_subplots(rows=1, cols=1), 2, None, ["A"])


def test_normalisation_check_warns_and_checks() -> None:
    """``normalisation_check`` reports the totals ``normalization_check`` reports."""
    from process_improve.chemistry import normalisation_check

    chem = pd.DataFrame({"a": [1.0, 2.0, 1.0], "b": [1.0, 2.0, 9.0]})
    (totals, outside), _ = _caught(lambda: normalisation_check(chem), "normalisation_check")
    expected_totals, expected_outside = normalization_check(chem)
    pd.testing.assert_series_equal(totals, expected_totals)
    pd.testing.assert_series_equal(outside, expected_outside)


def test_normalise_family_warns_and_names_the_family() -> None:
    """``normalise_family`` names the family ``normalize_family`` names."""
    found, _ = _caught(lambda: design_catalog.normalise_family("CCD"), "normalise_family")
    assert found == design_catalog.normalize_family("CCD") == "ccd"


def test_least_favourable_means_warns_and_gives_the_means() -> None:
    """``least_favourable_means`` gives the means ``least_favorable_means`` gives."""
    from process_improve.univariate import least_favourable_means

    found, _ = _caught(lambda: least_favourable_means(75, 4), "least_favourable_means")
    assert found == least_favorable_means(75, 4) == [-37.5, 0.0, 0.0, 37.5]


def test_reproduce_reference_catalogue_warns_and_reproduces() -> None:
    """``reproduce_reference_catalogue`` gives the report ``reproduce_reference_catalog`` gives."""
    found, _ = _caught(
        lambda: reference_catalogue.reproduce_reference_catalogue("ullmann-doe-2012"), "reproduce_reference_catalogue"
    )
    pd.testing.assert_frame_equal(found, reference_catalogue.reproduce_reference_catalog("ullmann-doe-2012"))


@pytest.mark.parametrize("module", ["process_improve.multivariate", "process_improve.multivariate.methods"])
def test_uncentred_data_warning_is_the_same_class(module: str) -> None:
    """``UncentredDataWarning`` is ``UncenteredDataWarning``, so a filter or a check with either name matches."""
    import importlib

    package = importlib.import_module(module)
    former, _ = _caught(lambda: package.UncentredDataWarning, "UncentredDataWarning")
    assert former is UncenteredDataWarning
    with pytest.warns(former, match="un-centered"):
        PLS(n_components=1, scale=False).fit(pd.DataFrame({"x": [10.0, 11, 12, 14]}), pd.Series([20.0, 22, 23, 27]))
    with pytest.raises(AttributeError, match="no attribute 'NoSuchWarning'"):
        _ = package.NoSuchWarning


@pytest.mark.parametrize("estimator", [PLS, PLSDA, PRM])
def test_warn_on_uncentred_keyword_warns_and_is_taken(estimator: type) -> None:
    """``warn_on_uncentred=...`` is taken as ``warn_on_uncentered=...``, and survives ``clone`` under its new name."""
    from sklearn.base import clone

    model, _ = _caught(lambda: estimator(n_components=1, warn_on_uncentred=False), "warn_on_uncentred")
    assert model.get_params()["warn_on_uncentered"] is False
    assert "warn_on_uncentred" not in model.get_params()
    assert clone(model).get_params()["warn_on_uncentered"] is False
    with pytest.raises(TypeError, match="unexpected keyword argument 'colour'"):
        estimator(n_components=1, colour="red")


def test_a_fit_with_warn_on_uncentred_false_is_quiet() -> None:
    """The former keyword silences the un-centered check as the new one does."""
    x, y = pd.DataFrame({"x": [10.0, 11, 12, 14]}), pd.Series([20.0, 22, 23, 27])
    model, _ = _caught(lambda: PLS(n_components=1, scale=False, warn_on_uncentred=False), "warn_on_uncentred")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UncenteredDataWarning)
        model.fit(x, y)
