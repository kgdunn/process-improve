"""Optimal designs on units with known covariates, each used once: samples, or times (#788 A7).

Goos and Jones (Section 9.2) choose 18 of 40 polypropylene samples, each with its EPDM, ethylene and
EVA content, and set four factors for each, D-optimally for the main effects. With 50 starts the
search chooses the same 18 samples; with 200 it finds a design a little better than the printed one.
Their trend-robust 2^3 (Table 9.8) is the case where every unit is used: the eight run times.
"""

from __future__ import annotations

import doctest

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor
from process_improve.experiments import designs_covariates as covariates_module
from process_improve.experiments.designs_covariates import CovariateOptions, covariate_design
from process_improve.experiments.designs_utils import categorical_effect_columns, sorted_level_ranks
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments

CODED = [Factor(name=n, low=-1, high=1) for n in ("x1", "x2", "x3")]
GASES = ["Activation gas 1", "Activation gas 2", "Etching gas"]
PROCESS = [
    Factor(name="flow", low=-1, high=1),
    Factor(name="power", low=-1, high=1),
    Factor(name="time", low=-1, high=1),
    Factor(name="gas", type="categorical", levels=GASES),
]


def catalogue(entries: list[dict], name: str) -> pd.DataFrame:
    [entry] = [e for e in entries if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


def coded_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """EPDM on 0.01 to 0.10 and ethylene on 0 to 0.15, as the book codes them; EVA as it is."""
    return pd.DataFrame(
        {"epdm": (frame["epdm"] - 0.055) / 0.045, "ethylene": (frame["ethylene"] - 0.075) / 0.075, "eva": frame["eva"]},
        index=frame.index,
    )


@pytest.fixture(scope="module")
def samples() -> pd.DataFrame:
    return coded_covariates(
        catalogue(reference_designs("goos-jones-2011"), "goos-jones-polypropylene-samples").set_index("sample")
    )


@pytest.fixture(scope="module")
def printed_log_det() -> float:
    """Return log|X'X| of the printed 18 runs, coded as the search codes them."""
    runs = catalogue(reference_experiments("goos-jones-2011"), "goos-jones-polypropylene")
    covariates = coded_covariates(runs)
    gas = categorical_effect_columns(sorted_level_ranks(GASES), 3)[[GASES.index(g) for g in runs["gas"]]]
    columns = [
        np.ones(len(runs)),
        (runs["flow"] - 1500) / 500,
        (runs["power"] - 1250) / 750,
        (runs["time"] - 8.5) / 6.5,
        *gas.T,
        covariates["epdm"],
        covariates["ethylene"],
        np.where(covariates["eva"] == "Yes", 1.0, -1.0),
    ]
    x = np.column_stack(columns)
    return float(np.linalg.slogdet(x.T @ x)[1])


# --------------------------------------------------------------------------- the sources
@pytest.mark.slow
@pytest.mark.dataset
def test_the_search_chooses_the_printed_polypropylene_samples(samples, printed_log_det):
    """With 50 starts: the 18 samples of Table 9.2, within 0.01% of the printed design's D-efficiency."""
    design, meta = covariate_design(samples, PROCESS, 18, CovariateOptions(n_starts=50), random_state=1)
    assert sorted(meta["units"]) == list(range(1, 19))
    relative = np.exp((meta["log_det_information"] - printed_log_det) / 9)
    assert relative > 0.9999
    assert list(design.columns) == ["epdm", "ethylene", "eva", "flow", "power", "time", "gas"]


@pytest.mark.slow
@pytest.mark.dataset
def test_more_starts_find_a_polypropylene_design_better_than_the_printed_one(samples, printed_log_det):
    """With 200 starts: other samples, and the printed design is then 99.95% D-efficient."""
    _, meta = covariate_design(samples, PROCESS, 18, CovariateOptions(n_starts=200), random_state=3)
    assert meta["log_det_information"] > printed_log_det
    assert round(float(np.exp((printed_log_det - meta["log_det_information"]) / 9)), 4) == 0.9995


@pytest.mark.dataset
def test_the_trend_robust_order_is_d_optimal_with_the_trend_in_the_model():
    """Table 9.8: with the time in the model, the best order makes every main effect orthogonal to it."""
    printed = catalogue(reference_designs("goos-jones-2011"), "goos-jones-trend-robust-main-effects")
    times = pd.DataFrame({"time": -1 + 2 * (printed["time"] - 1) / 7})
    x = np.column_stack([np.ones(8), times["time"], printed[["x1", "x2", "x3"]]])
    design, meta = covariate_design(times, CODED, 8, CovariateOptions(), random_state=1)
    assert meta["log_det_information"] == pytest.approx(np.linalg.slogdet(x.T @ x)[1], abs=1e-9)
    assert meta["log_det_information"] == pytest.approx(np.log(8**4 * float(times["time"] @ times["time"])))
    np.testing.assert_allclose(design[["x1", "x2", "x3"]].T @ design["time"], 0.0, atol=1e-12)


# --------------------------------------------------------------------------- what the search returns
def test_a_categorical_covariate_is_effect_coded_and_the_units_are_named():
    units = pd.DataFrame(
        {"size": np.linspace(-1, 1, 12), "supplier": ["a", "b", "c"] * 4}, index=[f"u{i}" for i in range(12)]
    )
    design, meta = covariate_design(units, CODED[:2], 8, CovariateOptions(), random_state=0)
    assert len(meta["units"]) == len(set(meta["units"])) == 8
    assert set(design.index) == set(meta["units"]) <= set(units.index)
    assert list(design.columns) == ["size", "supplier", "x1", "x2"]


@pytest.mark.parametrize("criterion", ["a_optimal", "i_optimal"])
def test_the_a_and_i_criteria_are_searched_too(criterion):
    units = pd.DataFrame({"t": np.linspace(-1, 1, 10)})
    _, meta = covariate_design(units, CODED, 8, CovariateOptions(criterion=criterion, n_starts=3), random_state=0)
    assert np.isfinite(meta["trace_criterion"])


def test_the_docstring_example_prints_what_it_shows():
    result = doctest.testmod(covariates_module, report=False)
    assert result.attempted > 0
    assert result.failed == 0


# --------------------------------------------------------------------------- what is refused
UNITS = pd.DataFrame({"t": np.linspace(-1, 1, 8)})


@pytest.mark.parametrize(
    ("units", "n_runs", "options", "message"),
    [
        (UNITS, 6, CovariateOptions(criterion="e_optimal"), "searched for on"),
        (UNITS, 9, CovariateOptions(), "Choose from 1 to 8"),
        (UNITS, 6, CovariateOptions(n_starts=0), "at least one start"),
        (UNITS, 4, CovariateOptions(), "cannot estimate the 5 terms"),
        (UNITS.rename(columns={"t": "x1"}), 6, CovariateOptions(), r"factors \['x1'\] are also covariates"),
    ],
)
def test_choices_the_search_cannot_make_are_refused(units, n_runs, options, message):
    with pytest.raises(ValueError, match=message):
        covariate_design(units, CODED, n_runs, options)
