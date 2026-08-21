# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Loader functions for the batch datasets bundled with the package.

Each loader returns the standard batch-data dictionary used throughout
:mod:`process_improve.batch`: keys are batch identifiers, values are per-batch
dataframes with identical, all-numeric columns (one column per tag). See
:mod:`process_improve.batch.data_input` for the format definitions and
converters to the melted and wide representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .data_input import melted_to_dict

if TYPE_CHECKING:
    from collections.abc import Hashable

_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "batch"


def _load_melted_csv(
    filename: str, batch_id_col: str, drop_columns: list[str] | None = None
) -> dict[Hashable, pd.DataFrame]:
    """Read a bundled melted CSV and return it as a batch-data dictionary.

    The ``batch_id_col`` column is consumed as the dictionary key and dropped
    from each per-batch dataframe (the key carries that information), along
    with any extra ``drop_columns``. Row indexes are reset per batch, so each
    batch counts its own samples from zero.
    """
    melted = pd.read_csv(_DATASETS_DIR / filename)
    batches = melted_to_dict(melted, batch_id_col=batch_id_col)
    to_drop = [batch_id_col, *(drop_columns or [])]
    return {batch_id: batch.drop(columns=to_drop).reset_index(drop=True) for batch_id, batch in batches.items()}


def load_nylon() -> dict[Hashable, pd.DataFrame]:
    """Return the nylon autoclave reactor batch dataset.

    Trajectory data from an industrial nylon polymerization autoclave,
    used widely in the batch analysis and monitoring literature. Variables
    ``Tag01`` to ``Tag10`` are temperatures, pressures, and flows recorded
    during each batch. Batch durations vary slightly (113 to 135 samples),
    so resample or align the batches to a common length before unfolding
    (see :func:`process_improve.batch.resample_to_reference`).

    Returns
    -------
    dict[Hashable, pd.DataFrame]
        Standard batch-data dictionary: 57 batches, each a dataframe of
        10 numeric tag columns.

    Source
    ------
    Kassidas, A., "Fault Detection and Diagnosis in Dynamic Multivariable
    Chemical Processes Using Speech Recognition Methods", PhD thesis,
    McMaster University, 1997. Also analyzed in Wold, Kettaneh-Wold,
    MacGregor and Dunn, "Batch Process Modeling and MSPC", Comprehensive
    Chemometrics, Elsevier, 2009.

    Examples
    --------
    >>> from process_improve.batch.datasets import load_nylon
    >>> batches = load_nylon()
    >>> len(batches)
    57
    """
    return _load_melted_csv("nylon.csv", batch_id_col="batch_id")


def load_dryer() -> dict[Hashable, pd.DataFrame]:
    """Return the batch dryer dataset.

    Trajectory data from an industrial batch drying process (a critical step
    in the manufacture of an agricultural chemical). Each batch records ten
    process tags plus ``ClockTime``, the wall-time sample counter:

    - ``CollectorTankLevel``: level of the solvent collector tank
    - ``DifferentialPressure``: differential pressure in the dryer
    - ``DryerPressure``: pressure in the dryer
    - ``AgitatorPower``: power to the agitator
    - ``AgitatorTorque``: torque resistance for the agitator
    - ``AgitatorSpeed``: agitator speed
    - ``JacketTemperatureSP``: set point for the jacket heating medium
    - ``JacketTemperature``: temperature of the jacket heating medium
    - ``DryerTemperatureSP``: set point for the temperature inside the dryer
    - ``DryerTemp``: temperature inside the dryer
    - ``ClockTime``: sample counter (samples assumed evenly spaced)

    The batches have varying durations, so this dataset is a realistic
    candidate for alignment (see :func:`process_improve.batch.batch_dtw` and
    :func:`process_improve.batch.resample_to_reference`).

    Returns
    -------
    dict[Hashable, pd.DataFrame]
        Standard batch-data dictionary: 71 batches, each a dataframe of
        11 numeric columns (10 tags plus ``ClockTime``).

    Source
    ------
    Garcia-Munoz, S., "Batch process improvement using latent variable
    methods", PhD thesis, McMaster University, 2004. Also analyzed in Wold,
    Kettaneh-Wold, MacGregor and Dunn, "Batch Process Modeling and MSPC",
    Comprehensive Chemometrics, Elsevier, 2009.

    Examples
    --------
    >>> from process_improve.batch.datasets import load_dryer
    >>> batches = load_dryer()
    >>> "DryerTemp" in next(iter(batches.values())).columns
    True
    """
    return _load_melted_csv("dryer.csv", batch_id_col="batch_id")


def load_batch_fake_data() -> dict[Hashable, pd.DataFrame]:
    """Return a small synthetic batch dataset.

    Simulated trajectory data for quick examples and tests: two temperature
    tags and one pressure tag per batch, plus ``UCI_minutes`` (minutes since
    the start of the batch). The wall-clock timestamp column in the raw CSV
    is dropped, so all returned columns are numeric.

    Returns
    -------
    dict[Hashable, pd.DataFrame]
        Standard batch-data dictionary of synthetic batches, each a dataframe
        with columns ``UCI_minutes``, ``Temp1``, ``Temp2``, and ``Pressure1``.

    Examples
    --------
    >>> from process_improve.batch.datasets import load_batch_fake_data
    >>> batches = load_batch_fake_data()
    >>> sorted(next(iter(batches.values())).columns)
    ['Pressure1', 'Temp1', 'Temp2', 'UCI_minutes']
    """
    return _load_melted_csv("batch-fake-data.csv", batch_id_col="Batch", drop_columns=["DateTime"])
