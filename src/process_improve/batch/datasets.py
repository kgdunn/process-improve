# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Loader functions for the batch datasets bundled with, or hosted for, the package.

Each trajectory loader returns the standard batch-data dictionary used
throughout :mod:`process_improve.batch`: keys are batch identifiers, values are
per-batch dataframes with identical, all-numeric columns (one column per tag).
See :mod:`process_improve.batch.data_input` for the format definitions and
converters to the melted and wide representations.

Three datasets are bundled (:func:`load_nylon`, :func:`load_dryer`,
:func:`load_batch_fake_data`). Three larger case-study datasets are hosted on
`openmv.net <https://openmv.net>`_ and downloaded on demand
(:func:`load_dupont`, :func:`load_fmc`, :func:`load_sbr`); the download is
bounded by ``settings.dataset_fetch_timeout`` and every failure surfaces as a
``RuntimeError`` naming the URL (see :mod:`process_improve._remote_data`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.utils import Bunch

from process_improve._remote_data import read_remote_csv, read_remote_excel

from .data_input import melted_to_dict

if TYPE_CHECKING:
    from collections.abc import Hashable

_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "batch"

DUPONT_URL = "https://openmv.net/file/polymerization.csv"
"""Hosted copy of the DuPont batch polymerization data (:func:`load_dupont`)."""

FMC_URL = "https://openmv.net/file/batch-dryer.xlsx"
"""Hosted copy of the aligned, four-block FMC batch dryer data (:func:`load_fmc`)."""

SBR_URL = "https://openmv.net/file/sbr-batch-reactor.xlsx"
"""Hosted copy of the simulated SBR batch reactor data (:func:`load_sbr`)."""


def _melted_to_batches(
    melted: pd.DataFrame, batch_id_col: str, drop_columns: list[str] | None = None
) -> dict[Hashable, pd.DataFrame]:
    """Split a melted table into the batch-data dictionary.

    The ``batch_id_col`` column is consumed as the dictionary key and dropped
    from each per-batch dataframe (the key carries that information), along
    with any extra ``drop_columns``. Row indexes are reset per batch, so each
    batch counts its own samples from zero.
    """
    batches = melted_to_dict(melted, batch_id_col=batch_id_col)
    to_drop = [batch_id_col, *(drop_columns or [])]
    return {batch_id: batch.drop(columns=to_drop).reset_index(drop=True) for batch_id, batch in batches.items()}


def _load_melted_csv(
    filename: str, batch_id_col: str, drop_columns: list[str] | None = None
) -> dict[Hashable, pd.DataFrame]:
    """Read a bundled melted CSV and return it as a batch-data dictionary."""
    return _melted_to_batches(pd.read_csv(_DATASETS_DIR / filename), batch_id_col, drop_columns)


def _per_batch_table(sheet: pd.DataFrame, batch_ids: list[Hashable]) -> pd.DataFrame:
    """Index a one-row-per-batch sheet by ``batch_id`` in the trajectory order."""
    return sheet.set_index("batch_id").reindex(batch_ids)


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


def load_dupont(*, url: str | None = None, timeout: float | None = None) -> dict[Hashable, pd.DataFrame]:
    """Return the DuPont industrial batch polymerization dataset (downloaded).

    The worked example of Nomikos and MacGregor (1995): 55 batches from an
    industrial batch polymerization reactor, each already aligned to 100 equal
    time intervals, with ten process measurements per interval. Values are
    scaled for confidentiality and there are no missing values. The ten tags,
    in file order, are ``TempR-1``, ``TempR-2``, ``TempR-3`` (reactor
    temperatures), ``Press-1`` (a pressure), ``Flow-1`` (a feed flow),
    ``TempH-1`` and ``TempC-1`` (heating- and cooling-medium temperatures),
    ``Press-2`` and ``Press-3`` (pressures) and ``Flow-2`` (a feed flow).

    The final quality of each batch is not part of the dataset. The paper
    reports that batches 40, 41, 42, 50, 51, 53, 54 and 55 had a quality
    measurement well outside the acceptable limit, that batches 38, 45, 46,
    49 and 52 were above or very close to it, and that batch 49 was barely
    acceptable. Batches 50 to 55 stand out in the score plot and batch 49 in
    the SPE.

    Parameters
    ----------
    url : str, optional
        Where to download from. Defaults to :data:`DUPONT_URL`; pass a mirror
        or a ``file://`` URL to read a local copy.
    timeout : float, optional
        Download budget in seconds. Defaults to ``settings.dataset_fetch_timeout``.

    Returns
    -------
    dict[Hashable, pd.DataFrame]
        Standard batch-data dictionary: batch identifiers 1 to 55, each a
        dataframe of 100 rows and the 10 numeric tag columns. The ``time``
        column of the hosted file is dropped, since it is identical in every
        batch.

    Raises
    ------
    RuntimeError
        When the download fails or times out.

    Source
    ------
    Nomikos, P. and MacGregor, J.F., "Multivariate SPC Charts for Monitoring
    Batch Processes", Technometrics, 37(1), 41-59, 1995. Hosted at
    https://openmv.net/info/polymerization.

    Examples
    --------
    >>> from process_improve.batch import load_dupont
    >>> batches = load_dupont()  # doctest: +SKIP
    >>> len(batches), batches[1].shape  # doctest: +SKIP
    (55, (100, 10))
    """
    melted = read_remote_csv(DUPONT_URL if url is None else url, timeout=timeout)
    return _melted_to_batches(melted, batch_id_col="batch_id", drop_columns=["time"])


def load_fmc(*, url: str | None = None, timeout: float | None = None) -> Bunch:
    """Return the aligned, four-block FMC batch dryer dataset (downloaded).

    An industrial batch drying step in the manufacture of an agricultural
    chemical: wet cake (solid plus embedded solvent) is charged, dried through
    three recipe phases (solvent collection, temperature ramp, cool-down), and
    the solvent is collected in a side tank. This is the multiblock case study
    of Garcia-Munoz et al. (2003), with the trajectories already aligned
    within each phase to 325 samples per batch. Compare :func:`load_dryer`,
    the raw, unaligned trajectories of the same process.

    The four blocks are:

    - ``X``: the batch trajectories, ten tags plus ``ClockTime``, the wall
      time at each aligned sample, which after alignment is itself a
      trajectory that carries the time-warping information.
    - ``Zchem``: eleven initial-condition chemistry measurements of the cake,
      ``Z1`` to ``Z11``.
    - ``Zop``: nine initial operating conditions (levels, temperatures, the
      durations of the recipe steps, the temperature slope, the cake weight).
    - ``Y``: eight final quality attributes, ``Y1`` to ``Y11`` (not all
      numbers are used) and ``SolventConc``.

    The data contain genuine missing values, kept as ``NaN``: 1410 cells in
    ``X``, 134 in ``Zchem`` and 21 in ``Y``. Thirteen batches have no
    chemistry measurements at all; the original study excluded them, and
    their identifiers are returned as ``missing_chemistry`` so the exclusion
    can be reproduced.

    Parameters
    ----------
    url : str, optional
        Where to download from. Defaults to :data:`FMC_URL`; pass a mirror or
        a ``file://`` URL to read a local copy. Reading the workbook needs
        ``openpyxl`` (the ``batch`` extra).
    timeout : float, optional
        Download budget in seconds. Defaults to ``settings.dataset_fetch_timeout``.

    Returns
    -------
    sklearn.utils.Bunch
        With fields ``X`` (standard batch-data dictionary of 59 batches, each
        325 rows by 11 columns), ``Y`` (59 x 8), ``Zop`` (59 x 9) and
        ``Zchem`` (59 x 11), the last three indexed by batch identifier in
        the same order as the keys of ``X``, plus ``batch_ids`` (the 59
        non-consecutive identifiers) and ``missing_chemistry`` (the 13
        identifiers without chemistry data).

    Raises
    ------
    RuntimeError
        When the download fails or times out.

    Source
    ------
    Garcia-Munoz, S., Kourti, T., MacGregor, J.F., Mateos, A.G. and Murphy,
    G., "Troubleshooting of an Industrial Batch Process Using Multivariate
    Methods", Industrial and Engineering Chemistry Research, 42, 3592-3601,
    2003. Hosted at https://openmv.net/info/batch-dryer.

    Examples
    --------
    >>> from process_improve.batch import load_fmc
    >>> fmc = load_fmc()  # doctest: +SKIP
    >>> len(fmc.X), fmc.Y.shape, fmc.Zop.shape, fmc.Zchem.shape  # doctest: +SKIP
    (59, (59, 8), (59, 9), (59, 11))
    """
    sheets = read_remote_excel(FMC_URL if url is None else url, timeout=timeout)
    batches = _melted_to_batches(sheets["X_batch"], batch_id_col="batch_id")
    batch_ids = list(batches)
    return Bunch(
        X=batches,
        Y=_per_batch_table(sheets["Y_quality"], batch_ids),
        Zop=_per_batch_table(sheets["Z_operations"], batch_ids),
        Zchem=_per_batch_table(sheets["Z_chemistry"], batch_ids),
        batch_ids=batch_ids,
        missing_chemistry=[15, 16, 17, 18, 33, 34, 35, 36, 37, 38, 39, 40, 63],
    )


def load_sbr(*, url: str | None = None, timeout: float | None = None) -> Bunch:
    """Return the simulated SBR batch reactor dataset (downloaded).

    Styrene-butadiene rubber (SBR) emulsion polymerization, simulated from a
    first-principles model for the batch monitoring work of Nomikos (1995):
    53 batches of 200 samples with nine trajectories, and five final quality
    attributes per batch. Because the data are simulated, the fault is
    known: batches 34 and 37 both received 30% more organic impurity in the
    butadiene feed, from the start of batch 37 and partway through batch 34.
    The two feed-flow trajectories are constant in the simulation.

    Parameters
    ----------
    url : str, optional
        Where to download from. Defaults to :data:`SBR_URL`; pass a mirror or
        a ``file://`` URL to read a local copy. Reading the workbook needs
        ``openpyxl`` (the ``batch`` extra).
    timeout : float, optional
        Download budget in seconds. Defaults to ``settings.dataset_fetch_timeout``.

    Returns
    -------
    sklearn.utils.Bunch
        With fields ``X`` (standard batch-data dictionary: batch identifiers
        1 to 53, each 200 rows by the 9 tags ``StyreneFlow``,
        ``ButadieneFlow``, ``FeedTemp``, ``ReactorTemp``, ``CoolingTemp``,
        ``JacketTemp``, ``LatexDensity``, ``Conversion`` and
        ``EnergyReleased``), ``Y`` (53 x 5: ``Composition``,
        ``ParticleSize``, ``Branching``, ``CrossLinking`` and
        ``Polydispersity``, indexed by batch identifier), ``trajectory_tags``
        (the six tags the original study modelled, without the feed tags) and
        ``fault_batches`` (``[34, 37]``).

    Raises
    ------
    RuntimeError
        When the download fails or times out.

    Source
    ------
    Nomikos, P., "Statistical process control of batch processes", PhD
    thesis, McMaster University, 1995. Hosted at
    https://openmv.net/info/sbr-batch-reactor.

    Examples
    --------
    >>> from process_improve.batch import load_sbr
    >>> sbr = load_sbr()  # doctest: +SKIP
    >>> len(sbr.X), sbr.X[1].shape, sbr.Y.shape  # doctest: +SKIP
    (53, (200, 9), (53, 5))
    """
    sheets = read_remote_excel(SBR_URL if url is None else url, timeout=timeout)
    batches = _melted_to_batches(sheets["X_batch"], batch_id_col="batch_id", drop_columns=["time"])
    return Bunch(
        X=batches,
        Y=_per_batch_table(sheets["Y_quality"], list(batches)),
        trajectory_tags=["ReactorTemp", "CoolingTemp", "JacketTemp", "LatexDensity", "Conversion", "EnergyReleased"],
        fault_batches=[34, 37],
    )
