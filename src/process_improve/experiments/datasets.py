# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from process_improve.config import settings

_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "experiments"


def _read_remote_csv(url: str, timeout: float | None = None) -> pd.DataFrame:
    """Fetch a sample-dataset CSV from a (hard-coded, trusted) remote host.

    The URL is not user-supplied. The download is bounded by an explicit
    timeout (``settings.dataset_fetch_timeout``, default 30 s, overridable via
    ``PROCESS_IMPROVE_DATASET_FETCH_TIMEOUT``), so a black-holing host raises
    instead of hanging the caller indefinitely (#508). Network failures,
    timeouts, and parse failures are surfaced as a clear ``RuntimeError``
    (``TimeoutError`` and ``urllib.error.URLError`` are both ``OSError``
    subclasses) rather than a lower-level error, so callers get an actionable
    message. Note that the content is fetched over the network and is
    therefore trusted only as far as the remote host is.

    Parameters
    ----------
    url : str
        The https URL of the CSV file to fetch.
    timeout : float, optional
        Seconds to wait before giving up. Defaults to
        ``settings.dataset_fetch_timeout``.
    """
    if timeout is None:
        timeout = settings.dataset_fetch_timeout
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https URLs
            payload = response.read()
        return pd.read_csv(io.BytesIO(payload))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not download the sample dataset from {url!r}: {exc}. "
            "Check your network connection; this dataset is fetched from a remote host."
        ) from exc


def distillateflow() -> pd.DataFrame:
    """Return the flow rate of distillate from the top of a distillation column.

    These are actual data, taken 1 minute apart in time, of the flow rate leaving
    the top of a continuous distillation column (data are from a 31 day period
    in time). The data are fetched from the canonical hosted location on
    openmv.net rather than bundled with the package.

    Dimensions
    ----------
    A data frame containing 44640 observations of 1 variable.

    Source
    ------
    http://openmv.net/info/distillate-flow


    """
    return _read_remote_csv("https://openmv.net/file/distillate-flow.csv")


def pollutant() -> pd.DataFrame:
    """
    Return water treatment example data from BHH2, Ch 5, Question 19.

    Description
    -----------
    The data are from the first 8 rows of the pollutant water treatment example
    n the book by Box, Hunter and Hunter, 2nd edition, Chapter 5, Question 19.

    The 3 factors (C, T, and S) are in coded units where:
    C = -1 is chemical brand A; C = +1 is chemical brand B
    T = -1 is 72F for treatment temperature; T = +1 is 100F for the temperature
    S = -1 is No stirring; S = +1 is with fast stirring

    The outcome variable is:
    y = the pollutant amount in the discharge [lb/day].

    The aim is to find treatment conditions that MINIMIZE the amount of pollutant
    discharged each day, where the limit is 10 lb/day.

    Dimensions
    ----------
    A data frame containing 8 observations of 4 variables (C, S, T and y).

    Source
    ------
    Box, G. E. P. and Hunter, J. S. and Hunter, W. G.r, Statistics for
    Experimenters, Wiley, 2nd edition, Chapter 5, Question 19, page 232.

    """
    return pd.read_csv(_DATASETS_DIR / "pollutant.csv")


def oildoe() -> pd.DataFrame:
    """
    Return industrial designed experiment data to improve the volumetric heat capacity of
    a product.

    Description
    -----------

    Four materials: A, B, C and D are added in a blend to achieve a desired
    heat capacity, the response variable, y.

    The amounts were varied in a factorial manner for the 4 materials.

    The data are scaled and coded for confidentiality. All that may be
    disclosed is that variable C is either added ("Yes") or not added not
    added ("No"). The data are fetched from the canonical hosted location
    on openmv.net rather than bundled with the package.

    Dimensions
    ----------
    A data frame containing 19 observations of 5 variables (A, B, C, D, and
    the response, y).

    Source
    ------
    http://openmv.net/info/oil-company-doe
    Data from a confidential industrial source.

    """
    return _read_remote_csv("https://openmv.net/file/oil-company-doe.csv")


def golf() -> pd.DataFrame:
    """
    Return full factorial experiment data to maximize a golfer's driving distance.

    A full factorial experiment with four factors run by a golf enthusiast. The
    objective of the experiments was for the golfer to maximize her driving distance
    at a specific tee off location on her local golf course. The golfer considered
    the following factors:

    H = Tee height (cm)
    N = Holes: number of golf balls played for prior to experimental tee shot
    C = Club type
    T = Time of day (on the 24 hour clock)

    The data are in standard order, however the actual experiments were run in
    random order.

    Coded values for H, N, C and T should be used in the linear regression
    model analysis, with -1 representing the low value and +1 the high value.


    Dimensions
    ----------
    A data frame containing 16 observations of 4 variables (H, N, C, T) and a
    column y, as a response variable. `C` and `T` are stored as text labels
    (``"Callaway"`` / ``"Titleist"`` and ``"9:00"`` / ``"14:00"``); code them
    to -1 / +1 before fitting a linear model.

    Source
    ------
    A MOOC on Design of Experiments, "Experimentation for Improvement",
    https://learnche.org

    """
    return pd.read_csv(_DATASETS_DIR / "golf.csv")


def boilingpot() -> pd.DataFrame:
    """
    Return full factorial experiment data for stove-top boiling of water.

    Description
    -----------

    The data are from boiling water in a pot under various conditions. The
    response variable, y, is the time taken, in minutes to reach 90 degrees
    Celsius. Accurately measuring the time to actual boiling is hard, hence
    the 90 degrees Celsius point is used instead.

    Three factors are varied in a full factorial manner (the first 8
    observations). The data are in standard order, however the actual
    experiments were run in random order. The last 3 rows are runs close to,
    or interior to the factorial.

    Factors varied were:

    A = Amount of water: low level was 500 mL, and high level was 600 mL
    B = Lid off (low level) or lid on (high level)
    C = Size of pot used: low level was 2 L, and high level was 3 L.


    Coded values for A, B and C should be used in the linear regression model
    analysis, with -1 representing the low value and +1 the high value.

    Dimensions
    ----------
    A data frame containing 11 observations of 4 variables (A, B, C, with y as
    a response variable.

    Source
    ------
    MOOC on Design of Experiments, "Experimentation for Improvement",
    https://learnche.org

    """
    return pd.read_csv(_DATASETS_DIR / "boilingpot.csv")


def solar() -> pd.DataFrame:
    """
    Return solar panel example data from Box, Hunter and Hunter, 2nd edition, Chapter 5,
    page 230.


    Description
    ------------
    The data are from a solar panel simulation case study.

    The original source that Box, Hunter and Hunter used is
    https://www.sciencedirect.com/science/article/abs/pii/0038092X67900515

    A theoretical model for a commercial system was made. A 2^4 factorial
    design was used (center point is not included in this dataset).

    The factors are dimensionless groups
    (https://en.wikipedia.org/wiki/Dimensionless_quantity), related to:

    A = total daily insolation,
    B = the tank capacity,
    C = the water flow through the absorber,
    D = solar intermittency coming in.

    All 4 factors are coded as -1 for the low level, and +1 for the high lever.

    The responses variables are
    y1: collection efficiency, and
    y2: the energy delivery efficiency.

    Dimensions
    ----------
    A data frame containing 16 observations of 6 variables (A, B, C, D, with
    y1 and y2 as responses.)

    Source
    ------
    Box, G. E. P. and Hunter, J. S. and Hunter, W. G., Statistics for
    Experimenters, 2nd edition, Wiley, Chapter 5, page 230.

    """
    return pd.read_csv(_DATASETS_DIR / "solar.csv")


#: Every dataset loader in this module, keyed by the name used in the R
#: package's ``data(<name>)`` call. ``oildoe`` is also reachable under the
#: aliases the R package documents for it.
_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "boilingpot": boilingpot,
    "distillateflow": distillateflow,
    "golf": golf,
    "oildoe": oildoe,
    "oil.doe": oildoe,
    "oilDOE": oildoe,
    "pollutant": pollutant,
    "solar": solar,
}


def data(dataset: str) -> pd.DataFrame:
    """Return the ``dataset`` given by the string name.

    The Python counterpart of R's ``data(<name>)``: a single dispatcher over
    the loaders in this module, for callers that hold the dataset name as a
    string (a CLI argument, a config file, a tool call) rather than as an
    identifier.

    Parameters
    ----------
    dataset : str
        Name of the dataset. One of ``"boilingpot"``, ``"distillateflow"``,
        ``"golf"``, ``"oildoe"``, ``"pollutant"``, ``"solar"``. The aliases
        ``"oil.doe"`` and ``"oilDOE"`` also resolve to :func:`oildoe`.

    Returns
    -------
    pd.DataFrame
        The dataset, exactly as returned by the corresponding loader.

    Raises
    ------
    ValueError
        If *dataset* is not a known name.

    Examples
    --------
    >>> data("pollutant").shape
    (8, 4)

    Notes
    -----
    ``"distillateflow"`` and ``"oildoe"`` are fetched over the network from
    openmv.net; the rest are bundled with the package.
    """
    try:
        loader = _LOADERS[dataset]
    except KeyError:
        known = ", ".join(sorted({"boilingpot", "distillateflow", "golf", "oildoe", "pollutant", "solar"}))
        raise ValueError(f"Unknown dataset {dataset!r}. Available datasets are: {known}.") from None
    return loader()
