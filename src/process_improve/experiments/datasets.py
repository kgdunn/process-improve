# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import logging
import time
import urllib.error
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "experiments"

# Retry policy for the live sample-data fetches.  Three attempts at 1s and 2s of
# backoff costs at most three seconds on a genuine outage, and rides out the
# momentary 5xx blips that would otherwise fail a whole run.
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_SECONDS = 1.0

# Server-side and connection failures worth a second look.  A 404 or a 403 will
# say the same thing next time, so those are not retried.
_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _is_transient(exc: Exception) -> bool:
    """Return True when *exc* looks like a blip rather than a settled answer.

    A parse failure (``ValueError``) means the fetch succeeded and the content
    was wrong, so retrying cannot help.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _TRANSIENT_HTTP_STATUS
    return isinstance(exc, OSError)


def _read_remote_csv(url: str, *, attempts: int = _FETCH_ATTEMPTS) -> pd.DataFrame:
    """Fetch a sample-dataset CSV from a (hard-coded, trusted) remote host.

    The URL is not user-supplied. Network or parse failures are surfaced as a
    clear ``RuntimeError`` rather than a lower-level ``URLError`` / parser error,
    so callers get an actionable message. Note that the content is fetched over
    the network and is therefore trusted only as far as the remote host is.

    A transient failure is retried with exponential backoff before giving up.
    Sample data is fetched live, so a momentary blip at the remote host would
    otherwise fail a whole test run or documentation build; a 502 from the host
    has done exactly that. Only failures that look transient are retried: a
    parse error, or an HTTP status that will not change on a second attempt, is
    raised immediately rather than paying the backoff for nothing.

    Parameters
    ----------
    url : str
        The dataset URL. Hard-coded by the calling loader, never user-supplied.
    attempts : int
        Maximum number of tries, including the first. Defaults to
        :data:`_FETCH_ATTEMPTS`; pass ``1`` to disable retrying. Must be at
        least 1.

    Returns
    -------
    pandas.DataFrame
        The parsed CSV.

    Raises
    ------
    RuntimeError
        If the dataset could not be fetched or parsed. The message names the URL
        and says the data comes from a remote host, so a caller can tell an
        outage apart from a genuine bug.
    ValueError
        If *attempts* is less than 1, which would otherwise fetch nothing and
        report a failure that never happened.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1; got {attempts}.")

    attempt = 0
    while True:
        attempt += 1
        try:
            return pd.read_csv(url)
        except (OSError, ValueError) as exc:
            if attempt >= attempts or not _is_transient(exc):
                raise RuntimeError(
                    f"Could not download the sample dataset from {url!r}: {exc}. "
                    "Check your network connection; this dataset is fetched from a remote host."
                ) from exc
            delay = _FETCH_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.info(
                "Fetching %s failed (%s); retrying in %.1fs (attempt %d of %d).", url, exc, delay, attempt + 1, attempts
            )
            time.sleep(delay)


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


def pollutant() -> None:
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


def golf() -> None:
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
    column y, as a response variable.

    Source
    ------
    A MOOC on Design of Experiments: ``Experimentation for Improvement'',
    https://learnche.org

    """


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
    MOOC on Design of Experiments: ``Experimentation for Improvement'',
    https://learnche.org

    """
    return pd.read_csv(_DATASETS_DIR / "boilingpot.csv")


def solar() -> None:
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


def data(dataset: str) -> pd.DataFrame:
    """Return the ``dataset`` given by the string name.

    This is a planned dispatcher that has not been implemented yet. It is kept
    as a typed stub so that its public signature is preserved; calling it raises
    :class:`NotImplementedError` rather than silently returning ``None``.
    """
    raise NotImplementedError(f"datasets.data({dataset!r}) is not implemented yet.")
