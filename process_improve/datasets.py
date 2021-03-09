# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.

import pandas as pd


def distillateflow():
    """The flow rate of distillate from the top of a distillation column.

    These are actual data, taken 1 minute apart in time, of the flow rate leaving
    the top of a continuous distillation column (data are from a 31 day period
    in time).

    Dimensions
    ----------
    A data frame containing 44640 observations of 1 variable.

    Source
    ------
    http://openmv.net/info/distillate-flow


    Example
    -------

    """
    pass


def pollutant():
    """
    Water treatment example from BHH2, Ch 5, Question 19.

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

    Example
    -------

    """


def oildoe():
    """
    Industrial designed experiment to improve the volumetric heat capacity of
    a product.

    Description
    -----------

    Four materials: A, B, C and D are added in a blend to achieve a desired
    heat capacity, the response variable, y.

    The amounts were varied in a factorial manner for the 4 materials.

    The data are scaled and coded for confidentiality. All that may be
    disclosed is that variable C is either added ("Yes") or not added not
    added ("No").

    Dimensions
    ----------
    A data frame containing 19 observations of 5 variables (A, B, C, D, and
    the response, y).

    Source
    ------
    http://openmv.net/info/oil-company-doe
    Data from a confidential industrial source.

    Example
    -------


    """
    pass


def golf():
    """
    Full factorial experiments to maximize a golfer's driving distance.

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

    Example
    -------
    """
    pass


def boilingpot():
    """
    Full factorial experiments for stove-top boiling of water.

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

    Example
    -------
    """
    pass


def solar():
    """
    Solar panel example from Box, Hunter and Hunter, 2nd edition, Chapter 5,
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

    Example
    -------

    """
    pass


def data(dataset: str) -> pd.DataFrame:
    """
    Returns the ``dataset`` given by the string name.
    """
