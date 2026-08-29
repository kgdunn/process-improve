"""(c) Kevin Dunn, 2010-2026. MIT License.

Preprocessing a product-by-compound block of concentrations or peak areas.

One row per product, one column per compound, values being concentrations or
integrated peak areas. Getting such a block ready for a PLS against a block of
sensory attributes is four decisions, and the order they are taken in is fixed:

.. code-block:: text

    trim  ->  transform  ->  centre  ->  scale

Trimming first, because a compound seen in two of forty products has no
concentration worth transforming. Transforming before centring, because the
range ratio that decides between a log and a linear scale is a property of the
raw values. Centring before scaling, because a scaling constant estimated
around the wrong centre is the wrong constant.

Three things in here are easy to get wrong, and each is wrong quietly:

* **A zero is not self-describing.** It is either a concentration below a
  detection limit (a *rounded* zero, the compound is there) or a compound that
  is genuinely absent (an *essential* zero). Those need opposite handling, and
  the distinction cannot be recovered from an exported table.
  :func:`classify_zero_states` therefore defaults to ``"unknown"`` and makes
  the caller declare. It never defaults to censored.
* **A trimmed compound is not a discarded compound.** For a rare compound the
  binary fingerprint of where it appears at all often carries more than its
  concentration does, so :func:`trim_by_prevalence` returns a presence layer
  covering every compound, not only the kept ones.
* **Preprocessing constants must not see the test rows.** Every fitting
  function here has an ``apply_fitted_*`` partner that replays the constants
  computed elsewhere, so held-out rows can be preprocessed with training
  constants alone. Without that pair, honest nested cross-validation is not
  possible: the transform offsets and the scaling constants would both have
  seen the row being predicted.

References
----------
Martin-Fernandez, Barcelo-Vidal and Pawlowsky-Glahn, "Dealing with zeros and
missing values in compositional data sets using nonparametric imputation",
Mathematical Geology, 35(3), 253-278, 2003.

van den Berg, Hoefsloot, Westerhuis and others, "Centering, scaling, and
transformations: improving the biological information content of metabolomics
data", BMC Genomics, 7, 142, 2006, doi:10.1186/1471-2164-7-142.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The three zero states. ``unknown`` is the default and is not a failure: it
#: records that nobody has yet said which of the other two applies.
ZERO_STATES: tuple[str, ...] = ("rounded", "essential", "unknown")

#: The transform rules :func:`choose_transform` can return.
TRANSFORM_RULES: tuple[str, ...] = ("log", "linear", "ambiguous")

#: Scaling methods understood by :func:`center_and_scale`.
SCALING_METHODS: tuple[str, ...] = ("autoscale", "pareto")

#: A divisor at or below this is treated as no spread at all and replaced by 1.0,
#: mirroring :class:`~process_improve.multivariate.MCUVScaler`, so a constant
#: column passes through unchanged rather than becoming inf or NaN.
_TINY = float(np.finfo(float).tiny) ** 0.5


def _require_numeric_block(chem: pd.DataFrame, caller: str) -> pd.DataFrame:
    """Validate a product-by-compound block and return it as floats."""
    if not isinstance(chem, pd.DataFrame):
        raise TypeError(f"{caller} needs a pandas DataFrame of products (rows) by compounds (columns).")
    if chem.shape[1] == 0:
        raise ValueError(f"{caller} was given a block with no compounds (no columns).")
    if chem.shape[0] == 0:
        raise ValueError(
            f"{caller} was given a block with no products (no rows), so there is nothing to "
            "compute from. This usually means an upstream filter removed every row."
        )
    numeric = chem.apply(pd.to_numeric, errors="coerce").astype(float)
    if numeric.isna().all(axis=None) and not chem.isna().all(axis=None):
        raise ValueError(f"{caller} could not read any numeric values out of the block; check the column dtypes.")
    return numeric


def classify_zero_states(
    chem: pd.DataFrame,
    declared: dict | None = None,
    lod: dict | None = None,
) -> pd.DataFrame:
    """Record, per compound, what a zero in that column is taken to mean.

    A zero is either *rounded* (left-censored: the compound is present at a
    concentration below the detection limit) or *essential* (structurally
    absent: the compound is not there). The two want opposite handling, and no
    amount of looking at an exported table recovers which one applies, so this
    function does not guess. A compound nobody has spoken for is ``"unknown"``.

    **Never default to censored.** Classifying a zero as left-censored asserts a
    latent value below a detection limit, which is a claim about the chemistry,
    not a convenience. Declaring a detection limit for a compound *is* that
    claim, so passing one in ``lod`` classifies the compound as ``"rounded"``.

    Parameters
    ----------
    chem : pandas.DataFrame
        Products (rows) by compounds (columns) of concentrations or peak areas.
    declared : dict or None
        Explicit per-compound states, ``{compound: "rounded" | "essential" |
        "unknown"}``. Takes precedence over ``lod``. Compound names not in the
        block are an error, since a typo would otherwise silently leave the
        compound unknown.
    lod : dict or None
        Per-compound limits of detection, ``{compound: float}``. A compound with
        a finite, positive limit and no entry in ``declared`` is classified
        ``"rounded"``.

    Returns
    -------
    pandas.DataFrame
        One row per compound, in the block's column order, with columns:

        ``compound``
            The compound name.
        ``zero_state``
            ``"rounded"``, ``"essential"`` or ``"unknown"``.
        ``source``
            How the state was arrived at: ``"declared"``, ``"lod"``, or
            ``"default"``.
        ``lod``
            The declared limit of detection, or ``NaN``.
        ``n_zero``, ``n_nonzero``, ``n_missing``
            Cell counts, so a compound with no zeros at all (whose state is
            therefore moot) is easy to spot.

    Raises
    ------
    ValueError
        If the block is empty, a declared state is not one of
        :data:`ZERO_STATES`, or ``declared`` / ``lod`` names a compound the
        block does not have.

    Examples
    --------
    >>> states = classify_zero_states(chem, lod={"linalool": 0.02})
    >>> states.query("zero_state == 'unknown' and n_zero > 0")["compound"].tolist()
    """
    numeric = _require_numeric_block(chem, "classify_zero_states")
    declared = dict(declared or {})
    lod = dict(lod or {})

    known = set(numeric.columns)
    for name, mapping in (("declared", declared), ("lod", lod)):
        unknown_names = [key for key in mapping if key not in known]
        if unknown_names:
            raise ValueError(
                f"{name} names compound(s) {sorted(unknown_names)} that are not columns of the block. "
                "A typo here would silently leave the compound unclassified, so it is an error."
            )
    bad_states = {key: value for key, value in declared.items() if value not in ZERO_STATES}
    if bad_states:
        raise ValueError(f"declared states must be one of {list(ZERO_STATES)}; got {bad_states}.")

    rows: list[dict[str, object]] = []
    for compound in numeric.columns:
        column = numeric[compound]
        limit = float(lod[compound]) if compound in lod else float("nan")
        if compound in declared:
            state, source = str(declared[compound]), "declared"
        elif np.isfinite(limit) and limit > 0:
            state, source = "rounded", "lod"
        else:
            state, source = "unknown", "default"
        rows.append(
            {
                "compound": str(compound),
                "zero_state": state,
                "source": source,
                "lod": limit,
                "n_zero": int((column == 0).sum()),
                "n_nonzero": int((column != 0).sum()),
                "n_missing": int(column.isna().sum()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["compound", "zero_state", "source", "lod", "n_zero", "n_nonzero", "n_missing"],
    )


def trim_by_prevalence(
    chem: pd.DataFrame,
    min_nonzero: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the block by how often each compound is seen at all.

    A compound detected in one or two products has no concentration worth
    modelling: any apparent relationship is a line through two points. It is not
    thereby uninformative, though, so nothing is thrown away. The third return
    value is a presence layer covering **every** compound, kept and dropped
    alike, which for a rare compound is often the more useful representation:
    "this compound appears in exactly the three products the panel called
    'green'" is a finding, and it is one a concentration column full of zeros
    states badly.

    Parameters
    ----------
    chem : pandas.DataFrame
        Products (rows) by compounds (columns).
    min_nonzero : int, default 3
        Keep a compound when it is non-zero in at least this many products.

    Returns
    -------
    (kept, dropped, presence) : tuple of pandas.DataFrame
        ``kept`` and ``dropped`` partition the columns of ``chem``, both keeping
        every row. ``presence`` has the same shape as ``chem`` and holds 1.0
        where a compound was detected, 0.0 where it was not, and ``NaN`` where
        the measurement is missing: a missing measurement is not an absence.

    Raises
    ------
    ValueError
        If the block is empty or ``min_nonzero`` is negative.

    Examples
    --------
    >>> kept, dropped, presence = trim_by_prevalence(chem, min_nonzero=3)
    >>> presence[dropped.columns].sum().sort_values(ascending=False).head()
    """
    numeric = _require_numeric_block(chem, "trim_by_prevalence")
    if int(min_nonzero) < 0:
        raise ValueError(f"min_nonzero must be >= 0; got {min_nonzero!r}.")

    detected = numeric > 0
    presence = detected.astype(float).where(numeric.notna())
    counts = detected.sum(axis=0)
    keep_mask = counts >= int(min_nonzero)

    kept = numeric.loc[:, keep_mask.to_numpy()]
    dropped = numeric.loc[:, (~keep_mask).to_numpy()]
    return kept, dropped, presence


def normalisation_check(chem: pd.DataFrame, factor: float = 1.8) -> tuple[pd.Series, pd.Series]:
    """Report the row totals, and which of them sit outside a fold-band around the median.

    A product-by-compound block that has been normalised to a constant sum has
    row totals that are all equal; one that has not may still be fine, but a row
    whose total is several fold away from its neighbours usually means something
    mechanical rather than chemical (a different injection volume, a dilution
    that was not recorded, an integration that dropped a peak). Either way the
    answer changes what the rest of the pipeline should do, so it is worth
    looking before transforming.

    Parameters
    ----------
    chem : pandas.DataFrame
        Products (rows) by compounds (columns).
    factor : float, default 1.8
        Half-width of the accepted band, as a fold change: a row is reported
        when its total is above ``median * factor`` or below ``median /
        factor``. Must be greater than 1.

    Returns
    -------
    (totals, outside) : tuple of pandas.Series
        ``totals`` is the row sum for every product, ``NaN`` for a row with no
        measurements at all. ``outside`` is the subset of ``totals`` beyond the
        band, in the block's row order; it is empty when every row is inside.

    Raises
    ------
    ValueError
        If the block is empty, ``factor`` is not greater than 1, or the median
        row total is not positive (which leaves no band to compare against).

    Examples
    --------
    >>> totals, outside = normalisation_check(chem)
    >>> outside / totals.median()  # how far out, as a fold change
    """
    numeric = _require_numeric_block(chem, "normalisation_check")
    if not np.isfinite(factor) or factor <= 1:
        raise ValueError(f"factor must be a finite fold change greater than 1; got {factor!r}.")

    totals = numeric.sum(axis=1, min_count=1)
    median = float(totals.median(skipna=True))
    if not np.isfinite(median) or median <= 0:
        raise ValueError(
            f"the median row total is {median!r}, so there is no band to compare rows against. "
            "Check that the block holds non-negative concentrations or peak areas."
        )
    outside = totals[(totals > median * factor) | (totals < median / factor)]
    return totals, outside


def choose_transform(col: pd.Series, ratio_log: float = 10.0, ratio_linear: float = 3.0) -> str:
    """Decide whether a compound should be modelled on a log or a linear scale.

    The decision is made on the range ratio of the **detected** values: the
    largest divided by the smallest, ignoring zeros and missing cells. A
    compound spanning orders of magnitude is multiplicative and belongs on a log
    scale; one varying by a factor of two or three is additive and does not. In
    between there is no evidence either way and the honest answer is
    ``"ambiguous"``, which :func:`apply_transform` resolves with a caller-chosen
    default rather than a coin toss.

    Parameters
    ----------
    col : pandas.Series
        One compound's values across products.
    ratio_log : float, default 10.0
        A range ratio at or above this chooses ``"log"``.
    ratio_linear : float, default 3.0
        A range ratio at or below this chooses ``"linear"``.

    Returns
    -------
    str
        One of :data:`TRANSFORM_RULES`. ``"linear"`` is returned when a log is
        not applicable at all (a negative value present, or fewer than two
        detected values to form a ratio from).

    Raises
    ------
    ValueError
        If ``ratio_log`` is not strictly greater than ``ratio_linear``, which
        would make the two rules overlap.

    Examples
    --------
    >>> choose_transform(chem["limonene"])
    'log'
    """
    if not (np.isfinite(ratio_log) and np.isfinite(ratio_linear)) or ratio_log <= ratio_linear:
        raise ValueError(
            f"ratio_log must be strictly greater than ratio_linear, or the two rules overlap; "
            f"got ratio_log={ratio_log!r}, ratio_linear={ratio_linear!r}."
        )
    values = pd.to_numeric(pd.Series(col), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size and float(values.min()) < 0:
        # A log is undefined here, so there is no decision left to make.
        return "linear"
    detected = values[values > 0]
    if detected.size < 2:
        return "linear"
    ratio = float(detected.max() / detected.min())
    if ratio >= ratio_log:
        return "log"
    if ratio <= ratio_linear:
        return "linear"
    return "ambiguous"


def _log_offset(values: np.ndarray, lod_value: float) -> float:
    """Return the substitution value used for non-detects before a log transform.

    Half the detection limit is the standard substitution when a limit is known;
    with no limit, half the smallest value actually seen is the usual stand-in.
    Both are imputations, which is why :func:`center_and_scale` is told which
    cells they landed in.
    """
    if np.isfinite(lod_value) and lod_value > 0:
        return float(lod_value) / 2.0
    detected = values[np.isfinite(values) & (values > 0)]
    if detected.size == 0:
        # Defensive, and unreachable through apply_transform today: the "log"
        # rule requires two detected values to form a range ratio from, so a
        # column with none never gets here. Kept so a future caller meets a
        # constant rather than a ValueError from min() on an empty array. Any
        # positive placeholder gives log(1) = 0 for the whole column, which is
        # the only honest constant when nothing was ever detected.
        return 1.0
    return float(detected.min()) / 2.0


def _apply_log(values: np.ndarray, offset: float) -> np.ndarray:
    """Substitute non-detects with ``offset`` and take base-10 logs, keeping NaN as NaN."""
    substituted = np.where(np.isnan(values), np.nan, np.where(values > 0, values, offset))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(substituted)


def apply_transform(
    chem: pd.DataFrame,
    lod: dict | None = None,
    ambiguous: str = "linear",
    ratio_log: float = 10.0,
    ratio_linear: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform every compound by the rule :func:`choose_transform` picks for it.

    A ``"log"`` compound has its non-detects replaced (by half the declared
    detection limit, or half the smallest value seen) and is then taken to base
    10. A ``"linear"`` compound passes through untouched: a zero on a linear
    scale is a usable number and needs no substitution.

    The substitution is an imputation, and it is the only imputation this module
    performs. It is what makes ``detected_only=True`` in :func:`center_and_scale`
    meaningful, and what makes it meaningless before this step has run.

    Parameters
    ----------
    chem : pandas.DataFrame
        Products (rows) by compounds (columns). Trim first: see
        :func:`trim_by_prevalence`.
    lod : dict or None
        Per-compound limits of detection, ``{compound: float}``. Used to set the
        substitution value for a log-transformed compound.
    ambiguous : {"linear", "log"}, default "linear"
        What to do with a compound whose range ratio decides nothing. The
        default keeps the values as they are, which is the smaller claim.
    ratio_log, ratio_linear : float
        Passed to :func:`choose_transform`.

    Returns
    -------
    (transformed, applied) : tuple of pandas.DataFrame
        ``transformed`` has the shape and labels of ``chem``. ``applied`` has one
        row per compound with columns ``compound``, ``rule`` (``"log"`` or
        ``"linear"``), ``offset`` (the substitution value; 0.0 for a linear
        compound), ``range_ratio``, and ``chosen_by`` (``"range_ratio"`` or
        ``"ambiguous_default"``). Feed it to :func:`apply_fitted_transform` to
        replay these decisions on held-out rows.

    Raises
    ------
    ValueError
        If the block is empty, ``ambiguous`` is not ``"linear"`` or ``"log"``,
        ``lod`` names a compound the block does not have, or the two ratio
        thresholds overlap.

    Examples
    --------
    >>> transformed, applied = apply_transform(kept, lod={"linalool": 0.02})
    >>> applied.query("rule == 'log'")["compound"].tolist()
    """
    numeric = _require_numeric_block(chem, "apply_transform")
    if ambiguous not in ("linear", "log"):
        raise ValueError(f"ambiguous must be 'linear' or 'log'; got {ambiguous!r}.")
    lod = dict(lod or {})
    unknown_names = [key for key in lod if key not in set(numeric.columns)]
    if unknown_names:
        raise ValueError(f"lod names compound(s) {sorted(unknown_names)} that are not columns of the block.")

    out = {}
    rows: list[dict[str, object]] = []
    for compound in numeric.columns:
        values = numeric[compound].to_numpy(dtype=float)
        rule = choose_transform(numeric[compound], ratio_log=ratio_log, ratio_linear=ratio_linear)
        chosen_by = "range_ratio"
        if rule == "ambiguous":
            rule, chosen_by = ambiguous, "ambiguous_default"

        detected = values[np.isfinite(values) & (values > 0)]
        ratio = float(detected.max() / detected.min()) if detected.size >= 2 else float("nan")

        if rule == "log":
            offset = _log_offset(values, float(lod.get(compound, float("nan"))))
            out[compound] = _apply_log(values, offset)
        else:
            offset = 0.0
            out[compound] = values

        rows.append(
            {
                "compound": str(compound),
                "rule": rule,
                "offset": float(offset),
                "range_ratio": ratio,
                "chosen_by": chosen_by,
            }
        )

    transformed = pd.DataFrame(out, index=numeric.index, columns=numeric.columns)
    applied = pd.DataFrame(rows, columns=["compound", "rule", "offset", "range_ratio", "chosen_by"])
    return transformed, applied


def apply_fitted_transform(chem: pd.DataFrame, applied: pd.DataFrame) -> pd.DataFrame:
    """Replay a transform table computed elsewhere, without re-deriving it.

    This is the half of :func:`apply_transform` that must be used on held-out
    rows. Re-deriving the rule and the offset from the test rows would let those
    rows influence their own preprocessing, and the cross-validated score would
    then be measuring something other than out-of-sample performance.

    Parameters
    ----------
    chem : pandas.DataFrame
        Products (rows) by compounds (columns). Every column must have a row in
        ``applied``.
    applied : pandas.DataFrame
        The second return value of :func:`apply_transform`; needs at least the
        ``compound``, ``rule`` and ``offset`` columns.

    Returns
    -------
    pandas.DataFrame
        The transformed block, with the labels of ``chem``.

    Raises
    ------
    ValueError
        If ``applied`` is missing a required column, names a rule other than
        ``"log"`` or ``"linear"``, or has no entry for some column of ``chem``.

    Examples
    --------
    >>> train_t, applied = apply_transform(chem.iloc[train_rows])
    >>> test_t = apply_fitted_transform(chem.iloc[test_rows], applied)
    """
    numeric = _require_numeric_block(chem, "apply_fitted_transform")
    required = ("compound", "rule", "offset")
    missing_columns = [column for column in required if column not in applied.columns]
    if missing_columns:
        raise ValueError(f"applied must carry the columns {list(required)}; missing {missing_columns}.")

    names = [str(name) for name in applied["compound"]]
    rules = dict(zip(names, (str(rule) for rule in applied["rule"]), strict=True))
    offsets = dict(zip(names, (float(offset) for offset in applied["offset"]), strict=True))

    unknown_compounds = [str(column) for column in numeric.columns if str(column) not in rules]
    if unknown_compounds:
        raise ValueError(
            f"applied has no entry for compound(s) {sorted(unknown_compounds)}. Preprocessing a "
            "held-out block with constants that were never fitted for it is not something this "
            "function will guess at."
        )
    bad_rules = sorted({rule for rule in rules.values() if rule not in ("log", "linear")})
    if bad_rules:
        raise ValueError(f"applied names unknown transform rule(s) {bad_rules}; expected 'log' or 'linear'.")

    out = {}
    for compound in numeric.columns:
        values = numeric[compound].to_numpy(dtype=float)
        name = str(compound)
        out[compound] = _apply_log(values, offsets[name]) if rules[name] == "log" else values
    return pd.DataFrame(out, index=numeric.index, columns=numeric.columns)


def center_and_scale(
    transformed: pd.DataFrame,
    detected: pd.DataFrame,
    method: str = "autoscale",
    *,
    detected_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Centre and scale a transformed block, returning the constants for replay.

    ``"autoscale"`` divides by the standard deviation (``ddof=1``, matching
    :class:`~process_improve.multivariate.MCUVScaler`), giving every compound
    equal say. ``"pareto"`` divides by its square root, which keeps some of the
    original variance structure and is the usual choice when the large peaks are
    meant to stay large.

    Parameters
    ----------
    transformed : pandas.DataFrame
        Output of :func:`apply_transform`.
    detected : pandas.DataFrame
        A 1 / 0 / ``NaN`` layer of the same shape, marking which cells were
        actually observed; the ``presence`` return of :func:`trim_by_prevalence`,
        restricted to the same columns. Only read when ``detected_only=True``,
        but required either way so that a caller cannot switch the flag on
        without having the mask to hand.
    method : {"autoscale", "pareto"}, default "autoscale"
        The divisor: the standard deviation, or its square root.
    detected_only : bool, keyword-only, default False
        Compute the centring and scaling constants from the detected cells
        alone.

        **Leave this off unless imputation has actually run.** The rule it
        implements, that constants must not be estimated from imputed values, is
        sound and easy to over-apply. On a column whose zeros are real
        observations of "not detected", excluding them puts every one of those
        zeros many standard deviations below a centre estimated from a handful
        of detected values. The column becomes effectively binary with a very
        large magnitude, and since PLS follows variance, the components then
        track how *sparse* a variable is rather than how it relates to the
        response: every attribute comes back with the same handful of rare
        compounds at the top of its list. Switch it on after
        :func:`apply_transform` has substituted non-detects for a log-scaled
        compound, and not before.

    Returns
    -------
    (scaled, constants) : tuple of pandas.DataFrame
        ``scaled`` has the labels of ``transformed``. ``constants`` has one row
        per compound with columns ``compound``, ``center``, ``divisor``,
        ``method`` and ``n_used``. ``divisor`` is what the centred values were
        **divided** by, not a multiplier; feed the table to
        :func:`apply_fitted_center_scale` rather than reapplying it by hand.

    Raises
    ------
    ValueError
        If the block is empty, ``method`` is not one of
        :data:`SCALING_METHODS`, or ``detected`` does not line up with
        ``transformed``.

    Examples
    --------
    >>> scaled, constants = center_and_scale(transformed, presence[transformed.columns])
    >>> constants.sort_values("divisor").head()
    """
    numeric = _require_numeric_block(transformed, "center_and_scale")
    if method not in SCALING_METHODS:
        raise ValueError(f"method must be one of {list(SCALING_METHODS)}; got {method!r}.")
    if not isinstance(detected, pd.DataFrame):
        raise TypeError("detected must be a DataFrame of the same shape as transformed.")
    if list(detected.columns) != list(transformed.columns) or list(detected.index) != list(transformed.index):
        raise ValueError(
            "detected must have exactly the rows and columns of transformed, so that a cell in one "
            f"means the same cell in the other. Got {detected.shape} against {numeric.shape}."
        )

    mask = detected.to_numpy(dtype=float) > 0
    values = numeric.to_numpy(dtype=float)
    usable = np.isfinite(values) & (mask if detected_only else True)

    rows: list[dict[str, object]] = []
    centred = np.empty_like(values)
    for position, compound in enumerate(numeric.columns):
        column = values[:, position]
        selected = column[usable[:, position]]
        n_used = int(selected.size)
        centre = float(np.mean(selected)) if n_used else 0.0
        spread = float(np.std(selected, ddof=1)) if n_used > 1 else 0.0
        if not np.isfinite(spread) or spread <= _TINY:
            divisor = 1.0
        else:
            divisor = spread if method == "autoscale" else float(np.sqrt(spread))
        centred[:, position] = (column - centre) / divisor
        rows.append(
            {
                "compound": str(compound),
                "center": centre,
                "divisor": divisor,
                "method": method,
                "n_used": n_used,
            }
        )

    scaled = pd.DataFrame(centred, index=numeric.index, columns=numeric.columns)
    constants = pd.DataFrame(rows, columns=["compound", "center", "divisor", "method", "n_used"])
    return scaled, constants


def apply_fitted_center_scale(transformed: pd.DataFrame, constants: pd.DataFrame) -> pd.DataFrame:
    """Replay centring and scaling constants computed elsewhere.

    The partner of :func:`center_and_scale`, for held-out rows. ``divisor`` is
    divided by, not multiplied by: it is the standard deviation (or its square
    root), the same quantity :func:`center_and_scale` divided the training rows
    by.

    Parameters
    ----------
    transformed : pandas.DataFrame
        Products (rows) by compounds (columns), already transformed with
        :func:`apply_fitted_transform`. Every column must have a row in
        ``constants``.
    constants : pandas.DataFrame
        The second return value of :func:`center_and_scale`; needs at least the
        ``compound``, ``center`` and ``divisor`` columns.

    Returns
    -------
    pandas.DataFrame
        The scaled block, with the labels of ``transformed``.

    Raises
    ------
    ValueError
        If ``constants`` is missing a required column, has no entry for some
        column of ``transformed``, or carries a non-positive divisor.

    Examples
    --------
    >>> train_s, constants = center_and_scale(train_t, train_presence)
    >>> test_s = apply_fitted_center_scale(test_t, constants)
    """
    numeric = _require_numeric_block(transformed, "apply_fitted_center_scale")
    required = ("compound", "center", "divisor")
    missing_columns = [column for column in required if column not in constants.columns]
    if missing_columns:
        raise ValueError(f"constants must carry the columns {list(required)}; missing {missing_columns}.")

    names = [str(name) for name in constants["compound"]]
    centre_by_name = dict(zip(names, (float(value) for value in constants["center"]), strict=True))
    divisor_by_name = dict(zip(names, (float(value) for value in constants["divisor"]), strict=True))

    wanted = [str(column) for column in numeric.columns]
    unknown_compounds = [name for name in wanted if name not in centre_by_name]
    if unknown_compounds:
        raise ValueError(
            f"constants has no entry for compound(s) {sorted(unknown_compounds)}. Preprocessing a "
            "held-out block with constants that were never fitted for it is not something this "
            "function will guess at."
        )

    centres = np.array([centre_by_name[name] for name in wanted], dtype=float)
    divisors = np.array([divisor_by_name[name] for name in wanted], dtype=float)
    if np.any(~np.isfinite(divisors)) or np.any(divisors <= 0):
        raise ValueError(
            "constants carries a non-positive or non-finite divisor. center_and_scale writes 1.0 "
            "for a constant column, so a value at or below zero means the table was edited or "
            "built by hand."
        )
    return pd.DataFrame(
        (numeric.to_numpy(dtype=float) - centres) / divisors,
        index=numeric.index,
        columns=numeric.columns,
    )
