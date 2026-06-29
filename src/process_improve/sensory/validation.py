"""(c) Kevin Dunn, 2010-2026. MIT License.

Schema validation for descriptive panel data.

Every analysis in this subpackage enters through :func:`validate_descriptive`,
which coerces a caller-supplied table into the canonical ``descriptive_long``
schema and validates a product-covariate table alongside it. For now only the
``observational`` mode is supported: the covariate columns are measured
descriptors of products whose formulation is unknown. The ``designed`` mode
(covariate columns being controlled factor levels, analysed as effects) is
stubbed and raises ``NotImplementedError``; it is planned for a later release.
The mode is recorded on the result and decides how
:func:`process_improve.sensory.analysis.analyze_descriptive` relates the
attributes back to the product.

The validated result carries a content hash and is cached, so downstream tools
can refuse to run on data that has not passed validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

#: Required columns of the ``descriptive_long`` schema, in canonical order.
DESCRIPTIVE_LONG_COLUMNS: tuple[str, ...] = (
    "panelist_id",
    "session",
    "product",
    "attribute",
    "replicate",
    "score",
)

#: Columns coerced to stripped strings (categorical identifiers).
_LABEL_COLUMNS: tuple[str, ...] = ("panelist_id", "product", "attribute")

Mode = Literal["designed", "observational"]

#: Cache of validated results keyed by content hash, so a downstream tool can
#: confirm a frame was validated without re-running the checks.
_VALIDATED_CACHE: dict[str, ValidationResult] = {}


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_descriptive`.

    Attributes
    ----------
    ok : bool
        ``True`` when no blocking errors were found. Downstream analysis
        refuses to run when this is ``False``.
    mode : str
        ``"designed"`` or ``"observational"``; how the covariate table is
        interpreted.
    normalized_df : pandas.DataFrame or None
        The panel data coerced to the ``descriptive_long`` schema.
    covariates : pandas.DataFrame or None
        The product-covariate table, indexed by ``product``.
    warnings : list of str
        Non-blocking issues the caller should see.
    errors : list of str
        Blocking issues; non-empty implies ``ok is False``.
    content_hash : str or None
        Stable hash of the normalized inputs and mode.
    stats : dict
        Summary counts (number of panelists, products, attributes, etc.).
    """

    ok: bool
    mode: Mode
    normalized_df: pd.DataFrame | None
    covariates: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    content_hash: str | None = None
    stats: dict = field(default_factory=dict)


def _collapsed_label_warnings(labels: pd.Series, column: str) -> list[str]:
    """Warn when distinct labels differ only by case or surrounding whitespace."""
    groups: dict[str, set[str]] = {}
    for raw in labels.dropna().unique():
        key = str(raw).strip().lower()
        groups.setdefault(key, set()).add(str(raw))
    return [
        f"Column {column!r} has labels that differ only by case or whitespace: "
        f"{sorted(variants)}. Consider harmonising them."
        for variants in groups.values()
        if len(variants) > 1
    ]


def _content_hash(panel: pd.DataFrame, covariates: pd.DataFrame, mode: str) -> str:
    """Return a stable SHA-256 hash of the normalized inputs and mode."""
    hasher = hashlib.sha256()
    hasher.update(panel.to_csv(index=False).encode())
    hasher.update(covariates.to_csv(index=True).encode())
    hasher.update(mode.encode())
    return hasher.hexdigest()


def _normalise_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    """Return the covariate table indexed by ``product``.

    Accepts either a ``product`` column or a frame already indexed by product.
    """
    cov = covariates.copy()
    if "product" in cov.columns:
        cov["product"] = cov["product"].astype(str).str.strip()
        cov = cov.set_index("product")
    else:
        cov.index = cov.index.astype(str).str.strip()
        cov.index.name = "product"
    return cov


def validate_descriptive(  # noqa: PLR0912, PLR0913, PLR0915, C901
    panel: pd.DataFrame,
    covariates: pd.DataFrame,
    mode: Mode,
    *,
    score_min: float | None = None,
    score_max: float | None = None,
    balance_warn: float = 0.05,
    balance_error: float = 0.20,
) -> ValidationResult:
    """Validate panel data against the ``descriptive_long`` schema.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long-format panel data; must contain the columns listed in
        :data:`DESCRIPTIVE_LONG_COLUMNS`.
    covariates : pandas.DataFrame
        Product-covariate table. Either has a ``product`` column or is indexed
        by product. In ``observational`` mode the remaining columns are measured
        numeric descriptors. ``designed`` mode (the columns being controlled
        factor levels) is not implemented yet.
    mode : {"observational"}
        How the covariate table is interpreted. Only ``"observational"`` is
        supported for now; ``"designed"`` raises ``NotImplementedError`` and is
        planned for a later release.
    score_min, score_max : float or None
        Optional inclusive bounds for the ``score`` column; out-of-range values
        are reported as a warning.
    balance_warn, balance_error : float
        Missing-cell fractions (of the full panelist x product x attribute x
        replicate grid) above which a warning or a blocking error is raised.

    Returns
    -------
    ValidationResult
        See the class docstring. When ``ok`` is ``True`` the result is also
        stored in an in-process cache keyed by ``content_hash``.

    Examples
    --------
    >>> result = validate_descriptive(panel_df, descriptors_df, mode="observational")
    >>> result.ok
    True
    """
    if mode not in ("designed", "observational"):
        raise ValueError(f"mode must be 'designed' or 'observational', got {mode!r}.")
    if mode == "designed":
        raise NotImplementedError(
            "Designed (DoE) covariate handling is not implemented yet; use "
            "mode='observational'. Designed-mode validation and the OMARS-based "
            "relate step are planned for a later release."
        )

    warnings: list[str] = []
    errors: list[str] = []

    # --- Required columns ----------------------------------------------
    missing = [c for c in DESCRIPTIVE_LONG_COLUMNS if c not in panel.columns]
    if missing:
        errors.append(f"Panel data is missing required columns: {missing}.")
        return ValidationResult(
            ok=False, mode=mode, normalized_df=None, covariates=None, errors=errors
        )

    df = panel.loc[:, list(DESCRIPTIVE_LONG_COLUMNS)].copy()

    # --- Dtype coercion ------------------------------------------------
    for col in _LABEL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    n_before = df["score"].notna().sum()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    n_unparsed = int(n_before - df["score"].notna().sum())
    if n_unparsed:
        warnings.append(f"{n_unparsed} score value(s) could not be parsed as numeric and became missing.")

    # --- Encoding sanity ----------------------------------------------
    for col in ("product", "attribute"):
        warnings.extend(_collapsed_label_warnings(df[col], col))

    # --- Score range ---------------------------------------------------
    if score_min is not None or score_max is not None:
        lo = score_min if score_min is not None else -float("inf")
        hi = score_max if score_max is not None else float("inf")
        out_of_range = df["score"].dropna()
        n_out = int(((out_of_range < lo) | (out_of_range > hi)).sum())
        if n_out:
            warnings.append(
                f"{n_out} score value(s) fall outside the expected range "
                f"[{score_min}, {score_max}]."
            )

    # --- Balance audit -------------------------------------------------
    n_panelist = df["panelist_id"].nunique()
    n_product = df["product"].nunique()
    n_attribute = df["attribute"].nunique()
    n_replicate = df["replicate"].nunique()
    expected = n_panelist * n_product * n_attribute * n_replicate
    present = df.dropna(subset=["score"]).drop_duplicates(
        subset=["panelist_id", "product", "attribute", "replicate"]
    ).shape[0]
    missing_fraction = 0.0 if expected == 0 else 1.0 - present / expected
    if missing_fraction > balance_error:
        errors.append(
            f"Panel is badly unbalanced: {missing_fraction:.1%} of the full "
            f"panelist x product x attribute x replicate grid is missing "
            f"(error threshold {balance_error:.0%})."
        )
    elif missing_fraction > balance_warn:
        warnings.append(
            f"Panel is unbalanced: {missing_fraction:.1%} of the full grid is "
            f"missing (warning threshold {balance_warn:.0%})."
        )

    # --- Covariate table ----------------------------------------------
    cov = _normalise_covariates(covariates)
    panel_products = set(df["product"].unique())
    cov_products = set(cov.index)
    absent = sorted(panel_products - cov_products)
    if absent:
        errors.append(f"These products have no row in the covariate table: {absent}.")

    # Observational mode only for now (designed mode is rejected above).
    non_numeric = [c for c in cov.columns if not pd.api.types.is_numeric_dtype(cov[c])]
    if non_numeric:
        errors.append(f"Observational descriptors must be numeric; non-numeric columns: {non_numeric}.")
    else:
        n_missing_cov = int(cov.isna().sum().sum())
        if n_missing_cov:
            warnings.append(f"Covariate table has {n_missing_cov} missing descriptor value(s).")

    ok = not errors
    content_hash = _content_hash(df, cov, mode) if ok else None
    stats = {
        "n_rows": int(df.shape[0]),
        "n_panelists": int(n_panelist),
        "n_products": int(n_product),
        "n_attributes": int(n_attribute),
        "n_replicates": int(n_replicate),
        "n_sessions": int(df["session"].nunique()),
        "n_covariates": int(cov.shape[1]),
        "missing_fraction": float(missing_fraction),
    }
    result = ValidationResult(
        ok=ok,
        mode=mode,
        normalized_df=df if ok else None,
        covariates=cov if ok else None,
        warnings=warnings,
        errors=errors,
        content_hash=content_hash,
        stats=stats,
    )
    if ok and content_hash is not None:
        _VALIDATED_CACHE[content_hash] = result
    return result


def is_validated(content_hash: str) -> bool:
    """Return ``True`` if ``content_hash`` refers to a cached validated result."""
    return content_hash in _VALIDATED_CACHE
