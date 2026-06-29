"""(c) Kevin Dunn, 2010-2026. MIT License.

Deterministic reshaping of panel data into the ``descriptive_long`` schema.

Raw panel data usually arrives wide (one column per attribute) or already long.
The front end (or a code sandbox) parses the spreadsheet into rows; this module
performs the reshape *deterministically* and self-checks it, so the bulk
transform never goes through an LLM's tokens (which would risk silent
mis-mapping on real-sized panels).

:func:`reshape_to_long` takes an explicit column mapping (the caller, possibly
an LLM, decides which column is which) and a ``layout`` flag, melts to long when
needed, and verifies a set of round-trip invariants: the grand mean, the mean
per attribute, the mean per panelist, and the count of non-missing cells must be
identical before and after the reshape. A mismatch (for example product and
attribute columns swapped) raises rather than silently corrupting every
downstream statistic.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from process_improve.sensory.validation import DESCRIPTIVE_LONG_COLUMNS

Layout = Literal["long", "wide_by_attribute", "wide_by_product"]

#: Tolerances for the before/after round-trip invariant comparison.
_ABS_TOL = 1e-8
_REL_TOL = 1e-6

#: Canonical row order of the long output (sample-major). Ordering does not
#: affect any analysis, but a stable order keeps the validated content hash
#: independent of the input order.
_CANONICAL_SORT = ["product", "attribute", "panelist_id", "session", "replicate"]


def _series_mean_map(frame: pd.DataFrame, key: str, value: str) -> dict[str, float]:
    """Mean of ``value`` grouped by ``key``, as a plain ``{label: mean}`` dict."""
    grouped = frame.groupby(key, observed=True)[value].mean()
    return {str(k): float(v) for k, v in grouped.items()}


def _compare_maps(before: dict[str, float], after: dict[str, float]) -> float:
    """Return the largest absolute difference between two ``{label: mean}`` maps."""
    keys = set(before) | set(after)
    return max((abs(before.get(k, np.nan) - after.get(k, np.nan)) for k in keys), default=0.0)


def reshape_to_long(  # noqa: C901, PLR0912, PLR0915
    data: pd.DataFrame,
    *,
    layout: Layout,
    mapping: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reshape panel data into the ``descriptive_long`` schema, with round-trip checks.

    Parameters
    ----------
    data : pandas.DataFrame
        The parsed panel table (rows already read from the spreadsheet).
    layout : {"long", "wide_by_attribute", "wide_by_product"}
        ``"long"`` passes through (renaming columns and canonicalising);
        ``"wide_by_attribute"`` melts attribute columns into rows (rows carry
        the product); ``"wide_by_product"`` melts product columns into rows
        (rows carry the attribute, i.e. one panelist-by-product matrix per
        attribute, stacked with an attribute label column).
    mapping : dict
        Explicit column roles:

        * ``panelist_id`` (required): column name.
        * ``session`` and ``replicate`` (optional): column names; default to a
          constant 1 when absent.
        * For ``wide_by_attribute``: ``product`` (required) and ``attributes``
          (the list of attribute columns; if omitted, all non-id columns).
        * For ``wide_by_product``: ``attribute`` (required, the block-label
          column) and ``products`` (the list of product columns; if omitted,
          all non-id columns).
        * For ``long``: ``product``, ``attribute`` and ``score`` column names.
        * ``ignore`` (optional): columns to drop before reshaping (nuisance
          columns such as a site or batch code). When the attribute/product
          list is omitted, "all remaining columns" excludes these.

    Returns
    -------
    long_df : pandas.DataFrame
        Data in the canonical ``descriptive_long`` schema, sample-major sorted.
    checks : dict
        The round-trip invariants (grand mean, per-attribute and per-panelist
        max differences, cell counts) and ``ok``.

    Raises
    ------
    ValueError
        If required mapping columns are missing, the data is means-only (no
        panelist column), or a round-trip invariant fails (which signals a
        wrong column mapping).
    """
    if layout not in ("long", "wide_by_attribute", "wide_by_product"):
        raise ValueError(
            f"layout must be 'long', 'wide_by_attribute', or 'wide_by_product', got {layout!r}."
        )

    # Drop any nuisance columns up front, so "all remaining columns" (when the
    # attribute/product list is omitted) automatically excludes them.
    ignore = mapping.get("ignore") or []
    if ignore:
        data = data.drop(columns=[c for c in ignore if c in data.columns])

    panelist_col = mapping.get("panelist_id")
    if not panelist_col:
        raise ValueError(
            "mapping must name a 'panelist_id' column. Means-only data (no panelist column) cannot be "
            "used: the Mixed Assessor Model needs panelist-level scores."
        )
    if panelist_col not in data.columns:
        raise ValueError(f"mapping['panelist_id'] = {panelist_col!r} is not a column in the data.")
    product_col = mapping.get("product")
    session_col = mapping.get("session")
    replicate_col = mapping.get("replicate")

    # --- Pre-reshape marginal aggregates -------------------------------
    if layout == "long":
        attribute_col = mapping.get("attribute")
        score_col = mapping.get("score")
        for role, col in (("product", product_col), ("attribute", attribute_col), ("score", score_col)):
            if not col or col not in data.columns:
                raise ValueError(f"mapping[{role!r}] = {col!r} is not a column in the data (required for long layout).")
        long_df = data.rename(
            columns={
                panelist_col: "panelist_id",
                product_col: "product",
                attribute_col: "attribute",
                score_col: "score",
                **({session_col: "session"} if session_col else {}),
                **({replicate_col: "replicate"} if replicate_col else {}),
            }
        ).copy()
        long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
        grand_before = float(np.nanmean(long_df["score"].to_numpy()))
        n_cells_before = int(long_df["score"].notna().sum())
        attr_mean_before = _series_mean_map(long_df, "attribute", "score")
        panelist_mean_before = _series_mean_map(long_df, "panelist_id", "score")
    else:
        # Both wide layouts melt a block of value columns into one long column.
        # wide_by_attribute: rows carry the product, the columns are attributes.
        # wide_by_product: rows carry the attribute (a block label), the columns
        # are products - i.e. one panelist x product matrix per attribute, stacked.
        if layout == "wide_by_attribute":
            if not product_col or product_col not in data.columns:
                raise ValueError(f"mapping['product'] = {product_col!r} is required for wide_by_attribute layout.")
            row_key_col, melted_name, value_key = product_col, "attribute", "attributes"
        else:  # wide_by_product
            row_key_col = mapping.get("attribute")
            if not row_key_col or row_key_col not in data.columns:
                raise ValueError(f"mapping['attribute'] = {row_key_col!r} is required for wide_by_product layout.")
            melted_name, value_key = "product", "products"

        id_cols = [c for c in (panelist_col, row_key_col, session_col, replicate_col) if c]
        value_cols = mapping.get(value_key)
        if value_cols is None:
            value_cols = [c for c in data.columns if c not in id_cols]
        if not value_cols:
            raise ValueError(f"No {melted_name} columns found for {layout} layout.")
        missing_vals = [c for c in value_cols if c not in data.columns]
        if missing_vals:
            raise ValueError(f"These {melted_name} columns are not in the data: {missing_vals}.")

        wide = data.copy()
        for col in value_cols:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
        cell_block = wide[value_cols]
        grand_before = float(np.nanmean(cell_block.to_numpy()))
        n_cells_before = int(cell_block.notna().to_numpy().sum())
        panelist_mean_before = {
            str(pid): float(np.nanmean(grp[value_cols].to_numpy()))
            for pid, grp in wide.groupby(panelist_col, observed=True)
        }
        if layout == "wide_by_attribute":
            attr_mean_before = {str(a): float(np.nanmean(wide[a].to_numpy())) for a in value_cols}
        else:  # attribute is a row key, so group rows by it
            attr_mean_before = {
                str(attr): float(np.nanmean(grp[value_cols].to_numpy()))
                for attr, grp in wide.groupby(row_key_col, observed=True)
            }

        row_key_target = "product" if layout == "wide_by_attribute" else "attribute"
        long_df = wide.melt(
            id_vars=id_cols, value_vars=value_cols, var_name=melted_name, value_name="score"
        ).rename(columns={panelist_col: "panelist_id", row_key_col: row_key_target})
        if session_col:
            long_df = long_df.rename(columns={session_col: "session"})
        if replicate_col:
            long_df = long_df.rename(columns={replicate_col: "replicate"})

    # --- Defaults and dtypes -------------------------------------------
    if "session" not in long_df.columns:
        long_df["session"] = 1
    if "replicate" not in long_df.columns:
        long_df["replicate"] = 1
    for col in ("panelist_id", "product", "attribute"):
        long_df[col] = long_df[col].astype(str).str.strip()
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

    # --- Post-reshape marginal aggregates ------------------------------
    grand_after = float(np.nanmean(long_df["score"].to_numpy()))
    n_cells_after = int(long_df["score"].notna().sum())
    attr_mean_after = _series_mean_map(long_df, "attribute", "score")
    panelist_mean_after = _series_mean_map(long_df, "panelist_id", "score")

    grand_diff = abs(grand_before - grand_after)
    attr_diff = _compare_maps(attr_mean_before, attr_mean_after)
    panelist_diff = _compare_maps(panelist_mean_before, panelist_mean_after)
    tol = _ABS_TOL + _REL_TOL * abs(grand_before)
    ok = (
        grand_diff <= tol
        and attr_diff <= tol
        and panelist_diff <= tol
        and n_cells_before == n_cells_after
    )
    checks = {
        "ok": ok,
        "grand_mean_before": grand_before,
        "grand_mean_after": grand_after,
        "grand_mean_diff": grand_diff,
        "per_attribute_max_diff": attr_diff,
        "per_panelist_max_diff": panelist_diff,
        "n_cells_before": n_cells_before,
        "n_cells_after": n_cells_after,
    }
    if not ok:
        raise ValueError(
            "Round-trip check failed after reshaping; the column mapping is likely wrong. "
            f"grand_mean_diff={grand_diff:.3g}, per_attribute_max_diff={attr_diff:.3g}, "
            f"per_panelist_max_diff={panelist_diff:.3g}, "
            f"cells before/after={n_cells_before}/{n_cells_after}."
        )

    long_df = long_df.loc[:, list(DESCRIPTIVE_LONG_COLUMNS)]
    long_df = long_df.sort_values(_CANONICAL_SORT, kind="stable").reset_index(drop=True)
    return long_df, checks
