"""End-to-end tutorial example for the descriptive-panel pipeline.

A single synthetic scenario drives both this test and the documentation worked
example, so the two stay in agreement. The data is entirely synthetic and
generic: ten assessors score six products on nine attributes (wide layout, with
a nuisance ``site`` column and a few missing cells), and a per-product table of
instrumental covariates carries both genuine mechanistic correlates and
spurious proxies/artifacts. Three assessors are planted to misbehave:

* ``J07`` scores at random (disagrees with the panel),
* ``J03`` rates everything high (a location shift),
* ``J09`` uses only the middle of the scale (a compressed range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.sensory import (
    DESCRIPTIVE_LONG_COLUMNS,
    analyze_descriptive,
    mixed_assessor_model,
    panel_scorecard,
    reshape_to_long,
    validate_descriptive,
)

PRODUCTS = [f"Product {chr(ord('A') + i)}" for i in range(12)]
ATTRIBUTES = [
    "Aroma intensity",
    "Sweetness",
    "Sourness",
    "Bitterness",
    "Firmness",
    "Juiciness",
    "Colour intensity",
    "Aftertaste",
    "Liking",
]
ASSESSORS = [f"J{i:02d}" for i in range(1, 11)]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def make_panel_and_covariates(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a wide panel table (with a nuisance ``site`` column) and covariates."""
    rng = np.random.default_rng(seed)

    # True per-product mean for each attribute (products genuinely differ).
    latent = {attr: rng.uniform(3.0, 7.0, len(PRODUCTS)) for attr in ATTRIBUTES}
    # Liking is driven by sweetness (up) and sourness (down), plus noise.
    liking_signal = 0.8 * _zscore(latent["Sweetness"]) - 0.5 * _zscore(latent["Sourness"])
    latent["Liking"] = np.clip(5.0 + liking_signal + rng.normal(0, 0.2, len(PRODUCTS)), 1.0, 9.0)

    # Per-assessor baseline offset (normal assessors use the scale similarly).
    offset = dict(zip(ASSESSORS, rng.normal(0, 0.3, len(ASSESSORS)), strict=True))

    rows = []
    for j in ASSESSORS:
        for p_idx, product in enumerate(PRODUCTS):
            row = {"Assessor": j, "Product": product, "site": "Site1" if p_idx % 2 == 0 else "Site2"}
            for attr in ATTRIBUTES:
                mean_pa = latent[attr][p_idx]
                centre = latent[attr].mean()
                if j == "J07":  # random: no relation to the products
                    score = rng.normal(5.0, 1.5)
                elif j == "J09":  # compressor: uses half the range
                    score = centre + 0.5 * (mean_pa - centre) + offset[j] + rng.normal(0, 0.25)
                elif j == "J03":  # shifted high
                    score = mean_pa + 2.0 + rng.normal(0, 0.25)
                else:  # normal assessor
                    score = mean_pa + offset[j] + rng.normal(0, 0.25)
                row[attr] = float(np.clip(score, 0.0, 10.0))
            rows.append(row)
    panel = pd.DataFrame(rows)

    # A few missing cells, to exercise the balance handling.
    for r, c in [(5, "Bitterness"), (17, "Aftertaste"), (40, "Firmness")]:
        panel.loc[r, c] = np.nan

    # Instrumental covariates per product: genuine mechanistic correlates ...
    brix = latent["Sweetness"] + rng.normal(0, 0.08, len(PRODUCTS))
    titratable_acidity = latent["Sourness"] + rng.normal(0, 0.08, len(PRODUCTS))
    covariates = pd.DataFrame(
        {
            "product": PRODUCTS,
            "brix": brix,  # -> Sweetness
            "titratable_acidity": titratable_acidity,  # -> Sourness
            "polyphenols": latent["Bitterness"] + rng.normal(0, 0.08, len(PRODUCTS)),  # -> Bitterness
            "volatile_oav": latent["Aroma intensity"] + rng.normal(0, 0.08, len(PRODUCTS)),  # -> Aroma
            # ... and spurious proxies / artifacts that correlate in-sample only.
            "refractive_index": brix + rng.normal(0, 0.08, len(PRODUCTS)),  # rides on brix
            "density": brix + rng.normal(0, 0.10, len(PRODUCTS)),  # rides on brix
            "total_dissolved_solids": brix + titratable_acidity + rng.normal(0, 0.08, len(PRODUCTS)),  # aggregate
            "price_tier": latent["Liking"] + rng.normal(0, 0.08, len(PRODUCTS)),  # artifact, tracks Liking
        }
    )
    return panel, covariates


def _assoc_row(assoc: pd.DataFrame, attribute: str, descriptor: str) -> pd.Series:
    return assoc[(assoc["attribute"] == attribute) & (assoc["descriptor"] == descriptor)].iloc[0]


def test_pipeline_end_to_end():
    panel, covariates = make_panel_and_covariates()

    # Step 1: reshape wide -> long, ignoring the nuisance 'site' column.
    long_df, checks = reshape_to_long(
        panel,
        layout="wide_by_attribute",
        mapping={"panelist_id": "Assessor", "product": "Product", "ignore": ["site"]},
    )
    assert checks["ok"]
    assert list(long_df.columns) == list(DESCRIPTIVE_LONG_COLUMNS)
    assert set(long_df["attribute"]) == set(ATTRIBUTES)
    assert "site" not in long_df.columns

    # Step 2: validate.
    validated = validate_descriptive(long_df, covariates, mode="observational", score_min=0, score_max=10)
    assert validated.ok

    # Step 3: panel check - the random assessor is flagged.
    card = panel_scorecard(long_df)
    assert "J07" in card.flagged

    # The Mixed Assessor Model recovers the planted scale usage on Sweetness.
    mam = mixed_assessor_model(long_df)
    sweet = mam.scaling[mam.scaling["attribute"] == "Sweetness"].set_index("panelist_id")
    assert sweet.loc["J09", "beta"] < 0.7  # compressor
    assert sweet.loc["J03", "offset"] == sweet["offset"].max()  # high rater
    normal = sweet.drop(["J03", "J07", "J09"])
    assert (normal["beta"].sub(1).abs() < 0.35).all()  # the rest track the panel

    # Once the random assessor is removed, the leftover assessor-by-product
    # interaction is mostly scale usage, so the MAM product F-test (which uses
    # the genuine disagreement as the error) beats the classical one.
    cleaned = long_df[long_df["panelist_id"] != "J07"]
    mam_clean = mixed_assessor_model(cleaned)
    ftest = mam_clean.ftests[mam_clean.ftests["attribute"] == "Sweetness"].iloc[0]
    assert ftest["f_product_mam"] > ftest["f_product_classical"]

    # Step 4: correct (align all assessors) and relate to the covariates.
    result = analyze_descriptive(validated, correction="align")
    assert result.correction == "align"
    assoc = pd.DataFrame(result.relate["associations"])

    # Genuine correlates are significant ...
    assert _assoc_row(assoc, "Sweetness", "brix")["significant"]
    assert _assoc_row(assoc, "Sourness", "titratable_acidity")["significant"]
    assert _assoc_row(assoc, "Liking", "price_tier")["significant"]
    # ... but so are the spurious proxies (the trap): they correlate in-sample.
    assert _assoc_row(assoc, "Sweetness", "refractive_index")["significant"]
    assert _assoc_row(assoc, "Sweetness", "density")["significant"]
    # A descriptor with no causal path to an unrelated attribute stays non-significant.
    assert not _assoc_row(assoc, "Sourness", "brix")["significant"]


def test_means_only_table_is_refused():
    # A product-by-attribute summary (no panelist column) cannot drive the MAM.
    panel, _ = make_panel_and_covariates()
    means_only = panel.drop(columns=["Assessor"]).groupby("Product", as_index=False)[ATTRIBUTES].mean()
    with pytest.raises(ValueError, match="panelist"):
        reshape_to_long(
            means_only,
            layout="wide_by_attribute",
            mapping={"product": "Product", "ignore": ["site"]},
        )
