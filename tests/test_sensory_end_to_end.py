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

PRODUCTS = [f"Product {chr(ord('A') + i)}" for i in range(18)]
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


def _sig(values: np.ndarray, figures: int = 4) -> np.ndarray:
    """Round to a fixed number of significant figures (3-4, as a lab would report)."""
    return np.array([float(f"{v:.{figures}g}") for v in np.asarray(values, dtype=float)])


def _loguniform(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map u in [0, 1] onto [lo, hi] on a log scale (right-skewed, as in real product frames)."""
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def make_panel_and_covariates(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a wide panel table (integer scores, nuisance ``site`` column) and covariates.

    Each product has a hidden intensity ``u`` in [0, 1] per mechanism; the
    attribute means are linear in ``u`` while the instrumental covariates are
    realistic physical quantities monotone in the same ``u`` (so genuine
    covariates correlate with their attribute). The spurious covariates are
    coupled to the mechanistic ones (e.g. refractive index ~ 1.333 + 0.00142 *
    Brix), as they are physically, so they correlate in-sample without a direct
    causal path to perception. Panel scores are integers on a 0-10 scale.
    """
    rng = np.random.default_rng(seed)
    n = len(PRODUCTS)

    # Hidden per-product intensity (one per mechanism) and the attribute means.
    u = {key: rng.uniform(0.0, 1.0, n) for key in ATTRIBUTES}
    liking_u = np.clip(0.7 * u["Sweetness"] + 0.3 * (1 - u["Sourness"]) + rng.normal(0, 0.05, n), 0, 1)
    u["Liking"] = liking_u
    attr_mean = {attr: 1.0 + 8.0 * u[attr] for attr in ATTRIBUTES}  # means on a 1-9 band

    # Mechanistic covariates: realistic units / ranges, monotone in the same u.
    brix = 0.5 + 64.5 * u["Sweetness"]  # deg Bx, 0.5-65
    acidity = _loguniform(u["Sourness"], 0.3, 60.0)  # g/L, right-skewed
    polyphenols = _loguniform(u["Bitterness"], 50.0, 4000.0)  # mg/L GAE
    aroma_oav = _loguniform(u["Aroma intensity"], 1.0, 2000.0)  # OAV
    viscosity = _loguniform(u["Firmness"], 1.0, 10000.0)  # mPa.s
    # Spurious / proxy covariates, physically coupled to the mechanistic ones.
    refractive_index = 1.333 + 0.00142 * brix + rng.normal(0, 0.0008, n)  # rides on Brix
    specific_gravity = np.clip(0.998 + 0.0045 * brix + rng.normal(0, 0.004, n), 0.98, 1.35)  # rides on Brix
    conductivity = np.clip(0.05 + 0.45 * acidity + rng.normal(0, 0.3, n), 0.05, 30.0)  # rides on acid ions
    tds = np.clip(50 + 700 * brix + 200 * acidity + rng.normal(0, 200, n), 50, 50000)  # aggregate
    price = _loguniform(liking_u, 0.30, 50.0)  # EUR/L, artifact tracking Liking
    serving_temperature = rng.uniform(2.0, 70.0, n)  # measurement condition; unrelated

    covariates = pd.DataFrame(
        {
            "product": PRODUCTS,
            "brix": _sig(brix),
            "titratable_acidity": _sig(acidity),
            "polyphenols": _sig(polyphenols),
            "aroma_oav": _sig(aroma_oav),
            "viscosity": _sig(viscosity),
            "refractive_index": _sig(refractive_index),
            "specific_gravity": _sig(specific_gravity),
            "conductivity": _sig(conductivity),
            "total_dissolved_solids": _sig(tds),
            "price": _sig(price, 3),
            "serving_temperature": _sig(serving_temperature, 3),
        }
    )

    # Panel: integer 0-10 scores, with three planted atypical assessors.
    offset = dict(zip(ASSESSORS, rng.normal(0, 0.3, len(ASSESSORS)), strict=True))
    rows = []
    for j in ASSESSORS:
        for p_idx, product in enumerate(PRODUCTS):
            row: dict[str, object] = {
                "Assessor": j,
                "Product": product,
                "site": "Site1" if p_idx % 2 == 0 else "Site2",
            }
            for attr in ATTRIBUTES:
                mean_pa = attr_mean[attr][p_idx]
                centre = attr_mean[attr].mean()
                if j == "J07":  # random: no relation to the products
                    score = rng.normal(5.0, 1.5)
                elif j == "J09":  # compressor: uses half the range
                    score = centre + 0.5 * (mean_pa - centre) + offset[j] + rng.normal(0, 0.4)
                elif j == "J03":  # shifted high
                    score = mean_pa + 2.0 + rng.normal(0, 0.4)
                else:  # normal assessor
                    score = mean_pa + offset[j] + rng.normal(0, 0.4)
                row[attr] = int(np.clip(round(score), 0, 10))
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel[ATTRIBUTES] = panel[ATTRIBUTES].astype("Int64")  # integer ratings (nullable for missing cells)

    # A few missing cells, to exercise the balance and missing-data handling.
    for r, c in [(5, "Bitterness"), (17, "Aftertaste"), (40, "Firmness")]:
        panel.loc[r, c] = pd.NA

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
    assert _assoc_row(assoc, "Liking", "price")["significant"]
    # ... but so are the spurious proxies (the trap): they correlate in-sample.
    assert _assoc_row(assoc, "Sweetness", "refractive_index")["significant"]
    assert _assoc_row(assoc, "Sweetness", "specific_gravity")["significant"]
    # A descriptor with no causal path to an unrelated attribute stays non-significant.
    assert not _assoc_row(assoc, "Sourness", "brix")["significant"]
    # Every attribute is related, including the ones with missing cells - a NaN
    # cell must not silently drop an attribute (regression guard).
    assert set(assoc["attribute"]) == set(ATTRIBUTES)
    assert _assoc_row(assoc, "Bitterness", "polyphenols")["significant"]


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
