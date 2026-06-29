"""Tests for the descriptive panel-data pipeline in ``process_improve.sensory``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.sensory import (
    DESCRIPTIVE_LONG_COLUMNS,
    analyze_descriptive,
    panel_scorecard,
    validate_descriptive,
)
from process_improve.univariate.metrics import benjamini_hochberg

PRODUCTS = list("UVWXYZT")
FACTOR_1 = [-1, -1, 0, 0, 1, 1, 0]
FACTOR_2 = [-1, 1, -1, 1, -1, 1, 0]


def _design() -> pd.DataFrame:
    return pd.DataFrame({"product": PRODUCTS, "f1": FACTOR_1, "f2": FACTOR_2})


def _panel(*, anomalous: str | None = "P8", seed: int = 0) -> pd.DataFrame:
    """Build a panel where attribute A is driven by f1 and B is noise.

    The panelist named by ``anomalous`` (if any) scores at random, so it
    neither agrees with the panel nor discriminates between products.
    """
    rng = np.random.default_rng(seed)
    truth = dict(zip(PRODUCTS, FACTOR_1, strict=True))
    rows = []
    for pid in [f"P{i}" for i in range(1, 9)]:
        bias = rng.normal(0, 0.3)
        is_anom = pid == anomalous
        for prod in PRODUCTS:
            for attr in ("A", "B"):
                for rep in (1, 2):
                    if is_anom:
                        score = rng.normal(5, 1)
                    else:
                        base = 5 + (2 * truth[prod] if attr == "A" else 0)
                        score = base + bias + rng.normal(0, 0.4)
                    rows.append(
                        {
                            "panelist_id": pid,
                            "session": 1,
                            "product": prod,
                            "attribute": attr,
                            "replicate": rep,
                            "score": score,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_designed_happy_path():
    result = validate_descriptive(_panel(), _design(), mode="designed", score_min=0, score_max=10)
    assert result.ok
    assert result.errors == []
    assert result.content_hash is not None
    assert result.stats["n_products"] == len(PRODUCTS)
    assert set(DESCRIPTIVE_LONG_COLUMNS).issubset(result.normalized_df.columns)


def test_validate_missing_column_is_error():
    panel = _panel().drop(columns=["score"])
    result = validate_descriptive(panel, _design(), mode="designed")
    assert not result.ok
    assert any("missing required columns" in e for e in result.errors)
    assert result.normalized_df is None


def test_validate_missing_product_in_covariates_is_error():
    design = _design().iloc[:-1]  # drop product T
    result = validate_descriptive(_panel(), design, mode="designed")
    assert not result.ok
    assert any("no row in the covariate table" in e for e in result.errors)


def test_validate_out_of_range_score_warns():
    panel = _panel()
    panel.loc[0, "score"] = 99.0
    result = validate_descriptive(panel, _design(), mode="designed", score_min=0, score_max=10)
    assert result.ok
    assert any("outside the expected range" in w for w in result.warnings)


def test_validate_unbalanced_panel_errors():
    panel = _panel()
    # Punch random holes across the grid (every panelist/product/attribute is
    # still present, so the full grid stays the same size) to exceed the 20%
    # missing-cell error threshold.
    panel = panel.sample(frac=0.7, random_state=0).reset_index(drop=True)
    result = validate_descriptive(panel, _design(), mode="designed")
    assert not result.ok
    assert any("unbalanced" in e for e in result.errors)


def test_validate_bad_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        validate_descriptive(_panel(), _design(), mode="nonsense")


def test_content_hash_is_stable():
    a = validate_descriptive(_panel(), _design(), mode="designed")
    b = validate_descriptive(_panel(), _design(), mode="designed")
    assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


def test_bh_matches_statsmodels():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5, 0.9])
    expected = multipletests(p, alpha=0.05, method="fdr_bh")[1]
    got = benjamini_hochberg(p, alpha=0.05).p_adjusted
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_bh_q_values_are_monotone_in_rank():
    p = np.array([0.04, 0.001, 0.20, 0.01])
    q = benjamini_hochberg(p).p_adjusted
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_bh_empty_input():
    out = benjamini_hochberg([])
    assert out.p_adjusted.size == 0


# ---------------------------------------------------------------------------
# Panel scorecard
# ---------------------------------------------------------------------------


def test_scorecard_flags_planted_anomaly():
    card = panel_scorecard(_panel(anomalous="P8"))
    assert "P8" in card.flagged
    assert "P8" in card.reasons


def test_scorecard_clean_panel_has_no_flags():
    card = panel_scorecard(_panel(anomalous=None))
    assert card.flagged == []


def test_dropping_panelist_changes_means():
    validated = validate_descriptive(_panel(), _design(), mode="designed")
    kept = analyze_descriptive(validated, drop_panelists=None)
    dropped = analyze_descriptive(validated, drop_panelists="auto")
    assert "P8" in dropped.dropped
    assert kept.product_means.shape == dropped.product_means.shape
    merged = kept.product_means.merge(
        dropped.product_means, on=["product", "attribute"], suffixes=("_keep", "_drop")
    )
    assert not np.allclose(merged["mean_keep"], merged["mean_drop"])


# ---------------------------------------------------------------------------
# Relate: designed
# ---------------------------------------------------------------------------


def test_relate_designed_recovers_active_factor():
    validated = validate_descriptive(_panel(), _design(), mode="designed", score_min=0, score_max=10)
    result = analyze_descriptive(validated, drop_panelists="auto")
    terms = pd.DataFrame(result.relate["terms"])
    a_terms = terms[terms["attribute"] == "A"].set_index("factor")
    # f1 drives attribute A; f2 does not.
    assert a_terms.loc["f1", "significant"]
    assert not a_terms.loc["f2", "significant"]
    assert a_terms.loc["f1", "effect"] > a_terms.loc["f2", "effect"]


def test_relate_designed_q_values_monotone():
    validated = validate_descriptive(_panel(), _design(), mode="designed")
    result = analyze_descriptive(validated, drop_panelists="auto")
    terms = pd.DataFrame(result.relate["terms"]).sort_values("p_value")
    assert np.all(np.diff(terms["q_value"].to_numpy()) >= -1e-12)


def test_relate_designed_cross_checks_analyze_experiment():
    from process_improve.experiments.analysis import analyze_experiment
    from process_improve.sensory.analysis import aggregate_to_product

    validated = validate_descriptive(_panel(), _design(), mode="designed")
    agg = aggregate_to_product(validated.normalized_df)
    design = validated.covariates.copy()
    design["y"] = agg["A"].reindex(design.index).to_numpy()
    direct = analyze_experiment(
        design.reset_index(drop=True), response_column="y", analysis_type="coefficients", model="main_effects"
    )
    direct_f1 = next(c["coefficient"] for c in direct["coefficients"] if c["term"] == "f1")

    result = analyze_descriptive(validated, drop_panelists=None)
    terms = pd.DataFrame(result.relate["terms"])
    mine_f1 = terms[(terms["attribute"] == "A") & (terms["factor"] == "f1")]["coefficient"].iloc[0]
    assert mine_f1 == pytest.approx(direct_f1, rel=1e-6)


# ---------------------------------------------------------------------------
# Relate: observational
# ---------------------------------------------------------------------------


def test_relate_observational_finds_descriptor():
    rng = np.random.default_rng(3)
    obs = pd.DataFrame(
        {
            "product": PRODUCTS,
            "sodium": [0.2, 0.3, 0.5, 0.55, 0.9, 1.0, 0.5],  # tracks f1 / attribute A
            "fat": rng.normal(3, 1, len(PRODUCTS)),
        }
    )
    validated = validate_descriptive(_panel(), obs, mode="observational")
    result = analyze_descriptive(validated)
    assoc = pd.DataFrame(result.relate["associations"])
    a_sodium = assoc[(assoc["attribute"] == "A") & (assoc["descriptor"] == "sodium")].iloc[0]
    a_fat = assoc[(assoc["attribute"] == "A") & (assoc["descriptor"] == "fat")].iloc[0]
    assert a_sodium["significant"]
    assert abs(a_sodium["r"]) > abs(a_fat["r"])
    assert {"descriptor", "vip"}.issubset(pd.DataFrame(result.relate["vip"]).columns)


def test_relate_observational_requires_numeric_descriptors():
    obs = pd.DataFrame({"product": PRODUCTS, "grade": list("AABBCCA")})
    result = validate_descriptive(_panel(), obs, mode="observational")
    assert not result.ok
    assert any("must be numeric" in e for e in result.errors)


def test_analyze_refuses_unvalidated():
    bad = validate_descriptive(_panel().drop(columns=["score"]), _design(), mode="designed")
    with pytest.raises(ValueError, match="requires a validated dataset"):
        analyze_descriptive(bad)
