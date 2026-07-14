"""Tests for the descriptive panel-data pipeline in ``process_improve.sensory``.

Only the observational relate mode is implemented for now; the designed (DoE)
mode is a stub and is covered by the not-implemented tests below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.sensory import (
    DESCRIPTIVE_LONG_COLUMNS,
    align_scores,
    analyze_descriptive,
    mixed_assessor_model,
    panel_scorecard,
    validate_descriptive,
)
from process_improve.sensory.analysis import _collinear_clusters, discriminate_observational, relate_designed
from process_improve.sensory.ingest import reshape_to_long
from process_improve.univariate.metrics import benjamini_hochberg

PRODUCTS = list("UVWXYZT")
# Attribute A is built to vary across products with this pattern, so an
# observational descriptor that tracks it should be recoverable.
ATTR_A_DRIVER = [-1, -1, 0, 0, 1, 1, 0]


def _obs(*, seed: int = 3, drop_last: bool = False) -> pd.DataFrame:
    """Observational descriptors: ``sodium`` tracks attribute A; ``fat`` is noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "product": PRODUCTS,
            "sodium": [0.2, 0.3, 0.5, 0.55, 0.9, 1.0, 0.5],
            "fat": rng.normal(3, 1, len(PRODUCTS)),
        }
    )
    return df.iloc[:-1] if drop_last else df


def _panel(*, anomalous: str | None = "P8", seed: int = 0) -> pd.DataFrame:
    """Build a panel where attribute A varies by product and B is noise.

    The panelist named by ``anomalous`` (if any) scores at random, so it
    neither agrees with the panel nor discriminates between products.
    """
    rng = np.random.default_rng(seed)
    truth = dict(zip(PRODUCTS, ATTR_A_DRIVER, strict=True))
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


def test_validate_observational_happy_path():
    result = validate_descriptive(_panel(), _obs(), mode="observational", score_min=0, score_max=10)
    assert result.ok
    assert result.errors == []
    assert result.content_hash is not None
    assert result.stats["n_products"] == len(PRODUCTS)
    assert set(DESCRIPTIVE_LONG_COLUMNS).issubset(result.normalized_df.columns)


def test_validate_designed_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        validate_descriptive(_panel(), _obs(), mode="designed")


def test_validate_missing_column_is_error():
    panel = _panel().drop(columns=["score"])
    result = validate_descriptive(panel, _obs(), mode="observational")
    assert not result.ok
    assert any("missing required columns" in e for e in result.errors)
    assert result.normalized_df is None


def test_validate_missing_product_in_covariates_drops_and_warns():
    # A panel product with no covariate row cannot be related (the relate looks
    # covariates up by product), so it is dropped with a warning and the relate
    # proceeds on the matched intersection, rather than blocking.
    result = validate_descriptive(_panel(), _obs(drop_last=True), mode="observational")
    assert result.ok
    assert result.errors == []
    assert any("no row in the covariate table" in w and "dropped" in w for w in result.warnings)
    # The unmatched product is gone from the normalised panel.
    assert PRODUCTS[-1] not in set(result.normalized_df["product"])
    assert result.stats["n_products"] == len(PRODUCTS) - 1


def test_validate_no_products_match_is_error():
    # When nothing lines up at all, that is a genuine blocking error.
    obs = _obs()
    obs["product"] = [f"other-{p}" for p in obs["product"]]
    result = validate_descriptive(_panel(), obs, mode="observational")
    assert not result.ok
    assert any("No panel product has a matching row" in e for e in result.errors)


def test_validate_capitalised_product_column_still_aligns():
    # A covariate table whose identifier column is not exactly lowercase
    # "product" (e.g. "Product") must still align, not fall back to the index.
    obs = _obs().rename(columns={"product": "Product"})
    result = validate_descriptive(_panel(), obs, mode="observational")
    assert result.ok, result.errors
    assert result.covariates.index.name == "product"
    assert set(result.covariates.index) == set(PRODUCTS)


def test_validate_duplicate_covariate_rows_collapse_with_warning():
    obs = _obs()
    dup = pd.concat([obs, obs.iloc[[0]].assign(sodium=obs.iloc[0]["sodium"] + 0.4)], ignore_index=True)
    result = validate_descriptive(_panel(), dup, mode="observational")
    assert result.ok, result.errors
    assert any("duplicate rows" in w for w in result.warnings)
    # Index is unique after collapsing; the duplicated product keeps one (mean) row.
    assert result.covariates.index.is_unique
    assert result.covariates.loc[PRODUCTS[0], "sodium"] == pytest.approx(obs.iloc[0]["sodium"] + 0.2)


def test_validate_out_of_range_score_warns():
    panel = _panel()
    panel.loc[0, "score"] = 99.0
    result = validate_descriptive(panel, _obs(), mode="observational", score_min=0, score_max=10)
    assert result.ok
    assert any("outside the expected range" in w for w in result.warnings)


def test_validate_unbalanced_panel_warns_but_does_not_block():
    # Punch random holes across the grid to exceed the 20% missing-cell
    # threshold. Because the relate aggregates to product means, imbalance is
    # surfaced as a warning and does not block the analysis.
    panel = _panel().sample(frac=0.7, random_state=0).reset_index(drop=True)
    result = validate_descriptive(panel, _obs(), mode="observational")
    assert result.ok
    assert not any("unbalanced" in e for e in result.errors)
    assert any("unbalanced" in w for w in result.warnings)


def test_validate_bad_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        validate_descriptive(_panel(), _obs(), mode="nonsense")


def test_content_hash_is_stable():
    a = validate_descriptive(_panel(), _obs(), mode="observational")
    b = validate_descriptive(_panel(), _obs(), mode="observational")
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
    validated = validate_descriptive(_panel(), _obs(), mode="observational")
    kept = analyze_descriptive(validated, drop_panelists=None, discriminator=False)
    dropped = analyze_descriptive(validated, drop_panelists="auto", discriminator=False)
    assert "P8" in dropped.dropped
    assert kept.product_means.shape == dropped.product_means.shape
    merged = kept.product_means.merge(
        dropped.product_means, on=["product", "attribute"], suffixes=("_keep", "_drop")
    )
    assert not np.allclose(merged["mean_keep"], merged["mean_drop"])


# ---------------------------------------------------------------------------
# Relate: designed (stub)
# ---------------------------------------------------------------------------


def test_relate_designed_is_stub():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        relate_designed(pd.DataFrame(), pd.DataFrame())


# ---------------------------------------------------------------------------
# Relate: observational
# ---------------------------------------------------------------------------


def test_relate_observational_finds_descriptor():
    validated = validate_descriptive(_panel(), _obs(), mode="observational")
    result = analyze_descriptive(validated, discriminator=False)
    assoc = pd.DataFrame(result.relate["associations"])
    a_sodium = assoc[(assoc["attribute"] == "A") & (assoc["descriptor"] == "sodium")].iloc[0]
    a_fat = assoc[(assoc["attribute"] == "A") & (assoc["descriptor"] == "fat")].iloc[0]
    assert a_sodium["significant"]
    assert abs(a_sodium["r"]) > abs(a_fat["r"])
    assert {"descriptor", "vip"}.issubset(pd.DataFrame(result.relate["vip"]).columns)


def test_relate_observational_q_values_monotone():
    validated = validate_descriptive(_panel(), _obs(), mode="observational")
    result = analyze_descriptive(validated, discriminator=False)
    assoc = pd.DataFrame(result.relate["associations"]).sort_values("p_value")
    assert np.all(np.diff(assoc["q_value"].to_numpy()) >= -1e-12)


def test_collinear_clusters_groups_correlated_descriptors():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(20)
    block = pd.DataFrame(
        {"a": base, "b": base + 0.001 * rng.standard_normal(20), "c": rng.standard_normal(20)}
    )
    clusters = _collinear_clusters(block, threshold=0.95)
    assert clusters["a"] == clusters["b"]  # near-identical columns group together
    assert clusters["c"] != clusters["a"]  # an independent column is its own cluster


def test_discriminator_gate_and_clusters():
    products = [f"P{i}" for i in range(9)]
    rng = np.random.default_rng(3)
    u = np.linspace(0.0, 1.0, 9) + rng.normal(0, 0.02, 9)
    agg = pd.DataFrame(
        {"A": 2.0 * u + rng.normal(0, 0.05, 9), "B": rng.normal(0, 1, 9)}, index=products
    )
    cov = pd.DataFrame(
        {"d1": u, "d2": u + 0.005 * rng.normal(0, 1, 9), "d3": rng.normal(0, 1, 9)}, index=products
    )
    disc = discriminate_observational(agg, cov, n_components=1, n_permutations=49, random_state=0)

    # The collinear pair shares a cluster; the noise descriptor does not.
    assert disc["clusters"]["d1"] == disc["clusters"]["d2"] != disc["clusters"]["d3"]

    gate = pd.DataFrame(disc["per_attribute"]).set_index("attribute")
    assert bool(gate.loc["A", "predictable"])  # A is driven by d1/d2
    assert not bool(gate.loc["B", "predictable"])  # B is noise, not predictable

    desc = pd.DataFrame(disc["descriptors"])
    assert set(desc.columns) >= {"selectivity_ratio", "p_value", "q_value", "discriminator_significant"}
    # Nothing is flagged for the unpredictable attribute, and the noise
    # descriptor is never flagged.
    assert not desc[desc["attribute"] == "B"]["discriminator_significant"].any()
    assert not desc[desc["descriptor"] == "d3"]["discriminator_significant"].any()


def test_relate_observational_requires_numeric_descriptors():
    obs = pd.DataFrame({"product": PRODUCTS, "grade": list("AABBCCA")})
    result = validate_descriptive(_panel(), obs, mode="observational")
    assert not result.ok
    assert any("must be numeric" in e for e in result.errors)


def test_analyze_refuses_unvalidated():
    bad = validate_descriptive(_panel().drop(columns=["score"]), _obs(), mode="observational")
    with pytest.raises(ValueError, match="requires a validated dataset"):
        analyze_descriptive(bad)


# ---------------------------------------------------------------------------
# Mixed Assessor Model: scaling and alignment
# ---------------------------------------------------------------------------


def _scaling_panel(*, seed: int = 0, n_products: int = 8):
    """Panel where P0 compresses (beta<1), P1 expands (beta>1), P2 rates high."""
    rng = np.random.default_rng(seed)
    products = [f"prod{i}" for i in range(n_products)]
    effect = dict(zip(products, rng.normal(0, 2, n_products), strict=True))
    betas = {"P0": 0.4, "P1": 1.6}
    offsets = {"P2": 2.0}
    rows = []
    for pid in [f"P{i}" for i in range(8)]:
        slope = betas.get(pid, 1.0)
        off = offsets.get(pid, 0.0)
        for prod in products:
            for rep in (1, 2):
                rows.append(  # noqa: PERF401
                    {
                        "panelist_id": pid,
                        "session": 1,
                        "product": prod,
                        "attribute": "A",
                        "replicate": rep,
                        "score": 5 + off + slope * (2 * effect[prod]) + rng.normal(0, 0.25),
                    }
                )
    return pd.DataFrame(rows)


def test_mam_recovers_scaling_coefficients():
    mam = mixed_assessor_model(_scaling_panel())
    beta = mam.scaling.set_index("panelist_id")["beta"]
    assert beta["P0"] < 0.7  # compressor
    assert beta["P1"] > 1.3  # expander
    others = beta.drop(["P0", "P1"])
    assert (others.abs().sub(1).abs() < 0.3).all()  # the rest use the scale like the panel
    offset = mam.scaling.set_index("panelist_id")["offset"]
    assert offset["P2"] == max(offset)  # the high rater has the largest offset


def test_mam_ftest_more_powerful_than_classical():
    mam = mixed_assessor_model(_scaling_panel())
    row = mam.ftests.iloc[0]
    # Removing scaling heterogeneity shrinks the error term, so the MAM product
    # F-statistic exceeds the classical one.
    assert row["f_product_mam"] > row["f_product_classical"]


def test_align_harmonizes_all_panelists():
    panel = _scaling_panel()
    aligned = align_scores(panel, method="both")
    beta_after = mixed_assessor_model(aligned).scaling.set_index("panelist_id")["beta"]
    # After alignment everyone uses the scale like the panel (beta ~ 1).
    assert (beta_after.sub(1).abs() < 0.2).all()


def test_align_is_approximately_idempotent():
    # A second alignment barely moves the scores: the first pass already brought
    # every panelist's scaling to ~1, so re-aligning applies only a tiny residual.
    panel = _scaling_panel()
    once = align_scores(panel, method="both")
    twice = align_scores(once, method="both")
    merged = once.merge(
        twice, on=["panelist_id", "product", "attribute", "replicate", "session"], suffixes=("_1", "_2")
    )
    assert np.allclose(merged["score_1"], merged["score_2"], atol=0.05)


def test_align_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        align_scores(_scaling_panel(), method="nonsense")


def test_analyze_correction_align_changes_means_and_reports_mam():
    panel = _scaling_panel()
    obs = pd.DataFrame({"product": sorted(panel["product"].unique()), "d": range(panel["product"].nunique())})
    validated = validate_descriptive(panel, obs, mode="observational")
    none = analyze_descriptive(validated, correction="none", discriminator=False)
    aligned = analyze_descriptive(validated, correction="align", discriminator=False)
    assert aligned.correction == "align"
    assert not aligned.mam.scaling.empty
    merged = none.product_means.merge(
        aligned.product_means, on=["product", "attribute"], suffixes=("_none", "_align")
    )
    assert not np.allclose(merged["mean_none"], merged["mean_align"])


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def test_tool_panel_check_returns_scorecard_and_mam():
    import json

    from process_improve.tool_spec import execute_tool_call

    panel = _scaling_panel().to_dict(orient="records")
    out = execute_tool_call("sensory_panel_check", {"panel": panel, "align": True})
    json.dumps(out)  # must be JSON-serialisable for the front end
    assert out["ok"]
    beta = {r["panelist_id"]: r["beta"] for r in out["mam"]["scaling"]}
    assert beta["P0"] < 0.7  # compressor recovered through the tool
    assert beta["P1"] > 1.3  # expander
    assert "aligned_panel" in out
    assert {"scorecard", "ftests"}.issubset({*out, *out["mam"]})


def test_tool_panel_check_missing_columns():
    from process_improve.tool_spec import execute_tool_call

    out = execute_tool_call("sensory_panel_check", {"panel": [{"panelist_id": "P1", "score": 5}]})
    assert not out["ok"]
    assert any("missing required columns" in e for e in out["errors"])


def _wide_panel(*, seed: int = 0):
    """Wide-by-attribute table: rows = assessor x sample x rep, one column per attribute."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in ["P1", "P2", "P3"]:
        for prod in ["A", "B", "C"]:
            for rep in (1, 2):
                rows.append(  # noqa: PERF401
                    {
                        "Assessor": pid,
                        "Sample": prod,
                        "Rep": rep,
                        "Salty": rng.normal(5, 1),
                        "Bitter": rng.normal(3, 1),
                    }
                )
    return pd.DataFrame(rows)


def test_reshape_wide_to_long_roundtrip():
    wide = _wide_panel()
    long_df, checks = reshape_to_long(
        wide,
        layout="wide_by_attribute",
        mapping={"panelist_id": "Assessor", "product": "Sample", "replicate": "Rep"},
    )
    assert checks["ok"]
    assert list(long_df.columns) == list(DESCRIPTIVE_LONG_COLUMNS)
    assert long_df.shape[0] == 3 * 3 * 2 * 2  # panelists x products x reps x attributes
    # Grand mean preserved and rows in canonical sample-major order.
    assert checks["grand_mean_before"] == pytest.approx(checks["grand_mean_after"])
    canonical = long_df.sort_values(
        ["product", "attribute", "panelist_id", "session", "replicate"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(long_df, canonical)


def test_reshape_wide_by_product_roundtrip():
    # One panelist x product matrix per attribute, stacked with an attribute label column.
    rng = np.random.default_rng(1)
    rows = []
    for attr in ("Salty", "Bitter"):
        for pid in ("P1", "P2", "P3"):
            for rep in (1, 2):
                rows.append(  # noqa: PERF401
                    {"Assessor": pid, "Attribute": attr, "Rep": rep, "A": rng.normal(5, 1), "B": rng.normal(4, 1)}
                )
    matrices = pd.DataFrame(rows)
    long_df, checks = reshape_to_long(
        matrices,
        layout="wide_by_product",
        mapping={"panelist_id": "Assessor", "attribute": "Attribute", "replicate": "Rep"},
    )
    assert checks["ok"]
    assert list(long_df.columns) == list(DESCRIPTIVE_LONG_COLUMNS)
    assert long_df.shape[0] == 2 * 3 * 2 * 2  # attributes x panelists x reps x products
    assert set(long_df["product"]) == {"A", "B"}
    assert set(long_df["attribute"]) == {"Salty", "Bitter"}
    assert checks["grand_mean_before"] == pytest.approx(checks["grand_mean_after"])


def test_reshape_defaults_session_and_replicate():
    wide = _wide_panel().drop(columns=["Rep"])
    long_df, _ = reshape_to_long(
        wide, layout="wide_by_attribute", mapping={"panelist_id": "Assessor", "product": "Sample"}
    )
    assert (long_df["session"] == 1).all()
    assert (long_df["replicate"] == 1).all()


def test_reshape_long_passthrough_is_canonical():
    panel = _panel(anomalous=None).rename(
        columns={"panelist_id": "who", "product": "sample", "attribute": "attr", "score": "value"}
    )
    long_df, checks = reshape_to_long(
        panel,
        layout="long",
        mapping={
            "panelist_id": "who",
            "product": "sample",
            "attribute": "attr",
            "score": "value",
            "session": "session",
            "replicate": "replicate",
        },
    )
    assert checks["ok"]
    assert set(long_df["attribute"]) == {"A", "B"}


def test_reshape_means_only_is_refused():
    with pytest.raises(ValueError, match="panelist"):
        reshape_to_long(
            pd.DataFrame({"Sample": ["A", "B"], "Salty": [5.0, 6.0]}),
            layout="wide_by_attribute",
            mapping={"product": "Sample", "attributes": ["Salty"]},
        )


def test_reshape_missing_attribute_column_raises():
    with pytest.raises(ValueError, match="not in the data"):
        reshape_to_long(
            _wide_panel(),
            layout="wide_by_attribute",
            mapping={"panelist_id": "Assessor", "product": "Sample", "attributes": ["Salty", "Sweetness"]},
        )


def test_validate_hash_is_order_independent():
    wide = _wide_panel()
    long_df, _ = reshape_to_long(
        wide, layout="wide_by_attribute", mapping={"panelist_id": "Assessor", "product": "Sample", "replicate": "Rep"}
    )
    cov = pd.DataFrame({"product": ["A", "B", "C"], "d": [1.0, 2.0, 3.0]})
    h1 = validate_descriptive(long_df, cov, mode="observational").content_hash
    shuffled = long_df.sample(frac=1, random_state=3).reset_index(drop=True)
    h2 = validate_descriptive(shuffled, cov, mode="observational").content_hash
    assert h1 == h2


def test_tool_reshape_to_long_dispatch():
    from process_improve.tool_spec import execute_tool_call

    wide = _wide_panel().to_dict(orient="records")
    out = execute_tool_call(
        "sensory_reshape_to_long",
        {
            "data": wide,
            "layout": "wide_by_attribute",
            "panelist_id": "Assessor",
            "product": "Sample",
            "replicate": "Rep",
        },
    )
    assert out["ok"]
    assert out["checks"]["ok"]
    assert len(out["long"]) == 36
    bad = execute_tool_call(
        "sensory_reshape_to_long",
        {"data": wide, "layout": "wide_by_attribute", "panelist_id": "Assessor", "product": "Missing"},
    )
    assert not bad["ok"]


def test_tool_analyze_exposes_correction_and_mam():
    from process_improve.tool_spec import execute_tool_call

    panel = _scaling_panel()
    products = sorted(panel["product"].unique())
    payload = {
        "panel": panel.to_dict(orient="records"),
        "covariates": [{"product": p, "d": i} for i, p in enumerate(products)],
        "mode": "observational",
        "correction": "align",
        "discriminator": False,
    }
    out = execute_tool_call("sensory_analyze_descriptive", payload)
    assert out["ok"]
    assert out["correction"] == "align"
    ftest = out["mam"]["ftests"][0]
    assert ftest["f_product_mam"] > ftest["f_product_classical"]
