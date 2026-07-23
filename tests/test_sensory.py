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
    panel_consistency,
    panel_scorecard,
    validate_descriptive,
)
from process_improve.sensory.analysis import (
    _collinear_clusters,
    _jackknife_correlation,
    discriminate_observational,
    permutation_column_null,
    relate_designed,
)
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


# Products for the high-leverage case study. ``Q5`` is both a response outlier on
# attribute C and the sole product carrying the ``spike`` descriptor.
LEVERAGE_PRODUCTS = [f"Q{i}" for i in range(12)]
_SPIKE_INDEX = 5


def _leverage_case(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel + covariates isolating a genuine driver from a single-support spike.

    ``genuine`` varies smoothly across all products and drives attribute A.
    ``spike`` is zero on every product except one (``Q5``); attribute C is flat
    except at that same product, where it is a large outlier. The spike-vs-C
    correlation is therefore created entirely by one high-leverage product: strong
    in-sample, but it collapses the moment that product is removed. An
    influence-robust relate must keep genuine-on-A and demote spike-on-C.
    """
    rng = np.random.default_rng(seed)
    n = len(LEVERAGE_PRODUCTS)
    genuine = np.linspace(0.2, 1.0, n)
    spike = np.zeros(n)
    spike[_SPIKE_INDEX] = 1.0
    cov = pd.DataFrame(
        {
            "product": LEVERAGE_PRODUCTS,
            "genuine": genuine,
            "spike": spike,
            "noise": rng.normal(0.0, 1.0, n),
        }
    )
    a_mean = 4.0 * genuine
    c_mean = np.zeros(n)
    c_mean[_SPIKE_INDEX] = 5.0  # one product is a large outlier on attribute C
    rows = []
    for pid in [f"J{i}" for i in range(6)]:
        bias = rng.normal(0.0, 0.2)
        for j, prod in enumerate(LEVERAGE_PRODUCTS):
            for rep in (1, 2):
                rows.append(
                    {"panelist_id": pid, "session": 1, "product": prod, "attribute": "A",
                     "replicate": rep, "score": 5.0 + a_mean[j] + bias + rng.normal(0, 0.15)}
                )
                rows.append(
                    {"panelist_id": pid, "session": 1, "product": prod, "attribute": "C",
                     "replicate": rep, "score": 5.0 + c_mean[j] + bias + rng.normal(0, 0.15)}
                )
    return pd.DataFrame(rows), cov


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


def test_consistency_flags_the_noise_rater():
    """A panelist scoring at random is indistinguishable from a reshuffle of its own scores."""
    result = panel_consistency(_panel(anomalous="P8"), n_permutations=199, random_state=0)
    table = result.table
    assert {"discrimination", "p_discrimination", "agreement", "p_agreement", "consistent"} == set(table.columns)

    # The random rater cannot beat its own permutations: high discrimination p-value,
    # not consistent, and the least consistent of the whole panel.
    assert "P8" in result.inconsistent
    assert table.loc["P8", "p_discrimination"] > 0.5
    assert table["p_discrimination"].idxmax() == "P8"
    assert table.loc["P8", "agreement"] < 0.5

    # The strongest genuine raters clear their own-permutation null.
    assert table.loc["P5", "consistent"]
    assert table.loc["P4", "consistent"]


def _row(panelist: str, product: str, attribute: str, replicate: int, score: float) -> dict:
    """Build a single descriptive_long row."""
    return {
        "panelist_id": panelist,
        "session": 1,
        "product": product,
        "attribute": attribute,
        "replicate": replicate,
        "score": score,
    }


def test_consistency_handles_degenerate_panelists():
    """A flat rater and a single-product rater cannot be assessed and are not consistent."""
    products = list("WXYZ")
    truth = {"W": -1, "X": 0, "Y": 1, "Z": 2}
    rows = [_row("good", prod, "A", rep, 5.0 + 2.0 * truth[prod]) for prod in products for rep in (1, 2)]
    rows += [_row("flat", prod, "A", rep, 5.0) for prod in products for rep in (1, 2)]  # no variance
    rows += [_row("one", "W", "A", rep, 6.0) for rep in (1, 2)]  # only one product
    result = panel_consistency(pd.DataFrame(rows), n_permutations=50, random_state=0)
    table = result.table

    assert np.isnan(table.loc["flat", "p_discrimination"])
    assert np.isnan(table.loc["flat", "agreement"])
    assert not bool(table.loc["flat", "consistent"])
    assert not bool(table.loc["one", "consistent"])
    assert {"flat", "one"}.issubset(result.inconsistent)


def test_consistency_is_deterministic_and_validates():
    """Same seed gives identical output; a non-positive permutation count is rejected."""
    panel = _panel(anomalous="P8")
    first = panel_consistency(panel, n_permutations=99, random_state=1)
    second = panel_consistency(panel, n_permutations=99, random_state=1)
    assert first.table.equals(second.table)
    assert first.inconsistent == second.inconsistent
    with pytest.raises(ValueError, match="n_permutations must be at least 1"):
        panel_consistency(panel, n_permutations=0)


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


def test_jackknife_correlation_edge_cases():
    """The jackknife helper's degenerate branches behave sensibly."""
    rng = np.random.default_rng(0)

    # Too few observations: the jackknife is undefined, so not robust.
    se, robust, n = _jackknife_correlation(np.arange(3.0), np.arange(3.0) + 1.0, 0.05)
    assert n == 3
    assert not robust
    assert np.isnan(se)

    # A smooth relationship supported across all points is robust.
    x = np.linspace(0.0, 1.0, 12)
    y = 2.0 * x + rng.normal(0, 0.05, 12)
    se, robust, n = _jackknife_correlation(x, y, 0.05)
    assert robust
    assert se > 0.0

    # A single-support spike collapses when its one point is removed: not robust.
    xs = np.zeros(12)
    xs[5] = 1.0
    ys = rng.normal(0, 1, 12)
    ys[5] += 5.0
    _, robust, _ = _jackknife_correlation(xs, ys, 0.05)
    assert not robust

    # A perfectly stable correlation (zero jackknife spread) is robust when non-zero.
    z = np.linspace(0.0, 1.0, 8)
    se, robust, _ = _jackknife_correlation(z, z, 0.05)
    assert se == 0.0
    assert robust


def test_jackknife_delete_two_demotes_two_support_spike():
    """Leave-one-out keeps a two-point spike; delete-two removes it, genuine survives."""
    rng = np.random.default_rng(0)
    n = 15
    # A predictor non-zero on exactly two products, both extreme on the response.
    xs = np.zeros(n)
    xs[13] = xs[14] = 1.0
    ys = rng.normal(0, 1, n)
    ys[13] += 6.0
    ys[14] += 6.0
    _, robust_d1, _ = _jackknife_correlation(xs, ys, 0.05, max_deletions=1)
    _, robust_d2, _ = _jackknife_correlation(xs, ys, 0.05, max_deletions=2)
    assert robust_d1  # leave-one-out is blind to a two-point effect
    assert not robust_d2  # removing both supporting points collapses it

    # A genuine broad driver survives both.
    xg = np.linspace(0.0, 1.0, n)
    yg = 2.0 * xg + rng.normal(0, 0.2, n)
    _, g1, _ = _jackknife_correlation(xg, yg, 0.05, max_deletions=1)
    _, g2, _ = _jackknife_correlation(xg, yg, 0.05, max_deletions=2)
    assert g1
    assert g2

    with pytest.raises(ValueError, match="max_deletions must be at least 1"):
        _jackknife_correlation(xg, yg, 0.05, max_deletions=0)


def test_jackknife_breakdown_demotes_via_nonsignificance_and_sign_flip():
    """Delete-two demotes through the non-significance and sign-flip paths, not only support removal."""
    # Moderate continuous correlation (no zeros, so no deletion makes a constant column):
    # robust to any single deletion, but removing the two most influential points drops
    # the remaining correlation below significance while keeping its sign.
    x = np.array([-1.1, -0.81, -0.78, -0.75, -0.73, -0.45, -0.25, 0.13, 0.27, 0.48, 0.84, 0.86])
    y = np.array([-1.69, -0.9, -2.43, -2.68, -0.21, -3.6, -0.55, 0.85, -1.41, -1.42, -0.45, 1.96])
    _, a1, _ = _jackknife_correlation(x, y, 0.05, max_deletions=1)
    _, a2, _ = _jackknife_correlation(x, y, 0.05, max_deletions=2)
    assert a1
    assert not a2

    # Two extreme points create a positive correlation; removing them leaves a negative
    # trend, so the effect reverses direction under a two-point deletion.
    xb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21.0])
    yb = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 30, 31.0])
    _, b1, _ = _jackknife_correlation(xb, yb, 0.05, max_deletions=1)
    _, b2, _ = _jackknife_correlation(xb, yb, 0.05, max_deletions=2)
    assert b1
    assert not b2


def test_relate_influence_deletions_two_demotes_two_support_spike():
    """`influence_deletions=2` demotes a two-product spike the default gate keeps."""
    n = len(LEVERAGE_PRODUCTS)
    rng = np.random.default_rng(3)
    genuine = np.linspace(0.2, 1.0, n)
    two_spike = np.zeros(n)
    two_spike[_SPIKE_INDEX] = two_spike[_SPIKE_INDEX + 1] = 1.0

    a_mean = 4.0 * genuine
    # Attribute C is flat except at the two products that carry the spike.
    c_mean = np.zeros(n)
    c_mean[_SPIKE_INDEX] = c_mean[_SPIKE_INDEX + 1] = 5.0
    rows = []
    for pid in [f"J{i}" for i in range(6)]:
        bias = rng.normal(0.0, 0.2)
        for j, prod in enumerate(LEVERAGE_PRODUCTS):
            for rep in (1, 2):
                rows.append({"panelist_id": pid, "session": 1, "product": prod, "attribute": "A",
                             "replicate": rep, "score": 5.0 + a_mean[j] + bias + rng.normal(0, 0.15)})
                rows.append({"panelist_id": pid, "session": 1, "product": prod, "attribute": "C",
                             "replicate": rep, "score": 5.0 + c_mean[j] + bias + rng.normal(0, 0.15)})
    panel = pd.DataFrame(rows)
    cov = pd.DataFrame({"product": LEVERAGE_PRODUCTS, "genuine": genuine, "two_spike": two_spike})
    validated = validate_descriptive(panel, cov, mode="observational")

    def spike_robust(deletions: int) -> bool:
        result = analyze_descriptive(validated, discriminator=False, influence_deletions=deletions)
        assoc = pd.DataFrame(result.relate["associations"])
        row = assoc[(assoc["attribute"] == "C") & (assoc["descriptor"] == "two_spike")].iloc[0]
        return bool(row["influence_robust"])

    assert spike_robust(1)  # default leave-one-out keeps the two-support spike
    assert not spike_robust(2)  # delete-two demotes it


def test_relate_marginal_demotes_single_support_spike():
    """A single high-leverage observation must not create a significant association.

    The ``spike`` descriptor is non-zero on exactly one product, which is also a
    large outlier on attribute C, so its in-sample Pearson correlation is strong and
    would pass a bare FDR gate. The leave-one-out jackknife demotes it (removing that
    one product collapses the correlation), while the genuine multi-product driver on
    attribute A survives.
    """
    panel, cov = _leverage_case()
    validated = validate_descriptive(panel, cov, mode="observational")
    result = analyze_descriptive(validated, discriminator=False)
    assoc = pd.DataFrame(result.relate["associations"])

    # New influence-robustness fields are surfaced per association.
    assert {"jackknife_se", "influence_robust", "n_supporting"}.issubset(assoc.columns)

    spike_c = assoc[(assoc["attribute"] == "C") & (assoc["descriptor"] == "spike")].iloc[0]
    genuine_a = assoc[(assoc["attribute"] == "A") & (assoc["descriptor"] == "genuine")].iloc[0]

    # The spike correlates strongly in-sample yet rests on one product: not robust,
    # so not significant, even though its raw p-value is tiny.
    assert abs(spike_c["r"]) > 0.8
    assert spike_c["p_value"] < 0.05
    assert not spike_c["influence_robust"]
    assert not spike_c["significant"]

    # The genuine driver is supported across all products: robust and significant.
    assert genuine_a["influence_robust"]
    assert genuine_a["significant"]


def test_discriminator_demotes_single_support_spike():
    """The cross-validated discriminator must also demote a single-support spike."""
    n = len(LEVERAGE_PRODUCTS)
    rng = np.random.default_rng(2)
    genuine = np.linspace(0.0, 1.0, n)
    spike = np.zeros(n)
    spike[_SPIKE_INDEX] = 1.0
    agg = pd.DataFrame(
        {
            "A": 4.0 * genuine + rng.normal(0, 0.1, n),
            "C": np.where(np.arange(n) == _SPIKE_INDEX, 5.0, 0.0) + rng.normal(0, 0.1, n),
        },
        index=LEVERAGE_PRODUCTS,
    )
    cov = pd.DataFrame(
        {"genuine": genuine, "spike": spike, "noise": rng.normal(0, 1, n)}, index=LEVERAGE_PRODUCTS
    )
    disc = discriminate_observational(agg, cov, n_components=1, n_permutations=99, random_state=0)
    desc = pd.DataFrame(disc["descriptors"])

    assert "jackknife_significant" in desc.columns
    # The spike's predictive weight rests on one product, so its jackknife interval
    # spans zero and it is never confirmed.
    spike_rows = desc[desc["descriptor"] == "spike"]
    assert not spike_rows["jackknife_significant"].any()
    assert not spike_rows["discriminator_significant"].any()
    # The genuine driver stays jackknife-stable on the attribute it drives, so the
    # jackknife gate demotes the single-support spike without over-pruning a real
    # driver's predictive coefficient.
    genuine_a = desc[(desc["descriptor"] == "genuine") & (desc["attribute"] == "A")].iloc[0]
    assert genuine_a["jackknife_significant"]


def test_discriminator_small_sample_skips_jackknife_gate():
    """With too few products the LOO gate is skipped, so nothing is confirmed.

    Below the cross-validation floor the per-coefficient jackknife cannot be run, so
    ``jackknife_significant`` defaults to ``False`` and no descriptor is confirmed.
    """
    products = [f"P{i}" for i in range(4)]
    rng = np.random.default_rng(0)
    u = np.linspace(0.0, 1.0, 4)
    agg = pd.DataFrame({"A": 2.0 * u + rng.normal(0, 0.05, 4)}, index=products)
    cov = pd.DataFrame({"d1": u, "d2": rng.normal(0, 1, 4)}, index=products)

    disc = discriminate_observational(agg, cov, n_components=1, n_permutations=19, random_state=0)
    desc = pd.DataFrame(disc["descriptors"])
    assert not desc["jackknife_significant"].any()
    assert not desc["discriminator_significant"].any()


def _null_case(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product-mean table + covariates with two genuine drivers and six noise columns."""
    rng = np.random.default_rng(seed)
    n = 20
    products = [f"P{i}" for i in range(n)]
    driver1 = np.linspace(0.0, 1.0, n) + rng.normal(0, 0.03, n)
    driver2 = rng.normal(0, 1, n)
    cov = pd.DataFrame(
        {"driver1": driver1, "driver2": driver2, **{f"noise{j}": rng.normal(0, 1, n) for j in range(6)}},
        index=products,
    )
    cov.index.name = "product"
    agg = pd.DataFrame(
        {"A": 3.0 * driver1 + rng.normal(0, 0.2, n), "B": 2.0 * driver2 + rng.normal(0, 0.2, n)},
        index=products,
    )
    return agg, cov


def test_permutation_null_flags_drivers_over_noise():
    """Genuine drivers clear the permuted-column null; noise columns fall below them."""
    agg, cov = _null_case()
    result = permutation_column_null(agg, cov, n_iter=12, min_knockoffs=7, random_state=0)
    assert result["ok"]
    assert result["n_descriptors"] == 8
    assert result["n_knockoffs"] == 7
    by_name = {r["descriptor"]: r for r in result["descriptors"]}

    for driver in ("driver1", "driver2"):
        assert by_name[driver]["vip_exceeds_null"]
        assert by_name[driver]["cv_beta_exceeds_null"]

    # Every noise column scores below both drivers on VIP (robust to platform jitter near
    # the threshold), and no noise column clears the beta null.
    min_driver_vip = min(by_name["driver1"]["vip"], by_name["driver2"]["vip"])
    for j in range(6):
        noise = by_name[f"noise{j}"]
        assert noise["vip"] < min_driver_vip
        assert not noise["cv_beta_exceeds_null"]


def test_permutation_null_ignore_drops_columns():
    """`ignore` removes named descriptors from the fit and the output."""
    agg, cov = _null_case()
    result = permutation_column_null(agg, cov, ignore=["noise0", "noise1"], n_iter=5, random_state=0)
    names = [r["descriptor"] for r in result["descriptors"]]
    assert result["n_descriptors"] == 6
    assert result["ignored"] == ["noise0", "noise1"]
    assert "noise0" not in names
    assert "noise1" not in names


def test_permutation_null_unknown_ignore_name_raises():
    """A descriptor name that is not in the block fails loudly rather than silently."""
    agg, cov = _null_case()
    with pytest.raises(ValueError, match="not in the covariate block"):
        permutation_column_null(agg, cov, ignore=["not_a_real_column"], n_iter=2)


def test_permutation_null_is_deterministic():
    """The same seed gives identical thresholds and flags."""
    agg, cov = _null_case()
    first = permutation_column_null(agg, cov, n_iter=5, random_state=1)
    second = permutation_column_null(agg, cov, n_iter=5, random_state=1)
    assert first["descriptors"] == second["descriptors"]


def test_permutation_null_validates_parameters():
    """Out-of-range fraction / quantile / counts are rejected."""
    agg, cov = _null_case()
    with pytest.raises(ValueError, match="fraction must be positive"):
        permutation_column_null(agg, cov, fraction=0.0, n_iter=2)
    with pytest.raises(ValueError, match="quantile must be in"):
        permutation_column_null(agg, cov, quantile=1.5, n_iter=2)
    with pytest.raises(ValueError, match="at least 1"):
        permutation_column_null(agg, cov, min_knockoffs=0, n_iter=2)


def test_permutation_null_max_knockoffs_caps_the_count():
    """`max_knockoffs` bounds the knockoff count even when the fraction is larger."""
    agg, cov = _null_case()
    result = permutation_column_null(agg, cov, fraction=0.9, max_knockoffs=3, n_iter=3, random_state=0)
    assert result["n_knockoffs"] == 3


def test_permutation_null_too_few_descriptors_returns_not_ok():
    """Fewer than two descriptors after ignoring cannot form a null."""
    agg, cov = _null_case()
    keep_two = cov[["driver1", "driver2"]]
    result = permutation_column_null(agg, keep_two, ignore=["driver2"], n_iter=2)
    assert not result["ok"]
    assert "at least 2 descriptors" in result["reason"]


def test_permutation_null_degenerate_block_returns_not_ok():
    """A no-variance descriptor block degrades gracefully instead of crashing."""
    products = [f"P{i}" for i in range(8)]
    cov = pd.DataFrame(
        {"c1": np.ones(8), "c2": np.full(8, 2.0), "c3": np.full(8, 3.0)}, index=products
    )
    agg = pd.DataFrame({"A": np.arange(8.0)}, index=products)
    result = permutation_column_null(agg, cov, n_iter=3, min_knockoffs=2, random_state=0)
    assert not result["ok"]
    assert "singular" in result["reason"]


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
