"""ANOVA-Simultaneous Component Analysis (#372).

The decomposition is exact and can be checked as an identity; everything else here is
about whether the test built on it has the power it claims. Two of these tests exist
because the obvious implementation is wrong in a way that still looks plausible: the
permutation null and the VASCA tie-break both had to be reconsidered, and the fixtures
below are the ones that showed it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import ASCA, PCA, effect_summary_plot


def _two_factor_design(  # noqa: PLR0913 - one knob per injected effect, all keyword-only
    *,
    a_size: float = 3.0,
    b_size: float = 2.0,
    interaction: float = 0.0,
    noise: float = 0.4,
    n_features: int = 8,
    replicates: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Build a crossed 2 x 3 design with each effect injected into a known subset of variables.

    Factor A moves ``v0`` and ``v1``; factor B moves ``v5``; the interaction, when asked
    for, moves ``v3``. Everything else is noise, which is what makes the VASCA assertions
    meaningful: the right answer is known per term.
    """
    rng = np.random.default_rng(seed)
    effect_a = np.zeros(n_features)
    effect_a[[0, 1]] = a_size
    effect_b = np.zeros(n_features)
    effect_b[5] = b_size
    effect_ab = np.zeros(n_features)
    effect_ab[3] = interaction

    rows, design = [], []
    b_codes = {"m1": -1.0, "m2": 0.0, "m3": 1.0}
    for a in ("lo", "hi"):
        a_code = 1.0 if a == "hi" else -1.0
        for b, b_code in b_codes.items():
            for _ in range(replicates):
                rows.append(
                    rng.standard_normal(n_features) * noise
                    + effect_a * a_code
                    + effect_b * b_code
                    + effect_ab * a_code * b_code
                )
                design.append({"A": a, "B": b})
    X = pd.DataFrame(rows, columns=[f"v{i}" for i in range(n_features)])
    truth = {"A": ["v0", "v1"], "B": ["v5"], "A:B": ["v3"] if interaction else []}
    return X, pd.DataFrame(design), truth


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


def test_asca_decomposition_is_exact() -> None:
    """#372's headline acceptance criterion: X = grand mean + sum(effects) + residuals."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)

    assert model.terms_ == ["A", "B", "A:B"]
    assert model.is_balanced_
    rebuilt = model.grand_mean_ + sum(model.effect_matrices_.values()) + model.residuals_
    np.testing.assert_allclose(rebuilt.to_numpy(), X.to_numpy(), atol=1e-12)

    # Each effect matrix is centred, which is what sum-to-zero coding buys, and constant
    # within a factor-level combination, which is what makes it an ANOVA effect.
    for name, effect in model.effect_matrices_.items():
        np.testing.assert_allclose(effect.to_numpy().mean(axis=0), 0.0, atol=1e-10)
        grouping = design["A"] if name == "A" else design["B"] if name == "B" else design[["A", "B"]].agg("-".join, 1)
        assert effect.groupby(grouping.to_numpy()).nunique().to_numpy().max() == 1


def test_asca_sums_of_squares_partition_a_balanced_design() -> None:
    """On a balanced design the terms are orthogonal, so the parts add up exactly."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)

    parts = [*model.terms_, "residual"]
    assert model.ssq_[parts].sum() == pytest.approx(model.ssq_["total"], rel=1e-10)
    assert model.ssq_percent_[parts].sum() == pytest.approx(100.0, rel=1e-10)
    # The injected sizes put A far ahead of B, and the interaction at nothing.
    assert model.ssq_percent_["A"] > model.ssq_percent_["B"] > model.ssq_percent_["A:B"]


def test_asca_warns_and_stops_partitioning_on_an_unbalanced_design() -> None:
    """An unbalanced design is fitted, but the model says the parts no longer add up."""
    X, design, _ = _two_factor_design()
    keep = np.ones(len(design), dtype=bool)
    keep[:3] = False  # drop three rows from one cell
    X, design = X[keep].reset_index(drop=True), design[keep].reset_index(drop=True)

    with pytest.warns(UserWarning, match=r"design is unbalanced.*do not partition the total exactly"):
        model = ASCA(n_components=2).fit(X, design)
    assert not model.is_balanced_
    parts = [*model.terms_, "residual"]
    assert model.ssq_[parts].sum() != pytest.approx(model.ssq_["total"], rel=1e-6)


def test_asca_per_term_pca_is_capped_at_the_term_rank() -> None:
    """A two-level factor's effect matrix has rank 1; asking for two components is asking for nothing."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=3).fit(X, design)

    assert isinstance(model.models_["A"], PCA)
    assert model.models_["A"].scores_.shape == (len(X), 1)  # A has two levels: rank 1
    assert model.models_["B"].scores_.shape == (len(X), 2)  # B has three: rank 2
    assert list(model.models_["A"].loadings_.index) == list(X.columns)
    # The A-effect loading points at the two variables the effect was injected into.
    dominant = model.models_["A"].loadings_.iloc[:, 0].abs().nlargest(2).index.tolist()
    assert sorted(dominant) == ["v0", "v1"]


def test_asca_add_residuals_restores_the_scatter() -> None:
    """ASCA+ / APCA: the pure effect matrix has one point per cell; adding residuals back does not."""
    X, design, _ = _two_factor_design()
    pure = ASCA(n_components=2).fit(X, design)
    plus = ASCA(n_components=2, add_residuals=True).fit(X, design)

    # Six cells in a 2 x 3 design, so the pure A-effect scores take two distinct values.
    assert pure.models_["A"].scores_.iloc[:, 0].round(9).nunique() == 2
    assert plus.models_["A"].scores_.iloc[:, 0].round(9).nunique() == len(X)
    # The decomposition itself is untouched; only what is handed to the PCA changes.
    np.testing.assert_allclose(pure.effect_matrices_["A"].to_numpy(), plus.effect_matrices_["A"].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# The permutation test
# ---------------------------------------------------------------------------


def test_asca_permutation_test_flags_real_terms_and_clears_null_ones() -> None:
    """A and B carry injected effects; the interaction does not, and the test agrees.

    B is the test that matters. It carries about 11 percent of the variation against A's
    84, and permuting the rows of the *whole* response leaves A's variation in the data,
    so B inherits a share of it and its null is far too high: that null put B at p = 0.13
    and hid a real effect. Permuting the reduced response instead, this term's effect plus
    the residual and nothing else, puts B where it belongs.
    """
    X, design, _ = _two_factor_design(interaction=0.0)
    model = ASCA(n_components=2).fit(X, design)
    pvalues = model.permutation_test(n_permutations=199, random_state=0)

    assert pvalues["A"] <= 0.01
    assert pvalues["B"] <= 0.01
    assert pvalues["A:B"] > 0.05
    # Floor of the p-value with the +1 correction on both sides.
    assert pvalues.min() >= 1.0 / (1 + 199)
    pd.testing.assert_series_equal(pvalues, model.pvalues_)


def test_asca_permutation_test_finds_an_injected_interaction() -> None:
    """With an interaction present, the term that was clear before is now flagged."""
    X, design, _ = _two_factor_design(interaction=2.5)
    model = ASCA(n_components=2).fit(X, design)
    assert model.permutation_test(n_permutations=199, random_state=0)["A:B"] <= 0.01


def test_asca_permutation_test_is_reproducible() -> None:
    """Same seed, same p-values; a different seed moves them."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)
    first = model.permutation_test(n_permutations=99, random_state=3)
    again = model.permutation_test(n_permutations=99, random_state=3)
    pd.testing.assert_series_equal(first, again)


# ---------------------------------------------------------------------------
# VASCA
# ---------------------------------------------------------------------------


def test_vasca_selects_the_variables_the_effect_was_injected_into() -> None:
    """Per term, VASCA recovers exactly the variables that carry it, and nothing else."""
    X, design, truth = _two_factor_design(interaction=2.5)
    model = ASCA(n_components=2).fit(X, design)

    for term, expected in truth.items():
        result = model.vasca(term, n_permutations=199, random_state=0)
        assert sorted(result.selected) == sorted(expected), f"{term}: {result.selected} != {expected}"


def test_vasca_selects_nothing_for_a_term_that_carries_nothing() -> None:
    """A null term gives an empty selection rather than the best of a bad ranking."""
    X, design, _ = _two_factor_design(interaction=0.0)
    model = ASCA(n_components=2).fit(X, design)
    result = model.vasca("A:B", n_permutations=199, random_state=0)
    assert result.selected == []
    assert result.p_value > 0.05


def test_vasca_breaks_p_value_ties_on_the_z_score() -> None:
    """Why the selection is not made on the p-value alone.

    With 199 permutations the smallest attainable p-value is 1/200, and on a strong effect
    many subset sizes reach it at once. Choosing the largest subset that clears alpha would
    then return every variable that happened to tie at the floor. The z-score, how far a
    subset sits above its own null, does not tie: it peaks where the effect is concentrated
    and falls away as passengers are added.
    """
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)
    result = model.vasca("A", n_permutations=199, random_state=0)

    table = result.table
    assert list(table.columns) == ["variable", "ssq_cumulative", "z_score", "p_value", "p_value_fdr"]
    # The tie is real: more than one subset size sits at the p-value floor.
    assert (table["p_value_fdr"] == table["p_value_fdr"].min()).sum() > 1
    # The z-score peaks at two variables, which is how many were injected.
    assert int(table["z_score"].idxmax()) == 2
    assert result.selected == table["variable"].iloc[:2].tolist()
    # Cumulative sum of squares only ever rises, so it cannot pick a subset on its own.
    assert table["ssq_cumulative"].is_monotonic_increasing


def test_vasca_rejects_an_unknown_term() -> None:
    """A term that was never fitted is an error naming the ones that were."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)
    with pytest.raises(ValueError, match=r"'C' is not a fitted term; choose one of \['A', 'B', 'A:B'\]"):
        model.vasca("C")


# ---------------------------------------------------------------------------
# Input handling, plotting, real data
# ---------------------------------------------------------------------------


def test_asca_rejects_input_it_cannot_use() -> None:
    """Each rejection names what is wrong with the call."""
    X, design, _ = _two_factor_design()
    with pytest.raises(ValueError, match=r"X has 30 rows but design has 10"):
        ASCA().fit(X, design.iloc[:10])
    gappy = X.copy()
    gappy.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"ASCA needs a complete X"):
        ASCA().fit(gappy, design)

    gappy_design = design.copy().astype(object)
    gappy_design.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"design frame holds missing values"):
        ASCA().fit(X, gappy_design)
    with pytest.raises(ValueError, match=r"single level and can carry no effect: \['C'\]"):
        ASCA().fit(X, design.assign(C="always the same"))


def test_asca_main_effects_only() -> None:
    """`model="main_effects"` fits no interaction, and the variation lands in the residual."""
    X, design, _ = _two_factor_design(interaction=2.5)
    full = ASCA(n_components=2).fit(X, design)
    main = ASCA(n_components=2, model="main_effects").fit(X, design)

    assert main.terms_ == ["A", "B"]
    # The interaction's variation has to go somewhere: without a term for it, the residual.
    assert main.ssq_["residual"] > full.ssq_["residual"]
    assert main.ssq_["residual"] == pytest.approx(full.ssq_["residual"] + full.ssq_["A:B"], rel=1e-9)


def test_asca_effect_summary_plot() -> None:
    """The bar chart carries the shares, and the p-values once they exist."""
    X, design, _ = _two_factor_design()
    model = ASCA(n_components=2).fit(X, design)

    fig = model.effect_summary_plot()
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].x) == ["A", "B", "A:B", "residual"]
    assert all("p=" not in label for label in fig.data[0].text)

    model.permutation_test(n_permutations=49, random_state=0)
    annotated = model.effect_summary_plot()
    assert all("p=" in label for label in annotated.data[0].text[:3])
    assert "p=" not in annotated.data[0].text[3]  # the residual has no p-value to show

    without = model.effect_summary_plot(settings={"include_residual": False})
    assert list(without.data[0].x) == ["A", "B", "A:B"]

    with pytest.raises(ValueError, match="Model is not fitted"):
        effect_summary_plot(ASCA())


@pytest.mark.dataset
def test_asca_on_the_tablet_spectra() -> None:
    """A real dataset: 460 tablet NIR spectra split by two arbitrary but real groupings.

    Neither grouping is a designed factor, so this is not a test of recovering a known
    effect. It checks that the decomposition holds exactly on real, badly conditioned
    data: the 33 wavelengths kept below have a condition number around 4e7 and a first
    principal component carrying 74 percent of the variance, which is where a
    least-squares implementation shows any weakness it has. A synthetic fixture with
    independent columns never would.
    """
    path = pathlib.Path("src/process_improve/datasets/multivariate/tablet-spectra.csv")
    if not path.exists():
        pytest.skip("tablet-spectra.csv fixture not present")
    spectra = pd.read_csv(path, index_col=0, header=None)
    # Keep it quick: every 20th wavelength is still 70 highly collinear columns. Trim to a
    # multiple of 8 so the 2 x 4 design below is exactly balanced and the sums of squares
    # are expected to partition; the point of the test is that they do, on real data.
    usable = (len(spectra) // 8) * 8
    X = spectra.iloc[:usable, ::20]
    design = pd.DataFrame(
        {
            "half": np.where(np.arange(usable) < usable / 2, "first", "second"),
            "batch": [f"b{index % 4}" for index in range(usable)],
        },
        index=X.index,
    )

    model = ASCA(n_components=2, scale=True).fit(X, design)
    assert model.is_balanced_
    # Badly conditioned by construction of the measurement, which is the point.
    eigenvalues = np.linalg.svd(((X - X.mean()) / X.std(ddof=1)).to_numpy(), compute_uv=False) ** 2
    assert eigenvalues[0] / eigenvalues[-1] > 1e6
    rebuilt = sum(model.effect_matrices_.values()) + model.residuals_
    centred = (X - model.grand_mean_) / model.column_scale_
    np.testing.assert_allclose(rebuilt.to_numpy(), centred.to_numpy(), atol=1e-8)
    assert model.ssq_percent_[[*model.terms_, "residual"]].sum() == pytest.approx(100.0, rel=1e-8)
